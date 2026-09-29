from __future__ import annotations

import fcntl
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .corpus import ANNCorpus, as_tensor, exact_topk, load_corpus
from .index import (BUDGET_GRID, IndexSpec, IndexStorage, build_index, empty_index,
                    prepare, search, set_budget)

# ASSUMPTIONS: dict[str, str] = {
#     "ground_truth": (
#         "Recall is measured against the EXACT top-k of the clean query over the CLEAN "
#         "corpus — what an undefended system with brute-force search returns. So a "
#         "defended index is charged for both the defense's distortion and the index's "
#         "approximation. index_recall isolates the second: even the undefended system "
#         "loses it, and a target above index_recall is unachievable for every defense."
#     ),
#     "queries": (
#         "Queries are documents from the indexed corpus (the first --ann-queries rows "
#         "of the seeded sample), self excluded — the same doc-as-query convention as "
#         "the brute-force recall@k, so the two numbers differ only in the search. With "
#         "--ann-query-side defended (default) the query is that document's defended "
#         "vector, which is exactly what the brute-force metric compared; 'clean' "
#         "queries a defended index with an undefended query, the deployment where only "
#         "stored vectors are perturbed."
#     ),
#     "whole_corpus_defended": (
#         "The defense is applied to EVERY indexed document on every probe, and the index "
#         "is rebuilt. Perturbing only the queries' neighbourhoods would under-count the "
#         "distractors a defense promotes into the top-k."
#     ),
#     "storage_normalises": (
#         "With --ann-metric cosine the index stores L2-normalised vectors (the standard "
#         "faiss cosine recipe), and in ANN mode that stored vector is what the attack "
#         "inverts. vec2text, lapmech and remote_rag deliberately do not renormalise; a "
#         "cosine index discards that magnitude, so under ANN storage vec2text becomes "
#         "gaussian by construction. That is a property of storing in a cosine index, "
#         "not a bug. --ann-metric ip keeps raw vectors."
#     ),
#     "budget_fixed": (
#         "Utility is measured at ONE search budget (--ann-budget: efSearch for HNSW, "
#         "nprobe for IVF) — the operator's production setting. A defense that needs "
#         "more budget to recover is reported through the surcharge, not by quietly "
#         "raising the budget for it."
#     ),
#     "partition_ceiling": (
#         "With --partition the treatment arms are stored in the partitioned, rotated "
#         "index, but the undefended CEILING and the two floors are measured in the "
#         "ordinary unpartitioned index. leak_norm is the fraction of the undefended "
#         "system's leakage that survives, so its denominator must not already include "
#         "the defense being evaluated. The keyed rotation alone is its own arm "
#         "(--defense keyed_rotation)."
#     ),
#     "partition_utility": (
#         "Partitioned utility is recall of the full partitioned system — routing, "
#         "per-cell HNSW and the base defense — against exact clean truth, at "
#         "(nprobe=--kr-nprobe, efSearch=--ann-budget). The swept budget is nprobe. "
#         "index_recall is then the undefended corpus in the partitioned index, so a "
#         "matched knob under --partition is solved against a different cap than without."
#     ),
#     "hnsw_nondeterminism": (
#         "faiss inserts into HNSW with many threads, so two builds over IDENTICAL "
#         "vectors differ. Measured on 522k Quora docs, recall@10 over 4 rebuilds: std "
#         "0.0018 at 256 faiss threads (range 0.961-0.966) and 0.0006 at 32 threads — "
#         "more threads, more insertion-order variation, more jitter. This is the noise "
#         "floor of every ANN recall number here: it is below the 0.01 matching "
#         "tolerance but NOT negligible against it, so a matched arm's recall is "
#         "determined to about +-0.002, and differences between arms smaller than that "
#         "are not real. Averaging draws (--match-verify-repeats) shrinks it as "
#         "1/sqrt(n); running defenses in parallel (PAR>1) shrinks it too, by giving "
#         "each process fewer threads."
#     ),
# }


@dataclass
class AnnUtility:
    defense: str
    recall: float
    recall_se: float
    exact_recall: float
    index_recall: float
    self_recall: float
    k: int
    spec: IndexSpec
    n_docs: int
    n_queries: int
    query_side: str
    curve: list[tuple[int, float, float]] = field(default_factory=list)  # budget, recall, self
    clean_self_base: float = float("nan")
    clean_curve: dict[int, float] = field(default_factory=dict)          # budget -> recall
    budget_param: str | None = None     # efSearch | nprobe (partitioned) | None (exact)
    budget: int = 0
    partition: dict[str, Any] = field(default_factory=dict)  # partition metrics, if any

    @property
    def surcharge(self) -> float:
        if not self.curve or not self.clean_curve:
            return float("nan")
        need = self.clean_self_base
        clean = _crossing(sorted(self.clean_curve.items()), need)
        return _crossing([(b, s) for b, _, s in self.curve], need) / clean

    def to_dict(self) -> dict[str, Any]:
        return {
            "defense": self.defense, "k": self.k,
            "ann_recall": self.recall, "ann_recall_se": self.recall_se,
            "exact_recall": self.exact_recall, "index_recall": self.index_recall,
            "self_recall": self.self_recall, "surcharge": self.surcharge,
            "index": self.spec.factory, "metric": self.spec.metric,
            "budget_param": self.budget_param, "budget": self.budget,
            "ef_search": self.spec.budget, "ef_construction": self.spec.ef_construction,
            "n_docs": self.n_docs, "n_queries": self.n_queries,
            "query_side": self.query_side,
            "curve": [list(c) for c in self.curve],
            "partition": self.partition,
        }


@dataclass
class CachedUtility:
    recall: float
    recall_se: float
    exact_recall: float
    index_recall: float
    self_recall: float
    surcharge: float
    partition: dict[str, Any]
    budget_param: str | None
    budget: int
    k: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CachedUtility":
        return cls(recall=d["ann_recall"], recall_se=d.get("ann_recall_se", float("nan")),
                   exact_recall=d.get("exact_recall", float("nan")),
                   index_recall=d.get("index_recall", float("nan")),
                   self_recall=d.get("self_recall", float("nan")),
                   surcharge=d.get("surcharge", float("nan")),
                   partition=d.get("partition") or {},
                   budget_param=d.get("budget_param"), budget=d.get("budget", 0),
                   k=d.get("k", 10))


UTILITY_CACHE = "ann_utility.json"


def utility_key(args: Any, arm: str) -> str:
    return f"{arm}|{probe_tag(args)}|r{args.match_repeats}|s{args.seed}"


def read_utility(directory: "Path", key: str) -> CachedUtility | None:
    f = Path(directory) / UTILITY_CACHE
    if not f.exists():
        return None
    try:
        hit = json.loads(f.read_text()).get(key)
    except (OSError, json.JSONDecodeError):
        return None
    return CachedUtility.from_dict(hit) if hit else None


def _merge_json(f: "Path", key: str, value: Any) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            text = fh.read()
            payload = json.loads(text) if text.strip() else {}
            payload[key] = value
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload, indent=2))
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_utility(directory: "Path", key: str, u: "AnnUtility") -> None:
    _merge_json(Path(directory) / UTILITY_CACHE, key, u.to_dict())


def _crossing(curve: list[tuple[int, float]], need: float) -> float:
    prev = None
    for b, v in curve:
        if v >= need:
            if prev is None or v == prev[1]:
                return float(b)
            t = (need - prev[1]) / (v - prev[1])
            return float(2 ** (math.log2(prev[0]) + t * (math.log2(b) - math.log2(prev[0]))))
        prev = (b, v)
    return float("inf")


def _drop_self(I: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    is_self = I == rows[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")
    return np.take_along_axis(I, order, axis=1)[:, :k]


def overlap(I: np.ndarray, T: np.ndarray) -> float:
    k = T.shape[1]
    hits = (I[:, :, None] == T[:, None, :]).any(axis=2).sum(axis=1)
    return float(hits.mean() / k)


class _FlatSystem:
    def __init__(self, index: Any, spec: IndexSpec) -> None:
        self.index, self.spec = index, spec

    @property
    def budget_param(self) -> str | None:
        return self.spec.budget_param

    @property
    def budget(self) -> int:
        return self.spec.budget

    def grid(self) -> list[int]:
        if self.spec.budget_param is None:
            return [self.spec.budget]
        return sorted(set(BUDGET_GRID) | {self.spec.budget})

    def search(self, Q: np.ndarray, k: int, budget: int | None = None) -> np.ndarray:
        set_budget(self.index, self.spec, budget)
        I = search(self.index, Q, k)
        set_budget(self.index, self.spec)
        return I

    def report(self) -> dict[str, Any]:
        return {}


class AnnProbe:
    def __init__(
        self,
        corpus: ANNCorpus,
        spec: IndexSpec,
        *,
        k: int = 10,
        query_side: str = "defended",
        repeats: int = 3,
        seed: int = 42,
        device: str | None = None,
        partition: Any = None,
        cache_dir: "Path | None" = None,
    ) -> None:
        if query_side not in ("defended", "clean"):
            raise ValueError(f"query_side must be 'defended' or 'clean', got {query_side!r}")
        self.corpus = corpus
        self.spec = spec
        self.k = k
        self.query_side = query_side
        self.repeats = max(1, repeats)
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.partition = partition            # ANN.partitioned.PartitionSpec | None
        # Where clean_baseline() is cached. The undefended index is rebuilt by every
        # ladder invocation and every report otherwise — 40 s a time, partitioned.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._state: dict[str, dict] = {}     # loaded defense objects, per defense
        self._clean: dict[str, Any] | None = None
        if partition is not None:
            from .partitioned import check_flat

            check_flat(spec)

    @classmethod
    def from_args(cls, args: Any) -> "AnnProbe":
        return cls(load_corpus(args), spec_from_args(args), k=args.match_k,
                   query_side=args.ann_query_side, repeats=args.match_repeats,
                   seed=args.seed, device=args.device, partition=partition_from_args(args))

    def describe(self) -> str:
        sp = self.spec
        if self.partition is None:
            return f"{sp.factory} [{sp.metric}] @ {sp.budget_label}{sp.budget}"
        pt = self.partition
        return (f"partitioned {sp.factory} [{sp.metric}], m={pt.m} alpha={pt.alpha:g} "
                f"b={pt.branching}, @ (nprobe {pt.nprobe}, ef {sp.budget})"
                + ("" if pt.rotate else ", NO rotation"))

    def _defend(self, defense: str, kwargs: dict[str, Any]) -> np.ndarray:
        from attacker.algen.defenses import apply_defense

        if defense in ("none", "", None, "keyed_rotation"):
            if defense == "keyed_rotation" and self.partition is None:
                raise ValueError("keyed_rotation is measured through the partitioned "
                                 "index: pass --partition")
            return self.corpus.vectors
        X = as_tensor(self.corpus.vectors, self.device)
        kw = {k: v for k, v in kwargs.items() if k != "defense"}
        # Keyed on the checkpoint too, so two fits of one defense never share state.
        key = f"{defense}|{kw.get(f'{defense}_checkpoint')}"
        out, self._state[key] = apply_defense(X, defense, **kw, state=self._state.get(key))
        return out.detach().float().cpu().numpy()

    def _queries(self, prepared: np.ndarray) -> np.ndarray:
        if self.query_side == "defended":
            return np.ascontiguousarray(prepared[self.corpus.query_rows])
        return prepare(self.corpus.vectors[self.corpus.query_rows], self.spec.metric)

    def _system(self, P: np.ndarray, seed: int, need_stored: bool = True) -> tuple[Any, Any]:
        if self.partition is not None:
            from .partitioned import PartitionedIndex

            return PartitionedIndex.build(P, self.spec, self.partition, seed=seed,
                                          device=self.device), P
        index = build_index(P, self.spec, seed=seed)
        stored = IndexStorage.from_index(index, self.spec).round_trip(P) if need_stored else P
        return _FlatSystem(index, self.spec), stored

    def _clean_key(self) -> str:
        pt = f"|{self.partition.tag()}" if self.partition is not None else ""
        return (f"__clean__|{self.spec.tag()}{pt}|n{self.corpus.n}"
                f"|q{len(self.corpus.query_rows)}|{self.query_side}|k{self.k}|s{self.seed}")

    def clean_baseline(self) -> dict[str, Any]:
        if self._clean is None and self.cache_dir is not None:
            f = self.cache_dir / UTILITY_CACHE
            try:
                hit = json.loads(f.read_text()).get(self._clean_key()) if f.exists() else None
            except (OSError, json.JSONDecodeError):
                hit = None
            if hit:
                # JSON keys are strings; the curve is indexed by budget.
                self._clean = {**hit, "curve": {int(k): v for k, v in hit["curve"].items()}}
                print(f"[ann] clean {self.describe()}: index-only recall@{self.k} = "
                      f"{self._clean['index_recall']:.4f} (cached)")
        if self._clean is None:
            truth = self.corpus.truth(self.k, self.spec.metric)
            system, _ = self._system(prepare(self.corpus.vectors, self.spec.metric),
                                     self.seed, need_stored=False)
            Q = prepare(self.corpus.vectors, self.spec.metric)[self.corpus.query_rows]
            rows = self.corpus.query_rows
            curve = {b: overlap(_drop_self(system.search(Q, self.k + 1, b), rows, self.k), truth)
                     for b in system.grid()}
            self._clean = {"index_recall": curve.get(system.budget, float("nan")),
                           "curve": curve, "partition": system.report(),
                           "budget_param": system.budget_param, "budget": system.budget}
            print(f"[ann] clean {self.describe()}: index-only recall@{self.k} = "
                  f"{self._clean['index_recall']:.4f}")
            if self.cache_dir is not None:
                _merge_json(self.cache_dir / UTILITY_CACHE, self._clean_key(), self._clean)
        return self._clean

    def measure(
        self, defense: str, kwargs: dict[str, Any] | None = None, *, curve: bool = False,
        repeats: int | None = None, seed: int | None = None, decompose: bool = True,
    ) -> AnnUtility:

        kwargs = dict(kwargs or {})
        base = self.clean_baseline()
        truth = self.corpus.truth(self.k, self.spec.metric)
        rows = self.corpus.query_rows
        seed = self.seed if seed is None else seed
        n_rep = (1 if defense in ("none", "", None, "keyed_rotation")
                 else (repeats or self.repeats))

        scores: list[float] = []
        exact_r = self_r = float("nan")
        pts: list[tuple[int, float, float]] = []
        part: dict[str, Any] = {}
        for i in range(n_rep):
            torch.manual_seed(seed + i)
            np.random.seed(seed + i)
            D = self._defend(defense, kwargs)
            want = decompose or curve
            # Normalise once, then drop the raw defended copy before the index is built
            # peak RSS is set by how many full-corpus arrays are live at the same moment.
            P = prepare(D, self.spec.metric)
            del D
            system, stored = self._system(P, seed + i, need_stored=want)
            Q = self._queries(P)
            I = _drop_self(system.search(Q, self.k + 1), rows, self.k)
            scores.append(overlap(I, truth))
            if i == 0:
                part = system.report()
            if i == 0 and want:
                E = exact_topk(stored, Q, self.k, exclude=rows)
                exact_r, self_r = overlap(E, truth), overlap(I, E)
                if curve:
                    for b in system.grid():
                        Ib = _drop_self(system.search(Q, self.k + 1, b), rows, self.k)
                        pts.append((b, overlap(Ib, truth), overlap(Ib, E)))
            del system, stored, P

        se = float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
        return AnnUtility(
            defense=defense or "none", recall=float(np.mean(scores)), recall_se=se,
            exact_recall=exact_r, index_recall=base["index_recall"], self_recall=self_r,
            k=self.k, spec=self.spec, n_docs=self.corpus.n, n_queries=len(rows),
            query_side=self.query_side, curve=pts,
            clean_self_base=base["index_recall"], clean_curve=dict(base["curve"]),
            budget_param=base["budget_param"], budget=base["budget"], partition=part,
        )

    def probe_fn(
        self, defense: str, extra: dict[str, Any], knob_kwargs: Callable[[float], dict],
        *, repeats: int | None = None, seed: int | None = None,
    ) -> Callable[[float], tuple[float, float]]:

        def probe(value: float) -> tuple[float, float]:
            try:
                u = self.measure(defense, {**extra, **knob_kwargs(value)},
                                 repeats=repeats, seed=seed, decompose=False)
            except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
                print(f"  [undefined] {defense} at {value:g}: {type(exc).__name__}: {exc}")
                return float("nan"), float("nan")
            return u.recall, u.recall_se
        return probe

    def flat_storage(self) -> IndexStorage:
        st = IndexStorage.untrained(self.spec, self.corpus.vectors.shape[1])
        if st is None:
            raise ValueError(f"{self.spec.factory} needs a trained codec; use storage()")
        return st

    def storage(self, defense: str, kwargs: dict[str, Any] | None = None) -> Any:
        dim = self.corpus.vectors.shape[1]
        if self.partition is None:
            flat = IndexStorage.untrained(self.spec, dim)
            if flat is not None:
                return flat
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        D = prepare(self._defend(defense, dict(kwargs or {})), self.spec.metric)
        if self.partition is not None:
            from defense.keyed_rotation import KeyedRotation

            from .partitioned import PartitionedStorage

            kr = KeyedRotation(self.partition.config(self.spec.metric, self.seed),
                               device=self.device).fit(D)
            return PartitionedStorage(kr, self.spec, self.partition)
        index = empty_index(self.spec, dim)
        rng = np.random.default_rng(self.seed)
        n = min(len(D), self.spec.train_size)
        index.train(np.ascontiguousarray(D[rng.choice(len(D), size=n, replace=False)]))
        return IndexStorage.from_index(index, self.spec)


def add_ann_args(p: Any) -> None:
    g = p.add_argument_group("ANN utility (--match-metric ann)")
    g.add_argument("--ann-index", default="HNSW32,Flat",
                   help="faiss index_factory string: HNSW32,Flat | HNSW32,SQ8 | "
                        "IVF1024,Flat | IVF1024,PQ96 | Flat (exact baseline)")
    g.add_argument("--ann-metric", default="cosine", choices=["cosine", "ip"],
                   help="cosine stores unit vectors (the attack then sees them too); "
                        "ip stores raw vectors")
    g.add_argument("--ann-budget", type=int, default=64,
                   help="search budget: efSearch for HNSW, nprobe for IVF")
    g.add_argument("--ann-ef-construction", type=int, default=40)
    g.add_argument("--ann-docs", type=int, default=0,
                   help="documents in the index; 0 = the whole corpus minus targets")
    g.add_argument("--ann-queries", type=int, default=1000,
                   help="documents used as queries (self excluded)")
    g.add_argument("--ann-query-side", default="defended", choices=["defended", "clean"],
                   help="whether the query vector passes through the defense")
    g.add_argument("--ann-train-size", type=int, default=100_000,
                   help="rows used to train IVF/PQ/SQ codecs")

    g = p.add_argument_group(
        "locality-keyed rotation (--partition; defense/keyed_rotation.py, ANN/partitioned.py)")
    g.add_argument("--partition", action="store_true",
                   help="store every defended arm in a density-adaptive partitioned index "
                        "with a secret rotation per cell (composed over the base defense). "
                        "Needs --match-metric ann. The ceiling and floors stay unpartitioned")
    g.add_argument("--kr-m", type=int, default=1000,
                   help="target occupancy: cells are split until they hold <= m docs")
    g.add_argument("--kr-cells", type=int, default=None,
                   help="target CELL COUNT C; overrides --kr-m with the occupancy solved "
                        "for it by `python -m defense.keyed_rotation calibrate`. Use this "
                        "for a C-sweep: effective C is ~1.6x N/m, so deriving m as N/C "
                        "mislabels every point")
    g.add_argument("--kr-alpha", type=float, default=1.3,
                   help="hard cap per cell: alpha * m (balance vs routing fidelity)")
    g.add_argument("--kr-branching", type=int, default=8, help="k-means children per split")
    g.add_argument("--kr-candidates", type=int, default=16,
                   help="nearest centroids each doc may be assigned to under the cap")
    g.add_argument("--kr-nprobe", type=int, default=128,
                   help="cells searched per query (efSearch inside a cell is --ann-budget)")
    g.add_argument("--kr-key", default=None,
                   help="the operator's secret; default derived from --seed")
    g.add_argument("--kr-no-rotate", action="store_true",
                   help="control: the same partitioned index with NO rotation")


def spec_from_args(args: Any) -> IndexSpec:
    return IndexSpec(factory=args.ann_index, metric=args.ann_metric,
                     budget=args.ann_budget, ef_construction=args.ann_ef_construction,
                     train_size=args.ann_train_size)


def partition_from_args(args: Any) -> Any:
    if not getattr(args, "partition", False):
        return None
    from .partitioned import PartitionSpec

    m = int(args.kr_m)
    cells = getattr(args, "kr_cells", None)
    if cells:
        from defense.keyed_rotation import m_for_cells

        m = m_for_cells(str(args.attack_dataset), cells, args.kr_alpha, args.kr_branching)
        print(f"[ann] --kr-cells {cells} -> --kr-m {m} ({args.attack_dataset})")

    return PartitionSpec(m=m, alpha=args.kr_alpha, branching=args.kr_branching,
                         candidates=args.kr_candidates, nprobe=args.kr_nprobe,
                         key=args.kr_key or f"keyed-rotation-{args.seed}",
                         rotate=not args.kr_no_rotate)


def probe_tag(args: Any) -> str:
    tag = (f"{spec_from_args(args).tag()}|n{args.ann_docs or 'all'}"
           f"|q{args.ann_queries}|{args.ann_query_side}")
    pt = partition_from_args(args)
    return tag if pt is None else f"{tag}|{pt.utility_tag()}"


def bucket_key(args: Any, target: float) -> str:
    return f"ann@{args.match_k}={target:g}[{probe_tag(args)}]"


def ladder_subdir(args: Any, dim: int) -> str:
    spec = spec_from_args(args)
    tag = f"ann_{spec.storage_tag()}"
    if IndexStorage.untrained(spec, dim) is None:
        tag += f"_n{args.ann_docs or 'all'}"
    return tag


def partition_subdir(args: Any) -> str | None:
    import hashlib

    pt = partition_from_args(args)
    if pt is None:
        return None
    digest = hashlib.sha1(pt.key.encode()).hexdigest()[:6]
    return f"{pt.storage_tag()}_n{args.ann_docs or 'all'}_k{digest}"


__all__ = [
    "ASSUMPTIONS", "AnnUtility", "AnnProbe", "CachedUtility", "add_ann_args",
    "utility_key", "read_utility", "write_utility", "spec_from_args",
    "partition_from_args", "probe_tag", "bucket_key", "ladder_subdir", "partition_subdir",
    "overlap",
]
