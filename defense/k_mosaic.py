from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class KMosaicConfig:

    m: int = 1000                 
    alpha: float = 1.3            
    branching: int = 8            
    candidates: int = 16          
    kmeans_iter: int = 20
    key: str = "keyed-rotation-42"
    seed: int = 42
    metric: str = "cosine"       

    def tag(self) -> str:
        return f"kr_m{self.m}_a{self.alpha:g}_b{self.branching}"


def _normalise(X: np.ndarray) -> np.ndarray:
    return (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def fit_partition(Z: np.ndarray, cfg: KMosaicConfig) -> list[np.ndarray]:
    import faiss

    rng = np.random.default_rng(cfg.seed)
    d = Z.shape[1]
    queue: list[np.ndarray] = [np.arange(len(Z))]
    leaves: list[np.ndarray] = []
    n_split = 0
    while queue:
        idx = queue.pop()
        if len(idx) <= cfg.m:
            leaves.append(idx)
            continue
        b = int(min(cfg.branching, max(2, math.ceil(len(idx) / cfg.m))))
        km = faiss.Kmeans(d, b, niter=cfg.kmeans_iter, seed=cfg.seed + n_split,
                          spherical=cfg.metric == "cosine", verbose=False,
                          min_points_per_centroid=1, max_points_per_centroid=1 << 30)
        n_split += 1
        sub = np.ascontiguousarray(Z[idx])
        km.train(sub)
        _, a = km.index.search(sub, 1)
        parts = [idx[a[:, 0] == j] for j in range(b)]
        parts = [p for p in parts if len(p)]
        if len(parts) < 2:
            parts = [p for p in np.array_split(rng.permutation(idx), b) if len(p)]
        queue.extend(parts)
    return leaves


def _centroids(Z: np.ndarray, labels: np.ndarray, n: int, metric: str) -> np.ndarray:
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=n)
    C = np.zeros((n, Z.shape[1]), dtype=np.float32)
    nz = np.flatnonzero(counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])[nz]
    C[nz] = np.add.reduceat(Z[order], starts, axis=0) / counts[nz, None]
    return _normalise(C) if metric == "cosine" else C


def capped_assign(
    Z: np.ndarray, centroids: np.ndarray, cap: int, candidates: int
) -> np.ndarray:
    import faiss

    N, C = len(Z), len(centroids)
    r = min(candidates, C)
    index = faiss.IndexFlatIP(Z.shape[1])
    index.add(np.ascontiguousarray(centroids))
    S, I = index.search(np.ascontiguousarray(Z), r)

    labels = np.full(N, -1, dtype=np.int64)
    score = np.full(N, -np.inf, dtype=np.float32)
    ptr = np.zeros(N, dtype=np.int64)
    free = np.arange(N)
    for _ in range(r):
        if len(free) == 0:
            break
        prop_cell = I[free, ptr[free]]
        prop_score = S[free, ptr[free]]
        members = np.flatnonzero(labels >= 0)
        pts = np.concatenate([members, free])
        cells = np.concatenate([labels[members], prop_cell])
        scs = np.concatenate([score[members], prop_score])
        order = np.lexsort((-scs, cells))
        pts, cells, scs = pts[order], cells[order], scs[order]
        start = np.searchsorted(cells, cells, side="left")
        rank = np.arange(len(cells)) - start
        keep = rank < cap
        labels[:] = -1
        score[:] = -np.inf
        labels[pts[keep]] = cells[keep]
        score[pts[keep]] = scs[keep]
        rejected = pts[~keep]
        ptr[rejected] += 1
        free = rejected[ptr[rejected] < r]
        exhausted = rejected[ptr[rejected] >= r]
        if len(exhausted):
            free = np.concatenate([free, exhausted])
            break
    left = np.flatnonzero(labels < 0)
    if len(left):
        counts = np.bincount(labels[labels >= 0], minlength=C)
        sims = Z[left] @ centroids.T
        for i, row in zip(left, sims):
            for c in np.argsort(-row):
                if counts[c] < cap:
                    labels[i] = c
                    counts[c] += 1
                    break
    return labels


def gini(sizes: np.ndarray) -> float:
    x = np.sort(np.asarray(sizes, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def partition_metrics(
    labels: np.ndarray, n_cells: int, cfg: KMosaicConfig,
    nearest: np.ndarray | None = None, dim: int | None = None,
) -> dict[str, Any]:

    sizes = np.bincount(labels, minlength=n_cells)
    N = int(sizes.sum())
    nonempty = sizes[sizes > 0]
    out = {
        "n_docs": N,
        "cells_nominal": int(n_cells),
        "cells_effective": int(len(nonempty)),
        "list_min": int(nonempty.min()) if len(nonempty) else 0,
        "list_median": float(np.median(nonempty)) if len(nonempty) else 0.0,
        "list_max": int(sizes.max()) if len(sizes) else 0,
        "max_share": float(sizes.max() / N) if N else float("nan"),
        "gini": gini(nonempty),
        "effective_security": float(N / sizes.max()) if N else float("nan"),
        "max_over_m": float(sizes.max() / cfg.m) if cfg.m else float("nan"),
        "m": cfg.m, "alpha": cfg.alpha, "branching": cfg.branching,
    }
    if nearest is not None:
        out["voronoi_agreement"] = float((nearest == labels).mean())
    if dim is not None:
        out["pair_budget_at_d"] = float(dim * N / sizes.max()) if N else float("nan")
    return out


_ROT_CACHE: dict[tuple[str, int, int, str], torch.Tensor] = {}

PRF_VERSION = 2


def _prf_generator(key: str, c: int) -> "np.random.Generator":
    digest = hmac.new(key.encode(), f"cell:{c}|v{PRF_VERSION}".encode(), hashlib.sha256).digest()
    return np.random.Generator(np.random.Philox(key=int.from_bytes(digest[:16], "little")))


class KMosaic:
    """A fitted partition plus its keys. ``protect`` stores ``R_c x`` in cell ``c``."""

    def __init__(self, cfg: KMosaicConfig | None = None, device: str | None = None):
        self.cfg = cfg or KMosaicConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.centroids: np.ndarray | None = None      # (C, d), plaintext space
        self.labels: np.ndarray | None = None         # bulk assignment of the fit set
        self._rot: dict[int, torch.Tensor] = {}
        self._cent_t: torch.Tensor | None = None


    def fit(self, Z: np.ndarray) -> "KMosaic":
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        if self.cfg.metric == "cosine":
            Z = _normalise(Z)
        leaves = fit_partition(Z, self.cfg)
        init = np.empty(len(Z), dtype=np.int64)
        for c, idx in enumerate(leaves):
            init[idx] = c
        cent = _centroids(Z, init, len(leaves), self.cfg.metric)
        cap = max(1, int(math.floor(self.cfg.alpha * self.cfg.m)))
        self.labels = capped_assign(Z, cent, cap, self.cfg.candidates)
        self.centroids = _centroids(Z, self.labels, len(leaves), self.cfg.metric)
        self._cent_t = None
        self._nearest = self.route(Z)
        return self

    @property
    def n_cells(self) -> int:
        return 0 if self.centroids is None else len(self.centroids)

    def report(self, dim: int | None = None) -> dict[str, Any]:
        if self.labels is None:
            raise RuntimeError("report() before fit()")
        return partition_metrics(self.labels, self.n_cells, self.cfg,
                                 nearest=getattr(self, "_nearest", None),
                                 dim=dim or self.centroids.shape[1])

    def _centroids_t(self) -> torch.Tensor:
        if self._cent_t is None:
            self._cent_t = torch.as_tensor(self.centroids, device=self.device)
        return self._cent_t

    def route(self, X: Any, nprobe: int = 1) -> np.ndarray:
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        out = []
        C = self._centroids_t()
        for s in range(0, len(Xt), 8192):
            out.append((Xt[s:s + 8192] @ C.T).topk(min(nprobe, len(C)), dim=1).indices.cpu())
        I = torch.cat(out).numpy() if out else np.zeros((0, nprobe), dtype=np.int64)
        return I[:, 0] if nprobe == 1 else I

    def rotation(self, c: int) -> torch.Tensor:
        c = int(c)
        if c not in self._rot:
            self.prepare_rotations([c])
        return self._rot[c]

    def prepare_rotations(self, cells: Any) -> None:
        d = self.centroids.shape[1]
        todo = []
        for c in {int(x) for x in np.asarray(cells).ravel()}:
            hit = _ROT_CACHE.get((self.cfg.key, d, c, str(self.device)))
            if hit is not None:
                self._rot[c] = hit
            elif c not in self._rot:
                todo.append(c)
        for s in range(0, len(todo), 64):
            chunk = todo[s:s + 64]
            A = torch.stack([
                torch.from_numpy(_prf_generator(self.cfg.key, c).standard_normal((d, d)))
                for c in chunk])
            Q, R = torch.linalg.qr(A.to(self.device))
            Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)  # Haar
            for c, q in zip(chunk, Q.to(torch.float32)):
                self._rot[c] = q
                _ROT_CACHE[(self.cfg.key, d, c, str(self.device))] = q

    def rotate(self, X: torch.Tensor, cells: np.ndarray, inverse: bool = False) -> torch.Tensor:
        out = torch.empty_like(X)
        cells_t = torch.as_tensor(cells, device=X.device)
        self.prepare_rotations(np.unique(cells))
        for c in np.unique(cells):
            rows = cells_t == int(c)
            R = self.rotation(int(c)).to(X.device, X.dtype)
            out[rows] = X[rows] @ (R if inverse else R.T)
        return out

    def protect(self, X: Any, cells: np.ndarray | None = None) -> Any:
        is_t = torch.is_tensor(X)
        Xt = X if is_t else torch.as_tensor(np.asarray(X, dtype=np.float32))
        dev = Xt.device
        Xt = Xt.to(self.device, torch.float32)
        cells = self.route(Xt.cpu().numpy()) if cells is None else np.asarray(cells)
        out = self.rotate(Xt, cells)
        return out.to(dev, X.dtype) if is_t else out.cpu().numpy()


    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "centroids": self.centroids,
                    "labels": self.labels}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "KMosaic":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(KMosaicConfig(**ck["cfg"]), device=device)
        obj.centroids, obj.labels = ck["centroids"], ck["labels"]
        return obj


def _selftest() -> bool:
    rng = np.random.default_rng(0)
    d = 64
    sizes = (rng.pareto(1.2, 40) * 200 + 20).astype(int)
    centers = rng.standard_normal((40, d)).astype(np.float32)
    Z = np.concatenate([c + 0.25 * rng.standard_normal((n, d)).astype(np.float32)
                        for c, n in zip(centers, sizes)])
    Z = _normalise(Z)
    cfg = KMosaicConfig(m=200, alpha=1.3, key="test")
    kr = KMosaic(cfg, device="cpu").fit(Z)
    rep = kr.report()
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    check("every row assigned", (kr.labels >= 0).all())
    check("no list above alpha*m", rep["list_max"] <= int(cfg.alpha * cfg.m),
          f"max {rep['list_max']} vs cap {int(cfg.alpha * cfg.m)}")
    check("partition is balanced", rep["gini"] < 0.25, f"gini {rep['gini']:.3f}")
    check("lists stay near m (C close to N/m)", rep["cells_effective"] < 2.5 * len(Z) / cfg.m,
          f"C={rep['cells_effective']} vs N/m={len(Z) / cfg.m:.0f}")
    check("most rows sit in their nearest cell", rep["voronoi_agreement"] > 0.8,
          f"{rep['voronoi_agreement']:.3f}")
    R = kr.rotation(3)
    check("R_c is orthogonal", torch.allclose(R @ R.T, torch.eye(d), atol=1e-5))
    check("R_c is a pure function of (key, c)",
          torch.allclose(R, KMosaic(cfg, "cpu")._with(kr).rotation(3)))
    check("different cells, different keys", not torch.allclose(kr.rotation(3), kr.rotation(4)))
    other = KMosaic(KMosaicConfig(m=200, key="other"), "cpu")._with(kr)
    check("different key, different R_c", not torch.allclose(other.rotation(3), R))
    X = torch.as_tensor(Z[:500])
    cells = kr.labels[:500]
    Y = kr.rotate(X, cells)
    same = cells[:, None] == cells[None, :]
    G, H = X @ X.T, Y @ Y.T
    check("inner products preserved within a cell", torch.allclose(G[same], H[same], atol=1e-5))
    check("not preserved across cells", not torch.allclose(G[~same], H[~same], atol=1e-2))
    check("rotation is invertible", torch.allclose(kr.rotate(Y, cells, inverse=True), X, atol=1e-5))
    print(f"  partition: {rep}")
    return ok


def _with(self: KMosaic, fitted: KMosaic) -> KMosaic:
    self.centroids, self.labels = fitted.centroids, fitted.labels
    return self


KMosaic._with = _with  


CALIB_PATH = Path(__file__).resolve().parents[1] / "metrics" / "outputs" / "cells_to_m.json"


def _calib_key(corpus: str, cells: int, alpha: float, branching: int) -> str:
    return f"{corpus}|C{int(cells)}|a{alpha:g}|b{int(branching)}"


def _leaf_count(Z: np.ndarray, cfg: KMosaicConfig, m: int) -> int:
    from dataclasses import replace

    return len(fit_partition(Z, replace(cfg, m=int(m))))


def solve_m_for_cells(
    Z: np.ndarray, target_cells: int, cfg: KMosaicConfig | None = None,
    *, tol: float = 0.05, max_iter: int = 16, verbose: bool = True,
) -> tuple[int, int]:
    cfg = cfg or KMosaicConfig()
    N = len(Z)
    target = int(target_cells)
    if not 1 <= target <= N:
        raise ValueError(f"target_cells must be in [1, {N:,}], got {target}")

    lo, hi = 1, max(2, N)                     # C(m=1) = N, C(m=N) = 1
    best: tuple[int, int] | None = None
    for it in range(max_iter):
        mid = (lo + hi) // 2
        c = _leaf_count(Z, cfg, mid)
        if verbose:
            print(f"[calib] iter {it + 1:2d}  m={mid:>9,}  ->  C={c:>9,}  (target {target:,})")
        if best is None or abs(c - target) < abs(best[1] - target):
            best = (mid, c)
        if abs(c - target) <= max(1, int(tol * target)):
            break
        if c > target:
            lo = mid + 1                      # too many cells -> raise occupancy
        else:
            hi = mid - 1
        if lo > hi:
            break
    assert best is not None
    return best


def m_for_cells(corpus: str, cells: int, alpha: float = 1.3, branching: int = 8) -> int:
    import json

    key = _calib_key(corpus, cells, alpha, branching)
    table = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else {}
    if key not in table:
        raise SystemExit(
            f"--kr-cells {cells}: no solved occupancy for {key!r} in {CALIB_PATH}.\n"
            f"Solve it once (minutes, one-time per C), then re-run:\n"
            f"  python -m defense.k_mosaic calibrate \\\n"
            f"      --vectors ANN/cache/<model>__{corpus}__n<N>__<hash>.npy \\\n"
            f"      --corpus {corpus} --cells {cells} "
            f"--alpha {alpha:g} --branching {branching}"
        )
    return int(table[key]["m"])


def _calibrate_cli(argv: list[str]) -> int:
    import argparse
    import fcntl
    import json
    import os

    ap = argparse.ArgumentParser(
        prog="python -m defense.k_mosaic calibrate",
        description="Solve the occupancy m whose partition has ~C cells, and record it "
                    "so --kr-cells C can use it. Read-only apart from the table it writes.")
    ap.add_argument("--vectors", required=True,
                    help="cached corpus embeddings, e.g. ANN/cache/gtr-base__quora__n*.npy")
    ap.add_argument("--cells", type=int, required=True, help="target cell count C")
    ap.add_argument("--corpus", required=True, help="corpus label, must match --attack-dataset")
    ap.add_argument("--alpha", type=float, default=1.3)
    ap.add_argument("--branching", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--metric", default="cosine", choices=["cosine", "ip"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=0.05, help="accept |C-target| <= tol*target")
    ap.add_argument("--full-fit", action="store_true",
                    help="after solving, run the capped assignment and print the partition "
                         "metrics the paper reports (effective C, max list, Gini)")
    ap.add_argument("--dry-run", action="store_true", help="solve and print, write nothing")
    a = ap.parse_args(argv)

    Z = np.load(a.vectors, mmap_mode="r")
    print(f"calib {a.vectors}: {Z.shape[0]:,} x {Z.shape[1]} -> target C={a.cells:,}")
    cfg = KMosaicConfig(alpha=a.alpha, branching=a.branching, candidates=a.candidates,
                              metric=a.metric, seed=a.seed)
    Zc = np.ascontiguousarray(Z, dtype=np.float32)
    m, c = solve_m_for_cells(Zc, a.cells, cfg, tol=a.tol)
    print(f"calib solved: m={m:,}  ->  C={c:,}  (target {a.cells:,}, "
          f"off by {100 * abs(c - a.cells) / a.cells:.1f}%)")

    record: dict[str, Any] = {"m": int(m), "cells_fit": int(c), "cells_target": int(a.cells),
                              "n_docs": int(Z.shape[0]), "dim": int(Z.shape[1]),
                              "vectors": os.path.basename(a.vectors)}
    if a.full_fit:
        from dataclasses import replace

        kr = KMosaic(replace(cfg, m=m)).fit(Zc)
        rep = kr.report()
        record["report"] = rep
        print("calib partition after capped assignment:")
        for k in ("cells_nominal", "cells_effective", "list_min", "list_median", "list_max",
                  "max_share", "gini", "effective_security", "voronoi_agreement"):
            if k in rep:
                print(f"           {k:>20}: {rep[k]}")

    if a.dry_run:
        print("[calib] --dry-run: table not written")
        return 0

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = _calib_key(a.corpus, a.cells, a.alpha, a.branching)

    with open(CALIB_PATH, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            text = fh.read()
            table = json.loads(text) if text.strip() else {}
            table[key] = record
            fh.seek(0)
            fh.truncate()
            json.dump(table, fh, indent=1, sort_keys=True)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"calib wrote {key!r} -> m={m:,} in {CALIB_PATH}")
    print(f"calib now run the ladder with:  --kr-cells {a.cells}   (or PART_CELLS={a.cells})")
    return 0


if __name__ == "__main__":  
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        sys.exit(_calibrate_cli(sys.argv[2:]))

    # for k, v in AMBIGUITIES.items():
    #     print(f"\n[{k}]")
    #     for field_name, text in v.items():
    #         print(f"  {field_name:>8}: {text}")
    print("\nselftest")
    sys.exit(0 if _selftest() else 1)
