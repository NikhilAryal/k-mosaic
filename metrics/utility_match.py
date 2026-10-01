from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from attacker.algen.defenses import DEFENSES, apply_defense


@dataclass(frozen=True)
class KnobSpec:
    defense: str
    knob: str                       
    lo: float
    hi: float
    note: str = ""

    def kwargs(self, value: float) -> dict[str, Any]:
        if self.defense == "vec2text":
            return {"vec2text_noise_level": value}
        if self.defense == "remote_rag":
            return {"remote_rag_radius": value}
        if self.defense == "gaussian":
            return {"noise_level": value}
        return {"epsilon": value}


KNOBS: dict[str, KnobSpec] = {
    "sparse": KnobSpec("sparse", "epsilon", 1.0, 1e4,
                       "higher eps = less noise. With --sparse-epsilon-scale per_dim "
                       "the usable range on unit-norm vectors is ~50-2000"),
    "cmag": KnobSpec("cmag", "epsilon", 0.1, 1e3,
                     "higher eps = less noise; the paper sweeps 1.6-40"),
    "vec2text": KnobSpec("vec2text", "lambda", 1e-4, 2.0,
                         "higher lambda = MORE noise; Table 7's knee is 0.01"),
    "remote_rag": KnobSpec("remote_rag", "r", 1e-3, 5.0,
                           "higher r = MORE noise; Table 6 sweeps 0.03-0.1"),
    "lapmech": KnobSpec("lapmech", "epsilon", 1.0, 1e5,
                        "Gamma(d, eps) radius, so E|Z| ~ d/eps at d=768"),
    "purmech": KnobSpec("purmech", "epsilon", 0.1, 1e3,
                        "higher eps = smaller rotation angle"),
    "dp_gaussian": KnobSpec("dp_gaussian", "epsilon", 1.0, 1e5,
                            "sigma = sqrt(2 ln(1.25/delta)) * Delta / eps"),
    "gaussian": KnobSpec("gaussian", "sigma", 1e-4, 10.0,
                         "higher sigma = MORE noise; no privacy guarantee"),
}

NO_KNOB: dict[str, str] = {
    "eguard": "alpha is a training hyperparameter of g_p; matching would mean "
              "retraining the projection network per candidate value. Fit a ladder "
              "of guards at several alphas and pick by measured recall instead.",
    "idct": "the knob is an integer subset count (2, 3, 4...), so utility takes a "
            "handful of discrete values and cannot be solved to a target.",
    "shuffling": "a permutation has no magnitude.",
    "wet": "the circulant transform has no magnitude parameter.",
    "masking": "overwrites one coordinate; nothing to tune.",
    "k_mosaic": "structural: the rotation costs no recall by construction, and the "
                      "security parameter is the cell count, set by --kr-m. Its utility "
                      "cost is the partition's, reported at --kr-nprobe.",
    "none": "no defense, no knob.",
}



def recall_at_k(original: torch.Tensor, protected: torch.Tensor, k: int = 10) -> float:
    a = torch.nn.functional.normalize(original.float(), dim=-1)
    b = torch.nn.functional.normalize(protected.float(), dim=-1)
    n = a.shape[0]
    kk = min(k, n - 1)
    eye = torch.eye(n, device=a.device, dtype=torch.bool)
    neg_inf = torch.finfo(a.dtype).min
    top_a = (a @ a.t()).masked_fill(eye, neg_inf).topk(kk, dim=-1).indices
    top_b = (b @ b.t()).masked_fill(eye, neg_inf).topk(kk, dim=-1).indices
    return float(
        np.mean([len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / kk for i in range(n)])
    )


def measure_utility(
    X: torch.Tensor,
    defense: str,
    value: float,
    *,
    k: int = 10,
    repeats: int = 3,
    seed: int = 42,
    extra: dict[str, Any] | None = None,
) -> tuple[float, float]:
    spec = KNOBS[defense]
    scores = []
    for i in range(max(1, repeats)):
        torch.manual_seed(seed + i)
        np.random.seed(seed + i)
        try:
            protected, _ = apply_defense(X, defense, **{**(extra or {}), **spec.kwargs(value)})
        except (ZeroDivisionError, ValueError, FloatingPointError) as exc:
            print(f"  [undefined] {defense} at {spec.knob}={value:g}: "
                  f"{type(exc).__name__}: {exc}")
            return float("nan"), float("nan")
        scores.append(recall_at_k(X, protected, k=k))
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
    return mean, se


@dataclass
class MatchResult:

    defense: str
    knob: str
    target: float
    value: float = float("nan")
    utility: float = float("nan")
    utility_se: float = float("nan")
    converged: bool = False
    reason: str = ""
    direction: str = ""             
    bracket: tuple[float, float] = (float("nan"), float("nan"))
    endpoint_utility: tuple[float, float] = (float("nan"), float("nan"))
    plateau_width: float = float("nan")
    iterations: int = 0
    trace: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "defense": self.defense, "knob": self.knob, "target_recall": self.target,
            "matched_value": self.value, "measured_recall": self.utility,
            "recall_se": self.utility_se, "converged": self.converged,
            "reason": self.reason, "direction": self.direction,
            "bracket": list(self.bracket), "endpoint_utility": list(self.endpoint_utility),
            "plateau_width_decades": self.plateau_width, "iterations": self.iterations,
            "trace": [list(t) for t in self.trace],
        }


def match_utility(
    X: torch.Tensor,
    defense: str,
    target: float,
    *,
    k: int = 10,
    repeats: int = 3,
    seed: int = 42,
    tolerance: float = 0.01,
    max_iter: int = 20,
    extra: dict[str, Any] | None = None,
    verbose: bool = True,
    probe_fn: "Callable[[float], tuple[float, float]] | None" = None,
    verify_fn: "Callable[[float], tuple[float, float]] | None" = None,
    metric_label: str = "recall@k",
) -> MatchResult:

    if defense in NO_KNOB:
        return MatchResult(defense, "-", target, reason=f"no continuous knob: {NO_KNOB[defense]}")
    if defense not in KNOBS:
        return MatchResult(defense, "-", target,
                           reason=f"unknown defense {defense!r}; known: {sorted(KNOBS)}")

    spec = KNOBS[defense]
    out = MatchResult(defense, spec.knob, target)

    probe = probe_fn or (
        lambda v: measure_utility(X, defense, v, k=k, repeats=repeats, seed=seed, extra=extra)
    )

    lo, hi = spec.lo, spec.hi
    u_lo, _ = probe(lo)
    u_hi, _ = probe(hi)
    if math.isnan(u_lo) or math.isnan(u_hi):
        out.bracket = (lo, hi)
        out.endpoint_utility = (u_lo, u_hi)
        out.reason = (
            f"{defense} is numerically undefined at one end of its bracket "
            f"({spec.knob} in [{lo:g}, {hi:g}] gave recall [{u_lo}, {u_hi}]). The "
            f"mechanism cannot be evaluated across the range a match would need, so "
        )
        return out
    if verbose:
        print(f"Match: {defense}: probing bracket {spec.knob}=[{lo:g}, {hi:g}] "
              f"-> {metric_label} [{u_lo:.4f}, {u_hi:.4f}]")

    if not (min(u_lo, u_hi) <= target <= max(u_lo, u_hi)):
        if (u_hi > u_lo) == (target > max(u_lo, u_hi)):
            hi *= 100.0
        else:
            lo /= 100.0
        u_lo, _ = probe(lo)
        u_hi, _ = probe(hi)
        if verbose:
            print(f"Match: widened to [{lo:g}, {hi:g}] -> {metric_label} [{u_lo:.4f}, {u_hi:.4f}]")

    out.bracket = (lo, hi)
    out.endpoint_utility = (u_lo, u_hi)
    out.direction = "increasing" if u_hi > u_lo else "decreasing"

    if not (min(u_lo, u_hi) <= target <= max(u_lo, u_hi)):
        out.reason = (
            f"target {metric_label}={target:.3f} is outside what {defense} can reach: the "
            f"widened bracket {spec.knob} in [{lo:g}, {hi:g}] spans recall "
            f"[{min(u_lo, u_hi):.4f}, {max(u_lo, u_hi):.4f}]. Pick a target inside "
            f"that range, or widen KNOBS[{defense!r}]."
        )
        return out

    lo_l, hi_l = math.log10(lo), math.log10(hi)
    increasing = u_hi > u_lo
    value = u = se = float("nan")
    for i in range(max_iter):
        mid_l = (lo_l + hi_l) / 2
        value = 10 ** mid_l
        u, se = probe(value)
        out.trace.append((value, u))
        out.iterations = i + 1
        if verbose:
            print(f"  iter {i + 1:>2}  {spec.knob}={value:<12.5g} {metric_label}={u:.4f} "
                  f"(+-{se:.4f})  target={target:.4f}")
        if abs(u - target) <= tolerance:
            break
        if (u < target) == increasing:
            lo_l = mid_l
        else:
            hi_l = mid_l

    out.value, out.utility, out.utility_se = value, u, se
    if probe_fn is not None:
        check_u, check_se = (verify_fn or probe_fn)(value)
    else:
        check_u, check_se = measure_utility(
            X, defense, value, k=k, repeats=max(repeats, 5), seed=seed + 1000, extra=extra
        )
    out.utility, out.utility_se = check_u, check_se
    out.converged = abs(check_u - target) <= tolerance
    if not out.converged:
        out.reason = (
            f"bisection stopped at {metric_label}={check_u:.4f} on an independent "
            f"re-measure, {abs(check_u - target):.4f} from target {target:.3f} "
            f"(tolerance {tolerance:.3f}). Either the curve is not monotone here or "
            f"{max_iter} iterations were not enough. Do NOT report this as matched."
        )
    if check_se > tolerance / 2:
        out.reason += (
            f" [noise] {metric_label} has SE {check_se:.4f} against a tolerance of "
            f"{tolerance:.3f}; raise --repeats or loosen --tolerance."
        )

    out.plateau_width = _plateau_width(probe, value, target, tolerance)
    return out


def _plateau_width(
    probe: Callable[[float], tuple[float, float]],
    value: float,
    target: float,
    tolerance: float,
) -> float:
    lo = hi = math.log10(value)
    for step in (0.25, 0.5, 1.0, 2.0):
        if abs(probe(10 ** (math.log10(value) - step))[0] - target) <= tolerance:
            lo = math.log10(value) - step
        else:
            break
    for step in (0.25, 0.5, 1.0, 2.0):
        if abs(probe(10 ** (math.log10(value) + step))[0] - target) <= tolerance:
            hi = math.log10(value) + step
        else:
            break
    return hi - lo


def eval_embeddings_for(args: argparse.Namespace) -> torch.Tensor:
    import train_algen as algen_pipeline
    from attacker.data import build_splits, collect_target_ids, find_embedding_sets, victim_embedder
    from models import EmbeddingSet

    paths = algen_pipeline.victim_paths(args)
    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    holdout = collect_target_ids(find_embedding_sets("all"), args.attack_dataset)
    splits = build_splits(
        args.attack_dataset, holdout_ids=holdout,
        n_train=0, n_val=args.match_samples, n_align=0, seed=args.seed,
    )
    print(f"Match: utility slice: {len(splits.val)} held-out {args.attack_dataset} docs "
          f"through {embedder}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(
        embedder.encode(splits.val, show_progress=True), dtype=torch.float32, device=device
    )


def add_match_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("utility matching")
    g.add_argument("--target-recall", type=float, default=None,
                   help="recall@k every defense is tuned to. The protocol's "
                        "utility-matched arm; without it, cross-defense comparisons "
                        "are comparisons of noise budgets, not of mechanisms")
    g.add_argument("--match-k", type=int, default=10, help="k in recall@k")
    g.add_argument("--match-samples", type=int, default=500,
                   help="held-out documents the utility is measured on")
    g.add_argument("--match-repeats", type=int, default=3,
                   help="draws averaged per candidate DURING bisection; most mechanisms "
                        "are stochastic. Each draw is a full index build, so this is the "
                        "main cost knob of the match stage")
    g.add_argument("--match-verify-repeats", type=int, default=None,
                   help="draws for the independent re-measure that decides `converged` "
                        "(default: max(--match-repeats, 3)). Averaging the VERIFY step "
                        "rather than every bisection probe keeps the reported operating "
                        "point as well determined while halving the builds")
    g.add_argument("--match-tolerance", type=float, default=0.01,
                   help="how close to --target-recall counts as matched")
    g.add_argument("--match-max-iter", type=int, default=20)
    g.add_argument("--match-metric", default="ann", choices=["ann", "recall"],
                   help="the utility the knob is solved against and that the ladder "
                        "reports. 'ann' (the traced path) = recall@k under a faiss "
                        "index over the whole corpus (see --ann-*), and the attack "
                        "then inverts the vectors as the index stores them; 'recall' = "
                        "brute-force recall@k on a 500-doc slice, kept only as the "
                        "bisection's fallback probe")
    from ANN import add_ann_args

    add_ann_args(p)


def build_parser() -> argparse.ArgumentParser:
    import train_algen as algen_pipeline

    p = algen_pipeline.build_parser()
    p.prog = "python -m metrics.utility_match"
    p.description = "Solve a defense's privacy knob for a fixed retrieval-utility target."
    add_match_args(p)
    p.add_argument("--assumptions", action="store_true",
                   help="print what this solver takes on faith, and exit")
    p.add_argument("--match-out", default="metrics/outputs/matched.json")
    return p


def _verify_repeats(args: argparse.Namespace) -> int:
    return int(args.match_verify_repeats or max(args.match_repeats, 3))


def _metric_label(args: argparse.Namespace) -> str:
    return {"ann": f"ANN recall@{args.match_k}"}.get(args.match_metric,
                                                     f"recall@{args.match_k}")


def main(argv: list[str] | None = None) -> int:
    from main import load_config

    from .ladder import CONFIG_SECTIONS

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        parser.set_defaults(**load_config(
            args.config, parser=parser,
            sections=CONFIG_SECTIONS,
            section_name_key={"dataset": "dataset", "model": "model", "defense": "defense"},
        ))
        args = parser.parse_args(argv)

    if args.defense not in DEFENSES:
        raise SystemExit(f"unknown defense {args.defense!r}; known: {sorted(DEFENSES)}")
    if args.target_recall is None:
        raise SystemExit("--target-recall is required (e.g. 0.90)")

    from metrics.ladder import _defense_kwargs

    extra = {k: v for k, v in _defense_kwargs(args, args.defense).items()
             if k not in ("defense", "epsilon", "noise_level", "vec2text_noise_level",
                          "remote_rag_radius")}
    probe = verify = None
    X = None
    ann_info: dict[str, Any] = {}
    if args.match_metric == "ann":
        from ANN import AnnProbe, probe_tag

        ann = AnnProbe.from_args(args)

        base = ann.clean_baseline()["index_recall"]
        if base < args.target_recall - args.match_tolerance:
            raise SystemExit(
                f"the undefended index only reaches ANN recall@{args.match_k}={base:.4f} "
                f"at {ann.spec.budget_label}{ann.spec.budget}, below the target "
                f"{args.target_recall:.3f}. Raise --ann-budget or lower the target."
            )
        if args.defense in KNOBS:
            probe = ann.probe_fn(args.defense, extra, KNOBS[args.defense].kwargs)
            verify = ann.probe_fn(args.defense, extra, KNOBS[args.defense].kwargs,
                                  repeats=_verify_repeats(args), seed=args.seed + 1000)
        ann_info = {"ann_probe": probe_tag(args), "ann_index_recall": base,
                    "ann_docs": ann.corpus.n}
    else:
        X = eval_embeddings_for(args)
    result = match_utility(
        X, args.defense, args.target_recall,
        k=args.match_k, repeats=args.match_repeats, seed=args.seed,
        tolerance=args.match_tolerance, max_iter=args.match_max_iter, extra=extra,
        probe_fn=probe, verify_fn=verify,
        metric_label=_metric_label(args),
    )

    print(f"\n{'=' * 84}")
    if result.knob == "-" or math.isnan(result.value):

        print(f"[not matchable] {result.defense}")
        print(f"{'=' * 84}")
        print(f"  {result.reason}")
        key = "no_continuous_knob" if result.knob == "-" else "numerically_undefined_ranges"
        print(f"\n  See ASSUMPTIONS[{key!r}] "
              f"(python -m metrics.utility_match --assumptions).")
        return 1
    print(f"[matched] {result.defense}: {result.knob} = "
          f"{result.value:.6g}  ->  {_metric_label(args)} = {result.utility:.4f} "
          f"(+-{result.utility_se:.4f}), target {result.target:.3f}")
    print(f"{'=' * 84}")
    print(f"  converged      {result.converged}")
    print(f"  direction      {_metric_label(args)} is {result.direction} in {result.knob}")
    print(f"  bracket        [{result.bracket[0]:g}, {result.bracket[1]:g}] "
          f"-> {_metric_label(args)} "
          f"[{result.endpoint_utility[0]:.4f}, {result.endpoint_utility[1]:.4f}]")
    print(f"  plateau        {result.plateau_width:.2f} decades of {result.knob} "
          f"satisfy the target")
    if result.plateau_width >= 1.0:
        print(f"^ under-determined: the matched value was chosen by the\n"
              f"  bisection path, not by the constraint. See\n")
    if result.reason:
        print(f"  NOTE           {result.reason}")

    out = Path(args.match_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.match_metric == "ann":
        from ANN import bucket_key

        bucket = bucket_key(args, args.target_recall)
    else:
        bucket = f"{args.match_metric}@{args.match_k}={args.target_recall:g}"
    entry = {
        **result.to_dict(),
        "k": args.match_k, "samples": args.match_samples, "seed": args.seed, **ann_info,
    }

    with open(out, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            text = fh.read()
            payload = json.loads(text) if text.strip() else {}
            payload.setdefault(str(args.attack_dataset), {}).setdefault(
                bucket, {})[args.defense] = entry
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(payload, indent=2))
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    print(f"\n Out: {out}")
    return 0 if result.converged else 1


if __name__ == "__main__":
    sys.exit(main())
