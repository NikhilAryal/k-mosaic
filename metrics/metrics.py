from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class MetricSpec:

    key: str 
    label: str
    scale: float
    min_headroom: float
    notes: str = ""


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("token_f1", "Token F1", 1.0, 0.10,
               "bag-of-content recovery; the most robust of the four"),
    MetricSpec("token_recall", "Token recall", 1.0, 0.10,
               "TOKEN-level, not recall@k — see HEADER_WARNING"),
    MetricSpec("bleu", "BLEU", 100.0, 10.0,
               "sacrebleu 0-100; surface fidelity, degrades faster than token F1"),
    MetricSpec("rougeL", "ROUGE-L", 1.0, 0.10,
               "longest-common-subsequence overlap; surface-sensitive"),
)

HEADER_WARNING = (
    "token_recall is TOKEN-level bag-of-words overlap between the reconstruction and "
    "the original text. recall@k is RETRIEVAL utility, neighbour overlap in the "
    "index. They are unrelated; do not read one as the other."
)

CEILINGS = {
    "undefended": "the undefended attack: the protocol's own definition. Bounded "
                  "in practice by what the decoder can reconstruct at all.",
    "oracle": "the same decoder on the TRUE target embeddings, alignment bypassed "
              "(AttackResult.oracle_metrics). The decoder ceiling, stated directly.",
}


@dataclass
class LeakNorm:
    metric: str
    defended: float
    ceiling: float
    floor: float
    ceiling_arm: str = "undefended"
    value: float = float("nan")
    headroom: float = float("nan")
    min_headroom: float = float("nan")
    valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "leak_norm": self.value,
            "m_defended": self.defended,
            "m_ceiling": self.ceiling,
            "m_floor": self.floor,
            "ceiling_arm": self.ceiling_arm,
            "headroom": self.headroom,
            "min_headroom": self.min_headroom,
            "valid": self.valid,
        }


def leak_norm(
    defended: float,
    ceiling: float,
    floor: float,
    *,
    metric: str = "",
    min_headroom: float = 0.0,
    ceiling_arm: str = "undefended",
) -> LeakNorm:
    headroom = ceiling - floor
    out = LeakNorm(
        metric=metric,
        defended=defended,
        ceiling=ceiling,
        floor=floor,
        ceiling_arm=ceiling_arm,
        headroom=headroom,
        min_headroom=min_headroom,
        valid=headroom >= min_headroom and headroom > 0,
    )
    out.value = (defended - floor) / headroom if headroom > 0 else float("nan")
    return out


def arm_metrics(rows: Sequence[dict], section: str = "test_results") -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in METRICS:
        vals = [float(r[section][spec.key]) for r in rows if spec.key in r.get(section, {})]
        out[spec.key] = mean(vals) if vals else float("nan")
    return out


def floor_metrics(*floors: Sequence[dict]) -> dict[str, float]:
    per_arm = [arm_metrics(rows) for rows in floors if rows]
    if not per_arm:
        raise ValueError("floor_metrics needs at least one floor arm")
    return {
        spec.key: max(
            (a[spec.key] for a in per_arm if a[spec.key] == a[spec.key]),  # drop NaN
            default=float("nan"),
        )
        for spec in METRICS
    }


def gate_for(spec: MetricSpec, min_headroom: float | None) -> float:
    return spec.min_headroom if min_headroom is None else min_headroom * spec.scale


def leak_norms_for_arm(
    defended: Sequence[dict],
    ceiling: dict[str, float],
    floor: dict[str, float],
    *,
    ceiling_arm: str = "undefended",
    min_headroom: float | None = None,
) -> dict[str, LeakNorm]:
    d = arm_metrics(defended)
    return {
        spec.key: leak_norm(
            d[spec.key], ceiling[spec.key], floor[spec.key],
            metric=spec.key, min_headroom=gate_for(spec, min_headroom),
            ceiling_arm=ceiling_arm,
        )
        for spec in METRICS
    }


OUTCOMES = {
    "no_signal": "the ceiling is too close to the floor; this run says nothing about "
                 "the defense. Bottleneck is the corpus or the attacker, not the defense.",
    "thesis_holds": "leak_norm 0.85-1.0: the defense costs utility and buys almost "
                    "nothing against alignment attacks.",
    "heterogeneity": "leak_norm 0.3-0.6: the defense genuinely impedes alignment, and "
                     "reconstruction-attack rankings did not predict that it would.",
    "protection_is_destruction": "leak_norm near 0, but only at a setting where "
                                 "recall@k has collapsed. Not protection = destruction "
                                 "of the embedding space.",
    "unclassified": "outside the spec's bands. Report the value; do not round it into "
                    "an adjacent verdict.",
}

DEFAULT_UTILITY_FLOOR = 0.5


def classify(
    norms: dict[str, LeakNorm],
    *,
    headline: str = "token_f1",
    recall_at_k: float | None = None,
    utility_floor: float = DEFAULT_UTILITY_FLOOR,
) -> tuple[str, str]:
    ln = norms.get(headline)
    if ln is None:
        return "unclassified", f"no {headline} in this arm"
    if not ln.valid:
        return (
            "no_signal",
            f"headroom {ln.headroom:.3f} < {ln.min_headroom:.3f} on {headline} "
            f"(ceiling {ln.ceiling:.3f} vs floor {ln.floor:.3f})",
        )

    v = ln.value
    if v <= 0.15:
        if recall_at_k is None:
            return (
                "unclassified",
                f"{headline} leak_norm={v:.2f} is near zero, but no recall@k was "
                f"supplied — cannot separate protection from destruction",
            )
        if recall_at_k < utility_floor:
            return (
                "protection_is_destruction",
                f"{headline} leak_norm={v:.2f} with recall@k={recall_at_k:.3f} "
                f"< {utility_floor:.2f}: the space is destroyed, not protected",
            )
        return (
            "heterogeneity",
            f"{headline} leak_norm={v:.2f} at recall@k={recall_at_k:.3f}: the defense "
            f"removed nearly all available leakage while keeping utility",
        )
    if 0.85 <= v <= 1.0:
        return "thesis_holds", f"{headline} leak_norm={v:.2f}: leakage essentially intact"
    if 0.3 <= v <= 0.6:
        return "heterogeneity", f"{headline} leak_norm={v:.2f}: partial, real protection"
    return (
        "unclassified",
        f"{headline} leak_norm={v:.2f} falls in a gap between the spec's bands",
    )


def classify_ladder(
    per_defense: dict[str, dict[str, LeakNorm]],
    *,
    headline: str = "token_f1",
    spread_threshold: float = 0.25,
    utility: dict[str, float] | None = None,
) -> tuple[str, str]:
    valid = {
        name: n[headline].value
        for name, n in per_defense.items()
        if headline in n and n[headline].valid
    }
    if not valid:
        return "no_signal", "no defended arm has usable headroom on " + headline
    if len(valid) == 1:
        name, v = next(iter(valid.items()))
        return classify({headline: per_defense[name][headline]}, headline=headline,
                        recall_at_k=(utility or {}).get(name))

    spread = max(valid.values()) - min(valid.values())
    if spread > spread_threshold:
        lo = min(valid, key=valid.get)
        hi = max(valid, key=valid.get)
        return (
            "heterogeneity",
            f"{headline} leak_norm spans {spread:.2f} across defenses "
            f"({lo}={valid[lo]:.2f} .. {hi}={valid[hi]:.2f}): protection is "
            f"heterogeneous and not predicted by reconstruction rankings",
        )
    if all(v >= 0.85 for v in valid.values()):
        return (
            "thesis_holds",
            f"every defense sits at {headline} leak_norm >= 0.85 "
            f"({', '.join(f'{k}={v:.2f}' for k, v in valid.items())})",
        )
    return (
        "unclassified",
        f"{headline} leak_norm clustered at "
        f"{', '.join(f'{k}={v:.2f}' for k, v in valid.items())} — outside the bands",
    )


@dataclass
class Ladder:
    floor: dict[str, float]
    ceiling: dict[str, float]
    ceiling_arm: str
    raw: dict[str, dict[str, float]] = field(default_factory=dict)   
    norms: dict[str, dict[str, LeakNorm]] = field(default_factory=dict)
    utility: dict[str, float] = field(default_factory=dict)          
    ndcg: dict[str, dict[str, float]] = field(default_factory=dict) 
    oracle: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ceiling_arm": self.ceiling_arm,
            "ceiling_arm_note": CEILINGS.get(self.ceiling_arm, ""),
            "floor": self.floor,
            "ceiling": self.ceiling,
            "oracle": self.oracle,
            "raw": self.raw,
            "utility_recall_at_k": self.utility,
            "utility_ndcg": self.ndcg,
            "leak_norm": {
                arm: {k: v.to_dict() for k, v in n.items()} for arm, n in self.norms.items()
            },
            "verdict": dict(zip(("outcome", "reason"),
                                classify_ladder(self.norms, utility=self.utility))),
        }


def build_ladder(
    arms: dict[str, Sequence[dict]],
    *,
    floor_arms: Iterable[str] = ("floor_prior_only", "floor_random_vector"),
    ceiling_arm: str = "undefended",
    utility: dict[str, float] | None = None,
    ndcg: dict[str, dict[str, float]] | None = None,
    min_headroom: float | None = None,
) -> Ladder:
    floors = [arms[name] for name in floor_arms if name in arms]
    if not floors:
        raise ValueError(
            f"no floor arm found among {list(arms)}; the protocol needs at least one "
            f"of {list(floor_arms)}. Without a floor, leak_norm is undefined,run "
            f"`python -m metrics.ladder` rather than computing deltas by hand."
        )
    if ceiling_arm not in arms:
        raise ValueError(f"ceiling arm {ceiling_arm!r} not in {list(arms)}")

    floor = floor_metrics(*floors)
    ceiling = arm_metrics(arms[ceiling_arm])
    oracle = arm_metrics(arms[ceiling_arm], section="oracle_results")

    treatment = [a for a in arms if a not in set(floor_arms) and a != ceiling_arm]
    return Ladder(
        floor=floor,
        ceiling=ceiling,
        ceiling_arm=ceiling_arm,
        oracle=oracle,
        raw={a: arm_metrics(rows) for a, rows in arms.items()},
        norms={
            a: leak_norms_for_arm(arms[a], ceiling, floor, ceiling_arm=ceiling_arm,
                                  min_headroom=min_headroom)
            for a in treatment
        },
        utility=dict(utility or {}),
        ndcg=dict(ndcg or {}),
    )


def print_ladder(
    ladder: Ladder, *, headline: str = "token_f1", utility_label: str = "recall@k",
    min_headroom: float | None = None
) -> None:
    keys = [s.key for s in METRICS]
    order = ["floor_prior_only", "floor_random_vector"]
    order += [a for a in ladder.raw if a not in order]

    print(f"\n{'=' * 100}")
    print(" raw metrics — floors first, then the ceiling, then the treatments")
    print(f"{'=' * 100}")
    print(f"  {'arm':<34} " + " ".join(f"{s.label:>13}" for s in METRICS))
    for arm in order:
        if arm not in ladder.raw:
            continue
        m = ladder.raw[arm]
        marker = "  <- ceiling" if arm == ladder.ceiling_arm else ""
        print(f"  {arm[:34]:<34} " + " ".join(f"{m[k]:>13.4f}" for k in keys) + marker)
    print(f"  {'FLOOR (max of the two)':<34} " + " ".join(f"{ladder.floor[k]:>13.4f}" for k in keys))
    if any(v == v for v in ladder.oracle.values()):
        print(f"  {'oracle (decoder ceiling)':<34} "
              + " ".join(f"{ladder.oracle.get(k, float('nan')):>13.4f}" for k in keys))

    print(f"\n{'=' * 100}")
    print(f"leak_norm: (m_defended - m_floor) / (m_{ladder.ceiling_arm} - m_floor) "
          f"— 1.0 = defense removed nothing, 0.0 = attack is at its prior")
    print(f"{'=' * 100}")
    has_ndcg = bool(ladder.ndcg)
    ndcg_head = f" {'NDCG@10':>9}" if has_ndcg else ""
    print(f"  {'arm':<34} " + " ".join(f"{s.label:>13}" for s in METRICS)
          + f" {utility_label:>10}{ndcg_head}  verdict")
    for arm, norms in ladder.norms.items():
        cells = []
        for k in keys:
            n = norms[k]
            cells.append(f"{n.value:>13.3f}" if n.valid else f"{'no signal':>13}")
        rk = ladder.utility.get(arm, float("nan"))
        outcome, _ = classify(norms, headline=headline, recall_at_k=ladder.utility.get(arm))
        nd = ""
        if has_ndcg:
            r = ladder.ndcg.get(arm, {})
            key = next((k for k in r if k.startswith("ndcg_at_") and not k.endswith("undefended")), None)
            nd = f" {r[key]:>9.4f}" if key else f" {'—':>9}"
        print(f"  {arm[:34]:<34} " + " ".join(cells) + f" {rk:>10.3f}{nd}  {outcome}")

    if ladder.ndcg:
        base = next((v.get(f"ndcg_at_{v.get('k', 10)}_undefended")
                     or v.get("ndcg_at_10_undefended") for v in ladder.ndcg.values()), None)
        print(
            f"\n  recall@k and NDCG@10 answer different questions. recall@k is neighbour\n"
            f"  PRESERVATION against the undefended index ."
            + (f", undefended baseline {base:.4f}" if base else "")
            + ".\n  Where they disagree, that disagreement is the result: a defense can reshuffle\n"
            f"  neighbours without moving relevance, or preserve them while wrecking the ranking."
        )
    print(f"\n  {HEADER_WARNING}")
    if min_headroom is not None:
        print(f"\n headroom threshold OVERRIDDEN to {min_headroom:g} "
              f"(spec default 0.10 on a 0-1 metric). Every leak_norm below is gated on "
              f"that, not on the protocol's value , report it with the number.")
    for spec in METRICS:
        head = ladder.ceiling[spec.key] - ladder.floor[spec.key]
        gate = gate_for(spec, min_headroom)
        if head < gate:
            print(
                f"no signal {spec.label}: ceiling - floor = {head:.4f} < "
                f"{gate:.4f}. That column says nothing about any defense."
            )
    outcome, reason = classify_ladder(ladder.norms, headline=headline,
                                      utility=ladder.utility)
    print(f"\n verdict: {outcome}: {reason}")
    print(f"            {OUTCOMES[outcome]}")
