from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attacker import get_attack
from attacker.data import find_embedding_sets
from attacker.utils import set_seed

from .metrics import build_ladder, print_ladder


CONFIG_SECTIONS = ("pretrain", "finetune", "eguard", "sparse", "cmag", "vec2text",
                   "remote_rag", "defense", "attack", "ladder", "steer", "teia",
                   "zero2text")
from .utility_match import add_match_args, eval_embeddings_for, match_utility

FLOOR_ARMS = {"floor_prior_only": "floor_prior", "floor_random_vector": "floor_random"}

FLOORS_FOR = {
    "algen": FLOOR_ARMS,
    "steer": FLOOR_ARMS,
    "teia": {"floor_prior_only": "teia_floor_prior",
             "floor_random_vector": "teia_floor_random"},

    "zero2text": {"floor_prior_only": "zero2text_floor_prior",
                  "floor_random_vector": "zero2text_floor_random"},
}

_ALGEN_ONLY = ("align_samples", "reg_lambda", "align_text", "attention_mask")


def build_parser() -> argparse.ArgumentParser:
    import train_algen as algen_pipeline

    p = algen_pipeline.build_parser()
    p.prog = "python -m metrics.ladder"
    p.description = (
        "Run the floor / ceiling / treatment ladder and report normalised leakage."
    )
    g = p.add_argument_group("ladder")
    g.add_argument("--checkpoint", default=None,
                   help="ALGEN run directory. Default: the finetune stage implied by "
                        "--config, i.e. the generator train_algen.py would attack with")
    g.add_argument("--floor-prior", default="mean", choices=["mean", "zero"],
                   help="how floor_prior builds its constant vector")
    g.add_argument("--floor-source", default="shuffled", choices=["shuffled", "random"],
                   help="how floor_random destroys the pair correspondence")
    g.add_argument("--headline", default="token_f1",
                   help="metric the outcome taxonomy is decided on. The spec's choice "
                        "is token_f1: the most robust of the four, so the least likely "
                        "to hit its floor for reasons unrelated to the defense")
    g.add_argument("--recall-at-k", type=float, default=None,
                   help="the defended arm's retrieval utility. Without it, an arm at ")
    g.add_argument("--attack", default="algen",
                   choices=["algen", "steer", "teia", "zero2text"],
                   help="which threat model runs the treatment and ceiling arms. ")
    g.add_argument("--defense-scope", default="targets", choices=["targets", "both"],
                   help="steer only: where the defense is applied. 'targets' is "
                        "STEER's model; 'both' reproduces ALGEN's.")
    g.add_argument("--min-headroom", type=float, default=None,
                   help="OVERRIDE the headroom gate, on the 0-1 metric scale")
    g.add_argument("--ladder-out", default="metrics/outputs",
                   help="where cached arms and the ladder JSON are written")
    g.add_argument("--algen-cell-maps", action="store_true",
                   help="algen + --partition only: fit one ridge map PER CELL on the ")
    g.add_argument("--algen-cell-min-pairs", type=int, default=1,
                   help="with --algen-cell-maps: fewest pairs a cell needs for its own ")
    from attacker.zero2text.cli import add_attack_args as _add_z2t_args

    _add_z2t_args(p)

    add_match_args(p)
    p.set_defaults(align_samples=200, reg_lambda=0.1)
    return p


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    from main import load_config

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        parser.set_defaults(
            **load_config(
                args.config, parser=parser,
                sections=CONFIG_SECTIONS,
                section_name_key={"dataset": "dataset", "model": "model",
                                  "defense": "defense"},
            )
        )
        args = parser.parse_args(argv)
    return args


def parser_default_ladder_out() -> str:
    return "metrics/outputs"


def knob_for(args: argparse.Namespace, defense: str) -> float:
    import train_algen as algen_pipeline

    sweep = {
        "sparse": algen_pipeline._sparse_epsilons,
        "cmag": algen_pipeline._cmag_epsilons,
        "vec2text": algen_pipeline._vec2text_noise_levels,
        "remote_rag": algen_pipeline._remote_rag_radii,
    }.get(defense)
    if sweep is None:
        return float(args.epsilon)
    values = sweep(args)
    if len(values) > 1:
        print(
            f"warn: a {defense} sweep of {[f'{v:g}' for v in values]} was given, but the "
            f"ladder runs ONE treatment arm — using {values[0]:g}.\n"
            f"       For the whole curve use train_algen.py, which runs an arm per value."
        )
    return float(values[0])


def _attack_kwargs(args: argparse.Namespace, defense: str) -> dict[str, Any]:
    kw = dict(_defense_kwargs(args, defense))
    attack = getattr(args, "attack", "algen")
    if attack in ("steer", "teia", "zero2text"):
        kw["defense_scope"] = args.defense_scope
    if attack == "algen" and getattr(args, "algen_cell_maps", False):
        kw["cell_maps"] = True
        kw["cell_min_pairs"] = args.algen_cell_min_pairs
    return kw


def arm_label(args: argparse.Namespace, defense: str) -> str:
    import train_algen as algen_pipeline

    if defense in ("none", "", None):
        return "undefended"
    if defense == "k_mosaic":
        return defense if getattr(args, "attack", "algen") == "algen" else (
            f"{args.attack}_{defense}_scope-{getattr(args, 'defense_scope', 'targets')}")
    parts = [defense]

    attack = getattr(args, "attack", "algen")
    if attack != "algen":
        parts.insert(0, attack)
        parts.append(f"scope-{getattr(args, 'defense_scope', 'targets')}")
    if defense == "gaussian":
        parts.append(f"sigma{args.noise_level:g}")
    elif defense == "vec2text":
        parts.append(f"lam{knob_for(args, defense):g}")
    elif defense == "remote_rag":
        parts.append(f"r{knob_for(args, defense):g}")
    elif defense in ("sparse", "cmag", "lapmech", "purmech", "dp_gaussian"):
        parts.append(f"eps{knob_for(args, defense):g}")
    fn = {
        "eguard": algen_pipeline.eguard_checkpoint_for,
        "sparse": algen_pipeline.sparse_checkpoint_for,
        "cmag": algen_pipeline.cmag_checkpoint_for,
    }.get(defense)
    if fn is not None:
        parts.append(fn(args).stem.removeprefix(defense + "_"))
    return "_".join(parts)


def _defense_kwargs(args: argparse.Namespace, defense: str) -> dict[str, Any]:
    import train_algen as algen_pipeline

    if defense in ("none", "", "k_mosaic"):
        return {"defense": "none"}
    eps = knob_for(args, defense)
    v = algen_pipeline.vec2text_checkpoint_for(args)
    r = algen_pipeline.remote_rag_checkpoint_for(args)
    return {
        "defense": defense,
        "noise_level": args.noise_level,
        "epsilon": eps,
        "eguard_checkpoint": algen_pipeline.eguard_checkpoint_for(args) if defense == "eguard" else None,
        "sparse_checkpoint": algen_pipeline.sparse_checkpoint_for(args) if defense == "sparse" else None,
        "cmag_checkpoint": algen_pipeline.cmag_checkpoint_for(args) if defense == "cmag" else None,
        "vec2text_checkpoint": v if defense == "vec2text" and v.exists() else None,
        "vec2text_noise_level": eps if defense == "vec2text" else None,
        "remote_rag_checkpoint": r if defense == "remote_rag" and r.exists() else None,
        "remote_rag_radius": eps if defense == "remote_rag" else None,
    }


def run_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    attack: str,
    paths: list[Path],
    cache_dir: Path,
    extra: dict[str, Any] | None = None,
) -> list[dict]:
    cache = cache_dir / f"{arm}.json"
    if cache.exists() and not args.force:
        print(f"[skip] {arm}: cached at {cache}")
        return json.loads(cache.read_text())

    print(f"\n{'-' * 72}\n[arm: {arm}] attack={attack}\n{'-' * 72}")
    kw: dict[str, Any] = dict(
        checkpoint_dir=args.checkpoint,
        dataset=args.attack_dataset,
        align_samples=args.align_samples,
        reg_lambda=args.reg_lambda,
        seed=args.seed,
        device=args.device,
        align_text=args.align_text,
        attention_mask=args.attention_mask,
        **(extra or {}),
    )

    if attack.startswith("teia"):
        for k in _ALGEN_ONLY:
            kw.pop(k, None)

    if attack.startswith("zero2text"):
        from attacker.zero2text.cli import attack_kwargs as _z2t_kwargs

        kw.update(_z2t_kwargs(args))
    atk = get_attack(attack, **kw)
    summary = []
    for path in paths:
        result = atk.attack_file(path)
        result.show(args.show)
        summary.append(result.to_dict())

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(summary, indent=2))

    import train_algen as algen_pipeline
    algen_pipeline._free(atk)
    return summary


def main(argv: list[str] | None = None) -> int:
    import train_algen as algen_pipeline

    args = resolve_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
        return 0
    set_seed(args.seed)

    if args.checkpoint is None:
        args.checkpoint = algen_pipeline.stage_dirs(args)["finetune"]
    from attacker.algen.trainer import GeneratorTrainer

    if not GeneratorTrainer.is_complete(Path(args.checkpoint)):
        raise SystemExit(
            f"no usable generator at {args.checkpoint}\n"
            "Train one: python train_algen.py --stages pretrain,finetune"
        )

    if args.attack == "zero2text" and args.ladder_out == parser_default_ladder_out():
        raise SystemExit(
            "zero2text must not share ALGEN's ladder directory.\n"
            "  Its 'undefended' and floor arms are DIFFERENT arms under the SAME "
            "filenames, so it would\n"
            "  read ALGEN's cached ceiling as its own (or overwrite it under --force).\n"
            "  Re-run with:  --ladder-out metrics/outputs/zero2text"
        )

    if getattr(args, "algen_cell_maps", False):
        if args.attack != "algen":
            raise SystemExit("--algen-cell-maps applies to --attack algen only")
        if Path(args.ladder_out) == Path(parser_default_ladder_out()):
            raise SystemExit(
                "--algen-cell-maps must not share ALGEN's ladder directory: its arms "
                "have the global-map arms' names.\n"
                "  Re-run with:  --ladder-out metrics/outputs/algen_cellmaps")

    paths = algen_pipeline.victim_paths(args)
    defense = args.defense if args.defense not in ("none", "") else None

    cache_dir = Path(args.ladder_out) / Path(args.checkpoint).name / (
        f"k{args.align_samples}_ridge{args.reg_lambda}_{args.align_text}"
    )

    ann = args.match_metric == "ann"
    if defense == "k_mosaic" and not args.partition:
        print("ladder: --defense k_mosaic implies --partition")
        args.partition = True
    if args.partition and not ann:
        raise SystemExit("--partition is an index layout: it needs --match-metric ann")
    probe = None
    store: dict[str, Any] = {}
    treat_dir = cache_dir
    if ann:
        from ANN import AnnProbe, ladder_subdir, partition_subdir

        if args.dry_run:
            print("Ladder:: ANN mode   (corpus not loaded in a dry run)")
        else:
            probe = AnnProbe.from_args(args)
            cache_dir = cache_dir / ladder_subdir(args, probe.corpus.vectors.shape[1])
            probe.cache_dir = cache_dir        # clean baseline is shared by every arm

            store = {"storage": probe.flat_storage() if args.partition
                     else probe.storage("none")}
            treat_dir = cache_dir / partition_subdir(args) if args.partition else cache_dir
            print(f"Ladder:: ANN        {probe.corpus.describe()}; ceiling and floors read "
                  f"{store['storage'].describe()}")
            if args.partition:
                cellmaps = getattr(args, "algen_cell_maps", False)
                aware = cellmaps or (args.attack == "zero2text"
                                     and getattr(args, "z2t_local_pairs", 0) > 0)
                print(f"Ladder:: partition  {probe.describe()}")
                if not aware:
                    print(f"Ladder:: !! {args.attack} is NOT partition-aware: it fits one "
                          f"map across every cell, so leak_norm here is an\n"
                          f"Ladder:: !! UPPER BOUND on protection (defense/k_mosaic.py ")
                elif cellmaps:
                    print(f"Ladder:: !! algen IS partition-aware (--algen-cell-maps): one "
                          f"map per cell, fit on that cell's leaked pairs\n"
                          f"Ladder:: !! (min {args.algen_cell_min_pairs}, else the global "
                          f"map), so leak_norm here is NOT the one-map upper bound.")
                else:
                    print(f"Ladder:: !! {args.attack} IS partition-aware at "
                          f"--z2t-local-pairs {args.z2t_local_pairs}: it refits per target "
                          f"on its nearest pool\n"
                          f"Ladder:: !! entries, so leak_norm here is NOT the one-map upper "
                          f"bound.")

    print(f"Ladder:: generator  {args.checkpoint}")
    print(f"Ladder:: targets    {len(paths)} set(s), {args.attack_dataset}")
    print(f"Ladder:: treatment  {defense or '(none — floors and ceiling only)'}")
    print(f"Ladder:: cache      {cache_dir}"
          + (f"\nLadder:: treatment  cached in {treat_dir}" if treat_dir != cache_dir else ""))
    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    if ann:
        src = cache_dir.parent / "floor_prior_only.json"
        dst = cache_dir / "floor_prior_only.json"
        if src.exists() and not dst.exists() and not args.force:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.read_text())
            print(f"Ladder:: floor_prior_only reused from {src} (consumes no victim vector)")

    arms: dict[str, list[dict]] = {}
    arms["floor_prior_only"] = run_arm(
        args, arm="floor_prior_only", attack=FLOORS_FOR[args.attack]["floor_prior_only"],
        paths=paths,
        cache_dir=cache_dir, extra={"prior": args.floor_prior},
    )
    arms["floor_random_vector"] = run_arm(
        args, arm="floor_random_vector",
        attack=FLOORS_FOR[args.attack]["floor_random_vector"], paths=paths,
        cache_dir=cache_dir, extra={"source": args.floor_source, **store},
    )
    arms["undefended"] = run_arm(
        args, arm="undefended", attack=args.attack, paths=paths, cache_dir=cache_dir,
        extra={**_attack_kwargs(args, "none"), **store},
    )

    utility: dict[str, float] = {}
    matched = None
    if defense is not None and args.target_recall is not None:

        from .utility_match import KNOBS, _metric_label, _verify_repeats

        extra = {k: v for k, v in _defense_kwargs(args, defense).items()
                 if k not in ("defense", "epsilon", "noise_level",
                              "vec2text_noise_level", "remote_rag_radius")}
        X = None if ann else eval_embeddings_for(args)
        matched = match_utility(
            X, defense, args.target_recall, k=args.match_k, repeats=args.match_repeats,
            seed=args.seed, tolerance=args.match_tolerance, max_iter=args.match_max_iter,
            extra=extra, metric_label=_metric_label(args),
            probe_fn=(probe.probe_fn(defense, extra, KNOBS[defense].kwargs)
                      if ann and defense in KNOBS else None),
            verify_fn=(probe.probe_fn(defense, extra, KNOBS[defense].kwargs,
                                      repeats=_verify_repeats(args), seed=args.seed + 1000)
                       if ann and defense in KNOBS else None),
        )
        del X
        if not matched.converged:
            raise SystemExit(
                f"could not match {defense} to recall@{args.match_k}="
                f"{args.target_recall}:\n  {matched.reason}\n"
                "Refusing to attack at an unmatched point and label it matched."
            )
        args.epsilon = matched.value
        for flag in ("sparse_epsilon", "cmag_epsilon"):
            if defense == flag.split("_")[0]:
                setattr(args, flag, matched.value)
        if defense == "vec2text":
            args.vec2text_noise_level = matched.value
        if defense == "remote_rag":
            args.remote_rag_radius = matched.value
        print(f"Ladder:: matched {defense} {matched.knob}={matched.value:.6g} "
              f"-> recall@{args.match_k}={matched.utility:.4f}")

    if defense is not None:
        label = (
            f"{defense}@{'ann' if ann else 'recall'}{args.target_recall:g}"
            if matched is not None else arm_label(args, defense)
        )
        if ann:
            store = {"storage": probe.storage(defense, _defense_kwargs(args, defense))}
            if args.partition:
                print(f"Ladder:: treatment reads {store['storage'].describe()}")
        arms[label] = run_arm(
            args, arm=label, attack=args.attack, paths=paths, cache_dir=treat_dir,
            extra={**_attack_kwargs(args, defense), **store},
        )
        rk = (matched.utility if matched is not None else args.recall_at_k)
        if rk is not None:
            utility[label] = rk
        elif not ann:  # ANN mode measures it below, with the decomposition
            print(
                f"Warn: no recall@k for {label}: an arm at leak_norm ~ 0 cannot be "
                f"told apart from one that destroyed the space, so it will read\n"
                f"       'unclassified'. Pass --recall-at-k, or fit the defense with "
                f"`python train_algen.py --stages {defense}` so its report exists."
            )

    ann_report: dict[str, dict] = {}
    if ann:
        ref = "undefended (partitioned index)" if args.partition else "undefended"
        runs = [(ref, "none")] + ([(label, defense)] if defense is not None else [])
        from ANN import utility_key, write_utility

        for arm_name, arm_defense in runs:
            u = probe.measure(arm_defense, _defense_kwargs(args, arm_defense), curve=True)
            ann_report[arm_name] = u.to_dict()
            write_utility(treat_dir, utility_key(args, arm_name), u)
            if arm_name != ref and arm_name not in utility:
                utility[arm_name] = u.recall
            print(f"[ann] {arm_name:<28} ANN recall@{u.k} = {u.recall:.4f}  "
                  f"(exact {u.exact_recall:.4f}, index-only {u.index_recall:.4f}, "
                  f"surcharge {u.surcharge:.2f}x {u.budget_param})")
            if u.partition:
                pr = u.partition
                print(f"[ann] {'':<28} partition: {pr['cells_effective']} cells, largest "
                      f"{pr['list_max']} ({pr['max_share']:.2%}), gini {pr['gini']:.3f}, "
                      f"N/max {pr['effective_security']:.0f}, "
                      f"voronoi {pr.get('voronoi_agreement', float('nan')):.3f}")

    ladder = build_ladder(arms, utility=utility, min_headroom=args.min_headroom)
    print_ladder(ladder, headline=args.headline,
                 utility_label=f"ANN r@{args.match_k}" if ann else "recall@k",
                 min_headroom=args.min_headroom)
    if matched is not None:
        print(f"\n  [matched arm] {defense} {matched.knob}={matched.value:.6g}, "
              f"recall@{args.match_k}={matched.utility:.4f}+-{matched.utility_se:.4f}, "
              f"plateau {matched.plateau_width:.2f} decades")
        print("                Cross-defense comparison is valid ONLY between arms "
              "matched to the same target.")

    out = treat_dir / "ladder.json"
    payload = ladder.to_dict()
    if matched is not None:
        payload["utility_matched"] = matched.to_dict()
    if ann:
        payload["utility_ann"] = ann_report

    if out.exists():
        try:
            prior = json.loads(out.read_text())
            for key in ("utility_recall_at_k", "utility_ann"):
                merged = dict(prior.get(key) or {})
                merged.update(payload.get(key) or {})
                payload[key] = merged
        except (OSError, json.JSONDecodeError):
            pass
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  [out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
