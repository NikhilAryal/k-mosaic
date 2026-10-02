from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from attacker import get_attack
from attacker.algen.defenses import DEFENSES
from attacker.data import collect_target_ids, find_embedding_sets
from attacker.teia.attack import DEFENSE_SCOPES
from attacker.teia.cli import build_teia_trainer, defense_tag_for
from attacker.teia.trainer import TeiaTrainer
from attacker.utils import set_seed
from main import load_config
from models import EmbeddingSet

import train_algen as algen_pipeline
from train_algen import (
    DEFENSE_LABELS,
    DEFAULT_COMPARE_METRICS,
    LEARNED_DEFENSES,
    _cmag_epsilons,
    _print_comparison,
    _remote_rag_radii,
    _sparse_epsilons,
    _vec2text_noise_levels,
    cmag_checkpoint_for,
    eguard_checkpoint_for,
    remote_rag_checkpoint_for,
    run_cmag_stage,
    run_eguard_stage,
    run_remote_rag_stage,
    run_sparse_stage,
    run_vec2text_stage,
    sparse_checkpoint_for,
    vec2text_checkpoint_for,
    victim_paths,
)

STAGES = ("eguard", "sparse", "cmag", "vec2text", "remote_rag", "train", "attack")

COMPARE_METRICS = [m for m in DEFAULT_COMPARE_METRICS if m[0] != "align_cos"] + [
    ("recon_cos", "diagnostics", "recon_COS"),
    ("embed_sim", "diagnostics", "embed_similarity"),
]

UNUSED_FLAGS = (
    "pretrain_dataset", "pretrain_samples", "pretrain_val_samples",
    "pretrain_align_reserve", "pretrain_batch_size", "pretrain_lr", "pretrain_epochs",
    "finetune_dataset", "finetune_samples", "finetune_val_samples",
    "finetune_align_reserve", "finetune_batch_size", "finetune_lr", "finetune_epochs",
    "align_samples", "reg_lambda", "align_text", "attention_mask",
    "sweep", "sweep_k", "sweep_lambda", "model_name",
)


def build_parser() -> argparse.ArgumentParser:
    p = algen_pipeline.build_parser()
    p.prog = "train_teia.py"
    p.description = (
        "Invert data/embeddings with TEIA (arXiv:2406.10280): a surrogate encoder "
        "and a trained adapter stand in for query access the attacker does not have."
    )

    hide = set(UNUSED_FLAGS)
    for action in p._actions:  # argparse's only handle on a registered argument
        if action.dest in hide:
            action.help = argparse.SUPPRESS

    g = p.add_argument_group("stage 2 — TEIA training (arXiv:2406.10280 §3)")
    g.add_argument("--decoder-name", default="microsoft/DialoGPT-small",
                   help="the causal LM that decodes a victim vector.")
    g.add_argument("--surrogate-model", default="gte-base",
                   help="the frozen off-the-shelf encoder behind the adapter.")
    g.add_argument("--external-dataset", default="beir/trec-covid",
                   help="corpus D_S is drawn from.")
    g.add_argument("--leaked-samples", type=int, default=2000,
                   help="|D_L|, the leaked (text, vector) pairs.")
    g.add_argument("--external-samples", type=int, default=20000,
                   help="|D_S|, the external texts.")
    g.add_argument("--teia-val-samples", type=int, default=200,
                   help="held-out slice the decoder is selected on")
    g.add_argument("--teia-batch-size", type=int, default=16,
                   help="batch size; a weighted sampler keeps D_L and D_S "
                        "at roughly half each")
    g.add_argument("--teia-epochs", type=int, default=24)
    g.add_argument("--teia-max-length", type=int, default=32,
                   help="tokens the decoder reconstructs.")
    g.add_argument("--mapping-lambda", type=float, default=1.0,
                   help="weight on L_adv in Eq. 7")
    g.add_argument("--pivot-lambda", type=float, default=1.0,
                   help="weight on L_intra + L_inter (Eqs. 4, 6)")
    g.add_argument("--geia", action="store_true",
                   help="train the direct-attack baseline instead")
    g.add_argument("--compare-geia", action="store_true",
                   help="train and attack both arms")
    g.add_argument("--no-embed-similarity", action="store_true",
                   help="skip upstream's headline metric")

    g = p.add_argument_group("threat model")
    g.add_argument("--defense-scope", choices=sorted(DEFENSE_SCOPES), default="both",
                   help="'defense")
    p.set_defaults(stages=",".join(STAGES))
    return p


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI > config file > defaults, matching main.py's precedence."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        parser.set_defaults(
            **load_config(
                args.config,
                parser=parser,
                sections=("eguard", "sparse", "cmag", "vec2text", "remote_rag",
                          "defense", "attack", "teia"),
                section_name_key={"dataset": "dataset", "model": "model", "defense": "defense"},
            )
        )
        args = parser.parse_args(argv)
    return args


def _epsilons_for(args: argparse.Namespace, defense: str) -> list[float]:
    return {
        "sparse": _sparse_epsilons,
        "cmag": _cmag_epsilons,
        "vec2text": _vec2text_noise_levels,
        "remote_rag": _remote_rag_radii,
    }.get(defense, lambda a: [a.epsilon])(args)


def _sweep_note(defense: str, eps: float) -> str:
    if defense == "remote_rag":
        return f" r={eps:g}"
    if defense == "vec2text":
        return f" lambda={eps:g}"
    if defense in ("sparse", "cmag", "lapmech", "purmech", "dp_gaussian"):
        return f" epsilon={eps:g}"
    return ""


def defense_kwargs_for(args: argparse.Namespace, defense: str, eps: float | None) -> dict[str, Any]:
    eps = args.epsilon if eps is None else eps
    kw: dict[str, Any] = {
        "noise_level": args.noise_level,
        "epsilon": eps,
        "delta": args.delta if hasattr(args, "delta") else 1e-5,
        "eguard_checkpoint": eguard_checkpoint_for(args) if defense == "eguard" else None,
        "sparse_checkpoint": sparse_checkpoint_for(args) if defense == "sparse" else None,
        "cmag_checkpoint": cmag_checkpoint_for(args) if defense == "cmag" else None,
        "vec2text_noise_level": eps if defense == "vec2text" else None,
        "remote_rag_radius": eps if defense == "remote_rag" else None,
    }
    v = vec2text_checkpoint_for(args)
    r = remote_rag_checkpoint_for(args)
    kw["vec2text_checkpoint"] = v if defense == "vec2text" and v.exists() else None
    kw["remote_rag_checkpoint"] = r if defense == "remote_rag" and r.exists() else None
    return kw


def run_dir_for(args: argparse.Namespace, *, geia: bool, defense: str, eps: float | None) -> Path:
    return TeiaTrainer.run_dir_for(
        decoder_name=args.decoder_name,
        output_dir=args.output_dir,
        dataset=args.attack_dataset,
        external_dataset=args.external_dataset,
        surrogate_model=args.surrogate_model,
        max_length=args.teia_max_length,
        leaked_samples=args.leaked_samples,
        external_samples=args.external_samples,
        batch_size=args.teia_batch_size,
        num_epochs=args.teia_epochs,
        geia=geia,
        defense_tag=defense_tag_for(defense, eps, args.defense_scope),
    )


def training_arms(args: argparse.Namespace) -> list[tuple[str, bool, str, float | None]]:
    defended = args.defense if args.defense not in ("none", "") else None
    geias = [False, True] if args.compare_geia else [bool(args.geia)]

    arms: list[tuple[str, bool, str, float | None]] = []
    for geia in geias:
        name = "geia" if geia else "teia"
        if args.defense_scope == "targets" or defended is None:
            arms.append((name, geia, "none", None))
        else:
            if args.compare:
                arms.append((f"{name}_baseline", geia, "none", None))
            for eps in _epsilons_for(args, defended):
                label = f"{name}_{defended}"
                if len(_epsilons_for(args, defended)) > 1:
                    label += f"_eps{eps:g}"
                arms.append((label, geia, defended, eps))
    return arms


def run_train_stage(args: argparse.Namespace, holdout: list[str]) -> dict[str, Path]:
    paths = victim_paths(args)
    embset = EmbeddingSet.load(paths[0])

    out: dict[str, Path] = {}
    for label, geia, defense, eps in training_arms(args):
        run_dir = run_dir_for(args, geia=geia, defense=defense, eps=eps)
        out[label] = run_dir
        print(f"\n{'=' * 72}")
        print(f"[stage: train] {label} — geia={geia} defense={defense}"
              f"{_sweep_note(defense, eps) if eps is not None else ''}")
        print(f"{'=' * 72}")
        if TeiaTrainer.is_complete(run_dir) and not args.force:
            print(f"[skip] already finished: {run_dir}\n       (--force to retrain)")
            continue

        trainer = build_teia_trainer(
            victim_embset=embset,
            dataset=args.attack_dataset,
            external_dataset=args.external_dataset,
            surrogate_model=args.surrogate_model,
            decoder_name=args.decoder_name,
            output_dir=args.output_dir,
            max_length=args.teia_max_length,
            leaked_samples=args.leaked_samples,
            external_samples=args.external_samples,
            val_samples=args.teia_val_samples,
            batch_size=args.teia_batch_size,
            epochs=args.teia_epochs,
            mapping_lambda=args.mapping_lambda,
            pivot_lambda=args.pivot_lambda,
            geia=geia,
            holdout=holdout,
            seed=args.seed,
            device=args.device,
            defense=defense,
            defense_scope=args.defense_scope,
            defense_kwargs=defense_kwargs_for(args, defense, eps),
        )
        print(f"[run] {trainer.run_dir}")
        trainer.train()
        algen_pipeline._free(trainer)
    return out


def _attack_arm(
    args: argparse.Namespace,
    checkpoint: Path,
    paths: list[Path],
    *,
    label: str,
    defense: str,
    epsilon: float | None,
    out_root: Path,
) -> list[dict]:
    eps = args.epsilon if epsilon is None else epsilon
    print(f"\n{'-' * 72}")
    print(f"[arm: {label}] defense={defense}{_sweep_note(defense, eps)} "
          f"scope={args.defense_scope}")
    print(f"{'-' * 72}")

    kw = defense_kwargs_for(args, defense, eps)
    attacker = get_attack(
        "teia",
        checkpoint_dir=checkpoint,
        dataset=args.attack_dataset,
        seed=args.seed,
        device=args.device,
        defense=defense,
        defense_scope=args.defense_scope,
        embed_similarity=not args.no_embed_similarity,
        **kw,
    )

    defense_tag = ""
    if defense != "none":
        defense_tag = f"_{defense}{_sweep_note(defense, eps).replace(' ', '_').replace('=', '')}"
    summary = []
    for path in paths:
        tag = f"teia_{path.stem}_{label}{defense_tag}_scope-{args.defense_scope}"
        result = attacker.attack_file(path, out_dir=out_root / tag)
        result.show(args.show)
        summary.append(result.to_dict())

    algen_pipeline._free(attacker)
    return summary


def _print_arm_table(
    label: str, summary: list[dict], *, defense: str | None = None, knob: str = ""
) -> None:
    head = f"  [{label}]"
    if defense in ("none", ""):
        head += "   defense: none (undefended baseline)"
    elif defense:
        head += f"   defense: {DEFENSE_LABELS.get(defense, defense.capitalize())}{knob}"
    print(f"\n{head}")
    print(
        f"  {'embedding set':<46} {'n':>4} {'tokF1':>7} {'tokRec':>7} {'bleu':>7} "
        f"{'rougeL':>8} {'recon_cos':>10} {'val_rougeL':>11}"
    )
    for s_ in summary:
        print(
            f"  {s_['source'][:46]:<46} {s_['n']:>4} "
            f"{s_['test_results'].get('token_f1', float('nan')):>7.4f} "
            f"{s_['test_results'].get('token_recall', float('nan')):>7.4f} "
            f"{s_['test_results']['bleu']:>7.2f} "
            f"{s_['test_results']['rougeL']:>8.4f} "
            f"{s_['diagnostics'].get('recon_COS', float('nan')):>10.4f} "
            f"{s_['diagnostics'].get('val_rougeL', float('nan')):>11.4f}"
        )


def _mean(rows: list[dict], section: str, key: str) -> float:
    vals = [float(r[section][key]) for r in rows if key in r[section]]
    return sum(vals) / len(vals) if vals else float("nan")


def _print_table1(results: dict[str, list[dict]]) -> None:
    print(f"\n{'=' * 104}")
    print("[Table 1] transfer-attack performance — arXiv:2406.10280 §4.1 metrics, our runs")
    print(f"{'=' * 104}")
    print(
        f"  {'arm':<32} {'n':>4} {'Rouge-L':>9} {'BLEU':>7} {'tokF1':>7} "
        f"{'Cos':>7} {'recon_cos':>10} {'def_cos':>8} {'val_ppl':>9}"
    )
    for label, rows in results.items():
        if not rows:
            continue
        n = sum(int(r["n"]) for r in rows)
        def_cos = (
            _mean(rows, "diagnostics", "defense_COS")
            if any("defense_COS" in r["diagnostics"] for r in rows)
            else float("nan")
        )
        print(
            f"  {label[:32]:<32} {n:>4} "
            f"{_mean(rows, 'test_results', 'rougeL'):>9.4f} "
            f"{_mean(rows, 'test_results', 'bleu'):>7.2f} "
            f"{_mean(rows, 'test_results', 'token_f1'):>7.4f} "
            f"{_mean(rows, 'diagnostics', 'embed_similarity'):>7.4f} "
            f"{_mean(rows, 'diagnostics', 'recon_COS'):>10.4f} "
            f"{def_cos:>8.4f} "
            f"{_mean(rows, 'diagnostics', 'val_perplexity'):>9.2f}"
        )


def _print_geia_comparison(results: dict[str, list[dict]], pairs: list[tuple[str, str, str]]) -> None:
    print(f"\n{'=' * 104}")
    print("[ablation] GEIA direct attack vs TEIA transfer attack")
    print(f"{'=' * 104}")
    print(f"  {'arm':<32} {'metric':<16} {'geia':>10} {'teia':>10} {'improvement':>13}")
    for name, geia_label, teia_label in pairs:
        a, b = results.get(geia_label), results.get(teia_label)
        if not a or not b:
            continue
        first = True
        for metric, section, key in (
            ("rougeL", "test_results", "rougeL"),
            ("bleu", "test_results", "bleu"),
            ("token_f1", "test_results", "token_f1"),
            ("embed_sim (Cos)", "diagnostics", "embed_similarity"),
            ("recon_cos", "diagnostics", "recon_COS"),
        ):
            g, t = _mean(a, section, key), _mean(b, section, key)
            head = f"  {name[:32]:<32}" if first else " " * 34
            print(f"{head}  {metric:<16} {g:>10.4f} {t:>10.4f} "
                  f"{algen_pipeline._pct_change(g, t):>13}")
            first = False
        print()


def run_attack_stage(args: argparse.Namespace, checkpoints: dict[str, Path]) -> dict[str, list[dict]]:
    print(f"\n{'=' * 72}\n[stage: attack]\n{'=' * 72}")
    paths = victim_paths(args)
    out_root = Path(args.output_dir) / "teia" / "attacks"
    defended = args.defense if args.defense not in ("none", "") else None

    arms: list[tuple[str, Path, str, float | None]] = []
    for label, ckpt in checkpoints.items():
        if not TeiaTrainer.is_complete(ckpt):
            raise SystemExit(
                f"no usable decoder at {ckpt}\n"
                f"Train it first: python train_teia.py --stages train"
            )
        if args.defense_scope == "both":
            if label.endswith("_baseline") or defended is None:
                arms.append((label, ckpt, "none", None))
            else:
                eps = None
                for e in _epsilons_for(args, defended):
                    if len(_epsilons_for(args, defended)) == 1 or label.endswith(f"eps{e:g}"):
                        eps = e
                        break
                arms.append((label, ckpt, defended, eps))
        else:
            if args.compare or defended is None:
                arms.append((f"{label}_baseline" if defended else label, ckpt, "none", None))
            if defended is not None:
                for eps in _epsilons_for(args, defended):
                    suffix = "" if len(_epsilons_for(args, defended)) == 1 else f"_eps{eps:g}"
                    arms.append((f"{label}_{defended}{suffix}", ckpt, defended, eps))

    needed = {d for _, _, d, _ in arms}
    for name in LEARNED_DEFENSES:
        ckpt = {"eguard": eguard_checkpoint_for, "sparse": sparse_checkpoint_for,
                "cmag": cmag_checkpoint_for}[name](args)
        if name in needed and not ckpt.exists():
            raise SystemExit(
                f"defense {name!r} needs a fitted checkpoint, none at {ckpt}\n"
                f"Fit it first: python train_teia.py --stages {name}"
            )

    results: dict[str, list[dict]] = {}
    for label, ckpt, defense, eps in arms:
        results[label] = _attack_arm(
            args, ckpt, paths, label=label, defense=defense, epsilon=eps, out_root=out_root
        )

    out_root.mkdir(parents=True, exist_ok=True)
    for label, summary in results.items():
        with open(out_root / f"teia_summary_{label}.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'=' * 72}\n[results] {out_root}\n{'=' * 72}")
    arm_meta = {lbl: (d, e) for lbl, _, d, e in arms}
    for label, summary in results.items():
        d, e = arm_meta.get(label, (None, None))
        _print_arm_table(label, summary, defense=d, knob=_sweep_note(d, e) if d and e is not None else "")
    _print_table1(results)

    if args.compare and defended is not None:
        ckpt = {"sparse": sparse_checkpoint_for, "eguard": eguard_checkpoint_for,
                "cmag": cmag_checkpoint_for, "vec2text": vec2text_checkpoint_for,
                "remote_rag": remote_rag_checkpoint_for}.get(defended)
        report_path = None
        if ckpt is not None:
            path = ckpt(args)
            report_path = path.parent / f"{path.stem}_report.json"
        utility = (
            json.loads(report_path.read_text())
            if report_path is not None and report_path.exists() else None
        )
        for base_label in [l for l in results if l.endswith("_baseline")]:
            stem = base_label.removesuffix("_baseline")
            for label, _, defense, eps in arms:
                if defense == "none" or not label.startswith(stem):
                    continue
                u = utility
                if u and defended in ("sparse", "cmag", "vec2text", "remote_rag") and eps is not None:
                    by_eps = u.get("utility_by_epsilon", {})
                    if str(eps) in by_eps:
                        u = {**u, "utility_holdout": by_eps[str(eps)]}
                pretty = DEFENSE_LABELS.get(defended, defended.capitalize())
                pretty += _sweep_note(defended, eps if eps is not None else args.epsilon)
                _print_comparison(
                    results[base_label], results[label], u,
                    defense_label=pretty, metrics=COMPARE_METRICS,
                )

    if args.compare_geia:
        pairs = []
        for label in results:
            twin = label.replace("teia", "geia", 1)
            if label.startswith("teia") and twin in results:
                pairs.append((label.removeprefix("teia_") or "baseline", twin, label))
        _print_geia_comparison(results, pairs)

    comparison = out_root / (
        f"teia_comparison_{defended or 'none'}_scope-{args.defense_scope}.json"
    )
    with open(comparison, "w") as f:
        json.dump(
            {
                "threat_model": "teia (arXiv:2406.10280 §2.2)",
                "defense": args.defense,
                "defense_scope": args.defense_scope,
                "arms": [
                    {"label": l, "checkpoint": str(c), "defense": d, "epsilon": e}
                    for l, c, d, e in arms
                ],
                "results": results,
            },
            f, indent=2,
        )
    print(f"\n  [out] {comparison}")
    return results


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
        return 0

    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    if args.defense not in DEFENSES:
        raise SystemExit(f"unknown defense {args.defense!r}; known: {sorted(DEFENSES)}")
    if args.surrogate_model == args.victim_model:
        print(
            f"[warn] --surrogate-model equals --victim-model ({args.victim_model}). "
            f"eval measures this case and finds it only slightly better, but it is "
            f"not the paper's threat model: Assumption 1 says phi is hidden."
        )

    set_seed(args.seed)
    holdout = collect_target_ids(find_embedding_sets("all"), args.attack_dataset)

    print("[pipeline] TEIA threat model: the victim encoder is")
    print("           completely hidden: no weights, no architecture, NO query access.")
    print("           All the attacker has is a leaked (text, vector) pair set and an")
    print("           external corpus of its own, routed through a surrogate + adapter.")
    print(f"[pipeline] stages={stages} seed={args.seed}")
    print(f"[threat]   defense={args.defense!r} scope={args.defense_scope!r}")
    print(f"           {DEFENSE_SCOPES[args.defense_scope]}")
    print(f"[teia]     |D_L|={args.leaked_samples} |D_S|={args.external_samples} "
          f"surrogate={args.surrogate_model} decoder={args.decoder_name}")
    print(f"[holdout]  {len(holdout)} attack-target id(s) excluded from every pool")

    arms = training_arms(args)
    checkpoints = {label: run_dir_for(args, geia=g, defense=d, eps=e) for label, g, d, e in arms}
    for label, run_dir in checkpoints.items():
        state = "cached" if TeiaTrainer.is_complete(run_dir) else "to train"
        print(f"[{label:<20}] {state:<9} {run_dir}")
    if args.defense_scope == "both" and args.defense not in ("none", "") and len(arms) > 1:
        print(
            f"[note] defense_scope=both bakes the defense into the decoder, so this run\n"
            f"       needs {len(arms)} training runs. --defense-scope targets trains one\n"
            f"       and attacks every arm through it, at the cost of changing the threat\n"
            f"       model to STEER's."
        )

    defended = args.defense if args.defense not in ("none", "") else "eguard"
    for name, ckpt_fn in (("eguard", eguard_checkpoint_for), ("sparse", sparse_checkpoint_for),
                          ("cmag", cmag_checkpoint_for), ("vec2text", vec2text_checkpoint_for),
                          ("remote_rag", remote_rag_checkpoint_for)):
        if name not in stages and name != defended:
            continue
        ckpt = ckpt_fn(args)
        note = ""
        if name in ("sparse", "cmag", "vec2text", "remote_rag"):
            eps = _epsilons_for(args, name)
            note = f"  eps={eps[0]:g}" if len(eps) == 1 else f"  eps sweep={[f'{e:g}' for e in eps]}"
        print(f"[{name:<20}] {'cached' if ckpt.exists() else 'to fit':<9} {ckpt}{note}")

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    for name, runner in (("eguard", run_eguard_stage), ("sparse", run_sparse_stage),
                         ("cmag", run_cmag_stage), ("vec2text", run_vec2text_stage),
                         ("remote_rag", run_remote_rag_stage)):
        if name in stages:
            runner(args, holdout)

    if "train" in stages:
        checkpoints = run_train_stage(args, holdout)

    if "attack" in stages:
        run_attack_stage(args, checkpoints)

    return 0


if __name__ == "__main__":
    sys.exit(main())
