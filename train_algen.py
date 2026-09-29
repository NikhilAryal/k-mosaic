from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from attacker import get_attack
from attacker.algen.attack import ALGENAttacker
from attacker.data import (
    build_splits,
    collect_target_ids,
    find_embedding_sets,
    victim_embedder,
)
from attacker.algen.trainer import GeneratorTrainer
from attacker.utils import set_seed
from main import load_config
from models import EmbeddingSet

STAGES = ("pretrain", "finetune", "eguard", "sparse", "cmag", "vec2text", "remote_rag", "attack")

LEARNED_DEFENSES = ("eguard", "sparse", "cmag")

DEFENSE_LABELS = {"idct": "IDCT", "eguard": "Eguard", "sparse": "Sparse",
                  "cmag": "CMAG", "vec2text": "vec2text \u00a76",
                  "remote_rag": "RemoteRAG \u00a73.2.1"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_algen.py",
        description="Reproduce the full ALGEN inversion pipeline against data/embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="YAML config; CLI overrides it")
    p.add_argument("--print-config", action="store_true", help="print resolved settings and exit")
    p.add_argument("--dry-run", action="store_true", help="show the plan without running it")
    p.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated subset of {STAGES}",
    )
    p.add_argument("--force", action="store_true", help="retrain stages that already finished")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="attacker/outputs")
    p.add_argument("--model-name", default="google/flan-t5-base", help="the generator G")
    p.add_argument("--max-length", type=int, default=32, help="L: tokens G reconstructs")
    p.add_argument("--wandb-run-name", default=None)

    g = p.add_argument_group("stage 1 — pretrain (large, out-of-domain)")
    g.add_argument("--pretrain-dataset", default="beir/trec-covid")
    g.add_argument("--pretrain-samples", type=int, default=50000)
    g.add_argument("--pretrain-val-samples", type=int, default=200)
    g.add_argument("--pretrain-align-reserve", type=int, default=500)
    g.add_argument("--pretrain-batch-size", type=int, default=128)
    g.add_argument("--pretrain-lr", type=float, default=1e-4)
    g.add_argument("--pretrain-epochs", type=int, default=12)

    g = p.add_argument_group("stage 2 — finetune (small, in-domain)")
    g.add_argument("--finetune-dataset", default="nfcorpus")
    g.add_argument("--finetune-samples", type=int, default=3000)
    g.add_argument("--finetune-val-samples", type=int, default=200)
    g.add_argument("--finetune-align-reserve", type=int, default=200)
    g.add_argument("--finetune-batch-size", type=int, default=64)
    g.add_argument("--finetune-lr", type=float, default=5e-5, help="lower than stage 1: adapting, not learning")
    g.add_argument("--finetune-epochs", type=int, default=15)

    g = p.add_argument_group("stage 3 — eguard defense (defense/eguard.py)")
    g.add_argument("--eguard-checkpoint", default=None,
                   help="path to the trained g_p; default derives one under --output-dir")
    g.add_argument("--eguard-samples", type=int, default=2500,
                   help="victim documents whose embeddings train g_p (attack targets excluded)")
    g.add_argument("--eguard-eval-samples", type=int, default=500,
                   help="disjoint slice for the retrieval-utility report (recall@k, Spearman)")
    g.add_argument("--eguard-alpha", type=float, default=1.0,
                   help="alpha in Eq. 8: weight on the MI term. 0 disables the privacy half")
    g.add_argument("--eguard-epochs", type=int, default=25)
    g.add_argument("--eguard-lr", type=float, default=1e-4,
                   help="the paper's 2e-5 suits a pretrained backbone; g_p from scratch wants more")
    g.add_argument("--eguard-batch-size", type=int, default=64,
                   help="also the negative pool for InfoNCE and the similarity matrix")
    g.add_argument("--eguard-layers", type=int, default=4)
    g.add_argument("--eguard-backbone", default=None,
                   help="None builds g_p from scratch; 'roberta-large' is the paper's 24 layers")
    g.add_argument("--eguard-mi-estimator", default="infonce", choices=["infonce", "mine", "probe"])
    g.add_argument("--eguard-latent-model", default="gte-base",
                   help="g_a, the auxiliary text encoder. Must differ from the victim encoder")
    g.add_argument("--eguard-stochastic", action="store_true",
                   help="variational bottleneck: e' is sampled, and the KL upper-bounds I(e;e')")
    g.add_argument("--eguard-utility-k", type=int, default=10,
                   help="k for the recall@k neighbour-overlap report")

    g = p.add_argument_group("stage 3' — SPARSE defense (defense/sparse.py)")
    g.add_argument("--sparse-checkpoint", default=None,
                   help="path to the fitted mask; default derives one under --output-dir")
    g.add_argument("--sparse-samples", type=int, default=2500,
                   help="victim documents whose embeddings train the mask (targets excluded)")
    g.add_argument("--sparse-eval-samples", type=int, default=500,
                   help="disjoint slice for the retrieval-utility report")
    g.add_argument("--sparse-concept-name", default="entities",
                   help="label for the privacy concept C; provenance only")
    g.add_argument("--sparse-concept-entity-types", default="PERSON,ORG,GPE,DATE",
                   help="spaCy NER labels defining C (Appendix E). Empty string disables NER")
    g.add_argument("--sparse-concept-tokens", default="",
                   help="comma-separated explicit vocabulary for C; unioned with the NER labels. "
                        "The reproducible option — needs no spaCy install")
    g.add_argument("--sparse-spacy-model", default="en_core_web_sm")
    g.add_argument("--sparse-removal", default="delete", choices=["delete", "placeholder", "unk"],
                   help="how R(s,C) builds D- from D-plus; the paper never defines it")
    g.add_argument("--sparse-epsilon-scale", default="absolute", choices=["absolute", "per_dim"],
                   help="'absolute' takes eps literally (Alg. 1: E|Z| ~ n/eps, so eps=10 at n=768 "
                        "puts the noise 77x a unit-norm embedding). 'per_dim' reads eps as a "
                        "per-dimension budget, eps_eff = eps*n, giving E|Z| ~ 1/eps")
    g.add_argument("--sparse-epsilon", type=float, default=10.0,
                   help="privacy budget for the Mahalanobis mechanism. The paper sweeps "
                        "{5,10,20,30,40}; this is the one used unless --sparse-epsilon-sweep is set")
    g.add_argument("--sparse-epsilon-sweep", default=None,
                   help="comma-separated eps values to attack, e.g. '5,10,20,40'. The mask does "
                        "not depend on eps, so one fit covers the whole curve")
    g.add_argument("--sparse-lam", type=float, default=1e-3,
                   help="lambda in Eq. 5: sparsity vs separability. Watch sigma_max in the report")
    g.add_argument("--sparse-epochs", type=int, default=100)
    g.add_argument("--sparse-lr", type=float, default=1e-4,
                   help="Appendix H.2's lr, for the classifier P_theta")
    g.add_argument("--sparse-mask-lr", type=float, default=1e-2,
                   help="separate lr for the gate logits. At the paper's single 1e-4 the mask "
                        "cannot sparsify at all (log alpha travels ~lr x steps, a gate closes at "
                        "-2.4), which makes its own lambda sweep inert. <=0 restores that setup")
    g.add_argument("--sparse-batch-size", type=int, default=64)
    g.add_argument("--sparse-l0-sign", default="standard", choices=["standard", "paper"],
                   help="'paper' reproduces Eq. 4's literal leading minus, which is anti-sparse")
    g.add_argument("--sparse-utility-k", type=int, default=10,
                   help="k for the recall@k neighbour-overlap report")

    g = p.add_argument_group("stage 3'' — CMAG defense (defense/cmag.py)")
    g.add_argument("--cmag-checkpoint", default=None,
                   help="path to the fitted covering; default derives one under --output-dir")
    g.add_argument("--cmag-samples", type=int, default=2500,
                   help="victim documents whose embeddings build the covering (targets excluded)")
    g.add_argument("--cmag-eval-samples", type=int, default=500,
                   help="disjoint slice for the retrieval-utility report")
    g.add_argument("--cmag-group-size", type=int, default=100,
                   help="the paper's top-100 neighbourhoods. A group of m gives a rank-(m-1) "
                        "covariance, so m also decides how much of the space is left unnoised")
    g.add_argument("--cmag-min-group-size", type=int, default=None,
                   help="groups below this are merged into the nearest; default group_size//2")
    g.add_argument("--cmag-epsilon", type=float, default=16.0,
                   help="eps in {1.6, 3.2, ..., 40.0} in the paper's sweep")
    g.add_argument("--cmag-epsilon-sweep", default=None,
                   help="comma-separated eps values to attack, e.g. '1.6,8,16,40'. The covering "
                        "does not depend on eps, so one fit covers the whole curve")
    g.add_argument("--cmag-delta", type=float, default=1e-5,
                   help="delta_j in Theorem 3. The paper never says how to choose it; its code "
                        "uses 1/|X_j|^k with a hardcoded per-eps k table (delta ~ 1e-31 to 1e-58)")
    g.add_argument("--cmag-delta-mode", default="fixed", choices=["fixed", "power"],
                   help="'power' reproduces the released code's delta_j = |X_j|^-k")
    g.add_argument("--cmag-delta-exponent", type=float, default=15.0,
                   help="k, used only when --cmag-delta-mode power")
    g.add_argument("--cmag-variant", default="mahalanobis", choices=["mahalanobis", "euclidean"],
                   help="'euclidean' is the paper's CMAG(E) ablation: U = I and Euclidean d_0")
    g.add_argument("--cmag-u-power", default="sqrt", choices=["sqrt", "inv_sqrt"],
                   help="Def. 6 says U = Sigma^{1/2}; Section 4.2's prose says Sigma^{-1/2}")
    g.add_argument("--cmag-assign", default="centroid", choices=["centroid", "nearest_member"],
                   help="how an unseen embedding picks its group. The paper has no such notion")
    g.add_argument("--cmag-utility-k", type=int, default=10,
                   help="k for the recall@k neighbour-overlap report")

    g = p.add_argument_group("stage 3''' — vec2text §6 defense (defense/vec2text.py)")
    g.add_argument("--vec2text-checkpoint", default=None,
                   help="optional; the defense is stateless, so this is provenance only")
    g.add_argument("--vec2text-samples", type=int, default=2500,
                   help="victim documents used to record the corpus norm scale")
    g.add_argument("--vec2text-eval-samples", type=int, default=500,
                   help="disjoint slice for the retrieval-utility report")
    g.add_argument("--vec2text-noise-level", type=float, default=0.01,
                   help="lambda in phi_noisy(x) = phi(x) + lambda*eps. Table 7's knee is 0.01; "
                        "note Section 6's prose says 0.1, which its own Table 7 shows destroys "
                        "retrieval (NDCG@10 0.302 -> 0.002)")
    g.add_argument("--vec2text-noise-sweep", default=None,
                   help="comma-separated lambda values, e.g. '0,0.001,0.01,0.1,1'. Reproduces "
                        "the axis of Table 7; the defense is stateless so one run covers it")
    g.add_argument("--vec2text-noise-scale", default="absolute", choices=["absolute", "relative"],
                   help="'absolute' takes lambda literally (noise norm = lambda*sqrt(n), which is "
                        "what the paper's GTR-base numbers mean). 'relative' reads lambda as a "
                        "fraction of ||e|| for a non-unit-norm encoder")
    g.add_argument("--vec2text-renormalize", action="store_true",
                   help="project e' back onto the unit sphere. A no-op for retrieval utility, but "
                        "it reproduces the existing --defense gaussian baseline exactly")
    g.add_argument("--vec2text-utility-k", type=int, default=10,
                   help="k for the recall@k neighbour-overlap report")

    g = p.add_argument_group("stage 3'''' — RemoteRAG defense (defense/remote_rag.py)")
    g.add_argument("--remote-rag-checkpoint", default=None,
                   help="optional; the defense is stateless, so this is provenance only")
    g.add_argument("--remote-rag-samples", type=int, default=2500,
                   help="victim documents used to record the corpus norm scale")
    g.add_argument("--remote-rag-eval-samples", type=int, default=500,
                   help="disjoint slice for the retrieval-utility report")
    g.add_argument("--remote-rag-radius", type=float, default=0.05,
                   help="r, the perturbation radius. Table 6 sweeps {0.03,0.05,0.07,0.1}; "
                        "eps = n/r, so r=0.05 at n=768 means eps=15360")
    g.add_argument("--remote-rag-radius-sweep", default=None,
                   help="comma-separated r values, e.g. 0.03,0.05,0.07,0.1 — Table 6 axis")
    g.add_argument("--remote-rag-budget-mode", default="radius",
                   choices=["radius", "per_dim", "absolute"],
                   help="'radius' takes r directly (the paper's own knob). 'per_dim' reads "
                        "--remote-rag-epsilon as eps/n (Figure 2 uses eps = 10n). 'absolute' "
                        "takes eps literally — the reading under which SPARSE's eps=5..40 "
                        "destroys a unit-norm embedding with this same sampler")
    g.add_argument("--remote-rag-epsilon", type=float, default=10.0,
                   help="eps, used by --remote-rag-budget-mode per_dim/absolute")
    g.add_argument("--remote-rag-renormalize", action="store_true",
                   help="project e' back onto the unit sphere; the paper is silent and the "
                        "drift is tiny (norm ~ sqrt(1+r^2))")
    g.add_argument("--remote-rag-utility-k", type=int, default=10,
                   help="k for the recall@k neighbour-overlap report")

    g = p.add_argument_group("stage 4 — attack")
    g.add_argument("--embeddings", default="all", help="'all', a glob, or explicit .npz path(s)")
    g.add_argument("--victim-model", default="gtr-base", help="only attack sets from this encoder")
    g.add_argument("--attack-dataset", default="nfcorpus", help="corpus for pairs + ground truth")
    g.add_argument("--align-samples", type=int, default=200, help="k: known (text, vector) pairs")
    g.add_argument("--reg-lambda", type=float, default=0.1, help="ridge strength")
    g.add_argument("--align-text", choices=["full", "truncated"], default="truncated")
    g.add_argument(
        "--attention-mask",
        choices=["ones", "oracle"],
        default="ones",
        help="'oracle' passes the target text's real mask (leaks its length; "
        "upstream's behaviour). Inert on long documents, live on short ones.",
    )
    g.add_argument("--defense", default="none",
                   help="perturbation applied to victim vectors, and the defense stage 3 fits. "
                        "'eguard' uses g_p; 'sparse' uses the concept mask; the closed-form "
                        "baselines (lapmech, purmech, gaussian, ...) need no stage 3. "
                        "Settable from YAML as `defense: {method: sparse}`")
    g.add_argument("--compare", action="store_true",
                   help="run the attack on e and on e' and print the delta table. The defended "
                        "arm is --defense (eguard when it is left at 'none', for backwards "
                        "compatibility)")
    g.add_argument("--noise-level", type=float, default=0.0)
    g.add_argument("--epsilon", type=float, default=1.0)
    g.add_argument("--show", type=int, default=5)
    g.add_argument("--sweep", action="store_true", help="also grid k x lambda on the largest set")
    g.add_argument("--sweep-k", default="30,100,200")
    g.add_argument("--sweep-lambda", default="0.001,0.01,0.1,1,10")
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
                sections=("pretrain", "finetune", "eguard", "sparse", "cmag", "vec2text",
                          "remote_rag", "defense", "attack"),

                section_name_key={"dataset": "dataset", "model": "model", "defense": "defense"},
            )
        )
        args = parser.parse_args(argv)
    return args


def _free(trainer: Any) -> None:
    import gc

    import torch

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


SPLIT_KEYS = ("align_reserve", "val_samples", "seed")


def recorded_args(run_dir: Path) -> dict[str, Any]:
    args_path = Path(run_dir) / "training_args_and_best_models.json"
    if not args_path.exists():
        return {}
    try:
        return json.loads(args_path.read_text()).get("training_args", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def split_mismatches(run_dir: Path, requested: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    have = recorded_args(run_dir)
    if not have:
        return {}
    out = {}
    for key in SPLIT_KEYS:
        if key in requested and key in have and have[key] != requested[key]:
            out[key] = (have[key], requested[key])
    return out


def assert_reusable(run_dir: Path, requested: dict[str, Any], *, stage: str) -> None:
    bad = split_mismatches(run_dir, requested)
    if not bad:
        return
    lines = "\n".join(
        f"         {k}: checkpoint has {have!r}, this run asks for {want!r}"
        for k, (have, want) in bad.items()
    )
    raise SystemExit(
        f"[stale] {run_dir}\n"
        f"        holds a finished '{stage}' run built with different splits:\n"
        f"{lines}\n"
        f"        The directory name does not distinguish them, so reusing it would\n"
        f"        attack with the old splits while reporting the new settings.\n"
        f"        Either:\n"
        f"          --force                 retrain in place (replaces the cached run)\n"
        f"          --finetune-samples <n>  a different corpus budget names its own\n"
        f"        directory, so both runs coexist and the\n"
        f"        cached stage 1 is still reused\n"
        f"        (worked example: configs/algen_nfcorpus_k1000.yaml)"
    )


def stage_dirs(args: argparse.Namespace) -> dict[str, Path]:
    common = dict(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        weight_decay=1e-5,
    )
    return {
        "pretrain": GeneratorTrainer.run_dir_for(
            dataset=args.pretrain_dataset,
            train_samples=args.pretrain_samples,
            batch_size=args.pretrain_batch_size,
            learning_rate=args.pretrain_lr,
            num_epochs=args.pretrain_epochs,
            **common,
        ),
        "finetune": GeneratorTrainer.run_dir_for(
            dataset=args.finetune_dataset,
            train_samples=args.finetune_samples,
            batch_size=args.finetune_batch_size,
            learning_rate=args.finetune_lr,
            num_epochs=args.finetune_epochs,
            **common,
        ),
    }


def run_training_stage(
    args: argparse.Namespace,
    *,
    name: str,
    dataset: str,
    train_samples: int,
    val_samples: int,
    align_reserve: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    holdout: list[str],
    run_dir: Path,
    init_from: Path | None = None,
) -> Path:
    print(f"\n{'=' * 72}\n[stage: {name}] {dataset} — {train_samples} samples, {epochs} epochs")
    print(f"{'=' * 72}")

    requested = {"align_reserve": align_reserve, "val_samples": val_samples, "seed": args.seed}
    if GeneratorTrainer.is_complete(run_dir) and not args.force:
        assert_reusable(run_dir, requested, stage=name)
        print(f"skip: already finished: {run_dir}\n       (--force to retrain)")
        return run_dir

    splits = build_splits(
        dataset,
        holdout_ids=holdout,
        n_train=train_samples,
        n_val=val_samples,
        n_align=align_reserve,
        seed=args.seed,
    )
    print(f"splits: {splits.summary()}")

    trainer = GeneratorTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        dataset=dataset,
        train_texts=splits.train,
        val_texts=splits.val,
        train_samples=train_samples,
        val_samples=val_samples,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        num_epochs=epochs,
        wandb_run_name=args.wandb_run_name,
        device=args.device,
        holdout_ids=holdout,
        align_reserve=align_reserve,
        seed=args.seed,
    )
    if init_from is not None:
        trainer.init_from(init_from)
    print(f"[run] {trainer.run_dir}")
    trainer.train()
    out = trainer.run_dir
    _free(trainer)
    return out


def victim_paths(args: argparse.Namespace) -> list[Path]:

    paths = find_embedding_sets(args.embeddings)
    found = {p: EmbeddingSet.load(p) for p in paths}

    kept = list(paths)
    if args.victim_model:
        kept = [p for p in kept if found[p].model == args.victim_model]
    corpus = None
    if args.attack_dataset:
        try:
            from dataloader import get_dataset

            corpus = get_dataset(args.attack_dataset).name
        except Exception: 
            corpus = str(args.attack_dataset)
        kept = [p for p in kept if found[p].dataset == corpus]

    dropped = [p for p in paths if p not in kept]
    if dropped:
        print(
            f"skip: {len(dropped)} embedding set(s) not matching "
            f"model={args.victim_model!r} dataset={corpus!r}: "
            + ", ".join(sorted(p.name for p in dropped))
        )
    if not kept:
        inventory = "\n".join(
            f"       {p.name}  model={found[p].model!r} dataset={found[p].dataset!r}"
            for p in paths
        ) or "       (no .npz files found at all)"
        raise SystemExit(
            f"nothing to attack: no embedding set has model={args.victim_model!r} and "
            f"dataset={corpus!r}.\n"
            f"Available:\n{inventory}\n"
            "Adjust --victim-model / --attack-dataset / --embeddings."
        )
    return kept


def eguard_checkpoint_for(args: argparse.Namespace) -> Path:
    if args.eguard_checkpoint:
        return Path(args.eguard_checkpoint)

    dataset_tag = str(args.attack_dataset).replace("/", "-")
    name = (
        f"eguard_{args.victim_model}_{dataset_tag}"
        f"_a{args.eguard_alpha}_{args.eguard_mi_estimator}"
        f"_l{args.eguard_layers}_e{args.eguard_epochs}_n{args.eguard_samples}"
        f"{'_vib' if args.eguard_stochastic else ''}"
        f"{'_' + args.eguard_backbone.replace('/', '-') if args.eguard_backbone else ''}.pt"
    )
    return Path(args.output_dir) / "eguard" / name


def run_eguard_stage(args: argparse.Namespace, holdout: list[str]) -> Path:

    from defense import Eguard

    ckpt = eguard_checkpoint_for(args)
    print(f"\n{'=' * 72}\n[stage: eguard] fit g_p -> {ckpt}\n{'=' * 72}")

    paths = victim_paths(args)
    report_path = ckpt.parent / f"{ckpt.stem}_report.json"
    if ckpt.exists() and not args.force:
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if prior.get("holdout") != len(set(holdout)):
                print(
                    f"Warn: {ckpt.name} was fitted with a holdout of "
                    f"{prior.get('holdout')} id(s) on {prior.get('train_samples')} "
                    f"documents; this run holds out {len(set(holdout))}.\n"
                    f"       The splits have changed since it was trained — "
                    f"--force to refit."
                )
        print(f"Skip: already trained: {ckpt}\n       (--force to refit)")
        return ckpt

    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    print(f"Victim: {embedder}  (manifest of {paths[0].name})")

    dataset = args.attack_dataset
    n_train, n_eval = args.eguard_samples, args.eguard_eval_samples

    from dataloader import get_dataset

    held = set(holdout)
    pool = sum(1 for r in get_dataset(dataset).load().records if r.id not in held)
    if n_train + n_eval > pool:
        n_eval = max(2, min(n_eval, pool // 6))
        n_train = pool - n_eval
        print(
            f"Warn: {dataset} has only {pool} records left after holding out "
            f"{len(held)} attack target(s); shrinking to train={n_train} eval={n_eval}.\n"
            f"       A stale .npz in data/embeddings inflates the holdout — every file "
            f"there counts as a target."
        )
    splits = build_splits(
        dataset, holdout_ids=holdout, n_train=n_train, n_val=n_eval, n_align=0, seed=args.seed
    )
    print(f"Splits: {splits.summary()}")

    print(f"Embed: {len(splits.train)} training + {len(splits.val)} eval documents")
    e_train = embedder.encode(splits.train, show_progress=True)
    e_eval = embedder.encode(splits.val, show_progress=True)

    guard = Eguard(
        embset.dim,
        alpha=args.eguard_alpha,
        epochs=args.eguard_epochs,
        lr=args.eguard_lr,
        batch_size=args.eguard_batch_size,
        num_layers=args.eguard_layers,
        backbone=args.eguard_backbone,
        mi_estimator=args.eguard_mi_estimator,
        latent_model=args.eguard_latent_model,
        stochastic=args.eguard_stochastic,
        seed=args.seed,
        device=args.device,
    )
    print(f"[eguard] {guard}")
    if args.eguard_latent_model == args.victim_model:
        print(
            "Warn: g_a is the victim encoder; L_1 then reduces to decorrelating e' "
            "from e, which any rotation satisfies while protecting nothing."
        )
    guard.fit(e_train, splits.train)
    guard.save(ckpt)
    print(f"[out] {ckpt}")

    # What the defense costs the victim, measured out of sample.
    k = args.eguard_utility_k
    utility = guard.utility_report(e_eval, guard.protect(e_eval), k=k)
    print(f"\n[utility] held-out slice, n={len(splits.val)} — retrieval structure kept by g_p")
    for key, value in utility.items():
        if key != "n":
            print(f"    {key:<22} {value:.4f}")

    # And per target set, since those are the rows the attack actually reads.
    per_set = {}
    for path in paths:
        es = EmbeddingSet.load(path)
        if len(es) < 3:
            continue
        per_set[path.name] = guard.utility_report(es, guard.protect(es), k=min(k, len(es) - 1))

    report = {
        "checkpoint": str(ckpt),
        "victim_model": args.victim_model,
        "dataset": dataset,
        "train_samples": len(splits.train),
        "eval_samples": len(splits.val),
        "holdout": len(held),
        "config": vars(guard.cfg),
        "utility_holdout": utility,
        "utility_per_target_set": per_set,
        "history": guard.history,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[out] {report_path}")
    return ckpt


def _sparse_concept(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:

    def split(raw: Any) -> tuple[str, ...]:
        if raw is None or raw == "" or raw == []:
            return ()
        if isinstance(raw, (list, tuple)):
            return tuple(str(x).strip() for x in raw if str(x).strip())
        return tuple(x.strip() for x in str(raw).split(",") if x.strip())

    return split(args.sparse_concept_entity_types), split(args.sparse_concept_tokens)


def sparse_checkpoint_for(args: argparse.Namespace) -> Path:
    if args.sparse_checkpoint:
        return Path(args.sparse_checkpoint)
    dataset_tag = str(args.attack_dataset).replace("/", "-")
    types, tokens = _sparse_concept(args)
    concept_tag = args.sparse_concept_name or "concept"
    if types:
        concept_tag += "-" + "".join(t[:3].lower() for t in types)
    if tokens:
        concept_tag += f"-tok{len(tokens)}"
    name = (
        f"sparse_{args.victim_model}_{dataset_tag}"
        f"_{concept_tag}_lam{args.sparse_lam}_e{args.sparse_epochs}"
        f"_n{args.sparse_samples}_{args.sparse_removal}"
        f"{'_papersign' if args.sparse_l0_sign == 'paper' else ''}"
        f"{'_paperlr' if args.sparse_mask_lr <= 0 else ''}.pt"
    )
    return Path(args.output_dir) / "sparse" / name


def run_sparse_stage(args: argparse.Namespace, holdout: list[str]) -> Path:
    from defense import Sparse

    ckpt = sparse_checkpoint_for(args)
    print(f"\n{'=' * 72}\n[stage: sparse] fit the concept mask -> {ckpt}\n{'=' * 72}")

    paths = victim_paths(args)
    report_path = ckpt.parent / f"{ckpt.stem}_report.json"
    if ckpt.exists() and not args.force:
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if prior.get("holdout") != len(set(holdout)):
                print(
                    f"Warn: {ckpt.name} was fitted with a holdout of "
                    f"{prior.get('holdout')} id(s) on {prior.get('train_samples')} "
                    f"documents; this run holds out {len(set(holdout))}.\n"
                    f"       The splits have changed since it was trained — "
                    f"--force to refit."
                )
        print(f"[skip] already fitted: {ckpt}\n       (--force to refit)")
        return ckpt

    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    print(f"[victim] {embedder}  (manifest of {paths[0].name})")

    entity_types, concept_tokens = _sparse_concept(args)
    if not entity_types and not concept_tokens:
        raise SystemExit(
            "SPARSE needs a privacy concept C. Set --sparse-concept-entity-types "
            "(spaCy NER labels, e.g. PERSON,ORG,GPE) or --sparse-concept-tokens "
            "(an explicit comma-separated vocabulary, which needs no spaCy install)."
        )
    print(f"[concept] {args.sparse_concept_name!r}: entity_types={list(entity_types)} "
          f"tokens={len(concept_tokens)} removal={args.sparse_removal}")

    dataset = args.attack_dataset
    n_train, n_eval = args.sparse_samples, args.sparse_eval_samples

    from dataloader import get_dataset

    held = set(holdout)
    pool = sum(1 for r in get_dataset(dataset).load().records if r.id not in held)
    if n_train + n_eval > pool:
        n_eval = max(2, min(n_eval, pool // 6))
        n_train = pool - n_eval
        print(
            f"Warn: {dataset} has only {pool} records left after holding out "
            f"{len(held)} attack target(s); shrinking to train={n_train} eval={n_eval}."
        )
    splits = build_splits(
        dataset, holdout_ids=holdout, n_train=n_train, n_val=n_eval, n_align=0, seed=args.seed
    )
    print(f"[splits] {splits.summary()}")

    print(f"Embed: {len(splits.train)} training + {len(splits.val)} eval documents")
    e_train = embedder.encode(splits.train, show_progress=True)
    e_eval = embedder.encode(splits.val, show_progress=True)

    defender = Sparse(
        embset.dim,
        concept_name=args.sparse_concept_name,
        concept_entity_types=entity_types,
        concept_tokens=concept_tokens,
        spacy_model=args.sparse_spacy_model,
        removal=args.sparse_removal,
        epsilon=args.sparse_epsilon,
        epsilon_scale=args.sparse_epsilon_scale,
        lam=args.sparse_lam,
        epochs=args.sparse_epochs,
        lr=args.sparse_lr,
        mask_lr=None if args.sparse_mask_lr <= 0 else args.sparse_mask_lr,
        batch_size=args.sparse_batch_size,
        l0_sign=args.sparse_l0_sign,
        victim_model=args.victim_model,
        dataset=dataset,
        seed=args.seed,
        device=args.device,
    )
    print(f"[sparse] {defender}")
    if defender.cfg.mask_lr is None:
        print(
            "Warn: mask_lr is off, reproducing the paper's single lr=1e-4 over both the\n"
            "       classifier and the gates. log alpha then travels ~lr x steps, far short of\n"
            "       the -2.4 a gate needs to close, so the mask will not sparsify and Sigma\n"
            "       stays isotropic i.e. this run is LapMech wearing a mask. Check sigma_max."
        )

    from defense import ConceptExtractor, build_concept_pairs

    pairs = build_concept_pairs(splits.train, ConceptExtractor(defender.cfg), verbose=True)
    coverage = len(pairs) / max(len(splits.train), 1)
    if coverage < 0.05:
        print(
            f"Warn: only {len(pairs)}/{len(splits.train)} ({coverage:.1%}) training documents\n"
            f"       contain concept {args.sparse_concept_name!r}. The mask is being fitted on\n"
            f"       very few pairs and will mostly reflect noise. Widen the concept."
        )
    defender.fit(e_train, splits.train, pairs=pairs, encoder=embedder, verbose=True)
    defender.save(ckpt)
    print(f"[out] {ckpt}")

    privacy = defender.privacy_report()
    print(f"\n[mask] what Sigma does — sigma_max is the number that matters")
    for key in ("active_dims", "active_fraction", "sigma_min", "sigma_max",
                "separability_val_acc", "epsilon_effective", "expected_noise_norm",
                "expected_noise_norm_lapmech", "snr_vs_unit_norm"):
        value = privacy.get(key)
        if isinstance(value, float):
            print(f"    {key:<28} {value:.4f}")
        else:
            print(f"    {key:<28} {value}")
    if privacy["snr_vs_unit_norm"] < 0.5:
        print(
            f"Warn: the noise is {1 / privacy['snr_vs_unit_norm']:.0f}x the norm of a unit-length\n"
            f"       embedding at eps={args.sparse_epsilon:g}. Alg. 1 draws the radius from\n"
            f"       Gamma(n, 1/eps), so E|Z| ~ n/eps and the paper's eps range lands far outside\n"
            f"       what a normalised sentence vector survives. Use --sparse-epsilon-scale\n"
            f"       per_dim to read eps as a per-dimension budget."
        )
    if privacy["sigma_max"] < 1.5:
        print(
            "Warn: sigma_max < 1.5: the noise is spread almost evenly across dimensions, so\n"
            "       this is LapMech in all but name. Raise --sparse-lam or --sparse-mask-lr."
        )

    sweep = _sparse_epsilons(args)
    k = args.sparse_utility_k
    utility_by_eps = {}
    print(f"\n[utility] held-out slice, n={len(splits.val)} — retrieval structure kept by e'")
    for eps in sweep:
        u = defender.utility_report(
            e_eval, defender.protect(e_eval, epsilon=eps, seed=args.seed), k=k
        )
        utility_by_eps[str(eps)] = u
        print(f"    eps={eps:<6} " + "  ".join(
            f"{key}={value:.4f}" for key, value in u.items() if key != "n"
        ))
    utility = utility_by_eps[str(args.sparse_epsilon)] if str(args.sparse_epsilon) in utility_by_eps \
        else next(iter(utility_by_eps.values()))

    per_set = {}
    for path in paths:
        es = EmbeddingSet.load(path)
        if len(es) < 3:
            continue
        per_set[path.name] = defender.utility_report(
            es, defender.protect(es, epsilon=args.sparse_epsilon, seed=args.seed),
            k=min(k, len(es) - 1),
        )

    report = {
        "checkpoint": str(ckpt),
        "defense": "sparse",
        "victim_model": args.victim_model,
        "dataset": dataset,
        "train_samples": len(splits.train),
        "eval_samples": len(splits.val),
        "holdout": len(held),
        "config": vars(defender.cfg),
        "concept_pairs": pairs.summary(),
        "concept_coverage": coverage,
        "privacy_report": privacy,
        "utility_holdout": utility,
        "utility_by_epsilon": utility_by_eps,
        "utility_per_target_set": per_set,
        "history": defender.history,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[out] {report_path}")
    return ckpt


def cmag_checkpoint_for(args: argparse.Namespace) -> Path:
    if args.cmag_checkpoint:
        return Path(args.cmag_checkpoint)
    dataset_tag = str(args.attack_dataset).replace("/", "-")
    delta_tag = (
        f"d{args.cmag_delta:g}" if args.cmag_delta_mode == "fixed"
        else f"dpow{args.cmag_delta_exponent:g}"
    )
    name = (
        f"cmag_{args.victim_model}_{dataset_tag}"
        f"_{args.cmag_variant}_g{args.cmag_group_size}_{delta_tag}"
        f"_n{args.cmag_samples}"
        f"{'_invsqrt' if args.cmag_u_power == 'inv_sqrt' else ''}"
        f"{'_' + args.cmag_assign if args.cmag_assign != 'centroid' else ''}.pt"
    )
    return Path(args.output_dir) / "cmag" / name


def _cmag_epsilons(args: argparse.Namespace) -> list[float]:
    raw = args.cmag_epsilon_sweep
    if raw is None or raw == "" or raw == []:
        return [float(args.cmag_epsilon)]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def run_cmag_stage(args: argparse.Namespace, holdout: list[str]) -> Path:
    from defense import Cmag

    ckpt = cmag_checkpoint_for(args)
    print(f"\n{'=' * 72}\n[stage: cmag] build the covering -> {ckpt}\n{'=' * 72}")

    paths = victim_paths(args)
    report_path = ckpt.parent / f"{ckpt.stem}_report.json"
    if ckpt.exists() and not args.force:
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if prior.get("holdout") != len(set(holdout)):
                print(
                    f"Warn: {ckpt.name} was fitted with a holdout of "
                    f"{prior.get('holdout')} id(s) on {prior.get('train_samples')} "
                    f"documents; this run holds out {len(set(holdout))}.\n"
                    f"       The splits have changed since it was fitted — "
                    f"--force to refit."
                )
        print(f"[skip] already fitted: {ckpt}\n       (--force to refit)")
        return ckpt

    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    print(f"[victim] {embedder}  (manifest of {paths[0].name})")

    dataset = args.attack_dataset
    n_train, n_eval = args.cmag_samples, args.cmag_eval_samples

    from dataloader import get_dataset

    held = set(holdout)
    pool = sum(1 for r in get_dataset(dataset).load().records if r.id not in held)
    if n_train + n_eval > pool:
        n_eval = max(2, min(n_eval, pool // 6))
        n_train = pool - n_eval
        print(
            f"Warn: {dataset} has only {pool} records left after holding out "
            f"{len(held)} attack target(s); shrinking to train={n_train} eval={n_eval}."
        )
    splits = build_splits(
        dataset, holdout_ids=holdout, n_train=n_train, n_val=n_eval, n_align=0, seed=args.seed
    )
    print(f"[splits] {splits.summary()}")

    print(f"Embed: {len(splits.train)} covering + {len(splits.val)} eval documents")
    e_train = embedder.encode(splits.train, show_progress=True)
    e_eval = embedder.encode(splits.val, show_progress=True)

    if args.cmag_group_size > len(splits.train):
        raise SystemExit(
            f"--cmag-group-size {args.cmag_group_size} exceeds the {len(splits.train)} "
            f"documents available to build the covering from. Lower it, or raise "
            f"--cmag-samples."
        )

    defender = Cmag(
        embset.dim,
        group_size=args.cmag_group_size,
        min_group_size=args.cmag_min_group_size,
        epsilon=args.cmag_epsilon,
        delta=args.cmag_delta,
        delta_mode=args.cmag_delta_mode,
        delta_exponent=args.cmag_delta_exponent,
        variant=args.cmag_variant,
        u_power=args.cmag_u_power,
        assign=args.cmag_assign,
        victim_model=args.victim_model,
        dataset=dataset,
        seed=args.seed,
        device=args.device,
    )
    print(f"[cmag] {defender}")
    if args.cmag_u_power == "inv_sqrt":
        print(
            "Warn: u_power=inv_sqrt follows Section 4.2's prose, which contradicts\n"
            "       Definition 6, Algorithm 4 and the released code. It WHITENS instead of\n"
            "       matching the neighbourhood covariance, i.e. it puts the noise where\n"
            "       neighbours do not vary. See defense/cmag.py AMBIGUITIES['u_direction']."
        )
    defender.fit(e_train, verbose=True)
    defender.save(ckpt)
    print(f"[out] {ckpt}")

    privacy = defender.privacy_report()
    print("\n[covering] what the mechanism does — the span fraction is the number to read")
    for key in ("n_groups", "group_size_min", "group_size_mean", "group_size_max",
                "sigma_rank_mean", "noise_energy_in_span", "untouched_fraction",
                "d0_mean", "d0_max", "sigma_mean", "sigma_max",
                "expected_noise_norm", "snr_vs_unit_norm"):
        value = privacy.get(key)
        print(f"    {key:<24} {value:.4f}" if isinstance(value, float)
              else f"    {key:<24} {value}")
    if privacy["noise_energy_in_span"] > 0.99:
        print(
            f"Warn: {privacy['noise_energy_in_span']:.4%} of the noise energy sits in the\n"
            f"       rank-{privacy['sigma_rank_mean']:.0f} neighbourhood span, so roughly\n"
            f"       {privacy['untouched_fraction']:.0%} of the {privacy['dim']} coordinates leave the\n"
            f"       trust boundary almost unperturbed. That is the mechanism as specified,\n"
            f"       not a bug — but an inversion attacker keeps that subspace. Raise\n"
            f"       --cmag-group-size to shrink it."
        )
    if privacy["snr_vs_unit_norm"] < 0.5:
        print(
            f"Warn: the noise is {1 / privacy['snr_vs_unit_norm']:.0f}x the norm of a unit-length\n"
            f"       embedding at eps={args.cmag_epsilon:g}, delta={args.cmag_delta:g}. The paper's eps\n"
            f"       sweep was calibrated on unnormalised SimCSE vectors. Raise --cmag-epsilon\n"
            f"       or --cmag-delta."
        )

    sweep = _cmag_epsilons(args)
    k = args.cmag_utility_k
    utility_by_eps = {}
    print(f"\n[utility] held-out slice, n={len(splits.val)} — retrieval structure kept by e'")
    for eps in sweep:
        u = defender.utility_report(
            e_eval, defender.protect(e_eval, epsilon=eps, seed=args.seed), k=k
        )
        utility_by_eps[str(eps)] = u
        print(f"    eps={eps:<6} " + "  ".join(
            f"{key}={value:.4f}" for key, value in u.items() if key != "n"
        ))
    utility = (
        utility_by_eps[str(args.cmag_epsilon)] if str(args.cmag_epsilon) in utility_by_eps
        else next(iter(utility_by_eps.values()))
    )

    report = {
        "checkpoint": str(ckpt),
        "holdout": len(set(holdout)),
        "train_samples": len(splits.train),
        "eval_samples": len(splits.val),
        "privacy": privacy,
        "utility_holdout": utility,
        "utility_by_epsilon": utility_by_eps,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"out: {report_path}")
    return ckpt


def vec2text_checkpoint_for(args: argparse.Namespace) -> Path:

    if args.vec2text_checkpoint:
        return Path(args.vec2text_checkpoint)
    dataset_tag = str(args.attack_dataset).replace("/", "-")
    name = (
        f"vec2text_{args.victim_model}_{dataset_tag}_{args.vec2text_noise_scale}"
        f"_n{args.vec2text_samples}"
        f"{'_renorm' if args.vec2text_renormalize else ''}.pt"
    )
    return Path(args.output_dir) / "vec2text" / name


def _vec2text_noise_levels(args: argparse.Namespace) -> list[float]:
    raw = args.vec2text_noise_sweep
    if raw is None or raw == "" or raw == []:
        return [float(args.vec2text_noise_level)]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def run_vec2text_stage(args: argparse.Namespace, holdout: list[str]) -> Path:

    from defense import Vec2TextDefense

    ckpt = vec2text_checkpoint_for(args)
    print(f"\n{'=' * 72}\n[stage: vec2text] measure the cost of phi_noisy -> {ckpt}\n{'=' * 72}")

    paths = victim_paths(args)
    report_path = ckpt.parent / f"{ckpt.stem}_report.json"
    if ckpt.exists() and not args.force:
        print(f"[skip] already recorded: {ckpt}\n       (--force to redo)")
        return ckpt

    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    print(f"[victim] {embedder}  (manifest of {paths[0].name})")

    dataset = args.attack_dataset
    n_train, n_eval = args.vec2text_samples, args.vec2text_eval_samples

    from dataloader import get_dataset

    held = set(holdout)
    pool = sum(1 for r in get_dataset(dataset).load().records if r.id not in held)
    if n_train + n_eval > pool:
        n_eval = max(2, min(n_eval, pool // 6))
        n_train = pool - n_eval
        print(
            f"Warn: {dataset} has only {pool} records left after holding out "
            f"{len(held)} attack target(s); shrinking to train={n_train} eval={n_eval}."
        )
    splits = build_splits(
        dataset, holdout_ids=holdout, n_train=n_train, n_val=n_eval, n_align=0, seed=args.seed
    )
    print(f"[splits] {splits.summary()}")

    print(f"Embed: {len(splits.train)} scale + {len(splits.val)} eval documents")
    e_train = embedder.encode(splits.train, show_progress=True)
    e_eval = embedder.encode(splits.val, show_progress=True)

    defender = Vec2TextDefense(
        embset.dim,
        noise_level=args.vec2text_noise_level,
        noise_scale=args.vec2text_noise_scale,
        renormalize_output=args.vec2text_renormalize,
        victim_model=args.victim_model,
        dataset=dataset,
        seed=args.seed,
        device=args.device,
    )
    defender.fit(e_train, verbose=True)
    defender.save(ckpt)
    print(f"[vec2text] {defender}")
    print(f"[out] {ckpt}")
    if args.vec2text_renormalize:
        print(
            "[note] --vec2text-renormalize reproduces the existing --defense gaussian\n"
            "       baseline. It is a no-op for retrieval utility (scaling cannot change\n"
            "       direction); it only changes the raw magnitudes the attacker sees."
        )

    sweep = _vec2text_noise_levels(args)
    k = args.vec2text_utility_k
    utility_by_lambda = {}
    print(f"\n[utility] held-out slice, n={len(splits.val)} — retrieval structure kept by e'")
    print(f"    {'lambda':>8} {'noise/signal':>13}  " + "  ".join(
        f"{n}" for n in ("recall@k", "pearson", "spearman", "abs_sim_err")))
    for lam in sweep:
        u = defender.utility_report(
            e_eval, defender.protect(e_eval, noise_level=lam, seed=args.seed), k=k
        )
        utility_by_lambda[str(lam)] = u
        rep = defender.privacy_report(noise_level=lam)
        rec = next(v for key, v in u.items() if key.startswith("recall_at_"))
        print(f"    {lam:>8g} {rep['noise_to_signal']:>13.4f}  {rec:>8.4f}  "
              f"{u['sim_pearson']:>7.4f}  {u['sim_spearman']:>8.4f}  "
              f"{u['mean_abs_sim_error']:>11.4f}")
    utility = (
        utility_by_lambda[str(args.vec2text_noise_level)]
        if str(args.vec2text_noise_level) in utility_by_lambda
        else next(iter(utility_by_lambda.values()))
    )

    report = {
        "checkpoint": str(ckpt),
        "holdout": len(set(holdout)),
        "train_samples": len(splits.train),
        "eval_samples": len(splits.val),
        "privacy": defender.privacy_report(),
        "utility_holdout": utility,

        "utility_by_epsilon": utility_by_lambda,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[out] {report_path}")
    return ckpt


def remote_rag_checkpoint_for(args: argparse.Namespace) -> Path:
    if args.remote_rag_checkpoint:
        return Path(args.remote_rag_checkpoint)
    dataset_tag = str(args.attack_dataset).replace("/", "-")
    name = (
        f"remote_rag_{args.victim_model}_{dataset_tag}_{args.remote_rag_budget_mode}"
        f"_n{args.remote_rag_samples}"
        f"{'_renorm' if args.remote_rag_renormalize else ''}.pt"
    )
    return Path(args.output_dir) / "remote_rag" / name


def _remote_rag_radii(args: argparse.Namespace) -> list[float]:
    raw = args.remote_rag_radius_sweep
    if raw is None or raw == "" or raw == []:
        return [float(args.remote_rag_radius)]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def run_remote_rag_stage(args: argparse.Namespace, holdout: list[str]) -> Path:
    from defense import RemoteRag

    ckpt = remote_rag_checkpoint_for(args)
    print(f"\n{'=' * 72}\n[stage: remote_rag] cost of the perturbation -> {ckpt}\n{'=' * 72}")

    paths = victim_paths(args)
    report_path = ckpt.parent / f"{ckpt.stem}_report.json"
    if ckpt.exists() and not args.force:
        print(f"[skip] already recorded: {ckpt}\n       (--force to redo)")
        return ckpt

    embset = EmbeddingSet.load(paths[0])
    embedder = victim_embedder(embset, **({"device": args.device} if args.device else {}))
    print(f"[victim] {embedder}  (manifest of {paths[0].name})")

    dataset = args.attack_dataset
    n_train, n_eval = args.remote_rag_samples, args.remote_rag_eval_samples

    from dataloader import get_dataset

    held = set(holdout)
    pool = sum(1 for r in get_dataset(dataset).load().records if r.id not in held)
    if n_train + n_eval > pool:
        n_eval = max(2, min(n_eval, pool // 6))
        n_train = pool - n_eval
        print(
            f"Warn: {dataset} has only {pool} records left after holding out "
            f"{len(held)} attack target(s); shrinking to train={n_train} eval={n_eval}."
        )
    splits = build_splits(
        dataset, holdout_ids=holdout, n_train=n_train, n_val=n_eval, n_align=0, seed=args.seed
    )
    print(f"[splits] {splits.summary()}")

    print(f"Embed: {len(splits.train)} scale + {len(splits.val)} eval documents")
    e_train = embedder.encode(splits.train, show_progress=True)
    e_eval = embedder.encode(splits.val, show_progress=True)

    defender = RemoteRag(
        embset.dim,
        budget_mode=args.remote_rag_budget_mode,
        radius=args.remote_rag_radius,
        epsilon=args.remote_rag_epsilon,
        renormalize_output=args.remote_rag_renormalize,
        victim_model=args.victim_model,
        dataset=dataset,
        seed=args.seed,
        device=args.device,
    )
    defender.fit(e_train, verbose=True)
    defender.save(ckpt)
    print(f"remote_rag: {defender}")
    print(f"out: {ckpt}")

    sweep = _remote_rag_radii(args)
    k = args.remote_rag_utility_k
    utility_by_radius = {}
    print(f"\nutility: held-out slice, n={len(splits.val)} — retrieval structure kept by e'")
    print(f"    {'r':>8} {'eps = n/r':>10} {'angle deg':>10}  {'recall@k':>8}  "
          f"{'pearson':>7}  {'spearman':>8}")
    for r in sweep:
        u = defender.utility_report(
            e_eval, defender.protect(e_eval, radius=r, seed=args.seed), k=k
        )
        utility_by_radius[str(r)] = u
        rep = defender.privacy_report(radius=r)
        rec = next(v for key, v in u.items() if key.startswith("recall_at_"))
        print(f"    {r:>8g} {rep['epsilon']:>10.0f} {rep['expected_angle_shift_deg']:>10.2f}  "
              f"{rec:>8.4f}  {u['sim_pearson']:>7.4f}  {u['sim_spearman']:>8.4f}")
    utility = (
        utility_by_radius[str(args.remote_rag_radius)]
        if str(args.remote_rag_radius) in utility_by_radius
        else next(iter(utility_by_radius.values()))
    )

    report = {
        "checkpoint": str(ckpt),
        "holdout": len(set(holdout)),
        "train_samples": len(splits.train),
        "eval_samples": len(splits.val),
        "privacy": defender.privacy_report(),
        "utility_holdout": utility,
        "utility_by_epsilon": utility_by_radius,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[out] {report_path}")
    return ckpt


def _sparse_epsilons(args: argparse.Namespace) -> list[float]:
    raw = args.sparse_epsilon_sweep
    if raw is None or raw == "" or raw == []:
        return [float(args.sparse_epsilon)]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).split(",") if str(x).strip()]


def _knob_note(defense: str, eps: float | None) -> str:
    if eps is None or defense in ("none", "", None):
        return ""
    if defense == "remote_rag":
        return f" r={eps:g}"
    if defense == "vec2text":
        return f" lambda={eps:g}"
    if defense in ("sparse", "cmag", "lapmech", "purmech", "dp_gaussian"):
        return f" epsilon={eps:g}"
    if defense == "gaussian":
        return f" sigma={eps:g}"
    return ""


def _attack_arm(
    args: argparse.Namespace,
    checkpoint: Path,
    paths: list[Path],
    *,
    label: str,
    defense: str,
    eguard_checkpoint: Path | None,
    sparse_checkpoint: Path | None = None,
    cmag_checkpoint: Path | None = None,
    vec2text_checkpoint: Path | None = None,
    remote_rag_checkpoint: Path | None = None,
    epsilon: float | None = None,
    out_root: Path,
) -> list[dict]:
    eps = args.epsilon if epsilon is None else epsilon
    eps_note = _knob_note(defense, eps)
    print(f"\n{'-' * 72}\n[arm: {label}] defense={defense}{eps_note}\n{'-' * 72}")
    attacker = get_attack(
        "algen",
        checkpoint_dir=checkpoint,
        dataset=args.attack_dataset,
        align_samples=args.align_samples,
        reg_lambda=args.reg_lambda,
        seed=args.seed,
        device=args.device,
        align_text=args.align_text,
        attention_mask=args.attention_mask,
        defense=defense,
        noise_level=args.noise_level,
        epsilon=eps,
        eguard_checkpoint=eguard_checkpoint,
        sparse_checkpoint=sparse_checkpoint,
        cmag_checkpoint=cmag_checkpoint,
        vec2text_checkpoint=vec2text_checkpoint,
        vec2text_noise_level=eps if defense == "vec2text" else None,
        remote_rag_checkpoint=remote_rag_checkpoint,
        remote_rag_radius=eps if defense == "remote_rag" else None,
    )

    mask_tag = "" if args.attention_mask == "ones" else f"_mask{args.attention_mask}"
    defense_tag = ""
    if defense != "none":
        defense_tag = f"_{defense}"
        if defense == "eguard" and eguard_checkpoint is not None:
            defense_tag += "_" + Path(eguard_checkpoint).stem.removeprefix("eguard_")
        if defense == "sparse":
            if sparse_checkpoint is not None:
                defense_tag += "_" + Path(sparse_checkpoint).stem.removeprefix("sparse_")
            defense_tag += f"_eps{eps}"
        if defense == "cmag":
            if cmag_checkpoint is not None:
                defense_tag += "_" + Path(cmag_checkpoint).stem.removeprefix("cmag_")
            defense_tag += f"_eps{eps}"
        if defense == "vec2text":
            if vec2text_checkpoint is not None:
                defense_tag += "_" + Path(vec2text_checkpoint).stem.removeprefix("vec2text_")
            defense_tag += f"_lam{eps}"
        if defense == "remote_rag":
            if remote_rag_checkpoint is not None:
                defense_tag += "_" + Path(remote_rag_checkpoint).stem.removeprefix("remote_rag_")
            defense_tag += f"_r{eps}"

    summary = []
    for path in paths:
        tag = (
            f"attack_{path.stem}_align{args.align_samples}_ridge{args.reg_lambda}"
            f"_{args.align_text}{mask_tag}{defense_tag}"
        )
        result = attacker.attack_file(path, out_dir=out_root / tag)
        result.show(args.show)
        summary.append(result.to_dict())

    _free(attacker)
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
        f"{'rougeL':>8} {'oracF1':>7} {'align_cos':>10}"
    )
    for s_ in summary:
        print(
            f"  {s_['source'][:46]:<46} {s_['n']:>4} "
            f"{s_['test_results'].get('token_f1', float('nan')):>7.4f} "
            f"{s_['test_results'].get('token_recall', float('nan')):>7.4f} "
            f"{s_['test_results']['bleu']:>7.2f} "
            f"{s_['test_results']['rougeL']:>8.4f} "
            f"{s_['oracle_results'].get('token_f1', float('nan')):>7.4f} "
            f"{s_['diagnostics']['X_Y_test_COS']:>10.4f}"
        )


def _pct_change(base: float, new: float) -> str:
    """Relative change, with the degenerate cases spelled out rather than shown as inf."""
    if base == 0:
        return "  n/a" if new == 0 else "  new"
    return f"{(new - base) / abs(base) * 100:+5.0f}%"


#: Rows of the compare table: ``(display name, AttackResult section, key)``.
DEFAULT_COMPARE_METRICS = [
    ("token_f1", "test_results", "token_f1"),
    ("token_recall", "test_results", "token_recall"),
    ("bleu", "test_results", "bleu"),
    ("rougeL", "test_results", "rougeL"),
    ("align_cos", "diagnostics", "X_Y_test_COS"),
]


def _print_comparison(
    baseline: list[dict],
    defended: list[dict],
    utility: dict[str, Any] | None,
    *,
    defense_label: str = "Eguard",
    metrics: list[tuple[str, str, str]] | None = None,
) -> None:
    by_source = {s_["source"]: s_ for s_ in defended}
    metrics = metrics or DEFAULT_COMPARE_METRICS

    has_align = any(name == "align_cos" for name, _, _ in metrics)
    fixed = (
        "same generator, same k, same ridge, same seed" if has_align
        else "same decoder, same D_L, same D_S, same seed"
    )
    print(f"\n{'=' * 96}")
    print(f"[compare] attack quality against e vs e' ({defense_label}) — {fixed}")
    print(f"{'=' * 96}")
    defended_col = f"e' ({defense_label})"
    print(
        f"  {'embedding set':<46} {'n':>3}  {'metric':<18} "
        f"{'e (no defense)':>14} {defended_col:>12} {'change':>8}"
    )

    for base in baseline:
        src = base["source"]
        dfd = by_source.get(src)
        if dfd is None:
            continue
        first = True
        for name, section, key in metrics:
            b = float(base[section].get(key, float("nan")))
            d = float(dfd[section].get(key, float("nan")))
            head = f"  {src[:46]:<46} {base['n']:>3}" if first else " " * 51
            print(f"{head}  {name:<18} {b:>14.4f} {d:>12.4f} {_pct_change(b, d):>8}")
            first = False
        print()

    def mean(rows: list[dict], section: str, key: str) -> float:
        vals = [float(r[section][key]) for r in rows if key in r[section]]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"  {'MEAN over ' + str(len(baseline)) + ' set(s)':<46} {'':>3}")
    for name, section, key in metrics:
        b, d = mean(baseline, section, key), mean(defended, section, key)
        print(f"{' ' * 51}  {name:<18} {b:>14.4f} {d:>12.4f} {_pct_change(b, d):>8}")

    if utility:
        u = utility.get("utility_holdout", {})
        print(
            f"\n  [utility] what e' costs the victim — held-out slice, "
            f"n={int(u.get('n', 0))}, out of sample"
        )
        for key, value in u.items():
            if key != "n":
                print(f"{' ' * 51}  {key:<18} {value:>14.4f}")

    print(
        "\n  A drop in bleu/rougeL is the defense working; a drop in recall@k / spearman is\n"
        "  what it cost.\n"
        + (
            "  align_cos is the diagnostic: it says how much of the loss came from the\n"
            "  linear map failing to land in G's space, rather than from the generator.\n"
            if has_align else
            "  recon_cos is the diagnostic: it says whether the recovered text still lands\n"
            "  near the original vector, which a drop in rougeL alone does not tell you.\n"
        )
        + "  Sets with n < 16 carry error bars wider than the differences between them."
    )
    if defense_label.lower().startswith("sparse"):
        print(
            "\n  Read align_cos carefully here. Eguard is a deterministic map, so a low\n"
            "  align_cos means the ridge solve failed to find it; SPARSE is additive noise\n"
            "  redrawn per call, so there is no map to find and align_cos falls for a\n"
            "  different reason — the pairs and the targets carry independent noise. A\n"
            "  defense that survives refitting is not the same result as one that does not."
        )


def run_attack_stage(args: argparse.Namespace, checkpoint: Path) -> dict[str, list[dict]]:
    print(f"\n{'=' * 72}\n[stage: attack] {checkpoint}\n{'=' * 72}")
    if not GeneratorTrainer.is_complete(checkpoint):
        raise SystemExit(
            f"no usable generator at {checkpoint}\n"
            "Run the training stages first: python train_algen.py --stages pretrain,finetune"
        )

    paths = victim_paths(args)
    out_root = Path(checkpoint) / "attacks"
    eguard_ckpt = eguard_checkpoint_for(args)
    sparse_ckpt = sparse_checkpoint_for(args)
    cmag_ckpt = cmag_checkpoint_for(args)
    vec2text_ckpt = vec2text_checkpoint_for(args)
    remote_rag_ckpt = remote_rag_checkpoint_for(args)


    defended = args.defense if args.defense not in ("none", "") else "eguard"
    if defended == "sparse":
        epsilons = _sparse_epsilons(args)
    elif defended == "cmag":
        epsilons = _cmag_epsilons(args)
    elif defended == "vec2text":
        epsilons = _vec2text_noise_levels(args)
    elif defended == "remote_rag":
        epsilons = _remote_rag_radii(args)
    else:
        epsilons = [args.epsilon]

    if args.compare:
        if args.defense in ("none", ""):
            print(f"[note] --compare with no --defense; defending with {defended!r}.")
        arms = [("baseline", "none", None)]
        arms += [
            (defended if len(epsilons) == 1 else f"{defended}_eps{eps:g}", defended, eps)
            for eps in epsilons
        ]
    else:
        arms = [
            (args.defense if len(epsilons) == 1 else f"{args.defense}_eps{eps:g}",
             args.defense, eps)
            for eps in epsilons
        ]

    needed = {d for _, d, _ in arms}
    if "eguard" in needed and not eguard_ckpt.exists():
        raise SystemExit(
            f"defense 'eguard' needs a trained projection network, none at {eguard_ckpt}\n"
            "Fit it first: python train_algen.py --stages eguard"
        )
    if "sparse" in needed and not sparse_ckpt.exists():
        raise SystemExit(
            f"defense 'sparse' needs a fitted concept mask, none at {sparse_ckpt}\n"
            "Fit it first: python train_algen.py --stages sparse"
        )
    if "cmag" in needed and not cmag_ckpt.exists():
        raise SystemExit(
            f"defense 'cmag' needs a fitted covering, none at {cmag_ckpt}\n"
            "Fit it first: python train_algen.py --stages cmag"
        )

    results: dict[str, list[dict]] = {}
    for label, defense, eps in arms:
        results[label] = _attack_arm(
            args, checkpoint, paths,
            label=label, defense=defense,
            eguard_checkpoint=eguard_ckpt if defense == "eguard" else None,
            sparse_checkpoint=sparse_ckpt if defense == "sparse" else None,
            cmag_checkpoint=cmag_ckpt if defense == "cmag" else None,
            vec2text_checkpoint=(
                vec2text_ckpt if defense == "vec2text" and vec2text_ckpt.exists() else None
            ),
            remote_rag_checkpoint=(
                remote_rag_ckpt
                if defense == "remote_rag" and remote_rag_ckpt.exists() else None
            ),
            epsilon=eps,
            out_root=out_root,
        )

    out_root.mkdir(parents=True, exist_ok=True)
    for label, summary in results.items():
        suffix = "" if len(results) == 1 else f"_{label}"
        with open(out_root / f"pipeline_summary{suffix}.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n{'=' * 72}\n[results] {out_root}\n{'=' * 72}")
    arm_meta = {lbl: (d, e) for lbl, d, e in arms}
    for label, summary in results.items():
        d, e = arm_meta.get(label, (None, None))
        _print_arm_table(label, summary, defense=d, knob=_knob_note(d, e) if d else "")

    if args.compare:
        ckpt = {"sparse": sparse_ckpt, "eguard": eguard_ckpt, "cmag": cmag_ckpt,
                "vec2text": vec2text_ckpt if vec2text_ckpt.exists() else None,
                "remote_rag": (
                    remote_rag_ckpt if remote_rag_ckpt.exists() else None
                )}.get(defended)
        report_path = ckpt.parent / f"{ckpt.stem}_report.json" if ckpt is not None else None
        utility = (
            json.loads(report_path.read_text())
            if report_path is not None and report_path.exists()
            else None
        )

        for label, _, eps in arms[1:]:
            u = utility
            if u and defended in ("sparse", "cmag", "vec2text", "remote_rag") \
                    and eps is not None:
                by_eps = u.get("utility_by_epsilon", {})
                if str(eps) in by_eps:
                    u = {**u, "utility_holdout": by_eps[str(eps)]}
                else:
                    print(
                        f"\n  Warn: no utility measured at {defended} "
                        f"{_knob_note(defended, eps).strip() or eps}. The report holds "
                        f"{sorted(by_eps) or 'nothing'}, written when the defense was fit.\n"
                        f"         The [utility] block below is for THAT value, not this "
                        f"arm's — do not read it as this arm's cost.\n"
                        f"         Refit with `--stages {defended} --force` at this knob, "
                        f"or solve it with `python -m metrics.utility_match "
                        f"--defense {defended} --target-recall <r>`."
                    )
                    u = {**u, "utility_holdout": {
                        **u.get("utility_holdout", {}), "STALE_measured_at": sorted(by_eps)}}
            pretty = DEFENSE_LABELS.get(defended, defended.capitalize()) + (
                f", r={eps:g}" if defended == "remote_rag"
                else f", lambda={eps:g}" if defended == "vec2text"
                else f", eps={eps:g}" if defended in ("sparse", "cmag") else ""
            )
            _print_comparison(results["baseline"], results[label], u, defense_label=pretty)

        if defended in ("sparse", "cmag", "vec2text", "remote_rag") and len(arms) > 2:
            _print_epsilon_curve(results, arms, utility, defense=defended)

        stem = ckpt.stem.removeprefix(defended + "_") if ckpt is not None else defended
        comparison_name = f"comparison_{stem}.json"
        with open(out_root / comparison_name, "w") as f:
            json.dump(
                {"defense": defended, "baseline": results["baseline"],
                 "defended": {label: results[label] for label, _, _ in arms[1:]},
                 "defense_report": utility},
                f, indent=2,
            )
        print(f"\n  [out] {out_root / comparison_name}")

    return results


def _print_epsilon_curve(
    results: dict[str, list[dict]],
    arms: list[tuple[str, str, float | None]],
    utility: dict[str, Any] | None,
    defense: str = "sparse",
) -> None:

    def mean(rows: list[dict], section: str, key: str) -> float:
        vals = [float(r[section][key]) for r in rows if key in r[section]]
        return sum(vals) / len(vals) if vals else float("nan")

    by_eps = (utility or {}).get("utility_by_epsilon", {})
    print(f"\n{'=' * 96}")
    label = DEFENSE_LABELS.get(defense, defense)
    noun = {"cmag": "covering", "vec2text": "mechanism",
            "remote_rag": "mechanism"}.get(defense, "mask")
    axis = {"vec2text": "lambda", "remote_rag": "radius r"}.get(defense, "epsilon")
    print(f"[{defense}] privacy-utility trade-off — one {noun}, every {axis} ({label})")
    print(f"{'=' * 96}")
    print(
        f"  {axis:>9}  {'tokF1':>7} {'bleu':>7} {'rougeL':>8} {'align_cos':>10}  "
        f"{'recall@k':>9} {'spearman':>9}"
    )
    base = results["baseline"]
    print(
        f"  {('0 (none)' if defense in ('vec2text', 'remote_rag') else 'inf (none)'):>9}  "
        f"{mean(base,'test_results','token_f1'):>7.4f} "
        f"{mean(base,'test_results','bleu'):>7.2f} {mean(base,'test_results','rougeL'):>8.4f} "
        f"{mean(base,'diagnostics','X_Y_test_COS'):>10.4f}  {1.0:>9.4f} {1.0:>9.4f}"
    )
    for label, _, eps in arms[1:]:
        rows = results[label]
        u = by_eps.get(str(eps), {})
        rec = next((v for k, v in u.items() if k.startswith("recall_at_")), None)
        spear = u.get("sim_spearman")

        rec_s = f"{rec:>9.4f}" if rec is not None else f"{'--':>9}"
        spear_s = f"{spear:>9.4f}" if spear is not None else f"{'--':>9}"
        print(
            f"  {eps:>9g}  {mean(rows,'test_results','token_f1'):>7.4f} "
            f"{mean(rows,'test_results','bleu'):>7.2f} {mean(rows,'test_results','rougeL'):>8.4f} "
            f"{mean(rows,'diagnostics','X_Y_test_COS'):>10.4f}  {rec_s} {spear_s}"
        )
    print(
        "\n  The undefended row is the ceiling for both columns. A defense earns its place\n"
        "  where the attack columns fall faster than recall@k and spearman do."
    )


def run_sweep(args: argparse.Namespace, checkpoint: Path) -> None:
    import torch

    target = max(victim_paths(args), key=lambda p: len(EmbeddingSet.load(p)))
    embset = EmbeddingSet.load(target)
    print(f"\n{'=' * 72}\n[sweep] {target.name} (n={len(embset)})\n{'=' * 72}")

    ks = [int(x) for x in str(args.sweep_k).split(",")]
    lams = [float(x) for x in str(args.sweep_lambda).split(",")]
    rows = []
    for k in ks:
        for lam in lams:
            atk = get_attack(
                "algen",
                checkpoint_dir=checkpoint,
                dataset=args.attack_dataset,
                align_samples=k,
                reg_lambda=lam,
                seed=args.seed,
                device=args.device,
                align_text=args.align_text,
                verify=False,
            )
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                atk.fit(embset)
                res = atk.invert(embset, source=target.name)
            rows.append((k, lam, res.text_metrics["rougeL"], res.diagnostics["X_Y_test_COS"]))
            print(f"  k={k:<5} lambda={lam:<7} rougeL={rows[-1][2]:.4f} align_cos={rows[-1][3]:.4f}")
            _free(atk)
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r[2])
    print(f"\n  best: k={best[0]} lambda={best[1]} -> rougeL={best[2]:.4f} (align_cos={best[3]:.4f})")
    out = Path(checkpoint) / "attacks" / "sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"target": target.name, "align_text": args.align_text,
             "rows": [{"k": k, "reg_lambda": l, "rougeL": r, "align_cos": c} for k, l, r, c in rows],
             "best": {"k": best[0], "reg_lambda": best[1], "rougeL": best[2]}},
            f, indent=2,
        )
    print(f"out: {out}")


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)
    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
        return 0

    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(STAGES)}")

    set_seed(args.seed)

    sets = find_embedding_sets("all")
    holdout_for = lambda dataset: collect_target_ids(sets, dataset)  # noqa: E731
    holdout = holdout_for(args.attack_dataset)
    dirs = stage_dirs(args)

    print(f"[pipeline] stages={stages} seed={args.seed}")
    print(f"[holdout]  {len(holdout)} {args.attack_dataset} attack-target id(s) excluded "
          f"from its training")
    requested = {
        "pretrain": {"align_reserve": args.pretrain_align_reserve,
                     "val_samples": args.pretrain_val_samples, "seed": args.seed},
        "finetune": {"align_reserve": args.finetune_align_reserve,
                     "val_samples": args.finetune_val_samples, "seed": args.seed},
    }
    for name, d in dirs.items():
        if not GeneratorTrainer.is_complete(d):
            state = "to train"
        elif split_mismatches(d, requested[name]):
            state = "STALE"
        else:
            state = "cached"
        print(f"[{name:<9}] {state:<9} {d}")

    defended = args.defense if args.defense not in ("none", "") else "eguard"
    for name, ckpt in (("eguard", eguard_checkpoint_for(args)),
                       ("sparse", sparse_checkpoint_for(args)),
                       ("cmag", cmag_checkpoint_for(args)),
                       ("vec2text", vec2text_checkpoint_for(args)),
                       ("remote_rag", remote_rag_checkpoint_for(args))):
        if name not in stages and name != defended:
            continue
        note = ""
        if args.compare and name == defended:
            note = "  (--compare: attack runs on e and e')"
        if name in ("sparse", "cmag", "vec2text", "remote_rag"):
            eps = {"sparse": _sparse_epsilons, "cmag": _cmag_epsilons,
                   "vec2text": _vec2text_noise_levels,
                   "remote_rag": _remote_rag_radii}[name](args)
            note += f"  eps={eps[0]:g}" if len(eps) == 1 else f"  eps sweep={[f'{e:g}' for e in eps]}"
        print(f"[{name:<9}] {'cached' if ckpt.exists() else 'to fit':<9} {ckpt}{note}")
    if args.defense in LEARNED_DEFENSES and args.defense not in stages:
        print(
            f"[note] --defense {args.defense} but '{args.defense}' is not in --stages; "
            f"the attack will use the checkpoint above and fail if it is missing."
        )

    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    if "pretrain" in stages:
        run_training_stage(
            args, name="pretrain", dataset=args.pretrain_dataset,
            train_samples=args.pretrain_samples, val_samples=args.pretrain_val_samples,
            align_reserve=args.pretrain_align_reserve, batch_size=args.pretrain_batch_size,
            learning_rate=args.pretrain_lr, epochs=args.pretrain_epochs,
            holdout=holdout_for(args.pretrain_dataset), run_dir=dirs["pretrain"],
        )

    if "finetune" in stages:
        init = dirs["pretrain"] if GeneratorTrainer.is_complete(dirs["pretrain"]) else None
        if init is None:
            print(
                "Warn: no pretrained generator to warm-start from; stage 2 will train\n"
                "       from scratch on a corpus too small for it."
            )
        run_training_stage(
            args, name="finetune", dataset=args.finetune_dataset,
            train_samples=args.finetune_samples, val_samples=args.finetune_val_samples,
            align_reserve=args.finetune_align_reserve, batch_size=args.finetune_batch_size,
            learning_rate=args.finetune_lr, epochs=args.finetune_epochs,
            holdout=holdout_for(args.finetune_dataset), run_dir=dirs["finetune"],
            init_from=init,
        )

    if "eguard" in stages:
        run_eguard_stage(args, holdout)

    if "sparse" in stages:
        run_sparse_stage(args, holdout)

    if "cmag" in stages:
        run_cmag_stage(args, holdout)

    if "vec2text" in stages:
        run_vec2text_stage(args, holdout)

    if "remote_rag" in stages:
        run_remote_rag_stage(args, holdout)

    if "attack" in stages:
        run_attack_stage(args, dirs["finetune"])
        if args.sweep:
            run_sweep(args, dirs["finetune"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
