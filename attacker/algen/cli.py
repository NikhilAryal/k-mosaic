from __future__ import annotations

import argparse

from ..data import build_splits, collect_target_ids, find_embedding_sets
from .trainer import GeneratorTrainer


def add_train_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("algen — generator fine-tuning")
    g.add_argument("--model-name", default="google/flan-t5-base", help="the generator G")
    g.add_argument("--output-dir", default="attacker/outputs")
    g.add_argument("--dataset", default="nfcorpus", help="attacker-held corpus")
    g.add_argument("--max-length", type=int, default=32, help="L: tokens G reconstructs")
    g.add_argument("--train-samples", type=int, default=3000)
    g.add_argument("--val-samples", type=int, default=200)
    g.add_argument("--align-reserve", type=int, default=200,
                   help="texts withheld so the attack's few-shot pairs are unseen")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--learning-rate", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=1e-5)
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--init-from", default=None,
                   help="warm-start from another run directory's best checkpoint")
    g.add_argument("--holdout-from", default="all",
                   help="'all', a glob, or paths: ids excluded from training")


def train_algen_generator(args: argparse.Namespace) -> int:
    holdout = (
        collect_target_ids(find_embedding_sets(args.holdout_from, args.embedding_dir),
                           args.dataset)
        if args.holdout_from else []
    )
    if holdout:
        print(f"Holdout: excluding {len(holdout)} attack-target id(s) from training")

    splits = build_splits(
        args.dataset,
        holdout_ids=holdout,
        n_train=args.train_samples,
        n_val=args.val_samples,
        n_align=args.align_reserve,
        seed=args.seed,
    )
    print(f"[splits]:: {splits.summary()}")

    trainer = GeneratorTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        dataset=args.dataset,
        train_texts=splits.train,
        val_texts=splits.val,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        wandb_run_name=args.wandb_run_name,
        device=args.device,
        holdout_ids=holdout,
        align_reserve=args.align_reserve,
        seed=args.seed,
    )
    if args.init_from:
        trainer.init_from(args.init_from)
    print(f"[run] {trainer.run_dir}")
    trainer.train()
    print(
        f"\nDone. Attack with:\n"
        f"  python -m attacker attack --attack algen --checkpoint {trainer.run_dir}"
    )
    return 0
