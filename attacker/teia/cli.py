from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from models import EmbeddingSet

from ..algen.defenses import apply_defense
from ..data import build_splits, collect_target_ids, find_embedding_sets, victim_embedder
from .trainer import TeiaTrainer


def add_train_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("teia — joint adapter/decoder training")
    g.add_argument("--decoder-name", default="microsoft/DialoGPT-small",
                   help="the causal LM that decodes a victim vector back to text")
    g.add_argument("--output-dir", default="attacker/outputs")
    g.add_argument("--dataset", default="nfcorpus", help="corpus D_L is leaked from")
    g.add_argument("--external-dataset", default="beir/trec-covid",
                   help="corpus D_S is drawn from; the attacker owns it outright")
    g.add_argument("--surrogate-model", default="gte-base",
                   help="the frozen off-the-shelf encoder behind the adapter")
    g.add_argument("--max-length", type=int, default=32)
    g.add_argument("--leaked-samples", type=int, default=2000, help="|D_L|")
    g.add_argument("--external-samples", type=int, default=20000, help="|D_S|")
    g.add_argument("--val-samples", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--epochs", type=int, default=24)
    g.add_argument("--mapping-lambda", type=float, default=1.0)
    g.add_argument("--pivot-lambda", type=float, default=1.0)
    g.add_argument("--geia", action="store_true",
                   help="train the direct-attack baseline instead (no surrogate)")
    g.add_argument("--holdout-from", default="all",
                   help="'all', a glob, or paths: ids excluded from every pool")


def defense_tag_for(defense: str, epsilon: float | None, scope: str) -> str:
    if scope != "both" or defense in ("none", "", None):
        return ""
    tag = str(defense)
    if epsilon is not None and defense in (
        "sparse", "cmag", "vec2text", "remote_rag", "lapmech", "purmech",
        "dp_gaussian", "gaussian",
    ):
        tag += f"{epsilon:g}"
    return tag


def build_teia_trainer(
    *,
    victim_embset: EmbeddingSet,
    dataset: str,
    external_dataset: str,
    surrogate_model: str,
    decoder_name: str,
    output_dir: str | Path,
    max_length: int,
    leaked_samples: int,
    external_samples: int,
    val_samples: int,
    batch_size: int,
    epochs: int,
    mapping_lambda: float,
    pivot_lambda: float,
    geia: bool,
    holdout: Sequence[str],
    seed: int,
    device: str | None = None,
    defense: str = "none",
    defense_scope: str = "both",
    defense_kwargs: dict[str, Any] | None = None,
    build_loaders: bool = True,
) -> TeiaTrainer:
    defense_kwargs = dict(defense_kwargs or {})
    embedder = victim_embedder(victim_embset, **({"device": device} if device else {}))
    print(f"[victim] {embedder}")

    splits = build_splits(
        dataset,
        holdout_ids=holdout,
        n_train=0,
        n_val=val_samples,
        n_align=leaked_samples,
        seed=seed,
    )
    print(f"splits D_L={len(splits.align)} val={len(splits.val)} (from {dataset})")

    print(f"embed D_L + val through the victim encoder ({len(splits.align) + len(splits.val)} docs)")
    leaked_vectors = embedder.encode(splits.align, show_progress=True)
    val_vectors = embedder.encode(splits.val, show_progress=True)

    defense_state: dict = {}
    if defense not in ("none", "") and defense_scope == "both":
        print(f"defense applying {defense!r} to D_L and val (scope=both)")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        lv, defense_state = apply_defense(
            torch.tensor(leaked_vectors, dtype=torch.float32, device=dev),
            defense, state=defense_state, **defense_kwargs,
        )
        vv, defense_state = apply_defense(
            torch.tensor(val_vectors, dtype=torch.float32, device=dev),
            defense, state=defense_state, **defense_kwargs,
        )
        leaked_vectors = lv.detach().cpu().numpy()
        val_vectors = vv.detach().cpu().numpy()

    surrogate = None
    leaked_surrogate = external_texts = external_surrogate = None
    surrogate_dim = int(np.asarray(leaked_vectors).shape[1])
    if not geia:
        from models import get_model

        surrogate = get_model(surrogate_model, cache_dir=None, **({"device": device} if device else {}))
        print(f"surrogate {surrogate}  (frozen; only the adapter is trained)")
        leaked_surrogate = surrogate.encode(splits.align, show_progress=True)
        surrogate_dim = int(np.asarray(leaked_surrogate).shape[1])

        external_texts = _external_texts(external_dataset, external_samples, seed)
        print(f"embed D_S={len(external_texts)} external docs through the surrogate")
        external_surrogate = surrogate.encode(external_texts, show_progress=True)
    else:
        leaked_surrogate = np.zeros((len(splits.align), 1), dtype=np.float32)
        surrogate_dim = 1

    return TeiaTrainer(
        decoder_name=decoder_name,
        output_dir=output_dir,
        max_length=max_length,
        dataset=dataset,
        external_dataset=external_dataset,
        surrogate_model=surrogate_model,
        victim_model=victim_embset.model,
        victim_dim=int(np.asarray(leaked_vectors).shape[1]),
        surrogate_dim=surrogate_dim,
        leaked_texts=splits.align,
        leaked_vectors=leaked_vectors,
        leaked_surrogate=leaked_surrogate,
        external_texts=external_texts,
        external_surrogate=external_surrogate,
        val_texts=splits.val,
        val_vectors=val_vectors,
        leaked_samples=leaked_samples,
        external_samples=external_samples,
        val_samples=val_samples,
        batch_size=batch_size,
        num_epochs=epochs,
        mapping_lambda=mapping_lambda,
        pivot_lambda=pivot_lambda,
        geia=geia,
        defense=defense,
        defense_scope=defense_scope,
        defense_tag=defense_tag_for(defense, defense_kwargs.get("epsilon"), defense_scope),
        defense_state=defense_state,
        device=device,
        holdout_ids=holdout,
        seed=seed,
        build_loaders=build_loaders,
    )


def _external_texts(dataset: str, n: int, seed: int) -> list[str]:
    from dataloader import get_dataset

    records = get_dataset(dataset).load().records
    import random

    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    return [records[i].text for i in order[:n]]


def train_teia_decoder(args: argparse.Namespace) -> int:
    holdout = (
        collect_target_ids(find_embedding_sets(args.holdout_from, getattr(args, "embedding_dir", None)),
                           args.dataset)
        if args.holdout_from else []
    )
    paths = find_embedding_sets("all")
    if not paths:
        raise SystemExit("no .npz in data/embeddings — nothing to take a victim manifest from")

    trainer = build_teia_trainer(
        victim_embset=EmbeddingSet.load(paths[0]),
        dataset=args.dataset,
        external_dataset=args.external_dataset,
        surrogate_model=args.surrogate_model,
        decoder_name=args.decoder_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        leaked_samples=args.leaked_samples,
        external_samples=args.external_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        mapping_lambda=args.mapping_lambda,
        pivot_lambda=args.pivot_lambda,
        geia=args.geia,
        holdout=holdout,
        seed=args.seed,
        device=getattr(args, "device", None),
    )
    trainer.train()
    print(f"out: {trainer.run_dir}")
    return 0
