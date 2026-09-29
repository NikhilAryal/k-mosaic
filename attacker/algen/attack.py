from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from models import EmbeddingSet

from ..base import AttackResult, BaseAttack
from .align import LinearAligner
from ..data import (
    CorpusSplits,
    truncate_reference,
    build_splits,
    texts_for_ids,
    verify_victim_encoder,
    victim_embedder,
)
from .defenses import apply_defense
from ..metrics import eval_embeddings, eval_texts
from .trainer import GeneratorTrainer
from .utils import (
    add_punctuation_token_ids,
    check_normalization,
    get_Y_embeddings_from_tokens,
)


class ALGENAttacker(BaseAttack):
    name = "algen"
    requires_training = True

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        dataset: str | None = None,
        align_samples: int = 100,
        reg_lambda: float | None = 0.01,
        seed: int = 42,
        device: str | None = None,
        align_text: str = "truncated",
        attention_mask: str = "ones",
        defense: str = "none",
        noise_level: float = 0.0,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        eguard_checkpoint: str | Path | None = None,
        sparse_checkpoint: str | Path | None = None,
        cmag_checkpoint: str | Path | None = None,
        vec2text_checkpoint: str | Path | None = None,
        vec2text_noise_level: float | None = None,
        remote_rag_checkpoint: str | Path | None = None,
        remote_rag_radius: float | None = None,
        storage: Any = None,
        verify: bool = True,
        cell_maps: bool = False,
        cell_min_pairs: int = 1,
    ) -> None:
        if checkpoint_dir is None:
            raise SystemExit(
                "algen needs a trained generator: pass --checkpoint <run-dir>, or "
                "train one with `python -m attacker train --attack algen`."
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self._dataset_override = dataset  # None => resolve per file from the manifest
        self.align_samples = align_samples
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.align_text = align_text
        self.attention_mask = attention_mask
        self.defense = defense
        self.eguard_checkpoint = eguard_checkpoint
        self.sparse_checkpoint = sparse_checkpoint
        self.cmag_checkpoint = cmag_checkpoint
        self.vec2text_checkpoint = vec2text_checkpoint
        self.vec2text_noise_level = vec2text_noise_level
        self.remote_rag_checkpoint = remote_rag_checkpoint
        self.remote_rag_radius = remote_rag_radius
        self.storage = storage
        self.noise_level = noise_level
        self.epsilon = epsilon
        self.delta = delta
        self.verify = verify
        self.cell_maps = cell_maps
        self.cell_min_pairs = max(1, int(cell_min_pairs))

        self.trainer = GeneratorTrainer.from_checkpoint(self.checkpoint_dir, device=device)
        self.model = self.trainer.model
        self.device = self.trainer.device
        self.tokenizer = self.trainer.tokenizer
        self.max_length = self.trainer.max_length

        # Replayed from the checkpoint so the few-shot pairs land in the same
        # reserved slice of the corpus that training kept its hands off.
        targs = self.trainer.args
        self.align_dataset = self._dataset_override or targs.get("dataset") or None
        self.train_holdout_ids: list[str] = list(targs.get("holdout_ids", []))
        self.split_seed = targs.get("seed", seed)
        self.align_reserve = targs.get("align_reserve", 200)
        if self.align_samples > self.align_reserve:
            n_val = int(targs.get("val_samples", 0) or 0)
            n_train = int(targs.get("train_samples", 0) or 0)
            train_start = self.align_reserve + n_val
            leaked = max(0, min(self.align_samples, train_start + n_train) - train_start)
            print(
                f"[warn] --align-samples {self.align_samples} exceeds the "
                f"{self.align_reserve} texts reserved at training time.\n"
                f"       {leaked} of the {self.align_samples} few-shot pairs are texts "
                f"the generator was trained on, so the attack will read better than "
                f"it really is.\n"
                f"       For a clean run, retrain stage 2 with "
                f"--finetune-align-reserve {self.align_samples}. The corpus budget "
                f"will usually force a smaller --finetune-samples too, which gives "
                f"that run its own directory and leaves this generator intact.\n"
                f"       See configs/algen_nfcorpus_k1000.yaml."
            )

        self._splits: CorpusSplits | None = None
        self._aligner: LinearAligner | None = None
        self._defense_state: dict = {}

    def _truncate(self, texts: Sequence[str]) -> list[str]:
        return truncate_reference(texts, self.tokenizer, self.max_length, self.device)

    def _target_embeddings(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = add_punctuation_token_ids(texts, self.tokenizer, self.max_length, self.device)
        return get_Y_embeddings_from_tokens(tokens, self.model.encoder, normalization=True)

    def dataset_for(self, embset: EmbeddingSet) -> str:
        return self._dataset_override or embset.dataset

    def fit_alignment(self, embset: EmbeddingSet, holdout_ids: Sequence[str]) -> LinearAligner:
        embedder = victim_embedder(embset)
        print(f"[victim] {embedder}")


        dataset = self.align_dataset or self.dataset_for(embset)
        new = set(holdout_ids) - set(self.train_holdout_ids)
        if self.train_holdout_ids and new:
            raise SystemExit(
                f"{len(new)} attack target id(s) were not held out when this generator "
                f"was trained ({len(self.train_holdout_ids)} were), e.g. {sorted(new)[:3]}.\n"
                "They may be in its training text, and the few-shot pairs would no longer "
                "come from the reserved slice. Retrain stage 2 with these targets in "
                "data/embeddings.")
        holdout = sorted(set(self.train_holdout_ids) | set(holdout_ids))
        n_val = min(100, max(0, self.align_samples // 2))
        splits = build_splits(
            dataset,
            holdout_ids=holdout,
            n_train=0,
            n_val=n_val,
            n_align=self.align_samples,
            seed=self.split_seed,
        )
        self._splits = splits
        print(f"[splits] {splits.summary()}")

        align_texts, val_texts = splits.align, splits.val
        victim_inputs = align_texts if self.align_text == "full" else self._truncate(align_texts)
        victim_val_inputs = val_texts if self.align_text == "full" else self._truncate(val_texts)

        X = torch.tensor(embedder.encode(victim_inputs), dtype=torch.float32, device=self.device)
        Y = self._target_embeddings(align_texts)
        X_val = (
            torch.tensor(embedder.encode(victim_val_inputs), dtype=torch.float32, device=self.device)
            if val_texts
            else None
        )
        Y_val = self._target_embeddings(val_texts) if val_texts else None

        if self.defense != "none":
            X, self._defense_state = self._defend(X)
            if X_val is not None:
                X_val, _ = self._defend(X_val)
        X = self._stored(X)

        self._pair_cells = getattr(self.storage, "last_cells", None)
        X_val = self._stored(X_val)

        X, Y = self._corrupt_pairs(X, Y)
        check_normalization(X, "X align (victim)")
        check_normalization(Y, "Y align (generator encoder)")

        self._aligner = LinearAligner(self.reg_lambda).fit(X, Y, X_val, Y_val)
        print(f"[align] {self._aligner.report.to_dict()}")
        if self.cell_maps:
            # Kept for the per-cell solves in invert(): only the cells that hold a
            # target need a map, and those are not known until the targets are stored.
            self._cell_XY = (X, Y)
            if self._pair_cells is None:
                print("[align] cell_maps: storage is not partitioned -- global map only")
        return self._aligner

    def _cell_transform(
        self, X: torch.Tensor, target_cells: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self._aligner is not None
        pairs = getattr(self, "_pair_cells", None)
        if not self.cell_maps or pairs is None or target_cells is None:
            return self._aligner.transform(X), {}
        Xp, Yp = self._cell_XY
        pairs, tc = np.asarray(pairs), np.asarray(target_cells)
        out = self._aligner.transform(X).clone()
        n_used: list[int] = []          # pairs behind each target's map (0 = fallback)
        cell_cos: list[float] = []
        for c in np.unique(tc):
            rows = torch.as_tensor(np.nonzero(tc == c)[0], device=out.device)
            idx = np.nonzero(pairs == c)[0]
            if len(idx) < self.cell_min_pairs:
                n_used += [0] * len(rows)
                continue
            sel = torch.as_tensor(idx, device=Xp.device)
            A = LinearAligner(self.reg_lambda).fit(Xp[sel], Yp[sel])
            out[rows] = A.transform(X[rows]).to(out.dtype)
            n_used += [len(idx)] * len(rows)
            cell_cos.append(A.report.train_cos)
        u = np.asarray(n_used)
        on = u[u > 0]
        diag = {
            "min_pairs": self.cell_min_pairs,
            "target_cells": int(len(np.unique(tc))),
            "cells_fitted": len(cell_cos),
            "targets_on_cell_map": float((u > 0).mean()),
            "targets_fallback_global": float((u == 0).mean()),
            "pairs_per_map_mean": float(on.mean()) if len(on) else 0.0,
            "pairs_per_map_median": float(np.median(on)) if len(on) else 0.0,
            "pairs_per_map_max": int(on.max()) if len(on) else 0,
            "pairs_per_map_over_d": float(on.mean() / X.shape[1]) if len(on) else 0.0,
            "cell_train_cos_mean": float(np.mean(cell_cos)) if cell_cos else None,
        }
        print(f"[align] cell_maps: {diag}")
        return out, diag

    def _corrupt_pairs(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return X, Y

    def _stored(self, X: torch.Tensor | None) -> torch.Tensor | None:
        return X if X is None or self.storage is None else self.storage(X)

    def _partition_diagnostics(self, target_cells: Any, dim: int) -> dict[str, Any]:
        pairs = getattr(self, "_pair_cells", None)
        if pairs is None or target_cells is None:
            return {}
        pairs, target_cells = np.asarray(pairs), np.asarray(target_cells)
        counts = np.bincount(pairs, minlength=int(max(pairs.max(), target_cells.max())) + 1)
        per_target = counts[target_cells]
        return {
            "pairs": int(len(pairs)),
            "pair_cells_hit": int((counts > 0).sum()),
            "pairs_per_cell_max": int(counts.max()),
            "target_cell_pairs_mean": float(per_target.mean()),
            "target_cell_pairs_max": int(per_target.max()),
            "targets_with_zero_pairs": float((per_target == 0).mean()),
            "targets_with_ge_d_pairs": float((per_target >= dim).mean()),
        }

    def _defend(self, X: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return apply_defense(
            X,
            self.defense,
            noise_level=self.noise_level,
            epsilon=self.epsilon,
            delta=self.delta,
            eguard_checkpoint=self.eguard_checkpoint,
            sparse_checkpoint=self.sparse_checkpoint,
            cmag_checkpoint=self.cmag_checkpoint,
            vec2text_checkpoint=self.vec2text_checkpoint,
            vec2text_noise_level=self.vec2text_noise_level,
            remote_rag_checkpoint=self.remote_rag_checkpoint,
            remote_rag_radius=self.remote_rag_radius,
            state=self._defense_state,
        )

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        dataset = self.dataset_for(embset)
        full_refs = texts_for_ids(dataset, embset.ids)
        refs = self._truncate(full_refs)

        embedder = victim_embedder(embset)
        if self.verify:
            verify_victim_encoder(embset, embedder, full_refs)

        if self._aligner is None:      # invert() called directly, without fit()
            self.fit_alignment(embset, holdout_ids=embset.ids)
        assert self._aligner is not None

        X = torch.tensor(embset.vectors, dtype=torch.float32, device=self.device)
        if self.defense != "none":
            X, _ = self._defend(X)
        X = self._stored(X)
        target_cells = getattr(self.storage, "last_cells", None)
        check_normalization(X, "X target (victim)")

        X_aligned, cell_diag = self._cell_transform(X, target_cells)
        Y_true = self._target_embeddings(full_refs)
        test_cos, test_mse = eval_embeddings(X_aligned, Y_true)

        if self.attention_mask == "oracle":
            attention_mask = add_punctuation_token_ids(
                full_refs, self.tokenizer, self.max_length, self.device
            )["attention_mask"]
        else:
            attention_mask = torch.ones(
                X_aligned.size(0), self.max_length, dtype=torch.long, device=self.device
            )
        self.model.eval()

        def decode(hidden: torch.Tensor) -> list[str]:
            generated = self.model.generate(
                {"hidden_states": hidden, "attention_mask": attention_mask}
            )
            return [
                t.strip()
                for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            ]

        predictions = decode(X_aligned)
        # Same decoder, true embeddings: isolates the generator from the alignment.
        oracle_predictions = decode(Y_true)

        align_metrics = dict(self._aligner.report.to_dict())
        align_metrics.update(
            {"X_Y_test_COS": float(test_cos), "X_Y_test_MSEloss": float(test_mse)}
        )
        part = self._partition_diagnostics(target_cells, X.shape[1])
        if part:
            align_metrics["partition"] = part
        if cell_diag:
            align_metrics["cell_maps"] = cell_diag

        return AttackResult(
            attack=self.name,
            source=source or embset.model,
            ids=list(embset.ids),
            predictions=predictions,
            references=refs,
            full_references=full_refs,
            text_metrics=eval_texts(predictions, refs),
            text_metrics_full=eval_texts(predictions, full_refs),
            oracle_predictions=oracle_predictions,
            oracle_metrics=eval_texts(oracle_predictions, refs),
            diagnostics=align_metrics,
            config={
                "checkpoint": str(self.checkpoint_dir),
                "generator": self.trainer.args["model_name"],
                "victim_model": embset.model,
                "victim_model_id": (embset.meta or {}).get("model_id"),
                "victim_dataset": embset.dataset,
                "dataset": dataset,
                "align_dataset": self.align_dataset or dataset,
                "align_samples": self.align_samples,
                "align_text": self.align_text,
                "attention_mask": self.attention_mask,
                "reg_lambda": self.reg_lambda,
                "max_length": self.max_length,
                "defense": self.defense,
                "eguard_checkpoint": str(self.eguard_checkpoint) if self.eguard_checkpoint else None,
                "sparse_checkpoint": str(self.sparse_checkpoint) if self.sparse_checkpoint else None,
                "noise_level": self.noise_level,
                "epsilon": self.epsilon,
                "storage": self.storage.describe() if self.storage is not None else None,
                "seed": self.seed,
                "cell_maps": self.cell_maps,
                "cell_min_pairs": self.cell_min_pairs if self.cell_maps else None,
            },
        )

    def fit(self, embset: EmbeddingSet) -> None:
        self._aligner = None
        self._defense_state = {}
        self.fit_alignment(embset, holdout_ids=embset.ids)

    def save_artifacts(self, out_dir: Path) -> None:
        if self._aligner is not None:
            self._aligner.save(out_dir / "alignment.npz")

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {"align_cos": result.diagnostics.get("X_Y_test_COS", float("nan"))}

    @classmethod
    def add_train_args(cls, parser: Any) -> None:
        from .cli import add_train_args

        add_train_args(parser)

    @classmethod
    def train_from_args(cls, args: Any) -> int:
        from .cli import train_algen_generator

        return train_algen_generator(args)
