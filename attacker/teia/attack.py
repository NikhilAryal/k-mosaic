from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from models import EmbeddingSet

from ..base import AttackResult, BaseAttack
from ..data import texts_for_ids, truncate_reference, verify_victim_encoder, victim_embedder
from ..metrics import eval_embeddings, eval_texts
from ..algen.defenses import apply_defense
from .trainer import TeiaTrainer


DEFENSE_SCOPES = {
    "both": "TEIA: the leak comes out of the vector database, so D_L "
            "carries whatever the database stores and the decoder is trained on it",
    "targets": "STEER: the decoder is trained on clean vectors and meets the "
               "defense for the first time at attack time",
}


EMBED_SIM_MODEL = "sentence-transformers/all-mpnet-base-v2"


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (A * B).sum(axis=1)


class TeiaAttacker(BaseAttack):
    name = "teia"
    requires_training = True

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        dataset: str | None = None,
        seed: int = 42,
        device: str | None = None,
        defense: str = "none",
        defense_scope: str = "both",
        noise_level: float = 0.0,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        eguard_checkpoint: str | Path | None = None,
        sparse_checkpoint: str | Path | None = None,
        cmag_checkpoint: str | Path | None = None,
        idct_checkpoint: str | Path | None = None,
        vec2text_checkpoint: str | Path | None = None,
        vec2text_noise_level: float | None = None,
        remote_rag_checkpoint: str | Path | None = None,
        remote_rag_radius: float | None = None,
        storage: Any = None,
        embed_similarity: bool = True,
        verify: bool = True,
    ) -> None:
        if checkpoint_dir is None:
            raise SystemExit(
                "teia needs a trained decoder: pass --checkpoint <run-dir>, or "
                "train one with `python train_teia.py --stages train`."
            )
        if defense_scope not in DEFENSE_SCOPES:
            raise ValueError(
                f"defense_scope must be one of {sorted(DEFENSE_SCOPES)}, got {defense_scope!r}"
            )
        self.checkpoint_dir = Path(checkpoint_dir)
        self._dataset_override = dataset
        self.seed = seed
        self.defense = defense
        self.defense_scope = defense_scope
        self.noise_level = noise_level
        self.epsilon = epsilon
        self.delta = delta
        self.eguard_checkpoint = eguard_checkpoint
        self.sparse_checkpoint = sparse_checkpoint
        self.cmag_checkpoint = cmag_checkpoint
        self.idct_checkpoint = idct_checkpoint
        self.vec2text_checkpoint = vec2text_checkpoint
        self.vec2text_noise_level = vec2text_noise_level
        self.remote_rag_checkpoint = remote_rag_checkpoint
        self.remote_rag_radius = remote_rag_radius
        self.storage = storage
        self.embed_similarity = embed_similarity
        self.verify = verify

        self.trainer = TeiaTrainer.from_checkpoint(self.checkpoint_dir, device=device)
        self.device = self.trainer.device
        self.tokenizer = self.trainer.tokenizer
        self.max_length = self.trainer.max_length

        self._defense_state: dict = dict(self.trainer.defense_state)

        self._check_scope_matches_checkpoint()


    def _check_scope_matches_checkpoint(self) -> None:

        trained = self.trainer.args
        if self.defense_scope != trained.get("defense_scope", "both"):
            print(
                f"[Warning] attacking with defense_scope={self.defense_scope!r} but the "
                f"checkpoint was trained with {trained.get('defense_scope')!r}."
            )
        if self.defense_scope == "both" and trained.get("defense", "none") != self.defense:
            raise SystemExit(
                f"checkpoint mismatch: {self.checkpoint_dir}\n"
                f"  was trained with defense={trained.get('defense', 'none')!r}, this "
                f"attack applies {self.defense!r}.\n"
                f"  Under defense_scope='both' the defended vectors ARE the training\n"
                f"  data, so the decoder has to be retrained per defense. Either run\n"
                f"  `python train_teia.py --stages train,attack --defense {self.defense}`\n"
                f"  or switch to --defense-scope targets, which trains once on clean\n"
                f"  vectors and defends only the targets."
            )

    def dataset_for(self, embset: EmbeddingSet) -> str:
        return self._dataset_override or embset.dataset

    def _truncate(self, texts: Sequence[str]) -> list[str]:
        return truncate_reference(texts, self.tokenizer, self.max_length, self.device)

    def _defend(self, X: torch.Tensor) -> torch.Tensor:
        out, self._defense_state = apply_defense(
            X,
            self.defense,
            noise_level=self.noise_level,
            epsilon=self.epsilon,
            delta=self.delta,
            eguard_checkpoint=self.eguard_checkpoint,
            sparse_checkpoint=self.sparse_checkpoint,
            cmag_checkpoint=self.cmag_checkpoint,
            idct_checkpoint=self.idct_checkpoint,
            vec2text_checkpoint=self.vec2text_checkpoint,
            vec2text_noise_level=self.vec2text_noise_level,
            remote_rag_checkpoint=self.remote_rag_checkpoint,
            remote_rag_radius=self.remote_rag_radius,
            state=self._defense_state,
        )
        return out

    def _embed_similarity(self, predictions: Sequence[str], references: Sequence[str]) -> float:
        from models import get_model

        enc = get_model("sentence-transformer", model_id=EMBED_SIM_MODEL, cache_dir=None)
        pred = np.asarray(enc.encode([p if p.strip() else " " for p in predictions]), np.float32)
        ref = np.asarray(enc.encode(list(references)), np.float32)
        return float(_cosine(pred, ref).mean())


    def _attack_vectors(self, embset: EmbeddingSet, clean: torch.Tensor) -> torch.Tensor:
        out = self._defend(clean) if self.defense != "none" else clean
        return out if self.storage is None else self.storage(out)

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        dataset = self.dataset_for(embset)
        full_refs = texts_for_ids(dataset, embset.ids)
        refs = self._truncate(full_refs)

        if self.verify:
            verify_victim_encoder(embset, victim_embedder(embset), full_refs)

        clean = torch.tensor(embset.vectors, dtype=torch.float32, device=self.device)
        X = self._attack_vectors(embset, clean)

        predictions = self.trainer.generate(X)
        oracle_predictions = (
            self.trainer.generate(clean) if self.defense != "none" else []
        )

        diagnostics: dict[str, Any] = {}
        recon = np.asarray(
            victim_embedder(embset).encode(
                [p if p.strip() else " " for p in predictions]
            ),
            np.float32,
        )
        diagnostics["recon_COS"] = float(_cosine(recon, np.asarray(embset.vectors, np.float32)).mean())
        if self.defense != "none":
            d_cos, d_mse = eval_embeddings(X, clean)
            diagnostics["defense_COS"] = float(d_cos)
            diagnostics["defense_MSE"] = float(d_mse)
        if self.embed_similarity:
            diagnostics["embed_similarity"] = self._embed_similarity(predictions, refs)

        val = (self.trainer.best_models[0][1] if self.trainer.best_models else {}) or {}
        diagnostics["val_rougeL"] = float(val.get("rougeL", float("nan")))
        diagnostics["val_perplexity"] = float(val.get("perplexity", float("nan")))

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
            oracle_metrics=eval_texts(oracle_predictions, refs) if oracle_predictions else {},
            diagnostics=diagnostics,
            config={
                "checkpoint": str(self.checkpoint_dir),
                "threat_model": "teia (arXiv:2406.10280 §2.2)",
                "decoder": self.trainer.args["decoder_name"],
                "surrogate_model": self.trainer.args["surrogate_model"],
                "external_dataset": self.trainer.args["external_dataset"],
                "leaked_samples": self.trainer.args["leaked_samples"],
                "external_samples": self.trainer.args["external_samples"],
                "geia": self.trainer.args["geia"],
                "mapping_lambda": self.trainer.args["mapping_lambda"],
                "pivot_lambda": self.trainer.args["pivot_lambda"],
                "victim_model": embset.model,
                "victim_dataset": embset.dataset,
                "dataset": dataset,
                "max_length": self.max_length,
                "defense": self.defense,
                "defense_scope": self.defense_scope,
                "defense_scope_note": DEFENSE_SCOPES[self.defense_scope],
                "epsilon": self.epsilon,
                "seed": self.seed,
            },
        )

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {
            "embed_sim": result.diagnostics.get("embed_similarity", float("nan")),
            "recon_cos": result.diagnostics.get("recon_COS", float("nan")),
        }

    @classmethod
    def add_train_args(cls, parser: Any) -> None:
        from .cli import add_train_args

        add_train_args(parser)

    @classmethod
    def train_from_args(cls, args: Any) -> int:
        from .cli import train_teia_decoder

        return train_teia_decoder(args)
