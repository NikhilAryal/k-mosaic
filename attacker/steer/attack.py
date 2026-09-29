from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from models import EmbeddingSet

from ..algen.attack import ALGENAttacker
from ..base import AttackResult
from ..data import victim_embedder


DEFENSE_SCOPES = {
    "targets": "STEER  — the provider never observes a defended vector it has "
               "text for, so the alignment pairs are clean and the map is never refitted",
    "both": "ALGEN — the defense sits inside the victim boundary, so the attacker's "
            "query access returns defended vectors and the map is refitted through it",
}


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return (A * B).sum(axis=1)


class SteerAttacker(ALGENAttacker):
    name = "steer"
    requires_training = True

    def __init__(self, *args: Any, defense_scope: str = "targets", **kwargs: Any) -> None:
        if defense_scope not in DEFENSE_SCOPES:
            raise ValueError(
                f"defense_scope must be one of {sorted(DEFENSE_SCOPES)}, got {defense_scope!r}"
            )
        super().__init__(*args, **kwargs)
        self.defense_scope = defense_scope

    def fit_alignment(self, embset: EmbeddingSet, holdout_ids: Sequence[str]):
        if self.defense_scope != "targets":
            return super().fit_alignment(embset, holdout_ids)

        print(
            f"steer: defense_scope=targets: fitting the alignment on clean pairs "
            f"(defense {self.defense!r} applied to the targets only)"
        )
        saved, saved_storage = self.defense, self.storage
        self.defense, self.storage = "none", None
        try:
            return super().fit_alignment(embset, holdout_ids)
        finally:
            self.defense, self.storage = saved, saved_storage

    def _reconstruction_cos(
        self, embset: EmbeddingSet, predictions: Sequence[str]
    ) -> dict[str, float]:
        clean = np.asarray(embset.vectors, dtype=np.float32)
        embedder = victim_embedder(embset)
        recon = np.asarray(
            embedder.encode([p if p.strip() else " " for p in predictions]), dtype=np.float32
        )
        out = {"recon_COS": float(_cosine(recon, clean).mean())}

        if self.defense != "none":
            import torch

            X = torch.tensor(clean, dtype=torch.float32, device=self.device)
            defended, _ = self._defend(X)
            out["defense_COS"] = float(
                _cosine(defended.detach().cpu().numpy().astype(np.float32), clean).mean()
            )
        return out

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        result = super().invert(embset, source=source)
        result.attack = self.name
        result.diagnostics.update(self._reconstruction_cos(embset, result.predictions))
        result.config.update(
            {
                "threat_model": "steer",
                "defense_scope": self.defense_scope,
                "defense_scope_note": DEFENSE_SCOPES[self.defense_scope],
            }
        )
        return result

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {
            "align_cos": result.diagnostics.get("X_Y_test_COS", float("nan")),
            "recon_cos": result.diagnostics.get("recon_COS", float("nan")),
        }
