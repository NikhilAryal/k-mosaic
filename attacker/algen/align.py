from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..metrics import eval_embeddings


@dataclass
class AlignmentReport:
    train_cos: float
    train_mse: float
    val_cos: float | None = None
    val_mse: float | None = None
    reg_lambda: float | None = None
    n_pairs: int = 0
    source_dim: int = 0
    target_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "X_Y_COS": self.train_cos,
            "X_Y_MSEloss": self.train_mse,
            "X_Y_val_COS": self.val_cos,
            "X_Y_val_MSEloss": self.val_mse,
            "reg_lambda": self.reg_lambda or 0,
            "n_pairs": self.n_pairs,
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
        }


class LinearAligner:
    def __init__(self, reg_lambda: float | None = 0.01) -> None:
        self.reg_lambda = reg_lambda
        self.A: torch.Tensor | None = None
        self.report: AlignmentReport | None = None

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_val: torch.Tensor | None = None,
        Y_val: torch.Tensor | None = None,
    ) -> "LinearAligner":
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"paired matrices disagree: X={tuple(X.shape)}, Y={tuple(Y.shape)}")
        X = X.to(torch.float32)
        Y = Y.to(torch.float32).to(X.device)

        lhs = X.T @ X
        rhs = X.T @ Y
        if self.reg_lambda:
            ridge = self.reg_lambda * torch.eye(lhs.shape[0], dtype=X.dtype, device=X.device)
            self.A = torch.linalg.pinv(lhs + ridge) @ rhs
        else:
            self.A = torch.linalg.pinv(lhs) @ rhs

        train_cos, train_mse = eval_embeddings(X @ self.A, Y)
        val_cos = val_mse = None
        if X_val is not None and Y_val is not None and len(X_val):
            c, m = eval_embeddings(self.transform(X_val), Y_val.to(X.device))
            val_cos, val_mse = float(c), float(m)

        self.report = AlignmentReport(
            train_cos=float(train_cos),
            train_mse=float(train_mse),
            val_cos=val_cos,
            val_mse=val_mse,
            reg_lambda=self.reg_lambda,
            n_pairs=int(X.shape[0]),
            source_dim=int(self.A.shape[0]),
            target_dim=int(self.A.shape[1]),
        )
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.A is None:
            raise RuntimeError("LinearAligner.fit must be called before transform")
        return X.to(self.A.device, torch.float32) @ self.A

    __call__ = transform

    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        if self.A is None:
            raise RuntimeError("nothing to save: fit first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            A=self.A.detach().cpu().numpy(),
            manifest=json.dumps(
                {
                    "reg_lambda": self.reg_lambda,
                    "report": self.report.to_dict() if self.report else None,
                }
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "LinearAligner":
        with np.load(Path(path), allow_pickle=True) as z:
            manifest = json.loads(str(z["manifest"]))
            aligner = cls(reg_lambda=manifest.get("reg_lambda"))
            aligner.A = torch.tensor(z["A"], dtype=torch.float32, device=device)
        return aligner
