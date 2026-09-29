from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from models import EmbeddingSet


@dataclass
class AttackResult:
    attack: str                                   
    source: str                                   
    ids: list[str]                                
    predictions: list[str]                        
    references: list[str]                         
    full_references: list[str]                    
    text_metrics: dict[str, float] = field(default_factory=dict)
    text_metrics_full: dict[str, float] = field(default_factory=dict)
    oracle_predictions: list[str] = field(default_factory=list)
    oracle_metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)   # method-specific
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "source": self.source,
            "n": len(self.ids),
            "config": self.config,
            "test_results": self.text_metrics,
            "test_results_vs_full_text": self.text_metrics_full,
            "oracle_results": self.oracle_metrics,
            "diagnostics": self.diagnostics,
            "bottleneck": self.bottleneck(),
        }

    def bottleneck(self) -> str:
        if not self.oracle_metrics:
            return "n/a (this attack has no oracle)"
        oracle = self.oracle_metrics.get("rougeL", 0.0)
        attack = self.text_metrics.get("rougeL", 0.0)
        retention = attack / oracle if oracle > 0 else 0.0
        gen = (
            "generator strong" if oracle >= 0.45
            else "generator moderate" if oracle >= 0.25
            else "generator weak (more/better training data is the lever)"
        )
        mapping = (
            "mapping near-lossless" if retention >= 0.85
            else "mapping lossy" if retention >= 0.6
            else "mapping is losing most of the ceiling"
        )
        return f"oracle rougeL={oracle:.3f} [{gen}]; retention={retention:.0%} [{mapping}]"

    def to_frame(self):
        import pandas as pd

        cols: dict[str, Sequence[str]] = {
            "id": self.ids,
            "prediction": self.predictions,
            "reference": self.references,
            "full_reference": self.full_references,
        }
        if self.oracle_predictions:
            cols["oracle_prediction"] = self.oracle_predictions
        return pd.DataFrame(cols)

    def show(self, n: int = 5) -> None:
        print(f"\n[{self.attack}] {self.source}: {len(self.ids)} vector(s) inverted")
        for i in range(min(n, len(self.ids))):
            print(f"  id={self.ids[i]}")
            print(f"    recovered: {self.predictions[i]}")
            print(f"    original:  {self.references[i]}")
        print(f"  metrics (vs truncated reference): {self.text_metrics}")
        if self.oracle_metrics:
            print(f"  oracle  (method's own ceiling):   {self.oracle_metrics}")
        print(f"  bottleneck: {self.bottleneck()}")


class BaseAttack(ABC):

    # Registry key
    name: str = "base"
    requires_training: bool = False

    def fit(self, embset: EmbeddingSet) -> None:
        """Per-target setup. Default: nothing to do."""

    @abstractmethod
    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        """Recover text for every row of ``embset``."""

    @classmethod
    def add_train_args(cls, parser: Any) -> None:
        """Contribute this method's training flags to the ``train`` subcommand.
        """

    @classmethod
    def train_from_args(cls, args: Any) -> int:
        raise NotImplementedError(
            f"{cls.__name__} declares requires_training but implements no "
            "train_from_args; either implement it or set requires_training = False"
        )

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        """Extra columns for the cross-file summary table. Method-specific."""
        return {}

    def attack_file(
        self, path: str | Path, out_dir: str | Path | None = None
    ) -> AttackResult:
        path = Path(path)
        embset = EmbeddingSet.load(path)
        print(f"\n=== {path.name}: {embset!r} ===")
        self.fit(embset)
        result = self.invert(embset, source=path.name)

        if out_dir is not None:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            result.to_frame().to_csv(out / "results_texts.csv", index=False)
            with open(out / "results.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            self.save_artifacts(out)
            print(f"[out] {out}")
        return result

    def save_artifacts(self, out_dir: Path) -> None:
        """Persist anything method-specific alongside the results"""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
