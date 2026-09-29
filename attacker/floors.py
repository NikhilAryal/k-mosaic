from __future__ import annotations

from typing import Any, Sequence

import torch

from models import EmbeddingSet

from .algen.attack import ALGENAttacker
from .base import AttackResult
from .data import texts_for_ids
from .metrics import eval_texts

PRIORS = {
    "mean": "the mean of G's own encoder embeddings over the attacker's alignment "
            "texts, renormalised, the corpus prior, in the space the decoder reads",
    "zero": "a zero vector. Degenerate after G's internal F.normalize, but it is the "
            "literal reading of 'no embedding input' and worth having to compare",
}

SOURCES = {
    "shuffled": "permute X's rows against Y. Keeps X's exact marginal distribution "
                "and destroys only the correspondence, the stronger control",
    "random": "replace X with unit-norm Gaussian noise. Destroys the distribution "
              "too, so a gap against 'shuffled' is about X's geometry, not pairing",
}


class PriorOnlyFloor(ALGENAttacker):
    name = "floor_prior"

    def __init__(self, *args: Any, prior: str = "mean", **kwargs: Any) -> None:
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {sorted(PRIORS)}, got {prior!r}")
        super().__init__(*args, **kwargs)
        self.prior = prior

    def fit(self, embset: EmbeddingSet) -> None:
        self._prior_vector = None

    def _prior(self, embset: EmbeddingSet) -> torch.Tensor:
        if self.prior == "zero":
            return torch.zeros(1, self.model.embedder_dim, device=self.device)
        dataset = self.align_dataset or self.dataset_for(embset)
        from .data import build_splits

        splits = build_splits(
            dataset,
            holdout_ids=sorted(set(self.train_holdout_ids) | set(embset.ids)),
            n_train=0,
            n_val=0,
            n_align=self.align_samples,
            seed=self.split_seed,
        )
        Y = self._target_embeddings(splits.align)
        mean = Y.mean(dim=0, keepdim=True)
        return mean / mean.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        dataset = self.dataset_for(embset)
        full_refs = texts_for_ids(dataset, embset.ids)
        refs = self._truncate(full_refs)

        prior = self._prior(embset).expand(len(embset.ids), -1).contiguous()
        attention_mask = torch.ones(
            len(embset.ids), self.max_length, dtype=torch.long, device=self.device
        )
        self.model.eval()
        generated = self.model.generate(
            {"hidden_states": prior, "attention_mask": attention_mask}
        )
        predictions = [
            t.strip() for t in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

        print(f"floor_prior ={self.prior!r} -> {predictions[0][:80]!r}")

        return AttackResult(
            attack=self.name,
            source=source or embset.model,
            ids=list(embset.ids),
            predictions=predictions,
            references=refs,
            full_references=full_refs,
            text_metrics=eval_texts(predictions, refs),
            text_metrics_full=eval_texts(predictions, full_refs),
            diagnostics={"prior": self.prior, "distinct_predictions": len(set(predictions))},
            config={
                "arm": "floor_prior_only",
                "arm_type": "floor",
                "checkpoint": str(self.checkpoint_dir),
                "generator": self.trainer.args["model_name"],
                "prior": self.prior,
                "prior_note": PRIORS[self.prior],
                "victim_model": embset.model,
                "dataset": dataset,
                "align_samples": self.align_samples,
                "max_length": self.max_length,
                "defense": "none (a floor consumes no victim vector)",
                "seed": self.seed,
            },
        )

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {"distinct": result.diagnostics.get("distinct_predictions", 0)}


class RandomVectorFloor(ALGENAttacker):
    name = "floor_random"

    def __init__(self, *args: Any, source: str = "shuffled", **kwargs: Any) -> None:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}, got {source!r}")
        super().__init__(*args, **kwargs)
        self.source = source

    def _corrupt_pairs(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.source == "shuffled":
            perm = torch.randperm(X.shape[0], generator=g).to(X.device)
            X = X[perm]
            print(f"floor_random permuted {X.shape[0]} alignment pairs (seed={self.seed})")
        else:
            noise = torch.randn(X.shape, generator=g).to(X.device, X.dtype)
            X = noise / noise.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            print(f"floor_random replaced X with unit-norm Gaussian noise (seed={self.seed})")
        return X, Y

    def invert(self, embset: EmbeddingSet, source: str = "") -> AttackResult:
        result = super().invert(embset, source=source)
        result.attack = self.name
        result.config.update(
            {
                "arm": "floor_random_vector",
                "arm_type": "floor",
                "floor_source": self.source,
                "floor_source_note": SOURCES[self.source],
            }
        )
        return result

    def summary_row(self, result: AttackResult) -> dict[str, Any]:
        return {"align_cos": result.diagnostics.get("X_Y_test_COS", float("nan"))}


__all__ = ["PriorOnlyFloor", "RandomVectorFloor", "PRIORS", "SOURCES"]


class TeiaPriorFloor:
    name = "teia_floor_prior"

    def __new__(cls, *args: Any, **kwargs: Any):  
        from .teia.attack import TeiaAttacker

        prior = kwargs.pop("prior", "mean")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {sorted(PRIORS)}, got {prior!r}")
        base = type("_TeiaPriorFloor", (TeiaAttacker,), {
            "name": cls.name,
            "_attack_vectors": lambda self, embset, clean: _teia_prior(self, embset, clean, prior),
        })
        obj = base(*args, **kwargs)
        obj.prior = prior
        return obj


class TeiaRandomFloor:
    name = "teia_floor_random"

    def __new__(cls, *args: Any, **kwargs: Any):  
        from .teia.attack import TeiaAttacker

        source = kwargs.pop("source", "random")
        if source == "shuffled":
            print("teia_floor_random 'shuffled' has no TEIA analogue (no pairs to "
                  "permute; the true analogue retrains the adapter). Using noise.")
        base = type("_TeiaRandomFloor", (TeiaAttacker,), {
            "name": cls.name,
            "_attack_vectors": lambda self, embset, clean: _teia_random(self, clean),
        })
        obj = base(*args, **kwargs)
        obj.source = "random"
        return obj


def _teia_prior(self, embset, clean, prior: str):
    if prior == "zero":
        v = torch.zeros(1, clean.shape[1], device=clean.device)
    else:

        from .data import build_splits, victim_embedder

        a = self.trainer.args
        splits = build_splits(
            a.get("dataset", "nfcorpus"),
            holdout_ids=sorted(set(a.get("holdout_ids", [])) | set(embset.ids)),
            n_train=0, n_val=0, n_align=int(a.get("leaked_samples", 2000)),
            seed=int(a.get("seed", 42)),
        )
        vecs = victim_embedder(embset).encode(splits.align)
        v = torch.tensor(vecs, dtype=torch.float32, device=clean.device).mean(0, keepdim=True)
    v = v / v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    print(f"teia_floor_prior ={prior!r}, one constant vector for all "
          f"{clean.shape[0]} target(s)")
    return v.expand(clean.shape[0], -1).contiguous()


def _teia_random(self, clean):
    g = torch.Generator(device="cpu").manual_seed(self.seed)
    z = torch.randn(clean.shape, generator=g).to(clean.device, clean.dtype)
    print(f"teia_floor_random replaced {clean.shape[0]} target vector(s) with noise "
          f"(seed={self.seed})")
    return z / z.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)


__all__ += ["TeiaPriorFloor", "TeiaRandomFloor"]
