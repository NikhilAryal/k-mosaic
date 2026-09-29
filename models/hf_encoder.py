from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .base import BaseEmbedder


class HFEncoderEmbedder(BaseEmbedder):
    name = "hf-encoder"
    default_model_id = "bert-base-uncased"

    POOLINGS = ("mean", "cls", "last")

    def __init__(
        self,
        model_id: str | None = None,
        *,
        pooling: str = "mean",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        if pooling not in self.POOLINGS:
            raise ValueError(f"pooling must be one of {self.POOLINGS}, got {pooling!r}")
        super().__init__(model_id, **kwargs)
        self.pooling = pooling
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any = None

    def _load_model(self) -> Any:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  
            raise ImportError("transformers and torch are required") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        model.to(self.device)
        model.eval()
        self._torch = torch
        return model

    def _pool(self, hidden: Any, mask: Any) -> Any:
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "last":
            lengths = mask.sum(dim=1) - 1
            return hidden[self._torch.arange(hidden.size(0)), lengths]
        expanded = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        model = self.model 
        torch = self._torch
        out: list[np.ndarray] = []
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc=f"encode[{self.name}]")
            except ImportError:
                pass

        with torch.no_grad():
            for start in iterator:
                batch = list(texts[start : start + self.batch_size])
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length or 512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**enc).last_hidden_state
                pooled = self._pool(hidden, enc["attention_mask"])
                out.append(pooled.float().cpu().numpy())
        return np.vstack(out)

    def _fingerprint_parts(self) -> dict[str, Any]:
        return {**super()._fingerprint_parts(), "pooling": self.pooling}


class HashEmbedder(BaseEmbedder):
    name = "hash"

    def __init__(self, model_id: str | None = None, *, dim: int = 256, ngram: int = 3,
                 **kwargs: Any) -> None:
        kwargs.setdefault("cache_dir", None)  # cheap to recompute
        super().__init__(model_id or "hash", **kwargs)
        self._dim = dim
        self.ngram = ngram

    def _load_model(self) -> Any:
        return None

    def _encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            t = text.lower()
            for i in range(max(len(t) - self.ngram + 1, 1)):
                gram = t[i : i + self.ngram]
                h = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                bucket = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if h[4] % 2 else -1.0
                out[row, bucket] += sign
        return out

    def _fingerprint_parts(self) -> dict[str, Any]:
        return {**super()._fingerprint_parts(), "dim": self._dim, "ngram": self.ngram}
