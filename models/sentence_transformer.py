from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformer"
    default_model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        prompt: str | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_id, **kwargs)
        self.prompt = prompt
        self.trust_remote_code = trust_remote_code

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(
            self.model_id, device=self.device, trust_remote_code=self.trust_remote_code
        )
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        return model

    def _encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        if self.prompt:
            texts = [f"{self.prompt}{t}" for t in texts]
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalisation is the base class's job
        )

    def _fingerprint_parts(self) -> dict[str, Any]:
        return {**super()._fingerprint_parts(), "prompt": self.prompt}


class GTRBaseEmbedder(SentenceTransformerEmbedder):
    name = "gtr-base"
    default_model_id = "sentence-transformers/gtr-t5-base"

    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("max_seq_length", 128)  # Vec2Text/SPARSE operate on short text
        super().__init__(model_id, **kwargs)


class SentenceT5Embedder(SentenceTransformerEmbedder):
    name = "st5"
    default_model_id = "sentence-transformers/sentence-t5-base"

    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("max_seq_length", 128)
        super().__init__(model_id, **kwargs)


class GTEBaseEmbedder(SentenceTransformerEmbedder):
    name = "gte-base"
    default_model_id = "thenlper/gte-base"


class E5BaseEmbedder(SentenceTransformerEmbedder):
    name = "e5-base"
    default_model_id = "intfloat/e5-base-v2"

    def __init__(self, model_id: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("prompt", "passage: ")
        super().__init__(model_id, **kwargs)


class MiniLMEmbedder(SentenceTransformerEmbedder):
    name = "minilm"
    default_model_id = "sentence-transformers/all-MiniLM-L6-v2"
