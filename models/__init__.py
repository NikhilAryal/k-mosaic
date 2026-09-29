from __future__ import annotations

from typing import Any, Type

from .base import BaseEmbedder, EmbeddingSet
from .hf_encoder import HashEmbedder, HFEncoderEmbedder
from .sentence_transformer import (
    E5BaseEmbedder,
    GTEBaseEmbedder,
    GTRBaseEmbedder,
    SentenceT5Embedder,
    MiniLMEmbedder,
    SentenceTransformerEmbedder,
)

DEFAULT_MODEL = "gtr-base"

MODELS: dict[str, Type[BaseEmbedder]] = {
    "gtr-base": GTRBaseEmbedder,
    "st5": SentenceT5Embedder,
    "gte-base": GTEBaseEmbedder,
    "e5-base": E5BaseEmbedder,
    "minilm": MiniLMEmbedder,
    "sentence-transformer": SentenceTransformerEmbedder,
    "hf": HFEncoderEmbedder,
    "hash": HashEmbedder,
}


def register_model(name: str, cls: Type[BaseEmbedder]) -> None:
    MODELS[name] = cls


def get_model(name: str = DEFAULT_MODEL, **kwargs: Any) -> BaseEmbedder:
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name.split(":", 1)[1], **kwargs)
    if name.startswith("hf:"):
        return HFEncoderEmbedder(name.split(":", 1)[1], **kwargs)
    try:
        cls = MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(MODELS)} "
            "(or 'st:<checkpoint>' / 'hf:<checkpoint>')"
        ) from None
    return cls(**kwargs)


__all__ = [
    "BaseEmbedder",
    "EmbeddingSet",
    "SentenceTransformerEmbedder",
    "GTRBaseEmbedder",
    "GTEBaseEmbedder",
    "E5BaseEmbedder",
    "MiniLMEmbedder",
    "HFEncoderEmbedder",
    "HashEmbedder",
    "SentenceT5Embedder",
    "MODELS",
    "DEFAULT_MODEL",
    "get_model",
    "register_model",
]
