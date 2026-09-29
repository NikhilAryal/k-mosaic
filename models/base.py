from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class EmbeddingSet:

    vectors: np.ndarray
    ids: list[str]
    indices: list[int]
    model: str
    dataset: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: {len(self.ids)} vs {self.vectors.shape[0]}"
            )

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __repr__(self) -> str:
        return (
            f"EmbeddingSet(model={self.model!r}, dataset={self.dataset!r}, "
            f"n={len(self)}, dim={self.dim}, dtype={self.vectors.dtype})"
        )

    def row_of(self, record_id: str) -> np.ndarray:
        try:
            i = self.ids.index(record_id)
        except ValueError:
            raise KeyError(f"{record_id!r} not in this EmbeddingSet") from None
        return self.vectors[i]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=self.vectors,
            ids=np.array(self.ids, dtype=object),
            indices=np.array(self.indices, dtype=np.int64),
            manifest=json.dumps(
                {"model": self.model, "dataset": self.dataset, "meta": self.meta}
            ),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingSet":
        with np.load(Path(path), allow_pickle=True) as z:
            manifest = json.loads(str(z["manifest"]))
            return cls(
                vectors=z["vectors"],
                ids=[str(x) for x in z["ids"].tolist()],
                indices=[int(x) for x in z["indices"].tolist()],
                model=manifest["model"],
                dataset=manifest["dataset"],
                meta=manifest.get("meta", {}),
            )


class BaseEmbedder(ABC):
    name: str = "base"
    default_model_id: str | None = None

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        max_seq_length: int | None = None,
        cache_dir: str | Path | None = "data/embeddings",
        dtype: str = "float32",
    ) -> None:
        self.model_id = model_id or self.default_model_id or self.name
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.device = device or self._auto_device()
        self._model: Any = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Instantiate and return the backend model. Called lazily, once."""

    @abstractmethod
    def _encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        """Encode raw texts into an ``(n, d)`` array (pre-normalisation)."""

    def _fingerprint_parts(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "name": self.name,
            "model_id": self.model_id,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
            "dtype": self.dtype,
        }

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dim(self) -> int:
        cached = getattr(self, "_dim", None)
        if cached is None:
            cached = int(self.encode(["dimension probe"]).shape[1])
            self._dim = cached
        return cached

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, model_id={self.model_id!r}, "
            f"device={self.device!r}, normalize={self.normalize})"
        )

    def encode(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=self.dtype)
        vecs = np.asarray(self._encode(list(texts), show_progress=show_progress))
        if self.normalize:
            vecs = self.l2_normalize(vecs)
        return vecs.astype(self.dtype, copy=False)

    def embed(
        self,
        selection: Any,
        *,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> EmbeddingSet:
        cache_path = self.cache_path(selection) if use_cache else None
        if cache_path is not None and cache_path.exists():
            return EmbeddingSet.load(cache_path)

        vectors = self.encode(selection.texts, show_progress=show_progress)
        out = EmbeddingSet(
            vectors=vectors,
            ids=list(selection.ids),
            indices=list(selection.indices),
            model=self.name,
            dataset=selection.dataset,
            meta={**self._fingerprint_parts(), "device": self.device, "n": len(selection)},
        )
        if cache_path is not None:
            out.save(cache_path)
        return out

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = self.l2_normalize(np.atleast_2d(a))
        b = self.l2_normalize(np.atleast_2d(b))
        return a @ b.T

    def fingerprint(self, selection: Any) -> str:
        payload = {
            "encoder": self._fingerprint_parts(),
            "dataset": getattr(selection, "dataset", "unknown"),
            "ids": list(getattr(selection, "ids", [])),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def cache_path(self, selection: Any) -> Path | None:
        if self.cache_dir is None:
            return None
        dataset = str(getattr(selection, "dataset", "unknown")).replace("/", "_")
        return self.cache_dir / f"{self.name}__{dataset}__{self.fingerprint(selection)}.npz"

    @staticmethod
    def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, eps)
