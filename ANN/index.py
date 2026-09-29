from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - surfaced at first use
    raise ImportError("ANN needs faiss: pip install faiss-cpu") from exc


METRICS = {
    "cosine": "L2-normalise on insert and on query, rank by inner product",
    "ip": "store raw vectors, rank by inner product (magnitude kept)",
}

BUDGET_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class IndexSpec:
    factory: str = "HNSW32,Flat"
    metric: str = "cosine"
    budget: int = 64                # efSearch (HNSW) or nprobe (IVF)
    ef_construction: int = 40      
    train_size: int = 100_000       

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}, got {self.metric!r}")

    @property
    def budget_param(self) -> str | None:
        """faiss's name for the search budget, or ``None`` for an exact index."""
        if "IVF" in self.factory:
            return "nprobe"
        if "HNSW" in self.factory:
            return "efSearch"
        return None

    @property
    def budget_label(self) -> str:
        return {"nprobe": "nprobe", "efSearch": "ef"}.get(self.budget_param or "", "exact")

    def storage_tag(self) -> str:
        return _safe(f"{self.factory}_{self.metric}")

    def tag(self) -> str:
        """Everything that moves a recall number, the utility cache key."""
        b = f"{self.budget_label}{self.budget}" if self.budget_param else "exact"
        return _safe(f"{self.factory}_{self.metric}_{b}_efc{self.ef_construction}")

    def with_budget(self, budget: int) -> "IndexSpec":
        return IndexSpec(self.factory, self.metric, int(budget), self.ef_construction,
                         self.train_size)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def prepare(X: np.ndarray, metric: str) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    if metric == "cosine":
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(X, dtype=np.float32)


def _faiss_metric(metric: str) -> int:
    # Both similarities are inner products; cosine differs only in `prepare`.
    return faiss.METRIC_INNER_PRODUCT


def empty_index(spec: IndexSpec, dim: int) -> Any:
    index = faiss.index_factory(dim, spec.factory, _faiss_metric(spec.metric))
    hnsw = _hnsw_of(index)
    if hnsw is not None:
        hnsw.efConstruction = int(spec.ef_construction)
    return index


def _hnsw_of(index: Any) -> Any:
    """The HNSW graph of an index, if it has one at the top level."""
    idx = faiss.downcast_index(index)
    return getattr(idx, "hnsw", None)


def build_index(X: np.ndarray, spec: IndexSpec, *, seed: int = 0) -> Any:
    index = empty_index(spec, X.shape[1])
    if not index.is_trained:
        rng = np.random.default_rng(seed)
        n = min(len(X), spec.train_size)
        sample = X[rng.choice(len(X), size=n, replace=False)] if n < len(X) else X
        index.train(np.ascontiguousarray(sample))
    index.add(X)
    set_budget(index, spec)
    return index


def set_budget(index: Any, spec: IndexSpec, budget: int | None = None) -> None:
    if spec.budget_param is None:
        return
    faiss.ParameterSpace().set_index_parameter(
        index, spec.budget_param, int(spec.budget if budget is None else budget)
    )


def search(index: Any, Q: np.ndarray, k: int) -> np.ndarray:
    _, I = index.search(np.ascontiguousarray(Q, dtype=np.float32), int(k))
    return I


def _codec_of(index: Any) -> Any:
    idx = faiss.downcast_index(index)
    if hasattr(idx, "storage") and idx.storage is not None:
        return faiss.downcast_index(idx.storage)
    return idx


class IndexStorage:
    def __init__(self, spec: IndexSpec, codec: Any | None, lossless: bool) -> None:
        self.spec = spec
        self._codec = codec
        self.lossless = lossless

    @classmethod
    def from_index(cls, index: Any, spec: IndexSpec) -> "IndexStorage":
        codec = faiss.clone_index(_codec_of(index))
        codec.reset()
        lossless = isinstance(codec, faiss.IndexFlat)
        return cls(spec, None if lossless else codec, lossless)

    @classmethod
    def untrained(cls, spec: IndexSpec, dim: int) -> "IndexStorage | None":
        """Storage for a codec that needs no training, or ``None`` if it does."""
        index = empty_index(spec, dim)
        if not index.is_trained:
            return None
        return cls.from_index(index, spec)

    def round_trip(self, X: np.ndarray) -> np.ndarray:
        X = prepare(X, self.spec.metric)
        if self._codec is None:
            return X
        return np.ascontiguousarray(self._codec.sa_decode(self._codec.sa_encode(X)),
                                    dtype=np.float32)

    def __call__(self, X: Any) -> Any:
        import torch

        if torch.is_tensor(X):
            out = self.round_trip(X.detach().float().cpu().numpy())
            return torch.as_tensor(out, dtype=X.dtype, device=X.device)
        return self.round_trip(np.asarray(X))

    def describe(self) -> str:
        kind = "lossless (flat)" if self.lossless else "LOSSY codec"
        return f"{self.spec.factory} [{self.spec.metric}] storage, {kind}"

    def __repr__(self) -> str:
        return f"IndexStorage({self.describe()})"


__all__ = [
    "METRICS", "BUDGET_GRID", "IndexSpec", "IndexStorage", "prepare", "empty_index",
    "build_index", "set_budget", "search",
]
