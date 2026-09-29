from __future__ import annotations

from typing import Any, Type

from .base import BaseDataset, Record, Selection
from .beir import BEIR_SUBSETS, BeirDataset, FiQADataset, NFCorpusDataset, QuoraDataset
from .msmarco import MSMARCO_TOTAL, MsMarco1M, MsMarcoDataset

DATASETS: dict[str, Type[BaseDataset]] = {
    "beir": BeirDataset,
    "nfcorpus": NFCorpusDataset,
    "fiqa": FiQADataset,
    "quora": QuoraDataset,
    "msmarco": MsMarco1M,
    "msmarco-1m": MsMarco1M,
}


def register_dataset(name: str, cls: Type[BaseDataset]) -> None:
    DATASETS[name] = cls


def get_dataset(name: str, /, **kwargs: Any) -> BaseDataset:
    if name.startswith("beir/"):
        return BeirDataset(name.split("/", 1)[1], **kwargs)
    try:
        cls = DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {sorted(DATASETS)} "
            f"(or 'beir/<subset>' for any of {sorted(BEIR_SUBSETS)})"
        ) from None
    return cls(**kwargs)


__all__ = [
    "BaseDataset",
    "Record",
    "Selection",
    "BeirDataset",
    "NFCorpusDataset",
    "FiQADataset",
    "QuoraDataset",
    "MsMarcoDataset",
    "MsMarco1M",
    "MSMARCO_TOTAL",
    "BEIR_SUBSETS",
    "DATASETS",
    "get_dataset",
    "register_dataset",
]
