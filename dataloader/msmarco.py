from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from .base import Record
from .beir import BeirDataset

MSMARCO_TOTAL = 8_841_823

SAMPLING = {
    "uniform": "keep a record when hash(seed, id) < docs/TOTAL,  a uniform draw over the "
               "whole corpus, decided per record with no state",
    "prefix": "take the first `docs` rows of the stream. Cheap, and WRONG for MS MARCO: "
              "consecutive passages come from the same source document",
}


def _keep(doc_id: str, seed: int, rate: float) -> bool:
    h = hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64 < rate


class MsMarcoDataset(BeirDataset):
    def __init__(
        self,
        *,
        max_records: int | None = 1_000_000,
        sample: str = "uniform",
        sample_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if sample not in SAMPLING:
            raise ValueError(f"sample must be one of {sorted(SAMPLING)}, got {sample!r}")
        kwargs.pop("subset", None)
        kwargs.setdefault("include_title", False)
        super().__init__("msmarco", max_records=max_records, **kwargs)
        self.sample = sample
        self.sample_seed = sample_seed
        self.name = f"beir/{self.subset_tag}"

    @property
    def subset_tag(self) -> str:
        if self.max_records is None:
            return "msmarco"
        n = self.max_records
        size = f"{n // 1_000_000}m" if n >= 1_000_000 and n % 1_000_000 == 0 else f"{n}"
        return (f"msmarco-{size}" if self.sample == "uniform" and self.sample_seed == 42
                else f"msmarco-{size}-{self.sample}{self.sample_seed}")

    @property
    def local_corpus_path(self) -> Path:
        return self.root / self.subset_tag / "corpus.jsonl"

    def _load_local(self):
        n = sum(1 for line in self.local_corpus_path.open(encoding="utf-8") if line.strip())
        if self.max_records is not None and n < self.max_records:
            raise SystemExit(
                f"{self.local_corpus_path} holds {n:,} records but {self.name} expects "
                f"{self.max_records:,}. A freeze was interrupted. Delete the file and "
                f"re-run: python -m dataloader.msmarco --docs {self.max_records} --freeze"
            )
        return super()._load_local()

    def _load_hf(self) -> Iterator[Record]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  
            raise ImportError(
                "MS MARCO needs either a frozen corpus at "
                f"{self.local_corpus_path} (python -m dataloader.msmarco --freeze) "
                "or the `datasets` package installed."
            ) from exc

        rate = 1.0 if self.max_records is None else min(
            1.0, (self.max_records / MSMARCO_TOTAL) * 1.02)   # 2% headroom, then truncate
        ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
        kept = 0
        for i, obj in enumerate(ds):
            doc_id = str(obj.get("_id", i))
            if self.sample == "uniform" and not _keep(doc_id, self.sample_seed, rate):
                continue
            kept += 1
            yield Record(id=doc_id, text=self._compose(obj.get("title"), obj.get("text", "")),
                         meta={"title": obj.get("title", ""), "subset": self.subset_tag})
            if self.max_records is not None and kept >= self.max_records:
                break    

    def freeze(self, root: str | Path | None = None) -> Path:
        root = Path(root) if root is not None else self.root
        out = root / self.subset_tag / "corpus.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps({"_id": rec.id, "title": "", "text": rec.text},
                                    ensure_ascii=False) + "\n")
        return out


class MsMarco1M(MsMarcoDataset):
    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("max_records") is None:
            kwargs["max_records"] = 1_000_000
        super().__init__(**kwargs)


def _main(argv: list[str] | None = None) -> int:  
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="python -m dataloader.msmarco", description=__doc__)
    p.add_argument("--docs", type=int, default=1_000_000, help="subsample size")
    p.add_argument("--sample", default="uniform", choices=sorted(SAMPLING))
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--freeze", action="store_true", help="write the local corpus.jsonl")
    p.add_argument("--root", default=None)
    a = p.parse_args(argv)

    ds = MsMarcoDataset(max_records=a.docs, sample=a.sample, sample_seed=a.sample_seed)
    print(f"msmarco {ds.name}: sampling {a.docs:,} of {MSMARCO_TOTAL:,} "
          f"({SAMPLING[a.sample].split(' — ')[0]})")
    if ds.local_corpus_path.exists() and not a.freeze:
        print(f"msmarco frozen copy already at {ds.local_corpus_path}")
    ds.load()
    print(f"msmarco loaded {len(ds):,} records; first id={ds.records[0].id} "
          f"text={ds.records[0].text[:60]!r}")
    lens = [len(r.text.split()) for r in ds.records[:20000]]
    print(f"msmarco median words/passage (first 20k): {sorted(lens)[len(lens) // 2]}")
    if a.freeze:
        out = ds.freeze(a.root)
        mb = out.stat().st_size / 2**20
        print(f"msmarco froze {len(ds):,} records -> {out} ({mb:.0f} MB)")
    return 0


if __name__ == "__main__": 
    import sys

    sys.exit(_main())
