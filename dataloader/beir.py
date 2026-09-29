from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .base import BaseDataset, Record

BEIR_SUBSETS: dict[str, int] = {
    "nfcorpus": 3_633,
    "scifact": 5_183,
    "arguana": 8_674,
    "scidocs": 25_657,
    "fiqa": 57_638,
    "trec-covid": 171_332,
    "quora": 522_931,
    "nq": 2_681_468,
    "hotpotqa": 5_233_329,
    "msmarco": 8_841_823,
}


class BeirDataset(BaseDataset):
    name = "beir"

    def __init__(
        self,
        subset: str = "nfcorpus",
        *,
        root: str | Path | None = None,
        include_title: bool = True,
        hf_repo_template: str = "BeIR/{subset}",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.subset = subset
        self.include_title = include_title
        self.hf_repo_template = hf_repo_template
        self.name = f"beir/{subset}"
        self.root = Path(root) if root is not None else self.cache_dir / "beir"

    @property
    def local_corpus_path(self) -> Path:
        return self.root / self.subset / "corpus.jsonl"

    def _compose(self, title: str | None, text: str) -> str:
        title = (title or "").strip()
        if self.include_title and title:
            return f"{title} {text.strip()}".strip()
        return text.strip()

    def _load(self) -> Iterable[Record]:
        if self.local_corpus_path.exists():
            return self._load_local()
        return self._load_hf()

    def _load_local(self) -> Iterator[Record]:
        with self.local_corpus_path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield Record(
                    id=str(obj.get("_id", obj.get("id", i))),
                    text=self._compose(obj.get("title"), obj.get("text", "")),
                    meta={"title": obj.get("title", ""), "subset": self.subset},
                )

    def _load_hf(self) -> Iterator[Record]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  
            raise ImportError(
                "BEIR needs either a local corpus.jsonl at "
                f"{self.local_corpus_path} or the `datasets` package installed."
            ) from exc

        repo = self.hf_repo_template.format(subset=self.subset)
        ds = load_dataset(repo, "corpus", split="corpus", streaming=True)
        for i, obj in enumerate(ds):
            yield Record(
                id=str(obj.get("_id", i)),
                text=self._compose(obj.get("title"), obj.get("text", "")),
                meta={"title": obj.get("title", ""), "subset": self.subset},
            )

    def queries(self, limit: int | None = None) -> list[Record]:
        local = self.root / self.subset / "queries.jsonl"
        out: list[Record] = []
        if local.exists():
            with local.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    out.append(Record(id=str(obj["_id"]), text=obj.get("text", ""),
                                      meta={"subset": self.subset, "kind": "query"}))
                    if limit and len(out) >= limit:
                        break
            return out

        from datasets import load_dataset 

        repo = self.hf_repo_template.format(subset=self.subset)
        ds = load_dataset(repo, "queries", split="queries", streaming=True)
        for obj in ds:
            out.append(Record(id=str(obj["_id"]), text=obj.get("text", ""),
                              meta={"subset": self.subset, "kind": "query"}))
            if limit and len(out) >= limit:
                break
        return out

    def qrels(self, split: str = "test") -> dict[str, dict[str, int]]:
        local = self.root / self.subset / "qrels" / f"{split}.tsv"
        rels: dict[str, dict[str, int]] = {}
        if local.exists():
            with local.open(encoding="utf-8") as fh:
                header = fh.readline()
                if "query" not in header.lower():
                    fh.seek(0)
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    rels.setdefault(qid, {})[did] = score
            return rels

        from datasets import load_dataset  

        ds = load_dataset(f"BeIR/{self.subset}-qrels", split=split)
        for row in ds:
            rels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
        return rels

    def freeze(self, root: str | Path | None = None) -> Path:
        root = Path(root) if root is not None else self.root
        out = root / self.subset / "corpus.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                title = rec.meta.get("title", "")
                if self.include_title:
                    obj = {"_id": rec.id, "title": "", "text": rec.text, "orig_title": title}
                else:
                    obj = {"_id": rec.id, "title": title, "text": rec.text}
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return out


class NFCorpusDataset(BeirDataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("subset", None)
        super().__init__("nfcorpus", **kwargs)


class FiQADataset(BeirDataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("subset", None)
        super().__init__("fiqa", **kwargs)


class QuoraDataset(BeirDataset):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("subset", None)
        super().__init__("quora", **kwargs)
