from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from dataloader import BEIR_SUBSETS, DATASETS, Selection, get_dataset
from models import DEFAULT_MODEL, MODELS, EmbeddingSet, get_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load a corpus, select records, embed them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("config")
    g.add_argument("--config", default=None,
                   help="YAML file of run arguments. CLI flags override its values.")
    g.add_argument("--print-config", action="store_true",
                   help="Print the resolved (config + CLI) arguments and exit")

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", default="nfcorpus",
                   help=f"One of {sorted(DATASETS)} or beir/<subset> "
                        f"({sorted(BEIR_SUBSETS)})")
    g.add_argument("--data-root", default="data",
                   help="Cache/corpus root (BEIR looks for <root>/beir/<subset>/)")
    g.add_argument("--max-records", type=int, default=None,
                   help="Truncate the corpus at load time (streaming early-stop)")

    g = p.add_argument_group("selection (precedence: indices > ids > contains > n)")
    g.add_argument("--n", type=int, default=16,
                   help="Number of records to draw at random")
    g.add_argument("--indices", default=None,
                   help="Comma-separated corpus positions, e.g. 0,7,42")
    g.add_argument("--ids", default=None,
                   help="Comma-separated corpus-native record ids")
    g.add_argument("--contains", default=None,
                   help="Keep only records containing this substring (case-insensitive)")
    g.add_argument("--seed", type=int, default=0, help="RNG seed for the random draw")
    g.add_argument("--all", action="store_true",
                   help="Select the whole (possibly truncated) corpus; ignores --n")

    g = p.add_argument_group("model")
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"One of {sorted(MODELS)} or st:<ckpt> / hf:<ckpt>")
    g.add_argument("--model-id", default=None,
                   help="Override the checkpoint for the chosen model class")
    g.add_argument("--device", default=None, help="cuda | cuda:0 | cpu | mps")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--max-seq-length", type=int, default=None)
    g.add_argument("--no-normalize", action="store_true",
                   help="Skip L2 normalisation of output vectors")

    g = p.add_argument_group("output")
    g.add_argument("--out", default=None,
                   help="Write the EmbeddingSet to this .npz path")
    g.add_argument("--embed-cache", default="data/embeddings",
                   help="Directory for cached embeddings ('none' to disable)")
    g.add_argument("--no-cache", action="store_true", help="Ignore and bypass the cache")
    g.add_argument("--preview", type=int, default=3,
                   help="How many selected records to print")
    g.add_argument("--preview-chars", type=int, default=160)
    g.add_argument("--show-similarity", action="store_true",
                   help="Print the cosine similarity matrix over the selection")
    g.add_argument("--stats-json", default=None,
                   help="Write a run manifest (dataset/model/selection stats) here")
    g.add_argument("--samples", type=int, default=3,
                   help="Random (id, index, embedding) triples to print at the end; 0 disables")
    g.add_argument("--sample-dims", type=int, default=8,
                   help="Leading vector components to show per sampled embedding")
    g.add_argument("--sample-seed", type=int, default=None,
                   help="Seed for the sample draw (default: --seed, so a run is repeatable)")
    return p


def parse_int_list(raw: str | list | None) -> list[int] | None:
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def parse_str_list(raw: str | list | None) -> list[str] | None:
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [x for x in (s.strip() for s in raw.split(",")) if x]


SECTION_NAME_KEY = {"dataset": "dataset", "model": "model"}
SECTIONS = ("dataset", "selection", "model", "output")


def load_config(
    path: str | Path,
    parser: argparse.ArgumentParser | None = None,
    sections: tuple[str, ...] = SECTIONS,
    section_name_key: dict[str, str] | None = None,
) -> dict:
    try:
        import yaml
    except ImportError as exc: 
        raise SystemExit("--config needs PyYAML: pip install pyyaml") from exc

    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config must be a YAML mapping, got {type(raw).__name__}")

    parser = parser if parser is not None else build_parser()
    section_name_key = section_name_key if section_name_key is not None else SECTION_NAME_KEY
    known = {a.dest for a in parser._actions} - {"help", "config", "print_config"}
    flat: dict = {}

    def put(key: str, value, section: str | None = None) -> None:
        dest = str(key).replace("-", "_")

        if section and dest in ("name", "method") and section in section_name_key:
            dest = section_name_key[section]

        if dest not in known and section and f"{section}_{dest}" in known:
            dest = f"{section}_{dest}"
        if dest not in known:
            where = f"{section}.{key}" if section else str(key)
            raise SystemExit(
                f"{path}: unknown config key {where!r}. Known keys: {sorted(known)}"
            )
        flat[dest] = value

    for key, value in raw.items():
        section = str(key).replace("-", "_")
        if isinstance(value, dict) and section in sections:
            for k, v in value.items():
                put(k, v, section)
        else:
            put(key, value)
    return flat


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        parser.set_defaults(**load_config(args.config))
        args = parser.parse_args(argv)
    return args


def build_dataset(args: argparse.Namespace):
    kwargs: dict = {
        "cache_dir": args.data_root,
        "max_records": args.max_records,
        "seed": args.seed,
    }
    return get_dataset(args.dataset, **kwargs)


def build_model(args: argparse.Namespace):
    cache_dir = None if (args.no_cache or args.embed_cache == "none") else args.embed_cache
    kwargs: dict = {
        "device": args.device,
        "batch_size": args.batch_size,
        "normalize": not args.no_normalize,
        "max_seq_length": args.max_seq_length,
        "cache_dir": cache_dir,
    }
    if args.model_id:
        kwargs["model_id"] = args.model_id
    return get_model(args.model, **kwargs)


def select(dataset, args: argparse.Namespace) -> Selection:
    return dataset.select(
        n=None if args.all else args.n,
        indices=parse_int_list(args.indices),
        ids=parse_str_list(args.ids),
        contains=args.contains,
        seed=args.seed,
    )


def report(selection: Selection, emb: EmbeddingSet, args: argparse.Namespace) -> None:
    print(f"\nselection {len(selection)} record(s) from {selection.dataset}")
    for rec, idx in list(zip(selection.records, selection.indices))[: args.preview]:
        text = rec.text.replace("\n", " ")
        if len(text) > args.preview_chars:
            text = text[: args.preview_chars] + "…"
        print(f"  [{idx}] {rec.id}: {text}")
    if len(selection) > args.preview:
        print(f"  … {len(selection) - args.preview} more")

    norms = np.linalg.norm(emb.vectors, axis=1)
    print(f"\nembeddings {emb}")
    print(f"  norm  mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")

    if args.show_similarity and len(emb) > 1:
        sims = emb.vectors @ emb.vectors.T
        off = sims[~np.eye(len(emb), dtype=bool)]
        print(f"  cosine off-diagonal: mean={off.mean():.4f} "
              f"min={off.min():.4f} max={off.max():.4f}")
        width = min(len(emb), 8)
        print(f"  similarity matrix (first {width}×{width}):")
        for row in sims[:width, :width]:
            print("    " + " ".join(f"{v:6.3f}" for v in row))


def print_samples(selection: Selection, emb: EmbeddingSet, args: argparse.Namespace) -> None:
    if args.samples <= 0 or len(emb) == 0:
        return

    seed = args.seed if args.sample_seed is None else args.sample_seed
    rng = random.Random(seed)
    k = min(args.samples, len(emb))
    rows = sorted(rng.sample(range(len(emb)), k))
    dims = max(1, min(args.sample_dims, emb.dim))
    tail = " …" if dims < emb.dim else ""

    print(f"\nsamples {k} random row(s) of {len(emb)} "
          f"(seed={seed}; showing dims 0-{dims - 1} of {emb.dim})")
    for row in rows:
        vec = emb.vectors[row]
        head = " ".join(f"{v:+.4f}" for v in vec[:dims])
        text = selection.records[row].text.replace("\n", " ")[:48]
        print(f"  row={row:<6d} id={emb.ids[row]:<16s} index={emb.indices[row]:<8d} "
              f"‖v‖={float(np.linalg.norm(vec)):.4f}")
        print(f"    embedding: [{head}{tail}]")
        print(f"    text:      {text}…")


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(argv)

    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0

    dataset = build_dataset(args)
    dataset.load()
    print(f"dataset {dataset}  {dataset.stats()}")

    selection = select(dataset, args)

    model = build_model(args)
    print(f"model {model}")

    emb = model.embed(selection, use_cache=not args.no_cache)
    report(selection, emb, args)

    if args.out:
        path = emb.save(args.out)
        print(f"\nsaved {path}")
    elif model.cache_dir is not None and not args.no_cache:
        print(f"\ncached {model.cache_path(selection)}")

    if args.stats_json:
        manifest = {
            "dataset": dataset.stats(),
            "model": {"name": model.name, "model_id": model.model_id,
                      "device": model.device, "normalize": model.normalize,
                      "dim": emb.dim},
            "selection": {"n": len(selection), "indices": selection.indices,
                          "ids": selection.ids, "seed": args.seed},
            "args": vars(args),
        }
        out = Path(args.stats_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"manifest {out}")

    print_samples(selection, emb, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
