from __future__ import annotations

import json
import sys
from pathlib import Path


def _args(argv: list[str], extra) -> "object":
    from main import load_config

    from metrics.ladder import CONFIG_SECTIONS, build_parser
    p = build_parser()
    extra(p)
    args = p.parse_args(argv)
    if args.config:
        p.set_defaults(**load_config(
            args.config, parser=p, sections=CONFIG_SECTIONS,
            section_name_key={"dataset": "dataset", "model": "model", "defense": "defense"},
        ))
        args = p.parse_args(argv)
    return args


def cmd_probe(argv: list[str]) -> int:
    from metrics.ladder import _defense_kwargs

    from .probe import AnnProbe

    def extra(p):
        p.add_argument("--curve", action="store_true", help="also sweep the search budget")

    args = _args(argv, extra)
    probe = AnnProbe.from_args(args)
    u = probe.measure(args.defense, _defense_kwargs(args, args.defense), curve=args.curve)
    print(f"\n[ann] {args.defense} on {probe.corpus.describe()}")
    print(f"      {probe.describe()}, queries {u.query_side}")
    print(f"  ANN recall@{u.k}    {u.recall:.4f} +- {u.recall_se:.4f}   <- utility")
    print(f"  exact recall@{u.k}  {u.exact_recall:.4f}   (defense only)")
    print(f"  index recall@{u.k}  {u.index_recall:.4f}   (clean index, index only)")
    print(f"  self recall@{u.k}   {u.self_recall:.4f}   (ANN vs exact on the defended index)")
    if u.partition:
        print("\n  partition: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in u.partition.items()))
    if u.curve:
        label = u.budget_param or "exact"
        print(f"\n  {label:>7} {'recall':>8} {'self':>8}")
        for b, r, s in u.curve:
            print(f"  {b:>7} {r:>8.4f} {s:>8.4f}")
        print(f"  surcharge: {u.surcharge:.2f}x {label}")
    return 0


def cmd_knob(argv: list[str]) -> int:
    from .probe import bucket_key

    def extra(p):
        p.add_argument("--target", type=float, required=True)
        p.add_argument("--match-out", default="metrics/outputs/matched.json")

    args = _args(argv, extra)
    f = Path(args.match_out)
    d = json.loads(f.read_text()).get(str(args.attack_dataset), {}) if f.exists() else {}
    v = d.get(bucket_key(args, args.target), {}).get(args.defense)
    print(f"{v['matched_value']:.10g}" if v and v.get("converged") else "")
    return 0


def cmd_selftest(argv: list[str]) -> int:
    import numpy as np
    import torch

    from .corpus import ANNCorpus
    from .index import IndexSpec, IndexStorage, build_index, prepare
    from .probe import AnnProbe, _drop_self, overlap

    rng = np.random.default_rng(0)
    # Clustered data so neighbourhoods exist; unit norm like gtr-base.
    centers = rng.standard_normal((200, 128)).astype(np.float32)
    X = centers[rng.integers(0, 200, 20_000)] + 0.3 * rng.standard_normal((20_000, 128)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    corpus = ANNCorpus(vectors=X, ids=[str(i) for i in range(len(X))], dataset="synthetic",
                       model="none", query_rows=np.arange(500))
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    spec = IndexSpec("HNSW32,Flat", "cosine", 64)
    st = IndexStorage.untrained(spec, 128)
    Y = 3.0 * X[:5]
    check("flat storage is lossless", st is not None and st.lossless)
    check("cosine storage stores unit vectors", np.allclose(st(Y), X[:5], atol=1e-6))
    check("ip storage keeps magnitude",
          np.allclose(IndexStorage.untrained(IndexSpec("HNSW32,Flat", "ip"), 128)(Y), Y))
    check("storage round-trips torch tensors",
          torch.allclose(st(torch.tensor(Y)), torch.tensor(X[:5]), atol=1e-6))

    idx = build_index(prepare(Y, "cosine"), spec)
    check("storage == index reconstruct",
          np.allclose(IndexStorage.from_index(idx, spec)(Y), idx.reconstruct_n(0, 5)))
    sq = IndexSpec("HNSW32,SQ8", "cosine", 64)
    idx = build_index(prepare(X, "cosine"), sq)
    s8 = IndexStorage.from_index(idx, sq)
    err = float(np.abs(s8(X[:100]) - X[:100]).max())
    check("SQ8 storage is lossy but close", (not s8.lossless) and 0 < err < 0.05, f"max err {err:.4f}")
    check("SQ8 storage == index reconstruct", np.allclose(s8(X[:5]), idx.reconstruct_n(0, 5)))

    I = np.array([[3, 1, 2], [4, 5, 6]])
    check("drop_self removes own row", _drop_self(I, np.array([1, 9]), 2).tolist() == [[3, 2], [4, 5]])

    probe = AnnProbe(corpus, spec, k=10, repeats=2, seed=0)
    exact = AnnProbe(corpus, IndexSpec("Flat", "cosine"), k=10, repeats=1, seed=0)
    none = probe.measure("none")
    er = exact.measure("none").recall
    check("undefended exact index has recall ~1", er > 0.999, f"{er:.4f}")
    check("undefended ANN recall ~ index recall", abs(none.recall - none.index_recall) < 3e-3,
          f"{none.recall:.4f} vs {none.index_recall:.4f}")
    lo = probe.measure("gaussian", {"noise_level": 0.005}, curve=True)
    hi = probe.measure("gaussian", {"noise_level": 0.05})
    check("more noise, less ANN recall", hi.recall < lo.recall < none.recall + 1e-3,
          f"{none.recall:.3f} > {lo.recall:.3f} > {hi.recall:.3f}")
    check("ANN recall <= exact recall (+jitter)", lo.recall <= lo.exact_recall + 0.01,
          f"{lo.recall:.3f} vs {lo.exact_recall:.3f}")
    check("budget curve is non-decreasing in recall",
          all(b[1] >= a[1] - 0.01 for a, b in zip(lo.curve, lo.curve[1:])))
    check("surcharge is finite", lo.surcharge == lo.surcharge and lo.surcharge < float("inf"),
          f"{lo.surcharge:.2f}x")

    loose = AnnProbe(corpus, IndexSpec("HNSW8,Flat", "cosine", 16), k=10, repeats=1, seed=0)
    sc0 = loose.measure("none", curve=True).surcharge
    check("undefended surcharge is ~1.0x", 0.75 < sc0 < 1.33, f"{sc0:.2f}x")

    from metrics.utility_match import recall_at_k

    Xt = torch.tensor(X[:2000])
    torch.manual_seed(0)
    noisy = Xt + 0.02 * torch.randn_like(Xt)
    bf = recall_at_k(Xt, noisy, k=10)
    small = ANNCorpus(vectors=X[:2000], ids=[str(i) for i in range(2000)], dataset="s",
                      model="none", query_rows=np.arange(2000))
    ex = AnnProbe(small, IndexSpec("Flat", "cosine"), k=10, repeats=1, seed=0)
    ex._defend = lambda d, kw: noisy.numpy()          # the identical draw
    check("exact-index probe == brute-force recall_at_k",
          abs(ex.measure("gaussian").recall - bf) < 1e-6, f"{bf:.4f}")


    from .partitioned import PartitionSpec, PartitionedStorage

    pt = PartitionSpec(m=500, nprobe=8, key="selftest")
    rot = AnnProbe(corpus, spec, k=10, repeats=1, seed=0, partition=pt)
    ur = rot.measure("none", curve=True)
 
    U = rng.standard_normal((20_000, 32)).astype(np.float32)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    flat_corpus = ANNCorpus(vectors=U, ids=[str(i) for i in range(len(U))], dataset="u",
                            model="none", query_rows=np.arange(500))
    uc = AnnProbe(flat_corpus, spec, k=10, repeats=1, seed=0, partition=pt).measure("none", curve=True)
    un = AnnProbe(flat_corpus, spec, k=10, repeats=1, seed=0,
                  partition=PartitionSpec(m=500, nprobe=8, key="selftest", rotate=False)
                  ).measure("none")
    check("rotation costs no recall given the partition", abs(uc.recall - un.recall) < 0.01,
          f"{uc.recall:.4f} rotated vs {un.recall:.4f} not")
    check("nprobe buys recall back", uc.curve[-1][1] > uc.curve[0][1] + 0.2,
          f"nprobe {uc.curve[0][0]}: {uc.curve[0][1]:.3f} -> {uc.curve[-1][0]}: {uc.curve[-1][1]:.3f}")
    pr = ur.partition
    check("partition respects the cap", pr["list_max"] <= int(pt.alpha * pt.m),
          f"largest list {pr['list_max']}, {pr['cells_effective']} cells, gini {pr['gini']:.3f}")
    st = rot.storage("none")
    Y = st(X[:300])
    cells = st.last_cells
    same = cells[:, None] == cells[None, :]
    G, H = X[:300] @ X[:300].T, Y @ Y.T
    check("partitioned storage is a per-cell isometry", isinstance(st, PartitionedStorage)
          and np.allclose(G[same], H[same], atol=1e-4) and not np.allclose(G[~same], H[~same], atol=1e-2))
    lin = np.linalg.lstsq(Y, X[:300], rcond=None)[0]
    one_map = float(np.linalg.norm(Y @ lin - X[:300]) / np.linalg.norm(X[:300]))
    check("one global linear map cannot undo per-cell keys", one_map > 0.2,
          f"relative residual {one_map:.2f}")

    print("\nselftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_assumptions(argv: list[str]) -> int:
    # from defense.k_mosaic import AMBIGUITIES

    from .partitioned import __doc__ as part_doc
    from .probe import ASSUMPTIONS

    print("=" * 88 + "\nANN utility probe (ANN/probe.py ASSUMPTIONS)\n" + "=" * 88)
    for k, v in ASSUMPTIONS.items():
        print(f"\n[{k}]\n  {v}")
    print("\n" + "=" * 88 + "\nPartitioned index (ANN/partitioned.py)\n" + "=" * 88)
    print(part_doc)
    print("=" * 88 + "\nk_mosaic (defense/k_mosaic.py AMBIGUITIES)\n"
          + "=" * 88)
    # for k, v in AMBIGUITIES.items():
    #     print(f"\n[{k}]")
    #     for f, text in v.items():
    #         print(f"  {f:>8}: {text}")
    return 0


COMMANDS = {"probe": cmd_probe, "knob": cmd_knob, "selftest": cmd_selftest,
            "assumptions": cmd_assumptions}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
