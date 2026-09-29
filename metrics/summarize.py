from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .metrics import METRICS, build_ladder, classify

ATTACK_NAMES = ("algen", "steer", "teia", "zero2text")

_ARM = re.compile(
    r"^(?:(?P<attack>" + "|".join(a for a in ATTACK_NAMES if a != "algen") + r")_)?"
    r"(?P<d>dp_gaussian|remote_rag|gaussian|lapmech|purmech|sparse|cmag|vec2text"
    r"|eguard|idct|shuffling|wet|masking|keyed_rotation)"
    r"(?:_scope-(?P<scope>targets|both))?"
    r"(?:_(?P<kind>sigma|eps|lam|r)(?P<val>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))?"
)
_FIELD = {
    "gaussian": "noise_level", "lapmech": "epsilon", "dp_gaussian": "epsilon",
    "purmech": "epsilon", "sparse": "sparse_epsilon", "cmag": "cmag_epsilon",
    "vec2text": "vec2text_noise_level", "remote_rag": "remote_rag_radius",
}


class Arm(NamedTuple):

    attack: str         
    defense: str
    scope: str | None   
    field: str | None   
    value: float | None
    kind: str | None     # sigma | eps | lam | r

    @property
    def knob(self) -> str:
        sym = {"sigma": "\u03c3", "eps": "\u03b5", "lam": "\u03bb", "r": "r"}
        return "\u2014" if self.value is None else f"{sym[self.kind]}={self.value:g}"


def parse_arm_full(name: str) -> Arm | None:
    m = _ARM.match(name)
    if not m:
        return None
    d = m.group("d")
    v = m.group("val")
    return Arm(
        attack=m.group("attack") or "algen",
        defense=d,
        scope=m.group("scope"),
        field=_FIELD.get(d) if v is not None else None,
        value=float(v) if v is not None else None,
        kind=m.group("kind"),
    )


def parse_arm(name: str) -> tuple[str, str | None, float | None] | None:
    m = _ARM.match(name)
    if not m:
        return None
    d = m.group("d")
    if m.group("val") is None:
        return d, None, None
    return d, _FIELD.get(d), float(m.group("val"))


def load_arms(d: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in sorted(d.glob("*.json")):
        try:
            payload = json.loads(q.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list) and payload and all(
            isinstance(r, dict) and "test_results" in r for r in payload
        ):
            out[q.stem] = payload
    return out


def _buckets(dataset: str, ann_tag: str | None = None) -> dict[str, dict[str, float]]:
    f = Path("metrics/outputs/matched.json")
    if not f.exists():
        return {}
    try:
        d = json.loads(f.read_text()).get(str(dataset), {})
    except (OSError, json.JSONDecodeError):
        return {}
    if ann_tag is not None:
        d = {b: a for b, a in d.items() if b.startswith("ann@") and b.endswith(f"[{ann_tag}]")}
    return {b: {k: v["matched_value"] for k, v in arms.items() if v.get("converged")}
            for b, arms in d.items()}


def _bucket_title(b: str) -> str:
    if b.startswith("ann@"):
        head = b.split("[", 1)[0]                 
        k, target = head[4:].split("=")
        return f"matched on ANN recall@{k} = {float(target):.2f}"
    return {"recall@10=0.9": "matched on recall@10 = 0.90",
            "no knob": "no continuous knob \u2014 cannot be matched"}.get(b, b)


def _bucket_of(a: "Arm", buckets: dict[str, dict[str, float]]) -> str:
    if a.value is None:
        return "no knob"
    for b, solved in buckets.items():
        v = solved.get(a.defense)
        if v is not None and abs(v - a.value) <= abs(v) * 1e-4:
            return b
    return "other"


def _sortkey(r: dict[str, Any]) -> float:
    for a in ATTACK_NAMES:
        if r.get(a) is not None:
            return r[a].value
    return float("inf")


def _utility(args, probe, directory, arm: str, defense: str, kwargs, key_args=None):
    from ANN import read_utility, utility_key, write_utility
    key = utility_key(key_args or args, arm)
    if not args.force:
        hit = read_utility(directory, key)
        if hit is not None:
            return hit
    u = probe.measure(defense, kwargs, curve=True)
    write_utility(directory, key, u)
    return u


def compare_flows(args, d: Path, treat: Path) -> int:
    from ANN import AnnProbe, load_corpus, partition_from_args, spec_from_args

    base = {k: v for k, v in load_arms(d).items()
            if k in ("floor_prior_only", "floor_random_vector", "undefended")}
    flat_arms = {k: v for k, v in load_arms(d).items() if k not in base}
    part_arms = load_arms(treat)
    if not part_arms:
        raise SystemExit(f"no partitioned arms at {treat}")
    mh = getattr(args, "min_headroom", None)
    lad_flat = build_ladder({**base, **flat_arms}, min_headroom=mh)
    lad_part = build_ladder({**base, **part_arms}, min_headroom=mh)

    corpus = load_corpus(args)                      # one corpus, both probes
    spec = spec_from_args(args)
    kw = dict(k=args.match_k, query_side=args.ann_query_side, repeats=args.match_repeats,
              seed=args.seed, device=args.device)
    probes = {"flat": AnnProbe(corpus, spec, partition=None, cache_dir=d, **kw),
              "part": AnnProbe(corpus, spec, partition=partition_from_args(args),
                               cache_dir=treat, **kw)}
    key_args = {"flat": argparse.Namespace(**{**vars(args), "partition": False}),
                "part": args}

    names = sorted(set(flat_arms) | set(part_arms))
    attacks = sorted({(parse_arm_full(a) or _A).attack for a in names})
    print(f"\ncompare: {len(names)} arm(s), WITHOUT vs WITH the partitioned rotated index")
    print(f"compare floors and ceiling are shared: floor={lad_flat.floor['token_f1']:.4f} "
          f"ceiling={lad_flat.ceiling['token_f1']:.4f}")
    rows = []
    for arm in names:
        parsed = parse_arm(arm)
        if parsed is None:
            continue
        defense, field, value = parsed
        if field and value is not None:
            setattr(args, field, value); args.epsilon = value
        args.defense = defense
        from .ladder import _defense_kwargs
        info = parse_arm_full(arm) or _A
        cell = {"attack": info.attack, "defense": defense, "knob": info.knob}
        for flow, lad, pool in (("flat", lad_flat, flat_arms), ("part", lad_part, part_arms)):
            if arm not in pool:
                continue
            u = _utility(args, probes[flow], d if flow == "flat" else treat, arm, defense,
                         _defense_kwargs(args, defense), key_args=key_args[flow])
            cell[f"{flow}_recall"] = u.recall
            cell[f"{flow}_leak"] = lad.norms[arm]["token_f1"]
            cell[f"{flow}_f1"] = lad.raw[arm]["token_f1"]
            cell[f"{flow}_knob"] = info.knob
            if flow == "part":
                cell["cells"] = u.partition.get("cells_effective")
        rows.append(cell)

    rows = _pair_across_knobs(rows)

    cell_maps = bool(getattr(args, "algen_cell_maps", False))
    local = int(getattr(args, "z2t_local_pairs", 0) or 0) > 0
    if cell_maps:
        tag = "algen-cellmap" + ("-norot" if getattr(args, "kr_no_rotate", False) else "")
        rows = [{**r, "attack": tag if r["attack"] == "algen" else r["attack"]} for r in rows]
        attacks = [tag if a == "algen" else a for a in attacks]
    _print_compare(rows, attacks, partition_aware=cell_maps or local)
    out = treat / "compare_partition.json"
    out.write_text(json.dumps([{k: (v.to_dict() if hasattr(v, "to_dict") else v)
                                for k, v in r.items()} for r in rows], indent=2, default=str))
    print(f"\n  out= {out}")
    return 0


def _pair_across_knobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["attack"], r["defense"]), []).append(r)
    out: list[dict[str, Any]] = []
    for grp in groups.values():
        flat_only = [r for r in grp if "flat_leak" in r and "part_leak" not in r]
        part_only = [r for r in grp if "part_leak" in r and "flat_leak" not in r]
        if len(flat_only) == 1 and len(part_only) == 1:
            f, pt = flat_only[0], part_only[0]
            merged = {**f, **{k: v for k, v in pt.items()
                              if k.startswith("part_") or k == "cells"}}
            merged["paired_across_knobs"] = True
            out.extend(r for r in grp if r is not f and r is not pt)
            out.append(merged)
        else:
            out.extend(grp)
    return out


def _print_compare(rows: list[dict[str, Any]], attacks: list[str],
                   partition_aware: bool = False) -> None:
    aw = max(len("attack"), *map(len, attacks)) + 1
    acol = f"{'attack':<{aw}}"
    paired = any(r.get("paired_across_knobs") for r in rows)
    kw = 14 if paired else 0                    
    hdr = (f"  {'':<{aw}}{'':<15}{'':>12}"
           + f"{'--- WITHOUT partition ---':^26}"
           + f"{'---- WITH partition ----':^{26 + kw}}" + f"{'':>13}")
    sub = (f"  {acol}{'defense':<15}{'knob':>12}{'ANN r@10':>9}{'token F1':>9}{'leak':>8}"
           + (f"{'knob':>{kw}}" if paired else "")
           + f"{'ANN r@10':>9}{'token F1':>9}{'leak':>8}{'delta leak':>13}")
    print(f"\n{hdr}\n{sub}\n  {'-' * (len(sub) - 2)}")
    fmt = lambda v, w, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float) and v == v else f"{'-':>{w}}")
    for r in sorted(rows, key=lambda r: (r["attack"],
                                         r.get("part_leak").value if r.get("part_leak") else 9)):
        fl, pl = r.get("flat_leak"), r.get("part_leak")
        delta = (pl.value - fl.value) if (fl and pl and fl.valid and pl.valid) else float("nan")
        if not paired:
            pknob = ""
        elif r.get("paired_across_knobs"):
            pknob = f"{r['part_knob']:>{kw}}"
        else:
            pknob = f"{('=' if 'part_leak' in r else '-'):>{kw}}"
        print(f"  {r['attack']:<{aw}}"
              f"{r['defense']:<15}{r['knob']:>12}"
              + fmt(r.get("flat_recall"), 9) + fmt(r.get("flat_f1"), 9, 4)
              + (f"{fl.value:>8.3f}" if fl and fl.valid else f"{'-':>8}")
              + pknob
              + fmt(r.get("part_recall"), 9) + fmt(r.get("part_f1"), 9, 4)
              + (f"{pl.value:>8.3f}" if pl and pl.valid else f"{'-':>8}")
              + (f"{delta:>+13.3f}" if delta == delta else f"{'-':>13}"))
    if paired:
        print("\n  WITH-side knob: '=' same knob as WITHOUT, '-' no partitioned arm. A value "
              "means the matcher\n  solved the partitioned layout to a DIFFERENT knob (the "
              "partition moved undefended recall);\n  such rows are paired at matched ANN "
              "recall, not at the same knob, so their delta leak\n  mixes the layout with "
              "a changed defense strength.")
    print(f"\n  Same defense, two storage layouts. leak_norm 1.0 = the defense "
          f"removed nothing;\n  0.0 = the attack is at its corpus prior; below 0 = the "
          f"attack scores WORSE than the prior,\n  which is what a single ridge map fitted "
          f"across mutually rotated cells produces.")
    aware = ", ".join(attacks).upper()
    if partition_aware:
        print(f"  !! {aware} IS partition-aware (a map per cell / per target), so the WITH "
              f"column is NOT the one-map\n     upper bound: it is what an attacker who "
              f"knows the mechanism (but not the keys) recovers.")
    else:
        print(f"  !! {aware} fits ONE map across every cell here, so the WITH column is an "
              f"UPPER BOUND on protection.\n"
              f"     zero2text can be made partition-aware with --z2t-local-pairs k "
              f"(./run_zero2text.sh local).")


class _A:
    knob = "\u2014"
    attack = "algen"


def main(argv: list[str] | None = None) -> int:
    from main import load_config

    from .ladder import CONFIG_SECTIONS, build_parser
    from .utility_match import eval_embeddings_for

    p = build_parser()
    p.prog = "python -m metrics.summarize"
    p.add_argument("--ladder-dir", default=None,
                   help="ladder cache directory; default is the newest for this config")
    p.add_argument("--compare-partition", action="store_true",
                   help="one table with every arm WITH and WITHOUT the partitioned, "
                        "rotated index (needs both ladders run; --partition selects which "
                        "partition settings to compare against)")
    args0 = p.parse_args(argv)
    if args0.config:
        p.set_defaults(**load_config(
            args0.config, parser=p,
            sections=CONFIG_SECTIONS,
            section_name_key={"dataset": "dataset", "model": "model", "defense": "defense"},
        ))
        args = p.parse_args(argv)
    else:
        args = args0

    ann = args.match_metric == "ann"
    probe = None
    if args.partition and not ann:
        raise SystemExit("--partition is an index layout: it needs --match-metric ann")
    if ann:
        from ANN import AnnProbe, ladder_subdir, partition_subdir, probe_tag

        probe = AnnProbe.from_args(args)

    if args.ladder_dir:
        d = Path(args.ladder_dir)
    elif args.checkpoint:
        d = (Path(args.ladder_out) / Path(args.checkpoint).name
             / f"k{args.align_samples}_ridge{args.reg_lambda}_{args.align_text}")
    else:
        tag = str(args.attack_dataset).replace("/", "_")
        dirs = sorted(Path(args.ladder_out).glob(f"{tag}_*/k*"))
        if not dirs:
            raise SystemExit(f"no ladder output under {args.ladder_out} for {tag}")
        if len(dirs) > 1:
            print(f"Warn {len(dirs)} ladder directories match {tag}_*/k*; using the "
                  f"newest by name. Pass --checkpoint (or --ladder-dir) to choose:\n"
                  + "\n".join(f"         {x}" for x in dirs))
        d = dirs[-1]
    if ann and not args.ladder_dir:
        d = d / ladder_subdir(args, probe.corpus.vectors.shape[1])
    if not d.exists():
        raise SystemExit(f"no ladder output at {d} -- run the ladder first")
    if probe is not None:
        probe.cache_dir = d

    if args.partition:
        treat = d / partition_subdir(args)
        if not treat.exists():
            raise SystemExit(f"no partitioned arms at {treat} -- run the ladder with "
                             f"--partition first")
        if args.compare_partition:
            return compare_flows(args, d, treat)
        base_arms = {k: v for k, v in load_arms(d).items()
                     if k in ("floor_prior_only", "floor_random_vector", "undefended")}
        arms = {**base_arms, **load_arms(treat)}
        out_dir = treat
    else:
        arms = load_arms(d)
        out_dir = d
    if not arms:
        raise SystemExit(f"no arm files in {d} — run the ladder first")
    lad = build_ladder(arms, min_headroom=getattr(args, "min_headroom", None))
    treatments = sorted(lad.norms)
    print(f"[summarize] {out_dir}\n[summarize] {len(treatments)} treatment arm(s); "
          f"re-measuring utility at each arm's own knob")

    from .ladder import _defense_kwargs

    X = None if ann else eval_embeddings_for(args)
    util: dict[str, float] = {}
    ann_util: dict[str, Any] = {}

    for arm in treatments:
        parsed = parse_arm(arm)
        if parsed is None:
            continue
        defense, field, value = parsed
        if field and value is not None:
            setattr(args, field, value)
            args.epsilon = value  
        args.defense = defense
        try:
            u = _utility(args, probe, out_dir, arm, defense, _defense_kwargs(args, defense))
            ann_util[arm], util[arm] = u, u.recall
        except Exception as exc:
            print(f"  [warn] recall@k failed for {arm}: {type(exc).__name__}: {exc}")

    head = lad.ceiling["token_f1"] - lad.floor["token_f1"]
    print(f"\n  {args.attack_dataset} / ALGEN    floor={lad.floor['token_f1']:.4f}  "
          f"ceiling={lad.ceiling['token_f1']:.4f}  "
          f"oracle={lad.oracle.get('token_f1', float('nan')):.4f}  "
          f"headroom={head:.4f} " + ("OK" if head >= 0.10 else "< 0.10 -> NO SIGNAL"))
    if ann:
        print(f"  ANN: {probe.describe()} over {probe.corpus.describe()}, "
              f"queries {probe.query_side}")
        base = probe.clean_baseline()
        print(f"  undefended index-only ANN recall@{probe.k} = "
              f"{base['index_recall']:.4f}   (the cap for every arm"
              + (", measured in the partitioned index)" if args.partition else ")"))
        print(f"  attacks invert the vectors as this index stores them")
        if args.partition:
            pr = base["partition"]
            print(f"  partition (undefended corpus): {pr['cells_effective']} cells "
                  f"(nominal {pr['cells_nominal']}), lists {pr['list_min']}-{pr['list_max']} "
                  f"(median {pr['list_median']:.0f}), largest {pr['max_share']:.2%}, "
                  f"gini {pr['gini']:.3f}, N/max {pr['effective_security']:.0f}, "
                  f"voronoi {pr['voronoi_agreement']:.3f}")
            print(f"  ceiling and floors: UNPARTITIONED index. !! ALGEN is not partition-aware"
                  f" -> leak_norm is an\n  UPPER BOUND on protection "
                  f"(defense/keyed_rotation.py AMBIGUITIES['attacker_not_partition_aware']).")
    print()

    buckets = _buckets(args.attack_dataset, probe_tag(args) if ann else None)
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in treatments:
        a = parse_arm_full(arm)
        if a is None:
            continue
        key = (_bucket_of(a, buckets), a.defense, a.knob)
        r = rows.setdefault(key, {"recall": float("nan"), "ann": None})
        r[a.attack] = lad.norms[arm]["token_f1"]
        r[f"{a.attack}_rouge"] = lad.norms[arm]["rougeL"]
        r[f"{a.attack}_f1"] = lad.raw[arm]["token_f1"]   # raw, before normalising
        if arm in util:
            r["recall"] = util[arm]
        if arm in ann_util:
            r["ann"] = ann_util[arm]
        # Pairs in each target's own cell: what a per-cell attacker would have had.
        diag = [x.get("diagnostics", {}).get("partition") for x in arms.get(arm, [])]
        diag = [x for x in diag if x]
        if diag:
            r["p_cell"] = float(np.mean([x["target_cell_pairs_mean"] for x in diag]))

    order = {"recall@10=0.9": 0, "no knob": 2, "other": 3}

    ucol = f"ANN r@{args.match_k}" if ann else "recall@10"
    col2 = f"exact r@{args.match_k}" if ann else "NDCG kept"

    f1_und = lad.ceiling["token_f1"]
    present = [a for a in ATTACK_NAMES
               if any(a in r for r in rows.values())]
    pair = "algen" in present and "steer" in present
    verdict_of = "steer" if "steer" in present else present[-1] if present else None
    part = bool(args.partition)
    pcols = f"{'cells':>7}{'N/max':>7}{'P/cell':>8}" if part else ""
    grp = (f"  {'':<15}{'':>12}{'':>11}{'':>11}" + (f"{'':>7}" if ann else "")
           + (f"{'':>22}" if part else "")
           + f"{'--- token F1 ---':^{9 * (len(present) + 1)}}"
           f"{'-- leak_norm --':^{9 * len(present) + (8 if pair else 0)}}")
    hdr = (f"  {'defense':<15}{'knob':>12}{ucol:>11}{col2:>11}"
           + (f"{'np x' if part else 'ef x':>7}" if ann else "") + pcols + f"{'undef':>9}"
           + "".join(f"{a.upper():>9}" for a in present)
           + "".join(f"{a.upper():>9}" for a in present)
           + (f"{'swing':>8}" if pair else "")
           + (f"  verdict ({verdict_of.upper()})" if verdict_of else ""))
    for b in sorted({k[0] for k in rows},
                    key=lambda x: order.get(x, -1 if x.startswith("ann@") else 9)):
        title = _bucket_title(b)
        print(f"\n  {title}\n  {'-' * (len(hdr) - 2)}")
        print(grp)
        print(hdr)
        sub = {k: v for k, v in rows.items() if k[0] == b}
        for (_, defense, knob), r in sorted(sub.items(), key=lambda kv: _sortkey(kv[1])):
            def cell(who: str) -> str:
                n = r.get(who)
                return "-" if n is None else (f"{n.value:.3f}" if n.valid else "no sig")
            swing = ""
            if pair:
                a, b = r.get("algen"), r.get("steer")
                swing = (f"{b.value - a.value:+.2f}" if a and b and a.valid and b.valid
                         else "-")
                swing = f"{swing:>8}"
            verdict = "-"
            n = r.get(verdict_of) if verdict_of else None
            if n is not None:
                verdict = classify(
                    {"token_f1": n, "rougeL": r.get(f"{verdict_of}_rouge", n)},
                    recall_at_k=None if r["recall"] != r["recall"] else r["recall"],
                )[0]
            def raw(who: str) -> str:
                v = r.get(f"{who}_f1")
                return "-" if v is None or v != v else f"{v:.4f}"
            if ann:
                u = r["ann"]
                c2 = f"{u.exact_recall:.3f}" if u is not None else "-"
                sc = f"{u.surcharge:.1f}" if u is not None and u.surcharge == u.surcharge else "-"
                c2 = f"{c2:>11}{sc:>7}"
                if part:
                    pr = u.partition if u is not None else {}
                    pc = r.get("p_cell")
                    cells = str(pr["cells_effective"]) if pr else "-"
                    nmax = f"{pr['effective_security']:.0f}" if pr else "-"
                    pcell = f"{pc:.1f}" if pc is not None else "-"
                    c2 += f"{cells:>7}{nmax:>7}{pcell:>8}"
            else:
                c2 = f"{'-':>11}"
            print(f"  {defense:<15}{knob:>12}{r['recall']:>11.3f}"
                  f"{c2}"
                  f"{f1_und:>9.4f}"
                  + "".join(f"{raw(a):>9}" for a in present)
                  + "".join(f"{cell(a):>9}" for a in present)
                  + swing + f"  {verdict}")

    if part:
        print(f"\n  cells = non-empty cells of the partition fitted on this arm's defended "
              f"corpus; N/max = N / largest\n  list, the effective frame count; P/cell = "
              f"mean leaked pairs sharing a target's cell — below\n  "
              f"d={probe.corpus.vectors.shape[1]} a per-cell attacker is underdetermined. "
              f"np x = nprobe surcharge.")
    if ann:
        bx = "np x" if part else "ef x"   
        print(f"\n  ANN r@k = recall@k of the defended index at the production search budget,"
              f" against exact\n  search over the CLEAN corpus. exact r@k = the same with"
              f" exact search on the defended\n  vectors (the defense's own cost). {bx} ="
              f" search-budget multiple the defended index needs\n  to be as navigable as"
              f" the clean one; inf = never on the grid.")
    print(f"\n  token F1 is bag-of-words overlap between the RECOVERED TEXT and the"
          f" original; 'undef'\n  is the undefended ceiling ({f1_und:.4f}), shared by both"
          f" attacks. Floor = {lad.floor['token_f1']:.4f}.")
    print(f"  leak_norm on token_f1: 1.0 = defense removed nothing, 0.0 = attack is at its"
          f" corpus prior.\n  swing = STEER - ALGEN. ALGEN refits its map through the defense;"
          f" STEER fits on clean\n  pairs and cannot adapt, so a large negative swing means the"
          f" defense was only ever\n  surviving because the attacker was allowed to observe it.")

    out = out_dir / "summary.json"
    out.write_text(json.dumps(
        {"dir": str(d), "floor": lad.floor, "ceiling": lad.ceiling, "oracle": lad.oracle,
         "headroom_token_f1": head,
         "recall_at_k": util,
         "utility_metric": args.match_metric,
         "ann": {a: u.to_dict() for a, u in ann_util.items()},
         "leak_norm": {a: {k: v.to_dict() for k, v in n.items()}
                       for a, n in lad.norms.items()}}, indent=2, default=str))
    print(f"\n  [out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
