# k_mosaic

How much text an embedding-inversion attacker recovers from a vector database, per
defense, at a matched retrieval cost (ANN recall@10 = 0.90), on a flat HNSW index vs a
partitioned index with a secret rotation per cell. Headline metric:
`leak_norm = (defended − floor) / (ceiling − floor)` on token F1.

- **Attacks:** ALGEN, STEER, TEIA, Zero2Text, plus the adaptive per-cell ALGEN
  (`algen-cellmap`) and its no-rotation control (`algen-cellmap-norot`).
- **Defenses:** eguard, idct, gaussian, lapmech, sparse, cmag, vec2text, remote_rag,
  plus k_mosaic on the partitioned index.
- **Corpora:** Quora (523k docs) and MS MARCO (a uniform 1M subsample).
- **Victim encoders:** gtr-base (default) and st5.

## Setup

```bash
cd k_mosaic                      # .python-version selects the pyenv env `kmosaic` (Python 3.11.13)
pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements.lock.txt` pins the exact environment the results were produced with.
`requirements.txt` lists only the direct dependencies.

Models and datasets download to `~/.cache/huggingface` on first use (~25 GB). Quora
streams from the Hugging Face hub; MS MARCO is frozen locally by the `prepare` stage.

## Running

```bash
bash experiments/quora.sh   [STAGE ...] [FLAGS]
bash experiments/msmarco.sh [STAGE ...] [FLAGS]
```

**Stages** run in this order. With no stage given, all of them run (except `status`).

| Stage | Does |
|---|---|
| `prepare` | MS MARCO only: freeze the 1M subsample to `data/beir/msmarco-1m/` |
| `embed` | 3 × 128 target vectors (seeds 1, 2, 3) → `data/embeddings/` |
| `pretrain` | ALGEN stage 1 on trec-covid (shared by every corpus and victim) |
| `train` | ALGEN stage 2: the generator for this corpus and leaked-pair count |
| `defenses` | fit eguard, sparse, cmag, vec2text, remote_rag |
| `teia-train` | TEIA's GPT-2 decoder (GTE-base surrogate) |
| `match` | solve each defense's knob to ANN recall@10 = 0.90, per layout (also calibrates `--cell-sweep` counts) |
| `ladder` | attack every defense, one arm at a time |
| `report` | flat-vs-partitioned tables |
| `status` | read-only: what's trained, knobs, arms done/failed/skipped, what's running |

**Flags**

| Flag | Default | Meaning |
|---|---|---|
| `--leaked-samples N` | 1000 | leaked (text, vector) pairs: ALGEN/STEER alignment pairs and TEIA's D_L |
| `--cells C` | `N/m` | cell count of the partitioned index (`N/m` = default occupancy m=1000, ≈828 cells on Quora) |
| `--cell-sweep "C …"` | `N/m` | extra cell counts, e.g. `"20 200 2000"` |
| `--victim M` | gtr-base | victim encoder (e.g. `st5`) |
| `--attacks "…"` | `algen steer teia zero2text` | for `ladder`/`report` only; add `algen-cellmap algen-cellmap-norot` for the adaptive attack |
| `--cell-min-pairs N` | 1 | per-cell ALGEN: fewest pairs a cell needs for its own map |
| `--min-headroom X` | 0.10 | validity gate on ceiling − floor (MS MARCO needs 0.05) |
| `--gpu N` / `--cores N` | 0 / all | CUDA device / CPU threads (set both when runs share the machine) |

## Full pipeline from scratch

If possible, run each command inside `tmux`. Commands within a numbered block can run in parallel.

```bash
# 0. inputs — one after another, before anything else
bash experiments/msmarco.sh prepare embed
bash experiments/quora.sh embed
bash experiments/quora.sh embed --victim st5

# 1. shared ALGEN stage 1 — once
bash experiments/quora.sh pretrain

# 2. training + knobs (parallel)
bash experiments/quora.sh   defenses match --cell-sweep "20 200 2000"  --gpu 0 --cores 64
bash experiments/msmarco.sh defenses match --min-headroom 0.05          --gpu 1 --cores 32
bash experiments/quora.sh   --victim st5                                 --gpu 2 --cores 32   # full st5 pipeline
bash experiments/quora.sh train teia-train --gpu 3 --cores 32 && bash experiments/msmarco.sh train teia-train --gpu 3 --cores 32

# 3. ladders + reports, after ALL of step 2 (parallel; split by attack, never split algen/steer)
bash experiments/quora.sh ladder report --attacks "algen steer" --cell-sweep "20 200 2000" --gpu 0 --cores 48
bash experiments/quora.sh ladder report --attacks teia          --cell-sweep "20 200 2000" --gpu 1 --cores 48
bash experiments/quora.sh ladder report --attacks zero2text     --cell-sweep "20 200 2000" --gpu 2 --cores 48
bash experiments/msmarco.sh ladder report --attacks "algen steer" --min-headroom 0.05 --gpu 0 --cores 48   # then teia, zero2text alike

# 4. leaked-pair sweep (parallel; each trains its own generator + decoder)
bash experiments/quora.sh --leaked-samples 2000 --gpu 0 --cores 48
bash experiments/quora.sh --leaked-samples 4000 --gpu 1 --cores 48
```

**Adaptive per-cell ALGEN** (Quora; needs step 2's `train` and `match --cell-sweep`):

```bash
# phase 1: floors, flat arms, C=N/m
bash experiments/quora.sh ladder --attacks algen-cellmap --gpu 0 --cores 48
# phase 2 (parallel)
bash experiments/quora.sh ladder --attacks algen-cellmap       --cell-sweep "20 200 2000"          --gpu 0 --cores 48
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 20   --cell-sweep "N/m 20"   --gpu 1 --cores 32
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 200  --cell-sweep 200        --gpu 2 --cores 32
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 2000 --cell-sweep 2000       --gpu 3 --cores 32
bash experiments/quora.sh report --attacks "algen algen-cellmap algen-cellmap-norot" --cell-sweep "20 200 2000"
```

To run sequentially instead, use one command:
`ladder --attacks "algen-cellmap algen-cellmap-norot" --cell-sweep "20 200 2000"`.

**Monitor progress:**

```bash
bash experiments/quora.sh status --cell-sweep "20 200 2000"
```

Pass the same `--leaked-samples`, `--victim`, `--attacks` and cell flags as the run you're
checking.

## Outputs

| Path | Contents |
|---|---|
| `data/` | frozen corpora, target sets |
| `ANN/cache/` | encoded corpora (1.6–3 GB each) |
| `attacker/outputs/verify/` | ALGEN stage 1; `<victim>/` defenses and TEIA decoders; `<victim>/pairs<N>/` generators |
| `metrics/outputs/verify/<victim>/` | ladder arms, `matched.json` (knobs), `logs/`, `zero2text/`, `algen_cellmaps/` |

Each report table is also written as `compare_partition.json` next to its arms.

## Rules for parallel runs

- **Never train the same thing twice at once.** Run shared stages (`embed`, `pretrain`,
  `defenses`, `match`, a new `--victim`) in one process first.


## Reproducibility

- **Seeds:** 42 everywhere (configs, splits, defense fits, and the k_mosaic HMAC key `keyed-rotation-42`, kept from
  VecSec so the rotations and folder hashes are unchanged); targets
  use seeds 1, 2, 3; the MS MARCO subsample uses `sample_seed=42`.
- **Training is deterministic** given the same inputs.
- **HNSW index builds are not deterministic:**, expect small differences in leak_norm (+- 0.01)
- **Knobs are matched to 0.90 ± 0.01**, on a coarse log grid, so different corpora often
  get identical knob values.
- **A knob that doesn't converge skips its arms** and prints `!! … knob did not
  converge`. Known cases: st5 gaussian, and remote_rag at C=200.
