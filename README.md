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
cd k_mosaic                      
# create a virtual environment with Python 3.11.13 and activate it
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


## Full pipeline from scratch

If possible, run each command inside `tmux`. Commands within a numbered block can run in parallel using --gpu flag

```bash
# 0. inputs — one after another, before anything else (~10-20 mins)
bash experiments/msmarco.sh prepare embed
bash experiments/quora.sh embed
bash experiments/quora.sh embed --victim st5

# 1. shared ALGEN stage 1 — once (~30 mins)
bash experiments/quora.sh pretrain

# 2. training + knobs (parallel) (~6-7 hrs)
bash experiments/quora.sh   defenses match --cell-sweep "20 200 2000"  
bash experiments/msmarco.sh defenses match --min-headroom 0.05          
bash experiments/quora.sh   --victim st5                             # full st5 pipeline
bash experiments/quora.sh train teia-train  
bash experiments/msmarco.sh train teia-train 

# 3. ladders + reports, after ALL of step 2 ends (parallel; split by attack, never split algen/steer) (~6-7 hrs)
bash experiments/quora.sh ladder report --attacks "algen steer" --cell-sweep "20 200 2000" 
bash experiments/quora.sh ladder report --attacks teia          --cell-sweep "20 200 2000" 
bash experiments/quora.sh ladder report --attacks zero2text     --cell-sweep "20 200 2000" 
bash experiments/msmarco.sh ladder report --attacks "algen steer" --min-headroom 0.05    # then teia, zero2text alike

# 4. leaked-pair sweep after step 3 ends (parallel; each trains its own generator + decoder)
bash experiments/quora.sh --leaked-samples 2000 
bash experiments/quora.sh --leaked-samples 4000 
```

**Adaptive per-cell ALGEN** (Quora; needs step 2's `train` and `match --cell-sweep`):

```bash
# phase 1: floors, flat arms, C=N/m
bash experiments/quora.sh ladder --attacks algen-cellmap 
# phase 2 (parallel)
bash experiments/quora.sh ladder --attacks algen-cellmap       --cell-sweep "20 200 2000" 
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 20   --cell-sweep "N/m 20" 
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 200  --cell-sweep 200       
bash experiments/quora.sh ladder --attacks algen-cellmap-norot --cells 2000 --cell-sweep 2000      
bash experiments/quora.sh report --attacks "algen algen-cellmap algen-cellmap-norot" --cell-sweep "20 200 2000"
```


**Monitor progress:**

```bash
bash experiments/quora.sh status --cell-sweep "20 200 2000"
```

Pass the same `--leaked-samples`, `--victim`, `--attacks` and cell flags as the run you're
checking.

## Rules for parallel runs

- **Never train the same thing twice at once.** Run shared stages (`embed`, `pretrain`,
  `defenses`, `match`, a new `--victim`) in one process first.


## Reproducibility

- **Training is deterministic** given the same inputs.
- **HNSW index builds are not deterministic:**, expect small differences in F1 score and leak_norm (2nd decimal places)
- **Knobs are matched to 0.90 ± 0.01**, on a coarse log grid, so different corpora often
  get identical knob values.