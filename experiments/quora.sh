#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ablations -- edit here or override on the command line
LEAKED=1000        # leaked (text, vector) pairs: ALGEN/STEER align_samples and TEIA's D_L
CELLS=N/m          # cell count of the partitioned index; N/m = whatever PART_M=1000 gives
CELL_SWEEP=N/m     # more cell counts to run the partitioned index at, e.g. "20 200 2000"
VICTIM=gtr-base    # victim embedding model
GPU=0              # CUDA device
CORES=             # faiss/OpenMP threads; empty = all. SET IT for parallel runs
MIN_HEADROOM=                       # empty = the spec's 0.10 validity gate
ATTACK_LIST="algen steer teia zero2text"   # ladder/report only; ALGEN+STEER share one
                                           # results tree, so keep them in ONE process
                   # adaptive: algen-cellmap = one ridge map PER CELL (partition-aware ALGEN),
                   # algen-cellmap-norot = same, cells without keys (control); both share a
                   # tree, so keep them in ONE process too. Partitioned layouts only.
CELL_MIN_PAIRS=1   # algen-cellmap: fewest leaked pairs a cell needs for its own map

STAGES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --leaked-samples) LEAKED=$2; shift ;;
    --cells)          CELLS=$2; shift ;;
    --cell-sweep)     CELL_SWEEP=$2; shift ;;
    --victim)         VICTIM=$2; shift ;;
    --gpu)            GPU=$2; shift ;;
    --cores)          CORES=$2; shift ;;
    --attacks)        ATTACK_LIST=$2; shift ;;
    --min-headroom)   MIN_HEADROOM=$2; shift ;;
    --cell-min-pairs) CELL_MIN_PAIRS=$2; shift ;;
    embed|pretrain|train|defenses|teia-train|match|ladder|report|status) STAGES="$STAGES $1" ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
stage() { [ -z "$STAGES" ] || [[ " $STAGES " == *" $1 "* ]]; }

# experiment settings
DATASET=quora                       # corpus name used by main.py and the configs
ETAG=quora                          # corpus name used in the target .npz file names
ALGEN_CFG=configs/algen_quora.yaml
STEER_CFG=configs/steer_quora.yaml
TEIA_CFG=configs/teia_quora.yaml
TARGET=0.90                         # ANN recall@10 every defense knob is solved to
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false
[ -n "$CORES" ] && export OMP_NUM_THREADS=$CORES

# where everything goes
ROOT=attacker/outputs/verify                   # ALGEN stage 1 (victim/corpus independent)
SHARED=$ROOT/$VICTIM                           # defenses + TEIA decoders, shared across LEAKED
OUT=$SHARED/pairs$LEAKED                       # ALGEN generator for this LEAKED
LAD=metrics/outputs/verify/$VICTIM             # ladder arms + solved knobs
mkdir -p "$OUT/google_flan-t5-base" "$LAD/logs"
[[ " $STAGES " == *" status "* ]] || exec > >(tee -a "$LAD/logs/$DATASET-$(date +%Y%m%d-%H%M%S)-$$.log") 2>&1
echo "[$(date '+%F %T')] $DATASET  stages=${STAGES:- all}  attacks=\"$ATTACK_LIST\" min_headroom=${MIN_HEADROOM:-0.10}  LEAKED=$LEAKED CELLS=$CELLS CELL_SWEEP=$CELL_SWEEP VICTIM=$VICTIM GPU=$GPU CORES=${CORES:-all}"

SCOPE=(--victim-model "$VICTIM" --embeddings "${VICTIM}__${ETAG}__*.npz" --output-dir "$OUT"
       --align-samples "$LEAKED" --finetune-align-reserve "$LEAKED")
ANN=(--match-metric ann --ann-index HNSW32,Flat --ann-budget 64 --ann-docs 0
     --match-repeats 1 --match-verify-repeats 3)
MATCH=(--match-out "$LAD/matched.json")          # solver + knob lookup only
TEIA=(--decoder-name gpt2 --surrogate-model gte-base --external-dataset beir/trec-covid
      --leaked-samples "$LEAKED" --external-samples 20000 --teia-batch-size 16
      --teia-epochs 24 --teia-max-length 32)
Z2T=(--attack zero2text --defense-scope both --z2t-llm Qwen/Qwen2.5-0.5B-Instruct
     --z2t-rounds 3 --z2t-seed-pool 2048 --z2t-keep-best 128 --z2t-pool-cap 3072
     --z2t-local-pairs 0)
MH=(); [ -n "$MIN_HEADROOM" ] && MH=(--min-headroom "$MIN_HEADROOM")

ATTACKS=($ATTACK_LIST)
for a in "${ATTACKS[@]}"; do
  case $a in algen|steer|teia|zero2text|algen-cellmap|algen-cellmap-norot) ;; *) echo "unknown attack: $a" >&2; exit 2 ;; esac
done
has() { [[ " $ATTACK_LIST " == *" $1 "* ]]; }
# per-cell ALGEN writes arms under the global-map names, so it gets its own tree
CMLAD=$LAD/algen_cellmaps; [ "$CELL_MIN_PAIRS" = 1 ] || CMLAD=${CMLAD}_min$CELL_MIN_PAIRS
CELLMAP=(--algen-cell-maps --algen-cell-min-pairs "$CELL_MIN_PAIRS")
adaptive() { [[ $1 == algen-cellmap-norot ]]; } # control only; cellmap runs flat       
FIXED=(eguard idct)                                   # no knob
TUNABLE=(gaussian lapmech sparse cmag vec2text remote_rag)
declare -A FLAG=([gaussian]=--noise-level [lapmech]=--epsilon [sparse]=--sparse-epsilon
                 [cmag]=--cmag-epsilon [vec2text]=--vec2text-noise-level
                 [remote_rag]=--remote-rag-radius)
LAYOUTS="flat $(printf '%s\n' $CELLS $CELL_SWEEP | awk '!seen[$0]++' | tr '\n' ' ')"

# layout <flat|C> -> index flags (C = cell count, N/m = default occupancy m=1000)
layout() {
  [ "$1" = flat ] && return
  echo --partition --kr-m 1000 --kr-alpha 1.3 --kr-nprobe 128
  [ "$1" = N/m ] || echo --kr-cells "$1"
}

# gen_dir <pretrain|finetune> [flags] -> ALGEN generator directory
gen_dir() {
  python3 - "$ALGEN_CFG" "$@" <<'PY'
import sys, train_algen as a
print(a.stage_dirs(a.resolve_args(["--config", sys.argv[1], *sys.argv[3:]]))[sys.argv[2]])
PY
}

teia_dir() {
  python3 - "$OUT" "$DATASET" "$LEAKED" <<'PY'
import sys
from attacker.teia.trainer import TeiaTrainer
out, ds, leaked = sys.argv[1:4]
print(TeiaTrainer.run_dir_for(decoder_name="gpt2", output_dir=out, dataset=ds,
      external_dataset="beir/trec-covid", surrogate_model="gte-base", max_length=32,
      leaked_samples=int(leaked), external_samples=20000, batch_size=16, num_epochs=24))
PY
}

# knob <defense> <layout> -> solved knob value, empty if not solved
knob() {
  python3 -m ANN knob --config "$ALGEN_CFG" "${SCOPE[@]}" "${ANN[@]}" "${MATCH[@]}" $(layout "$2") \
    --defense "$1" --target "$TARGET" 2>/dev/null | tail -1 || true
}

# arm <attack> <defense> <layout> -> one ladder arm
arm() {
  local a=$1 d=$2 l=$3 v="" args=()
  case $a in
    algen)     args=(--config "$ALGEN_CFG" --ladder-out "$LAD") ;;
    steer)     args=(--config "$STEER_CFG" --ladder-out "$LAD" --attack steer --defense-scope targets) ;;
    teia)      args=(--config "$TEIA_CFG" --ladder-out "$LAD" --attack teia --defense-scope targets
                     --checkpoint "$(teia_dir)") ;;
    zero2text) args=(--config "$ALGEN_CFG" --ladder-out "$LAD/zero2text" "${Z2T[@]}") ;;
    algen-cellmap)       args=(--config "$ALGEN_CFG" --ladder-out "$CMLAD" "${CELLMAP[@]}") ;;
    algen-cellmap-norot) args=(--config "$ALGEN_CFG" --ladder-out "$CMLAD" "${CELLMAP[@]}" --kr-no-rotate) ;;
  esac
  if [ -n "${FLAG[$d]:-}" ]; then
    v=$(knob "$d" "$l")
    [ -n "$v" ] || { echo "!! no solved knob"; echo "[skipped] LEAKED=$LEAKED $a / $d / $l"; return; }
    args+=("${FLAG[$d]}" "$v")
  fi
  echo "#### $a / $d / $l ${v:+(${FLAG[$d]} $v)} ####"
  python3 -u -m metrics.ladder "${args[@]}" "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") "${MH[@]}" \
    --defense "$d" --show 0 && echo "[done] LEAKED=$LEAKED $a / $d / $l" \
                            || echo "[failed] LEAKED=$LEAKED $a / $d / $l"
}

###############################################################

if [[ " $STAGES " == *" status "* ]]; then
  ok() { compgen -G "$2" >/dev/null && echo "  ok       $1" || echo "  MISSING  $1"; }
  echo "== $DATASET  LEAKED=$LEAKED  VICTIM=$VICTIM  layouts: $LAYOUTS =="
  echo "-- training --"
  for s in 1 2 3; do ok "targets s$s" "data/embeddings/${VICTIM}__${ETAG}__s$s.npz"; done
  ok "ALGEN stage 1" "$(gen_dir pretrain --output-dir $ROOT)/training_args_and_best_models.json"
  ok "ALGEN generator" "$(gen_dir finetune "${SCOPE[@]}")/training_args_and_best_models.json"
  for d in eguard sparse cmag vec2text remote_rag; do
    ok "defense $d" "$SHARED/$d/${d}_${VICTIM}_${DATASET}_*.pt"
  done
  ok "TEIA decoder" "$(teia_dir)/training_args_and_best_models.json"

  echo "-- knobs (ANN recall@10 = $TARGET; empty = not solved) --"
  printf '  %-12s' defense; for l in $LAYOUTS; do printf ' %14s' "$l"; done; echo
  for d in "${TUNABLE[@]}"; do
    printf '  %-12s' "$d"; for l in $LAYOUTS; do printf ' %14s' "$(knob $d "$l")"; done; echo
  done

  echo "-- ladder arms (done / failed / skipped of total, from the logs) --"
  marks=$(cat "$LAD"/logs/$DATASET-*.log 2>/dev/null | tr '\r' '\n' | grep -E '^\[(done|failed|skipped)\] ' || true)
  count() {  # count <done|failed|skipped> <attack> <layout> -> distinct arms
    echo "$marks" | awk -v m="[$1]" -v k="LEAKED=$LEAKED" -v a="$2" -v l="$3" \
      '$1==m && $2==k && $3==a && $NF==l {print $5}' | sort -u | wc -l
  }
  for a in "${ATTACKS[@]}"; do
    for l in $LAYOUTS; do
      [ "$l" = flat ] && adaptive "$a" && continue
      n=$(( ${#FIXED[@]} + ${#TUNABLE[@]} )); [ "$l" = flat ] || n=$((n + 1))
      printf '  %-10s %-6s %2s / %s / %s of %s\n' "$a" "$l" \
        "$(count done "$a" "$l")" "$(count failed "$a" "$l")" "$(count skipped "$a" "$l")" "$n"
    done
  done

  echo "-- now --"
  pgrep -af "$(basename "$0")" | grep -v " status" | sed 's/^/  running: /' || echo "  not running"
  last=$(ls -t "$LAD"/logs/$DATASET-*.log 2>/dev/null | head -1)
  if [ -n "$last" ]; then
    echo "  newest log: $last (last written $(( ($(date +%s) - $(date -r "$last" +%s)) / 60 )) min ago)"
    tr '\r' '\n' < "$last" | grep -E '^#### ' | tail -1 | sed 's/^/  last arm started: /'
    echo "  tracebacks in it: $(tr '\r' '\n' < "$last" | grep -c 'Traceback (most recent call last)')"
  fi
  exit 0
fi

# ALGEN stage 1 is shared by every corpus and victim; link it (and the attacker-
# independent defenses / TEIA decoders) into this LEAKED's scope
PT=$(gen_dir pretrain --output-dir $ROOT)
# create links only when missing: parallel runs share them, and ln -sfn replaces
[ -L "$OUT/google_flan-t5-base/$(basename "$PT")" ] || ln -sfn "$PWD/$PT" "$OUT/google_flan-t5-base/$(basename "$PT")"
for d in eguard sparse cmag vec2text remote_rag teia; do
  mkdir -p "$SHARED/$d"; [ -L "$OUT/$d" ] || ln -sfn "../$d" "$OUT/$d"
done

if stage embed; then      # victim target sets (3 x 128)
  for seed in 1 2 3; do
    f=data/embeddings/${VICTIM}__${ETAG}__s${seed}.npz
    [ -f "$f" ] || python3 -u main.py --dataset $DATASET --model "$VICTIM" --n 128 --seed $seed --out "$f"
  done
fi

if stage pretrain; then   # ALGEN stage 1 (trec-covid)
  python3 -u train_algen.py --config $ALGEN_CFG --stages pretrain --output-dir $ROOT
fi

if stage train; then      # ALGEN stage 2 (the generator ALGEN, STEER and Zero2Text decode through)
  python3 -u train_algen.py --config $ALGEN_CFG --stages finetune "${SCOPE[@]}"
fi

if stage defenses; then   # learned defenses
  for d in eguard sparse cmag vec2text remote_rag; do
    python3 -u train_algen.py --config $ALGEN_CFG --stages $d "${SCOPE[@]}"
  done
fi

if stage teia-train; then # TEIA decoder
  python3 -u train_teia.py --config $TEIA_CFG --stages train "${SCOPE[@]}" "${TEIA[@]}" \
    --defense-scope targets
fi

if stage match; then      # solve each tunable defense's knob to ANN recall@10 = TARGET, per layout
  for l in $LAYOUTS; do
    if [ "$l" != flat ] && [ "$l" != N/m ] && \
       ! grep -q "\"$DATASET|C$l|" metrics/outputs/cells_to_m.json 2>/dev/null; then
      python3 -u -m defense.keyed_rotation calibrate --corpus $DATASET --cells "$l" --full-fit \
        --vectors "$(ls -t ANN/cache/${VICTIM}__${DATASET}__n*.npy | head -1)"
    fi
    for d in "${TUNABLE[@]}"; do
      [ -n "$(knob $d "$l")" ] && continue
      python3 -u -m metrics.utility_match --config $ALGEN_CFG --defense $d "${SCOPE[@]}" "${ANN[@]}" \
        "${MATCH[@]}" $(layout "$l") --target-recall $TARGET --match-tolerance 0.01 \
        || echo "!! $d / $l: knob did not converge -- its ladder arms will be skipped"
    done
  done
fi

if stage ladder; then     # every attack x every defense, one by one
  for a in "${ATTACKS[@]}"; do
    for l in $LAYOUTS; do
      [ "$l" = flat ] && adaptive "$a" && continue
      defenses=("${FIXED[@]}" "${TUNABLE[@]}")
      [ "$l" = flat ] || defenses=(keyed_rotation "${defenses[@]}")
      for d in "${defenses[@]}"; do arm "$a" "$d" "$l"; done
    done
  done
fi

if stage report; then     # WITH-vs-WITHOUT partition tables (ALGEN + STEER share one table)
  for l in $LAYOUTS; do
    [ "$l" = flat ] && continue
    echo "======== report: cells=$l ========"
    if has algen || has steer; then
      python3 -u -m metrics.summarize --config $ALGEN_CFG --ladder-out "$LAD" --checkpoint "$(gen_dir finetune "${SCOPE[@]}")" \
        "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") "${MH[@]}" --compare-partition
    fi
    if has teia; then
      python3 -u -m metrics.summarize --config $TEIA_CFG --ladder-out "$LAD" --checkpoint "$(teia_dir)" \
        "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") "${MH[@]}" --compare-partition
    fi
    if has zero2text; then
      python3 -u -m metrics.summarize --config $ALGEN_CFG --ladder-out "$LAD/zero2text" --checkpoint "$(gen_dir finetune "${SCOPE[@]}")" \
        "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") "${MH[@]}" --compare-partition
    fi
    if has algen-cellmap; then
      echo "---- per-cell ALGEN (adaptive), cells=$l ----"
      python3 -u -m metrics.summarize --config $ALGEN_CFG --ladder-out "$CMLAD" --checkpoint "$(gen_dir finetune "${SCOPE[@]}")" \
        "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") "${CELLMAP[@]}" "${MH[@]}" --compare-partition
    fi
    if has algen-cellmap-norot; then
      echo "---- per-cell ALGEN, NO rotation (control), cells=$l ----"
      python3 -u -m metrics.summarize --config $ALGEN_CFG --ladder-out "$CMLAD" --checkpoint "$(gen_dir finetune "${SCOPE[@]}")" \
        "${SCOPE[@]}" "${ANN[@]}" $(layout "$l") --kr-no-rotate "${CELLMAP[@]}" "${MH[@]}" --compare-partition
    fi
  done
fi
