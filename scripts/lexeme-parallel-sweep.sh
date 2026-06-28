#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Lexeme softmax parallel map-reduce scaling sweep.
#
# Runs the same lexeme training job at multiple --lexeme-map-reduce-workers
# values and records wall time + final trace fields.  The bit-identical gate
# checks that final_weight_hash and weight_delta_l1 match across all runs.
#
# Usage:
#   scripts/lexeme-parallel-sweep.sh                    # local defaults
#   scripts/lexeme-parallel-sweep.sh --workers 1,2,4,8  # custom worker list
#
# Environment overrides:
#   NSRL_TOKENS   Path to .tokens.u16 file (default: data/processed/simplewiki-expository-v1/v4096.tokens.u16)
#   NSRL_MODEL    Path to .nsrllm or .nsrllex model file (default: web/assets/model.nsrllm)
#   NSRL_VOCAB    Path to .vocab.tsv file (default: data/processed/simplewiki-expository-v1/v4096.vocab.tsv)
#   NSRL_EPOCHS   Training epochs (default: 3)
#   NSRL_MAX_WINDOWS  Max training windows (default: 50000)
#   NSRL_BATCH_WINDOWS  Batch windows per update (default: 32)
#   NSRL_SEQ_LEN  Context sequence length (default: 16)
#   NSRL_BIN      nsrl-train binary (default: target/release/nsrl-train)
# ─────────────────────────────────────────────────────────────────────────────

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ── Defaults ─────────────────────────────────────────────────────────────────
tokens="${NSRL_TOKENS:-data/processed/simplewiki-expository-v1/v4096.tokens.u16}"
model="${NSRL_MODEL:-web/assets/model.nsrllm}"
vocab="${NSRL_VOCAB:-data/processed/simplewiki-expository-v1/v4096.vocab.tsv}"
epochs="${NSRL_EPOCHS:-3}"
max_windows="${NSRL_MAX_WINDOWS:-50000}"
batch_windows="${NSRL_BATCH_WINDOWS:-32}"
seq_len="${NSRL_SEQ_LEN:-16}"
bin="${NSRL_BIN:-target/release/nsrl-train}"

# ── Parse --workers ──────────────────────────────────────────────────────────
worker_list="1,2,4,8"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) worker_list="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--workers 1,2,4,8]"
      echo ""
      echo "Environment: NSRL_TOKENS NSRL_MODEL NSRL_VOCAB NSRL_EPOCHS NSRL_MAX_WINDOWS NSRL_BATCH_WINDOWS NSRL_SEQ_LEN NSRL_BIN"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── Validate inputs ──────────────────────────────────────────────────────────
for f in "$tokens" "$model" "$vocab"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f" >&2
    exit 1
  fi
done

# ── Build the binary if needed ───────────────────────────────────────────────
if [[ ! -x "$bin" ]] || [[ "$bin" -ot crates/nsrl-train/src/lib.rs ]]; then
  echo "Building release binary..."
  cargo build --release -p nsrl-train
fi

# ── Output directory ─────────────────────────────────────────────────────────
sweep_dir="data/processed/lexeme-parallel-sweep-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$sweep_dir"
results_tsv="${sweep_dir}/results.tsv"
summary_txt="${sweep_dir}/summary.txt"

echo -e "workers\twall_seconds\tupdates\tfinal_weight_hash\tweight_delta_l1\tfinal_accuracy_per_mille\tbit_identical" > "$results_tsv"

# ── Run the sweep ────────────────────────────────────────────────────────────
baseline_hash=""
baseline_delta=""
all_match="YES"

IFS=',' read -ra workers <<< "$worker_list"
for w in "${workers[@]}"; do
  echo ""
  echo "═══ workers=$w ═══"
  trace_file="${sweep_dir}/trace-w${w}.jsonl"
  start_time=$(date +%s%N)

  "$bin" \
    --mode lexeme-softmax \
    --tokens "$tokens" \
    --model "$model" \
    --vocab "$vocab" \
    --seq-len "$seq_len" \
    --lexeme-context-features ordered \
    --stride 1 \
    --max-windows "$max_windows" \
    --epochs "$epochs" \
    --batch-windows "$batch_windows" \
    --lexeme-map-reduce-workers "$w" \
    --lr-shift 18 \
    --lr-shift-decay-windows $((max_windows / 2)) \
    --lr-shift-decay-step 1 \
    --max-lr-shift 23 \
    --max-weight-delta 1 \
    --target-frequency-cap 4096 \
    --frequency-weight-min-q15 4096 \
    --quality-weight-profile cruft-aware \
    --train-lexeme-embeddings \
    --trace "$trace_file" \
    > "${sweep_dir}/stdout-w${w}.txt" 2> "${sweep_dir}/stderr-w${w}.txt"

  end_time=$(date +%s%N)
  wall_seconds=$(( (end_time - start_time) / 1000000000 ))

  # Parse trace JSON for key fields
  trace_line=$(cat "$trace_file")
  updates=$(echo "$trace_line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['training']['updates'])")
  final_hash=$(echo "$trace_line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['final_weight_hash'])")
  delta_l1=$(echo "$trace_line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']['weight_delta_l1'])")
  accuracy=$(echo "$trace_line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['metrics']['final_accuracy_per_mille'])")

  # Bit-identical check against first run (workers=1 baseline)
  if [[ -z "$baseline_hash" ]]; then
    baseline_hash="$final_hash"
    baseline_delta="$delta_l1"
    match="baseline"
  elif [[ "$final_hash" == "$baseline_hash" && "$delta_l1" == "$baseline_delta" ]]; then
    match="YES"
  else
    match="NO"
    all_match="NO"
  fi

  echo "  wall=${wall_seconds}s  updates=$updates  hash=$final_hash  delta_l1=$delta_l1  accuracy=${accuracy}per_mille  identical=$match"

  echo -e "${w}\t${wall_seconds}\t${updates}\t${final_hash}\t${delta_l1}\t${accuracy}\t${match}" >> "$results_tsv"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══ Summary ═══"
echo "Results: $results_tsv"
cat "$results_tsv"
echo ""
if [[ "$all_match" == "YES" ]]; then
  echo "✅ All runs bit-identical (final_weight_hash + weight_delta_l1 match)"
else
  echo "❌ Bit-identical check FAILED — see results.tsv"
fi

# Speedup table relative to workers=1
echo ""
echo "Speedup vs serial (workers=1):"
python3 - "$results_tsv" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
baseline = None
for r in rows:
    if r["workers"] == "1":
        baseline = float(r["wall_seconds"])
        break
if baseline is None or baseline == 0:
    print("  (no workers=1 baseline found)")
else:
    print(f"  {'workers':>8}  {'wall_s':>8}  {'speedup':>8}  {'efficiency':>10}")
    for r in rows:
        w = int(r["workers"])
        t = float(r["wall_seconds"])
        speedup = baseline / t if t > 0 else 0
        eff = speedup / w * 100 if w > 0 else 0
        print(f"  {w:>8}  {t:>8.1f}  {speedup:>8.2f}x  {eff:>9.1f}%")
PY

echo "$all_match" > "${sweep_dir}/bit_identical.txt"
echo ""
echo "Sweep complete. Output in: $sweep_dir"
