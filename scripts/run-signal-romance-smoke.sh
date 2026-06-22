#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Build, tokenize, train, and evaluate the local Signal romance voice smoke model.

Common knobs:
  OUT_DIR=data/processed/signal-romance-smoke
  SIGNAL_SFT=/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-sft.jsonl
  SIGNAL_VOICE=/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-voice.txt
  SIGNAL_REPEAT=6
  STYLE_BYTES=8000
  SEQ_LEN=128
  MAX_WINDOWS=32768
  EPOCHS=1
  BATCH_WINDOWS=2
  RESUME_FROM=data/processed/signal-romance-smoke/signal-romance.nsrlmt
  MODEL_OUT=data/processed/signal-romance-smoke/signal-romance.nsrlmt
  EVAL_COUNT=20
  BUILD=1
  TRAIN=1
  EVAL=1
  STRICT_EVAL=0
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${OUT_DIR:-data/processed/signal-romance-smoke}"
signal_sft="${SIGNAL_SFT:-/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-sft.jsonl}"
signal_voice="${SIGNAL_VOICE:-/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-voice.txt}"
signal_repeat="${SIGNAL_REPEAT:-6}"
style_bytes="${STYLE_BYTES:-8000}"
style_chunk_bytes="${STYLE_CHUNK_BYTES:-256}"
style_every_frames="${STYLE_EVERY_FRAMES:-16}"
eval_count="${EVAL_COUNT:-20}"
seq_len="${SEQ_LEN:-128}"
stride="${STRIDE:-1}"
max_windows="${MAX_WINDOWS:-32768}"
epochs="${EPOCHS:-1}"
batch_windows="${BATCH_WINDOWS:-2}"
window_offset="${WINDOW_OFFSET:-0}"
progress_interval_batches="${PROGRESS_INTERVAL_BATCHES:-128}"
resume_from="${RESUME_FROM:-}"
build="${BUILD:-1}"
train="${TRAIN:-1}"
eval="${EVAL:-1}"
strict_eval="${STRICT_EVAL:-0}"

corpus="$out_dir/corpus.txt"
tokens="$out_dir/corpus.tokens.u8"
token_trace="$out_dir/corpus.tokens.trace.jsonl"
model="${MODEL_OUT:-$out_dir/signal-romance.nsrlmt}"
train_trace="$out_dir/signal-romance.train.trace.jsonl"
progress="$out_dir/signal-romance.progress.jsonl"
command_out="$out_dir/command.txt"

mkdir -p "$out_dir"

echo "out_dir=$out_dir"
echo "signal_sft=$signal_sft"
echo "signal_voice=$signal_voice"
echo "seq_len=$seq_len"
echo "max_windows=$max_windows"
if [[ -n "$resume_from" ]]; then
  echo "resume_from=$resume_from"
fi
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

node scripts/build-signal-romance-corpus.mjs \
  --out-dir "$out_dir" \
  --signal-sft "$signal_sft" \
  --signal-voice "$signal_voice" \
  --signal-repeat "$signal_repeat" \
  --eval-count "$eval_count" \
  --style-bytes "$style_bytes" \
  --style-chunk-bytes "$style_chunk_bytes" \
  --style-every-frames "$style_every_frames"

if [[ "$build" != "0" ]]; then
  date -u +"build_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo build --release -q -p nsrl-corpus -p nsrl-train
else
  echo "build_skipped=1"
fi

corpus_bin="${NSRL_CORPUS_BIN:-target/release/nsrl-corpus}"
train_bin="${NSRL_TRAIN_BIN:-target/release/nsrl-train}"
for required_bin in "$corpus_bin" "$train_bin"; do
  if [[ ! -x "$required_bin" ]]; then
    echo "missing executable: $required_bin" >&2
    echo "Run with BUILD=1 after the Rust workspace compiles, or set NSRL_CORPUS_BIN/NSRL_TRAIN_BIN." >&2
    exit 2
  fi
done

date -u +"tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
"$corpus_bin" tokenize \
  --corpus "$corpus" \
  --tokens-out "$tokens" \
  --trace "$token_trace" \
  --seq-len "$seq_len" \
  --stride "$stride" \
  --text-profile identity \
  --preview-tokens 32

if [[ "$train" != "0" ]]; then
  train_cmd=(
    "$train_bin"
    --mode mini-transformer-mlp
    --tokens "$tokens"
    --seq-len "$seq_len"
    --stride "$stride"
    --window-offset "$window_offset"
    --batch-windows "$batch_windows"
    --max-windows "$max_windows"
    --epochs "$epochs"
    --mini-transformer-attention linear
    --mini-transformer-position nope
    --adaptive-rule-shifts
    --adaptive-rule-interval-batches 128
    --mini-transformer-trace-detail summary
    --model-out "$model"
    --trace "$train_trace"
    --progress-out "$progress"
    --progress-interval-batches "$progress_interval_batches"
  )
  if [[ -n "$resume_from" ]]; then
    if [[ ! -f "$resume_from" ]]; then
      echo "RESUME_FROM does not exist: $resume_from" >&2
      exit 2
    fi
    train_cmd+=(--resume-from "$resume_from")
  fi
  printf '%q ' "${train_cmd[@]}" > "$command_out"
  printf '\n' >> "$command_out"
  date -u +"train_started_at=%Y-%m-%dT%H:%M:%SZ"
  "${train_cmd[@]}"
fi

if [[ "$eval" != "0" ]]; then
  eval_args=(
    scripts/eval-signal-romance.mjs
    --run-dir "$out_dir"
    --model "$model"
    --tokens "$tokens"
    --count "$eval_count"
  )
  if [[ "$strict_eval" != "0" ]]; then
    eval_args+=(--fail-on-threshold)
  fi
  date -u +"eval_started_at=%Y-%m-%dT%H:%M:%SZ"
  NSRL_TRAIN_BIN="$train_bin" node "${eval_args[@]}"
fi

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "model=$model"
echo "tokens=$tokens"
echo "eval_report=$out_dir/eval/eval-report.json"
