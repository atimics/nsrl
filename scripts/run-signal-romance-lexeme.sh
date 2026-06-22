#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Build, train, and evaluate the Signal romance voice with NSRL lexeme tokens.

This is the practical training path for the current tiny voice model. It trains
the ranker-to-voice copy reflex first; radio/sci-fi style lanes can be added
after grounding is reliable.

Common knobs:
  OUT_DIR=data/processed/signal-romance-deploy-lexeme
  SIGNAL_REPEAT=40
  STYLE_BYTES=0
  FETCH_SOURCES=0
  EVAL_COUNT=0
  SOFTMAX_SEQ_LEN=16
  SOFTMAX_WINDOWS=150000
  SOFTMAX_EPOCHS=3
  BUILD_CORPUS=1
  TRAIN=1
  EVAL=1
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${OUT_DIR:-data/processed/signal-romance-deploy-lexeme}"
signal_sft="${SIGNAL_SFT:-/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-sft.jsonl}"
signal_voice="${SIGNAL_VOICE:-/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-voice.txt}"
signal_repeat="${SIGNAL_REPEAT:-40}"
style_bytes="${STYLE_BYTES:-0}"
style_chunk_bytes="${STYLE_CHUNK_BYTES:-256}"
style_every_frames="${STYLE_EVERY_FRAMES:-16}"
eval_count="${EVAL_COUNT:-0}"
fetch_sources="${FETCH_SOURCES:-0}"
build_corpus="${BUILD_CORPUS:-1}"
train="${TRAIN:-1}"
eval="${EVAL:-1}"

lexeme_seq_len="${LEXEME_SEQ_LEN:-16}"
vocab_max="${VOCAB_MAX:-1024}"
freq_cap="${FREQ_CAP:-4096}"
embed_dim="${EMBED_DIM:-16}"
embed_windows="${EMBED_WINDOWS:-65536}"
softmax_seq_len="${SOFTMAX_SEQ_LEN:-16}"
softmax_windows="${SOFTMAX_WINDOWS:-150000}"
softmax_epochs="${SOFTMAX_EPOCHS:-3}"
softmax_lr_shift="${SOFTMAX_LR_SHIFT:-18}"
softmax_max_lr_shift="${SOFTMAX_MAX_LR_SHIFT:-23}"

mkdir -p "$out_dir"

corpus="$out_dir/corpus.txt"
tokens="$out_dir/v${vocab_max}.tokens.u16"
vocab="$out_dir/v${vocab_max}.vocab.tsv"
token_trace="$out_dir/v${vocab_max}.tokens.trace.jsonl"

echo "out_dir=$out_dir"
echo "signal_repeat=$signal_repeat"
echo "style_bytes=$style_bytes"
echo "softmax_seq_len=$softmax_seq_len"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

if [[ "$build_corpus" != "0" ]]; then
  if [[ "$fetch_sources" != "0" ]]; then
    node scripts/fetch-signal-romance-sources.mjs \
      --out-dir "${STYLE_SOURCE_DIR:-data/processed/signal-romance-sources}"
  fi

  build_args=(
    scripts/build-signal-romance-corpus.mjs
    --out-dir "$out_dir"
    --signal-sft "$signal_sft"
    --signal-voice "$signal_voice"
    --signal-repeat "$signal_repeat"
    --eval-count "$eval_count"
    --style-bytes "$style_bytes"
    --style-chunk-bytes "$style_chunk_bytes"
    --style-every-frames "$style_every_frames"
  )
  if [[ -n "${STYLE_SOURCE_DIR:-}" ]]; then
    build_args+=(--style-source-dir "$STYLE_SOURCE_DIR")
  fi

  node scripts/build-signal-romance-corpus.mjs \
    "${build_args[@]:1}"

  target/release/nsrl-corpus lexeme-tokenize \
    --corpus "$corpus" \
    --tokens-out "$tokens" \
    --vocab-out "$vocab" \
    --trace "$token_trace" \
    --seq-len "$lexeme_seq_len" \
    --stride 1 \
    --max-vocab "$vocab_max" \
    --lexeme-vocab-profile balanced \
    --lexeme-frequency-cap "$freq_cap" \
    --preview-tokens 32
fi

vocab_size="$(
  node -e 'const fs=require("fs"); const trace=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(String(trace.vocab.size));' "$token_trace"
)"
embedding="$out_dir/v${vocab_size}.nsrllex"
embedding_trace="$out_dir/v${vocab_size}.embedding.trace.jsonl"
model="$out_dir/v${vocab_size}-seq${softmax_seq_len}.nsrllm"
softmax_trace="$out_dir/v${vocab_size}-seq${softmax_seq_len}.softmax.trace.jsonl"

echo "vocab_size=$vocab_size"
echo "model=$model"

if [[ "$train" != "0" ]]; then
  target/release/nsrl-train \
    --mode lexeme-embedding \
    --tokens "$tokens" \
    --vocab "$vocab" \
    --model-out "$embedding" \
    --trace "$embedding_trace" \
    --vocab-size "$vocab_size" \
    --embedding-dim "$embed_dim" \
    --context-radius 2 \
    --stride 1 \
    --max-windows "$embed_windows" \
    --epochs 1 \
    --lr-shift 8 \
    --concept-frequency-cap "$freq_cap" \
    --frequency-weight-min-q15 4096 \
    --quality-weight-profile cruft-aware

  target/release/nsrl-train \
    --mode lexeme-softmax \
    --tokens "$tokens" \
    --vocab "$vocab" \
    --model "$embedding" \
    --model-out "$model" \
    --trace "$softmax_trace" \
    --seq-len "$softmax_seq_len" \
    --lexeme-context-features ordered \
    --stride 1 \
    --max-windows "$softmax_windows" \
    --epochs "$softmax_epochs" \
    --lr-shift "$softmax_lr_shift" \
    --lr-shift-decay-windows "$((softmax_windows / 2))" \
    --lr-shift-decay-step 1 \
    --max-lr-shift "$softmax_max_lr_shift" \
    --max-weight-delta 1 \
    --target-frequency-cap "$freq_cap" \
    --frequency-weight-min-q15 4096 \
    --quality-weight-profile cruft-aware
fi

if [[ "$eval" != "0" ]]; then
  prompts="$out_dir/eval-prompts.jsonl"
  if [[ ! -s "$prompts" ]]; then
    prompts="$out_dir/frames.jsonl"
  fi
  NSRL_TRAIN_BIN=target/release/nsrl-train node scripts/eval-signal-romance.mjs \
    --backend lexeme \
    --run-dir "$out_dir" \
    --model "$model" \
    --vocab "$vocab" \
    --prompts "$prompts" \
    --out-dir "$out_dir/lexeme-eval" \
    --count "${LEXEME_EVAL_COUNT:-20}" \
    --max-new-tokens "${LEXEME_MAX_NEW_TOKENS:-20}"
fi

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
