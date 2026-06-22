#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-crowley-bard-aphorism-v2}"
out_dir="${OUT_DIR:-data/processed/$run_id}"
vocab_size="${VOCAB_SIZE:-4096}"
embed_dim="${EMBED_DIM:-16}"
freq_cap="${FREQ_CAP:-4096}"
embed_windows="${EMBED_WINDOWS:-131072}"
softmax_windows="${SOFTMAX_WINDOWS:-131072}"
softmax_seq_len="${SOFTMAX_SEQ_LEN:-16}"
lexeme_context_features="${LEXEME_CONTEXT_FEATURES:-ordered}"
softmax_lr_shift="${SOFTMAX_LR_SHIFT:-21}"
softmax_max_lr_shift="${SOFTMAX_MAX_LR_SHIFT:-23}"
crowley_bytes="${CROWLEY_BYTES:-420000}"
blake_bytes="${BLAKE_BYTES:-160000}"
shakespeare_bytes="${SHAKESPEARE_BYTES:-100000}"
raw_count="${RAW_TWEET_COUNT:-96}"
keep_count="${KEEP_TWEET_COUNT:-24}"

mkdir -p "$out_dir"

corpus="$out_dir/corpus.txt"
tokens="$out_dir/v${vocab_size}.tokens.u16"
vocab_tsv="$out_dir/v${vocab_size}.vocab.tsv"
token_trace="$out_dir/v${vocab_size}.tokens.trace.jsonl"
embedding="$out_dir/v${vocab_size}.nsrllex"
embedding_trace="$out_dir/v${vocab_size}.embedding.trace.jsonl"
softmax="$out_dir/v${vocab_size}.nsrllm"
softmax_trace="$out_dir/v${vocab_size}.softmax.trace.jsonl"
manifest="$out_dir/manifest.json"

echo "run_id=$run_id"
echo "out_dir=$out_dir"
echo "crowley_bytes=$crowley_bytes"
echo "blake_bytes=$blake_bytes"
echo "shakespeare_bytes=$shakespeare_bytes"
echo "softmax_seq_len=$softmax_seq_len"
echo "lexeme_context_features=$lexeme_context_features"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

node scripts/build-crowley-bard-aphorism-corpus.mjs \
  --out-dir "$out_dir" \
  --crowley-bytes "$crowley_bytes" \
  --blake-bytes "$blake_bytes" \
  --shakespeare-bytes "$shakespeare_bytes"

date -u +"tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-corpus -- lexeme-tokenize \
  --corpus "$corpus" \
  --tokens-out "$tokens" \
  --vocab-out "$vocab_tsv" \
  --trace "$token_trace" \
  --seq-len 32 \
  --stride 1 \
  --max-vocab "$vocab_size" \
  --lexeme-vocab-profile balanced \
  --lexeme-frequency-cap "$freq_cap" \
  --preview-tokens 32

date -u +"embedding_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens "$tokens" \
  --vocab "$vocab_tsv" \
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

date -u +"softmax_started_at=%Y-%m-%dT%H:%M:%SZ"
cargo run --release -q -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens "$tokens" \
  --vocab "$vocab_tsv" \
  --model "$embedding" \
  --model-out "$softmax" \
  --trace "$softmax_trace" \
  --seq-len "$softmax_seq_len" \
  --lexeme-context-features "$lexeme_context_features" \
  --stride 1 \
  --max-windows "$softmax_windows" \
  --epochs 1 \
  --lr-shift "$softmax_lr_shift" \
  --lr-shift-decay-windows "$((softmax_windows / 2))" \
  --lr-shift-decay-step 1 \
  --max-lr-shift "$softmax_max_lr_shift" \
  --max-weight-delta 1 \
  --target-frequency-cap "$freq_cap" \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware

node -e '
const fs = require("fs");
const outDir = process.argv[1];
const runId = process.argv[2];
const vocabSize = Number(process.argv[3]);
const embedDim = Number(process.argv[4]);
const freqCap = Number(process.argv[5]);
const embedWindows = Number(process.argv[6]);
const softmaxWindows = Number(process.argv[7]);
const softmaxSeqLen = Number(process.argv[8]);
const contextFeatures = process.argv[9];
const manifestPath = `${outDir}/manifest.json`;
const aphorismManifest = JSON.parse(fs.readFileSync(`${outDir}/aphorism-manifest.json`, "utf8"));
fs.writeFileSync(manifestPath, JSON.stringify({
  schema: "nsrl.crowley_bard_aphorism_lexeme_run.v1",
  run_id: runId,
  out_dir: outDir,
  aphorism_corpus: aphorismManifest,
  vocab_size: vocabSize,
  embedding_dim: embedDim,
  frequency_cap: freqCap,
  embedding_windows: embedWindows,
  softmax_windows: softmaxWindows,
  softmax_seq_len: softmaxSeqLen,
  lexeme_context_features: contextFeatures,
  corpus: `${outDir}/corpus.txt`,
  tokens: `${outDir}/v${vocabSize}.tokens.u16`,
  vocab: `${outDir}/v${vocabSize}.vocab.tsv`,
  embedding: `${outDir}/v${vocabSize}.nsrllex`,
  model: `${outDir}/v${vocabSize}.nsrllm`
}, null, 2) + "\n");
' "$out_dir" "$run_id" "$vocab_size" "$embed_dim" "$freq_cap" "$embed_windows" "$softmax_windows" "$softmax_seq_len" "$lexeme_context_features"

date -u +"generation_started_at=%Y-%m-%dT%H:%M:%SZ"
node scripts/generate-crowley-bard-tweets.mjs \
  --run-dir "$out_dir" \
  --out-dir "$out_dir/tweets-strict" \
  --raw-count "$raw_count" \
  --keep-count "$keep_count" \
  --min-chars 60 \
  --max-chars 240

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
echo "corpus=$corpus"
echo "tokens=$tokens"
echo "vocab=$vocab_tsv"
echo "model=$softmax"
echo "manifest=$manifest"
echo "tweets=$out_dir/tweets-strict/tweets.md"
