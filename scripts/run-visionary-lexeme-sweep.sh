#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

corpus="${1:-data/processed/visionary-wikibard-corpus.txt}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_dir="data/processed/visionary-lexeme-sweep-${run_id}"
mkdir -p "$out_dir"

if [[ ! -f "$corpus" ]]; then
  echo "missing corpus: $corpus" >&2
  exit 1
fi

echo "run_id=$run_id"
echo "corpus=$corpus"
echo "out_dir=$out_dir"
date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"

run_size() {
  local label="$1"
  local vocab="$2"
  local windows="$3"
  local stride="$4"
  local embed_dim="$5"
  local freq_cap="$6"

  local prefix="${out_dir}/${label}-v${vocab}-w${windows}"
  local tokens="${prefix}.tokens.u16"
  local vocab_tsv="${prefix}.vocab.tsv"
  local token_trace="${prefix}.tokens.trace.jsonl"
  local embedding="${prefix}.nsrllex"
  local embedding_trace="${prefix}.embedding.trace.jsonl"
  local softmax="${prefix}.nsrllm"
  local softmax_trace="${prefix}.softmax.trace.jsonl"
  local sample_trace="${prefix}.sample.trace.jsonl"
  local sample_text="${prefix}.sample.txt"

  echo "== ${label}: vocab=${vocab} windows=${windows} stride=${stride} embed_dim=${embed_dim} cap=${freq_cap}"
  date -u +"${label}_tokenize_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo run --release -q -p nsrl-corpus -- lexeme-tokenize \
    --corpus "$corpus" \
    --tokens-out "$tokens" \
    --vocab-out "$vocab_tsv" \
    --trace "$token_trace" \
    --seq-len 32 \
    --stride 1 \
    --max-vocab "$vocab" \
    --lexeme-vocab-profile balanced \
    --lexeme-frequency-cap "$freq_cap" \
    --preview-tokens 32

  date -u +"${label}_embedding_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-embedding \
    --tokens "$tokens" \
    --vocab "$vocab_tsv" \
    --model-out "$embedding" \
    --trace "$embedding_trace" \
    --vocab-size "$vocab" \
    --embedding-dim "$embed_dim" \
    --context-radius 2 \
    --stride "$stride" \
    --max-windows "$windows" \
    --epochs 1 \
    --lr-shift 8 \
    --concept-frequency-cap "$freq_cap" \
    --frequency-weight-min-q15 4096 \
    --quality-weight-profile cruft-aware

  date -u +"${label}_softmax_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-softmax \
    --tokens "$tokens" \
    --vocab "$vocab_tsv" \
    --model "$embedding" \
    --model-out "$softmax" \
    --trace "$softmax_trace" \
    --seq-len 8 \
    --stride "$stride" \
    --max-windows "$windows" \
    --epochs 1 \
    --lr-shift 20 \
    --lr-shift-decay-windows "$((windows / 2))" \
    --lr-shift-decay-step 1 \
    --max-lr-shift 22 \
    --max-weight-delta 1 \
    --target-frequency-cap "$freq_cap" \
    --frequency-weight-min-q15 4096 \
    --quality-weight-profile cruft-aware

  date -u +"${label}_sample_started_at=%Y-%m-%dT%H:%M:%SZ"
  cargo run --release -q -p nsrl-train -- \
    --mode lexeme-generate \
    --model "$softmax" \
    --vocab "$vocab_tsv" \
    --tokens "$tokens" \
    --prompt "to be or not to be" \
    --max-new-tokens 220 \
    --decode sample \
    --sample-seed 7 \
    --top-k 8 \
    --repeat-window 32 \
    --repeat-penalty-shift 1 \
    --max-repeat-run 3 \
    --corpus-prior \
    --strict-adjacency \
    --text-out "$sample_text" \
    --trace "$sample_trace"

  date -u +"${label}_finished_at=%Y-%m-%dT%H:%M:%SZ"
  echo "${label}_sample=$sample_text"
}

run_size small 1024 4096 25000 16 1024
run_size medium 2048 8192 12500 16 2048
run_size large 4096 16384 6250 16 4096

date -u +"finished_at=%Y-%m-%dT%H:%M:%SZ"
