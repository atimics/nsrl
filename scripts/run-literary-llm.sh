#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_LITERARY_OUT_DIR:-data/local-runs/crowley-shakespeare-blake}"
seq_len="${NSRL_LITERARY_SEQ_LEN:-32}"
max_windows="${NSRL_LITERARY_MAX_WINDOWS:-4096}"
epochs="${NSRL_LITERARY_EPOCHS:-1}"
batch_windows="${NSRL_LITERARY_BATCH_WINDOWS:-4}"
workers="${NSRL_LITERARY_WORKERS:-4}"
holdout_bytes="${NSRL_LITERARY_HOLDOUT_BYTES:-8192}"
adaptive_attention="${NSRL_LITERARY_ADAPTIVE_ATTENTION:-0}"

default_shakespeare_texts="data/processed/crowley-bard-focused-v1/shakespeare.body.txt"
default_blake_texts="data/processed/blake-poems.clean.txt:data/processed/blake-marriage-heaven-hell.clean.txt:data/processed/crowley-bard-sources/blake-poems-yeats.clean.txt:data/processed/crowley-bard-sources/blake-songs.clean.txt"
default_crowley_texts="data/processed/crowley-household-gods.clean.txt:data/processed/crowley-tannhauser.clean.txt"
while IFS= read -r source; do
  default_crowley_texts+=":$source"
done < <(find data/processed/crowley-bard-sources -maxdepth 1 -type f -name 'crowley-*.clean.txt' | sort)

IFS=':' read -r -a shakespeare_sources <<< "${NSRL_SHAKESPEARE_TEXTS:-$default_shakespeare_texts}"
IFS=':' read -r -a blake_sources <<< "${NSRL_BLAKE_TEXTS:-$default_blake_texts}"
IFS=':' read -r -a crowley_sources <<< "${NSRL_CROWLEY_TEXTS:-$default_crowley_texts}"

for value in "$seq_len" "$max_windows" "$epochs" "$batch_windows" "$workers" "$holdout_bytes" "$adaptive_attention"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "literary run settings must be non-negative integers: $value" >&2
    exit 1
  fi
done
if ((seq_len < 1 || max_windows < 1 || epochs < 1 || batch_windows < 1 || workers < 1)); then
  echo "sequence length, windows, epochs, batch size, and workers must be positive" >&2
  exit 1
fi

for source in "${shakespeare_sources[@]}" "${blake_sources[@]}" "${crowley_sources[@]}"; do
  if [[ ! -s "$source" ]]; then
    echo "missing literary source: $source" >&2
    exit 1
  fi
done

mkdir -p "$out_dir"

builder_args=(--out-dir "$out_dir" --holdout-bytes-per-author "$holdout_bytes")
for source in "${shakespeare_sources[@]}"; do builder_args+=(--shakespeare "$source"); done
for source in "${blake_sources[@]}"; do builder_args+=(--blake "$source"); done
for source in "${crowley_sources[@]}"; do builder_args+=(--crowley "$source"); done
node scripts/build-literary-corpus.mjs "${builder_args[@]}"

cargo build --release -p nsrl-corpus -p nsrl-train \
  --bin nsrl-corpus --bin nsrl-train --bin nsrl-mini-transformer-eval

token_budget="$(wc -c < "$out_dir/corpus.txt" | tr -d ' ')"
if [[ -n "${NSRL_LITERARY_STRIDE:-}" ]]; then
  stride="$NSRL_LITERARY_STRIDE"
else
  trainable_tokens=$((token_budget > seq_len ? token_budget - seq_len : 1))
  stride=$(((trainable_tokens + max_windows - 1) / max_windows))
  ((stride > 0)) || stride=1
fi

target/release/nsrl-corpus tokenize \
  --corpus "$out_dir/corpus.txt" \
  --tokens-out "$out_dir/tokens.u8" \
  --trace "$out_dir/tokens.trace.jsonl" \
  --seq-len "$seq_len" \
  --stride "$stride" \
  --text-profile ascii-lower

if ((holdout_bytes > 0)); then
  holdout_token_budget="$(wc -c < "$out_dir/holdout.txt" | tr -d ' ')"
  holdout_max_windows="${NSRL_LITERARY_HOLDOUT_WINDOWS:-4096}"
  holdout_trainable=$((holdout_token_budget > seq_len ? holdout_token_budget - seq_len : 1))
  holdout_stride=$(((holdout_trainable + holdout_max_windows - 1) / holdout_max_windows))
  ((holdout_stride > 0)) || holdout_stride=1
  target/release/nsrl-corpus tokenize \
    --corpus "$out_dir/holdout.txt" \
    --tokens-out "$out_dir/holdout.tokens.u8" \
    --trace "$out_dir/holdout.tokens.trace.jsonl" \
    --seq-len "$seq_len" \
    --stride "$holdout_stride" \
    --text-profile ascii-lower
fi

train_args=(
  --mode mini-transformer-mlp
  --tokens "$out_dir/tokens.u8"
  --model-out "$out_dir/model.nsrlmt"
  --epochs "$epochs"
  --seq-len "$seq_len"
  --stride "$stride"
  --max-windows "$max_windows"
  --batch-windows "$batch_windows"
  --tokenizer ascii-lower
  --mini-transformer-attention linear
  --mini-transformer-position nope
  --mini-transformer-batch-mode map-reduce
  --mini-transformer-map-reduce-workers "$workers"
  --mini-transformer-trace-detail summary
  --trace "$out_dir/train.trace.jsonl"
  --progress-out "$out_dir/progress.jsonl"
  --progress-interval-batches 64
)
if ((adaptive_attention != 0)); then
  train_args+=(--adaptive-attention-shifts)
fi
target/release/nsrl-train "${train_args[@]}"

if ((holdout_bytes > 0)); then
  target/release/nsrl-mini-transformer-eval \
    --tokens "$out_dir/holdout.tokens.u8" \
    --model "$out_dir/model.nsrlmt" \
    --stride "$holdout_stride" \
    --max-windows "$holdout_max_windows" \
    --attention linear \
    --position nope \
    --out "$out_dir/holdout.eval.jsonl"
fi

target/release/nsrl-train \
  --mode mini-transformer-generate \
  --model "$out_dir/model.nsrlmt" \
  --tokens "$out_dir/tokens.u8" \
  --prompt "the soul" \
  --max-new-tokens 128 \
  --decode sample \
  --top-k 8 \
  --sample-seed 7 \
  --tokenizer ascii-lower \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --printable-only \
  --repeat-window 32 \
  --repeat-penalty-shift 2 \
  --corpus-prior \
  --text-out "$out_dir/sample.txt" \
  --generated-only \
  --trace "$out_dir/sample.trace.jsonl"

echo "model:  $out_dir/model.nsrlmt"
echo "sample: $out_dir/sample.txt"
echo "trace:  $out_dir/train.trace.jsonl"
