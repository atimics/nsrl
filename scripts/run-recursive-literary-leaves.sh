#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"
max_jobs="${NSRL_RECURSIVE_SWARM_MAX_JOBS:-0}"
only_author="${NSRL_RECURSIVE_SWARM_ONLY_AUTHOR:-}"
workers="${NSRL_RECURSIVE_SWARM_WORKERS:-8}"
skip_existing="${NSRL_RECURSIVE_SWARM_SKIP_EXISTING:-1}"
completed=0
skipped=0

if [[ ! -s "$out_dir/preflight.json" ]] || ! jq -e '.preparation_ready == true' "$out_dir/preflight.json" >/dev/null; then
  echo "recursive swarm preparation is missing or not ready; run scripts/prepare-recursive-literary-swarm.sh" >&2
  exit 1
fi

cargo build --release -p nsrl-train \
  --bin nsrl-train --bin nsrl-mini-transformer-eval

while IFS=$'\t' read -r expert_id author variant text_path tokens_path model_path \
  train_trace_path seq_len max_windows stride window_offset; do
  if [[ "$expert_id" == "expert_id" ]]; then continue; fi
  if [[ -n "$only_author" && "$author" != "$only_author" ]]; then continue; fi
  if ((max_jobs > 0 && completed >= max_jobs)); then break; fi
  if ((skip_existing != 0)) && [[ -s "$model_path" && -s "$train_trace_path" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  expert_dir="$(dirname "$model_path")"
  mkdir -p "$expert_dir"
  target/release/nsrl-train \
    --mode mini-transformer-mlp \
    --tokens "$tokens_path" \
    --model-out "$model_path" \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$window_offset" \
    --max-windows "$max_windows" \
    --batch-windows 4 \
    --tokenizer ascii-lower \
    --mini-transformer-attention linear \
    --mini-transformer-position nope \
    --mini-transformer-batch-mode map-reduce \
    --mini-transformer-map-reduce-workers "$workers" \
    --mini-transformer-trace-detail summary \
    --trace "$train_trace_path" \
    --progress-out "$expert_dir/progress.jsonl" \
    --progress-interval-batches 64

  for eval_author in crowley shakespeare blake; do
    eval_tokens="$out_dir/splits/$eval_author/final-test.tokens.u8"
    eval_bytes="$(wc -c < "$eval_tokens" | tr -d ' ')"
    eval_trainable=$((eval_bytes > seq_len ? eval_bytes - seq_len : 1))
    eval_stride=$(((eval_trainable + 1023) / 1024))
    ((eval_stride > 0)) || eval_stride=1
    target/release/nsrl-mini-transformer-eval \
      --tokens "$eval_tokens" \
      --model "$model_path" \
      --stride "$eval_stride" \
      --max-windows 1024 \
      --attention linear \
      --position nope \
      --out "$expert_dir/eval-$eval_author.jsonl"
  done
  completed=$((completed + 1))
done < "$out_dir/leaf-jobs.tsv"

echo "completed leaf jobs: $completed"
echo "skipped existing leaf jobs: $skipped"
