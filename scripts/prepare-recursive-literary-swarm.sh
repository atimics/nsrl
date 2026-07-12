#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"
source_manifest="${NSRL_RECURSIVE_SWARM_SOURCE_MANIFEST:-data/local-runs/literary-scale-8k-seq32-fixed/corpus.manifest.json}"
seq_len="${NSRL_RECURSIVE_SWARM_SEQ_LEN:-32}"
leaf_windows="${NSRL_RECURSIVE_SWARM_LEAF_WINDOWS:-8192}"

manifest_path="$(node scripts/build-recursive-literary-swarm-experiment.mjs \
  --source-manifest "$source_manifest" \
  --out-dir "$out_dir" \
  --seq-len "$seq_len" \
  --leaf-windows "$leaf_windows")"

cargo build --release -p nsrl-corpus --bin nsrl-corpus

tail -n +2 "$out_dir/leaf-jobs.tsv" | while IFS=$'\t' read -r \
  expert_id author variant text_path tokens_path model_path train_trace_path \
  job_seq_len max_windows stride window_offset; do
  target/release/nsrl-corpus tokenize \
    --corpus "$text_path" \
    --tokens-out "$tokens_path" \
    --trace "${tokens_path%.u8}.trace.jsonl" \
    --seq-len "$job_seq_len" \
    --stride "$stride" \
    --text-profile ascii-lower
done

for author in crowley shakespeare blake; do
  for split in leaf-train router-train router-calibration final-test; do
    text_path="$out_dir/splits/$author/$split.txt"
    tokens_path="$out_dir/splits/$author/$split.tokens.u8"
    target/release/nsrl-corpus tokenize \
      --corpus "$text_path" \
      --tokens-out "$tokens_path" \
      --trace "$out_dir/splits/$author/$split.tokens.trace.jsonl" \
      --seq-len "$seq_len" \
      --stride 1 \
      --text-profile ascii-lower
  done
done

node scripts/check-recursive-literary-swarm-experiment.mjs \
  --manifest "$manifest_path" \
  --out "$out_dir/preflight.json"

echo "manifest:  $manifest_path"
echo "preflight: $out_dir/preflight.json"
echo "leaf jobs: $out_dir/leaf-jobs.tsv"
