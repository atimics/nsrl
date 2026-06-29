#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_SOLOMON_MULTIMODAL_OUT_DIR:-data/processed/key-solomon-goetia-multimodal-v1}"
text_index="${NSRL_SOLOMON_MULTIMODAL_TEXT_INDEX:-web/assets/solomon-spirit-text-signatures.tsv}"
model="${NSRL_SOLOMON_MULTIMODAL_MODEL:-$out_dir/model.nsrlmod}"

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$out_dir"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-multimodal -- train \
  --tokens "$out_dir/corpus.tokens.u16" \
  --model-out "$model"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-multimodal -- sample \
  --model "$model" \
  --out-dir "$out_dir/sample-bael" \
  --prompt "seal of Bael" \
  --top-k 1

cargo run --quiet -p nsrl-train --bin nsrl-solomon-multimodal -- sample \
  --model "$model" \
  --out-dir "$out_dir/sample-stolas" \
  --prompt "seal of Stolas" \
  --top-k 1

echo "Solomon multimodal smoke wrote $out_dir"
