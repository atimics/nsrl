#!/usr/bin/env bash
# Build the Solomon-seal denoise dataset, train the integer 3-layer conv
# denoiser, and sample a 64-seal preview grid. Pure integer arithmetic
# end to end (no floating point). Deterministic: same seeds in, same seals out.
#
# Source: Project Gutenberg PG72679 (Key of Solomon / Goetia), public domain.
# See docs/solomon-seal-denoise.md.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset_root="data/processed/key-solomon-goetia-denoise-v1"
model_dir="${dataset_root}/baseline-three-layer-conv"
samples_dir="${model_dir}/samples"

epochs="${EPOCHS:-8}"
samples="${SAMPLES:-64}"
candidate_multiplier="${CANDIDATE_MULTIPLIER:-4}"
diversity_weight="${DIVERSITY_WEIGHT:-1}"
init="noise"

manifest="data/processed/key-solomon-goetia-bitmaps-pg72679/slices/manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "missing sliced source manifest: $manifest" >&2
  echo "regenerate the PG72679 seal-grid slices first (see docs/solomon-seal-denoise.md)." >&2
  exit 1
fi

echo "[1/3] building denoise dataset -> ${dataset_root}"
cargo run --release -q -p nsrl-train --bin nsrl-build-solomon-bitmap-denoise-dataset

echo "[2/3] training 3-layer integer conv denoiser (epochs=${epochs}) -> ${model_dir}/model.nsrlcv3"
cargo run --release -q -p nsrl-train --bin nsrl-bitmap-three-layer-denoise -- --epochs "$epochs"

echo "[3/3] sampling ${samples} seals (init=${init}) -> ${samples_dir}"
cargo run --release -q -p nsrl-train --bin nsrl-bitmap-sample -- \
  --samples "$samples" \
  --init "$init" \
  --candidate-multiplier "$candidate_multiplier" \
  --diversity-weight "$diversity_weight"

echo "done. preview grid under ${samples_dir}/"
