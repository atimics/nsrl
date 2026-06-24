#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Train the Solomon text-conditioned bitmap denoiser on native Linux/Graviton.

Common knobs:
  NSRL_SOLOMON_DENOISE_EPOCHS=8
  NSRL_SOLOMON_DENOISE_DATASET=data/processed/key-solomon-goetia-denoise-v1
  NSRL_SOLOMON_TEXT_INDEX=data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv
  NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR=data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv
  NSRL_SOLOMON_TEXT_DENOISE_MODEL=data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch
  NSRL_S3_URI=s3://bucket/prefix

The output model must be an NSRLTCH artifact with at least 30 channels so the
learned 16x16 layout signature reaches the denoiser.
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

dataset_root="${NSRL_SOLOMON_DENOISE_DATASET:-data/processed/key-solomon-goetia-denoise-v1}"
text_index="${NSRL_SOLOMON_TEXT_INDEX:-data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv}"
out_dir="${NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR:-${dataset_root}/text-multichannel-conv}"
model_out="${NSRL_SOLOMON_TEXT_DENOISE_MODEL:-${out_dir}/model.nsrltch}"
epochs="${NSRL_SOLOMON_DENOISE_EPOCHS:-8}"
preview_pairs="${NSRL_SOLOMON_DENOISE_PREVIEW_PAIRS:-32}"
image_size="${NSRL_SOLOMON_DENOISE_IMAGE_SIZE:-128}"
timesteps="${NSRL_SOLOMON_DENOISE_TIMESTEPS:-8}"
layers="${NSRL_SOLOMON_DENOISE_LAYERS:-3}"
hidden_shift="${NSRL_SOLOMON_DENOISE_HIDDEN_SHIFT:-1}"
output_shift="${NSRL_SOLOMON_DENOISE_OUTPUT_SHIFT:-9}"
learning_shift="${NSRL_SOLOMON_DENOISE_LEARNING_SHIFT:-24}"
bias_learning_shift="${NSRL_SOLOMON_DENOISE_BIAS_LEARNING_SHIFT:-30}"
target_dir="${CARGO_TARGET_DIR:-target}"
release_bin_dir="${target_dir%/}/release"

for required in "$dataset_root/pairs/train.input.ink${image_size}.u8" \
  "$dataset_root/pairs/train.target.ink${image_size}.u8" \
  "$dataset_root/rows/train.pairs.jsonl" \
  "$text_index"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

node - "$dataset_root" <<'JS'
const fs = require("node:fs");
const path = require("node:path");

const datasetRoot = process.argv[2];
for (const split of ["train", "eval"]) {
  const rowsPath = path.join(datasetRoot, "rows", `${split}.pairs.jsonl`);
  const counts = new Map();
  for (const line of fs.readFileSync(rowsPath, "utf8").split(/\r?\n/)) {
    if (!line) continue;
    const row = JSON.parse(line);
    counts.set(row.corruption, (counts.get(row.corruption) ?? 0) + 1);
  }
  const noiseCount = counts.get("noise-seed") ?? 0;
  if (noiseCount <= 0) {
    throw new Error(`${rowsPath}: missing noise-seed pairs; rebuild with --corruptions-per-image 10`);
  }
  console.log(`${split}_noise_seed_pairs=${noiseCount}`);
}
JS

sync_artifacts() {
  if [[ -n "${NSRL_S3_URI:-}" ]]; then
    aws s3 sync "$out_dir" "${NSRL_S3_URI%/}/text-denoiser/${out_dir##*/}" --only-show-errors
  fi
}

echo "[1/3] building text denoiser trainer"
cargo build --release -p nsrl-train --no-default-features \
  --bin nsrl-bitmap-multichannel-denoise

echo "[2/3] training NSRLTCH denoiser epochs=${epochs} -> ${model_out}"
"${release_bin_dir}/nsrl-bitmap-multichannel-denoise" \
  --dataset "$dataset_root" \
  --text-index "$text_index" \
  --out-dir "$out_dir" \
  --model-out "$model_out" \
  --epochs "$epochs" \
  --image-size "$image_size" \
  --timesteps "$timesteps" \
  --layers "$layers" \
  --hidden-shift "$hidden_shift" \
  --output-shift "$output_shift" \
  --learning-shift "$learning_shift" \
  --bias-learning-shift "$bias_learning_shift" \
  --preview-pairs "$preview_pairs"

echo "[3/3] checking trained denoiser header"
node scripts/check-solomon-denoiser-model.mjs --model "$model_out"
sync_artifacts
