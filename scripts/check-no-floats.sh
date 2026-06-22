#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

patterns=(
  '\bf32\b'
  '\bf64\b'
  '\bas[[:space:]]+f32\b'
  '\bas[[:space:]]+f64\b'
  '\.sqrt[[:space:]]*\('
  '\.powf[[:space:]]*\('
  '\.round[[:space:]]*\('
  '[0-9][0-9_]*\.[0-9]'
)

failed=0
for pattern in "${patterns[@]}"; do
  if rg --line-number --glob '*.rs' --regexp "$pattern" crates; then
    failed=1
  fi
done

dataset_patterns=(
  'Number\.isFinite'
  '\.toFixed[[:space:]]*\('
  'rng[[:space:]]*\([[:space:]]*\)[[:space:]]*<'
  '[0-9][0-9_]*\.[0-9]'
)

for pattern in "${dataset_patterns[@]}"; do
  if rg --line-number --regexp "$pattern" scripts/build-solomon-bitmap-denoise-dataset.mjs; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "banned floating-point syntax found in checked source" >&2
  exit 1
fi
