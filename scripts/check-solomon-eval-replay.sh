#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

required=(
  "data/processed/key-solomon-goetia-latent-v1/model.nsrllat"
  "data/processed/key-solomon-goetia-latent-v1/prompts.jsonl"
  "data/processed/key-solomon-goetia-latent-v1/gold.tsv"
  "data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv"
)

missing=0
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing replay fixture: $path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Solomon eval replay requires generated local fixtures; run the Solomon corpus/model build first." >&2
  exit 1
fi

cargo run --quiet -p nsrl-train --bin nsrl-solomon-eval -- \
  --timestamp 0 \
  --no-ledger \
  --no-partition \
  --expect-row-hash b79c83bd >/dev/null
