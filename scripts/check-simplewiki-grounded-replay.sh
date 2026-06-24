#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest_default="data/processed/simplewiki-expository-v1/topic-earth-curriculum-holo-sentence-stop-smoke-20260621/paragraph-bestof-earth3-lastprompt-grounded16-20260621/manifest.json"
manifest="${MANIFEST:-$manifest_default}"
timestamp="${TIMESTAMP:-0}"

if [[ "${FAST:-0}" == "1" ]]; then
  bpt_max_windows="${BPT_MAX_WINDOWS:-1024}"
  expected_row_hash="${EXPECTED_ROW_HASH:-ba35b515}"
  out="${OUT:-target/nsrl-simplewiki-grounded-eval.fast.jsonl}"
else
  bpt_max_windows="${BPT_MAX_WINDOWS:-10000}"
  expected_row_hash="${EXPECTED_ROW_HASH:-ba2028c5}"
  out="${OUT:-target/nsrl-simplewiki-grounded-eval.full.jsonl}"
fi

if [[ ! -f "$manifest" ]]; then
  echo "missing SimpleWiki grounded manifest: $manifest" >&2
  echo "Build or fetch the topic-earth paragraph artifacts before running this replay gate." >&2
  exit 1
fi

mkdir -p "$(dirname "$out")"

cargo run --quiet -p nsrl-train --bin nsrl-simplewiki-grounded-eval -- \
  --manifest "$manifest" \
  --timestamp "$timestamp" \
  --bpt-max-windows "$bpt_max_windows" \
  --expect-row-hash "$expected_row_hash" \
  > "$out"

printf 'simplewiki_grounded_replay_passed\trow_hash=%s\tbpt_max_windows=%s\tout=%s\n' \
  "$expected_row_hash" \
  "$bpt_max_windows" \
  "$out"
