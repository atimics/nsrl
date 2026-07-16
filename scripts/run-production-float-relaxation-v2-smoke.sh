#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

integer_dir="${NSRL_PRODUCTION_SMOKE_OUT_DIR:-data/experiments/production-model-v1/p10m-smoke}"
out_dir="${NSRL_PRODUCTION_FLOAT_RELAXATION_OUT_DIR:-data/experiments/production-model-v1/p10m-float-relaxation-v2}"
tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"

if [[ ! -f "$integer_dir/initial.nsrlpm" ]]; then
  scripts/run-production-model-v1-smoke.sh
fi

python3 scripts/production-float-twin-v1.py \
  --math-contract integer-relaxation-v2 \
  --model "$integer_dir/initial.nsrlpm" \
  --tokens "$tokens" \
  --out "$out_dir/trained.npz" \
  --trace "$out_dir/train.json" \
  --context-tokens 4 \
  --max-windows 8 \
  --epochs 2 \
  --batch-windows 4 \
  --learning-rate-millionths 1000

node -e '
const fs = require("node:fs");
const trace = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
if (trace.schema !== "nsrl.production_float_relaxation.v2"
  || trace.math_contract?.id !== "integer-relaxation-v2"
  || trace.math_contract?.softmax_base !== "2"
  || trace.math_contract?.nll_unit !== "bits"
  || trace.math_contract?.rms_epsilon !== 2 ** -30) {
  throw new Error("float relaxation v2 math contract mismatch");
}
' "$out_dir/train.json"

echo "production-model p10m faithful float relaxation v2 smoke passed"
