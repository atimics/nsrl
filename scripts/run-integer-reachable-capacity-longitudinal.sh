#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-data/experiments/integer-reachable-capacity-v1/longitudinal}"
model="data/local-runs/literary-rms-adam-8k-shift8/model.nsrlmt"
train_tokens="data/local-runs/literary-scale-8k-seq32/tokens.u8"
heldout_tokens="data/local-runs/literary-scale-8k-seq32/holdout.tokens.u8"
early_dir="data/experiments/integer-reachable-capacity-v1"
binary="target/release/nsrl-mini-transformer-low-rank-expert"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-mini-transformer-low-rank-expert

if [[ ! -s "$out_dir/zero-heldout.eval.json" ]]; then
  "$binary" eval \
    --tokens "$heldout_tokens" --model "$model" \
    --expert "$early_dir/rank8-shift4-carryoff.nsrlle" \
    --trace "$out_dir/zero-heldout.eval.json.tmp" \
    --stride 1 --max-windows 4096
  mv "$out_dir/zero-heldout.eval.json.tmp" "$out_dir/zero-heldout.eval.json"
fi

for rank in 8 16 32; do
  for shift in 0 1 2 3 4; do
    for carry in off on; do
      id="rank${rank}-shift${shift}-carry${carry}"
      if [[ -s "$out_dir/$id.train.json" && -s "$out_dir/$id.eval.json" \
        && -s "$out_dir/$id.nsrlle" ]]; then
        echo "longitudinal reuse: $id"
        continue
      fi
      train_args=(
        train --tokens "$train_tokens" --model "$model"
        --out "$out_dir/$id.nsrlle.tmp" --trace "$out_dir/$id.train.json.tmp"
        --rank "$rank" --epochs 1 --stride 1 --max-windows 2048
        --batch-windows 32 --learning-rate 16384 --learning-rate-shift "$shift"
      )
      if [[ "$carry" == "off" ]]; then train_args+=(--no-error-feedback); fi
      echo "longitudinal train/eval: $id"
      "$binary" "${train_args[@]}"
      mv "$out_dir/$id.nsrlle.tmp" "$out_dir/$id.nsrlle"
      mv "$out_dir/$id.train.json.tmp" "$out_dir/$id.train.json"
      "$binary" eval \
        --tokens "$heldout_tokens" --model "$model" \
        --expert "$out_dir/$id.nsrlle" --trace "$out_dir/$id.eval.json.tmp" \
        --stride 1 --max-windows 4096
      mv "$out_dir/$id.eval.json.tmp" "$out_dir/$id.eval.json"
    done
  done
done

node scripts/summarize-integer-reachable-capacity-longitudinal.mjs \
  --input-dir "$out_dir" \
  --out "$out_dir/report.json"
node scripts/summarize-integer-reachable-capacity-longitudinal.mjs \
  --input-dir "$out_dir" \
  --out benchmarks/integer-reachable-capacity-v1/longitudinal.json

echo "integer reachable-capacity longitudinal study written to $out_dir/report.json"
