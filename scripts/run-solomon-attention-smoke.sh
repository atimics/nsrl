#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_SOLOMON_ATTENTION_OUT_DIR:-data/processed/key-solomon-goetia-attention-v1}"
text_index="${NSRL_SOLOMON_ATTENTION_TEXT_INDEX:-web/assets/solomon-spirit-text-signatures.tsv}"
model="${NSRL_SOLOMON_ATTENTION_MODEL:-$out_dir/model.nsrllmm}"
epochs="${NSRL_SOLOMON_ATTENTION_EPOCHS:-1}"
seq_len="${NSRL_SOLOMON_ATTENTION_SEQ_LEN:-32}"
stride="${NSRL_SOLOMON_ATTENTION_STRIDE:-1}"
window_offset="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET:-0}"
max_windows="${NSRL_SOLOMON_ATTENTION_MAX_WINDOWS:-256}"
batch_windows="${NSRL_SOLOMON_ATTENTION_BATCH_WINDOWS:-8}"
text_only_repeats="${NSRL_SOLOMON_ATTENTION_TEXT_ONLY_REPEATS:-0}"
name_opening_repeats="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS:-0}"
learning_rate="${NSRL_SOLOMON_ATTENTION_LEARNING_RATE:-1}"
output_lr_shift="${NSRL_SOLOMON_ATTENTION_OUTPUT_LR_SHIFT:-18}"
mlp_lr_shift="${NSRL_SOLOMON_ATTENTION_MLP_LR_SHIFT:-16}"
embed_lr_shift="${NSRL_SOLOMON_ATTENTION_EMBED_LR_SHIFT:-14}"
attention_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_LR_SHIFT:-24}"
attention_q_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_Q_LR_SHIFT:-18}"
attention_qk_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_QK_LR_SHIFT:-18}"

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$out_dir" \
  --pad-context "$seq_len" \
  --text-only-repeats "$text_only_repeats" \
  --name-opening-repeats "$name_opening_repeats"

train_args=(
  train
  --tokens "$out_dir/corpus.tokens.u8"
  --model-out "$model"
  --epochs "$epochs"
  --seq-len "$seq_len"
  --stride "$stride"
  --window-offset "$window_offset"
  --batch-windows "$batch_windows"
  --learning-rate "$learning_rate"
  --output-lr-shift "$output_lr_shift"
  --mlp-lr-shift "$mlp_lr_shift"
  --embed-lr-shift "$embed_lr_shift"
  --attention-lr-shift "$attention_lr_shift"
  --attention-q-lr-shift "$attention_q_lr_shift"
  --attention-qk-lr-shift "$attention_qk_lr_shift"
)
if [[ "$max_windows" == "none" ]]; then
  train_args+=(--max-windows none)
else
  train_args+=(--max-windows "$max_windows")
fi

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- "${train_args[@]}"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- eval \
  --model "$model" \
  --tokens "$out_dir/corpus.tokens.u8" \
  --conditioning-examples "$out_dir/examples.jsonl" \
  --eval-max-examples 4 \
  > "$out_dir/attention-eval.json"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$model" \
  --out-dir "$out_dir/attention-sample-bael" \
  --prompt "seal of Bael" \
  --min-text-tokens 16 \
  --max-text-tokens 220 \
  --repeat-run-cap 4 \
  --top-k 1

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$model" \
  --out-dir "$out_dir/attention-sample-stolas" \
  --prompt "seal of Stolas" \
  --min-text-tokens 16 \
  --max-text-tokens 220 \
  --repeat-run-cap 4 \
  --top-k 1

grep -Fqx "Solomon selects Bael: He maketh thee to go Invisible." \
  "$out_dir/attention-sample-bael/text.txt"
grep -Fqx "Solomon selects Stolas: He teacheth the Art of Astronomy, and the Virtues of Herbs and Precious Stones." \
  "$out_dir/attention-sample-stolas/text.txt"
test "$(wc -c < "$out_dir/attention-sample-bael/image.ink16.u8")" -eq 256
test "$(wc -c < "$out_dir/attention-sample-stolas/image.ink16.u8")" -eq 256
grep -Fq '"conditioning_primary_name":"Bael"' "$out_dir/attention-sample-bael/sample.json"
grep -Fq '"conditioning_primary_name":"Stolas"' "$out_dir/attention-sample-stolas/sample.json"
grep -Fq '"schema":"nsrl.solomon_attention_eval_trace.v1"' "$out_dir/attention-eval.json"

echo "Solomon attention smoke wrote $out_dir"
