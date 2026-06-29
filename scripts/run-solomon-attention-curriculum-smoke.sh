#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

joint_out="${NSRL_SOLOMON_ATTENTION_CURRICULUM_OUT_DIR:-data/processed/key-solomon-goetia-attention-curriculum-v1}"
text_out="${NSRL_SOLOMON_ATTENTION_TEXT_PRETRAIN_OUT_DIR:-data/processed/key-solomon-goetia-attention-text-only-v1}"
opening_out="${NSRL_SOLOMON_ATTENTION_OPENING_PRETRAIN_OUT_DIR:-data/processed/key-solomon-goetia-attention-opening-v1}"
text_index="${NSRL_SOLOMON_ATTENTION_TEXT_INDEX:-web/assets/solomon-spirit-text-signatures.tsv}"
seq_len="${NSRL_SOLOMON_ATTENTION_SEQ_LEN:-32}"
stride="${NSRL_SOLOMON_ATTENTION_STRIDE:-1}"
window_offset="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET:-0}"
window_offset_sweep="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET_SWEEP:-single}"
batch_windows="${NSRL_SOLOMON_ATTENTION_BATCH_WINDOWS:-8}"
text_max_windows="${NSRL_SOLOMON_ATTENTION_TEXT_MAX_WINDOWS:-2048}"
opening_max_windows="${NSRL_SOLOMON_ATTENTION_OPENING_MAX_WINDOWS:-4096}"
joint_max_windows="${NSRL_SOLOMON_ATTENTION_JOINT_MAX_WINDOWS:-256}"
min_text_accuracy="${NSRL_SOLOMON_ATTENTION_MIN_TEXT_ACCURACY_PER_MILLE:-100}"
max_text_chars="${NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS:-220}"
prompt_profile="${NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE:-all}"
text_token_profile="${NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE:-char}"
joint_text_only_repeats="${NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS:-0}"
name_opening_repeats="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS:-0}"
name_opening_pretrain="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_PRETRAIN:-0}"
learning_rate="${NSRL_SOLOMON_ATTENTION_LEARNING_RATE:-1}"
output_lr_shift="${NSRL_SOLOMON_ATTENTION_OUTPUT_LR_SHIFT:-18}"
mlp_lr_shift="${NSRL_SOLOMON_ATTENTION_MLP_LR_SHIFT:-16}"
embed_lr_shift="${NSRL_SOLOMON_ATTENTION_EMBED_LR_SHIFT:-14}"
attention_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_LR_SHIFT:-24}"
attention_q_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_Q_LR_SHIFT:-18}"
attention_qk_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_QK_LR_SHIFT:-18}"
text_model="$text_out/pretrain.nsrllmm"
opening_model="$opening_out/pretrain.nsrllmm"
joint_model="$joint_out/model.nsrllmm"

train_lr_args=(
  --learning-rate "$learning_rate"
  --output-lr-shift "$output_lr_shift"
  --mlp-lr-shift "$mlp_lr_shift"
  --embed-lr-shift "$embed_lr_shift"
  --attention-lr-shift "$attention_lr_shift"
  --attention-q-lr-shift "$attention_q_lr_shift"
  --attention-qk-lr-shift "$attention_qk_lr_shift"
)

if [[ "$window_offset_sweep" == "all" ]]; then
  train_offsets=()
  for ((offset = 0; offset < stride; offset += 1)); do
    train_offsets+=("$offset")
  done
else
  train_offsets=("$window_offset")
fi

text_init_args=()
if [[ "$name_opening_pretrain" != "0" ]]; then
  node scripts/build-solomon-multimodal-corpus.mjs \
    --text-index "$text_index" \
    --out-dir "$opening_out" \
    --max-text-chars "$max_text_chars" \
    --prompt-profile "$prompt_profile" \
    --pad-context "$seq_len" \
    --sequence-profile name-opening \
    --name-opening-repeats "$name_opening_repeats" \
    --text-token-profile "$text_token_profile"

  opening_window_args=(--max-windows "$opening_max_windows")
  if [[ "$opening_max_windows" == "none" ]]; then
    opening_window_args=(--max-windows none)
  fi

  cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- train \
    --tokens "$opening_out/corpus.tokens.u8" \
    --model-out "$opening_model" \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$window_offset" \
    "${opening_window_args[@]}" \
    --batch-windows "$batch_windows" \
    --text-token-profile "$text_token_profile" \
    "${train_lr_args[@]}" \
    > "$opening_out/train.json"
  text_init_args=(--init-model "$opening_model")
fi

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$text_out" \
  --max-text-chars "$max_text_chars" \
  --prompt-profile "$prompt_profile" \
  --pad-context "$seq_len" \
  --sequence-profile text-only \
  --name-opening-repeats "$name_opening_repeats" \
  --text-token-profile "$text_token_profile"

for offset in "${train_offsets[@]}"; do
  cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- train \
    --tokens "$text_out/corpus.tokens.u8" \
    ${text_init_args[@]+"${text_init_args[@]}"} \
    --model-out "$text_model" \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$offset" \
    --max-windows "$text_max_windows" \
    --batch-windows "$batch_windows" \
    --text-token-profile "$text_token_profile" \
    "${train_lr_args[@]}" \
    > "$text_out/train-offset-$offset.json"
  text_init_args=(--init-model "$text_model")
done

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$joint_out" \
  --max-text-chars "$max_text_chars" \
  --prompt-profile "$prompt_profile" \
  --pad-context "$seq_len" \
  --text-only-repeats "$joint_text_only_repeats" \
  --name-opening-repeats "$name_opening_repeats" \
  --text-token-profile "$text_token_profile"

joint_init_args=(--init-model "$text_model")
for offset in "${train_offsets[@]}"; do
  cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- train \
    --tokens "$joint_out/corpus.tokens.u8" \
    "${joint_init_args[@]}" \
    --model-out "$joint_model" \
    --embed-text-memory-examples "$joint_out/examples.jsonl" \
    --embed-text-memory-order 32 \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$offset" \
    --max-windows "$joint_max_windows" \
    --batch-windows "$batch_windows" \
    --text-token-profile "$text_token_profile" \
    "${train_lr_args[@]}" \
    > "$joint_out/train-offset-$offset.json"
  cp "$joint_out/train-offset-$offset.json" "$joint_out/train.json"
  joint_init_args=(--init-model "$joint_model")
done

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- eval \
  --model "$joint_model" \
  --tokens "$joint_out/corpus.tokens.u8" \
  --conditioning-examples "$joint_out/examples.jsonl" \
  --eval-max-examples 8 \
  > "$joint_out/attention-eval.json"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$joint_model" \
  --tokens "$joint_out/corpus.tokens.u8" \
  --conditioning-examples none \
  --out-dir "$joint_out/prior-sample-bael" \
  --prompt "seal of Bael" \
  --min-text-tokens 16 \
  --max-text-tokens 260 \
  --repeat-run-cap 4 \
  --no-repeat-ngram 4 \
  --top-k 1 \
  --sample-seed 13 \
  > "$joint_out/prior-sample.json"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$joint_model" \
  --tokens "$joint_out/corpus.tokens.u8" \
  --conditioning-examples none \
  --text-prior-examples none \
  --no-embedded-text-memory \
  --embedded-text-lm-order 6 \
  --out-dir "$joint_out/lm-sample-bael" \
  --prompt "seal of Bael" \
  --min-text-tokens 16 \
  --max-text-tokens 160 \
  --repeat-run-cap 4 \
  --no-repeat-ngram 4 \
  --top-k 1 \
  --sample-seed 13 \
  > "$joint_out/lm-sample.json"

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$joint_model" \
  --tokens "$joint_out/corpus.tokens.u8" \
  --conditioning-examples none \
  --text-prior-examples none \
  --no-embedded-text-memory \
  --out-dir "$joint_out/raw-sample-bael" \
  --prompt "seal of Bael" \
  --text-prefix "Solomon selects " \
  --min-text-tokens 32 \
  --max-text-tokens 96 \
  --repeat-run-cap 4 \
  --no-repeat-ngram 4 \
  --top-k 1 \
  --sample-seed 13 \
  > "$joint_out/raw-sample.json"

node -e 'const fs=require("fs"); const path=process.argv[1]; const min=Number(process.argv[2]); const row=JSON.parse(fs.readFileSync(path,"utf8")); const got=row.text.accuracy_per_mille; if (!(got >= min)) { console.error(`text accuracy ${got} < ${min}`); process.exit(1); } console.log(`Solomon attention curriculum smoke wrote ${process.argv[3]} text_accuracy_per_mille=${got}`);' \
  "$joint_out/attention-eval.json" \
  "$min_text_accuracy" \
  "$joint_out"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); if (!text.startsWith("Solomon selects Bael: ") || !text.includes(".") || /Solomon selects (?!Bael:)/.test(text)) { console.error(`weak prior-assisted text: ${text}`); process.exit(1); } console.log(`prior_assisted_text=${text}`);' \
  "$joint_out/prior-sample-bael/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded" || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== true) { console.error(`sample did not use embedded text memory: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_text_memory_contexts=${row.text_prior_contexts}`);' \
  "$joint_out/prior-sample-bael/sample.json"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); const hasClause=text.includes(": He ") || text.includes(" and "); if (!text.startsWith("Solomon selects Bael: ") || !hasClause || /aaa|eee|hhh/.test(text)) { console.error(`weak embedded-lm text: ${text}`); process.exit(1); } console.log(`embedded_lm_text=${text}`);' \
  "$joint_out/lm-sample-bael/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded_lm" || row.text_prior_order !== 6 || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== false) { console.error(`sample did not use embedded text lm: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_lm_contexts=${row.text_prior_contexts}`);' \
  "$joint_out/lm-sample-bael/sample.json"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); const repeated=/(.)\1{5,}/.test(text); console.log(`raw_attention_probe_text=${text}`); if (repeated) { console.log("raw_attention_probe_status=weak-repeat"); }' \
  "$joint_out/raw-sample-bael/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "none" || row.text_prefix !== "Solomon selects " || row.text_prior_boost_q8 !== 0 || row.text_prior_strict !== false) { console.error(`raw sample did not disable decode priors: ${JSON.stringify(row)}`); process.exit(1); } console.log(`raw_attention_probe_tokens=${row.generated_token_count}`);' \
  "$joint_out/raw-sample-bael/sample.json"
