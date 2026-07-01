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
joint_max_windows="${NSRL_SOLOMON_ATTENTION_JOINT_MAX_WINDOWS:-512}"
joint_target_phase="${NSRL_SOLOMON_ATTENTION_JOINT_TARGET_PHASE:-all}"
target_segment="${NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT:-all}"
min_text_accuracy="${NSRL_SOLOMON_ATTENTION_MIN_TEXT_ACCURACY_PER_MILLE:-100}"
max_text_chars="${NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS:-220}"
prompt_profile="${NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE:-all}"
text_token_profile="${NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE:-char}"
joint_text_only_repeats="${NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS:-0}"
name_initial_repeats="${NSRL_SOLOMON_ATTENTION_NAME_INITIAL_REPEATS:-0}"
name_opening_repeats="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS:-0}"
name_opening_pretrain="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_PRETRAIN:-0}"
name_copy_init="${NSRL_SOLOMON_ATTENTION_NAME_COPY_INIT:-0}"
name_copy_repair="${NSRL_SOLOMON_ATTENTION_NAME_COPY_REPAIR:-0}"
name_copy_repair_preserve_body_output="${NSRL_SOLOMON_ATTENTION_NAME_COPY_REPAIR_PRESERVE_BODY_OUTPUT:-0}"
body_scaffold="${NSRL_SOLOMON_ATTENTION_BODY_SCAFFOLD:-0}"
body_opening_repair="${NSRL_SOLOMON_ATTENTION_BODY_OPENING_REPAIR:-0}"
learning_rate="${NSRL_SOLOMON_ATTENTION_LEARNING_RATE:-1}"
output_lr_shift="${NSRL_SOLOMON_ATTENTION_OUTPUT_LR_SHIFT:-18}"
mlp_lr_shift="${NSRL_SOLOMON_ATTENTION_MLP_LR_SHIFT:-16}"
embed_lr_shift="${NSRL_SOLOMON_ATTENTION_EMBED_LR_SHIFT:-14}"
attention_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_LR_SHIFT:-24}"
attention_q_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_Q_LR_SHIFT:-18}"
attention_qk_lr_shift="${NSRL_SOLOMON_ATTENTION_ATTENTION_QK_LR_SHIFT:-18}"
target_frequency_cap="${NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_CAP:-0}"
target_frequency_min_weight_q15="${NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_MIN_WEIGHT_Q15:-4096}"
argmax_margin_weight_q15="${NSRL_SOLOMON_ATTENTION_ARGMAX_MARGIN_WEIGHT_Q15:-0}"
joint_learning_rate="${NSRL_SOLOMON_ATTENTION_JOINT_LEARNING_RATE:-2}"
joint_output_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_OUTPUT_LR_SHIFT:-20}"
joint_mlp_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_MLP_LR_SHIFT:-18}"
joint_embed_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_EMBED_LR_SHIFT:-16}"
joint_attention_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_ATTENTION_LR_SHIFT:-26}"
joint_attention_q_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_ATTENTION_Q_LR_SHIFT:-20}"
joint_attention_qk_lr_shift="${NSRL_SOLOMON_ATTENTION_JOINT_ATTENTION_QK_LR_SHIFT:-20}"
joint_target_frequency_cap="${NSRL_SOLOMON_ATTENTION_JOINT_TARGET_FREQUENCY_CAP:-$target_frequency_cap}"
joint_target_frequency_min_weight_q15="${NSRL_SOLOMON_ATTENTION_JOINT_TARGET_FREQUENCY_MIN_WEIGHT_Q15:-$target_frequency_min_weight_q15}"
joint_argmax_margin_weight_q15="${NSRL_SOLOMON_ATTENTION_JOINT_ARGMAX_MARGIN_WEIGHT_Q15:-$argmax_margin_weight_q15}"
joint_target_segment="${NSRL_SOLOMON_ATTENTION_JOINT_TARGET_SEGMENT:-$target_segment}"
reject_loss_regression="${NSRL_SOLOMON_ATTENTION_REJECT_LOSS_REGRESSION:-0}"
embedded_text_lm_order="${NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_ORDER:-12}"
embedded_text_lm_min_order="${NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_MIN_ORDER:-3}"
embedded_text_lm_strict="${NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_STRICT:-1}"
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
  --target-frequency-cap "$target_frequency_cap"
  --target-frequency-min-weight-q15 "$target_frequency_min_weight_q15"
  --argmax-margin-weight-q15 "$argmax_margin_weight_q15"
  --target-segment "$target_segment"
)
if [[ "$reject_loss_regression" != "0" ]]; then
  train_lr_args+=(--reject-loss-regression)
fi
if [[ "$name_copy_init" != "0" ]]; then
  train_lr_args+=(--solomon-name-copy-init)
fi
if [[ "$name_copy_repair" != "0" ]]; then
  train_lr_args+=(--solomon-name-copy-repair)
fi
if [[ "$name_copy_repair_preserve_body_output" != "0" ]]; then
  train_lr_args+=(--solomon-name-copy-repair-preserve-body-output)
fi
if [[ "$body_scaffold" != "0" ]]; then
  train_lr_args+=(--solomon-body-scaffold)
fi
if [[ "$body_opening_repair" != "0" ]]; then
  train_lr_args+=(--solomon-body-opening-repair)
fi
joint_train_lr_args=(
  --learning-rate "$joint_learning_rate"
  --output-lr-shift "$joint_output_lr_shift"
  --mlp-lr-shift "$joint_mlp_lr_shift"
  --embed-lr-shift "$joint_embed_lr_shift"
  --attention-lr-shift "$joint_attention_lr_shift"
  --attention-q-lr-shift "$joint_attention_q_lr_shift"
  --attention-qk-lr-shift "$joint_attention_qk_lr_shift"
  --target-frequency-cap "$joint_target_frequency_cap"
  --target-frequency-min-weight-q15 "$joint_target_frequency_min_weight_q15"
  --argmax-margin-weight-q15 "$joint_argmax_margin_weight_q15"
  --target-segment "$joint_target_segment"
)
if [[ "$reject_loss_regression" != "0" ]]; then
  joint_train_lr_args+=(--reject-loss-regression)
fi
if [[ "$name_copy_init" != "0" ]]; then
  joint_train_lr_args+=(--solomon-name-copy-init)
fi
if [[ "$name_copy_repair" != "0" ]]; then
  joint_train_lr_args+=(--solomon-name-copy-repair)
fi
if [[ "$name_copy_repair_preserve_body_output" != "0" ]]; then
  joint_train_lr_args+=(--solomon-name-copy-repair-preserve-body-output)
fi
if [[ "$body_scaffold" != "0" ]]; then
  joint_train_lr_args+=(--solomon-body-scaffold)
fi
if [[ "$body_opening_repair" != "0" ]]; then
  joint_train_lr_args+=(--solomon-body-opening-repair)
fi
joint_target_args=()
if [[ "$joint_target_phase" != "all" ]]; then
  joint_target_args=(--target-phase "$joint_target_phase")
fi
embedded_lm_args=(
  --embedded-text-lm-order "$embedded_text_lm_order"
  --text-prior-min-order "$embedded_text_lm_min_order"
)
if [[ "$embedded_text_lm_strict" != "0" ]]; then
  embedded_lm_args+=(--text-prior-strict)
fi

check_attention_train_loss() {
  node -e 'const fs=require("fs"); const row=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); if (!(row.final_probability_error_q15 <= row.initial_probability_error_q15)) { console.error(`attention train loss increased in ${process.argv[1]}: ${row.initial_probability_error_q15} -> ${row.final_probability_error_q15}`); process.exit(1); } console.log(`attention_train_loss_delta=${row.probability_error_delta_i64}`);' \
    "$1"
}

check_attention_train_updated() {
  node -e 'const fs=require("fs"); const row=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); if (!(row.updates > 0 && row.accepted_batches > 0)) { console.error(`attention train accepted no updates in ${process.argv[1]}: updates=${row.updates} accepted_batches=${row.accepted_batches}`); process.exit(1); } console.log(`attention_train_updates=${row.updates} accepted_batches=${row.accepted_batches}`);' \
    "$1"
}

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
    --name-initial-repeats "$name_initial_repeats" \
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
  check_attention_train_loss "$opening_out/train.json"
  text_init_args=(--init-model "$opening_model")
fi

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$text_out" \
  --max-text-chars "$max_text_chars" \
  --prompt-profile "$prompt_profile" \
  --pad-context "$seq_len" \
  --sequence-profile text-only \
  --name-initial-repeats "$name_initial_repeats" \
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
  check_attention_train_loss "$text_out/train-offset-$offset.json"
  text_init_args=(--init-model "$text_model")
done

node scripts/build-solomon-multimodal-corpus.mjs \
  --text-index "$text_index" \
  --out-dir "$joint_out" \
  --max-text-chars "$max_text_chars" \
  --prompt-profile "$prompt_profile" \
  --pad-context "$seq_len" \
  --text-only-repeats "$joint_text_only_repeats" \
  --name-initial-repeats "$name_initial_repeats" \
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
    ${joint_target_args[@]+"${joint_target_args[@]}"} \
    "${joint_train_lr_args[@]}" \
    > "$joint_out/train-offset-$offset.json"
  check_attention_train_loss "$joint_out/train-offset-$offset.json"
  check_attention_train_updated "$joint_out/train-offset-$offset.json"
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
  "${embedded_lm_args[@]}" \
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

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- sample \
  --model "$joint_model" \
  --tokens "$joint_out/corpus.tokens.u8" \
  --conditioning-examples none \
  --text-prior-examples none \
  --no-embedded-text-memory \
  --out-dir "$joint_out/opening-sample-bael" \
  --prompt "seal of Bael" \
  --text-prefix "Solomon selects " \
  --min-text-tokens 32 \
  --max-text-tokens 96 \
  --repeat-run-cap 4 \
  --no-repeat-ngram 4 \
  --top-k 1 \
  --sample-seed 13 \
  --prompt-name-opening-prior \
  > "$joint_out/opening-sample.json"

node -e 'const fs=require("fs"); const path=process.argv[1]; const min=Number(process.argv[2]); const row=JSON.parse(fs.readFileSync(path,"utf8")); const got=row.text.accuracy_per_mille; if (!(got >= min)) { console.error(`text accuracy ${got} < ${min}`); process.exit(1); } console.log(`Solomon attention curriculum smoke wrote ${process.argv[3]} text_accuracy_per_mille=${got}`);' \
  "$joint_out/attention-eval.json" \
  "$min_text_accuracy" \
  "$joint_out"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); if (!text.startsWith("Solomon selects Bael: ") || !text.includes(".") || /Solomon selects (?!Bael:)/.test(text)) { console.error(`weak prior-assisted text: ${text}`); process.exit(1); } console.log(`prior_assisted_text=${text}`);' \
  "$joint_out/prior-sample-bael/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded" || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== true || row.image_prior_source !== "embedded" || row.image_prior_tokens !== 256) { console.error(`sample did not use embedded text/image memory: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_text_memory_contexts=${row.text_prior_contexts} embedded_image_tokens=${row.image_prior_tokens}`);' \
  "$joint_out/prior-sample-bael/sample.json"

node scripts/check-solomon-attention-web-quality.mjs \
  --model "$joint_model" \
  --text-index "$text_index" \
  --all-names \
  --summary

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); const hasClause=text.includes(": He ") || text.includes(" and "); if (!text.startsWith("Solomon selects Bael: ") || !hasClause || /aaa|eee|hhh/.test(text)) { console.error(`weak embedded-lm text: ${text}`); process.exit(1); } console.log(`embedded_lm_text=${text}`);' \
  "$joint_out/lm-sample-bael/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const expectedOrder=Number(process.argv[2]); const expectedMinOrder=Number(process.argv[3]); const expectedStrict=process.argv[4] !== "0"; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded_lm" || row.text_prior_order !== expectedOrder || row.text_prior_min_order !== expectedMinOrder || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== expectedStrict || row.image_prior_source !== "none") { console.error(`sample did not use embedded text lm: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_lm_contexts=${row.text_prior_contexts}`);' \
  "$joint_out/lm-sample-bael/sample.json" \
  "$embedded_text_lm_order" \
  "$embedded_text_lm_min_order" \
  "$embedded_text_lm_strict"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); const repeated=/(.)\1{5,}/.test(text); console.log(`raw_attention_probe_text=${text}`); if (repeated) { console.log("raw_attention_probe_status=weak-repeat"); }' \
  "$joint_out/raw-sample-bael/text.txt"
node scripts/check-solomon-attention-raw-quality.mjs \
  --text "$joint_out/raw-sample-bael/text.txt" \
  --prompt "seal of Bael" \
  --label raw_attention_probe

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "none" || row.image_prior_source !== "none" || row.text_prefix !== "Solomon selects " || row.text_prior_boost_q8 !== 0 || row.text_prior_strict !== false) { console.error(`raw sample did not disable decode priors: ${JSON.stringify(row)}`); process.exit(1); } console.log(`raw_attention_probe_tokens=${row.generated_token_count}`);' \
  "$joint_out/raw-sample-bael/sample.json"

node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); if (!text.startsWith("Solomon selects Bael: He")) { console.error(`weak prompt-name opening text: ${text}`); process.exit(1); } console.log(`prompt_name_opening_text=${text}`);' \
  "$joint_out/opening-sample-bael/text.txt"
node scripts/check-solomon-attention-raw-quality.mjs \
  --text "$joint_out/opening-sample-bael/text.txt" \
  --prompt "seal of Bael" \
  --label prompt_name_opening

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "none" || row.image_prior_source !== "none" || row.text_prefix !== "Solomon selects " || row.prompt_name_opening_prior !== true) { console.error(`opening sample did not isolate prompt-name prior: ${JSON.stringify(row)}`); process.exit(1); } console.log(`prompt_name_opening_tokens=${row.generated_token_count}`);' \
  "$joint_out/opening-sample-bael/sample.json"
