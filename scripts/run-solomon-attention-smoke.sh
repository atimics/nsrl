#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_SOLOMON_ATTENTION_OUT_DIR:-data/processed/key-solomon-goetia-attention-v1}"
text_index="${NSRL_SOLOMON_ATTENTION_TEXT_INDEX:-web/assets/solomon-spirit-text-signatures.tsv}"
heldout_prompts="${NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS:-data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl}"
model="${NSRL_SOLOMON_ATTENTION_MODEL:-$out_dir/model.nsrllmm}"
epochs="${NSRL_SOLOMON_ATTENTION_EPOCHS:-1}"
seq_len="${NSRL_SOLOMON_ATTENTION_SEQ_LEN:-32}"
stride="${NSRL_SOLOMON_ATTENTION_STRIDE:-1}"
window_offset="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET:-0}"
max_windows="${NSRL_SOLOMON_ATTENTION_MAX_WINDOWS:-256}"
batch_windows="${NSRL_SOLOMON_ATTENTION_BATCH_WINDOWS:-8}"
batch_mode="${NSRL_SOLOMON_ATTENTION_BATCH_MODE:-serial}"
map_reduce_workers="${NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS:-1}"
target_segment="${NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT:-all}"
prompt_profile="${NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE:-all}"
text_token_profile="${NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE:-char}"
corpus_version="${NSRL_SOLOMON_ATTENTION_CORPUS_VERSION:-v1}"
image_token_profile="${NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE:-}"
eval_max_examples="${NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES:-}"
eval_max_targets_per_task_phase="${NSRL_SOLOMON_ATTENTION_EVAL_MAX_TARGETS_PER_TASK_PHASE:-}"
attention_denoiser_model="${NSRL_SOLOMON_ATTENTION_DENOISER_MODEL:-}"
attention_denoise_samples="${NSRL_SOLOMON_ATTENTION_DENOISE_SAMPLES:-1}"
attention_denoise_candidate_multiplier="${NSRL_SOLOMON_ATTENTION_DENOISE_CANDIDATE_MULTIPLIER:-1}"
attention_denoise_passes="${NSRL_SOLOMON_ATTENTION_DENOISE_PASSES:-1}"
attention_denoise_max_output_signature_distance="${NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_SIGNATURE_DISTANCE:-}"
attention_denoise_min_output_ink_range="${NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_INK_RANGE:-1}"
attention_denoise_max_output_retrieval_rank="${NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK:-1}"
attention_denoise_min_output_retrieval_margin="${NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN:-1}"
attention_denoise_min_unique_targets="${NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS:-0}"
quality_min_total_top5="${NSRL_SOLOMON_V2_MIN_TOTAL_TOP5_PER_MILLE:-0}"
quality_min_text_top5="${NSRL_SOLOMON_V2_MIN_TEXT_TOP5_PER_MILLE:-0}"
quality_min_image_top5="${NSRL_SOLOMON_V2_MIN_IMAGE_TOP5_PER_MILLE:-0}"
quality_min_task_targets="${NSRL_SOLOMON_V2_MIN_TASK_TARGETS:-}"
quality_min_task_top5="${NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE:-}"
quality_min_phase_targets="${NSRL_SOLOMON_V2_MIN_PHASE_TARGETS:-}"
quality_min_direction_accuracy="${NSRL_SOLOMON_V2_MIN_DIRECTION_ACCURACY_PER_MILLE:-}"
quality_min_direction_top5="${NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE:-}"
quality_min_direction_top10="${NSRL_SOLOMON_V2_MIN_DIRECTION_TOP10_PER_MILLE:-}"
quality_require_heldout_prompts="${NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS:-1}"
quality_min_heldout_prompt_rows="${NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS:-72}"
quality_min_match_yes_top1="${NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1:-72}"
quality_min_match_no_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1:-72}"
quality_min_match_no_image_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1:-72}"
quality_min_match_no_prompt_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1:-72}"
quality_min_retrieval_margin="${NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN:-1}"
quality_require_architecture="${NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE:-1}"
quality_min_d_model="${NSRL_SOLOMON_V2_MIN_D_MODEL:-0}"
quality_min_heads="${NSRL_SOLOMON_V2_MIN_HEADS:-0}"
quality_min_hidden_dim="${NSRL_SOLOMON_V2_MIN_HIDDEN_DIM:-0}"
quality_min_layers="${NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS:-0}"
quality_min_context_seq_len="${NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN:-0}"
quality_require_promoted_small_profile="${NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE:-0}"
quality_require_image_token_profile="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE:-symbolic16}"
quality_require_image_token_channels="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS:-ink,edge,component,radial,direction}"
quality_require_image_channel_token_stats="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS:-1}"
quality_min_image_channel_distinct_bins="${NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS:-2}"
quality_require_directional_groups="${NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS:-1}"
quality_require_identity_inference="${NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE:-1}"
quality_require_curriculum_stages="${NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES:-0}"
quality_require_denoise_bridge="${NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE:-0}"
quality_require_denoise_output_identity="${NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY:-$quality_require_denoise_bridge}"
quality_min_denoise_bridge_unique_targets="${NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS:-$attention_denoise_min_unique_targets}"
quality_require_grounded_corpus="${NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS:-0}"
quality_min_grounded_source_overlap="${NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS:-2}"
quality_min_grounded_attribute_source_overlap="${NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS:-8}"
quality_max_grounded_source_placeholder_rows="${NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS:-0}"
quality_max_grounded_attribute_generic_rank_rows="${NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS:-0}"
quality_require_confidence_trace="${NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE:-1}"
quality_generative_eval="${NSRL_SOLOMON_V2_GENERATIVE_EVAL:-}"
quality_require_generative_eval="${NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL:-0}"
quality_require_generative_output_identity="${NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY:-$quality_require_generative_eval}"
quality_min_generated_top5="${NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_PER_MILLE:-0}"
quality_min_generated_top5_16="${NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE:-0}"
quality_min_generated_top5_px="${NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_PX_PER_MILLE:-0}"
quality_min_generated_retrieval_top1="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE:-0}"
quality_min_generated_retrieval_top5="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE:-0}"
quality_min_generated_retrieval_margin="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN:-0}"
quality_min_generated_prompt_rows="${NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS:-0}"
quality_min_latent_top5="${NSRL_SOLOMON_V2_MIN_LATENT_TOP5_PER_MILLE:-0}"
quality_max_generated_mean_rank="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_RANK_Q8:-0}"
quality_max_generated_mean_rank_16="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_RANK_16_Q8:-0}"
quality_max_generated_mean_rank_px="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_RANK_PX_Q8:-0}"
quality_max_generated_mean_target_distance="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_Q8:-0}"
quality_max_generated_mean_target_distance_16="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8:-0}"
quality_max_generated_mean_target_distance_px="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_PX_Q8:-0}"
text_only_repeats="${NSRL_SOLOMON_ATTENTION_TEXT_ONLY_REPEATS:-0}"
name_initial_repeats="${NSRL_SOLOMON_ATTENTION_NAME_INITIAL_REPEATS:-0}"
name_opening_repeats="${NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS:-0}"
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
reject_loss_regression="${NSRL_SOLOMON_ATTENTION_REJECT_LOSS_REGRESSION:-0}"

heldout_prompt_gate_args=(
  --prompts "$heldout_prompts"
  --min-heldout-prompt-rows "$quality_min_heldout_prompt_rows"
  --min-match-yes-top1 "$quality_min_match_yes_top1"
  --min-match-no-top1 "$quality_min_match_no_top1"
  --min-match-no-image-top1 "$quality_min_match_no_image_top1"
  --min-match-no-prompt-top1 "$quality_min_match_no_prompt_top1"
)
if [[ "$quality_require_heldout_prompts" != "0" ]]; then
  heldout_prompt_gate_args+=(--require-heldout-prompts)
fi

denoise_bridge_gate_args=(
  --min-output-ink-range "$attention_denoise_min_output_ink_range"
  --max-output-retrieval-rank "$attention_denoise_max_output_retrieval_rank"
  --min-output-retrieval-margin "$attention_denoise_min_output_retrieval_margin"
  --min-unique-targets "$attention_denoise_min_unique_targets"
)
if [[ -n "$attention_denoise_max_output_signature_distance" ]]; then
  denoise_bridge_gate_args+=(--max-output-signature-distance "$attention_denoise_max_output_signature_distance")
fi
task_eval_gate_args=()
if [[ -n "$quality_require_image_token_profile" ]]; then
  task_eval_gate_args+=(--require-image-token-profile "$quality_require_image_token_profile")
fi
if [[ -n "$quality_require_image_token_channels" ]]; then
  task_eval_gate_args+=(--require-image-token-channels "$quality_require_image_token_channels")
fi
if [[ "$quality_require_image_channel_token_stats" != "0" ]]; then
  task_eval_gate_args+=(
    --require-image-channel-token-stats
    --min-image-channel-distinct-bins "$quality_min_image_channel_distinct_bins"
  )
fi
if [[ -n "$quality_min_task_targets" ]]; then
  task_eval_gate_args+=(--min-task-targets "$quality_min_task_targets")
fi
if [[ -n "$quality_min_phase_targets" && "$quality_min_phase_targets" != "0" ]]; then
  task_eval_gate_args+=(--min-phase-targets "$quality_min_phase_targets")
fi
if [[ -n "$quality_min_direction_accuracy" && "$quality_min_direction_accuracy" != "0" ]]; then
  task_eval_gate_args+=(--min-direction-accuracy "$quality_min_direction_accuracy")
fi
if [[ -n "$quality_min_direction_top5" && "$quality_min_direction_top5" != "0" ]]; then
  task_eval_gate_args+=(--min-direction-top5 "$quality_min_direction_top5")
fi
if [[ -n "$quality_min_direction_top10" && "$quality_min_direction_top10" != "0" ]]; then
  task_eval_gate_args+=(--min-direction-top10 "$quality_min_direction_top10")
fi
if [[ "$quality_require_directional_groups" != "0" ]]; then
  task_eval_gate_args+=(--require-directional-groups)
fi

score_generative_eval_retrieval() {
  local eval_path="$1"
  local retrieval_head="$2"
  local eval_dir="$eval_path"
  if [[ -z "$eval_path" || ! -f "$retrieval_head" ]]; then
    return 0
  fi
  if [[ "$eval_path" == */summary.tsv ]]; then
    eval_dir="${eval_path%/summary.tsv}"
  elif [[ "$eval_path" == "summary.tsv" ]]; then
    eval_dir="."
  fi
  eval_dir="${eval_dir%/}"
  if [[ -z "$eval_dir" ]]; then
    eval_dir="."
  fi
  if [[ -f "$eval_dir/summary.tsv" && -f "$eval_dir/samples.tsv" ]]; then
    node scripts/score-solomon-generative-eval-retrieval.mjs \
      --generative-eval "$eval_dir" \
      --retrieval-head "$retrieval_head"
  fi
}

if [[ -z "$eval_max_examples" ]]; then
  if [[ "$corpus_version" == "v2" ]]; then
    eval_max_examples=32
  else
    eval_max_examples=4
  fi
fi
if [[ -z "$eval_max_targets_per_task_phase" && "$corpus_version" == "v2" ]]; then
  eval_max_targets_per_task_phase=4
fi
if [[ -z "$quality_min_phase_targets" && "$corpus_version" == "v2" ]]; then
  quality_min_phase_targets="special=1,prompt=1,text=1,image=1"
  task_eval_gate_args+=(--min-phase-targets "$quality_min_phase_targets")
fi

build_args=(
  --text-index "$text_index"
  --out-dir "$out_dir"
  --prompt-profile "$prompt_profile"
  --pad-context "$seq_len"
  --text-only-repeats "$text_only_repeats"
  --name-initial-repeats "$name_initial_repeats"
  --name-opening-repeats "$name_opening_repeats"
  --text-token-profile "$text_token_profile"
  --corpus-version "$corpus_version"
)
if [[ -n "$image_token_profile" ]]; then
  build_args+=(--image-token-profile "$image_token_profile")
fi

node scripts/build-solomon-multimodal-corpus.mjs "${build_args[@]}"

train_args=(
  train
  --tokens "$out_dir/corpus.tokens.u8"
  --conditioning-examples "$out_dir/examples.jsonl"
  --model-out "$model"
  --epochs "$epochs"
  --seq-len "$seq_len"
  --stride "$stride"
  --window-offset "$window_offset"
  --batch-windows "$batch_windows"
  --batch-mode "$batch_mode"
  --map-reduce-workers "$map_reduce_workers"
  --text-token-profile "$text_token_profile"
  --target-segment "$target_segment"
  --learning-rate "$learning_rate"
  --output-lr-shift "$output_lr_shift"
  --mlp-lr-shift "$mlp_lr_shift"
  --embed-lr-shift "$embed_lr_shift"
  --attention-lr-shift "$attention_lr_shift"
  --attention-q-lr-shift "$attention_q_lr_shift"
  --attention-qk-lr-shift "$attention_qk_lr_shift"
)
if [[ "$reject_loss_regression" != "0" ]]; then
  train_args+=(--reject-loss-regression)
fi
if [[ "$name_copy_init" != "0" ]]; then
  train_args+=(--solomon-name-copy-init)
fi
if [[ "$name_copy_repair" != "0" ]]; then
  train_args+=(--solomon-name-copy-repair)
fi
if [[ "$name_copy_repair_preserve_body_output" != "0" ]]; then
  train_args+=(--solomon-name-copy-repair-preserve-body-output)
fi
if [[ "$body_scaffold" != "0" ]]; then
  train_args+=(--solomon-body-scaffold)
fi
if [[ "$body_opening_repair" != "0" ]]; then
  train_args+=(--solomon-body-opening-repair)
fi
if [[ "$max_windows" == "none" ]]; then
  train_args+=(--max-windows none)
else
  train_args+=(--max-windows "$max_windows")
fi

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- "${train_args[@]}" \
  > "$out_dir/train.json"

node -e 'const fs=require("fs"); const row=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); if (!(row.final_probability_error_q15 <= row.initial_probability_error_q15)) { console.error(`attention train loss increased: ${row.initial_probability_error_q15} -> ${row.final_probability_error_q15}`); process.exit(1); } console.log(`attention_train_loss_delta=${row.probability_error_delta_i64}`);' \
  "$out_dir/train.json"

eval_args=(
  eval
  --model "$model"
  --tokens "$out_dir/corpus.tokens.u8"
  --conditioning-examples "$out_dir/examples.jsonl"
  --eval-max-examples "$eval_max_examples"
)
if [[ -n "$eval_max_targets_per_task_phase" && "$eval_max_targets_per_task_phase" != "none" ]]; then
  eval_args+=(--eval-max-targets-per-task-phase "$eval_max_targets_per_task_phase")
fi

cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- "${eval_args[@]}" \
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
if [[ "$corpus_version" == "v2" ]]; then
  node scripts/check-solomon-attention-task-eval.mjs \
    --eval "$out_dir/attention-eval.json" \
    --examples "$out_dir/examples.jsonl" \
    --manifest "$out_dir/manifest.json" \
    --require-corpus-version v2 \
    "${task_eval_gate_args[@]}"
  grounded_corpus_path="$out_dir/grounded-corpus.json"
  node scripts/check-solomon-v2-grounded-corpus.mjs \
    --examples "$out_dir/examples.jsonl" \
    --text-index "$text_index" \
    --min-source-overlap-tokens "$quality_min_grounded_source_overlap" \
    --min-attribute-source-overlap-tokens "$quality_min_grounded_attribute_source_overlap" \
    --max-source-placeholder-rows "$quality_max_grounded_source_placeholder_rows" \
    --max-attribute-generic-rank-rows "$quality_max_grounded_attribute_generic_rank_rows" \
    --out "$grounded_corpus_path"
  node scripts/check-solomon-v2-retrieval-spine.mjs \
    --examples "$out_dir/examples.jsonl" \
    --tokens "$out_dir/corpus.tokens.u8" \
    --text-index "$text_index" \
    "${heldout_prompt_gate_args[@]}"
  node scripts/train-solomon-v2-retrieval-head.mjs \
    --examples "$out_dir/examples.jsonl" \
    --tokens "$out_dir/corpus.tokens.u8" \
    --text-index "$text_index" \
    "${heldout_prompt_gate_args[@]}" \
    --min-retrieval-margin "$quality_min_retrieval_margin" \
    --model-out "$out_dir/retrieval-head.json" \
    --eval-out "$out_dir/retrieval-head-eval.json"
  node scripts/check-solomon-attention-sample-binding.mjs \
    --sample-dir "$out_dir/attention-sample-bael" \
    --sample-dir "$out_dir/attention-sample-stolas" \
    --text-index "$text_index" \
    --retrieval-head "$out_dir/retrieval-head.json" \
    --require-retrieval-head \
    --out "$out_dir/sample-binding.json"
  node scripts/infer-solomon-v2-identity.mjs \
    --retrieval-head "$out_dir/retrieval-head.json" \
    --text-index "$text_index" \
    --text "seal of Bael" \
    --text "seal of Stolas" \
    --sample-dir "$out_dir/attention-sample-bael" \
    --sample-dir "$out_dir/attention-sample-stolas" \
    --require-sample-agreement \
    --require-source-evidence \
    --out "$out_dir/identity-inference.json"
fi

node scripts/check-solomon-generation-integrity.mjs \
  --sample-dir "$out_dir/attention-sample-bael" \
  --sample-dir "$out_dir/attention-sample-stolas" \
  --out "$out_dir/generation-integrity.json"

denoise_bridge_path=""
if [[ -n "$attention_denoiser_model" ]]; then
  cargo run --quiet -p nsrl-train --bin nsrl-bitmap-sample -- \
    --model "$attention_denoiser_model" \
    --out-dir "$out_dir/denoise-from-attention-bael" \
    --samples "$attention_denoise_samples" \
    --candidate-multiplier "$attention_denoise_candidate_multiplier" \
    --passes "$attention_denoise_passes" \
    --preview-columns "$attention_denoise_samples" \
    --attention-plan "$out_dir/attention-sample-bael/image.ink16.u8" \
    --prompt "seal of Bael" \
    --seed "solomon-attention-plan-bael" \
    --workers 1
  cargo run --quiet -p nsrl-train --bin nsrl-bitmap-sample -- \
    --model "$attention_denoiser_model" \
    --out-dir "$out_dir/denoise-from-attention-stolas" \
    --samples "$attention_denoise_samples" \
    --candidate-multiplier "$attention_denoise_candidate_multiplier" \
    --passes "$attention_denoise_passes" \
    --preview-columns "$attention_denoise_samples" \
    --attention-plan "$out_dir/attention-sample-stolas/image.ink16.u8" \
    --prompt "seal of Stolas" \
    --seed "solomon-attention-plan-stolas" \
    --workers 1
  denoise_bridge_path="$out_dir/denoise-bridge.json"
  denoise_bridge_identity_args=()
  if [[ "$corpus_version" == "v2" && -f "$out_dir/retrieval-head.json" ]]; then
    denoise_bridge_identity_args=(
      --text-index "$text_index"
      --retrieval-head "$out_dir/retrieval-head.json"
      --require-retrieval-head
    )
  fi
  node scripts/check-solomon-attention-denoise-bridge.mjs \
    --pair "$out_dir/attention-sample-bael:$out_dir/denoise-from-attention-bael" \
    --pair "$out_dir/attention-sample-stolas:$out_dir/denoise-from-attention-stolas" \
    "${denoise_bridge_gate_args[@]}" \
    "${denoise_bridge_identity_args[@]}" \
    --out "$denoise_bridge_path"
  node scripts/check-solomon-generation-integrity.mjs \
    --sample-dir "$out_dir/denoise-from-attention-bael" \
    --sample-dir "$out_dir/denoise-from-attention-stolas" \
    --expected-latent-target-source attention-plan \
    --out "$out_dir/denoise-generation-integrity.json"
fi

if [[ "$corpus_version" == "v2" ]]; then
  score_generative_eval_retrieval "$quality_generative_eval" "$out_dir/retrieval-head.json"

  quality_report_args=(
    --eval "$out_dir/attention-eval.json"
    --examples "$out_dir/examples.jsonl"
    --manifest "$out_dir/manifest.json"
    --retrieval-head "$out_dir/retrieval-head.json"
    --retrieval-head-eval "$out_dir/retrieval-head-eval.json"
    --sample-binding "$out_dir/sample-binding.json"
    --generation-integrity "$out_dir/generation-integrity.json"
    --identity-inference "$out_dir/identity-inference.json"
    --min-total-top5-per-mille "$quality_min_total_top5"
    --min-text-top5-per-mille "$quality_min_text_top5"
    --min-image-top5-per-mille "$quality_min_image_top5"
    --min-heldout-prompt-rows "$quality_min_heldout_prompt_rows"
    --require-corpus-version v2
    --require-image-token-profile "$quality_require_image_token_profile"
    --require-image-token-channels "$quality_require_image_token_channels"
    --min-image-channel-distinct-bins "$quality_min_image_channel_distinct_bins"
    --min-match-yes-top1 "$quality_min_match_yes_top1"
    --min-match-no-top1 "$quality_min_match_no_top1"
    --min-match-no-image-top1 "$quality_min_match_no_image_top1"
    --min-match-no-prompt-top1 "$quality_min_match_no_prompt_top1"
    --min-retrieval-margin "$quality_min_retrieval_margin"
    --min-grounded-source-overlap-tokens "$quality_min_grounded_source_overlap"
    --min-grounded-attribute-source-overlap-tokens "$quality_min_grounded_attribute_source_overlap"
    --max-grounded-source-placeholder-rows "$quality_max_grounded_source_placeholder_rows"
    --max-grounded-attribute-generic-rank-rows "$quality_max_grounded_attribute_generic_rank_rows"
    --min-d-model "$quality_min_d_model"
    --min-heads "$quality_min_heads"
    --min-hidden-dim "$quality_min_hidden_dim"
    --min-transformer-layers "$quality_min_layers"
    --min-context-seq-len "$quality_min_context_seq_len"
    --min-generated-top5-per-mille "$quality_min_generated_top5"
    --min-generated-top5-16-per-mille "$quality_min_generated_top5_16"
    --min-generated-top5-px-per-mille "$quality_min_generated_top5_px"
    --min-generated-retrieval-top1-per-mille "$quality_min_generated_retrieval_top1"
    --min-generated-retrieval-top5-per-mille "$quality_min_generated_retrieval_top5"
    --min-generated-retrieval-margin "$quality_min_generated_retrieval_margin"
    --min-generated-prompt-rows "$quality_min_generated_prompt_rows"
    --min-denoise-bridge-unique-targets "$quality_min_denoise_bridge_unique_targets"
    --min-latent-top5-per-mille "$quality_min_latent_top5"
    --max-generated-mean-rank-q8 "$quality_max_generated_mean_rank"
    --max-generated-mean-rank-16-q8 "$quality_max_generated_mean_rank_16"
    --max-generated-mean-rank-px-q8 "$quality_max_generated_mean_rank_px"
    --max-generated-mean-target-distance-q8 "$quality_max_generated_mean_target_distance"
    --max-generated-mean-target-distance-16-q8 "$quality_max_generated_mean_target_distance_16"
    --max-generated-mean-target-distance-px-q8 "$quality_max_generated_mean_target_distance_px"
    --out "$out_dir/quality-report.json"
  )
  if [[ -n "$quality_min_task_top5" ]]; then
    quality_report_args+=(--min-task-top5-per-mille "$quality_min_task_top5")
  fi
  if [[ -n "$quality_min_task_targets" ]]; then
    quality_report_args+=(--min-task-targets "$quality_min_task_targets")
  fi
  if [[ -n "$quality_min_phase_targets" && "$quality_min_phase_targets" != "0" ]]; then
    quality_report_args+=(--min-phase-targets "$quality_min_phase_targets")
  fi
  if [[ "$quality_require_image_channel_token_stats" != "0" ]]; then
    quality_report_args+=(--require-image-channel-token-stats)
  fi
  if [[ "$quality_require_heldout_prompts" != "0" ]]; then
    quality_report_args+=(--require-heldout-prompts)
  fi
  if [[ "$quality_require_identity_inference" != "0" ]]; then
    quality_report_args+=(--require-identity-inference)
  fi
  if [[ "$quality_require_curriculum_stages" != "0" ]]; then
    quality_report_args+=(--require-curriculum-stages)
  fi
  if [[ "$quality_require_denoise_bridge" != "0" ]]; then
    quality_report_args+=(--require-denoise-bridge)
  fi
  if [[ "$quality_require_denoise_output_identity" != "0" ]]; then
    quality_report_args+=(--require-denoise-output-identity)
  fi
  if [[ "$quality_require_grounded_corpus" != "0" ]]; then
    quality_report_args+=(--require-grounded-corpus)
  fi
  if [[ "$quality_require_confidence_trace" != "0" ]]; then
    quality_report_args+=(--require-confidence-trace)
  fi
  if [[ "$quality_require_generative_eval" != "0" ]]; then
    quality_report_args+=(--require-generative-eval)
  fi
  if [[ "$quality_require_generative_output_identity" != "0" ]]; then
    quality_report_args+=(--require-generative-output-identity)
  fi
  if [[ -n "$denoise_bridge_path" ]]; then
    quality_report_args+=(--denoise-bridge "$denoise_bridge_path")
  fi
  if [[ -n "${grounded_corpus_path:-}" ]]; then
    quality_report_args+=(--grounded-corpus "$grounded_corpus_path")
  fi
  if [[ -n "$quality_generative_eval" ]]; then
    quality_report_args+=(--generative-eval "$quality_generative_eval")
  fi
  if [[ "$quality_require_architecture" != "0" ]]; then
    quality_report_args+=(--require-architecture-profile)
  fi
  if [[ "$quality_require_promoted_small_profile" != "0" ]]; then
    quality_report_args+=(--require-promoted-small-profile)
  fi
  node scripts/check-solomon-v2-quality-report.mjs \
    "${quality_report_args[@]}"
fi

echo "Solomon attention smoke wrote $out_dir"
