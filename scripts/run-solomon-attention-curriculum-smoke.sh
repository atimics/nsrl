#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

joint_out="${NSRL_SOLOMON_ATTENTION_CURRICULUM_OUT_DIR:-data/processed/key-solomon-goetia-attention-curriculum-v1}"
text_out="${NSRL_SOLOMON_ATTENTION_TEXT_PRETRAIN_OUT_DIR:-data/processed/key-solomon-goetia-attention-text-only-v1}"
opening_out="${NSRL_SOLOMON_ATTENTION_OPENING_PRETRAIN_OUT_DIR:-data/processed/key-solomon-goetia-attention-opening-v1}"
text_index="${NSRL_SOLOMON_ATTENTION_TEXT_INDEX:-web/assets/solomon-spirit-text-signatures.tsv}"
heldout_prompts="${NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS:-data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl}"
seq_len="${NSRL_SOLOMON_ATTENTION_SEQ_LEN:-32}"
stride="${NSRL_SOLOMON_ATTENTION_STRIDE:-1}"
window_offset="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET:-0}"
window_offset_sweep="${NSRL_SOLOMON_ATTENTION_WINDOW_OFFSET_SWEEP:-single}"
batch_windows="${NSRL_SOLOMON_ATTENTION_BATCH_WINDOWS:-8}"
batch_mode="${NSRL_SOLOMON_ATTENTION_BATCH_MODE:-serial}"
map_reduce_workers="${NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS:-1}"
text_max_windows="${NSRL_SOLOMON_ATTENTION_TEXT_MAX_WINDOWS:-2048}"
opening_max_windows="${NSRL_SOLOMON_ATTENTION_OPENING_MAX_WINDOWS:-4096}"
joint_max_windows="${NSRL_SOLOMON_ATTENTION_JOINT_MAX_WINDOWS:-512}"
joint_target_phase="${NSRL_SOLOMON_ATTENTION_JOINT_TARGET_PHASE:-all}"
v2_curriculum_stages="${NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES:-}"
v2_curriculum_required_stages="${NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES:-$v2_curriculum_stages}"
v2_stage_max_windows="${NSRL_SOLOMON_ATTENTION_V2_STAGE_MAX_WINDOWS:-$joint_max_windows}"
v2_stage_epochs="${NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS:-1}"
v2_native_bind_epochs="${NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS:-2}"
v2_native_bind_max_windows="${NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_MAX_WINDOWS:-$v2_stage_max_windows}"
target_segment="${NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT:-all}"
min_text_accuracy="${NSRL_SOLOMON_ATTENTION_MIN_TEXT_ACCURACY_PER_MILLE:-100}"
max_text_chars="${NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS:-220}"
prompt_profile="${NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE:-all}"
text_token_profile="${NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE:-char}"
joint_corpus_version="${NSRL_SOLOMON_ATTENTION_JOINT_CORPUS_VERSION:-${NSRL_SOLOMON_ATTENTION_CORPUS_VERSION:-v1}}"
joint_image_token_profile="${NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE:-${NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE:-}}"
joint_eval_max_examples="${NSRL_SOLOMON_ATTENTION_JOINT_EVAL_MAX_EXAMPLES:-${NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES:-8}}"
joint_eval_max_targets_per_task_phase="${NSRL_SOLOMON_ATTENTION_JOINT_EVAL_MAX_TARGETS_PER_TASK_PHASE:-${NSRL_SOLOMON_ATTENTION_EVAL_MAX_TARGETS_PER_TASK_PHASE:-}}"
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
quality_require_curriculum_stages="${NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES:-}"
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

if [[ -z "$quality_require_curriculum_stages" ]]; then
  if [[ -n "$v2_curriculum_stages" ]]; then
    quality_require_curriculum_stages=1
  else
    quality_require_curriculum_stages=0
  fi
fi
if [[ -z "$joint_eval_max_targets_per_task_phase" && "$joint_corpus_version" == "v2" ]]; then
  joint_eval_max_targets_per_task_phase=4
fi
if [[ -z "$quality_min_phase_targets" && "$joint_corpus_version" == "v2" ]]; then
  quality_min_phase_targets="special=1,prompt=1,text=1,image=1"
fi

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

v2_stage_filter_args() {
  local stage="$1"
  case "$stage" in
    identity)
      stage_filter_args=(--tasks "identify,image-to-text,explain")
      ;;
    image)
      stage_filter_args=(--tasks "text-to-image,description-to-image,image-to-text")
      ;;
    text-to-image)
      stage_filter_args=(--tasks "text-to-image,description-to-image")
      ;;
    description-to-image)
      stage_filter_args=(--tasks "description-to-image")
      ;;
    image-to-text)
      stage_filter_args=(--tasks "image-to-text,image-to-explain,text-image-explain,image-to-attributes")
      ;;
    explain)
      stage_filter_args=(--tasks "explain,image-to-explain,text-image-explain,image-to-attributes")
      ;;
    match)
      stage_filter_args=(--tasks match)
      ;;
    hard-negative | hard-negatives)
      stage_filter_args=(--tasks match --match-labels no --match-roles "image,prompt")
      ;;
    native-bind)
      stage_filter_args=(--tasks "canonical-joint,identify,text-to-image,description-to-image,image-to-text,image-to-explain,text-image-explain,image-to-attributes,explain")
      ;;
    all)
      stage_filter_args=(--tasks "canonical-joint,identify,text-to-image,image-to-text,image-to-explain,text-image-explain,image-to-attributes,explain,description-to-image,match")
      ;;
    *)
      echo "unknown v2 curriculum stage: $stage" >&2
      exit 2
      ;;
  esac
}

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
    --conditioning-examples "$opening_out/examples.jsonl" \
    --model-out "$opening_model" \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$window_offset" \
    "${opening_window_args[@]}" \
    --batch-windows "$batch_windows" \
    --batch-mode "$batch_mode" \
    --map-reduce-workers "$map_reduce_workers" \
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
      --conditioning-examples "$text_out/examples.jsonl" \
      ${text_init_args[@]+"${text_init_args[@]}"} \
    --model-out "$text_model" \
    --epochs 1 \
    --seq-len "$seq_len" \
    --stride "$stride" \
    --window-offset "$offset" \
    --max-windows "$text_max_windows" \
    --batch-windows "$batch_windows" \
    --batch-mode "$batch_mode" \
    --map-reduce-workers "$map_reduce_workers" \
    --text-token-profile "$text_token_profile" \
    "${train_lr_args[@]}" \
    > "$text_out/train-offset-$offset.json"
  check_attention_train_loss "$text_out/train-offset-$offset.json"
  text_init_args=(--init-model "$text_model")
done

joint_build_args=(
  --text-index "$text_index"
  --out-dir "$joint_out"
  --max-text-chars "$max_text_chars"
  --prompt-profile "$prompt_profile"
  --pad-context "$seq_len"
  --text-only-repeats "$joint_text_only_repeats"
  --name-initial-repeats "$name_initial_repeats"
  --name-opening-repeats "$name_opening_repeats"
  --text-token-profile "$text_token_profile"
  --corpus-version "$joint_corpus_version"
)
if [[ -n "$joint_image_token_profile" ]]; then
  joint_build_args+=(--image-token-profile "$joint_image_token_profile")
fi

node scripts/build-solomon-multimodal-corpus.mjs "${joint_build_args[@]}"

joint_init_args=(--init-model "$text_model")
if [[ "$joint_corpus_version" == "v2" && -n "$v2_curriculum_stages" ]]; then
  IFS=',' read -r -a v2_stage_names <<< "$v2_curriculum_stages"
  v2_stage_index=0
  v2_stage_init_args=(--init-model "$text_model")
  v2_stage_check_args=()
  for v2_stage_name in "${v2_stage_names[@]}"; do
    v2_stage_name="${v2_stage_name//[[:space:]]/}"
    if [[ -z "$v2_stage_name" ]]; then
      continue
    fi
    v2_stage_filter_args "$v2_stage_name"
    v2_stage_out="$joint_out/v2-stage-$v2_stage_index-$v2_stage_name"
    node scripts/filter-solomon-multimodal-corpus.mjs \
      --input-dir "$joint_out" \
      --out-dir "$v2_stage_out" \
      "${stage_filter_args[@]}"
    v2_stage_check_args+=(--stage-dir "$v2_stage_out")

    stage_epochs="$v2_stage_epochs"
    stage_max_windows="$v2_stage_max_windows"
    if [[ "$v2_stage_name" == "native-bind" ]]; then
      stage_epochs="$v2_native_bind_epochs"
      stage_max_windows="$v2_native_bind_max_windows"
    fi
    v2_stage_window_args=(--max-windows "$stage_max_windows")
    if [[ "$stage_max_windows" == "none" ]]; then
      v2_stage_window_args=(--max-windows none)
    fi

    for offset in "${train_offsets[@]}"; do
      cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- train \
        --tokens "$v2_stage_out/corpus.tokens.u8" \
        --conditioning-examples "$v2_stage_out/examples.jsonl" \
        "${v2_stage_init_args[@]}" \
        --model-out "$joint_model" \
        --embed-text-memory-examples "$joint_out/examples.jsonl" \
        --embed-text-memory-order 32 \
        --epochs "$stage_epochs" \
        --seq-len "$seq_len" \
        --stride "$stride" \
        --window-offset "$offset" \
        "${v2_stage_window_args[@]}" \
        --batch-windows "$batch_windows" \
        --batch-mode "$batch_mode" \
        --map-reduce-workers "$map_reduce_workers" \
        --text-token-profile "$text_token_profile" \
        ${joint_target_args[@]+"${joint_target_args[@]}"} \
        "${joint_train_lr_args[@]}" \
        > "$v2_stage_out/train-offset-$offset.json"
      check_attention_train_loss "$v2_stage_out/train-offset-$offset.json"
      check_attention_train_updated "$v2_stage_out/train-offset-$offset.json"
      cp "$v2_stage_out/train-offset-$offset.json" "$v2_stage_out/train.json"
      v2_stage_init_args=(--init-model "$joint_model")
    done
    v2_stage_index=$((v2_stage_index + 1))
  done
  if (( v2_stage_index == 0 )); then
    echo "NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES selected no stages" >&2
    exit 2
  fi
  node scripts/check-solomon-v2-curriculum-stages.mjs \
    "${v2_stage_check_args[@]}" \
    --min-stages "$v2_stage_index" \
    --min-native-bind-epochs "$v2_native_bind_epochs" \
    --require-stage-names "$v2_curriculum_required_stages" \
    --out "$joint_out/curriculum-stages.json"
  joint_init_args=(--init-model "$joint_model")
fi

for offset in "${train_offsets[@]}"; do
  cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- train \
    --tokens "$joint_out/corpus.tokens.u8" \
    --conditioning-examples "$joint_out/examples.jsonl" \
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
    --batch-mode "$batch_mode" \
    --map-reduce-workers "$map_reduce_workers" \
    --text-token-profile "$text_token_profile" \
    ${joint_target_args[@]+"${joint_target_args[@]}"} \
    "${joint_train_lr_args[@]}" \
    > "$joint_out/train-offset-$offset.json"
  check_attention_train_loss "$joint_out/train-offset-$offset.json"
  check_attention_train_updated "$joint_out/train-offset-$offset.json"
  cp "$joint_out/train-offset-$offset.json" "$joint_out/train.json"
  joint_init_args=(--init-model "$joint_model")
done

joint_eval_args=(
  eval
  --model "$joint_model"
  --tokens "$joint_out/corpus.tokens.u8"
  --conditioning-examples "$joint_out/examples.jsonl"
  --eval-max-examples "$joint_eval_max_examples"
)
if [[ -n "$joint_eval_max_targets_per_task_phase" && "$joint_eval_max_targets_per_task_phase" != "none" ]]; then
  joint_eval_args+=(--eval-max-targets-per-task-phase "$joint_eval_max_targets_per_task_phase")
fi
cargo run --quiet -p nsrl-train --bin nsrl-solomon-attention -- "${joint_eval_args[@]}" \
  > "$joint_out/attention-eval.json"

if [[ "$joint_corpus_version" == "v2" ]]; then
  node scripts/check-solomon-attention-task-eval.mjs \
    --eval "$joint_out/attention-eval.json" \
    --examples "$joint_out/examples.jsonl" \
    --manifest "$joint_out/manifest.json" \
    --require-corpus-version v2 \
    "${task_eval_gate_args[@]}"
  grounded_corpus_path="$joint_out/grounded-corpus.json"
  node scripts/check-solomon-v2-grounded-corpus.mjs \
    --examples "$joint_out/examples.jsonl" \
    --text-index "$text_index" \
    --min-source-overlap-tokens "$quality_min_grounded_source_overlap" \
    --min-attribute-source-overlap-tokens "$quality_min_grounded_attribute_source_overlap" \
    --max-source-placeholder-rows "$quality_max_grounded_source_placeholder_rows" \
    --max-attribute-generic-rank-rows "$quality_max_grounded_attribute_generic_rank_rows" \
    --out "$grounded_corpus_path"
  node scripts/check-solomon-v2-retrieval-spine.mjs \
    --examples "$joint_out/examples.jsonl" \
    --tokens "$joint_out/corpus.tokens.u8" \
    --text-index "$text_index" \
    "${heldout_prompt_gate_args[@]}"
  node scripts/train-solomon-v2-retrieval-head.mjs \
    --examples "$joint_out/examples.jsonl" \
    --tokens "$joint_out/corpus.tokens.u8" \
    --text-index "$text_index" \
    "${heldout_prompt_gate_args[@]}" \
    --min-retrieval-margin "$quality_min_retrieval_margin" \
    --model-out "$joint_out/retrieval-head.json" \
    --eval-out "$joint_out/retrieval-head-eval.json"
fi

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
  --out-dir "$joint_out/prior-sample-stolas" \
  --prompt "seal of Stolas" \
  --min-text-tokens 16 \
  --max-text-tokens 260 \
  --repeat-run-cap 4 \
  --no-repeat-ngram 4 \
  --top-k 1 \
  --sample-seed 17 \
  > "$joint_out/prior-sample-stolas.json"

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
node -e 'const fs=require("fs"); const path=process.argv[1]; const text=fs.readFileSync(path,"utf8").trim(); if (!text.startsWith("Solomon selects Stolas: ") || !text.includes(".") || /Solomon selects (?!Stolas:)/.test(text)) { console.error(`weak prior-assisted text: ${text}`); process.exit(1); } console.log(`prior_assisted_text=${text}`);' \
  "$joint_out/prior-sample-stolas/text.txt"

node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded" || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== true || row.image_prior_source !== "embedded" || row.image_prior_tokens !== 256) { console.error(`sample did not use embedded text/image memory: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_text_memory_contexts=${row.text_prior_contexts} embedded_image_tokens=${row.image_prior_tokens}`);' \
  "$joint_out/prior-sample-bael/sample.json"
node -e 'const fs=require("fs"); const path=process.argv[1]; const row=JSON.parse(fs.readFileSync(path,"utf8")); if (row.conditioning_primary_name || row.text_prior_source !== "embedded" || row.text_prior_contexts <= 0 || row.text_prior_boost_q8 <= 0 || row.text_prior_strict !== true || row.image_prior_source !== "embedded" || row.image_prior_tokens !== 256) { console.error(`sample did not use embedded text/image memory: ${JSON.stringify(row)}`); process.exit(1); } console.log(`embedded_text_memory_contexts=${row.text_prior_contexts} embedded_image_tokens=${row.image_prior_tokens}`);' \
  "$joint_out/prior-sample-stolas/sample.json"
if [[ "$joint_corpus_version" == "v2" ]]; then
  node scripts/check-solomon-attention-sample-binding.mjs \
    --sample-dir "$joint_out/prior-sample-bael" \
    --sample-dir "$joint_out/prior-sample-stolas" \
    --text-index "$text_index" \
    --retrieval-head "$joint_out/retrieval-head.json" \
    --require-retrieval-head \
    --out "$joint_out/prior-sample-binding.json"
  node scripts/infer-solomon-v2-identity.mjs \
    --retrieval-head "$joint_out/retrieval-head.json" \
    --text-index "$text_index" \
    --text "seal of Bael" \
    --text "seal of Stolas" \
    --sample-dir "$joint_out/prior-sample-bael" \
    --sample-dir "$joint_out/prior-sample-stolas" \
    --require-sample-agreement \
    --require-source-evidence \
    --out "$joint_out/identity-inference.json"
fi

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

node scripts/check-solomon-generation-integrity.mjs \
  --sample-dir "$joint_out/prior-sample-bael" \
  --sample-dir "$joint_out/prior-sample-stolas" \
  --sample-dir "$joint_out/lm-sample-bael" \
  --sample-dir "$joint_out/raw-sample-bael" \
  --sample-dir "$joint_out/opening-sample-bael" \
  --out "$joint_out/generation-integrity.json"

denoise_bridge_path=""
if [[ -n "$attention_denoiser_model" ]]; then
  cargo run --quiet -p nsrl-train --bin nsrl-bitmap-sample -- \
    --model "$attention_denoiser_model" \
    --out-dir "$joint_out/denoise-from-prior-bael" \
    --samples "$attention_denoise_samples" \
    --candidate-multiplier "$attention_denoise_candidate_multiplier" \
    --passes "$attention_denoise_passes" \
    --preview-columns "$attention_denoise_samples" \
    --attention-plan "$joint_out/prior-sample-bael/image.ink16.u8" \
    --prompt "seal of Bael" \
    --seed "solomon-curriculum-attention-plan-bael" \
    --workers 1
  cargo run --quiet -p nsrl-train --bin nsrl-bitmap-sample -- \
    --model "$attention_denoiser_model" \
    --out-dir "$joint_out/denoise-from-prior-stolas" \
    --samples "$attention_denoise_samples" \
    --candidate-multiplier "$attention_denoise_candidate_multiplier" \
    --passes "$attention_denoise_passes" \
    --preview-columns "$attention_denoise_samples" \
    --attention-plan "$joint_out/prior-sample-stolas/image.ink16.u8" \
    --prompt "seal of Stolas" \
    --seed "solomon-curriculum-attention-plan-stolas" \
    --workers 1
  denoise_bridge_path="$joint_out/denoise-bridge.json"
  denoise_bridge_identity_args=()
  if [[ "$joint_corpus_version" == "v2" && -f "$joint_out/retrieval-head.json" ]]; then
    denoise_bridge_identity_args=(
      --text-index "$text_index"
      --retrieval-head "$joint_out/retrieval-head.json"
      --require-retrieval-head
    )
  fi
  node scripts/check-solomon-attention-denoise-bridge.mjs \
    --pair "$joint_out/prior-sample-bael:$joint_out/denoise-from-prior-bael" \
    --pair "$joint_out/prior-sample-stolas:$joint_out/denoise-from-prior-stolas" \
    "${denoise_bridge_gate_args[@]}" \
    "${denoise_bridge_identity_args[@]}" \
    --out "$denoise_bridge_path"
  node scripts/check-solomon-generation-integrity.mjs \
    --sample-dir "$joint_out/denoise-from-prior-bael" \
    --sample-dir "$joint_out/denoise-from-prior-stolas" \
    --expected-latent-target-source attention-plan \
    --out "$joint_out/denoise-generation-integrity.json"
fi

if [[ "$joint_corpus_version" == "v2" ]]; then
  score_generative_eval_retrieval "$quality_generative_eval" "$joint_out/retrieval-head.json"

  quality_report_args=(
    --eval "$joint_out/attention-eval.json"
    --examples "$joint_out/examples.jsonl"
    --manifest "$joint_out/manifest.json"
    --retrieval-head "$joint_out/retrieval-head.json"
    --retrieval-head-eval "$joint_out/retrieval-head-eval.json"
    --sample-binding "$joint_out/prior-sample-binding.json"
    --generation-integrity "$joint_out/generation-integrity.json"
    --identity-inference "$joint_out/identity-inference.json"
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
    --out "$joint_out/quality-report.json"
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
  if [[ -n "$v2_curriculum_required_stages" ]]; then
    quality_report_args+=(--require-curriculum-stage-names "$v2_curriculum_required_stages")
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
  if [[ -f "$joint_out/curriculum-stages.json" ]]; then
    quality_report_args+=(--curriculum-stages "$joint_out/curriculum-stages.json")
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
