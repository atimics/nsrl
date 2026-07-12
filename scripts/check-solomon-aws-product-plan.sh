#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${NSRL_AWS_PRODUCT_PLAN_CHECK_ROOT:-/tmp/nsrl-aws-product-plan-check}"
run_name="${NSRL_AWS_PRODUCT_PLAN_CHECK_NAME:-solomon-e2e-product-plan-check-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
run_dir="${run_root}/${run_name}"
self_test="${NSRL_AWS_PRODUCT_PLAN_CHECK_SELF_TEST:-1}"

NSRL_PIPELINE_RUN_ROOT="$run_root" \
NSRL_PIPELINE_RUN_NAME="$run_name" \
NSRL_S3_URI="s3://nsrl-product-plan-check/solomon" \
  scripts/aws/run-solomon-end-to-end.sh --dry-run

node scripts/check-solomon-aws-product-plan.mjs \
  --run-dir "$run_dir" \
  --out "$run_dir/product-plan-check.json"

if [[ "$self_test" != "0" ]]; then
  node scripts/check-solomon-attention-denoise-bridge-self-test.mjs
  node scripts/check-solomon-promotion-bundle-self-test.mjs

  missing_gate_run_dir="${run_dir}-missing-gate"
  mkdir -p "$missing_gate_run_dir"
  cp "$run_dir/run.env" "$missing_gate_run_dir/run.env"
  cp "$run_dir/plan.tsv" "$missing_gate_run_dir/plan.tsv"
  cp "$run_dir/promotion.tsv" "$missing_gate_run_dir/promotion.tsv"
  node -e '
const fs = require("fs");
const dir = process.argv[1];
const originalDir = dir.endsWith("-missing-gate") ? dir.slice(0, -"-missing-gate".length) : "";
let env = fs.readFileSync(`${dir}/run.env`, "utf8");
let plan = fs.readFileSync(`${dir}/plan.tsv`, "utf8");
let promotion = fs.readFileSync(`${dir}/promotion.tsv`, "utf8");
if (originalDir) {
  env = env.split(originalDir).join(dir);
  plan = plan.split(originalDir).join(dir);
  promotion = promotion.split(originalDir).join(dir);
}
plan = plan
  .split("\n")
  .filter((line) => line === "" || !line.startsWith("promotion-bundle-check\t"))
  .join("\n");
fs.writeFileSync(`${dir}/run.env`, env);
fs.writeFileSync(`${dir}/plan.tsv`, plan.endsWith("\n") ? plan : `${plan}\n`);
fs.writeFileSync(`${dir}/promotion.tsv`, promotion);
' "$missing_gate_run_dir"
  if node scripts/check-solomon-aws-product-plan.mjs \
    --run-dir "$missing_gate_run_dir" \
    --out "$missing_gate_run_dir/product-plan-check.json"; then
    echo "expected Solomon AWS product plan without promotion-bundle-check to fail" >&2
    exit 1
  fi
  echo "solomon_aws_product_plan_missing_gate_check: $missing_gate_run_dir/product-plan-check.json"

  source_arch_run_dir="${run_dir}-source-arch"
  source_arch_repo="${source_arch_run_dir}/fake-repo"
  mkdir -p "$source_arch_repo/crates/nsrl-train-core/src"
  cp "$run_dir/run.env" "$source_arch_run_dir/run.env"
  cp "$run_dir/plan.tsv" "$source_arch_run_dir/plan.tsv"
  cp "$run_dir/promotion.tsv" "$source_arch_run_dir/promotion.tsv"
  node -e '
const fs = require("fs");
const dir = process.argv[1];
const fakeRepo = process.argv[2];
const originalDir = dir.endsWith("-source-arch") ? dir.slice(0, -"-source-arch".length) : "";
let env = fs.readFileSync(`${dir}/run.env`, "utf8");
let plan = fs.readFileSync(`${dir}/plan.tsv`, "utf8");
let promotion = fs.readFileSync(`${dir}/promotion.tsv`, "utf8");
if (originalDir) {
  env = env.split(originalDir).join(dir);
  plan = plan.split(originalDir).join(dir);
  promotion = promotion.split(originalDir).join(dir);
}
env = env.replace(/^repo_root=.*$/m, `repo_root=${fakeRepo}`);
fs.writeFileSync(`${dir}/run.env`, env);
fs.writeFileSync(`${dir}/plan.tsv`, plan);
fs.writeFileSync(`${dir}/promotion.tsv`, promotion);
fs.writeFileSync(`${fakeRepo}/crates/nsrl-train-core/src/lib.rs`, [
  "pub const MINI_TRANSFORMER_D_MODEL: usize = 64;",
  "pub const MINI_TRANSFORMER_HEADS: usize = 2;",
  "pub const MINI_TRANSFORMER_HIDDEN_DIM: usize = 256;",
  "",
].join("\n"));
' "$source_arch_run_dir" "$source_arch_repo"
  if node scripts/check-solomon-aws-product-plan.mjs \
    --run-dir "$source_arch_run_dir" \
    --out "$source_arch_run_dir/product-plan-check.json"; then
    echo "expected Solomon AWS product plan with drifted train-core source architecture to fail" >&2
    exit 1
  fi
  node -e '
const fs = require("fs");
const report = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const errors = (report.errors || []).map(String);
if (!errors.some((error) => error.includes("train-core head_dim 32 != 64"))) {
  console.error(`expected train-core head_dim failure, got ${JSON.stringify(errors)}`);
  process.exit(1);
}
' "$source_arch_run_dir/product-plan-check.json"
  echo "solomon_aws_product_plan_source_arch_check: $source_arch_run_dir/product-plan-check.json"

  broken_run_dir="${run_dir}-broken"
  mkdir -p "$broken_run_dir"
  cp "$run_dir/run.env" "$broken_run_dir/run.env"
  cp "$run_dir/plan.tsv" "$broken_run_dir/plan.tsv"
  cp "$run_dir/promotion.tsv" "$broken_run_dir/promotion.tsv"
  node -e '
const fs = require("fs");
const dir = process.argv[1];
const originalDir = dir.endsWith("-broken") ? dir.slice(0, -"-broken".length) : "";
let env = fs.readFileSync(`${dir}/run.env`, "utf8");
if (originalDir) env = env.split(originalDir).join(dir);
env = env.replace(/attention_image_token_profile=symbolic16/g, "attention_image_token_profile=ink16");
env = env.replace(/attention_joint_image_token_profile=symbolic16/g, "attention_joint_image_token_profile=ink16");
env = env.replace(/attention_batch_mode=map-reduce/g, "attention_batch_mode=serial");
env = env.replace(/attention_map_reduce_workers=0/g, "attention_map_reduce_workers=1");
env = env.replace(/attention_require_image_token_profile=symbolic16/g, "attention_require_image_token_profile=ink16");
env = env.replace(/attention_require_image_token_channels=ink,edge,component,radial,direction/g, "attention_require_image_token_channels=ink");
env = env.replace(/attention_require_image_channel_token_stats=1/g, "attention_require_image_channel_token_stats=0");
env = env.replace(/attention_require_directional_groups=1/g, "attention_require_directional_groups=0");
env = env.replace(/attention_min_image_channel_distinct_bins=2/g, "attention_min_image_channel_distinct_bins=1");
env = env.replace(/attention_v2_curriculum_required_stages=identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind/g, "attention_v2_curriculum_required_stages=identity,text-to-image");
env = env.replace(/attention_v2_stage_epochs=1/g, "attention_v2_stage_epochs=0");
env = env.replace(/attention_v2_native_bind_epochs=2/g, "attention_v2_native_bind_epochs=1");
env = env.replace(/generative_prompts=data\/processed\/key-solomon-goetia-latent-v1\/prompts-expanded\.jsonl/g, "generative_prompts=none");
env = env.replace(/generative_eval_permille=200/g, "generative_eval_permille=180");
env = env.replace(/generative_limit=72/g, "generative_limit=8");
env = env.replace(/attention_heldout_prompts=data\/processed\/key-solomon-goetia-latent-v1\/prompts-expanded\.jsonl/g, "attention_heldout_prompts=none");
env = env.replace(/attention_require_heldout_prompts=1/g, "attention_require_heldout_prompts=0");
env = env.replace(/attention_min_heldout_prompt_rows=72/g, "attention_min_heldout_prompt_rows=0");
env = env.replace(/attention_min_match_yes_top1=72/g, "attention_min_match_yes_top1=0");
env = env.replace(/attention_min_match_no_top1=72/g, "attention_min_match_no_top1=0");
env = env.replace(/attention_min_match_no_image_top1=72/g, "attention_min_match_no_image_top1=0");
env = env.replace(/attention_min_match_no_prompt_top1=72/g, "attention_min_match_no_prompt_top1=0");
env = env.replace(/attention_min_retrieval_margin=1/g, "attention_min_retrieval_margin=0");
env = env.replace(/attention_require_identity_inference=1/g, "attention_require_identity_inference=0");
env = env.replace(/attention_require_grounded_corpus=1/g, "attention_require_grounded_corpus=0");
env = env.replace(/attention_min_source_overlap_tokens=2/g, "attention_min_source_overlap_tokens=0");
env = env.replace(/attention_min_attribute_source_overlap_tokens=8/g, "attention_min_attribute_source_overlap_tokens=0");
env = env.replace(/attention_max_source_placeholder_rows=0/g, "attention_max_source_placeholder_rows=1");
env = env.replace(/attention_max_attribute_generic_rank_rows=0/g, "attention_max_attribute_generic_rank_rows=1");
env = env.replace(/attention_require_architecture_profile=1/g, "attention_require_architecture_profile=0");
env = env.replace(/attention_min_d_model=128/g, "attention_min_d_model=64");
env = env.replace(/attention_min_heads=2/g, "attention_min_heads=1");
env = env.replace(/attention_min_hidden_dim=256/g, "attention_min_hidden_dim=128");
env = env.replace(/attention_min_transformer_layers=2/g, "attention_min_transformer_layers=1");
env = env.replace(/attention_min_context_seq_len=384/g, "attention_min_context_seq_len=32");
env = env.replace(/attention_require_denoise_output_identity=1/g, "attention_require_denoise_output_identity=0");
env = env.replace(/attention_denoise_max_output_retrieval_rank=1/g, "attention_denoise_max_output_retrieval_rank=5");
env = env.replace(/attention_denoise_min_output_retrieval_margin=1/g, "attention_denoise_min_output_retrieval_margin=0");
env = env.replace(/attention_denoise_min_unique_targets=2/g, "attention_denoise_min_unique_targets=1");
env = env.replace(/attention_seq_len=512/g, "attention_seq_len=32");
env = env.replace(/attention_eval_max_examples=none/g, "attention_eval_max_examples=8");
env = env.replace(/attention_require_generative_eval=1/g, "attention_require_generative_eval=0");
env = env.replace(/attention_require_generative_output_identity=1/g, "attention_require_generative_output_identity=0");
env = env.replace(/require_graviton=1/g, "require_graviton=0");
env = env.replace(/require_s3_artifacts=1/g, "require_s3_artifacts=0");
env = env.replace(/s3_uri=s3:\/\/nsrl-product-plan-check\/solomon/g, "s3_uri=none");
env = env.replace(/s3_pipeline_uri=s3:\/\/nsrl-product-plan-check\/solomon\/pipelines\/[^\n]*/g, "s3_pipeline_uri=none");
env = env.replace(/promotion_bundle_check=1/g, "promotion_bundle_check=0");
env = env.replace(/attention_generative_eval=[^\n]*/g, "attention_generative_eval=none");
env = env.replace(/attention_min_generated_prompt_rows=72/g, "attention_min_generated_prompt_rows=8");
env = env.replace(/attention_min_generated_top5_16_per_mille=1/g, "attention_min_generated_top5_16_per_mille=0");
env = env.replace(/attention_min_generated_retrieval_top1_per_mille=1000/g, "attention_min_generated_retrieval_top1_per_mille=0");
env = env.replace(/attention_min_generated_retrieval_top5_per_mille=1000/g, "attention_min_generated_retrieval_top5_per_mille=0");
env = env.replace(/attention_min_generated_retrieval_margin=1/g, "attention_min_generated_retrieval_margin=0");
env = env.replace(/attention_max_generated_mean_target_distance_16_q8=7000000/g, "attention_max_generated_mean_target_distance_16_q8=9000000");
env = env.replace(/attention_min_task_targets=all=72/g, "attention_min_task_targets=all=0");
env = env.replace(/attention_min_task_top5_per_mille=all=1/g, "attention_min_task_top5_per_mille=all=0");
env = env.replace(/attention_min_phase_targets=all=72/g, "attention_min_phase_targets=all=0");
env = env.replace(/attention_require_promoted_small_profile=1/g, "attention_require_promoted_small_profile=0");
fs.writeFileSync(`${dir}/run.env`, env);
let plan = fs.readFileSync(`${dir}/plan.tsv`, "utf8");
if (originalDir) plan = plan.split(originalDir).join(dir);
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE=symbolic16/g, "NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE=ink16");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE=symbolic16/g, "NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE=ink16");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce/g, "NSRL_SOLOMON_ATTENTION_BATCH_MODE=serial");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0/g, "NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE=symbolic16/g, "NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE=ink16");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS=ink,edge,component,radial,direction/g, "NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS=ink");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS=1/g, "NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS=1/g, "NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS=2/g, "NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS=1");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES=identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind/g, "NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES=identity,text-to-image");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS=1/g, "NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS=0");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS=2/g, "NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS=1");
plan = plan.replace(/--prompts data\/processed\/key-solomon-goetia-latent-v1\/prompts-expanded\.jsonl/g, "--prompts none");
plan = plan.replace(/--partition eval/g, "--partition train");
plan = plan.replace(/--eval-permille 200/g, "--eval-permille 180");
plan = plan.replace(/--limit 72/g, "--limit 8");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS=data\/processed\/key-solomon-goetia-latent-v1\/prompts-expanded\.jsonl/g, "NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS=none");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS=1/g, "NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS=72/g, "NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1=72/g, "NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1=72/g, "NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1=72/g, "NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1=72/g, "NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN=1/g, "NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN=0");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE=1/g, "NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE=0");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS=1/g, "NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS=2/g, "NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS=8/g, "NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS=0/g, "NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS=0/g, "NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=1/g, "NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_D_MODEL=128/g, "NSRL_SOLOMON_V2_MIN_D_MODEL=64");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_HEADS=2/g, "NSRL_SOLOMON_V2_MIN_HEADS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_HIDDEN_DIM=256/g, "NSRL_SOLOMON_V2_MIN_HIDDEN_DIM=128");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS=2/g, "NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN=384/g, "NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN=32");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY=1/g, "NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY=0");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK=1/g, "NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK=5");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN=1/g, "NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN=0");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS=2/g, "NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS=1");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS=2/g, "NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS=1");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_SEQ_LEN=512/g, "NSRL_SOLOMON_ATTENTION_SEQ_LEN=32");
plan = plan.replace(/NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=none/g, "NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=8");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL=1/g, "NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL=0");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY=1/g, "NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY=0");
plan = plan.replace(/NSRL_SOLOMON_V2_GENERATIVE_EVAL=[^ ]+/g, "NSRL_SOLOMON_V2_GENERATIVE_EVAL=none");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS=72/g, "NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS=8");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE=1/g, "NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE=1000/g, "NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE=1000/g, "NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN=1/g, "NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8=7000000/g, "NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8=9000000");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=72/g, "NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=1/g, "NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=0");
plan = plan.replace(/NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=all=72/g, "NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=all=0");
plan = plan.replace(/NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1/g, "NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=0");
fs.writeFileSync(`${dir}/plan.tsv`, plan);
let promotion = fs.readFileSync(`${dir}/promotion.tsv`, "utf8");
if (originalDir) promotion = promotion.split(originalDir).join(dir);
fs.writeFileSync(`${dir}/promotion.tsv`, promotion);
' "$broken_run_dir"
  if node scripts/check-solomon-aws-product-plan.mjs \
    --run-dir "$broken_run_dir" \
    --out "$broken_run_dir/product-plan-check.json"; then
    echo "expected broken Solomon AWS product plan to fail" >&2
    exit 1
  fi
  echo "solomon_aws_product_plan_negative_check: $broken_run_dir/product-plan-check.json"
fi

echo "solomon_aws_product_plan_check: $run_dir/product-plan-check.json"
