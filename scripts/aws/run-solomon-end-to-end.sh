#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'USAGE'
Run the Solomon pipeline end-to-end on a native Linux/Graviton runner.

Default stages:
  dataset,denoiser,prior,generative-eval,attention-curriculum
  plus the final promotion-bundle-check product gate

Common knobs:
  NSRL_PIPELINE_RUN_NAME=solomon-e2e-001
  NSRL_PIPELINE_RUN_ROOT=data/aws-pipelines
  NSRL_SOLOMON_AWS_STAGES=dataset,denoiser,prior,generative-eval,attention-curriculum
  NSRL_S3_URI=s3://bucket/prefix  # required for product runs by default

Stage knobs:
  NSRL_SOLOMON_DENOISE_DATASET=<run-dir>/denoise-dataset
  NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR=<run-dir>/text-denoiser
  NSRL_SOLOMON_TEXT_DENOISE_MODEL=<run-dir>/text-denoiser/model.nsrltch
  NSRL_SOLOMON_ATTENTION_DENOISER_MODEL=<run-dir>/text-denoiser/model.nsrltch
  NSRL_SOLOMON_ATTENTION_DENOISER_MODEL=none  # opt out of attention denoise bridge
  NSRL_SOLOMON_PRIOR_RUN_NAME=prior
  NSRL_SOLOMON_GENERATIVE_EVAL_PROMPTS=data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl
  NSRL_SOLOMON_GENERATIVE_EVAL_PERMILLE=200
  NSRL_SOLOMON_GENERATIVE_EVAL_LIMIT=72
  NSRL_SOLOMON_ATTENTION_CORPUS_VERSION=v2
  NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE=chunked
  NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE=symbolic16
  NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS=data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl
  NSRL_SOLOMON_ATTENTION_SEQ_LEN=512
  NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce
  NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0  # auto-scale to online CPUs
  NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=auto-online-processors
  NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_INK_RANGE=1
  NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_SIGNATURE_DISTANCE=<measured-threshold>
  NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS=2

Optional stages:
  NSRL_SOLOMON_AWS_STAGES=dataset,denoiser,prior,generative-eval,multimodal,attention
  NSRL_SOLOMON_AWS_STAGES=dataset,denoiser,prior,generative-eval,multimodal,attention,attention-curriculum
  NSRL_SOLOMON_AWS_STAGES=all

Optional S3 input hydrate:
  NSRL_PIPELINE_FETCH_INPUTS=1
  NSRL_PIPELINE_INPUT_S3_URI=s3://bucket/prefix/inputs

Dry run:
  scripts/aws/run-solomon-end-to-end.sh --dry-run
  NSRL_PIPELINE_DRY_RUN=1 scripts/aws/run-solomon-end-to-end.sh

Dry run resolves the same defaults, writes run.env, artifacts.tsv, plan.tsv,
and per-stage status/log files, including the final promotion-bundle-check,
but does not execute stages or sync S3.

Artifacts are gathered under:
  <run-root>/<run-name>/
and synced, when NSRL_S3_URI is set, to:
  s3://bucket/prefix/pipelines/<run-name>/

Real product runs require NSRL_S3_URI unless
NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=0 is set for a local diagnostic run.

Promotion evidence is summarized in:
  <run-root>/<run-name>/promotion.tsv

Successful runs also write:
  <run-root>/<run-name>/pipeline-complete.json
USAGE
}

dry_run="${NSRL_PIPELINE_DRY_RUN:-0}"
while (($# > 0)); do
  case "$1" in
    --help | -h)
      print_usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "try: scripts/aws/run-solomon-end-to-end.sh --help" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$dry_run" == "true" || "$dry_run" == "yes" ]]; then
  dry_run=1
fi
if [[ "$dry_run" != "0" && "$dry_run" != "1" ]]; then
  echo "NSRL_PIPELINE_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
pipeline_run_name="${NSRL_PIPELINE_RUN_NAME:-${NSRL_RUN_NAME:-solomon-e2e-${timestamp}}}"
pipeline_run_root="${NSRL_PIPELINE_RUN_ROOT:-data/aws-pipelines}"
pipeline_run_dir="${pipeline_run_root}/${pipeline_run_name}"
log_dir="${pipeline_run_dir}/logs"
mkdir -p "$log_dir"

stage_csv="${NSRL_SOLOMON_AWS_STAGES:-dataset,denoiser,prior,generative-eval,attention-curriculum}"
if [[ "$stage_csv" == "all" ]]; then
  stage_csv="dataset,denoiser,prior,generative-eval,multimodal,attention,attention-curriculum"
fi
stage_csv="${stage_csv//[[:space:]]/}"
IFS=',' read -r -a stage_list <<< "$stage_csv"

for stage in "${stage_list[@]}"; do
  [[ -z "$stage" ]] && continue
  case "$stage" in
    dataset|denoiser|prior|generative-eval|multimodal|attention|attention-curriculum)
      ;;
    *)
      echo "unknown NSRL_SOLOMON_AWS_STAGES entry: $stage" >&2
      exit 2
      ;;
  esac
done

has_stage() {
  local wanted="$1"
  local stage
  for stage in "${stage_list[@]}"; do
    if [[ "$stage" == "$wanted" ]]; then
      return 0
    fi
  done
  return 1
}

sync_pipeline_artifacts() {
  if [[ "$dry_run" != "0" ]]; then
    return 0
  fi
  if [[ -n "$s3_uri" ]]; then
    aws s3 sync "$pipeline_run_dir" "$s3_pipeline_uri" --only-show-errors
  fi
}

run_stage() {
  local stage="$1"
  shift
  local log="${log_dir}/${stage}.log"
  local status_path="${log_dir}/${stage}.status"
  local started
  local finished
  local status
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$dry_run" != "0" ]]; then
    echo "[${stage}] dry-run ${started}"
    {
      echo "stage=${stage}"
      echo "started_at=${started}"
      echo "dry_run=1"
      echo "command=$*"
    } > "$log"
    finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    {
      echo "stage=${stage}"
      echo "started_at=${started}"
      echo "finished_at=${finished}"
      echo "status=0"
      echo "dry_run=1"
      echo "log=${log}"
    } > "$status_path"
    printf '%s\t%s\n' "$stage" "$*" >> "${pipeline_run_dir}/plan.tsv"
    return 0
  fi
  echo "[${stage}] start ${started}"
  printf '%s\t%s\n' "$stage" "$*" >> "${pipeline_run_dir}/plan.tsv"
  set +e
  {
    echo "stage=${stage}"
    echo "started_at=${started}"
    echo "command=$*"
    "$@"
  } > >(tee "$log") 2>&1
  status=$?
  set -e
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {
    echo "stage=${stage}"
    echo "started_at=${started}"
    echo "finished_at=${finished}"
    echo "status=${status}"
    echo "log=${log}"
  } > "$status_path"
  sync_pipeline_artifacts
  if [[ "$status" -ne 0 ]]; then
    echo "[${stage}] failed; see ${log}" >&2
    exit "$status"
  fi
  echo "[${stage}] done ${finished}"
}

fetch_ec2_metadata_value() {
  local metadata_path="$1"
  if [[ -z "$ec2_metadata_token" ]]; then
    return 0
  fi
  curl -fsS --max-time 1 \
    -H "X-aws-ec2-metadata-token: ${ec2_metadata_token}" \
    "http://169.254.169.254/latest/meta-data/${metadata_path}" 2>/dev/null || true
}

capture_ec2_metadata() {
  ec2_metadata_token="$(curl -fsS --max-time 1 -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
    "http://169.254.169.254/latest/api/token" 2>/dev/null || true)"
  if [[ -z "$ec2_metadata_token" ]]; then
    return 0
  fi
  ec2_instance_id="$(fetch_ec2_metadata_value instance-id)"
  ec2_instance_type="$(fetch_ec2_metadata_value instance-type)"
  ec2_availability_zone="$(fetch_ec2_metadata_value placement/availability-zone)"
  ec2_region="$(fetch_ec2_metadata_value placement/region)"
  ec2_instance_lifecycle="$(fetch_ec2_metadata_value instance-life-cycle)"
}

write_run_metadata() {
  local metadata="${pipeline_run_dir}/run.env"
  {
    echo "schema=nsrl.solomon_aws_pipeline.v1"
    echo "run_name=${pipeline_run_name}"
    echo "run_dir=${pipeline_run_dir}"
    echo "stages=${stage_csv}"
    echo "dry_run=${dry_run}"
    echo "runner_kernel=${runner_kernel}"
    echo "runner_arch=${runner_arch}"
    echo "require_graviton=${require_graviton}"
    echo "ec2_metadata_required=${require_ec2_metadata}"
    echo "ec2_instance_id=${ec2_instance_id}"
    echo "ec2_instance_type=${ec2_instance_type}"
    echo "ec2_availability_zone=${ec2_availability_zone}"
    echo "ec2_region=${ec2_region}"
    echo "ec2_instance_lifecycle=${ec2_instance_lifecycle}"
    echo "require_s3_artifacts=${require_s3_artifacts}"
    echo "s3_uri=${s3_uri}"
    echo "s3_pipeline_uri=${s3_pipeline_uri}"
    echo "promotion_manifest=${promotion_manifest}"
    echo "pipeline_complete_report=${pipeline_complete_report}"
    echo "promotion_bundle_check=${promotion_bundle_check}"
    echo "created_at=${timestamp}"
    echo "repo_root=${repo_root}"
    echo "cargo_target_dir=${CARGO_TARGET_DIR:-target}"
    echo "release_bin_dir=${CARGO_TARGET_DIR:-target}/release"
    echo "generative_prompts=${generative_prompts}"
    echo "generative_eval_permille=${generative_eval_permille}"
    echo "generative_limit=${generative_limit}"
    echo "generative_eval_run=${generative_out_dir}/${generative_run_name}"
    echo "generative_eval_summary=${generative_out_dir}/${generative_run_name}/summary.tsv"
    echo "attention_curriculum_out_dir=${attention_curriculum_out_dir}"
    echo "attention_curriculum_quality_report=${attention_curriculum_out_dir}/quality-report.json"
    echo "attention_corpus_version=${attention_corpus_version}"
    echo "attention_joint_corpus_version=${attention_joint_corpus_version}"
    echo "attention_text_token_profile=${attention_text_token_profile}"
    echo "attention_image_token_profile=${attention_image_token_profile}"
    echo "attention_joint_image_token_profile=${attention_joint_image_token_profile}"
    echo "attention_batch_mode=${attention_batch_mode}"
    echo "attention_map_reduce_workers=${attention_map_reduce_workers}"
    echo "attention_cpu_scaling_policy=${attention_cpu_scaling_policy}"
    echo "attention_map_reduce_auto_workers=${attention_map_reduce_auto_workers}"
    echo "attention_effective_map_reduce_workers=${attention_effective_map_reduce_workers}"
    echo "attention_seq_len=${attention_seq_len}"
    echo "attention_eval_max_examples=${attention_eval_max_examples}"
    echo "attention_v2_curriculum_stages=${attention_v2_curriculum_stages}"
    echo "attention_v2_curriculum_required_stages=${attention_v2_curriculum_required_stages}"
    echo "attention_v2_stage_epochs=${attention_v2_stage_epochs}"
    echo "attention_v2_native_bind_epochs=${attention_v2_native_bind_epochs}"
    echo "attention_require_image_token_profile=${attention_require_image_token_profile}"
    echo "attention_require_image_token_channels=${attention_require_image_token_channels}"
    echo "attention_require_image_channel_token_stats=${attention_require_image_channel_token_stats}"
    echo "attention_min_image_channel_distinct_bins=${attention_min_image_channel_distinct_bins}"
    echo "attention_require_directional_groups=${attention_require_directional_groups}"
    echo "attention_heldout_prompts=${attention_heldout_prompts}"
    echo "attention_require_heldout_prompts=${attention_require_heldout_prompts}"
    echo "attention_min_heldout_prompt_rows=${attention_min_heldout_prompt_rows}"
    echo "attention_min_match_yes_top1=${attention_min_match_yes_top1}"
    echo "attention_min_match_no_top1=${attention_min_match_no_top1}"
    echo "attention_min_match_no_image_top1=${attention_min_match_no_image_top1}"
    echo "attention_min_match_no_prompt_top1=${attention_min_match_no_prompt_top1}"
    echo "attention_min_retrieval_margin=${attention_min_retrieval_margin}"
    echo "attention_min_direction_accuracy_per_mille=${attention_min_direction_accuracy_per_mille}"
    echo "attention_min_direction_top5_per_mille=${attention_min_direction_top5_per_mille}"
    echo "attention_min_direction_top10_per_mille=${attention_min_direction_top10_per_mille}"
    echo "attention_require_identity_inference=${attention_require_identity_inference}"
    echo "attention_require_grounded_corpus=${attention_require_grounded_corpus}"
    echo "attention_min_source_overlap_tokens=${attention_min_source_overlap_tokens}"
    echo "attention_min_attribute_source_overlap_tokens=${attention_min_attribute_source_overlap_tokens}"
    echo "attention_max_source_placeholder_rows=${attention_max_source_placeholder_rows}"
    echo "attention_max_attribute_generic_rank_rows=${attention_max_attribute_generic_rank_rows}"
    echo "attention_require_architecture_profile=${attention_require_architecture_profile}"
    echo "attention_min_d_model=${attention_min_d_model}"
    echo "attention_min_heads=${attention_min_heads}"
    echo "attention_min_hidden_dim=${attention_min_hidden_dim}"
    echo "attention_min_transformer_layers=${attention_min_transformer_layers}"
    echo "attention_min_context_seq_len=${attention_min_context_seq_len}"
    echo "attention_require_promoted_small_profile=${attention_require_promoted_small_profile}"
    echo "attention_require_confidence_trace=${attention_require_confidence_trace}"
    echo "attention_require_denoise_bridge=${attention_require_denoise_bridge}"
    echo "attention_require_denoise_output_identity=${attention_require_denoise_output_identity}"
    echo "attention_denoise_max_output_retrieval_rank=${attention_denoise_max_output_retrieval_rank}"
    echo "attention_denoise_min_output_retrieval_margin=${attention_denoise_min_output_retrieval_margin}"
    echo "attention_denoise_min_unique_targets=${attention_denoise_min_unique_targets}"
    echo "attention_generative_eval=${attention_generative_eval}"
    echo "attention_require_generative_eval=${attention_require_generative_eval}"
    echo "attention_require_generative_output_identity=${attention_require_generative_output_identity}"
    echo "attention_min_generated_prompt_rows=${attention_min_generated_prompt_rows}"
    echo "attention_min_generated_top5_16_per_mille=${attention_min_generated_top5_16_per_mille}"
    echo "attention_min_generated_retrieval_top1_per_mille=${attention_min_generated_retrieval_top1_per_mille}"
    echo "attention_min_generated_retrieval_top5_per_mille=${attention_min_generated_retrieval_top5_per_mille}"
    echo "attention_min_generated_retrieval_margin=${attention_min_generated_retrieval_margin}"
    echo "attention_max_generated_mean_target_distance_q8=${attention_max_generated_mean_target_distance_q8}"
    echo "attention_max_generated_mean_target_distance_16_q8=${attention_max_generated_mean_target_distance_16_q8}"
    echo "attention_max_generated_mean_target_distance_px_q8=${attention_max_generated_mean_target_distance_px_q8}"
    echo "attention_min_task_targets=${attention_min_task_targets}"
    echo "attention_min_task_top5_per_mille=${attention_min_task_top5_per_mille}"
    echo "attention_min_phase_targets=${attention_min_phase_targets}"
    echo "uname=$(uname -a)"
    echo "processor_count=${online_processors}"
  } > "$metadata"
}

write_artifact_manifest() {
  local manifest="${pipeline_run_dir}/artifacts.tsv"
  if [[ "$dry_run" != "0" ]]; then
    {
      printf 'stage\tartifact\tpath\n'
      [[ -f "${pipeline_run_dir}/run.env" ]] && printf 'pipeline\trun_env\trun.env\n'
      [[ -f "${pipeline_run_dir}/plan.tsv" ]] && printf 'pipeline\tplan\tplan.tsv\n'
      [[ -f "${pipeline_run_dir}/artifacts.tsv" ]] && printf 'pipeline\tartifacts\tartifacts.tsv\n'
      [[ -f "$promotion_manifest" ]] && printf 'pipeline\tpromotion_manifest\t%s\n' "$promotion_manifest"
      [[ -f "$pipeline_complete_report" ]] && printf 'pipeline\tpipeline_complete\t%s\n' "$pipeline_complete_report"
    } > "$manifest"
    return 0
  fi
  {
    printf 'stage\tartifact\tpath\n'
    [[ -f "${pipeline_run_dir}/run.env" ]] && printf 'pipeline\trun_env\trun.env\n'
    [[ -f "${pipeline_run_dir}/plan.tsv" ]] && printf 'pipeline\tplan\tplan.tsv\n'
    [[ -f "${pipeline_run_dir}/artifacts.tsv" ]] && printf 'pipeline\tartifacts\tartifacts.tsv\n'
    [[ -f "$promotion_manifest" ]] && printf 'pipeline\tpromotion_manifest\t%s\n' "$promotion_manifest"
    [[ -f "$pipeline_complete_report" ]] && printf 'pipeline\tpipeline_complete\t%s\n' "$pipeline_complete_report"
    [[ -f "${pipeline_run_dir}/promotion-bundle-check.json" ]] && printf 'pipeline\tpromotion_bundle_check\t%s\n' "${pipeline_run_dir}/promotion-bundle-check.json"
    [[ -d "$denoise_dataset" ]] && printf 'dataset\troot\t%s\n' "$denoise_dataset"
    [[ -f "$text_denoise_model" ]] && printf 'denoiser\tmodel\t%s\n' "$text_denoise_model"
    [[ -f "$latent_model" ]] && printf 'prior\tlatent_model\t%s\n' "$latent_model"
    [[ -f "${prior_run_dir}/smoke-check.json" ]] && printf 'prior\tsmoke_check\t%s\n' "${prior_run_dir}/smoke-check.json"
    [[ -d "${generative_out_dir}/${generative_run_name}" ]] && printf 'generative-eval\trun\t%s\n' "${generative_out_dir}/${generative_run_name}"
    [[ -f "${generative_out_dir}/${generative_run_name}/summary.tsv" ]] && printf 'generative-eval\tsummary\t%s\n' "${generative_out_dir}/${generative_run_name}/summary.tsv"
    [[ -f "${generative_out_dir}/${generative_run_name}/samples.tsv" ]] && printf 'generative-eval\tsamples\t%s\n' "${generative_out_dir}/${generative_run_name}/samples.tsv"
    [[ -f "${multimodal_out_dir}/model.nsrlmod" ]] && printf 'multimodal\tmodel\t%s\n' "${multimodal_out_dir}/model.nsrlmod"
    [[ -f "${attention_out_dir}/model.nsrllmm" ]] && printf 'attention\tmodel\t%s\n' "${attention_out_dir}/model.nsrllmm"
    [[ -f "${attention_out_dir}/manifest.json" ]] && printf 'attention\tcorpus_manifest\t%s\n' "${attention_out_dir}/manifest.json"
    [[ -f "${attention_out_dir}/attention-eval.json" ]] && printf 'attention\tattention_eval\t%s\n' "${attention_out_dir}/attention-eval.json"
    [[ -f "${attention_out_dir}/retrieval-head.json" ]] && printf 'attention\tretrieval_head\t%s\n' "${attention_out_dir}/retrieval-head.json"
    [[ -f "${attention_out_dir}/retrieval-head-eval.json" ]] && printf 'attention\tretrieval_head_eval\t%s\n' "${attention_out_dir}/retrieval-head-eval.json"
    [[ -f "${attention_out_dir}/grounded-corpus.json" ]] && printf 'attention\tgrounded_corpus\t%s\n' "${attention_out_dir}/grounded-corpus.json"
    [[ -f "${attention_out_dir}/sample-binding.json" ]] && printf 'attention\tsample_binding\t%s\n' "${attention_out_dir}/sample-binding.json"
    [[ -f "${attention_out_dir}/identity-inference.json" ]] && printf 'attention\tidentity_inference\t%s\n' "${attention_out_dir}/identity-inference.json"
    [[ -f "${attention_out_dir}/generation-integrity.json" ]] && printf 'attention\tgeneration_integrity\t%s\n' "${attention_out_dir}/generation-integrity.json"
    [[ -f "${attention_out_dir}/denoise-bridge.json" ]] && printf 'attention\tdenoise_bridge\t%s\n' "${attention_out_dir}/denoise-bridge.json"
    [[ -f "${attention_out_dir}/denoise-generation-integrity.json" ]] && printf 'attention\tdenoise_generation_integrity\t%s\n' "${attention_out_dir}/denoise-generation-integrity.json"
    [[ -f "${attention_out_dir}/quality-report.json" ]] && printf 'attention\tquality_report\t%s\n' "${attention_out_dir}/quality-report.json"
    [[ -f "${attention_curriculum_out_dir}/model.nsrllmm" ]] && printf 'attention-curriculum\tmodel\t%s\n' "${attention_curriculum_out_dir}/model.nsrllmm"
    [[ -f "${attention_curriculum_out_dir}/manifest.json" ]] && printf 'attention-curriculum\tcorpus_manifest\t%s\n' "${attention_curriculum_out_dir}/manifest.json"
    [[ -f "${attention_curriculum_out_dir}/attention-eval.json" ]] && printf 'attention-curriculum\tattention_eval\t%s\n' "${attention_curriculum_out_dir}/attention-eval.json"
    [[ -f "${attention_curriculum_out_dir}/retrieval-head.json" ]] && printf 'attention-curriculum\tretrieval_head\t%s\n' "${attention_curriculum_out_dir}/retrieval-head.json"
    [[ -f "${attention_curriculum_out_dir}/retrieval-head-eval.json" ]] && printf 'attention-curriculum\tretrieval_head_eval\t%s\n' "${attention_curriculum_out_dir}/retrieval-head-eval.json"
    [[ -f "${attention_curriculum_out_dir}/curriculum-stages.json" ]] && printf 'attention-curriculum\tcurriculum_stages\t%s\n' "${attention_curriculum_out_dir}/curriculum-stages.json"
    [[ -f "${attention_curriculum_out_dir}/grounded-corpus.json" ]] && printf 'attention-curriculum\tgrounded_corpus\t%s\n' "${attention_curriculum_out_dir}/grounded-corpus.json"
    [[ -f "${attention_curriculum_out_dir}/prior-sample-binding.json" ]] && printf 'attention-curriculum\tsample_binding\t%s\n' "${attention_curriculum_out_dir}/prior-sample-binding.json"
    [[ -f "${attention_curriculum_out_dir}/identity-inference.json" ]] && printf 'attention-curriculum\tidentity_inference\t%s\n' "${attention_curriculum_out_dir}/identity-inference.json"
    [[ -f "${attention_curriculum_out_dir}/generation-integrity.json" ]] && printf 'attention-curriculum\tgeneration_integrity\t%s\n' "${attention_curriculum_out_dir}/generation-integrity.json"
    [[ -f "${attention_curriculum_out_dir}/denoise-bridge.json" ]] && printf 'attention-curriculum\tdenoise_bridge\t%s\n' "${attention_curriculum_out_dir}/denoise-bridge.json"
    [[ -f "${attention_curriculum_out_dir}/denoise-generation-integrity.json" ]] && printf 'attention-curriculum\tdenoise_generation_integrity\t%s\n' "${attention_curriculum_out_dir}/denoise-generation-integrity.json"
    [[ -f "${attention_curriculum_out_dir}/quality-report.json" ]] && printf 'attention-curriculum\tquality_report\t%s\n' "${attention_curriculum_out_dir}/quality-report.json"
  } > "$manifest"
}

write_promotion_manifest() {
  local manifest="$promotion_manifest"
  {
    printf 'product\tstage\tartifact\tpath\trequired\n'
    printf 'solomon-v1\tpipeline\trun_env\trun.env\t1\n'
    printf 'solomon-v1\tpipeline\tplan\tplan.tsv\t1\n'
    printf 'solomon-v1\tpipeline\tartifacts\tartifacts.tsv\t1\n'
    printf 'solomon-v1\tattention-curriculum\tquality_report\tattention-curriculum/quality-report.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tmodel\tattention-curriculum/model.nsrllmm\t1\n'
    printf 'solomon-v1\tattention-curriculum\tcorpus_manifest\tattention-curriculum/manifest.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tattention_eval\tattention-curriculum/attention-eval.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tretrieval_head\tattention-curriculum/retrieval-head.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tretrieval_head_eval\tattention-curriculum/retrieval-head-eval.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tcurriculum_stages\tattention-curriculum/curriculum-stages.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tsample_binding\tattention-curriculum/prior-sample-binding.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tidentity_inference\tattention-curriculum/identity-inference.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tgrounded_corpus\tattention-curriculum/grounded-corpus.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tgeneration_integrity\tattention-curriculum/generation-integrity.json\t1\n'
    printf 'solomon-v1\tattention-curriculum\tdenoise_bridge\tattention-curriculum/denoise-bridge.json\t%s\n' "$attention_require_denoise_bridge"
    printf 'solomon-v1\tattention-curriculum\tdenoise_generation_integrity\tattention-curriculum/denoise-generation-integrity.json\t%s\n' "$attention_require_denoise_output_identity"
    printf 'solomon-v1\tgenerative-eval\trun\tgenerative-eval/%s\t%s\n' "$generative_run_name" "$attention_require_generative_eval"
    printf 'solomon-v1\tgenerative-eval\tsummary\tgenerative-eval/%s/summary.tsv\t%s\n' "$generative_run_name" "$attention_require_generative_eval"
  } > "$manifest"
}

write_completion_report() {
  PIPELINE_COMPLETE_REPORT="$pipeline_complete_report" \
  PIPELINE_RUN_NAME="$pipeline_run_name" \
  PIPELINE_RUN_DIR="$pipeline_run_dir" \
  PIPELINE_DRY_RUN="$dry_run" \
  PIPELINE_STAGES="$stage_csv" \
  PIPELINE_LOG_DIR="logs" \
  PIPELINE_RUN_ENV="run.env" \
  PIPELINE_PLAN="plan.tsv" \
  PIPELINE_ARTIFACTS="artifacts.tsv" \
  PIPELINE_PROMOTION_MANIFEST="promotion.tsv" \
  PIPELINE_PROMOTION_BUNDLE_CHECK="$promotion_bundle_check" \
  PIPELINE_PROMOTION_BUNDLE_CHECK_PATH="promotion-bundle-check.json" \
  PIPELINE_QUALITY_REPORT="attention-curriculum/quality-report.json" \
  PIPELINE_COMPLETE_REF="pipeline-complete.json" \
  PIPELINE_RUNNER_KERNEL="$runner_kernel" \
  PIPELINE_RUNNER_ARCH="$runner_arch" \
  PIPELINE_ONLINE_PROCESSORS="$online_processors" \
  PIPELINE_REQUIRE_GRAVITON="$require_graviton" \
  PIPELINE_REQUIRE_EC2_METADATA="$require_ec2_metadata" \
  PIPELINE_EC2_INSTANCE_ID="$ec2_instance_id" \
  PIPELINE_EC2_INSTANCE_TYPE="$ec2_instance_type" \
  PIPELINE_EC2_AVAILABILITY_ZONE="$ec2_availability_zone" \
  PIPELINE_EC2_REGION="$ec2_region" \
  PIPELINE_EC2_INSTANCE_LIFECYCLE="$ec2_instance_lifecycle" \
  PIPELINE_REQUIRE_S3_ARTIFACTS="$require_s3_artifacts" \
  PIPELINE_S3_URI="$s3_uri" \
  PIPELINE_S3_PIPELINE_URI="$s3_pipeline_uri" \
  PIPELINE_ATTENTION_CORPUS_VERSION="$attention_corpus_version" \
  PIPELINE_ATTENTION_JOINT_CORPUS_VERSION="$attention_joint_corpus_version" \
  PIPELINE_ATTENTION_TEXT_TOKEN_PROFILE="$attention_text_token_profile" \
  PIPELINE_ATTENTION_IMAGE_TOKEN_PROFILE="$attention_image_token_profile" \
  PIPELINE_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE="$attention_joint_image_token_profile" \
  PIPELINE_ATTENTION_SEQ_LEN="$attention_seq_len" \
  PIPELINE_ATTENTION_CURRICULUM_STAGES="$attention_v2_curriculum_stages" \
  PIPELINE_ATTENTION_CURRICULUM_REQUIRED_STAGES="$attention_v2_curriculum_required_stages" \
  PIPELINE_ATTENTION_V2_STAGE_EPOCHS="$attention_v2_stage_epochs" \
  PIPELINE_ATTENTION_V2_NATIVE_BIND_EPOCHS="$attention_v2_native_bind_epochs" \
  PIPELINE_ATTENTION_IMAGE_TOKEN_CHANNELS="$attention_require_image_token_channels" \
  PIPELINE_ATTENTION_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS="$attention_require_image_channel_token_stats" \
  PIPELINE_ATTENTION_REQUIRE_DIRECTIONAL_GROUPS="$attention_require_directional_groups" \
  PIPELINE_ATTENTION_HELDOUT_PROMPTS="$attention_heldout_prompts" \
  PIPELINE_ATTENTION_MIN_HELDOUT_PROMPT_ROWS="$attention_min_heldout_prompt_rows" \
  PIPELINE_ATTENTION_MIN_TASK_TARGETS="$attention_min_task_targets" \
  PIPELINE_ATTENTION_MIN_PHASE_TARGETS="$attention_min_phase_targets" \
  PIPELINE_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS="$attention_denoise_min_unique_targets" \
  PIPELINE_ATTENTION_GENERATIVE_EVAL="$attention_generative_eval" \
  PIPELINE_ATTENTION_REQUIRE_GENERATIVE_EVAL="$attention_require_generative_eval" \
  PIPELINE_ATTENTION_REQUIRE_GENERATIVE_OUTPUT_IDENTITY="$attention_require_generative_output_identity" \
  PIPELINE_ATTENTION_MIN_GENERATED_PROMPT_ROWS="$attention_min_generated_prompt_rows" \
  PIPELINE_ATTENTION_MIN_GENERATED_TOP5_16_PER_MILLE="$attention_min_generated_top5_16_per_mille" \
  PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE="$attention_min_generated_retrieval_top1_per_mille" \
  PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE="$attention_min_generated_retrieval_top5_per_mille" \
  PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_MARGIN="$attention_min_generated_retrieval_margin" \
  PIPELINE_ATTENTION_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8="$attention_max_generated_mean_target_distance_16_q8" \
  PIPELINE_ATTENTION_BATCH_MODE="$attention_batch_mode" \
  PIPELINE_ATTENTION_MAP_REDUCE_WORKERS="$attention_map_reduce_workers" \
  PIPELINE_ATTENTION_CPU_SCALING_POLICY="$attention_cpu_scaling_policy" \
  PIPELINE_ATTENTION_MAP_REDUCE_AUTO_WORKERS="$attention_map_reduce_auto_workers" \
  PIPELINE_ATTENTION_EFFECTIVE_MAP_REDUCE_WORKERS="$attention_effective_map_reduce_workers" \
  node -e '
const fs = require("node:fs");
const path = require("node:path");
const env = process.env;
const out = env.PIPELINE_COMPLETE_REPORT;
const productStages = String(env.PIPELINE_STAGES || "").split(",").filter(Boolean);
const csv = (value) => String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
const intValue = (value) => (/^-?\d+$/.test(String(value || "").trim()) ? Number(value) : 0);
const boolValue = (value) => value === "1" || value === "true";
const hasPromotionCheck = env.PIPELINE_PROMOTION_BUNDLE_CHECK !== "0";
const resolveRunRef = (ref) => {
  const text = String(ref || "");
  if (!text) return "";
  return path.isAbsolute(text) ? text : path.join(env.PIPELINE_RUN_DIR || "", text);
};
function promotionArtifactMap(manifestRef) {
  const filePath = resolveRunRef(manifestRef);
  if (!filePath || !fs.existsSync(filePath)) return {};
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  if (lines[0] !== "product\tstage\tartifact\tpath\trequired") return {};
  const artifacts = {};
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const fields = line.split("\t");
    if (fields.length !== 5) continue;
    const [product, , artifact, artifactPath, required] = fields;
    if (product === "solomon-v1" && (required === "1" || required === "true")) {
      artifacts[artifact] = artifactPath;
    }
  }
  return artifacts;
}
const promotedArtifacts = promotionArtifactMap(env.PIPELINE_PROMOTION_MANIFEST);
const stages = hasPromotionCheck ? [...productStages, "promotion-bundle-check"] : productStages;
const statusFiles = Object.fromEntries(
  stages.map((stage) => [stage, path.join(env.PIPELINE_LOG_DIR || "", `${stage}.status`)]),
);
const report = {
  schema: "nsrl.solomon_aws_pipeline_complete.v1",
  ok: true,
  generated_at: new Date().toISOString(),
  run_name: env.PIPELINE_RUN_NAME || "",
  run_dir: env.PIPELINE_RUN_DIR || "",
  dry_run: env.PIPELINE_DRY_RUN === "1",
  stages,
  product_stages: productStages,
  runner: {
    kernel: env.PIPELINE_RUNNER_KERNEL || "",
    arch: env.PIPELINE_RUNNER_ARCH || "",
    online_processors: Number(env.PIPELINE_ONLINE_PROCESSORS || 0),
    require_graviton: env.PIPELINE_REQUIRE_GRAVITON === "1",
    ec2: {
      metadata_required: env.PIPELINE_REQUIRE_EC2_METADATA === "1",
      instance_id: env.PIPELINE_EC2_INSTANCE_ID || "",
      instance_type: env.PIPELINE_EC2_INSTANCE_TYPE || "",
      availability_zone: env.PIPELINE_EC2_AVAILABILITY_ZONE || "",
      region: env.PIPELINE_EC2_REGION || "",
      instance_lifecycle: env.PIPELINE_EC2_INSTANCE_LIFECYCLE || "",
    },
  },
  s3: {
    required: env.PIPELINE_REQUIRE_S3_ARTIFACTS === "1",
    uri: env.PIPELINE_S3_URI || "",
    pipeline_uri: env.PIPELINE_S3_PIPELINE_URI || "",
  },
  product_config: {
    attention: {
      corpus_version: env.PIPELINE_ATTENTION_CORPUS_VERSION || "",
      joint_corpus_version: env.PIPELINE_ATTENTION_JOINT_CORPUS_VERSION || "",
      text_token_profile: env.PIPELINE_ATTENTION_TEXT_TOKEN_PROFILE || "",
      image_token_profile: env.PIPELINE_ATTENTION_IMAGE_TOKEN_PROFILE || "",
      joint_image_token_profile: env.PIPELINE_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE || "",
      seq_len: intValue(env.PIPELINE_ATTENTION_SEQ_LEN),
      curriculum: {
        stages: csv(env.PIPELINE_ATTENTION_CURRICULUM_STAGES),
        required_stages: csv(env.PIPELINE_ATTENTION_CURRICULUM_REQUIRED_STAGES),
        stage_epochs: intValue(env.PIPELINE_ATTENTION_V2_STAGE_EPOCHS),
        native_bind_epochs: intValue(env.PIPELINE_ATTENTION_V2_NATIVE_BIND_EPOCHS),
      },
      image_token_channels: csv(env.PIPELINE_ATTENTION_IMAGE_TOKEN_CHANNELS),
      require_image_channel_token_stats: boolValue(env.PIPELINE_ATTENTION_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS),
      require_directional_groups: boolValue(env.PIPELINE_ATTENTION_REQUIRE_DIRECTIONAL_GROUPS),
      heldout_prompts: env.PIPELINE_ATTENTION_HELDOUT_PROMPTS || "",
      min_heldout_prompt_rows: intValue(env.PIPELINE_ATTENTION_MIN_HELDOUT_PROMPT_ROWS),
      min_task_targets: env.PIPELINE_ATTENTION_MIN_TASK_TARGETS || "",
      min_phase_targets: env.PIPELINE_ATTENTION_MIN_PHASE_TARGETS || "",
      denoise_min_unique_targets: intValue(env.PIPELINE_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS),
      generation: {
        generative_eval: env.PIPELINE_ATTENTION_GENERATIVE_EVAL || "",
        require_generative_eval: boolValue(env.PIPELINE_ATTENTION_REQUIRE_GENERATIVE_EVAL),
        require_generative_output_identity: boolValue(env.PIPELINE_ATTENTION_REQUIRE_GENERATIVE_OUTPUT_IDENTITY),
        min_generated_prompt_rows: intValue(env.PIPELINE_ATTENTION_MIN_GENERATED_PROMPT_ROWS),
        min_generated_top5_16_per_mille: intValue(env.PIPELINE_ATTENTION_MIN_GENERATED_TOP5_16_PER_MILLE),
        min_generated_retrieval_top1_per_mille: intValue(env.PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE),
        min_generated_retrieval_top5_per_mille: intValue(env.PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE),
        min_generated_retrieval_margin: intValue(env.PIPELINE_ATTENTION_MIN_GENERATED_RETRIEVAL_MARGIN),
        max_generated_mean_target_distance_16_q8: intValue(env.PIPELINE_ATTENTION_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8),
      },
      cpu_scaling: {
        batch_mode: env.PIPELINE_ATTENTION_BATCH_MODE || "",
        map_reduce_workers: env.PIPELINE_ATTENTION_MAP_REDUCE_WORKERS || "",
        policy: env.PIPELINE_ATTENTION_CPU_SCALING_POLICY || "",
        auto_workers: boolValue(env.PIPELINE_ATTENTION_MAP_REDUCE_AUTO_WORKERS),
        effective_workers: intValue(env.PIPELINE_ATTENTION_EFFECTIVE_MAP_REDUCE_WORKERS),
      },
    },
  },
  artifacts: {
    ...promotedArtifacts,
    run_env: env.PIPELINE_RUN_ENV || "",
    plan: env.PIPELINE_PLAN || "",
    artifacts: env.PIPELINE_ARTIFACTS || "",
    promotion_manifest: env.PIPELINE_PROMOTION_MANIFEST || "",
    promotion_bundle_check: hasPromotionCheck ? env.PIPELINE_PROMOTION_BUNDLE_CHECK_PATH || "" : "",
    quality_report: env.PIPELINE_QUALITY_REPORT || "",
    pipeline_complete: env.PIPELINE_COMPLETE_REF || out || "",
  },
  status_files: statusFiles,
};
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`, "utf8");
'
}

add_optional_dataset_arg() {
  local var_name="$1"
  local flag="$2"
  local value="${!var_name:-}"
  if [[ -n "$value" ]]; then
    dataset_args+=("$flag" "$value")
  fi
}

denoise_dataset="${NSRL_SOLOMON_DENOISE_DATASET:-${pipeline_run_dir}/denoise-dataset}"
text_denoise_out_dir="${NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR:-${pipeline_run_dir}/text-denoiser}"
text_denoise_model="${NSRL_SOLOMON_TEXT_DENOISE_MODEL:-${text_denoise_out_dir}/model.nsrltch}"
sampler_model="${NSRL_SOLOMON_DENOISE_MODEL:-$text_denoise_model}"
prior_run_name="${NSRL_SOLOMON_PRIOR_RUN_NAME:-prior}"
prior_run_root="${NSRL_SOLOMON_PRIOR_RUN_ROOT:-$pipeline_run_dir}"
prior_run_dir="${prior_run_root}/${prior_run_name}"
latent_model="${NSRL_SOLOMON_LATENT_MODEL:-${prior_run_dir}/latent/model.nsrllat}"
generative_out_dir="${NSRL_SOLOMON_GENERATIVE_EVAL_OUT_DIR:-${pipeline_run_dir}/generative-eval}"
generative_run_name="${NSRL_SOLOMON_GENERATIVE_EVAL_RUN_NAME:-current}"
generative_limit="${NSRL_SOLOMON_GENERATIVE_EVAL_LIMIT:-72}"
generative_eval_permille="${NSRL_SOLOMON_GENERATIVE_EVAL_PERMILLE:-200}"
generative_latent_model="${NSRL_SOLOMON_GENERATIVE_EVAL_LATENT_MODEL:-current=${latent_model}}"
generative_retrieval_head="${NSRL_SOLOMON_GENERATIVE_EVAL_RETRIEVAL_HEAD:-}"
generative_require_retrieval_head="${NSRL_SOLOMON_GENERATIVE_EVAL_REQUIRE_RETRIEVAL_HEAD:-0}"
multimodal_out_dir="${NSRL_SOLOMON_MULTIMODAL_OUT_DIR:-${pipeline_run_dir}/multimodal}"
attention_out_dir="${NSRL_SOLOMON_ATTENTION_OUT_DIR:-${pipeline_run_dir}/attention}"
attention_curriculum_out_dir="${NSRL_SOLOMON_ATTENTION_CURRICULUM_OUT_DIR:-${pipeline_run_dir}/attention-curriculum}"
promotion_manifest="${NSRL_SOLOMON_PROMOTION_MANIFEST:-${pipeline_run_dir}/promotion.tsv}"
pipeline_complete_report="${NSRL_SOLOMON_PIPELINE_COMPLETE:-${pipeline_run_dir}/pipeline-complete.json}"
promotion_bundle_check="${NSRL_SOLOMON_CHECK_PROMOTION_BUNDLE:-1}"
require_s3_artifacts="${NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS:-1}"
if [[ "$require_s3_artifacts" == "true" || "$require_s3_artifacts" == "yes" ]]; then
  require_s3_artifacts=1
fi
if [[ "$require_s3_artifacts" == "false" || "$require_s3_artifacts" == "no" ]]; then
  require_s3_artifacts=0
fi
if [[ "$require_s3_artifacts" != "0" && "$require_s3_artifacts" != "1" ]]; then
  echo "NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS must be 0 or 1" >&2
  exit 2
fi
s3_uri="${NSRL_S3_URI:-}"
s3_pipeline_uri=""
if [[ -n "$s3_uri" ]]; then
  case "$s3_uri" in
    s3://*)
      s3_pipeline_uri="${s3_uri%/}/pipelines/${pipeline_run_name}"
      ;;
    *)
      echo "NSRL_S3_URI must start with s3:// when set" >&2
      exit 2
      ;;
  esac
fi
if [[ "$dry_run" == "0" && "$require_s3_artifacts" != "0" && -z "$s3_uri" ]]; then
  echo "Solomon product runs require NSRL_S3_URI so promotion artifacts sync durably." >&2
  echo "Set NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=0 only for an intentional local diagnostic run." >&2
  exit 2
fi
runner_kernel="$(uname -s)"
runner_arch="$(uname -m)"
online_processors="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
case "$online_processors" in
  ''|*[!0-9]*)
    online_processors=1
    ;;
  0)
    online_processors=1
    ;;
esac
require_graviton="${NSRL_SOLOMON_REQUIRE_GRAVITON:-1}"
if [[ "$require_graviton" == "true" || "$require_graviton" == "yes" ]]; then
  require_graviton=1
fi
if [[ "$require_graviton" == "false" || "$require_graviton" == "no" ]]; then
  require_graviton=0
fi
if [[ "$require_graviton" != "0" && "$require_graviton" != "1" ]]; then
  echo "NSRL_SOLOMON_REQUIRE_GRAVITON must be 0 or 1" >&2
  exit 2
fi
require_ec2_metadata="${NSRL_SOLOMON_REQUIRE_EC2_METADATA:-$require_graviton}"
if [[ "$require_ec2_metadata" == "true" || "$require_ec2_metadata" == "yes" ]]; then
  require_ec2_metadata=1
fi
if [[ "$require_ec2_metadata" == "false" || "$require_ec2_metadata" == "no" ]]; then
  require_ec2_metadata=0
fi
if [[ "$require_ec2_metadata" != "0" && "$require_ec2_metadata" != "1" ]]; then
  echo "NSRL_SOLOMON_REQUIRE_EC2_METADATA must be 0 or 1" >&2
  exit 2
fi
ec2_metadata_token=""
ec2_instance_id=""
ec2_instance_type=""
ec2_availability_zone=""
ec2_region=""
ec2_instance_lifecycle=""
if [[ "$dry_run" == "0" && "$require_graviton" != "0" ]]; then
  case "${runner_kernel}:${runner_arch}" in
    Linux:aarch64|Linux:arm64)
      ;;
    *)
      echo "Solomon product runs require a Linux ARM64/Graviton runner; got ${runner_kernel}/${runner_arch}." >&2
      echo "Set NSRL_SOLOMON_REQUIRE_GRAVITON=0 only for an intentional non-Graviton diagnostic run." >&2
      exit 2
      ;;
  esac
fi
if [[ "$dry_run" == "0" && "$require_ec2_metadata" != "0" ]]; then
  capture_ec2_metadata
  if [[ -z "$ec2_instance_id" || -z "$ec2_instance_type" ]]; then
    echo "Solomon product runs require EC2 IMDSv2 instance metadata; could not read instance-id and instance-type." >&2
    echo "Set NSRL_SOLOMON_REQUIRE_EC2_METADATA=0 only for an intentional non-EC2 diagnostic run." >&2
    exit 2
  fi
  case "$ec2_instance_type" in
    c6g.*|c6gd.*|c6gn.*|c7g.*|c7gd.*|c7gn.*|c8g.*|c8gd.*|c8gn.*|m6g.*|m6gd.*|m7g.*|m7gd.*|m8g.*|m8gd.*|r6g.*|r6gd.*|r7g.*|r7gd.*|r8g.*|r8gd.*|t6g.*)
      ;;
    *)
      echo "Solomon product runs require an EC2 Graviton instance type; IMDS reported ${ec2_instance_type}." >&2
      exit 2
      ;;
  esac
fi
attention_batch_mode="${NSRL_SOLOMON_ATTENTION_BATCH_MODE:-map-reduce}"
attention_map_reduce_workers="${NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS:-0}"
attention_cpu_scaling_policy="${NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY:-auto-online-processors}"
if [[ "$attention_batch_mode" == "map-reduce" && "$attention_map_reduce_workers" == "0" ]]; then
  attention_map_reduce_auto_workers=1
  attention_effective_map_reduce_workers="$online_processors"
elif [[ "$attention_batch_mode" == "map-reduce" ]]; then
  attention_map_reduce_auto_workers=0
  attention_effective_map_reduce_workers="$attention_map_reduce_workers"
else
  attention_map_reduce_auto_workers=0
  attention_effective_map_reduce_workers=1
fi
attention_corpus_version="${NSRL_SOLOMON_ATTENTION_CORPUS_VERSION:-v2}"
attention_joint_corpus_version="${NSRL_SOLOMON_ATTENTION_JOINT_CORPUS_VERSION:-$attention_corpus_version}"
attention_text_token_profile="${NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE:-chunked}"
attention_image_token_profile="${NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE:-symbolic16}"
attention_joint_image_token_profile="${NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE:-$attention_image_token_profile}"
attention_seq_len="${NSRL_SOLOMON_ATTENTION_SEQ_LEN:-512}"
attention_eval_max_examples="${NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES:-none}"
attention_v2_curriculum_stages="${NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES:-identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind}"
attention_v2_curriculum_required_stages="${NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES:-$attention_v2_curriculum_stages}"
attention_v2_stage_epochs="${NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS:-1}"
attention_v2_native_bind_epochs="${NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS:-2}"
attention_require_image_token_profile="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE:-symbolic16}"
attention_require_image_token_channels="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS:-ink,edge,component,radial,direction}"
attention_require_image_channel_token_stats="${NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS:-1}"
attention_min_image_channel_distinct_bins="${NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS:-2}"
attention_require_directional_groups="${NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS:-1}"
attention_heldout_prompts="${NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS:-data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl}"
generative_prompts="${NSRL_SOLOMON_GENERATIVE_EVAL_PROMPTS:-$attention_heldout_prompts}"
attention_require_heldout_prompts="${NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS:-1}"
attention_min_heldout_prompt_rows="${NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS:-72}"
attention_min_match_yes_top1="${NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1:-72}"
attention_min_match_no_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1:-72}"
attention_min_match_no_image_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1:-72}"
attention_min_match_no_prompt_top1="${NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1:-72}"
attention_min_retrieval_margin="${NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN:-1}"
attention_require_identity_inference="${NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE:-1}"
attention_require_grounded_corpus="${NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS:-1}"
attention_min_source_overlap_tokens="${NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS:-2}"
attention_min_attribute_source_overlap_tokens="${NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS:-8}"
attention_max_source_placeholder_rows="${NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS:-0}"
attention_max_attribute_generic_rank_rows="${NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS:-0}"
attention_require_architecture_profile="${NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE:-1}"
attention_min_d_model="${NSRL_SOLOMON_V2_MIN_D_MODEL:-128}"
attention_min_heads="${NSRL_SOLOMON_V2_MIN_HEADS:-2}"
attention_min_hidden_dim="${NSRL_SOLOMON_V2_MIN_HIDDEN_DIM:-256}"
attention_min_transformer_layers="${NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS:-2}"
attention_min_context_seq_len="${NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN:-384}"
attention_denoiser_model="${NSRL_SOLOMON_ATTENTION_DENOISER_MODEL:-}"
if [[ "$attention_denoiser_model" == "none" || "$attention_denoiser_model" == "0" ]]; then
  attention_denoiser_model=""
elif [[ -z "$attention_denoiser_model" ]]; then
  if [[ -n "${NSRL_SOLOMON_DENOISE_MODEL:-}" ]]; then
    attention_denoiser_model="$sampler_model"
  elif has_stage denoiser; then
    attention_denoiser_model="$sampler_model"
  fi
fi
attention_require_denoise_bridge="${NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE:-0}"
if [[ -z "${NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE+x}" && -n "$attention_denoiser_model" ]]; then
  attention_require_denoise_bridge=1
fi
attention_denoise_max_output_retrieval_rank="${NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK:-1}"
attention_denoise_min_output_retrieval_margin="${NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN:-1}"
attention_denoise_min_unique_targets="${NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS:-2}"
attention_require_denoise_output_identity="${NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY:-$attention_require_denoise_bridge}"
attention_curriculum_require_stages="${NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES:-1}"
attention_require_confidence_trace="${NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE:-1}"
attention_require_promoted_small_profile="${NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE:-1}"
attention_generative_eval="${NSRL_SOLOMON_V2_GENERATIVE_EVAL:-}"
if [[ -z "$attention_generative_eval" ]] && has_stage generative-eval; then
  attention_generative_eval="${generative_out_dir}/${generative_run_name}"
fi
attention_require_generative_eval="${NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL:-0}"
if [[ -z "${NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL+x}" && -n "$attention_generative_eval" ]] && has_stage generative-eval; then
  attention_require_generative_eval=1
fi
attention_require_generative_output_identity="${NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY:-$attention_require_generative_eval}"
attention_min_generated_top5_16_per_mille="${NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE:-1}"
attention_min_generated_retrieval_top1_per_mille="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE:-1000}"
attention_min_generated_retrieval_top5_per_mille="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE:-1000}"
attention_min_generated_retrieval_margin="${NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN:-1}"
attention_min_generated_prompt_rows="${NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS:-72}"
attention_max_generated_mean_target_distance_q8="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_Q8:-0}"
attention_max_generated_mean_target_distance_16_q8="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8:-7000000}"
attention_max_generated_mean_target_distance_px_q8="${NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_PX_Q8:-0}"
attention_min_task_targets="${NSRL_SOLOMON_V2_MIN_TASK_TARGETS:-all=72}"
attention_min_task_top5_per_mille="${NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE:-all=1}"
attention_min_phase_targets="${NSRL_SOLOMON_V2_MIN_PHASE_TARGETS:-all=72}"
attention_min_direction_accuracy_per_mille="${NSRL_SOLOMON_V2_MIN_DIRECTION_ACCURACY_PER_MILLE:-}"
attention_min_direction_top5_per_mille="${NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE:-all=1}"
attention_min_direction_top10_per_mille="${NSRL_SOLOMON_V2_MIN_DIRECTION_TOP10_PER_MILLE:-}"

write_run_metadata

printf 'stage\tcommand\n' > "${pipeline_run_dir}/plan.tsv"

if [[ "${NSRL_PIPELINE_FETCH_INPUTS:-0}" != "0" ]]; then
  input_s3_uri="${NSRL_PIPELINE_INPUT_S3_URI:-}"
  if [[ -z "$input_s3_uri" ]]; then
    if [[ -z "${NSRL_S3_URI:-}" ]]; then
      echo "NSRL_PIPELINE_FETCH_INPUTS requires NSRL_PIPELINE_INPUT_S3_URI or NSRL_S3_URI" >&2
      exit 2
    fi
    input_s3_uri="${NSRL_S3_URI%/}/inputs"
  fi
  run_stage fetch-inputs aws s3 sync "$input_s3_uri" "$repo_root" --only-show-errors
fi

if has_stage dataset; then
  dataset_args=(node scripts/build-solomon-bitmap-denoise-dataset.mjs --out-dir "$denoise_dataset")
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_INPUT_MANIFEST --input-manifest
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_KINDS --kinds
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_CLEAN_AUGMENTATIONS --clean-augmentations
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_TARGET_CLEANING --target-cleaning
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_TARGET_CLEANING_STRENGTH --target-cleaning-strength
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_IMAGE_SIZE --image-size
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_CORRUPTIONS_PER_IMAGE --corruptions-per-image
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_TIMESTEPS --timesteps
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_EVAL_RATIO_PERMILLE --eval-ratio-permille
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_SEED --seed
  add_optional_dataset_arg NSRL_SOLOMON_DENOISE_PREVIEW_PAIRS --preview-pairs
  run_stage dataset "${dataset_args[@]}"
fi

if has_stage denoiser; then
  run_stage denoiser env \
    NSRL_SOLOMON_DENOISE_DATASET="$denoise_dataset" \
    NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR="$text_denoise_out_dir" \
    NSRL_SOLOMON_TEXT_DENOISE_MODEL="$text_denoise_model" \
    bash scripts/aws/run-solomon-text-denoiser-train.sh
fi

if has_stage prior; then
  run_stage prior env \
    NSRL_RUN_ROOT="$prior_run_root" \
    NSRL_RUN_NAME="$prior_run_name" \
    NSRL_SOLOMON_DENOISE_MODEL="$sampler_model" \
    bash scripts/aws/run-solomon-prior-smoke.sh
fi

if has_stage generative-eval; then
  generative_eval_args=(
    node scripts/run-solomon-generative-eval.mjs
    --out-dir "$generative_out_dir"
    --run-name "$generative_run_name"
    --sampler-model "$sampler_model"
    --latent-model "$generative_latent_model"
    --prompts "$generative_prompts"
    --partition eval
    --eval-permille "$generative_eval_permille"
    --limit "$generative_limit"
  )
  if [[ -n "$generative_retrieval_head" ]]; then
    generative_eval_args+=(--retrieval-head "$generative_retrieval_head")
  fi
  if [[ "$generative_require_retrieval_head" != "0" ]]; then
    generative_eval_args+=(--require-retrieval-head)
  fi
  run_stage generative-eval "${generative_eval_args[@]}"
fi

if has_stage multimodal; then
  run_stage multimodal env \
    NSRL_SOLOMON_MULTIMODAL_OUT_DIR="$multimodal_out_dir" \
    bash scripts/run-solomon-multimodal-smoke.sh
fi

if has_stage attention; then
  run_stage attention env \
    NSRL_SOLOMON_ATTENTION_OUT_DIR="$attention_out_dir" \
    NSRL_SOLOMON_ATTENTION_BATCH_MODE="$attention_batch_mode" \
    NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS="$attention_map_reduce_workers" \
    NSRL_SOLOMON_ATTENTION_CORPUS_VERSION="$attention_corpus_version" \
    NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE="$attention_text_token_profile" \
    NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE="$attention_image_token_profile" \
    NSRL_SOLOMON_ATTENTION_SEQ_LEN="$attention_seq_len" \
    NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES="$attention_eval_max_examples" \
    NSRL_SOLOMON_ATTENTION_DENOISER_MODEL="$attention_denoiser_model" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE="$attention_require_image_token_profile" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS="$attention_require_image_token_channels" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS="$attention_require_image_channel_token_stats" \
    NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS="$attention_min_image_channel_distinct_bins" \
    NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS="$attention_require_directional_groups" \
    NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS="$attention_heldout_prompts" \
    NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS="$attention_require_heldout_prompts" \
    NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS="$attention_min_heldout_prompt_rows" \
    NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1="$attention_min_match_yes_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1="$attention_min_match_no_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1="$attention_min_match_no_image_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1="$attention_min_match_no_prompt_top1" \
    NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN="$attention_min_retrieval_margin" \
    NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE="$attention_require_identity_inference" \
    NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS="$attention_require_grounded_corpus" \
    NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS="$attention_min_source_overlap_tokens" \
    NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS="$attention_min_attribute_source_overlap_tokens" \
    NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS="$attention_max_source_placeholder_rows" \
    NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS="$attention_max_attribute_generic_rank_rows" \
    NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE="$attention_require_architecture_profile" \
    NSRL_SOLOMON_V2_MIN_D_MODEL="$attention_min_d_model" \
    NSRL_SOLOMON_V2_MIN_HEADS="$attention_min_heads" \
    NSRL_SOLOMON_V2_MIN_HIDDEN_DIM="$attention_min_hidden_dim" \
    NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS="$attention_min_transformer_layers" \
    NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN="$attention_min_context_seq_len" \
    NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE="$attention_require_denoise_bridge" \
    NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY="$attention_require_denoise_output_identity" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK="$attention_denoise_max_output_retrieval_rank" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN="$attention_denoise_min_output_retrieval_margin" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS="$attention_denoise_min_unique_targets" \
    NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS="$attention_denoise_min_unique_targets" \
    NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE="$attention_require_confidence_trace" \
    NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE="$attention_require_promoted_small_profile" \
    NSRL_SOLOMON_V2_GENERATIVE_EVAL="$attention_generative_eval" \
    NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL="$attention_require_generative_eval" \
    NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY="$attention_require_generative_output_identity" \
    NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS="$attention_min_generated_prompt_rows" \
    NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE="$attention_min_generated_top5_16_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE="$attention_min_generated_retrieval_top1_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE="$attention_min_generated_retrieval_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN="$attention_min_generated_retrieval_margin" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_Q8="$attention_max_generated_mean_target_distance_q8" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8="$attention_max_generated_mean_target_distance_16_q8" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_PX_Q8="$attention_max_generated_mean_target_distance_px_q8" \
    NSRL_SOLOMON_V2_MIN_TASK_TARGETS="$attention_min_task_targets" \
    NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE="$attention_min_task_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_PHASE_TARGETS="$attention_min_phase_targets" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_ACCURACY_PER_MILLE="$attention_min_direction_accuracy_per_mille" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE="$attention_min_direction_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_TOP10_PER_MILLE="$attention_min_direction_top10_per_mille" \
    bash scripts/run-solomon-attention-smoke.sh
fi

if has_stage attention-curriculum; then
  run_stage attention-curriculum env \
    NSRL_SOLOMON_ATTENTION_CURRICULUM_OUT_DIR="$attention_curriculum_out_dir" \
    NSRL_SOLOMON_ATTENTION_BATCH_MODE="$attention_batch_mode" \
    NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS="$attention_map_reduce_workers" \
    NSRL_SOLOMON_ATTENTION_CORPUS_VERSION="$attention_corpus_version" \
    NSRL_SOLOMON_ATTENTION_JOINT_CORPUS_VERSION="$attention_joint_corpus_version" \
    NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE="$attention_text_token_profile" \
    NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE="$attention_image_token_profile" \
    NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE="$attention_joint_image_token_profile" \
    NSRL_SOLOMON_ATTENTION_SEQ_LEN="$attention_seq_len" \
    NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES="$attention_eval_max_examples" \
    NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES="$attention_v2_curriculum_stages" \
    NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES="$attention_v2_curriculum_required_stages" \
    NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS="$attention_v2_stage_epochs" \
    NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS="$attention_v2_native_bind_epochs" \
    NSRL_SOLOMON_ATTENTION_DENOISER_MODEL="$attention_denoiser_model" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE="$attention_require_image_token_profile" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS="$attention_require_image_token_channels" \
    NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS="$attention_require_image_channel_token_stats" \
    NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS="$attention_min_image_channel_distinct_bins" \
    NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS="$attention_require_directional_groups" \
    NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS="$attention_heldout_prompts" \
    NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS="$attention_require_heldout_prompts" \
    NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS="$attention_min_heldout_prompt_rows" \
    NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1="$attention_min_match_yes_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1="$attention_min_match_no_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1="$attention_min_match_no_image_top1" \
    NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1="$attention_min_match_no_prompt_top1" \
    NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN="$attention_min_retrieval_margin" \
    NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE="$attention_require_identity_inference" \
    NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS="$attention_require_grounded_corpus" \
    NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS="$attention_min_source_overlap_tokens" \
    NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS="$attention_min_attribute_source_overlap_tokens" \
    NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS="$attention_max_source_placeholder_rows" \
    NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS="$attention_max_attribute_generic_rank_rows" \
    NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE="$attention_require_architecture_profile" \
    NSRL_SOLOMON_V2_MIN_D_MODEL="$attention_min_d_model" \
    NSRL_SOLOMON_V2_MIN_HEADS="$attention_min_heads" \
    NSRL_SOLOMON_V2_MIN_HIDDEN_DIM="$attention_min_hidden_dim" \
    NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS="$attention_min_transformer_layers" \
    NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN="$attention_min_context_seq_len" \
    NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES="$attention_curriculum_require_stages" \
    NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE="$attention_require_denoise_bridge" \
    NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY="$attention_require_denoise_output_identity" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK="$attention_denoise_max_output_retrieval_rank" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN="$attention_denoise_min_output_retrieval_margin" \
    NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS="$attention_denoise_min_unique_targets" \
    NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS="$attention_denoise_min_unique_targets" \
    NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE="$attention_require_confidence_trace" \
    NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE="$attention_require_promoted_small_profile" \
    NSRL_SOLOMON_V2_GENERATIVE_EVAL="$attention_generative_eval" \
    NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL="$attention_require_generative_eval" \
    NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY="$attention_require_generative_output_identity" \
    NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS="$attention_min_generated_prompt_rows" \
    NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE="$attention_min_generated_top5_16_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE="$attention_min_generated_retrieval_top1_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE="$attention_min_generated_retrieval_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN="$attention_min_generated_retrieval_margin" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_Q8="$attention_max_generated_mean_target_distance_q8" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8="$attention_max_generated_mean_target_distance_16_q8" \
    NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_PX_Q8="$attention_max_generated_mean_target_distance_px_q8" \
    NSRL_SOLOMON_V2_MIN_TASK_TARGETS="$attention_min_task_targets" \
    NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE="$attention_min_task_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_PHASE_TARGETS="$attention_min_phase_targets" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_ACCURACY_PER_MILLE="$attention_min_direction_accuracy_per_mille" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE="$attention_min_direction_top5_per_mille" \
    NSRL_SOLOMON_V2_MIN_DIRECTION_TOP10_PER_MILLE="$attention_min_direction_top10_per_mille" \
    bash scripts/run-solomon-attention-curriculum-smoke.sh
fi

write_promotion_manifest
write_artifact_manifest
if [[ "$promotion_bundle_check" != "0" ]]; then
  run_stage promotion-bundle-check node scripts/check-solomon-promotion-bundle.mjs \
    --promotion "$promotion_manifest" \
    --out "${pipeline_run_dir}/promotion-bundle-check.json"
  write_artifact_manifest
fi
write_completion_report
write_artifact_manifest
sync_pipeline_artifacts

echo "pipeline_run_dir: $pipeline_run_dir"
echo "artifact_manifest: ${pipeline_run_dir}/artifacts.tsv"
echo "promotion_manifest: $promotion_manifest"
if [[ -f "${pipeline_run_dir}/promotion-bundle-check.json" ]]; then
  echo "promotion_bundle_check: ${pipeline_run_dir}/promotion-bundle-check.json"
fi
echo "pipeline_complete: $pipeline_complete_report"
if [[ "$dry_run" != "0" ]]; then
  echo "dry_run_plan: ${pipeline_run_dir}/plan.tsv"
fi
if [[ -n "$s3_pipeline_uri" ]]; then
  echo "s3_pipeline: $s3_pipeline_uri"
fi
