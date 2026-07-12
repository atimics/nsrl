#!/usr/bin/env node

import childProcess from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_run_artifacts_check.v1";
const completionSchema = "nsrl.solomon_aws_pipeline_complete.v1";
const promotionCheckSchema = "nsrl.solomon_promotion_bundle_check.v1";
const requiredProductStages = [
  "dataset",
  "denoiser",
  "prior",
  "generative-eval",
  "attention-curriculum",
];
const requiredStages = [...requiredProductStages, "promotion-bundle-check"];
const requiredArtifactNames = [
  "run_env",
  "plan",
  "artifacts",
  "promotion_manifest",
  "pipeline_complete",
  "promotion_bundle_check",
  "quality_report",
  "model",
  "corpus_manifest",
  "attention_eval",
  "retrieval_head",
  "retrieval_head_eval",
  "curriculum_stages",
  "sample_binding",
  "identity_inference",
  "grounded_corpus",
  "generation_integrity",
  "denoise_bridge",
  "denoise_generation_integrity",
  "run",
  "summary",
];
const requiredCurriculumStages = [
  "identity",
  "image",
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "explain",
  "hard-negative",
  "native-bind",
];
const requiredNativeEvalTasks = [
  "canonical-joint",
  "identify",
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
  "match",
];
const requiredNativeEvalPhases = ["special", "prompt", "text", "image"];
const requiredDirectionalGroups = [
  "text_prompt_to_image_plan",
  "seal_image_to_text",
  "text_and_seal_to_explanation",
  "identity_source_binding",
];
const requiredNativeBindEpochs = 2;
const requiredImageTokenChannels = ["ink", "edge", "component", "radial", "direction"];
const gravitonInstance = /^(?:c|m|r|t)(?:6|7|8)g[dn]?\./;
const ec2InstanceIdPattern = /^i-[0-9a-f]{8,}$/;

const defaults = {
  runDir: "",
  outPath: "",
  requireRealRun: true,
  requireGravitonRunner: true,
  requireS3Artifacts: true,
  requireCompletionReport: true,
  requireEc2Metadata: true,
  validatePromotionBundle: true,
  requireProductConfig: true,
};

function usage() {
  console.log([
    "Usage: check-solomon-aws-run-artifacts.mjs --run-dir PATH [options]",
    "",
    "Checks a completed Solomon AWS product run after the pipeline artifacts have",
    "been synced locally. By default it requires a real Graviton/S3 run, all",
    "stage status files at status=0, pipeline-complete.json, and a passing",
    "promotion-bundle check against the run promotion.tsv.",
    "",
    "Options:",
    "  --out PATH",
    "  --allow-dry-run",
    "  --allow-non-graviton-runner",
    "  --allow-missing-s3-artifacts",
    "  --allow-missing-completion-report",
    "  --allow-missing-ec2-metadata",
    "  --skip-promotion-bundle-validation",
    "  --allow-weak-product-config",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--run-dir") {
      config.runDir = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--allow-dry-run") {
      config.requireRealRun = false;
    } else if (arg === "--allow-non-graviton-runner") {
      config.requireGravitonRunner = false;
    } else if (arg === "--allow-missing-s3-artifacts") {
      config.requireS3Artifacts = false;
    } else if (arg === "--allow-missing-completion-report") {
      config.requireCompletionReport = false;
    } else if (arg === "--allow-missing-ec2-metadata") {
      config.requireEc2Metadata = false;
    } else if (arg === "--skip-promotion-bundle-validation") {
      config.validatePromotionBundle = false;
    } else if (arg === "--allow-weak-product-config") {
      config.requireProductConfig = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.runDir) {
    throw new Error("--run-dir is required");
  }
  config.runDir = path.resolve(config.runDir);
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readKeyValueFile(filePath) {
  const rows = {};
  const text = fs.readFileSync(filePath, "utf8");
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) {
      continue;
    }
    const equals = line.indexOf("=");
    if (equals < 0) {
      throw new Error(`${filePath}:${index + 1}: expected key=value`);
    }
    rows[line.slice(0, equals)] = line.slice(equals + 1);
  }
  return rows;
}

function readTsv(filePath, expectedHeader) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  const lines = text ? text.split(/\r?\n/) : [];
  if (lines[0] !== expectedHeader) {
    throw new Error(`${filePath} must start with ${expectedHeader.replace(/\t/g, "\\t")}`);
  }
  const keys = expectedHeader.split("\t");
  return lines.slice(1).filter((line) => line.trim()).map((line, index) => {
    const fields = line.split("\t");
    if (fields.length !== keys.length) {
      throw new Error(`${filePath}:${index + 2}: expected ${keys.length} tab-separated fields`);
    }
    return Object.fromEntries(keys.map((key, fieldIndex) => [key, fields[fieldIndex]]));
  });
}

function readJsonIfPresent(filePath, errors, label) {
  if (!fs.existsSync(filePath)) {
    errors.push(`${label} ${filePath} is missing`);
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    errors.push(`${label} ${filePath} is not valid JSON: ${error.message}`);
    return null;
  }
}

function resolveRunPath(runDir, ref) {
  if (!ref) {
    return "";
  }
  const candidates = [];
  if (path.isAbsolute(ref)) {
    candidates.push(ref);
    const parts = path.resolve(ref).split(path.sep).filter(Boolean);
    const runName = path.basename(runDir);
    const runIndex = parts.lastIndexOf(runName);
    if (runIndex >= 0) {
      candidates.push(path.join(runDir, ...parts.slice(runIndex + 1)));
    }
  } else {
    candidates.push(path.join(runDir, ref), path.resolve(ref));
  }
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0];
}

function hasS3Uri(value) {
  return String(value || "").startsWith("s3://");
}

function parseList(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseInteger(value) {
  const text = String(value ?? "").trim();
  if (!/^-?\d+$/.test(text)) {
    return Number.NaN;
  }
  return Number(text);
}

function booleanish(value) {
  return value === true || value === "1" || value === "true";
}

function attentionConfigFromEnv(env) {
  return {
    corpus_version: env.attention_corpus_version || "",
    joint_corpus_version: env.attention_joint_corpus_version || "",
    text_token_profile: env.attention_text_token_profile || "",
    image_token_profile: env.attention_image_token_profile || "",
    joint_image_token_profile: env.attention_joint_image_token_profile || "",
    seq_len: parseInteger(env.attention_seq_len),
    curriculum_stages: parseList(env.attention_v2_curriculum_stages),
    curriculum_required_stages: parseList(env.attention_v2_curriculum_required_stages),
    v2_stage_epochs: parseInteger(env.attention_v2_stage_epochs),
    native_bind_epochs: parseInteger(env.attention_v2_native_bind_epochs),
    image_token_channels: parseList(env.attention_require_image_token_channels),
    require_image_channel_token_stats: booleanish(env.attention_require_image_channel_token_stats),
    require_directional_groups: booleanish(env.attention_require_directional_groups),
    heldout_prompts: env.attention_heldout_prompts || "",
    min_heldout_prompt_rows: parseInteger(env.attention_min_heldout_prompt_rows),
    min_task_targets: env.attention_min_task_targets || "",
    min_phase_targets: env.attention_min_phase_targets || "",
    denoise_min_unique_targets: parseInteger(env.attention_denoise_min_unique_targets),
    generation: {
      generative_eval: env.attention_generative_eval || "",
      require_generative_eval: booleanish(env.attention_require_generative_eval),
      require_generative_output_identity: booleanish(env.attention_require_generative_output_identity),
      min_generated_prompt_rows: parseInteger(env.attention_min_generated_prompt_rows),
      min_generated_top5_16_per_mille: parseInteger(env.attention_min_generated_top5_16_per_mille),
      min_generated_retrieval_top1_per_mille: parseInteger(env.attention_min_generated_retrieval_top1_per_mille),
      min_generated_retrieval_top5_per_mille: parseInteger(env.attention_min_generated_retrieval_top5_per_mille),
      min_generated_retrieval_margin: parseInteger(env.attention_min_generated_retrieval_margin),
      max_generated_mean_target_distance_16_q8: parseInteger(env.attention_max_generated_mean_target_distance_16_q8),
    },
    cpu_scaling: {
      batch_mode: env.attention_batch_mode || "",
      map_reduce_workers: env.attention_map_reduce_workers || "",
      policy: env.attention_cpu_scaling_policy || "",
      auto_workers: booleanish(env.attention_map_reduce_auto_workers),
      effective_workers: parseInteger(env.attention_effective_map_reduce_workers),
    },
  };
}

function attentionConfigFromCompletion(completion) {
  const attention = completion?.product_config?.attention || {};
  const cpuScaling = attention.cpu_scaling || {};
  const curriculum = attention.curriculum || {};
  const generation = attention.generation || {};
  return {
    corpus_version: attention.corpus_version || "",
    joint_corpus_version: attention.joint_corpus_version || "",
    text_token_profile: attention.text_token_profile || "",
    image_token_profile: attention.image_token_profile || "",
    joint_image_token_profile: attention.joint_image_token_profile || "",
    seq_len: Number(attention.seq_len || 0),
    curriculum_stages: Array.isArray(curriculum.stages) ? curriculum.stages : [],
    curriculum_required_stages: Array.isArray(curriculum.required_stages) ? curriculum.required_stages : [],
    v2_stage_epochs: Number(curriculum.stage_epochs || 0),
    native_bind_epochs: Number(curriculum.native_bind_epochs || 0),
    image_token_channels: Array.isArray(attention.image_token_channels) ? attention.image_token_channels : [],
    require_image_channel_token_stats: attention.require_image_channel_token_stats === true,
    require_directional_groups: attention.require_directional_groups === true,
    heldout_prompts: attention.heldout_prompts || "",
    min_heldout_prompt_rows: Number(attention.min_heldout_prompt_rows || 0),
    min_task_targets: attention.min_task_targets || "",
    min_phase_targets: attention.min_phase_targets || "",
    denoise_min_unique_targets: Number(attention.denoise_min_unique_targets || 0),
    generation: {
      generative_eval: generation.generative_eval || "",
      require_generative_eval: generation.require_generative_eval === true,
      require_generative_output_identity: generation.require_generative_output_identity === true,
      min_generated_prompt_rows: Number(generation.min_generated_prompt_rows || 0),
      min_generated_top5_16_per_mille: Number(generation.min_generated_top5_16_per_mille || 0),
      min_generated_retrieval_top1_per_mille: Number(generation.min_generated_retrieval_top1_per_mille || 0),
      min_generated_retrieval_top5_per_mille: Number(generation.min_generated_retrieval_top5_per_mille || 0),
      min_generated_retrieval_margin: Number(generation.min_generated_retrieval_margin || 0),
      max_generated_mean_target_distance_16_q8: Number(generation.max_generated_mean_target_distance_16_q8 || 0),
    },
    cpu_scaling: {
      batch_mode: cpuScaling.batch_mode || "",
      map_reduce_workers: String(cpuScaling.map_reduce_workers ?? ""),
      policy: cpuScaling.policy || "",
      auto_workers: cpuScaling.auto_workers === true,
      effective_workers: Number(cpuScaling.effective_workers || 0),
    },
  };
}

function requireIncludesAll(actual, expected, label, errors) {
  const missing = expected.filter((item) => !actual.includes(item));
  if (missing.length > 0) {
    errors.push(`${label} missing ${missing.join(",")}`);
  }
}

function requireEqual(actual, expected, label, errors) {
  if (actual !== expected) {
    errors.push(`${label} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function requireArrayEqual(actual, expected, label, errors) {
  if (actual.length !== expected.length || actual.some((item, index) => item !== expected[index])) {
    errors.push(`${label} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function validateProductAttentionConfig(attention, label, runnerProcessorCount, config, errors) {
  if (!config.requireProductConfig) {
    return;
  }
  requireEqual(attention.corpus_version, "v2", `${label}.corpus_version`, errors);
  requireEqual(attention.joint_corpus_version, "v2", `${label}.joint_corpus_version`, errors);
  requireEqual(attention.text_token_profile, "chunked", `${label}.text_token_profile`, errors);
  requireEqual(attention.image_token_profile, "symbolic16", `${label}.image_token_profile`, errors);
  requireEqual(attention.joint_image_token_profile, "symbolic16", `${label}.joint_image_token_profile`, errors);
  if (!Number.isInteger(attention.seq_len) || attention.seq_len < 384 || attention.seq_len > 768) {
    errors.push(`${label}.seq_len ${JSON.stringify(attention.seq_len)} is not in [384, 768]`);
  }
  requireIncludesAll(attention.curriculum_stages, requiredCurriculumStages, `${label}.curriculum.stages`, errors);
  requireIncludesAll(attention.curriculum_required_stages, requiredCurriculumStages, `${label}.curriculum.required_stages`, errors);
  if (!Number.isInteger(attention.v2_stage_epochs) || attention.v2_stage_epochs !== 1) {
    errors.push(`${label}.curriculum.stage_epochs ${JSON.stringify(attention.v2_stage_epochs)} != 1`);
  }
  if (!Number.isInteger(attention.native_bind_epochs) || attention.native_bind_epochs < requiredNativeBindEpochs) {
    errors.push(
      `${label}.curriculum.native_bind_epochs ${JSON.stringify(attention.native_bind_epochs)} < ${requiredNativeBindEpochs}`,
    );
  }
  requireIncludesAll(attention.image_token_channels, requiredImageTokenChannels, `${label}.image_token_channels`, errors);
  if (attention.require_image_channel_token_stats !== true) {
    errors.push(`${label}.require_image_channel_token_stats is not true`);
  }
  if (attention.require_directional_groups !== true) {
    errors.push(`${label}.require_directional_groups is not true`);
  }
  if (!attention.heldout_prompts) {
    errors.push(`${label}.heldout_prompts is missing`);
  }
  if (!Number.isInteger(attention.min_heldout_prompt_rows) || attention.min_heldout_prompt_rows < 72) {
    errors.push(`${label}.min_heldout_prompt_rows ${JSON.stringify(attention.min_heldout_prompt_rows)} < 72`);
  }
  if (attention.min_task_targets !== "all=72") {
    errors.push(`${label}.min_task_targets ${JSON.stringify(attention.min_task_targets)} != "all=72"`);
  }
  if (attention.min_phase_targets !== "all=72") {
    errors.push(`${label}.min_phase_targets ${JSON.stringify(attention.min_phase_targets)} != "all=72"`);
  }
  if (!Number.isInteger(attention.denoise_min_unique_targets) || attention.denoise_min_unique_targets < 2) {
    errors.push(`${label}.denoise_min_unique_targets ${JSON.stringify(attention.denoise_min_unique_targets)} < 2`);
  }

  const generation = attention.generation || {};
  if (!generation.generative_eval) {
    errors.push(`${label}.generation.generative_eval is missing`);
  }
  if (generation.require_generative_eval !== true) {
    errors.push(`${label}.generation.require_generative_eval is not true`);
  }
  if (generation.require_generative_output_identity !== true) {
    errors.push(`${label}.generation.require_generative_output_identity is not true`);
  }
  if (!Number.isInteger(generation.min_generated_prompt_rows) || generation.min_generated_prompt_rows < 72) {
    errors.push(`${label}.generation.min_generated_prompt_rows ${JSON.stringify(generation.min_generated_prompt_rows)} < 72`);
  }
  if (!Number.isInteger(generation.min_generated_top5_16_per_mille) || generation.min_generated_top5_16_per_mille < 1) {
    errors.push(`${label}.generation.min_generated_top5_16_per_mille ${JSON.stringify(generation.min_generated_top5_16_per_mille)} < 1`);
  }
  if (
    !Number.isInteger(generation.min_generated_retrieval_top1_per_mille) ||
    generation.min_generated_retrieval_top1_per_mille < 1000
  ) {
    errors.push(
      `${label}.generation.min_generated_retrieval_top1_per_mille ${JSON.stringify(generation.min_generated_retrieval_top1_per_mille)} < 1000`,
    );
  }
  if (
    !Number.isInteger(generation.min_generated_retrieval_top5_per_mille) ||
    generation.min_generated_retrieval_top5_per_mille < 1000
  ) {
    errors.push(
      `${label}.generation.min_generated_retrieval_top5_per_mille ${JSON.stringify(generation.min_generated_retrieval_top5_per_mille)} < 1000`,
    );
  }
  if (
    !Number.isInteger(generation.min_generated_retrieval_margin) ||
    generation.min_generated_retrieval_margin < 1
  ) {
    errors.push(
      `${label}.generation.min_generated_retrieval_margin ${JSON.stringify(generation.min_generated_retrieval_margin)} < 1`,
    );
  }

  const cpu = attention.cpu_scaling || {};
  requireEqual(cpu.batch_mode, "map-reduce", `${label}.cpu_scaling.batch_mode`, errors);
  requireEqual(cpu.map_reduce_workers, "0", `${label}.cpu_scaling.map_reduce_workers`, errors);
  requireEqual(cpu.policy, "auto-online-processors", `${label}.cpu_scaling.policy`, errors);
  if (cpu.auto_workers !== true) {
    errors.push(`${label}.cpu_scaling.auto_workers is not true`);
  }
  if (!Number.isInteger(cpu.effective_workers) || cpu.effective_workers < 1) {
    errors.push(`${label}.cpu_scaling.effective_workers ${JSON.stringify(cpu.effective_workers)} < 1`);
  }
  if (Number.isInteger(runnerProcessorCount) && runnerProcessorCount > 0 && cpu.effective_workers !== runnerProcessorCount) {
    errors.push(`${label}.cpu_scaling.effective_workers ${cpu.effective_workers} != runner online processors ${runnerProcessorCount}`);
  }
}

function compareProductAttentionConfig(envAttention, completionAttention, config, errors) {
  if (!config.requireProductConfig) {
    return;
  }
  const scalarKeys = [
    "corpus_version",
    "joint_corpus_version",
    "text_token_profile",
    "image_token_profile",
    "joint_image_token_profile",
    "seq_len",
    "require_image_channel_token_stats",
    "require_directional_groups",
    "heldout_prompts",
    "min_heldout_prompt_rows",
    "min_task_targets",
    "min_phase_targets",
    "denoise_min_unique_targets",
    "v2_stage_epochs",
    "native_bind_epochs",
  ];
  for (const key of scalarKeys) {
    requireEqual(
      envAttention[key],
      completionAttention[key],
      `product_config mismatch: run.env.attention.${key} vs pipeline-complete.product_config.attention.${key}`,
      errors,
    );
  }
  for (const key of ["curriculum_stages", "curriculum_required_stages", "image_token_channels"]) {
    requireArrayEqual(
      envAttention[key],
      completionAttention[key],
      `product_config mismatch: run.env.attention.${key} vs pipeline-complete.product_config.attention.${key}`,
      errors,
    );
  }
  for (const key of [
    "generative_eval",
    "require_generative_eval",
    "require_generative_output_identity",
    "min_generated_prompt_rows",
    "min_generated_top5_16_per_mille",
    "min_generated_retrieval_top1_per_mille",
    "min_generated_retrieval_top5_per_mille",
    "min_generated_retrieval_margin",
    "max_generated_mean_target_distance_16_q8",
  ]) {
    requireEqual(
      envAttention.generation?.[key],
      completionAttention.generation?.[key],
      `product_config mismatch: run.env.attention.generation.${key} vs pipeline-complete.product_config.attention.generation.${key}`,
      errors,
    );
  }
  for (const key of ["batch_mode", "map_reduce_workers", "policy", "auto_workers", "effective_workers"]) {
    requireEqual(
      envAttention.cpu_scaling?.[key],
      completionAttention.cpu_scaling?.[key],
      `product_config mismatch: run.env.attention.cpu_scaling.${key} vs pipeline-complete.product_config.attention.cpu_scaling.${key}`,
      errors,
    );
  }
}

function samePathish(left, right, runDir) {
  if (!left || !right) {
    return false;
  }
  return normalize(resolveRunPath(runDir, left)) === normalize(resolveRunPath(runDir, right));
}

function normalize(filePath) {
  try {
    return fs.realpathSync.native(path.resolve(filePath));
  } catch (_error) {
    return path.resolve(filePath);
  }
}

function requireFile(filePath, label, errors) {
  if (!fs.existsSync(filePath)) {
    errors.push(`${label} ${filePath} is missing`);
    return false;
  }
  return true;
}

function checkRunEnv(env, config, errors) {
  if (env.schema !== "nsrl.solomon_aws_pipeline.v1") {
    errors.push(`run.env schema ${JSON.stringify(env.schema || "")} != nsrl.solomon_aws_pipeline.v1`);
  }
  if (config.requireRealRun && env.dry_run !== "0") {
    errors.push(`run.env dry_run ${JSON.stringify(env.dry_run || "")} != "0"`);
  }
  if (config.requireGravitonRunner) {
    if (env.require_graviton !== "1") {
      errors.push(`run.env require_graviton ${JSON.stringify(env.require_graviton || "")} != "1"`);
    }
    if (env.runner_kernel !== "Linux") {
      errors.push(`run.env runner_kernel ${JSON.stringify(env.runner_kernel || "")} != "Linux"`);
    }
    if (!["aarch64", "arm64"].includes(env.runner_arch || "")) {
      errors.push(`run.env runner_arch ${JSON.stringify(env.runner_arch || "")} is not ARM64/Graviton`);
    }
  }
  if (config.requireEc2Metadata) {
    if (env.ec2_metadata_required !== "1") {
      errors.push(`run.env ec2_metadata_required ${JSON.stringify(env.ec2_metadata_required || "")} != "1"`);
    }
    if (!ec2InstanceIdPattern.test(env.ec2_instance_id || "")) {
      errors.push(`run.env ec2_instance_id ${JSON.stringify(env.ec2_instance_id || "")} is not a valid EC2 instance id`);
    }
    if (!gravitonInstance.test(env.ec2_instance_type || "")) {
      errors.push(`run.env ec2_instance_type ${JSON.stringify(env.ec2_instance_type || "")} is not a Graviton EC2 family`);
    }
  }
  if (config.requireS3Artifacts) {
    if (env.require_s3_artifacts !== "1") {
      errors.push(`run.env require_s3_artifacts ${JSON.stringify(env.require_s3_artifacts || "")} != "1"`);
    }
    if (!hasS3Uri(env.s3_uri)) {
      errors.push(`run.env s3_uri ${JSON.stringify(env.s3_uri || "")} must start with s3://`);
    }
    if (!hasS3Uri(env.s3_pipeline_uri)) {
      errors.push(`run.env s3_pipeline_uri ${JSON.stringify(env.s3_pipeline_uri || "")} must start with s3://`);
    }
    const expectedPrefix = `${String(env.s3_uri || "").replace(/\/+$/, "")}/pipelines/`;
    if (env.s3_pipeline_uri && env.s3_uri && !String(env.s3_pipeline_uri).startsWith(expectedPrefix)) {
      errors.push(`run.env s3_pipeline_uri ${env.s3_pipeline_uri} is not under ${expectedPrefix}`);
    }
  }
  if (env.promotion_bundle_check !== "1") {
    errors.push(`run.env promotion_bundle_check ${JSON.stringify(env.promotion_bundle_check || "")} != "1"`);
  }
}

function checkPlan(planRows, errors) {
  const stages = planRows.map((row) => row.stage);
  for (const stage of requiredStages) {
    if (!stages.includes(stage)) {
      errors.push(`plan.tsv missing stage ${stage}`);
    }
  }
  return stages;
}

function checkStatusFiles(runDir, errors, config) {
  const summaries = {};
  for (const stage of requiredStages) {
    const statusPath = path.join(runDir, "logs", `${stage}.status`);
    if (!requireFile(statusPath, `${stage} status`, errors)) {
      continue;
    }
    let status = {};
    try {
      status = readKeyValueFile(statusPath);
    } catch (error) {
      errors.push(error.message);
      continue;
    }
    summaries[stage] = {
      path: statusPath,
      status: status.status || "",
      dry_run: status.dry_run || "0",
      started_at: status.started_at || "",
      finished_at: status.finished_at || "",
    };
    if (status.status !== "0") {
      errors.push(`${stage} status ${JSON.stringify(status.status || "")} != "0"`);
    }
    if (config.requireRealRun && status.dry_run === "1") {
      errors.push(`${stage} status is a dry-run status`);
    }
    if (!status.started_at || !status.finished_at) {
      errors.push(`${stage} status missing started_at or finished_at`);
    }
  }
  return summaries;
}

function checkArtifacts(runDir, artifactRows, errors) {
  const summary = {};
  const byName = new Map(artifactRows.map((row) => [row.artifact, row]));
  for (const artifact of requiredArtifactNames) {
    const row = byName.get(artifact);
    if (!row) {
      errors.push(`artifacts.tsv missing ${artifact}`);
      continue;
    }
    const resolved = resolveRunPath(runDir, row.path);
    summary[artifact] = {
      stage: row.stage,
      path: row.path,
      resolved,
      present: fs.existsSync(resolved),
    };
    if (!fs.existsSync(resolved)) {
      errors.push(`artifact ${artifact} ${resolved} is missing`);
    }
  }
  return summary;
}

function fileSha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function directoryFiles(dirPath) {
  const files = [];
  const stack = [dirPath];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(absolute);
      } else if (entry.isFile()) {
        files.push(absolute);
      }
    }
  }
  return files.sort((left, right) => left.localeCompare(right));
}

function syncedArtifactEntry(runDir, row) {
  const resolved = resolveRunPath(runDir, row.path);
  const base = {
    stage: row.stage,
    artifact: row.artifact,
    path: row.path,
    present: fs.existsSync(resolved),
    type: "missing",
    sha256: "",
    file_count: 0,
  };
  if (!base.present) {
    return base;
  }
  const stat = fs.statSync(resolved);
  if (stat.isDirectory()) {
    const files = directoryFiles(resolved).map((filePath) => ({
      path: path.relative(resolved, filePath).split(path.sep).join("/"),
      sha256: fileSha256(filePath),
    }));
    return {
      ...base,
      type: "directory",
      sha256: crypto.createHash("sha256").update(JSON.stringify(files)).digest("hex"),
      file_count: files.length,
    };
  }
  if (stat.isFile()) {
    return {
      ...base,
      type: "file",
      sha256: fileSha256(resolved),
      file_count: 1,
    };
  }
  return {
    ...base,
    type: "other",
  };
}

function summarizeSyncedArtifacts(runDir, artifactRows) {
  const entries = artifactRows
    .map((row) => syncedArtifactEntry(runDir, row))
    .sort((left, right) =>
      left.stage.localeCompare(right.stage) ||
      left.artifact.localeCompare(right.artifact) ||
      left.path.localeCompare(right.path)
    );
  return {
    schema: "nsrl.solomon_synced_artifacts.v1",
    artifact_count: entries.length,
    present_count: entries.filter((entry) => entry.present).length,
    file_count: entries.reduce((total, entry) => total + Number(entry.file_count || 0), 0),
    sha256: crypto.createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    entries,
  };
}

function requiredPromotionRows(promotionRows) {
  return promotionRows.filter((row) =>
    row.product === "solomon-v1" &&
    (row.required === "1" || row.required === "true")
  );
}

function checkPromotionArtifactIndex(runDir, promotionRows, artifactRows, errors) {
  const summary = {
    required: 0,
    indexed: 0,
    path_matched: 0,
    missing: [],
    mismatched: [],
  };
  const byStageArtifact = new Map();
  for (const row of artifactRows) {
    const key = `${row.stage}\0${row.artifact}`;
    if (!byStageArtifact.has(key)) {
      byStageArtifact.set(key, []);
    }
    byStageArtifact.get(key).push(row);
  }

  for (const promotionRow of requiredPromotionRows(promotionRows)) {
    summary.required += 1;
    const key = `${promotionRow.stage}\0${promotionRow.artifact}`;
    const candidates = byStageArtifact.get(key) || [];
    const label = `${promotionRow.stage}/${promotionRow.artifact}`;
    if (candidates.length === 0) {
      summary.missing.push(label);
      errors.push(`artifacts.tsv missing required promotion artifact ${label}`);
      continue;
    }
    summary.indexed += 1;
    if (!candidates.some((candidate) => samePathish(candidate.path, promotionRow.path, runDir))) {
      summary.mismatched.push(label);
      errors.push(`artifacts.tsv ${label} does not point at promotion.tsv path ${promotionRow.path}`);
      continue;
    }
    summary.path_matched += 1;
  }
  summary.ok = summary.required > 0 &&
    summary.indexed === summary.required &&
    summary.path_matched === summary.required &&
    summary.missing.length === 0 &&
    summary.mismatched.length === 0;
  return summary;
}

function checkCompletionPromotionArtifacts(runDir, completion, promotionRows, errors) {
  const summary = {
    required: 0,
    present: 0,
    path_matched: 0,
    missing: [],
    mismatched: [],
  };
  if (!completion || typeof completion !== "object") {
    return summary;
  }
  const artifacts = completion.artifacts || {};
  for (const promotionRow of requiredPromotionRows(promotionRows)) {
    summary.required += 1;
    const label = `${promotionRow.stage}/${promotionRow.artifact}`;
    const completionPath = artifacts[promotionRow.artifact];
    if (!completionPath) {
      summary.missing.push(label);
      errors.push(`pipeline-complete artifacts missing required promotion artifact ${label}`);
      continue;
    }
    summary.present += 1;
    if (!samePathish(completionPath, promotionRow.path, runDir)) {
      summary.mismatched.push(label);
      errors.push(`pipeline-complete artifacts ${label} does not match promotion.tsv path ${promotionRow.path}`);
      continue;
    }
    summary.path_matched += 1;
  }
  summary.ok = summary.required > 0 &&
    summary.present === summary.required &&
    summary.path_matched === summary.required &&
    summary.missing.length === 0 &&
    summary.mismatched.length === 0;
  return summary;
}

function checkCompletionReport(completion, config, env, errors) {
  if (!completion) {
    return {};
  }
  if (completion.schema !== completionSchema) {
    errors.push(`pipeline-complete schema ${JSON.stringify(completion.schema || "")} != ${completionSchema}`);
  }
  if (completion.ok !== true) {
    errors.push("pipeline-complete ok is not true");
  }
  if (config.requireRealRun && completion.dry_run !== false) {
    errors.push("pipeline-complete dry_run is not false");
  }
  if (completion.run_name && env.run_name && completion.run_name !== env.run_name) {
    errors.push(`pipeline-complete run_name ${completion.run_name} != run.env ${env.run_name}`);
  }
  const stages = Array.isArray(completion.stages) ? completion.stages : [];
  for (const stage of requiredStages) {
    if (!stages.includes(stage)) {
      errors.push(`pipeline-complete missing stage ${stage}`);
    }
  }
  const runner = completion.runner || {};
  if (config.requireGravitonRunner) {
    if (runner.kernel !== "Linux") {
      errors.push(`pipeline-complete runner.kernel ${JSON.stringify(runner.kernel || "")} != "Linux"`);
    }
    if (!["aarch64", "arm64"].includes(runner.arch || "")) {
      errors.push(`pipeline-complete runner.arch ${JSON.stringify(runner.arch || "")} is not ARM64/Graviton`);
    }
    if (runner.require_graviton !== true) {
      errors.push("pipeline-complete runner.require_graviton is not true");
    }
  }
  if (config.requireEc2Metadata) {
    const ec2 = runner.ec2 || {};
    if (ec2.metadata_required !== true) {
      errors.push("pipeline-complete runner.ec2.metadata_required is not true");
    }
    if (!ec2InstanceIdPattern.test(ec2.instance_id || "")) {
      errors.push(`pipeline-complete runner.ec2.instance_id ${JSON.stringify(ec2.instance_id || "")} is not a valid EC2 instance id`);
    }
    if (!gravitonInstance.test(ec2.instance_type || "")) {
      errors.push(`pipeline-complete runner.ec2.instance_type ${JSON.stringify(ec2.instance_type || "")} is not a Graviton EC2 family`);
    }
  }
  const s3 = completion.s3 || {};
  if (config.requireS3Artifacts) {
    if (s3.required !== true) {
      errors.push("pipeline-complete s3.required is not true");
    }
    if (!hasS3Uri(s3.uri)) {
      errors.push(`pipeline-complete s3.uri ${JSON.stringify(s3.uri || "")} must start with s3://`);
    }
    if (!hasS3Uri(s3.pipeline_uri)) {
      errors.push(`pipeline-complete s3.pipeline_uri ${JSON.stringify(s3.pipeline_uri || "")} must start with s3://`);
    }
  }
  const productConfig = attentionConfigFromCompletion(completion);
  validateProductAttentionConfig(productConfig, "pipeline-complete.product_config.attention", Number(runner.online_processors || 0), config, errors);
  return {
    run_name: completion.run_name || "",
    dry_run: completion.dry_run === true,
    stages,
    runner,
    s3,
    product_config: {
      attention: productConfig,
    },
  };
}

function checkPromotionBundle(config, promotionPath, promotionCheckPath, errors) {
  const existing = readJsonIfPresent(promotionCheckPath, errors, "promotion-bundle-check");
  if (existing) {
    if (existing.schema !== promotionCheckSchema) {
      errors.push(`promotion-bundle-check schema ${JSON.stringify(existing.schema || "")} != ${promotionCheckSchema}`);
    }
    if (existing.ok !== true) {
      errors.push("promotion-bundle-check ok is not true");
    }
  }
  if (!config.validatePromotionBundle) {
    return existing || {};
  }

  const scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-aws-run-artifacts-"));
  const outPath = path.join(scratchDir, "promotion-bundle-check.json");
  try {
    const result = childProcess.spawnSync(process.execPath, [
      "scripts/check-solomon-promotion-bundle.mjs",
      "--promotion",
      promotionPath,
      "--out",
      outPath,
    ], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    const report = fs.existsSync(outPath)
      ? JSON.parse(fs.readFileSync(outPath, "utf8"))
      : extractJsonObjects(result.stdout || "").find((item) => item.schema === promotionCheckSchema);
    if (result.status !== 0) {
      errors.push(`promotion bundle validation exited with status ${result.status}`);
    }
    if (!report) {
      errors.push("promotion bundle validation did not emit a JSON report");
      return existing || {};
    }
    if (report.schema !== promotionCheckSchema) {
      errors.push(`promotion bundle validation schema ${JSON.stringify(report.schema || "")} != ${promotionCheckSchema}`);
    }
    if (report.ok !== true) {
      errors.push("promotion bundle validation ok is not true");
    }
    if (Array.isArray(report.errors) && report.errors.length > 0) {
      errors.push(...report.errors.slice(0, 20).map((error) => `promotion bundle: ${error}`));
    }
    return report;
  } finally {
    fs.rmSync(scratchDir, { recursive: true, force: true });
  }
}

function checkQualityReportNativeProductEval(qualityReport, attention, errors) {
  const summary = {
    min_task_targets: qualityReport?.model_only_quality_floor?.min_task_targets || "",
    min_task_top5_per_mille: qualityReport?.model_only_quality_floor?.min_task_top5_per_mille || "",
    min_phase_targets: qualityReport?.model_only_quality_floor?.min_phase_targets || "",
    min_denoise_bridge_unique_targets: Number(
      qualityReport?.model_only_quality_floor?.min_denoise_bridge_unique_targets || 0,
    ),
    native_task_targets_ok: false,
    native_phase_targets_ok: false,
    native_directional_groups_ok: false,
    denoise_bridge_targets_ok: false,
    weakest_task_targets: null,
    weakest_phase_targets: null,
    weakest_task_top5_per_mille: null,
  };
  if (!qualityReport || typeof qualityReport !== "object") {
    return summary;
  }

  if (summary.min_task_targets !== "all=72") {
    errors.push(`quality-report model_only_quality_floor.min_task_targets ${JSON.stringify(summary.min_task_targets)} != "all=72"`);
  }
  if (summary.min_task_top5_per_mille !== "all=1") {
    errors.push(`quality-report model_only_quality_floor.min_task_top5_per_mille ${JSON.stringify(summary.min_task_top5_per_mille)} != "all=1"`);
  }
  if (summary.min_phase_targets !== "all=72") {
    errors.push(`quality-report model_only_quality_floor.min_phase_targets ${JSON.stringify(summary.min_phase_targets)} != "all=72"`);
  }
  if (attention.min_task_targets === "all=72" && summary.min_task_targets !== attention.min_task_targets) {
    errors.push("quality-report native task target floor does not match product config");
  }
  if (attention.min_phase_targets === "all=72" && summary.min_phase_targets !== attention.min_phase_targets) {
    errors.push("quality-report native phase target floor does not match product config");
  }
  if (Number(attention.denoise_min_unique_targets || 0) >= 2) {
    if (summary.min_denoise_bridge_unique_targets !== Number(attention.denoise_min_unique_targets || 0)) {
      errors.push("quality-report denoise bridge unique-target floor does not match product config");
    }
    const denoiseUniqueTargets = Number(qualityReport.denoise_bridge?.expected_unique_targets || 0);
    const generationUniqueTargets = Number(qualityReport.confidence_trace?.generation_bridge?.expected_unique_targets || 0);
    summary.denoise_bridge_targets_ok =
      denoiseUniqueTargets >= Number(attention.denoise_min_unique_targets || 0) &&
      generationUniqueTargets === denoiseUniqueTargets &&
      qualityReport.denoise_bridge?.target_coverage_ok === true &&
      qualityReport.confidence_trace?.generation_bridge?.target_coverage_ok === true;
    if (!summary.denoise_bridge_targets_ok) {
      errors.push("quality-report denoise bridge target coverage does not meet product config");
    }
  } else {
    summary.denoise_bridge_targets_ok = true;
  }

  const nativeTasks = qualityReport.confidence_trace?.native_task_eval?.tasks || {};
  const taskSummaries = requiredNativeEvalTasks.map((task) => {
    const metric = nativeTasks[task] || {};
    return {
      task,
      targets: Number(metric.targets || 0),
      invalid_contexts: Number(metric.invalid_contexts || 0),
      top5_accuracy_per_mille: Number(metric.top5_accuracy_per_mille || 0),
    };
  });
  summary.weakest_task_targets = [...taskSummaries].sort((left, right) => left.targets - right.targets || left.task.localeCompare(right.task))[0] || null;
  summary.weakest_task_top5_per_mille =
    [...taskSummaries].sort((left, right) => left.top5_accuracy_per_mille - right.top5_accuracy_per_mille || left.task.localeCompare(right.task))[0] || null;
  summary.native_task_targets_ok = taskSummaries.every(
    (item) => item.targets >= 72 && item.invalid_contexts === 0 && item.top5_accuracy_per_mille >= 1,
  );
  for (const item of taskSummaries) {
    if (item.targets < 72) {
      errors.push(`quality-report native task eval ${item.task} targets ${item.targets} < 72`);
    }
    if (item.invalid_contexts !== 0) {
      errors.push(`quality-report native task eval ${item.task} invalid_contexts ${item.invalid_contexts} != 0`);
    }
    if (item.top5_accuracy_per_mille < 1) {
      errors.push(`quality-report native task eval ${item.task} top5_accuracy_per_mille ${item.top5_accuracy_per_mille} < 1`);
    }
  }

  const nativePhases = qualityReport.attention_eval?.phases || {};
  const phaseSummaries = requiredNativeEvalPhases.map((phase) => {
    const metric = nativePhases[phase] || {};
    return {
      phase,
      targets: Number(metric.targets || 0),
      invalid_contexts: Number(metric.invalid_contexts || 0),
    };
  });
  summary.weakest_phase_targets =
    [...phaseSummaries].sort((left, right) => left.targets - right.targets || left.phase.localeCompare(right.phase))[0] || null;
  summary.native_phase_targets_ok = phaseSummaries.every((item) => item.targets >= 72 && item.invalid_contexts === 0);
  for (const item of phaseSummaries) {
    if (item.targets < 72) {
      errors.push(`quality-report native phase eval ${item.phase} targets ${item.targets} < 72`);
    }
    if (item.invalid_contexts !== 0) {
      errors.push(`quality-report native phase eval ${item.phase} invalid_contexts ${item.invalid_contexts} != 0`);
    }
  }

  const directional = qualityReport.confidence_trace?.directional_native_eval || {};
  const groups = directional.groups || {};
  summary.native_directional_groups_ok =
    directional.ok === true && requiredDirectionalGroups.every((group) => groups[group]?.ok === true);
  if (directional.ok !== true) {
    errors.push("quality-report directional native eval is not ok");
  }
  for (const group of requiredDirectionalGroups) {
    if (groups[group]?.ok !== true) {
      errors.push(`quality-report directional native eval group ${group} is not ok`);
    }
  }

  return summary;
}

function checkQualityReportGeneratedProduct(qualityReport, attention, errors) {
  const generation = attention.generation || {};
  const requiredRows = Math.max(Number(generation.min_generated_prompt_rows || 0), 72);
  const promptProvenance = qualityReport?.generative_eval?.evidence?.prompt_provenance || {};
  const confidencePromptProvenance =
    qualityReport?.confidence_trace?.product_generation?.prompt_provenance || {};
  const productGeneration = qualityReport?.confidence_trace?.product_generation || {};
  const summary = {
    required_rows: requiredRows,
    product_generation_ready: qualityReport?.product_generation_ready === true,
    generative_eval_present: qualityReport?.generative_eval?.present === true,
    generative_eval_ok: qualityReport?.generative_eval?.ok === true,
    product_floor_ok: qualityReport?.generative_eval?.product_floor?.ok === true,
    confidence_product_floor_ok: productGeneration.product_floor_ok === true,
    output_identity_required: productGeneration.output_identity_required === true,
    output_identity_ok: productGeneration.matching_model_output_identity?.ok === true,
    selected_prompt_eligible_rows_recorded:
      promptProvenance.selected_prompt_eligible_rows_recorded === true,
    selected_prompt_eligible_rows:
      Number(promptProvenance.selected_prompt_eligible_rows || 0),
    selected_prompt_eligible_rows_match:
      promptProvenance.selected_prompt_eligible_rows_match === true,
    selected_prompt_eligible_unique_targets_recorded:
      promptProvenance.selected_prompt_eligible_unique_targets_recorded === true,
    selected_prompt_eligible_unique_targets:
      Number(promptProvenance.selected_prompt_eligible_unique_targets || 0),
    selected_prompt_eligible_unique_targets_match:
      promptProvenance.selected_prompt_eligible_unique_targets_match === true,
    selected_prompt_hash_match: promptProvenance.selected_prompt_hash_match === true,
    confidence_selected_prompt_eligible_rows:
      Number(confidencePromptProvenance.selected_prompt_eligible_rows || 0),
    confidence_selected_prompt_eligible_unique_targets:
      Number(confidencePromptProvenance.selected_prompt_eligible_unique_targets || 0),
    best_retrieval_top1_per_mille: Number(productGeneration.best_retrieval_top1_per_mille || 0),
    best_retrieval_min_margin: Number(productGeneration.best_retrieval_min_margin || 0),
  };
  if (!qualityReport || typeof qualityReport !== "object") {
    return summary;
  }
  if (generation.require_generative_eval !== true) {
    return summary;
  }

  if (summary.product_generation_ready !== true) {
    errors.push("quality-report product_generation_ready is not true");
  }
  if (summary.generative_eval_present !== true) {
    errors.push("quality-report generative_eval.present is not true");
  }
  if (summary.generative_eval_ok !== true) {
    errors.push("quality-report generative_eval.ok is not true");
  }
  if (summary.product_floor_ok !== true) {
    errors.push("quality-report generative_eval.product_floor.ok is not true");
  }
  if (summary.confidence_product_floor_ok !== true) {
    errors.push("quality-report confidence product_generation.product_floor_ok is not true");
  }
  if (generation.require_generative_output_identity === true && summary.output_identity_ok !== true) {
    errors.push("quality-report confidence product_generation output identity is not ok");
  }
  if (summary.selected_prompt_eligible_rows_recorded !== true) {
    errors.push("quality-report generated selected_prompt_eligible_rows is not recorded");
  }
  if (summary.selected_prompt_eligible_rows < requiredRows) {
    errors.push(
      `quality-report generated selected held-out prompt rows ${summary.selected_prompt_eligible_rows} < ${requiredRows}`,
    );
  }
  if (summary.selected_prompt_eligible_rows_match !== true) {
    errors.push("quality-report generated selected held-out prompt rows do not match recompute");
  }
  if (summary.selected_prompt_eligible_unique_targets_recorded !== true) {
    errors.push("quality-report generated selected_prompt_eligible_unique_targets is not recorded");
  }
  if (summary.selected_prompt_eligible_unique_targets < requiredRows) {
    errors.push(
      `quality-report generated selected held-out unique targets ${summary.selected_prompt_eligible_unique_targets} < ${requiredRows}`,
    );
  }
  if (summary.selected_prompt_eligible_unique_targets_match !== true) {
    errors.push("quality-report generated selected held-out unique targets do not match recompute");
  }
  if (summary.selected_prompt_hash_match !== true) {
    errors.push("quality-report generated selected prompt hash does not match recompute");
  }
  if (
    summary.confidence_selected_prompt_eligible_rows !== summary.selected_prompt_eligible_rows ||
    summary.confidence_selected_prompt_eligible_unique_targets !== summary.selected_prompt_eligible_unique_targets
  ) {
    errors.push("quality-report confidence product_generation prompt provenance does not match generative_eval evidence");
  }
  if (summary.best_retrieval_top1_per_mille < 1000) {
    errors.push(`quality-report generated retrieval top1 ${summary.best_retrieval_top1_per_mille} < 1000`);
  }
  if (summary.best_retrieval_min_margin <= 0) {
    errors.push(`quality-report generated retrieval min margin ${summary.best_retrieval_min_margin} <= 0`);
  }

  return summary;
}

function extractJsonObjects(text) {
  const objects = [];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "{") {
      continue;
    }
    const end = matchingJsonObjectEnd(text, index);
    if (end < 0) {
      continue;
    }
    const candidate = text.slice(index, end + 1);
    try {
      objects.push(JSON.parse(candidate));
      index = end;
    } catch {
      // Non-JSON log text can contain braces.
    }
  }
  return objects;
}

function matchingJsonObjectEnd(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    } else if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }
  return -1;
}

function promotionArtifactPath(runDir, promotionRows, artifactName, fallback) {
  const row = promotionRows.find((item) => item.artifact === artifactName);
  return resolveRunPath(runDir, row?.path || fallback);
}

function check(config) {
  const errors = [];
  const runEnvPath = path.join(config.runDir, "run.env");
  const planPath = path.join(config.runDir, "plan.tsv");
  const artifactsPath = path.join(config.runDir, "artifacts.tsv");
  const promotionPath = path.join(config.runDir, "promotion.tsv");
  const completionPath = path.join(config.runDir, "pipeline-complete.json");
  const promotionCheckPath = path.join(config.runDir, "promotion-bundle-check.json");

  requireFile(runEnvPath, "run.env", errors);
  requireFile(planPath, "plan.tsv", errors);
  requireFile(artifactsPath, "artifacts.tsv", errors);
  requireFile(promotionPath, "promotion.tsv", errors);

  const env = fs.existsSync(runEnvPath) ? readKeyValueFile(runEnvPath) : {};
  checkRunEnv(env, config, errors);
  const productConfig = {
    attention: attentionConfigFromEnv(env),
  };
  validateProductAttentionConfig(productConfig.attention, "run.env.attention", Number(env.processor_count || 0), config, errors);

  let planRows = [];
  if (fs.existsSync(planPath)) {
    try {
      planRows = readTsv(planPath, "stage\tcommand");
    } catch (error) {
      errors.push(error.message);
    }
  }
  const planStages = checkPlan(planRows, errors);

  let artifactRows = [];
  if (fs.existsSync(artifactsPath)) {
    try {
      artifactRows = readTsv(artifactsPath, "stage\tartifact\tpath");
    } catch (error) {
      errors.push(error.message);
    }
  }
  const artifacts = checkArtifacts(config.runDir, artifactRows, errors);
  const syncedArtifacts = summarizeSyncedArtifacts(config.runDir, artifactRows);

  let promotionRows = [];
  if (fs.existsSync(promotionPath)) {
    try {
      promotionRows = readTsv(promotionPath, "product\tstage\tartifact\tpath\trequired");
    } catch (error) {
      errors.push(error.message);
    }
  }
  const promotionArtifactIndex = checkPromotionArtifactIndex(config.runDir, promotionRows, artifactRows, errors);

  const completion = config.requireCompletionReport || fs.existsSync(completionPath)
    ? readJsonIfPresent(completionPath, errors, "pipeline-complete")
    : null;
  const completionSummary = checkCompletionReport(completion, config, env, errors);
  if (completion) {
    compareProductAttentionConfig(
      productConfig.attention,
      completionSummary.product_config?.attention || {},
      config,
      errors,
    );
  }
  const completionArtifactMap = checkCompletionPromotionArtifacts(config.runDir, completion, promotionRows, errors);
  if (completionSummary && typeof completionSummary === "object") {
    completionSummary.artifact_map = completionArtifactMap;
  }
  if (completion?.artifacts?.promotion_manifest && !samePathish(completion.artifacts.promotion_manifest, promotionPath, config.runDir)) {
    errors.push("pipeline-complete artifacts.promotion_manifest does not point at promotion.tsv");
  }
  if (completion?.artifacts?.quality_report) {
    const expectedQuality = promotionArtifactPath(config.runDir, promotionRows, "quality_report", "attention-curriculum/quality-report.json");
    if (!samePathish(completion.artifacts.quality_report, expectedQuality, config.runDir)) {
      errors.push("pipeline-complete artifacts.quality_report does not match promotion.tsv quality_report");
    }
  }

  const statuses = checkStatusFiles(config.runDir, errors, config);
  const promotion = checkPromotionBundle(config, promotionPath, promotionCheckPath, errors);

  const qualityReportPath = promotionArtifactPath(config.runDir, promotionRows, "quality_report", "attention-curriculum/quality-report.json");
  const qualityReport = readJsonIfPresent(qualityReportPath, errors, "quality-report");
  const nativeProductEval = checkQualityReportNativeProductEval(qualityReport, productConfig.attention, errors);
  const generatedProduct = checkQualityReportGeneratedProduct(qualityReport, productConfig.attention, errors);

  return {
    schema,
    ok: errors.length === 0,
    run_dir: config.runDir,
    run_name: env.run_name || path.basename(config.runDir),
    dry_run: env.dry_run === "1",
    runner: {
      kernel: env.runner_kernel || "",
      arch: env.runner_arch || "",
      require_graviton: env.require_graviton === "1",
      processor_count: Number(env.processor_count || 0),
      ec2: {
        metadata_required: env.ec2_metadata_required === "1",
        instance_id: env.ec2_instance_id || "",
        instance_type: env.ec2_instance_type || "",
        availability_zone: env.ec2_availability_zone || "",
        region: env.ec2_region || "",
        instance_lifecycle: env.ec2_instance_lifecycle || "",
      },
    },
    s3: {
      required: env.require_s3_artifacts === "1",
      uri: env.s3_uri || "",
      pipeline_uri: env.s3_pipeline_uri || "",
    },
    plan_stages: planStages,
    completion: completionSummary,
    product_config: productConfig,
    statuses,
    artifacts,
    synced_artifacts: syncedArtifacts,
    promotion: {
      path: promotionPath,
      validation_ran: config.validatePromotionBundle,
      ok: promotion?.ok === true,
      schema: promotion?.schema || "",
      confidence_label: promotion?.confidence_label || "",
      ready_flags: promotion?.ready_flags || {},
      artifact_index: promotionArtifactIndex,
    },
    quality_report: {
      path: qualityReportPath,
      schema: qualityReport?.schema || "",
      ok: qualityReport?.ok === true,
      confidence_label: qualityReport?.confidence_trace?.label || "",
      full_product_generation: qualityReport?.confidence_trace?.label === "strong-bidirectional-product-generation",
      native_product_eval: nativeProductEval,
      generated_product: generatedProduct,
    },
    errors,
  };
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

try {
  const config = parseArgs(process.argv.slice(2));
  const report = check(config);
  writeReport(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
