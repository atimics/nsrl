#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildFixture as buildPromotionBundleFixture } from "./check-solomon-promotion-bundle-self-test.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_run_artifacts_self_test.v1";
const stages = [
  "dataset",
  "denoiser",
  "prior",
  "generative-eval",
  "attention-curriculum",
  "promotion-bundle-check",
];

function usage() {
  console.log([
    "Usage: check-solomon-aws-run-artifacts-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic completed-run fixtures and checks that the AWS run artifact",
    "gate accepts a good real-run shell while rejecting dry-run, non-Graviton,",
    "missing-status, bad-S3, failed-promotion, and stale-promotion cases.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "", keep: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function writeText(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, text, "utf8");
}

export function makeFixture(root, name, mutate = () => {}) {
  const runDir = path.join(root, name);
  const state = {
    runDir,
    runName: name,
    dryRun: false,
    runnerKernel: "Linux",
    runnerArch: "aarch64",
    requireGraviton: true,
    requireEc2Metadata: true,
    ec2InstanceId: "i-0123456789abcdef0",
    ec2InstanceType: "c8g.4xlarge",
    ec2AvailabilityZone: "us-east-1a",
    ec2Region: "us-east-1",
    ec2InstanceLifecycle: "",
    requireS3: true,
    s3Uri: "s3://nsrl-product-run-check/solomon",
    processorCount: 16,
    attentionCorpusVersion: "v2",
    attentionJointCorpusVersion: "v2",
    attentionTextTokenProfile: "chunked",
    attentionImageTokenProfile: "symbolic16",
    attentionJointImageTokenProfile: "symbolic16",
    attentionSeqLen: 512,
    attentionCurriculumStages: [
      "identity",
      "image",
      "text-to-image",
      "description-to-image",
      "image-to-text",
      "explain",
      "hard-negative",
      "native-bind",
    ],
    attentionCurriculumRequiredStages: [
      "identity",
      "image",
      "text-to-image",
      "description-to-image",
      "image-to-text",
      "explain",
      "hard-negative",
      "native-bind",
    ],
    attentionV2StageEpochs: 1,
    attentionNativeBindEpochs: 2,
    attentionImageTokenChannels: ["ink", "edge", "component", "radial", "direction"],
    attentionRequireImageChannelTokenStats: true,
    attentionRequireDirectionalGroups: true,
    attentionHeldoutPrompts: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
    attentionMinHeldoutPromptRows: 72,
    attentionMinTaskTargets: "all=72",
    attentionMinPhaseTargets: "all=72",
    attentionDenoiseMinUniqueTargets: 2,
    attentionGenerativeEval: "generative-eval/current",
    attentionRequireGenerativeEval: true,
    attentionRequireGenerativeOutputIdentity: true,
    attentionMinGeneratedPromptRows: 72,
    attentionMinGeneratedTop516PerMille: 1,
    attentionMinGeneratedRetrievalTop1PerMille: 1000,
    attentionMinGeneratedRetrievalTop5PerMille: 1000,
    attentionMinGeneratedRetrievalMargin: 1,
    attentionMaxGeneratedMeanTargetDistance16Q8: 7000000,
    attentionBatchMode: "map-reduce",
    attentionMapReduceWorkers: "0",
    attentionCpuScalingPolicy: "auto-online-processors",
    attentionMapReduceAutoWorkers: true,
    attentionEffectiveMapReduceWorkers: 16,
    promotionOk: true,
    corruptQualityAfterCheck: false,
    corruptNativeEvalAfterCheck: false,
    corruptGeneratedCoverageAfterCheck: false,
    omitArtifactIndex: "",
    omitCompletionArtifact: "",
    omitStatus: "",
  };
  mutate(state);
  buildPromotionBundleFixture(runDir);
  fs.mkdirSync(path.join(runDir, "logs"), { recursive: true });
  fs.mkdirSync(path.join(runDir, "attention-curriculum"), { recursive: true });
  fs.mkdirSync(path.join(runDir, "generative-eval", "current"), { recursive: true });

  writeText(path.join(runDir, "run.env"), [
    "schema=nsrl.solomon_aws_pipeline.v1",
    `run_name=${state.runName}`,
    `run_dir=${runDir}`,
    `stages=${stages.slice(0, -1).join(",")}`,
    `dry_run=${state.dryRun ? 1 : 0}`,
    `runner_kernel=${state.runnerKernel}`,
    `runner_arch=${state.runnerArch}`,
    `require_graviton=${state.requireGraviton ? 1 : 0}`,
    `ec2_metadata_required=${state.requireEc2Metadata ? 1 : 0}`,
    `ec2_instance_id=${state.ec2InstanceId}`,
    `ec2_instance_type=${state.ec2InstanceType}`,
    `ec2_availability_zone=${state.ec2AvailabilityZone}`,
    `ec2_region=${state.ec2Region}`,
    `ec2_instance_lifecycle=${state.ec2InstanceLifecycle}`,
    `require_s3_artifacts=${state.requireS3 ? 1 : 0}`,
    `s3_uri=${state.s3Uri}`,
    `s3_pipeline_uri=${state.s3Uri ? `${state.s3Uri}/pipelines/${state.runName}` : ""}`,
    "promotion_manifest=promotion.tsv",
    "pipeline_complete_report=pipeline-complete.json",
    "promotion_bundle_check=1",
    `processor_count=${state.processorCount}`,
    `attention_corpus_version=${state.attentionCorpusVersion}`,
    `attention_joint_corpus_version=${state.attentionJointCorpusVersion}`,
    `attention_text_token_profile=${state.attentionTextTokenProfile}`,
    `attention_image_token_profile=${state.attentionImageTokenProfile}`,
    `attention_joint_image_token_profile=${state.attentionJointImageTokenProfile}`,
    `attention_seq_len=${state.attentionSeqLen}`,
    `attention_v2_curriculum_stages=${state.attentionCurriculumStages.join(",")}`,
    `attention_v2_curriculum_required_stages=${state.attentionCurriculumRequiredStages.join(",")}`,
    `attention_v2_stage_epochs=${state.attentionV2StageEpochs}`,
    `attention_v2_native_bind_epochs=${state.attentionNativeBindEpochs}`,
    `attention_require_image_token_channels=${state.attentionImageTokenChannels.join(",")}`,
    `attention_require_image_channel_token_stats=${state.attentionRequireImageChannelTokenStats ? 1 : 0}`,
    `attention_require_directional_groups=${state.attentionRequireDirectionalGroups ? 1 : 0}`,
    `attention_heldout_prompts=${state.attentionHeldoutPrompts}`,
    `attention_min_heldout_prompt_rows=${state.attentionMinHeldoutPromptRows}`,
    `attention_min_task_targets=${state.attentionMinTaskTargets}`,
    `attention_min_phase_targets=${state.attentionMinPhaseTargets}`,
    `attention_denoise_min_unique_targets=${state.attentionDenoiseMinUniqueTargets}`,
    `attention_generative_eval=${state.attentionGenerativeEval}`,
    `attention_require_generative_eval=${state.attentionRequireGenerativeEval ? 1 : 0}`,
    `attention_require_generative_output_identity=${state.attentionRequireGenerativeOutputIdentity ? 1 : 0}`,
    `attention_min_generated_prompt_rows=${state.attentionMinGeneratedPromptRows}`,
    `attention_min_generated_top5_16_per_mille=${state.attentionMinGeneratedTop516PerMille}`,
    `attention_min_generated_retrieval_top1_per_mille=${state.attentionMinGeneratedRetrievalTop1PerMille}`,
    `attention_min_generated_retrieval_top5_per_mille=${state.attentionMinGeneratedRetrievalTop5PerMille}`,
    `attention_min_generated_retrieval_margin=${state.attentionMinGeneratedRetrievalMargin}`,
    `attention_max_generated_mean_target_distance_16_q8=${state.attentionMaxGeneratedMeanTargetDistance16Q8}`,
    `attention_batch_mode=${state.attentionBatchMode}`,
    `attention_map_reduce_workers=${state.attentionMapReduceWorkers}`,
    `attention_cpu_scaling_policy=${state.attentionCpuScalingPolicy}`,
    `attention_map_reduce_auto_workers=${state.attentionMapReduceAutoWorkers ? 1 : 0}`,
    `attention_effective_map_reduce_workers=${state.attentionEffectiveMapReduceWorkers}`,
    "",
  ].join("\n"));

  writeText(path.join(runDir, "plan.tsv"), [
    "stage\tcommand",
    ...stages.map((stage) => `${stage}\t${stage} command`),
    "",
  ].join("\n"));

  const artifactRows = [
    ["pipeline", "run_env", "run.env"],
    ["pipeline", "plan", "plan.tsv"],
    ["pipeline", "artifacts", "artifacts.tsv"],
    ["pipeline", "promotion_manifest", "promotion.tsv"],
    ["pipeline", "pipeline_complete", "pipeline-complete.json"],
    ["pipeline", "promotion_bundle_check", "promotion-bundle-check.json"],
    ["attention-curriculum", "quality_report", "attention-curriculum/quality-report.json"],
    ["attention-curriculum", "model", "attention-curriculum/model.nsrllmm"],
    ["attention-curriculum", "corpus_manifest", "attention-curriculum/manifest.json"],
    ["attention-curriculum", "attention_eval", "attention-curriculum/attention-eval.json"],
    ["attention-curriculum", "retrieval_head", "attention-curriculum/retrieval-head.json"],
    ["attention-curriculum", "retrieval_head_eval", "attention-curriculum/retrieval-head-eval.json"],
    ["attention-curriculum", "curriculum_stages", "attention-curriculum/curriculum-stages.json"],
    ["attention-curriculum", "sample_binding", "attention-curriculum/prior-sample-binding.json"],
    ["attention-curriculum", "identity_inference", "attention-curriculum/identity-inference.json"],
    ["attention-curriculum", "grounded_corpus", "attention-curriculum/grounded-corpus.json"],
    ["attention-curriculum", "generation_integrity", "attention-curriculum/generation-integrity.json"],
    ["attention-curriculum", "denoise_bridge", "attention-curriculum/denoise-bridge.json"],
    ["attention-curriculum", "denoise_generation_integrity", "attention-curriculum/denoise-generation-integrity.json"],
    ["generative-eval", "run", "generative-eval/current"],
    ["generative-eval", "summary", "generative-eval/current/summary.tsv"],
  ].filter(([, artifact]) => artifact !== state.omitArtifactIndex);
  writeText(path.join(runDir, "artifacts.tsv"), [
    "stage\tartifact\tpath",
    ...artifactRows.map((row) => row.join("\t")),
    "",
  ].join("\n"));

  for (const stage of stages) {
    if (stage === state.omitStatus) {
      continue;
    }
    writeText(path.join(runDir, "logs", `${stage}.status`), [
      `stage=${stage}`,
      "started_at=2026-07-03T00:00:00Z",
      "finished_at=2026-07-03T00:00:01Z",
      "status=0",
      `dry_run=${state.dryRun ? 1 : 0}`,
      `log=logs/${stage}.log`,
      "",
    ].join("\n"));
  }

  if (state.promotionOk) {
    const result = childProcess.spawnSync(process.execPath, [
      "scripts/check-solomon-promotion-bundle.mjs",
      "--promotion",
      path.join(runDir, "promotion.tsv"),
      "--out",
      path.join(runDir, "promotion-bundle-check.json"),
    ], {
      cwd: repoRoot,
      encoding: "utf8",
    });
    if (result.status !== 0) {
      throw new Error(`synthetic promotion bundle failed unexpectedly\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
  } else {
    writeText(path.join(runDir, "promotion-bundle-check.json"), `${JSON.stringify({
      schema: "nsrl.solomon_promotion_bundle_check.v1",
      ok: false,
      promotion: "promotion.tsv",
      quality: { ready_flags: {} },
      errors: ["synthetic promotion failure"],
    }, null, 2)}\n`);
  }

  if (state.corruptQualityAfterCheck) {
    const qualityPath = path.join(runDir, "attention-curriculum", "quality-report.json");
    const quality = JSON.parse(fs.readFileSync(qualityPath, "utf8"));
    quality.product_generation_ready = false;
    quality.confidence_trace.product_generation.matching_model_output_identity.ok = false;
    writeText(qualityPath, `${JSON.stringify(quality, null, 2)}\n`);
  }
  if (state.corruptNativeEvalAfterCheck) {
    const qualityPath = path.join(runDir, "attention-curriculum", "quality-report.json");
    const quality = JSON.parse(fs.readFileSync(qualityPath, "utf8"));
    quality.model_only_quality_floor.min_task_targets = "all=1";
    quality.model_only_quality_floor.min_phase_targets = "all=1";
    quality.confidence_trace.native_task_eval.tasks["image-to-text"].targets = 2;
    quality.attention_eval.phases.image.targets = 2;
    writeText(qualityPath, `${JSON.stringify(quality, null, 2)}\n`);
  }
  if (state.corruptGeneratedCoverageAfterCheck) {
    const qualityPath = path.join(runDir, "attention-curriculum", "quality-report.json");
    const quality = JSON.parse(fs.readFileSync(qualityPath, "utf8"));
    const promptProvenance = quality.generative_eval.evidence.prompt_provenance;
    promptProvenance.selected_prompt_eligible_rows = 1;
    promptProvenance.selected_prompt_eligible_unique_targets = 1;
    quality.confidence_trace.product_generation.prompt_provenance.selected_prompt_eligible_rows = 1;
    quality.confidence_trace.product_generation.prompt_provenance.selected_prompt_eligible_unique_targets = 1;
    writeText(qualityPath, `${JSON.stringify(quality, null, 2)}\n`);
  }

  const completionArtifacts = {
    run_env: "run.env",
    plan: "plan.tsv",
    artifacts: "artifacts.tsv",
    quality_report: "attention-curriculum/quality-report.json",
    model: "attention-curriculum/model.nsrllmm",
    corpus_manifest: "attention-curriculum/manifest.json",
    attention_eval: "attention-curriculum/attention-eval.json",
    retrieval_head: "attention-curriculum/retrieval-head.json",
    retrieval_head_eval: "attention-curriculum/retrieval-head-eval.json",
    curriculum_stages: "attention-curriculum/curriculum-stages.json",
    sample_binding: "attention-curriculum/prior-sample-binding.json",
    identity_inference: "attention-curriculum/identity-inference.json",
    grounded_corpus: "attention-curriculum/grounded-corpus.json",
    generation_integrity: "attention-curriculum/generation-integrity.json",
    denoise_bridge: "attention-curriculum/denoise-bridge.json",
    denoise_generation_integrity: "attention-curriculum/denoise-generation-integrity.json",
    run: "generative-eval/current",
    summary: "generative-eval/current/summary.tsv",
    promotion_manifest: "promotion.tsv",
    promotion_bundle_check: "promotion-bundle-check.json",
    pipeline_complete: "pipeline-complete.json",
  };
  if (state.omitCompletionArtifact) {
    delete completionArtifacts[state.omitCompletionArtifact];
  }

  writeText(path.join(runDir, "pipeline-complete.json"), `${JSON.stringify({
    schema: "nsrl.solomon_aws_pipeline_complete.v1",
    ok: true,
    generated_at: "2026-07-03T00:00:02.000Z",
    run_name: state.runName,
    run_dir: runDir,
    dry_run: state.dryRun,
    stages,
    product_stages: stages.slice(0, -1),
    runner: {
      kernel: state.runnerKernel,
      arch: state.runnerArch,
      online_processors: state.processorCount,
      require_graviton: state.requireGraviton,
      ec2: {
        metadata_required: state.requireEc2Metadata,
        instance_id: state.ec2InstanceId,
        instance_type: state.ec2InstanceType,
        availability_zone: state.ec2AvailabilityZone,
        region: state.ec2Region,
        instance_lifecycle: state.ec2InstanceLifecycle,
      },
    },
    s3: {
      required: state.requireS3,
      uri: state.s3Uri,
      pipeline_uri: state.s3Uri ? `${state.s3Uri}/pipelines/${state.runName}` : "",
    },
    product_config: {
      attention: {
        corpus_version: state.attentionCorpusVersion,
        joint_corpus_version: state.attentionJointCorpusVersion,
        text_token_profile: state.attentionTextTokenProfile,
        image_token_profile: state.attentionImageTokenProfile,
        joint_image_token_profile: state.attentionJointImageTokenProfile,
        seq_len: state.completionAttentionSeqLen ?? state.attentionSeqLen,
        curriculum: {
          stages: state.attentionCurriculumStages,
          required_stages: state.attentionCurriculumRequiredStages,
          stage_epochs: state.completionAttentionV2StageEpochs ?? state.attentionV2StageEpochs,
          native_bind_epochs: state.completionAttentionNativeBindEpochs ?? state.attentionNativeBindEpochs,
        },
        image_token_channels: state.attentionImageTokenChannels,
        require_image_channel_token_stats: state.attentionRequireImageChannelTokenStats,
        require_directional_groups: state.attentionRequireDirectionalGroups,
        heldout_prompts: state.attentionHeldoutPrompts,
        min_heldout_prompt_rows: state.attentionMinHeldoutPromptRows,
        min_task_targets: state.attentionMinTaskTargets,
        min_phase_targets: state.attentionMinPhaseTargets,
        denoise_min_unique_targets:
          state.completionDenoiseMinUniqueTargets ?? state.attentionDenoiseMinUniqueTargets,
        generation: {
          generative_eval: state.completionAttentionGenerativeEval ?? state.attentionGenerativeEval,
          require_generative_eval: state.completionRequireGenerativeEval ?? state.attentionRequireGenerativeEval,
          require_generative_output_identity:
            state.completionRequireGenerativeOutputIdentity ?? state.attentionRequireGenerativeOutputIdentity,
          min_generated_prompt_rows:
            state.completionMinGeneratedPromptRows ?? state.attentionMinGeneratedPromptRows,
          min_generated_top5_16_per_mille:
            state.completionMinGeneratedTop516PerMille ?? state.attentionMinGeneratedTop516PerMille,
          min_generated_retrieval_top1_per_mille:
            state.completionMinGeneratedRetrievalTop1PerMille ?? state.attentionMinGeneratedRetrievalTop1PerMille,
          min_generated_retrieval_top5_per_mille:
            state.completionMinGeneratedRetrievalTop5PerMille ?? state.attentionMinGeneratedRetrievalTop5PerMille,
          min_generated_retrieval_margin:
            state.completionMinGeneratedRetrievalMargin ?? state.attentionMinGeneratedRetrievalMargin,
          max_generated_mean_target_distance_16_q8:
            state.completionMaxGeneratedMeanTargetDistance16Q8 ?? state.attentionMaxGeneratedMeanTargetDistance16Q8,
        },
        cpu_scaling: {
          batch_mode: state.attentionBatchMode,
          map_reduce_workers: state.attentionMapReduceWorkers,
          policy: state.attentionCpuScalingPolicy,
          auto_workers: state.attentionMapReduceAutoWorkers,
          effective_workers: state.attentionEffectiveMapReduceWorkers,
        },
      },
    },
    artifacts: completionArtifacts,
  }, null, 2)}\n`);
  return runDir;
}

function runChecker(runDir) {
  return childProcess.spawnSync(process.execPath, [
    "scripts/check-solomon-aws-run-artifacts.mjs",
    "--run-dir",
    runDir,
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function extractReport(stdout) {
  const start = stdout.indexOf("{");
  if (start < 0) {
    return null;
  }
  return JSON.parse(stdout.slice(start));
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-aws-run-artifacts-self-test-"));
  const cases = [];
  try {
    const definitions = [
      {
        name: "good",
        expectOk: true,
        mutate: () => {},
      },
      {
        name: "bad-dry-run",
        expectOk: false,
        mutate: (state) => {
          state.dryRun = true;
        },
      },
      {
        name: "bad-non-graviton",
        expectOk: false,
        mutate: (state) => {
          state.runnerKernel = "Linux";
          state.runnerArch = "x86_64";
          state.ec2InstanceType = "c7i.4xlarge";
        },
      },
      {
        name: "bad-missing-ec2-metadata",
        expectOk: false,
        mutate: (state) => {
          state.requireEc2Metadata = false;
          state.ec2InstanceId = "";
          state.ec2InstanceType = "";
        },
      },
      {
        name: "bad-missing-stage-status",
        expectOk: false,
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-fixed-attention-workers",
        expectOk: false,
        mutate: (state) => {
          state.attentionMapReduceWorkers = "4";
          state.attentionMapReduceAutoWorkers = false;
          state.attentionEffectiveMapReduceWorkers = 4;
        },
      },
      {
        name: "bad-cpu-scaling-policy",
        expectOk: false,
        mutate: (state) => {
          state.attentionCpuScalingPolicy = "fixed-workers";
        },
      },
      {
        name: "bad-image-token-profile",
        expectOk: false,
        mutate: (state) => {
          state.attentionImageTokenProfile = "raw16";
          state.attentionJointImageTokenProfile = "raw16";
        },
      },
      {
        name: "bad-native-bind-epochs",
        expectOk: false,
        mutate: (state) => {
          state.attentionNativeBindEpochs = 1;
        },
      },
      {
        name: "bad-native-product-eval-scope",
        expectOk: false,
        mutate: (state) => {
          state.corruptNativeEvalAfterCheck = true;
        },
      },
      {
        name: "bad-generated-product-coverage",
        expectOk: false,
        mutate: (state) => {
          state.corruptGeneratedCoverageAfterCheck = true;
        },
      },
      {
        name: "bad-artifact-index-missing-curriculum-stages",
        expectOk: false,
        requiredError: "artifacts.tsv missing required promotion artifact attention-curriculum/curriculum_stages",
        mutate: (state) => {
          state.omitArtifactIndex = "curriculum_stages";
        },
      },
      {
        name: "bad-completion-missing-curriculum-stages-artifact",
        expectOk: false,
        requiredError: "pipeline-complete artifacts missing required promotion artifact attention-curriculum/curriculum_stages",
        mutate: (state) => {
          state.omitCompletionArtifact = "curriculum_stages";
        },
      },
      {
        name: "bad-completion-env-config-mismatch",
        expectOk: false,
        mutate: (state) => {
          state.completionAttentionSeqLen = 384;
          state.completionDenoiseMinUniqueTargets = 1;
        },
      },
      {
        name: "bad-completion-generative-config-mismatch",
        expectOk: false,
        mutate: (state) => {
          state.completionMinGeneratedRetrievalTop1PerMille = 1;
          state.completionMinGeneratedRetrievalMargin = 0;
        },
      },
      {
        name: "bad-s3",
        expectOk: false,
        mutate: (state) => {
          state.s3Uri = "";
        },
      },
      {
        name: "bad-promotion-check",
        expectOk: false,
        mutate: (state) => {
          state.promotionOk = false;
        },
      },
      {
        name: "bad-promotion-validation",
        expectOk: false,
        mutate: (state) => {
          state.corruptQualityAfterCheck = true;
        },
      },
    ];
    for (const item of definitions) {
      const runDir = makeFixture(root, item.name, item.mutate);
      const result = runChecker(runDir);
      const report = extractReport(result.stdout || "");
      const actualOk = result.status === 0 && report?.ok === true;
      const requiredErrorOk = item.requiredError
        ? (report?.errors || []).some((error) => String(error).includes(item.requiredError))
        : true;
      cases.push({
        name: item.name,
        expect_ok: item.expectOk,
        ok: actualOk === item.expectOk &&
          requiredErrorOk &&
          (item.expectOk ? Boolean(report?.synced_artifacts?.sha256) : true),
        status: result.status,
        checker_ok: report?.ok === true,
        synced_artifacts_sha256: report?.synced_artifacts?.sha256 || "",
        synced_artifacts_present:
          Number(report?.synced_artifacts?.present_count || 0) ===
          Number(report?.synced_artifacts?.artifact_count || -1),
        required_error: item.requiredError || "",
        required_error_ok: requiredErrorOk,
        errors: report?.errors || [],
      });
    }
    const digestRunDir = makeFixture(root, "digest-changes-after-artifact-tamper", () => {});
    const beforeDigestReport = extractReport(runChecker(digestRunDir).stdout || "");
    fs.appendFileSync(path.join(digestRunDir, "attention-curriculum", "model.nsrllmm"), "\n# digest tamper\n", "utf8");
    const afterDigestReport = extractReport(runChecker(digestRunDir).stdout || "");
    cases.push({
      name: "digest-changes-after-artifact-tamper",
      expect_ok: true,
      ok: Boolean(beforeDigestReport?.synced_artifacts?.sha256) &&
        Boolean(afterDigestReport?.synced_artifacts?.sha256) &&
        beforeDigestReport.synced_artifacts.sha256 !== afterDigestReport.synced_artifacts.sha256,
      status: afterDigestReport?.ok === true ? 0 : 1,
      checker_ok: afterDigestReport?.ok === true,
      before_synced_artifacts_sha256: beforeDigestReport?.synced_artifacts?.sha256 || "",
      after_synced_artifacts_sha256: afterDigestReport?.synced_artifacts?.sha256 || "",
      errors: afterDigestReport?.errors || [],
    });
    const report = {
      schema,
      ok: cases.every((item) => item.ok),
      root,
      kept: config.keep,
      cases,
    };
    if (config.outPath) {
      writeText(config.outPath, `${JSON.stringify(report, null, 2)}\n`);
    }
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) {
      process.exit(1);
    }
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(2);
  }
}
