#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_release_candidate_check.v1";
const diagnosticSchema = "nsrl.solomon_product_diagnostic_check.v1";
const objectiveSchema = "nsrl.solomon_objective_coverage_check.v1";
const expectedReleaseGap = "no synced real Graviton product run was supplied with --aws-run-dir";
const requiredProductStages = [
  "dataset",
  "denoiser",
  "prior",
  "generative-eval",
  "attention-curriculum",
];
const requiredAwsLaunchExecuteGuardCases = [
  "bad-execute-missing-explicit-s3-blocks-before-aws",
  "bad-execute-missing-explicit-artifact-blocks-before-aws",
  "bad-execute-prelaunch-blocks-before-aws",
  "good-execute-records-launch-result",
  "good-execute-command-matches-launch-manifest",
  "good-execute-command-matches-launch-manifest-with-profile",
];
const requiredAwsReleaseProofCases = [
  "bad-mismatched-s3-pipeline",
  "bad-missing-stage-status",
  "bad-native-product-eval-scope",
  "bad-missing-required-launch-dir",
  "bad-dry-run-launch-dir",
  "bad-missing-launch-instance-id",
  "bad-launch-user-data-sha256",
  "bad-launch-command-image-id",
  "bad-launch-command-instance-type",
  "bad-launch-command-user-data",
  "bad-launch-command-tag-specification",
  "bad-launch-command-security-groups",
  "bad-launch-command-subnet",
  "bad-launch-command-key-name",
  "bad-launch-command-region",
  "bad-launch-command-profile",
  "bad-missing-launch-result",
  "bad-launch-result-sha256",
  "bad-launch-result-image-id",
  "bad-launch-result-instance-type",
  "bad-launch-result-subnet",
  "bad-launch-result-security-groups",
  "bad-launch-run-instance-mismatch",
  "bad-post-run-proof-command",
];
const requiredAwsRunArtifactCases = [
  "good",
  "bad-dry-run",
  "bad-non-graviton",
  "bad-missing-ec2-metadata",
  "bad-missing-stage-status",
  "bad-fixed-attention-workers",
  "bad-cpu-scaling-policy",
  "bad-image-token-profile",
  "bad-native-bind-epochs",
  "bad-native-product-eval-scope",
  "bad-generated-product-coverage",
  "bad-artifact-index-missing-curriculum-stages",
  "bad-completion-missing-curriculum-stages-artifact",
  "digest-changes-after-artifact-tamper",
  "bad-completion-env-config-mismatch",
  "bad-completion-generative-config-mismatch",
  "bad-s3",
  "bad-promotion-check",
  "bad-promotion-validation",
];
const requiredAwsRunFetchCases = [
  "good",
  "good-run-name",
  "bad-mismatched-s3-pipeline",
  "bad-missing-status",
  "bad-stale-promotion",
  "bad-native-product-eval-scope",
];
const requiredAwsLiveLaunchReadinessCases = [
  "good-explicit-s3-artifact",
  "bad-missing-explicit-s3-artifact",
  "bad-missing-explicit-ami",
];
const localNextOperatorAction = [
  "run scripts/check-solomon-aws-live-launch-readiness.sh",
  "then scripts/aws/launch-solomon-product-run.sh --execute",
  "keep the executed launch directory with launch.json and launch-result.json",
  "then after the S3 run completes run scripts/aws/prove-solomon-product-run.sh --s3-pipeline-uri <executed launch s3_pipeline_uri> --launch-dir <executed launch dir> --require-launch-dir",
].join(", ");
const requiredMaxGeneratedMeanTargetDistance16Q8 = 7000000;
const requiredMinTaskTargets = "all=72";
const requiredMinTaskTop5PerMille = "all=1";
const requiredMinPhaseTargets = "all=72";
const requiredNativeBindEpochs = 2;
const requiredDModel = 128;
const requiredHeads = 2;
const requiredHeadDim = 64;
const requiredMinHiddenDim = 256;
const requiredMaxHiddenDim = 512;
const requiredMinTransformerLayers = 2;
const requiredMinContextSeqLen = 384;
const requiredMinSeqLen = 384;
const requiredMaxSeqLen = 768;
const requiredMinGeneratedRetrievalMargin = 1;
const requiredMinDenoiseBridgeUniqueTargets = 2;
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

function usage() {
  console.log([
    "Usage: check-solomon-release-candidate.mjs --diagnostic PATH [options]",
    "",
    "Checks that a Solomon product diagnostic is a no-spend release candidate:",
    "all local product/objective proof is green, AWS Graviton launch handoff",
    "evidence is green, and the only tolerated release gap is the absent synced",
    "real Graviton run. Pass --require-release to require completed-run proof.",
    "",
    "Options:",
    "  --objective PATH             use an existing objective coverage report",
    "  --objective-out PATH         when --objective is omitted, write objective coverage here",
    "  --out PATH                   write this release-candidate report",
    "  --require-release            require release_product_proof/release_objective_proof",
    "  --allow-extra-release-gaps   tolerate extra remaining release evidence",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = {
    diagnosticPath: "",
    objectivePath: "",
    objectiveOutPath: "",
    outPath: "",
    requireRelease: false,
    allowExtraReleaseGaps: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--diagnostic") {
      config.diagnosticPath = requireValue(argv, ++index, arg);
    } else if (arg === "--objective") {
      config.objectivePath = requireValue(argv, ++index, arg);
    } else if (arg === "--objective-out") {
      config.objectiveOutPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-release") {
      config.requireRelease = true;
    } else if (arg === "--allow-extra-release-gaps") {
      config.allowExtraReleaseGaps = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.diagnosticPath) {
    throw new Error("--diagnostic PATH is required");
  }
  if (config.objectivePath && config.objectiveOutPath) {
    throw new Error("--objective and --objective-out cannot both be supplied");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  const resolved = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function sameSequence(values, expected) {
  const actual = (values || []).map(String);
  return actual.length === expected.length && expected.every((item, index) => actual[index] === item);
}

function includesAll(values, expected) {
  const set = new Set((values || []).map(String));
  return expected.every((item) => set.has(item));
}

function postRunProofCommandProblems(label, row) {
  const command = Array.isArray(row?.post_run_proof_command)
    ? row.post_run_proof_command.map(String)
    : [];
  const s3PipelineUri = String(row?.s3_pipeline_uri || "");
  const s3Index = command.indexOf("--s3-pipeline-uri");
  const launchDirIndex = command.indexOf("--launch-dir");
  return [
    command.includes("scripts/aws/prove-solomon-product-run.sh")
      ? ""
      : `${label} post-run proof command missing scripts/aws/prove-solomon-product-run.sh`,
    s3PipelineUri.startsWith("s3://")
      ? ""
      : `${label} s3_pipeline_uri is missing`,
    s3Index >= 0 && command[s3Index + 1] === s3PipelineUri
      ? ""
      : `${label} post-run proof command does not match s3_pipeline_uri`,
    launchDirIndex >= 0 && Boolean(command[launchDirIndex + 1])
      ? ""
      : `${label} post-run proof command missing --launch-dir`,
    command.includes("--require-launch-dir")
      ? ""
      : `${label} post-run proof command missing --require-launch-dir`,
  ].filter(Boolean);
}

function denoiseRunnerProblems(runner, requiredFloor) {
  const bridgePairCount = Number(runner?.bridge_pair_count || 0);
  const requiredBridgePairCount = Math.max(Number(runner?.required_bridge_pair_count || 0), requiredFloor);
  return [
    runner?.ok === true ? "" : "denoise runner proof is not ok",
    runner?.present === true ? "" : "denoise runner source proof is missing",
    runner?.min_unique_targets_arg === true ? "" : "denoise runner does not pass --min-unique-targets",
    runner?.quality_min_unique_targets_arg === true
      ? ""
      : "denoise runner does not pass --min-denoise-bridge-unique-targets",
    bridgePairCount >= requiredBridgePairCount
      ? ""
      : `denoise runner bridge-pair count is below ${requiredBridgePairCount}`,
  ].filter(Boolean);
}

function checkEntry(diagnostic, name) {
  return (diagnostic.checks || []).find((entry) => entry?.name === name) || {};
}

function requirement(key, label, ok, evidence = {}, missing = []) {
  return {
    key,
    label,
    ok: ok === true,
    evidence,
    missing: missing.filter(Boolean),
  };
}

function tailLines(text, maxLines) {
  const lines = String(text || "").split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function objectiveOutputPath(config) {
  if (config.objectiveOutPath) {
    return path.resolve(config.objectiveOutPath);
  }
  if (config.outPath) {
    const parsed = path.parse(path.resolve(config.outPath));
    return path.join(parsed.dir, `${parsed.name}-objective-coverage.json`);
  }
  return path.join(fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-release-candidate-")), "objective-coverage.json");
}

function loadObjectiveCoverage(config) {
  if (config.objectivePath) {
    return {
      generated: false,
      path: path.resolve(config.objectivePath),
      status: null,
      stdout_tail: "",
      stderr_tail: "",
      report: readJson(config.objectivePath),
    };
  }

  const outPath = objectiveOutputPath(config);
  const args = [
    "scripts/check-solomon-objective-coverage.mjs",
    "--diagnostic",
    config.diagnosticPath,
    "--out",
    outPath,
  ];
  if (config.requireRelease) {
    args.push("--require-release");
  }
  const result = childProcess.spawnSync(process.execPath, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  return {
    generated: true,
    path: outPath,
    status: result.status,
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
    report: fs.existsSync(outPath) ? readJson(outPath) : null,
  };
}

function releaseGapOk(values, config) {
  const gaps = (values || []).map(String).filter(Boolean);
  if (gaps.length === 0) {
    return true;
  }
  if (config.allowExtraReleaseGaps) {
    return gaps.includes(expectedReleaseGap);
  }
  return gaps.length === 1 && gaps[0] === expectedReleaseGap;
}

function buildAwsHandoff(evidence) {
  const productPlan = evidence["aws-product-plan"] || {};
  const launchPlan = evidence["aws-launch-plan"] || {};
  const prelaunch = evidence["aws-prelaunch-readiness"] || {};
  const launchExecuteGuard = evidence["aws-launch-execute-guard-self-test"] || {};
  const liveLaunchReadiness = evidence["aws-live-launch-readiness-self-test"] || {};
  const releaseProof = evidence["aws-release-proof-self-test"] || {};
  const runArtifacts = evidence["aws-run-artifacts-self-test"] || {};
  const runFetch = evidence["aws-run-fetch-self-test"] || {};
  return {
    product_plan: {
      stages: productPlan.stages || [],
      required_plan_stages: productPlan.required_plan_stages || [],
      promotion_bundle_check: productPlan.promotion_bundle_check === true,
      require_graviton: productPlan.runner?.require_graviton === true,
      s3_required: productPlan.s3_required === true,
      cpu_scaling: productPlan.attention?.cpu_scaling || {},
      curriculum_stages: productPlan.attention?.curriculum_stages || [],
      curriculum_required_stages: productPlan.attention?.curriculum_required_stages || [],
      native_bind_epochs: Number(productPlan.attention?.native_bind_epochs || 0),
      min_direction_top5_per_mille: productPlan.attention?.min_direction_top5_per_mille || "",
      min_task_targets: productPlan.attention?.min_task_targets || "",
      min_task_top5_per_mille: productPlan.attention?.min_task_top5_per_mille || "",
      min_phase_targets: productPlan.attention?.min_phase_targets || "",
      require_promoted_small_profile: productPlan.attention?.require_promoted_small_profile === true,
      seq_len: Number(productPlan.attention?.seq_len || 0),
      require_architecture_profile: productPlan.attention?.require_architecture_profile === true,
      min_d_model: Number(productPlan.attention?.min_d_model || 0),
      min_heads: Number(productPlan.attention?.min_heads || 0),
      target_head_dim: Number(productPlan.attention?.target_head_dim || 0),
      min_hidden_dim: Number(productPlan.attention?.min_hidden_dim || 0),
      max_hidden_dim: Number(productPlan.attention?.max_hidden_dim || 0),
      min_transformer_layers: Number(productPlan.attention?.min_transformer_layers || 0),
      min_context_seq_len: Number(productPlan.attention?.min_context_seq_len || 0),
      train_core_architecture: productPlan.attention?.train_core_architecture || {},
      curriculum_denoise_runner: productPlan.attention?.curriculum_denoise_runner || {},
      require_generative_eval: productPlan.attention?.require_generative_eval === true,
      require_generative_output_identity: productPlan.attention?.require_generative_output_identity === true,
      min_generated_prompt_rows: Number(productPlan.attention?.min_generated_prompt_rows || 0),
      generated_prompt_rows: Number(productPlan.generated_prompt_rows || 0),
      generated_unique_targets: Number(productPlan.generated_unique_targets || 0),
      min_generated_top5_16_per_mille: Number(productPlan.attention?.min_generated_top5_16_per_mille || 0),
      min_generated_retrieval_top1_per_mille: Number(productPlan.attention?.min_generated_retrieval_top1_per_mille || 0),
      min_generated_retrieval_top5_per_mille: Number(productPlan.attention?.min_generated_retrieval_top5_per_mille || 0),
      min_generated_retrieval_margin: Number(productPlan.attention?.min_generated_retrieval_margin || 0),
      max_generated_mean_target_distance_16_q8:
        Number(productPlan.attention?.max_generated_mean_target_distance_16_q8 || 0),
      denoise_min_unique_targets: Number(productPlan.attention?.denoise_min_unique_targets || 0),
    },
    launch_plan: {
      dry_run: launchPlan.dry_run === true,
      graviton_instance: launchPlan.graviton_instance === true,
      ec2_metadata_required: launchPlan.ec2_metadata_required === true,
      product_stages: launchPlan.product_stages || [],
      cpu_scaling: launchPlan.cpu_scaling || {},
      s3_uri: launchPlan.s3_uri || "",
      s3_pipeline_uri: launchPlan.s3_pipeline_uri || "",
      artifact_s3_uri: launchPlan.artifact_s3_uri || "",
      post_run_proof_command: launchPlan.post_run_proof_command || [],
    },
    prelaunch_readiness: {
      launch_ready: prelaunch.launch_ready === true,
      dry_run: prelaunch.dry_run === true,
      graviton_instance: prelaunch.graviton_instance === true,
      ec2_metadata_required: prelaunch.ec2_metadata_required === true,
      product_stages: prelaunch.product_stages || [],
      cpu_scaling: prelaunch.cpu_scaling || {},
      s3_uri: prelaunch.s3_uri || "",
      s3_pipeline_uri: prelaunch.s3_pipeline_uri || "",
      artifact_s3_uri: prelaunch.artifact_s3_uri || "",
      post_run_proof_command: prelaunch.post_run_proof_command || [],
    },
    release_proof_cases: releaseProof.cases || [],
    run_artifact_cases: runArtifacts.cases || [],
    run_fetch_cases: runFetch.cases || [],
    live_launch_readiness_cases: liveLaunchReadiness.cases || [],
    launch_execute_guard_cases: launchExecuteGuard.cases || [],
  };
}

function buildReport(config, diagnostic, objectiveResult) {
  const objective = objectiveResult.report || {};
  const evidence = diagnostic.evidence || {};
  const awsHandoff = buildAwsHandoff(evidence);
  const awsProductCheck = checkEntry(diagnostic, "aws-product-plan");
  const awsLaunchCheck = checkEntry(diagnostic, "aws-launch-plan");
  const awsPrelaunchCheck = checkEntry(diagnostic, "aws-prelaunch-readiness");
  const diagnosticPath = path.resolve(config.diagnosticPath);
  const objectiveDiagnosticPath = objective.diagnostic_path ? path.resolve(objective.diagnostic_path) : "";
  const objectiveStatusOk = objectiveResult.generated ? objectiveResult.status === 0 : true;
  const releaseProductProof = diagnostic.release_product_proof === true;
  const releaseObjectiveProof = objective.release_objective_proof === true;
  const releaseGaps = [
    ...(diagnostic.remaining_product_evidence || []),
    ...(objective.remaining_release_evidence || []),
  ].map(String).filter(Boolean);
  const productPlan = awsHandoff.product_plan;
  const launchPlan = awsHandoff.launch_plan;
  const prelaunch = awsHandoff.prelaunch_readiness;
  const productCpuScaling = productPlan.cpu_scaling || {};
  const launchCpuScaling = launchPlan.cpu_scaling || {};
  const prelaunchCpuScaling = prelaunch.cpu_scaling || {};
  const launchProofCommandProblems = postRunProofCommandProblems("launch plan", launchPlan);
  const prelaunchProofCommandProblems = postRunProofCommandProblems("prelaunch readiness", prelaunch);
  const productDenoiseRunnerProblems = denoiseRunnerProblems(
    productPlan.curriculum_denoise_runner,
    Math.max(productPlan.denoise_min_unique_targets, requiredMinDenoiseBridgeUniqueTargets),
  );
  const requirements = [
    requirement(
      "product_diagnostic_green",
      "product diagnostic ran the full local proof spine",
      diagnostic.schema === diagnosticSchema &&
        diagnostic.ok === true &&
        diagnostic.full_product_proof === true &&
        diagnostic.local_product_proof === true &&
        Array.isArray(diagnostic.skipped) &&
        diagnostic.skipped.length === 0,
      {
        schema: diagnostic.schema || "",
        ok: diagnostic.ok === true,
        full_product_proof: diagnostic.full_product_proof === true,
        local_product_proof: diagnostic.local_product_proof === true,
        skipped: diagnostic.skipped || [],
      },
      [
        diagnostic.schema === diagnosticSchema ? "" : `diagnostic schema ${JSON.stringify(diagnostic.schema || "")} != ${diagnosticSchema}`,
        diagnostic.ok === true ? "" : "diagnostic ok is not true",
        diagnostic.full_product_proof === true ? "" : "full_product_proof is not true",
        diagnostic.local_product_proof === true ? "" : "local_product_proof is not true",
        Array.isArray(diagnostic.skipped) && diagnostic.skipped.length === 0
          ? ""
          : `diagnostic has skipped checks: ${(diagnostic.skipped || []).join(", ")}`,
      ],
    ),
    requirement(
      "objective_coverage_green",
      "objective coverage maps the diagnostic to the narrow multimodal model objective",
      objective.schema === objectiveSchema &&
        objectiveStatusOk &&
        objective.ok === true &&
        objective.diagnostic_ok === true &&
        objective.local_objective_proof === true &&
        (!config.requireRelease || releaseObjectiveProof) &&
        (!objectiveDiagnosticPath || objectiveDiagnosticPath === diagnosticPath),
      {
        path: objectiveResult.path,
        generated: objectiveResult.generated,
        status: objectiveResult.status,
        schema: objective.schema || "",
        ok: objective.ok === true,
        diagnostic_path: objective.diagnostic_path || "",
        local_objective_proof: objective.local_objective_proof === true,
        release_objective_proof: releaseObjectiveProof,
        missing: objective.missing || [],
      },
      [
        objective.schema === objectiveSchema ? "" : `objective schema ${JSON.stringify(objective.schema || "")} != ${objectiveSchema}`,
        objectiveStatusOk ? "" : `objective coverage command exited ${objectiveResult.status}`,
        objective.ok === true ? "" : "objective coverage ok is not true",
        objective.diagnostic_ok === true ? "" : "objective diagnostic_ok is not true",
        objective.local_objective_proof === true ? "" : "local_objective_proof is not true",
        !config.requireRelease || releaseObjectiveProof ? "" : "release_objective_proof is not true",
        !objectiveDiagnosticPath || objectiveDiagnosticPath === diagnosticPath
          ? ""
          : `objective diagnostic_path ${objectiveDiagnosticPath} != ${diagnosticPath}`,
        ...(objective.missing || []).map((item) => `objective coverage: ${item}`),
      ],
    ),
    requirement(
      "aws_graviton_handoff_green",
      "AWS plan, launch plan, and prelaunch readiness are green for Graviton CPU scaling",
      awsProductCheck.ok === true &&
        awsLaunchCheck.ok === true &&
        awsPrelaunchCheck.ok === true &&
        includesAll(productPlan.stages, requiredProductStages) &&
        productPlan.promotion_bundle_check === true &&
        productPlan.require_graviton === true &&
        productPlan.s3_required === true &&
        productCpuScaling.policy === "auto-online-processors" &&
        productCpuScaling.auto_workers === true &&
        sameSequence(productPlan.curriculum_stages, requiredCurriculumStages) &&
        sameSequence(productPlan.curriculum_required_stages, requiredCurriculumStages) &&
        productPlan.native_bind_epochs >= requiredNativeBindEpochs &&
        productPlan.min_direction_top5_per_mille === "all=1" &&
        productPlan.min_task_targets === requiredMinTaskTargets &&
        productPlan.min_task_top5_per_mille === requiredMinTaskTop5PerMille &&
        productPlan.min_phase_targets === requiredMinPhaseTargets &&
        productPlan.require_promoted_small_profile === true &&
        productPlan.require_architecture_profile === true &&
        productPlan.seq_len >= requiredMinSeqLen &&
        productPlan.seq_len <= requiredMaxSeqLen &&
        productPlan.min_d_model === requiredDModel &&
        productPlan.min_heads === requiredHeads &&
        productPlan.target_head_dim === requiredHeadDim &&
        productPlan.min_hidden_dim >= requiredMinHiddenDim &&
        productPlan.max_hidden_dim >= productPlan.min_hidden_dim &&
        productPlan.max_hidden_dim <= requiredMaxHiddenDim &&
        productPlan.min_transformer_layers >= requiredMinTransformerLayers &&
        productPlan.min_context_seq_len >= requiredMinContextSeqLen &&
        productPlan.require_generative_eval === true &&
        productPlan.require_generative_output_identity === true &&
        productPlan.min_generated_prompt_rows >= 72 &&
        productPlan.generated_prompt_rows >= 72 &&
        productPlan.generated_unique_targets >= 72 &&
        productPlan.min_generated_top5_16_per_mille >= 1 &&
        productPlan.min_generated_retrieval_top1_per_mille >= 1000 &&
        productPlan.min_generated_retrieval_top5_per_mille >= 1000 &&
        productPlan.min_generated_retrieval_margin >= requiredMinGeneratedRetrievalMargin &&
        Number.isInteger(productPlan.max_generated_mean_target_distance_16_q8) &&
        productPlan.max_generated_mean_target_distance_16_q8 >= 1 &&
        productPlan.max_generated_mean_target_distance_16_q8 <= requiredMaxGeneratedMeanTargetDistance16Q8 &&
        productPlan.denoise_min_unique_targets >= requiredMinDenoiseBridgeUniqueTargets &&
        productDenoiseRunnerProblems.length === 0 &&
        includesAll(awsHandoff.release_proof_cases, requiredAwsReleaseProofCases) &&
        includesAll(awsHandoff.run_artifact_cases, requiredAwsRunArtifactCases) &&
        includesAll(awsHandoff.run_fetch_cases, requiredAwsRunFetchCases) &&
        includesAll(awsHandoff.live_launch_readiness_cases, requiredAwsLiveLaunchReadinessCases) &&
        includesAll(awsHandoff.launch_execute_guard_cases, requiredAwsLaunchExecuteGuardCases) &&
        launchPlan.dry_run === true &&
        launchPlan.graviton_instance === true &&
        launchPlan.ec2_metadata_required === true &&
        launchCpuScaling.policy === "auto-online-processors" &&
        launchProofCommandProblems.length === 0 &&
        prelaunch.launch_ready === true &&
        prelaunch.dry_run === true &&
        prelaunch.graviton_instance === true &&
        prelaunch.ec2_metadata_required === true &&
        prelaunchCpuScaling.policy === "auto-online-processors" &&
        prelaunchProofCommandProblems.length === 0,
      {
        aws_product_check_ok: awsProductCheck.ok === true,
        aws_launch_check_ok: awsLaunchCheck.ok === true,
        aws_prelaunch_check_ok: awsPrelaunchCheck.ok === true,
        ...awsHandoff,
      },
      [
        awsProductCheck.ok === true ? "" : "aws-product-plan check is not ok",
        awsLaunchCheck.ok === true ? "" : "aws-launch-plan check is not ok",
        awsPrelaunchCheck.ok === true ? "" : "aws-prelaunch-readiness check is not ok",
        includesAll(productPlan.stages, requiredProductStages)
          ? ""
          : `product plan stages missing one of ${requiredProductStages.join(", ")}`,
        productPlan.promotion_bundle_check === true ? "" : "product plan is missing promotion-bundle-check",
        productPlan.require_graviton === true ? "" : "product plan does not require Graviton",
        productPlan.s3_required === true ? "" : "product plan does not require S3 artifacts",
        productCpuScaling.policy === "auto-online-processors" ? "" : "product CPU scaling policy is not auto-online-processors",
        productCpuScaling.auto_workers === true ? "" : "product CPU scaling auto_workers is not true",
        sameSequence(productPlan.curriculum_stages, requiredCurriculumStages) ? "" : "product curriculum stage order is not the required order",
        sameSequence(productPlan.curriculum_required_stages, requiredCurriculumStages)
          ? ""
          : "required curriculum stage order is not the required order",
        productPlan.native_bind_epochs >= requiredNativeBindEpochs
          ? ""
          : `native-bind epoch floor is below ${requiredNativeBindEpochs}`,
        productPlan.min_direction_top5_per_mille === "all=1" ? "" : "per-direction top-5 floor is not all=1",
        productPlan.min_task_targets === requiredMinTaskTargets
          ? ""
          : `per-task target floor is not ${requiredMinTaskTargets}`,
        productPlan.min_task_top5_per_mille === requiredMinTaskTop5PerMille
          ? ""
          : `per-task top-5 floor is not ${requiredMinTaskTop5PerMille}`,
        productPlan.min_phase_targets === requiredMinPhaseTargets
          ? ""
          : `per-phase target floor is not ${requiredMinPhaseTargets}`,
        productPlan.require_promoted_small_profile === true ? "" : "product plan does not require promoted small profile",
        productPlan.require_architecture_profile === true ? "" : "product plan does not require architecture profile",
        productPlan.seq_len >= requiredMinSeqLen && productPlan.seq_len <= requiredMaxSeqLen
          ? ""
          : `product seq_len is outside ${requiredMinSeqLen}-${requiredMaxSeqLen}`,
        productPlan.min_d_model === requiredDModel
          ? ""
          : `product min_d_model is not ${requiredDModel}`,
        productPlan.min_heads === requiredHeads
          ? ""
          : `product min_heads is not ${requiredHeads}`,
        productPlan.target_head_dim === requiredHeadDim
          ? ""
          : `product target_head_dim is not ${requiredHeadDim}`,
        productPlan.min_hidden_dim >= requiredMinHiddenDim
          ? ""
          : `product min_hidden_dim is below ${requiredMinHiddenDim}`,
        productPlan.max_hidden_dim >= productPlan.min_hidden_dim &&
        productPlan.max_hidden_dim <= requiredMaxHiddenDim
          ? ""
          : `product max_hidden_dim is outside ${productPlan.min_hidden_dim}-${requiredMaxHiddenDim}`,
        productPlan.min_transformer_layers >= requiredMinTransformerLayers
          ? ""
          : `product min_transformer_layers is below ${requiredMinTransformerLayers}`,
        productPlan.min_context_seq_len >= requiredMinContextSeqLen
          ? ""
          : `product min_context_seq_len is below ${requiredMinContextSeqLen}`,
        productPlan.require_generative_eval === true ? "" : "product plan does not require generated eval",
        productPlan.require_generative_output_identity === true ? "" : "product plan does not require generated output identity",
        productPlan.min_generated_prompt_rows >= 72 ? "" : "generated prompt-row floor is below 72",
        productPlan.generated_prompt_rows >= 72 ? "" : "generated held-out prompt rows are below 72",
        productPlan.generated_unique_targets >= 72 ? "" : "generated held-out unique targets are below 72",
        productPlan.min_generated_top5_16_per_mille >= 1 ? "" : "generated 16x16 top-5 floor is below 1",
        productPlan.min_generated_retrieval_top1_per_mille >= 1000 ? "" : "generated retrieval top-1 floor is below 1000",
        productPlan.min_generated_retrieval_top5_per_mille >= 1000 ? "" : "generated retrieval top-5 floor is below 1000",
        productPlan.min_generated_retrieval_margin >= requiredMinGeneratedRetrievalMargin
          ? ""
          : `generated retrieval margin floor is below ${requiredMinGeneratedRetrievalMargin}`,
        Number.isInteger(productPlan.max_generated_mean_target_distance_16_q8) &&
        productPlan.max_generated_mean_target_distance_16_q8 >= 1 &&
        productPlan.max_generated_mean_target_distance_16_q8 <= requiredMaxGeneratedMeanTargetDistance16Q8
          ? ""
          : `generated 16x16 target-distance cap is outside 1-${requiredMaxGeneratedMeanTargetDistance16Q8}`,
        productPlan.denoise_min_unique_targets >= requiredMinDenoiseBridgeUniqueTargets
          ? ""
          : `denoise bridge unique-target floor is below ${requiredMinDenoiseBridgeUniqueTargets}`,
        ...productDenoiseRunnerProblems,
        includesAll(awsHandoff.release_proof_cases, requiredAwsReleaseProofCases)
          ? ""
          : `release proof self-test missing one of ${requiredAwsReleaseProofCases.join(", ")}`,
        includesAll(awsHandoff.run_artifact_cases, requiredAwsRunArtifactCases)
          ? ""
          : `run artifact self-test missing one of ${requiredAwsRunArtifactCases.join(", ")}`,
        includesAll(awsHandoff.run_fetch_cases, requiredAwsRunFetchCases)
          ? ""
          : `run fetch self-test missing one of ${requiredAwsRunFetchCases.join(", ")}`,
        includesAll(awsHandoff.live_launch_readiness_cases, requiredAwsLiveLaunchReadinessCases)
          ? ""
          : `live launch readiness self-test missing one of ${requiredAwsLiveLaunchReadinessCases.join(", ")}`,
        includesAll(awsHandoff.launch_execute_guard_cases, requiredAwsLaunchExecuteGuardCases)
          ? ""
          : `launch execute guard self-test missing one of ${requiredAwsLaunchExecuteGuardCases.join(", ")}`,
        launchPlan.dry_run === true ? "" : "launch plan is not a no-spend dry run",
        launchPlan.graviton_instance === true ? "" : "launch plan is not Graviton",
        launchPlan.ec2_metadata_required === true ? "" : "launch plan does not require EC2 metadata",
        launchCpuScaling.policy === "auto-online-processors" ? "" : "launch CPU scaling policy is not auto-online-processors",
        ...launchProofCommandProblems,
        prelaunch.launch_ready === true ? "" : "prelaunch readiness is not green",
        prelaunch.dry_run === true ? "" : "prelaunch plan is not dry-run evidence",
        prelaunch.graviton_instance === true ? "" : "prelaunch plan is not Graviton",
        prelaunch.ec2_metadata_required === true ? "" : "prelaunch plan does not require EC2 metadata",
        prelaunchCpuScaling.policy === "auto-online-processors" ? "" : "prelaunch CPU scaling policy is not auto-online-processors",
        ...prelaunchProofCommandProblems,
      ],
    ),
    requirement(
      "release_gap_accounted",
      "remaining release evidence is either complete or exactly the known synced-run gap",
      config.requireRelease
        ? releaseProductProof && releaseObjectiveProof && releaseGaps.length === 0
        : releaseGapOk(diagnostic.remaining_product_evidence, config) && releaseGapOk(objective.remaining_release_evidence, config),
      {
        require_release: config.requireRelease,
        release_product_proof: releaseProductProof,
        release_objective_proof: releaseObjectiveProof,
        diagnostic_remaining_product_evidence: diagnostic.remaining_product_evidence || [],
        objective_remaining_release_evidence: objective.remaining_release_evidence || [],
        expected_no_spend_gap: expectedReleaseGap,
      },
      config.requireRelease
        ? [
          releaseProductProof ? "" : "release_product_proof is not true",
          releaseObjectiveProof ? "" : "release_objective_proof is not true",
          releaseGaps.length === 0 ? "" : `remaining release evidence is not empty: ${releaseGaps.join("; ")}`,
        ]
        : [
          releaseGapOk(diagnostic.remaining_product_evidence, config)
            ? ""
            : `diagnostic release gaps are not limited to: ${expectedReleaseGap}`,
          releaseGapOk(objective.remaining_release_evidence, config)
            ? ""
            : `objective release gaps are not limited to: ${expectedReleaseGap}`,
        ],
    ),
  ];

  const ok = requirements.every((item) => item.ok);
  const candidateState = ok
    ? (releaseProductProof && releaseObjectiveProof ? "release-proof" : "local-release-candidate")
    : "not-ready";
  return {
    schema,
    ok,
    candidate_state: candidateState,
    require_release: config.requireRelease,
    generated_at: new Date().toISOString(),
    diagnostic: {
      path: diagnosticPath,
      schema: diagnostic.schema || "",
      ok: diagnostic.ok === true,
      full_product_proof: diagnostic.full_product_proof === true,
      local_product_proof: diagnostic.local_product_proof === true,
      release_product_proof: releaseProductProof,
      remaining_product_evidence: diagnostic.remaining_product_evidence || [],
      skipped: diagnostic.skipped || [],
    },
    objective_coverage: {
      path: objectiveResult.path,
      generated: objectiveResult.generated,
      status: objectiveResult.status,
      schema: objective.schema || "",
      ok: objective.ok === true,
      local_objective_proof: objective.local_objective_proof === true,
      release_objective_proof: releaseObjectiveProof,
      remaining_release_evidence: objective.remaining_release_evidence || [],
      missing: objective.missing || [],
      stdout_tail: objectiveResult.generated ? objectiveResult.stdout_tail : "",
      stderr_tail: objectiveResult.generated ? objectiveResult.stderr_tail : "",
    },
    aws_handoff: awsHandoff,
    next_operator_action: releaseProductProof && releaseObjectiveProof
      ? "archive release-proof.json with the promoted Solomon product artifacts"
      : localNextOperatorAction,
    requirements,
    errors: requirements.filter((item) => !item.ok).flatMap((item) => item.missing.map((reason) => `${item.key}: ${reason}`)),
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const diagnostic = readJson(config.diagnosticPath);
  const objectiveResult = loadObjectiveCoverage(config);
  const report = buildReport(config, diagnostic, objectiveResult);
  writeReport(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
