#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_product_diagnostic_check.v1";

const expectedSchemas = {
  "v2-corpus-contract": "nsrl.solomon_v2_corpus_contract_check.v1",
  "symbolic-image-self-test": "nsrl.solomon_symbolic_image_self_test.v1",
  "token-layout-self-test": "nsrl.solomon_token_layout_self_test.v1",
  "heldout-retrieval-proof": "nsrl.solomon_heldout_retrieval_proof.v1",
  "heldout-retrieval-proof-self-test": "nsrl.solomon_heldout_retrieval_proof_self_test.v1",
  "grounded-corpus-self-test": "nsrl.solomon_v2_grounded_corpus_self_test.v1",
  "native-directional-eval": "nsrl.solomon_native_directional_eval_smoke.v1",
  "prior-smoke-self-test": "nsrl.solomon_prior_smoke_self_test.v1",
  "task-eval-self-test": "nsrl.solomon_attention_task_eval_self_test.v1",
  "quality-report-self-test": "nsrl.solomon_v2_quality_report_self_test.v1",
  "objective-coverage-self-test": "nsrl.solomon_objective_coverage_self_test.v1",
  "release-candidate-self-test": "nsrl.solomon_release_candidate_self_test.v1",
  "generative-eval-provenance": "nsrl.solomon_generative_eval_provenance_check.v1",
  "generation-integrity-self-test": "nsrl.solomon_generation_integrity_self_test.v1",
  "sample-binding-self-test": "nsrl.solomon_attention_sample_binding_self_test.v1",
  "denoise-bridge-self-test": "nsrl.solomon_attention_denoise_bridge_self_test.v1",
  "promotion-bundle-self-test": "nsrl.solomon_promotion_bundle_self_test.v1",
  "aws-product-plan": "nsrl.solomon_aws_product_plan_check.v1",
  "aws-launch-plan": "nsrl.solomon_aws_product_launch_plan_check.v1",
  "aws-prelaunch-readiness": "nsrl.solomon_aws_prelaunch_readiness_check.v1",
  "aws-live-launch-readiness-self-test": "nsrl.solomon_aws_live_launch_readiness_self_test.v1",
  "aws-launch-execute-guard-self-test": "nsrl.solomon_aws_launch_execute_guard_self_test.v1",
  "aws-run-artifacts-self-test": "nsrl.solomon_aws_run_artifacts_self_test.v1",
  "aws-run-fetch-self-test": "nsrl.solomon_aws_run_fetch_self_test.v1",
  "aws-release-proof-self-test": "nsrl.solomon_aws_product_release_proof_self_test.v1",
  "aws-run-artifacts": "nsrl.solomon_aws_run_artifacts_check.v1",
};

function usage() {
  console.log([
    "Usage: check-solomon-product-diagnostic.mjs [options]",
    "",
    "Runs the local Solomon product proof spine and emits one JSON diagnostic.",
    "By default this includes the v2 corpus contract, the real held-out retrieval",
    "proof, the held-out retrieval contract self-test, the grounded-corpus",
    "contract self-test, the promoted-context native directional eval smoke, the prior-smoke",
    "contract self-test, the synthetic",
    "task-eval, quality-report, objective-coverage, and release-candidate contract self-tests, provenance",
    "self-tests, the generation-integrity guardrail self-test, the generated sample-binding self-test, the denoise bridge",
    "self-test, the promotion bundle self-test, the completed-run artifact",
    "self-test, the completed-run fetch self-test, the release-proof wrapper",
    "self-test, the AWS Graviton dry-run plan check, the EC2 launch-plan",
    "and prelaunch-readiness checks with duplicate shell self-tests disabled,",
    "the live-launch-readiness self-test that requires explicit S3 handoff",
    "inputs, and the execute-path guard self-test that proves bad plans fail",
    "before AWS.",
    "",
    "Options:",
    "  --out PATH                  write the diagnostic JSON to PATH",
    "  --keep                      keep the AWS dry-run scratch directory",
    "  --fast                      skip the slower corpus, held-out, and native proofs",
    "  --skip-corpus-contract      skip check-solomon-v2-corpus-contract.mjs",
    "  --skip-heldout-retrieval    skip check-solomon-heldout-retrieval-proof.mjs",
    "  --skip-native               skip check-solomon-native-directional-eval-smoke.mjs",
    "  --skip-aws-plan             skip check-solomon-aws-product-plan.sh",
    "  --skip-aws-launch           skip check-solomon-aws-launch-plan.sh",
    "  --aws-run-dir PATH           also verify a synced real Graviton run directory",
    "  --require-aws-run           fail unless --aws-run-dir is supplied and passes",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = {
    outPath: "",
    keep: false,
    skipCorpusContract: false,
    skipHeldoutRetrieval: false,
    skipNative: false,
    skipAwsPlan: false,
    skipAwsLaunch: false,
    awsRunDir: "",
    requireAwsRun: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
    } else if (arg === "--fast") {
      config.skipCorpusContract = true;
      config.skipHeldoutRetrieval = true;
      config.skipNative = true;
    } else if (arg === "--skip-corpus-contract") {
      config.skipCorpusContract = true;
    } else if (arg === "--skip-heldout-retrieval") {
      config.skipHeldoutRetrieval = true;
    } else if (arg === "--skip-native") {
      config.skipNative = true;
    } else if (arg === "--skip-aws-plan") {
      config.skipAwsPlan = true;
    } else if (arg === "--skip-aws-launch") {
      config.skipAwsLaunch = true;
    } else if (arg === "--aws-run-dir") {
      config.awsRunDir = requireValue(argv, ++index, arg);
    } else if (arg === "--require-aws-run") {
      config.requireAwsRun = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.requireAwsRun && !config.awsRunDir) {
    throw new Error("--require-aws-run requires --aws-run-dir PATH");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function productChecks(config, scratchRoot) {
  const checks = [];
  if (!config.skipCorpusContract) {
    checks.push({
      name: "v2-corpus-contract",
      command: process.execPath,
      args: ["scripts/check-solomon-v2-corpus-contract.mjs"],
    });
  }
  if (!config.skipHeldoutRetrieval) {
    checks.push({
      name: "heldout-retrieval-proof",
      command: process.execPath,
      args: ["scripts/check-solomon-heldout-retrieval-proof.mjs"],
    });
  }
  if (!config.skipNative) {
    checks.push({
      name: "native-directional-eval",
      command: process.execPath,
      args: ["scripts/check-solomon-native-directional-eval-smoke.mjs"],
    });
  }
  checks.push(
    {
      name: "symbolic-image-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-symbolic-image-self-test.mjs"],
    },
    {
      name: "token-layout-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-token-layout-self-test.mjs"],
    },
    {
      name: "heldout-retrieval-proof-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-heldout-retrieval-proof-self-test.mjs"],
    },
    {
      name: "grounded-corpus-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-v2-grounded-corpus-self-test.mjs"],
    },
    {
      name: "prior-smoke-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-prior-smoke-self-test.mjs"],
    },
    {
      name: "task-eval-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-attention-task-eval-self-test.mjs"],
    },
    {
      name: "quality-report-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-v2-quality-report-self-test.mjs"],
    },
    {
      name: "objective-coverage-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-objective-coverage-self-test.mjs"],
    },
    {
      name: "release-candidate-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-release-candidate-self-test.mjs"],
    },
    {
      name: "generative-eval-provenance",
      command: process.execPath,
      args: ["scripts/check-solomon-generative-eval-provenance.mjs"],
    },
    {
      name: "generation-integrity-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-generation-integrity-self-test.mjs"],
    },
    {
      name: "sample-binding-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-attention-sample-binding-self-test.mjs"],
    },
    {
      name: "denoise-bridge-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-attention-denoise-bridge-self-test.mjs"],
    },
    {
      name: "promotion-bundle-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-promotion-bundle-self-test.mjs"],
    },
    {
      name: "aws-run-artifacts-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-aws-run-artifacts-self-test.mjs"],
    },
    {
      name: "aws-run-fetch-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-aws-run-fetch-self-test.mjs"],
    },
    {
      name: "aws-release-proof-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-aws-release-proof-self-test.mjs"],
    },
  );
  if (!config.skipAwsPlan) {
    checks.push({
      name: "aws-product-plan",
      command: "bash",
      args: ["scripts/check-solomon-aws-product-plan.sh"],
      env: {
        NSRL_AWS_PRODUCT_PLAN_CHECK_ROOT: scratchRoot,
        NSRL_AWS_PRODUCT_PLAN_CHECK_NAME: `solomon-product-diagnostic-${Date.now()}-${process.pid}`,
        NSRL_AWS_PRODUCT_PLAN_CHECK_SELF_TEST: "0",
      },
    });
  }
  if (!config.skipAwsLaunch) {
    checks.push({
      name: "aws-launch-plan",
      command: "bash",
      args: ["scripts/check-solomon-aws-launch-plan.sh"],
      env: {
        NSRL_AWS_LAUNCH_PLAN_CHECK_ROOT: scratchRoot,
        NSRL_AWS_LAUNCH_PLAN_CHECK_NAME: `solomon-product-diagnostic-launch-${Date.now()}-${process.pid}`,
        NSRL_AWS_LAUNCH_PLAN_CHECK_SELF_TEST: "0",
      },
    });
    checks.push({
      name: "aws-prelaunch-readiness",
      command: "bash",
      args: ["scripts/check-solomon-aws-prelaunch-readiness.sh"],
      env: {
        NSRL_AWS_PRELAUNCH_READINESS_CHECK_ROOT: scratchRoot,
        NSRL_AWS_PRELAUNCH_READINESS_CHECK_NAME: `solomon-product-diagnostic-prelaunch-${Date.now()}-${process.pid}`,
        NSRL_AWS_PRELAUNCH_READINESS_SELF_TEST: "0",
      },
    });
    checks.push({
      name: "aws-live-launch-readiness-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-aws-live-launch-readiness-self-test.mjs"],
    });
    checks.push({
      name: "aws-launch-execute-guard-self-test",
      command: process.execPath,
      args: ["scripts/check-solomon-aws-launch-execute-guard-self-test.mjs"],
    });
  }
  if (config.awsRunDir) {
    checks.push({
      name: "aws-run-artifacts",
      command: process.execPath,
      args: [
        "scripts/check-solomon-aws-run-artifacts.mjs",
        "--run-dir",
        config.awsRunDir,
      ],
    });
  }
  return checks;
}

function skippedChecks(config) {
  const skipped = [];
  if (config.skipCorpusContract) skipped.push("v2-corpus-contract");
  if (config.skipHeldoutRetrieval) skipped.push("heldout-retrieval-proof");
  if (config.skipNative) skipped.push("native-directional-eval");
  if (config.skipAwsPlan) skipped.push("aws-product-plan");
  if (config.skipAwsLaunch) skipped.push("aws-launch-plan");
  return skipped;
}

function runCheck(check) {
  const started = Date.now();
  const result = childProcess.spawnSync(check.command, check.args, {
    cwd: repoRoot,
    encoding: "utf8",
    env: { ...process.env, ...(check.env || {}) },
  });
  const durationMs = Date.now() - started;
  const jsonObjects = extractJsonObjects(result.stdout || "");
  const parsed = selectPrimaryJson(check.name, jsonObjects);
  const schemaOk = parsed ? parsed.schema === expectedSchemas[check.name] : false;
  const exitOk = result.status === 0;
  const parsedOk = parsed ? parsed.ok !== false : false;
  const ok = exitOk && schemaOk && parsedOk;
  const entry = {
    name: check.name,
    ok,
    command: commandLine(check.command, check.args),
    duration_ms: durationMs,
    status: result.status,
    signal: result.signal || "",
    schema: parsed?.schema || "",
    summary: summarizeCheck(check.name, parsed),
  };
  if (!ok) {
    entry.errors = [];
    if (!exitOk) entry.errors.push(`command exited with status ${result.status}`);
    if (!parsed) entry.errors.push("missing JSON report");
    if (parsed && !schemaOk) entry.errors.push(`schema ${JSON.stringify(parsed.schema || "")} did not match ${expectedSchemas[check.name]}`);
    if (parsed && parsed.ok === false) entry.errors.push("JSON report returned ok=false");
    if (Array.isArray(parsed?.errors) && parsed.errors.length > 0) {
      entry.errors.push(...parsed.errors.slice(0, 20).map(String));
    }
    entry.stdout_tail = tailLines(result.stdout || "", 80);
    entry.stderr_tail = tailLines(result.stderr || "", 80);
  }
  return { entry, parsed };
}

function selectPrimaryJson(name, jsonObjects) {
  const expected = expectedSchemas[name];
  return jsonObjects.find((item) => item && item.schema === expected) || jsonObjects[jsonObjects.length - 1] || null;
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
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        objects.push(parsed);
      }
      index = end;
    } catch {
      // Logs can contain braces; only complete JSON objects are diagnostic evidence.
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

function summarizeCheck(name, report) {
  if (!report) {
    return {};
  }
  if (name === "v2-corpus-contract") {
    return {
      examples: report.examples || 0,
      token_count: report.token_count || 0,
      task_counts: report.task_counts || {},
      image_token_profile: report.image_token_profile || "",
      image_token_channels: report.image_token_channels || [],
      image_token_channel_stats: report.image_token_channel_stats || {},
      hard_negative_roles: report.hard_negative_roles || {},
      identity_binding_coverage: report.identity_binding_coverage || {},
      source_provenance: report.source_provenance || {},
      task_marker_integrity: report.task_marker_integrity || {},
      task_modality_integrity: report.task_modality_integrity || {},
      image_channel_marker_integrity: report.image_channel_marker_integrity || {},
      retrieval_model_hash: report.retrieval_model_hash || "",
      retrieval_head: report.retrieval_head || {},
      negative_cases: (report.negative_cases || []).map((item) => item.name || item),
    };
  }
  if (name === "native-directional-eval") {
    return {
      eval_scope: report.eval_scope || {},
      architecture: report.architecture || {},
      integer_trace: report.integer_trace || {},
      output_heads: outputHeadTargets(report.output_heads || {}),
      tasks: report.tasks || {},
      task_phases: report.task_phases || {},
      task_phase_tasks: report.task_phase_tasks || 0,
      directional_groups: report.directional_groups || {},
    };
  }
  if (name === "heldout-retrieval-proof") {
    return {
      corpus: {
        rows: report.corpus?.rows || 0,
        corpus_version: report.corpus?.corpus_version || "",
        image_token_profile: report.corpus?.image_token_profile || "",
        image_token_channels: report.corpus?.image_token_channels || [],
      },
      retrieval_head: report.retrieval_head || {},
      heldout_prompts: {
        rows: report.heldout_prompts?.rows || 0,
        unique_targets: report.heldout_prompts?.unique_targets || 0,
        prompts_hash: report.heldout_prompts?.prompts_hash || "",
        metric: report.heldout_prompts?.metric || {},
      },
      known_prompts: report.known_prompts || {},
      image_to_text: report.image_to_text || {},
      match: report.match || {},
    };
  }
  if (name === "heldout-retrieval-proof-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "symbolic-image-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "token-layout-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
      canonical_layout: report.canonical_layout || {},
    };
  }
  if (name === "grounded-corpus-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "generative-eval-provenance") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
      clean_sample: report.clean_sample || {},
      posthoc_sample: report.posthoc_sample || {},
    };
  }
  if (name === "generation-integrity-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "prior-smoke-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "task-eval-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "quality-report-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "objective-coverage-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "sample-binding-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "denoise-bridge-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "promotion-bundle-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "aws-run-artifacts-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "aws-run-fetch-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "aws-release-proof-self-test") {
    return {
      include_slow_positive: report.include_slow_positive === true,
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "aws-live-launch-readiness-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  if (name === "release-candidate-self-test") {
    const cases = Array.isArray(report.cases) ? report.cases : [];
    const nextActionCases = {};
    for (const item of cases) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const expected = Array.isArray(item.expected_next_action_includes)
        ? item.expected_next_action_includes.map(String)
        : [];
      if (expected.length === 0) {
        continue;
      }
      nextActionCases[String(item.name || "")] = {
        expected_next_action_includes: expected,
        matched_next_action_includes: Array.isArray(item.matched_next_action_includes)
          ? item.matched_next_action_includes.map(String)
          : [],
      };
    }
    return {
      cases: cases.map((item) => item.name || item),
      next_action_cases: nextActionCases,
    };
  }
  if (name === "aws-run-artifacts") {
    return {
      run_dir: report.run_dir || "",
      run_name: report.run_name || "",
      dry_run: report.dry_run === true,
      runner: report.runner || {},
      s3: report.s3 || {},
      plan_stages: report.plan_stages || [],
      completion: report.completion || {},
      product_config: report.product_config || {},
      promotion: report.promotion || {},
      quality_report: report.quality_report || {},
    };
  }
  if (name === "aws-product-plan") {
    return {
      stages: report.stages || [],
      required_plan_stages: report.required_plan_stages || [],
      promotion_bundle_check: report.promotion_bundle_check === true,
      runner: report.runner || {},
      s3_required: report.s3?.required === true,
      attention: {
        corpus_version: report.attention?.corpus_version || "",
        text_token_profile: report.attention?.text_token_profile || "",
        image_token_profile: report.attention?.image_token_profile || "",
        seq_len: report.attention?.seq_len || 0,
        cpu_scaling: report.attention?.cpu_scaling || {},
        eval_max_examples: report.attention?.eval_max_examples || "",
        v2_stage_epochs: report.attention?.v2_stage_epochs || 0,
        native_bind_epochs: report.attention?.native_bind_epochs || 0,
        curriculum_stages: report.attention?.curriculum_stages || [],
        curriculum_required_stages: report.attention?.curriculum_required_stages || [],
        require_directional_groups: report.attention?.require_directional_groups === true,
        min_direction_accuracy_per_mille: report.attention?.min_direction_accuracy_per_mille || "",
        min_direction_top5_per_mille: report.attention?.min_direction_top5_per_mille || "",
        min_direction_top10_per_mille: report.attention?.min_direction_top10_per_mille || "",
        min_task_targets: report.attention?.min_task_targets || "",
        min_task_top5_per_mille: report.attention?.min_task_top5_per_mille || "",
        min_phase_targets: report.attention?.min_phase_targets || "",
        require_promoted_small_profile: report.attention?.require_promoted_small_profile === true,
        require_architecture_profile: report.attention?.require_architecture_profile === true,
        min_d_model: report.attention?.min_d_model || 0,
        min_heads: report.attention?.min_heads || 0,
        target_head_dim: report.attention?.target_head_dim || 0,
        min_hidden_dim: report.attention?.min_hidden_dim || 0,
        max_hidden_dim: report.attention?.max_hidden_dim || 0,
        min_transformer_layers: report.attention?.min_transformer_layers || 0,
        min_context_seq_len: report.attention?.min_context_seq_len || 0,
        train_core_architecture: report.attention?.train_core_architecture || {},
        curriculum_denoise_runner: report.attention?.curriculum_denoise_runner || {},
        require_generative_eval: report.attention?.require_generative_eval === true,
        require_generative_output_identity: report.attention?.require_generative_output_identity === true,
        min_generated_prompt_rows: report.attention?.min_generated_prompt_rows || 0,
        min_generated_top5_16_per_mille: report.attention?.min_generated_top5_16_per_mille || 0,
        min_generated_retrieval_top1_per_mille:
          report.attention?.min_generated_retrieval_top1_per_mille || 0,
        min_generated_retrieval_top5_per_mille:
          report.attention?.min_generated_retrieval_top5_per_mille || 0,
        min_generated_retrieval_margin:
          report.attention?.min_generated_retrieval_margin || 0,
        max_generated_mean_target_distance_16_q8:
          report.attention?.max_generated_mean_target_distance_16_q8 || 0,
        denoise_min_unique_targets: report.attention?.denoise_min_unique_targets || 0,
      },
      generated_prompt_rows: report.generative_eval?.prompt_artifact?.selected_prompt_eligible_rows || 0,
      generated_unique_targets: report.generative_eval?.prompt_artifact?.selected_eligible_unique_targets || 0,
    };
  }
  if (name === "aws-launch-plan") {
    return {
      dry_run: report.dry_run === true,
      instance_type: report.instance_type || "",
      graviton_instance: report.graviton_instance === true,
      ec2_metadata_required: report.ec2_metadata_required === true,
      product_stages: report.product_stages || [],
      cpu_scaling: report.cpu_scaling || {},
      s3_uri: report.s3_uri || "",
      s3_pipeline_uri: report.s3_pipeline_uri || "",
      artifact_s3_uri: report.artifact_s3_uri || "",
      post_run_proof_command: report.post_run_proof_command || [],
    };
  }
  if (name === "aws-prelaunch-readiness") {
    return {
      launch_ready: report.launch_ready === true,
      dry_run: report.dry_run === true,
      ami_id: report.ami_id || "",
      instance_type: report.instance_type || "",
      graviton_instance: report.graviton_instance === true,
      ec2_metadata_required: report.ec2_metadata_required === true,
      iam_instance_profile: report.iam_instance_profile || "",
      subnet_id: report.subnet_id || "",
      security_group_ids: report.security_group_ids || [],
      product_stages: report.product_stages || [],
      cpu_scaling: report.cpu_scaling || {},
      s3_uri: report.s3_uri || "",
      s3_pipeline_uri: report.s3_pipeline_uri || "",
      artifact_s3_uri: report.artifact_s3_uri || "",
      post_run_proof_command: report.post_run_proof_command || [],
    };
  }
  if (name === "aws-launch-execute-guard-self-test") {
    return {
      cases: (report.cases || []).map((item) => item.name || item),
    };
  }
  return {};
}

function outputHeadTargets(outputHeads) {
  const summary = {};
  for (const [name, head] of Object.entries(outputHeads)) {
    summary[name] = {
      source: head?.source || "",
      targets: Number(head?.stats?.targets || 0),
    };
  }
  return summary;
}

function commandLine(command, args) {
  return [command, ...args].map(shellQuote).join(" ");
}

function shellQuote(value) {
  const text = String(value);
  return /^[A-Za-z0-9_./:=+-]+$/.test(text) ? text : `'${text.replace(/'/g, "'\\''")}'`;
}

function tailLines(text, maxLines) {
  const lines = String(text).split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  const resolved = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const scratchRoot = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-product-diagnostic-"));
  const started = Date.now();
  const startedAt = new Date(started).toISOString();
  const checks = [];
  const evidence = {};
  let completed = false;
  try {
    for (const check of productChecks(config, scratchRoot)) {
      const { entry, parsed } = runCheck(check);
      checks.push(entry);
      evidence[check.name] = entry.summary;
      if (!entry.ok && parsed) {
        evidence[check.name].raw_ok = parsed.ok === true;
      }
    }
    completed = true;
    const skipped = skippedChecks(config);
    const ok = checks.every((check) => check.ok);
    const coreChecks = checks.filter((check) => check.name !== "aws-run-artifacts");
    const localProductProof = coreChecks.every((check) => check.ok) && skipped.length === 0;
    const awsRunCheck = checks.find((check) => check.name === "aws-run-artifacts");
    const awsRunEvidenceProvided = Boolean(config.awsRunDir);
    const awsRunEvidenceOk = awsRunCheck?.ok === true;
    const releaseProductProof = localProductProof && awsRunEvidenceOk;
    const finished = Date.now();
    const report = {
      schema,
      ok,
      full_product_proof: localProductProof,
      local_product_proof: localProductProof,
      release_product_proof: releaseProductProof,
      live_product_evidence: {
        required: config.requireAwsRun,
        provided: awsRunEvidenceProvided,
        ok: awsRunEvidenceOk,
        run_dir: config.awsRunDir || "",
        check: awsRunCheck
          ? {
            name: awsRunCheck.name,
            ok: awsRunCheck.ok,
            status: awsRunCheck.status,
            schema: awsRunCheck.schema,
            summary: awsRunCheck.summary,
          }
          : null,
      },
      remaining_product_evidence: remainingProductEvidence({
        localProductProof,
        skipped,
        awsRunEvidenceProvided,
        awsRunEvidenceOk,
      }),
      started_at: startedAt,
      finished_at: new Date(finished).toISOString(),
      duration_ms: finished - started,
      repo_root: repoRoot,
      aws_scratch_dir: scratchRoot,
      aws_scratch_kept: config.keep,
      skipped,
      checks,
      evidence,
    };
    writeReport(config.outPath, report);
    console.log(JSON.stringify(report, null, 2));
    if (!ok || (config.requireAwsRun && !releaseProductProof)) {
      process.exitCode = 1;
    }
  } finally {
    if (!config.keep && completed) {
      fs.rmSync(scratchRoot, { recursive: true, force: true });
    } else if (!config.keep) {
      console.error(`aws_scratch_dir: ${scratchRoot}`);
    }
  }
}

function remainingProductEvidence(state) {
  const remaining = [];
  if (!state.localProductProof) {
    remaining.push("local product proof is incomplete or one of the default checks was skipped");
  }
  if (!state.awsRunEvidenceProvided) {
    remaining.push("no synced real Graviton product run was supplied with --aws-run-dir");
  } else if (!state.awsRunEvidenceOk) {
    remaining.push("synced real Graviton product run failed completed-run artifact validation");
  }
  return remaining;
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
