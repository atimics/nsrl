#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function usage() {
  console.log(
    [
      "Usage: node scripts/check-nsrl-mme-v0-self-test.mjs [--keep]",
      "",
      "Builds tiny synthetic NSRL-MME v0 fixtures and proves the headline",
      "multimodal eval scorer rejects missing, replay-only, incomplete, and weak",
      "evidence while accepting a green confidence trace.",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { keep: false };
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function writeJson(filePath, data) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function runScorer(paths) {
  return childProcess.spawnSync(
    process.execPath,
    [
      "scripts/check-nsrl-mme-v0.mjs",
      "--quality-report",
      paths.quality,
      "--objective-coverage",
      paths.objective,
      "--out",
      paths.out,
    ],
    {
      cwd: repoRoot,
      encoding: "utf8",
      maxBuffer: 1024 * 1024 * 8,
    },
  );
}

function runCase(root, name, options) {
  const caseRoot = path.join(root, name);
  const paths = {
    quality: path.join(caseRoot, "quality-report.json"),
    objective: path.join(caseRoot, "objective-coverage.json"),
    out: path.join(caseRoot, "nsrl-mme-v0.json"),
  };
  fs.mkdirSync(caseRoot, { recursive: true });
  if (options.quality !== null) {
    writeJson(paths.quality, options.quality);
  }
  if (options.objective !== null) {
    writeJson(paths.objective, options.objective);
  }
  const result = runScorer(paths);
  const report = JSON.parse(fs.readFileSync(paths.out, "utf8"));
  assertEqual(report.schema, "nsrl.multimodal_llm_eval.v0", `${name} schema`);
  assertEqual(result.status, options.expectedStatusCode, `${name} exit status`);
  assertEqual(report.status, options.expectedStatus, `${name} report status`);
  if (options.expectedScore !== undefined) {
    assertEqual(report.headline_score_per_mille, options.expectedScore, `${name} score`);
  }
  for (const expectedError of options.expectedErrors || []) {
    if (!report.errors.some((error) => error.includes(expectedError))) {
      throw new Error(`${name} missing expected error ${JSON.stringify(expectedError)}; errors=${JSON.stringify(report.errors)}`);
    }
  }
  return {
    name,
    status: report.status,
    score: report.headline_score_per_mille,
    exit_status: result.status,
  };
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function syntheticObjective() {
  return {
    schema: "nsrl.solomon_objective_coverage_check.v1",
    ok: true,
    diagnostic_ok: true,
    local_objective_proof: true,
    release_objective_proof: false,
    missing: [],
  };
}

function syntheticQuality(options = {}) {
  const quality = {
    schema: "nsrl.solomon_v2_quality_report.v1",
    ok: true,
    generation_integrity: {
      ok: true,
    },
    multimodal_replay: {
      diagnostic_only: true,
    },
  };
  if (options.replayOnly) {
    return quality;
  }
  quality.confidence_trace = syntheticConfidenceTrace(options);
  return quality;
}

function syntheticConfidenceTrace(options = {}) {
  const generatedSampleCount = options.generatedSampleCount ?? 72;
  const scores = {
    text_prompt_to_image_plan: options.weakFamily === "text_prompt_to_image_plan" ? 600 : 800,
    seal_image_to_text: options.weakFamily === "seal_image_to_text" ? 600 : 800,
    text_and_seal_to_explanation: options.weakFamily === "text_and_seal_to_explanation" ? 600 : 800,
    identity_source_binding: options.weakFamily === "identity_source_binding" ? 600 : 800,
  };
  const groups = Object.fromEntries(
    Object.entries(scores).map(([key, score]) => [
      key,
      syntheticDirectionalGroup(key, score),
    ]),
  );
  if (options.missingFamily) {
    delete groups[options.missingFamily];
  }
  return {
    label: "strong-bidirectional-product-generation",
    directional_native_eval: {
      ok: !options.missingFamily,
      groups,
      errors: [],
    },
    source_grounding: {
      grounded_corpus_present: true,
      grounded_corpus_ok: true,
      grounded_source_provenance: true,
      text_queries_have_source_text: true,
      image_queries_have_source_text: true,
      sample_queries_have_source_text: true,
      sample_source_text_evidence: true,
      generated_text_source_evidence: true,
      generated_text_image_agreement: true,
      expected_generated_text_agreement: true,
    },
    product_generation: {
      present: true,
      heldout_partition_ready: true,
      trace_integrity_ok: true,
      product_floor_ok: true,
      sample_count: generatedSampleCount,
      prompt_provenance: {
        selected_prompt_eligible_rows: generatedSampleCount,
        selected_prompt_eligible_rows_match: true,
        selected_prompt_eligible_unique_targets: generatedSampleCount,
        selected_prompt_eligible_unique_targets_match: true,
        selected_prompt_hash_match: true,
        sample_prompt_sets_match: true,
      },
      output_identity: {
        required: true,
        rows: generatedSampleCount,
        scored_rows: generatedSampleCount,
        identity_rows: generatedSampleCount,
        positive_margin_rows: generatedSampleCount,
        min_margin: 1,
        ok: true,
      },
      best_retrieval_top1_per_mille: 800,
    },
    cross_modal_agreement: {
      match_yes: syntheticAgreementMetric(72),
      match_no: syntheticAgreementMetric(72),
      wrong_image_negatives: syntheticAgreementMetric(72),
      wrong_prompt_negatives: syntheticAgreementMetric(72),
    },
  };
}

function syntheticDirectionalGroup(key, score) {
  return {
    label: key,
    tasks: [],
    targets: 72,
    stats: {
      targets: 72,
      correct: Math.floor((score * 72) / 1000),
      invalid_contexts: 0,
      accuracy_per_mille: score,
      top5_accuracy_per_mille: score,
      top10_accuracy_per_mille: score,
      mean_target_rank_per_mille: 1,
      mean_target_margin_q8: 128,
    },
    task_targets: {},
    phase_targets: {},
    ok: true,
    errors: [],
  };
}

function syntheticAgreementMetric(count) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: 1,
    mean_margin: 4,
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-mme-v0-self-test-"));
  try {
    const cases = [
      runCase(root, "missing-quality", {
        quality: null,
        objective: syntheticObjective(),
        expectedStatusCode: 1,
        expectedStatus: "missing",
        expectedScore: null,
        expectedErrors: ["missing quality-report.json"],
      }),
      runCase(root, "replay-only", {
        quality: syntheticQuality({ replayOnly: true }),
        objective: syntheticObjective(),
        expectedStatusCode: 1,
        expectedStatus: "incomplete",
        expectedScore: null,
        expectedErrors: ["quality report confidence_trace"],
      }),
      runCase(root, "incomplete-directional-family", {
        quality: syntheticQuality({ missingFamily: "seal_image_to_text" }),
        objective: syntheticObjective(),
        expectedStatusCode: 1,
        expectedStatus: "incomplete",
        expectedScore: null,
        expectedErrors: ["missing directional group seal_image_to_text"],
      }),
      runCase(root, "passing-confidence-trace", {
        quality: syntheticQuality(),
        objective: syntheticObjective(),
        expectedStatusCode: 0,
        expectedStatus: "passed",
        expectedScore: 800,
      }),
      runCase(root, "tiny-generated-output", {
        quality: syntheticQuality({ generatedSampleCount: 2 }),
        objective: syntheticObjective(),
        expectedStatusCode: 1,
        expectedStatus: "failed",
        expectedScore: 800,
        expectedErrors: ["generated_output_integrity gate is not green"],
      }),
      runCase(root, "weak-floor-score", {
        quality: syntheticQuality({ weakFamily: "identity_source_binding" }),
        objective: syntheticObjective(),
        expectedStatusCode: 1,
        expectedStatus: "failed",
        expectedScore: 600,
        expectedErrors: ["headline score 600 < target 700"],
      }),
    ];
    console.log(JSON.stringify({
      schema: "nsrl.multimodal_llm_eval_self_test.v0",
      ok: true,
      cases,
    }, null, 2));
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
