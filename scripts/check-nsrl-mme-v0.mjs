#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.multimodal_llm_eval.v0";
const qualityReportSchema = "nsrl.solomon_v2_quality_report.v1";
const objectiveCoverageSchema = "nsrl.solomon_objective_coverage_check.v1";

const defaults = {
  qualityReportPath: "",
  objectiveCoveragePath: "",
  outPath: "",
  targetScorePerMille: 700,
  minimumRowsPerFamily: 72,
  strict: true,
};

const directionalFamilies = [
  {
    key: "text_prompt_to_image_plan",
    label: "text prompt -> symbolic image plan",
  },
  {
    key: "seal_image_to_text",
    label: "seal image -> identity / attributes / source text",
  },
  {
    key: "text_and_seal_to_explanation",
    label: "text + seal -> grounded explanation / match",
  },
  {
    key: "identity_source_binding",
    label: "prompt/name -> identity / source binding",
  },
];

function usage() {
  console.log(
    [
      "Usage: node scripts/check-nsrl-mme-v0.mjs [options]",
      "",
      "Builds the executable NSRL-MME v0 headline multimodal LLM eval artifact.",
      "The check reads Solomon quality/objective proof evidence and writes a",
      "`nsrl-mme-v0.json`-shaped report with status, floor score, components, gates,",
      "and errors.",
      "",
      "Options:",
      "  --quality-report PATH          quality-report.json to score",
      "  --objective-coverage PATH      objective-coverage.json to gate the score",
      "  --out PATH                     write JSON report to PATH",
      "  --target-score-per-mille N     pass floor target (default 700)",
      "  --minimum-rows-per-family N    row floor per component (default 72)",
      "  --no-strict                    exit 0 even when the eval does not pass",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--quality-report") {
      config.qualityReportPath = requireValue(argv, ++index, arg);
    } else if (arg === "--objective-coverage") {
      config.objectiveCoveragePath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--target-score-per-mille") {
      config.targetScorePerMille = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--minimum-rows-per-family") {
      config.minimumRowsPerFamily = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--no-strict") {
      config.strict = false;
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

function parseNonNegativeInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return parsed;
}

function resolvePath(relativeOrAbsolutePath) {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.join(repoRoot, relativeOrAbsolutePath);
}

function relativePath(relativeOrAbsolutePath) {
  if (!relativeOrAbsolutePath) {
    return "";
  }
  return path.relative(repoRoot, resolvePath(relativeOrAbsolutePath));
}

function latestEvidencePath(fileName) {
  const dataRoot = path.join(repoRoot, "data");
  if (!fs.existsSync(dataRoot)) {
    return "";
  }
  const matches = [];
  walk(dataRoot, (filePath) => {
    if (path.basename(filePath) === fileName) {
      const stat = fs.statSync(filePath);
      matches.push({ path: filePath, modifiedAt: stat.mtime.toISOString() });
    }
  });
  matches.sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt));
  return matches[0]?.path || "";
}

function walk(dir, visit) {
  let entries = [];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name === ".git" || entry.name === "target" || entry.name === "node_modules") {
      continue;
    }
    const filePath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(filePath, visit);
    } else if (entry.isFile()) {
      visit(filePath);
    }
  }
}

function readOptionalJson(filePath, label) {
  const input = {
    path: filePath ? relativePath(filePath) : "",
    present: false,
    data: null,
    error: "",
  };
  if (!filePath) {
    return input;
  }
  const resolved = resolvePath(filePath);
  if (!fs.existsSync(resolved)) {
    input.path = relativePath(resolved);
    return input;
  }
  input.present = true;
  input.path = relativePath(resolved);
  try {
    input.data = JSON.parse(fs.readFileSync(resolved, "utf8"));
  } catch (error) {
    input.error = `${label} JSON could not be parsed: ${error instanceof Error ? error.message : String(error)}`;
  }
  return input;
}

function buildReport(config, inputs) {
  const quality = inputs.quality.data;
  const objective = inputs.objective.data;
  const confidence = quality?.confidence_trace || null;
  const missingEvidence = [];
  const inputErrors = [];

  if (!inputs.quality.present) {
    missingEvidence.push("quality-report.json");
  }
  if (!inputs.objective.present) {
    missingEvidence.push("objective-coverage.json");
  }
  if (inputs.quality.error) {
    inputErrors.push(inputs.quality.error);
  }
  if (inputs.objective.error) {
    inputErrors.push(inputs.objective.error);
  }
  if (inputs.quality.present && quality?.schema !== qualityReportSchema) {
    inputErrors.push(`quality report schema ${JSON.stringify(quality?.schema || "")} != ${qualityReportSchema}`);
  }
  if (inputs.objective.present && objective?.schema !== objectiveCoverageSchema) {
    inputErrors.push(`objective coverage schema ${JSON.stringify(objective?.schema || "")} != ${objectiveCoverageSchema}`);
  }
  if (inputs.quality.present && (!confidence || typeof confidence !== "object" || Array.isArray(confidence))) {
    missingEvidence.push("quality report confidence_trace");
  }

  const metricComponents = confidence
    ? [
        ...directionalFamilies.map((family) => directionalComponent(confidence, family, config)),
        hardNegativeComponent(confidence, config),
      ]
    : [];
  const gates = [
    qualityReportGate(inputs.quality, quality),
    objectiveCoverageGate(inputs.objective, objective),
    sourceGroundingGate(confidence),
    generatedOutputGate(confidence, quality, config),
  ];

  const allMetricsMeasured = metricComponents.length > 0
    && metricComponents.every(
      (component) =>
        component.score_per_mille !== null &&
        component.rows >= config.minimumRowsPerFamily,
    );
  const score = allMetricsMeasured
    ? Math.min(...metricComponents.map((component) => component.score_per_mille))
    : null;
  const weakest = score === null
    ? null
    : metricComponents
        .map((component) => ({
          key: component.key,
          label: component.label,
          score_per_mille: component.score_per_mille,
        }))
        .sort((left, right) => left.score_per_mille - right.score_per_mille || left.key.localeCompare(right.key))[0];
  const gatesGreen = gates.every((gate) => gate.ok === true);
  const errors = collectErrors({
    inputErrors,
    missingEvidence,
    metricComponents,
    gates,
    score,
    targetScorePerMille: config.targetScorePerMille,
    allMetricsMeasured,
  });
  const structuralEvidenceComplete = inputs.quality.present
    && inputs.objective.present
    && inputErrors.length === 0
    && missingEvidence.length === 0;

  let status = "missing";
  if (inputs.quality.present) {
    status = structuralEvidenceComplete && allMetricsMeasured ? "failed" : "incomplete";
    if (structuralEvidenceComplete && allMetricsMeasured && gatesGreen && score >= config.targetScorePerMille) {
      status = "passed";
    }
  }

  return {
    schema,
    id: "nsrl-mme-v0",
    label: "NSRL-MME v0 multimodal LLM eval",
    generated_at: new Date().toISOString(),
    status,
    ok: status === "passed",
    headline_score_per_mille: score,
    score_per_mille: score,
    target_score_per_mille: config.targetScorePerMille,
    minimum_rows_per_family: config.minimumRowsPerFamily,
    target_met: status === "passed",
    weakest_component: weakest,
    headline_metric:
      "minimum per-mille score across model-native multimodal task families",
    policy:
      "Sampler, replay, browser-probe, and memory-assisted sample metrics are diagnostics only; they do not define the headline score.",
    inputs: {
      quality_report: inputs.quality.path,
      objective_coverage: inputs.objective.path,
    },
    evidence: {
      quality_report: {
        path: inputs.quality.path,
        present: inputs.quality.present,
        schema: quality?.schema || "",
        ok: quality?.ok === true,
      },
      objective_coverage: {
        path: inputs.objective.path,
        present: inputs.objective.present,
        schema: objective?.schema || "",
        ok: objective?.ok === true,
        local_objective_proof: objective?.local_objective_proof === true,
        release_objective_proof: objective?.release_objective_proof === true,
      },
      confidence_label: confidence?.label || "",
    },
    metric_components: metricComponents,
    gates,
    missing_evidence: [...new Set(missingEvidence)],
    errors,
  };
}

function directionalComponent(confidence, family, config) {
  const group = confidence.directional_native_eval?.groups?.[family.key];
  const stats = group?.stats || {};
  const rows = Number(group?.targets || stats.targets || 0);
  const score = numberOrNull(stats.top5_accuracy_per_mille);
  const invalidContexts = Number(group?.invalid_contexts || stats.invalid_contexts || 0);
  const errors = [];
  if (!group || typeof group !== "object" || Array.isArray(group)) {
    errors.push(`missing directional group ${family.key}`);
  }
  if (rows < config.minimumRowsPerFamily) {
    errors.push(`rows ${rows} < ${config.minimumRowsPerFamily}`);
  }
  if (score === null) {
    errors.push("missing top5_accuracy_per_mille");
  }
  if (invalidContexts !== 0) {
    errors.push(`invalid_contexts ${invalidContexts} != 0`);
  }
  if (group && group.ok !== true) {
    errors.push("directional group ok is not true");
  }
  if (Array.isArray(group?.errors)) {
    errors.push(...group.errors.slice(0, 6));
  }
  return {
    key: family.key,
    label: family.label,
    kind: "model_native_directional_task",
    score_metric: "top5_accuracy_per_mille",
    score_per_mille: score,
    rows,
    ok: errors.length === 0,
    source: `confidence_trace.directional_native_eval.groups.${family.key}.stats`,
    errors: [...new Set(errors)],
  };
}

function hardNegativeComponent(confidence, config) {
  const cross = confidence.cross_modal_agreement || {};
  const metrics = [
    ["match_yes", "match yes"],
    ["match_no", "match no"],
    ["wrong_image_negatives", "wrong-image hard negatives"],
    ["wrong_prompt_negatives", "wrong-prompt hard negatives"],
  ].map(([key, label]) => {
    const metric = cross[key] || {};
    return {
      key,
      label,
      rows: Number(metric.count || 0),
      top1: Number(metric.top1 || 0),
      score_per_mille: confidenceTop1PerMille(metric),
      min_margin: numberOrNull(metric.min_margin),
    };
  });
  const scores = metrics.map((metric) => metric.score_per_mille).filter((value) => value !== null);
  const rows = metrics.length > 0 ? Math.min(...metrics.map((metric) => metric.rows)) : 0;
  const errors = [];
  for (const metric of metrics) {
    if (metric.rows < config.minimumRowsPerFamily) {
      errors.push(`${metric.label} rows ${metric.rows} < ${config.minimumRowsPerFamily}`);
    }
    if (metric.score_per_mille === null) {
      errors.push(`${metric.label} missing top1 score`);
    }
    if (metric.rows > 0 && metric.top1 !== metric.rows) {
      errors.push(`${metric.label} top1 ${metric.top1} != rows ${metric.rows}`);
    }
    if (metric.min_margin !== null && metric.min_margin <= 0) {
      errors.push(`${metric.label} min_margin ${metric.min_margin} <= 0`);
    }
  }
  return {
    key: "hard_negative_match",
    label: "match / no-match hard-negative agreement",
    kind: "model_native_cross_modal_agreement",
    score_metric: "minimum top1_per_mille across match yes/no and hard negatives",
    score_per_mille: scores.length === metrics.length ? Math.min(...scores) : null,
    rows,
    ok: errors.length === 0,
    source: "confidence_trace.cross_modal_agreement",
    submetrics: metrics,
    errors: [...new Set(errors)],
  };
}

function qualityReportGate(input, quality) {
  const failed = [];
  if (!input.present) failed.push("missing");
  if (input.error) failed.push("invalid_json");
  if (input.present && quality?.schema !== qualityReportSchema) failed.push("schema");
  if (quality?.ok !== true) failed.push("ok");
  if (!quality?.confidence_trace) failed.push("confidence_trace");
  return {
    key: "quality_report",
    label: "green quality report with confidence trace",
    kind: "gate",
    ok: failed.length === 0,
    source: input.path || "quality-report.json",
    failed,
  };
}

function objectiveCoverageGate(input, objective) {
  const failed = [];
  if (!input.present) failed.push("missing");
  if (input.error) failed.push("invalid_json");
  if (input.present && objective?.schema !== objectiveCoverageSchema) failed.push("schema");
  if (objective?.ok !== true) failed.push("ok");
  if (objective?.local_objective_proof !== true) failed.push("local_objective_proof");
  return {
    key: "objective_coverage",
    label: "local objective coverage proof",
    kind: "gate",
    ok: failed.length === 0,
    source: input.path || "objective-coverage.json",
    failed,
  };
}

function sourceGroundingGate(confidence) {
  const source = confidence?.source_grounding || {};
  const checks = {
    grounded_corpus_present: source.grounded_corpus_present === true,
    grounded_corpus_ok: source.grounded_corpus_ok === true,
    grounded_source_provenance: source.grounded_source_provenance === true,
    text_queries_have_source_text: source.text_queries_have_source_text === true,
    image_queries_have_source_text: source.image_queries_have_source_text === true,
    sample_queries_have_source_text: source.sample_queries_have_source_text === true,
    sample_source_text_evidence: source.sample_source_text_evidence === true,
    generated_text_source_evidence: source.generated_text_source_evidence === true,
    generated_text_image_agreement: source.generated_text_image_agreement === true,
    expected_generated_text_agreement: source.expected_generated_text_agreement === true,
  };
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([key]) => key);
  return {
    key: "source_grounding",
    label: "source-grounded text/image evidence",
    kind: "gate",
    ok: failed.length === 0,
    source: "confidence_trace.source_grounding",
    failed,
  };
}

function generatedOutputGate(confidence, quality, config) {
  const generation = confidence?.product_generation || {};
  const integrity = quality?.generation_integrity || {};
  const promptProvenance = generation.prompt_provenance || {};
  const outputIdentity = generation.output_identity || {};
  const sampleCount = Number(generation.sample_count || 0);
  const selectedEligibleRows = Number(promptProvenance.selected_prompt_eligible_rows || 0);
  const selectedEligibleUniqueTargets = Number(promptProvenance.selected_prompt_eligible_unique_targets || 0);
  const checks = {
    product_generation_present: generation.present === true,
    heldout_partition_ready: generation.heldout_partition_ready === true,
    generated_sample_count_ready: sampleCount >= config.minimumRowsPerFamily,
    selected_prompt_rows_ready: selectedEligibleRows >= config.minimumRowsPerFamily,
    selected_unique_targets_ready: selectedEligibleUniqueTargets >= config.minimumRowsPerFamily,
    selected_prompt_provenance_ok:
      promptProvenance.selected_prompt_eligible_rows_match === true &&
      promptProvenance.selected_prompt_eligible_unique_targets_match === true &&
      promptProvenance.selected_prompt_hash_match === true &&
      promptProvenance.sample_prompt_sets_match === true,
    trace_integrity_ok: generation.trace_integrity_ok === true && integrity.ok !== false,
    product_floor_ok: generation.product_floor_ok === true,
    generated_output_identity_ok: outputIdentity.required === true ? outputIdentity.ok === true : true,
  };
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([key]) => key);
  return {
    key: "generated_output_integrity",
    label: "held-out generated output integrity",
    kind: "gate",
    ok: failed.length === 0,
    source: "confidence_trace.product_generation + generation_integrity",
    sample_count: sampleCount,
    minimum_rows_per_family: config.minimumRowsPerFamily,
    selected_prompt_eligible_rows: selectedEligibleRows,
    selected_prompt_eligible_unique_targets: selectedEligibleUniqueTargets,
    best_retrieval_top1_per_mille: numberOrNull(generation.best_retrieval_top1_per_mille),
    failed,
  };
}

function collectErrors(input) {
  const errors = [...input.inputErrors];
  for (const missing of input.missingEvidence) {
    errors.push(`missing ${missing}`);
  }
  if (!input.allMetricsMeasured) {
    errors.push("headline metric components are incomplete");
  }
  for (const component of input.metricComponents) {
    if (!component.ok) {
      const detail = component.errors.length > 0 ? `: ${component.errors.join("; ")}` : "";
      errors.push(`${component.key} is not green${detail}`);
    }
  }
  for (const gate of input.gates) {
    if (!gate.ok) {
      errors.push(`${gate.key} gate is not green: ${gate.failed.join(", ")}`);
    }
  }
  if (input.score !== null && input.score < input.targetScorePerMille) {
    errors.push(`headline score ${input.score} < target ${input.targetScorePerMille}`);
  }
  return [...new Set(errors)];
}

function confidenceTop1PerMille(metric) {
  const count = Number(metric?.count || 0);
  const top1 = Number(metric?.top1 || 0);
  if (count <= 0) {
    return null;
  }
  return Math.floor((top1 * 1000) / count);
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function writeOutput(outPath, report) {
  if (!outPath) {
    return;
  }
  const resolved = resolvePath(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const qualityReportPath = config.qualityReportPath || latestEvidencePath("quality-report.json");
  const objectiveCoveragePath = config.objectiveCoveragePath || latestEvidencePath("objective-coverage.json");
  const inputs = {
    quality: readOptionalJson(qualityReportPath, "quality report"),
    objective: readOptionalJson(objectiveCoveragePath, "objective coverage"),
  };
  const report = buildReport(config, inputs);
  writeOutput(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (config.strict && report.status !== "passed") {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
