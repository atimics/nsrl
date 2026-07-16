#!/usr/bin/env node

import fs from "node:fs";

const options = parseArgs(process.argv.slice(2));
const required = ["combined", "transformer-only", "suffix-memory-only", "out"];
for (const name of required) {
  if (!options[name]) throw new Error(`--${name} is required`);
}

const rows = Object.fromEntries(
  ["combined", "transformer-only", "suffix-memory-only"].map((mode) => [
    mode,
    JSON.parse(fs.readFileSync(options[mode], "utf8")),
  ]),
);

const combined = rows.combined;
for (const [mode, row] of Object.entries(rows)) {
  if (row.ablation?.mode !== mode) {
    throw new Error(`${mode} trace has mode ${row.ablation?.mode ?? "missing"}`);
  }
  if (!row.ablation.suffix_memory_present) {
    throw new Error(`${mode} trace does not identify an installed suffix memory`);
  }
  for (const field of ["token_hash", "window_hash", "windows"]) {
    if (row.data?.[field] !== combined.data?.[field]) {
      throw new Error(`${mode} trace differs on data.${field}`);
    }
  }
  if (row.ablation.source_model_hash !== combined.ablation.source_model_hash) {
    throw new Error(`${mode} trace uses a different source model`);
  }
}

const metrics = Object.fromEntries(
  Object.entries(rows).map(([mode, row]) => [
    mode,
    {
      evaluated_model_hash: row.ablation.evaluated_model_hash,
      mistakes: row.evaluation.mistakes,
      accuracy_per_mille: row.evaluation.accuracy_per_mille,
      probability_error_q15: row.evaluation.probability_error_q15,
      mean_probability_error_q15: row.evaluation.mean_probability_error_q15,
      unique_predicted_tokens: row.evaluation.unique_predicted_tokens,
      logits_hash: row.evaluation.logits_hash,
    },
  ]),
);

const report = {
  schema: "nsrl.integer_transformer_component_ablation.v1",
  contract: "integer-transformer-proof-v1",
  promotion_evidence: false,
  source_model_hash: combined.ablation.source_model_hash,
  data: combined.data,
  metrics,
  contrasts: {
    suffix_memory_added_to_transformer: contrast(
      metrics.combined,
      metrics["transformer-only"],
    ),
    transformer_logits_added_to_suffix_memory: contrast(
      metrics.combined,
      metrics["suffix-memory-only"],
    ),
  },
};

fs.writeFileSync(options.out, `${JSON.stringify(report, null, 2)}\n`);

function contrast(combinedMetrics, ablatedMetrics) {
  return {
    mistake_reduction: ablatedMetrics.mistakes - combinedMetrics.mistakes,
    accuracy_gain_per_mille:
      combinedMetrics.accuracy_per_mille - ablatedMetrics.accuracy_per_mille,
    probability_error_reduction_q15:
      ablatedMetrics.probability_error_q15 - combinedMetrics.probability_error_q15,
    distinct_logits: combinedMetrics.logits_hash !== ablatedMetrics.logits_hash,
  };
}

function parseArgs(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`invalid argument near ${key ?? "end"}`);
    }
    parsed[key.slice(2)] = value;
  }
  return parsed;
}
