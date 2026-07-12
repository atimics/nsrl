#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const schema = "nsrl.integer_transformer_candidate_health.v1";
const adamSchema = "nsrl.training_mini_transformer_integer_adam_trace.v1";
const evalSchema = "nsrl.mini_transformer_eval.v1";

export function checkCandidateHealth(trace, policy = {}, evaluation = null) {
  const resolvedPolicy = {
    expectedProfile: policy.expectedProfile || "",
    expectedQuantizationProfile: policy.expectedQuantizationProfile || "",
    requireRmsNorm: policy.requireRmsNorm ?? true,
    minTransformerLayers: policy.minTransformerLayers ?? 2,
    requireAttentionDeltas: policy.requireAttentionDeltas ?? true,
    requireLossImprovement: policy.requireLossImprovement ?? true,
    maxRejectedBatches: policy.maxRejectedBatches ?? 0,
    maxMlpSaturations: policy.maxMlpSaturations ?? 0,
    maxAttentionSaturations: policy.maxAttentionSaturations ?? 0,
    maxResidualSaturations: policy.maxResidualSaturations ?? 0,
    minUniquePredictions: policy.minUniquePredictions ?? 8,
    maxPredictionSharePerMille: policy.maxPredictionSharePerMille ?? 900,
  };
  const errors = [];
  const checks = [];
  const examinedWindows = Math.max(1, integer(trace?.data?.examined_windows));
  const check = (key, ok, actual, expected) => {
    checks.push({ key, ok, actual, expected });
    if (!ok) errors.push(`${key}: got ${JSON.stringify(actual)}, expected ${expected}`);
  };

  check("schema", trace?.schema === adamSchema, trace?.schema ?? null, adamSchema);
  if (resolvedPolicy.expectedProfile) {
    check(
      "architecture_profile",
      trace?.model?.architecture_profile === resolvedPolicy.expectedProfile,
      trace?.model?.architecture_profile ?? null,
      resolvedPolicy.expectedProfile,
    );
  }
  if (resolvedPolicy.expectedQuantizationProfile) {
    check(
      "quantization_profile",
      trace?.model?.quantization_profile === resolvedPolicy.expectedQuantizationProfile,
      trace?.model?.quantization_profile ?? null,
      resolvedPolicy.expectedQuantizationProfile,
    );
  }
  check(
    "transformer_layers",
    integer(trace?.model?.transformer_layers) >= resolvedPolicy.minTransformerLayers,
    trace?.model?.transformer_layers ?? null,
    `>= ${resolvedPolicy.minTransformerLayers}`,
  );
  if (resolvedPolicy.requireRmsNorm) {
    check("rms_norm_enabled", trace?.model?.rms_norm_enabled === true, trace?.model?.rms_norm_enabled ?? null, true);
  }
  check("train_scope", trace?.training?.train_scope === "all", trace?.training?.train_scope ?? null, "all");
  check("accepted_windows", integer(trace?.updates?.accepted_windows) > 0, trace?.updates?.accepted_windows ?? null, "> 0");
  check("accepted_batches", integer(trace?.updates?.accepted_batches) > 0, trace?.updates?.accepted_batches ?? null, "> 0");
  check(
    "rejected_batches",
    integer(trace?.updates?.rejected_batches) >= 0
      && integer(trace?.updates?.rejected_batches) <= resolvedPolicy.maxRejectedBatches,
    trace?.updates?.rejected_batches ?? null,
    `<= ${resolvedPolicy.maxRejectedBatches}`,
  );
  if (resolvedPolicy.requireLossImprovement) {
    check(
      "probability_error_improved",
      integer(trace?.loss?.final_probability_error_q15) < integer(trace?.loss?.initial_probability_error_q15),
      {
        initial: trace?.loss?.initial_probability_error_q15 ?? null,
        final: trace?.loss?.final_probability_error_q15 ?? null,
      },
      "final < initial",
    );
  }
  if (resolvedPolicy.requireAttentionDeltas) {
    for (const projection of ["q", "k", "v", "o"]) {
      const key = `attention_${projection}`;
      check(`${key}_delta_l1`, integer(trace?.delta_l1?.[key]) > 0, trace?.delta_l1?.[key] ?? null, "> 0");
    }
  }
  check(
    "mlp_saturation_count",
    integer(trace?.saturation?.mlp) >= 0
      && integer(trace?.saturation?.mlp) <= resolvedPolicy.maxMlpSaturations,
    trace?.saturation?.mlp ?? null,
    `<= ${resolvedPolicy.maxMlpSaturations}`,
  );
  check(
    "attention_saturation_count",
    integer(trace?.saturation?.attention) >= 0
      && integer(trace?.saturation?.attention) <= resolvedPolicy.maxAttentionSaturations,
    trace?.saturation?.attention ?? null,
    `<= ${resolvedPolicy.maxAttentionSaturations}`,
  );
  check(
    "residual_saturation_count",
    integer(trace?.saturation?.residual) >= 0
      && integer(trace?.saturation?.residual) <= resolvedPolicy.maxResidualSaturations,
    trace?.saturation?.residual ?? null,
    `<= ${resolvedPolicy.maxResidualSaturations}`,
  );
  if (evaluation) {
    check("eval_schema", evaluation?.schema === evalSchema, evaluation?.schema ?? null, evalSchema);
    check(
      "eval_invalid_forward_count",
      integer(evaluation?.evaluation?.invalid_forward_count) === 0,
      evaluation?.evaluation?.invalid_forward_count ?? null,
      "0",
    );
    check(
      "unique_predicted_tokens",
      integer(evaluation?.evaluation?.unique_predicted_tokens) >= resolvedPolicy.minUniquePredictions,
      evaluation?.evaluation?.unique_predicted_tokens ?? null,
      `>= ${resolvedPolicy.minUniquePredictions}`,
    );
    check(
      "most_predicted_token_share_per_mille",
      integer(evaluation?.evaluation?.most_predicted_token_share_per_mille) >= 0
        && integer(evaluation?.evaluation?.most_predicted_token_share_per_mille) <= resolvedPolicy.maxPredictionSharePerMille,
      evaluation?.evaluation?.most_predicted_token_share_per_mille ?? null,
      `<= ${resolvedPolicy.maxPredictionSharePerMille}`,
    );
  }

  return {
    schema,
    ok: errors.length === 0,
    policy: resolvedPolicy,
    metrics: {
      examined_windows: trace?.data?.examined_windows ?? null,
      saturation_per_examined_window: {
        mlp: integer(trace?.saturation?.mlp) / examinedWindows,
        attention: integer(trace?.saturation?.attention) / examinedWindows,
        residual: integer(trace?.saturation?.residual) / examinedWindows,
      },
      evaluation: evaluation ? {
        unique_predicted_tokens: evaluation?.evaluation?.unique_predicted_tokens ?? null,
        most_predicted_token: evaluation?.evaluation?.most_predicted_token ?? null,
        most_predicted_token_count: evaluation?.evaluation?.most_predicted_token_count ?? null,
        most_predicted_token_share_per_mille: evaluation?.evaluation?.most_predicted_token_share_per_mille ?? null,
      } : null,
    },
    checks,
    errors,
  };
}

function integer(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : -1;
}

function parseArgs(argv) {
  const config = {
    trace: "",
    evaluation: "",
    out: "",
    policy: {},
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      index += 1;
      if (index >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[index];
    };
    switch (arg) {
      case "--trace": config.trace = next(); break;
      case "--eval": config.evaluation = next(); break;
      case "--out": config.out = next(); break;
      case "--expected-profile": config.policy.expectedProfile = next(); break;
      case "--expected-quantization-profile": config.policy.expectedQuantizationProfile = next(); break;
      case "--min-transformer-layers": config.policy.minTransformerLayers = nonnegative(next(), arg); break;
      case "--max-rejected-batches": config.policy.maxRejectedBatches = nonnegative(next(), arg); break;
      case "--max-mlp-saturations": config.policy.maxMlpSaturations = nonnegative(next(), arg); break;
      case "--max-attention-saturations": config.policy.maxAttentionSaturations = nonnegative(next(), arg); break;
      case "--max-residual-saturations": config.policy.maxResidualSaturations = nonnegative(next(), arg); break;
      case "--min-unique-predictions": config.policy.minUniquePredictions = nonnegative(next(), arg); break;
      case "--max-prediction-share-per-mille": config.policy.maxPredictionSharePerMille = nonnegative(next(), arg); break;
      case "--allow-no-rms-norm": config.policy.requireRmsNorm = false; break;
      case "--allow-dead-attention": config.policy.requireAttentionDeltas = false; break;
      case "--allow-loss-regression": config.policy.requireLossImprovement = false; break;
      case "--help":
        console.log("Usage: check-integer-transformer-candidate-health.mjs --trace PATH [--eval PATH] [--out PATH] [--expected-profile NAME] [--min-transformer-layers N] [--max-rejected-batches N] [--max-mlp-saturations N] [--max-attention-saturations N] [--max-residual-saturations N] [--min-unique-predictions N] [--max-prediction-share-per-mille N] [--allow-no-rms-norm] [--allow-dead-attention] [--allow-loss-regression]");
        process.exit(0);
      default: throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!config.trace) throw new Error("--trace is required");
  return config;
}

function nonnegative(value, label) {
  if (!/^\d+$/.test(value)) throw new Error(`${label} requires a nonnegative integer`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${label} exceeds the safe integer range`);
  return parsed;
}

function main() {
  let config;
  let trace;
  let evaluation = null;
  try {
    config = parseArgs(process.argv.slice(2));
    trace = JSON.parse(fs.readFileSync(path.resolve(config.trace), "utf8"));
    if (config.evaluation) evaluation = JSON.parse(fs.readFileSync(path.resolve(config.evaluation), "utf8"));
  } catch (error) {
    console.error(`candidate health check: ${error.message}`);
    process.exit(2);
  }
  const report = {
    ...checkCandidateHealth(trace, config.policy, evaluation),
    trace: path.resolve(config.trace),
    evaluation: config.evaluation ? path.resolve(config.evaluation) : null,
  };
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (config.out) fs.writeFileSync(path.resolve(config.out), output);
  else process.stdout.write(output);
  process.exit(report.ok ? 0 : 1);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
