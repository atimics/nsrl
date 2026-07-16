#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const candidateSchema = "nsrl.mini_transformer_eval.v1";
const trainSchema = "nsrl.training_mini_transformer_integer_adam_trace.v1";
const baselinePath = "benchmarks/integer-transformer-proof-v1/baselines.tsv";

function parseArgs(argv) {
  const config = { dir: "", out: "" };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--dir") config.dir = argv[++index] ?? "";
    else if (argv[index] === "--out") config.out = argv[++index] ?? "";
    else throw new Error(`unknown argument: ${argv[index]}`);
  }
  if (!config.dir || !config.out) throw new Error("--dir PATH and --out PATH are required");
  return config;
}

function loadBaselines() {
  const lines = fs.readFileSync(baselinePath, "utf8").trim().split("\n").slice(1);
  return lines.map((line) => {
    const fields = line.split("\t");
    return { system: fields[5], mistakes: Number(fields[7]), probability_error_q15: Number(fields[8]) };
  });
}

function loadRows(directory) {
  return fs.readdirSync(directory)
    .filter((name) => name.endsWith(".eval.json"))
    .filter((name) => !name.startsWith("base-"))
    .map((name) => {
      const variant = name.slice(0, -".eval.json".length);
      const evaluation = JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
      const trace = JSON.parse(fs.readFileSync(path.join(directory, `${variant}.train.jsonl`), "utf8"));
      if (evaluation.schema !== candidateSchema || trace.schema !== trainSchema) {
        throw new Error(`${variant} has an unexpected schema`);
      }
      if (evaluation.ablation?.suffix_memory_present !== (evaluation.ablation?.mode === "transformer-only")) {
        throw new Error(`${variant} does not have explicit suffix-memory provenance`);
      }
      return {
        variant,
        artifact: path.join(directory, `${variant}.nsrlmt`),
        artifact_model_hash: evaluation.ablation.evaluated_model_hash,
        attention: trace.training.attention_kind,
        position: trace.training.position,
        epochs: trace.training.epochs,
        windows: trace.data.windows,
        examined_windows: trace.data.examined_windows,
        batch_windows: trace.training.batch_windows,
        adam_step_shift: trace.optimizer.step_shift,
        margin_weight_q15: trace.training.argmax_margin_weight_q15,
        target_frequency_cap: trace.training.target_frequency_cap,
        accepted_batches: trace.updates.accepted_batches,
        rejected_batches: trace.updates.rejected_batches,
        mistakes: evaluation.evaluation.mistakes,
        probability_error_q15: evaluation.evaluation.probability_error_q15,
        unique_predictions: evaluation.evaluation.unique_predicted_tokens,
        max_prediction_share_per_mille: evaluation.evaluation.most_predicted_token_share_per_mille,
        logits_hash: evaluation.evaluation.logits_hash,
      };
    });
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const baselines = loadBaselines();
  const maxAllowedMistakes = Math.min(...baselines.map((row) => row.mistakes));
  const maxAllowedProbabilityError = Math.min(...baselines.map((row) => row.probability_error_q15));
  const candidates = loadRows(config.dir)
    .map((row) => ({
      ...row,
      passed: row.mistakes <= maxAllowedMistakes
        && row.probability_error_q15 < maxAllowedProbabilityError,
    }))
    .sort((left, right) => left.mistakes - right.mistakes
      || left.probability_error_q15 - right.probability_error_q15
      || left.variant.localeCompare(right.variant));
  if (candidates.length !== 16) throw new Error(`expected 16 candidates, found ${candidates.length}`);
  const best = candidates[0];
  const output = {
    schema: "nsrl.integer_transformer_successor_sweep.v1",
    contract: "integer-transformer-proof-v1",
    artifact_requirement: "serialized_model_has_no_installed_suffix_memory",
    gate: {
      probability_error_q15_strictly_below: maxAllowedProbabilityError,
      mistakes_at_most: maxAllowedMistakes,
    },
    candidates,
    conclusion: {
      passed_candidates: candidates.filter((row) => row.passed).length,
      best_variant: best.variant,
      best_mistakes: best.mistakes,
      best_probability_error_q15: best.probability_error_q15,
      mistake_gap_to_gate: best.mistakes - maxAllowedMistakes,
      probability_error_margin_to_gate: maxAllowedProbabilityError - best.probability_error_q15,
      status: best.passed ? "pass" : "blocked_on_top1_generalization",
    },
  };
  fs.mkdirSync(path.dirname(path.resolve(config.out)), { recursive: true });
  fs.writeFileSync(config.out, `${JSON.stringify(output, null, 2)}\n`);
  console.log(JSON.stringify(output.conclusion));
}

main();
