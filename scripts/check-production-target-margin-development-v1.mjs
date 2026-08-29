#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const selectionBytes = fs.readFileSync(config.selection);
const selection = JSON.parse(selectionBytes);
const trainBytes = fs.readFileSync(config.train);
const train = JSON.parse(trainBytes);
const sourceDevBytes = fs.readFileSync(config.sourceDev);
const sourceDev = JSON.parse(sourceDevBytes);
const candidateDevBytes = fs.readFileSync(config.candidateDev);
const candidateDev = JSON.parse(candidateDevBytes);

assert(contract.schema === "nsrl.production_target_margin_contract.v1",
  "target-margin contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}
assert(selection.schema === "nsrl.production_target_margin_preflight_selection.v1"
  && selection.contract.sha256 === sha256(contractBytes)
  && selection.preflight_passed === true,
"target-margin preflight binding is invalid");
assert(train.schema === "nsrl.production_target_margin_train.v1",
  "target-margin training trace schema is invalid");
assert(sourceDev.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateDev.schema === "nsrl.production_model_canonical_eval.v2",
"target-margin development evaluation schema is invalid");

const expected = contract.training;
assert(train.training.context_tokens === expected.context_tokens
  && train.training.windows === expected.windows
  && train.training.evaluation_windows === expected.evaluation_windows
  && train.training.targets_per_window === expected.targets_per_window
  && train.training.epochs === expected.epochs
  && train.training.batch_windows === expected.batch_windows
  && train.training.optimizer_steps === expected.optimizer_steps
  && train.training.total_optimizer_step === expected.optimizer_steps
  && train.training.margin_q8 === expected.margin_q8
  && train.training.feature_shift === selection.selected_feature_shift,
"target-margin full training geometry is invalid");
assert(sourceDev.model_hash === contract.source.model_hash
  && candidateDev.model_hash === train.hashes.final_model
  && sourceDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && candidateDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && sourceDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
"target-margin development bindings are invalid");

const sourceNll = sourceDev.evaluation.total_nll_millibits;
const candidateNll = candidateDev.evaluation.total_nll_millibits;
const initial = train.evaluation.initial;
const final = train.evaluation.final;
const gates = contract.development_gates;
const candidateBytes = fs.readFileSync(config.candidate);
const replayBytes = fs.readFileSync(config.replay);
const optimizerBytes = fs.readFileSync(config.optimizer);
const replayOptimizerBytes = fs.readFileSync(config.replayOptimizer);
const measurements = {
  selected_feature_shift: selection.selected_feature_shift,
  training_initial_mean_target_rank_x1000: initial.mean_target_rank_x1000,
  training_final_mean_target_rank_x1000: final.mean_target_rank_x1000,
  training_rank_improvement_per_mille:
    Math.floor((initial.mean_target_rank_x1000 - final.mean_target_rank_x1000) * 1000
      / Math.max(1, initial.mean_target_rank_x1000)),
  training_initial_mistakes: initial.mistakes,
  training_final_mistakes: final.mistakes,
  training_initial_top5_hits: initial.top5_hits,
  training_final_top5_hits: final.top5_hits,
  training_initial_top10_hits: initial.top10_hits,
  training_final_top10_hits: final.top10_hits,
  training_initial_margin_satisfied: initial.margin_satisfied,
  training_final_margin_satisfied: final.margin_satisfied,
  development_source_total_nll_millibits: sourceNll,
  development_candidate_total_nll_millibits: candidateNll,
  development_nll_delta_millibits: candidateNll - sourceNll,
  development_nll_regression_per_mille:
    Math.max(0, Math.ceil((candidateNll - sourceNll) * 1000 / sourceNll)),
  source_development_residual_saturation_count: sourceDev.health.residual_saturation_count,
  candidate_development_residual_saturation_count:
    candidateDev.health.residual_saturation_count,
};
const results = {
  schedule_complete: train.cursor.schedule_complete === gates.schedule_complete,
  total_optimizer_steps: train.training.total_optimizer_step === gates.total_optimizer_steps,
  output_matrix_movement_minimum:
    train.training.movement_l1 >= gates.output_matrix_movement_minimum,
  frozen_parameters_unchanged: train.gates.frozen_parameters_unchanged === true,
  output_bias_unchanged: train.gates.output_bias_unchanged === true,
  weight_saturation_maximum:
    train.health.weight_saturation_count <= gates.weight_saturation_maximum,
  candidate_model_exact_restart_replay:
    sha256(candidateBytes) === sha256(replayBytes),
  optimizer_state_exact_restart_replay:
    sha256(optimizerBytes) === sha256(replayOptimizerBytes),
  training_mean_target_rank_improvement_per_mille_minimum:
    measurements.training_rank_improvement_per_mille
      >= gates.training_mean_target_rank_improvement_per_mille_minimum,
  training_mistakes_strictly_improve: final.mistakes < initial.mistakes,
  training_top5_hits_strictly_improve: final.top5_hits > initial.top5_hits,
  training_top10_hits_strictly_improve: final.top10_hits > initial.top10_hits,
  training_margin_satisfied_strictly_improves:
    final.margin_satisfied > initial.margin_satisfied,
  development_nll_regression_per_mille_maximum:
    measurements.development_nll_regression_per_mille
      <= gates.development_nll_regression_per_mille_maximum,
  development_residual_saturation_must_not_increase:
    candidateDev.health.residual_saturation_count
      <= sourceDev.health.residual_saturation_count,
};
const passed = Object.values(results).every(Boolean);
const result = {
  schema: "nsrl.production_target_margin_development_gate.v1",
  contract: binding(config.contract, contractBytes),
  preflight: binding(config.selection, selectionBytes),
  training: binding(config.train, trainBytes),
  candidate: artifactBinding(config.candidate, candidateBytes),
  replay_candidate: artifactBinding(config.replay, replayBytes),
  optimizer: artifactBinding(config.optimizer, optimizerBytes),
  replay_optimizer: artifactBinding(config.replayOptimizer, replayOptimizerBytes),
  source_development: binding(config.sourceDev, sourceDevBytes),
  candidate_development: binding(config.candidateDev, candidateDevBytes),
  candidate_model_hash: candidateDev.model_hash,
  measurements,
  gates: results,
  development_gate_passed: passed,
  public_test_authorized: passed,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "target-margin development gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  development_gate_passed: passed,
  public_test_authorized: passed,
  gates: results,
  out: config.out,
})}\n`);
if (!passed) process.exitCode = 1;

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, sha256: sha256(bytes)};
}

function artifactBinding(file, bytes) {
  return {path: file, bytes: bytes.length, sha256: sha256(bytes)};
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(args) {
  const config = {check: false};
  const names = {
    "--contract": "contract", "--selection": "selection", "--train": "train",
    "--candidate": "candidate", "--replay": "replay", "--optimizer": "optimizer",
    "--replay-optimizer": "replayOptimizer", "--source-dev": "sourceDev",
    "--candidate-dev": "candidateDev", "--out": "out",
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--check") config.check = true;
    else {
      const name = names[args[index]];
      if (!name) throw new Error(`unknown argument ${args[index]}`);
      config[name] = args[++index] || "";
    }
  }
  for (const name of Object.values(names)) assert(config[name], `missing ${name}`);
  return config;
}
