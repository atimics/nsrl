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
const sourceRolloutBytes = fs.readFileSync(config.sourceRollout);
const sourceRollout = JSON.parse(sourceRolloutBytes);
const candidateRolloutBytes = fs.readFileSync(config.candidateRollout);
const candidateRollout = JSON.parse(candidateRolloutBytes);

assert(contract.schema === "nsrl.production_target_margin_trust_region_contract.v1",
  "target-margin trust-region contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}
assert(selection.schema
  === "nsrl.production_target_margin_trust_region_preflight_selection.v1"
  && selection.contract.sha256 === sha256(contractBytes)
  && selection.preflight_passed === true,
"target-margin trust-region preflight binding is invalid");
assert(train.schema === "nsrl.production_target_margin_train.v1",
  "target-margin training trace schema is invalid");
assert(sourceDev.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateDev.schema === "nsrl.production_model_canonical_eval.v2"
  && sourceRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && candidateRollout.schema === "nsrl.production_rollout_divergence_audit.v1",
"target-margin development evaluation schema is invalid");

const expected = contract.training;
assert(train.training.context_tokens === expected.context_tokens
  && train.training.windows === expected.windows
  && train.training.window_schedule_windows === expected.window_schedule_windows
  && train.training.evaluation_windows === expected.evaluation_windows
  && train.training.targets_per_window === expected.targets_per_window
  && train.training.epochs === expected.epochs
  && train.training.batch_windows === expected.batch_windows
  && train.training.optimizer_steps === expected.optimizer_steps
  && train.training.total_optimizer_step === expected.optimizer_steps
  && train.training.margin_q8 === expected.margin_q8
  && train.training.feature_shift === selection.selected_feature_shift
  && train.descent_guard.windows === expected.descent_guard_windows,
"target-margin full training geometry is invalid");
assert(sourceDev.model_hash === contract.source.model_hash
  && candidateDev.model_hash === train.hashes.final_model
  && sourceRollout.bindings.model_hash === contract.source.model_hash
  && candidateRollout.bindings.model_hash === train.hashes.final_model
  && sourceDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && candidateDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && sourceDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && sourceRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
"target-margin development bindings are invalid");

const candidateBytes = fs.readFileSync(config.candidate);
const replayBytes = fs.readFileSync(config.replay);
const optimizerBytes = fs.readFileSync(config.optimizer);
const replayOptimizerBytes = fs.readFileSync(config.replayOptimizer);
const guard = train.descent_guard;
const sourceNll = sourceDev.evaluation.total_nll_millibits;
const candidateNll = candidateDev.evaluation.total_nll_millibits;
const sourceRank = sourceRollout.teacher_forced.mean_target_rank;
const candidateRank = candidateRollout.teacher_forced.mean_target_rank;
const measurements = {
  selected_feature_shift: selection.selected_feature_shift,
  window_schedule_rank_hash: train.training.window_schedule_rank_hash,
  descent_guard_window_rank_hash: guard.window_rank_hash,
  descent_guard_batches_evaluated: guard.batches_evaluated,
  descent_guard_batches_accepted: guard.batches_accepted,
  descent_guard_batches_rejected: guard.batches_rejected,
  descent_guard_initial_nll_millibits: guard.initial_nll_millibits,
  descent_guard_final_nll_millibits: guard.final_nll_millibits,
  descent_guard_initial_mean_target_rank_x1000:
    guard.initial_evaluation.mean_target_rank_x1000,
  descent_guard_final_mean_target_rank_x1000:
    guard.final_evaluation.mean_target_rank_x1000,
  descent_guard_rank_improvement_per_mille: improvementPerMille(
    guard.initial_evaluation.mean_target_rank_x1000,
    guard.final_evaluation.mean_target_rank_x1000,
  ),
  descent_guard_initial_top10_hits: guard.initial_evaluation.top10_hits,
  descent_guard_final_top10_hits: guard.final_evaluation.top10_hits,
  development_source_total_nll_millibits: sourceNll,
  development_candidate_total_nll_millibits: candidateNll,
  development_nll_delta_millibits: candidateNll - sourceNll,
  development_nll_regression_per_mille:
    Math.max(0, Math.ceil((candidateNll - sourceNll) * 1000 / sourceNll)),
  source_development_residual_saturation_count: sourceDev.health.residual_saturation_count,
  candidate_development_residual_saturation_count:
    candidateDev.health.residual_saturation_count,
  source_development_teacher_forced_top1_matches:
    sourceRollout.teacher_forced.top1_matches,
  candidate_development_teacher_forced_top1_matches:
    candidateRollout.teacher_forced.top1_matches,
  source_development_teacher_forced_mean_target_rank: sourceRank,
  candidate_development_teacher_forced_mean_target_rank: candidateRank,
  development_teacher_forced_rank_improvement_per_mille:
    improvementPerMille(sourceRank, candidateRank),
  source_development_teacher_forced_mean_target_probability_q15:
    sourceRollout.teacher_forced.mean_target_probability_q15,
  candidate_development_teacher_forced_mean_target_probability_q15:
    candidateRollout.teacher_forced.mean_target_probability_q15,
};
const gates = contract.development_gates;
const results = {
  schedule_complete: train.cursor.schedule_complete === gates.schedule_complete,
  total_optimizer_steps: train.training.total_optimizer_step === gates.total_optimizer_steps,
  output_matrix_movement_minimum:
    train.training.movement_l1 >= gates.output_matrix_movement_minimum,
  accepted_guard_batches_minimum:
    guard.batches_accepted >= gates.accepted_guard_batches_minimum,
  frozen_parameters_unchanged: train.gates.frozen_parameters_unchanged === true,
  output_bias_unchanged: train.gates.output_bias_unchanged === true,
  weight_saturation_maximum:
    train.health.weight_saturation_count <= gates.weight_saturation_maximum,
  candidate_model_exact_restart_replay: sha256(candidateBytes) === sha256(replayBytes),
  optimizer_state_exact_restart_replay:
    sha256(optimizerBytes) === sha256(replayOptimizerBytes),
  window_schedule_matches_preflight:
    train.training.window_schedule_rank_hash
      === selection.selected_window_schedule_rank_hash,
  descent_guard_matches_preflight:
    guard.window_rank_hash === selection.selected_descent_guard_window_rank_hash
      && guard.initial_nll_millibits === selection.selected_guard_initial_nll_millibits
      && JSON.stringify(guard.initial_evaluation)
        === JSON.stringify(selection.selected_guard_initial_evaluation),
  descent_guard_nll_strictly_improves:
    guard.final_nll_millibits < guard.initial_nll_millibits,
  descent_guard_mean_target_rank_improvement_per_mille_minimum:
    measurements.descent_guard_rank_improvement_per_mille
      >= gates.descent_guard_mean_target_rank_improvement_per_mille_minimum,
  descent_guard_top10_hits_must_not_decrease:
    guard.final_evaluation.top10_hits >= guard.initial_evaluation.top10_hits,
  development_nll_regression_per_mille_maximum:
    measurements.development_nll_regression_per_mille
      <= gates.development_nll_regression_per_mille_maximum,
  development_residual_saturation_must_not_increase:
    candidateDev.health.residual_saturation_count
      <= sourceDev.health.residual_saturation_count,
  development_teacher_forced_mean_target_rank_improvement_per_mille_minimum:
    measurements.development_teacher_forced_rank_improvement_per_mille
      >= gates.development_teacher_forced_mean_target_rank_improvement_per_mille_minimum,
  development_teacher_forced_top1_matches_minimum:
    candidateRollout.teacher_forced.top1_matches
      >= gates.development_teacher_forced_top1_matches_minimum,
  development_teacher_forced_mean_target_probability_must_not_decrease:
    candidateRollout.teacher_forced.mean_target_probability_q15
      >= sourceRollout.teacher_forced.mean_target_probability_q15,
};
const passed = Object.values(results).every(Boolean);
const result = {
  schema: "nsrl.production_target_margin_trust_region_development_gate.v1",
  contract: binding(config.contract, contractBytes),
  preflight: binding(config.selection, selectionBytes),
  training: binding(config.train, trainBytes),
  candidate: binding(config.candidate, candidateBytes),
  replay_candidate: binding(config.replay, replayBytes),
  optimizer: binding(config.optimizer, optimizerBytes),
  replay_optimizer: binding(config.replayOptimizer, replayOptimizerBytes),
  source_development: binding(config.sourceDev, sourceDevBytes),
  candidate_development: binding(config.candidateDev, candidateDevBytes),
  source_development_rollout: binding(config.sourceRollout, sourceRolloutBytes),
  candidate_development_rollout: binding(config.candidateRollout, candidateRolloutBytes),
  candidate_model_hash: candidateDev.model_hash,
  measurements,
  gates: results,
  development_gate_passed: passed,
  public_test_authorized: passed,
  public_test_opened: false,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "target-margin trust-region development gate does not byte-replay");
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

function improvementPerMille(source, candidate) {
  return Math.floor((source - candidate) * 1000 / Math.max(1, source));
}

function binding(file, bytes) {
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
    "--candidate-dev": "candidateDev", "--source-rollout": "sourceRollout",
    "--candidate-rollout": "candidateRollout", "--out": "out",
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
