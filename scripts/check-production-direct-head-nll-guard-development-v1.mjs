#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const trainBytes = fs.readFileSync(config.train);
const train = JSON.parse(trainBytes);
const replayTrainBytes = fs.readFileSync(config.replayTrain);
const replayTrain = JSON.parse(replayTrainBytes);
const sourceDevBytes = fs.readFileSync(config.sourceDev);
const sourceDev = JSON.parse(sourceDevBytes);
const candidateDevBytes = fs.readFileSync(config.candidateDev);
const candidateDev = JSON.parse(candidateDevBytes);
const sourceRolloutBytes = fs.readFileSync(config.sourceRollout);
const sourceRollout = JSON.parse(sourceRolloutBytes);
const candidateRolloutBytes = fs.readFileSync(config.candidateRollout);
const candidateRollout = JSON.parse(candidateRolloutBytes);

assert(contract.schema === "nsrl.production_direct_head_nll_guard_contract.v1",
  "direct-head NLL guard contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}
assert(train.schema === "nsrl.direct_head_train.v1"
  && replayTrain.schema === "nsrl.direct_head_train.v1",
"direct-head training trace schema is invalid");
assert(sourceDev.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateDev.schema === "nsrl.production_model_canonical_eval.v2"
  && sourceRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && candidateRollout.schema === "nsrl.production_rollout_divergence_audit.v1",
"direct-head development evaluation schema is invalid");

const expected = contract.training;
assert(train.objective === contract.implementation.objective
  && train.method === "gradient_ranked_probe_scored_coordinate_descent"
  && train.training.context_tokens === expected.context_tokens
  && train.training.train_windows === expected.train_windows
  && train.training.dev_windows === expected.guard_windows
  && train.training.candidates_per_round === expected.candidates_per_round
  && train.training.max_rounds === expected.max_rounds
  && train.training.require_dev_nll_nonworsening
    === expected.require_guard_nll_nonworsening
  && train.training.probability_gradient_fractional_bits
    === expected.probability_gradient_fractional_bits
  && train.training.probability_normalization === expected.probability_normalization
  && train.training.sample_seed === expected.sample_seed
  && train.window_selection === expected.window_selection,
"direct-head training geometry is invalid");
assert(train.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && train.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
  && train.hashes.initial_model === contract.source.model_hash
  && sourceDev.model_hash === contract.source.model_hash
  && candidateDev.model_hash === train.hashes.final_model
  && sourceRollout.bindings.model_hash === contract.source.model_hash
  && candidateRollout.bindings.model_hash === train.hashes.final_model
  && sourceDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && candidateDev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && sourceDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && sourceRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
"direct-head development binding mismatch");

const candidateBytes = fs.readFileSync(config.candidate);
const replayBytes = fs.readFileSync(config.replay);
const sourceNll = sourceDev.evaluation.total_nll_millibits;
const candidateNll = candidateDev.evaluation.total_nll_millibits;
const sourceRank = sourceRollout.teacher_forced.mean_target_rank;
const candidateRank = candidateRollout.teacher_forced.mean_target_rank;
const roundSaturation = train.rounds.reduce(
  (sum, round) => sum + round.weight_saturation_count, 0);
const guardRegressionQ20 = Math.max(0,
  train.quality.final_dev_nll_q20 - train.quality.initial_dev_nll_q20);
const trainingDocuments = uniqueDocuments(train.window_bindings.train);
const guardDocuments = uniqueDocuments(train.window_bindings.dev);
const rejectedCandidates = train.rounds
  .filter((round) => round.dev_guard_rejected)
  .map((round) => ({
    round: round.round,
    output_weight_coordinate: round.output_weight_coordinate,
    output_bias_coordinate: round.output_bias_coordinate,
    train_nll_improvement_q20: round.best_delta_train_nll_q20,
    guard_nll_improvement_q20: round.best_delta_dev_nll_q20,
    applied_delta: round.applied_delta,
  }));
const measurements = {
  training_documents: trainingDocuments,
  guard_documents: guardDocuments,
  training_guard_document_overlap:
    trainingDocuments.filter((document) => guardDocuments.includes(document)),
  training_initial_nll_q20: train.quality.initial_train_nll_q20,
  training_final_nll_q20: train.quality.final_train_nll_q20,
  training_nll_improvement_q20:
    train.quality.initial_train_nll_q20 - train.quality.final_train_nll_q20,
  guard_initial_nll_q20: train.quality.initial_dev_nll_q20,
  guard_final_nll_q20: train.quality.final_dev_nll_q20,
  guard_nll_regression_q20: guardRegressionQ20,
  guard_initial_mistakes: train.quality.initial_dev_mistakes,
  guard_final_mistakes: train.quality.final_dev_mistakes,
  rounds: train.stats.rounds,
  descent_steps: train.stats.total_descent_steps,
  candidates_evaluated: train.stats.total_candidates_evaluated,
  guard_rejections: train.stats.dev_guard_rejections,
  round_weight_saturation_count: roundSaturation,
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
  rejected_candidates: rejectedCandidates,
};
const expectedGates = contract.development_gates;
const gates = {
  candidate_model_exact_rerun_replay:
    sha256(candidateBytes) === sha256(replayBytes),
  training_trace_exact_rerun_replay:
    sha256(trainBytes) === sha256(replayTrainBytes)
      && JSON.stringify(train) === JSON.stringify(replayTrain),
  source_model_was_a_candidate:
    train.hashes.initial_model === contract.source.model_hash,
  model_movement_required:
    !expectedGates.model_movement_required
      || train.hashes.final_model !== train.hashes.initial_model,
  descent_steps_minimum:
    train.stats.total_descent_steps >= expectedGates.descent_steps_minimum,
  training_nll_strictly_improves:
    train.quality.final_train_nll_q20 < train.quality.initial_train_nll_q20,
  guard_nll_regression_q20_maximum:
    guardRegressionQ20 <= expectedGates.guard_nll_regression_q20_maximum,
  guard_mistakes_must_not_increase:
    !expectedGates.guard_mistakes_must_not_increase
      || train.quality.final_dev_mistakes <= train.quality.initial_dev_mistakes,
  frozen_parameters_unchanged:
    train.gates.frozen_parameters_unchanged
      === expectedGates.frozen_parameters_unchanged,
  every_applied_round_is_guard_safe:
    train.rounds.every((round) => round.applied_delta === 0
      || (!round.dev_guard_rejected && round.best_delta_train_nll_q20 > 0
        && round.best_delta_dev_nll_q20 >= 0)),
  every_rejected_round_is_atomic:
    train.rounds.every((round) => !round.dev_guard_rejected
      || (round.applied_delta === 0 && !round.function_visible)),
  weight_saturation_maximum:
    roundSaturation <= expectedGates.weight_saturation_maximum,
  development_nll_regression_per_mille_maximum:
    measurements.development_nll_regression_per_mille
      <= expectedGates.development_nll_regression_per_mille_maximum,
  development_residual_saturation_must_not_increase:
    candidateDev.health.residual_saturation_count
      <= sourceDev.health.residual_saturation_count,
  development_teacher_forced_mean_target_rank_improvement_per_mille_minimum:
    measurements.development_teacher_forced_rank_improvement_per_mille
      >= expectedGates.development_teacher_forced_mean_target_rank_improvement_per_mille_minimum,
  development_teacher_forced_top1_must_not_decrease:
    !expectedGates.development_teacher_forced_top1_must_not_decrease
      || candidateRollout.teacher_forced.top1_matches
        >= sourceRollout.teacher_forced.top1_matches,
  development_teacher_forced_mean_target_probability_must_not_decrease:
    !expectedGates.development_teacher_forced_mean_target_probability_must_not_decrease
      || candidateRollout.teacher_forced.mean_target_probability_q15
        >= sourceRollout.teacher_forced.mean_target_probability_q15,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_nll_guard_development_gate.v1",
  contract: binding(config.contract, contractBytes),
  training: binding(config.train, trainBytes),
  replay_training: binding(config.replayTrain, replayTrainBytes),
  candidate: binding(config.candidate, candidateBytes),
  replay_candidate: binding(config.replay, replayBytes),
  source_development: binding(config.sourceDev, sourceDevBytes),
  candidate_development: binding(config.candidateDev, candidateDevBytes),
  source_development_rollout: binding(config.sourceRollout, sourceRolloutBytes),
  candidate_development_rollout: binding(config.candidateRollout, candidateRolloutBytes),
  candidate_model_hash: candidateDev.model_hash,
  measurements,
  gates,
  development_gate_passed: passed,
  public_test_authorized: passed,
  public_test_opened: false,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "direct-head development gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  development_gate_passed: passed,
  public_test_authorized: passed,
  gates,
  out: config.out,
})}\n`);
if (!passed) process.exitCode = 1;

function improvementPerMille(source, candidate) {
  return Math.floor((source - candidate) * 1000 / Math.max(1, source));
}

function uniqueDocuments(bindings) {
  return [...new Set(bindings.map((binding) => binding.document))].sort((a, b) => a - b);
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
    "--contract": "contract", "--train": "train",
    "--replay-train": "replayTrain", "--candidate": "candidate",
    "--replay": "replay", "--source-dev": "sourceDev",
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
