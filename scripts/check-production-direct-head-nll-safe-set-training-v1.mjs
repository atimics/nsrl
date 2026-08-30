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
const candidateBytes = fs.readFileSync(config.candidate);
const replayBytes = fs.readFileSync(config.replay);

assert(contract.schema === "nsrl.production_direct_head_nll_safe_set_contract.v1",
  "direct-head NLL safe-set contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}
assert(train.schema === "nsrl.direct_head_train.v1"
  && replayTrain.schema === "nsrl.direct_head_train.v1",
"direct-head training trace schema is invalid");

const expected = contract.training;
assert(train.objective === contract.implementation.objective
  && train.method === contract.implementation.method
  && train.training.context_tokens === expected.context_tokens
  && train.training.train_windows === expected.train_windows
  && train.training.dev_windows === expected.guard_windows
  && train.training.candidates_per_round === expected.candidates_per_round
  && train.training.max_rounds === expected.max_rounds
  && train.training.require_dev_nll_nonworsening
    === expected.require_guard_nll_nonworsening
  && train.training.exact_safe_set_selection === expected.exact_safe_set_selection
  && train.training.probability_gradient_fractional_bits
    === expected.probability_gradient_fractional_bits
  && train.training.probability_normalization === expected.probability_normalization
  && train.training.sample_seed === expected.sample_seed
  && train.window_selection === expected.window_selection,
"direct-head safe-set training geometry is invalid");
assert(train.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
  && train.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
  && train.hashes.initial_model === contract.source.model_hash,
"direct-head safe-set training binding mismatch");

const trainingDocuments = uniqueDocuments(train.window_bindings.train);
const guardDocuments = uniqueDocuments(train.window_bindings.dev);
const overlap = trainingDocuments.filter((document) => guardDocuments.includes(document));
assert(overlap.length === 0, "training and guard documents overlap");

let priorTrainNll = train.quality.initial_train_nll_q20;
let priorGuardNll = train.quality.initial_dev_nll_q20;
let exactTraceConsistent = true;
let onlySafeSelected = true;
let bestSafeSelected = true;
let emptySafeSetSelectedSource = true;
const selectedMoves = [];
const guardRejectedDirections = [];
for (const [roundIndex, round] of train.rounds.entries()) {
  const exact = round.exact_candidates;
  assert(Array.isArray(exact), `round ${roundIndex} exact candidate trace is missing`);
  const expectedDirections = round.candidates_evaluated
    * contract.training_gates.directions_per_evaluable_coordinate;
  exactTraceConsistent &&= round.exact_directions_evaluated === exact.length
    && exact.length === expectedDirections;

  for (let index = 0; index < exact.length; index += 2) {
    const negative = exact[index];
    const positive = exact[index + 1];
    exactTraceConsistent &&= positive !== undefined
      && negative.global_coordinate === positive.global_coordinate
      && negative.parameter_group === positive.parameter_group
      && negative.local_coordinate === positive.local_coordinate
      && negative.proposed_delta === -1
      && positive.proposed_delta === 1;
  }
  for (const candidate of exact) {
    const trainDescent = candidate.train_nll_improvement_q20
      > expected.minimum_train_nll_delta_q20;
    const guardSafe = candidate.dev_nll_improvement_q20 >= 0;
    const safe = trainDescent && guardSafe;
    exactTraceConsistent &&= candidate.train_descent === trainDescent
      && candidate.dev_guard_safe === guardSafe
      && candidate.safe === safe;
    onlySafeSelected &&= !candidate.selected || safe;
    if (trainDescent && !guardSafe) {
      guardRejectedDirections.push({
        round: round.round,
        global_coordinate: candidate.global_coordinate,
        parameter_group: candidate.parameter_group,
        local_coordinate: candidate.local_coordinate,
        proposed_delta: candidate.proposed_delta,
        train_nll_improvement_q20: candidate.train_nll_improvement_q20,
        guard_nll_improvement_q20: candidate.dev_nll_improvement_q20,
      });
    }
  }

  const safe = exact.filter((candidate) => candidate.safe).sort(compareSafeCandidates);
  const selected = exact.filter((candidate) => candidate.selected);
  const coordinatesWithDescent = new Set(exact
    .filter((candidate) => candidate.train_descent)
    .map((candidate) => candidate.global_coordinate));
  exactTraceConsistent &&= round.exact_safe_candidates === safe.length
    && round.candidates_with_descent === coordinatesWithDescent.size
    && round.exact_guard_rejections
      === exact.filter((candidate) => candidate.train_descent && !candidate.dev_guard_safe).length;
  if (safe.length > 0) {
    bestSafeSelected &&= selected.length === 1
      && sameCandidate(selected[0], safe[0])
      && round.applied_delta === selected[0].proposed_delta
      && round.best_delta_train_nll_q20 === selected[0].train_nll_improvement_q20
      && round.best_delta_dev_nll_q20 === selected[0].dev_nll_improvement_q20
      && (selected[0].parameter_group === "output_bias"
        ? round.output_bias_coordinate === selected[0].local_coordinate
          && round.output_weight_coordinate === null
        : round.output_weight_coordinate === selected[0].local_coordinate
          && round.output_bias_coordinate === null)
      && round.source_selected === false;
    if (selected.length === 1) {
      selectedMoves.push({
        round: round.round,
        global_coordinate: selected[0].global_coordinate,
        parameter_group: selected[0].parameter_group,
        local_coordinate: selected[0].local_coordinate,
        applied_delta: selected[0].proposed_delta,
        train_nll_improvement_q20: selected[0].train_nll_improvement_q20,
        guard_nll_improvement_q20: selected[0].dev_nll_improvement_q20,
      });
    }
  } else {
    emptySafeSetSelectedSource &&= selected.length === 0
      && round.applied_delta === 0
      && round.source_selected === true
      && roundIndex === train.rounds.length - 1;
  }

  const selectedTrainImprovement = selected.length === 1
    ? selected[0].train_nll_improvement_q20 : 0;
  const selectedGuardImprovement = selected.length === 1
    ? selected[0].dev_nll_improvement_q20 : 0;
  exactTraceConsistent &&= round.train_nll_q20_after
      === priorTrainNll - selectedTrainImprovement
    && round.dev_nll_q20_after === priorGuardNll - selectedGuardImprovement;
  priorTrainNll = round.train_nll_q20_after;
  priorGuardNll = round.dev_nll_q20_after;
}
exactTraceConsistent &&= priorTrainNll === train.quality.final_train_nll_q20
  && priorGuardNll === train.quality.final_dev_nll_q20
  && train.stats.total_exact_directions_evaluated
    === train.rounds.reduce((sum, round) => sum + round.exact_directions_evaluated, 0)
  && train.stats.total_exact_safe_candidates
    === train.rounds.reduce((sum, round) => sum + round.exact_safe_candidates, 0)
  && train.stats.total_exact_guard_rejections
    === train.rounds.reduce((sum, round) => sum + round.exact_guard_rejections, 0)
  && train.stats.total_candidates_evaluated
    === train.rounds.reduce((sum, round) => sum + round.candidates_evaluated, 0)
  && train.stats.total_descent_steps === selectedMoves.length;

const roundSaturation = train.rounds.reduce(
  (sum, round) => sum + round.weight_saturation_count, 0);
const guardRegressionQ20 = Math.max(0,
  train.quality.final_dev_nll_q20 - train.quality.initial_dev_nll_q20);
const measurements = {
  training_documents: trainingDocuments,
  guard_documents: guardDocuments,
  training_guard_document_overlap: overlap,
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
  ranked_coordinates_evaluated: train.stats.total_candidates_evaluated,
  exact_directions_evaluated: train.stats.total_exact_directions_evaluated,
  exact_safe_candidates: train.stats.total_exact_safe_candidates,
  exact_guard_rejections: train.stats.total_exact_guard_rejections,
  round_weight_saturation_count: roundSaturation,
  selected_moves: selectedMoves,
  guard_rejected_directions: guardRejectedDirections,
};
const expectedGates = contract.training_gates;
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
    !expectedGates.training_nll_strictly_improves
      || train.quality.final_train_nll_q20 < train.quality.initial_train_nll_q20,
  guard_nll_regression_q20_maximum:
    guardRegressionQ20 <= expectedGates.guard_nll_regression_q20_maximum,
  guard_mistakes_must_not_increase:
    !expectedGates.guard_mistakes_must_not_increase
      || train.quality.final_dev_mistakes <= train.quality.initial_dev_mistakes,
  frozen_parameters_unchanged:
    train.gates.frozen_parameters_unchanged
      === expectedGates.frozen_parameters_unchanged,
  weight_saturation_maximum:
    roundSaturation <= expectedGates.weight_saturation_maximum,
  every_direction_trace_must_be_consistent:
    !expectedGates.every_direction_trace_must_be_consistent || exactTraceConsistent,
  only_safe_candidates_may_be_selected:
    !expectedGates.only_safe_candidates_may_be_selected || onlySafeSelected,
  selected_candidate_must_be_best_safe_candidate:
    !expectedGates.selected_candidate_must_be_best_safe_candidate || bestSafeSelected,
  empty_safe_set_must_select_source:
    !expectedGates.empty_safe_set_must_select_source || emptySafeSetSelectedSource,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_nll_safe_set_training_gate.v1",
  contract: binding(config.contract, contractBytes),
  training: binding(config.train, trainBytes),
  replay_training: binding(config.replayTrain, replayTrainBytes),
  candidate: binding(config.candidate, candidateBytes),
  replay_candidate: binding(config.replay, replayBytes),
  candidate_model_hash: train.hashes.final_model,
  measurements,
  gates,
  training_gate_passed: passed,
  public_development_authorized: passed,
  public_development_opened: false,
  public_test_opened: false,
  open_generation_opened: false,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "direct-head safe-set training gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  training_gate_passed: passed,
  public_development_authorized: passed,
  gates,
  out: config.out,
})}\n`);
if (!passed) process.exitCode = 1;

function compareSafeCandidates(left, right) {
  return right.train_nll_improvement_q20 - left.train_nll_improvement_q20
    || left.global_coordinate - right.global_coordinate
    || left.proposed_delta - right.proposed_delta;
}

function sameCandidate(left, right) {
  return left.global_coordinate === right.global_coordinate
    && left.proposed_delta === right.proposed_delta;
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
    "--replay": "replay", "--out": "out",
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
