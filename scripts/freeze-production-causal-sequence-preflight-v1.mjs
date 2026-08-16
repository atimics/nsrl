#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const readRun = (name) => JSON.parse(fs.readFileSync(path.join(config.runDir, name), "utf8"));
const sourceDev = readRun("source-dev.json");
const sourceTest = readRun("source-test.json");
const midpoint = readRun("train-midpoint.json");
const final = readRun("train-final.json");
const replay = readRun("train-replay.json");
const candidateDev = readRun("candidate-dev.json");
const candidateTestPath = path.join(config.runDir, "candidate-test.json");
const candidateTest = fs.existsSync(candidateTestPath) ? readRun("candidate-test.json") : null;

assert([
  "nsrl.production_causal_sequence_preflight_contract.v1",
  "nsrl.production_causal_tail_context_contract.v1",
  "nsrl.production_causal_tail_stability_contract.v1",
].includes(contract.schema),
  "causal sequence contract schema is invalid");
assert(contract.authorization?.hidden_panel_access === false
  && contract.authorization?.paid_scaling === false,
"causal sequence contract exceeds local public-development authorization");
verifyInput(contract.source.model_path, contract.source.artifact_sha256);
verifyInput(contract.bindings.tokenizer_path, contract.bindings.tokenizer_sha256);
verifyInput(contract.bindings.train_tokens_path, contract.bindings.train_tokens_sha256);
verifyInput(contract.bindings.dev_tokens_path, contract.bindings.dev_tokens_sha256);
verifyInput(contract.bindings.test_tokens_path, contract.bindings.test_tokens_sha256);
for (const artifact of contract.derivation?.artifacts ?? []) {
  verifyInput(artifact.path, artifact.sha256);
}

const expectedTraining = contract.training;
const midpointSteps = expectedTraining.midpoint_optimizer_steps
  ?? Math.floor(expectedTraining.optimizer_steps / 2);
const finalSteps = expectedTraining.optimizer_steps - midpointSteps;
const midpointTargets = midpointSteps * expectedTraining.batch_windows
  * expectedTraining.targets_per_window;
const finalTargets = finalSteps * expectedTraining.batch_windows
  * expectedTraining.targets_per_window;
const targetMeanShift = Math.log2(expectedTraining.targets_per_window);
assert(Number.isInteger(targetMeanShift), "causal target count must be a power of two");
for (const [name, trace] of [["midpoint", midpoint], ["final", final], ["replay", replay]]) {
  assert(trace.schema === "nsrl.production_full_train_smoke.v1"
    && trace.profile === contract.profile
    && trace.parameter_count === contract.parameter_count,
  `${name} training trace identity is invalid`);
  assert(trace.bindings?.tokenizer_hash === contract.bindings.tokenizer_hash
    && trace.bindings?.token_stream_hash === contract.bindings.train_token_stream_hash,
  `${name} training trace bindings are invalid`);
  assert(trace.training?.context_tokens === expectedTraining.context_tokens
    && trace.training?.windows === expectedTraining.windows
    && trace.training?.window_selection === expectedTraining.window_selection
    && trace.training?.target_policy === expectedTraining.target_policy
    && trace.training?.targets_per_window === expectedTraining.targets_per_window
    && trace.training?.target_mean_shift === targetMeanShift
    && trace.training?.mean_reduction === "parameter_update_power_of_two_shift"
    && (expectedTraining.evaluation_windows === undefined
      || trace.training?.evaluation_windows === expectedTraining.evaluation_windows)
    && trace.training?.epochs === expectedTraining.epochs
    && trace.training?.batch_windows === expectedTraining.batch_windows
    && trace.training?.probability_gradient_fractional_bits
      === expectedTraining.probability_gradient_fractional_bits
    && trace.training?.probability_normalization
      === expectedTraining.probability_normalization
    && trace.training?.output_backward_shift === expectedTraining.output_backward_shift
    && sameJson(trace.training?.learning_rate_shifts, expectedTraining.learning_rate_shifts),
  `${name} training schedule does not match the prospective contract`);
  if (expectedTraining.effective_probability_adjusted_shifts) {
    assert(trace.training?.effective_output_learning_rate_shift
        === expectedTraining.effective_probability_adjusted_shifts.output
      && trace.training?.effective_bias_learning_rate_shift
        === expectedTraining.effective_probability_adjusted_shifts.bias,
    `${name} probability-adjusted update shifts do not match the prospective contract`);
  }
  if (expectedTraining.embedding_learning_rate_boost_shift !== undefined) {
    assert((trace.training?.embedding_learning_rate_boost_shift ?? 0)
      === expectedTraining.embedding_learning_rate_boost_shift,
    `${name} embedding learning-rate boost does not match the prospective contract`);
  }
  if (expectedTraining.training_workers !== undefined) {
    assert(trace.training?.training_workers === expectedTraining.training_workers,
      `${name} training worker count does not match the prospective contract`);
  }
  if (expectedTraining.atomic_saturation_policy !== undefined) {
    assert(trace.transaction?.saturation_policy === expectedTraining.atomic_saturation_policy
      && trace.transaction?.rejected_batch === null
      && trace.gates?.saturated_batch_rejection_enabled === true
      && trace.gates?.saturated_batch_rejected_atomically === false,
    `${name} atomic saturation policy does not match the prospective contract`);
  }
}
assert(midpoint.training.optimizer_steps === midpointSteps
  && midpoint.training.total_optimizer_step === midpointSteps
  && midpoint.training.supervised_targets === midpointTargets
  && midpoint.cursor.start_window === 0
  && midpoint.cursor.next_window
    === midpointSteps * expectedTraining.batch_windows
  && midpoint.cursor.schedule_complete === false,
"causal sequence midpoint cursor is invalid");
assert(final.training.optimizer_steps === finalSteps
  && final.training.total_optimizer_step === expectedTraining.optimizer_steps
  && final.training.supervised_targets === finalTargets
  && final.cursor.start_window
    === midpointSteps * expectedTraining.batch_windows
  && final.cursor.next_epoch === 1
  && final.cursor.next_window === 0
  && final.cursor.schedule_complete === true,
"causal sequence final cursor is invalid");
assert(midpoint.hashes.initial_model === contract.source.model_hash
  && final.hashes.initial_model === midpoint.hashes.final_model
  && replay.hashes.initial_model === midpoint.hashes.final_model
  && final.hashes.final_model === replay.hashes.final_model,
"causal sequence model-hash chain is invalid");
assert(fs.readFileSync(path.join(config.runDir, "candidate.nsrlpm"))
  .equals(fs.readFileSync(path.join(config.runDir, "replay.nsrlpm")))
  && fs.readFileSync(path.join(config.runDir, "candidate.nsrlpo"))
    .equals(fs.readFileSync(path.join(config.runDir, "replay.nsrlpo")))
  && fs.readFileSync(path.join(config.runDir, "train-final.json"))
    .equals(fs.readFileSync(path.join(config.runDir, "train-replay.json"))),
"causal sequence midpoint replay is not byte-identical");

verifyEval(sourceDev, contract.bindings.dev_token_stream_hash, contract.source.model_hash,
  contract.evaluation.source_dev_residual_saturation_count ?? 0);
verifyEval(sourceTest, contract.bindings.test_token_stream_hash, contract.source.model_hash,
  contract.evaluation.source_test_residual_saturation_count ?? 0);
verifyEval(candidateDev, contract.bindings.dev_token_stream_hash, final.hashes.final_model, null);
if (candidateTest) {
  verifyEval(candidateTest, contract.bindings.test_token_stream_hash, final.hashes.final_model, null);
}
assert(sourceDev.evaluation.total_nll_millibits
    === contract.evaluation.source_dev_total_nll_millibits
  && sourceTest.evaluation.total_nll_millibits
    === contract.evaluation.source_test_total_nll_millibits,
"causal sequence source evaluation does not match the frozen baseline");

const developmentImproved = candidateDev.evaluation.total_nll_millibits
  < sourceDev.evaluation.total_nll_millibits;
assert(Boolean(candidateTest) === developmentImproved,
  "test candidate must be scored exactly when development improves");
const testImproved = Boolean(candidateTest)
  && candidateTest.evaluation.total_nll_millibits
    < sourceTest.evaluation.total_nll_millibits;
const trunkGroups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v", "o",
  "up", "gate", "down",
];
const movement = Object.fromEntries(Object.keys(expectedTraining.learning_rate_shifts)
  .map((group) => [group,
    (midpoint.movement_l1?.[group] ?? 0) + (final.movement_l1?.[group] ?? 0)]));
const requiredMovementGroups = contract.gates?.required_parameter_groups ?? trunkGroups;
assert(Array.isArray(requiredMovementGroups)
  && requiredMovementGroups.length > 0
  && new Set(requiredMovementGroups).size === requiredMovementGroups.length
  && requiredMovementGroups.every((group) => Object.hasOwn(movement, group)),
"causal sequence required movement groups are invalid");
const requiredParameterGroupsMoved = requiredMovementGroups
  .every((group) => movement[group] > 0);
const frozenParameterGroups = contract.gates?.frozen_parameter_groups ?? [];
assert(Array.isArray(frozenParameterGroups)
  && new Set(frozenParameterGroups).size === frozenParameterGroups.length
  && frozenParameterGroups.every((group) => Object.hasOwn(movement, group))
  && frozenParameterGroups.every((group) => !requiredMovementGroups.includes(group)),
"causal sequence frozen parameter groups are invalid");
const frozenParameterGroupsUnchanged = frozenParameterGroups
  .every((group) => movement[group] === 0);
const exactRestartReplay = true;
const zeroSaturation = [midpoint, final, replay, candidateDev, candidateTest]
  .filter(Boolean)
  .every((trace) => Object.values(trace.health).every((value) => value === 0));
const gates = {
  development_total_nll_strictly_improves: developmentImproved,
  test_total_nll_strictly_improves: testImproved,
  ...(contract.gates?.required_parameter_groups === undefined
    ? {all_eleven_trunk_groups_move: requiredParameterGroupsMoved}
    : {required_parameter_groups_move: requiredParameterGroupsMoved}),
  ...(contract.gates?.frozen_parameter_groups === undefined
    ? {}
    : {frozen_parameter_groups_unchanged: frozenParameterGroupsUnchanged}),
  gradient_residual_and_weight_saturation_max: zeroSaturation,
  exact_restart_replay: exactRestartReplay,
};
const preflightPassed = Object.values(gates).every(Boolean);
const postflightRequired = contract.authorization?.postflight_quality_gate_required === true;

const artifactNames = [
  "source-dev.json", "source-test.json", "train-midpoint.json", "midpoint.nsrlpm",
  "midpoint.nsrlpo", "train-final.json", "candidate.nsrlpm", "candidate.nsrlpo",
  "train-replay.json", "replay.nsrlpm", "replay.nsrlpo", "candidate-dev.json",
  ...(candidateTest ? ["candidate-test.json"] : []),
];
const result = {
  schema: "nsrl.production_causal_sequence_preflight.v1",
  contract: binding(config.contract, contractBytes),
  source: {
    model_hash: contract.source.model_hash,
    development: sourceDev.evaluation,
    development_health: sourceDev.health,
    test: sourceTest.evaluation,
    test_health: sourceTest.health,
  },
  candidate: {
    model_hash: final.hashes.final_model,
    development: candidateDev.evaluation,
    development_health: candidateDev.health,
    test: candidateTest?.evaluation ?? null,
    test_health: candidateTest?.health ?? null,
  },
  training: {
    target_policy: expectedTraining.target_policy,
    targets_per_window: expectedTraining.targets_per_window,
    ...(expectedTraining.training_workers === undefined
      ? {}
      : {training_workers: expectedTraining.training_workers}),
    supervised_targets: midpoint.training.supervised_targets
      + final.training.supervised_targets,
    optimizer_steps: final.training.total_optimizer_step,
    movement_l1: movement,
  },
  deltas: {
    development_total_nll_millibits:
      candidateDev.evaluation.total_nll_millibits
        - sourceDev.evaluation.total_nll_millibits,
    test_total_nll_millibits: candidateTest
      ? candidateTest.evaluation.total_nll_millibits
        - sourceTest.evaluation.total_nll_millibits
      : null,
  },
  gates,
  preflight_passed: preflightPassed,
  open_generation_rerun_authorized: preflightPassed && !postflightRequired,
  test_candidate_scored: Boolean(candidateTest),
  provenance: {
    binary: binding(config.binary, fs.readFileSync(config.binary)),
    training_source: binding("crates/nsrl-train/src/production/training.rs",
      fs.readFileSync("crates/nsrl-train/src/production/training.rs")),
    cli_source: binding("crates/nsrl-train/src/bin/nsrl-production-model.rs",
      fs.readFileSync("crates/nsrl-train/src/bin/nsrl-production-model.rs")),
    artifacts: Object.fromEntries(artifactNames.map((name) => [name,
      binding(path.join(config.runDir, name), fs.readFileSync(path.join(config.runDir, name)))])),
  },
  known_non_claims: contract.known_non_claims,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "causal sequence preflight checkpoint does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: config.check,
  candidate: result.candidate.model_hash,
  deltas: result.deltas,
  gates,
  preflight_passed: preflightPassed,
  out: config.out,
})}\n`);

function verifyEval(trace, tokenStreamHash, modelHash, expectedResidualSaturation) {
  assert(trace.schema === "nsrl.production_model_canonical_eval.v2"
    && trace.objective === "integer_base2_softmax_nll_millibits"
    && trace.profile === contract.profile
    && trace.parameter_count === contract.parameter_count
    && trace.bindings?.tokenizer_hash === contract.bindings.tokenizer_hash
    && trace.bindings?.token_stream_hash === tokenStreamHash
    && trace.evaluation?.context_tokens === contract.evaluation.context_tokens
    && trace.evaluation?.windows === contract.evaluation.windows
    && trace.evaluation?.zero_probability_windows === 0
    && (expectedResidualSaturation === null
      || trace.health?.residual_saturation_count === expectedResidualSaturation)
    && trace.model_hash === modelHash,
  "causal sequence canonical evaluation is invalid");
}

function verifyInput(file, expectedSha256) {
  assert(sha256(fs.readFileSync(file)) === expectedSha256, `${file} SHA-256 mismatch`);
}

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, fnv64: fnv64(bytes), sha256: sha256(bytes)};
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function fnv64(bytes) {
  let value = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    value = ((value ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function parseArgs(args) {
  const config = {
    contract: "benchmarks/production-model-v1/p10m-causal-sequence-preflight-v1-contract.json",
    runDir: "data/experiments/production-model-v1/p10m-causal-sequence-preflight-v1",
    binary: "target/release/nsrl-production-model",
    out: "benchmarks/production-model-v1/p10m-causal-sequence-preflight-v1.json",
    check: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--contract") config.contract = args[++index] || "";
    else if (args[index] === "--run-dir") config.runDir = args[++index] || "";
    else if (args[index] === "--binary") config.binary = args[++index] || "";
    else if (args[index] === "--out") config.out = args[++index] || "";
    else if (args[index] === "--check") config.check = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  assert(config.contract && config.runDir && config.binary && config.out,
    "causal sequence preflight paths must not be empty");
  return config;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
