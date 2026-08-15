#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let contractPath = "";
let runDir = "";
let outPath = "";
let triggerOnly = false;
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--trigger-only") triggerOnly = true;
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !runDir || !outPath) {
  throw new Error("--contract, --run-dir, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const allZero = (values) => Object.values(values).every((value) => value === 0);
const deltaValue = (group) => group.total ?? group;

const contractBytes = await readFile(contractPath);
const contract = JSON.parse(contractBytes);
assert(
  contract.schema === "nsrl.production_representation_health_trigger_contract.v1",
  "unexpected health trigger contract schema",
);
for (const artifact of contract.bindings.artifacts) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const [prefix, trigger, prefixModel, triggerModel, prefixOptimizer, triggerOptimizer] =
  await Promise.all([
    json(path.join(runDir, "prefix", "train.json")),
    json(path.join(runDir, "trigger", "train.json")),
    readFile(path.join(runDir, "prefix", "model.nsrlpm")),
    readFile(path.join(runDir, "trigger", "model.nsrlpm")),
    readFile(path.join(runDir, "prefix", "optimizer.nsrlpo")),
    readFile(path.join(runDir, "trigger", "optimizer.nsrlpo")),
  ]);

const expectedTraining = contract.training;
const trainingShapeMatches = [prefix, trigger].every((trace) =>
  trace.schema === "nsrl.production_full_train_smoke.v1"
    && trace.profile === contract.profile
    && trace.parameter_count === contract.parameter_count
    && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
    && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
    && trace.training.context_tokens === expectedTraining.context_tokens
    && trace.training.windows === expectedTraining.windows
    && trace.training.targets_per_window === expectedTraining.targets_per_window
    && trace.training.training_workers === expectedTraining.training_workers
    && trace.training.evaluation_windows === expectedTraining.evaluation_windows
    && trace.training.epochs === expectedTraining.epochs
    && trace.training.batch_windows === expectedTraining.batch_windows
    && trace.training.embedding_residual_flush === expectedTraining.embedding_residual_flush
    && trace.training.descent_guard_windows === expectedTraining.descent_guard_windows
    && trace.training.descent_guard_candidate_family === expectedTraining.candidate_family
    && trace.training.signed_block_feasibility === expectedTraining.feasibility
    && trace.training.output_backward_shift === expectedTraining.output_backward_shift
    && trace.training.probability_gradient_fractional_bits
      === expectedTraining.probability_gradient_fractional_bits
    && trace.training.probability_normalization === expectedTraining.probability_normalization
    && same(trace.training.learning_rate_shifts, expectedTraining.learning_rate_shifts)
    && trace.gates.signed_block_trust_region_enabled === true
    && trace.gates.signed_block_zero_guard_residual_saturation_enabled === true
    && trace.gates.signed_block_zero_saturation_feasibility_enforced === true
    && trace.gates.training_only_descent_guard_enabled === true
    && trace.gates.descent_guard_update_windows_disjoint === true
    && trace.gates.saturated_batch_rejection_enabled === true
);

const prefixMatches = prefix.training.optimizer_steps === contract.trigger.prefix_optimizer_steps
  && prefix.training.total_optimizer_step === contract.trigger.prefix_optimizer_steps
  && prefix.cursor.start_epoch === 0
  && prefix.cursor.start_window === 0
  && prefix.cursor.next_epoch === 0
  && prefix.cursor.next_window === contract.trigger.trigger_start_window
  && prefix.cursor.schedule_complete === false
  && prefix.hashes.initial_model === contract.source.model_hash
  && prefix.hashes.final_model === contract.source.model_hash
  && prefix.signed_block_trust_region.evaluated_batches === 0
  && prefix.signed_block_trust_region.selected_batches === 0
  && prefix.signed_block_trust_region.last_selection === null
  && prefix.descent_guard.initial_nll_millibits === contract.source.guard_nll_millibits
  && prefix.descent_guard.final_nll_millibits === contract.source.guard_nll_millibits
  && prefix.descent_guard.window_rank_hash === contract.training.descent_guard_window_rank_hash
  && prefix.descent_guard.update_window_overlap_count === 0
  && allZero(prefix.movement_l1)
  && allZero(prefix.health)
  && sha256(prefixModel) === contract.source.model_sha256;

const selection = trigger.signed_block_trust_region.last_selection;
const triggerMatches = trigger.training.optimizer_steps === 1
  && trigger.training.total_optimizer_step === contract.trigger.total_optimizer_step
  && trigger.cursor.start_epoch === 0
  && trigger.cursor.start_window === contract.trigger.trigger_start_window
  && trigger.cursor.next_epoch === 0
  && trigger.cursor.next_window === contract.trigger.trigger_next_window
  && trigger.cursor.schedule_complete === false
  && trigger.hashes.initial_model === contract.source.model_hash
  && trigger.hashes.final_model === contract.expected.model_hash
  && trigger.signed_block_trust_region.evaluated_batches === 1
  && trigger.signed_block_trust_region.selected_batches === 1
  && selection !== null
  && selection.attempted_total_optimizer_step === contract.trigger.total_optimizer_step
  && selection.start_window === contract.trigger.trigger_start_window
  && selection.windows === contract.training.batch_windows
  && selection.candidates_evaluated === contract.expected.candidates_evaluated
  && selection.zero_saturation_candidates === contract.expected.zero_saturation_candidates
  && selection.before_nll_millibits === contract.source.guard_nll_millibits
  && selection.forward_nll_millibits === contract.expected.forward_nll_millibits
  && selection.forward_residual_saturation_count
    === contract.expected.forward_residual_saturation_count
  && selection.selected_nll_millibits === contract.expected.selected_nll_millibits
  && selection.selected_residual_saturation_count === 0
  && same(selection.selected_steps, contract.expected.selected_steps)
  && same(selection.selected_movement_l1, contract.expected.selected_movement_l1)
  && trigger.descent_guard.initial_nll_millibits === contract.source.guard_nll_millibits
  && trigger.descent_guard.final_nll_millibits === contract.expected.selected_nll_millibits
  && trigger.descent_guard.window_rank_hash === contract.training.descent_guard_window_rank_hash
  && trigger.descent_guard.update_window_overlap_count === 0
  && trigger.descent_guard.accepted_batches === 1
  && trigger.descent_guard.rejected_batches === 0
  && allZero(trigger.health);

const chainMatches = trigger.hashes.initial_model === prefix.hashes.final_model
  && sha256(prefixModel) === contract.source.model_sha256
  && sha256(prefixOptimizer) !== sha256(triggerOptimizer)
  && sha256(prefixModel) !== sha256(triggerModel);
const triggerPassed = trainingShapeMatches && prefixMatches && triggerMatches && chainMatches;
assert(triggerPassed, "prospective health-constrained step-408 trigger failed");

if (triggerOnly) {
  process.stdout.write(`${JSON.stringify({
    schema: "nsrl.production_representation_health_trigger_preflight.v1",
    outcome: "health_constrained_step_408_trigger_passed",
    model_hash: trigger.hashes.final_model,
    out: outPath,
  })}\n`);
  process.exit(0);
}

const [development, saturation, delta] = await Promise.all([
  json(path.join(runDir, "development.json")),
  json(path.join(runDir, "saturation.json")),
  json(path.join(runDir, "delta.json")),
]);
const movement = Object.fromEntries(
  Object.entries(delta.groups).map(([group, value]) => [group, deltaValue(value).l1]),
);
const onlyExpectedGroupsMoved = Object.entries(movement).every(([group, value]) =>
  contract.gates.allowed_moving_parameter_groups.includes(group) ? value > 0 : value === 0);
const evidenceBindingsMatch = development.model_hash === trigger.hashes.final_model
  && saturation.bindings.model_hash === trigger.hashes.final_model
  && delta.bindings.candidate_model_hash === trigger.hashes.final_model
  && delta.bindings.source_model_hash === contract.source.model_hash
  && development.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash;
const developmentDelta = development.evaluation.total_nll_millibits
  - contract.source.development_nll_millibits;
const postflightPassed = development.evaluation.total_nll_millibits
    === contract.expected.development_nll_millibits
  && developmentDelta < 0
  && development.health.residual_saturation_count === 0
  && saturation.aggregate.residual_saturation_count
    <= contract.gates.manifest_residual_saturation_max
  && onlyExpectedGroupsMoved
  && evidenceBindingsMatch;
const allGatesPassed = triggerPassed && postflightPassed;
const result = {
  schema: "nsrl.production_representation_health_trigger.v1",
  checked: check,
  objective: contract.objective,
  outcome: allGatesPassed
    ? "health_constrained_step_408_trigger_confirmed"
    : "health_constrained_step_408_postflight_failed",
  contract: { path: contractPath, sha256: sha256(contractBytes) },
  source: {
    model_hash: contract.source.model_hash,
    model_sha256: contract.source.model_sha256,
    guard_nll_millibits: contract.source.guard_nll_millibits,
    development_nll_millibits: contract.source.development_nll_millibits,
  },
  prefix: {
    total_optimizer_step: prefix.training.total_optimizer_step,
    next_window: prefix.cursor.next_window,
    model_hash: prefix.hashes.final_model,
    optimizer_state_hash: prefix.hashes.optimizer_state,
    model_sha256: sha256(prefixModel),
    optimizer_sha256: sha256(prefixOptimizer),
  },
  trigger: {
    total_optimizer_step: trigger.training.total_optimizer_step,
    next_window: trigger.cursor.next_window,
    model_hash: trigger.hashes.final_model,
    model_sha256: sha256(triggerModel),
    optimizer_state_hash: trigger.hashes.optimizer_state,
    optimizer_sha256: sha256(triggerOptimizer),
    selection,
    health: trigger.health,
  },
  development: {
    total_nll_millibits: development.evaluation.total_nll_millibits,
    delta_millibits: developmentDelta,
    residual_saturation_count: development.health.residual_saturation_count,
  },
  manifest_residual_saturation_count: saturation.aggregate.residual_saturation_count,
  source_relative_movement_l1: movement,
  gates: {
    training_shape_and_bindings_match: trainingShapeMatches,
    no_model_movement_before_trigger: prefixMatches,
    predicted_health_constrained_selection_exact: triggerMatches,
    model_and_optimizer_chain_match: chainMatches,
    development_strictly_improved: developmentDelta < 0,
    development_and_manifest_health_passed: postflightPassed,
    movement_isolated_to_expected_groups: onlyExpectedGroupsMoved,
    evidence_bindings_match: evidenceBindingsMatch,
    development_not_read_before_trigger_passed: true,
    test_partition_not_read: true,
    all_trigger_gates_passed: allGatesPassed,
  },
  authorization: contract.authorization,
  known_non_claims: contract.known_non_claims,
};

const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "health trigger checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
assert(allGatesPassed, "health-constrained step-408 postflight failed");
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  model_hash: result.trigger.model_hash,
  development_delta_millibits: result.development.delta_millibits,
  all_gates_passed: result.gates.all_trigger_gates_passed,
  out: outPath,
})}\n`);
