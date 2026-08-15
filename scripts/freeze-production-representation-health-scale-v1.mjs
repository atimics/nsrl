#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let contractPath = "";
let runDir = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
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
const addObjects = (values) => {
  const result = {};
  for (const value of values) {
    for (const [key, count] of Object.entries(value)) {
      result[key] = (result[key] ?? 0) + count;
    }
  }
  return result;
};
const deltaValue = (group) => group.total ?? group;

const contractBytes = await readFile(contractPath);
const contract = JSON.parse(contractBytes);
assert(
  contract.schema === "nsrl.production_representation_health_scale_contract.v1",
  "unexpected health scale contract schema",
);
for (const artifact of contract.bindings.artifacts) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const traces = await Promise.all(
  Array.from({ length: contract.training.chunks }, (_, interval) =>
    json(path.join(runDir, `chunk-${interval}`, "train.json"))),
);
const finalModelPath = path.join(runDir, `chunk-${contract.training.chunks - 1}`, "model.nsrlpm");
const finalOptimizerPath = path.join(
  runDir,
  `chunk-${contract.training.chunks - 1}`,
  "optimizer.nsrlpo",
);
const [finalModel, finalOptimizer, development, saturation, delta] = await Promise.all([
  readFile(finalModelPath),
  readFile(finalOptimizerPath),
  json(path.join(runDir, "development.json")),
  json(path.join(runDir, "saturation.json")),
  json(path.join(runDir, "delta.json")),
]);

let scheduleValid = true;
let cursorChainValid = true;
let modelHashChainValid = true;
let guardChainValid = true;
let transactionHealthValid = true;
let signedFeasibilityValid = true;
let previousModelHash = contract.trigger.model_hash;
let previousGuardNll = contract.trigger.guard_nll_millibits;
let previousTotalStep = contract.trigger.total_optimizer_step;
let previousNextWindow = contract.trigger.next_window;
const expectedSteps = contract.training.optimizer_steps_by_chunk;

for (const [interval, trace] of traces.entries()) {
  const expectedTotalStep = previousTotalStep + expectedSteps[interval];
  const expectedNextWindow = expectedTotalStep === contract.training.optimizer_steps
    ? 0
    : expectedTotalStep * contract.training.batch_windows;
  const finalChunk = interval + 1 === traces.length;
  scheduleValid &&= trace.schema === "nsrl.production_full_train_smoke.v1"
    && trace.profile === contract.profile
    && trace.parameter_count === contract.parameter_count
    && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
    && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
    && trace.training.context_tokens === contract.training.context_tokens
    && trace.training.windows === contract.training.windows
    && trace.training.targets_per_window === contract.training.targets_per_window
    && trace.training.training_workers === contract.training.training_workers
    && trace.training.evaluation_windows === contract.training.evaluation_windows
    && trace.training.epochs === contract.training.epochs
    && trace.training.batch_windows === contract.training.batch_windows
    && trace.training.optimizer_steps === expectedSteps[interval]
    && trace.training.total_optimizer_step === expectedTotalStep
    && trace.training.supervised_targets
      === expectedSteps[interval] * contract.training.batch_windows
        * contract.training.targets_per_window
    && trace.training.embedding_residual_flush === contract.training.embedding_residual_flush
    && trace.training.descent_guard_windows === contract.training.descent_guard_windows
    && trace.training.descent_guard_candidate_family === contract.training.candidate_family
    && trace.training.signed_block_feasibility === contract.training.feasibility
    && trace.training.output_backward_shift === contract.training.output_backward_shift
    && trace.training.probability_gradient_fractional_bits
      === contract.training.probability_gradient_fractional_bits
    && trace.training.probability_normalization === contract.training.probability_normalization
    && same(trace.training.learning_rate_shifts, contract.training.learning_rate_shifts);
  cursorChainValid &&= trace.cursor.start_epoch === 0
    && trace.cursor.start_window === previousNextWindow
    && trace.cursor.next_epoch === (finalChunk ? 1 : 0)
    && trace.cursor.next_window === expectedNextWindow
    && trace.cursor.schedule_complete === finalChunk;
  modelHashChainValid &&= trace.hashes.initial_model === previousModelHash;
  previousModelHash = trace.hashes.final_model;
  previousTotalStep = expectedTotalStep;
  previousNextWindow = expectedNextWindow;

  const guard = trace.descent_guard;
  guardChainValid &&= guard.surface === contract.training.descent_guard_surface
    && guard.window_rank_hash === contract.training.descent_guard_window_rank_hash
    && guard.update_window_overlap_count === 0
    && guard.initial_nll_millibits === previousGuardNll
    && guard.final_nll_millibits <= guard.initial_nll_millibits
    && guard.accepted_batches + guard.rejected_batches === guard.evaluated_batches;
  previousGuardNll = guard.final_nll_millibits;

  transactionHealthValid &&= trace.transaction.rejected_batch === null
    && allZero(trace.health)
    && trace.gates.saturated_batch_rejection_enabled === true;
  signedFeasibilityValid &&= trace.gates.training_only_descent_guard_enabled === true
    && trace.gates.descent_guard_update_windows_disjoint === true
    && trace.gates.signed_block_trust_region_enabled === true
    && trace.gates.signed_block_source_candidate_guarantees_nonworsening === true
    && trace.gates.signed_block_zero_guard_residual_saturation_enabled === true
    && trace.gates.signed_block_zero_saturation_feasibility_enforced === true
    && trace.signed_block_trust_region.zero_guard_residual_saturation_required === true;
  const selection = trace.signed_block_trust_region.last_selection;
  signedFeasibilityValid &&= selection === null
    || (selection.candidates_evaluated === contract.gates.signed_block_candidates_evaluated
      && selection.zero_saturation_candidates > 0
      && selection.selected_residual_saturation_count === 0
      && selection.selected_nll_millibits <= selection.before_nll_millibits);
}

const movement = Object.fromEntries(
  Object.entries(delta.groups).map(([group, value]) => [group, deltaValue(value).l1]),
);
const requiredGroupsMoved = contract.gates.required_parameter_groups
  .every((group) => movement[group] > 0);
const candidateFamilyIsolated = Object.entries(movement).every(([group, value]) =>
  contract.gates.allowed_moving_parameter_groups.includes(group) || value === 0);
const trainingHealth = addObjects(traces.map((trace) => trace.health));
const developmentNll = development.evaluation.total_nll_millibits;
const developmentDelta = developmentNll - contract.source.development_nll_millibits;
const manifestSaturation = saturation.aggregate.residual_saturation_count;
const numericHealthPassed = allZero(trainingHealth)
  && development.health.residual_saturation_count === 0
  && manifestSaturation <= contract.gates.manifest_residual_saturation_max;
const finalTrace = traces.at(-1);
const evidenceBindingsValid = finalTrace.hashes.final_model === development.model_hash
  && finalTrace.hashes.final_model === saturation.bindings.model_hash
  && finalTrace.hashes.final_model === delta.bindings.candidate_model_hash
  && contract.source.model_hash === delta.bindings.source_model_hash
  && development.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash;
const triggerBound = traces[0].hashes.initial_model === contract.trigger.model_hash
  && traces[0].cursor.start_window === contract.trigger.next_window;
const allGatesPassed = scheduleValid
  && cursorChainValid
  && modelHashChainValid
  && guardChainValid
  && transactionHealthValid
  && signedFeasibilityValid
  && requiredGroupsMoved
  && candidateFamilyIsolated
  && numericHealthPassed
  && evidenceBindingsValid
  && triggerBound
  && developmentDelta < 0;
const outcome = allGatesPassed
  ? "health_constrained_broader_horizon_replicated"
  : !scheduleValid || !cursorChainValid || !modelHashChainValid || !triggerBound
    ? "health_constrained_durable_chain_failed"
    : !numericHealthPassed || !transactionHealthValid || !signedFeasibilityValid
      ? "health_constrained_numeric_health_failed"
      : !guardChainValid
        ? "health_constrained_guard_failed"
        : !requiredGroupsMoved || !candidateFamilyIsolated
          ? "health_constrained_representation_isolation_failed"
          : developmentDelta >= 0
            ? "health_constrained_development_did_not_improve"
            : "health_constrained_evidence_binding_failed";

const chunks = traces.map((trace, interval) => ({
  interval,
  start_window: trace.cursor.start_window,
  next_epoch: trace.cursor.next_epoch,
  next_window: trace.cursor.next_window,
  optimizer_steps: trace.training.optimizer_steps,
  total_optimizer_step: trace.training.total_optimizer_step,
  initial_model_hash: trace.hashes.initial_model,
  final_model_hash: trace.hashes.final_model,
  optimizer_state_hash: trace.hashes.optimizer_state,
  movement_l1: trace.movement_l1,
  health: trace.health,
  descent_guard: trace.descent_guard,
  signed_block_trust_region: trace.signed_block_trust_region,
}));
const result = {
  schema: "nsrl.production_representation_health_scale.v1",
  checked: check,
  objective: contract.objective,
  outcome,
  contract: { path: contractPath, sha256: sha256(contractBytes) },
  source_model_hash: contract.source.model_hash,
  trigger: contract.trigger,
  candidate: {
    model_hash: finalTrace.hashes.final_model,
    model_sha256: sha256(finalModel),
    optimizer_state_hash: finalTrace.hashes.optimizer_state,
    optimizer_sha256: sha256(finalOptimizer),
    total_optimizer_step: finalTrace.training.total_optimizer_step,
    schedule_complete: finalTrace.cursor.schedule_complete,
  },
  chunks,
  aggregate: {
    source_relative_movement_l1: movement,
    training_health: trainingHealth,
    descent_guard_initial_nll_millibits: contract.trigger.guard_nll_millibits,
    descent_guard_final_nll_millibits: finalTrace.descent_guard.final_nll_millibits,
    descent_guard_evaluated_batches_after_trigger: traces.reduce(
      (sum, trace) => sum + trace.descent_guard.evaluated_batches,
      0,
    ),
    signed_block_evaluated_batches_after_trigger: traces.reduce(
      (sum, trace) => sum + trace.signed_block_trust_region.evaluated_batches,
      0,
    ),
    signed_block_selected_batches_after_trigger: traces.reduce(
      (sum, trace) => sum + trace.signed_block_trust_region.selected_batches,
      0,
    ),
  },
  development: {
    source_total_nll_millibits: contract.source.development_nll_millibits,
    candidate_total_nll_millibits: developmentNll,
    delta_millibits: developmentDelta,
  },
  health: {
    development_residual_saturation_count: development.health.residual_saturation_count,
    manifest_residual_saturation_count: manifestSaturation,
  },
  gates: {
    trigger_checkpoint_bound: triggerBound,
    complete_durable_schedule_and_cursor_chain: scheduleValid && cursorChainValid,
    model_hash_chain_complete: modelHashChainValid,
    training_only_guard_nonworsening: guardChainValid,
    zero_saturation_signed_feasibility_enforced: signedFeasibilityValid,
    no_batch_rejected_for_saturation: transactionHealthValid,
    training_development_manifest_saturation_zero: numericHealthPassed,
    required_parameter_groups_moved: requiredGroupsMoved,
    movement_isolated_to_signed_candidate_family: candidateFamilyIsolated,
    evidence_bindings_match_final_candidate: evidenceBindingsValid,
    development_strictly_improved: developmentDelta < 0,
    test_partition_not_read: true,
    all_broader_horizon_gates_passed: allGatesPassed,
  },
  authorization: contract.authorization,
  known_non_claims: contract.known_non_claims,
};

const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "health scale checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  total_optimizer_step: result.candidate.total_optimizer_step,
  development_delta_millibits: result.development.delta_millibits,
  all_gates_passed: result.gates.all_broader_horizon_gates_passed,
  out: outPath,
})}\n`);
