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
  contract.schema === "nsrl.production_representation_scale_contract.v1",
  "unexpected representation scale contract schema",
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
const finalModelPath = path.join(
  runDir,
  `chunk-${contract.training.chunks - 1}`,
  "model.nsrlpm",
);
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

const effectiveShifts = contract.training.learning_rate_shifts;
const expectedChunkSteps = contract.training.optimizer_steps / contract.training.chunks;
const expectedChunkWindows = contract.training.windows / contract.training.chunks;
const expectedChunkTargets = expectedChunkWindows * contract.training.targets_per_window;
let hashChainValid = true;
let cursorChainValid = true;
let guardChainValid = true;
let scheduleValid = true;
let transactionHealthValid = true;
let previousModelHash = contract.source.model_hash;
let previousGuardNll = null;
let guardWindowRankHash = null;

for (const [interval, trace] of traces.entries()) {
  assert(trace.schema === "nsrl.production_full_train_smoke.v1", "unexpected training trace");
  const expectedStartWindow = interval * expectedChunkWindows;
  const expectedEndWindow = (interval + 1) * expectedChunkWindows;
  const finalChunk = interval + 1 === contract.training.chunks;
  scheduleValid &&= trace.profile === contract.profile
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
    && trace.training.optimizer_steps === expectedChunkSteps
    && trace.training.supervised_targets === expectedChunkTargets
    && trace.training.embedding_residual_flush === contract.training.embedding_residual_flush
    && trace.training.descent_guard_windows === contract.training.descent_guard_windows
    && trace.training.descent_guard_candidate_family
      === contract.training.signed_block_candidate_family
    && trace.training.output_backward_shift === contract.training.output_backward_shift
    && trace.training.probability_gradient_fractional_bits
      === contract.training.probability_gradient_fractional_bits
    && trace.training.probability_normalization === contract.training.probability_normalization
    && same(trace.training.learning_rate_shifts, effectiveShifts);
  cursorChainValid &&= trace.training.total_optimizer_step
      === (interval + 1) * expectedChunkSteps
    && trace.cursor.start_epoch === 0
    && trace.cursor.start_window === expectedStartWindow
    && trace.cursor.next_epoch === (finalChunk ? 1 : 0)
    && trace.cursor.next_window === (finalChunk ? 0 : expectedEndWindow)
    && trace.cursor.schedule_complete === finalChunk;
  hashChainValid &&= trace.hashes.initial_model === previousModelHash;
  previousModelHash = trace.hashes.final_model;

  const guard = trace.descent_guard;
  if (guardWindowRankHash === null) guardWindowRankHash = guard.window_rank_hash;
  guardChainValid &&= guard.surface === contract.training.descent_guard_surface
    && guard.window_rank_hash === guardWindowRankHash
    && guard.window_rank_hash !== "0x0000000000000000"
    && guard.update_window_overlap_count === 0
    && guard.final_nll_millibits <= guard.initial_nll_millibits
    && (previousGuardNll === null || guard.initial_nll_millibits === previousGuardNll)
    && guard.accepted_batches + guard.rejected_batches === guard.evaluated_batches;
  previousGuardNll = guard.final_nll_millibits;

  transactionHealthValid &&= trace.transaction.rejected_batch === null
    && allZero(trace.health)
    && trace.gates.saturated_batch_rejection_enabled === true
    && trace.gates.training_only_descent_guard_enabled === true
    && trace.gates.signed_block_trust_region_enabled === true
    && trace.gates.signed_block_source_candidate_guarantees_nonworsening === true;
}

const movement = Object.fromEntries(
  Object.entries(delta.groups).map(([group, value]) => [group, deltaValue(value).l1]),
);
const aggregateAttemptedMovement = addObjects(traces.map((trace) => trace.movement_l1));
const requiredGroupsMoved = contract.gates.required_parameter_groups
  .every((group) => movement[group] > 0);
const frozenGroupsUnchanged = contract.gates.frozen_parameter_groups
  .every((group) => movement[group] === 0);
const signedEvaluatedBatches = traces.reduce(
  (sum, trace) => sum + trace.signed_block_trust_region.evaluated_batches,
  0,
);
const signedSelectedBatches = traces.reduce(
  (sum, trace) => sum + trace.signed_block_trust_region.selected_batches,
  0,
);
const signedLastSelections = traces.flatMap((trace, interval) => {
  const selection = trace.signed_block_trust_region.last_selection;
  return selection === null ? [] : [{ interval, ...selection }];
});
const signedBlockPassed = signedEvaluatedBatches
    >= contract.gates.signed_block_evaluated_batches_min
  && signedSelectedBatches >= contract.gates.signed_block_selected_batches_min
  && signedLastSelections.every((selection) =>
    selection.candidates_evaluated === contract.gates.signed_block_candidates_evaluated
      && selection.selected_nll_millibits <= selection.before_nll_millibits);
const trainingHealth = addObjects(traces.map((trace) => trace.health));
const developmentNll = development.evaluation.total_nll_millibits;
const developmentDelta = developmentNll - contract.source.development_total_nll_millibits;
const developmentImproved = developmentDelta < 0;
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
const allGatesPassed = scheduleValid
  && cursorChainValid
  && hashChainValid
  && guardChainValid
  && transactionHealthValid
  && signedBlockPassed
  && numericHealthPassed
  && requiredGroupsMoved
  && frozenGroupsUnchanged
  && evidenceBindingsValid
  && developmentImproved;
const outcome = allGatesPassed
  ? "broader_signed_representation_horizon_replicated"
  : !scheduleValid || !cursorChainValid || !hashChainValid
    ? "durable_training_chain_failed"
    : !numericHealthPassed || !transactionHealthValid
      ? "broader_horizon_numeric_health_failed"
      : !guardChainValid || !signedBlockPassed
        ? "broader_horizon_signed_guard_failed"
        : !requiredGroupsMoved || !frozenGroupsUnchanged
          ? "broader_horizon_representation_isolation_failed"
          : !developmentImproved
            ? "broader_horizon_did_not_improve_development"
            : "broader_horizon_evidence_binding_failed";

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
  schema: "nsrl.production_representation_scale.v1",
  checked: check,
  objective: contract.objective,
  outcome,
  contract: { path: contractPath, sha256: sha256(contractBytes) },
  source_model_hash: contract.source.model_hash,
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
    attempted_movement_l1: aggregateAttemptedMovement,
    source_relative_movement_l1: movement,
    training_health: trainingHealth,
    descent_guard_window_rank_hash: guardWindowRankHash,
    descent_guard_initial_nll_millibits: traces[0].descent_guard.initial_nll_millibits,
    descent_guard_final_nll_millibits: finalTrace.descent_guard.final_nll_millibits,
    descent_guard_evaluated_batches: traces.reduce(
      (sum, trace) => sum + trace.descent_guard.evaluated_batches,
      0,
    ),
    descent_guard_accepted_batches: traces.reduce(
      (sum, trace) => sum + trace.descent_guard.accepted_batches,
      0,
    ),
    descent_guard_rejected_batches: traces.reduce(
      (sum, trace) => sum + trace.descent_guard.rejected_batches,
      0,
    ),
    signed_block_evaluated_batches: signedEvaluatedBatches,
    signed_block_selected_batches: signedSelectedBatches,
    signed_block_last_selections: signedLastSelections,
  },
  development: {
    source_total_nll_millibits: contract.source.development_total_nll_millibits,
    candidate_total_nll_millibits: developmentNll,
    delta_millibits: developmentDelta,
  },
  health: {
    development_residual_saturation_count: development.health.residual_saturation_count,
    manifest_residual_saturation_count: manifestSaturation,
  },
  gates: {
    all_durable_chunks_match_schedule: scheduleValid,
    cursor_and_model_hash_chain_complete: cursorChainValid && hashChainValid,
    training_only_descent_guard_nonworsening: guardChainValid,
    signed_block_trust_region_passed: signedBlockPassed,
    no_batch_rejected_for_saturation: transactionHealthValid,
    training_development_manifest_saturation_zero: numericHealthPassed,
    required_parameter_groups_moved: requiredGroupsMoved,
    frozen_parameter_groups_unchanged: frozenGroupsUnchanged,
    evidence_bindings_match_final_candidate: evidenceBindingsValid,
    development_strictly_improved: developmentImproved,
    test_partition_not_read: true,
    all_broader_horizon_replication_gates_passed: allGatesPassed,
  },
  authorization: {
    diagnostic_only: true,
    test_evaluation: false,
    quality_postflight: false,
    quality_promotion: false,
    open_generation_rerun: false,
    hidden_panel_access: false,
    paid_scaling: false,
  },
  known_non_claims: contract.known_non_claims,
};

const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "representation scale checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  total_optimizer_step: result.candidate.total_optimizer_step,
  development_delta_millibits: result.development.delta_millibits,
  all_gates_passed: result.gates.all_broader_horizon_replication_gates_passed,
  out: outPath,
})}\n`);
