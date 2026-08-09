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

const contractBytes = await readFile(contractPath);
const contract = JSON.parse(contractBytes);
assert(
  contract.schema === "nsrl.production_representation_restart_replay_contract.v1",
  "unexpected restart replay contract schema",
);

for (const artifact of contract.bindings.artifacts) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const files = {
  partialTrace: path.join(runDir, "partial.json"),
  partialModel: path.join(runDir, "partial.nsrlpm"),
  partialOptimizer: path.join(runDir, "partial.nsrlpo"),
  resumeTrace: path.join(runDir, "resume.json"),
  replayModel: path.join(runDir, "replay.nsrlpm"),
  replayOptimizer: path.join(runDir, "replay.nsrlpo"),
};
const [
  partial,
  partialModel,
  partialOptimizer,
  resume,
  replayModel,
  replayOptimizer,
] = await Promise.all([
  json(files.partialTrace),
  readFile(files.partialModel),
  readFile(files.partialOptimizer),
  json(files.resumeTrace),
  readFile(files.replayModel),
  readFile(files.replayOptimizer),
]);

for (const trace of [partial, resume]) {
  assert(trace.schema === "nsrl.production_full_train_smoke.v1", "unexpected trace schema");
  assert(
    trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash,
    "training binding mismatch",
  );
  assert(
    trace.training.context_tokens === contract.training.context_tokens
      && trace.training.windows === contract.training.windows
      && trace.training.targets_per_window === contract.training.targets_per_window
      && trace.training.batch_windows === contract.training.batch_windows
      && trace.training.descent_guard_windows === contract.training.descent_guard_windows
      && trace.training.descent_guard_candidate_family
        === contract.training.signed_block_candidate_family,
    "training geometry mismatch",
  );
  assert(
    trace.training.probability_normalization === contract.training.probability_normalization
      && trace.training.probability_gradient_fractional_bits
        === contract.training.probability_gradient_fractional_bits
      && trace.training.output_backward_shift === contract.training.output_backward_shift,
    "training numeric policy mismatch",
  );
  assert(
    same(trace.training.learning_rate_shifts, contract.training.learning_rate_shifts),
    "training schedule mismatch",
  );
  assert(
    trace.transaction.rejected_batch === null
      && allZero(trace.health)
      && trace.gates.saturated_batch_rejection_enabled === true
      && trace.gates.training_only_descent_guard_enabled === true
      && trace.gates.signed_block_trust_region_enabled === true,
    "transaction or numeric-health gate failed",
  );
}

const partialAtDeclaredBoundary =
  partial.training.optimizer_steps === contract.replay.partial_optimizer_steps
  && partial.training.total_optimizer_step === contract.replay.partial_optimizer_steps
  && partial.cursor.start_epoch === 0
  && partial.cursor.start_window === 0
  && partial.cursor.next_epoch === 0
  && partial.cursor.next_window === contract.replay.partial_next_window
  && partial.cursor.schedule_complete === false
  && partial.hashes.initial_model === contract.source.model_hash
  && partial.hashes.final_model === contract.source.model_hash
  && partial.signed_block_trust_region.evaluated_batches === 0
  && partial.descent_guard.evaluated_batches === 0
  && allZero(partial.movement_l1);

const expectedSelection = contract.replay.expected_signed_selection;
const selection = resume.signed_block_trust_region.last_selection;
const selectedEventReplayedExactly =
  resume.cursor.start_epoch === 0
  && resume.cursor.start_window === contract.replay.partial_next_window
  && resume.training.optimizer_steps === contract.replay.resume_optimizer_steps
  && resume.training.total_optimizer_step === contract.training.optimizer_steps
  && resume.cursor.next_epoch === 1
  && resume.cursor.next_window === 0
  && resume.cursor.schedule_complete === true
  && resume.hashes.initial_model === contract.source.model_hash
  && resume.hashes.final_model === contract.reference.final_model_hash
  && resume.hashes.optimizer_state === contract.reference.final_optimizer_state_hash
  && resume.signed_block_trust_region.evaluated_batches === 1
  && resume.signed_block_trust_region.selected_batches === 1
  && selection.attempted_total_optimizer_step === contract.replay.selected_optimizer_step
  && selection.start_window === contract.replay.partial_next_window
  && selection.candidates_evaluated === contract.replay.candidates_evaluated
  && selection.before_nll_millibits === expectedSelection.before_nll_millibits
  && selection.forward_nll_millibits === expectedSelection.forward_nll_millibits
  && selection.selected_nll_millibits === expectedSelection.selected_nll_millibits
  && same(selection.selected_steps, expectedSelection.steps)
  && same(selection.selected_movement_l1, expectedSelection.movement_l1)
  && same(resume.movement_l1, expectedSelection.movement_l1);

const replayModelSha256 = sha256(replayModel);
const replayOptimizerSha256 = sha256(replayOptimizer);
const modelByteIdentical = replayModelSha256 === contract.reference.final_model_sha256;
const optimizerByteIdentical =
  replayOptimizerSha256 === contract.reference.final_optimizer_sha256;
const allGatesPassed = partialAtDeclaredBoundary
  && selectedEventReplayedExactly
  && modelByteIdentical
  && optimizerByteIdentical;

const result = {
  schema: "nsrl.production_representation_restart_replay.v1",
  checked: check,
  objective: contract.objective,
  outcome: allGatesPassed
    ? "selected_signed_event_exactly_restart_replayed"
    : "restart_replay_failed",
  contract: {
    path: contractPath,
    sha256: sha256(contractBytes),
  },
  split: {
    completed_optimizer_steps: partial.training.total_optimizer_step,
    next_epoch: partial.cursor.next_epoch,
    next_window: partial.cursor.next_window,
    model_hash: partial.hashes.final_model,
    model_sha256: sha256(partialModel),
    optimizer_state_hash: partial.hashes.optimizer_state,
    optimizer_sha256: sha256(partialOptimizer),
    guard_nll_millibits: partial.descent_guard.final_nll_millibits,
  },
  replay: {
    invocation_optimizer_steps: resume.training.optimizer_steps,
    total_optimizer_step: resume.training.total_optimizer_step,
    schedule_complete: resume.cursor.schedule_complete,
    final_model_hash: resume.hashes.final_model,
    final_model_sha256: replayModelSha256,
    final_optimizer_state_hash: resume.hashes.optimizer_state,
    final_optimizer_sha256: replayOptimizerSha256,
    signed_selection: selection,
    descent_guard: resume.descent_guard,
    health: resume.health,
  },
  reference: contract.reference,
  gates: {
    split_immediately_before_selected_event: partialAtDeclaredBoundary,
    disk_model_optimizer_binding_accepted: true,
    selected_event_decision_trace_exact: selectedEventReplayedExactly,
    final_model_byte_identical: modelByteIdentical,
    final_optimizer_byte_identical: optimizerByteIdentical,
    training_saturation_zero: allZero(resume.health),
    no_batch_rejected: resume.transaction.rejected_batch === null,
    test_partition_not_read: true,
    all_restart_replay_gates_passed: allGatesPassed,
  },
  authorization: {
    diagnostic_only: true,
    test_evaluation: false,
    quality_postflight: false,
    quality_promotion: false,
    open_generation_rerun: false,
    paid_scaling: false,
  },
};

const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "restart replay checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  all_gates_passed: result.gates.all_restart_replay_gates_passed,
  out: outPath,
})}\n`);
