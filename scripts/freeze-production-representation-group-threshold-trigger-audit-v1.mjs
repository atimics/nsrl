#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let contractPath = "";
let auditPath = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--audit") auditPath = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !auditPath || !outPath) {
  throw new Error("--contract, --audit, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const sum = (values) => Object.values(values).reduce((total, value) => total + value, 0);

const contract = await json(contractPath);
assert(
  contract.schema === "nsrl.production_representation_group_threshold_trigger_audit_contract.v1",
  "unexpected representation group threshold trigger contract schema",
);
for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.source.optimizer_path, sha256: contract.source.optimizer_sha256 },
  { path: contract.source.trace_path, sha256: contract.source.trace_sha256 },
  { path: contract.failure_witness.trace_path, sha256: contract.failure_witness.trace_sha256 },
  ...contract.derivation.artifacts,
  ...Object.values(contract.derivation.residual_audits),
]) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

for (const group of ["k", "v", "o"]) {
  const expected = contract.derivation.residual_audits[group];
  const residualAudit = await json(expected.path);
  assert(
    residualAudit.schema === "nsrl.production_optimizer_residual_audit.v1"
      && residualAudit.group.name === group,
    `unexpected ${group} residual audit binding`,
  );
  const threshold = residualAudit.group.thresholds.find(
    (row) => row.effective_shift === expected.first_crossing_effective_shift,
  );
  assert(
    threshold?.coordinates_at_threshold === expected.coordinates_at_threshold
      && threshold.coordinates_at_threshold > 0,
    `${group} selected residual threshold mismatch`,
  );
  const moreDamped = residualAudit.group.thresholds
    .filter((row) => row.effective_shift > expected.first_crossing_effective_shift);
  assert(
    moreDamped.every((row) => row.coordinates_at_threshold === 0),
    `${group} selection is not the first nonzero threshold`,
  );
}
const embeddingResidualAudit = await json(contract.derivation.residual_audits.embeddings.path);
assert(
  embeddingResidualAudit.group.name === "embeddings"
    && embeddingResidualAudit.group.maximum_absolute_residual
      === contract.derivation.residual_audits.embeddings.maximum_absolute_residual,
  "embedding residual audit mismatch",
);

const [sourceTrace, failureTrace, audit, modelBytes, optimizerBytes] = await Promise.all([
  json(contract.source.trace_path),
  json(contract.failure_witness.trace_path),
  json(auditPath),
  readFile(contract.source.model_path),
  readFile(contract.source.optimizer_path),
]);
assert(
  sourceTrace.hashes.final_model === contract.source.model_hash
    && sourceTrace.hashes.optimizer_state === contract.source.optimizer_state_hash
    && sourceTrace.training.total_optimizer_step === contract.source.total_optimizer_step
    && sourceTrace.cursor.next_window === contract.source.next_window,
  "source checkpoint mismatch",
);
for (const field of [
  "gradient_saturation_count",
  "residual_saturation_count",
  "weight_saturation_count",
]) {
  assert(
    failureTrace.health[field] === contract.failure_witness[field],
    `failure witness ${field} mismatch`,
  );
}

assert(audit.schema === "nsrl.production_saturation_backoff_audit.v1", "unexpected audit schema");
assert(
  audit.bindings.model_hash === contract.source.model_hash
    && audit.bindings.optimizer_state_hash === contract.source.optimizer_state_hash
    && audit.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
    && audit.bindings.token_stream_hash === contract.bindings.train_token_stream_hash,
  "audit binding mismatch",
);
assert(
  audit.source_cursor.total_optimizer_step === contract.source.total_optimizer_step
    && audit.source_cursor.next_window === contract.source.next_window,
  "audit cursor mismatch",
);
const cli = contract.implementation.cli_learning_rate_shifts;
assert(
  audit.audit.mode === contract.audit.mode
    && audit.audit.candidate_backward_quantization
      === contract.implementation.backward_quantization
    && audit.audit.candidate_embedding_learning_rate_shift === cli.embeddings
    && audit.audit.candidate_embedding_learning_rate_boost_shift === cli.embedding_boost
    && audit.audit.candidate_k_learning_rate_shift === cli.k
    && audit.audit.candidate_v_learning_rate_shift === cli.v
    && audit.audit.candidate_o_learning_rate_shift === cli.o
    && audit.audit.candidate_flush_batched_embedding_residuals === true
    && audit.audit.candidate_schedule_hash_rebound_in_memory_only === true
    && audit.audit.candidate_artifacts_persisted === false,
  "candidate schedule or isolation mismatch",
);
assert(
  same(audit.audit.output_backward_shifts, contract.audit.candidate_output_backward_shifts)
    && audit.rows.length === 1,
  "candidate shift or row count mismatch",
);

const row = audit.rows[0];
const trace = row.train_trace;
const effective = contract.implementation.effective_learning_rate_shifts;
assert(
  row.output_backward_shift === contract.implementation.output_backward_shift
    && trace.training.output_backward_shift === contract.implementation.output_backward_shift
    && trace.training.embedding_learning_rate_boost_shift === cli.embedding_boost
    && trace.training.learning_rate_shifts.embeddings === effective.embeddings
    && trace.training.learning_rate_shifts.k === effective.k
    && trace.training.learning_rate_shifts.v === effective.v
    && trace.training.learning_rate_shifts.o === effective.o,
  "candidate effective schedule mismatch",
);
assert(
  trace.training.embedding_residual_flush === "all_batch_touched_tokens"
    && trace.gates.batched_embedding_residual_flush === true,
  "batch-complete embedding residual flush is not active",
);

const rejected = trace.transaction?.rejected_batch ?? null;
const attemptedHealth = rejected === null ? trace.health : {
  gradient_saturation_count: rejected.gradient_saturation_count,
  residual_saturation_count: rejected.residual_saturation_count,
  weight_saturation_count: rejected.weight_saturation_count,
};
const movement = rejected?.movement_l1 ?? trace.movement_l1;
const accepted = rejected === null
  && sum(attemptedHealth) <= contract.gates.gradient_residual_and_weight_saturation_max
  && trace.training.total_optimizer_step === contract.source.total_optimizer_step + 1
  && trace.cursor.next_window === contract.source.next_window + trace.training.batch_windows;
if (!accepted) {
  assert(
    trace.training.total_optimizer_step === contract.source.total_optimizer_step
      && trace.cursor.next_window === contract.source.next_window,
    "rejected trigger batch advanced durable state",
  );
}
const movedGroups = contract.gates.required_moving_groups
  .filter((group) => movement[group] > 0);
const requiredGroupsMoved = movedGroups.length === contract.gates.required_moving_groups.length;
const allGatesPassed = accepted && requiredGroupsMoved;
const outcome = !accepted
  ? "group_threshold_trigger_batch_rejected"
  : !requiredGroupsMoved
    ? "group_threshold_trigger_safe_but_required_groups_not_live"
    : "group_threshold_trigger_safe_and_required_groups_live";

const result = {
  schema: "nsrl.production_representation_group_threshold_trigger_audit.v1",
  checked: check,
  objective: contract.objective,
  outcome,
  implementation: contract.implementation,
  source: {
    total_optimizer_step: contract.source.total_optimizer_step,
    next_window: contract.source.next_window,
    model_hash: contract.source.model_hash,
    model_sha256: sha256(modelBytes),
    optimizer_state_hash: contract.source.optimizer_state_hash,
    optimizer_sha256: sha256(optimizerBytes),
  },
  trigger_batch: {
    accepted,
    attempted_health: attemptedHealth,
    movement_l1: movement,
    gradient_nonzero_count: rejected?.gradient_nonzero_count
      ?? trace.diagnostics.gradient_nonzero_count,
    update_nonzero_count: rejected?.update_nonzero_count
      ?? trace.diagnostics.update_nonzero_count,
    final_model_hash: trace.hashes.final_model,
    final_optimizer_state_hash: trace.hashes.optimizer_state,
  },
  gates: {
    exact_trigger_batch_committed: accepted,
    atomic_saturation_guard_active: trace.transaction?.saturation_policy === "reject_batch_stop",
    gradient_residual_and_weight_saturation_zero: sum(attemptedHealth) === 0,
    batch_complete_embedding_residual_flush_active: true,
    required_groups_moved: requiredGroupsMoved,
    source_model_and_optimizer_unchanged: true,
    candidate_artifacts_not_persisted: true,
    test_partition_not_read: true,
    fresh_full_horizon_run_authorized: allGatesPassed,
  },
  authorization: {
    diagnostic_only: true,
    fresh_full_horizon_run: allGatesPassed,
    optimizer_schedule_change: false,
    candidate_checkpoint: false,
    test_evaluation: false,
    quality_promotion: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "group threshold checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  all_gates_passed: allGatesPassed,
  out: outPath,
})}\n`);
