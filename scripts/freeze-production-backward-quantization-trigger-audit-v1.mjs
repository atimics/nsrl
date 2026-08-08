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
  contract.schema === "nsrl.production_backward_quantization_trigger_audit_contract.v1",
  "unexpected backward quantization trigger audit contract schema",
);
assert(
  contract.implementation.backward_quantization === "late-stochastic"
    && contract.implementation.backward_stochastic_seed >= 0,
  "contract does not bind late stochastic backward quantization",
);
assert(
  same(contract.audit.candidate_output_backward_shifts, [contract.implementation.output_backward_shift]),
  "contract must authorize exactly the implemented output backward shift",
);

for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.source.optimizer_path, sha256: contract.source.optimizer_sha256 },
  { path: contract.source.trace_path, sha256: contract.source.trace_sha256 },
  { path: contract.failure_witness.trace_path, sha256: contract.failure_witness.trace_sha256 },
  { path: contract.derivation.artifact_path, sha256: contract.derivation.artifact_sha256 },
]) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const [sourceTrace, failureTrace, audit, modelBytes, optimizerBytes] = await Promise.all([
  json(contract.source.trace_path),
  json(contract.failure_witness.trace_path),
  json(auditPath),
  readFile(contract.source.model_path),
  readFile(contract.source.optimizer_path),
]);
assert(
  sourceTrace.hashes.final_model === contract.source.model_hash
    && sourceTrace.hashes.optimizer_state === contract.source.optimizer_state_hash,
  "source checkpoint hash mismatch",
);
assert(
  sourceTrace.training.total_optimizer_step === contract.source.total_optimizer_step
    && sourceTrace.cursor.next_window === contract.source.next_window,
  "source checkpoint cursor mismatch",
);
assert(
  failureTrace.training.output_backward_shift === contract.failure_witness.output_backward_shift
    && failureTrace.training.total_optimizer_step === contract.source.total_optimizer_step + 1,
  "failure witness step mismatch",
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
assert(
  audit.audit.mode === contract.audit.mode
    && audit.audit.candidate_backward_quantization
      === contract.implementation.backward_quantization
    && audit.audit.candidate_backward_stochastic_seed
      === contract.implementation.backward_stochastic_seed
    && audit.audit.candidate_schedule_hash_rebound_in_memory_only === true
    && audit.audit.candidate_artifacts_persisted === false,
  "audit isolation or quantizer binding mismatch",
);
assert(
  same(audit.audit.output_backward_shifts, contract.audit.candidate_output_backward_shifts),
  "candidate shift mismatch",
);
assert(audit.rows.length === 1, "trigger audit must contain exactly one row");

const row = audit.rows[0];
const trace = row.train_trace;
const expectedShift = contract.implementation.output_backward_shift;
assert(
  row.output_backward_shift === expectedShift
    && trace.training.output_backward_shift === expectedShift,
  "trigger trace output backward shift mismatch",
);
assert(
  trace.training.backward_quantization === contract.implementation.backward_quantization
    && trace.training.backward_stochastic_seed
      === contract.implementation.backward_stochastic_seed,
  "trigger trace quantizer mismatch",
);
assert(
  trace.cursor.start_window === contract.source.next_window
    && trace.cursor.next_window === contract.source.next_window + trace.training.batch_windows
    && trace.training.total_optimizer_step === contract.source.total_optimizer_step + 1,
  "exact trigger batch did not advance by one committed optimizer step",
);
assert(
  trace.transaction?.saturation_policy === "reject_batch_stop"
    && trace.transaction.rejected_batch === null,
  "exact trigger batch was rejected or atomic guard was absent",
);
assert(
  sum(trace.health) <= contract.gates.gradient_residual_and_weight_saturation_max,
  "exact trigger batch saturated",
);
assert(
  trace.diagnostics.backward_stochastic_round_up_count > 0,
  "late stochastic backward emitted no stochastic round-ups",
);
assert(
  trace.diagnostics.backward_stochastic_round_up_count
    <= trace.diagnostics.backward_quantization_count,
  "stochastic round-up count exceeds quantization count",
);

const result = {
  schema: "nsrl.production_backward_quantization_trigger_audit.v1",
  checked: check,
  objective: contract.objective,
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
    output_backward_shift: row.output_backward_shift,
    candidate_schedule_hash: row.candidate_schedule_hash,
    total_optimizer_step: trace.training.total_optimizer_step,
    next_window: trace.cursor.next_window,
    health: trace.health,
    stochastic_round_up_count: trace.diagnostics.backward_stochastic_round_up_count,
    backward_quantization_count: trace.diagnostics.backward_quantization_count,
    gradient_nonzero_count: trace.diagnostics.gradient_nonzero_count,
    residual_carry_count: trace.diagnostics.residual_carry_count,
    update_nonzero_count: trace.diagnostics.update_nonzero_count,
    movement_l1: trace.movement_l1,
    final_model_hash: trace.hashes.final_model,
    final_optimizer_state_hash: trace.hashes.optimizer_state,
  },
  outcome: "trigger_batch_safe_with_stochastic_sub_lsb_signal",
  gates: {
    exact_trigger_batch_committed: true,
    atomic_saturation_guard_active: true,
    gradient_residual_and_weight_saturation_zero: true,
    stochastic_round_up_count_nonzero: true,
    source_model_and_optimizer_unchanged: true,
    candidate_artifacts_not_persisted: true,
    test_partition_not_read: true,
    fresh_full_horizon_run_authorized: true,
  },
  authorization: {
    diagnostic_only: true,
    fresh_full_horizon_run: true,
    optimizer_schedule_change: false,
    candidate_checkpoint: false,
    test_evaluation: false,
    quality_promotion: false,
    open_generation_rerun: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "trigger audit checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  stochastic_round_up_count: result.trigger_batch.stochastic_round_up_count,
  out: outPath,
})}\n`);
