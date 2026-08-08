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
const contract = await json(contractPath);
assert(
  contract.schema === "nsrl.production_saturation_backoff_audit_contract.v1",
  "unexpected saturation backoff contract schema",
);
for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.source.optimizer_path, sha256: contract.source.optimizer_sha256 },
  { path: contract.source.trace_path, sha256: contract.source.trace_sha256 },
  { path: contract.failure_witness.trace_path, sha256: contract.failure_witness.trace_sha256 },
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
    && failureTrace.training.total_optimizer_step
      === contract.source.total_optimizer_step + 1,
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
  audit.audit.mode === contract.implementation.audit_mode
    && audit.audit.candidate_schedule_hash_rebound_in_memory_only === true
    && audit.audit.candidate_artifacts_persisted === false,
  "audit isolation claim mismatch",
);
assert(
  same(audit.audit.output_backward_shifts, contract.audit.candidate_output_backward_shifts),
  "candidate shift range mismatch",
);
assert(audit.rows.length === contract.audit.candidate_output_backward_shifts.length,
  "candidate row count mismatch");

const rows = audit.rows.map((row, index) => {
  const expectedShift = contract.audit.candidate_output_backward_shifts[index];
  const trace = row.train_trace;
  assert(row.output_backward_shift === expectedShift, `candidate shift mismatch at row ${index}`);
  assert(
    trace.training.output_backward_shift === expectedShift
      && trace.cursor.start_window === contract.source.next_window,
    `candidate trace binding mismatch at shift ${expectedShift}`,
  );
  const rejected = trace.transaction?.rejected_batch ?? null;
  const attemptedHealth = rejected === null ? trace.health : {
    gradient_saturation_count: rejected.gradient_saturation_count,
    residual_saturation_count: rejected.residual_saturation_count,
    weight_saturation_count: rejected.weight_saturation_count,
  };
  const saturationTotal = Object.values(attemptedHealth)
    .reduce((sum, value) => sum + value, 0);
  const accepted = rejected === null
    && saturationTotal === 0
    && trace.training.total_optimizer_step === contract.source.total_optimizer_step + 1
    && trace.cursor.next_window === contract.source.next_window + trace.training.batch_windows;
  if (!accepted) {
    assert(
      trace.training.total_optimizer_step === contract.source.total_optimizer_step
        && trace.cursor.next_window === contract.source.next_window,
      `rejected candidate advanced durable state at shift ${expectedShift}`,
    );
  }
  return {
    output_backward_shift: expectedShift,
    candidate_schedule_hash: row.candidate_schedule_hash,
    accepted,
    attempted_health: attemptedHealth,
    attempted_movement_l1: rejected?.movement_l1 ?? trace.movement_l1,
    attempted_saturation_by_group: rejected?.saturation_by_group
      ?? trace.diagnostics.saturation_by_group,
    final_model_hash: trace.hashes.final_model,
    final_optimizer_state_hash: trace.hashes.optimizer_state,
  };
});

const base = rows[0];
assert(base.output_backward_shift === contract.failure_witness.output_backward_shift,
  "base shift is not the failure witness shift");
assert(!base.accepted, "base failure witness was unexpectedly accepted");
assert(
  same(base.attempted_health, {
    gradient_saturation_count: contract.failure_witness.gradient_saturation_count,
    residual_saturation_count: contract.failure_witness.residual_saturation_count,
    weight_saturation_count: contract.failure_witness.weight_saturation_count,
  }),
  "base shift did not reproduce the failure witness",
);
const selected = rows.find((row) => row.accepted) ?? null;
const result = {
  schema: "nsrl.production_saturation_backoff_audit.v1",
  checked: check,
  objective: contract.objective,
  source: {
    total_optimizer_step: contract.source.total_optimizer_step,
    next_window: contract.source.next_window,
    model_hash: contract.source.model_hash,
    model_sha256: sha256(modelBytes),
    optimizer_state_hash: contract.source.optimizer_state_hash,
    optimizer_sha256: sha256(optimizerBytes),
  },
  rows,
  selection: selected === null ? null : {
    minimum_safe_output_backward_shift: selected.output_backward_shift,
    additional_backward_damping_bits:
      selected.output_backward_shift - contract.failure_witness.output_backward_shift,
    final_model_hash: selected.final_model_hash,
    final_optimizer_state_hash: selected.final_optimizer_state_hash,
  },
  gates: {
    base_failure_witness_reproduced: true,
    same_source_model_optimizer_and_batch: true,
    source_artifacts_unchanged: true,
    minimum_safe_shift_identified: selected !== null,
    test_partition_not_read: true,
  },
  authorization: {
    read_only_counterfactual_only: true,
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
  assert(existing === unchecked || existing === rendered, "saturation backoff checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  base_reproduced: result.gates.base_failure_witness_reproduced,
  minimum_safe_output_backward_shift:
    result.selection?.minimum_safe_output_backward_shift ?? null,
  out: outPath,
})}\n`);
