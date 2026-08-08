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
const contract = await json(contractPath);
assert(
  contract.schema === "nsrl.production_atomic_saturation_guard_contract.v1",
  "unexpected atomic saturation guard contract schema",
);

for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.source.optimizer_path, sha256: contract.source.optimizer_sha256 },
  { path: contract.source.trace_path, sha256: contract.source.trace_sha256 },
  {
    path: contract.reference_safe_checkpoint.model_path,
    sha256: contract.reference_safe_checkpoint.model_sha256,
  },
  {
    path: contract.reference_safe_checkpoint.optimizer_path,
    sha256: contract.reference_safe_checkpoint.optimizer_sha256,
  },
  {
    path: contract.reference_safe_checkpoint.trace_path,
    sha256: contract.reference_safe_checkpoint.trace_sha256,
  },
  { path: contract.failure_witness.trace_path, sha256: contract.failure_witness.trace_sha256 },
  {
    path: contract.failure_witness.localization_path,
    sha256: contract.failure_witness.localization_sha256,
  },
  {
    path: contract.failure_witness.trigger_window_audit_path,
    sha256: contract.failure_witness.trigger_window_audit_sha256,
  },
]) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const [sourceTrace, referenceTrace, failureTrace, localization, triggerAudit] = await Promise.all([
  json(contract.source.trace_path),
  json(contract.reference_safe_checkpoint.trace_path),
  json(contract.failure_witness.trace_path),
  json(contract.failure_witness.localization_path),
  json(contract.failure_witness.trigger_window_audit_path),
]);
assert(sourceTrace.hashes.final_model === contract.source.model_hash, "source model mismatch");
assert(
  sourceTrace.hashes.optimizer_state === contract.source.optimizer_state_hash,
  "source optimizer mismatch",
);
assert(
  sourceTrace.training.total_optimizer_step === contract.source.total_optimizer_step
    && sourceTrace.cursor.next_window === contract.source.next_window,
  "source cursor mismatch",
);
assert(
  referenceTrace.hashes.final_model === contract.reference_safe_checkpoint.model_hash
    && referenceTrace.hashes.optimizer_state
      === contract.reference_safe_checkpoint.optimizer_state_hash,
  "reference safe checkpoint hash mismatch",
);
assert(
  referenceTrace.training.total_optimizer_step
    === contract.reference_safe_checkpoint.total_optimizer_step
    && referenceTrace.cursor.next_window === contract.reference_safe_checkpoint.next_window,
  "reference safe checkpoint cursor mismatch",
);
assert(
  failureTrace.training.total_optimizer_step === contract.expected.rejected_batch.attempted_total_optimizer_step,
  "failure witness optimizer step mismatch",
);
assert(same(failureTrace.health, {
  gradient_saturation_count: contract.expected.rejected_batch.gradient_saturation_count,
  residual_saturation_count: contract.expected.rejected_batch.residual_saturation_count,
  weight_saturation_count: contract.expected.rejected_batch.weight_saturation_count,
}), "failure witness saturation signature mismatch");
assert(
  same(failureTrace.movement_l1, contract.expected.rejected_batch.movement_l1),
  "failure witness movement signature mismatch",
);
assert(
  localization.first_unsafe_interval.safe_total_optimizer_step === contract.expected.total_optimizer_step
    && localization.first_unsafe_interval.unsafe_total_optimizer_step
      === contract.expected.rejected_batch.attempted_total_optimizer_step,
  "failure localization boundary mismatch",
);
const triggerRows = triggerAudit.selected.filter(
  (row) => row.optimizer_step === contract.expected.rejected_batch.attempted_total_optimizer_step,
);
assert(
  triggerRows.length === contract.expected.rejected_batch.windows
    && triggerRows[0].selected_index === contract.expected.rejected_batch.start_window
    && triggerRows.every((row, index) => row.batch_offset === index),
  "trigger window provenance mismatch",
);

const files = {
  trace: path.join(runDir, "train.json"),
  model: path.join(runDir, "candidate.nsrlpm"),
  optimizer: path.join(runDir, "optimizer.nsrlpo"),
  development: path.join(runDir, "development.json"),
  saturation: path.join(runDir, "saturation.json"),
};
const [trace, modelBytes, optimizerBytes, development, saturation] = await Promise.all([
  json(files.trace),
  readFile(files.model),
  readFile(files.optimizer),
  json(files.development),
  json(files.saturation),
]);

assert(trace.schema === "nsrl.production_full_train_smoke.v1", "unexpected train trace schema");
assert(
  trace.training.total_optimizer_step === contract.expected.total_optimizer_step
    && trace.training.optimizer_steps === contract.expected.committed_optimizer_steps,
  "committed optimizer step mismatch",
);
assert(
  trace.training.supervised_targets === contract.expected.committed_supervised_targets,
  "committed supervised-target count mismatch",
);
assert(
  same(trace.training.learning_rate_shifts, contract.training.learning_rate_shifts),
  "guard replay schedule mismatch",
);
assert(
  trace.cursor.start_window === contract.source.next_window
    && trace.cursor.next_window === contract.expected.next_window
    && trace.cursor.schedule_complete === false,
  "guard replay cursor mismatch",
);
assert(same(trace.health, contract.expected.committed_health), "committed health is not clean");
assert(
  same(trace.movement_l1, contract.expected.committed_movement_l1),
  "committed movement differs from the expected safe step",
);
assert(
  trace.transaction?.saturation_policy === contract.implementation.policy,
  "saturation rejection policy mismatch",
);
const rejected = trace.transaction?.rejected_batch;
assert(rejected, "saturated batch was not rejected");
for (const field of [
  "attempted_total_optimizer_step",
  "start_epoch",
  "start_window",
  "windows",
  "supervised_targets",
  "gradient_saturation_count",
  "residual_saturation_count",
  "weight_saturation_count",
]) {
  assert(rejected[field] === contract.expected.rejected_batch[field], `rejected ${field} mismatch`);
}
for (const field of ["movement_l1", "update_nonzero_count", "saturation_by_group"]) {
  assert(
    same(rejected[field], contract.expected.rejected_batch[field]),
    `rejected ${field} mismatch`,
  );
}
assert(
  trace.gates.saturated_batch_rejection_enabled === true
    && trace.gates.saturated_batch_rejected_atomically === true,
  "atomic rejection gates did not pass",
);
assert(
  trace.hashes.final_model === contract.reference_safe_checkpoint.model_hash
    && trace.hashes.optimizer_state === contract.reference_safe_checkpoint.optimizer_state_hash,
  "guarded output hashes differ from the reference safe checkpoint",
);

const modelSha256 = sha256(modelBytes);
const optimizerSha256 = sha256(optimizerBytes);
assert(
  modelSha256 === contract.reference_safe_checkpoint.model_sha256,
  "guarded model bytes differ from the reference safe checkpoint",
);
assert(
  optimizerSha256 === contract.reference_safe_checkpoint.optimizer_sha256,
  "guarded optimizer bytes differ from the reference safe checkpoint",
);
assert(
  development.model_hash === trace.hashes.final_model
    && development.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
  "development evaluation binding mismatch",
);
assert(
  development.evaluation.total_nll_millibits
    === contract.expected.development_total_nll_millibits
    && development.evaluation.total_nll_millibits
      - contract.baseline.development_total_nll_millibits
      === contract.expected.development_delta_from_full_v1_millibits
    && development.health.residual_saturation_count
      === contract.expected.development_residual_saturation_count,
  "development health mismatch",
);
assert(
  saturation.bindings.model_hash === trace.hashes.final_model
    && saturation.aggregate.residual_saturation_count
      === contract.expected.manifest_residual_saturation_count,
  "manifest residual saturation mismatch",
);

const result = {
  schema: "nsrl.production_atomic_saturation_guard.v1",
  checked: check,
  objective: contract.objective,
  source: {
    total_optimizer_step: contract.source.total_optimizer_step,
    model_hash: contract.source.model_hash,
    optimizer_state_hash: contract.source.optimizer_state_hash,
  },
  committed: {
    total_optimizer_step: trace.training.total_optimizer_step,
    next_window: trace.cursor.next_window,
    model_hash: trace.hashes.final_model,
    model_sha256: modelSha256,
    optimizer_state_hash: trace.hashes.optimizer_state,
    optimizer_sha256: optimizerSha256,
    health: trace.health,
    movement_l1: trace.movement_l1,
  },
  rejected_batch: rejected,
  evaluation: {
    development_total_nll_millibits: development.evaluation.total_nll_millibits,
    development_delta_from_full_v1_millibits:
      development.evaluation.total_nll_millibits
        - contract.baseline.development_total_nll_millibits,
    development_residual_saturation_count: development.health.residual_saturation_count,
    manifest_residual_saturation_count: saturation.aggregate.residual_saturation_count,
  },
  gates: {
    failure_witness_bound: true,
    trigger_batch_provenance_bound: true,
    saturated_batch_rejected_atomically: true,
    model_bytes_equal_reference_safe_checkpoint: true,
    optimizer_bytes_equal_reference_safe_checkpoint: true,
    committed_training_health_zero: Object.values(trace.health).every((value) => value === 0),
    development_better_than_full_v1:
      development.evaluation.total_nll_millibits
        < contract.baseline.development_total_nll_millibits,
    manifest_residual_saturation_zero: saturation.aggregate.residual_saturation_count === 0,
    test_partition_not_read: true,
  },
  authorization: {
    diagnostic_guard_replay_only: true,
    test_evaluation: false,
    quality_promotion: false,
    open_generation_rerun: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "atomic guard checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  committed_step: result.committed.total_optimizer_step,
  rejected_step: result.rejected_batch.attempted_total_optimizer_step,
  atomic: result.gates.saturated_batch_rejected_atomically,
  out: outPath,
})}\n`);
