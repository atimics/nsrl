#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let optimizerPath = "";
let tracePath = "";
let planPath = "benchmarks/production-model-v1/scaling-plan.json";
let outPath = "";
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--optimizer") optimizerPath = process.argv[++index];
  else if (arg === "--trace") tracePath = process.argv[++index];
  else if (arg === "--plan") planPath = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else throw new Error(`unknown argument: ${arg}`);
}
if (!optimizerPath || !tracePath) throw new Error("--optimizer and --trace are required");

const [optimizer, trace, plan] = await Promise.all([
  readFile(optimizerPath),
  readFile(tracePath, "utf8").then(JSON.parse),
  readFile(planPath, "utf8").then(JSON.parse),
]);
if (optimizer.subarray(0, 8).toString() !== "NSRLPO2\n" || optimizer.readUInt32LE(8) !== 2) {
  throw new Error("unsupported production optimizer artifact");
}
const residualCount = Number(optimizer.readBigUInt64LE(68));
if (!Number.isSafeInteger(residualCount) || optimizer.length !== 84 + residualCount * 8) {
  throw new Error("production optimizer residual length mismatch");
}
let checksum = 0xcbf29ce484222325n;
for (const byte of optimizer.subarray(0, optimizer.length - 8)) {
  checksum ^= BigInt(byte);
  checksum = BigInt.asUintN(64, checksum * 0x100000001b3n);
}
if (checksum !== optimizer.readBigUInt64LE(optimizer.length - 8)) {
  throw new Error("production optimizer checksum mismatch");
}
const point = plan.points.find(({ id }) => id === trace.profile);
if (!point || point.parameter_count !== trace.parameter_count || residualCount !== point.parameter_count) {
  throw new Error("optimizer, trace, and scaling-plan profile mismatch");
}
const { d_model: dModel, hidden_dim: hiddenDim, layers } = point;
const vocab = plan.tokenizer.vocab_size;
const groupLengths = {
  embeddings: vocab * dModel,
  attention_rms: layers * dModel,
  mlp_rms: layers * dModel,
  final_rms: dModel,
  q: layers * dModel * dModel,
  k: layers * dModel * dModel,
  v: layers * dModel * dModel,
  o: layers * dModel * dModel,
  up: layers * dModel * hiddenDim,
  gate: layers * dModel * hiddenDim,
  down: layers * hiddenDim * dModel,
  output: vocab * dModel,
  bias: vocab,
};
if (Object.values(groupLengths).reduce((sum, length) => sum + length, 0) !== residualCount) {
  throw new Error("production optimizer group geometry mismatch");
}

let offset = 76;
const groups = [];
for (const [group, length] of Object.entries(groupLengths)) {
  const groupOffset = offset;
  let maximum = 0n;
  let nonzero = 0;
  for (let index = 0; index < length; index += 1, offset += 8) {
    const value = optimizer.readBigInt64LE(offset);
    const magnitude = value < 0 ? -value : value;
    if (magnitude !== 0n) nonzero += 1;
    if (magnitude > maximum) maximum = magnitude;
  }
  const currentShift = trace.training.learning_rate_shifts[group];
  const boundaryShift = maximum === 0n ? null : maximum.toString(2).length;
  const threshold = boundaryShift === null ? null : 1n << BigInt(boundaryShift - 1);
  let predictedCrossings = 0;
  if (threshold !== null) {
    for (let index = 0; index < length; index += 1) {
      const value = optimizer.readBigInt64LE(groupOffset + index * 8);
      const magnitude = value < 0 ? -value : value;
      if (magnitude >= threshold) predictedCrossings += 1;
    }
  }
  groups.push({
    group,
    parameters: length,
    gradient_nonzero_count: trace.diagnostics.gradient_nonzero_count[group],
    update_nonzero_count: trace.diagnostics.update_nonzero_count[group],
    current_shift: currentShift,
    residual_nonzero_parameters: nonzero,
    maximum_absolute_residual: maximum.toString(),
    boundary_shift: boundaryShift,
    required_shift_reduction: boundaryShift === null
      ? null
      : Math.max(0, currentShift - boundaryShift),
    predicted_parameter_crossings_at_boundary: predictedCrossings,
  });
}
const candidates = groups
  .filter((row) => !["output", "bias"].includes(row.group)
    && row.gradient_nonzero_count > 0
    && row.update_nonzero_count === 0
    && row.required_shift_reduction > 0)
  .sort((left, right) => left.required_shift_reduction - right.required_shift_reduction
    || right.predicted_parameter_crossings_at_boundary
      - left.predicted_parameter_crossings_at_boundary
    || left.group.localeCompare(right.group));
const recommendation = candidates[0] ? {
  policy: "minimum_single_group_boundary_reduction_v1",
  group: candidates[0].group,
  source_shift: candidates[0].current_shift,
  candidate_shift: candidates[0].boundary_shift,
  shift_reduction: candidates[0].required_shift_reduction,
  predicted_parameter_crossings: candidates[0].predicted_parameter_crossings_at_boundary,
} : null;
const analysis = {
  schema: "nsrl.production_optimizer_residual_analysis.v1",
  profile: point.id,
  parameter_count: point.parameter_count,
  source: {
    optimizer_bytes: optimizer.length,
    optimizer_sha256: createHash("sha256").update(optimizer).digest("hex"),
    optimizer_state_hash: trace.hashes.optimizer_state,
    trace_final_model_hash: trace.hashes.final_model,
    optimizer_step: Number(optimizer.readBigUInt64LE(36)),
  },
  groups,
  recommendation,
  known_limitations: [
    "boundary_is_inferred_from_accumulated_residual_state_not_a_loss_optimizer",
    "candidate_requires_fresh_run_with_liveness_and_heldout_gates",
    "single_group_action_is_a_bounded_policy_bootstrap_not_a_learned_controller",
  ],
};
const rendered = `${JSON.stringify(analysis, null, 2)}\n`;
if (outPath) await writeFile(outPath, rendered);
else process.stdout.write(rendered);
