#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const planPath = process.argv[2] ?? "benchmarks/production-model-v1/scaling-plan.json";
const plan = JSON.parse(await readFile(planPath, "utf8"));
if (plan.schema !== "nsrl.production_model_scaling_plan.v1" || plan.contract_id !== "production-model-v1") {
  throw new Error("invalid production model scaling plan");
}
const checkpointBytes = await readFile(plan.corpus_checkpoint.path);
const checkpointHash = createHash("sha256").update(checkpointBytes).digest("hex");
if (checkpointHash !== plan.corpus_checkpoint.sha256) throw new Error("corpus checkpoint hash mismatch");
const checkpoint = JSON.parse(checkpointBytes);
if (checkpoint.tokenizer.actual_vocab_size !== plan.tokenizer.vocab_size
  || checkpoint.tokenizer.artifact_hash_fnv64 !== plan.tokenizer.artifact_hash_fnv64) {
  throw new Error("tokenizer binding mismatch");
}
for (const split of ["train", "dev", "test"]) {
  if (checkpoint.encodings[split].token_hash_fnv64 !== plan.corpus_checkpoint[`${split}_token_hash_fnv64`]) {
    throw new Error(`${split} token binding mismatch`);
  }
}
const expectedBands = [[8_000_000, 12_000_000], [18_000_000, 24_000_000], [26_000_000, 34_000_000]];
for (const [index, point] of plan.points.entries()) {
  if (point.d_model % point.heads !== 0 || point.hidden_dim !== point.d_model * 3) {
    throw new Error(`invalid shape: ${point.id}`);
  }
  const vocab = plan.tokenizer.vocab_size;
  const perLayer = 4 * point.d_model * point.d_model
    + 3 * point.d_model * point.hidden_dim
    + 2 * point.d_model;
  const parameters = 2 * vocab * point.d_model
    + point.layers * perLayer
    + point.d_model
    + vocab;
  if (parameters !== point.parameter_count) throw new Error(`parameter count mismatch: ${point.id}`);
  const [minimum, maximum] = expectedBands[index];
  if (parameters < minimum || parameters > maximum) throw new Error(`parameter band mismatch: ${point.id}`);
  if (JSON.stringify(point.pair) !== JSON.stringify(["integer", "float"])) throw new Error(`missing matched pair: ${point.id}`);
}
if (JSON.stringify(plan.run_order) !== JSON.stringify(plan.points.map((point) => point.id))) {
  throw new Error("run order must follow increasing scale points");
}
const smokeBytes = await readFile(plan.smoke_checkpoint.path);
if (createHash("sha256").update(smokeBytes).digest("hex") !== plan.smoke_checkpoint.sha256) {
  throw new Error("p10m smoke checkpoint hash mismatch");
}
const smoke = JSON.parse(smokeBytes);
if (smoke.schema !== "nsrl.production_model_smoke_checkpoint.v1"
  || smoke.parameter_count !== plan.points[0].parameter_count
  || smoke.bindings.tokenizer_hash !== plan.tokenizer.artifact_hash_fnv64
  || smoke.bindings.token_stream_hash !== plan.corpus_checkpoint.train_token_hash_fnv64
  || smoke.training.final_mistakes !== 0
  || smoke.health.weight_saturation_count !== 0
  || smoke.health.residual_saturation_count !== 0
  || smoke.gates.variable_vocab_artifact !== true
  || smoke.gates.tokenizer_bound_u32_stream !== true
  || smoke.gates.full_layer_backward !== false
  || smoke.gates.float_twin !== false) {
  throw new Error("p10m smoke checkpoint gate failed");
}
const status = plan.implementation_status;
if (!status.variable_vocab_artifact_ready
  || !status.u32_forward_runtime_ready
  || !status.output_head_smoke_runtime_ready
  || !status.p10m_smoke_completed
  || status.full_layer_backward_ready
  || status.float_twin_runner_ready
  || status.training_started
  || status.next_gate !== "full_layer_backward_and_float_twin_runner") {
  throw new Error("production implementation status is inconsistent");
}
console.log(JSON.stringify({
  schema: "nsrl.production_model_scaling_plan_check.v1",
  ok: true,
  points: plan.points.map(({ id, parameter_count }) => ({ id, parameter_count })),
  p10m_smoke: { final_mistakes: smoke.training.final_mistakes, zero_saturation: smoke.gates.zero_saturation },
  next_gate: plan.implementation_status.next_gate,
}));
