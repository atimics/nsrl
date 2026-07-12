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
console.log(JSON.stringify({
  schema: "nsrl.production_model_scaling_plan_check.v1",
  ok: true,
  points: plan.points.map(({ id, parameter_count }) => ({ id, parameter_count })),
  next_gate: plan.implementation_status.next_gate,
}));
