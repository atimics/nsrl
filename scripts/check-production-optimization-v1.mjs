#!/usr/bin/env node

import { readFile } from "node:fs/promises";

const performancePath = process.argv[2] ?? "benchmarks/production-model-v1/prepilot-performance.json";
const [performance, full, float] = await Promise.all([
  readFile(performancePath, "utf8").then(JSON.parse),
  readFile("benchmarks/production-model-v1/p10m-full-smoke.json", "utf8").then(JSON.parse),
  readFile("benchmarks/production-model-v1/p10m-float-smoke.json", "utf8").then(JSON.parse),
]);
if (performance.schema !== "nsrl.production_preflight_performance.v1"
  || JSON.stringify(performance.results.map((row) => row.context_tokens)) !== JSON.stringify([4, 16, 64, 256])
  || performance.results.some((row) => row.integer_milliseconds <= 0
    || row.float_milliseconds <= 0
    || row.integer_weight_saturation_count !== 0
    || row.integer_gradient_saturation_count !== 0
    || row.integer_optimizer_state_bytes !== 74_541_140
    || row.float_attention_algorithm !== "causal_recurrent_linear")
  || full.training.optimizer !== "integer_residual_sgd"
  || full.training.batch_windows !== 4
  || full.restart.byte_identical_model !== true
  || full.restart.byte_identical_optimizer !== true
  || Object.values(full.diagnostics.saturation_by_group).some((count) => count !== 0)
  || float.training.attention_algorithm !== "causal_recurrent_linear"
  || float.training.batch_windows !== 4) {
  throw new Error("production optimization checkpoint failed");
}
console.log(JSON.stringify({
  schema: "nsrl.production_optimization_check.v1",
  ok: true,
  contexts: performance.results.map(({ context_tokens, integer_milliseconds, float_milliseconds }) => ({
    context_tokens, integer_milliseconds, float_milliseconds,
  })),
  exact_restart: true,
  zero_integer_saturation: true,
}));
