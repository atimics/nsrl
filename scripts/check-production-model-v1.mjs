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
const [
  fullBytes, floatBytes, optimizationBytes, pilotContractBytes, pilotBytes, stabilizationBytes,
] = await Promise.all([
  readFile(plan.full_train_checkpoint.path),
  readFile(plan.float_twin_checkpoint.path),
  readFile(plan.prepilot_optimization_checkpoint.path),
  readFile(plan.pilot_contract.path),
  readFile(plan.pilot_checkpoint.path),
  readFile(plan.integer_stabilization_checkpoint.path),
]);
if (createHash("sha256").update(fullBytes).digest("hex") !== plan.full_train_checkpoint.sha256
  || createHash("sha256").update(floatBytes).digest("hex") !== plan.float_twin_checkpoint.sha256
  || createHash("sha256").update(optimizationBytes).digest("hex") !== plan.prepilot_optimization_checkpoint.sha256
  || createHash("sha256").update(pilotContractBytes).digest("hex") !== plan.pilot_contract.sha256
  || createHash("sha256").update(pilotBytes).digest("hex") !== plan.pilot_checkpoint.sha256
  || createHash("sha256").update(stabilizationBytes).digest("hex")
    !== plan.integer_stabilization_checkpoint.sha256) {
  throw new Error("production training checkpoint hash mismatch");
}
const full = JSON.parse(fullBytes);
const float = JSON.parse(floatBytes);
const optimization = JSON.parse(optimizationBytes);
const pilotContract = JSON.parse(pilotContractBytes);
const pilot = JSON.parse(pilotBytes);
const stabilization = JSON.parse(stabilizationBytes);
if (full.schema !== "nsrl.production_full_train_smoke_checkpoint.v1"
  || float.schema !== "nsrl.production_float_twin_smoke_checkpoint.v1"
  || full.parameter_count !== plan.points[0].parameter_count
  || float.parameter_count !== plan.points[0].parameter_count
  || full.bindings.tokenizer_hash !== plan.tokenizer.artifact_hash_fnv64
  || float.bindings.tokenizer_hash !== plan.tokenizer.artifact_hash_fnv64
  || full.bindings.token_stream_hash !== plan.corpus_checkpoint.train_token_hash_fnv64
  || float.bindings.token_stream_hash !== plan.corpus_checkpoint.train_token_hash_fnv64
  || full.training.context_tokens !== float.training.context_tokens
  || full.training.windows !== float.training.windows
  || full.training.epochs !== float.training.epochs
  || full.training.batch_windows !== float.training.batch_windows
  || full.training.optimizer !== "integer_residual_sgd"
  || float.training.attention_algorithm !== "causal_recurrent_linear"
  || full.restart.byte_identical_model !== true
  || full.restart.byte_identical_optimizer !== true
  || Object.values(full.diagnostics.saturation_by_group).some((count) => count !== 0)
  || optimization.schema !== "nsrl.production_preflight_performance.v1"
  || pilotContract.schema !== "nsrl.production_pilot_contract.v1"
  || pilotContract.profile !== "p10m"
  || pilotContract.schedule.train_windows !== 1024
  || pilotContract.schedule.dev_windows !== 256
  || pilotContract.schedule.context_tokens !== 64
  || pilot.schema !== "nsrl.production_pilot_checkpoint.v1"
  || pilot.contract.profile !== "p10m"
  || pilot.contract.parameter_count !== plan.points[0].parameter_count
  || pilot.contract.bindings.tokenizer_hash !== plan.tokenizer.artifact_hash_fnv64
  || pilot.contract.bindings.train_token_stream_hash !== plan.corpus_checkpoint.train_token_hash_fnv64
  || pilot.contract.bindings.dev_token_stream_hash !== plan.corpus_checkpoint.dev_token_hash_fnv64
  || pilot.integer.training.windows !== pilotContract.schedule.train_windows
  || pilot.integer.dev_final.windows !== pilotContract.schedule.dev_windows
  || pilot.float.evaluation.windows !== pilotContract.schedule.dev_windows
  || pilot.restart.durable_model_sha256 !== pilot.restart.midpoint_model_sha256
  || pilot.restart.durable_optimizer_sha256 !== pilot.restart.midpoint_optimizer_sha256
  || pilot.gates.midpoint_restart_byte_identical !== true
  || pilot.gates.integer_full_gradient_path_sustained !== false
  || pilot.gates.integer_saturation_zero !== false
  || pilot.gates.integer_dev_loss_nonincreasing !== false
  || pilot.gates.float_dev_loss_nonincreasing !== true
  || pilot.gates.integer_float_dev_regression_within_limit !== false
  || pilot.promotion_eligible !== false
  || stabilization.schema !== "nsrl.production_integer_stabilization_preflight.v1"
  || stabilization.parameter_count !== plan.points[0].parameter_count
  || stabilization.source_pilot.sha256 !== plan.pilot_checkpoint.sha256
  || stabilization.bindings.tokenizer_hash !== plan.tokenizer.artifact_hash_fnv64
  || stabilization.bindings.train_token_stream_hash !== plan.corpus_checkpoint.train_token_hash_fnv64
  || stabilization.bindings.dev_token_stream_hash !== plan.corpus_checkpoint.dev_token_hash_fnv64
  || stabilization.initialization.output_init_amplitude !== 1
  || stabilization.initialization.output_forward_shift !== 14
  || stabilization.schedule.output_backward_shift !== 8
  || stabilization.schedule.train_windows !== 256
  || stabilization.schedule.dev_windows !== 256
  || stabilization.evaluation.final.total_millibits
    > stabilization.evaluation.initial.total_millibits
  || stabilization.training.health.gradient_saturation_count !== 0
  || stabilization.training.health.weight_saturation_count !== 0
  || !Object.values(stabilization.gates).every(Boolean)
  || stabilization.preflight_eligible !== true
  || stabilization.next_gate !== "p10m_stabilized_pilot_replay"
  || plan.pilot_contract.runner !== "aws_graviton_parallel_lanes"
  || JSON.stringify(optimization.results.map((row) => row.context_tokens)) !== JSON.stringify([4, 16, 64, 256])
  || full.moved_parameter_groups.length !== 13
  || float.moved_parameter_groups.length !== 13
  || !Object.values(full.gates).every(Boolean)
  || !Object.values(float.gates).every(Boolean)) {
  throw new Error("matched integer/float p10m smoke gate failed");
}
const status = plan.implementation_status;
if (!status.variable_vocab_artifact_ready
  || !status.u32_forward_runtime_ready
  || !status.output_head_smoke_runtime_ready
  || !status.p10m_smoke_completed
  || !status.full_layer_backward_ready
  || !status.full_layer_backward_smoke_completed
  || !status.float_twin_runner_ready
  || !status.float_twin_smoke_completed
  || !status.prepilot_optimization_completed
  || !status.controlled_p10m_pilot_launched
  || !status.controlled_p10m_pilot_completed
  || status.controlled_p10m_pilot_promotion_eligible !== false
  || !status.integer_shift_stabilization_preflight_completed
  || !status.integer_shift_stabilization_preflight_eligible
  || !status.training_started
  || status.next_gate !== "p10m_stabilized_pilot_replay") {
  throw new Error("production implementation status is inconsistent");
}
console.log(JSON.stringify({
  schema: "nsrl.production_model_scaling_plan_check.v1",
  ok: true,
  points: plan.points.map(({ id, parameter_count }) => ({ id, parameter_count })),
  p10m_smoke: { final_mistakes: smoke.training.final_mistakes, zero_saturation: smoke.gates.zero_saturation },
  full_backward: { moved_parameter_groups: full.moved_parameter_groups.length, final_mistakes: full.training.final_mistakes },
  float_twin: { moved_parameter_groups: float.moved_parameter_groups.length, final_mistakes: float.training.final_mistakes },
  pilot: {
    promotion_eligible: pilot.promotion_eligible,
    integer_final_mean_millibits: pilot.integer.dev_final.mean_millibits,
    float_final_mean_millibits: pilot.float.evaluation.final_mean_millibits,
  },
  stabilization: {
    preflight_eligible: stabilization.preflight_eligible,
    dev_total_millibits_delta: stabilization.evaluation.total_millibits_delta,
    moved_parameter_groups: stabilization.training.moved_parameter_groups.length,
  },
  next_gate: plan.implementation_status.next_gate,
}));
