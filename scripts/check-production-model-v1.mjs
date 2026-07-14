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
  stabilizedAttemptBytes, stabilizedContractBytes, stabilizedBytes, livenessBytes,
] = await Promise.all([
  readFile(plan.full_train_checkpoint.path),
  readFile(plan.float_twin_checkpoint.path),
  readFile(plan.prepilot_optimization_checkpoint.path),
  readFile(plan.pilot_contract.path),
  readFile(plan.pilot_checkpoint.path),
  readFile(plan.integer_stabilization_checkpoint.path),
  readFile(plan.stabilized_pilot_attempt.path),
  readFile(plan.stabilized_pilot_contract.path),
  readFile(plan.stabilized_pilot_checkpoint.path),
  readFile(plan.training_liveness_checkpoint.path),
]);
if (createHash("sha256").update(fullBytes).digest("hex") !== plan.full_train_checkpoint.sha256
  || createHash("sha256").update(floatBytes).digest("hex") !== plan.float_twin_checkpoint.sha256
  || createHash("sha256").update(optimizationBytes).digest("hex") !== plan.prepilot_optimization_checkpoint.sha256
  || createHash("sha256").update(pilotContractBytes).digest("hex") !== plan.pilot_contract.sha256
  || createHash("sha256").update(pilotBytes).digest("hex") !== plan.pilot_checkpoint.sha256
  || createHash("sha256").update(stabilizationBytes).digest("hex")
    !== plan.integer_stabilization_checkpoint.sha256
  || createHash("sha256").update(stabilizedAttemptBytes).digest("hex")
    !== plan.stabilized_pilot_attempt.sha256
  || createHash("sha256").update(stabilizedContractBytes).digest("hex")
    !== plan.stabilized_pilot_contract.sha256
  || createHash("sha256").update(stabilizedBytes).digest("hex")
    !== plan.stabilized_pilot_checkpoint.sha256
  || createHash("sha256").update(livenessBytes).digest("hex")
    !== plan.training_liveness_checkpoint.sha256) {
  throw new Error("production training checkpoint hash mismatch");
}
const full = JSON.parse(fullBytes);
const float = JSON.parse(floatBytes);
const optimization = JSON.parse(optimizationBytes);
const pilotContract = JSON.parse(pilotContractBytes);
const pilot = JSON.parse(pilotBytes);
const stabilization = JSON.parse(stabilizationBytes);
const stabilizedAttempt = JSON.parse(stabilizedAttemptBytes);
const stabilizedContract = JSON.parse(stabilizedContractBytes);
const stabilized = JSON.parse(stabilizedBytes);
const liveness = JSON.parse(livenessBytes);
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
  || stabilizedAttempt.schema !== "nsrl.production_stabilized_pilot_attempt.v1"
  || stabilizedAttempt.outcome !== "early_stopped"
  || stabilizedAttempt.failed_gate !== "complete_gradient_path"
  || stabilizedAttempt.integer_chunk_0.heldout_delta !== 0
  || stabilizedAttempt.integer_chunk_0.gradient_saturation_count !== 0
  || stabilizedAttempt.integer_chunk_0.weight_saturation_count !== 0
  || stabilizedContract.schema !== "nsrl.production_stabilized_pilot_contract.v2"
  || stabilizedContract.source_preflight.sha256 !== plan.integer_stabilization_checkpoint.sha256
  || stabilizedContract.source_attempt.sha256 !== plan.stabilized_pilot_attempt.sha256
  || stabilizedContract.schedule.train_windows !== 1024
  || stabilizedContract.schedule.integer_learning_rate_shifts.output !== 34
  || stabilizedContract.schedule.source_to_pilot_non_output_shift_delta !== 2
  || stabilizedContract.schedule.source_to_pilot_output_shift_delta !== 0
  || stabilized.schema !== "nsrl.production_stabilized_pilot_checkpoint.v1"
  || stabilized.contract_sha256 !== plan.stabilized_pilot_contract.sha256
  || stabilized.integer.training.windows !== stabilizedContract.schedule.train_windows
  || stabilized.integer.dev_final.total_millibits > stabilized.integer.dev_initial.total_millibits
  || stabilized.integer.health.gradient_saturation_count !== 0
  || stabilized.integer.health.weight_saturation_count !== 0
  || stabilized.integer.moved_parameter_groups.length !== 1
  || stabilized.integer.moved_parameter_groups[0] !== "output"
  || stabilized.float.evaluation.final_mean_millibits
    > stabilized.float.evaluation.initial_mean_millibits
  || stabilized.comparison.integer_regression_vs_float_per_mille > 150
  || stabilized.restart.durable_model_sha256 !== stabilized.restart.midpoint_model_sha256
  || stabilized.restart.durable_optimizer_sha256 !== stabilized.restart.midpoint_optimizer_sha256
  || !Object.values(stabilized.gates).every(Boolean)
  || stabilized.replay_eligible !== true
  || stabilized.next_gate !== "p10m_trunk_unlock_preflight"
  || liveness.schema !== "nsrl.production_training_liveness_audit.v1"
  || liveness.parameter_count !== plan.points[0].parameter_count
  || liveness.interval.windows !== 64
  || liveness.interval.total_windows !== 256
  || liveness.negative_control.event.classification !== "output_unlock_timeout"
  || liveness.negative_control.event.dead !== true
  || liveness.micro_probe.interval_windows !== 16
  || liveness.micro_probe.output_unlock_deadline_intervals !== 4
  || liveness.micro_probe.trunk_activation_deadline_intervals !== 3
  || liveness.micro_probe.observed_output_unlock_interval !== 3
  || liveness.micro_probe.observed_trunk_activation_interval !== 6
  || liveness.micro_probe.trunk_update_observed_by_256_windows !== false
  || !Object.values(liveness.gates).every(Boolean)
  || liveness.audit_eligible !== true
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
  || !status.stabilized_pilot_first_attempt_early_stopped
  || !status.stabilized_pilot_replay_completed
  || !status.stabilized_pilot_replay_eligible
  || !status.training_liveness_audit_completed
  || !status.training_liveness_audit_eligible
  || !status.phase_aware_liveness_ready
  || !status.residual_saturation_observable
  || !status.training_started
  || status.next_gate !== "p10m_trunk_unlock_preflight") {
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
  stabilized_pilot: {
    replay_eligible: stabilized.replay_eligible,
    integer_final_mean_millibits: stabilized.integer.dev_final.mean_millibits,
    float_final_mean_millibits: stabilized.float.evaluation.final_mean_millibits,
    integer_float_regression_per_mille: stabilized.comparison.integer_regression_vs_float_per_mille,
    moved_parameter_groups: stabilized.integer.moved_parameter_groups.length,
  },
  training_liveness: {
    audit_eligible: liveness.audit_eligible,
    output_unlock_interval: liveness.micro_probe.observed_output_unlock_interval,
    trunk_activation_interval: liveness.micro_probe.observed_trunk_activation_interval,
    trunk_update_observed: liveness.micro_probe.trunk_update_observed_by_256_windows,
  },
  next_gate: plan.implementation_status.next_gate,
}));
