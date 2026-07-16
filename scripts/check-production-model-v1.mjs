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
const [normalizedWideContractBytes, normalizedWideBytes] = await Promise.all([
  readFile(plan.normalized_wide_gradient_preflight_contract.path),
  readFile(plan.normalized_wide_gradient_preflight_checkpoint.path),
]);
if (createHash("sha256").update(normalizedWideContractBytes).digest("hex")
    !== plan.normalized_wide_gradient_preflight_contract.sha256
  || createHash("sha256").update(normalizedWideBytes).digest("hex")
    !== plan.normalized_wide_gradient_preflight_checkpoint.sha256) {
  throw new Error("normalized wide-gradient checkpoint hash mismatch");
}
const normalizedWide = JSON.parse(normalizedWideBytes);
if (normalizedWide.schema
    !== "nsrl.production_normalized_wide_gradient_preflight_checkpoint.v1"
  || normalizedWide.parameter_count !== plan.points[0].parameter_count
  || normalizedWide.safety_eligible !== true
  || normalizedWide.integer_precision_bottleneck_resolved !== true
  || normalizedWide.dev_quality_gain !== false
  || normalizedWide.outcome !== "integer_precision_recovered_without_dev_gain"
  || normalizedWide.precision_effect.up_boundary_materialized_updates !== 155
  || normalizedWide.precision_effect.up_boundary_feature_changed_windows !== 84
  || normalizedWide.precision_effect.output_boundary_target_probability_changed_windows !== 3
  || normalizedWide.quality_effect.output_boundary_dev_total_millibits_delta !== 415
  || normalizedWide.replay.selected_model_sha256
    !== normalizedWide.replay.replay_model_sha256
  || normalizedWide.replay.selected_optimizer_sha256
    !== normalizedWide.replay.replay_optimizer_sha256
  || normalizedWide.next_gate !== "p10m_target_aligned_integer_objective_review") {
  throw new Error("normalized wide-gradient checkpoint gate failed");
}
const [
  fullBytes, floatBytes, optimizationBytes, pilotContractBytes, pilotBytes, stabilizationBytes,
  stabilizedAttemptBytes, stabilizedContractBytes, stabilizedBytes, livenessBytes,
  trunkUnlockContractBytes, trunkUnlockBytes, kContractBytes, kStabilizationBytes,
  kvContractBytes, kvPilotBytes, kvReadinessContractBytes, kvReadinessBytes,
  gateContractBytes, gatePreflightBytes, upContractBytes, upUsefulBytes,
  upShift22ContractBytes, upShift22Bytes, upFunctionalContractBytes, upFunctionalBytes,
  upForwardScaleContractBytes, upForwardScaleBytes,
  upForwardTrainingContractBytes, upForwardTrainingBytes,
  targetProbabilityContractBytes, targetProbabilityBytes,
  wideProbabilityContractBytes, wideProbabilityBytes,
  probabilityNormalizationContractBytes, probabilityNormalizationBytes,
  normalizationAttributionContractBytes, normalizationAttributionBytes,
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
  readFile(plan.trunk_unlock_contract.path),
  readFile(plan.trunk_unlock_preflight_checkpoint.path),
  readFile(plan.k_stabilization_contract.path),
  readFile(plan.k_stabilization_preflight_checkpoint.path),
  readFile(plan.kv_boundary_pilot_contract.path),
  readFile(plan.kv_boundary_pilot_checkpoint.path),
  readFile(plan.kv_scaling_readiness_contract.path),
  readFile(plan.kv_scaling_readiness_checkpoint.path),
  readFile(plan.gate_boundary_preflight_contract.path),
  readFile(plan.gate_boundary_preflight_checkpoint.path),
  readFile(plan.up_useful_update_contract.path),
  readFile(plan.up_useful_update_checkpoint.path),
  readFile(plan.up_shift22_breakthrough_contract.path),
  readFile(plan.up_shift22_breakthrough_checkpoint.path),
  readFile(plan.up_functional_comparison_contract.path),
  readFile(plan.up_functional_comparison_checkpoint.path),
  readFile(plan.up_forward_scale_sensitivity_contract.path),
  readFile(plan.up_forward_scale_sensitivity_checkpoint.path),
  readFile(plan.up_forward_scale_training_contract.path),
  readFile(plan.up_forward_scale_training_checkpoint.path),
  readFile(plan.target_probability_resolution_contract.path),
  readFile(plan.target_probability_resolution_checkpoint.path),
  readFile(plan.wide_probability_gradient_preflight_contract.path),
  readFile(plan.wide_probability_gradient_preflight_checkpoint.path),
  readFile(plan.probability_normalization_accuracy_contract.path),
  readFile(plan.probability_normalization_accuracy_checkpoint.path),
  readFile(plan.probability_normalization_signal_attribution_contract.path),
  readFile(plan.probability_normalization_signal_attribution_checkpoint.path),
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
    !== plan.training_liveness_checkpoint.sha256
  || createHash("sha256").update(trunkUnlockContractBytes).digest("hex")
    !== plan.trunk_unlock_contract.sha256
  || createHash("sha256").update(trunkUnlockBytes).digest("hex")
    !== plan.trunk_unlock_preflight_checkpoint.sha256
  || createHash("sha256").update(kContractBytes).digest("hex")
    !== plan.k_stabilization_contract.sha256
  || createHash("sha256").update(kStabilizationBytes).digest("hex")
    !== plan.k_stabilization_preflight_checkpoint.sha256
  || createHash("sha256").update(kvContractBytes).digest("hex")
    !== plan.kv_boundary_pilot_contract.sha256
  || createHash("sha256").update(kvPilotBytes).digest("hex")
    !== plan.kv_boundary_pilot_checkpoint.sha256
  || createHash("sha256").update(kvReadinessContractBytes).digest("hex")
    !== plan.kv_scaling_readiness_contract.sha256
  || createHash("sha256").update(kvReadinessBytes).digest("hex")
    !== plan.kv_scaling_readiness_checkpoint.sha256
  || createHash("sha256").update(gateContractBytes).digest("hex")
    !== plan.gate_boundary_preflight_contract.sha256
  || createHash("sha256").update(gatePreflightBytes).digest("hex")
    !== plan.gate_boundary_preflight_checkpoint.sha256
  || createHash("sha256").update(upContractBytes).digest("hex")
    !== plan.up_useful_update_contract.sha256
  || createHash("sha256").update(upUsefulBytes).digest("hex")
    !== plan.up_useful_update_checkpoint.sha256
  || createHash("sha256").update(upShift22ContractBytes).digest("hex")
    !== plan.up_shift22_breakthrough_contract.sha256
  || createHash("sha256").update(upShift22Bytes).digest("hex")
    !== plan.up_shift22_breakthrough_checkpoint.sha256
  || createHash("sha256").update(upFunctionalContractBytes).digest("hex")
    !== plan.up_functional_comparison_contract.sha256
  || createHash("sha256").update(upFunctionalBytes).digest("hex")
    !== plan.up_functional_comparison_checkpoint.sha256
  || createHash("sha256").update(upForwardScaleContractBytes).digest("hex")
    !== plan.up_forward_scale_sensitivity_contract.sha256
  || createHash("sha256").update(upForwardScaleBytes).digest("hex")
    !== plan.up_forward_scale_sensitivity_checkpoint.sha256
  || createHash("sha256").update(upForwardTrainingContractBytes).digest("hex")
    !== plan.up_forward_scale_training_contract.sha256
  || createHash("sha256").update(upForwardTrainingBytes).digest("hex")
    !== plan.up_forward_scale_training_checkpoint.sha256
  || createHash("sha256").update(targetProbabilityContractBytes).digest("hex")
    !== plan.target_probability_resolution_contract.sha256
  || createHash("sha256").update(targetProbabilityBytes).digest("hex")
    !== plan.target_probability_resolution_checkpoint.sha256
  || createHash("sha256").update(wideProbabilityContractBytes).digest("hex")
    !== plan.wide_probability_gradient_preflight_contract.sha256
  || createHash("sha256").update(wideProbabilityBytes).digest("hex")
    !== plan.wide_probability_gradient_preflight_checkpoint.sha256
  || createHash("sha256").update(probabilityNormalizationContractBytes).digest("hex")
    !== plan.probability_normalization_accuracy_contract.sha256
  || createHash("sha256").update(probabilityNormalizationBytes).digest("hex")
    !== plan.probability_normalization_accuracy_checkpoint.sha256
  || createHash("sha256").update(normalizationAttributionContractBytes).digest("hex")
    !== plan.probability_normalization_signal_attribution_contract.sha256
  || createHash("sha256").update(normalizationAttributionBytes).digest("hex")
    !== plan.probability_normalization_signal_attribution_checkpoint.sha256) {
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
const trunkUnlockContract = JSON.parse(trunkUnlockContractBytes);
const trunkUnlock = JSON.parse(trunkUnlockBytes);
const kContract = JSON.parse(kContractBytes);
const kStabilization = JSON.parse(kStabilizationBytes);
const kvContract = JSON.parse(kvContractBytes);
const kvPilot = JSON.parse(kvPilotBytes);
const kvReadinessContract = JSON.parse(kvReadinessContractBytes);
const kvReadiness = JSON.parse(kvReadinessBytes);
const gateContract = JSON.parse(gateContractBytes);
const gatePreflight = JSON.parse(gatePreflightBytes);
const upContract = JSON.parse(upContractBytes);
const upUseful = JSON.parse(upUsefulBytes);
const upShift22Contract = JSON.parse(upShift22ContractBytes);
const upShift22 = JSON.parse(upShift22Bytes);
const upFunctionalContract = JSON.parse(upFunctionalContractBytes);
const upFunctional = JSON.parse(upFunctionalBytes);
const upForwardScaleContract = JSON.parse(upForwardScaleContractBytes);
const upForwardScale = JSON.parse(upForwardScaleBytes);
const upForwardTrainingContract = JSON.parse(upForwardTrainingContractBytes);
const upForwardTraining = JSON.parse(upForwardTrainingBytes);
const targetProbabilityContract = JSON.parse(targetProbabilityContractBytes);
const targetProbability = JSON.parse(targetProbabilityBytes);
const wideProbabilityContract = JSON.parse(wideProbabilityContractBytes);
const wideProbability = JSON.parse(wideProbabilityBytes);
const probabilityNormalizationContract = JSON.parse(probabilityNormalizationContractBytes);
const probabilityNormalization = JSON.parse(probabilityNormalizationBytes);
const normalizationAttributionContract = JSON.parse(normalizationAttributionContractBytes);
const normalizationAttribution = JSON.parse(normalizationAttributionBytes);
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
  || trunkUnlockContract.schema !== "nsrl.production_trunk_unlock_preflight_contract.v1"
  || trunkUnlockContract.source_liveness.sha256 !== plan.training_liveness_checkpoint.sha256
  || trunkUnlockContract.candidate.group !== "v"
  || trunkUnlockContract.candidate.candidate_shift !== 30
  || trunkUnlockContract.schedule.train_windows !== 256
  || trunkUnlockContract.liveness_policy.require_trunk_update_by_interval !== 3
  || trunkUnlock.schema !== "nsrl.production_trunk_unlock_preflight_checkpoint.v1"
  || trunkUnlock.contract.sha256 !== plan.trunk_unlock_contract.sha256
  || trunkUnlock.source_liveness.sha256 !== plan.training_liveness_checkpoint.sha256
  || trunkUnlock.candidate.group !== "v"
  || trunkUnlock.candidate.candidate_shift !== 30
  || trunkUnlock.intervals[3].movement_l1.v !== 269
  || trunkUnlock.heldout.total_millibits_delta !== -415
  || trunkUnlock.restart.final_model_sha256 !== trunkUnlock.restart.replay_model_sha256
  || trunkUnlock.restart.final_optimizer_sha256 !== trunkUnlock.restart.replay_optimizer_sha256
  || !Object.values(trunkUnlock.gates).every(Boolean)
  || trunkUnlock.preflight_eligible !== true
  || trunkUnlock.next_gate !== "p10m_trunk_unlock_pilot_contract"
  || kContract.schema !== "nsrl.production_k_stabilization_preflight_contract.v1"
  || kContract.source_liveness.sha256 !== plan.training_liveness_checkpoint.sha256
  || kContract.candidate.group !== "k"
  || kContract.candidate.source_shift !== 35
  || kContract.candidate.candidate_shift !== 26
  || kContract.candidate.predicted_parameter_crossings !== 4192
  || kStabilization.schema !== "nsrl.production_k_stabilization_preflight_checkpoint.v1"
  || kStabilization.contract.sha256 !== plan.k_stabilization_contract.sha256
  || kStabilization.candidate.group !== "k"
  || kStabilization.candidate.candidate_shift !== 26
  || kStabilization.intervals[3].update_nonzero_count.k !== 5184
  || kStabilization.intervals[3].movement_l1.k !== 5184
  || kStabilization.heldout.total_millibits_delta !== -830
  || kStabilization.restart.final_model_sha256
    !== kStabilization.restart.replay_model_sha256
  || kStabilization.restart.final_optimizer_sha256
    !== kStabilization.restart.replay_optimizer_sha256
  || !Object.values(kStabilization.gates).every(Boolean)
  || kStabilization.preflight_eligible !== true
  || kStabilization.next_gate !== "p10m_k_stabilized_boundary_pilot_contract"
  || kvContract.schema !== "nsrl.production_kv_boundary_pilot_contract.v1"
  || kvContract.schedule.train_windows !== 1024
  || kvContract.schedule.candidate_learning_rate_shifts.k !== 26
  || kvContract.schedule.candidate_learning_rate_shifts.v !== 30
  || kvPilot.schema !== "nsrl.production_kv_boundary_pilot_checkpoint.v1"
  || kvPilot.contract.sha256 !== plan.kv_boundary_pilot_contract.sha256
  || kvPilot.intervals.length !== 8
  || kvPilot.heldout.total_millibits_delta !== -5209
  || kvPilot.restart.final_model_sha256 !== kvPilot.restart.replay_model_sha256
  || kvPilot.restart.final_optimizer_sha256 !== kvPilot.restart.replay_optimizer_sha256
  || !Object.values(kvPilot.gates).every(Boolean)
  || kvPilot.pilot_eligible !== true
  || kvPilot.next_gate !== "p10m_kv_scaling_readiness_review"
  || kvReadinessContract.schema !== "nsrl.production_kv_scaling_readiness_contract.v1"
  || kvReadinessContract.source_pilot.checkpoint_sha256
    !== plan.kv_boundary_pilot_checkpoint.sha256
  || kvReadinessContract.schedule.train_windows !== 2048
  || kvReadinessContract.schedule.chunks !== 8
  || kvReadinessContract.schedule.midpoint_window !== 1024
  || kvReadinessContract.authorization.paid_cloud_execution !== false
  || kvReadiness.schema !== "nsrl.production_kv_scaling_readiness_checkpoint.v1"
  || kvReadiness.contract.sha256 !== plan.kv_scaling_readiness_contract.sha256
  || kvReadiness.integer.chunks.length !== 8
  || kvReadiness.float.chunks.length !== 8
  || kvReadiness.integer.heldout.total_millibits_delta !== -5209
  || kvReadiness.float.heldout.delta_millibits !== -98
  || kvReadiness.comparison.integer_regression_per_mille !== 12
  || kvReadiness.integer.restart.final_model_sha256
    !== kvReadiness.integer.restart.replay_model_sha256
  || kvReadiness.integer.restart.final_optimizer_sha256
    !== kvReadiness.integer.restart.replay_optimizer_sha256
  || kvReadiness.float.restart.byte_identical_tensors !== true
  || kvReadiness.integer.residual_analysis.recommendation.group !== "gate"
  || kvReadiness.integer.residual_analysis.recommendation.candidate_shift !== 23
  || !Object.values(kvReadiness.gates).every(Boolean)
  || kvReadiness.readiness_eligible !== true
  || kvReadiness.paid_scale_authorized !== false
  || kvReadiness.next_gate !== "p10m_gate_boundary_preflight_contract"
  || gateContract.schema !== "nsrl.production_gate_boundary_preflight_contract.v1"
  || gateContract.source_readiness.checkpoint_sha256
    !== plan.kv_scaling_readiness_checkpoint.sha256
  || gateContract.schedule.train_windows !== 2048
  || gateContract.schedule.midpoint_window !== 1024
  || gateContract.candidate.group !== "gate"
  || gateContract.candidate.candidate_shift !== 23
  || gateContract.authorization.paid_cloud_execution !== false
  || gatePreflight.schema !== "nsrl.production_gate_boundary_preflight_checkpoint.v1"
  || gatePreflight.contract.sha256 !== plan.gate_boundary_preflight_contract.sha256
  || gatePreflight.intervals.length !== 8
  || gatePreflight.candidate.actual_update_count !== 26
  || gatePreflight.candidate.first_movement_window !== 768
  || gatePreflight.heldout.total_millibits_delta !== -5209
  || gatePreflight.restart.final_model_sha256
    !== gatePreflight.restart.replay_model_sha256
  || gatePreflight.restart.final_optimizer_sha256
    !== gatePreflight.restart.replay_optimizer_sha256
  || gatePreflight.final_residual_analysis.recommendation.group !== "up"
  || gatePreflight.final_residual_analysis.recommendation.candidate_shift !== 23
  || gatePreflight.final_residual_analysis.recommendation.predicted_parameter_crossings !== 6
  || !Object.values(gatePreflight.gates).every(Boolean)
  || gatePreflight.preflight_eligible !== true
  || gatePreflight.paid_scale_authorized !== false
  || gatePreflight.next_gate !== "p10m_up_boundary_preflight_contract"
  || upContract.schema !== "nsrl.production_up_useful_update_contract.v1"
  || upContract.source_gate_preflight.checkpoint_sha256
    !== plan.gate_boundary_preflight_checkpoint.sha256
  || upContract.candidate.group !== "up"
  || upContract.candidate.candidate_shift !== 23
  || upContract.quality_gates.minimum_mean_millibit_improvement_vs_source !== 1
  || upContract.authorization.paid_cloud_execution !== false
  || upUseful.schema !== "nsrl.production_up_useful_update_checkpoint.v1"
  || upUseful.contract.sha256 !== plan.up_useful_update_contract.sha256
  || upUseful.intervals.length !== 8
  || upUseful.candidate.actual_update_count !== 26
  || upUseful.candidate.first_movement_window !== 768
  || upUseful.heldout.total_millibits_delta_vs_source !== 0
  || upUseful.heldout.mean_millibits_delta_vs_source !== 0
  || upUseful.restart.final_model_sha256 !== upUseful.restart.replay_model_sha256
  || upUseful.restart.final_optimizer_sha256 !== upUseful.restart.replay_optimizer_sha256
  || !Object.values(upUseful.safety_gates).every(Boolean)
  || Object.values(upUseful.quality_gates).every(Boolean)
  || upUseful.safety_eligible !== true
  || upUseful.quality_breakthrough !== false
  || upUseful.promotion_eligible !== false
  || upUseful.outcome !== "safe_reachability_without_source_gain"
  || upUseful.paid_scale_authorized !== false
  || upUseful.next_gate !== "p10m_up_useful_update_shift_sweep_contract"
  || upShift22Contract.schema !== "nsrl.production_up_shift_breakthrough_contract.v1"
  || upShift22Contract.source_up_useful_update.checkpoint_sha256
    !== plan.up_useful_update_checkpoint.sha256
  || upShift22Contract.candidate.candidate_shift !== 22
  || upShift22Contract.candidate.predicted_parameter_crossings !== 22983
  || upShift22Contract.selection_policy.test_split_opened_after_selection !== true
  || upShift22Contract.authorization.paid_cloud_execution !== false
  || upShift22.schema !== "nsrl.production_up_shift_breakthrough_checkpoint.v1"
  || upShift22.contract.sha256 !== plan.up_shift22_breakthrough_contract.sha256
  || upShift22.intervals.length !== 8
  || upShift22.candidate.actual_update_count !== 101543
  || upShift22.candidate.first_movement_window !== 512
  || upShift22.selection.selected_interval !== 3
  || upShift22.evaluation.selected_dev_total_millibits_delta_vs_source !== 0
  || upShift22.evaluation.selected_test_total_millibits_delta_vs_source !== 1245
  || upShift22.evaluation.selected_test_mean_millibits_delta_vs_source !== 5
  || upShift22.replay.selected_model_sha256 !== upShift22.replay.replay_model_sha256
  || upShift22.replay.selected_optimizer_sha256
    !== upShift22.replay.replay_optimizer_sha256
  || !Object.values(upShift22.safety_gates).every(Boolean)
  || upShift22.safety_eligible !== true
  || upShift22.discovery_passed !== false
  || upShift22.confirmation_passed !== false
  || upShift22.promotion_eligible !== false
  || upShift22.outcome !== "no_dev_discovery"
  || upShift22.paid_scale_authorized !== false
  || upShift22.next_gate !== "p10m_integer_objective_quality_review"
  || upFunctionalContract.schema !== "nsrl.production_functional_comparison_contract.v1"
  || upFunctionalContract.source_model.up_shift !== 23
  || upFunctionalContract.candidate_model.up_shift !== 22
  || upFunctionalContract.evaluation.windows !== 256
  || upFunctionalContract.authorization.paid_cloud_execution !== false
  || upFunctional.schema !== "nsrl.production_functional_comparison_checkpoint.v1"
  || upFunctional.contract.sha256 !== plan.up_functional_comparison_contract.sha256
  || upFunctional.classification !== "weight_updates_masked_before_final_features"
  || upFunctional.comparison.quality.total_millibits_delta !== 0
  || upFunctional.comparison.quality.equal_loss_windows !== 256
  || upFunctional.comparison.functional_delta.feature_changed_windows !== 0
  || upFunctional.comparison.functional_delta.logits_changed_windows !== 0
  || upFunctional.comparison.functional_delta.probabilities_changed_windows !== 0
  || !Object.values(upFunctional.gates).every(Boolean)
  || upFunctional.diagnostic_eligible !== true
  || upFunctional.promotion_eligible !== false
  || upFunctional.paid_scale_authorized !== false
  || upFunctional.next_gate !== "p10m_up_forward_scale_sensitivity_contract"
  || upForwardScaleContract.schema
    !== "nsrl.production_forward_scale_sensitivity_contract.v1"
  || upForwardScaleContract.source_diagnostic.checkpoint_sha256
    !== plan.up_functional_comparison_checkpoint.sha256
  || JSON.stringify(upForwardScaleContract.selection_policy.row_order)
    !== JSON.stringify([10, 9, 8, 7])
  || upForwardScaleContract.selection_policy.no_test_split_access !== true
  || upForwardScaleContract.authorization.paid_cloud_execution !== false
  || upForwardScale.schema !== "nsrl.production_forward_scale_sensitivity_checkpoint.v1"
  || upForwardScale.contract.sha256 !== plan.up_forward_scale_sensitivity_contract.sha256
  || upForwardScale.rows.length !== 4
  || upForwardScale.rows[0].functional_delta.feature_changed_windows !== 0
  || upForwardScale.rows[1].functional_delta.feature_changed_windows !== 0
  || upForwardScale.rows[2].functional_delta.feature_changed_windows !== 0
  || upForwardScale.rows[3].up_forward_shift !== 7
  || upForwardScale.rows[3].functional_delta.feature_changed_windows !== 250
  || upForwardScale.rows[3].functional_delta.logits_changed_windows !== 250
  || upForwardScale.rows[3].functional_delta.probabilities_changed_windows !== 124
  || upForwardScale.rows[3].functional_delta.target_probability_changed_windows !== 0
  || upForwardScale.rows[3].health.source_residual_saturation_count !== 0
  || upForwardScale.rows[3].health.candidate_residual_saturation_count !== 0
  || upForwardScale.selection.selected_up_forward_shift !== 7
  || !Object.values(upForwardScale.gates).every(Boolean)
  || upForwardScale.diagnostic_eligible !== true
  || upForwardScale.outcome !== "safe_functional_boundary_found"
  || upForwardScale.promotion_eligible !== false
  || upForwardScale.paid_scale_authorized !== false
  || upForwardScale.next_gate !== "p10m_up_forward_scale_training_contract"
  || upForwardTrainingContract.schema
    !== "nsrl.production_forward_scale_training_contract.v1"
  || upForwardTrainingContract.source_sensitivity.checkpoint_sha256
    !== plan.up_forward_scale_sensitivity_checkpoint.sha256
  || upForwardTrainingContract.initialization.up_forward_shift !== 7
  || upForwardTrainingContract.schedule.learning_rate_shifts.up !== 22
  || upForwardTrainingContract.schedule.forward_shifts.up !== 7
  || upForwardTrainingContract.selection_policy.test_split_access !== false
  || upForwardTrainingContract.authorization.paid_cloud_execution !== false
  || upForwardTraining.schema !== "nsrl.production_forward_scale_training_checkpoint.v1"
  || upForwardTraining.contract.sha256 !== plan.up_forward_scale_training_contract.sha256
  || upForwardTraining.intervals.length !== 4
  || upForwardTraining.training_candidate.actual_up_update_count !== 50568
  || upForwardTraining.training_candidate.first_up_movement_window !== 512
  || upForwardTraining.selection.selected_interval !== 3
  || upForwardTraining.selection.selected_window !== 1024
  || upForwardTraining.evaluation.selected_dev_total_millibits_delta_vs_source !== 0
  || upForwardTraining.evaluation.selected_dev_mean_millibits_delta_vs_source !== 0
  || upForwardTraining.replay.selected_model_sha256
    !== upForwardTraining.replay.replay_model_sha256
  || upForwardTraining.replay.selected_optimizer_sha256
    !== upForwardTraining.replay.replay_optimizer_sha256
  || !Object.values(upForwardTraining.safety_gates).every(Boolean)
  || upForwardTraining.safety_eligible !== true
  || upForwardTraining.quality_discovery !== false
  || upForwardTraining.promotion_eligible !== false
  || upForwardTraining.outcome !== "safe_functional_training_without_dev_gain"
  || upForwardTraining.paid_scale_authorized !== false
  || upForwardTraining.next_gate !== "p10m_target_probability_resolution_review"
  || targetProbabilityContract.schema
    !== "nsrl.production_target_probability_resolution_contract.v1"
  || targetProbabilityContract.source_training.checkpoint_sha256
    !== plan.up_forward_scale_training_checkpoint.sha256
  || JSON.stringify(targetProbabilityContract.evaluation.probability_fractional_bits)
    !== JSON.stringify([15, 19, 23, 27, 31])
  || targetProbabilityContract.selection_policy.no_test_split_access !== true
  || targetProbabilityContract.authorization.paid_cloud_execution !== false
  || targetProbability.schema
    !== "nsrl.production_target_probability_resolution_checkpoint.v1"
  || targetProbability.contract.sha256
    !== plan.target_probability_resolution_contract.sha256
  || targetProbability.precision_rows.length !== 5
  || targetProbability.precision_rows[0].fractional_bits !== 15
  || targetProbability.precision_rows[0].uniform_probability_floor !== 4
  || targetProbability.precision_rows[0].source_target.unique_values !== 3
  || targetProbability.precision_rows[0].delta.target_probability_changed_windows !== 0
  || targetProbability.precision_rows[1].fractional_bits !== 19
  || targetProbability.precision_rows[1].delta.target_probability_changed_windows !== 1
  || targetProbability.precision_rows[2].fractional_bits !== 23
  || targetProbability.precision_rows[2].delta.target_probability_changed_windows !== 13
  || targetProbability.precision_rows[4].fractional_bits !== 31
  || targetProbability.precision_rows[4].delta.target_probability_changed_windows !== 13
  || targetProbability.selection.selected_fractional_bits !== 19
  || targetProbability.compatibility.q15_requantization_exact !== true
  || !Object.values(targetProbability.gates).every(Boolean)
  || targetProbability.diagnostic_eligible !== true
  || targetProbability.outcome !== "wider_precision_recovers_target_signal"
  || targetProbability.promotion_eligible !== false
  || targetProbability.paid_scale_authorized !== false
  || targetProbability.next_gate !== "p10m_wide_probability_gradient_preflight_contract"
  || wideProbabilityContract.schema
    !== "nsrl.production_wide_probability_gradient_preflight_contract.v1"
  || wideProbabilityContract.source_resolution.checkpoint_sha256
    !== plan.target_probability_resolution_checkpoint.sha256
  || JSON.stringify(
    wideProbabilityContract.schedule.candidate_probability_gradient_fractional_bits,
  ) !== JSON.stringify([19, 23])
  || wideProbabilityContract.selection_policy.test_split_access !== false
  || wideProbabilityContract.authorization.paid_cloud_execution !== false
  || wideProbability.schema
    !== "nsrl.production_wide_probability_gradient_preflight_checkpoint.v1"
  || wideProbability.contract.sha256
    !== plan.wide_probability_gradient_preflight_contract.sha256
  || wideProbability.candidates.length !== 2
  || wideProbability.candidates[0].probability_gradient_fractional_bits !== 19
  || wideProbability.candidates[1].probability_gradient_fractional_bits !== 23
  || wideProbability.candidates.some((row) =>
    row.dev.total_millibits !== wideProbability.q15_control.window_256_dev_total_millibits)
  || wideProbability.precision_effect.classification
    !== "wide_probability_information_residual_only_at_256_windows"
  || wideProbability.precision_effect.all_candidate_models_byte_identical_to_q15_control
    !== true
  || wideProbability.precision_effect.all_candidate_optimizers_differ_from_q15_control
    !== true
  || wideProbability.selection.selected_fractional_bits !== 19
  || wideProbability.selection.selected_dev_total_millibits_delta_vs_q15_control !== 0
  || wideProbability.replay.selected_model_sha256
    !== wideProbability.replay.replay_model_sha256
  || wideProbability.replay.selected_optimizer_sha256
    !== wideProbability.replay.replay_optimizer_sha256
  || !Object.values(wideProbability.safety_gates).every(Boolean)
  || wideProbability.safety_eligible !== true
  || wideProbability.quality_gain !== false
  || wideProbability.promotion_eligible !== false
  || wideProbability.outcome !== "wide_precision_no_preflight_gain"
  || wideProbability.paid_scale_authorized !== false
  || wideProbability.next_gate !== "p10m_probability_normalization_accuracy_review"
  || probabilityNormalizationContract.schema
    !== "nsrl.production_probability_normalization_accuracy_contract.v1"
  || probabilityNormalizationContract.source_preflight.checkpoint_sha256
    !== plan.wide_probability_gradient_preflight_checkpoint.sha256
  || JSON.stringify(
    probabilityNormalizationContract.evaluation.normalizations.map(({ id }) => id),
  ) !== JSON.stringify([
    "legacy_q31_lut", "q47_lut", "q47_newton1", "q47_exact_division",
  ])
  || probabilityNormalizationContract.selection_policy.maximum_mass_error_ppm !== 1000
  || probabilityNormalizationContract.selection_policy.no_test_split_access !== true
  || probabilityNormalizationContract.authorization.paid_cloud_execution !== false
  || probabilityNormalization.schema
    !== "nsrl.production_probability_normalization_accuracy_checkpoint.v1"
  || probabilityNormalization.contract.sha256
    !== plan.probability_normalization_accuracy_contract.sha256
  || probabilityNormalization.normalization_rows.length !== 4
  || probabilityNormalization.normalization_rows[0].normalization !== "legacy_q31_lut"
  || probabilityNormalization.normalization_rows[0].mass.source_error_max_ppm !== 98925
  || probabilityNormalization.normalization_rows[0].delta
    .target_probability_changed_windows !== 13
  || probabilityNormalization.normalization_rows[2].normalization !== "q47_newton1"
  || probabilityNormalization.normalization_rows[2].mass.source_error_max_ppm !== 98
  || probabilityNormalization.normalization_rows[2].mass.candidate_error_max_ppm !== 83
  || probabilityNormalization.normalization_rows[2].delta
    .target_probability_changed_windows !== 5
  || probabilityNormalization.normalization_rows[3].normalization !== "q47_exact_division"
  || probabilityNormalization.normalization_rows[3].mass.source_error_max_ppm !== 73
  || probabilityNormalization.normalization_rows[3].mass.candidate_error_max_ppm !== 74
  || probabilityNormalization.normalization_rows[3].delta
    .target_probability_changed_windows !== 4
  || probabilityNormalization.accuracy_effect.best_nondivision_normalization !== "q47_newton1"
  || probabilityNormalization.accuracy_effect.best_nondivision_meets_mass_threshold !== true
  || probabilityNormalization.accuracy_effect.classification
    !== "mass_accuracy_recovered_but_legacy_target_signal_requires_attribution"
  || !Object.values(probabilityNormalization.gates).every(Boolean)
  || probabilityNormalization.diagnostic_eligible !== true
  || probabilityNormalization.selection.selected_normalization !== null
  || probabilityNormalization.outcome !== "normalization_accuracy_not_recovered"
  || probabilityNormalization.promotion_eligible !== false
  || probabilityNormalization.paid_scale_authorized !== false
  || probabilityNormalization.next_gate
    !== "p10m_probability_normalization_signal_attribution_review"
  || normalizationAttributionContract.schema
    !== "nsrl.production_probability_normalization_signal_attribution_contract.v1"
  || normalizationAttributionContract.source_normalization_review.checkpoint_sha256
    !== plan.probability_normalization_accuracy_checkpoint.sha256
  || normalizationAttributionContract.selection_policy.candidate_normalization
    !== "q47_newton1"
  || normalizationAttributionContract.selection_policy.accuracy_ceiling
    !== "q47_exact_division"
  || normalizationAttributionContract.selection_policy
    .maximum_candidate_probability_error_q23_units !== 1
  || normalizationAttributionContract.selection_policy.no_test_split_access !== true
  || normalizationAttributionContract.authorization.paid_cloud_execution !== false
  || normalizationAttribution.schema
    !== "nsrl.production_probability_normalization_signal_attribution_checkpoint.v1"
  || normalizationAttribution.contract.sha256
    !== plan.probability_normalization_signal_attribution_contract.sha256
  || normalizationAttribution.methods.length !== 3
  || JSON.stringify(normalizationAttribution.methods[2].target_changed_window_indices)
    !== JSON.stringify([6, 79, 173, 193])
  || JSON.stringify(normalizationAttribution.set_attribution.newton_exact_overlap_indices)
    !== JSON.stringify([6, 79, 173, 193])
  || JSON.stringify(normalizationAttribution.set_attribution.newton_only_indices)
    !== JSON.stringify([174])
  || normalizationAttribution.set_attribution.exact_missing_from_newton_indices.length !== 0
  || normalizationAttribution.methods[1].source_error_vs_exact.probability_error_max !== 1
  || normalizationAttribution.methods[1].candidate_error_vs_exact.probability_error_max !== 1
  || normalizationAttribution.methods[1].source_error_vs_exact.target_error_max !== 1
  || normalizationAttribution.methods[1].candidate_error_vs_exact.target_error_max !== 1
  || normalizationAttribution.legacy_only_attribution.windows !== 9
  || normalizationAttribution.legacy_only_attribution.target_logit_changed_windows !== 0
  || normalizationAttribution.legacy_only_attribution.target_weight_changed_windows !== 9
  || normalizationAttribution.legacy_only_attribution.normalization_sum_changed_windows !== 9
  || normalizationAttribution.legacy_only_attribution.all_exact_q23_deltas_zero !== true
  || !Object.values(normalizationAttribution.gates).every(Boolean)
  || !Object.values(normalizationAttribution.candidate_gates).every(Boolean)
  || normalizationAttribution.diagnostic_eligible !== true
  || normalizationAttribution.candidate_eligible !== true
  || normalizationAttribution.selection.selected_normalization !== "q47_newton1"
  || normalizationAttribution.outcome !== "newton_normalization_attributed_and_ready"
  || normalizationAttribution.promotion_eligible !== false
  || normalizationAttribution.paid_scale_authorized !== false
  || normalizationAttribution.next_gate !== "p10m_normalized_wide_gradient_preflight_contract"
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
  || !status.residual_boundary_policy_ready
  || !status.trunk_unlock_preflight_completed
  || !status.trunk_unlock_preflight_eligible
  || !status.k_stabilization_preflight_completed
  || !status.k_stabilization_preflight_eligible
  || !status.kv_boundary_pilot_completed
  || !status.kv_boundary_pilot_eligible
  || !status.kv_scaling_readiness_completed
  || !status.kv_scaling_readiness_eligible
  || !status.matched_integer_float_2048_ready
  || status.paid_scale_authorized !== false
  || !status.gate_boundary_preflight_completed
  || !status.gate_boundary_preflight_eligible
  || !status.up_useful_update_completed
  || !status.up_useful_update_safety_eligible
  || status.up_useful_update_quality_breakthrough !== false
  || !status.up_shift22_breakthrough_completed
  || !status.up_shift22_breakthrough_safety_eligible
  || status.up_shift22_dev_discovery_passed !== false
  || status.up_shift22_test_confirmation_passed !== false
  || !status.up_functional_comparison_completed
  || !status.up_functional_comparison_eligible
  || !status.up_forward_masking_localized
  || !status.up_forward_scale_sensitivity_completed
  || !status.up_forward_scale_sensitivity_eligible
  || !status.up_safe_functional_boundary_found
  || status.up_selected_forward_shift !== 7
  || !status.up_forward_scale_training_completed
  || !status.up_forward_scale_training_safety_eligible
  || status.up_forward_scale_training_quality_discovery !== false
  || !status.target_probability_resolution_review_completed
  || !status.target_probability_resolution_review_eligible
  || status.target_probability_minimum_detectable_fractional_bits !== 19
  || status.target_probability_full_q31_coverage_fractional_bits !== 23
  || !status.wide_probability_gradient_preflight_completed
  || !status.wide_probability_gradient_preflight_safety_eligible
  || status.wide_probability_gradient_preflight_quality_gain !== false
  || !status.wide_probability_information_residual_only_at_256_windows
  || !status.probability_normalization_accuracy_review_completed
  || !status.probability_normalization_accuracy_review_eligible
  || !status.probability_normalization_mass_accuracy_recovered
  || status.probability_normalization_candidate_selected !== false
  || !status.probability_normalization_legacy_signal_requires_attribution
  || !status.probability_normalization_signal_attribution_completed
  || !status.probability_normalization_signal_attribution_eligible
  || status.probability_normalization_selected_method !== "q47_newton1"
  || !status.probability_normalization_exact_signal_preserved
  || status.probability_normalization_max_exact_error_q23_units !== 1
  || status.probability_normalization_legacy_only_windows_attributed !== 9
  || !status.normalized_wide_gradient_preflight_ready
  || !status.normalized_wide_gradient_preflight_completed
  || !status.normalized_wide_gradient_preflight_safety_eligible
  || !status.normalized_wide_gradient_exact_replay
  || !status.normalized_integer_precision_bottleneck_resolved
  || status.normalized_up_boundary_materialized_updates !== 155
  || status.normalized_up_boundary_feature_changed_windows !== 84
  || status.normalized_output_boundary_target_probability_changed_windows !== 3
  || status.normalized_wide_gradient_dev_quality_gain !== false
  || !status.exact_reachable_update_gate_ready
  || !status.training_started
  || status.next_gate !== "p10m_target_aligned_integer_objective_review") {
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
  trunk_unlock: {
    preflight_eligible: trunkUnlock.preflight_eligible,
    group: trunkUnlock.candidate.group,
    candidate_shift: trunkUnlock.candidate.candidate_shift,
    movement_l1: trunkUnlock.intervals[3].movement_l1.v,
    heldout_total_millibits_delta: trunkUnlock.heldout.total_millibits_delta,
    exact_restart: trunkUnlock.gates.midpoint_restart_model_byte_identical
      && trunkUnlock.gates.midpoint_restart_optimizer_byte_identical,
  },
  k_stabilization: {
    preflight_eligible: kStabilization.preflight_eligible,
    candidate_shift: kStabilization.candidate.candidate_shift,
    update_nonzero_count: kStabilization.intervals[3].update_nonzero_count.k,
    heldout_total_millibits_delta: kStabilization.heldout.total_millibits_delta,
    exact_restart: kStabilization.gates.midpoint_restart_model_byte_identical
      && kStabilization.gates.midpoint_restart_optimizer_byte_identical,
  },
  kv_boundary_pilot: {
    pilot_eligible: kvPilot.pilot_eligible,
    moved_parameter_groups: kvPilot.intervals[7].moved_parameter_groups,
    heldout_total_millibits_delta: kvPilot.heldout.total_millibits_delta,
    exact_restart: kvPilot.gates.midpoint_restart_model_byte_identical
      && kvPilot.gates.midpoint_restart_optimizer_byte_identical,
  },
  kv_scaling_readiness: {
    readiness_eligible: kvReadiness.readiness_eligible,
    integer_final_mean_millibits: kvReadiness.comparison.integer_final_mean_millibits,
    float_final_mean_millibits: kvReadiness.comparison.float_final_mean_millibits,
    integer_regression_per_mille: kvReadiness.comparison.integer_regression_per_mille,
    residual_candidate: kvReadiness.integer.residual_analysis.recommendation.group,
    exact_integer_restart:
      kvReadiness.integer.restart.final_model_sha256
        === kvReadiness.integer.restart.replay_model_sha256
      && kvReadiness.integer.restart.final_optimizer_sha256
        === kvReadiness.integer.restart.replay_optimizer_sha256,
    exact_float_restart: kvReadiness.float.restart.byte_identical_tensors,
  },
  gate_boundary_preflight: {
    preflight_eligible: gatePreflight.preflight_eligible,
    actual_update_count: gatePreflight.candidate.actual_update_count,
    first_movement_window: gatePreflight.candidate.first_movement_window,
    heldout_total_millibits_delta: gatePreflight.heldout.total_millibits_delta,
    residual_candidate: gatePreflight.final_residual_analysis.recommendation.group,
    exact_restart:
      gatePreflight.restart.final_model_sha256
        === gatePreflight.restart.replay_model_sha256
      && gatePreflight.restart.final_optimizer_sha256
        === gatePreflight.restart.replay_optimizer_sha256,
  },
  up_useful_update: {
    outcome: upUseful.outcome,
    safety_eligible: upUseful.safety_eligible,
    quality_breakthrough: upUseful.quality_breakthrough,
    actual_update_count: upUseful.candidate.actual_update_count,
    heldout_total_millibits_delta_vs_source:
      upUseful.heldout.total_millibits_delta_vs_source,
    exact_restart:
      upUseful.restart.final_model_sha256 === upUseful.restart.replay_model_sha256
      && upUseful.restart.final_optimizer_sha256
        === upUseful.restart.replay_optimizer_sha256,
  },
  up_shift22_breakthrough: {
    outcome: upShift22.outcome,
    safety_eligible: upShift22.safety_eligible,
    actual_update_count: upShift22.candidate.actual_update_count,
    selected_window: upShift22.selection.selected_window,
    dev_total_millibits_delta_vs_source:
      upShift22.evaluation.selected_dev_total_millibits_delta_vs_source,
    test_total_millibits_delta_vs_source:
      upShift22.evaluation.selected_test_total_millibits_delta_vs_source,
    exact_replay:
      upShift22.replay.selected_model_sha256 === upShift22.replay.replay_model_sha256
      && upShift22.replay.selected_optimizer_sha256
        === upShift22.replay.replay_optimizer_sha256,
  },
  up_functional_comparison: {
    diagnostic_eligible: upFunctional.diagnostic_eligible,
    classification: upFunctional.classification,
    feature_changed_windows:
      upFunctional.comparison.functional_delta.feature_changed_windows,
    logits_changed_windows: upFunctional.comparison.functional_delta.logits_changed_windows,
    probabilities_changed_windows:
      upFunctional.comparison.functional_delta.probabilities_changed_windows,
  },
  up_forward_scale_sensitivity: {
    diagnostic_eligible: upForwardScale.diagnostic_eligible,
    outcome: upForwardScale.outcome,
    selected_up_forward_shift: upForwardScale.selection.selected_up_forward_shift,
    feature_changed_windows:
      upForwardScale.selection.selected_row.functional_delta.feature_changed_windows,
    logits_changed_windows:
      upForwardScale.selection.selected_row.functional_delta.logits_changed_windows,
    probabilities_changed_windows:
      upForwardScale.selection.selected_row.functional_delta.probabilities_changed_windows,
    target_probability_changed_windows:
      upForwardScale.selection.selected_row.functional_delta
        .target_probability_changed_windows,
  },
  up_forward_scale_training: {
    outcome: upForwardTraining.outcome,
    safety_eligible: upForwardTraining.safety_eligible,
    quality_discovery: upForwardTraining.quality_discovery,
    actual_up_update_count: upForwardTraining.training_candidate.actual_up_update_count,
    selected_window: upForwardTraining.selection.selected_window,
    dev_total_millibits_delta_vs_source:
      upForwardTraining.evaluation.selected_dev_total_millibits_delta_vs_source,
    exact_replay:
      upForwardTraining.replay.selected_model_sha256
        === upForwardTraining.replay.replay_model_sha256
      && upForwardTraining.replay.selected_optimizer_sha256
        === upForwardTraining.replay.replay_optimizer_sha256,
  },
  target_probability_resolution: {
    outcome: targetProbability.outcome,
    diagnostic_eligible: targetProbability.diagnostic_eligible,
    selected_fractional_bits: targetProbability.selection.selected_fractional_bits,
    q15_target_probability_changed_windows:
      targetProbability.precision_rows[0].delta.target_probability_changed_windows,
    q23_target_probability_changed_windows:
      targetProbability.precision_rows[2].delta.target_probability_changed_windows,
    q31_target_probability_changed_windows:
      targetProbability.precision_rows[4].delta.target_probability_changed_windows,
  },
  wide_probability_gradient_preflight: {
    outcome: wideProbability.outcome,
    safety_eligible: wideProbability.safety_eligible,
    quality_gain: wideProbability.quality_gain,
    selected_fractional_bits: wideProbability.selection.selected_fractional_bits,
    dev_total_millibits_delta_vs_q15_control:
      wideProbability.selection.selected_dev_total_millibits_delta_vs_q15_control,
    precision_effect: wideProbability.precision_effect.classification,
    exact_replay:
      wideProbability.replay.selected_model_sha256
        === wideProbability.replay.replay_model_sha256
      && wideProbability.replay.selected_optimizer_sha256
        === wideProbability.replay.replay_optimizer_sha256,
  },
  probability_normalization_accuracy: {
    outcome: probabilityNormalization.outcome,
    diagnostic_eligible: probabilityNormalization.diagnostic_eligible,
    selected_normalization: probabilityNormalization.selection.selected_normalization,
    best_nondivision_normalization:
      probabilityNormalization.accuracy_effect.best_nondivision_normalization,
    legacy_source_mass_error_max_ppm:
      probabilityNormalization.normalization_rows[0].mass.source_error_max_ppm,
    newton_source_mass_error_max_ppm:
      probabilityNormalization.normalization_rows[2].mass.source_error_max_ppm,
    exact_source_mass_error_max_ppm:
      probabilityNormalization.normalization_rows[3].mass.source_error_max_ppm,
    legacy_target_probability_changed_windows:
      probabilityNormalization.normalization_rows[0].delta
        .target_probability_changed_windows,
    newton_target_probability_changed_windows:
      probabilityNormalization.normalization_rows[2].delta
        .target_probability_changed_windows,
    exact_target_probability_changed_windows:
      probabilityNormalization.normalization_rows[3].delta
        .target_probability_changed_windows,
  },
  probability_normalization_signal_attribution: {
    outcome: normalizationAttribution.outcome,
    candidate_eligible: normalizationAttribution.candidate_eligible,
    selected_normalization: normalizationAttribution.selection.selected_normalization,
    exact_target_windows:
      normalizationAttribution.methods[2].target_changed_window_indices,
    newton_only_windows: normalizationAttribution.set_attribution.newton_only_indices,
    exact_missing_from_newton:
      normalizationAttribution.set_attribution.exact_missing_from_newton_indices,
    legacy_only_windows: normalizationAttribution.legacy_only_attribution.windows,
    maximum_probability_error_q23_units:
      normalizationAttribution.methods[1].source_error_vs_exact.probability_error_max,
  },
  normalized_wide_gradient_preflight: {
    outcome: normalizedWide.outcome,
    safety_eligible: normalizedWide.safety_eligible,
    integer_precision_bottleneck_resolved:
      normalizedWide.integer_precision_bottleneck_resolved,
    up_boundary_materialized_updates:
      normalizedWide.precision_effect.up_boundary_materialized_updates,
    up_boundary_feature_changed_windows:
      normalizedWide.precision_effect.up_boundary_feature_changed_windows,
    output_boundary_target_probability_changed_windows:
      normalizedWide.precision_effect.output_boundary_target_probability_changed_windows,
    output_boundary_dev_total_millibits_delta:
      normalizedWide.quality_effect.output_boundary_dev_total_millibits_delta,
    exact_replay:
      normalizedWide.replay.selected_model_sha256
        === normalizedWide.replay.replay_model_sha256
      && normalizedWide.replay.selected_optimizer_sha256
        === normalizedWide.replay.replay_optimizer_sha256,
  },
  next_gate: plan.implementation_status.next_gate,
}));
