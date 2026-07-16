#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-kv-scaling-readiness";
let outPath = "benchmarks/production-model-v1/p10m-kv-scaling-readiness.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-kv-scaling-readiness-contract.json";
const groups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];
const sortedGroups = [...groups].sort();
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const readRunJson = async (name) => readJson(path.join(runDir, name));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const exactGroupKeys = (value) => value && sameJson(Object.keys(value).sort(), sortedGroups);
const validCounts = (value) => exactGroupKeys(value)
  && Object.values(value).every((count) => Number.isSafeInteger(count) && count >= 0);

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const sourceContractPath = contract.source_pilot.contract_path;
  const sourceCheckpointPath = contract.source_pilot.checkpoint_path;
  const inputPaths = [
    contract.bindings.tokenizer_path,
    contract.bindings.train_tokens_path,
    contract.bindings.dev_tokens_path,
  ];
  const [sourceContractBytes, sourceCheckpointBytes, tokenizerBytes, trainBytes,
    devBytes, init, initialModel, devInitial, finalModel, finalOptimizer,
    replayModel, replayOptimizer, replayTrace, residualAnalysis,
    floatFinal, floatReplay, floatEquality, ...rows] = await Promise.all([
    readFile(sourceContractPath),
    readFile(sourceCheckpointPath),
    ...inputPaths.map((file) => readFile(file)),
    readRunJson("init.json"),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readRunJson("integer-dev-initial.json"),
    readFile(path.join(runDir, "integer-model-7.nsrlpm")),
    readFile(path.join(runDir, "integer-optimizer-7.nsrlpo")),
    readFile(path.join(runDir, "integer-replay-final.nsrlpm")),
    readFile(path.join(runDir, "integer-replay-final.nsrlpo")),
    readRunJson("integer-replay.json"),
    readRunJson("integer-residual-analysis.json"),
    readFile(path.join(runDir, "float-7.npz")),
    readRunJson("float-replay.json"),
    readRunJson("float-replay-equality.json"),
    ...Array.from({ length: 8 }, (_, chunk) => [
      readRunJson(`integer-train-${chunk}.json`),
      readRunJson(`integer-dev-${chunk}.json`),
      readRunJson(`integer-event-${chunk}.json`),
      readRunJson(`float-${chunk}.json`),
      readRunJson(`float-event-${chunk}.json`),
    ]).flat(),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const integerTraces = [];
  const integerDev = [];
  const integerEvents = [];
  const floatTraces = [];
  const floatEvents = [];
  for (let chunk = 0; chunk < 8; chunk += 1) {
    integerTraces.push(rows[chunk * 5]);
    integerDev.push(rows[chunk * 5 + 1]);
    integerEvents.push(rows[chunk * 5 + 2]);
    floatTraces.push(rows[chunk * 5 + 3]);
    floatEvents.push(rows[chunk * 5 + 4]);
  }

  const schedule = contract.schedule;
  const expectedShifts = schedule.integer_learning_rate_shifts;
  const movedByUpdate = (trace) => groups
    .filter((group) => trace.diagnostics.update_nonzero_count[group] > 0).sort();
  const exactMovement = integerTraces.every((trace) => {
    if (!validCounts(trace.diagnostics.update_nonzero_count)
      || !validCounts(trace.movement_l1)) return false;
    const updates = movedByUpdate(trace);
    const movement = groups.filter((group) => trace.movement_l1[group] > 0).sort();
    const declared = [...trace.moved_parameter_groups].sort();
    return sameJson(updates, movement)
      && sameJson(updates, declared)
      && (trace.hashes.initial_model !== trace.hashes.final_model) === (updates.length > 0);
  });
  const integerScheduleExact = integerTraces.every((trace, chunk) => {
    const expectedNextWindow = (chunk + 1) * schedule.windows_per_chunk;
    return trace.schema === "nsrl.production_full_train_smoke.v1"
      && trace.profile === contract.profile
      && trace.parameter_count === contract.parameter_count
      && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
      && trace.training.context_tokens === schedule.context_tokens
      && trace.training.windows === schedule.train_windows
      && trace.training.evaluation_windows
        === schedule.integer_evaluation_windows_per_chunk
      && trace.training.epochs === schedule.epochs
      && trace.training.batch_windows === schedule.batch_windows
      && trace.training.optimizer_steps === schedule.optimizer_steps_per_integer_chunk
      && trace.training.total_optimizer_step
        === (chunk + 1) * schedule.optimizer_steps_per_integer_chunk
      && sameJson(trace.training.learning_rate_shifts, expectedShifts)
      && trace.training.output_backward_shift === schedule.output_backward_shift
      && trace.cursor.start_window === chunk * schedule.windows_per_chunk
      && trace.cursor.next_window === (chunk === 7 ? 0 : expectedNextWindow)
      && trace.cursor.next_epoch === (chunk === 7 ? 1 : 0)
      && trace.cursor.schedule_complete === (chunk === 7);
  });
  const integerChainExact = integerTraces.every((trace, chunk) =>
    trace.hashes.initial_model === (chunk === 0
      ? contract.initialization.model_hash
      : integerTraces[chunk - 1].hashes.final_model));
  const integerHealthExact = integerTraces.every((trace) =>
    trace.health.gradient_saturation_count === 0
      && (trace.health.residual_saturation_count ?? 0) === 0
      && trace.health.weight_saturation_count === 0
      && validCounts(trace.diagnostics.saturation_by_group)
      && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
      && validCounts(trace.diagnostics.residual_saturation_by_group)
      && Object.values(trace.diagnostics.residual_saturation_by_group)
        .every((count) => count === 0));
  const integerGradientPathExact = integerTraces.every((trace) =>
    validCounts(trace.diagnostics.gradient_nonzero_count)
      && groups.every((group) => trace.diagnostics.gradient_nonzero_count[group] > 0));
  const movedAfterMidpoint = (group) => integerTraces.slice(4).some((trace) =>
    trace.diagnostics.update_nonzero_count[group] > 0 && trace.movement_l1[group] > 0);

  const floatScheduleExact = floatTraces.every((trace, chunk) =>
    trace.schema === "nsrl.production_float_twin_smoke.v1"
      && trace.profile === contract.profile
      && trace.parameter_count === contract.parameter_count
      && trace.bindings.integer_initial_model_hash === contract.initialization.model_hash
      && trace.bindings.integer_artifact_sha256 === contract.initialization.artifact_sha256
      && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
      && trace.training.context_tokens === schedule.context_tokens
      && trace.training.start_window === chunk * schedule.windows_per_chunk
      && trace.training.windows === schedule.windows_per_chunk
      && trace.training.evaluation_windows
        === schedule.float_training_evaluation_windows_per_chunk
      && trace.training.epochs === schedule.epochs
      && trace.training.batch_windows === schedule.batch_windows
      && trace.training.learning_rate_millionths
        === schedule.float_learning_rate_millionths
      && trace.evaluation.token_stream_hash === contract.bindings.dev_token_stream_hash
      && trace.evaluation.context_tokens === schedule.context_tokens
      && trace.evaluation.windows === schedule.dev_windows);
  const floatChainExact = floatTraces.every((trace, chunk) => chunk === 0
    || trace.tensor_hashes.initial === floatTraces[chunk - 1].tensor_hashes.final);
  const floatAllGroupsMove = floatTraces.every((trace) =>
    trace.gates.all_parameter_groups_moved === true
      && sameJson([...trace.moved_parameter_groups].sort(), sortedGroups));
  const floatHealthy = floatTraces.every((trace) =>
    trace.gates.all_parameters_finite === true
      && trace.gates.loss_nonincreasing === true
      && trace.gates.tensor_hash_changed === true);

  const initialMean = devInitial.evaluation.mean_millibits;
  const integerFinalMean = integerDev[7].evaluation.mean_millibits;
  const floatInitialMean = floatTraces[0].evaluation.initial_mean_millibits;
  const floatFinalMean = floatTraces[7].evaluation.final_mean_millibits;
  const integerFloatRegressionPerMille = Math.max(
    0,
    Math.ceil(((integerFinalMean - floatFinalMean) * 1000) / floatFinalMean),
  );
  const inputHashes = {
    tokenizer_sha256: sha256(tokenizerBytes),
    train_tokens_sha256: sha256(trainBytes),
    dev_tokens_sha256: sha256(devBytes),
  };
  const residualGroupsComplete = residualAnalysis.schema
      === "nsrl.production_optimizer_residual_analysis.v1"
    && residualAnalysis.profile === contract.profile
    && residualAnalysis.parameter_count === contract.parameter_count
    && residualAnalysis.groups.length === groups.length
    && sameJson(residualAnalysis.groups.map(({ group }) => group).sort(), sortedGroups)
    && residualAnalysis.groups.every((row) => Number.isSafeInteger(row.parameters)
      && row.parameters > 0
      && Number.isSafeInteger(row.gradient_nonzero_count)
      && Number.isSafeInteger(row.update_nonzero_count));
  const gates = {
    source_pilot_hashes_match:
      sha256(sourceContractBytes) === contract.source_pilot.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_pilot.checkpoint_sha256,
    source_pilot_eligible: sourceCheckpoint.pilot_eligible === true,
    bound_input_hashes_match:
      inputHashes.tokenizer_sha256 === contract.bindings.tokenizer_sha256
      && inputHashes.train_tokens_sha256 === contract.bindings.train_tokens_sha256
      && inputHashes.dev_tokens_sha256 === contract.bindings.dev_tokens_sha256,
    deterministic_initialization_matches:
      init.model_hash === contract.initialization.model_hash
      && init.initialization_seed === contract.initialization.seed
      && init.output_init_amplitude === contract.initialization.output_init_amplitude
      && init.output_forward_shift === contract.initialization.output_forward_shift
      && sha256(initialModel) === contract.initialization.artifact_sha256,
    integer_schedule_and_chain_exact: integerScheduleExact && integerChainExact,
    integer_k_and_v_move_by_window_256:
      integerEvents[0].required_trunk_group_observations?.k === true
      && integerEvents[0].required_trunk_group_observations?.v === true
      && integerTraces[0].diagnostics.update_nonzero_count.k > 0
      && integerTraces[0].diagnostics.update_nonzero_count.v > 0,
    integer_k_and_v_move_after_midpoint:
      movedAfterMidpoint("k") && movedAfterMidpoint("v"),
    integer_all_gradient_paths_active: integerGradientPathExact,
    integer_saturation_zero: integerHealthExact,
    integer_exact_reachable_update_consistency: exactMovement,
    integer_all_chunks_live: integerEvents.every((event) =>
      event.dead === false && event.full_gradient_path === true),
    integer_heldout_nonincreasing_every_chunk: integerDev.every((row) =>
      row.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
    integer_heldout_improves_at_completion:
      integerDev[7].evaluation.total_millibits
        < devInitial.evaluation.total_millibits,
    integer_midpoint_restart_byte_identical:
      sha256(finalModel) === sha256(replayModel)
      && sha256(finalOptimizer) === sha256(replayOptimizer),
    integer_replay_schedule_complete_and_healthy:
      replayTrace.training.optimizer_steps === schedule.integer_replay_optimizer_steps
      && replayTrace.cursor.start_window === schedule.midpoint_window
      && replayTrace.cursor.next_window === 0
      && replayTrace.cursor.next_epoch === 1
      && replayTrace.cursor.schedule_complete === true
      && replayTrace.health.gradient_saturation_count === 0
      && (replayTrace.health.residual_saturation_count ?? 0) === 0
      && replayTrace.health.weight_saturation_count === 0,
    float_schedule_and_chain_exact: floatScheduleExact && floatChainExact,
    float_all_parameter_groups_move: floatAllGroupsMove,
    float_all_chunks_healthy: floatHealthy
      && floatEvents.every((event) => event.continue_training === true),
    float_heldout_mean_millibits_nonincreasing_every_chunk: floatEvents.every((event) =>
      event.heldout_current_mean_millibits <= floatInitialMean),
    float_heldout_improves_at_completion: floatFinalMean < floatInitialMean
      && floatTraces[7].evaluation.final_loss_millionths
        < floatTraces[0].evaluation.initial_loss_millionths,
    float_midpoint_restart_tensor_identical:
      floatEquality.schema === "nsrl.production_float_artifact_equality.v1"
      && floatEquality.byte_identical_tensors === true
      && floatReplay.training.start_window === schedule.midpoint_window
      && floatReplay.training.windows === schedule.float_replay_windows
      && floatReplay.tensor_hashes.initial === floatTraces[3].tensor_hashes.final
      && floatReplay.tensor_hashes.final === floatTraces[7].tensor_hashes.final,
    matched_integer_float_budget:
      integerTraces[7].training.total_optimizer_step * schedule.batch_windows
        === schedule.train_windows
      && floatTraces.reduce((sum, trace) => sum + trace.training.windows, 0)
        === schedule.train_windows,
    integer_float_dev_regression_within_150_per_mille:
      integerFloatRegressionPerMille
        <= contract.completion_gates.integer_float_dev_regression_max_per_mille,
    remaining_group_residual_analysis_complete: residualGroupsComplete
      && residualAnalysis.source.optimizer_sha256 === sha256(finalOptimizer)
      && residualAnalysis.source.optimizer_state_hash
        === integerTraces[7].hashes.optimizer_state,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const readinessEligible = Object.values(gates).every(Boolean);
  const recommendation = residualAnalysis.recommendation;
  const nextGate = readinessEligible && recommendation
    ? `p10m_${recommendation.group}_boundary_preflight_contract`
    : readinessEligible
      ? "p20m_matched_scaling_contract_review"
      : "p10m_kv_scaling_readiness_review";

  return {
    schema: "nsrl.production_kv_scaling_readiness_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_pilot: contract.source_pilot,
    bindings: {
      ...contract.bindings,
      observed_sha256: inputHashes,
    },
    initialization: init,
    integer: {
      optimizer: "integer_residual_sgd",
      chunks: integerTraces.map((trace, chunk) => ({
        chunk,
        start_window: trace.cursor.start_window,
        next_window: trace.cursor.next_window,
        optimizer_steps: trace.training.optimizer_steps,
        total_optimizer_step: trace.training.total_optimizer_step,
        initial_model_hash: trace.hashes.initial_model,
        final_model_hash: trace.hashes.final_model,
        optimizer_state_hash: trace.hashes.optimizer_state,
        moved_parameter_groups: trace.moved_parameter_groups,
        update_nonzero_count: trace.diagnostics.update_nonzero_count,
        movement_l1: trace.movement_l1,
        dev: integerDev[chunk].evaluation,
        liveness: integerEvents[chunk],
      })),
      heldout: {
        initial: devInitial.evaluation,
        final: integerDev[7].evaluation,
        total_millibits_delta:
          integerDev[7].evaluation.total_millibits
            - devInitial.evaluation.total_millibits,
      },
      restart: {
        midpoint_window: schedule.midpoint_window,
        final_model_sha256: sha256(finalModel),
        replay_model_sha256: sha256(replayModel),
        final_optimizer_sha256: sha256(finalOptimizer),
        replay_optimizer_sha256: sha256(replayOptimizer),
      },
      residual_analysis: residualAnalysis,
      artifacts: {
        model: { bytes: finalModel.length, sha256: sha256(finalModel) },
        optimizer: { bytes: finalOptimizer.length, sha256: sha256(finalOptimizer) },
      },
    },
    float: {
      optimizer: "float32_sgd_reference",
      chunks: floatTraces.map((trace, chunk) => ({
        chunk,
        start_window: trace.training.start_window,
        next_window: trace.training.start_window + trace.training.windows,
        initial_tensor_hash: trace.tensor_hashes.initial,
        final_tensor_hash: trace.tensor_hashes.final,
        moved_parameter_groups: trace.moved_parameter_groups,
        training_initial_mean_millibits: trace.training.initial_mean_millibits,
        training_final_mean_millibits: trace.training.final_mean_millibits,
        heldout_initial_loss_millionths: trace.evaluation.initial_loss_millionths,
        heldout_final_loss_millionths: trace.evaluation.final_loss_millionths,
        heldout_initial_mean_millibits: trace.evaluation.initial_mean_millibits,
        heldout_final_mean_millibits: trace.evaluation.final_mean_millibits,
        health: floatEvents[chunk],
      })),
      heldout: {
        initial_loss_millionths: floatTraces[0].evaluation.initial_loss_millionths,
        final_loss_millionths: floatTraces[7].evaluation.final_loss_millionths,
        loss_delta_millionths: floatTraces[7].evaluation.final_loss_millionths
          - floatTraces[0].evaluation.initial_loss_millionths,
        initial_mean_millibits: floatInitialMean,
        final_mean_millibits: floatFinalMean,
        delta_millibits: floatFinalMean - floatInitialMean,
        chunk_zero_sub_millibit_loss_delta_millionths:
          floatTraces[0].evaluation.final_loss_millionths
            - floatTraces[0].evaluation.initial_loss_millionths,
        gate_resolution: "rounded_mean_millibits",
      },
      restart: floatEquality,
      artifact: { bytes: floatFinal.length, sha256: sha256(floatFinal) },
    },
    comparison: {
      matched_train_windows: schedule.train_windows,
      integer_final_mean_millibits: integerFinalMean,
      float_final_mean_millibits: floatFinalMean,
      integer_minus_float_mean_millibits: integerFinalMean - floatFinalMean,
      integer_regression_per_mille: integerFloatRegressionPerMille,
      maximum_regression_per_mille:
        contract.completion_gates.integer_float_dev_regression_max_per_mille,
    },
    gates,
    readiness_eligible: readinessEligible,
    paid_scale_authorized: false,
    next_gate: nextGate,
    known_non_claims: contract.known_non_claims,
  };
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = await readJson(outPath);
}
const failedGates = Object.entries(checkpoint.gates ?? {})
  .filter(([, passed]) => !passed).map(([gate]) => gate);
const currentContractBytes = await readFile(contractPath);
const expectedNextGate = checkpoint.integer?.residual_analysis?.recommendation
  ? `p10m_${checkpoint.integer.residual_analysis.recommendation.group}_boundary_preflight_contract`
  : "p20m_matched_scaling_contract_review";
if (checkpoint.schema !== "nsrl.production_kv_scaling_readiness_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.readiness_eligible !== true
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== expectedNextGate
  || failedGates.length > 0) {
  throw new Error(
    `production K+V scaling readiness is invalid; failed gates: ${failedGates.join(",")}`,
  );
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production K+V scaling readiness checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_kv_scaling_readiness_check.v1",
    ok: true,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, readiness_eligible: true }));
}
