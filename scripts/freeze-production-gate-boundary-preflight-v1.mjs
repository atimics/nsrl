#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-gate-boundary-preflight";
let outPath = "benchmarks/production-model-v1/p10m-gate-boundary-preflight.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-gate-boundary-preflight-contract.json";
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
  const [sourceContractBytes, sourceReadinessBytes, tokenizerBytes, trainBytes,
    devBytes, init, initialModel, devInitial, replayTrace, residualAnalysis,
    finalModel, finalOptimizer, replayModel, replayOptimizer, ...rows] = await Promise.all([
    readFile(contract.source_readiness.contract_path),
    readFile(contract.source_readiness.checkpoint_path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.train_tokens_path),
    readFile(contract.bindings.dev_tokens_path),
    readRunJson("init.json"),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readRunJson("dev-initial.json"),
    readRunJson("replay.json"),
    readRunJson("residual-analysis.json"),
    readFile(path.join(runDir, "model-7.nsrlpm")),
    readFile(path.join(runDir, "optimizer-7.nsrlpo")),
    readFile(path.join(runDir, "replay-final.nsrlpm")),
    readFile(path.join(runDir, "replay-final.nsrlpo")),
    ...Array.from({ length: 8 }, (_, interval) => [
      readRunJson(`train-${interval}.json`),
      readRunJson(`dev-${interval}.json`),
      readRunJson(`event-${interval}.json`),
    ]).flat(),
  ]);
  const sourceReadiness = JSON.parse(sourceReadinessBytes);
  const traces = [];
  const dev = [];
  const events = [];
  for (let interval = 0; interval < 8; interval += 1) {
    traces.push(rows[interval * 3]);
    dev.push(rows[interval * 3 + 1]);
    events.push(rows[interval * 3 + 2]);
  }

  const schedule = contract.schedule;
  const baseShifts = schedule.base_learning_rate_shifts;
  const candidateShifts = {
    ...baseShifts,
    [contract.candidate.group]: contract.candidate.candidate_shift,
  };
  const sourceAnalysis = sourceReadiness.integer.residual_analysis;
  const sourceGate = sourceAnalysis.groups.find(({ group }) => group === "gate");
  const candidateMatchesRecommendation = sameJson(
    sourceAnalysis.recommendation,
    {
      policy: contract.residual_policy.id,
      group: contract.candidate.group,
      source_shift: contract.candidate.source_shift,
      candidate_shift: contract.candidate.candidate_shift,
      shift_reduction: contract.candidate.shift_reduction,
      predicted_parameter_crossings: contract.candidate.predicted_parameter_crossings,
    },
  );
  const movedByUpdate = (trace) => groups
    .filter((group) => trace.diagnostics.update_nonzero_count[group] > 0).sort();
  const exactMovement = traces.every((trace) => {
    if (!validCounts(trace.diagnostics.update_nonzero_count)
      || !validCounts(trace.movement_l1)) return false;
    const updates = movedByUpdate(trace);
    const movement = groups.filter((group) => trace.movement_l1[group] > 0).sort();
    const declared = [...trace.moved_parameter_groups].sort();
    return sameJson(updates, movement)
      && sameJson(updates, declared)
      && (trace.hashes.initial_model !== trace.hashes.final_model) === (updates.length > 0);
  });
  const scheduleExact = traces.every((trace, interval) => {
    const expectedNextWindow = (interval + 1) * schedule.interval_windows;
    return trace.schema === "nsrl.production_full_train_smoke.v1"
      && trace.profile === contract.profile
      && trace.parameter_count === contract.parameter_count
      && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
      && trace.training.context_tokens === schedule.context_tokens
      && trace.training.windows === schedule.train_windows
      && trace.training.evaluation_windows
        === schedule.integer_evaluation_windows_per_interval
      && trace.training.epochs === schedule.epochs
      && trace.training.batch_windows === schedule.batch_windows
      && trace.training.optimizer_steps === schedule.interval_optimizer_steps
      && trace.training.total_optimizer_step
        === (interval + 1) * schedule.interval_optimizer_steps
      && sameJson(trace.training.learning_rate_shifts, candidateShifts)
      && trace.training.output_backward_shift === schedule.output_backward_shift
      && trace.cursor.start_window === interval * schedule.interval_windows
      && trace.cursor.next_window === (interval === 7 ? 0 : expectedNextWindow)
      && trace.cursor.next_epoch === (interval === 7 ? 1 : 0)
      && trace.cursor.schedule_complete === (interval === 7);
  });
  const modelChainExact = traces.every((trace, interval) =>
    trace.hashes.initial_model === (interval === 0
      ? contract.initialization.model_hash
      : traces[interval - 1].hashes.final_model));
  const exactHealth = traces.every((trace) =>
    trace.health.gradient_saturation_count === 0
      && (trace.health.residual_saturation_count ?? 0) === 0
      && trace.health.weight_saturation_count === 0
      && validCounts(trace.diagnostics.saturation_by_group)
      && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
      && validCounts(trace.diagnostics.residual_saturation_by_group)
      && Object.values(trace.diagnostics.residual_saturation_by_group)
        .every((count) => count === 0));
  const fullGradientPath = traces.every((trace) =>
    validCounts(trace.diagnostics.gradient_nonzero_count)
      && groups.every((group) => trace.diagnostics.gradient_nonzero_count[group] > 0));
  const gateUpdateCount = traces.reduce(
    (sum, trace) => sum + trace.diagnostics.update_nonzero_count.gate,
    0,
  );
  const firstGateMovementInterval = traces.findIndex((trace) =>
    trace.diagnostics.update_nonzero_count.gate > 0);
  const finalGroupsComplete = residualAnalysis.schema
      === "nsrl.production_optimizer_residual_analysis.v1"
    && residualAnalysis.profile === contract.profile
    && residualAnalysis.parameter_count === contract.parameter_count
    && residualAnalysis.groups.length === groups.length
    && sameJson(residualAnalysis.groups.map(({ group }) => group).sort(), sortedGroups);
  const inputHashes = {
    tokenizer_sha256: sha256(tokenizerBytes),
    train_tokens_sha256: sha256(trainBytes),
    dev_tokens_sha256: sha256(devBytes),
  };

  const gates = {
    source_readiness_hashes_match:
      sha256(sourceContractBytes) === contract.source_readiness.contract_sha256
      && sha256(sourceReadinessBytes) === contract.source_readiness.checkpoint_sha256,
    source_readiness_eligible: sourceReadiness.readiness_eligible === true
      && sourceReadiness.paid_scale_authorized === false,
    source_optimizer_matches: sourceAnalysis.source.optimizer_sha256
      === contract.source_optimizer.sha256
      && sourceAnalysis.source.optimizer_state_hash
        === contract.source_optimizer.optimizer_state_hash
      && sourceAnalysis.source.optimizer_step === contract.source_optimizer.optimizer_step,
    candidate_matches_source_residual_recommendation: candidateMatchesRecommendation
      && sourceGate?.current_shift === contract.candidate.source_shift
      && sourceGate?.boundary_shift === contract.candidate.candidate_shift
      && sourceGate?.update_nonzero_count === 0
      && sourceGate?.predicted_parameter_crossings_at_boundary
        === contract.candidate.predicted_parameter_crossings,
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
    candidate_schedule_and_chain_exact: scheduleExact && modelChainExact,
    only_gate_shift_changed: Object.keys(baseShifts).every((group) =>
      candidateShifts[group] === (group === "gate"
        ? contract.candidate.candidate_shift
        : baseShifts[group])),
    gate_gradient_and_carry_present: traces.every((trace) =>
      trace.diagnostics.gradient_nonzero_count.gate > 0
        && trace.diagnostics.residual_carry_count.gate > 0),
    gate_moves_by_window_2048: gateUpdateCount > 0
      && events[7].required_trunk_group_observations?.gate === true,
    k_and_v_remain_reachable: traces.every((trace) =>
      trace.diagnostics.update_nonzero_count.k > 0
        && trace.diagnostics.update_nonzero_count.v > 0),
    only_k_v_gate_and_output_move: traces.every((trace) =>
      trace.moved_parameter_groups.every((group) =>
        ["k", "v", "gate", "output"].includes(group))),
    exact_reachable_update_consistency: exactMovement,
    all_13_gradient_paths_active: fullGradientPath,
    all_saturation_zero: exactHealth,
    all_intervals_live: events.every((event) =>
      event.dead === false && event.full_gradient_path === true),
    heldout_nonincreasing_vs_lane_initial: dev.every((row) =>
      row.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
    heldout_improves_at_completion: dev[7].evaluation.total_millibits
      < devInitial.evaluation.total_millibits,
    midpoint_restart_model_byte_identical: sha256(finalModel) === sha256(replayModel),
    midpoint_restart_optimizer_byte_identical:
      sha256(finalOptimizer) === sha256(replayOptimizer),
    replay_schedule_complete_and_healthy:
      replayTrace.training.optimizer_steps === schedule.replay_optimizer_steps
      && replayTrace.cursor.start_window === schedule.midpoint_window
      && replayTrace.cursor.next_window === 0
      && replayTrace.cursor.next_epoch === 1
      && replayTrace.cursor.schedule_complete === true
      && replayTrace.health.gradient_saturation_count === 0
      && (replayTrace.health.residual_saturation_count ?? 0) === 0
      && replayTrace.health.weight_saturation_count === 0,
    final_residual_analysis_complete: finalGroupsComplete
      && residualAnalysis.source.optimizer_sha256 === sha256(finalOptimizer)
      && residualAnalysis.source.optimizer_state_hash === traces[7].hashes.optimizer_state,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const preflightEligible = Object.values(gates).every(Boolean);
  const recommendation = residualAnalysis.recommendation;
  const nextGate = preflightEligible && recommendation
    ? `p10m_${recommendation.group}_boundary_preflight_contract`
    : preflightEligible
      ? "p20m_matched_scaling_contract_review"
      : "p10m_gate_boundary_policy_review";

  return {
    schema: "nsrl.production_gate_boundary_preflight_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_readiness: contract.source_readiness,
    source_optimizer: contract.source_optimizer,
    bindings: { ...contract.bindings, observed_sha256: inputHashes },
    initialization: init,
    candidate: {
      ...contract.candidate,
      learning_rate_shifts: candidateShifts,
      actual_update_count: gateUpdateCount,
      first_movement_interval: firstGateMovementInterval,
      first_movement_window: firstGateMovementInterval < 0
        ? null
        : (firstGateMovementInterval + 1) * schedule.interval_windows,
    },
    intervals: traces.map((trace, interval) => ({
      interval,
      start_window: trace.cursor.start_window,
      next_window: trace.cursor.next_window,
      initial_model_hash: trace.hashes.initial_model,
      final_model_hash: trace.hashes.final_model,
      optimizer_state_hash: trace.hashes.optimizer_state,
      moved_parameter_groups: trace.moved_parameter_groups,
      update_nonzero_count: trace.diagnostics.update_nonzero_count,
      movement_l1: trace.movement_l1,
      gate_gradient_nonzero_count: trace.diagnostics.gradient_nonzero_count.gate,
      gate_residual_carry_count: trace.diagnostics.residual_carry_count.gate,
      health: trace.health,
      dev: dev[interval].evaluation,
      liveness: events[interval],
    })),
    heldout: {
      initial: devInitial.evaluation,
      final: dev[7].evaluation,
      total_millibits_delta:
        dev[7].evaluation.total_millibits - devInitial.evaluation.total_millibits,
    },
    restart: {
      midpoint_window: schedule.midpoint_window,
      final_model_sha256: sha256(finalModel),
      replay_model_sha256: sha256(replayModel),
      final_optimizer_sha256: sha256(finalOptimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    final_residual_analysis: residualAnalysis,
    artifacts: {
      model: { bytes: finalModel.length, sha256: sha256(finalModel) },
      optimizer: { bytes: finalOptimizer.length, sha256: sha256(finalOptimizer) },
    },
    gates,
    preflight_eligible: preflightEligible,
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
const currentContractBytes = await readFile(contractPath);
const failedGates = Object.entries(checkpoint.gates ?? {})
  .filter(([, passed]) => !passed).map(([gate]) => gate);
const expectedNextGate = checkpoint.final_residual_analysis?.recommendation
  ? `p10m_${checkpoint.final_residual_analysis.recommendation.group}_boundary_preflight_contract`
  : "p20m_matched_scaling_contract_review";
if (checkpoint.schema !== "nsrl.production_gate_boundary_preflight_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.candidate?.group !== "gate"
  || checkpoint.candidate?.candidate_shift !== 23
  || checkpoint.preflight_eligible !== true
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== expectedNextGate
  || failedGates.length > 0) {
  throw new Error(
    `production gate boundary preflight is invalid; failed gates: ${failedGates.join(",")}`,
  );
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production gate boundary preflight checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_gate_boundary_preflight_check.v1",
    ok: true,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, preflight_eligible: true }));
}
