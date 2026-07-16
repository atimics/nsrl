#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-up-useful-update";
let outPath = "benchmarks/production-model-v1/p10m-up-useful-update.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-up-useful-update-contract.json";
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

function outcomeFor(safetyEligible, qualityBreakthrough) {
  if (!safetyEligible) return "failed_safety";
  return qualityBreakthrough
    ? "quality_breakthrough"
    : "safe_reachability_without_source_gain";
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, tokenizerBytes, trainBytes,
    devBytes, init, initialModel, devInitial, replayTrace, residualAnalysis,
    finalModel, finalOptimizer, replayModel, replayOptimizer, ...rows] = await Promise.all([
    readFile(contract.source_gate_preflight.contract_path),
    readFile(contract.source_gate_preflight.checkpoint_path),
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
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
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
  const sourceAnalysis = sourceCheckpoint.final_residual_analysis;
  const sourceUp = sourceAnalysis.groups.find(({ group }) => group === "up");
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
  const groupMoves = (group, selectedTraces = traces) => selectedTraces.some((trace) =>
    trace.diagnostics.update_nonzero_count[group] > 0 && trace.movement_l1[group] > 0);
  const upUpdateCount = traces.reduce(
    (sum, trace) => sum + trace.diagnostics.update_nonzero_count.up,
    0,
  );
  const firstUpMovementInterval = traces.findIndex((trace) =>
    trace.diagnostics.update_nonzero_count.up > 0);
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

  const safetyGates = {
    source_gate_preflight_hashes_match:
      sha256(sourceContractBytes) === contract.source_gate_preflight.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_gate_preflight.checkpoint_sha256,
    source_gate_preflight_eligible: sourceCheckpoint.preflight_eligible === true
      && sourceCheckpoint.paid_scale_authorized === false,
    source_optimizer_matches: sourceAnalysis.source.optimizer_sha256
      === contract.source_optimizer.sha256
      && sourceAnalysis.source.optimizer_state_hash
        === contract.source_optimizer.optimizer_state_hash
      && sourceAnalysis.source.optimizer_step === contract.source_optimizer.optimizer_step,
    source_heldout_matches: sameJson(sourceCheckpoint.heldout.final, contract.source_heldout),
    candidate_matches_source_residual_recommendation: candidateMatchesRecommendation
      && sourceUp?.current_shift === contract.candidate.source_shift
      && sourceUp?.boundary_shift === contract.candidate.candidate_shift
      && sourceUp?.update_nonzero_count === 0
      && sourceUp?.predicted_parameter_crossings_at_boundary
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
    only_up_shift_changed: Object.keys(baseShifts).every((group) =>
      candidateShifts[group] === (group === "up"
        ? contract.candidate.candidate_shift
        : baseShifts[group])),
    up_gradient_and_carry_present: traces.every((trace) =>
      trace.diagnostics.gradient_nonzero_count.up > 0
        && trace.diagnostics.residual_carry_count.up > 0),
    up_moves_by_window_2048: upUpdateCount > 0
      && events[7].required_trunk_group_observations?.up === true,
    k_v_and_gate_remain_reachable: traces.every((trace) =>
      trace.diagnostics.update_nonzero_count.k > 0
        && trace.diagnostics.update_nonzero_count.v > 0)
      && groupMoves("gate") && groupMoves("gate", traces.slice(4)),
    only_k_v_up_gate_and_output_move: traces.every((trace) =>
      trace.moved_parameter_groups.every((group) =>
        ["k", "v", "up", "gate", "output"].includes(group))),
    exact_reachable_update_consistency: exactMovement,
    all_13_gradient_paths_active: fullGradientPath,
    all_saturation_zero: exactHealth,
    all_intervals_live: events.every((event) =>
      event.dead === false && event.full_gradient_path === true),
    heldout_nonincreasing_vs_lane_initial: dev.every((row) =>
      row.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
    heldout_improves_vs_lane_initial: dev[7].evaluation.total_millibits
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
  const finalHeldout = dev[7].evaluation;
  const qualityGates = {
    final_total_millibits_beats_source_by_required_margin:
      finalHeldout.total_millibits
        <= contract.quality_gates.candidate_final_total_millibits_max,
    final_mean_millibits_beats_source:
      finalHeldout.mean_millibits
        <= contract.quality_gates.candidate_final_mean_millibits_max,
    measured_mean_improvement_meets_minimum:
      contract.source_heldout.mean_millibits - finalHeldout.mean_millibits
        >= contract.quality_gates.minimum_mean_millibit_improvement_vs_source,
  };
  const safetyEligible = Object.values(safetyGates).every(Boolean);
  const qualityBreakthrough = safetyEligible && Object.values(qualityGates).every(Boolean);
  const outcome = outcomeFor(safetyEligible, qualityBreakthrough);
  const recommendation = residualAnalysis.recommendation;
  const nextGate = qualityBreakthrough
    ? recommendation
      ? `p10m_${recommendation.group}_boundary_preflight_contract`
      : "p20m_matched_scaling_contract_review"
    : safetyEligible
      ? "p10m_up_useful_update_shift_sweep_contract"
      : "p10m_up_boundary_policy_review";

  return {
    schema: "nsrl.production_up_useful_update_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_gate_preflight: contract.source_gate_preflight,
    source_optimizer: contract.source_optimizer,
    bindings: { ...contract.bindings, observed_sha256: inputHashes },
    initialization: init,
    candidate: {
      ...contract.candidate,
      learning_rate_shifts: candidateShifts,
      actual_update_count: upUpdateCount,
      first_movement_interval: firstUpMovementInterval,
      first_movement_window: firstUpMovementInterval < 0
        ? null
        : (firstUpMovementInterval + 1) * schedule.interval_windows,
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
      up_gradient_nonzero_count: trace.diagnostics.gradient_nonzero_count.up,
      up_residual_carry_count: trace.diagnostics.residual_carry_count.up,
      health: trace.health,
      dev: dev[interval].evaluation,
      liveness: events[interval],
    })),
    heldout: {
      initial: devInitial.evaluation,
      source: contract.source_heldout,
      final: finalHeldout,
      total_millibits_delta_vs_initial:
        finalHeldout.total_millibits - devInitial.evaluation.total_millibits,
      total_millibits_delta_vs_source:
        finalHeldout.total_millibits - contract.source_heldout.total_millibits,
      mean_millibits_delta_vs_source:
        finalHeldout.mean_millibits - contract.source_heldout.mean_millibits,
      mistakes_delta_vs_source: finalHeldout.mistakes - contract.source_heldout.mistakes,
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
    safety_gates: safetyGates,
    quality_gates: qualityGates,
    safety_eligible: safetyEligible,
    quality_breakthrough: qualityBreakthrough,
    promotion_eligible: qualityBreakthrough,
    outcome,
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
const expectedSafety = Object.values(checkpoint.safety_gates ?? {}).every(Boolean);
const expectedQuality = expectedSafety
  && Object.values(checkpoint.quality_gates ?? {}).every(Boolean);
const expectedOutcome = outcomeFor(expectedSafety, expectedQuality);
const expectedNextGate = expectedQuality
  ? checkpoint.final_residual_analysis?.recommendation
    ? `p10m_${checkpoint.final_residual_analysis.recommendation.group}_boundary_preflight_contract`
    : "p20m_matched_scaling_contract_review"
  : expectedSafety
    ? "p10m_up_useful_update_shift_sweep_contract"
    : "p10m_up_boundary_policy_review";
if (checkpoint.schema !== "nsrl.production_up_useful_update_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.candidate?.group !== "up"
  || checkpoint.candidate?.candidate_shift !== 23
  || checkpoint.safety_eligible !== expectedSafety
  || checkpoint.quality_breakthrough !== expectedQuality
  || checkpoint.promotion_eligible !== expectedQuality
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== expectedNextGate) {
  throw new Error("production up useful-update checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production up useful-update checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_up_useful_update_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    promotion_eligible: checkpoint.promotion_eligible,
  }));
}
