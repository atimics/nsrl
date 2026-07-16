#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-up-shift22-breakthrough";
let outPath = "benchmarks/production-model-v1/p10m-up-shift22-breakthrough.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-up-shift22-breakthrough-contract.json";
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

function outcomeFor(safetyEligible, discoveryPassed, confirmationPassed) {
  if (!safetyEligible) return "failed_safety";
  if (!discoveryPassed) return "no_dev_discovery";
  return confirmationPassed
    ? "confirmed_quality_breakthrough"
    : "dev_discovery_not_test_confirmed";
}

function nextGateFor(outcome) {
  if (outcome === "confirmed_quality_breakthrough") {
    return "p10m_up_shift22_breakthrough_replication_contract";
  }
  if (outcome === "dev_discovery_not_test_confirmed") {
    return "p10m_up_shift22_generalization_review";
  }
  if (outcome === "no_dev_discovery") return "p10m_integer_objective_quality_review";
  return "p10m_up_shift22_safety_review";
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, sourceModel, sourceOptimizer,
    tokenizerBytes, trainBytes, devBytes, testBytes, init, initialModel, devInitial,
    sourceTest, selectedTest, replayTrace, replayModel, replayOptimizer,
    residualAnalysis, ...rows] = await Promise.all([
    readFile(contract.source_up_useful_update.contract_path),
    readFile(contract.source_up_useful_update.checkpoint_path),
    readFile(contract.source_artifacts.model_path),
    readFile(contract.source_artifacts.optimizer_path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.train_tokens_path),
    readFile(contract.bindings.dev_tokens_path),
    readFile(contract.bindings.test_tokens_path),
    readRunJson("init.json"),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readRunJson("dev-initial.json"),
    readRunJson("test-source.json"),
    readRunJson("test-selected.json"),
    readRunJson("replay.json"),
    readFile(path.join(runDir, "replay-selected.nsrlpm")),
    readFile(path.join(runDir, "replay-selected.nsrlpo")),
    readRunJson("residual-analysis-selected.json"),
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
  const selectedInterval = dev.reduce((best, row, interval) =>
    row.evaluation.total_millibits < dev[best].evaluation.total_millibits
      ? interval
      : best, 0);
  const selectedTrace = traces[selectedInterval];
  const selectedDev = dev[selectedInterval].evaluation;
  const selectedModel = await readFile(path.join(runDir, `model-${selectedInterval}.nsrlpm`));
  const selectedOptimizer = await readFile(
    path.join(runDir, `optimizer-${selectedInterval}.nsrlpo`),
  );
  const schedule = contract.schedule;
  const shifts = schedule.learning_rate_shifts;
  const sourceUp = sourceCheckpoint.final_residual_analysis.groups
    .find(({ group }) => group === contract.candidate.group);
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
      && sameJson(trace.training.learning_rate_shifts, shifts)
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
  const upUpdateCount = traces.reduce(
    (sum, trace) => sum + trace.diagnostics.update_nonzero_count.up,
    0,
  );
  const firstUpMovementInterval = traces.findIndex((trace) =>
    trace.diagnostics.update_nonzero_count.up > 0);
  const inputHashes = {
    tokenizer_sha256: sha256(tokenizerBytes),
    train_tokens_sha256: sha256(trainBytes),
    dev_tokens_sha256: sha256(devBytes),
    test_tokens_sha256: sha256(testBytes),
  };
  const selectedSteps = (selectedInterval + 1) * schedule.interval_optimizer_steps;
  const selectedNextWindow = selectedInterval === 7
    ? 0
    : (selectedInterval + 1) * schedule.interval_windows;
  const selectedNextEpoch = selectedInterval === 7 ? 1 : 0;
  const finalGroupsComplete = residualAnalysis.schema
      === "nsrl.production_optimizer_residual_analysis.v1"
    && residualAnalysis.profile === contract.profile
    && residualAnalysis.parameter_count === contract.parameter_count
    && residualAnalysis.groups.length === groups.length
    && sameJson(residualAnalysis.groups.map(({ group }) => group).sort(), sortedGroups);
  const sourceEvidenceMatches = sourceUp?.current_shift === contract.candidate.source_shift
    && sourceUp?.boundary_shift === contract.candidate.candidate_shift
    && sourceUp?.predicted_parameter_crossings_at_boundary
      === contract.candidate.predicted_parameter_crossings
    && sourceCheckpoint.candidate.actual_update_count === contract.candidate.source_update_count
    && sourceCheckpoint.heldout.final.total_millibits === contract.source_dev.total_millibits;

  const safetyGates = {
    source_safe_no_gain_outcome_matches:
      sha256(sourceContractBytes) === contract.source_up_useful_update.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_up_useful_update.checkpoint_sha256
      && sourceCheckpoint.outcome === contract.source_up_useful_update.required_outcome
      && sourceCheckpoint.safety_eligible === true
      && sourceCheckpoint.quality_breakthrough === false,
    source_artifacts_and_residual_evidence_match:
      sha256(sourceModel) === contract.source_artifacts.model_sha256
      && sha256(sourceOptimizer) === contract.source_artifacts.optimizer_sha256
      && sourceCheckpoint.artifacts.model.sha256 === contract.source_artifacts.model_sha256
      && sourceCheckpoint.artifacts.optimizer.sha256
        === contract.source_artifacts.optimizer_sha256
      && sourceCheckpoint.intervals[7].final_model_hash
        === contract.source_artifacts.model_hash
      && sourceCheckpoint.intervals[7].optimizer_state_hash
        === contract.source_artifacts.optimizer_state_hash
      && sourceEvidenceMatches,
    bound_input_hashes_match:
      inputHashes.tokenizer_sha256 === contract.bindings.tokenizer_sha256
      && inputHashes.train_tokens_sha256 === contract.bindings.train_tokens_sha256
      && inputHashes.dev_tokens_sha256 === contract.bindings.dev_tokens_sha256
      && inputHashes.test_tokens_sha256 === contract.bindings.test_tokens_sha256,
    deterministic_initialization_matches:
      init.model_hash === contract.initialization.model_hash
      && init.initialization_seed === contract.initialization.seed
      && init.output_init_amplitude === contract.initialization.output_init_amplitude
      && init.output_forward_shift === contract.initialization.output_forward_shift
      && sha256(initialModel) === contract.initialization.artifact_sha256,
    candidate_schedule_and_chain_exact: scheduleExact && modelChainExact,
    up_is_only_modified_shift:
      shifts.up === contract.candidate.candidate_shift
      && shifts.gate === 23
      && shifts.k === 26
      && shifts.v === 30,
    all_eight_intervals_complete: traces.length === schedule.intervals
      && dev.length === schedule.intervals && events.length === schedule.intervals,
    up_crosses_exact_integer_boundaries_by_window_1024:
      upUpdateCount > 0
      && traces.slice(0, 4).some((trace) => trace.diagnostics.update_nonzero_count.up > 0)
      && events[3].required_trunk_group_observations?.up === true,
    only_k_v_up_gate_and_output_move: traces.every((trace) =>
      trace.moved_parameter_groups.every((group) =>
        ["k", "v", "up", "gate", "output"].includes(group))),
    exact_reachable_update_consistency: exactMovement,
    all_13_gradient_paths_active: fullGradientPath,
    all_saturation_zero: exactHealth,
    all_intervals_live: events.every((event) =>
      event.dead === false && event.full_gradient_path === true),
    heldout_total_millibits_nonincreasing_vs_lane_initial: dev.every((row) =>
      row.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
    selected_checkpoint_is_deterministic_dev_minimum:
      dev.every((row, interval) =>
        selectedDev.total_millibits <= row.evaluation.total_millibits
        && (selectedDev.total_millibits !== row.evaluation.total_millibits
          || selectedInterval <= interval)),
    selected_checkpoint_replay_byte_identical:
      sha256(selectedModel) === sha256(replayModel)
      && sha256(selectedOptimizer) === sha256(replayOptimizer)
      && replayTrace.training.optimizer_steps === selectedSteps
      && replayTrace.training.total_optimizer_step === selectedSteps
      && replayTrace.cursor.start_epoch === 0
      && replayTrace.cursor.start_window === 0
      && replayTrace.cursor.next_epoch === selectedNextEpoch
      && replayTrace.cursor.next_window === selectedNextWindow
      && replayTrace.cursor.schedule_complete === (selectedInterval === 7)
      && replayTrace.health.gradient_saturation_count === 0
      && (replayTrace.health.residual_saturation_count ?? 0) === 0
      && replayTrace.health.weight_saturation_count === 0,
    test_split_binding_and_single_open_policy_match:
      contract.selection_policy.test_split_opened_after_selection === true
      && contract.selection_policy.test_evaluations.source === 1
      && contract.selection_policy.test_evaluations.selected_candidate === 1
      && sourceTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
      && selectedTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
      && sourceTest.evaluation.context_tokens === schedule.context_tokens
      && selectedTest.evaluation.context_tokens === schedule.context_tokens
      && sourceTest.evaluation.windows === schedule.test_windows
      && selectedTest.evaluation.windows === schedule.test_windows
      && sourceTest.model_hash === contract.source_artifacts.model_hash
      && selectedTest.model_hash === selectedTrace.hashes.final_model
      && sourceTest.health.residual_saturation_count === 0
      && selectedTest.health.residual_saturation_count === 0,
    selected_residual_analysis_complete: finalGroupsComplete
      && residualAnalysis.source.optimizer_sha256 === sha256(selectedOptimizer)
      && residualAnalysis.source.optimizer_state_hash === selectedTrace.hashes.optimizer_state,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const discoveryGates = {
    selected_dev_total_millibits_beats_source_by_required_margin:
      selectedDev.total_millibits
        <= contract.quality_gates.discovery.candidate_dev_total_millibits_max,
    selected_dev_mean_millibits_beats_source:
      selectedDev.mean_millibits
        <= contract.quality_gates.discovery.candidate_dev_mean_millibits_max,
    selected_dev_mean_improvement_meets_minimum:
      contract.source_dev.mean_millibits - selectedDev.mean_millibits
        >= contract.quality_gates.discovery.minimum_mean_millibit_improvement_vs_source,
  };
  const sourceTestEval = sourceTest.evaluation;
  const selectedTestEval = selectedTest.evaluation;
  const confirmationGates = {
    selected_test_total_millibits_beats_source_by_required_margin:
      selectedTestEval.total_millibits
        <= sourceTestEval.total_millibits
          - contract.quality_gates.confirmation.candidate_test_total_millibits_margin,
    selected_test_mean_millibits_beats_source:
      selectedTestEval.mean_millibits <= sourceTestEval.mean_millibits
        - contract.quality_gates.confirmation.minimum_test_mean_millibit_improvement_vs_source,
    selected_test_mean_improvement_meets_minimum:
      sourceTestEval.mean_millibits - selectedTestEval.mean_millibits
        >= contract.quality_gates.confirmation.minimum_test_mean_millibit_improvement_vs_source,
  };
  const safetyEligible = Object.values(safetyGates).every(Boolean);
  const discoveryPassed = safetyEligible && Object.values(discoveryGates).every(Boolean);
  const confirmationPassed = discoveryPassed
    && Object.values(confirmationGates).every(Boolean);
  const outcome = outcomeFor(safetyEligible, discoveryPassed, confirmationPassed);

  return {
    schema: "nsrl.production_up_shift_breakthrough_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_up_useful_update: contract.source_up_useful_update,
    source_artifacts: contract.source_artifacts,
    bindings: { ...contract.bindings, observed_sha256: inputHashes },
    initialization: init,
    candidate: {
      ...contract.candidate,
      learning_rate_shifts: shifts,
      actual_update_count: upUpdateCount,
      first_movement_interval: firstUpMovementInterval,
      first_movement_window: firstUpMovementInterval < 0
        ? null
        : (firstUpMovementInterval + 1) * schedule.interval_windows,
    },
    selection: {
      policy: contract.selection_policy,
      selected_interval: selectedInterval,
      selected_window: (selectedInterval + 1) * schedule.interval_windows,
      selected_optimizer_steps: selectedSteps,
      selected_model_hash: selectedTrace.hashes.final_model,
      selected_optimizer_state_hash: selectedTrace.hashes.optimizer_state,
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
    evaluation: {
      lane_initial_dev: devInitial.evaluation,
      source_dev: contract.source_dev,
      selected_dev: selectedDev,
      selected_dev_total_millibits_delta_vs_source:
        selectedDev.total_millibits - contract.source_dev.total_millibits,
      selected_dev_mean_millibits_delta_vs_source:
        selectedDev.mean_millibits - contract.source_dev.mean_millibits,
      source_test: sourceTestEval,
      selected_test: selectedTestEval,
      selected_test_total_millibits_delta_vs_source:
        selectedTestEval.total_millibits - sourceTestEval.total_millibits,
      selected_test_mean_millibits_delta_vs_source:
        selectedTestEval.mean_millibits - sourceTestEval.mean_millibits,
    },
    replay: {
      selected_model_sha256: sha256(selectedModel),
      replay_model_sha256: sha256(replayModel),
      selected_optimizer_sha256: sha256(selectedOptimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    selected_residual_analysis: residualAnalysis,
    artifacts: {
      model: { bytes: selectedModel.length, sha256: sha256(selectedModel) },
      optimizer: { bytes: selectedOptimizer.length, sha256: sha256(selectedOptimizer) },
    },
    safety_gates: safetyGates,
    discovery_gates: discoveryGates,
    confirmation_gates: confirmationGates,
    safety_eligible: safetyEligible,
    discovery_passed: discoveryPassed,
    confirmation_passed: confirmationPassed,
    promotion_eligible: confirmationPassed,
    outcome,
    paid_scale_authorized: false,
    next_gate: nextGateFor(outcome),
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
const expectedDiscovery = expectedSafety
  && Object.values(checkpoint.discovery_gates ?? {}).every(Boolean);
const expectedConfirmation = expectedDiscovery
  && Object.values(checkpoint.confirmation_gates ?? {}).every(Boolean);
const expectedOutcome = outcomeFor(expectedSafety, expectedDiscovery, expectedConfirmation);
if (checkpoint.schema !== "nsrl.production_up_shift_breakthrough_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.candidate?.group !== "up"
  || checkpoint.candidate?.candidate_shift !== 22
  || checkpoint.selection?.selected_interval < 0
  || checkpoint.selection?.selected_interval > 7
  || checkpoint.safety_eligible !== expectedSafety
  || checkpoint.discovery_passed !== expectedDiscovery
  || checkpoint.confirmation_passed !== expectedConfirmation
  || checkpoint.promotion_eligible !== expectedConfirmation
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production up shift22 breakthrough checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production up shift22 breakthrough checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_up_shift_breakthrough_check.v1",
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
