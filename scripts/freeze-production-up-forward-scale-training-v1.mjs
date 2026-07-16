#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-up-forward-scale-training";
let outPath = "benchmarks/production-model-v1/p10m-up-forward-scale-training.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-up-forward-scale-training-contract.json";
const groups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];
const sortedGroups = [...groups].sort();
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const readRunJson = async (name) => readJson(path.join(runDir, name));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const validCounts = (value) => value
  && sameJson(Object.keys(value).sort(), sortedGroups)
  && Object.values(value).every((count) => Number.isSafeInteger(count) && count >= 0);

function outcomeFor(safetyEligible, qualityDiscovery) {
  if (!safetyEligible) return "failed_safety";
  return qualityDiscovery
    ? "dev_quality_discovery"
    : "safe_functional_training_without_dev_gain";
}

function nextGateFor(outcome) {
  if (outcome === "dev_quality_discovery") {
    return "p10m_up_forward_scale_hidden_confirmation_contract";
  }
  if (outcome === "safe_functional_training_without_dev_gain") {
    return "p10m_target_probability_resolution_review";
  }
  return "p10m_up_forward_scale_training_safety_review";
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, tokenizerBytes, trainBytes,
    devBytes, init, initialModel, devInitial, replayTrace, replayModel,
    replayOptimizer, residualAnalysis, ...rows] = await Promise.all([
    readFile(contract.source_sensitivity.contract_path),
    readFile(contract.source_sensitivity.checkpoint_path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.train_tokens_path),
    readFile(contract.bindings.dev_tokens_path),
    readRunJson("init.json"),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readRunJson("dev-initial.json"),
    readRunJson("replay.json"),
    readFile(path.join(runDir, "replay-selected.nsrlpm")),
    readFile(path.join(runDir, "replay-selected.nsrlpo")),
    readRunJson("residual-analysis-selected.json"),
    ...Array.from({ length: 4 }, (_, interval) => [
      readRunJson(`train-${interval}.json`),
      readRunJson(`dev-${interval}.json`),
      readRunJson(`event-${interval}.json`),
    ]).flat(),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const traces = [];
  const dev = [];
  const events = [];
  for (let interval = 0; interval < 4; interval += 1) {
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
  const movedByUpdate = (trace) => groups
    .filter((group) => trace.diagnostics.update_nonzero_count[group] > 0).sort();
  const exactMovement = traces.every((trace) => {
    if (!validCounts(trace.diagnostics.update_nonzero_count)
      || !validCounts(trace.movement_l1)) return false;
    const updates = movedByUpdate(trace);
    const movement = groups.filter((group) => trace.movement_l1[group] > 0).sort();
    return sameJson(updates, movement)
      && sameJson(updates, [...trace.moved_parameter_groups].sort())
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
      && sameJson(trace.training.learning_rate_shifts, schedule.learning_rate_shifts)
      && sameJson(trace.forward_shifts, schedule.forward_shifts)
      && trace.training.output_backward_shift === schedule.output_backward_shift
      && trace.cursor.start_window === interval * schedule.interval_windows
      && trace.cursor.next_window === (interval === 3 ? 0 : expectedNextWindow)
      && trace.cursor.next_epoch === (interval === 3 ? 1 : 0)
      && trace.cursor.schedule_complete === (interval === 3);
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
  const selectedSteps = (selectedInterval + 1) * schedule.interval_optimizer_steps;
  const selectedNextWindow = selectedInterval === 3
    ? 0
    : (selectedInterval + 1) * schedule.interval_windows;
  const selectedNextEpoch = selectedInterval === 3 ? 1 : 0;
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
    source_sensitivity_matches:
      sha256(sourceContractBytes) === contract.source_sensitivity.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_sensitivity.checkpoint_sha256
      && sourceCheckpoint.diagnostic_eligible === true
      && sourceCheckpoint.outcome === contract.source_sensitivity.required_outcome
      && sourceCheckpoint.selection.selected_up_forward_shift
        === contract.source_sensitivity.selected_up_forward_shift,
    bound_input_hashes_match:
      inputHashes.tokenizer_sha256 === contract.bindings.tokenizer_sha256
      && inputHashes.train_tokens_sha256 === contract.bindings.train_tokens_sha256
      && inputHashes.dev_tokens_sha256 === contract.bindings.dev_tokens_sha256,
    deterministic_initialization_and_forward_scale_match:
      init.model_hash === contract.initialization.model_hash
      && init.initialization_seed === contract.initialization.seed
      && init.output_init_amplitude === contract.initialization.output_init_amplitude
      && init.output_forward_shift === contract.initialization.output_forward_shift
      && init.up_forward_shift === contract.initialization.up_forward_shift
      && sha256(initialModel) === contract.initialization.artifact_sha256,
    four_interval_schedule_and_chain_exact: scheduleExact && modelChainExact,
    exact_reachable_update_consistency: exactMovement,
    all_13_gradient_paths_active: fullGradientPath,
    all_saturation_zero: exactHealth,
    all_intervals_live: events.every((event) =>
      event.dead === false && event.full_gradient_path === true),
    up_moves_by_window_512: upUpdateCount > 0
      && events[1].required_trunk_group_observations?.up === true,
    heldout_nonincreasing_vs_lane_initial: dev.every((row) =>
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
      && sameJson(replayTrace.forward_shifts, schedule.forward_shifts)
      && replayTrace.cursor.start_window === 0
      && replayTrace.cursor.next_window === selectedNextWindow
      && replayTrace.cursor.next_epoch === selectedNextEpoch
      && replayTrace.cursor.schedule_complete === (selectedInterval === 3)
      && replayTrace.health.gradient_saturation_count === 0
      && (replayTrace.health.residual_saturation_count ?? 0) === 0
      && replayTrace.health.weight_saturation_count === 0,
    selected_residual_analysis_complete: finalGroupsComplete
      && residualAnalysis.source.optimizer_sha256 === sha256(selectedOptimizer)
      && residualAnalysis.source.optimizer_state_hash === selectedTrace.hashes.optimizer_state,
    test_split_not_accessed: contract.selection_policy.test_split_access === false,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const qualityGates = {
    selected_dev_total_millibits_beats_source_by_required_margin:
      selectedDev.total_millibits <= contract.quality_gates.candidate_dev_total_millibits_max,
    selected_dev_mean_millibits_beats_source:
      selectedDev.mean_millibits <= contract.quality_gates.candidate_dev_mean_millibits_max,
    selected_dev_mean_improvement_meets_minimum:
      contract.source_dev.mean_millibits - selectedDev.mean_millibits
        >= contract.quality_gates.minimum_mean_millibit_improvement_vs_source,
  };
  const safetyEligible = Object.values(safetyGates).every(Boolean);
  const qualityDiscovery = safetyEligible && Object.values(qualityGates).every(Boolean);
  const outcome = outcomeFor(safetyEligible, qualityDiscovery);
  return {
    schema: "nsrl.production_forward_scale_training_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_sensitivity: contract.source_sensitivity,
    bindings: { ...contract.bindings, observed_sha256: inputHashes },
    initialization: init,
    training_candidate: {
      up_learning_rate_shift: schedule.learning_rate_shifts.up,
      up_forward_shift: schedule.forward_shifts.up,
      actual_up_update_count: upUpdateCount,
      first_up_movement_interval: firstUpMovementInterval,
      first_up_movement_window: firstUpMovementInterval < 0
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
      moved_parameter_groups: trace.moved_parameter_groups,
      update_nonzero_count: trace.diagnostics.update_nonzero_count,
      movement_l1: trace.movement_l1,
      forward_shifts: trace.forward_shifts,
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
    },
    replay: {
      selected_model_sha256: sha256(selectedModel),
      replay_model_sha256: sha256(replayModel),
      selected_optimizer_sha256: sha256(selectedOptimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    selected_residual_analysis: residualAnalysis,
    safety_gates: safetyGates,
    quality_gates: qualityGates,
    safety_eligible: safetyEligible,
    quality_discovery: qualityDiscovery,
    promotion_eligible: false,
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
const expectedQuality = expectedSafety
  && Object.values(checkpoint.quality_gates ?? {}).every(Boolean);
const expectedOutcome = outcomeFor(expectedSafety, expectedQuality);
if (checkpoint.schema !== "nsrl.production_forward_scale_training_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.training_candidate?.up_learning_rate_shift !== 22
  || checkpoint.training_candidate?.up_forward_shift !== 7
  || checkpoint.selection?.selected_interval < 0
  || checkpoint.selection?.selected_interval > 3
  || checkpoint.safety_eligible !== expectedSafety
  || checkpoint.quality_discovery !== expectedQuality
  || checkpoint.promotion_eligible !== false
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production up forward-scale training checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production up forward-scale training checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_forward_scale_training_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    quality_discovery: checkpoint.quality_discovery,
  }));
}
