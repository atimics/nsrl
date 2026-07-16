#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const args = process.argv.slice(2);
const checkOnly = args.includes("--check");
const valueAfter = (flag, fallback) => {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] : fallback;
};
const runDir = valueAfter(
  "--run-dir",
  "data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight",
);
const outPath = valueAfter(
  "--out",
  "benchmarks/production-model-v1/p10m-normalized-wide-gradient-preflight.json",
);
const contractPath = "benchmarks/production-model-v1/p10m-normalized-wide-gradient-preflight-contract.json";
const legacyQ23OptimizerPath =
  "data/experiments/production-model-v1/p10m-wide-probability-gradient-preflight/optimizer-q23.nsrlpo";

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (path) => JSON.parse(await readFile(path, "utf8"));
const allGroups = (values) => values && Object.keys(values).length === 13;
const zeroHealth = (trace) => trace.health.gradient_saturation_count === 0
  && trace.health.residual_saturation_count === 0
  && trace.health.weight_saturation_count === 0
  && allGroups(trace.diagnostics.saturation_by_group)
  && Object.values(trace.diagnostics.saturation_by_group).every((value) => value === 0)
  && allGroups(trace.diagnostics.residual_saturation_by_group)
  && Object.values(trace.diagnostics.residual_saturation_by_group).every((value) => value === 0);

async function loadLane(suffix) {
  const [model, optimizer, trace, dev, residual] = await Promise.all([
    readFile(`${runDir}/model-q23-newton${suffix}.nsrlpm`),
    readFile(`${runDir}/optimizer-q23-newton${suffix}.nsrlpo`),
    readJson(`${runDir}/train-q23-newton${suffix}.json`),
    readJson(`${runDir}/dev-q23-newton${suffix}.json`),
    readJson(`${runDir}/residual-q23-newton${suffix}.json`),
  ]);
  return { model, optimizer, trace, dev, residual };
}

async function buildCheckpoint() {
  const [
    contractBytes,
    contract,
    sourceAttributionBytes,
    tokenizerBytes,
    trainBytes,
    devBytes,
    initialModel,
    legacyQ23Optimizer,
    control,
    upBoundary,
    outputBoundary,
    upComparison,
    outputComparison,
    replayModel,
    replayOptimizer,
    replayTrace,
  ] = await Promise.all([
    readFile(contractPath),
    readJson(contractPath),
    readFile("benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json"),
    readFile("data/processed/production-corpus-v1/tokenizer.nsrlbpe"),
    readFile("data/processed/production-corpus-v1/train.nsrltok"),
    readFile("data/processed/production-corpus-v1/dev.nsrltok"),
    readFile("data/experiments/production-model-v1/p10m-up-forward-scale-training/initial.nsrlpm"),
    readFile(legacyQ23OptimizerPath),
    loadLane(""),
    loadLane("-up21"),
    loadLane("-output33"),
    readJson(`${runDir}/compare-q23-newton-up21.json`),
    readJson(`${runDir}/compare-q23-newton-output33.json`),
    readFile(`${runDir}/replay-selected.nsrlpm`),
    readFile(`${runDir}/replay-selected.nsrlpo`),
    readJson(`${runDir}/replay.json`),
  ]);

  const sourceAttribution = JSON.parse(sourceAttributionBytes);
  const lanes = [control, upBoundary, outputBoundary];
  const scheduleMatches = lanes.every(({ trace }) =>
    trace.schema === "nsrl.production_full_train_smoke.v1"
      && trace.profile === contract.profile
      && trace.parameter_count === contract.parameter_count
      && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
      && trace.training.context_tokens === contract.schedule.context_tokens
      && trace.training.windows === contract.schedule.train_windows
      && trace.training.optimizer_steps === contract.schedule.optimizer_steps
      && trace.training.probability_gradient_fractional_bits
        === contract.schedule.probability_gradient_fractional_bits
      && trace.training.probability_normalization
        === contract.schedule.probability_normalization
      && trace.training.output_backward_shift === contract.schedule.output_backward_shift);
  const devMatches = lanes.every(({ dev }) =>
    dev.schema === "nsrl.production_model_eval.v1"
      && dev.profile === contract.profile
      && dev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && dev.evaluation.context_tokens === contract.schedule.context_tokens
      && dev.evaluation.windows === contract.schedule.dev_windows);
  const gradientsActive = lanes.every(({ trace }) =>
    allGroups(trace.diagnostics.gradient_nonzero_count)
      && Object.values(trace.diagnostics.gradient_nonzero_count).every((value) => value > 0));
  const allSaturationZero = lanes.every(({ trace }) => zeroHealth(trace));
  const normalizedControlModelMatchesQ15 = sha256(control.model) === contract.q15_control.model_sha256;
  const normalizedControlOptimizerDiffersFromLegacy =
    sha256(control.optimizer) !== sha256(legacyQ23Optimizer);
  const upMaterialized = upBoundary.trace.training.learning_rate_shifts.up === 21
    && upBoundary.trace.movement_l1.up > 0
    && upComparison.functional_delta.feature_changed_windows > 0
    && upComparison.functional_delta.logits_changed_windows > 0;
  const outputRecoveredTargetSignal =
    outputBoundary.trace.training.learning_rate_shifts.output === 33
      && outputBoundary.trace.movement_l1.output > control.trace.movement_l1.output
      && outputComparison.functional_delta.target_logit_changed_windows > 0
      && outputComparison.functional_delta.target_probability_changed_windows > 0;
  const noDevGain = lanes.every(({ dev }) =>
    dev.evaluation.total_millibits >= contract.q15_control.dev_total_millibits);
  const exactReplay = sha256(control.model) === sha256(replayModel)
    && sha256(control.optimizer) === sha256(replayOptimizer)
    && replayTrace.training.probability_normalization === "q47_newton1";

  const safetyGates = {
    source_normalization_attribution_matches:
      sha256(sourceAttributionBytes)
        === contract.source_normalization_attribution.checkpoint_sha256
      && sourceAttribution.outcome
        === contract.source_normalization_attribution.required_outcome
      && sourceAttribution.selection.selected_normalization
        === contract.source_normalization_attribution.selected_normalization,
    bound_inputs_and_initial_model_match:
      sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(trainBytes) === contract.bindings.train_tokens_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256
      && sha256(initialModel) === contract.initialization.sha256,
    schedules_and_dev_shapes_match: scheduleMatches && devMatches,
    all_13_gradient_paths_active: gradientsActive,
    all_saturation_zero: allSaturationZero,
    normalization_changes_optimizer_not_materialized_control:
      normalizedControlModelMatchesQ15 && normalizedControlOptimizerDiffersFromLegacy,
    normalized_control_exact_replay: exactReplay,
    up_boundary_materializes_functional_model_signal: upMaterialized,
    output_boundary_recovers_target_probability_signal: outputRecoveredTargetSignal,
    test_split_not_accessed: contract.gates.test_split_access === false,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false,
  };
  const safetyEligible = Object.values(safetyGates).every(Boolean);
  const qualityGain = safetyEligible && !noDevGain;
  const outcome = safetyEligible
    ? qualityGain
      ? "integer_precision_and_dev_quality_recovered"
      : "integer_precision_recovered_without_dev_gain"
    : "failed_safety";

  const laneSummary = (lane) => ({
    training: lane.trace.training,
    movement_l1: lane.trace.movement_l1,
    moved_parameter_groups: lane.trace.moved_parameter_groups,
    health: lane.trace.health,
    dev: lane.dev.evaluation,
    model_sha256: sha256(lane.model),
    optimizer_sha256: sha256(lane.optimizer),
    residual_recommendation: lane.residual.recommendation,
  });

  return {
    schema: "nsrl.production_normalized_wide_gradient_preflight_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    bindings: contract.bindings,
    initialization: contract.initialization,
    numeric_path: {
      probability_fractional_bits: 23,
      normalization: "q47_newton1",
      reciprocal_fractional_bits: 47,
      integer_newton_steps: 1,
      optimizer_residual_type: "i64",
    },
    lanes: {
      normalized_control: laneSummary(control),
      up_boundary: {
        ...laneSummary(upBoundary),
        functional_delta_vs_control: upComparison.functional_delta,
        quality_vs_control: upComparison.quality,
      },
      output_boundary: {
        ...laneSummary(outputBoundary),
        functional_delta_vs_control: outputComparison.functional_delta,
        quality_vs_control: outputComparison.quality,
      },
    },
    precision_effect: {
      normalized_control_model_byte_identical_to_q15:
        normalizedControlModelMatchesQ15,
      normalized_control_optimizer_differs_from_legacy_q23:
        normalizedControlOptimizerDiffersFromLegacy,
      up_boundary_materialized_updates: upBoundary.trace.movement_l1.up,
      up_boundary_feature_changed_windows:
        upComparison.functional_delta.feature_changed_windows,
      output_boundary_target_logit_changed_windows:
        outputComparison.functional_delta.target_logit_changed_windows,
      output_boundary_target_probability_changed_windows:
        outputComparison.functional_delta.target_probability_changed_windows,
      classification: "precision_reaches_target_probability_at_output_boundary",
    },
    quality_effect: {
      normalized_control_dev_total_millibits_delta:
        control.dev.evaluation.total_millibits - contract.q15_control.dev_total_millibits,
      up_boundary_dev_total_millibits_delta:
        upBoundary.dev.evaluation.total_millibits - contract.q15_control.dev_total_millibits,
      output_boundary_dev_total_millibits_delta:
        outputBoundary.dev.evaluation.total_millibits - contract.q15_control.dev_total_millibits,
      classification: noDevGain ? "no_dev_gain" : "dev_gain",
    },
    replay: {
      selected_model_sha256: sha256(control.model),
      replay_model_sha256: sha256(replayModel),
      selected_optimizer_sha256: sha256(control.optimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    selection: {
      selected_probability_fractional_bits: 23,
      selected_normalization: "q47_newton1",
      selected_up_learning_rate_shift: 22,
      selected_output_learning_rate_shift: 34,
      aggressive_materialization_boundary_promoted: false,
    },
    safety_gates: safetyGates,
    safety_eligible: safetyEligible,
    integer_precision_bottleneck_resolved: safetyEligible,
    dev_quality_gain: qualityGain,
    promotion_eligible: false,
    outcome,
    paid_scale_authorized: false,
    next_gate: safetyEligible && !qualityGain
      ? "p10m_target_aligned_integer_objective_review"
      : "p10m_normalized_wide_gradient_safety_review",
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
const contractBytes = await readFile(contractPath);
const expectedSafety = Object.values(checkpoint.safety_gates ?? {}).every(Boolean);
if (checkpoint.schema
    !== "nsrl.production_normalized_wide_gradient_preflight_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(contractBytes)
  || checkpoint.safety_eligible !== expectedSafety
  || checkpoint.integer_precision_bottleneck_resolved !== expectedSafety
  || checkpoint.outcome !== "integer_precision_recovered_without_dev_gain"
  || checkpoint.dev_quality_gain !== false
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== "p10m_target_aligned_integer_objective_review") {
  throw new Error("normalized wide-gradient preflight checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("normalized wide-gradient preflight checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_normalized_wide_gradient_preflight_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
    target_probability_changed_windows:
      checkpoint.precision_effect.output_boundary_target_probability_changed_windows,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    target_probability_changed_windows:
      checkpoint.precision_effect.output_boundary_target_probability_changed_windows,
  }));
}
