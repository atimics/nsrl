#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-wide-probability-gradient-preflight";
let outPath = "benchmarks/production-model-v1/p10m-wide-probability-gradient-preflight.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-wide-probability-gradient-preflight-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const groups = ["embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias"];

function outcomeFor(safetyEligible, qualityGain) {
  if (!safetyEligible) return "failed_safety";
  return qualityGain ? "wide_precision_preflight_gain" : "wide_precision_no_preflight_gain";
}

function nextGateFor(outcome) {
  if (outcome === "wide_precision_preflight_gain") {
    return "p10m_wide_probability_gradient_training_contract";
  }
  if (outcome === "wide_precision_no_preflight_gain") {
    return "p10m_probability_normalization_accuracy_review";
  }
  return "p10m_wide_probability_gradient_safety_review";
}

function validGroupCounts(value) {
  return value && sameJson(Object.keys(value).sort(), [...groups].sort())
    && Object.values(value).every((count) => Number.isSafeInteger(count) && count >= 0);
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const bits = contract.schedule.candidate_probability_gradient_fractional_bits;
  const [sourceContractBytes, sourceCheckpointBytes, q15CheckpointBytes, q15Model,
    q15Optimizer, tokenizerBytes, trainBytes, devBytes, initialModel, ...artifacts] = await Promise.all([
    readFile(contract.source_resolution.contract_path),
    readFile(contract.source_resolution.checkpoint_path),
    readFile(contract.q15_control.checkpoint_path),
    readFile(contract.q15_control.model_path),
    readFile(contract.q15_control.optimizer_path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.train_tokens_path),
    readFile(contract.bindings.dev_tokens_path),
    readFile(contract.initialization.path),
    ...bits.flatMap((fractionalBits) => [
      readJson(path.join(runDir, `train-q${fractionalBits}.json`)),
      readJson(path.join(runDir, `dev-q${fractionalBits}.json`)),
      readFile(path.join(runDir, `model-q${fractionalBits}.nsrlpm`)),
      readFile(path.join(runDir, `optimizer-q${fractionalBits}.nsrlpo`)),
      readJson(path.join(runDir, `residual-q${fractionalBits}.json`)),
    ]),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const q15Checkpoint = JSON.parse(q15CheckpointBytes);
  const rows = bits.map((fractionalBits, index) => ({
    fractionalBits,
    trace: artifacts[index * 5],
    dev: artifacts[index * 5 + 1],
    model: artifacts[index * 5 + 2],
    optimizer: artifacts[index * 5 + 3],
    residualAnalysis: artifacts[index * 5 + 4],
  }));
  const selected = [...rows].sort((left, right) =>
    left.dev.evaluation.total_millibits - right.dev.evaluation.total_millibits
      || left.fractionalBits - right.fractionalBits)[0];
  const [replayTrace, replayModel, replayOptimizer] = await Promise.all([
    readJson(path.join(runDir, "replay.json")),
    readFile(path.join(runDir, "replay-selected.nsrlpm")),
    readFile(path.join(runDir, "replay-selected.nsrlpo")),
  ]);
  const schedule = contract.schedule;
  const traceShapeExact = rows.every(({ fractionalBits, trace }) => {
    const delta = fractionalBits - 15;
    return trace.schema === "nsrl.production_full_train_smoke.v1"
      && trace.profile === contract.profile
      && trace.parameter_count === contract.parameter_count
      && trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash
      && trace.training.context_tokens === schedule.context_tokens
      && trace.training.windows === schedule.train_windows
      && trace.training.evaluation_windows === schedule.evaluation_windows
      && trace.training.epochs === schedule.epochs
      && trace.training.batch_windows === schedule.batch_windows
      && trace.training.optimizer_steps === schedule.optimizer_steps
      && trace.training.total_optimizer_step === schedule.optimizer_steps
      && sameJson(trace.training.learning_rate_shifts, schedule.learning_rate_shifts)
      && trace.training.output_backward_shift === schedule.output_backward_shift
      && trace.training.probability_gradient_fractional_bits === fractionalBits
      && trace.training.probability_gradient_shift_delta === delta
      && trace.training.effective_output_backward_shift
        === schedule.output_backward_shift + delta
      && trace.training.effective_output_learning_rate_shift
        === schedule.learning_rate_shifts.output + delta
      && trace.training.effective_bias_learning_rate_shift
        === schedule.learning_rate_shifts.bias + delta
      && sameJson(trace.forward_shifts, schedule.forward_shifts)
      && trace.cursor.start_window === 0
      && trace.cursor.next_window === schedule.processed_windows
      && trace.cursor.next_epoch === 0
      && trace.cursor.schedule_complete === false;
  });
  const devShapeExact = rows.every(({ dev }) =>
    dev.schema === "nsrl.production_model_eval.v1"
      && dev.profile === contract.profile
      && dev.parameter_count === contract.parameter_count
      && dev.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && dev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && dev.evaluation.context_tokens === schedule.context_tokens
      && dev.evaluation.windows === schedule.dev_windows);
  const safetyGates = {
    source_resolution_matches:
      sha256(sourceContractBytes) === contract.source_resolution.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_resolution.checkpoint_sha256
      && sourceCheckpoint.outcome === contract.source_resolution.required_outcome
      && sourceCheckpoint.selection.selected_fractional_bits
        === contract.source_resolution.minimum_detectable_fractional_bits
      && sourceCheckpoint.precision_rows.find(({ fractional_bits }) => fractional_bits === 23)
        .delta.target_probability_changed_windows
        === sourceCheckpoint.precision_rows.find(({ fractional_bits }) => fractional_bits === 31)
          .delta.target_probability_changed_windows,
    q15_control_matches:
      sha256(q15CheckpointBytes) === contract.q15_control.checkpoint_sha256
      && sha256(q15Model) === contract.q15_control.model_sha256
      && sha256(q15Optimizer) === contract.q15_control.optimizer_sha256
      && q15Checkpoint.intervals[0].dev.total_millibits
        === contract.q15_control.window_256_dev_total_millibits
      && q15Checkpoint.intervals[0].dev.mean_millibits
        === contract.q15_control.window_256_dev_mean_millibits,
    bound_inputs_and_initial_model_match:
      sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(trainBytes) === contract.bindings.train_tokens_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256
      && sha256(initialModel) === contract.initialization.sha256,
    predeclared_candidates_and_schedule_exact:
      sameJson(bits, [19, 23]) && traceShapeExact && devShapeExact,
    scale_compensation_exact: rows.every(({ fractionalBits, trace }) => {
      const delta = fractionalBits - 15;
      return trace.training.effective_output_backward_shift
          - trace.training.output_backward_shift === delta
        && trace.training.effective_output_learning_rate_shift
          - trace.training.learning_rate_shifts.output === delta
        && trace.training.effective_bias_learning_rate_shift
          - trace.training.learning_rate_shifts.bias === delta;
    }),
    all_13_gradient_paths_active: rows.every(({ trace }) =>
      validGroupCounts(trace.diagnostics.gradient_nonzero_count)
      && groups.every((group) => trace.diagnostics.gradient_nonzero_count[group] > 0)),
    all_saturation_zero: rows.every(({ trace }) =>
      trace.health.gradient_saturation_count === 0
      && trace.health.residual_saturation_count === 0
      && trace.health.weight_saturation_count === 0
      && validGroupCounts(trace.diagnostics.saturation_by_group)
      && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
      && validGroupCounts(trace.diagnostics.residual_saturation_by_group)
      && Object.values(trace.diagnostics.residual_saturation_by_group)
        .every((count) => count === 0)),
    movement_and_update_accounting_exact: rows.every(({ trace }) =>
      validGroupCounts(trace.movement_l1)
      && validGroupCounts(trace.diagnostics.update_nonzero_count)
      && groups.every((group) =>
        (trace.movement_l1[group] > 0)
          === (trace.diagnostics.update_nonzero_count[group] > 0))
      && sameJson([...trace.moved_parameter_groups].sort(), groups
        .filter((group) => trace.movement_l1[group] > 0).sort())),
    selection_policy_applied_exactly:
      selected.dev.evaluation.total_millibits
        === Math.min(...rows.map(({ dev }) => dev.evaluation.total_millibits))
      && !rows.some(({ fractionalBits, dev }) =>
        dev.evaluation.total_millibits === selected.dev.evaluation.total_millibits
        && fractionalBits < selected.fractionalBits),
    selected_checkpoint_replay_byte_identical:
      replayTrace.training.probability_gradient_fractional_bits === selected.fractionalBits
      && replayTrace.training.optimizer_steps === schedule.optimizer_steps
      && sha256(selected.model) === sha256(replayModel)
      && sha256(selected.optimizer) === sha256(replayOptimizer),
    test_split_not_accessed:
      contract.selection_policy.test_split_access === false
      && rows.every(({ dev }) =>
        dev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash),
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const qualityGates = {
    selected_dev_total_millibits_beats_q15_control:
      selected.dev.evaluation.total_millibits
        <= contract.quality_gates.selected_dev_total_millibits_below_q15_control,
    selected_dev_mean_millibits_at_most_q15_control:
      selected.dev.evaluation.mean_millibits
        <= contract.quality_gates.selected_dev_mean_millibits_at_most_q15_control,
  };
  const safetyEligible = Object.values(safetyGates).every(Boolean);
  const qualityGain = safetyEligible && Object.values(qualityGates).every(Boolean);
  const outcome = outcomeFor(safetyEligible, qualityGain);
  return {
    schema: "nsrl.production_wide_probability_gradient_preflight_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_resolution: contract.source_resolution,
    q15_control: contract.q15_control,
    bindings: contract.bindings,
    initialization: contract.initialization,
    precision_effect: {
      classification: rows.every(({ model }) => sha256(model) === sha256(q15Model))
        && rows.every(({ optimizer }) => sha256(optimizer) !== sha256(q15Optimizer))
        ? "wide_probability_information_residual_only_at_256_windows"
        : "wide_probability_information_reaches_model_at_256_windows",
      all_candidate_models_byte_identical_to_q15_control:
        rows.every(({ model }) => sha256(model) === sha256(q15Model)),
      all_candidate_optimizers_differ_from_q15_control:
        rows.every(({ optimizer }) => sha256(optimizer) !== sha256(q15Optimizer)),
      q15_model_sha256: sha256(q15Model),
      q15_optimizer_sha256: sha256(q15Optimizer),
    },
    candidates: rows.map(({ fractionalBits, trace, dev, model, optimizer,
      residualAnalysis }) => ({
      probability_gradient_fractional_bits: fractionalBits,
      training: trace.training,
      forward_shifts: trace.forward_shifts,
      moved_parameter_groups: trace.moved_parameter_groups,
      movement_l1: trace.movement_l1,
      update_nonzero_count: trace.diagnostics.update_nonzero_count,
      health: trace.health,
      dev: dev.evaluation,
      model_sha256: sha256(model),
      optimizer_sha256: sha256(optimizer),
      model_byte_identical_to_q15_control: sha256(model) === sha256(q15Model),
      optimizer_byte_identical_to_q15_control: sha256(optimizer) === sha256(q15Optimizer),
      residual_analysis: {
        output: residualAnalysis.groups.find(({ group }) => group === "output"),
        bias: residualAnalysis.groups.find(({ group }) => group === "bias"),
        recommendation: residualAnalysis.recommendation,
      },
    })),
    selection: {
      policy: contract.selection_policy,
      selected_fractional_bits: selected.fractionalBits,
      selected_dev: selected.dev.evaluation,
      selected_dev_total_millibits_delta_vs_q15_control:
        selected.dev.evaluation.total_millibits
          - contract.q15_control.window_256_dev_total_millibits,
      selected_dev_mean_millibits_delta_vs_q15_control:
        selected.dev.evaluation.mean_millibits
          - contract.q15_control.window_256_dev_mean_millibits,
    },
    replay: {
      selected_model_sha256: sha256(selected.model),
      replay_model_sha256: sha256(replayModel),
      selected_optimizer_sha256: sha256(selected.optimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    safety_gates: safetyGates,
    quality_gates: qualityGates,
    safety_eligible: safetyEligible,
    quality_gain: qualityGain,
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
const expectedQuality = expectedSafety && Object.values(checkpoint.quality_gates ?? {}).every(Boolean);
const expectedOutcome = outcomeFor(expectedSafety, expectedQuality);
if (checkpoint.schema !== "nsrl.production_wide_probability_gradient_preflight_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || !sameJson(checkpoint.candidates?.map(({ probability_gradient_fractional_bits }) =>
    probability_gradient_fractional_bits), [19, 23])
  || checkpoint.safety_eligible !== expectedSafety
  || checkpoint.quality_gain !== expectedQuality
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production wide probability-gradient preflight checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production wide probability-gradient preflight checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_wide_probability_gradient_preflight_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
    selected_fractional_bits: checkpoint.selection.selected_fractional_bits,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    selected_fractional_bits: checkpoint.selection.selected_fractional_bits,
  }));
}
