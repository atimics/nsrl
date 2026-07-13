#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-stabilized-pilot";
let outPath = "benchmarks/production-model-v1/p10m-stabilized-pilot.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${arg}`);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (name) => JSON.parse(await readFile(path.join(runDir, name), "utf8"));

function sumObjects(values) {
  const result = {};
  for (const value of values) {
    for (const [key, count] of Object.entries(value)) result[key] = (result[key] ?? 0) + count;
  }
  return result;
}

async function buildCheckpoint() {
  const [contractBytes, preflightBytes, init, integerInitial, ...rest] = await Promise.all([
    readFile("benchmarks/production-model-v1/p10m-stabilized-pilot-contract.json"),
    readFile("benchmarks/production-model-v1/p10m-stabilization.json"),
    readJson("init.json"),
    readJson("integer-dev-initial.json"),
    ...[0, 1, 2, 3].map((index) => readJson(`integer-durable-${index}.json`)),
    ...[0, 1, 2, 3].map((index) => readJson(`integer-dev-durable-${index}.json`)),
    ...[0, 1, 2, 3].map((index) => readJson(`early-stop-durable-${index}.json`)),
    ...[0, 1].map((index) => readJson(`integer-midpoint-${index}.json`)),
    ...[0, 1, 2, 3].map((index) => readJson(`float-${index}.json`)),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readFile(path.join(runDir, "integer-durable-3.nsrlpm")),
    readFile(path.join(runDir, "integer-durable-3.nsrlpo")),
    readFile(path.join(runDir, "integer-midpoint-1.nsrlpm")),
    readFile(path.join(runDir, "integer-midpoint-1.nsrlpo")),
    readFile(path.join(runDir, "float-3.npz")),
  ]);
  const contract = JSON.parse(contractBytes);
  const preflight = JSON.parse(preflightBytes);
  const durable = rest.slice(0, 4);
  const devChunks = rest.slice(4, 8);
  const earlyStops = rest.slice(8, 12);
  const midpoint = rest.slice(12, 14);
  const floatChunks = rest.slice(14, 18);
  const [initialModel, model, optimizer, midpointModel, midpointOptimizer, floatArtifact] = rest.slice(18);

  const diagnostics = Object.fromEntries([
    "gradient_nonzero_count", "residual_carry_count", "update_nonzero_count", "saturation_by_group",
  ].map((name) => [name, sumObjects(durable.map((trace) => trace.diagnostics[name]))]));
  for (const name of ["backward_ste_rescue_count", "backward_quantization_count"]) {
    diagnostics[name] = durable.reduce((total, trace) => total + trace.diagnostics[name], 0);
  }
  diagnostics.backward_ste_rescue_per_million = Math.floor(
    diagnostics.backward_ste_rescue_count * 1_000_000
      / Math.max(1, diagnostics.backward_quantization_count),
  );
  const health = {
    gradient_saturation_count: durable.reduce(
      (total, trace) => total + trace.health.gradient_saturation_count, 0,
    ),
    weight_saturation_count: durable.reduce(
      (total, trace) => total + trace.health.weight_saturation_count, 0,
    ),
  };
  const fullGradientPathByChunk = durable.map((trace) => (
    Object.keys(trace.diagnostics.gradient_nonzero_count).length === 13
      && Object.values(trace.diagnostics.gradient_nonzero_count).every((count) => count > 0)
  ));
  const movedGroups = [...new Set(durable.flatMap((trace) => trace.moved_parameter_groups))].sort();
  const floatMovedGroups = [...new Set(floatChunks.flatMap((trace) => trace.moved_parameter_groups))].sort();
  const integerFinal = devChunks[3];
  const floatEvaluation = {
    token_stream_hash: floatChunks[0].evaluation.token_stream_hash,
    context_tokens: floatChunks[0].evaluation.context_tokens,
    windows: floatChunks[0].evaluation.windows,
    initial_loss_millionths: floatChunks[0].evaluation.initial_loss_millionths,
    final_loss_millionths: floatChunks[3].evaluation.final_loss_millionths,
    initial_mean_millibits: floatChunks[0].evaluation.initial_mean_millibits,
    final_mean_millibits: floatChunks[3].evaluation.final_mean_millibits,
    initial_mistakes: floatChunks[0].evaluation.initial_mistakes,
    final_mistakes: floatChunks[3].evaluation.final_mistakes,
  };
  const integerFinalMillibits = integerFinal.evaluation.mean_millibits;
  const floatFinalMillibits = floatEvaluation.final_mean_millibits;
  const regressionPerMille = Math.round(
    (integerFinalMillibits - floatFinalMillibits) * 1000 / Math.max(1, floatFinalMillibits),
  );
  const gates = {
    source_preflight_eligible: preflight.preflight_eligible === true
      && sha256(preflightBytes) === contract.source_preflight.sha256,
    initialization_locked: init.initialization_seed === contract.initialization.seed
      && init.output_init_amplitude === contract.initialization.output_init_amplitude
      && init.output_forward_shift === contract.initialization.output_forward_shift,
    scaled_schedule_locked: JSON.stringify(durable[0].training.learning_rate_shifts)
      === JSON.stringify(contract.schedule.integer_learning_rate_shifts)
      && durable[0].training.output_backward_shift === contract.schedule.output_backward_shift,
    all_early_stop_gates_passed: earlyStops.length === 4
      && earlyStops.every((result) => result.continue_training === true),
    integer_dev_total_millibits_nonincreasing:
      integerFinal.evaluation.total_millibits <= integerInitial.evaluation.total_millibits,
    float_dev_loss_nonincreasing:
      floatEvaluation.final_mean_millibits <= floatEvaluation.initial_mean_millibits,
    integer_float_dev_regression_within_limit:
      regressionPerMille <= contract.completion_gates.integer_dev_regression_vs_float_max_per_mille,
    integer_parameter_movement: movedGroups.length > 0,
    all_float_parameter_groups_moved: floatMovedGroups.length === 13,
    integer_saturation_zero:
      health.gradient_saturation_count === 0 && health.weight_saturation_count === 0,
    integer_full_gradient_path_sustained: fullGradientPathByChunk.every(Boolean),
    midpoint_restart_byte_identical:
      sha256(model) === sha256(midpointModel) && sha256(optimizer) === sha256(midpointOptimizer),
    schedules_complete:
      durable[3].cursor.schedule_complete === true && midpoint[1].cursor.schedule_complete === true,
  };

  return {
    schema: "nsrl.production_stabilized_pilot_checkpoint.v1",
    contract,
    source_preflight_sha256: sha256(preflightBytes),
    contract_sha256: sha256(contractBytes),
    initialization: {
      seed: init.initialization_seed,
      output_init_amplitude: init.output_init_amplitude,
      output_forward_shift: init.output_forward_shift,
      model_hash: init.model_hash,
    },
    integer: {
      training: {
        optimizer: durable[0].training.optimizer,
        backward: durable[0].training.backward,
        context_tokens: durable[0].training.context_tokens,
        windows: durable[0].training.windows,
        evaluation_windows_per_chunk: durable[0].training.evaluation_windows,
        epochs: durable[0].training.epochs,
        batch_windows: durable[0].training.batch_windows,
        optimizer_steps: durable.reduce((total, trace) => total + trace.training.optimizer_steps, 0),
        initial_mistakes: durable[0].training.initial_mistakes,
        final_mistakes: durable[3].training.final_mistakes,
        learning_rate_shifts: durable[0].training.learning_rate_shifts,
        output_backward_shift: durable[0].training.output_backward_shift,
      },
      cursor: durable[3].cursor,
      chunks: durable.map((trace, index) => ({
        training: trace.training,
        cursor: trace.cursor,
        gradient_path_complete: fullGradientPathByChunk[index],
        heldout_evaluation: devChunks[index].evaluation,
        early_stop: earlyStops[index],
        diagnostics: trace.diagnostics,
        health: trace.health,
        moved_parameter_groups: trace.moved_parameter_groups,
      })),
      diagnostics,
      health,
      moved_parameter_groups: movedGroups,
      dev_initial: integerInitial.evaluation,
      dev_final: integerFinal.evaluation,
      artifacts: {
        initial_model: { bytes: initialModel.length, sha256: sha256(initialModel) },
        model: { bytes: model.length, sha256: sha256(model) },
        optimizer: { bytes: optimizer.length, sha256: sha256(optimizer) },
      },
    },
    float: {
      training: {
        context_tokens: 64,
        windows: floatChunks.reduce((total, trace) => total + trace.training.windows, 0),
        chunks: floatChunks.map((trace) => trace.training),
      },
      evaluation: floatEvaluation,
      moved_parameter_groups: floatMovedGroups,
      artifact: { bytes: floatArtifact.length, sha256: sha256(floatArtifact) },
    },
    restart: {
      midpoint_optimizer_steps: midpoint[0].training.optimizer_steps,
      resume_start_window: midpoint[1].cursor.start_window,
      durable_model_sha256: sha256(model),
      midpoint_model_sha256: sha256(midpointModel),
      durable_optimizer_sha256: sha256(optimizer),
      midpoint_optimizer_sha256: sha256(midpointOptimizer),
    },
    comparison: {
      integer_final_mean_millibits: integerFinalMillibits,
      float_final_mean_millibits: floatFinalMillibits,
      integer_regression_vs_float_per_mille: regressionPerMille,
    },
    gates,
    replay_eligible: Object.values(gates).every(Boolean),
    next_gate: Object.values(gates).every(Boolean)
      ? "p10m_trunk_unlock_preflight"
      : "p10m_stabilization_review",
  };
}

function validate(value) {
  const schedule = value.contract?.schedule;
  if (value.schema !== "nsrl.production_stabilized_pilot_checkpoint.v1"
    || value.contract?.schema !== "nsrl.production_stabilized_pilot_contract.v1"
    || schedule?.context_tokens !== 64 || schedule?.train_windows !== 1024
    || schedule?.dev_windows !== 256 || schedule?.batch_windows !== 4
    || schedule?.source_to_pilot_shift_delta !== 2
    || schedule?.integer_learning_rate_shifts?.output !== 36
    || schedule?.integer_learning_rate_shifts?.k !== 35
    || value.initialization?.output_init_amplitude !== 1
    || value.initialization?.output_forward_shift !== 14
    || value.integer?.training?.windows !== 1024
    || value.integer?.training?.optimizer_steps !== 256
    || value.integer?.dev_final?.windows !== 256
    || value.float?.evaluation?.windows !== 256
    || value.restart?.midpoint_optimizer_steps !== 128
    || value.integer?.chunks?.length !== 4
    || value.integer.chunks.some((chunk) => chunk.early_stop?.continue_training !== true)
    || typeof value.replay_eligible !== "boolean") {
    throw new Error("production stabilized pilot checkpoint is structurally invalid");
  }
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = JSON.parse(await readFile(outPath, "utf8"));
}
validate(checkpoint);
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production stabilized pilot checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_stabilized_pilot_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, replay_eligible: checkpoint.replay_eligible }));
}
