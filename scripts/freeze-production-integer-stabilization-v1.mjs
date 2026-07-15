#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-stabilization";
let outPath = "benchmarks/production-model-v1/p10m-stabilization.json";
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

async function buildCheckpoint() {
  const [pilotBytes, init, train, devInitial, devFinal, initialModel, trainedModel, optimizer] = await Promise.all([
    readFile("benchmarks/production-model-v1/p10m-pilot.json"),
    readJson("init.json"),
    readJson("train.json"),
    readJson("dev-initial.json"),
    readJson("dev-final.json"),
    readFile(path.join(runDir, "initial.nsrlpm")),
    readFile(path.join(runDir, "trained.nsrlpm")),
    readFile(path.join(runDir, "optimizer.nsrlpo")),
  ]);
  const gradientCounts = train.diagnostics.gradient_nonzero_count;
  const saturationByGroup = train.diagnostics.saturation_by_group;
  const learningRateShifts = train.training.learning_rate_shifts;
  const gates = {
    bounded_schedule_complete: train.training.windows === 256
      && train.training.optimizer_steps === 64
      && train.cursor.schedule_complete === true,
    dev_total_millibits_nonincreasing:
      devFinal.evaluation.total_millibits <= devInitial.evaluation.total_millibits,
    fixed_probe_mistakes_nonincreasing:
      train.training.final_mistakes <= train.training.initial_mistakes,
    complete_gradient_path: Object.keys(gradientCounts).length === 13
      && Object.values(gradientCounts).every((count) => count > 0),
    parameter_movement: train.moved_parameter_groups.length > 0,
    zero_gradient_saturation: train.health.gradient_saturation_count === 0,
    zero_weight_saturation: train.health.weight_saturation_count === 0
      && Object.values(saturationByGroup).every((count) => count === 0),
    projection_specific_shift_schedule: Object.keys(learningRateShifts).length === 13
      && learningRateShifts.k > learningRateShifts.q
      && learningRateShifts.down === learningRateShifts.o,
    output_scale_decoupled: init.output_forward_shift === 14
      && train.training.output_backward_shift === 8,
    model_hash_changed: train.hashes.initial_model !== train.hashes.final_model,
  };
  return {
    schema: "nsrl.production_integer_stabilization_preflight.v1",
    profile: "p10m",
    parameter_count: train.parameter_count,
    source_pilot: {
      path: "benchmarks/production-model-v1/p10m-pilot.json",
      sha256: sha256(pilotBytes),
      promotion_eligible: JSON.parse(pilotBytes).promotion_eligible,
    },
    bindings: {
      tokenizer_hash: train.bindings.tokenizer_hash,
      train_token_stream_hash: train.bindings.token_stream_hash,
      dev_token_stream_hash: devInitial.bindings.token_stream_hash,
    },
    initialization: {
      seed: init.initialization_seed,
      output_init_amplitude: init.output_init_amplitude,
      output_forward_shift: init.output_forward_shift,
      model_hash: init.model_hash,
    },
    schedule: {
      context_tokens: train.training.context_tokens,
      train_windows: train.training.windows,
      dev_windows: devInitial.evaluation.windows,
      batch_windows: train.training.batch_windows,
      optimizer_steps: train.training.optimizer_steps,
      learning_rate_shifts: learningRateShifts,
      output_backward_shift: train.training.output_backward_shift,
      scaling_rule: "add_log2_window_multiplier_to_each_update_shift",
      reference_windows: 32,
      reference_to_validation_shift_delta: 3,
    },
    training: {
      initial_mistakes: train.training.initial_mistakes,
      final_mistakes: train.training.final_mistakes,
      moved_parameter_groups: train.moved_parameter_groups,
      movement_l1: train.movement_l1,
      diagnostics: train.diagnostics,
      health: train.health,
      hashes: train.hashes,
    },
    evaluation: {
      initial: devInitial.evaluation,
      final: devFinal.evaluation,
      total_millibits_delta:
        devFinal.evaluation.total_millibits - devInitial.evaluation.total_millibits,
    },
    artifacts: {
      initial_model: { bytes: initialModel.length, sha256: sha256(initialModel) },
      trained_model: { bytes: trainedModel.length, sha256: sha256(trainedModel) },
      optimizer: { bytes: optimizer.length, sha256: sha256(optimizer) },
    },
    gates,
    preflight_eligible: Object.values(gates).every(Boolean),
    next_gate: "p10m_stabilized_pilot_replay",
    known_non_claims: [
      "bounded_256_window_preflight_not_full_pilot",
      "active_output_initialization_changes_failed_pilot_candidate",
      "gradient_path_activity_does_not_require_every_group_to_move_in_preflight",
      "not_open_generation_quality",
    ],
  };
}

function validate(value) {
  if (value.schema !== "nsrl.production_integer_stabilization_preflight.v1"
    || value.profile !== "p10m"
    || value.parameter_count !== 9_317_632
    || value.source_pilot?.promotion_eligible !== false
    || value.schedule?.train_windows !== 256
    || value.schedule?.dev_windows !== 256
    || value.schedule?.context_tokens !== 64
    || value.schedule?.optimizer_steps !== 64
    || value.initialization?.output_init_amplitude !== 1
    || value.initialization?.output_forward_shift !== 14
    || value.schedule?.output_backward_shift !== 8
    || value.next_gate !== "p10m_stabilized_pilot_replay"
    || typeof value.preflight_eligible !== "boolean") {
    throw new Error("production integer stabilization checkpoint is structurally invalid");
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
    throw new Error("production integer stabilization checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_integer_stabilization_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, preflight_eligible: checkpoint.preflight_eligible }));
}
