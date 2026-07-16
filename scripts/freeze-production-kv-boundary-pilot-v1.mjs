#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-kv-boundary-pilot";
let outPath = "benchmarks/production-model-v1/p10m-kv-boundary-pilot.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-kv-boundary-pilot-contract.json";
const groups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const readRunJson = async (name) => readJson(path.join(runDir, name));

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [kBytes, vBytes, init, devInitial, replay, finalModel, finalOptimizer,
    replayModel, replayOptimizer, ...rows] = await Promise.all([
    readFile(contract.source_preflights.k.path),
    readFile(contract.source_preflights.v.path),
    readRunJson("init.json"),
    readRunJson("dev-initial.json"),
    readRunJson("replay.json"),
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
  const kPreflight = JSON.parse(kBytes);
  const vPreflight = JSON.parse(vBytes);
  const traces = [];
  const dev = [];
  const events = [];
  for (let interval = 0; interval < 8; interval += 1) {
    traces.push(rows[interval * 3]);
    dev.push(rows[interval * 3 + 1]);
    events.push(rows[interval * 3 + 2]);
  }
  const updateGroups = (trace) => groups
    .filter((group) => trace.diagnostics.update_nonzero_count[group] > 0).sort();
  const exactMovement = traces.every((trace) => {
    const updates = updateGroups(trace);
    const movement = groups.filter((group) => trace.movement_l1[group] > 0).sort();
    const declared = [...trace.moved_parameter_groups].sort();
    return JSON.stringify(updates) === JSON.stringify(movement)
      && JSON.stringify(updates) === JSON.stringify(declared)
      && (trace.hashes.initial_model !== trace.hashes.final_model) === (updates.length > 0);
  });
  const movedAfterMidpoint = (group) => traces.slice(4)
    .some((trace) => trace.diagnostics.update_nonzero_count[group] > 0
      && trace.movement_l1[group] > 0);
  const gates = {
    source_preflight_hashes_match: sha256(kBytes) === contract.source_preflights.k.sha256
      && sha256(vBytes) === contract.source_preflights.v.sha256,
    source_preflights_eligible: kPreflight.preflight_eligible === true
      && vPreflight.preflight_eligible === true,
    candidate_schedule_exact: traces.every((trace) => JSON.stringify(
      trace.training.learning_rate_shifts,
    ) === JSON.stringify(contract.schedule.candidate_learning_rate_shifts)),
    only_k_and_v_shifts_changed: Object.keys(contract.schedule.base_learning_rate_shifts)
      .every((group) => contract.schedule.candidate_learning_rate_shifts[group]
        === (["k", "v"].includes(group)
          ? contract.candidates[group].candidate_shift
          : contract.schedule.base_learning_rate_shifts[group])),
    both_move_by_window_256: events[1].required_trunk_group_observed === true
      && events[1].required_trunk_group_observations?.k === true
      && events[1].required_trunk_group_observations?.v === true,
    both_move_again_after_midpoint: movedAfterMidpoint("k") && movedAfterMidpoint("v"),
    only_k_v_and_output_move: traces.every((trace) => trace.moved_parameter_groups
      .every((group) => ["k", "v", "output"].includes(group))),
    exact_reachable_update_consistency: exactMovement,
    all_intervals_live: events.every((event) => event.dead === false),
    full_gradient_path_after_unlock: events.slice(1).every((event) =>
      event.full_gradient_path === true && event.active_gradient_groups.length === 13),
    k_v_gradient_and_carry_present: traces.slice(1).every((trace) =>
      trace.diagnostics.gradient_nonzero_count.k > 0
      && trace.diagnostics.gradient_nonzero_count.v > 0
      && trace.diagnostics.residual_carry_count.k > 0
      && trace.diagnostics.residual_carry_count.v > 0),
    all_saturation_zero: traces.every((trace) =>
      trace.health.gradient_saturation_count === 0
      && trace.health.residual_saturation_count === 0
      && trace.health.weight_saturation_count === 0
      && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
      && Object.values(trace.diagnostics.residual_saturation_by_group)
        .every((count) => count === 0)),
    heldout_nonincreasing_at_every_interval: dev.every((row) =>
      row.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
    heldout_improves_at_completion: dev[7].evaluation.total_millibits
      < devInitial.evaluation.total_millibits,
    schedule_complete: traces[7].cursor.schedule_complete === true,
    midpoint_restart_model_byte_identical: sha256(finalModel) === sha256(replayModel),
    midpoint_restart_optimizer_byte_identical: sha256(finalOptimizer) === sha256(replayOptimizer),
    replay_schedule_complete_and_healthy: replay.cursor.schedule_complete === true
      && replay.health.gradient_saturation_count === 0
      && replay.health.residual_saturation_count === 0
      && replay.health.weight_saturation_count === 0,
  };
  const eligible = Object.values(gates).every(Boolean);
  return {
    schema: "nsrl.production_kv_boundary_pilot_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_preflights: contract.source_preflights,
    initialization: init,
    candidate: {
      groups: ["k", "v"],
      learning_rate_shifts: contract.schedule.candidate_learning_rate_shifts,
    },
    intervals: traces.map((trace, interval) => ({
      interval,
      start_window: trace.cursor.start_window,
      next_window: trace.cursor.next_window,
      moved_parameter_groups: trace.moved_parameter_groups,
      update_nonzero_count: trace.diagnostics.update_nonzero_count,
      movement_l1: trace.movement_l1,
      k_residual_carry_count: trace.diagnostics.residual_carry_count.k,
      v_residual_carry_count: trace.diagnostics.residual_carry_count.v,
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
      midpoint_window: 512,
      final_model_sha256: sha256(finalModel),
      replay_model_sha256: sha256(replayModel),
      final_optimizer_sha256: sha256(finalOptimizer),
      replay_optimizer_sha256: sha256(replayOptimizer),
    },
    artifacts: {
      model: { bytes: finalModel.length, sha256: sha256(finalModel) },
      optimizer: { bytes: finalOptimizer.length, sha256: sha256(finalOptimizer) },
    },
    gates,
    pilot_eligible: eligible,
    next_gate: eligible ? "p10m_kv_scaling_readiness_review" : "p10m_kv_schedule_review",
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
if (checkpoint.schema !== "nsrl.production_kv_boundary_pilot_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.pilot_eligible !== true
  || failedGates.length > 0) {
  throw new Error(`production K+V boundary pilot is invalid; failed gates: ${failedGates.join(",")}`);
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production K+V boundary pilot checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_kv_boundary_pilot_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, pilot_eligible: checkpoint.pilot_eligible }));
}
