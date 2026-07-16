#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-k-stabilization-preflight";
let outPath = "benchmarks/production-model-v1/p10m-k-stabilization-preflight.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${arg}`);
}

const contractPath = "benchmarks/production-model-v1/p10m-k-stabilization-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readRunJson = async (name) => JSON.parse(await readFile(path.join(runDir, name), "utf8"));
const expectedGroups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];

async function buildCheckpoint() {
  const [contractBytes, livenessBytes, analysis, init, devInitial, replay,
    finalModel, finalOptimizer, replayModel, replayOptimizer, ...intervalData] = await Promise.all([
    readFile(contractPath),
    readFile("benchmarks/production-model-v1/p10m-liveness-audit.json"),
    readRunJson("residual-analysis.json"),
    readRunJson("init.json"),
    readRunJson("dev-initial.json"),
    readRunJson("replay.json"),
    readFile(path.join(runDir, "model-3.nsrlpm")),
    readFile(path.join(runDir, "optimizer-3.nsrlpo")),
    readFile(path.join(runDir, "replay-final.nsrlpm")),
    readFile(path.join(runDir, "replay-final.nsrlpo")),
    ...[0, 1, 2, 3].flatMap((interval) => [
      readRunJson(`train-${interval}.json`),
      readRunJson(`dev-${interval}.json`),
      readRunJson(`event-${interval}.json`),
    ]),
  ]);
  const contract = JSON.parse(contractBytes);
  const liveness = JSON.parse(livenessBytes);
  const traces = [];
  const dev = [];
  const events = [];
  for (let interval = 0; interval < 4; interval += 1) {
    traces.push(intervalData[interval * 3]);
    dev.push(intervalData[interval * 3 + 1]);
    events.push(intervalData[interval * 3 + 2]);
  }
  const baseSchedule = contract.schedule.base_learning_rate_shifts;
  const candidateSchedule = {
    ...baseSchedule,
    [contract.candidate.group]: contract.candidate.candidate_shift,
  };
  const kAnalysis = analysis.groups.find((row) => row.group === "k");
  const exactUpdateGroups = (trace) => expectedGroups
    .filter((group) => trace.diagnostics.update_nonzero_count[group] > 0).sort();
  const exactMovement = traces.every((trace) => {
    const updateGroups = exactUpdateGroups(trace);
    const movementGroups = expectedGroups
      .filter((group) => trace.movement_l1[group] > 0).sort();
    const declaredGroups = [...trace.moved_parameter_groups].sort();
    return JSON.stringify(updateGroups) === JSON.stringify(movementGroups)
      && JSON.stringify(updateGroups) === JSON.stringify(declaredGroups)
      && (trace.hashes.initial_model !== trace.hashes.final_model)
        === (updateGroups.length > 0);
  });
  const gates = {
    source_liveness_hash_matches: sha256(livenessBytes) === contract.source_liveness.sha256,
    source_optimizer_matches: analysis.source.optimizer_sha256
      === contract.source_optimizer.sha256
      && analysis.source.optimizer_state_hash === contract.source_optimizer.optimizer_state_hash
      && analysis.source.optimizer_step === contract.source_optimizer.optimizer_step,
    k_boundary_prediction_matches: kAnalysis?.current_shift === contract.candidate.source_shift
      && kAnalysis?.boundary_shift === contract.candidate.candidate_shift
      && kAnalysis?.predicted_parameter_crossings_at_boundary
        === contract.candidate.predicted_parameter_crossings,
    candidate_schedule_exact: traces.every((trace) =>
      JSON.stringify(trace.training.learning_rate_shifts) === JSON.stringify(candidateSchedule)),
    only_k_shift_changed: Object.keys(baseSchedule).every((group) =>
      candidateSchedule[group] === (group === "k" ? 26 : baseSchedule[group])),
    all_intervals_live: events.every((event) => event.dead === false),
    full_gradient_path_after_activation: events.slice(1).every((event) =>
      event.full_gradient_path === true
      && JSON.stringify([...event.active_gradient_groups].sort())
        === JSON.stringify([...expectedGroups].sort())),
    k_gradient_and_carry_present: traces.slice(1).every((trace) =>
      trace.diagnostics.gradient_nonzero_count.k > 0
      && trace.diagnostics.residual_carry_count.k > 0),
    k_moves_by_deadline: events[3].required_trunk_group_observed === true
      && traces[3].diagnostics.update_nonzero_count.k > 0
      && traces[3].movement_l1.k > 0,
    no_early_k_update: traces.slice(0, 3)
      .every((trace) => trace.diagnostics.update_nonzero_count.k === 0),
    only_k_and_output_move: traces.every((trace) =>
      trace.moved_parameter_groups.every((group) => ["k", "output"].includes(group))),
    exact_reachable_update_consistency: exactMovement,
    all_saturation_zero: traces.every((trace) =>
      trace.health.gradient_saturation_count === 0
      && trace.health.residual_saturation_count === 0
      && trace.health.weight_saturation_count === 0
      && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
      && Object.values(trace.diagnostics.residual_saturation_by_group)
        .every((count) => count === 0)),
    heldout_nonincreasing: dev.every((row) => row.evaluation.total_millibits
      <= devInitial.evaluation.total_millibits),
    heldout_improves: dev[3].evaluation.total_millibits
      < devInitial.evaluation.total_millibits,
    schedule_complete: traces[3].cursor.schedule_complete === true,
    midpoint_restart_model_byte_identical: sha256(finalModel) === sha256(replayModel),
    midpoint_restart_optimizer_byte_identical: sha256(finalOptimizer) === sha256(replayOptimizer),
    replay_schedule_complete_and_healthy: replay.cursor.schedule_complete === true
      && replay.health.gradient_saturation_count === 0
      && replay.health.residual_saturation_count === 0
      && replay.health.weight_saturation_count === 0,
  };
  const eligible = Object.values(gates).every(Boolean);
  return {
    schema: "nsrl.production_k_stabilization_preflight_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_liveness: contract.source_liveness,
    initialization: init,
    residual_analysis: { source: analysis.source, k: kAnalysis },
    candidate: { ...contract.candidate, learning_rate_shifts: candidateSchedule },
    intervals: traces.map((trace, interval) => ({
      interval,
      start_window: trace.cursor.start_window,
      next_window: trace.cursor.next_window,
      moved_parameter_groups: trace.moved_parameter_groups,
      movement_l1: trace.movement_l1,
      update_nonzero_count: trace.diagnostics.update_nonzero_count,
      k_gradient_nonzero_count: trace.diagnostics.gradient_nonzero_count.k,
      k_residual_carry_count: trace.diagnostics.residual_carry_count.k,
      health: trace.health,
      dev: dev[interval].evaluation,
      liveness: events[interval],
    })),
    heldout: {
      initial: devInitial.evaluation,
      final: dev[3].evaluation,
      total_millibits_delta:
        dev[3].evaluation.total_millibits - devInitial.evaluation.total_millibits,
    },
    restart: {
      midpoint_window: 128,
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
    preflight_eligible: eligible,
    next_gate: eligible
      ? "p10m_k_stabilized_boundary_pilot_contract"
      : "p10m_k_stabilization_policy_review",
  };
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = JSON.parse(await readFile(outPath, "utf8"));
}
if (checkpoint.schema !== "nsrl.production_k_stabilization_preflight_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.candidate?.group !== "k"
  || checkpoint.candidate?.candidate_shift !== 26
  || !Object.values(checkpoint.gates).every(Boolean)
  || checkpoint.preflight_eligible !== true
  || checkpoint.next_gate !== "p10m_k_stabilized_boundary_pilot_contract") {
  const failedGates = Object.entries(checkpoint.gates ?? {})
    .filter(([, passed]) => !passed).map(([gate]) => gate);
  throw new Error(
    `production K stabilization preflight checkpoint is structurally invalid; failed gates: ${failedGates.join(",")}`,
  );
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production K stabilization preflight checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_k_stabilization_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, preflight_eligible: checkpoint.preflight_eligible }));
}
