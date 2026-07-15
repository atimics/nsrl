#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-trunk-unlock-preflight";
let outPath = "benchmarks/production-model-v1/p10m-trunk-unlock-preflight.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${arg}`);
}
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readRunJson = async (name) => JSON.parse(await readFile(path.join(runDir, name), "utf8"));
const expectedGroups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];

async function buildCheckpoint() {
  const [contractBytes, livenessBytes, analysis, init, devInitial, replay,
    finalModel, finalOptimizer, replayModel, replayOptimizer, ...intervalData] = await Promise.all([
    readFile("benchmarks/production-model-v1/p10m-trunk-unlock-contract.json"),
    readFile("benchmarks/production-model-v1/p10m-liveness-audit.json"),
    readRunJson("residual-analysis.json"), readRunJson("init.json"),
    readRunJson("dev-initial.json"), readRunJson("replay.json"),
    readFile(path.join(runDir, "model-3.nsrlpm")),
    readFile(path.join(runDir, "optimizer-3.nsrlpo")),
    readFile(path.join(runDir, "replay-final.nsrlpm")),
    readFile(path.join(runDir, "replay-final.nsrlpo")),
    ...[0, 1, 2, 3].flatMap((index) => [
      readRunJson(`train-${index}.json`),
      readRunJson(`dev-${index}.json`),
      readRunJson(`event-${index}.json`),
    ]),
  ]);
  const contract = JSON.parse(contractBytes);
  const liveness = JSON.parse(livenessBytes);
  const traces = [];
  const dev = [];
  const events = [];
  for (let index = 0; index < 4; index += 1) {
    traces.push(intervalData[index * 3]);
    dev.push(intervalData[index * 3 + 1]);
    events.push(intervalData[index * 3 + 2]);
  }
  const schedule = contract.schedule.base_learning_rate_shifts;
  const candidateSchedule = { ...schedule, [contract.candidate.group]: contract.candidate.candidate_shift };
  const intervalSummaries = traces.map((trace, interval) => ({
    interval,
    start_window: trace.cursor.start_window,
    next_window: trace.cursor.next_window,
    moved_parameter_groups: trace.moved_parameter_groups,
    movement_l1: trace.movement_l1,
    update_nonzero_count: trace.diagnostics.update_nonzero_count,
    active_gradient_groups: Object.entries(trace.diagnostics.gradient_nonzero_count)
      .filter(([, count]) => count > 0).map(([group]) => group),
    health: trace.health,
    dev: dev[interval].evaluation,
    liveness: events[interval],
  }));
  const gates = {
    source_liveness_hash_matches: sha256(livenessBytes) === contract.source_liveness.sha256,
    source_optimizer_hash_matches: analysis.source.optimizer_sha256
      === contract.source_optimizer.sha256,
    source_optimizer_state_matches: analysis.source.optimizer_state_hash
      === contract.source_optimizer.optimizer_state_hash
      && analysis.source.optimizer_step === contract.source_optimizer.optimizer_step,
    residual_policy_recommends_candidate: analysis.recommendation?.policy
      === contract.residual_policy.id
      && analysis.recommendation?.group === contract.candidate.group
      && analysis.recommendation?.candidate_shift === contract.candidate.candidate_shift
      && analysis.recommendation?.predicted_parameter_crossings
        === contract.candidate.predicted_parameter_crossings,
    candidate_schedule_exact: traces.every((trace) =>
      JSON.stringify(trace.training.learning_rate_shifts) === JSON.stringify(candidateSchedule)),
    only_v_shift_changed: Object.keys(schedule).every((group) =>
      candidateSchedule[group] === (group === "v" ? 30 : schedule[group])),
    all_intervals_live: events.every((event) => event.dead === false),
    output_unlocked_by_first_interval: events[0].output_unlocked === true,
    full_gradient_path_after_activation: events.slice(1)
      .every((event) => event.full_gradient_path === true
        && JSON.stringify([...event.active_gradient_groups].sort())
          === JSON.stringify([...expectedGroups].sort())),
    v_moves_by_deadline: events[3].trunk_update_observed === true
      && JSON.stringify(events[3].moved_trunk_groups) === JSON.stringify(["v"])
      && traces[3].movement_l1.v > 0,
    no_early_false_trunk_unlock: events.slice(0, 3)
      .every((event) => event.trunk_update_observed === false),
    only_v_and_output_move: traces.every((trace) => trace.moved_parameter_groups
      .every((group) => ["v", "output"].includes(group))),
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
  return {
    schema: "nsrl.production_trunk_unlock_preflight_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: "benchmarks/production-model-v1/p10m-trunk-unlock-contract.json",
      sha256: sha256(contractBytes) },
    source_liveness: contract.source_liveness,
    initialization: init,
    residual_analysis: analysis,
    candidate: { ...contract.candidate, learning_rate_shifts: candidateSchedule },
    interval: {
      windows: contract.schedule.interval_windows,
      optimizer_steps: contract.schedule.interval_optimizer_steps,
      count: contract.schedule.intervals,
      total_windows: contract.schedule.train_windows,
    },
    intervals: intervalSummaries,
    heldout: {
      initial: devInitial.evaluation,
      final: dev[3].evaluation,
      total_millibits_delta: dev[3].evaluation.total_millibits
        - devInitial.evaluation.total_millibits,
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
    preflight_eligible: Object.values(gates).every(Boolean),
    next_gate: Object.values(gates).every(Boolean)
      ? "p10m_trunk_unlock_pilot_contract"
      : "p10m_trunk_unlock_policy_review",
  };
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = JSON.parse(await readFile(outPath, "utf8"));
}
if (checkpoint.schema !== "nsrl.production_trunk_unlock_preflight_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.candidate?.group !== "v"
  || checkpoint.candidate?.candidate_shift !== 30
  || checkpoint.interval?.total_windows !== 256
  || checkpoint.heldout?.total_millibits_delta >= 0
  || !Object.values(checkpoint.gates).every(Boolean)
  || checkpoint.preflight_eligible !== true
  || checkpoint.next_gate !== "p10m_trunk_unlock_pilot_contract") {
  throw new Error("production trunk-unlock preflight checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production trunk-unlock preflight checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_trunk_unlock_preflight_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, preflight_eligible: checkpoint.preflight_eligible }));
}
