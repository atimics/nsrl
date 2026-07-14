#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-liveness-audit";
let outPath = "benchmarks/production-model-v1/p10m-liveness-audit.json";
let positiveMicroDir = "data/experiments/production-model-v1/p10m-liveness-micro-local";
let postUnlockMicroDir = "data/experiments/production-model-v1/p10m-liveness-post-unlock-micro-local";
let negativeMicroDir = "data/experiments/production-model-v1/p10m-liveness-negative-micro-local";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--positive-micro-dir") positiveMicroDir = process.argv[++index];
  else if (arg === "--post-unlock-micro-dir") postUnlockMicroDir = process.argv[++index];
  else if (arg === "--negative-micro-dir") negativeMicroDir = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${arg}`);
}
const readJson = async (name) => JSON.parse(await readFile(path.join(runDir, name), "utf8"));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const traceSummary = (trace, interval) => ({
  interval,
  start_window: trace.cursor.start_window,
  next_window: trace.cursor.next_window,
  output_movement: trace.movement_l1.output,
  active_gradient_groups: Object.entries(trace.diagnostics.gradient_nonzero_count)
    .filter(([, count]) => count > 0).map(([group]) => group),
  moved_parameter_groups: trace.moved_parameter_groups,
  health: trace.health,
});

async function readMicroTimeline(directory, count, intervalOffset) {
  try {
    return await Promise.all(Array.from({ length: count }, async (_, index) =>
      traceSummary(JSON.parse(await readFile(path.join(directory, `chunk-${index}.json`), "utf8")),
        intervalOffset + index)));
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

async function buildCheckpoint() {
const [init, devInitial, negative, negativeEvent, ...rest] = await Promise.all([
  readJson("init.json"), readJson("dev-initial.json"),
  readJson("negative-0.json"), readJson("negative-event-0.json"),
  ...[0, 1, 2, 3].map((index) => readJson(`positive-${index}.json`)),
  ...[0, 1, 2, 3].map((index) => readJson(`positive-dev-${index}.json`)),
  ...[0, 1, 2, 3].map((index) => readJson(`positive-event-${index}.json`)),
  readFile(path.join(runDir, "positive-3.nsrlpm")),
  readFile(path.join(runDir, "positive-3.nsrlpo")),
]);
const positive = rest.slice(0, 4);
const dev = rest.slice(4, 8);
const events = rest.slice(8, 12);
const [model, optimizer] = rest.slice(12);
const [positiveMicro, postUnlockMicro, negativeMicro] = await Promise.all([
  readMicroTimeline(positiveMicroDir, 4, 0),
  readMicroTimeline(postUnlockMicroDir, 4, 4),
  readMicroTimeline(negativeMicroDir, 4, 0),
]);
const positiveTimeline = positiveMicro && postUnlockMicro
  ? [...positiveMicro, ...postUnlockMicro]
  : null;
const gates = {
  negative_control_classified_dead: negativeEvent.dead === true
    && negativeEvent.classification === "output_unlock_timeout",
  negative_control_stopped_at_64_windows: negative.cursor.next_window === 64,
  positive_control_all_intervals_live: events.every((event) => event.dead === false),
  positive_control_output_unlocked: events.some((event) => event.output_moved),
  positive_control_full_gradient_path_after_activation: events
    .filter((event) => event.previous_phase !== "output_unlock")
    .every((event) => event.full_gradient_path),
  positive_control_zero_saturation: positive.every((trace) =>
    trace.health.gradient_saturation_count === 0
      && (trace.health.residual_saturation_count ?? 0) === 0
      && trace.health.weight_saturation_count === 0),
  positive_control_heldout_nonincreasing: dev.every((trace) =>
    trace.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
  positive_schedule_complete: positive[3].cursor.schedule_complete === true,
};
if (positiveTimeline && negativeMicro) {
  const firstOutputMovement = positiveTimeline.find((trace) => trace.output_movement > 0)?.interval;
  const firstFullGradientPath = positiveTimeline.find(
    (trace) => trace.active_gradient_groups.length === 13,
  )?.interval;
  gates.micro_output_unlock_at_measured_deadline = firstOutputMovement === 3;
  gates.micro_trunk_activation_at_measured_deadline = firstFullGradientPath === 6;
  gates.micro_negative_control_stays_locked = negativeMicro.every(
    (trace) => trace.output_movement === 0 && trace.active_gradient_groups.length === 3,
  );
  gates.micro_zero_residual_saturation = [...positiveTimeline, ...negativeMicro].every(
    (trace) => (trace.health.residual_saturation_count ?? 0) === 0,
  );
  gates.trunk_update_starvation_detectable = positive.every(
    (trace) => trace.moved_parameter_groups.every((group) => group === "output"),
  );
}
return {
  schema: "nsrl.production_training_liveness_audit.v1",
  profile: "p10m",
  parameter_count: positive[0].parameter_count,
  initialization: init,
  interval: { optimizer_steps: 16, windows: 64, total_windows: 256 },
  negative_control: { trace: negative, event: negativeEvent },
  positive_control: {
    traces: positive,
    heldout_initial: devInitial.evaluation,
    heldout_by_interval: dev.map((trace) => trace.evaluation),
    events,
    artifacts: {
      model: { bytes: model.length, sha256: sha256(model) },
      optimizer: { bytes: optimizer.length, sha256: sha256(optimizer) },
    },
  },
  micro_probe: positiveTimeline && negativeMicro ? {
    interval_windows: 16,
    output_unlock_deadline_intervals: 4,
    trunk_activation_deadline_intervals: 3,
    positive_timeline: positiveTimeline,
    negative_timeline: negativeMicro,
    observed_output_unlock_interval: positiveTimeline.find(
      (trace) => trace.output_movement > 0,
    )?.interval,
    observed_trunk_activation_interval: positiveTimeline.find(
      (trace) => trace.active_gradient_groups.length === 13,
    )?.interval,
    trunk_update_observed_by_256_windows: positive.some(
      (trace) => trace.moved_parameter_groups.some((group) => group !== "output"),
    ),
  } : null,
  gates,
  audit_eligible: Object.values(gates).every(Boolean),
};
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = JSON.parse(await readFile(outPath, "utf8"));
}
if (checkpoint.schema !== "nsrl.production_training_liveness_audit.v1"
  || checkpoint.interval?.windows !== 64
  || checkpoint.negative_control?.event?.classification !== "output_unlock_timeout"
  || (checkpoint.micro_probe !== null
    && (checkpoint.micro_probe?.observed_output_unlock_interval !== 3
      || checkpoint.micro_probe?.observed_trunk_activation_interval !== 6
      || checkpoint.micro_probe?.trunk_update_observed_by_256_windows !== false))
  || !Object.values(checkpoint.gates).every(Boolean)
  || checkpoint.audit_eligible !== true) {
  throw new Error("production training liveness audit is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production training liveness audit checkpoint is stale");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_training_liveness_audit_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({ out: outPath, audit_eligible: checkpoint.audit_eligible }));
}
