#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let tracePath = "";
let stateInPath = "";
let stateOutPath = "";
let devInitialPath = "";
let devCurrentPath = "";
let eventPath = "";
let interval = -1;
let expectDead = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--trace") tracePath = process.argv[++index];
  else if (arg === "--state-in") stateInPath = process.argv[++index];
  else if (arg === "--state-out") stateOutPath = process.argv[++index];
  else if (arg === "--dev-initial") devInitialPath = process.argv[++index];
  else if (arg === "--dev-current") devCurrentPath = process.argv[++index];
  else if (arg === "--event-out") eventPath = process.argv[++index];
  else if (arg === "--interval") interval = Number(process.argv[++index]);
  else if (arg === "--expect-dead") expectDead = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!tracePath || !stateOutPath || !eventPath || interval < 0) {
  throw new Error("--trace, --state-out, --event-out, and --interval are required");
}

const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const trace = await readJson(tracePath);
const previous = stateInPath
  ? await readJson(stateInPath)
  : {
      schema: "nsrl.production_training_liveness_state.v1",
      phase: "output_unlock",
      output_unlocked: false,
      consecutive_dead_intervals: 0,
    };
const devInitial = devInitialPath ? await readJson(devInitialPath) : null;
const devCurrent = devCurrentPath ? await readJson(devCurrentPath) : null;
const gradientCounts = trace.diagnostics.gradient_nonzero_count;
const activeGroups = Object.keys(gradientCounts).filter((group) => gradientCounts[group] > 0).sort();
const outputMoved = trace.movement_l1.output > 0;
const outputUnlocked = previous.output_unlocked || outputMoved;
const fullGradientPath = activeGroups.length === 13;
const saturationZero = trace.health.gradient_saturation_count === 0
  && trace.health.weight_saturation_count === 0
  && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0);
const heldoutNonincreasing = !devInitial || !devCurrent
  || devCurrent.evaluation.total_millibits <= devInitial.evaluation.total_millibits;

let classification = "live";
if (!saturationZero) classification = "saturation";
else if (!outputUnlocked) classification = "output_unlock_timeout";
else if (!fullGradientPath) classification = "post_unlock_gradient_path_loss";
else if (!heldoutNonincreasing) classification = "heldout_regression";
const dead = classification !== "live";
const state = {
  schema: "nsrl.production_training_liveness_state.v1",
  interval,
  phase: outputUnlocked ? "trunk_live" : "output_unlock",
  output_unlocked: outputUnlocked,
  consecutive_dead_intervals: dead ? previous.consecutive_dead_intervals + 1 : 0,
  model_hash: trace.hashes.final_model,
};
const event = {
  schema: "nsrl.production_training_liveness_event.v1",
  interval,
  optimizer_steps: trace.training.optimizer_steps,
  start_window: trace.cursor.start_window,
  next_window: trace.cursor.next_window,
  previous_phase: previous.phase,
  phase: state.phase,
  output_moved: outputMoved,
  output_unlocked: outputUnlocked,
  active_gradient_groups: activeGroups,
  full_gradient_path: fullGradientPath,
  gradient_saturation_count: trace.health.gradient_saturation_count,
  weight_saturation_count: trace.health.weight_saturation_count,
  heldout_total_millibits_delta: devInitial && devCurrent
    ? devCurrent.evaluation.total_millibits - devInitial.evaluation.total_millibits
    : null,
  classification,
  dead,
};
await Promise.all([
  writeFile(stateOutPath, `${JSON.stringify(state, null, 2)}\n`),
  writeFile(eventPath, `${JSON.stringify(event, null, 2)}\n`),
]);
console.log(JSON.stringify(event));
if (expectDead ? !dead : dead) process.exitCode = 3;
