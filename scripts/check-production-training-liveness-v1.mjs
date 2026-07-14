#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

const requiredGroups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];
let tracePath = "";
let stateInPath = "";
let stateOutPath = "";
let devInitialPath = "";
let devCurrentPath = "";
let eventPath = "";
let interval = -1;
let expectDead = false;
let requireTrunkUpdateBy = null;
let outputUnlockDeadlineIntervals = 1;
let trunkActivationDeadlineIntervals = 1;
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
  else if (arg === "--require-trunk-update-by-interval") {
    requireTrunkUpdateBy = Number(process.argv[++index]);
  }
  else if (arg === "--output-unlock-deadline-intervals") {
    outputUnlockDeadlineIntervals = Number(process.argv[++index]);
  }
  else if (arg === "--trunk-activation-deadline-intervals") {
    trunkActivationDeadlineIntervals = Number(process.argv[++index]);
  }
  else throw new Error(`unknown argument: ${arg}`);
}
if (!tracePath || !stateOutPath || !eventPath || interval < 0) {
  throw new Error("--trace, --state-out, --event-out, and --interval are required");
}
if (!Number.isInteger(outputUnlockDeadlineIntervals) || outputUnlockDeadlineIntervals < 1
  || !Number.isInteger(trunkActivationDeadlineIntervals) || trunkActivationDeadlineIntervals < 1
  || (requireTrunkUpdateBy !== null
    && (!Number.isInteger(requireTrunkUpdateBy) || requireTrunkUpdateBy < 0))) {
  throw new Error("liveness deadlines must be bounded integer intervals");
}

const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const trace = await readJson(tracePath);
const policy = {
  output_unlock_deadline_intervals: outputUnlockDeadlineIntervals,
  trunk_activation_deadline_intervals: trunkActivationDeadlineIntervals,
  require_trunk_update_by_interval: requireTrunkUpdateBy,
};
const previous = stateInPath
  ? await readJson(stateInPath)
  : {
      schema: "nsrl.production_training_liveness_state.v1",
      phase: "output_unlock",
      output_unlocked: false,
      trunk_update_observed: false,
      phase_age_intervals: 0,
      consecutive_dead_intervals: 0,
      history_hash: "0".repeat(64),
      policy,
    };
if ((!stateInPath && interval !== 0)
  || previous.schema !== "nsrl.production_training_liveness_state.v1"
  || !["output_unlock", "trunk_activation", "trunk_live"].includes(previous.phase)
  || !/^[0-9a-f]{64}$/.test(previous.history_hash)
  || (stateInPath && previous.interval + 1 !== interval)
  || (stateInPath && previous.model_hash !== trace.hashes.initial_model)
  || (stateInPath && previous.consecutive_dead_intervals !== 0)
  || (stateInPath && JSON.stringify(previous.policy) !== JSON.stringify(policy))) {
  throw new Error("liveness state does not bind the next model interval");
}
const devInitial = devInitialPath ? await readJson(devInitialPath) : null;
const devCurrent = devCurrentPath ? await readJson(devCurrentPath) : null;
const gradientCounts = trace.diagnostics.gradient_nonzero_count;
const exactGroupKeys = (value) => value && JSON.stringify(Object.keys(value).sort())
  === JSON.stringify([...requiredGroups].sort());
const validCounts = (value) => Object.values(value)
  .every((count) => Number.isSafeInteger(count) && count >= 0);
if (!exactGroupKeys(gradientCounts) || !validCounts(gradientCounts)
  || !exactGroupKeys(trace.diagnostics.saturation_by_group)
  || !validCounts(trace.diagnostics.saturation_by_group)
  || !exactGroupKeys(trace.diagnostics.residual_saturation_by_group)
  || !validCounts(trace.diagnostics.residual_saturation_by_group)
  || !trace.moved_parameter_groups.every((group) => requiredGroups.includes(group))) {
  throw new Error("training trace does not contain the exact production parameter groups");
}
const activeGroups = requiredGroups.filter((group) => gradientCounts[group] > 0).sort();
const outputMoved = trace.movement_l1.output > 0;
const trunkMoved = trace.moved_parameter_groups.some((group) => group !== "output");
const trunkUpdateObserved = (previous.trunk_update_observed ?? false) || trunkMoved;
const outputUnlocked = previous.output_unlocked || outputMoved;
const fullGradientPath = activeGroups.length === 13;
const residualSaturationCount = trace.health.residual_saturation_count ?? 0;
const saturationZero = trace.health.gradient_saturation_count === 0
  && residualSaturationCount === 0
  && trace.health.weight_saturation_count === 0
  && Object.values(trace.diagnostics.saturation_by_group).every((count) => count === 0)
  && Object.values(trace.diagnostics.residual_saturation_by_group)
    .every((count) => count === 0);
const heldoutNonincreasing = !devInitial || !devCurrent
  || devCurrent.evaluation.total_millibits <= devInitial.evaluation.total_millibits;

let phase = "trunk_live";
if (!outputUnlocked) phase = "output_unlock";
else if (!fullGradientPath && previous.phase !== "trunk_live") phase = "trunk_activation";
const phaseAgeIntervals = phase === previous.phase
  ? (previous.phase_age_intervals ?? 0) + 1
  : 0;
let classification = "live";
if (!saturationZero) classification = "saturation";
else if (!outputUnlocked && phaseAgeIntervals >= outputUnlockDeadlineIntervals) {
  classification = "output_unlock_timeout";
} else if (phase === "trunk_activation" && !fullGradientPath
  && phaseAgeIntervals >= trunkActivationDeadlineIntervals) {
  classification = "trunk_activation_timeout";
} else if (previous.phase === "trunk_live" && !fullGradientPath) {
  classification = "post_unlock_gradient_path_loss";
} else if (requireTrunkUpdateBy !== null && interval >= requireTrunkUpdateBy
  && !trunkUpdateObserved) classification = "trunk_update_timeout";
else if (!heldoutNonincreasing) classification = "heldout_regression";
const dead = classification !== "live";
const state = {
  schema: "nsrl.production_training_liveness_state.v1",
  interval,
  phase,
  output_unlocked: outputUnlocked,
  trunk_update_observed: trunkUpdateObserved,
  phase_age_intervals: phaseAgeIntervals,
  consecutive_dead_intervals: dead ? previous.consecutive_dead_intervals + 1 : 0,
  model_hash: trace.hashes.final_model,
  policy,
};
const event = {
  schema: "nsrl.production_training_liveness_event.v1",
  interval,
  optimizer_steps: trace.training.optimizer_steps,
  start_window: trace.cursor.start_window,
  next_window: trace.cursor.next_window,
  previous_phase: previous.phase,
  phase: state.phase,
  phase_age_intervals: phaseAgeIntervals,
  output_moved: outputMoved,
  output_unlocked: outputUnlocked,
  trunk_moved: trunkMoved,
  trunk_update_observed: trunkUpdateObserved,
  active_gradient_groups: activeGroups,
  full_gradient_path: fullGradientPath,
  gradient_saturation_count: trace.health.gradient_saturation_count,
  residual_saturation_count: residualSaturationCount,
  weight_saturation_count: trace.health.weight_saturation_count,
  heldout_total_millibits_delta: devInitial && devCurrent
    ? devCurrent.evaluation.total_millibits - devInitial.evaluation.total_millibits
    : null,
  classification,
  dead,
  policy,
};
state.history_hash = createHash("sha256")
  .update(previous.history_hash ?? "0".repeat(64))
  .update(JSON.stringify(event))
  .digest("hex");
event.history_hash = state.history_hash;
await Promise.all([
  writeFile(stateOutPath, `${JSON.stringify(state, null, 2)}\n`),
  writeFile(eventPath, `${JSON.stringify(event, null, 2)}\n`),
]);
console.log(JSON.stringify(event));
if (expectDead ? !dead : dead) process.exitCode = 3;
