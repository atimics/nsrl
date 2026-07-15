#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import process from "node:process";

let trainPath = "";
let initialDevPath = "";
let currentDevPath = "";
let outPath = "";
let chunk = -1;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--train") trainPath = process.argv[++index];
  else if (arg === "--initial-dev") initialDevPath = process.argv[++index];
  else if (arg === "--current-dev") currentDevPath = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--chunk") chunk = Number(process.argv[++index]);
  else throw new Error(`unknown argument: ${arg}`);
}
if (!trainPath || !initialDevPath || !currentDevPath || !outPath || chunk < 0) {
  throw new Error("--train, --initial-dev, --current-dev, --out, and --chunk are required");
}

const [train, initialDev, currentDev] = await Promise.all(
  [trainPath, initialDevPath, currentDevPath].map(async (file) => JSON.parse(await readFile(file, "utf8"))),
);
const gradientCounts = train.diagnostics?.gradient_nonzero_count ?? {};
const gates = {
  gradient_saturation_zero: train.health?.gradient_saturation_count === 0,
  weight_saturation_zero: train.health?.weight_saturation_count === 0,
  complete_gradient_path: Object.keys(gradientCounts).length === 13
    && Object.values(gradientCounts).every((count) => count > 0),
  heldout_dev_total_millibits_nonincreasing:
    currentDev.evaluation?.total_millibits <= initialDev.evaluation?.total_millibits,
};
const result = {
  schema: "nsrl.production_stabilized_pilot_early_stop.v1",
  chunk,
  initial_dev_total_millibits: initialDev.evaluation.total_millibits,
  current_dev_total_millibits: currentDev.evaluation.total_millibits,
  current_dev_delta: currentDev.evaluation.total_millibits - initialDev.evaluation.total_millibits,
  gradient_saturation_count: train.health.gradient_saturation_count,
  weight_saturation_count: train.health.weight_saturation_count,
  active_gradient_groups: Object.keys(gradientCounts).filter((key) => gradientCounts[key] > 0).sort(),
  gates,
  continue_training: Object.values(gates).every(Boolean),
};
await writeFile(outPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result));
if (!result.continue_training) process.exitCode = 3;
