#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";

let tracePath = "";
let baselinePath = "";
let previousPath = "";
let outPath = "";
let chunk = -1;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--trace") tracePath = process.argv[++index];
  else if (process.argv[index] === "--baseline") baselinePath = process.argv[++index];
  else if (process.argv[index] === "--previous") previousPath = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--chunk") chunk = Number(process.argv[++index]);
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}
if (!tracePath || !baselinePath || !outPath || chunk < 0) {
  throw new Error("--trace, --baseline, --out, and --chunk are required");
}

const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const trace = await readJson(tracePath);
const baseline = await readJson(baselinePath);
const previous = previousPath ? await readJson(previousPath) : null;
const expectedStart = chunk * 256;
const baselineMean = baseline.evaluation.initial_mean_millibits;
const gates = {
  schema_and_geometry_match: trace.schema === "nsrl.production_float_twin_smoke.v1"
    && trace.profile === "p10m"
    && trace.parameter_count === 9317632
    && trace.training.context_tokens === 64
    && trace.training.start_window === expectedStart
    && trace.training.windows === 256
    && trace.training.batch_windows === 4,
  chain_hash_matches: previous === null
    ? trace.bindings.integer_initial_model_hash === "0x0e808e354809141d"
    : trace.tensor_hashes.initial === previous.tensor_hashes.final,
  all_parameter_groups_moved: trace.gates.all_parameter_groups_moved === true
    && trace.moved_parameter_groups.length === 13,
  all_parameters_finite: trace.gates.all_parameters_finite === true,
  training_loss_nonincreasing: trace.gates.loss_nonincreasing === true,
  heldout_nonincreasing_vs_lane_initial:
    trace.evaluation.final_mean_millibits <= baselineMean,
  tensor_hash_changed: trace.gates.tensor_hash_changed === true,
};
const event = {
  schema: "nsrl.production_scaling_readiness_float_chunk.v1",
  chunk,
  start_window: trace.training.start_window,
  next_window: trace.training.start_window + trace.training.windows,
  initial_tensor_hash: trace.tensor_hashes.initial,
  final_tensor_hash: trace.tensor_hashes.final,
  heldout_initial_mean_millibits: baselineMean,
  heldout_current_mean_millibits: trace.evaluation.final_mean_millibits,
  heldout_delta_millibits: trace.evaluation.final_mean_millibits - baselineMean,
  moved_parameter_groups: trace.moved_parameter_groups,
  gates,
  continue_training: Object.values(gates).every(Boolean),
};
await writeFile(outPath, `${JSON.stringify(event, null, 2)}\n`);
console.log(JSON.stringify(event));
if (!event.continue_training) process.exitCode = 3;
