#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-liveness-audit";
let outPath = "benchmarks/production-model-v1/p10m-liveness-audit.json";
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else throw new Error(`unknown argument: ${arg}`);
}
const readJson = async (name) => JSON.parse(await readFile(path.join(runDir, name), "utf8"));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
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
const gates = {
  negative_control_classified_dead: negativeEvent.dead === true
    && negativeEvent.classification === "output_unlock_timeout",
  negative_control_stopped_at_64_windows: negative.cursor.next_window === 64,
  positive_control_all_intervals_live: events.every((event) => event.dead === false),
  positive_control_output_unlocked: events.some((event) => event.output_moved),
  positive_control_full_gradient_path: events.every((event) => event.full_gradient_path),
  positive_control_zero_saturation: positive.every((trace) =>
    trace.health.gradient_saturation_count === 0 && trace.health.weight_saturation_count === 0),
  positive_control_heldout_nonincreasing: dev.every((trace) =>
    trace.evaluation.total_millibits <= devInitial.evaluation.total_millibits),
  positive_schedule_complete: positive[3].cursor.schedule_complete === true,
};
const checkpoint = {
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
  gates,
  audit_eligible: Object.values(gates).every(Boolean),
};
await writeFile(outPath, `${JSON.stringify(checkpoint, null, 2)}\n`);
console.log(JSON.stringify({ out: outPath, audit_eligible: checkpoint.audit_eligible }));
