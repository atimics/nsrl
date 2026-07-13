#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-full-smoke";
let outPath = "benchmarks/production-model-v1/p10m-full-smoke.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${arg}`);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function validate(checkpoint) {
  if (checkpoint.schema !== "nsrl.production_full_train_smoke_checkpoint.v1"
    || checkpoint.profile !== "p10m"
    || checkpoint.parameter_count !== 9_317_632
    || checkpoint.bindings.tokenizer_hash !== "0xf4fe71d93c438c1a"
    || checkpoint.bindings.token_stream_hash !== "0x97e5254c31c27bda"
    || checkpoint.training.optimizer !== "integer_residual_sgd"
    || checkpoint.training.backward !== "full_quantized_straight_through"
    || checkpoint.training.windows !== 8
    || checkpoint.training.batch_windows !== 4
    || checkpoint.training.optimizer_steps !== 4
    || checkpoint.training.final_mistakes > checkpoint.training.initial_mistakes
    || checkpoint.moved_parameter_groups.length !== 13
    || checkpoint.artifacts.optimizer.bytes !== 74_541_140
    || checkpoint.cursor.schedule_complete !== true
    || checkpoint.restart.byte_identical_model !== true
    || checkpoint.restart.byte_identical_optimizer !== true
    || Object.values(checkpoint.diagnostics.gradient_nonzero_count).some((count) => count <= 0)
    || Object.values(checkpoint.diagnostics.saturation_by_group).some((count) => count !== 0)
    || checkpoint.hashes.initial_model === checkpoint.hashes.final_model
    || checkpoint.gates.all_parameter_groups_moved !== true
    || checkpoint.gates.model_hash_changed !== true
    || checkpoint.gates.resumable_optimizer_state !== true) {
    throw new Error("production full-train smoke gate failed");
  }
}

async function buildCheckpoint() {
  const tracePath = path.join(runDir, "train.json");
  const modelPath = path.join(runDir, "trained.nsrlpm");
  const optimizerPath = path.join(runDir, "optimizer.nsrlpo");
  const resumeTracePath = path.join(runDir, "resume.json");
  const resumedModelPath = path.join(runDir, "resumed.nsrlpm");
  const resumedOptimizerPath = path.join(runDir, "resumed.nsrlpo");
  const [traceBytes, modelBytes, optimizerBytes, resumeTraceBytes, resumedModelBytes, resumedOptimizerBytes] = await Promise.all([
    readFile(tracePath), readFile(modelPath), readFile(optimizerPath), readFile(resumeTracePath),
    readFile(resumedModelPath), readFile(resumedOptimizerPath),
  ]);
  const trace = JSON.parse(traceBytes);
  const resumeTrace = JSON.parse(resumeTraceBytes);
  if (trace.schema !== "nsrl.production_full_train_smoke.v1") throw new Error("invalid full-train trace");
  return {
    schema: "nsrl.production_full_train_smoke_checkpoint.v1",
    profile: trace.profile,
    parameter_count: trace.parameter_count,
    bindings: trace.bindings,
    training: trace.training,
    movement_l1: trace.movement_l1,
    moved_parameter_groups: trace.moved_parameter_groups,
    diagnostics: trace.diagnostics,
    cursor: trace.cursor,
    health: trace.health,
    hashes: trace.hashes,
    artifacts: {
      trained_model: { bytes: modelBytes.length, sha256: sha256(modelBytes) },
      optimizer: { bytes: optimizerBytes.length, sha256: sha256(optimizerBytes) },
      trace_sha256: sha256(traceBytes),
    },
    restart: {
      resumed_optimizer_steps: resumeTrace.training.optimizer_steps,
      resume_start_epoch: resumeTrace.cursor.start_epoch,
      resume_start_window: resumeTrace.cursor.start_window,
      schedule_complete: resumeTrace.cursor.schedule_complete,
      byte_identical_model: sha256(modelBytes) === sha256(resumedModelBytes),
      byte_identical_optimizer: sha256(optimizerBytes) === sha256(resumedOptimizerBytes),
      trace_sha256: sha256(resumeTraceBytes),
    },
    gates: trace.gates,
    known_non_claims: trace.known_non_claims,
  };
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
  if (await readFile(outPath, "utf8") !== rendered) throw new Error("production full-train checkpoint is stale");
  console.log(JSON.stringify({ schema: "nsrl.production_full_train_smoke_checkpoint_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(outPath);
}
