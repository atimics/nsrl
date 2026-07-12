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
    || checkpoint.training.optimizer !== "integer_stateless_sgd"
    || checkpoint.training.backward !== "full_quantized_straight_through"
    || checkpoint.training.windows !== 8
    || checkpoint.training.optimizer_steps !== 16
    || checkpoint.training.final_mistakes > checkpoint.training.initial_mistakes
    || checkpoint.moved_parameter_groups.length !== 13
    || checkpoint.artifacts.optimizer.bytes !== 60
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
  const [traceBytes, modelBytes, optimizerBytes] = await Promise.all([
    readFile(tracePath), readFile(modelPath), readFile(optimizerPath),
  ]);
  const trace = JSON.parse(traceBytes);
  if (trace.schema !== "nsrl.production_full_train_smoke.v1") throw new Error("invalid full-train trace");
  return {
    schema: "nsrl.production_full_train_smoke_checkpoint.v1",
    profile: trace.profile,
    parameter_count: trace.parameter_count,
    bindings: trace.bindings,
    training: trace.training,
    movement_l1: trace.movement_l1,
    moved_parameter_groups: trace.moved_parameter_groups,
    health: trace.health,
    hashes: trace.hashes,
    artifacts: {
      trained_model: { bytes: modelBytes.length, sha256: sha256(modelBytes) },
      optimizer: { bytes: optimizerBytes.length, sha256: sha256(optimizerBytes) },
      trace_sha256: sha256(traceBytes),
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
