#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-float-smoke";
let outPath = "benchmarks/production-model-v1/p10m-float-smoke.json";
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
  const gates = checkpoint.gates ?? {};
  if (checkpoint.schema !== "nsrl.production_float_twin_smoke_checkpoint.v1"
    || checkpoint.profile !== "p10m"
    || checkpoint.parameter_count !== 9_317_632
    || checkpoint.bindings.integer_initial_model_hash !== "0x9f278ab8d99e096c"
    || checkpoint.bindings.tokenizer_hash !== "0xf4fe71d93c438c1a"
    || checkpoint.bindings.token_stream_hash !== "0x97e5254c31c27bda"
    || checkpoint.training.windows !== 8
    || checkpoint.training.batch_windows !== 4
    || checkpoint.training.attention_algorithm !== "causal_recurrent_linear"
    || checkpoint.training.final_loss_millionths > checkpoint.training.initial_loss_millionths
    || checkpoint.training.final_mistakes > checkpoint.training.initial_mistakes
    || checkpoint.moved_parameter_groups.length !== 13
    || !Object.values(gates).every(Boolean)) {
    throw new Error("production float twin smoke gate failed");
  }
}

async function buildCheckpoint() {
  const tracePath = path.join(runDir, "train.json");
  const traceBytes = await readFile(tracePath);
  const trace = JSON.parse(traceBytes);
  if (trace.schema !== "nsrl.production_float_twin_smoke.v1") throw new Error("invalid float trace");
  return {
    schema: "nsrl.production_float_twin_smoke_checkpoint.v1",
    profile: trace.profile,
    parameter_count: trace.parameter_count,
    bindings: trace.bindings,
    training: trace.training,
    movement_trillionths: trace.movement_trillionths,
    moved_parameter_groups: trace.moved_parameter_groups,
    tensor_hashes: trace.tensor_hashes,
    trace_sha256: sha256(traceBytes),
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
  if (await readFile(outPath, "utf8") !== rendered) throw new Error("production float checkpoint is stale");
  console.log(JSON.stringify({ schema: "nsrl.production_float_twin_smoke_checkpoint_check.v1", ok: true }));
} else {
  await writeFile(outPath, rendered);
  console.log(outPath);
}
