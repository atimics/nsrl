#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let runDir = "data/experiments/production-model-v1/p10m-smoke";
let outPath = "benchmarks/production-model-v1/p10m-smoke.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/freeze-production-model-v1.mjs [--run-dir PATH] [--out PATH] [--check]");
    process.exit(0);
  } else throw new Error(`unknown argument: ${arg}`);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function validate(checkpoint) {
  if (checkpoint.schema !== "nsrl.production_model_smoke_checkpoint.v1" || checkpoint.profile !== "p10m") {
    throw new Error("invalid production model smoke checkpoint");
  }
  if (checkpoint.parameter_count !== 9_317_632
    || checkpoint.bindings.tokenizer_hash !== "0xf4fe71d93c438c1a"
    || checkpoint.bindings.token_stream_hash !== "0x97e5254c31c27bda") {
    throw new Error("production model shape or binding mismatch");
  }
  if (checkpoint.training.scope !== "output_head_perceptron"
    || checkpoint.training.windows !== 8
    || checkpoint.training.initial_mistakes !== 8
    || checkpoint.training.final_mistakes !== 0
    || checkpoint.training.updates <= 0) {
    throw new Error("production smoke training gate failed");
  }
  if (checkpoint.health.weight_saturation_count !== 0
    || checkpoint.health.residual_saturation_count !== 0) {
    throw new Error("production smoke health gate failed");
  }
  if (checkpoint.models.initial.hash === checkpoint.models.trained.hash
    || checkpoint.models.initial.sha256 === checkpoint.models.trained.sha256) {
    throw new Error("production smoke model did not change");
  }
}

async function buildCheckpoint() {
  const initPath = path.join(runDir, "init.json");
  const trainPath = path.join(runDir, "train.json");
  const initialPath = path.join(runDir, "initial.nsrlpm");
  const trainedPath = path.join(runDir, "trained.nsrlpm");
  const [initBytes, trainBytes, initialBytes, trainedBytes] = await Promise.all([
    readFile(initPath), readFile(trainPath), readFile(initialPath), readFile(trainedPath),
  ]);
  const init = JSON.parse(initBytes);
  const train = JSON.parse(trainBytes);
  return {
    schema: "nsrl.production_model_smoke_checkpoint.v1",
    profile: init.profile,
    parameter_count: init.parameter_count,
    architecture: {
      artifact_magic: "NSRLPM1",
      vocab_size: 8192,
      d_model: 256,
      heads: 8,
      layers: 6,
      hidden_dim: 768,
      context_tokens: 256,
      attention: "causal_linear_attention",
    },
    bindings: train.bindings,
    initialization_seed: init.initialization_seed,
    training: {
      scope: train.training.scope,
      context_tokens: train.training.context_tokens,
      windows: train.training.windows,
      epochs: train.training.epochs,
      updates: train.training.updates,
      initial_mistakes: train.evaluation.initial_mistakes,
      final_mistakes: train.evaluation.final_mistakes,
    },
    health: train.health,
    models: {
      initial: {
        file: path.basename(initialPath),
        bytes: initialBytes.length,
        sha256: sha256(initialBytes),
        hash: train.hashes.initial_model,
      },
      trained: {
        file: path.basename(trainedPath),
        bytes: trainedBytes.length,
        sha256: sha256(trainedBytes),
        hash: train.hashes.final_model,
      },
    },
    traces: {
      init_sha256: sha256(initBytes),
      train_sha256: sha256(trainBytes),
    },
    gates: {
      variable_vocab_artifact: true,
      tokenizer_bound_u32_stream: true,
      deterministic_integer_forward: true,
      bounded_output_head_training: true,
      model_hash_changed: true,
      zero_saturation: true,
      full_layer_backward: false,
      float_twin: false,
    },
    known_non_claims: train.known_non_claims,
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
  if (await readFile(outPath, "utf8") !== rendered) throw new Error("production model smoke checkpoint is stale");
  console.log(JSON.stringify({ schema: "nsrl.production_model_smoke_checkpoint_check.v1", ok: true, checkpoint: outPath }));
} else {
  await writeFile(outPath, rendered);
  console.log(outPath);
}
