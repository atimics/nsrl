#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const files = {
  run: "run.json",
  modeling: "modeling.json",
  samples: "samples.jsonl",
  decoder_traces: "decoder-traces.jsonl",
  result: "result.json",
};
const bytes = Object.fromEntries(Object.entries(files).map(([name, file]) => [
  name, fs.readFileSync(path.join(config.runDir, file)),
]));
const run = JSON.parse(bytes.run);
const modeling = JSON.parse(bytes.modeling);
const result = JSON.parse(bytes.result);

assert(run.schema === "nsrl.open_generation_run.v1"
  && modeling.schema === "nsrl.open_generation_modeling.v1"
  && result.schema === "nsrl.open_generation_development_result.v1",
"open-generation development artifacts have invalid schemas");
assert(run.bindings.candidate_model_fnv64 === result.candidate.model_fnv64
  && modeling.bindings.candidate_model_hash === result.candidate.model_fnv64
  && run.bindings.candidate_tokenizer_fnv64 === result.candidate.tokenizer_fnv64
  && modeling.bindings.candidate_tokenizer_hash === result.candidate.tokenizer_fnv64,
"open-generation candidate bindings disagree");
assert(result.promotion_passed === false,
  "development-only evidence must not claim model promotion");

const checkpoint = {
  schema: "nsrl.open_generation_development_checkpoint.v1",
  contract: "open-generation-v1",
  partition: "development",
  candidate: result.candidate,
  execution: {
    decoder: run.execution,
    sampling: run.sampling,
    counts: run.counts,
    cache: run.cache,
    residual_saturation_count: run.residual_saturation_count,
    forbidden_assistance: run.forbidden_assistance,
  },
  modeling: result.modeling,
  generation: {
    thresholds: result.thresholds,
    metrics: result.metrics,
    gates: result.gates,
  },
  evidence_layers: result.evidence_layers,
  missing_evidence: result.missing_evidence,
  promotion_passed: result.promotion_passed,
  artifacts: Object.fromEntries(Object.entries(files).map(([name, file]) => {
    const value = bytes[name];
    return [name, {
      path: path.join(config.runDir, file),
      bytes: value.length,
      fnv64: hex64(fnv64(value)),
      sha256: crypto.createHash("sha256").update(value).digest("hex"),
    }];
  })),
  runner_bindings: {
    generation_source_fnv64: run.bindings.runner_source_fnv64,
    generation_binary_fnv64: run.bindings.runner_binary_fnv64,
    modeling_source_fnv64: modeling.bindings.runner_source_fnv64,
    modeling_binary_fnv64: modeling.bindings.runner_binary_fnv64,
  },
  known_non_claims: [
    "development_baseline_not_model_promotion",
    "required_modeling_baselines_not_measured",
    "human_preference_not_measured",
    "hidden_panel_not_opened_or_scored",
  ],
};
const output = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "open-generation development checkpoint does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  schema: checkpoint.schema,
  checked: config.check,
  candidate: checkpoint.candidate,
  development_generation_passed: checkpoint.generation.gates.development_generation_passed,
  promotion_passed: checkpoint.promotion_passed,
  out: config.out,
})}\n`);

function parseArgs(args) {
  const config = {
    runDir: "data/experiments/open-generation-v1/p10m-kv-scaling-baseline",
    out: "benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json",
    check: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--run-dir") config.runDir = args[++index] || "";
    else if (args[index] === "--out") config.out = args[++index] || "";
    else if (args[index] === "--check") config.check = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  assert(config.runDir && config.out, "--run-dir and --out must not be empty");
  return config;
}

function fnv64(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of value) {
    hash = ((hash ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return hash;
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
