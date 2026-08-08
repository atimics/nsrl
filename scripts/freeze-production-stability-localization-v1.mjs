#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let contractPath = "";
let runDir = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !runDir || !outPath) {
  throw new Error("--contract, --run-dir, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const contract = await json(contractPath);
assert(contract.schema === "nsrl.production_stability_localization_contract.v1",
  "unexpected localization contract schema");

for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.source.optimizer_path, sha256: contract.source.optimizer_sha256 },
  { path: contract.source.trace_path, sha256: contract.source.trace_sha256 },
  ...contract.derivation.artifacts,
]) {
  assert(sha256(await readFile(artifact.path)) === artifact.sha256,
    `artifact hash mismatch: ${artifact.path}`);
}

const midpointTrace = await json(contract.source.trace_path);
assert(midpointTrace.hashes.final_model === contract.source.model_hash,
  "midpoint model hash does not match contract");
assert(midpointTrace.hashes.optimizer_state === contract.source.optimizer_state_hash,
  "midpoint optimizer hash does not match contract");
assert(midpointTrace.training.total_optimizer_step === contract.source.total_optimizer_step,
  "midpoint optimizer step does not match contract");
assert(same(midpointTrace.training.learning_rate_shifts, contract.training.learning_rate_shifts),
  "midpoint schedule does not match localization contract");

const frozen = contract.gates.frozen_parameter_groups;
const trainingSaturation = (trace) => Object.values(trace.health)
  .reduce((sum, value) => sum + value, 0);
const deltaValue = (group) => group.total ?? group;
const pointPaths = (step) => {
  if (step === 256) return {
    trace: contract.source.trace_path,
    development: "benchmarks/production-model-v1/p10m-causal-tail-representation-v2-midpoint-development.json",
    saturation: "benchmarks/open-generation-v1/p10m-causal-tail-representation-v2-midpoint-residual-saturation.json",
    delta: "benchmarks/production-model-v1/p10m-causal-tail-representation-v2-midpoint-parameter-delta.json",
  };
  if (step === 512) return {
    trace: "data/experiments/production-model-v1/p10m-causal-tail-representation-v2/train-final.json",
    development: "data/experiments/production-model-v1/p10m-causal-tail-representation-v2/candidate-dev.json",
    saturation: "benchmarks/open-generation-v1/p10m-causal-tail-representation-v2-residual-saturation.json",
    delta: "benchmarks/production-model-v1/p10m-causal-tail-representation-v2-final-parameter-delta.json",
  };
  return {
    trace: path.join(runDir, `train-step-${step}.json`),
    development: path.join(runDir, `development-step-${step}.json`),
    saturation: path.join(runDir, `saturation-step-${step}.json`),
    delta: path.join(runDir, `delta-step-${step}.json`),
  };
};

const steps = [
  contract.derivation.known_safe_total_optimizer_step,
  ...contract.derivation.probe_total_optimizer_steps,
  contract.derivation.known_unsafe_total_optimizer_step,
];
const points = [];
for (const step of steps) {
  const files = pointPaths(step);
  const [trace, development, saturation, delta] = await Promise.all([
    json(files.trace), json(files.development), json(files.saturation), json(files.delta),
  ]);
  assert(trace.training.total_optimizer_step === step,
    `trace optimizer step mismatch at ${step}`);
  assert(same(trace.training.learning_rate_shifts, contract.training.learning_rate_shifts),
    `trace schedule mismatch at ${step}`);
  assert(development.model_hash === trace.hashes.final_model,
    `development model mismatch at ${step}`);
  assert(saturation.bindings.model_hash === trace.hashes.final_model,
    `saturation model mismatch at ${step}`);
  assert(delta.bindings.candidate_model_hash === trace.hashes.final_model,
    `delta model mismatch at ${step}`);
  assert(development.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
    `development stream mismatch at ${step}`);
  const frozenUnchanged = frozen.every((group) => deltaValue(delta.groups[group]).l1 === 0);
  const trainSat = trainingSaturation(trace);
  const inferenceSat = saturation.aggregate.residual_saturation_count;
  const developmentTotal = development.evaluation.total_nll_millibits;
  const developmentDelta = developmentTotal - contract.baseline.development_total_nll_millibits;
  const safe = developmentDelta < 0 && trainSat === 0 && inferenceSat === 0 && frozenUnchanged;
  points.push({
    total_optimizer_step: step,
    model_hash: trace.hashes.final_model,
    development_total_nll_millibits: developmentTotal,
    development_delta_from_full_v1_millibits: developmentDelta,
    training_saturation_count: trainSat,
    manifest_residual_saturation_count: inferenceSat,
    frozen_parameter_groups_unchanged: frozenUnchanged,
    movement_l1: Object.fromEntries(Object.entries(delta.groups)
      .map(([group, value]) => [group, deltaValue(value).l1])),
    safe,
  });
}

assert(points[0].safe, "contracted safe endpoint is not safe");
assert(!points.at(-1).safe, "contracted unsafe endpoint is not unsafe");
let firstUnsafeInterval = null;
for (let index = 1; index < points.length; index += 1) {
  if (points[index - 1].safe && !points[index].safe) {
    firstUnsafeInterval = {
      safe_total_optimizer_step: points[index - 1].total_optimizer_step,
      unsafe_total_optimizer_step: points[index].total_optimizer_step,
    };
    break;
  }
}
assert(firstUnsafeInterval !== null,
  "localization did not find an adjacent safe-to-unsafe interval");

const result = {
  schema: "nsrl.production_stability_localization.v1",
  checked: check,
  objective: contract.objective,
  source_model_hash: contract.source.model_hash,
  baseline_model_hash: contract.baseline.model_hash,
  points,
  first_unsafe_interval: firstUnsafeInterval,
  gates: {
    known_safe_endpoint_confirmed: points[0].safe,
    known_unsafe_endpoint_confirmed: !points.at(-1).safe,
    frozen_parameter_groups_unchanged_at_all_points:
      points.every((point) => point.frozen_parameter_groups_unchanged),
    adjacent_safe_to_unsafe_interval_found: firstUnsafeInterval !== null,
    test_partition_not_read: true,
  },
  authorization: {
    diagnostic_only: true,
    test_evaluation: false,
    quality_promotion: false,
    open_generation_rerun: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered,
    "localization checkpoint differs from recomputed evidence");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(JSON.stringify({
  schema: result.schema,
  checked: check,
  first_unsafe_interval: result.first_unsafe_interval,
  points: result.points.map((point) => ({
    step: point.total_optimizer_step,
    development_delta: point.development_delta_from_full_v1_millibits,
    training_saturation: point.training_saturation_count,
    inference_saturation: point.manifest_residual_saturation_count,
    safe: point.safe,
  })),
  out: outPath,
}) + "\n");
