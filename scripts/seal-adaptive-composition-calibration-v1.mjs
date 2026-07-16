#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const precontractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-precalibration-contract.json";
const calibrationDirectory = process.argv[3]
  ?? "data/experiments/production-model-v1/p10m-adaptive-composition-v1/execution";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json";

const fail = (message) => { throw new Error(`adaptive composition calibration seal: ${message}`); };
const assert = (condition, message) => { if (!condition) fail(message); };
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const bind = (artifactPath) => {
  const bytes = fs.readFileSync(artifactPath);
  return {path: artifactPath, sha256: sha256(bytes), bytes: bytes.length};
};
const verifyBindings = (value, label = "bindings") => {
  if (!value || typeof value !== "object") return;
  if (typeof value.path === "string" && typeof value.sha256 === "string"
    && Number.isSafeInteger(value.bytes)) {
    const actual = bind(value.path);
    assert(actual.sha256 === value.sha256 && actual.bytes === value.bytes,
      `${label} changed after pre-calibration freeze`);
    return;
  }
  for (const [key, child] of Object.entries(value)) verifyBindings(child, `${label}.${key}`);
};

const precontractBytes = fs.readFileSync(precontractPath);
const precontract = JSON.parse(precontractBytes);
assert(precontract.schema === "nsrl.adaptive_composition_execution_contract.v1"
  && precontract.analysis_role === "frozen_after_fitting_before_calibration",
"wrong pre-calibration contract");
verifyBindings(precontract.bindings);

const calibrationManifestPath = path.join(calibrationDirectory, "calibration-manifest.json");
const calibrationManifest = JSON.parse(fs.readFileSync(calibrationManifestPath));
assert(calibrationManifest.schema === "nsrl.adaptive_composition_calibration.v1"
  && calibrationManifest.analysis_role === "calibration_only_before_adaptive_endpoint"
  && calibrationManifest.cube_rows === 19_992
  && calibrationManifest.source_scores === 357
  && !calibrationManifest.adaptive_outcomes_read
  && !calibrationManifest.endpoint_outcomes_read,
"calibration manifest crossed the adaptive or endpoint firewall");

const contract = {
  ...precontract,
  analysis_role: "frozen_after_calibration_before_adaptive_endpoint",
  bindings: {
    ...precontract.bindings,
    precalibration_contract: {
      path: precontractPath, sha256: sha256(precontractBytes), bytes: precontractBytes.length,
    },
    calibration: {
      cube: bind(path.join(calibrationDirectory, "calibration-cube.tsv")),
      scores: bind(path.join(calibrationDirectory, "calibration-scores.tsv")),
      corrections: bind(path.join(calibrationDirectory, "corrections.tsv")),
      manifest: bind(calibrationManifestPath),
    },
  },
  calibration: {
    ...precontract.calibration,
    artifacts_frozen_before_adaptive_endpoint: true,
    adaptive_outcomes_read_at_freeze: false,
    endpoint_outcomes_read_at_freeze: false,
  },
};
const bytes = Buffer.from(`${JSON.stringify(contract, null, 2)}\n`);
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: contract.schema, output: outputPath, sha256: sha256(bytes),
  calibration_cube_frozen: true, adaptive_outcomes_read: false,
  endpoint_outcomes_read: false,
}, null, 2)}\n`);
