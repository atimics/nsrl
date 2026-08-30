#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const experiment = path.join(root, "benchmarks/q22-solomon-prospective-v1");
const evidence = path.join(experiment, "evidence");
const expectedContract = "b3b2e2d2648802ac99395c0d1110207409f240cc5e4b83d8c89186c36238ccc0";
const expectedEvaluation = "9270ea2b72af90235407bd7924a0864b8eba35b2969e1657ed1c15bf04449519";

assert(sha256(path.join(experiment, "contract.json")) === expectedContract, "contract digest drifted");
verifyManifest();

const result = readJson("result.json");
const freeze = readJson("models-frozen.json");
const opened = readJson("evaluation-opened.json");
assert(result.schema === "nsrl.q22_solomon_prospective_result.v1", "result schema drifted");
assert(result.contract_sha256 === expectedContract, "result contract binding drifted");
assert(result.eval_sha256 === expectedEvaluation, "result evaluation binding drifted");
assert(result.outcome === "go" && result.family_passed === true, "frozen outcome is not go");
assert(result.minimum_operation_exact_rate_ppm === 1000000, "minimum seed exact rate drifted");
assert(result.mean_operation_exact_rate_ppm === 1000000, "mean seed exact rate drifted");
assert(result.all_seed_agreement_cases === 500, "agreement case count drifted");
assert(result.all_seed_agreement_rate_ppm === 1000000, "agreement rate drifted");

assert(freeze.schema === "nsrl.q22_model_freeze.v1", "model freeze schema drifted");
assert(freeze.contract_sha256 === expectedContract, "model freeze contract binding drifted");
assert(freeze.evaluation_opened === false, "models were not frozen before evaluation");
assert(sha256(path.join(evidence, "models-frozen.json")) === result.model_freeze_sha256, "result model freeze binding drifted");
assert(opened.schema === "nsrl.q22_evaluation_open.v1", "evaluation-open schema drifted");
assert(opened.contract_sha256 === expectedContract, "evaluation-open contract binding drifted");
assert(opened.model_freeze_sha256 === result.model_freeze_sha256, "evaluation opened against different models");
assert(opened.eval_sha256 === expectedEvaluation, "evaluation-open dataset binding drifted");
assert(opened.retraining_allowed === false, "evaluation record permits retraining");
assert(sha256(path.join(evidence, "promotion-inputs.blind.tsv")) === opened.blind_inputs_sha256, "blind input binding drifted");

assert(freeze.models.length === 3, "model freeze must contain three seeds");
const predictions = [];
for (const seed of [1, 2, 3]) {
  const model = freeze.models.find((entry) => entry.seed === seed);
  assert(model, `seed ${seed} is absent from the model freeze`);
  assert(sha256(path.join(evidence, model.model)) === model.model_sha256, `seed ${seed} model drifted`);
  assert(sha256(path.join(evidence, model.train_trace)) === model.train_trace_sha256, `seed ${seed} train trace drifted`);
  const check = readJson(`seed${seed}.check.json`);
  const recorded = result.seeds.find((entry) => entry.seed === seed);
  assert(recorded && JSON.stringify(recorded) === JSON.stringify({ seed, ...check }), `seed ${seed} result/check mismatch`);
  assert(check.cases === 500 && check.operation_exact === 500 && check.operation_exact_rate_ppm === 1000000 && check.valid === true, `seed ${seed} exact check drifted`);
  predictions.push(fs.readFileSync(path.join(evidence, `seed${seed}.predictions.tsv`)));
}
assert(predictions[0].equals(predictions[1]) && predictions[0].equals(predictions[2]), "cross-seed predictions diverged");

const blindLines = fs.readFileSync(path.join(evidence, "promotion-inputs.blind.tsv"), "utf8").trimEnd().split("\n");
assert(blindLines.shift() === "id\tinput", "blind input header drifted");
assert(blindLines.length === 500 && blindLines.every((line) => line.split("\t").length === 2), "blind input contains an unexpected column or row count");

console.log(JSON.stringify({
  schema: "nsrl.q22_solomon_evidence_check.v1",
  outcome: result.outcome,
  seeds: result.seeds.map((entry) => ({ seed: entry.seed, operation_exact_rate_ppm: entry.operation_exact_rate_ppm })),
  all_seed_agreement_rate_ppm: result.all_seed_agreement_rate_ppm,
  valid: true,
}));

function verifyManifest() {
  const lines = fs.readFileSync(path.join(evidence, "MANIFEST.sha256"), "utf8").trimEnd().split("\n");
  const expectedFiles = [];
  for (const line of lines) {
    const match = /^([0-9a-f]{64})  ([A-Za-z0-9.-]+)$/.exec(line);
    assert(match, "evidence manifest contains an invalid row");
    const [, expected, name] = match;
    expectedFiles.push(name);
    assert(sha256(path.join(evidence, name)) === expected, `evidence artifact drifted: ${name}`);
  }
  const actualFiles = fs.readdirSync(evidence).filter((name) => name !== "MANIFEST.sha256").sort();
  assert(JSON.stringify(actualFiles) === JSON.stringify(expectedFiles.sort()), "evidence manifest coverage drifted");
}

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(evidence, name), "utf8"));
}

function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
