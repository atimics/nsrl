#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const experiment = path.join(root, "benchmarks/q22-compositional-solomon-prospective-v1");
const evidence = path.join(experiment, "evidence");
const expectedContract = "f5fdd260ae7e7ef2fdab85bb4aaf0aaebb2716015b3e8026a677aaf8fa8d0ae4";
const expectedEvaluation = "42902d61d9edc2ceb5a33875deddbdd9e9b523b06fb1aa13a22249acbf2983fe";
const operations = [
  "quantity.add",
  "quantity.multiply",
  "quantity.add-rational",
  "quantity.convert",
  "quantity.solve-linear",
];
const expectedSeedRates = new Map([[1, 425000], [2, 430000], [3, 533000]]);
const expectedPerClass = new Map([
  [1, [0, 595000, 750000, 750000, 30000]],
  [2, [165000, 750000, 410000, 750000, 75000]],
  [3, [250000, 605000, 295000, 960000, 555000]],
]);

assert(sha256(path.join(experiment, "contract.json")) === expectedContract, "contract digest drifted");
verifyManifest();

const result = readJson("result.json");
const freeze = readJson("models-frozen.json");
const opened = readJson("evaluation-opened.json");
assert(result.schema === "nsrl.q22_compositional_solomon_prospective_result.v1", "result schema drifted");
assert(result.contract_sha256 === expectedContract, "result contract binding drifted");
assert(result.eval_sha256 === expectedEvaluation, "result evaluation binding drifted");
assert(result.outcome === "no_go" && result.family_passed === false, "frozen outcome is not no_go");
assert(result.minimum_operation_exact_rate_ppm === 425000, "minimum seed exact rate drifted");
assert(result.minimum_per_class_exact_rate_ppm === 0, "minimum seed-class rate drifted");
assert(result.mean_operation_exact_rate_ppm === 462666, "mean seed exact rate drifted");
assert(result.prefix_only_exact_rate_ppm === 200000, "prefix-only baseline drifted");
assert(result.minimum_margin_over_prefix_ppm === 225000, "minimum baseline margin drifted");
assert(result.all_seed_agreement_cases === 531, "agreement case count drifted");
assert(result.all_seed_agreement_rate_ppm === 531000, "agreement rate drifted");

assert(freeze.schema === "nsrl.q22_compositional_model_freeze.v1", "model freeze schema drifted");
assert(freeze.contract_sha256 === expectedContract, "model freeze contract binding drifted");
assert(freeze.evaluation_opened === false, "models were not frozen before evaluation");
assert(sha256(path.join(evidence, "models-frozen.json")) === result.model_freeze_sha256, "result model freeze binding drifted");
assert(opened.schema === "nsrl.q22_compositional_evaluation_open.v1", "evaluation-open schema drifted");
assert(opened.contract_sha256 === expectedContract, "evaluation-open contract binding drifted");
assert(opened.model_freeze_sha256 === result.model_freeze_sha256, "evaluation opened against different models");
assert(opened.eval_sha256 === expectedEvaluation, "evaluation-open dataset binding drifted");
assert(opened.retraining_allowed === false, "evaluation record permits retraining");
assert(sha256(path.join(evidence, "promotion-inputs.blind.tsv")) === opened.blind_inputs_sha256, "blind input binding drifted");

assert(freeze.models.length === 3, "model freeze must contain three seeds");
const predictionSets = [];
for (const seed of [1, 2, 3]) {
  const model = freeze.models.find((entry) => entry.seed === seed);
  assert(model, `seed ${seed} is absent from the model freeze`);
  assert(sha256(path.join(evidence, model.model)) === model.model_sha256, `seed ${seed} model drifted`);
  assert(sha256(path.join(evidence, model.train_trace)) === model.train_trace_sha256, `seed ${seed} train trace drifted`);
  const check = readJson(`seed${seed}.check.json`);
  const recorded = result.seeds.find((entry) => entry.seed === seed);
  assert(recorded && JSON.stringify(recorded) === JSON.stringify({ seed, ...check }), `seed ${seed} result/check mismatch`);
  assert(check.cases === 1000 && check.valid === true, `seed ${seed} exact check is invalid`);
  assert(check.operation_exact_rate_ppm === expectedSeedRates.get(seed), `seed ${seed} exact rate drifted`);
  assert(JSON.stringify(Object.keys(check.per_class_exact_rate_ppm)) === JSON.stringify(operations), `seed ${seed} class order drifted`);
  assert(JSON.stringify(Object.values(check.per_class_exact_rate_ppm)) === JSON.stringify(expectedPerClass.get(seed)), `seed ${seed} per-class rates drifted`);
  predictionSets.push(readPredictions(`seed${seed}.predictions.tsv`));
}

const ids = [...predictionSets[0].keys()];
assert(ids.length === 1000, "prediction count drifted");
assert(predictionSets.every((rows) => rows.size === ids.length && ids.every((id) => rows.has(id))), "seed prediction identities disagree");
const agreements = ids.filter((id) => predictionSets.every((rows) => rows.get(id) === predictionSets[0].get(id))).length;
assert(agreements === 531, "recomputed cross-seed agreement drifted");

const blindLines = fs.readFileSync(path.join(evidence, "promotion-inputs.blind.tsv"), "utf8").trimEnd().split("\n");
assert(blindLines.shift() === "id\tinput", "blind input header drifted");
assert(blindLines.length === 1000 && blindLines.every((line) => line.split("\t").length === 2), "blind input contains an unexpected column or row count");

console.log(JSON.stringify({
  schema: "nsrl.q22_compositional_solomon_evidence_check.v1",
  outcome: result.outcome,
  seeds: result.seeds.map((entry) => ({ seed: entry.seed, operation_exact_rate_ppm: entry.operation_exact_rate_ppm })),
  minimum_per_class_exact_rate_ppm: result.minimum_per_class_exact_rate_ppm,
  all_seed_agreement_rate_ppm: result.all_seed_agreement_rate_ppm,
  valid: true,
}));

function readPredictions(name) {
  const lines = fs.readFileSync(path.join(evidence, name), "utf8").trimEnd().split("\n");
  assert(lines.shift() === "id\tmodel_request", `${name} header drifted`);
  const rows = new Map();
  for (const line of lines) {
    const fields = line.split("\t");
    assert(fields.length === 2 && !rows.has(fields[0]) && operations.includes(fields[1]), `${name} contains an invalid row`);
    rows.set(fields[0], fields[1]);
  }
  return rows;
}

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
