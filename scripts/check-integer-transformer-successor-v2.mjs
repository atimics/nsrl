#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const directory = path.join(root, "benchmarks/integer-transformer-successor-v2");
const manifestPath = path.join(directory, "manifest.tsv");
const manifestHeader = "schema\tcontract\ttrain\teval\tcontext\tstride\ttargets\tdataset_hash\tcandidate\tcandidate_artifact_hash\tcandidate_hash\tmodel_hash\trunner\trunner_hash\tassistance\tfloat_model\tfloat_model_hash\tfloat_runner\tfloat_runner_hash";
const resultHeader = "schema\tcontract\tsuite\tpartition\tdataset_hash\tcandidate_hash\tmodel_hash\trunner_hash\tassistance_hash\tsystem\ttargets\tmistakes\ttotal_nll_millibits\tzero_probability_windows\treplay_hash";
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;

function hashParts(parts) {
  let hash = fnvOffset;
  for (const part of parts) {
    for (const byte of part) hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
}

const manifestLines = fs.readFileSync(manifestPath, "utf8").trimEnd().split("\n");
assert.equal(manifestLines.length, 2);
assert.equal(manifestLines[0], manifestHeader);
const fields = manifestLines[1].split("\t");
assert.equal(fields.length, 19);
assert.equal(fields[0], "nsrl.integer_transformer_successor_manifest.v2");
assert.equal(fields[1], "integer-transformer-successor-v2");
assert.equal(fields[4], "64");
assert.equal(fields[5], "1");
assert.equal(fields[6], "5896");
assert.equal(fields[7], "0x8fe7b86378f81951");
assert.equal(fields[10], "0x6ffd37de48a3121b");
assert.equal(fields[11], "0x391adc5e1d1a8713");
assert.equal(fields[14], "suffix-memory=off,retrieval=off,routing-oracle=off");

const train = fs.readFileSync(path.resolve(directory, fields[2]));
const evaluation = fs.readFileSync(path.resolve(directory, fields[3]));
assert.equal(hashParts([train, Buffer.from([255]), evaluation]), fields[7]);
assert.equal(evaluation.length - Number(fields[4]), Number(fields[6]));
for (const [fileIndex, hashIndex] of [[8, 9], [12, 13], [15, 16], [17, 18]]) {
  const bytes = fs.readFileSync(path.resolve(directory, fields[fileIndex]));
  assert.equal(hashParts([bytes]), fields[hashIndex], `${fields[fileIndex]} hash drifted`);
}

const candidate = readJson("candidate.eval.json");
assert.equal(candidate.schema, "nsrl.mini_transformer_eval.v1");
assert.equal(candidate.data.windows, 5896);
assert.equal(candidate.model.hash, fields[11]);
assert.deepEqual(candidate.ablation, {
  mode: "transformer-only",
  source_model_hash: fields[10],
  evaluated_model_hash: fields[11],
  source_suffix_memory_present: true,
  suffix_memory_enabled: false,
  retrieval_enabled: false,
  routing_oracle_enabled: false,
});
assert.equal(candidate.evaluation.invalid_forward_count, 0);
assert.equal(candidate.evaluation.mistakes, 5094);

const floatTransformer = readJson("float-transformer.eval.json");
assert.equal(floatTransformer.schema, "nsrl.float_transformer_eval.v1");
assert.equal(floatTransformer.dataset_hash, fields[7]);
assert.equal(floatTransformer.targets, 5896);
assert.equal(floatTransformer.model_hash, fields[16]);
assert.equal(floatTransformer.runner_hash, fields[18]);
assert.equal(floatTransformer.architecture.kind, "causal-float-transformer");
assert.equal(floatTransformer.architecture.attention, "scaled-dot-product-softmax");
assert.equal(floatTransformer.training.trained_parameters, "all");
assert.equal(Object.keys(floatTransformer.training.moved_values).length, 10);
assert(Object.values(floatTransformer.training.moved_values).every((count) => count > 0));
assert(floatTransformer.training.final_batch_nll_nats < floatTransformer.training.first_batch_nll_nats);

const resultLines = fs.readFileSync(path.join(directory, "results.tsv"), "utf8").trimEnd().split("\n");
assert.equal(resultLines[0], resultHeader);
assert.equal(resultLines.length, 6);
const expected = [
  ["transformer-only", 5094, 115010055, 2916, "0xa53bf1c82509f825"],
  ["uniform", 5896, 47168000, 0, "0x71ff602c473f82ad"],
  ["retrieval", 2505, 38293936, 0, "0x50c878d3bfa8eafa"],
  ["byte-ngram", 2574, 36952920, 0, "0x01abae2938a3a85a"],
  ["float-transformer", 4497, 23216345, 9, "0x0f1b504ce4e4531b"],
];
for (let index = 0; index < expected.length; index += 1) {
  const row = resultLines[index + 1].split("\t");
  assert.equal(row.length, 15);
  assert.equal(row[4], fields[7]);
  assert.equal(row[5], fields[10]);
  assert.equal(row[6], fields[11]);
  assert.equal(row[7], fields[13]);
  assert.equal(row[8], "0x83e30b9ff0fe6c77");
  assert.deepEqual(
    [row[9], Number(row[11]), Number(row[12]), Number(row[13]), row[14]],
    expected[index],
  );
}

const check = readJson("check.json");
assert.equal(check.schema, "nsrl.integer_transformer_successor_result.v2");
assert.equal(check.dataset_hash, fields[7]);
assert.equal(check.targets, 5896);
assert.equal(check.candidate_hash, fields[10]);
assert.equal(check.model_hash, fields[11]);
assert.equal(check.runner_hash, fields[13]);
assert.equal(check.float_model_hash, fields[16]);
assert.equal(check.float_runner_hash, fields[18]);
assert.equal(check.passed, false);
assert.equal(check.candidate.total_nll_millibits, expected[0][2]);
assert.deepEqual(check.baselines.map((row) => row.system), expected.slice(1).map((row) => row[0]));
assert(check.baselines.every((row) => check.candidate.total_nll_millibits > row.total_nll_millibits));

console.log(JSON.stringify({
  checked: true,
  contract: fields[1],
  dataset_hash: fields[7],
  targets: Number(fields[6]),
  status: "falsified",
  candidate_total_nll_millibits: check.candidate.total_nll_millibits,
  best_baseline: check.baselines.reduce((best, row) =>
    row.total_nll_millibits < best.total_nll_millibits ? row : best),
}));
