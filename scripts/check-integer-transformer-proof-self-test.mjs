#!/usr/bin/env node

import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  generateBaselineTsv,
  loadProofManifest,
} from "./run-integer-transformer-proof-baselines.mjs";
import {
  buildProofResults,
  stableU8SliceHash,
} from "./build-integer-transformer-proof-results.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = path.join(root, "benchmarks/integer-transformer-proof-v1/manifest.tsv");
const manifest = loadProofManifest(manifestPath);
const first = generateBaselineTsv(manifest);
const second = generateBaselineTsv(manifest);
assert.equal(first, second, "baseline generation must be byte deterministic");
const lines = first.trimEnd().split("\n");
assert.equal(lines.length, 4);
assert.deepEqual(
  lines.slice(1).map((line) => line.split("\t")[5]),
  ["retrieval", "byte-ngram", "float-reference"],
);
assert(lines.slice(1).every((line) => line.includes(`\t${manifest.datasetHash}\t`)));
assert(manifest.targets >= manifest.minTargets);
const candidateTrace = {
  schema: "nsrl.mini_transformer_eval.v1",
  data: {
    token_count: manifest.evalBytes.length,
    token_hash: stableU8SliceHash(manifest.evalBytes),
    windows: manifest.targets,
  },
  model: { seq_len: manifest.context },
  evaluation: {
    stride: manifest.stride,
    mistakes: manifest.targets,
    probability_error_q15: manifest.targets * 65500,
    invalid_forward_count: 0,
    logits_hash: "0xaaaaaaaaaaaaaaaa",
  },
};
const full = buildProofResults(manifest, first, candidateTrace);
assert.equal(full.trimEnd().split("\n").length, 5);
assert.equal(full.trimEnd().split("\n")[1].split("\t")[5], "candidate");
assert.throws(() => buildProofResults(
  manifest,
  first,
  { ...candidateTrace, data: { ...candidateTrace.data, windows: manifest.targets - 1 } },
));
console.log(JSON.stringify({ passed: true, dataset_hash: manifest.datasetHash, targets: manifest.targets }));
