#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadProofManifest } from "./run-integer-transformer-proof-baselines.mjs";

const resultSchema = "nsrl.integer_transformer_proof_result.v1";
const contractId = "integer-transformer-proof-v1";
const resultHeader = "schema\tcontract\tsuite\tpartition\tdataset_hash\tsystem\ttargets\tmistakes\tprobability_error_q15\treplay_hash";
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;

export function buildProofResults(manifest, baselineTsv, candidateTrace) {
  const baselineLines = baselineTsv.trimEnd().split("\n");
  if (baselineLines[0] !== resultHeader || baselineLines.length !== 4) {
    throw new Error("baseline artifact must contain the proof header and three rows");
  }
  const expectedSystems = ["retrieval", "byte-ngram", "float-reference"];
  baselineLines.slice(1).forEach((line, index) => {
    const fields = line.split("\t");
    if (fields.length !== 10
      || fields[0] !== resultSchema
      || fields[1] !== contractId
      || fields[2] !== "substrate"
      || fields[3] !== "eval"
      || fields[4] !== manifest.datasetHash
      || fields[5] !== expectedSystems[index]
      || Number(fields[6]) !== manifest.targets) {
      throw new Error(`baseline row ${index + 2} does not match the frozen manifest`);
    }
  });

  const expectedTokenHash = stableU8SliceHash(manifest.evalBytes);
  if (candidateTrace.schema !== "nsrl.mini_transformer_eval.v1"
    || candidateTrace.data?.token_count !== manifest.evalBytes.length
    || candidateTrace.data?.token_hash !== expectedTokenHash
    || candidateTrace.data?.windows !== manifest.targets
    || candidateTrace.model?.seq_len !== manifest.context
    || candidateTrace.evaluation?.stride !== manifest.stride
    || candidateTrace.evaluation?.invalid_forward_count !== 0) {
    throw new Error("candidate trace does not match the frozen corpus and evaluation geometry");
  }
  for (const field of ["mistakes", "probability_error_q15"]) {
    if (!Number.isSafeInteger(candidateTrace.evaluation[field]) || candidateTrace.evaluation[field] < 0) {
      throw new Error(`candidate ${field} must be a non-negative safe integer`);
    }
  }
  if (!/^0x[0-9a-f]{16}$/i.test(candidateTrace.evaluation.logits_hash)) {
    throw new Error("candidate logits_hash must be a 64-bit hexadecimal replay hash");
  }
  const candidate = [
    resultSchema,
    contractId,
    "substrate",
    "eval",
    manifest.datasetHash,
    "candidate",
    manifest.targets,
    candidateTrace.evaluation.mistakes,
    candidateTrace.evaluation.probability_error_q15,
    candidateTrace.evaluation.logits_hash.toLowerCase(),
  ].join("\t");
  return `${resultHeader}\n${candidate}\n${baselineLines.slice(1).join("\n")}\n`;
}

export function stableU8SliceHash(bytes) {
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(bytes.length));
  let hash = fnvOffset;
  for (const byte of Buffer.concat([length, bytes])) {
    hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function parseArgs(argv) {
  const config = { manifest: "", baselines: "", candidateTrace: "", out: "" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") config.manifest = argv[++index] ?? "";
    else if (arg === "--baselines") config.baselines = argv[++index] ?? "";
    else if (arg === "--candidate-trace") config.candidateTrace = argv[++index] ?? "";
    else if (arg === "--out") config.out = argv[++index] ?? "";
    else throw new Error(`unknown argument: ${arg}`);
  }
  for (const [name, value] of Object.entries(config)) {
    if (!value) throw new Error(`--${name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)} PATH is required`);
  }
  return config;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const manifest = loadProofManifest(config.manifest);
  const output = buildProofResults(
    manifest,
    fs.readFileSync(config.baselines, "utf8"),
    JSON.parse(fs.readFileSync(config.candidateTrace, "utf8")),
  );
  fs.mkdirSync(path.dirname(path.resolve(config.out)), { recursive: true });
  fs.writeFileSync(config.out, output);
  console.log(JSON.stringify({ out: config.out, dataset_hash: manifest.datasetHash, targets: manifest.targets }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
