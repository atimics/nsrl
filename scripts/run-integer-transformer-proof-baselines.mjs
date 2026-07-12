#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const manifestSchema = "nsrl.integer_transformer_proof_manifest.v1";
const resultSchema = "nsrl.integer_transformer_proof_result.v1";
const contractId = "integer-transformer-proof-v1";
const resultHeader = "schema\tcontract\tsuite\tpartition\tdataset_hash\tsystem\ttargets\tmistakes\tprobability_error_q15\treplay_hash";
const q15One = 32767;
const byteClasses = 256;
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;

export function loadProofManifest(manifestPath, { allowUnfrozen = false } = {}) {
  const absolute = path.resolve(manifestPath);
  const lines = fs.readFileSync(absolute, "utf8").trimEnd().split("\n");
  const expectedHeader = "schema\tcontract\ttrain\teval\tcontext\tstride\tmin_targets\tdataset_hash";
  if (lines.length !== 2 || lines[0] !== expectedHeader) {
    throw new Error(`manifest must contain header ${expectedHeader} and one row`);
  }
  const fields = lines[1].split("\t");
  if (fields.length !== 8 || fields[0] !== manifestSchema || fields[1] !== contractId) {
    throw new Error("manifest schema or contract does not match integer-transformer-proof-v1");
  }
  const directory = path.dirname(absolute);
  const trainPath = path.resolve(directory, fields[2]);
  const evalPath = path.resolve(directory, fields[3]);
  const context = positiveInteger(fields[4], "context");
  const stride = positiveInteger(fields[5], "stride");
  const minTargets = positiveInteger(fields[6], "min_targets");
  const train = fs.readFileSync(trainPath);
  const evalBytes = fs.readFileSync(evalPath);
  const datasetHash = hex64(hashParts([train, Buffer.from([255]), evalBytes]));
  if (!allowUnfrozen && fields[7] !== datasetHash) {
    throw new Error(`manifest dataset_hash ${fields[7]} does not match ${datasetHash}`);
  }
  const targets = targetCount(evalBytes.length, context, stride);
  if (targets < minTargets) {
    throw new Error(`benchmark has ${targets} targets, below frozen minimum ${minTargets}`);
  }
  return {
    absolute,
    trainPath,
    evalPath,
    train,
    evalBytes,
    context,
    stride,
    minTargets,
    datasetHash,
    targets,
  };
}

export function generateBaselineTsv(manifest) {
  const ngramTables = buildTables(manifest.train, 4);
  const retrievalTables = buildTables(manifest.train, manifest.context);
  const systems = [
    evaluateSystem(manifest, "retrieval", (context) => {
      for (const order of [manifest.context, 32, 16, 8, 4, 2]) {
        if (order > context.length) continue;
        const counts = lookup(retrievalTables, context, order);
        if (counts) return quantizeIntegerCounts(counts, 1);
      }
      return quantizeIntegerCounts(ngramTables[0].get(""), 1);
    }),
    evaluateSystem(manifest, "byte-ngram", (context) => {
      for (let order = 4; order >= 1; order -= 1) {
        const counts = lookup(ngramTables, context, order);
        if (counts) return quantizeIntegerCounts(counts, 1);
      }
      return quantizeIntegerCounts(ngramTables[0].get(""), 1);
    }),
    evaluateSystem(manifest, "float-reference", (context) => {
      const weights = [0.05, 0.1, 0.15, 0.25, 0.45];
      const mixed = Array(byteClasses).fill(0);
      let usedWeight = 0;
      for (let order = 0; order <= 4; order += 1) {
        const counts = order === 0 ? ngramTables[0].get("") : lookup(ngramTables, context, order);
        if (!counts) continue;
        const distribution = floatingDistribution(counts, 0.5);
        for (let byte = 0; byte < byteClasses; byte += 1) {
          mixed[byte] += distribution[byte] * weights[order];
        }
        usedWeight += weights[order];
      }
      return quantizeFloatingProbabilities(mixed.map((value) => value / usedWeight));
    }),
  ];
  return `${resultHeader}\n${systems.map((row) => resultRow(manifest, row)).join("\n")}\n`;
}

function evaluateSystem(manifest, system, distributionForContext) {
  let mistakes = 0;
  let probabilityErrorQ15 = 0;
  let replay = hashBytes(Buffer.from(`${contractId}\0${system}\0`, "utf8"));
  for (let start = 0; start + manifest.context < manifest.evalBytes.length; start += manifest.stride) {
    const end = start + manifest.context;
    const context = manifest.evalBytes.subarray(start, end);
    const target = manifest.evalBytes[end];
    const probabilities = distributionForContext(context);
    const predicted = argmax(probabilities);
    if (predicted !== target) mistakes += 1;
    let error = q15One - probabilities[target];
    for (let byte = 0; byte < byteClasses; byte += 1) {
      if (byte !== target) error += probabilities[byte];
    }
    probabilityErrorQ15 += error;
    replay = hashUpdate(replay, predicted);
    replay = hashUpdate(replay, target);
    replay = hashUpdate(replay, probabilities[target] & 255);
    replay = hashUpdate(replay, probabilities[target] >> 8);
  }
  return { system, targets: manifest.targets, mistakes, probabilityErrorQ15, replayHash: hex64(replay) };
}

function buildTables(bytes, maxOrder) {
  const tables = Array.from({ length: maxOrder + 1 }, () => new Map());
  tables[0].set("", new Uint32Array(byteClasses));
  for (let targetIndex = 0; targetIndex < bytes.length; targetIndex += 1) {
    tables[0].get("")[bytes[targetIndex]] += 1;
    const available = Math.min(maxOrder, targetIndex);
    for (let order = 1; order <= available; order += 1) {
      const key = bytes.subarray(targetIndex - order, targetIndex).toString("hex");
      let counts = tables[order].get(key);
      if (!counts) {
        counts = new Uint32Array(byteClasses);
        tables[order].set(key, counts);
      }
      counts[bytes[targetIndex]] += 1;
    }
  }
  return tables;
}

function lookup(tables, context, order) {
  return tables[order]?.get(context.subarray(context.length - order).toString("hex"));
}

function quantizeIntegerCounts(counts, smoothing) {
  const total = counts.reduce((sum, count) => sum + count, 0) + smoothing * byteClasses;
  const values = Array(byteClasses).fill(0);
  const remainders = [];
  let assigned = 0;
  for (let byte = 0; byte < byteClasses; byte += 1) {
    const numerator = (counts[byte] + smoothing) * q15One;
    values[byte] = Math.floor(numerator / total);
    assigned += values[byte];
    remainders.push({ byte, remainder: numerator % total });
  }
  remainders.sort((left, right) => right.remainder - left.remainder || left.byte - right.byte);
  for (let index = 0; index < q15One - assigned; index += 1) values[remainders[index].byte] += 1;
  return values;
}

function floatingDistribution(counts, smoothing) {
  const total = counts.reduce((sum, count) => sum + count, 0) + smoothing * byteClasses;
  return Array.from(counts, (count) => (count + smoothing) / total);
}

function quantizeFloatingProbabilities(probabilities) {
  const scaled = probabilities.map((value, byte) => {
    const exact = value * q15One;
    return { byte, value: Math.floor(exact), remainder: exact - Math.floor(exact) };
  });
  let assigned = scaled.reduce((sum, entry) => sum + entry.value, 0);
  scaled.sort((left, right) => right.remainder - left.remainder || left.byte - right.byte);
  for (let index = 0; assigned < q15One; index += 1, assigned += 1) scaled[index].value += 1;
  scaled.sort((left, right) => left.byte - right.byte);
  return scaled.map((entry) => entry.value);
}

function resultRow(manifest, row) {
  return [
    resultSchema,
    contractId,
    "substrate",
    "eval",
    manifest.datasetHash,
    row.system,
    row.targets,
    row.mistakes,
    row.probabilityErrorQ15,
    row.replayHash,
  ].join("\t");
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function targetCount(bytes, context, stride) {
  if (bytes <= context) return 0;
  return Math.ceil((bytes - context) / stride);
}

function hashParts(parts) {
  let hash = fnvOffset;
  for (const part of parts) {
    for (const byte of part) hash = hashUpdate(hash, byte);
  }
  return hash;
}

function hashBytes(bytes) {
  return hashParts([bytes]);
}

function hashUpdate(hash, byte) {
  return ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function positiveInteger(value, name) {
  if (!/^[1-9][0-9]*$/.test(value)) throw new Error(`${name} must be a positive integer`);
  return Number(value);
}

function parseArgs(argv) {
  const config = { manifest: "", out: "", printHash: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") config.manifest = argv[++index] ?? "";
    else if (arg === "--out") config.out = argv[++index] ?? "";
    else if (arg === "--print-hash") config.printHash = true;
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (!config.manifest) throw new Error("--manifest PATH is required");
  if (!config.printHash && !config.out) throw new Error("--out PATH is required");
  return config;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const manifest = loadProofManifest(config.manifest, { allowUnfrozen: config.printHash });
  if (config.printHash) {
    console.log(manifest.datasetHash);
    return;
  }
  const output = generateBaselineTsv(manifest);
  fs.mkdirSync(path.dirname(path.resolve(config.out)), { recursive: true });
  fs.writeFileSync(config.out, output);
  console.log(JSON.stringify({ out: config.out, dataset_hash: manifest.datasetHash, targets: manifest.targets }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
