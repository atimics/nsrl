#!/usr/bin/env node

import { readFile } from "node:fs/promises";

const [modelPath, tokenPath, contextArg = "64", windowsArg = "32"] = process.argv.slice(2);
if (!modelPath || !tokenPath) {
  throw new Error(
    "usage: analyze-production-attention-selectivity-v1.mjs MODEL.nsrlpm TOKENS.nsrltok [context] [windows]",
  );
}

const contextTokens = Number.parseInt(contextArg, 10);
const maximumWindows = Number.parseInt(windowsArg, 10);
if (!Number.isSafeInteger(contextTokens) || contextTokens <= 1
    || !Number.isSafeInteger(maximumWindows) || maximumWindows <= 0) {
  throw new Error("context and windows must be positive integers");
}

const modelBytes = await readFile(modelPath);
const tokenBytes = await readFile(tokenPath);

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const U64_MASK = 0xffffffffffffffffn;

function fnv1a(bytes) {
  let hash = FNV_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = (hash * FNV_PRIME) & U64_MASK;
  }
  return hash;
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function readModel(bytes) {
  if (bytes.subarray(0, 8).toString("ascii") !== "NSRLPM1\n") {
    throw new Error("bad NSRLPM1 model magic");
  }
  let offset = 8;
  const version = bytes.readUInt32LE(offset); offset += 4;
  if (version !== 1) throw new Error(`unsupported model version ${version}`);
  const config = {
    vocabSize: bytes.readUInt32LE(offset),
    dModel: bytes.readUInt32LE(offset + 4),
    heads: bytes.readUInt32LE(offset + 8),
    layers: bytes.readUInt32LE(offset + 12),
    hiddenDim: bytes.readUInt32LE(offset + 16),
    contextTokens: bytes.readUInt32LE(offset + 20),
  };
  offset += 24;
  const tokenizerHash = bytes.readBigUInt64LE(offset); offset += 8;
  const initializationSeed = bytes.readBigUInt64LE(offset); offset += 8;
  const shifts = [...bytes.subarray(offset, offset + 6)];
  offset += 6;

  const readI16 = (count) => {
    const start = offset;
    offset += count * 2;
    return {
      length: count,
      get(index) { return bytes.readInt16LE(start + index * 2); },
    };
  };
  const readI8 = (count) => {
    const start = offset;
    offset += count;
    return {
      length: count,
      get(index) { return bytes.readInt8(start + index); },
    };
  };

  const matrix = config.dModel * config.dModel;
  const embeddings = readI16(config.vocabSize * config.dModel);
  const attentionRms = readI16(config.layers * config.dModel);
  readI16(config.layers * config.dModel); // MLP RMS
  readI16(config.dModel); // final RMS
  const q = readI8(config.layers * matrix);
  const k = readI8(config.layers * matrix);

  if (offset > bytes.length - 8) throw new Error("truncated NSRLPM1 model tensors");
  const storedChecksum = bytes.readBigUInt64LE(bytes.length - 8);
  const computedChecksum = fnv1a(bytes.subarray(0, bytes.length - 8));
  if (storedChecksum !== computedChecksum) throw new Error("NSRLPM1 checksum mismatch");

  return {
    config, shifts, embeddings, attentionRms, q, k, tokenizerHash, initializationSeed,
    modelHash: computedChecksum,
  };
}

function readTokens(bytes) {
  if (bytes.subarray(0, 8).toString("ascii") !== "NSRLTOK1") {
    throw new Error("bad NSRLTOK1 token-stream magic");
  }
  const tokenizerHash = bytes.readBigUInt64LE(8);
  const count = Number(bytes.readBigUInt64LE(16));
  if (bytes.length !== 24 + count * 4) throw new Error("wrong NSRLTOK1 length");
  return {
    tokenizerHash,
    tokenStreamHash: fnv1a(bytes.subarray(24)),
    tokens: Array.from({ length: count }, (_, index) => bytes.readUInt32LE(24 + index * 4)),
  };
}

function windows(tokens, context, maximum) {
  const result = [];
  let document = [];
  let active = false;
  for (const token of tokens) {
    if (token === 256) {
      document = [];
      active = true;
    } else if (token === 257) {
      if (active && document.length > context) {
        for (let start = 0; start < document.length - context; start += 1) {
          result.push(document.slice(start, start + context));
          if (result.length >= maximum) return result;
        }
      }
      document = [];
      active = false;
    } else if (active) {
      document.push(token);
    }
  }
  return result;
}

function rhu(value, shift) {
  if (shift === 0) return value;
  return Math.floor((value + 2 ** (shift - 1)) / 2 ** shift);
}

function projectFirstLayer(model, token, weights) {
  const { dModel } = model.config;
  const embeddingStart = token * dModel;
  let squareSum = 0;
  for (let dim = 0; dim < dModel; dim += 1) {
    const value = model.embeddings.get(embeddingStart + dim);
    squareSum += value * value;
  }
  const rms = Math.sqrt(squareSum / dModel + 1);
  const normalized = Array.from({ length: dModel }, (_, dim) => {
    const value = model.embeddings.get(embeddingStart + dim);
    const gamma = model.attentionRms.get(dim);
    return Math.max(-32768, Math.min(32767, Math.round(value * gamma / rms)));
  });
  const output = new Int16Array(dModel);
  for (let out = 0; out < dModel; out += 1) {
    let accumulator = 0;
    const row = out * dModel;
    for (let input = 0; input < dModel; input += 1) {
      accumulator += weights.get(row + input) * normalized[input];
    }
    output[out] = Math.max(-32768, Math.min(32767, rhu(accumulator, model.shifts[0])));
  }
  return output;
}

function summarize(values) {
  const ordered = [...values].sort((a, b) => a - b);
  const mean = ordered.reduce((sum, value) => sum + value, 0) / ordered.length;
  const quantile = (p) => ordered[Math.min(ordered.length - 1, Math.floor(p * ordered.length))];
  return {
    mean,
    p10: quantile(0.10),
    median: quantile(0.50),
    p90: quantile(0.90),
    min: ordered[0],
    max: ordered.at(-1),
  };
}

function kernel(query, key, offset, headDim, kind) {
  let sum = 0;
  for (let dim = 0; dim < headDim; dim += 1) {
    const q = query[offset + dim];
    const k = key[offset + dim];
    if (kind === "offset32769") {
      sum += (q + 32769) * (k + 32769);
    } else if (kind === "relu1") {
      sum += (Math.max(q, 0) + 1) * (Math.max(k, 0) + 1);
    } else if (kind === "signed_split") {
      sum += (Math.max(q, 0) + 1) * (Math.max(k, 0) + 1);
      sum += (Math.max(-q, 0) + 1) * (Math.max(-k, 0) + 1);
    } else {
      throw new Error(`unknown kernel ${kind}`);
    }
  }
  return sum;
}

function distributionMetrics(raw) {
  const total = raw.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return null;
  const probabilities = raw.map((value) => value / total);
  const mean = 1 / probabilities.length;
  const variance = probabilities.reduce((sum, value) => sum + (value - mean) ** 2, 0)
    / probabilities.length;
  const entropy = -probabilities.reduce(
    (sum, value) => sum + (value > 0 ? value * Math.log(value) : 0),
    0,
  );
  return {
    coefficientOfVariation: Math.sqrt(variance) / mean,
    effectiveTokens: Math.exp(entropy),
    maximumShare: Math.max(...probabilities),
    oldestToNewestRatio: probabilities[0] / probabilities.at(-1),
  };
}

const model = readModel(modelBytes);
const tokenArtifact = readTokens(tokenBytes);
if (model.tokenizerHash !== tokenArtifact.tokenizerHash) {
  throw new Error(
    `model/token tokenizer hash mismatch: ${hex64(model.tokenizerHash)} != ${hex64(tokenArtifact.tokenizerHash)}`,
  );
}
if (tokenArtifact.tokens.some((token) => token >= model.config.vocabSize)) {
  throw new Error("token stream contains token outside model vocabulary");
}
if (contextTokens > model.config.contextTokens) throw new Error("context exceeds model contract");
const probeWindows = windows(tokenArtifact.tokens, contextTokens, maximumWindows);
if (probeWindows.length === 0) throw new Error("no complete document windows");

const kinds = ["offset32769", "relu1", "signed_split"];
const decays = [
  { name: "none", gamma: 1 },
  { name: "63_over_64", gamma: 63 / 64 },
  { name: "31_over_32", gamma: 31 / 32 },
];
const metrics = new Map();
for (const kind of kinds) {
  for (const decay of decays) {
    metrics.set(`${kind}:${decay.name}`, {
      coefficientOfVariation: [], effectiveTokens: [], maximumShare: [], oldestToNewestRatio: [],
    });
  }
}

const headDim = model.config.dModel / model.config.heads;
for (const context of probeWindows) {
  const qRows = context.map((token) => projectFirstLayer(model, token, model.q));
  const kRows = context.map((token) => projectFirstLayer(model, token, model.k));
  const query = qRows.at(-1);
  for (let head = 0; head < model.config.heads; head += 1) {
    const offset = head * headDim;
    for (const kind of kinds) {
      const base = kRows.map((key) => kernel(query, key, offset, headDim, kind));
      for (const decay of decays) {
        const decayed = base.map(
          (value, position) => value * decay.gamma ** (context.length - 1 - position),
        );
        const row = distributionMetrics(decayed);
        const target = metrics.get(`${kind}:${decay.name}`);
        for (const [name, value] of Object.entries(row)) target[name].push(value);
      }
    }
  }
}

const result = {
  schema: "nsrl.production_attention_selectivity_probe.v2",
  knownLimitations: [
    "diagnostic_float_reconstruction_of_first_layer_rmsnorm",
    "not_exact_integer_rmsnorm_or_decay_execution",
    "not_a_training_or_quality_result",
    "alternative_feature_maps_are_offline_counterfactuals",
  ],
  modelPath,
  tokenPath,
  bindings: {
    modelHash: hex64(model.modelHash),
    tokenizerHash: hex64(model.tokenizerHash),
    tokenStreamHash: hex64(tokenArtifact.tokenStreamHash),
    initializationSeed: hex64(model.initializationSeed),
  },
  profile: model.config,
  contextTokens,
  windows: probeWindows.length,
  headRows: probeWindows.length * model.config.heads,
  featureMaps: Object.fromEntries([...metrics].map(([name, row]) => [
    name,
    Object.fromEntries(Object.entries(row).map(([metric, values]) => [metric, summarize(values)])),
  ])),
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
