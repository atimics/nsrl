#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  sourceSamples: "",
  textIndex: "data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv",
  outDir: "data/processed/key-solomon-goetia-latent-v1/planner-eval",
  runName: "",
  limit: 72,
  sourceModel: "",
  latentModels: [],
};

const schema = "nsrl.solomon_latent_planner_eval.v1";
const signatureGrid = 16;
const signatureBins = signatureGrid * signatureGrid;
const contentWindow = 16;
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;
const stopwords = new Set([
  "a", "about", "after", "again", "all", "also", "an", "and", "any", "are",
  "as", "at", "be", "before", "both", "but", "by", "can", "etc", "for",
  "from", "great", "have", "he", "her", "him", "his", "in", "is", "it",
  "man", "many", "men", "must", "of", "or", "order", "seal", "shall",
  "she", "spirit", "spirits", "the", "this", "thou", "to", "unto", "upon",
  "which", "who", "will", "with",
]);

function usage() {
  console.log(
    [
      "Usage: run-solomon-latent-planner-eval.mjs --source-samples PATH --latent-model LABEL=PATH [options]",
      "",
      "Scores learned NSRLLAT1 prompt-to-signature plans on a held-out generative",
      "eval prompt list without running the bitmap sampler.",
      "",
      "Options:",
      "  --source-samples PATH",
      "  --source-model LABEL       optional filter when samples.tsv has multiple models",
      "  --text-index PATH",
      "  --latent-model LABEL=PATH  repeatable",
      "  --out-dir PATH",
      "  --run-name NAME",
      "  --limit N",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, latentModels: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--source-samples") {
      config.sourceSamples = requireValue(argv, ++index, arg);
    } else if (arg === "--source-model") {
      config.sourceModel = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--latent-model") {
      config.latentModels.push(parseLatentSpec(requireValue(argv, ++index, arg)));
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--run-name") {
      config.runName = requireValue(argv, ++index, arg);
    } else if (arg === "--limit") {
      config.limit = parsePositive(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.sourceSamples) {
    throw new Error("--source-samples is required");
  }
  if (config.latentModels.length === 0) {
    throw new Error("--latent-model is required");
  }
  if (!config.runName) {
    config.runName = `run-${timestamp()}`;
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePositive(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parseLatentSpec(spec) {
  const equals = spec.indexOf("=");
  if (equals <= 0 || equals === spec.length - 1) {
    throw new Error(`invalid --latent-model ${spec}; expected LABEL=PATH`);
  }
  return {
    label: sanitizeSlug(spec.slice(0, equals)),
    path: spec.slice(equals + 1),
  };
}

function requireFile(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  requireFile(config.sourceSamples, "--source-samples");
  requireFile(config.textIndex, "--text-index");
  for (const latent of config.latentModels) {
    requireFile(latent.path, `--latent-model ${latent.label}`);
  }

  const runDir = path.join(config.outDir, config.runName);
  fs.rmSync(runDir, { recursive: true, force: true });
  fs.mkdirSync(runDir, { recursive: true });

  const textRows = readTextIndex(config.textIndex);
  const prompts = selectSourcePrompts(readTsv(config.sourceSamples), config);
  if (prompts.length === 0) {
    throw new Error(`no source rows selected from ${config.sourceSamples}`);
  }

  const sampleRows = [];
  const summaryRows = [];
  for (const latent of config.latentModels) {
    const model = readLatentModel(latent.path);
    const modelHash = fnv64FileHex(latent.path);
    let top1 = 0;
    let top5 = 0;
    let rankTotal = 0;
    let distanceTotal = 0;
    for (const prompt of prompts) {
      const target = textRows.byNumber.get(Number(prompt.spirit_id));
      if (!target) {
        throw new Error(`missing text-index row for spirit_id ${prompt.spirit_id}`);
      }
      const planned = latentDecodedMetrics(model, prompt.prompt, target.signature, textRows.targets);
      if (planned.rank <= 1) {
        top1 += 1;
      }
      if (planned.rank <= 5) {
        top5 += 1;
      }
      rankTotal += planned.rank;
      distanceTotal += planned.distance;
      sampleRows.push({
        model: latent.label,
        prompt_hash: prompt.prompt_hash,
        spirit_id: prompt.spirit_id,
        target_name: target.name,
        rank: planned.rank,
        target_distance: planned.distance,
        prompt: prompt.prompt,
      });
    }
    summaryRows.push({
      model: latent.label,
      latent_model: latent.path,
      latent_model_hash: modelHash,
      source_samples: config.sourceSamples,
      source_model: config.sourceModel,
      prompts: prompts.length,
      top1,
      top5,
      top1_per_mille: Math.floor((top1 * 1000) / prompts.length),
      top5_per_mille: Math.floor((top5 * 1000) / prompts.length),
      mean_rank_q8: Math.floor((rankTotal * 256) / prompts.length),
      mean_target_distance_q8: Math.floor((distanceTotal * 256) / prompts.length),
    });
  }

  writeTsv(path.join(runDir, "samples.tsv"), sampleRows, [
    "model",
    "prompt_hash",
    "spirit_id",
    "target_name",
    "rank",
    "target_distance",
    "prompt",
  ]);
  writeTsv(path.join(runDir, "summary.tsv"), summaryRows, [
    "model",
    "latent_model",
    "latent_model_hash",
    "source_samples",
    "source_model",
    "prompts",
    "top1",
    "top5",
    "top1_per_mille",
    "top5_per_mille",
    "mean_rank_q8",
    "mean_target_distance_q8",
  ]);
  fs.writeFileSync(
    path.join(runDir, "config.json"),
    `${JSON.stringify(
      {
        schema,
        ...config,
        runDir,
        sourceSamplesHash: fnv64FileHex(config.sourceSamples),
        textIndexHash: fnv64FileHex(config.textIndex),
        selectedPromptHash: promptSelectionHash(prompts),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  console.log(JSON.stringify({ schema, run_dir: runDir, prompts: prompts.length, models: summaryRows.length }));
}

function readTsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  const lines = text.split(/\r?\n/);
  const header = lines.shift().split("\t");
  return lines.filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = { row_index: rowIndex + 2 };
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    return row;
  });
}

function selectSourcePrompts(rows, config) {
  const selected = [];
  const seen = new Set();
  for (const row of rows) {
    if (config.sourceModel && row.model !== config.sourceModel) {
      continue;
    }
    const spiritId = Number(row.spirit_id);
    if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72) {
      continue;
    }
    const promptHash = String(row.prompt_hash || "");
    const key = `${spiritId}:${promptHash || row.row_index}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    selected.push({
      prompt_hash: promptHash,
      spirit_id: spiritId,
      prompt: row.prompt || "",
    });
    if (selected.length >= config.limit) {
      break;
    }
  }
  return selected;
}

function readTextIndex(filePath) {
  const rows = readTsv(filePath);
  const byNumber = new Map();
  for (const row of rows) {
    const number = Number(row.number);
    const signature = parseSignature(row.signature_16x16, filePath);
    if (!Number.isInteger(number) || number < 1 || number > 72) {
      throw new Error(`${filePath}:${row.row_index} has invalid spirit number`);
    }
    byNumber.set(number, {
      number,
      name: row.primary_name || "",
      signature,
    });
  }
  return {
    byNumber,
    targets: [...byNumber.values()].sort((left, right) => left.number - right.number),
  };
}

function readLatentModel(filePath) {
  const bytes = fs.readFileSync(filePath);
  let offset = 0;
  function readBytes(count) {
    if (offset + count > bytes.length) {
      throw new Error(`${filePath} ended unexpectedly`);
    }
    const start = offset;
    offset += count;
    return bytes.subarray(start, offset);
  }
  function readU32() {
    const value = bytes.readUInt32LE(offset);
    offset += 4;
    return value;
  }
  function readI16() {
    const value = bytes.readInt16LE(offset);
    offset += 2;
    return value;
  }
  function readI8Vec(count) {
    const out = [];
    for (let index = 0; index < count; index += 1) {
      out.push(bytes.readInt8(offset));
      offset += 1;
    }
    return out;
  }
  function readI16Vec(count) {
    const out = [];
    for (let index = 0; index < count; index += 1) {
      out.push(readI16());
    }
    return out;
  }

  const magic = readBytes(8).toString("ascii");
  if (magic !== "NSRLLAT1") {
    throw new Error(`${filePath} is not an NSRLLAT1 model`);
  }
  const latentDim = readU32();
  const textFeatureCount = readU32();
  const modelSignatureBins = readU32();
  const textEncoderShift = readU32();
  readU32();
  const decoderShift = readU32();
  const modelSignatureGrid = readU32();
  if (
    latentDim <= 0 ||
    textFeatureCount <= 0 ||
    modelSignatureBins !== signatureBins ||
    modelSignatureGrid !== signatureGrid
  ) {
    throw new Error(`${filePath} has incompatible latent dimensions`);
  }
  const textWeights = readI8Vec(latentDim * textFeatureCount);
  const textBiases = readI16Vec(latentDim);
  readBytes(latentDim * signatureBins);
  readBytes(latentDim * 2);
  const decoderWeights = readI8Vec(signatureBins * latentDim);
  const decoderBiases = readI16Vec(signatureBins);
  if (offset !== bytes.length) {
    throw new Error(`${filePath} has ${bytes.length - offset} trailing bytes`);
  }
  return {
    latentDim,
    textFeatureCount,
    textEncoderShift,
    decoderShift,
    textWeights,
    textBiases,
    decoderWeights,
    decoderBiases,
  };
}

function latentDecodedMetrics(model, prompt, targetSignature, targetRows) {
  const features = textFeatures(prompt, model.textFeatureCount);
  const latent = encodeText(model, features);
  const signature = decodeSignature(model, latent);
  return {
    distance: signatureDistance(signature, targetSignature),
    rank: targetRank(signature, targetSignature, targetRows),
  };
}

function textFeatures(text, featureCount) {
  const features = new Array(featureCount).fill(0);
  const tokens = tokenizeText(text);
  if (tokens.length > 0 && tokens.length <= 4) {
    addTextFeature(features, "whole", tokens.join(" "), 0, 320);
  }
  tokens.forEach((token, position) => {
    addTextFeature(features, "tok", token, position, 72);
    if (tokens[position + 1]) {
      addTextFeature(features, "bi", `${token} ${tokens[position + 1]}`, position, 96);
    }
    if (tokens[position + 1] && tokens[position + 2]) {
      addTextFeature(features, "tri", `${token} ${tokens[position + 1]} ${tokens[position + 2]}`, position, 112);
    }
  });
  const content = contentTokens(tokens);
  if (content.length > 0 && content.length <= 5) {
    addTextFeature(features, "cwhole", content.join(" "), 0, 336);
    addTextFeature(features, "cset", sortedKey(content), 0, 336);
  }
  for (let index = 0; index < content.length; index += 1) {
    addTextFeature(features, "ctok", content[index], index, 128);
    if (content[index + 1]) {
      addTextFeature(features, "cbi", `${content[index]} ${content[index + 1]}`, index, 160);
    }
    if (content[index + 1] && content[index + 2]) {
      addTextFeature(
        features,
        "ctri",
        `${content[index]} ${content[index + 1]} ${content[index + 2]}`,
        index,
        176,
      );
    }
    const windowEnd = Math.min(content.length, index + contentWindow);
    for (let right = index + 1; right < windowEnd; right += 1) {
      addTextFeature(features, "skip2", `${content[index]} ${content[right]}`, index, 176);
      addTextFeature(features, "pair", sortedKey([content[index], content[right]]), index, 192);
      for (let third = right + 1; third < windowEnd; third += 1) {
        addTextFeature(features, "triple", sortedKey([content[index], content[right], content[third]]), index, 192);
      }
    }
  }
  return features;
}

function addTextFeature(features, namespace, text, position, base) {
  if (text.length < 2 || features.length === 0) return;
  const hash = hashParts([namespace, text]);
  const bin = hash % features.length;
  const value = Math.min(384, base + Math.min(28, text.length) * 6 + (position % 7) * 5);
  const signed = (hash >>> 31) === 0 ? value : -value;
  features[bin] = clamp(features[bin] + signed, -511, 511);
}

function encodeText(model, features) {
  const out = new Array(model.latentDim).fill(0);
  for (let dim = 0; dim < model.latentDim; dim += 1) {
    let acc = 0;
    const base = dim * model.textFeatureCount;
    for (let feature = 0; feature < model.textFeatureCount; feature += 1) {
      acc += model.textWeights[base + feature] * features[feature];
    }
    const value = signedRoundShift(acc, model.textEncoderShift) + model.textBiases[dim];
    out[dim] = clamp(value, -511, 511);
  }
  return out;
}

function decodeSignature(model, latent) {
  const out = new Array(signatureBins).fill(0);
  for (let bin = 0; bin < signatureBins; bin += 1) {
    let acc = model.decoderBiases[bin] * (2 ** model.decoderShift);
    const base = bin * model.latentDim;
    for (let dim = 0; dim < model.latentDim; dim += 1) {
      acc += model.decoderWeights[base + dim] * latent[dim];
    }
    out[bin] = clamp(signedRoundShift(acc, model.decoderShift), 0, 255);
  }
  return out;
}

function signedRoundShift(value, shift) {
  if (shift === 0) {
    return value;
  }
  const rounding = 2 ** (shift - 1);
  if (value >= 0) {
    return Math.trunc((value + rounding) / (2 ** shift));
  }
  return -Math.trunc((-value + rounding) / (2 ** shift));
}

function tokenizeText(text) {
  const tokens = [];
  let current = "";
  for (const ch of String(text)) {
    if (/^[A-Za-z0-9]$/.test(ch)) {
      current += ch.toLowerCase();
    } else if (current) {
      const normalized = normalizeToken(current);
      if (normalized.length >= 2) tokens.push(normalized);
      current = "";
    }
  }
  if (current) {
    const normalized = normalizeToken(current);
    if (normalized.length >= 2) tokens.push(normalized);
  }
  return tokens;
}

function normalizeToken(token) {
  if (["teach", "teacher", "teaches", "teacheth", "teaching"].includes(token)) return "teach";
  if (["know", "knows", "known", "knowing", "knoweth", "knowledge"].includes(token)) return "know";
  if (["make", "makes", "maketh", "making"].includes(token)) return "make";
  if (["discover", "discovers", "discovereth", "discovering"].includes(token)) return "discover";
  if (["produce", "produces", "produceth", "producing"].includes(token)) return "produce";
  if (["answer", "answers", "answereth", "answering"].includes(token)) return "answer";
  if (["virtue", "virtues"].includes(token)) return "virtue";
  if (["water", "waters"].includes(token)) return "water";
  if (["rush", "rushing", "rushings"].includes(token)) return "rush";
  if (["herb", "herbs"].includes(token)) return "herb";
  if (["stone", "stones"].includes(token)) return "stone";
  if (["science", "sciences"].includes(token)) return "science";
  if (token.length > 5 && token.endsWith("eth")) return token.slice(0, -3);
  if (token.length > 5 && token.endsWith("ing")) return token.slice(0, -3);
  if (token.length > 4 && token.endsWith("es")) return token.slice(0, -2);
  if (token.length > 3 && token.endsWith("s")) return token.slice(0, -1);
  return token;
}

function contentTokens(tokens) {
  return tokens.filter((token) => token.length >= 3 && !stopwords.has(token));
}

function sortedKey(tokens) {
  return [...tokens].sort().join("\x00");
}

function targetRank(signature, targetSignature, targetRows) {
  const positiveDistance = signatureDistance(signature, targetSignature);
  let rank = 1;
  for (const row of targetRows) {
    const rowSignature = row.signature;
    if (rowSignature === targetSignature) {
      continue;
    }
    const distance = signatureDistance(signature, rowSignature);
    if (distance < positiveDistance || (distance === positiveDistance && rowSignature !== targetSignature)) {
      rank += 1;
    }
  }
  return rank;
}

function signatureDistance(left, right) {
  let best = Number.MAX_SAFE_INTEGER;
  for (let variant = 0; variant < 8; variant += 1) {
    let distance = 0;
    for (let y = 0; y < signatureGrid; y += 1) {
      for (let x = 0; x < signatureGrid; x += 1) {
        const [sx, sy] = transformCoords(signatureGrid, x, y, variant);
        distance += Math.abs(left[y * signatureGrid + x] - right[sy * signatureGrid + sx]);
      }
    }
    best = Math.min(best, distance);
  }
  return best;
}

function transformCoords(size, x, y, variant) {
  const last = size - 1;
  switch (variant % 8) {
    case 0:
      return [x, y];
    case 1:
      return [last - x, y];
    case 2:
      return [x, last - y];
    case 3:
      return [last - x, last - y];
    case 4:
      return [y, x];
    case 5:
      return [last - y, x];
    case 6:
      return [y, last - x];
    default:
      return [last - y, last - x];
  }
}

function parseSignature(text, filePath) {
  const parts = String(text).split(",").map((value) => Number(value));
  if (parts.length !== signatureBins || parts.some((value) => !Number.isFinite(value))) {
    throw new Error(`${filePath} has invalid ${signatureBins}-bin signature`);
  }
  return parts;
}

function promptSelectionHash(prompts) {
  const lines = prompts
    .map((prompt) => [
      prompt.prompt_hash || "",
      prompt.spirit_id || "",
      prompt.prompt || "",
    ].join("\t"))
    .sort()
    .join("\n");
  return fnv64TextHex(`${lines}\n`);
}

function writeTsv(filePath, rows, header) {
  const lines = [
    header.join("\t"),
    ...rows.map((row) => header.map((key) => tsvEscape(row[key] ?? "")).join("\t")),
  ];
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function tsvEscape(value) {
  return String(value).replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function sanitizeSlug(value) {
  return String(value).replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function hashParts(parts) {
  let hash = 2166136261 >>> 0;
  for (const part of parts) {
    for (const byte of Buffer.from(String(part))) {
      hash ^= byte;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    hash ^= 255;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return (hash | 1) >>> 0;
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function fnv64TextHex(value) {
  return fnv64BytesHex(Buffer.from(String(value), "utf8"));
}

function fnv64BytesHex(bytes) {
  let hash = FNV64_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

main();
