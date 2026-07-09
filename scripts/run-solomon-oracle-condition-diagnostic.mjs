#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  sourceSamples: "",
  textIndex:
    "data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv",
  samplerModel: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch",
  retrievalHead: "",
  outDir: "data/processed/key-solomon-goetia-latent-v1/oracle-condition-diagnostic",
  runName: "",
  limit: 24,
  samples: 1,
  candidateMultiplier: 1,
  diversityWeight: 0,
  textWeight: 96,
  passes: 1,
  seed: "solomon-oracle-condition-diagnostic-v1",
  force: false,
};

const schema = "nsrl.solomon_oracle_condition_diagnostic.v1";
const grid = 16;
const bins = grid * grid;
const imageSize = 128;
const imageBytes = imageSize * imageSize;
const inkThreshold = 64;
const imageBase = 144;
const imageBins = 16;
const imageChannelInk = 11;
const imageChannelEdge = 12;
const imageChannelComponent = 13;
const imageChannelRadial = 14;
const imageChannelDirection = 15;
const channelNames = new Map([
  [imageChannelInk, "ink"],
  [imageChannelEdge, "edge"],
  [imageChannelComponent, "component"],
  [imageChannelRadial, "radial"],
  [imageChannelDirection, "direction"],
]);

function usage() {
  console.log(
    [
      "Usage: run-solomon-oracle-condition-diagnostic.mjs --source-samples PATH [options]",
      "",
      "Runs a diagnostic-only upper bound: the sampler receives the true 16x16",
      "target signature as an attention plan. Do not use this for headline evals.",
      "",
      "Options:",
      "  --source-samples PATH",
      "  --text-index PATH",
      "  --sampler-model PATH",
      "  --retrieval-head PATH",
      "  --out-dir PATH",
      "  --run-name NAME",
      "  --limit N",
      "  --samples N",
      "  --candidate-multiplier N",
      "  --diversity-weight N",
      "  --text-weight N",
      "  --passes N",
      "  --seed TEXT",
      "  --force",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--source-samples") {
      config.sourceSamples = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--sampler-model") {
      config.samplerModel = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHead = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--run-name") {
      config.runName = requireValue(argv, ++index, arg);
    } else if (arg === "--limit") {
      config.limit = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--samples") {
      config.samples = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--candidate-multiplier") {
      config.candidateMultiplier = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--diversity-weight") {
      config.diversityWeight = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--text-weight") {
      config.textWeight = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--passes") {
      config.passes = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--seed") {
      config.seed = requireValue(argv, ++index, arg);
    } else if (arg === "--force") {
      config.force = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.sourceSamples) {
    throw new Error("--source-samples is required");
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

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\..+$/, "Z");
}

function requireFile(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
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
      spirit_id: number,
      primary_name: row.primary_name || "",
      aliases: String(row.aliases || "").split("|").filter(Boolean),
      signature,
    });
  }
  return {
    byNumber,
    targets: [...byNumber.values()].sort((left, right) => left.spirit_id - right.spirit_id),
  };
}

function parseSignature(value, filePath) {
  const signature = String(value || "").split(",").map((part) => Number(part));
  if (signature.length !== bins || signature.some((part) => !Number.isFinite(part))) {
    throw new Error(`${filePath} has invalid ${bins}-bin signature`);
  }
  return signature;
}

function selectSourceRows(rows, limit) {
  const seen = new Set();
  const selected = [];
  for (const row of rows) {
    const spiritId = Number(row.spirit_id);
    const key = `${spiritId}:${row.prompt_hash || row.row_index}`;
    if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72 || seen.has(key)) {
      continue;
    }
    seen.add(key);
    selected.push(row);
    if (selected.length >= limit) {
      break;
    }
  }
  return selected;
}

function writePlan(filePath, signature) {
  const bytes = Buffer.alloc(bins);
  for (let index = 0; index < bins; index += 1) {
    bytes[index] = Math.max(0, Math.min(255, Math.round(signature[index] || 0)));
  }
  fs.writeFileSync(filePath, bytes);
}

function runCommand(label, command, args, logPath) {
  fs.mkdirSync(path.dirname(logPath), { recursive: true });
  const result = spawnSync(command, args, { encoding: "utf8" });
  const log = [
    `$ ${[command, ...args].join(" ")}`,
    "",
    result.stdout || "",
    result.stderr || "",
  ].join("\n");
  fs.writeFileSync(logPath, log, "utf8");
  if (result.status !== 0) {
    throw new Error(`${label} failed; see ${logPath}`);
  }
}

function sampleSignature(bytes) {
  const out = new Array(bins).fill(0);
  for (let gy = 0; gy < grid; gy += 1) {
    const y0 = Math.floor((gy * imageSize) / grid);
    const y1 = Math.floor(((gy + 1) * imageSize) / grid);
    for (let gx = 0; gx < grid; gx += 1) {
      const x0 = Math.floor((gx * imageSize) / grid);
      const x1 = Math.floor(((gx + 1) * imageSize) / grid);
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1; y += 1) {
        const row = y * imageSize;
        for (let x = x0; x < x1; x += 1) {
          sum += bytes[row + x];
          count += 1;
        }
      }
      out[gy * grid + gx] = count > 0 && Math.floor(sum / count) >= inkThreshold ? 255 : 0;
    }
  }
  return out;
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < bins; index += 1) {
    distance += Math.abs((left[index] || 0) - (right[index] || 0));
  }
  return distance;
}

function targetRank(signature, targetSignature, targets) {
  const positiveDistance = signatureDistance(signature, targetSignature);
  let rank = 1;
  let bestSpiritId = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const target of targets) {
    const distance = signatureDistance(signature, target.signature);
    if (
      distance < bestDistance ||
      (distance === bestDistance && target.spirit_id < bestSpiritId)
    ) {
      bestDistance = distance;
      bestSpiritId = target.spirit_id;
    }
    if (distance < positiveDistance) {
      rank += 1;
    }
  }
  return { rank, distance: positiveDistance, bestSpiritId, bestDistance };
}

function readRetrievalHead(filePath) {
  if (!filePath) {
    return null;
  }
  const model = JSON.parse(fs.readFileSync(filePath, "utf8"));
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(model.schema)}`);
  }
  return {
    ...model,
    image_head: {
      biases: model.image_head?.biases || [],
      weights: (model.image_head?.weights || []).map((entries) => new Map(entries)),
    },
  };
}

function symbolicImageOptions() {
  return {
    grid,
    imageBase,
    imageBins,
    channelTokens: {
      ink: imageChannelInk,
      edge: imageChannelEdge,
      component: imageChannelComponent,
      radial: imageChannelRadial,
      direction: imageChannelDirection,
    },
  };
}

function rankRetrievalImage(model, signature, count = 5) {
  const features = imageFeatures(
    solomonImage.symbolicImageTokens(signature, symbolicImageOptions()),
    model.feature_count,
  );
  const ranked = model.labels.map((label) => ({
    spirit_id: Number(label.spirit_id),
    primary_name: label.primary_name || "",
    score: scoreLabel(model.image_head, Number(label.label), features),
  }));
  ranked.sort((left, right) => right.score - left.score || left.spirit_id - right.spirit_id);
  return ranked.slice(0, count);
}

function scoreLabel(head, label, features) {
  let score = head.biases[label] || 0;
  const weights = head.weights[label] || new Map();
  for (const [feature, value] of features) {
    score += (weights.get(feature) || 0) * value;
  }
  return score;
}

function imageFeatures(image, featureCount) {
  const out = new Map();
  let channel = "ink";
  let position = 0;
  for (const token of image) {
    if (channelNames.has(token)) {
      channel = channelNames.get(token);
      position = 0;
      addHashedFeature(out, featureCount, "channel", channel, 32);
      continue;
    }
    const bin = token >= imageBase && token < imageBase + imageBins ? token - imageBase : token;
    addHashedFeature(out, featureCount, "ipos", `${channel}:${position}:${bin}`, 64);
    addHashedFeature(out, featureCount, "itok", `${channel}:${bin}`, 8);
    if (position % grid === 0) {
      addHashedFeature(
        out,
        featureCount,
        "irow",
        `${channel}:${Math.floor(position / grid)}:${bin}`,
        6,
      );
    }
    position += 1;
  }
  return [...out.entries()];
}

function addHashedFeature(out, featureCount, namespace, value, amount) {
  const hash = fnv32(`${namespace}\xff${value}`);
  const index = hash % featureCount;
  const sign = hash & 0x80000000 ? -1 : 1;
  out.set(index, Math.max(-127, Math.min(127, (out.get(index) || 0) + sign * amount)));
}

function fnv32(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function meanQ8(total, count) {
  return count > 0 ? Math.floor((total * 256) / count) : 0;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  requireFile(config.sourceSamples, "--source-samples");
  requireFile(config.textIndex, "--text-index");
  requireFile(config.samplerModel, "--sampler-model");
  if (config.retrievalHead) {
    requireFile(config.retrievalHead, "--retrieval-head");
  }

  const runDir = path.join(config.outDir, config.runName);
  if (fs.existsSync(runDir)) {
    if (!config.force) {
      throw new Error(`${runDir} already exists; pass --force or choose another --run-name`);
    }
    fs.rmSync(runDir, { recursive: true, force: true });
  }
  fs.mkdirSync(runDir, { recursive: true });

  runCommand(
    "build sampler",
    "cargo",
    ["build", "--release", "-q", "-p", "nsrl-train", "--bin", "nsrl-bitmap-sample"],
    path.join(runDir, "build.log"),
  );

  const sourceRows = selectSourceRows(readTsv(config.sourceSamples), config.limit);
  if (sourceRows.length === 0) {
    throw new Error(`no usable source rows found in ${config.sourceSamples}`);
  }
  const textIndex = readTextIndex(config.textIndex);
  const retrievalHead = readRetrievalHead(config.retrievalHead);
  const rows = [];
  let top1 = 0;
  let top5 = 0;
  let retrievalTop1 = 0;
  let retrievalTop5 = 0;
  let rankTotal = 0;
  let retrievalRankTotal = 0;
  let targetDistanceTotal = 0;

  for (const row of sourceRows) {
    const spiritId = Number(row.spirit_id);
    const target = textIndex.byNumber.get(spiritId);
    if (!target) {
      throw new Error(`missing target spirit_id ${spiritId}`);
    }
    const slug = `${String(rows.length + 1).padStart(3, "0")}-${row.prompt_hash || spiritId}`;
    const sampleDir = path.join(runDir, slug);
    fs.mkdirSync(sampleDir, { recursive: true });
    const planPath = path.join(sampleDir, "oracle-attention-plan.ink16.u8");
    writePlan(planPath, target.signature);
    runCommand(
      `sample ${slug}`,
      "target/release/nsrl-bitmap-sample",
      [
        "--model",
        config.samplerModel,
        "--attention-plan",
        planPath,
        "--prompt",
        row.prompt || target.primary_name,
        "--out-dir",
        sampleDir,
        "--samples",
        String(config.samples),
        "--candidate-multiplier",
        String(config.candidateMultiplier),
        "--diversity-weight",
        String(config.diversityWeight),
        "--text-weight",
        String(config.textWeight),
        "--passes",
        String(config.passes),
        "--preview-columns",
        String(config.samples),
        "--seed",
        `${config.seed}-${row.prompt_hash || slug}`,
        "--init",
        "noise",
      ],
      path.join(sampleDir, "sample.log"),
    );
    const rawSamplesPath = path.join(sampleDir, `samples.ink${imageSize}.u8`);
    const sampleBytes = fs.readFileSync(rawSamplesPath);
    if (sampleBytes.length < imageBytes) {
      throw new Error(
        `${rawSamplesPath} has ${sampleBytes.length} bytes, expected at least ${imageBytes}`,
      );
    }
    const signature = sampleSignature(sampleBytes.subarray(0, imageBytes));
    const ranked = targetRank(signature, target.signature, textIndex.targets);
    const retrievalRanked = retrievalHead
      ? rankRetrievalImage(retrievalHead, signature, retrievalHead.labels.length)
      : [];
    const retrievalRank = retrievalRanked.length > 0
      ? retrievalRanked.findIndex((entry) => entry.spirit_id === spiritId) + 1
      : 0;
    if (ranked.rank <= 1) {
      top1 += 1;
    }
    if (ranked.rank <= 5) {
      top5 += 1;
    }
    if (retrievalRank > 0) {
      if (retrievalRank <= 1) {
        retrievalTop1 += 1;
      }
      if (retrievalRank <= 5) {
        retrievalTop5 += 1;
      }
      retrievalRankTotal += retrievalRank;
    }
    rankTotal += ranked.rank;
    targetDistanceTotal += ranked.distance;
    rows.push({
      prompt_hash: row.prompt_hash || "",
      spirit_id: spiritId,
      target_name: target.primary_name,
      signature_rank: ranked.rank,
      signature_target_distance: ranked.distance,
      signature_top1_spirit_id: ranked.bestSpiritId,
      signature_top1_distance: ranked.bestDistance,
      retrieval_rank: retrievalRank || "",
      retrieval_top1_spirit_id: retrievalRanked[0]?.spirit_id || "",
      retrieval_top1_name: retrievalRanked[0]?.primary_name || "",
      prompt: row.prompt || target.primary_name,
      out_dir: sampleDir,
      attention_plan: planPath,
    });
  }

  const summary = {
    schema,
    diagnostic_only: true,
    diagnostic_warning:
      "oracle_conditioned attention plans use the known target signature; exclude from headline/product gates",
    run_dir: runDir,
    source_samples: config.sourceSamples,
    sampler_model: config.samplerModel,
    retrieval_head: config.retrievalHead || "",
    prompts: rows.length,
    samples: config.samples,
    candidate_multiplier: config.candidateMultiplier,
    passes: config.passes,
    text_weight: config.textWeight,
    signature_top1: top1,
    signature_top5: top5,
    signature_top1_per_mille: Math.floor((top1 * 1000) / rows.length),
    signature_top5_per_mille: Math.floor((top5 * 1000) / rows.length),
    mean_signature_rank_q8: meanQ8(rankTotal, rows.length),
    mean_signature_target_distance_q8: meanQ8(targetDistanceTotal, rows.length),
    retrieval_top1: retrievalHead ? retrievalTop1 : null,
    retrieval_top5: retrievalHead ? retrievalTop5 : null,
    retrieval_top1_per_mille: retrievalHead ? Math.floor((retrievalTop1 * 1000) / rows.length) : null,
    retrieval_top5_per_mille: retrievalHead ? Math.floor((retrievalTop5 * 1000) / rows.length) : null,
    mean_retrieval_rank_q8: retrievalHead ? meanQ8(retrievalRankTotal, rows.length) : null,
  };

  writeTsv(path.join(runDir, "samples.tsv"), rows, [
    "prompt_hash",
    "spirit_id",
    "target_name",
    "signature_rank",
    "signature_target_distance",
    "signature_top1_spirit_id",
    "signature_top1_distance",
    "retrieval_rank",
    "retrieval_top1_spirit_id",
    "retrieval_top1_name",
    "prompt",
    "out_dir",
    "attention_plan",
  ]);
  fs.writeFileSync(path.join(runDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  fs.writeFileSync(path.join(runDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(summary));
}

main();
