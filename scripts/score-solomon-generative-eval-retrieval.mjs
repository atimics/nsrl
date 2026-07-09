#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  inputPath: "",
  retrievalHeadPath: "",
  imageSize: 128,
  requireAll: true,
};

const GRID = 16;
const BINS = GRID * GRID;
const IMAGE_BASE = 144;
const IMAGE_BINS = 16;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
const CHANNEL_NAMES = new Map([
  [IMAGE_CHANNEL_INK, "ink"],
  [IMAGE_CHANNEL_EDGE, "edge"],
  [IMAGE_CHANNEL_COMPONENT, "component"],
  [IMAGE_CHANNEL_RADIAL, "radial"],
  [IMAGE_CHANNEL_DIRECTION, "direction"],
]);
const ALLOWED_TARGET_KEYS = new Set([
  "latent_target_source",
  "latent_target_number",
  "latent_target_name",
  "latent_target_score",
  "latent_target_latent_score",
  "latent_target_lexical_score",
  "latent_target_signature",
]);
const FREE_TEXT_VALUE_KEYS = new Set([
  "latent_prompt",
  "latent_target_name",
]);
const FORBIDDEN_KEY_PATTERNS = [
  /display[_-]?cleanup/i,
  /cleanup/i,
  /post[_-]?process/i,
  /postprocess/i,
  /oracle/i,
  /ground[_-]?truth/i,
  /guidance/i,
  /target[_-]?(pixel|pixels|bitmap|image|ink|seal|lookup|source|guidance)/i,
  /(pixel|pixels|bitmap|image|ink|seal)[_-]?target/i,
];
const BROAD_FORBIDDEN_VALUE =
  /target[-_\s]*(pixel|pixels|bitmap|image|ink|seal|signature)|ground[-_\s]*truth|oracle|retrieval[-_\s]*hybrid|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess|targetctx/i;
const SOURCE_FORBIDDEN_VALUE =
  /\btarget\b|target[-_\s]*(lookup|guidance|source)|retrieval[-_\s]*hybrid|ground[-_\s]*truth|oracle|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess/i;
const GENERATED_RETRIEVAL_FIELDS = [
  "generated_retrieval_rank",
  "generated_retrieval_margin",
  "generated_retrieval_top1_spirit_id",
  "generated_retrieval_top1_name",
  "generated_retrieval_identity",
];

function usage() {
  console.log(
    [
      "Usage: score-solomon-generative-eval-retrieval.mjs --generative-eval PATH --retrieval-head PATH",
      "",
      "Adds rendered-image retrieval identity metrics to an existing Solomon",
      "generative-eval run by reading samples.tsv, summary.tsv, and sample",
      "out_dir/samples.ink128.u8 files.",
      "",
      "Options:",
      "  --generative-eval PATH   run dir or summary.tsv",
      "  --run-dir PATH           alias for --generative-eval",
      "  --retrieval-head PATH",
      "  --image-size N",
      "  --allow-missing-samples",
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
    } else if (arg === "--generative-eval" || arg === "--run-dir") {
      config.inputPath = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = requireValue(argv, ++index, arg);
    } else if (arg === "--image-size") {
      config.imageSize = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--allow-missing-samples") {
      config.requireAll = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.inputPath) {
    throw new Error("--generative-eval is required");
  }
  if (!config.retrievalHeadPath) {
    throw new Error("--retrieval-head is required");
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

function resolveRunDir(inputPath) {
  if (fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory()) {
    return inputPath;
  }
  if (path.basename(inputPath) === "summary.tsv") {
    return path.dirname(inputPath);
  }
  return inputPath;
}

function readTsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return { header: [], rows: [] };
  }
  const lines = text.split(/\r?\n/);
  const header = lines[0].split("\t");
  const rows = lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = { __line: rowIndex + 2 };
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    return row;
  });
  return { header, rows };
}

function writeTsv(filePath, header, rows) {
  const lines = [
    header.join("\t"),
    ...rows.map((row) => header.map((key) => tsvEscape(row[key] ?? "")).join("\t")),
  ];
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function readJsonIfPresent(filePath) {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, row) {
  const dir = path.dirname(filePath);
  if (dir && dir !== ".") {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`, "utf8");
}

function tsvEscape(value) {
  return String(value).replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function ensureColumns(header, columns) {
  for (const column of columns) {
    if (!header.includes(column)) {
      header.push(column);
    }
  }
}

function readRetrievalHead(filePath) {
  const model = JSON.parse(fs.readFileSync(filePath, "utf8"));
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(model.schema)}`);
  }
  validateRetrievalHeadLabels(filePath, model.labels);
  return {
    ...model,
    image_head: {
      biases: model.image_head?.biases || [],
      weights: (model.image_head?.weights || []).map((entries) => new Map(entries)),
    },
  };
}

function validateRetrievalHeadLabels(filePath, labels) {
  if (!Array.isArray(labels) || labels.length !== 72) {
    throw new Error(`${filePath} retrieval head labels ${Array.isArray(labels) ? labels.length : 0} != 72`);
  }
  const ids = new Set();
  for (const label of labels) {
    const spiritId = Number(label?.spirit_id || 0);
    if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72) {
      throw new Error(`${filePath} retrieval head label has invalid spirit_id ${JSON.stringify(label?.spirit_id)}`);
    }
    ids.add(spiritId);
  }
  if (ids.size !== 72) {
    throw new Error(`${filePath} retrieval head labels must cover each spirit_id 1..72 exactly once`);
  }
}

function rawSamplesPath(row, imageSize) {
  if (!row.out_dir) {
    return "";
  }
  const expectedName = `samples.ink${imageSize}.u8`;
  const expectedPath = path.resolve(row.out_dir, expectedName);
  const trace = readCleanBitmapTrace(row);
  if (!trace.raw_samples) {
    throw new Error(`samples.tsv:${row.__line} trace raw_samples is missing`);
  }
  const matched = rawSampleReferenceCandidates(String(trace.raw_samples), row.out_dir)
    .find((candidate) => sameResolvedPath(candidate, expectedPath));
  if (!matched) {
    throw new Error(`samples.tsv:${row.__line} trace raw_samples must resolve to ${expectedName} in out_dir`);
  }
  return matched;
}

function readCleanBitmapTrace(row) {
  const tracePath = path.join(row.out_dir, "trace.json");
  if (!fs.existsSync(tracePath)) {
    throw new Error(`samples.tsv:${row.__line} missing trace.json for out_dir ${JSON.stringify(row.out_dir || "")}`);
  }
  const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  if (!trace || typeof trace !== "object" || Array.isArray(trace)) {
    throw new Error(`samples.tsv:${row.__line} trace.json is not a JSON object`);
  }
  if (trace.schema !== "nsrl.bitmap_sampler_trace.v1") {
    throw new Error(`samples.tsv:${row.__line} trace schema ${JSON.stringify(trace.schema || "")} != nsrl.bitmap_sampler_trace.v1`);
  }
  if (trace.latent_target_source !== "decoded-latent") {
    throw new Error(
      `samples.tsv:${row.__line} trace latent_target_source ${JSON.stringify(trace.latent_target_source || "")} != decoded-latent`,
    );
  }
  const violations = [];
  scanTraceObject(trace, [], violations);
  if (violations.length > 0) {
    throw new Error(`samples.tsv:${row.__line} trace ${violations[0].field}: ${violations[0].reason}`);
  }
  return trace;
}

function rawSampleReferenceCandidates(reference, sampleDir) {
  const candidates = [];
  if (path.isAbsolute(reference)) {
    candidates.push(path.resolve(reference));
  } else {
    candidates.push(path.resolve(sampleDir, reference));
    candidates.push(path.resolve(reference));
  }
  return [...new Set(candidates)];
}

function sameResolvedPath(left, right) {
  return path.resolve(left) === path.resolve(right);
}

function scanTraceObject(value, keyPath, violations) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      scanTraceObject(value[index], keyPath.concat(String(index)), violations);
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    const field = nextPath.join(".");
    if (isForbiddenKey(key) && !ALLOWED_TARGET_KEYS.has(key)) {
      violations.push({
        field,
        reason: "forbidden target-pixel, oracle, guidance, or cleanup field",
      });
    }
    if (typeof child === "string") {
      const reason = forbiddenValueReason(key, child);
      if (reason) {
        violations.push({ field, reason });
      }
    }
    scanTraceObject(child, nextPath, violations);
  }
}

function isForbiddenKey(key) {
  return FORBIDDEN_KEY_PATTERNS.some((pattern) => pattern.test(key));
}

function forbiddenValueReason(key, value) {
  if (isPathLikeKey(key) || isFreeTextValueKey(key)) {
    return "";
  }
  if (BROAD_FORBIDDEN_VALUE.test(value)) {
    return "forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value";
  }
  if (isSourceLikeKey(key) && SOURCE_FORBIDDEN_VALUE.test(value)) {
    return "forbidden generation source value";
  }
  return "";
}

function isFreeTextValueKey(key) {
  return FREE_TEXT_VALUE_KEYS.has(key);
}

function isPathLikeKey(key) {
  return /(path|file|dir|raw[_-]?samples|preview|pgm|model)$/i.test(key);
}

function isSourceLikeKey(key) {
  return /(source|mode|policy|method|strategy|guidance|cleanup|post[_-]?process|postprocess)$/i.test(key);
}

function clearGeneratedRetrievalFields(row) {
  for (const field of GENERATED_RETRIEVAL_FIELDS) {
    row[field] = "";
  }
}

function scoreSampleRow(row, retrievalHead, imageSize) {
  clearGeneratedRetrievalFields(row);
  const spiritId = Number(row.spirit_id || 0);
  if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72) {
    throw new Error(`samples.tsv:${row.__line} invalid spirit_id ${JSON.stringify(row.spirit_id)}`);
  }
  const filePath = rawSamplesPath(row, imageSize);
  if (!filePath || !fs.existsSync(filePath)) {
    throw new Error(`samples.tsv:${row.__line} missing raw samples for out_dir ${JSON.stringify(row.out_dir || "")}`);
  }
  const raw = fs.readFileSync(filePath);
  const metrics = generatedRetrievalMetrics(raw, imageSize, spiritId, retrievalHead);
  row.generated_retrieval_rank = metrics.best_rank;
  row.generated_retrieval_margin = metrics.best_margin ?? "";
  row.generated_retrieval_top1_spirit_id = metrics.top1_spirit_id ?? "";
  row.generated_retrieval_top1_name = metrics.top1_primary_name || "";
  row.generated_retrieval_identity = metrics.best_rank === 1 ? 1 : 0;
  return metrics;
}

function generatedRetrievalMetrics(raw, imageSize, targetSpiritId, retrievalHead) {
  const imageBytes = imageSize * imageSize;
  if (raw.length === 0 || raw.length % imageBytes !== 0) {
    throw new Error(`raw sample byte count ${raw.length} is not a positive multiple of ${imageBytes}`);
  }
  let best = null;
  for (let offset = 0; offset < raw.length; offset += imageBytes) {
    const image = raw.subarray(offset, offset + imageBytes);
    const signature = sampleSignature(image, imageSize);
    const ranked = rankRetrievalImage(retrievalHead, signature, retrievalHead.labels.length);
    const rank = retrievalTargetRank(ranked, targetSpiritId, retrievalHead.labels.length);
    const stats = scoreRankStats(ranked, targetSpiritId);
    const metrics = {
      best_rank: rank,
      best_margin: stats.margin,
      top1_spirit_id: ranked[0]?.spirit_id ?? null,
      top1_primary_name: ranked[0]?.primary_name ?? "",
    };
    if (
      !best ||
      metrics.best_rank < best.best_rank ||
      (metrics.best_rank === best.best_rank &&
        metrics.best_margin !== null &&
        (best.best_margin === null || metrics.best_margin > best.best_margin))
    ) {
      best = metrics;
    }
  }
  return best || {
    best_rank: retrievalMissRank(retrievalHead.labels.length),
    best_margin: null,
    top1_spirit_id: null,
    top1_primary_name: "",
  };
}

function retrievalTargetRank(ranked, targetSpiritId, labelCount) {
  const index = ranked.findIndex((row) => row.spirit_id === targetSpiritId);
  return index >= 0 ? index + 1 : retrievalMissRank(labelCount, ranked.length);
}

function retrievalMissRank(labelCount, rankedCount = 0) {
  return Math.max(72, Number(labelCount || 0), Number(rankedCount || 0)) + 1;
}

function sampleSignature(image, imageSize) {
  const sums = new Array(BINS).fill(0);
  const counts = new Array(BINS).fill(0);
  for (let y = 0; y < imageSize; y += 1) {
    const binY = Math.floor((y * GRID) / imageSize);
    for (let x = 0; x < imageSize; x += 1) {
      const binX = Math.floor((x * GRID) / imageSize);
      const bin = binY * GRID + binX;
      sums[bin] += image[y * imageSize + x];
      counts[bin] += 1;
    }
  }
  return sums.map((sum, index) => Math.floor((sum + Math.floor(counts[index] / 2)) / counts[index]));
}

function rankRetrievalImage(model, signature, count = 5) {
  const features = imageFeatures(symbolicImageTokens(signature), model.feature_count);
  const ranked = model.labels.map((label) => ({
    label: label.label,
    spirit_id: label.spirit_id,
    primary_name: label.primary_name,
    score: scoreLabel(model.image_head, label.label, features),
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

function scoreRankStats(ranked, targetSpiritId) {
  const target = ranked.find((row) => row.spirit_id === targetSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== targetSpiritId) || null;
  return {
    margin: target && runnerUp ? target.score - runnerUp.score : null,
  };
}

function symbolicImageTokens(signature) {
  return solomonImage.symbolicImageTokens(signature, symbolicImageOptions());
}

function symbolicImageOptions() {
  return {
    grid: GRID,
    imageBase: IMAGE_BASE,
    imageBins: IMAGE_BINS,
    channelTokens: {
      ink: IMAGE_CHANNEL_INK,
      edge: IMAGE_CHANNEL_EDGE,
      component: IMAGE_CHANNEL_COMPONENT,
      radial: IMAGE_CHANNEL_RADIAL,
      direction: IMAGE_CHANNEL_DIRECTION,
    },
  };
}

function imageFeatures(image, featureCount) {
  const out = new Map();
  let channel = "ink";
  let position = 0;
  for (const token of image) {
    if (CHANNEL_NAMES.has(token)) {
      channel = CHANNEL_NAMES.get(token);
      position = 0;
      addHashedFeature(out, featureCount, "channel", channel, 32);
      continue;
    }
    const bin = token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS ? token - IMAGE_BASE : token;
    addHashedFeature(out, featureCount, "ipos", `${channel}:${position}:${bin}`, 64);
    addHashedFeature(out, featureCount, "itok", `${channel}:${bin}`, 8);
    if (position % GRID === 0) {
      addHashedFeature(out, featureCount, "irow", `${channel}:${Math.floor(position / GRID)}:${bin}`, 6);
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

function summarizeModel(rows) {
  const count = rows.length;
  const top1 = rows.filter((row) => Number(row.generated_retrieval_rank || 0) === 1).length;
  const top5 = rows.filter((row) => {
    const rank = Number(row.generated_retrieval_rank || 0);
    return rank > 0 && rank <= 5;
  }).length;
  const rankTotal = rows.reduce((sum, row) => sum + Number(row.generated_retrieval_rank || 0), 0);
  const margins = rows
    .map((row) => Number(row.generated_retrieval_margin))
    .filter((value) => Number.isFinite(value));
  return {
    top1,
    top5,
    top1_per_mille: count === 0 ? 0 : Math.floor((top1 * 1000) / count),
    top5_per_mille: count === 0 ? 0 : Math.floor((top5 * 1000) / count),
    mean_rank_q8: count === 0 ? 0 : Math.floor((rankTotal * 256) / count),
    min_margin: margins.length === 0 ? "" : Math.min(...margins),
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const runDir = resolveRunDir(config.inputPath);
  const samplesPath = path.join(runDir, "samples.tsv");
  const summaryPath = path.join(runDir, "summary.tsv");
  const configPath = path.join(runDir, "config.json");
  if (!fs.existsSync(samplesPath)) {
    throw new Error(`samples.tsv not found: ${samplesPath}`);
  }
  if (!fs.existsSync(summaryPath)) {
    throw new Error(`summary.tsv not found: ${summaryPath}`);
  }
  const retrievalHead = readRetrievalHead(config.retrievalHeadPath);
  const samples = readTsv(samplesPath);
  const summary = readTsv(summaryPath);
  ensureColumns(samples.header, [
    ...GENERATED_RETRIEVAL_FIELDS,
  ]);
  ensureColumns(summary.header, [
    "generated_retrieval_top1",
    "generated_retrieval_top5",
    "generated_retrieval_top1_per_mille",
    "generated_retrieval_top5_per_mille",
    "mean_generated_retrieval_rank_q8",
    "min_generated_retrieval_margin",
  ]);

  const errors = [];
  let scored = 0;
  for (const row of samples.rows) {
    try {
      scoreSampleRow(row, retrievalHead, config.imageSize);
      scored += 1;
    } catch (error) {
      if (config.requireAll) {
        errors.push(error instanceof Error ? error.message : String(error));
      }
    }
  }
  const rowsByModel = new Map();
  for (const row of samples.rows) {
    if (!rowsByModel.has(row.model || "")) {
      rowsByModel.set(row.model || "", []);
    }
    rowsByModel.get(row.model || "").push(row);
  }
  for (const row of summary.rows) {
    const modelRows = rowsByModel.get(row.model || "") || [];
    const metrics = summarizeModel(modelRows);
    row.generated_retrieval_top1 = metrics.top1;
    row.generated_retrieval_top5 = metrics.top5;
    row.generated_retrieval_top1_per_mille = metrics.top1_per_mille;
    row.generated_retrieval_top5_per_mille = metrics.top5_per_mille;
    row.mean_generated_retrieval_rank_q8 = metrics.mean_rank_q8;
    row.min_generated_retrieval_margin = metrics.min_margin;
    const expected = Number(row.prompts || 0);
    if (config.requireAll && expected !== modelRows.length) {
      errors.push(`summary model ${row.model || "<missing-model>"} prompts ${expected} != scored rows ${modelRows.length}`);
    }
  }

  const result = {
    schema: "nsrl.solomon_generative_eval_retrieval_score.v1",
    ok: errors.length === 0,
    run_dir: runDir,
    retrieval_head: config.retrievalHeadPath,
    retrieval_head_model_hash: retrievalHead.model_hash || "",
    retrieval_head_feature_count: Number(retrievalHead.feature_count || 0),
    samples: samples.rows.length,
    scored,
    models: summary.rows.length,
    errors,
  };
  if (!result.ok) {
    console.log(JSON.stringify(result, null, 2));
    process.exit(1);
  }

  writeTsv(samplesPath, samples.header, samples.rows);
  writeTsv(summaryPath, summary.header, summary.rows);
  const runConfig = readJsonIfPresent(configPath);
  writeJson(configPath, {
    ...runConfig,
    retrievalHead: config.retrievalHeadPath,
    retrievalHeadModelHash: retrievalHead.model_hash || "",
    retrievalHeadFeatureCount: Number(retrievalHead.feature_count || 0),
  });
  console.log(JSON.stringify(result, null, 2));
}

main();
