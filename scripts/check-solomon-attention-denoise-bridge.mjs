#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  pairs: [],
  outPath: "",
  textIndexPath: "web/assets/solomon-spirit-text-signatures.tsv",
  retrievalHeadPath: "",
  requireRetrievalHead: false,
  maxOutputSignatureDistance: null,
  minOutputInkRange: 1,
  maxOutputRetrievalRank: 1,
  minOutputRetrievalMargin: 0,
  minUniqueTargets: 0,
};

const SIGNATURE_BINS = 256;
const GRID = 16;
const OUTPUT_IMAGE_SIZE = 128;
const IMAGE_BASE = 144;
const IMAGE_BINS = 16;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;
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
  "latent_target_plan",
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

function usage() {
  console.log(
    [
      "Usage: check-solomon-attention-denoise-bridge.mjs --pair ATTENTION_SAMPLE_DIR:DENOISE_DIR [--pair ...]",
      "",
      "Checks that a generated NSRLLMM1 16x16 image plan was used as the",
      "conditioning signature for a 128x128 NSRLTCH denoiser sample.",
      "",
      "Options:",
      "  --text-index PATH",
      "  --retrieval-head PATH",
      "  --require-retrieval-head",
      "  --max-output-signature-distance N",
      "  --min-output-ink-range N",
      "  --max-output-retrieval-rank N",
      "  --min-output-retrieval-margin N",
      "  --min-unique-targets N",
      "  --out PATH",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, pairs: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--pair") {
      config.pairs.push(parsePair(requireValue(argv, ++index, arg)));
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndexPath = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-retrieval-head") {
      config.requireRetrievalHead = true;
    } else if (arg === "--max-output-signature-distance") {
      config.maxOutputSignatureDistance = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-output-ink-range") {
      config.minOutputInkRange = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-output-retrieval-rank") {
      config.maxOutputRetrievalRank = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-output-retrieval-margin") {
      config.minOutputRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-unique-targets") {
      config.minUniqueTargets = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.pairs.length === 0) {
    throw new Error("--pair is required");
  }
  return config;
}

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function parsePositive(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePair(value) {
  const split = value.indexOf(":");
  if (split <= 0 || split === value.length - 1) {
    throw new Error(`invalid --pair ${value}; expected ATTENTION_SAMPLE_DIR:DENOISE_DIR`);
  }
  return {
    sampleDir: value.slice(0, split),
    denoiseDir: value.slice(split + 1),
  };
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readTextIndex(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  for (const column of ["number", "primary_name", "aliases", "signature_16x16"]) {
    if (!header.includes(column)) {
      throw new Error(`${filePath} is missing ${column}`);
    }
  }
  const indexOf = (column) => header.indexOf(column);
  return lines.filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const number = Number(fields[indexOf("number")]);
    const signature = fields[indexOf("signature_16x16")].split(",").map((part) => Number(part));
    if (!Number.isInteger(number) || number < 1 || number > 72) {
      throw new Error(`${filePath}:${rowIndex + 2} has invalid spirit number`);
    }
    if (signature.length !== SIGNATURE_BINS || signature.some((value) => !Number.isFinite(value))) {
      throw new Error(`${filePath}:${rowIndex + 2} has invalid 16x16 signature`);
    }
    return {
      label: number - 1,
      spirit_id: number,
      primary_name: fields[indexOf("primary_name")] || "",
      aliases: String(fields[indexOf("aliases")] || "").split("|").filter(Boolean),
      signature,
    };
  });
}

function readRetrievalHead(filePath) {
  if (!filePath) {
    return null;
  }
  const model = readJson(filePath);
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(model.schema)}`);
  }
  const recomputedModelHash = recomputeRetrievalHeadHash(model);
  return {
    ...model,
    raw: model,
    recomputed_model_hash: recomputedModelHash,
    hash_verified: Boolean(model.model_hash) && model.model_hash === recomputedModelHash,
    image_head: hydrateHead(model.image_head),
  };
}

function recomputeRetrievalHeadHash(model) {
  const copy = { ...model };
  delete copy.model_hash;
  return fnv64TextHex(JSON.stringify(copy));
}

function hydrateHead(head) {
  return {
    biases: head?.biases || [],
    weights: (head?.weights || []).map((entries) => new Map(entries)),
  };
}

function readPlan(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length !== SIGNATURE_BINS) {
    throw new Error(`${filePath} must contain ${SIGNATURE_BINS} bytes`);
  }
  return Array.from(bytes);
}

function checkPair(pair, config, context) {
  const errors = [];
  const samplePath = path.join(pair.sampleDir, "sample.json");
  const denoiseTracePath = path.join(pair.denoiseDir, "trace.json");
  const sample = readJson(samplePath);
  const trace = readJson(denoiseTracePath);
  const traceIntegrity = traceIntegrityReport(denoiseTracePath, trace);
  if (!traceIntegrity.ok) {
    for (const violation of traceIntegrity.violations) {
      errors.push(`${denoiseTracePath} ${violation.field}: ${violation.reason}`);
    }
  }
  const planPath = samplePathFor(pair.sampleDir, sample.image_ink16_u8 || "image.ink16.u8");
  const plan = readPlan(planPath);
  const tracePlanPath = trace.latent_target_plan || "";
  const denoiseModel = trace.model || "";
  const denoiseModelProvenance = denoiseModelInfo(denoiseModel, pair.denoiseDir, denoiseTracePath);
  if (!denoiseModel) {
    errors.push(`${denoiseTracePath} model is missing`);
  } else if (!denoiseModelProvenance.resolved_model) {
    errors.push(`${denoiseTracePath} model ${JSON.stringify(denoiseModel)} could not be resolved`);
  } else if (!denoiseModelProvenance.model_hash) {
    errors.push(`${denoiseTracePath} model ${JSON.stringify(denoiseModel)} hash is missing`);
  }
  const traceSignature = Array.isArray(trace.latent_target_signature)
    ? trace.latent_target_signature.map((value) => Number(value))
    : [];
  const rawSamplesPath = trace.raw_samples || path.join(pair.denoiseDir, `samples.ink${OUTPUT_IMAGE_SIZE}.u8`);
  const previewPath = trace.preview_pgm || path.join(pair.denoiseDir, "samples.pgm");
  const expected = context.spirits.length > 0 ? expectedSpiritForPrompt(sample.prompt || "", context.spirits) : null;

  expect(sample.schema, "nsrl.solomon_attention_sample_trace.v1", `${samplePath} schema`, errors);
  expect(trace.schema, "nsrl.bitmap_sampler_trace.v1", `${denoiseTracePath} schema`, errors);
  expect(trace.model_format, "NSRLTCH", `${denoiseTracePath} model_format`, errors);
  expect(trace.latent_target_source, "attention-plan", `${denoiseTracePath} latent_target_source`, errors);
  if (Number(trace.image_size || 0) !== OUTPUT_IMAGE_SIZE) {
    errors.push(`${denoiseTracePath} image_size ${trace.image_size || 0} != ${OUTPUT_IMAGE_SIZE}`);
  }
  if (Number(trace.feature_channels || 0) < 30) {
    errors.push(`${denoiseTracePath} feature_channels ${trace.feature_channels || 0} < 30`);
  }
  if (Number(trace.selected_count || 0) <= 0) {
    errors.push(`${denoiseTracePath} selected_count ${trace.selected_count || 0} <= 0`);
  }
  if (trace.latent_prompt !== sample.prompt) {
    errors.push(`${denoiseTracePath} latent_prompt ${JSON.stringify(trace.latent_prompt)} != sample prompt ${JSON.stringify(sample.prompt)}`);
  }
  if (!samePath(tracePlanPath, planPath)) {
    errors.push(`${denoiseTracePath} latent_target_plan ${JSON.stringify(tracePlanPath)} != ${JSON.stringify(planPath)}`);
  }
  if (traceSignature.length !== SIGNATURE_BINS) {
    errors.push(`${denoiseTracePath} latent_target_signature length ${traceSignature.length} != ${SIGNATURE_BINS}`);
  } else {
    for (let index = 0; index < SIGNATURE_BINS; index += 1) {
      if (traceSignature[index] !== plan[index]) {
        errors.push(`${denoiseTracePath} latent_target_signature differs from attention plan at bin ${index}`);
        break;
      }
    }
  }
  if (!fs.existsSync(rawSamplesPath)) {
    errors.push(`${rawSamplesPath} is missing`);
  } else {
    const expectedBytes = Math.max(1, Number(trace.samples || trace.selected_count || 1)) * OUTPUT_IMAGE_SIZE * OUTPUT_IMAGE_SIZE;
    const raw = fs.readFileSync(rawSamplesPath);
    if (raw.length !== expectedBytes) {
      errors.push(`${rawSamplesPath} size ${raw.length} != ${expectedBytes}`);
    }
    const outputStats = outputSignatureStats(raw, OUTPUT_IMAGE_SIZE, plan, expected, context, config);
    if (outputStats.min_ink_range < config.minOutputInkRange) {
      errors.push(`${rawSamplesPath} min output ink range ${outputStats.min_ink_range} < ${config.minOutputInkRange}`);
    }
    if (
      config.maxOutputSignatureDistance !== null &&
      outputStats.min_signature_distance > config.maxOutputSignatureDistance
    ) {
      errors.push(
        `${rawSamplesPath} min output signature distance ${outputStats.min_signature_distance} > ${config.maxOutputSignatureDistance}`,
      );
    }
    for (const error of outputStats.identity_errors || []) {
      errors.push(`${rawSamplesPath} ${error}`);
    }
    trace._bridge_output_stats = outputStats;
  }
  if (!fs.existsSync(previewPath)) {
    errors.push(`${previewPath} is missing`);
  }

  return {
    ok: errors.length === 0,
    sample_dir: pair.sampleDir,
    denoise_dir: pair.denoiseDir,
    prompt: sample.prompt || "",
    generated_text: sample.generated_text || "",
    expected_spirit_id: expected?.spirit_id ?? null,
    expected_primary_name: expected?.primary_name ?? "",
    attention_plan: planPath,
    denoise_trace: denoiseTracePath,
    denoise_model: denoiseModel,
    resolved_denoise_model: denoiseModelProvenance.resolved_model,
    denoise_model_hash: denoiseModelProvenance.model_hash,
    denoise_raw_samples: rawSamplesPath,
    denoise_preview_pgm: previewPath,
    latent_target_source: trace.latent_target_source || "",
    trace_integrity: traceIntegrity,
    output_signature: trace._bridge_output_stats || null,
    selected_min_text_distance: Number(trace.selected_min_text_distance || 0),
    selected_min_score: Number(trace.selected_min_score || 0),
    errors,
  };
}

function traceIntegrityReport(tracePath, trace) {
  const violations = [];
  scanTraceObject(trace, [], tracePath, violations);
  return {
    ok: violations.length === 0,
    violations,
  };
}

function scanTraceObject(value, keyPath, tracePath, violations) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      scanTraceObject(value[index], keyPath.concat(String(index)), tracePath, violations);
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    const field = nextPath.join(".");
    if (isForbiddenKey(key) && !ALLOWED_TARGET_KEYS.has(key)) {
      violations.push({
        path: tracePath,
        field,
        value: compactValue(child),
        reason: "forbidden target-pixel, oracle, guidance, or cleanup field",
      });
    }
    if (typeof child === "string") {
      const reason = forbiddenValueReason(key, child);
      if (reason) {
        violations.push({
          path: tracePath,
          field,
          value: compactValue(child),
          reason,
        });
      }
    }
    scanTraceObject(child, nextPath, tracePath, violations);
  }
}

function isForbiddenKey(key) {
  return FORBIDDEN_KEY_PATTERNS.some((pattern) => pattern.test(key));
}

function forbiddenValueReason(key, value) {
  if (BROAD_FORBIDDEN_VALUE.test(value)) {
    return "forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value";
  }
  if (isSourceLikeKey(key) && SOURCE_FORBIDDEN_VALUE.test(value)) {
    return "forbidden generation source value";
  }
  return "";
}

function isSourceLikeKey(key) {
  return /(source|mode|policy|method|strategy|guidance|cleanup|post[_-]?process|postprocess)$/i.test(key);
}

function compactValue(value) {
  if (Array.isArray(value)) {
    return `[array:${value.length}]`;
  }
  if (value && typeof value === "object") {
    return "{object}";
  }
  const text = String(value ?? "");
  return text.length > 96 ? `${text.slice(0, 93)}...` : text;
}

function outputSignatureStats(raw, imageSize, plan, expected, context, config) {
  const imageBytes = imageSize * imageSize;
  const count = Math.floor(raw.length / imageBytes);
  const samples = [];
  const identityErrors = [];
  for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
    const start = sampleIndex * imageBytes;
    const image = raw.subarray(start, start + imageBytes);
    const signature = downsampleSignature(image, imageSize);
    const distance = signatureDistance(signature, plan);
    const inkRange = imageInkRange(image);
    const sample = {
      index: sampleIndex,
      signature_distance: distance,
      ink_range: inkRange,
    };
    if (expected) {
      const signatureRanked = rankSignature(signature, context.spirits);
      const signatureRank = signatureRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
      sample.expected_spirit_id = expected.spirit_id;
      sample.expected_primary_name = expected.primary_name;
      sample.signature_rank = signatureRank;
      sample.signature_top1_spirit_id = signatureRanked[0]?.spirit_id ?? null;
      sample.signature_top1_primary_name = signatureRanked[0]?.primary_name ?? "";
    }
    if (expected && context.retrievalHead) {
      const imageRanked = rankRetrievalImage(context.retrievalHead, signature, context.retrievalHead.labels.length);
      const imageRank = imageRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
      const imageStats = scoreRankStats(imageRanked, expected.spirit_id);
      sample.retrieval_image_rank = imageRank;
      sample.retrieval_image_margin = imageStats.margin;
      sample.retrieval_image_top1_spirit_id = imageRanked[0]?.spirit_id ?? null;
      sample.retrieval_image_top1_primary_name = imageRanked[0]?.primary_name ?? "";
      sample.image_to_text_identity = imageRank === 1 && imageRanked[0]?.spirit_id === expected.spirit_id;
      if (imageRank < 1 || imageRank > config.maxOutputRetrievalRank) {
        identityErrors.push(`sample ${sampleIndex} output retrieval image rank ${imageRank} > ${config.maxOutputRetrievalRank}`);
      }
      if (imageStats.margin !== null && imageStats.margin < config.minOutputRetrievalMargin) {
        identityErrors.push(
          `sample ${sampleIndex} output retrieval margin ${imageStats.margin} < ${config.minOutputRetrievalMargin}`,
        );
      }
    }
    samples.push(sample);
  }
  const retrievalSamples = samples.filter((sample) => sample.image_to_text_identity !== undefined);
  return {
    samples: samples.length,
    min_signature_distance: samples.length === 0 ? null : Math.min(...samples.map((sample) => sample.signature_distance)),
    mean_signature_distance_q8:
      samples.length === 0
        ? null
        : Math.round((samples.reduce((sum, sample) => sum + sample.signature_distance, 0) * 256) / samples.length),
    min_ink_range: samples.length === 0 ? 0 : Math.min(...samples.map((sample) => sample.ink_range)),
    output_image_to_text_identification:
      retrievalSamples.length === 0
        ? null
        : retrievalSamples.every((sample) => sample.image_to_text_identity === true),
    min_retrieval_image_margin:
      retrievalSamples.length === 0
        ? null
        : Math.min(...retrievalSamples.map((sample) => sample.retrieval_image_margin ?? 0)),
    identity_errors: identityErrors,
    samples_detail: samples.slice(0, 8),
  };
}

function downsampleSignature(image, imageSize) {
  const sums = new Array(SIGNATURE_BINS).fill(0);
  const counts = new Array(SIGNATURE_BINS).fill(0);
  for (let y = 0; y < imageSize; y += 1) {
    const binY = Math.floor((y * 16) / imageSize);
    for (let x = 0; x < imageSize; x += 1) {
      const binX = Math.floor((x * 16) / imageSize);
      const bin = binY * 16 + binX;
      sums[bin] += image[y * imageSize + x];
      counts[bin] += 1;
    }
  }
  return sums.map((sum, index) => (counts[index] === 0 ? 0 : Math.floor((sum + Math.floor(counts[index] / 2)) / counts[index])));
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < SIGNATURE_BINS; index += 1) {
    distance += Math.abs((left[index] || 0) - (right[index] || 0));
  }
  return distance;
}

function imageInkRange(image) {
  let min = 255;
  let max = 0;
  for (const value of image) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return max - min;
}

function expectedSpiritForPrompt(prompt, spirits) {
  const promptKey = normalizeKey(prompt);
  const candidates = spirits
    .map((spirit) => ({
      spirit,
      score: spiritPromptScore(spirit, promptKey),
    }))
    .filter((row) => row.score > 0)
    .sort((left, right) => right.score - left.score || left.spirit.spirit_id - right.spirit.spirit_id);
  if (candidates.length === 0) {
    throw new Error(`prompt does not name a known spirit: ${prompt}`);
  }
  return candidates[0].spirit;
}

function spiritPromptScore(spirit, promptKey) {
  let score = 0;
  for (const name of [spirit.primary_name, ...spirit.aliases]) {
    const key = normalizeKey(name);
    if (!key) continue;
    if (promptKey === key || promptKey.startsWith(`${key} `)) {
      score = Math.max(score, 1_000_000 + key.length * 1000);
    } else if (` ${promptKey} `.includes(` ${key} `)) {
      score = Math.max(score, 100_000 + key.length * 100);
    }
  }
  return score;
}

function rankSignature(signature, spirits) {
  const ranked = spirits.map((spirit) => ({
    spirit_id: spirit.spirit_id,
    primary_name: spirit.primary_name,
    distance: signatureDistance(signature, spirit.signature),
  }));
  ranked.sort((left, right) => left.distance - right.distance || left.spirit_id - right.spirit_id);
  return ranked;
}

function rankRetrievalImage(model, signature, count = 5) {
  const features = imageFeatures(symbolicImageTokens(signature), model.feature_count);
  return rankHead(model.image_head, model.labels, features, count);
}

function rankHead(head, labels, features, count) {
  const ranked = labels.map((label) => ({
    label: label.label,
    spirit_id: label.spirit_id,
    primary_name: label.primary_name,
    score: scoreLabel(head, label.label, features),
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

function scoreRankStats(ranked, expectedSpiritId) {
  const expected = ranked.find((row) => row.spirit_id === expectedSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== expectedSpiritId) || null;
  return {
    score: expected?.score ?? null,
    runner_up_score: runnerUp?.score ?? null,
    margin: expected && runnerUp ? expected.score - runnerUp.score : null,
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

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function fnv64BytesHex(bytes) {
  let hash = FNV64_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64TextHex(value) {
  let hash = FNV64_OFFSET;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function denoiseModelInfo(modelRef, denoiseDir, tracePath) {
  if (!modelRef) {
    return {
      model: "",
      resolved_model: "",
      model_hash: "",
    };
  }
  const traceDir = path.dirname(tracePath);
  const resolvedModel = resolveReferencedPath(modelRef, [denoiseDir, traceDir]);
  return {
    model: modelRef,
    resolved_model: resolvedModel,
    model_hash: resolvedModel ? fnv64FileHex(resolvedModel) : "",
  };
}

function resolveReferencedPath(ref, baseDirs = []) {
  const candidates = path.isAbsolute(ref)
    ? [path.resolve(ref)]
    : [path.resolve(ref), ...baseDirs.map((baseDir) => path.resolve(baseDir, ref))];
  for (const candidate of [...new Set(candidates)]) {
    if (fs.existsSync(candidate)) {
      return normalizePath(candidate);
    }
  }
  return "";
}

function normalizePath(filePath) {
  const resolved = path.resolve(filePath);
  try {
    return fs.realpathSync.native(resolved);
  } catch (_error) {
    return resolved;
  }
}

function summarizeDenoiseModelProvenance(results) {
  const errors = [];
  const rows = results.map((result, index) => ({
    index,
    denoise_model: result.denoise_model || "",
    resolved_denoise_model: result.resolved_denoise_model || "",
    denoise_model_hash: result.denoise_model_hash || "",
  }));
  const hashes = [...new Set(rows.map((row) => row.denoise_model_hash).filter(Boolean))].sort();
  const resolvedModels = [...new Set(rows.map((row) => row.resolved_denoise_model).filter(Boolean))].sort();
  for (const row of rows) {
    if (!row.denoise_model) {
      errors.push(`denoise bridge result ${row.index} denoise_model is missing`);
    }
    if (!row.resolved_denoise_model) {
      errors.push(`denoise bridge result ${row.index} resolved_denoise_model is missing`);
    }
    if (!row.denoise_model_hash) {
      errors.push(`denoise bridge result ${row.index} denoise_model_hash is missing`);
    }
  }
  if (hashes.length !== 1) {
    errors.push(`denoise bridge expected exactly one denoiser model hash, found ${hashes.length}`);
  }
  return {
    ok: errors.length === 0,
    errors,
    denoise_model: resolvedModels.length === 1 ? rows.find((row) => row.resolved_denoise_model === resolvedModels[0])?.denoise_model || "" : "",
    resolved_denoise_model: resolvedModels.length === 1 ? resolvedModels[0] : "",
    denoise_model_hash: hashes.length === 1 ? hashes[0] : "",
    denoise_model_hashes: hashes,
    consistent: hashes.length === 1,
    results: rows,
  };
}

function summarizeRetrievalHeadProvenance(config, retrievalHead) {
  if (!retrievalHead) {
    return {
      ok: !config.requireRetrievalHead,
      required: config.requireRetrievalHead,
      retrieval_head: config.retrievalHeadPath || "",
      model_hash: "",
      recomputed_model_hash: "",
      hash_verified: false,
      feature_count: 0,
      label_count: 0,
      errors: config.requireRetrievalHead ? ["retrieval head is required but missing"] : [],
    };
  }
  const errors = [];
  if (!retrievalHead.model_hash) {
    errors.push("retrieval head model_hash is missing");
  }
  if (!retrievalHead.recomputed_model_hash) {
    errors.push("retrieval head recomputed model hash is missing");
  }
  if (retrievalHead.hash_verified !== true) {
    errors.push(
      `retrieval head model_hash ${retrievalHead.model_hash || ""} != recomputed ${retrievalHead.recomputed_model_hash || ""}`,
    );
  }
  return {
    ok: errors.length === 0,
    required: config.requireRetrievalHead,
    retrieval_head: config.retrievalHeadPath || "",
    model_hash: retrievalHead.model_hash || "",
    recomputed_model_hash: retrievalHead.recomputed_model_hash || "",
    hash_verified: retrievalHead.hash_verified === true,
    feature_count: Number(retrievalHead.feature_count || 0),
    label_count: Array.isArray(retrievalHead.labels) ? retrievalHead.labels.length : 0,
    errors,
  };
}

function normalizeKey(text) {
  return String(text || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, " ")
    .replace(/\[[0-9]+\]/g, " ")
    .toLowerCase()
    .replace(/[^a-z0-9']+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function expect(actual, expected, label, errors) {
  if (actual !== expected) {
    errors.push(`${label} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function samePath(left, right) {
  if (!left || !right) {
    return false;
  }
  return path.resolve(left) === path.resolve(right);
}

function samplePathFor(sampleDir, filePath) {
  if (path.isAbsolute(filePath)) {
    return filePath;
  }
  return path.join(sampleDir, filePath);
}

function writeJson(filePath, row) {
  const dir = path.dirname(filePath);
  if (dir && dir !== ".") {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`);
}

function targetCoverage(results, minUniqueTargets) {
  const ids = results
    .map((row) => Number(row.expected_spirit_id || 0))
    .filter((id) => Number.isInteger(id) && id >= 1 && id <= 72);
  const uniqueIds = Array.from(new Set(ids)).sort((left, right) => left - right);
  const missingIds = Array.from({ length: 72 }, (_unused, index) => index + 1).filter(
    (id) => !uniqueIds.includes(id),
  );
  const errors = [];
  if (minUniqueTargets > 0 && uniqueIds.length < minUniqueTargets) {
    errors.push(`denoise bridge unique targets ${uniqueIds.length} < ${minUniqueTargets}`);
  }
  return {
    min_unique_targets: minUniqueTargets,
    expected_spirit_ids: ids,
    unique_expected_spirit_ids: uniqueIds,
    expected_unique_targets: uniqueIds.length,
    missing_expected_spirit_ids: missingIds,
    target_coverage_ok: errors.length === 0,
    errors,
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const retrievalHead = readRetrievalHead(config.retrievalHeadPath);
  if (config.requireRetrievalHead && !retrievalHead) {
    throw new Error("--require-retrieval-head was set but --retrieval-head was not supplied");
  }
  const context = {
    spirits: retrievalHead ? readTextIndex(config.textIndexPath) : [],
    retrievalHead,
  };
  const results = config.pairs.map((pair) => checkPair(pair, config, context));
  const denoiseModelProvenance = summarizeDenoiseModelProvenance(results);
  const retrievalHeadProvenance = summarizeRetrievalHeadProvenance(config, retrievalHead);
  const coverage = targetCoverage(results, config.minUniqueTargets);
  const errors = results
    .flatMap((result) => result.errors)
    .concat(denoiseModelProvenance.errors, retrievalHeadProvenance.errors, coverage.errors);
  const result = {
    schema: "nsrl.solomon_attention_denoise_bridge_check.v1",
    ok: errors.length === 0,
    pairs: results.length,
    min_unique_targets: coverage.min_unique_targets,
    expected_spirit_ids: coverage.expected_spirit_ids,
    unique_expected_spirit_ids: coverage.unique_expected_spirit_ids,
    expected_unique_targets: coverage.expected_unique_targets,
    missing_expected_spirit_ids: coverage.missing_expected_spirit_ids,
    target_coverage_ok: coverage.target_coverage_ok,
    denoise_model: denoiseModelProvenance.denoise_model,
    resolved_denoise_model: denoiseModelProvenance.resolved_denoise_model,
    denoise_model_hash: denoiseModelProvenance.denoise_model_hash,
    denoise_model_hashes: denoiseModelProvenance.denoise_model_hashes,
    denoise_model_consistent: denoiseModelProvenance.consistent,
    denoise_model_provenance: denoiseModelProvenance,
    retrieval_head: config.retrievalHeadPath || null,
    retrieval_head_model_hash: retrievalHead?.model_hash || "",
    recomputed_retrieval_head_model_hash: retrievalHead?.recomputed_model_hash || "",
    retrieval_head_hash_verified: retrievalHead?.hash_verified === true,
    retrieval_head_feature_count: Number(retrievalHead?.feature_count || 0),
    retrieval_head_provenance: retrievalHeadProvenance,
    min_output_signature_distance:
      results.length === 0
        ? null
        : Math.min(...results.map((row) => row.output_signature?.min_signature_distance ?? Number.POSITIVE_INFINITY)),
    min_output_ink_range:
      results.length === 0
        ? null
        : Math.min(...results.map((row) => row.output_signature?.min_ink_range ?? 0)),
    output_image_to_text_identification:
      retrievalHead && results.length > 0
        ? results.every((row) => row.output_signature?.output_image_to_text_identification === true)
        : null,
    trace_integrity_ok: results.every((row) => row.trace_integrity?.ok === true),
    min_output_retrieval_image_margin:
      retrievalHead && results.length > 0
        ? Math.min(...results.map((row) => row.output_signature?.min_retrieval_image_margin ?? 0))
        : null,
    results,
    errors,
  };
  if (config.outPath) {
    writeJson(config.outPath, result);
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
