#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  sampleDirs: [],
  textIndexPath: "web/assets/solomon-spirit-text-signatures.tsv",
  retrievalHeadPath: "",
  outPath: "",
  maxSignatureRank: 1,
  maxRetrievalRank: 1,
  maxTextRank: 1,
  minSignatureMargin: 0,
  minRetrievalMargin: 0,
  minTextMargin: 0,
  requireRetrievalHead: false,
  maxMisses: 8,
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
const STOPWORDS = new Set([
  "a",
  "about",
  "after",
  "again",
  "all",
  "also",
  "an",
  "and",
  "any",
  "are",
  "as",
  "at",
  "be",
  "before",
  "both",
  "but",
  "by",
  "can",
  "etc",
  "for",
  "from",
  "great",
  "has",
  "have",
  "he",
  "her",
  "him",
  "his",
  "in",
  "is",
  "it",
  "man",
  "many",
  "men",
  "must",
  "of",
  "or",
  "order",
  "seal",
  "shall",
  "she",
  "spirit",
  "spirits",
  "the",
  "this",
  "thou",
  "to",
  "unto",
  "upon",
  "which",
  "who",
  "will",
  "with",
]);

function usage() {
  console.log(
    [
      "Usage: check-solomon-attention-sample-binding.mjs --sample-dir PATH [--sample-dir PATH...]",
      "",
      "Ranks generated NSRLLMM1 16x16 image plans against known Solomon seals",
      "and optionally asks retrieval-head.json to identify the generated image.",
      "",
      "Options:",
      "  --text-index PATH",
      "  --retrieval-head PATH",
      "  --out PATH",
      "  --max-signature-rank N",
      "  --max-retrieval-rank N",
      "  --max-text-rank N",
      "  --min-signature-margin N",
      "  --min-retrieval-margin N",
      "  --min-text-margin N",
      "  --require-retrieval-head",
      "  --max-misses N",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, sampleDirs: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--sample-dir") {
      config.sampleDirs.push(requireValue(argv, ++index, arg));
    } else if (arg === "--text-index") {
      config.textIndexPath = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--max-signature-rank") {
      config.maxSignatureRank = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-retrieval-rank") {
      config.maxRetrievalRank = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-text-rank") {
      config.maxTextRank = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-signature-margin") {
      config.minSignatureMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-retrieval-margin") {
      config.minRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-text-margin") {
      config.minTextMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-retrieval-head") {
      config.requireRetrievalHead = true;
    } else if (arg === "--max-misses") {
      config.maxMisses = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.sampleDirs.length === 0) {
    throw new Error("--sample-dir is required");
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
    if (signature.length !== BINS || signature.some((value) => !Number.isFinite(value))) {
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
  const model = JSON.parse(fs.readFileSync(filePath, "utf8"));
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(model.schema)}`);
  }
  return {
    ...model,
    text_head: hydrateHead(model.text_head),
    image_head: hydrateHead(model.image_head),
  };
}

function hydrateHead(head) {
  return {
    biases: head.biases || [],
    weights: (head.weights || []).map((entries) => new Map(entries)),
  };
}

function readSample(sampleDir) {
  const tracePath = path.join(sampleDir, "sample.json");
  const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  const imagePath = resolveSampleArtifact(sampleDir, trace.image_ink16_u8 || "image.ink16.u8");
  const bytes = fs.readFileSync(imagePath);
  if (bytes.length !== BINS) {
    throw new Error(`${imagePath} has ${bytes.length} bytes, expected ${BINS}`);
  }
  const signature = Array.from(bytes);
  return { sampleDir, trace, imagePath, signature };
}

function resolveSampleArtifact(sampleDir, artifactPath) {
  if (path.isAbsolute(artifactPath)) {
    return artifactPath;
  }
  const repoRelative = path.resolve(artifactPath);
  if (fs.existsSync(repoRelative)) {
    return repoRelative;
  }
  return path.resolve(sampleDir, artifactPath);
}

function expectedSpiritForSample(sample, spirits) {
  const promptKey = normalizeKey(sample.trace.prompt || "");
  const candidates = spirits
    .map((spirit) => ({
      spirit,
      score: spiritPromptScore(spirit, promptKey),
    }))
    .filter((row) => row.score > 0)
    .sort((left, right) => right.score - left.score || left.spirit.spirit_id - right.spirit.spirit_id);
  if (candidates.length === 0) {
    throw new Error(`${sample.sampleDir} prompt does not name a known spirit: ${sample.trace.prompt}`);
  }
  return candidates[0].spirit;
}

function spiritPromptScore(spirit, promptKey) {
  let score = 0;
  for (const name of [spirit.primary_name, ...spirit.aliases]) {
    const key = normalizeKey(name);
    if (!key) {
      continue;
    }
    if (promptKey === key || promptKey.startsWith(`${key} `)) {
      score = Math.max(score, 1_000_000 + key.length * 1000);
    } else if (` ${promptKey} `.includes(` ${key} `)) {
      score = Math.max(score, 100_000 + key.length * 100);
    }
  }
  return score;
}

function signatureRank(signature, spirits) {
  const ranked = spirits.map((spirit) => ({
    label: spirit.label,
    spirit_id: spirit.spirit_id,
    primary_name: spirit.primary_name,
    distance: signatureDistance(signature, spirit.signature),
  }));
  ranked.sort((left, right) => left.distance - right.distance || left.spirit_id - right.spirit_id);
  return ranked;
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < BINS; index += 1) {
    distance += Math.abs((left[index] || 0) - (right[index] || 0));
  }
  return distance;
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

function rankRetrievalImage(model, signature, count = 5) {
  const features = imageFeatures(symbolicImageTokens(signature), model.feature_count);
  return rankHead(model.image_head, model.labels, features, count);
}

function rankRetrievalText(model, text, count = 5) {
  const queryKey = normalizeKey(text);
  const features = textFeatures(text, model.feature_count);
  const ranked = [];
  for (const label of model.labels) {
    const sparseScore = scoreLabel(model.text_head, label.label, features);
    ranked.push({
      label: label.label,
      spirit_id: label.spirit_id,
      primary_name: label.primary_name,
      score: sparseScore + identityAnchorScore(model, label, queryKey),
    });
  }
  ranked.sort((left, right) => right.score - left.score || left.spirit_id - right.spirit_id);
  return ranked.slice(0, count);
}

function identityAnchorScore(model, label, queryKey) {
  const anchor = model.identity_anchor || { leading_boost: 0, mention_boost: 0 };
  let score = 0;
  for (const key of identityAnchorKeys(label)) {
    if (queryKey === key || queryKey.startsWith(`${key} `)) {
      score = Math.max(score, anchor.leading_boost + key.length * 1000);
    } else if (` ${queryKey} `.includes(` ${key} `)) {
      score = Math.max(score, anchor.mention_boost + key.length * 100);
    }
  }
  return score;
}

function identityAnchorKeys(label) {
  const keys = new Set();
  const add = (value) => {
    const key = normalizeKey(value);
    if (key) {
      keys.add(key);
    }
  };
  for (const name of [label.primary_name, ...(label.aliases || [])]) {
    add(name);
  }
  const spiritId = Number(label.spirit_id ?? label.number);
  if (Number.isInteger(spiritId)) {
    add(`seal id ${spiritId}`);
    add(`spirit ${spiritId}`);
    add(`goetic spirit ${spiritId}`);
  }
  return keys;
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
    if (position % 16 === 0) {
      addHashedFeature(out, featureCount, "irow", `${channel}:${Math.floor(position / 16)}:${bin}`, 6);
    }
    position += 1;
  }
  return [...out.entries()];
}

function textFeatures(text, featureCount) {
  const tokens = tokenize(text);
  const out = new Map();
  if (tokens.length === 0) return [];
  if (tokens.length <= 8) addHashedFeature(out, featureCount, "whole", tokens.join(" "), 96);
  addHashedFeature(out, featureCount, "lead1", tokens[0], 72);
  if (tokens[1]) addHashedFeature(out, featureCount, "lead2", `${tokens[0]} ${tokens[1]}`, 88);
  const content = tokens.filter((token) => token.length >= 3 && !STOPWORDS.has(token));
  if (content.length > 0 && content.length <= 8) {
    addHashedFeature(out, featureCount, "content", content.join(" "), 84);
    addHashedFeature(out, featureCount, "cset", [...content].sort().join(" "), 80);
  }
  for (let index = 0; index < tokens.length; index += 1) {
    addHashedFeature(out, featureCount, "tok", tokens[index], 12 + (index < 4 ? 8 : 0));
    if (tokens[index + 1]) addHashedFeature(out, featureCount, "bi", `${tokens[index]} ${tokens[index + 1]}`, 18);
    if (tokens[index + 1] && tokens[index + 2]) {
      addHashedFeature(out, featureCount, "tri", `${tokens[index]} ${tokens[index + 1]} ${tokens[index + 2]}`, 24);
    }
  }
  for (let left = 0; left < content.length; left += 1) {
    addHashedFeature(out, featureCount, "ctok", content[left], 18);
    for (let right = left + 1; right < Math.min(content.length, left + 10); right += 1) {
      addHashedFeature(out, featureCount, "pair", [content[left], content[right]].sort().join(" "), 20);
    }
  }
  return [...out.entries()];
}

function addHashedFeature(out, featureCount, namespace, value, amount) {
  const hash = fnv32(`${namespace}\xff${value}`);
  const index = hash % featureCount;
  const sign = hash & 0x80000000 ? -1 : 1;
  out.set(index, Math.max(-127, Math.min(127, (out.get(index) || 0) + sign * amount)));
}

function tokenize(text) {
  return normalizeKey(text)
    .split(/\s+/)
    .map(normalizeToken)
    .filter((token) => token.length >= 2);
}

function normalizeToken(token) {
  if (["teach", "teacher", "teaches", "teacheth", "teaching"].includes(token)) return "teach";
  if (["know", "knows", "known", "knowing", "knoweth", "knowledge"].includes(token)) return "know";
  if (["make", "makes", "maketh", "making"].includes(token)) return "make";
  if (["discover", "discovers", "discovereth", "discovering"].includes(token)) return "discover";
  if (["answer", "answers", "answereth", "answering"].includes(token)) return "answer";
  if (["virtue", "virtues"].includes(token)) return "virtue";
  if (["language", "languages", "tongue", "tongues"].includes(token)) return "language";
  if (["friend", "friends"].includes(token)) return "friend";
  if (["enemy", "enemies", "foe", "foes"].includes(token)) return "enemy";
  return token;
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

function fnv32(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function sampleResult(sample, expected, spirits, retrievalHead) {
  const signatureRanked = signatureRank(sample.signature, spirits);
  const signatureRankIndex = signatureRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
  const signatureStats = distanceRankStats(signatureRanked, expected.spirit_id);
  const result = {
    sample_dir: sample.sampleDir,
    prompt: sample.trace.prompt,
    image_ink16_u8: sample.imagePath,
    expected_spirit_id: expected.spirit_id,
    expected_primary_name: expected.primary_name,
    generated_text: sample.trace.generated_text,
    image_prior_source: sample.trace.image_prior_source || "",
    conditioning_primary_name: sample.trace.conditioning_primary_name || "",
    signature_rank: signatureRankIndex,
    signature_distance: signatureRanked[signatureRankIndex - 1]?.distance ?? null,
    signature_runner_up_distance: signatureStats.runner_up_distance,
    signature_margin: signatureStats.margin,
    signature_top5: signatureRanked.slice(0, 5),
  };
  if (retrievalHead) {
    const imageRanked = rankRetrievalImage(retrievalHead, sample.signature, retrievalHead.labels.length);
    const imageRank = imageRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
    const imageStats = scoreRankStats(imageRanked, expected.spirit_id);
    const textRanked = rankRetrievalText(
      retrievalHead,
      sample.trace.prompt || "",
      retrievalHead.labels.length,
    );
    const generatedText = String(sample.trace.generated_text || "").trim();
    const generatedTextRanked = generatedText
      ? rankRetrievalText(retrievalHead, generatedText, retrievalHead.labels.length)
      : [];
    const textRank = textRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
    const textStats = scoreRankStats(textRanked, expected.spirit_id);
    const generatedTextRank = generatedTextRanked.findIndex((row) => row.spirit_id === expected.spirit_id) + 1;
    const generatedTextStats = scoreRankStats(generatedTextRanked, expected.spirit_id);
    result.retrieval_image_rank = imageRank;
    result.retrieval_image_score = imageStats.score;
    result.retrieval_image_runner_up_score = imageStats.runner_up_score;
    result.retrieval_image_margin = imageStats.margin;
    result.retrieval_image_top5 = imageRanked.slice(0, 5);
    result.image_to_text_identity = {
      expected_spirit_id: expected.spirit_id,
      expected_primary_name: expected.primary_name,
      predicted_spirit_id: imageRanked[0]?.spirit_id ?? null,
      predicted_primary_name: imageRanked[0]?.primary_name ?? "",
      rank: imageRank,
      margin: imageStats.margin,
      ok: imageRank === 1 && imageRanked[0]?.spirit_id === expected.spirit_id,
    };
    result.retrieval_text_rank = textRank;
    result.retrieval_text_score = textStats.score;
    result.retrieval_text_runner_up_score = textStats.runner_up_score;
    result.retrieval_text_margin = textStats.margin;
    result.retrieval_text_top5 = textRanked.slice(0, 5);
    result.generated_text_rank = generatedTextRank;
    result.generated_text_score = generatedTextStats.score;
    result.generated_text_runner_up_score = generatedTextStats.runner_up_score;
    result.generated_text_margin = generatedTextStats.margin;
    result.generated_text_top5 = generatedTextRanked.slice(0, 5);
    result.generated_text_identity = {
      expected_spirit_id: expected.spirit_id,
      expected_primary_name: expected.primary_name,
      predicted_spirit_id: generatedTextRanked[0]?.spirit_id ?? null,
      predicted_primary_name: generatedTextRanked[0]?.primary_name ?? "",
      rank: generatedTextRank,
      margin: generatedTextStats.margin,
      ok: generatedTextRank === 1 && generatedTextRanked[0]?.spirit_id === expected.spirit_id,
    };
    result.text_image_agree =
      imageRanked[0]?.spirit_id === textRanked[0]?.spirit_id &&
      imageRanked[0]?.spirit_id === expected.spirit_id;
    result.generated_text_image_agree =
      generatedTextRanked[0]?.spirit_id === imageRanked[0]?.spirit_id &&
      imageRanked[0]?.spirit_id === expected.spirit_id;
    result.signature_retrieval_agree =
      signatureRanked[0]?.spirit_id === imageRanked[0]?.spirit_id &&
      imageRanked[0]?.spirit_id === expected.spirit_id;
  }
  result.confidence = {
    signature_rank: result.signature_rank,
    signature_margin: result.signature_margin,
    retrieval_image_rank: result.retrieval_image_rank ?? null,
    retrieval_image_margin: result.retrieval_image_margin ?? null,
    image_to_text_identity: result.image_to_text_identity?.ok ?? null,
    retrieval_text_rank: result.retrieval_text_rank ?? null,
    retrieval_text_margin: result.retrieval_text_margin ?? null,
    generated_text_rank: result.generated_text_rank ?? null,
    generated_text_margin: result.generated_text_margin ?? null,
    generated_text_identity: result.generated_text_identity?.ok ?? null,
    text_image_agree: result.text_image_agree ?? null,
    generated_text_image_agree: result.generated_text_image_agree ?? null,
    signature_retrieval_agree: result.signature_retrieval_agree ?? null,
  };
  return result;
}

function distanceRankStats(ranked, expectedSpiritId) {
  const expected = ranked.find((row) => row.spirit_id === expectedSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== expectedSpiritId) || null;
  return {
    distance: expected?.distance ?? null,
    runner_up_distance: runnerUp?.distance ?? null,
    margin:
      expected && runnerUp
        ? runnerUp.distance - expected.distance
        : null,
  };
}

function scoreRankStats(ranked, expectedSpiritId) {
  const expected = ranked.find((row) => row.spirit_id === expectedSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== expectedSpiritId) || null;
  return {
    score: expected?.score ?? null,
    runner_up_score: runnerUp?.score ?? null,
    margin:
      expected && runnerUp
        ? expected.score - runnerUp.score
        : null,
  };
}

function checkResult(result, config, hasRetrievalHead) {
  const errors = [];
  if (result.signature_rank < 1 || result.signature_rank > config.maxSignatureRank) {
    errors.push(`${result.sample_dir} signature rank ${result.signature_rank} > ${config.maxSignatureRank}`);
  }
  if (result.signature_margin !== null && result.signature_margin < config.minSignatureMargin) {
    errors.push(
      `${result.sample_dir} signature margin ${result.signature_margin} < ${config.minSignatureMargin}`,
    );
  }
  if (
    hasRetrievalHead &&
    (result.retrieval_image_rank < 1 || result.retrieval_image_rank > config.maxRetrievalRank)
  ) {
    errors.push(`${result.sample_dir} retrieval image rank ${result.retrieval_image_rank} > ${config.maxRetrievalRank}`);
  }
  if (
    hasRetrievalHead &&
    result.retrieval_image_margin !== null &&
    result.retrieval_image_margin < config.minRetrievalMargin
  ) {
    errors.push(
      `${result.sample_dir} retrieval image margin ${result.retrieval_image_margin} < ${config.minRetrievalMargin}`,
    );
  }
  if (
    hasRetrievalHead &&
    (result.retrieval_text_rank < 1 || result.retrieval_text_rank > config.maxTextRank)
  ) {
    errors.push(`${result.sample_dir} retrieval text rank ${result.retrieval_text_rank} > ${config.maxTextRank}`);
  }
  if (
    hasRetrievalHead &&
    result.retrieval_text_margin !== null &&
    result.retrieval_text_margin < config.minTextMargin
  ) {
    errors.push(
      `${result.sample_dir} retrieval text margin ${result.retrieval_text_margin} < ${config.minTextMargin}`,
    );
  }
  if (hasRetrievalHead && !String(result.generated_text || "").trim()) {
    errors.push(`${result.sample_dir} generated text is empty`);
  }
  if (
    hasRetrievalHead &&
    (result.generated_text_rank < 1 || result.generated_text_rank > config.maxTextRank)
  ) {
    errors.push(`${result.sample_dir} generated text rank ${result.generated_text_rank} > ${config.maxTextRank}`);
  }
  if (
    hasRetrievalHead &&
    result.generated_text_margin !== null &&
    result.generated_text_margin < config.minTextMargin
  ) {
    errors.push(
      `${result.sample_dir} generated text margin ${result.generated_text_margin} < ${config.minTextMargin}`,
    );
  }
  if (hasRetrievalHead && result.generated_text_identity?.ok !== true) {
    errors.push(`${result.sample_dir} generated text identity is not true`);
  }
  if (hasRetrievalHead && result.generated_text_image_agree !== true) {
    errors.push(`${result.sample_dir} generated text/image agreement is not true`);
  }
  return errors;
}

function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    const spirits = readTextIndex(config.textIndexPath);
    const retrievalHead = readRetrievalHead(config.retrievalHeadPath);
    const results = [];
    const errors = [];
    if (config.requireRetrievalHead && !retrievalHead) {
      errors.push("--require-retrieval-head was set but --retrieval-head was not supplied");
    }
    for (const sampleDir of config.sampleDirs) {
      const sample = readSample(sampleDir);
      const expected = expectedSpiritForSample(sample, spirits);
      const result = sampleResult(sample, expected, spirits, retrievalHead);
      results.push(result);
      errors.push(...checkResult(result, config, Boolean(retrievalHead)));
    }
    const summary = {
      schema: "nsrl.solomon_attention_sample_binding_check.v1",
      ok: errors.length === 0,
      text_index: config.textIndexPath,
      retrieval_head: config.retrievalHeadPath || null,
      retrieval_head_model_hash: retrievalHead?.model_hash || "",
      retrieval_head_feature_count: Number(retrievalHead?.feature_count || 0),
      samples: results.length,
      min_signature_margin:
        results.length === 0
          ? null
          : Math.min(...results.map((result) => result.signature_margin ?? 0)),
      min_retrieval_image_margin:
        retrievalHead && results.length > 0
          ? Math.min(...results.map((result) => result.retrieval_image_margin ?? 0))
          : null,
      image_to_text_identification:
        retrievalHead && results.length > 0
          ? results.every((result) => result.image_to_text_identity?.ok === true)
          : null,
      min_image_to_text_margin:
        retrievalHead && results.length > 0
          ? Math.min(...results.map((result) => result.image_to_text_identity?.margin ?? 0))
          : null,
      min_retrieval_text_margin:
        retrievalHead && results.length > 0
          ? Math.min(...results.map((result) => result.retrieval_text_margin ?? 0))
          : null,
      generated_text_identification:
        retrievalHead && results.length > 0
          ? results.every((result) => result.generated_text_identity?.ok === true)
          : null,
      min_generated_text_margin:
        retrievalHead && results.length > 0
          ? Math.min(...results.map((result) => result.generated_text_margin ?? 0))
          : null,
      text_image_agreement:
        retrievalHead && results.length > 0
          ? results.every((result) => result.text_image_agree === true)
          : null,
      generated_text_image_agreement:
        retrievalHead && results.length > 0
          ? results.every((result) => result.generated_text_image_agree === true)
          : null,
      signature_retrieval_agreement:
        retrievalHead && results.length > 0
          ? results.every((result) => result.signature_retrieval_agree === true)
          : null,
      results,
      errors: errors.slice(0, config.maxMisses),
    };
    if (config.outPath) {
      fs.mkdirSync(path.dirname(config.outPath), { recursive: true });
      fs.writeFileSync(config.outPath, `${JSON.stringify(summary)}\n`, "utf8");
    }
    console.log(JSON.stringify(summary));
    if (errors.length > 0) {
      console.error(`Solomon attention sample binding check failed with ${errors.length} error(s):`);
      for (const error of errors.slice(0, config.maxMisses)) {
        console.error(`- ${error}`);
      }
      process.exit(1);
    }
  } catch (error) {
    console.error(error.message);
    process.exit(2);
  }
}

main();
