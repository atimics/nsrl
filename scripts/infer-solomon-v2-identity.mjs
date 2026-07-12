#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  textIndexPath: "web/assets/solomon-spirit-text-signatures.tsv",
  retrievalHeadPath: "",
  texts: [],
  imageInk16Paths: [],
  sampleDirs: [],
  outPath: "",
  topK: 5,
  requireSampleAgreement: false,
  requireSourceEvidence: false,
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

const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

function usage() {
  console.log(
    [
      "Usage: infer-solomon-v2-identity.mjs --retrieval-head PATH [inputs...]",
      "",
      "Ranks Solomon v2 text queries, 16x16 seal plans, and generated sample",
      "directories with the sparse integer retrieval head.",
      "",
      "Inputs:",
      "  --text TEXT",
      "  --image-ink16 PATH",
      "  --image PATH                  Alias for --image-ink16",
      "  --sample-dir PATH",
      "",
      "Options:",
      "  --text-index PATH",
      "  --top-k N",
      "  --require-sample-agreement",
      "  --require-source-evidence",
      "  --max-misses N",
      "  --out PATH",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, texts: [], imageInk16Paths: [], sampleDirs: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndexPath = requireValue(argv, ++index, arg);
    } else if (arg === "--text") {
      config.texts.push(requireValue(argv, ++index, arg));
    } else if (arg === "--image-ink16" || arg === "--image") {
      config.imageInk16Paths.push(requireValue(argv, ++index, arg));
    } else if (arg === "--sample-dir") {
      config.sampleDirs.push(requireValue(argv, ++index, arg));
    } else if (arg === "--top-k") {
      config.topK = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-sample-agreement") {
      config.requireSampleAgreement = true;
    } else if (arg === "--require-source-evidence") {
      config.requireSourceEvidence = true;
    } else if (arg === "--max-misses") {
      config.maxMisses = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.retrievalHeadPath) {
    throw new Error("--retrieval-head is required");
  }
  if (config.texts.length + config.imageInk16Paths.length + config.sampleDirs.length === 0) {
    throw new Error("at least one --text, --image-ink16, or --sample-dir input is required");
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

function fnv64FileHex(filePath) {
  let hash = FNV64_OFFSET;
  for (const byte of fs.readFileSync(filePath)) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
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
  const valueAt = (fields, column) => {
    const index = indexOf(column);
    return index >= 0 ? fields[index] || "" : "";
  };
  return lines.filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const number = Number(valueAt(fields, "number"));
    const signature = valueAt(fields, "signature_16x16").split(",").map((part) => Number(part));
    if (!Number.isInteger(number) || number < 1 || number > 72) {
      throw new Error(`${filePath}:${rowIndex + 2} has invalid spirit number`);
    }
    if (signature.length !== BINS || signature.some((value) => !Number.isFinite(value))) {
      throw new Error(`${filePath}:${rowIndex + 2} has invalid 16x16 signature`);
    }
    return {
      label: number - 1,
      spirit_id: number,
      primary_name: valueAt(fields, "primary_name"),
      aliases: String(valueAt(fields, "aliases")).split("|").filter(Boolean),
      slice_id: valueAt(fields, "slice_id"),
      source_file: valueAt(fields, "source_file"),
      ink_128_u8: valueAt(fields, "ink_128_u8"),
      signature,
      source_text: valueAt(fields, "text"),
    };
  });
}

function readRetrievalHead(filePath) {
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

function readSignatureFile(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length !== BINS) {
    throw new Error(`${filePath} has ${bytes.length} bytes, expected ${BINS}`);
  }
  return Array.from(bytes);
}

function readSample(sampleDir) {
  const tracePath = path.join(sampleDir, "sample.json");
  const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  const imagePath = resolveSampleArtifact(sampleDir, trace.image_ink16_u8 || "image.ink16.u8");
  const textPath = path.join(sampleDir, "text.txt");
  const generatedText = fs.existsSync(textPath)
    ? fs.readFileSync(textPath, "utf8").trim()
    : String(trace.generated_text || "").trim();
  return {
    sample_dir: sampleDir,
    trace,
    prompt: trace.prompt || "",
    generated_text: generatedText,
    image_ink16_u8: imagePath,
    signature: readSignatureFile(imagePath),
  };
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

function inferTextQuery(text, model, spiritsById, topK) {
  const ranked = rankRetrievalText(model, text, model.labels.length);
  const top = decorateScoreRanking(ranked, spiritsById, topK);
  return {
    kind: "text",
    text,
    best: top[0] || null,
    top_k: top,
    source_evidence: {
      best_has_source_text: candidateHasSourceEvidence(top[0]),
      best_source_text_chars: sourceTextChars(top[0]),
    },
    confidence: scoreConfidence(ranked),
  };
}

function inferImageQuery(imagePath, signature, model, spirits, spiritsById, topK) {
  const retrievalRanked = rankRetrievalImage(model, signature, model.labels.length);
  const signatureRanked = rankSignature(signature, spirits);
  const retrievalTop = decorateScoreRanking(retrievalRanked, spiritsById, topK);
  const signatureTop = decorateDistanceRanking(signatureRanked, spiritsById, topK);
  return {
    kind: "image_ink16",
    image_ink16_u8: imagePath,
    best_retrieval: retrievalTop[0] || null,
    best_signature: signatureTop[0] || null,
    retrieval_signature_agree:
      retrievalTop[0]?.spirit_id !== undefined &&
      retrievalTop[0]?.spirit_id === signatureTop[0]?.spirit_id,
    retrieval_top_k: retrievalTop,
    signature_top_k: signatureTop,
    source_evidence: {
      best_retrieval_has_source_text: candidateHasSourceEvidence(retrievalTop[0]),
      best_signature_has_source_text: candidateHasSourceEvidence(signatureTop[0]),
      min_best_source_text_chars: Math.min(sourceTextChars(retrievalTop[0]), sourceTextChars(signatureTop[0])),
    },
    confidence: {
      retrieval_margin: scoreConfidence(retrievalRanked).top1_margin,
      signature_margin: distanceConfidence(signatureRanked).top1_margin,
    },
  };
}

function inferSample(sample, model, spirits, spiritsById, topK) {
  const promptRanked = sample.prompt ? rankRetrievalText(model, sample.prompt, model.labels.length) : [];
  const generatedRanked = sample.generated_text
    ? rankRetrievalText(model, sample.generated_text, model.labels.length)
    : [];
  const imageRanked = rankRetrievalImage(model, sample.signature, model.labels.length);
  const signatureRanked = rankSignature(sample.signature, spirits);
  const expected = expectedSpiritForPrompt(sample.prompt, spirits);
  const promptTop = decorateScoreRanking(promptRanked, spiritsById, topK);
  const generatedTop = decorateScoreRanking(generatedRanked, spiritsById, topK);
  const imageTop = decorateScoreRanking(imageRanked, spiritsById, topK);
  const signatureTop = decorateDistanceRanking(signatureRanked, spiritsById, topK);
  const imageBest = imageTop[0] || null;
  const signatureBest = signatureTop[0] || null;
  const promptBest = promptTop[0] || null;
  const generatedBest = generatedTop[0] || null;
  const promptImageAgree = promptBest && imageBest ? promptBest.spirit_id === imageBest.spirit_id : null;
  const generatedTextImageAgree =
    generatedBest && imageBest ? generatedBest.spirit_id === imageBest.spirit_id : null;
  const signatureRetrievalAgree =
    signatureBest && imageBest ? signatureBest.spirit_id === imageBest.spirit_id : null;
  const expectedImageAgree = expected && imageBest ? expected.spirit_id === imageBest.spirit_id : null;
  const expectedPromptAgree = expected && promptBest ? expected.spirit_id === promptBest.spirit_id : null;
  const expectedGeneratedTextAgree =
    expected && generatedBest ? expected.spirit_id === generatedBest.spirit_id : null;
  const textChecks = [promptImageAgree, generatedTextImageAgree].filter((value) => value !== null);
  const textImageAgree =
    textChecks.length > 0 &&
    textChecks.every(Boolean) &&
    signatureRetrievalAgree === true &&
    (expected ? expectedImageAgree === true : true);
  return {
    kind: "sample",
    sample_dir: sample.sample_dir,
    prompt: sample.prompt,
    generated_text: sample.generated_text,
    image_ink16_u8: sample.image_ink16_u8,
    expected: expected ? candidateMetadata(expected) : null,
    best_prompt_text: promptBest,
    best_generated_text: generatedBest,
    best_image_retrieval: imageBest,
    best_signature: signatureBest,
    prompt_retrieval_top_k: promptTop,
    generated_text_retrieval_top_k: generatedTop,
    image_retrieval_top_k: imageTop,
    signature_top_k: signatureTop,
    agreement: {
      text_image_agree: textImageAgree,
      prompt_image_agree: promptImageAgree,
      generated_text_image_agree: generatedTextImageAgree,
      signature_retrieval_agree: signatureRetrievalAgree,
      expected_image_agree: expectedImageAgree,
      expected_prompt_agree: expectedPromptAgree,
      expected_generated_text_agree: expectedGeneratedTextAgree,
    },
    source_evidence: {
      expected_has_source_text: candidateHasSourceEvidence(expected ? candidateMetadata(expected) : null),
      prompt_text_has_source_text: candidateHasSourceEvidence(promptBest),
      generated_text_has_source_text: candidateHasSourceEvidence(generatedBest),
      image_retrieval_has_source_text: candidateHasSourceEvidence(imageBest),
      signature_has_source_text: candidateHasSourceEvidence(signatureBest),
      min_source_text_chars: Math.min(
        sourceTextChars(promptBest),
        sourceTextChars(imageBest),
        sourceTextChars(signatureBest),
        expected ? sourceTextChars(candidateMetadata(expected)) : Number.POSITIVE_INFINITY,
      ),
    },
    confidence: {
      prompt_text_margin: scoreConfidence(promptRanked).top1_margin,
      generated_text_margin: scoreConfidence(generatedRanked).top1_margin,
      image_retrieval_margin: scoreConfidence(imageRanked).top1_margin,
      signature_margin: distanceConfidence(signatureRanked).top1_margin,
    },
  };
}

function expectedSpiritForPrompt(prompt, spirits) {
  const promptKey = normalizeKey(prompt || "");
  const candidates = spirits
    .map((spirit) => ({
      spirit,
      score: spiritPromptScore(spirit, promptKey),
    }))
    .filter((row) => row.score > 0)
    .sort((left, right) => right.score - left.score || left.spirit.spirit_id - right.spirit.spirit_id);
  return candidates[0]?.spirit || null;
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

function decorateScoreRanking(ranked, spiritsById, topK) {
  return ranked.slice(0, topK).map((row, index) => ({
    rank: index + 1,
    label: row.label,
    spirit_id: row.spirit_id,
    primary_name: row.primary_name,
    score: row.score,
    margin_to_next: ranked[index + 1] ? row.score - ranked[index + 1].score : null,
    ...candidateMetadata(spiritsById.get(row.spirit_id) || row),
  }));
}

function decorateDistanceRanking(ranked, spiritsById, topK) {
  return ranked.slice(0, topK).map((row, index) => ({
    rank: index + 1,
    label: row.label,
    spirit_id: row.spirit_id,
    primary_name: row.primary_name,
    distance: row.distance,
    margin_to_next: ranked[index + 1] ? ranked[index + 1].distance - row.distance : null,
    ...candidateMetadata(spiritsById.get(row.spirit_id) || row),
  }));
}

function candidateMetadata(spirit) {
  return {
    label: spirit.label,
    spirit_id: spirit.spirit_id,
    primary_name: spirit.primary_name,
    aliases: spirit.aliases || [],
    slice_id: spirit.slice_id || "",
    source_file: spirit.source_file || "",
    ink_128_u8: spirit.ink_128_u8 || "",
    source_text_excerpt: excerpt(spirit.source_text || ""),
    source_text_chars: String(spirit.source_text || "").length,
  };
}

function candidateHasSourceEvidence(candidate) {
  return sourceTextChars(candidate) > 0 && String(candidate?.source_text_excerpt || "").trim().length > 0;
}

function sourceTextChars(candidate) {
  const value = Number(candidate?.source_text_chars || 0);
  return Number.isFinite(value) ? value : 0;
}

function excerpt(text, limit = 320) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (compact.length <= limit) {
    return compact;
  }
  return `${compact.slice(0, limit - 3)}...`;
}

function scoreConfidence(ranked) {
  if (ranked.length < 2) {
    return { top1_margin: null };
  }
  return { top1_margin: ranked[0].score - ranked[1].score };
}

function distanceConfidence(ranked) {
  if (ranked.length < 2) {
    return { top1_margin: null };
  }
  return { top1_margin: ranked[1].distance - ranked[0].distance };
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

function scoreLabel(head, label, features) {
  let score = head.biases[label] || 0;
  const weights = head.weights[label] || new Map();
  for (const [feature, value] of features) {
    score += (weights.get(feature) || 0) * value;
  }
  return score;
}

function rankSignature(signature, spirits) {
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

function sampleSummary(samples) {
  if (samples.length === 0) {
    return {
      samples: 0,
      text_image_agreement: null,
      prompt_image_agreement: null,
      generated_text_image_agreement: null,
      signature_retrieval_agreement: null,
      expected_image_agreement: null,
      expected_generated_text_agreement: null,
      source_text_evidence: null,
      generated_text_source_evidence: null,
      min_source_text_chars: null,
      min_prompt_text_margin: null,
      min_generated_text_margin: null,
      min_image_retrieval_margin: null,
      min_signature_margin: null,
    };
  }
  return {
    samples: samples.length,
    text_image_agreement: samples.every((sample) => sample.agreement.text_image_agree === true),
    prompt_image_agreement: samples.every((sample) => sample.agreement.prompt_image_agree === true),
    generated_text_image_agreement: samples.every(
      (sample) => sample.agreement.generated_text_image_agree === true,
    ),
    signature_retrieval_agreement: samples.every(
      (sample) => sample.agreement.signature_retrieval_agree === true,
    ),
    expected_image_agreement: samples.every((sample) => sample.agreement.expected_image_agree === true),
    expected_generated_text_agreement: samples.every(
      (sample) => sample.agreement.expected_generated_text_agree === true,
    ),
    source_text_evidence: samples.every(
      (sample) =>
        sample.source_evidence.prompt_text_has_source_text === true &&
        sample.source_evidence.generated_text_has_source_text === true &&
        sample.source_evidence.image_retrieval_has_source_text === true &&
        sample.source_evidence.signature_has_source_text === true &&
        sample.source_evidence.expected_has_source_text === true,
    ),
    generated_text_source_evidence: samples.every(
      (sample) => sample.source_evidence.generated_text_has_source_text === true,
    ),
    min_source_text_chars: Math.min(...samples.map((sample) => sample.source_evidence.min_source_text_chars ?? 0)),
    min_prompt_text_margin: Math.min(
      ...samples.map((sample) => sample.confidence.prompt_text_margin ?? 0),
    ),
    min_generated_text_margin: Math.min(
      ...samples.map((sample) => sample.confidence.generated_text_margin ?? 0),
    ),
    min_image_retrieval_margin: Math.min(
      ...samples.map((sample) => sample.confidence.image_retrieval_margin ?? 0),
    ),
    min_signature_margin: Math.min(...samples.map((sample) => sample.confidence.signature_margin ?? 0)),
  };
}

function checkSamples(samples, config) {
  const errors = [];
  if (!config.requireSampleAgreement && !config.requireSourceEvidence) {
    return errors;
  }
  for (const sample of samples) {
    if (config.requireSampleAgreement && !sample.expected) {
      errors.push(`${sample.sample_dir} prompt does not name a known spirit`);
    }
    if (config.requireSampleAgreement && sample.agreement.text_image_agree !== true) {
      errors.push(`${sample.sample_dir} text/image agreement is not true`);
    }
    if (config.requireSampleAgreement && sample.agreement.generated_text_image_agree !== true) {
      errors.push(`${sample.sample_dir} generated text/image agreement is not true`);
    }
    if (config.requireSampleAgreement && sample.agreement.signature_retrieval_agree !== true) {
      errors.push(`${sample.sample_dir} signature/retrieval agreement is not true`);
    }
    if (config.requireSampleAgreement && sample.agreement.expected_image_agree !== true) {
      errors.push(`${sample.sample_dir} expected/image agreement is not true`);
    }
    if (config.requireSampleAgreement && sample.agreement.expected_generated_text_agree !== true) {
      errors.push(`${sample.sample_dir} expected/generated text agreement is not true`);
    }
    if (config.requireSourceEvidence) {
      if (sample.source_evidence.prompt_text_has_source_text !== true) {
        errors.push(`${sample.sample_dir} prompt text candidate is missing source text evidence`);
      }
      if (sample.source_evidence.generated_text_has_source_text !== true) {
        errors.push(`${sample.sample_dir} generated text candidate is missing source text evidence`);
      }
      if (sample.source_evidence.image_retrieval_has_source_text !== true) {
        errors.push(`${sample.sample_dir} image retrieval candidate is missing source text evidence`);
      }
      if (sample.source_evidence.signature_has_source_text !== true) {
        errors.push(`${sample.sample_dir} signature candidate is missing source text evidence`);
      }
      if (sample.source_evidence.expected_has_source_text !== true) {
        errors.push(`${sample.sample_dir} expected candidate is missing source text evidence`);
      }
    }
  }
  return errors;
}

function sourceSummary(textQueries, imageQueries, sampleQueries) {
  return {
    text_queries_have_source_text: textQueries.every((query) => query.source_evidence.best_has_source_text === true),
    image_queries_have_source_text: imageQueries.every(
      (query) =>
        query.source_evidence.best_retrieval_has_source_text === true &&
        query.source_evidence.best_signature_has_source_text === true,
    ),
    sample_queries_have_source_text: sampleQueries.every(
      (query) =>
        query.source_evidence.image_retrieval_has_source_text === true &&
        query.source_evidence.signature_has_source_text === true &&
        query.source_evidence.expected_has_source_text === true,
    ),
    min_text_query_source_chars: minOrNull(textQueries.map((query) => query.source_evidence.best_source_text_chars)),
    min_image_query_source_chars: minOrNull(
      imageQueries.map((query) => query.source_evidence.min_best_source_text_chars),
    ),
    min_sample_source_chars: minOrNull(sampleQueries.map((query) => query.source_evidence.min_source_text_chars)),
  };
}

function minOrNull(values) {
  const finite = values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  return finite.length === 0 ? null : Math.min(...finite);
}

function writeJson(filePath, row) {
  const dir = path.dirname(filePath);
  if (dir && dir !== ".") {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const spirits = readTextIndex(config.textIndexPath);
  const spiritsById = new Map(spirits.map((spirit) => [spirit.spirit_id, spirit]));
  const model = readRetrievalHead(config.retrievalHeadPath);
  const textQueries = config.texts.map((text) => inferTextQuery(text, model, spiritsById, config.topK));
  const imageQueries = config.imageInk16Paths.map((imagePath) =>
    inferImageQuery(
      imagePath,
      readSignatureFile(imagePath),
      model,
      spirits,
      spiritsById,
      config.topK,
    ),
  );
  const sampleQueries = config.sampleDirs.map((sampleDir) =>
    inferSample(readSample(sampleDir), model, spirits, spiritsById, config.topK),
  );
  const errors = checkSamples(sampleQueries, config);
  const summary = sampleSummary(sampleQueries);
  const sources = sourceSummary(textQueries, imageQueries, sampleQueries);
  const result = {
    schema: "nsrl.solomon_v2_identity_inference.v1",
    ok: errors.length === 0,
    text_index: config.textIndexPath,
    text_index_hash: fnv64FileHex(config.textIndexPath),
    retrieval_head: config.retrievalHeadPath,
    model_hash: model.model_hash || "",
    feature_count: Number(model.feature_count || 0),
    labels: Array.isArray(model.labels) ? model.labels.length : 0,
    top_k: config.topK,
    require_sample_agreement: config.requireSampleAgreement,
    require_source_evidence: config.requireSourceEvidence,
    query_count: textQueries.length + imageQueries.length + sampleQueries.length,
    text_queries: textQueries,
    image_queries: imageQueries,
    sample_queries: sampleQueries,
    source_summary: sources,
    sample_summary: summary,
    errors: errors.slice(0, config.maxMisses),
  };
  if (config.outPath) {
    writeJson(config.outPath, result);
  }
  console.log(JSON.stringify(result, null, 2));
  if (errors.length > 0) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
