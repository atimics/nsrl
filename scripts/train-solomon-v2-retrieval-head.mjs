#!/usr/bin/env node

import fs from "node:fs";

const defaults = {
  examplesPath: "",
  tokensPath: "",
  textIndexPath: "web/assets/solomon-spirit-text-signatures.tsv",
  promptsPath: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  modelOut: "",
  evalOut: "",
  featureCount: 16384,
  epochs: 16,
  seed: "solomon-v2-retrieval-head-v1",
  leadingAnchorBoost: 1_000_000_000,
  mentionAnchorBoost: 100_000_000,
  minKnownTop1: 72,
  minKnownTop5: 72,
  minHeldoutTop1PerMille: 1000,
  minHeldoutTop5PerMille: 1000,
  requireHeldoutPrompts: false,
  minHeldoutPromptRows: 0,
  minImageTop1: 72,
  minMatchYesTop1: 72,
  minMatchNoTop1: 72,
  minMatchNoImageTop1: 72,
  minMatchNoPromptTop1: 72,
  minRetrievalMargin: 1,
  maxMisses: 8,
};

const BOS = 1;
const PROMPT = 2;
const IMAGE = 4;
const EOS = 5;
const TEXT = 3;
const TASK_IDENTIFY = 10;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
const IMAGE_BASE = 144;
const IMAGE_BINS = 16;
const REVERSE_IMAGE_RETRIEVAL_TASKS = [
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
];
const FORWARD_IMAGE_PLAN_TASKS = [
  "text-to-image",
  "description-to-image",
];
const IMAGE_RETRIEVAL_TASKS = [
  ...FORWARD_IMAGE_PLAN_TASKS,
  ...REVERSE_IMAGE_RETRIEVAL_TASKS,
];
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
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
      "Usage: train-solomon-v2-retrieval-head.mjs --examples PATH --tokens PATH [options]",
      "",
      "Trains a tiny integer 72-way retrieval head over v2 Solomon text and",
      "symbolic image-token records, then gates known prompts, held-out prompts,",
      "image-to-text/source identity, wrong-seal matches, and wrong-prompt/name matches.",
      "",
      "Options:",
      "  --text-index PATH",
      "  --prompts PATH|none",
      "  --model-out PATH",
      "  --eval-out PATH",
      "  --feature-count N",
      "  --epochs N",
      "  --seed TEXT",
      "  --min-known-top1 N",
      "  --min-known-top5 N",
      "  --min-heldout-top1-per-mille N",
      "  --min-heldout-top5-per-mille N",
      "  --require-heldout-prompts",
      "  --min-heldout-prompt-rows N",
      "  --min-image-top1 N",
      "  --min-match-yes-top1 N",
      "  --min-match-no-top1 N",
      "  --min-match-no-image-top1 N",
      "  --min-match-no-prompt-top1 N",
      "  --min-retrieval-margin N",
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
    } else if (arg === "--examples") {
      config.examplesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--tokens") {
      config.tokensPath = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndexPath = requireValue(argv, ++index, arg);
    } else if (arg === "--prompts") {
      config.promptsPath = requireValue(argv, ++index, arg);
    } else if (arg === "--model-out") {
      config.modelOut = requireValue(argv, ++index, arg);
    } else if (arg === "--eval-out") {
      config.evalOut = requireValue(argv, ++index, arg);
    } else if (arg === "--feature-count") {
      config.featureCount = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--epochs") {
      config.epochs = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--seed") {
      config.seed = requireValue(argv, ++index, arg);
    } else if (arg === "--min-known-top1") {
      config.minKnownTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-known-top5") {
      config.minKnownTop5 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heldout-top1-per-mille") {
      config.minHeldoutTop1PerMille = parseRatePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heldout-top5-per-mille") {
      config.minHeldoutTop5PerMille = parseRatePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-heldout-prompts") {
      config.requireHeldoutPrompts = true;
    } else if (arg === "--min-heldout-prompt-rows") {
      config.minHeldoutPromptRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-image-top1") {
      config.minImageTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-yes-top1") {
      config.minMatchYesTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-top1") {
      config.minMatchNoTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-image-top1") {
      config.minMatchNoImageTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-prompt-top1") {
      config.minMatchNoPromptTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-retrieval-margin") {
      config.minRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.examplesPath) {
    throw new Error("--examples is required");
  }
  if (!config.tokensPath) {
    throw new Error("--tokens is required");
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

function parseRatePerMille(value, flag) {
  if (/^[0-9]+$/.test(value)) {
    const parsed = Number(value);
    if (parsed > 1000) {
      throw new Error(`${flag} must be <= 1000`);
    }
    return parsed;
  }
  if (/^(?:0(?:\.[0-9]+)?|1(?:\.0+)?)$/.test(value)) {
    return Math.round(Number(value) * 1000);
  }
  throw new Error(`${flag} requires a per-mille integer or 0.0-1.0 decimal`);
}

function readJsonl(path) {
  const text = fs.readFileSync(path, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  return text.split(/\r?\n/).filter(Boolean).map((line, rowIndex) => {
    const row = JSON.parse(line);
    row.__line = rowIndex + 1;
    return row;
  });
}

function readTextIndex(path) {
  const lines = fs.readFileSync(path, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  for (const column of ["number", "primary_name", "aliases", "text"]) {
    if (!header.includes(column)) {
      throw new Error(`${path} is missing required column ${column}`);
    }
  }
  const indexOf = (column) => header.indexOf(column);
  return lines.filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const number = Number(fields[indexOf("number")]);
    if (!Number.isInteger(number) || number < 1 || number > 72) {
      throw new Error(`${path}:${rowIndex + 2} has invalid spirit number`);
    }
    return {
      number,
      label: number - 1,
      primary_name: fields[indexOf("primary_name")] || "",
      aliases: String(fields[indexOf("aliases")] || "").split("|").filter(Boolean),
      text: fields[indexOf("text")] || "",
    };
  });
}

function readPromptRows(path) {
  if (!path || path === "none" || !fs.existsSync(path)) {
    return { rows: [], present: false, total_rows: 0, tier_counts: {}, source_counts: {}, unique_targets: 0 };
  }
  const allRows = readJsonl(path)
    .map((row) => ({
      spirit_id: normalizedId(row.spirit_id),
      label: normalizedId(row.spirit_id) === null ? null : normalizedId(row.spirit_id) - 1,
      text: String(row.text || row.prompt || ""),
      source: row.source || "prompt",
      tier: row.tier || "",
    }))
    .filter((row) => row.spirit_id !== null && row.label >= 0 && row.text);
  const rows = allRows.filter(isHeldoutPromptRow);
  return {
    rows,
    present: true,
    total_rows: allRows.length,
    tier_counts: countBy(allRows, (row) => row.tier || "unknown"),
    source_counts: countBy(allRows, (row) => row.source || "unknown"),
    evaluated_tier_counts: countBy(rows, (row) => row.tier || "unknown"),
    evaluated_source_counts: countBy(rows, (row) => row.source || "unknown"),
    unique_targets: new Set(rows.map((row) => row.spirit_id)).size,
  };
}

function isHeldoutPromptRow(row) {
  const tier = String(row.tier || "").toLowerCase();
  const source = String(row.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function countBy(rows, keyFn) {
  const out = {};
  for (const row of rows) {
    const key = String(keyFn(row) || "unknown");
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}

function buildTrainingRows({ spirits, examples, tokens, featureCount }) {
  const textRows = [];
  const imageRows = [];
  for (const spirit of spirits) {
    const names = [spirit.primary_name, ...spirit.aliases].filter(Boolean);
    for (const name of names) {
      textRows.push(textTrainRow(spirit.label, name, 64, featureCount));
      textRows.push(textTrainRow(spirit.label, `${name} ${name}`, 48, featureCount));
      textRows.push(textTrainRow(spirit.label, `seal of ${name}`, 48, featureCount));
      textRows.push(textTrainRow(spirit.label, `${name} goetic seal`, 40, featureCount));
    }
    textRows.push(textTrainRow(spirit.label, `${spirit.primary_name} ${spirit.text}`, 1, featureCount));
  }
  for (const row of examples) {
    const spiritId = normalizedId(row.spirit_id);
    if (spiritId === null) {
      continue;
    }
    const label = spiritId - 1;
    const matchLabel = String(row.match_label || row.text || "").toLowerCase();
    const negativeRole = matchNegativeRole(row);
    const negativeSpiritId = normalizedId(row.negative_spirit_id);
    if (row.prompt) {
      const promptLabel =
        row.task === "match" && matchLabel === "no" && negativeRole === "prompt" && negativeSpiritId !== null
          ? negativeSpiritId - 1
          : label;
      textRows.push(textTrainRow(promptLabel, row.prompt, row.task === "identify" ? 36 : 4, featureCount));
    }
    if (row.text && row.task !== "match") {
      const textWeight = ["explain", "image-to-explain", "text-image-explain", "image-to-attributes"].includes(row.task) ? 2 : 1;
      textRows.push(textTrainRow(label, row.text, textWeight, featureCount));
    }
    const image = imageVectorForRow(row, tokens);
    if (!image) {
      continue;
    }
    if (row.task === "match" && matchLabel === "no") {
      if (negativeRole === "prompt") {
        imageRows.push(imageTrainRow(label, image, 5, featureCount));
      } else if (negativeSpiritId !== null) {
        imageRows.push(imageTrainRow(negativeSpiritId - 1, image, 5, featureCount));
      }
    } else if ([...IMAGE_RETRIEVAL_TASKS, "match"].includes(row.task)) {
      imageRows.push(imageTrainRow(label, image, row.task === "match" ? 4 : 6, featureCount));
    }
  }
  return { textRows, imageRows };
}

function textTrainRow(label, text, weight, featureCount) {
  return {
    label,
    weight,
    features: textFeatures(text, featureCount),
  };
}

function imageTrainRow(label, image, weight, featureCount) {
  return {
    label,
    weight,
    features: imageFeatures(image, featureCount),
  };
}

function trainHead(rowCount, rows, config, headSeed) {
  const weights = Array.from({ length: rowCount }, () => new Map());
  const biases = new Int32Array(rowCount);
  let mistakes = 0;
  for (let epoch = 0; epoch < config.epochs; epoch += 1) {
    const order = shuffledIndices(rows.length, `${config.seed}:${headSeed}:${epoch}`);
    for (const rowIndex of order) {
      const row = rows[rowIndex];
      const prediction = predictLabel({ weights, biases }, row.features).label;
      if (prediction === row.label) {
        biases[row.label] += row.weight;
        continue;
      }
      mistakes += 1;
      updateWeights(weights[row.label], row.features, row.weight);
      updateWeights(weights[prediction], row.features, -row.weight);
      biases[row.label] += row.weight * 2;
      biases[prediction] -= row.weight * 2;
    }
  }
  return { weights, biases, mistakes };
}

function predictLabel(head, features) {
  return rankHead(head, features, 2)[0];
}

function rankHead(head, features, count = 5) {
  const ranked = [];
  for (let label = 0; label < head.weights.length; label += 1) {
    ranked.push({ label, score: scoreLabel(head, label, features) });
  }
  ranked.sort((left, right) => right.score - left.score || left.label - right.label);
  return ranked.slice(0, count);
}

function rankTextHead(head, spirits, text, config, count = 5) {
  const features = textFeatures(text, config.featureCount);
  const queryKey = normalizeKey(text);
  const ranked = [];
  for (let label = 0; label < head.weights.length; label += 1) {
    const spirit = spirits[label];
    ranked.push({
      label,
      score: scoreLabel(head, label, features) + identityAnchorScore(spirit, queryKey, config),
    });
  }
  ranked.sort((left, right) => right.score - left.score || left.label - right.label);
  return ranked.slice(0, count);
}

function identityAnchorScore(spirit, queryKey, config) {
  if (!spirit || !queryKey) {
    return 0;
  }
  let score = 0;
  for (const key of identityAnchorKeys(spirit)) {
    if (startsWithPhrase(queryKey, key)) {
      score = Math.max(score, config.leadingAnchorBoost + key.length * 1000);
    } else if (containsPhrase(queryKey, key)) {
      score = Math.max(score, config.mentionAnchorBoost + key.length * 100);
    }
  }
  return score;
}

function identityAnchorKeys(spirit) {
  const keys = new Set();
  const add = (value) => {
    const key = normalizeKey(value);
    if (key) {
      keys.add(key);
    }
  };
  for (const name of [spirit.primary_name, ...spirit.aliases]) {
    add(name);
  }
  const spiritId = normalizedId(spirit.spirit_id ?? spirit.number);
  if (spiritId !== null) {
    add(`seal id ${spiritId}`);
    add(`spirit ${spiritId}`);
    add(`goetic spirit ${spiritId}`);
  }
  return keys;
}

function startsWithPhrase(haystack, needle) {
  return haystack === needle || haystack.startsWith(`${needle} `);
}

function containsPhrase(haystack, needle) {
  return ` ${haystack} `.includes(` ${needle} `);
}

function scoreLabel(head, label, features) {
  let score = head.biases[label] || 0;
  const weights = head.weights[label];
  for (const [feature, value] of features) {
    score += (weights.get(feature) || 0) * value;
  }
  return score;
}

function updateWeights(weights, features, direction) {
  for (const [feature, value] of features) {
    const next = (weights.get(feature) || 0) + value * direction;
    if (next === 0) {
      weights.delete(feature);
    } else {
      weights.set(feature, clampInt32(next));
    }
  }
}

function clampInt32(value) {
  return Math.max(-2147483648, Math.min(2147483647, value));
}

function textFeatures(text, featureCount) {
  const tokens = tokenize(text);
  const out = new Map();
  if (tokens.length === 0) {
    return [];
  }
  if (tokens.length <= 8) {
    addHashedFeature(out, featureCount, "whole", tokens.join(" "), 96);
  }
  addHashedFeature(out, featureCount, "lead1", tokens[0], 72);
  if (tokens[1]) {
    addHashedFeature(out, featureCount, "lead2", `${tokens[0]} ${tokens[1]}`, 88);
  }
  const content = tokens.filter((token) => token.length >= 3 && !STOPWORDS.has(token));
  if (content.length > 0 && content.length <= 8) {
    addHashedFeature(out, featureCount, "content", content.join(" "), 84);
    addHashedFeature(out, featureCount, "cset", [...content].sort().join(" "), 80);
  }
  for (let index = 0; index < tokens.length; index += 1) {
    addHashedFeature(out, featureCount, "tok", tokens[index], 12 + (index < 4 ? 8 : 0));
    if (tokens[index + 1]) {
      addHashedFeature(out, featureCount, "bi", `${tokens[index]} ${tokens[index + 1]}`, 18);
    }
    if (tokens[index + 1] && tokens[index + 2]) {
      addHashedFeature(
        out,
        featureCount,
        "tri",
        `${tokens[index]} ${tokens[index + 1]} ${tokens[index + 2]}`,
        24,
      );
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

function addHashedFeature(out, featureCount, namespace, value, amount) {
  if (value === undefined || value === null || String(value).length === 0) {
    return;
  }
  const hash = fnv32(`${namespace}\xff${value}`);
  const index = hash % featureCount;
  const sign = hash & 0x80000000 ? -1 : 1;
  out.set(index, clampFeatureValue((out.get(index) || 0) + sign * amount));
}

function clampFeatureValue(value) {
  return Math.max(-127, Math.min(127, value));
}

function imageVectorForRow(row, tokens) {
  const offset = normalizedId(row.token_offset);
  const count = normalizedId(row.token_count);
  if (offset === null || count === null) {
    return null;
  }
  const slice = tokens.subarray(offset, offset + count);
  if (slice[0] !== BOS) {
    return null;
  }
  const imageIndex = slice.indexOf(IMAGE);
  if (imageIndex < 0) {
    return null;
  }
  let end = slice.length;
  for (let index = imageIndex + 1; index < slice.length; index += 1) {
    if (slice[index] === PROMPT || slice[index] === TEXT || slice[index] === EOS) {
      end = index;
      break;
    }
  }
  const image = Array.from(slice.subarray(imageIndex + 1, end));
  return image.length > 0 ? image : null;
}

function knownPromptQueries(examples) {
  const seen = new Set();
  const out = [];
  for (const row of examples) {
    if (row.task !== "identify" || !row.prompt) {
      continue;
    }
    const spiritId = normalizedId(row.spirit_id);
    if (spiritId === null) {
      continue;
    }
    const key = `${spiritId}:${normalizeKey(row.prompt)}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push({
      label: spiritId - 1,
      spirit_id: spiritId,
      text: row.prompt,
      identity_binding: row.identity_binding === true || row.identity_binding === "true",
      binding_kind: String(row.binding_kind || ""),
    });
  }
  return out;
}

function evaluateText(head, spirits, queries, config, maxMisses) {
  let top1 = 0;
  let top5 = 0;
  let rankSum = 0;
  let marginSum = 0;
  let marginCount = 0;
  let minMargin = null;
  const misses = [];
  for (const query of queries) {
    const ranked = rankTextHead(head, spirits, query.text, config, spirits.length);
    const rank = ranked.findIndex((row) => row.label === query.label) + 1;
    const margin = targetMargin(ranked, query.label);
    marginSum += margin;
    marginCount += 1;
    minMargin = minMargin === null ? margin : Math.min(minMargin, margin);
    rankSum += rank || spirits.length + 1;
    if (rank === 1) {
      top1 += 1;
    }
    if (rank > 0 && rank <= 5) {
      top5 += 1;
    }
    if (rank !== 1 && misses.length < maxMisses) {
      misses.push({
        spirit_id: query.spirit_id,
        text: query.text,
        binding_kind: query.binding_kind || null,
        identity_binding: query.identity_binding === true,
        rank,
        margin,
        top: ranked.slice(0, 5).map((item) => predictionJson(item, spirits)),
      });
    }
  }
  return evalSummary(queries.length, top1, top5, rankSum, misses, marginSum, marginCount, minMargin);
}

function evaluateIdentityBindings(head, spirits, queries, config, maxMisses) {
  const identityQueries = queries.filter((query) => query.identity_binding);
  const byKind = {};
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    byKind[kind] = evaluateText(
      head,
      spirits,
      identityQueries.filter((query) => query.binding_kind === kind),
      config,
      maxMisses,
    );
  }
  return {
    required_kinds: REQUIRED_IDENTITY_BINDING_KINDS,
    total: evaluateText(head, spirits, identityQueries, config, maxMisses),
    by_kind: byKind,
  };
}

function evaluateImage(head, spirits, examples, tokens, featureCount, maxMisses) {
  const rows = examples.filter((row) => REVERSE_IMAGE_RETRIEVAL_TASKS.includes(row.task));
  return evaluateImageRows(head, spirits, rows, tokens, featureCount, maxMisses);
}

function evaluateImageTasks(head, spirits, examples, tokens, featureCount, maxMisses) {
  return Object.fromEntries(
    IMAGE_RETRIEVAL_TASKS.map((task) => [
      task,
      evaluateImageRows(
        head,
        spirits,
        examples.filter((row) => row.task === task),
        tokens,
        featureCount,
        maxMisses,
      ),
    ]),
  );
}

function evaluateImageRows(head, spirits, rows, tokens, featureCount, maxMisses) {
  let top1 = 0;
  let top5 = 0;
  let rankSum = 0;
  let marginSum = 0;
  let marginCount = 0;
  let minMargin = null;
  const misses = [];
  for (const row of rows) {
    const spiritId = normalizedId(row.spirit_id);
    const image = imageVectorForRow(row, tokens);
    if (spiritId === null || !image) {
      continue;
    }
    const target = spiritId - 1;
    const ranked = rankHead(head, imageFeatures(image, featureCount), spirits.length);
    const rank = ranked.findIndex((item) => item.label === target) + 1;
    const margin = targetMargin(ranked, target);
    marginSum += margin;
    marginCount += 1;
    minMargin = minMargin === null ? margin : Math.min(minMargin, margin);
    rankSum += rank || spirits.length + 1;
    if (rank === 1) {
      top1 += 1;
    }
    if (rank > 0 && rank <= 5) {
      top5 += 1;
    }
    if (rank !== 1 && misses.length < maxMisses) {
      misses.push({
        spirit_id: spiritId,
        primary_name: row.primary_name,
        rank,
        margin,
        top: ranked.slice(0, 5).map((item) => predictionJson(item, spirits)),
      });
    }
  }
  return evalSummary(rows.length, top1, top5, rankSum, misses, marginSum, marginCount, minMargin);
}

function evaluateMatch(textHead, imageHead, spirits, examples, tokens, config, maxMisses) {
  let yesCount = 0;
  let yesTop1 = 0;
  let noCount = 0;
  let noTop1 = 0;
  const noByRole = {
    image: matchAccumulator(),
    prompt: matchAccumulator(),
  };
  const yes = matchAccumulator();
  const no = matchAccumulator();
  const misses = [];
  for (const row of examples.filter((item) => item.task === "match")) {
    const spiritId = normalizedId(row.spirit_id);
    const image = imageVectorForRow(row, tokens);
    if (spiritId === null || !image) {
      continue;
    }
    const textRanked = rankTextHead(textHead, spirits, row.prompt || "", config, spirits.length);
    const imageRanked = rankHead(imageHead, imageFeatures(image, config.featureCount), spirits.length);
    const textPrediction = textRanked[0];
    const imagePrediction = imageRanked[0];
    const label = String(row.match_label || row.text || "").toLowerCase();
    const negativeRole = matchNegativeRole(row);
    if (label === "yes") {
      yesCount += 1;
      const textTarget = spiritId - 1;
      const imageTarget = spiritId - 1;
      const margin = Math.min(targetMargin(textRanked, textTarget), targetMargin(imageRanked, imageTarget));
      recordMatchMargin(yes, margin);
      const ok = textPrediction.label === textTarget && imagePrediction.label === imageTarget;
      if (ok) {
        yesTop1 += 1;
        yes.top1 += 1;
      } else if (misses.length < maxMisses) {
        misses.push(matchMiss(row, spirits, textPrediction, imagePrediction, margin));
      }
    } else if (label === "no") {
      noCount += 1;
      const negativeSpiritId = normalizedId(row.negative_spirit_id);
      const textTarget = negativeRole === "prompt" && negativeSpiritId !== null ? negativeSpiritId - 1 : spiritId - 1;
      const imageTarget = negativeRole === "prompt" ? spiritId - 1 : negativeSpiritId === null ? null : negativeSpiritId - 1;
      const margin =
        imageTarget === null ? Number.MIN_SAFE_INTEGER : Math.min(targetMargin(textRanked, textTarget), targetMargin(imageRanked, imageTarget));
      recordMatchMargin(no, margin);
      recordMatchMargin(noByRole[negativeRole], margin);
      const ok =
        negativeRole === "prompt"
          ? negativeSpiritId !== null &&
            textPrediction.label === textTarget &&
            imagePrediction.label === imageTarget &&
            textPrediction.label !== imagePrediction.label
          : negativeSpiritId !== null &&
            textPrediction.label === textTarget &&
            imagePrediction.label === imageTarget &&
            textPrediction.label !== imagePrediction.label;
      if (ok) {
        noTop1 += 1;
        noByRole[negativeRole].top1 += 1;
        no.top1 += 1;
      } else if (misses.length < maxMisses) {
        misses.push(matchMiss(row, spirits, textPrediction, imagePrediction, margin));
      }
    }
  }
  return {
    yes: matchRoleSummary({ ...yes, count: yesCount, top1: yesTop1 }),
    no: matchRoleSummary({ ...no, count: noCount, top1: noTop1 }),
    no_by_role: {
      image: matchRoleSummary(noByRole.image),
      prompt: matchRoleSummary(noByRole.prompt),
    },
    misses,
  };
}

function matchAccumulator() {
  return { count: 0, top1: 0, margin_sum: 0, margin_count: 0, min_margin: null };
}

function recordMatchMargin(role, margin) {
  role.count += 1;
  role.margin_sum += margin;
  role.margin_count += 1;
  role.min_margin = role.min_margin === null ? margin : Math.min(role.min_margin, margin);
}

function matchRoleSummary(role) {
  return {
    count: role.count,
    top1: role.top1,
    top1_per_mille: role.count === 0 ? null : Math.round((role.top1 * 1000) / role.count),
    min_margin: role.margin_count === 0 ? null : role.min_margin,
    mean_margin: role.margin_count === 0 ? null : Math.round(role.margin_sum / role.margin_count),
  };
}

function matchMiss(row, spirits, textPrediction, imagePrediction, margin) {
  const miss = {
    label: row.match_label || row.text,
    spirit_id: row.spirit_id,
    margin,
    text_prediction: predictionJson(textPrediction, spirits),
    image_prediction: predictionJson(imagePrediction, spirits),
  };
  if (String(row.match_label || row.text || "").toLowerCase() === "no") {
    miss.negative_role = matchNegativeRole(row);
    miss.negative_spirit_id = row.negative_spirit_id;
  }
  return miss;
}

function matchNegativeRole(row) {
  const role = String(row.negative_role || "image").toLowerCase();
  return role === "prompt" || role === "text" || role === "name" ? "prompt" : "image";
}

function predictionJson(prediction, spirits) {
  const spirit = spirits[prediction.label];
  return {
    spirit_id: spirit?.number ?? prediction.label + 1,
    primary_name: spirit?.primary_name ?? "",
    score: prediction.score,
  };
}

function targetMargin(ranked, targetLabel) {
  const target = ranked.find((item) => item.label === targetLabel);
  if (!target) {
    return Number.MIN_SAFE_INTEGER;
  }
  const bestWrong = ranked.find((item) => item.label !== targetLabel);
  if (!bestWrong) {
    return target.score;
  }
  return target.score - bestWrong.score;
}

function evalSummary(count, top1, top5, rankSum, misses, marginSum = 0, marginCount = 0, minMargin = null) {
  return {
    count,
    top1,
    top5,
    top1_per_mille: count === 0 ? null : Math.round((top1 * 1000) / count),
    top5_per_mille: count === 0 ? null : Math.round((top5 * 1000) / count),
    mean_rank_per_mille: count === 0 ? null : Math.round((rankSum * 1000) / count),
    min_margin: marginCount === 0 ? null : minMargin,
    mean_margin: marginCount === 0 ? null : Math.round(marginSum / marginCount),
    misses,
  };
}

function checkThresholds(config, evalTrace) {
  const errors = [];
  if (evalTrace.known_prompts.top1 < config.minKnownTop1) {
    errors.push(`known prompt top1 ${evalTrace.known_prompts.top1} < ${config.minKnownTop1}`);
  }
  if (evalTrace.known_prompts.top5 < config.minKnownTop5) {
    errors.push(`known prompt top5 ${evalTrace.known_prompts.top5} < ${config.minKnownTop5}`);
  }
  requireMinMargin(errors, evalTrace.known_prompts, "known prompt", config.minRetrievalMargin);
  const identityTotal = evalTrace.identity_bindings?.total;
  if (!identityTotal || identityTotal.count <= 0) {
    errors.push("identity binding retrieval has no rows");
  } else if (identityTotal.top1 !== identityTotal.count) {
    errors.push(`identity binding retrieval top1 ${identityTotal.top1} != count ${identityTotal.count}`);
  }
  requireMinMargin(errors, identityTotal, "identity binding retrieval", config.minRetrievalMargin);
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    const metric = evalTrace.identity_bindings?.by_kind?.[kind];
    if (!metric || metric.count <= 0) {
      errors.push(`identity binding kind ${kind} has no rows`);
    } else if (metric.top1 !== metric.count) {
      errors.push(`identity binding kind ${kind} top1 ${metric.top1} != count ${metric.count}`);
    }
    requireMinMargin(errors, metric, `identity binding kind ${kind}`, config.minRetrievalMargin);
  }
  if (evalTrace.heldout_prompts && evalTrace.heldout_prompts.count > 0) {
    if (evalTrace.heldout_prompts.top1_per_mille < config.minHeldoutTop1PerMille) {
      errors.push(
        `held-out prompt top1 ${evalTrace.heldout_prompts.top1_per_mille} < ${config.minHeldoutTop1PerMille}`,
      );
    }
    if (evalTrace.heldout_prompts.top5_per_mille < config.minHeldoutTop5PerMille) {
      errors.push(
        `held-out prompt top5 ${evalTrace.heldout_prompts.top5_per_mille} < ${config.minHeldoutTop5PerMille}`,
      );
    }
    requireMinMargin(errors, evalTrace.heldout_prompts, "held-out prompt", config.minRetrievalMargin);
  }
  const heldoutRows = Number(evalTrace.heldout_prompt_rows || evalTrace.heldout_prompts?.count || 0);
  if (config.requireHeldoutPrompts && heldoutRows <= 0) {
    errors.push("held-out prompts are required but no prompt rows were evaluated");
  }
  if (heldoutRows < config.minHeldoutPromptRows) {
    errors.push(`held-out prompt rows ${heldoutRows} < ${config.minHeldoutPromptRows}`);
  }
  if (config.requireHeldoutPrompts && Number(evalTrace.heldout_prompt_unique_targets || 0) < 72) {
    errors.push(`held-out prompt unique targets ${Number(evalTrace.heldout_prompt_unique_targets || 0)} < 72`);
  }
  if (evalTrace.image_to_text.top1 < config.minImageTop1) {
    errors.push(`image-to-text/source top1 ${evalTrace.image_to_text.top1} < ${config.minImageTop1}`);
  }
  requireMinMargin(errors, evalTrace.image_to_text, "image-to-text/source", config.minRetrievalMargin);
  for (const task of IMAGE_RETRIEVAL_TASKS) {
    const taskMetric = evalTrace.image_tasks?.[task];
    if (!taskMetric || taskMetric.count <= 0) {
      errors.push(`${task} image retrieval has no rows`);
    } else if (taskMetric.top1 !== taskMetric.count) {
      errors.push(`${task} image retrieval top1 ${taskMetric.top1} != count ${taskMetric.count}`);
    }
    requireMinMargin(errors, taskMetric, `${task} image retrieval`, config.minRetrievalMargin);
  }
  if (evalTrace.match.yes.top1 < config.minMatchYesTop1) {
    errors.push(`match yes top1 ${evalTrace.match.yes.top1} < ${config.minMatchYesTop1}`);
  }
  if (evalTrace.match.no.top1 < config.minMatchNoTop1) {
    errors.push(`match no top1 ${evalTrace.match.no.top1} < ${config.minMatchNoTop1}`);
  }
  const noImageTop1 = Number(evalTrace.match.no_by_role?.image?.top1 || 0);
  const noPromptTop1 = Number(evalTrace.match.no_by_role?.prompt?.top1 || 0);
  if (noImageTop1 < config.minMatchNoImageTop1) {
    errors.push(`match no image top1 ${noImageTop1} < ${config.minMatchNoImageTop1}`);
  }
  if (noPromptTop1 < config.minMatchNoPromptTop1) {
    errors.push(`match no prompt top1 ${noPromptTop1} < ${config.minMatchNoPromptTop1}`);
  }
  requireMinMargin(errors, evalTrace.match.yes, "match yes", config.minRetrievalMargin);
  requireMinMargin(errors, evalTrace.match.no, "match no", config.minRetrievalMargin);
  requireMinMargin(errors, evalTrace.match.no_by_role?.image, "match no image", config.minRetrievalMargin);
  requireMinMargin(errors, evalTrace.match.no_by_role?.prompt, "match no prompt", config.minRetrievalMargin);
  return errors;
}

function requireMinMargin(errors, metric, label, floor) {
  const minimum = Number(floor || 0);
  if (minimum <= 0 || !metric || Number(metric.count || 0) <= 0) {
    return;
  }
  const margin = Number(metric.min_margin ?? Number.MIN_SAFE_INTEGER);
  if (margin < minimum) {
    errors.push(`${label} min_margin ${margin} < ${minimum}`);
  }
}

function serializeHead(head) {
  return {
    biases: Array.from(head.biases),
    weights: head.weights.map((weights) =>
      [...weights.entries()].sort((left, right) => left[0] - right[0]),
    ),
  };
}

function sparseNonZeroCount(head) {
  return head.weights.reduce((total, weights) => total + weights.size, 0);
}

function modelJson(config, spirits, textHead, imageHead, trainRows, evalTrace) {
  const model = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    corpus: {
      examples: config.examplesPath,
      examples_hash: evalTrace.examples_hash || "",
      tokens: config.tokensPath,
      tokens_hash: evalTrace.tokens_hash || "",
    },
    feature_count: config.featureCount,
    labels: spirits.map((spirit) => ({
      label: spirit.label,
      spirit_id: spirit.number,
      primary_name: spirit.primary_name,
      aliases: spirit.aliases,
    })),
    identity_anchor: {
      leading_boost: config.leadingAnchorBoost,
      mention_boost: config.mentionAnchorBoost,
      seal_id_templates: ["seal id {n}", "spirit {n}", "goetic spirit {n}"],
    },
    text_head: serializeHead(textHead),
    image_head: serializeHead(imageHead),
    training: {
      text_rows: trainRows.textRows.length,
      image_rows: trainRows.imageRows.length,
      epochs: config.epochs,
      seed: config.seed,
      text_mistakes: textHead.mistakes,
      image_mistakes: imageHead.mistakes,
      text_nonzero_weights: sparseNonZeroCount(textHead),
      image_nonzero_weights: sparseNonZeroCount(imageHead),
    },
    eval: {
      known_prompts: evalTrace.known_prompts,
      identity_bindings: evalTrace.identity_bindings,
      heldout_prompts: evalTrace.heldout_prompts,
      image_to_text: evalTrace.image_to_text,
      image_tasks: evalTrace.image_tasks,
      match: evalTrace.match,
    },
  };
  model.model_hash = fnv64Hex(JSON.stringify(model));
  return model;
}

function tokenize(text) {
  return normalizeKey(text)
    .split(/\s+/)
    .map(normalizeToken)
    .filter((token) => token.length >= 2);
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

function normalizedId(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function shuffledIndices(length, seed) {
  const indices = Array.from({ length }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swap = fnv32(`${seed}:${index}`) % (index + 1);
    [indices[index], indices[swap]] = [indices[swap], indices[index]];
  }
  return indices;
}

function fnv32(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function fnv64Hex(value) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64BytesHex(bytes) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    const examples = readJsonl(config.examplesPath);
    const tokens = fs.readFileSync(config.tokensPath);
    const corpusHashes = {
      examples_hash: fnv64FileHex(config.examplesPath),
      tokens_hash: fnv64BytesHex(tokens),
    };
    const spirits = readTextIndex(config.textIndexPath);
    const promptRows = readPromptRows(config.promptsPath);
    const trainRows = buildTrainingRows({
      spirits,
      examples,
      tokens,
      featureCount: config.featureCount,
    });
    const textHead = trainHead(spirits.length, trainRows.textRows, config, "text");
    const imageHead = trainHead(spirits.length, trainRows.imageRows, config, "image");
    const knownQueries = knownPromptQueries(examples);
    const known = evaluateText(
      textHead,
      spirits,
      knownQueries,
      config,
      config.maxMisses,
    );
    const identityBindings = evaluateIdentityBindings(
      textHead,
      spirits,
      knownQueries,
      config,
      config.maxMisses,
    );
    const heldout = promptRows.present
      ? evaluateText(textHead, spirits, promptRows.rows, config, config.maxMisses)
      : null;
    const image = evaluateImage(
      imageHead,
      spirits,
      examples,
      tokens,
      config.featureCount,
      config.maxMisses,
    );
    const imageTasks = evaluateImageTasks(
      imageHead,
      spirits,
      examples,
      tokens,
      config.featureCount,
      config.maxMisses,
    );
    const match = evaluateMatch(
      textHead,
      imageHead,
      spirits,
      examples,
      tokens,
      config,
      config.maxMisses,
    );
    const evalTrace = {
      schema: "nsrl.solomon_v2_retrieval_head_eval.v1",
      ok: true,
      model: config.modelOut || null,
      examples: config.examplesPath,
      examples_hash: corpusHashes.examples_hash,
      tokens: config.tokensPath,
      tokens_hash: corpusHashes.tokens_hash,
      text_index: config.textIndexPath,
      prompts: promptRows.present ? config.promptsPath : null,
      prompts_hash: promptRows.present ? fnv64FileHex(config.promptsPath) : "",
      prompt_rows_total: promptRows.total_rows || 0,
      heldout_prompt_tiers: promptRows.evaluated_tier_counts || {},
      heldout_prompt_sources: promptRows.evaluated_source_counts || {},
      heldout_prompt_unique_targets: promptRows.unique_targets || 0,
      feature_count: config.featureCount,
      epochs: config.epochs,
      text_training_rows: trainRows.textRows.length,
      image_training_rows: trainRows.imageRows.length,
      text_mistakes: textHead.mistakes,
      image_mistakes: imageHead.mistakes,
      known_prompts: known,
      identity_bindings: identityBindings,
      heldout_prompts: heldout,
      heldout_prompt_rows: promptRows.rows.length,
      image_to_text: image,
      image_tasks: imageTasks,
      match,
      errors: [],
    };
    evalTrace.errors = checkThresholds(config, evalTrace);
    evalTrace.ok = evalTrace.errors.length === 0;
    const model = modelJson(config, spirits, textHead, imageHead, trainRows, evalTrace);
    evalTrace.model_hash = model.model_hash;
    if (config.modelOut) {
      fs.writeFileSync(config.modelOut, `${JSON.stringify(model)}\n`, "utf8");
    }
    if (config.evalOut) {
      fs.writeFileSync(config.evalOut, `${JSON.stringify(evalTrace)}\n`, "utf8");
    }
    console.log(JSON.stringify(evalTrace));
    if (!evalTrace.ok) {
      console.error(`Solomon v2 retrieval head failed with ${evalTrace.errors.length} error(s):`);
      for (const error of evalTrace.errors) {
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
