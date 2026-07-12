#!/usr/bin/env node

import fs from "node:fs";

const defaults = {
  examplesPath: "",
  tokensPath: "",
  textIndexPath: "web/assets/solomon-spirit-text-signatures.tsv",
  promptsPath: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  expectSpirits: 72,
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
  maxMisses: 8,
};

const PAD = 0;
const BOS = 1;
const PROMPT = 2;
const TEXT = 3;
const IMAGE = 4;
const EOS = 5;
const TASK_TEXT_TO_IMAGE = 6;
const TASK_IMAGE_TO_TEXT = 7;
const TASK_MATCH = 8;
const TASK_EXPLAIN = 9;
const TASK_IDENTIFY = 10;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
const TEXT_BASE = 16;
const TEXT_COUNT = 128;
const IMAGE_BASE = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS = 16;
const TEXT_CHUNK_BASE = 160;

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
      "Usage: check-solomon-v2-retrieval-spine.mjs --examples PATH --tokens PATH [options]",
      "",
      "Checks that v2 Solomon multimodal artifacts support a narrow retrieval",
      "spine: text prompts rank the right spirit, image-token records rank their",
      "source spirit, and hard-negative match rows expose wrong-seal and",
      "wrong-prompt/name mismatches. It also requires explicit v2 identity",
      "bindings for primary names, aliases, and seal-ID prompts.",
      "",
      "Options:",
      "  --text-index PATH",
      "  --prompts PATH|none",
      "  --expect-spirits N",
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
      "  --max-misses N",
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
    } else if (arg === "--expect-spirits") {
      config.expectSpirits = parseNonNegative(requireValue(argv, ++index, arg), arg);
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
    } else if (arg === "--max-misses") {
      config.maxMisses = parseNonNegative(requireValue(argv, ++index, arg), arg);
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
      primary_name: fields[indexOf("primary_name")] || "",
      aliases: String(fields[indexOf("aliases")] || "").split("|").filter(Boolean),
      text: fields[indexOf("text")] || "",
    };
  });
}

function readPromptRows(path) {
  if (!path || path === "none") {
    return { rows: [], present: false, total_rows: 0, tier_counts: {}, source_counts: {}, unique_targets: 0 };
  }
  if (!fs.existsSync(path)) {
    return { rows: [], present: false, total_rows: 0, tier_counts: {}, source_counts: {}, unique_targets: 0 };
  }
  const allRows = readJsonl(path)
    .map((row) => ({
      spirit_id: normalizedId(row.spirit_id),
      text: String(row.text || row.prompt || ""),
      source: row.source || "prompt",
      tier: row.tier || "",
      prompt_hash: row.prompt_hash || "",
    }))
    .filter((row) => row.spirit_id !== null && row.text);
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

function buildTextProfiles(spirits, examples) {
  const profiles = new Map();
  for (const spirit of spirits) {
    const profile = new Map();
    addTextFeatures(profile, spirit.primary_name, 80);
    addTextFeatures(profile, `seal of ${spirit.primary_name}`, 80);
    addTextFeatures(profile, `${spirit.primary_name} goetic seal`, 64);
    for (const alias of spirit.aliases) {
      addTextFeatures(profile, alias, 72);
      addTextFeatures(profile, `seal of ${alias}`, 72);
    }
    addTextFeatures(profile, spirit.text, 2);
    profiles.set(spirit.number, profile);
  }
  for (const row of examples) {
    const spiritId = normalizedId(row.spirit_id);
    if (spiritId === null || !profiles.has(spiritId)) {
      continue;
    }
    const profile = profiles.get(spiritId);
    if (row.prompt) {
      addTextFeatures(profile, row.prompt, row.task === "identify" ? 96 : 28);
    }
    if (row.text && row.task !== "match") {
      const textWeight = ["explain", "image-to-explain", "text-image-explain", "image-to-attributes"].includes(row.task) ? 4 : 18;
      addTextFeatures(profile, row.text, textWeight);
    }
  }
  return profiles;
}

function addTextFeatures(profile, text, weight) {
  const tokens = tokenize(text);
  if (tokens.length === 0) {
    return;
  }
  if (tokens.length <= 6) {
    addFeature(profile, `whole:${tokens.join(" ")}`, weight * 12);
  }
  const content = tokens.filter((token) => !STOPWORDS.has(token));
  if (content.length > 0 && content.length <= 6) {
    addFeature(profile, `content:${content.join(" ")}`, weight * 14);
    addFeature(profile, `cset:${[...content].sort().join(" ")}`, weight * 12);
  }
  for (let index = 0; index < tokens.length; index += 1) {
    addFeature(profile, `tok:${tokens[index]}`, weight);
    if (tokens[index + 1]) {
      addFeature(profile, `bi:${tokens[index]} ${tokens[index + 1]}`, weight * 2);
    }
    if (tokens[index + 1] && tokens[index + 2]) {
      addFeature(
        profile,
        `tri:${tokens[index]} ${tokens[index + 1]} ${tokens[index + 2]}`,
        weight * 3,
      );
    }
  }
  for (let left = 0; left < content.length; left += 1) {
    addFeature(profile, `ctok:${content[left]}`, weight * 2);
    for (let right = left + 1; right < Math.min(content.length, left + 10); right += 1) {
      addFeature(profile, `pair:${[content[left], content[right]].sort().join(" ")}`, weight * 2);
    }
  }
}

function queryFeatures(text) {
  const features = new Map();
  addTextFeatures(features, text, 1);
  return features;
}

function addFeature(map, key, value) {
  map.set(key, (map.get(key) || 0) + value);
}

function rankText(spirits, profiles, query) {
  const features = queryFeatures(query);
  const queryKey = normalizeKey(query);
  const ranked = spirits.map((spirit) => {
    const profile = profiles.get(spirit.number);
    let score = 0;
    for (const [feature, value] of features) {
      score += value * (profile.get(feature) || 0);
    }
    score += spiritNameBoost(spirit, queryKey);
    return {
      spirit_id: spirit.number,
      primary_name: spirit.primary_name,
      score,
    };
  });
  ranked.sort((left, right) => right.score - left.score || left.spirit_id - right.spirit_id);
  return ranked;
}

function spiritNameBoost(spirit, queryKey) {
  let boost = 0;
  for (const name of [spirit.primary_name, ...spirit.aliases]) {
    const key = normalizeKey(name);
    if (!key) {
      continue;
    }
    if (startsWithPhrase(queryKey, key)) {
      boost = Math.max(boost, 100_000_000 + key.length * 100_000);
    } else if (containsPhrase(queryKey, key)) {
      boost = Math.max(boost, 10_000_000 + key.length * 10_000);
    }
  }
  return boost;
}

function startsWithPhrase(haystack, needle) {
  return haystack === needle || haystack.startsWith(`${needle} `);
}

function containsPhrase(haystack, needle) {
  return ` ${haystack} `.includes(` ${needle} `);
}

function evaluateTextQueries({ spirits, profiles, queries, maxMisses }) {
  let top1 = 0;
  let top5 = 0;
  let rankSum = 0;
  const misses = [];
  for (const query of queries) {
    const ranked = rankText(spirits, profiles, query.text);
    const rank = ranked.findIndex((row) => row.spirit_id === query.spirit_id) + 1;
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
        rank,
        top: ranked.slice(0, 5),
      });
    }
  }
  return querySummary(queries.length, top1, top5, rankSum, misses);
}

function querySummary(count, top1, top5, rankSum, misses) {
  return {
    count,
    top1,
    top5,
    top1_per_mille: count === 0 ? null : Math.round((top1 * 1000) / count),
    top5_per_mille: count === 0 ? null : Math.round((top5 * 1000) / count),
    mean_rank_per_mille: count === 0 ? null : Math.round((rankSum * 1000) / count),
    misses,
  };
}

function buildKnownQueries(examples) {
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
    out.push({ spirit_id: spiritId, text: row.prompt, source: "v2-identify" });
  }
  return out;
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

function canonicalImageVectors(examples, tokens) {
  const vectors = new Map();
  const candidates = examples.filter((row) => row.task === "text-to-image");
  for (const row of candidates) {
    const spiritId = normalizedId(row.spirit_id);
    if (spiritId === null || vectors.has(spiritId)) {
      continue;
    }
    const image = imageVectorForRow(row, tokens);
    if (image) {
      vectors.set(spiritId, image);
    }
  }
  return vectors;
}

function evaluateImageIdentity({ spirits, examples, tokens, canonical, maxMisses }) {
  let top1 = 0;
  let top5 = 0;
  let rankSum = 0;
  const misses = [];
  const queries = examples.filter((row) =>
    ["image-to-text", "image-to-explain", "text-image-explain", "image-to-attributes"].includes(row.task),
  );
  for (const row of queries) {
    const spiritId = normalizedId(row.spirit_id);
    const image = imageVectorForRow(row, tokens);
    if (spiritId === null || !image) {
      continue;
    }
    const ranked = rankImage(spirits, canonical, image);
    const rank = ranked.findIndex((item) => item.spirit_id === spiritId) + 1;
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
        top: ranked.slice(0, 5),
      });
    }
  }
  return querySummary(queries.length, top1, top5, rankSum, misses);
}

function evaluateMatchRows({ spirits, profiles, examples, tokens, canonical, maxMisses }) {
  const yesRows = [];
  const noRows = [];
  const noByRole = {
    image: [],
    prompt: [],
  };
  const misses = [];
  for (const row of examples.filter((item) => item.task === "match")) {
    const spiritId = normalizedId(row.spirit_id);
    const label = String(row.match_label || row.text || "").toLowerCase();
    const image = imageVectorForRow(row, tokens);
    if (spiritId === null || !image) {
      continue;
    }
    const rankedImage = rankImage(spirits, canonical, image);
    const rankedText = rankText(spirits, profiles, row.prompt || "");
    if (label === "yes") {
      const imageRank = rankedImage.findIndex((item) => item.spirit_id === spiritId) + 1;
      yesRows.push(imageRank);
      if (imageRank !== 1 && misses.length < maxMisses) {
        misses.push({
          label,
          spirit_id: spiritId,
          primary_name: row.primary_name,
          image_rank: imageRank,
          top_image: rankedImage.slice(0, 5),
        });
      }
    } else if (label === "no") {
      const negativeSpiritId = normalizedId(row.negative_spirit_id);
      const negativeRole = matchNegativeRole(row);
      const imageExpectedSpiritId = negativeRole === "prompt" ? spiritId : negativeSpiritId;
      const textExpectedSpiritId = negativeRole === "prompt" ? negativeSpiritId : null;
      const imageExpectedRank =
        rankedImage.findIndex((item) => item.spirit_id === imageExpectedSpiritId) + 1;
      const textExpectedRank =
        textExpectedSpiritId === null
          ? 0
          : rankedText.findIndex((item) => item.spirit_id === textExpectedSpiritId) + 1;
      const imageSourceRank = rankedImage.findIndex((item) => item.spirit_id === spiritId) + 1;
      const textSourceRank = rankedText.findIndex((item) => item.spirit_id === spiritId) + 1;
      const ok = negativeRole === "prompt"
        ? imageExpectedRank === 1 && textExpectedRank === 1 && textSourceRank !== 1
        : imageExpectedRank === 1 && imageSourceRank !== 1;
      const noRow = { negativeRole, ok, imageExpectedRank, textExpectedRank, imageSourceRank, textSourceRank };
      noRows.push(noRow);
      if (negativeRole === "image" || negativeRole === "prompt") {
        noByRole[negativeRole].push(noRow);
      }
      if (!ok && misses.length < maxMisses) {
        misses.push({
          label,
          negative_role: negativeRole,
          spirit_id: spiritId,
          negative_spirit_id: negativeSpiritId,
          primary_name: row.primary_name,
          negative_primary_name: row.negative_primary_name,
          image_expected_rank: imageExpectedRank,
          text_expected_rank: textExpectedRank,
          image_source_rank: imageSourceRank,
          text_source_rank: textSourceRank,
          top_image: rankedImage.slice(0, 5),
          top_text: rankedText.slice(0, 5),
        });
      }
    }
  }
  const yesTop1 = yesRows.filter((rank) => rank === 1).length;
  const noTop1 = noRows.filter((row) => row.ok).length;
  return {
    yes: {
      count: yesRows.length,
      top1: yesTop1,
      top1_per_mille: yesRows.length === 0 ? null : Math.round((yesTop1 * 1000) / yesRows.length),
    },
    no: {
      count: noRows.length,
      top1: noTop1,
      top1_per_mille: noRows.length === 0 ? null : Math.round((noTop1 * 1000) / noRows.length),
    },
    no_by_role: {
      image: matchRowsSummary(noByRole.image),
      prompt: matchRowsSummary(noByRole.prompt),
    },
    misses,
  };
}

function matchRowsSummary(rows) {
  const top1 = rows.filter((row) => row.ok).length;
  return {
    count: rows.length,
    top1,
    top1_per_mille: rows.length === 0 ? null : Math.round((top1 * 1000) / rows.length),
  };
}

function rankImage(spirits, canonical, image) {
  const ranked = [];
  for (const spirit of spirits) {
    const target = canonical.get(spirit.number);
    if (!target) {
      continue;
    }
    ranked.push({
      spirit_id: spirit.number,
      primary_name: spirit.primary_name,
      distance: imageDistance(image, target),
    });
  }
  ranked.sort((left, right) => left.distance - right.distance || left.spirit_id - right.spirit_id);
  return ranked;
}

function imageDistance(left, right) {
  const count = Math.min(left.length, right.length);
  let distance = Math.abs(left.length - right.length) * 1024;
  for (let index = 0; index < count; index += 1) {
    distance += Math.abs(left[index] - right[index]);
  }
  return distance;
}

function nearestImageTokenNegatives(canonical, spiritId) {
  const source = canonical.get(spiritId);
  if (!source) {
    return null;
  }
  let bestDistance = Number.POSITIVE_INFINITY;
  const spiritIds = new Set();
  for (const [candidateId, candidate] of canonical.entries()) {
    if (candidateId === spiritId) {
      continue;
    }
    const distance = imageDistance(source, candidate);
    if (distance < bestDistance) {
      bestDistance = distance;
      spiritIds.clear();
      spiritIds.add(candidateId);
    } else if (distance === bestDistance) {
      spiritIds.add(candidateId);
    }
  }
  if (!Number.isFinite(bestDistance) || spiritIds.size === 0) {
    return null;
  }
  return { distance: bestDistance, spirit_ids: spiritIds };
}

function validateCoverage({ config, spirits, examples, canonical }) {
  const errors = [];
  if (spirits.length !== config.expectSpirits) {
    errors.push(`text index has ${spirits.length} spirits, expected ${config.expectSpirits}`);
  }
  const exampleSpirits = new Set(
    examples.map((row) => normalizedId(row.spirit_id)).filter((value) => value !== null),
  );
  if (exampleSpirits.size !== config.expectSpirits) {
    errors.push(`examples cover ${exampleSpirits.size} spirits, expected ${config.expectSpirits}`);
  }
  if (canonical.size !== config.expectSpirits) {
    errors.push(`canonical image vectors cover ${canonical.size} spirits, expected ${config.expectSpirits}`);
  }
  for (const row of examples) {
    if (row.task !== "match") {
      continue;
    }
    const label = String(row.match_label || row.text || "").toLowerCase();
    if (label === "no") {
      const spiritId = normalizedId(row.spirit_id);
      const negativeSpiritId = normalizedId(row.negative_spirit_id);
      if (negativeSpiritId === null) {
        errors.push(`examples line ${row.__line}: no-match row is missing negative_spirit_id`);
      } else if (spiritId === negativeSpiritId) {
        errors.push(`examples line ${row.__line}: no-match row uses its own spirit as negative`);
      }
      const negativeRole = matchNegativeRole(row);
      if (row.negative_role && negativeRole !== "image" && negativeRole !== "prompt") {
        errors.push(`examples line ${row.__line}: no-match row has invalid negative_role ${JSON.stringify(row.negative_role)}`);
      }
      if (String(row.negative_selection || "") !== "nearest-image-token") {
        errors.push(`examples line ${row.__line}: no-match row negative_selection ${JSON.stringify(row.negative_selection || "")} != nearest-image-token`);
      }
      if (Number(row.negative_image_token_rank) !== 1) {
        errors.push(`examples line ${row.__line}: no-match row negative_image_token_rank ${JSON.stringify(row.negative_image_token_rank || "")} != 1`);
      }
      const nearest = nearestImageTokenNegatives(canonical, spiritId);
      const reportedDistance = Number(row.negative_image_token_distance);
      if (!Number.isInteger(reportedDistance) || reportedDistance <= 0) {
        errors.push(`examples line ${row.__line}: no-match row has invalid negative_image_token_distance ${JSON.stringify(row.negative_image_token_distance || "")}`);
      } else if (nearest) {
        if (reportedDistance !== nearest.distance) {
          errors.push(
            `examples line ${row.__line}: no-match row negative_image_token_distance ${reportedDistance} != nearest ${nearest.distance}`,
          );
        }
        if (!nearest.spirit_ids.has(negativeSpiritId)) {
          errors.push(
            `examples line ${row.__line}: no-match row negative_spirit_id ${negativeSpiritId} is not a nearest image-token neighbor of ${spiritId}`,
          );
        }
      }
    }
  }
  return errors;
}

function evaluateIdentityBindings({ spirits, examples, maxMisses }) {
  const spiritsById = new Map(spirits.map((spirit) => [spirit.number, spirit]));
  const expected = expectedIdentityBindings(spirits);
  const identifyKeys = new Set();
  const textToImageKeys = new Set();
  let identityRows = 0;
  const targetErrors = [];

  for (const row of examples) {
    if (row.identity_binding !== true && row.identity_binding !== "true") {
      continue;
    }
    identityRows += 1;
    const spiritId = normalizedId(row.spirit_id);
    const promptKey = normalizeKey(row.prompt || "");
    if (spiritId === null || !promptKey) {
      continue;
    }
    const key = `${spiritId}:${promptKey}`;
    if (row.task === "identify") {
      identifyKeys.add(key);
      const spirit = spiritsById.get(spiritId);
      if (spirit && normalizeKey(row.text || "") !== normalizeKey(spirit.primary_name)) {
        targetErrors.push({
          line: row.__line,
          spirit_id: spiritId,
          prompt: row.prompt,
          text: row.text,
          expected: spirit.primary_name,
        });
      }
    } else if (row.task === "text-to-image") {
      textToImageKeys.add(key);
    }
  }

  const byKind = {};
  const missingIdentify = [];
  const missingTextToImage = [];
  let identifyCovered = 0;
  let textToImageCovered = 0;
  for (const binding of expected) {
    const kind = binding.kind;
    if (!byKind[kind]) {
      byKind[kind] = {
        required: 0,
        identify_covered: 0,
        text_to_image_covered: 0,
      };
    }
    byKind[kind].required += 1;
    const key = `${binding.spirit_id}:${binding.prompt_key}`;
    if (identifyKeys.has(key)) {
      identifyCovered += 1;
      byKind[kind].identify_covered += 1;
    } else if (missingIdentify.length < maxMisses) {
      missingIdentify.push(binding);
    }
    if (textToImageKeys.has(key)) {
      textToImageCovered += 1;
      byKind[kind].text_to_image_covered += 1;
    } else if (missingTextToImage.length < maxMisses) {
      missingTextToImage.push(binding);
    }
  }

  return {
    required_prompts: expected.length,
    identity_rows: identityRows,
    identify: {
      rows: identifyKeys.size,
      covered: identifyCovered,
      missing: missingIdentify,
    },
    text_to_image: {
      rows: textToImageKeys.size,
      covered: textToImageCovered,
      missing: missingTextToImage,
    },
    by_kind: byKind,
    target_error_count: targetErrors.length,
    target_errors: targetErrors.slice(0, maxMisses),
  };
}

function expectedIdentityBindings(spirits) {
  const out = [];
  for (const spirit of spirits) {
    const seen = new Set();
    const add = (kind, prompt) => {
      const text = String(prompt || "").trim();
      const promptKey = normalizeKey(text);
      if (!promptKey || seen.has(promptKey)) {
        return;
      }
      seen.add(promptKey);
      out.push({
        spirit_id: spirit.number,
        primary_name: spirit.primary_name,
        kind,
        prompt: text,
        prompt_key: promptKey,
      });
    };
    add("primary-name", spirit.primary_name);
    add("primary-seal", `seal of ${spirit.primary_name}`);
    for (const alias of spirit.aliases) {
      add("alias", alias);
      add("alias-seal", `seal of ${alias}`);
    }
    add("seal-id", `seal id ${spirit.number}`);
    add("seal-id", `spirit ${spirit.number}`);
    add("seal-id", `goetic spirit ${spirit.number}`);
  }
  return out;
}

function checkIdentityBindingCoverage(identityBindings) {
  const errors = [];
  const required = Number(identityBindings.required_prompts || 0);
  if (required <= 0) {
    errors.push("identity binding coverage has no required prompts");
    return errors;
  }
  if (identityBindings.identify.covered < required) {
    errors.push(`identity identify bindings covered ${identityBindings.identify.covered}/${required}`);
  }
  if (identityBindings.text_to_image.covered < required) {
    errors.push(`identity text-to-image bindings covered ${identityBindings.text_to_image.covered}/${required}`);
  }
  if (identityBindings.target_error_count > 0) {
    errors.push(`identity identify bindings have ${identityBindings.target_error_count} target error(s)`);
  }
  return errors;
}

function matchNegativeRole(row) {
  const role = String(row.negative_role || "image").toLowerCase();
  if (role === "prompt" || role === "text" || role === "name") {
    return "prompt";
  }
  if (role === "image" || role === "seal") {
    return "image";
  }
  return role;
}

function checkThresholds(config, known, heldout, image, match, promptRows) {
  const errors = [];
  if (known.top1 < config.minKnownTop1) {
    errors.push(`known prompt top1 ${known.top1} < ${config.minKnownTop1}`);
  }
  if (known.top5 < config.minKnownTop5) {
    errors.push(`known prompt top5 ${known.top5} < ${config.minKnownTop5}`);
  }
  if (heldout && heldout.count > 0) {
    if (heldout.top1_per_mille < config.minHeldoutTop1PerMille) {
      errors.push(`held-out prompt top1 ${heldout.top1_per_mille} < ${config.minHeldoutTop1PerMille}`);
    }
    if (heldout.top5_per_mille < config.minHeldoutTop5PerMille) {
      errors.push(`held-out prompt top5 ${heldout.top5_per_mille} < ${config.minHeldoutTop5PerMille}`);
    }
  }
  const heldoutRows = Number(heldout?.count || 0);
  if (config.requireHeldoutPrompts && heldoutRows <= 0) {
    errors.push("held-out prompts are required but no prompt rows were evaluated");
  }
  if (heldoutRows < config.minHeldoutPromptRows) {
    errors.push(`held-out prompt rows ${heldoutRows} < ${config.minHeldoutPromptRows}`);
  }
  if (config.requireHeldoutPrompts && Number(promptRows?.unique_targets || 0) < config.expectSpirits) {
    errors.push(
      `held-out prompt unique targets ${Number(promptRows?.unique_targets || 0)} < ${config.expectSpirits}`,
    );
  }
  if (image.top1 < config.minImageTop1) {
    errors.push(`image-to-text/source top1 ${image.top1} < ${config.minImageTop1}`);
  }
  if (match.yes.top1 < config.minMatchYesTop1) {
    errors.push(`match yes top1 ${match.yes.top1} < ${config.minMatchYesTop1}`);
  }
  if (match.no.top1 < config.minMatchNoTop1) {
    errors.push(`match no negative top1 ${match.no.top1} < ${config.minMatchNoTop1}`);
  }
  const noImageTop1 = Number(match.no_by_role?.image?.top1 || 0);
  const noPromptTop1 = Number(match.no_by_role?.prompt?.top1 || 0);
  if (noImageTop1 < config.minMatchNoImageTop1) {
    errors.push(`match no image top1 ${noImageTop1} < ${config.minMatchNoImageTop1}`);
  }
  if (noPromptTop1 < config.minMatchNoPromptTop1) {
    errors.push(`match no prompt top1 ${noPromptTop1} < ${config.minMatchNoPromptTop1}`);
  }
  return errors;
}

function normalizedId(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function tokenize(text) {
  return normalizeKey(text).split(/\s+/).filter(Boolean);
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

function buildSummary({ config, spirits, examples, promptRows, promptRowsPresent, known, heldout, image, match, identityBindings, errors }) {
  const taskCounts = {};
  for (const row of examples) {
    const task = row.task || "legacy";
    taskCounts[task] = (taskCounts[task] || 0) + 1;
  }
  const imageProfiles = [...new Set(examples.map((row) => row.image_token_profile).filter(Boolean))].sort();
  return {
    schema: "nsrl.solomon_v2_retrieval_spine_check.v1",
    ok: errors.length === 0,
    examples: config.examplesPath,
    tokens: config.tokensPath,
    text_index: config.textIndexPath,
    prompts: promptRowsPresent ? config.promptsPath : null,
    prompts_hash: promptRowsPresent ? fnv64FileHex(config.promptsPath) : "",
    prompt_rows_total: promptRows.total_rows || 0,
    heldout_prompt_tiers: promptRows.evaluated_tier_counts || {},
    heldout_prompt_sources: promptRows.evaluated_source_counts || {},
    heldout_prompt_unique_targets: promptRows.unique_targets || 0,
    spirits: spirits.length,
    example_count: examples.length,
    task_counts: taskCounts,
    image_token_profiles: imageProfiles,
    known_prompts: known,
    heldout_prompts: heldout,
    heldout_prompt_rows: promptRows.rows?.length ?? promptRows.length,
    image_to_text: image,
    match,
    identity_bindings: identityBindings,
    errors,
  };
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
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

function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    const examples = readJsonl(config.examplesPath);
    const tokens = fs.readFileSync(config.tokensPath);
    const spirits = readTextIndex(config.textIndexPath);
    const profiles = buildTextProfiles(spirits, examples);
    const promptRows = readPromptRows(config.promptsPath);
    const knownQueries = buildKnownQueries(examples);
    const canonical = canonicalImageVectors(examples, tokens);
    const identityBindings = evaluateIdentityBindings({
      spirits,
      examples,
      maxMisses: config.maxMisses,
    });

    const known = evaluateTextQueries({
      spirits,
      profiles,
      queries: knownQueries,
      maxMisses: config.maxMisses,
    });
    const heldout = promptRows.present
      ? evaluateTextQueries({
          spirits,
          profiles,
          queries: promptRows.rows,
          maxMisses: config.maxMisses,
        })
      : null;
    const image = evaluateImageIdentity({
      spirits,
      examples,
      tokens,
      canonical,
      maxMisses: config.maxMisses,
    });
    const match = evaluateMatchRows({
      spirits,
      profiles,
      examples,
      tokens,
      canonical,
      maxMisses: config.maxMisses,
    });

    const errors = [
      ...validateCoverage({ config, spirits, examples, canonical }),
      ...checkIdentityBindingCoverage(identityBindings),
      ...checkThresholds(config, known, heldout, image, match, promptRows),
    ];
    const summary = buildSummary({
      config,
      spirits,
      examples,
      promptRows,
      promptRowsPresent: promptRows.present,
      known,
      heldout,
      image,
      match,
      identityBindings,
      errors,
    });
    console.log(JSON.stringify(summary));
    if (errors.length > 0) {
      console.error(`Solomon v2 retrieval spine check failed with ${errors.length} error(s):`);
      for (const error of errors) {
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
