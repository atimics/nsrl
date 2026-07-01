#!/usr/bin/env node

import fs from "node:fs";

const SHORT_WORDS = new Set([
  "a",
  "am",
  "an",
  "as",
  "at",
  "be",
  "by",
  "go",
  "he",
  "if",
  "in",
  "is",
  "it",
  "me",
  "my",
  "no",
  "of",
  "on",
  "or",
  "so",
  "to",
  "up",
  "we",
]);

function usage() {
  console.error(
    [
      "usage: node scripts/check-solomon-attention-raw-quality.mjs --text PATH",
      "       [--label NAME] [--prompt PROMPT] [--min-score N]",
      "       [--vocab-source PATH | --no-vocab]",
    ].join("\n"),
  );
}

let textPath = null;
let label = "raw_attention";
let minScore = null;
let vocabSource = "web/assets/solomon-spirit-text-signatures.tsv";
let prompt = null;

for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--text") {
    textPath = process.argv[++index] || null;
  } else if (arg === "--label") {
    label = process.argv[++index] || label;
  } else if (arg === "--prompt") {
    prompt = process.argv[++index] || null;
  } else if (arg === "--min-score") {
    minScore = Number(process.argv[++index]);
  } else if (arg === "--vocab-source") {
    vocabSource = process.argv[++index] || null;
  } else if (arg === "--no-vocab") {
    vocabSource = null;
  } else if (arg === "--help" || arg === "-h") {
    usage();
    process.exit(0);
  } else {
    console.error(`unknown argument: ${arg}`);
    usage();
    process.exit(2);
  }
}

if (!textPath) {
  usage();
  process.exit(2);
}

const text = fs.readFileSync(textPath, "utf8").trim();
const source = vocabSource && fs.existsSync(vocabSource) ? sourceMetadata(vocabSource) : null;
const metrics = rawQualityMetrics(text, {
  vocabulary: source?.vocabulary || null,
  names: source?.names || [],
  prompt,
});
const line = [
  `${label}_score=${metrics.score}`,
  `${label}_chars=${metrics.characters}`,
  `${label}_alpha_ratio_per_mille=${metrics.alphaRatioPerMille}`,
  `${label}_wordlike_ratio_per_mille=${metrics.wordlikeRatioPerMille}`,
  `${label}_repeat_ratio_per_mille=${metrics.repeatRatioPerMille}`,
  `${label}_longest_run=${metrics.longestRun}`,
  `${label}_unique_chars=${metrics.uniqueChars}`,
  `${label}_dominant_word_ratio_per_mille=${metrics.dominantWordRatioPerMille}`,
  `${label}_repeated_word_ngram_ratio_per_mille=${metrics.repeatedWordNgramRatioPerMille}`,
  `${label}_case_noise_ratio_per_mille=${metrics.caseNoiseRatioPerMille}`,
  `${label}_source_vocab_ratio_per_mille=${metrics.sourceVocabRatioPerMille}`,
  `${label}_prompt_name_match=${metrics.promptNameMatch}`,
  `${label}_other_name_penalty_per_mille=${metrics.otherNamePenaltyPerMille}`,
  `${label}_phrase_restart_penalty_per_mille=${metrics.phraseRestartPenaltyPerMille}`,
  `${label}_glued_repeat_penalty_per_mille=${metrics.gluedRepeatPenaltyPerMille}`,
].join(" ");
console.log(line);

if (minScore !== null && metrics.score < minScore) {
  console.error(`${label} score ${metrics.score} < ${minScore}: ${text}`);
  process.exit(1);
}

function rawQualityMetrics(value, context = {}) {
  const vocabulary = context.vocabulary || null;
  const expectedName = expectedNameForPrompt(context.prompt, context.names || []);
  const openingName = openingNameForText(value);
  const body = value.replace(/^Solomon selects(?: [A-Za-z-]+: He)?\s*/, "");
  const chars = [...body];
  const characters = chars.length;
  const alpha = chars.filter((char) => /[A-Za-z]/.test(char)).length;
  const uniqueChars = new Set(chars).size;
  const longestRun = longestRepeatedRun(chars);
  const repeated = repeatedCharacterCount(chars);
  const words = body.match(/[A-Za-z]{2,}/g) || [];
  const wordlike = words.filter(isWordlike).length;
  const wordStats = repeatedWordStats(words);
  const caseNoiseRatio = caseNoiseStats(words);
  const sourceVocabRatio = sourceVocabularyRatio(words, vocabulary);
  const promptNameMatch = promptNameMatchScore(openingName, expectedName);
  const otherNamePenalty = otherSpiritNamePenalty(value, expectedName, context.names || []);
  const wordlikeRatio = words.length === 0 ? 0 : wordlike / words.length;
  const alphaRatio = characters === 0 ? 0 : alpha / characters;
  const repeatRatio = characters === 0 ? 1 : repeated / characters;
  const uniqueScore = Math.min(uniqueChars, 18) / 18;
  const phraseRestartPenalty = Math.min(1, countPhrase(body, "Solomon selects") * 0.5);
  const gluedRepeatPenalty = hasGluedRepeatedWord(words) ? 1 : 0;
  const wordRepeatPenalty = Math.max(
    wordStats.repeatedNgramRatio,
    Math.max(0, wordStats.dominantWordRatio - 0.28) * 2,
  );
  const lexicalPenalty = clamp01(
    Math.max(phraseRestartPenalty, wordRepeatPenalty, gluedRepeatPenalty),
  );
  const sourcePenalty = sourceVocabRatio === null ? 0 : 1 - sourceVocabRatio;
  const promptPenalty = expectedName && promptNameMatch !== 1 ? 0.45 : 0;

  const score = clamp01(
    alphaRatio * 0.25 +
      wordlikeRatio * 0.35 +
      uniqueScore * 0.15 +
      (1 - repeatRatio) * 0.2 +
      (longestRun <= 3 ? 0.05 : 0) -
      lexicalPenalty * 0.45 -
      caseNoiseRatio * 0.35 -
      sourcePenalty * 0.35 -
      otherNamePenalty * 0.4 -
      promptPenalty,
  );

  return {
    score: Math.round(score * 100),
    characters,
    alphaRatioPerMille: Math.round(alphaRatio * 1000),
    wordlikeRatioPerMille: Math.round(wordlikeRatio * 1000),
    repeatRatioPerMille: Math.round(repeatRatio * 1000),
    longestRun,
    uniqueChars,
    dominantWordRatioPerMille: Math.round(wordStats.dominantWordRatio * 1000),
    repeatedWordNgramRatioPerMille: Math.round(wordStats.repeatedNgramRatio * 1000),
    caseNoiseRatioPerMille: Math.round(caseNoiseRatio * 1000),
    sourceVocabRatioPerMille:
      sourceVocabRatio === null ? -1 : Math.round(sourceVocabRatio * 1000),
    promptNameMatch,
    otherNamePenaltyPerMille: Math.round(otherNamePenalty * 1000),
    phraseRestartPenaltyPerMille: Math.round(phraseRestartPenalty * 1000),
    gluedRepeatPenaltyPerMille: Math.round(gluedRepeatPenalty * 1000),
  };
}

function sourceMetadata(tsvPath) {
  const text = fs.readFileSync(tsvPath, "utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  const textIndex = header.indexOf("text");
  const nameIndex = header.indexOf("primary_name");
  const aliasIndex = header.indexOf("aliases");
  const vocabulary = new Set(SHORT_WORDS);
  const names = [];
  for (const line of lines) {
    const fields = line.split("\t");
    if (nameIndex >= 0 && fields[nameIndex]) {
      names.push(fields[nameIndex]);
    }
    for (const index of [textIndex, nameIndex, aliasIndex]) {
      if (index >= 0) {
        addWordsToVocabulary(vocabulary, fields[index] || "");
      }
    }
  }
  return { vocabulary, names };
}

function addWordsToVocabulary(vocabulary, value) {
  for (const word of normalizeWords(value)) {
    vocabulary.add(word);
  }
}

function sourceVocabularyRatio(words, vocabulary) {
  if (!vocabulary || words.length === 0) {
    return null;
  }
  let known = 0;
  for (const word of words) {
    if (vocabulary.has(normalizeWord(word))) {
      known += 1;
    }
  }
  return known / words.length;
}

function expectedNameForPrompt(prompt, names) {
  if (!prompt) {
    return null;
  }
  const promptKey = normalizeTextKey(prompt);
  return (
    names.find((name) => promptContainsPhrase(promptKey, normalizeTextKey(name))) || null
  );
}

function openingNameForText(value) {
  const match = String(value || "").match(/^Solomon selects\s+([A-Za-z-]+):/);
  return match?.[1] || null;
}

function promptNameMatchScore(openingName, expectedName) {
  if (!expectedName) {
    return -1;
  }
  if (!openingName) {
    return 0;
  }
  return normalizeTextKey(openingName) === normalizeTextKey(expectedName) ? 1 : 0;
}

function otherSpiritNamePenalty(value, expectedName, names) {
  const textKey = ` ${normalizeTextKey(value)} `;
  let count = 0;
  for (const name of names) {
    const nameKey = normalizeTextKey(name);
    if (!nameKey || nameKey === normalizeTextKey(expectedName)) {
      continue;
    }
    if (promptContainsPhrase(textKey, nameKey)) {
      count += 1;
    }
  }
  return Math.min(1, count * 0.25);
}

function promptContainsPhrase(textKey, phraseKey) {
  return (
    textKey === phraseKey ||
    textKey.split(/\s+/).includes(phraseKey) ||
    textKey.includes(` ${phraseKey} `) ||
    textKey.startsWith(`${phraseKey} `) ||
    textKey.endsWith(` ${phraseKey}`)
  );
}

function caseNoiseStats(words) {
  if (words.length === 0) {
    return 0;
  }
  const noisy = words.filter(hasNoisyCasing).length;
  return noisy / words.length;
}

function repeatedWordStats(words) {
  const normalized = words.map((word) => word.toLowerCase());
  if (normalized.length === 0) {
    return {
      dominantWordRatio: 0,
      repeatedNgramRatio: 0,
    };
  }
  const counts = new Map();
  for (const word of normalized) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  const dominant = Math.max(...counts.values()) / normalized.length;
  let repeated = 0;
  let total = 0;
  for (let size = 2; size <= 4; size += 1) {
    if (normalized.length < size) {
      continue;
    }
    const seen = new Set();
    for (let index = 0; index + size <= normalized.length; index += 1) {
      const key = normalized.slice(index, index + size).join(" ");
      total += 1;
      if (seen.has(key)) {
        repeated += 1;
      } else {
        seen.add(key);
      }
    }
  }
  return {
    dominantWordRatio: dominant,
    repeatedNgramRatio: total === 0 ? 0 : repeated / total,
  };
}

function countPhrase(text, phrase) {
  let count = 0;
  let index = 0;
  while (index < text.length) {
    const found = text.indexOf(phrase, index);
    if (found < 0) {
      break;
    }
    count += 1;
    index = found + phrase.length;
  }
  return count;
}

function longestRepeatedRun(chars) {
  let best = 0;
  let run = 0;
  let last = null;
  for (const char of chars) {
    if (char === last) {
      run += 1;
    } else {
      last = char;
      run = 1;
    }
    best = Math.max(best, run);
  }
  return best;
}

function repeatedCharacterCount(chars) {
  let repeated = 0;
  let last = null;
  for (const char of chars) {
    if (char === last && /\S/.test(char)) {
      repeated += 1;
    }
    last = char;
  }
  return repeated;
}

function isWordlike(word) {
  const normalized = word.toLowerCase();
  if (hasNoisyCasing(word)) {
    return false;
  }
  if (normalized.length <= 2 && !SHORT_WORDS.has(normalized)) {
    return false;
  }
  if (/([a-z]{3,})\1/.test(normalized)) {
    return false;
  }
  if (/(.)\1{2,}/.test(normalized)) {
    return false;
  }
  if (!/[aeiouy]/.test(normalized)) {
    return false;
  }
  if (/^[bcdfghjklmnpqrstvwxz]{4,}$/i.test(normalized)) {
    return false;
  }
  return true;
}

function hasNoisyCasing(word) {
  return /[A-Z]/.test(word.slice(1));
}

function hasGluedRepeatedWord(words) {
  return words.some((word) => /([A-Za-z]{3,})\1/.test(word));
}

function normalizeWords(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .match(/[a-z]{2,}/g) || [];
}

function normalizeWord(value) {
  return normalizeWords(value)[0] || "";
}

function normalizeTextKey(value) {
  return normalizeWords(value).join(" ");
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}
