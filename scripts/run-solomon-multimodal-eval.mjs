#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";

const defaults = {
  model: "web/assets/solomon-multimodal.nsrlmod",
  textIndex: "web/assets/solomon-spirit-text-signatures.tsv",
  out: "docs/solomon-multimodal-eval.tsv",
  label: "deployed",
  maxTextChars: 220,
};

const MODEL_MAGIC = "NSRLMOD1";
const MODEL_VERSION = 1;
const PAD = 0;
const BOS = 1;
const PROMPT = 2;
const TEXT = 3;
const IMAGE = 4;
const EOS = 5;
const TEXT_BASE = 16;
const TEXT_COUNT = 128;
const IMAGE_BASE = TEXT_BASE + TEXT_COUNT;
const IMAGE_BINS = 16;
const SIGNATURE_GRID = 16;
const SIGNATURE_BINS = SIGNATURE_GRID * SIGNATURE_GRID;
const VOCAB_SIZE = IMAGE_BASE + IMAGE_BINS;
const MAX_CONTEXT_TOKENS = 64;
const CONTEXT_LENGTHS = [1, 2, 4, 8, 16, 32, 64];
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;
const schema = "nsrl.solomon_multimodal_eval.v1";

const config = parseArgs(process.argv.slice(2));
const modelBytes = readFileSync(config.model);
const model = readModel(modelBytes, config.model);
const examples = buildExamples(readRows(config.textIndex), config);
const tokens = examples.flatMap((example) => example.tokens);
const rebuiltTokenHash = fnv64Hex(tokens);
const modelTokenHash = hex64(model.tokenHash);

if (tokens.length !== Number(model.tokenCount)) {
  throw new Error(
    `rebuilt corpus has ${tokens.length} tokens, model was trained on ${model.tokenCount}`,
  );
}
if (rebuiltTokenHash !== modelTokenHash) {
  throw new Error(
    `rebuilt token hash ${rebuiltTokenHash} does not match model token hash ${modelTokenHash}`,
  );
}

const row = summarizeEval({
  model,
  examples,
  modelPath: config.model,
  textIndex: config.textIndex,
  label: config.label,
});
const header = [
  "model",
  "model_path",
  "text_index",
  "eval_scope",
  "examples",
  "token_count",
  "model_hash",
  "token_hash",
  "overall_count",
  "overall_top1_per_mille",
  "overall_top5_per_mille",
  "prompt_count",
  "prompt_top1_per_mille",
  "prompt_top5_per_mille",
  "text_count",
  "text_top1_per_mille",
  "text_top5_per_mille",
  "image_count",
  "image_top1_per_mille",
  "image_top5_per_mille",
  "special_count",
  "special_top1_per_mille",
  "special_top5_per_mille",
  "exact_examples",
  "exact_examples_per_mille",
  "mean_rank_q8",
  "prompt_mean_rank_q8",
  "text_mean_rank_q8",
  "image_mean_rank_q8",
  "context_hit_per_mille",
];
const output = `${header.join("\t")}\n${header.map((column) => row[column]).join("\t")}\n`;
if (config.out === "-") {
  process.stdout.write(output);
} else {
  writeFileSync(config.out, output, "utf8");
  console.log(JSON.stringify({ schema, out: config.out, row }));
}

function parseArgs(args) {
  const parsed = { ...defaults };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--model") {
      parsed.model = requiredValue(args, ++index, arg);
    } else if (arg === "--text-index") {
      parsed.textIndex = requiredValue(args, ++index, arg);
    } else if (arg === "--out") {
      parsed.out = requiredValue(args, ++index, arg);
    } else if (arg === "--label") {
      parsed.label = sanitizeLabel(requiredValue(args, ++index, arg));
    } else if (arg === "--max-text-chars") {
      parsed.maxTextChars = parsePositive(requiredValue(args, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return parsed;
}

function usage() {
  console.log(
    [
      "Usage: run-solomon-multimodal-eval.mjs [--model PATH] [--text-index PATH] [--out PATH]",
      "",
      "Scores the deployed NSRLMOD1 artifact on deterministic corpus replay.",
      "The corpus is rebuilt from the tracked Solomon text/signature TSV and must",
      "match the token hash embedded in the model before any row is emitted.",
    ].join("\n"),
  );
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function parsePositive(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function sanitizeLabel(value) {
  const label = String(value || "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  if (!label) {
    throw new Error("--label must not be empty");
  }
  return label;
}

function readRows(tsvPath) {
  const lines = readFileSync(tsvPath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  for (const column of ["number", "primary_name", "aliases", "text", "signature_16x16"]) {
    if (!header.includes(column)) {
      throw new Error(`${tsvPath} is missing required column ${column}`);
    }
  }
  return lines.filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.number = Number(row.number);
    if (!Number.isInteger(row.number) || row.number < 1 || row.number > 72) {
      throw new Error(`${tsvPath}:${rowIndex + 2} has invalid spirit number`);
    }
    row.signature = parseSignature(row.signature_16x16, tsvPath, rowIndex + 2);
    return row;
  });
}

function parseSignature(value, source, lineNumber) {
  const bins = String(value)
    .split(",")
    .map((part) => Number(part.trim()));
  if (
    bins.length !== SIGNATURE_BINS ||
    bins.some((value) => !Number.isInteger(value) || value < 0 || value > 255)
  ) {
    throw new Error(`${source}:${lineNumber} has invalid ${SIGNATURE_BINS}-bin signature`);
  }
  return bins;
}

function buildExamples(rows, config) {
  if (rows.length !== 72) {
    throw new Error(`expected 72 Solomon rows, found ${rows.length}`);
  }
  const examples = [];
  for (const row of rows) {
    const text = textForRow(row, config.maxTextChars);
    const image = imageTokens(row.signature);
    for (const prompt of promptsForRow(row)) {
      examples.push({
        spiritId: row.number,
        name: normalizeText(row.primary_name),
        prompt,
        tokens: [
          BOS,
          PROMPT,
          ...encodeTextTokens(prompt),
          TEXT,
          ...encodeTextTokens(text),
          IMAGE,
          ...image,
          EOS,
        ],
      });
    }
  }
  return examples;
}

function promptsForRow(row) {
  const aliases = String(row.aliases || "")
    .split("|")
    .map((alias) => normalizeText(alias))
    .filter(Boolean);
  const name = normalizeText(row.primary_name);
  return unique([
    "king solomon seal",
    name,
    `seal of ${name}`,
    `${name} goetic seal`,
    ...aliases.map((alias) => `seal of ${alias}`),
  ]);
}

function textForRow(row, maxTextChars) {
  const name = normalizeText(row.primary_name);
  const selected = selectSentence(row.text) || normalizeText(row.text);
  return truncateText(`Solomon selects ${name}: ${selected}`, maxTextChars);
}

function selectSentence(text) {
  const sentences = normalizeText(text)
    .split(/(?<=[.!?])\s+|;\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 24);
  const needles = [
    "maketh",
    "teaches",
    "teacheth",
    "giveth",
    "gives",
    "declare",
    "discover",
    "causeth",
    "heal",
    "office",
    "appeareth",
    "appears",
  ];
  const scored = sentences.map((sentence, index) => ({
    score: needles.reduce((sum, needle) => sum + (sentence.toLowerCase().includes(needle) ? 1 : 0), 0),
    index,
    sentence,
  }));
  scored.sort((left, right) => right.score - left.score || left.index - right.index);
  return scored[0]?.sentence ?? "";
}

function truncateText(text, maxChars) {
  const compact = normalizeText(text);
  if (compact.length <= maxChars) {
    return compact;
  }
  const clipped = compact.slice(0, maxChars);
  const lastSpace = clipped.lastIndexOf(" ");
  return (lastSpace > 80 ? clipped.slice(0, lastSpace) : clipped).replace(/[,:;]+$/g, "").trim();
}

function normalizeText(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, " - ")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/[^ -~]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function unique(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const normalized = normalizeText(value);
    const key = normalized.toLowerCase();
    if (!normalized || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(normalized);
  }
  return out;
}

function encodeTextTokens(text) {
  return [...normalizeText(text)].map((ch) => TEXT_BASE + Math.min(ch.charCodeAt(0), TEXT_COUNT - 1));
}

function imageTokens(signature) {
  return signature.map((value) => IMAGE_BASE + Math.min(IMAGE_BINS - 1, Math.floor((value * IMAGE_BINS) / 256)));
}

function summarizeEval({ model, examples, modelPath, textIndex, label }) {
  const stats = {
    overall: emptyStats(),
    prompt: emptyStats(),
    text: emptyStats(),
    image: emptyStats(),
    special: emptyStats(),
  };
  let exactExamples = 0;
  let contextHits = 0;
  let scoredTargets = 0;
  for (const example of examples) {
    const markers = markerIndexes(example.tokens);
    let exact = true;
    for (let index = 1; index < example.tokens.length; index += 1) {
      const target = example.tokens[index];
      const category = categoryForIndex(index, target, markers);
      const counts = modelCountsForPosition(model, example.tokens, index);
      if (counts.source === "context") {
        contextHits += 1;
      }
      scoredTargets += 1;
      const rank = rankTarget(counts.next, target, category.allowed);
      record(stats.overall, rank);
      record(stats[category.name], rank);
      if (rank !== 1) {
        exact = false;
      }
    }
    if (exact) {
      exactExamples += 1;
    }
  }
  return {
    model: label,
    model_path: modelPath,
    text_index: textIndex,
    eval_scope: "tracked_corpus_replay",
    examples: examples.length,
    token_count: String(model.tokenCount),
    model_hash: hex64(model.modelHash),
    token_hash: hex64(model.tokenHash),
    overall_count: stats.overall.count,
    overall_top1_per_mille: perMille(stats.overall.top1, stats.overall.count),
    overall_top5_per_mille: perMille(stats.overall.top5, stats.overall.count),
    prompt_count: stats.prompt.count,
    prompt_top1_per_mille: perMille(stats.prompt.top1, stats.prompt.count),
    prompt_top5_per_mille: perMille(stats.prompt.top5, stats.prompt.count),
    text_count: stats.text.count,
    text_top1_per_mille: perMille(stats.text.top1, stats.text.count),
    text_top5_per_mille: perMille(stats.text.top5, stats.text.count),
    image_count: stats.image.count,
    image_top1_per_mille: perMille(stats.image.top1, stats.image.count),
    image_top5_per_mille: perMille(stats.image.top5, stats.image.count),
    special_count: stats.special.count,
    special_top1_per_mille: perMille(stats.special.top1, stats.special.count),
    special_top5_per_mille: perMille(stats.special.top5, stats.special.count),
    exact_examples: exactExamples,
    exact_examples_per_mille: perMille(exactExamples, examples.length),
    mean_rank_q8: meanRankQ8(stats.overall),
    prompt_mean_rank_q8: meanRankQ8(stats.prompt),
    text_mean_rank_q8: meanRankQ8(stats.text),
    image_mean_rank_q8: meanRankQ8(stats.image),
    context_hit_per_mille: perMille(contextHits, scoredTargets),
  };
}

function markerIndexes(tokens) {
  const text = tokens.indexOf(TEXT, 2);
  const image = tokens.indexOf(IMAGE, text + 1);
  if (text < 0 || image < 0) {
    throw new Error("malformed multimodal example");
  }
  return { text, image };
}

function categoryForIndex(index, target, markers) {
  if (target === PROMPT || target === TEXT || target === IMAGE || target === EOS) {
    return { name: "special", allowed: isSpecialToken };
  }
  if (index > 1 && index < markers.text) {
    return { name: "prompt", allowed: isTextToken };
  }
  if (index > markers.text && index < markers.image) {
    return { name: "text", allowed: isTextOrStopToken };
  }
  if (index > markers.image) {
    return { name: "image", allowed: isImageToken };
  }
  throw new Error(`could not categorize token ${target} at ${index}`);
}

function modelCountsForPosition(model, tokens, position) {
  for (const contextLen of contextLengthsForPosition(position).reverse()) {
    const key = contextKey(tokens.slice(position - contextLen, position));
    const row = model.contexts.get(key);
    if (row) {
      return { source: "context", next: row.next };
    }
  }
  return { source: "unigram", next: model.unigram };
}

function contextLengthsForPosition(position) {
  const lengths = [];
  if (position > 0 && position <= MAX_CONTEXT_TOKENS) {
    lengths.push(position);
  }
  for (const length of CONTEXT_LENGTHS) {
    if (length <= position && !lengths.includes(length)) {
      lengths.push(length);
    }
  }
  return lengths.sort((left, right) => left - right);
}

function rankTarget(counts, target, allowed) {
  const candidates = counts
    .filter((entry) => allowed(entry.token))
    .sort((left, right) => right.count - left.count || left.token - right.token);
  const index = candidates.findIndex((entry) => entry.token === target);
  return index >= 0 ? index + 1 : candidates.length + 1;
}

function record(stats, rank) {
  stats.count += 1;
  stats.rankTotal += rank;
  if (rank <= 1) {
    stats.top1 += 1;
  }
  if (rank <= 5) {
    stats.top5 += 1;
  }
}

function emptyStats() {
  return { count: 0, top1: 0, top5: 0, rankTotal: 0 };
}

function perMille(value, total) {
  return total > 0 ? Math.floor((value * 1000) / total) : 0;
}

function meanRankQ8(stats) {
  return stats.count > 0 ? Math.floor((stats.rankTotal * 256) / stats.count) : 0;
}

function readModel(bytes, filePath) {
  const expectedHash = readU64At(bytes.length - 8);
  const actualHash = hashBytes(bytes.subarray(0, bytes.length - 8));
  if (expectedHash !== actualHash) {
    throw new Error(`${filePath} hash mismatch: expected ${hex64(expectedHash)}, got ${hex64(actualHash)}`);
  }
  let offset = 0;
  function readBytes(count) {
    if (offset + count > bytes.length) {
      throw new Error(`${filePath} ended unexpectedly`);
    }
    const out = bytes.subarray(offset, offset + count);
    offset += count;
    return out;
  }
  function readU32() {
    const value = bytes.readUInt32LE(offset);
    offset += 4;
    return value;
  }
  function readU64() {
    const value = readU64At(offset);
    offset += 8;
    return value;
  }
  function readU64At(position) {
    return bytes.readBigUInt64LE(position);
  }
  function expectU32(expected, label) {
    const actual = readU32();
    if (actual !== expected) {
      throw new Error(`${filePath} ${label} mismatch: expected ${expected}, got ${actual}`);
    }
  }
  function readCountList() {
    const count = readU32();
    const out = [];
    for (let index = 0; index < count; index += 1) {
      const token = readU32();
      const count = readU32();
      if (token < 0 || token >= VOCAB_SIZE) {
        throw new Error(`${filePath} has token ${token} outside NSRLMOD1 vocab`);
      }
      out.push({ token, count });
    }
    return out;
  }

  if (readBytes(8).toString("ascii") !== MODEL_MAGIC) {
    throw new Error(`${filePath} is not an NSRLMOD1 model`);
  }
  expectU32(MODEL_VERSION, "version");
  expectU32(VOCAB_SIZE, "vocab size");
  expectU32(TEXT_BASE, "text base");
  expectU32(TEXT_COUNT, "text count");
  expectU32(IMAGE_BASE, "image base");
  expectU32(IMAGE_BINS, "image bins");
  expectU32(SIGNATURE_GRID, "signature grid");
  const tokenCount = readU64();
  const tokenHash = readU64();
  readU32();
  const unigram = readCountList();
  expectU32(MAX_CONTEXT_TOKENS, "max context tokens");
  const contextCount = readU32();
  const contexts = new Map();
  for (let rowIndex = 0; rowIndex < contextCount; rowIndex += 1) {
    const contextLength = readU32();
    const context = [];
    for (let tokenIndex = 0; tokenIndex < contextLength; tokenIndex += 1) {
      context.push(readU32());
    }
    readU32();
    contexts.set(contextKey(context), { context, next: readCountList() });
  }
  if (offset !== bytes.length - 8) {
    throw new Error(`${filePath} has ${bytes.length - 8 - offset} trailing bytes`);
  }
  return { tokenCount, tokenHash, modelHash: expectedHash, unigram, contexts };
}

function isTextToken(token) {
  return token >= TEXT_BASE && token < TEXT_BASE + TEXT_COUNT;
}

function isImageToken(token) {
  return token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS;
}

function isTextOrStopToken(token) {
  return isTextToken(token) || token === IMAGE || token === EOS;
}

function isSpecialToken(token) {
  return token === BOS || token === PROMPT || token === TEXT || token === IMAGE || token === EOS || token === PAD;
}

function contextKey(tokens) {
  return tokens.join(",");
}

function fnv64Hex(tokens) {
  let state = FNV_OFFSET;
  for (const token of tokens) {
    state ^= BigInt(token & 0xff);
    state = (state * FNV_PRIME) & FNV_MASK;
    state ^= BigInt((token >> 8) & 0xff);
    state = (state * FNV_PRIME) & FNV_MASK;
  }
  return hex64(state);
}

function hashBytes(bytes) {
  let state = FNV_OFFSET;
  for (const byte of bytes) {
    state ^= BigInt(byte);
    state = (state * FNV_PRIME) & FNV_MASK;
  }
  return state;
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}
