#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { SolomonAttentionSampler } from "../web/attention-sampler.js";

const args = process.argv.slice(2);
let modelPath = "web/assets/solomon-attention.nsrllmm";
let textIndexPath = "web/assets/solomon-spirit-text-signatures.tsv";
let textPrefix = "Solomon selects ";
let topN = 10;
let allNames = false;
let summary = false;
const prompts = [];

for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === "--model") {
    modelPath = requiredValue(args, ++index, arg);
  } else if (arg === "--text-index") {
    textIndexPath = requiredValue(args, ++index, arg);
  } else if (arg === "--text-prefix") {
    textPrefix = requiredValue(args, ++index, arg);
  } else if (arg === "--top-n") {
    topN = Number(requiredValue(args, ++index, arg));
  } else if (arg === "--prompt") {
    prompts.push(requiredValue(args, ++index, arg));
  } else if (arg === "--all-names") {
    allNames = true;
  } else if (arg === "--summary") {
    summary = true;
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}

if (allNames) {
  prompts.push(...allNamePrompts(textIndexPath));
} else if (prompts.length === 0) {
  prompts.push("seal of Bael", "seal of Stolas", "seal of Marbas");
}

const sampler = new SolomonAttentionSampler(readFileSync(modelPath));
const rows = [];
for (const prompt of prompts) {
  rows.push(
    sampler.diagnoseMemoryContinuation(prompt, {
      seed: 13,
      textPrefix,
      topN,
    }),
  );
}

if (summary) {
  console.log(JSON.stringify(rankSummary(rows, { modelPath, textPrefix })));
} else {
  for (const row of rows) {
    console.log(JSON.stringify(row));
  }
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function allNamePrompts(tsvPath) {
  const lines = readFileSync(tsvPath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  const primaryNameIndex = header.indexOf("primary_name");
  if (primaryNameIndex < 0) {
    throw new Error(`${tsvPath} is missing primary_name`);
  }
  return lines
    .filter(Boolean)
    .map((line) => line.split("\t")[primaryNameIndex])
    .filter(Boolean)
    .map((name) => `seal of ${name}`);
}

function rankSummary(rows, context) {
  const ranked = rows.filter((row) => row.memoryFound && row.expectedRank !== null);
  const ranks = ranked.map((row) => row.expectedRank).sort((left, right) => left - right);
  const margins = ranked
    .map((row) => row.expectedMarginQ8)
    .filter((value) => value !== null)
    .sort((left, right) => left - right);
  const meanRank =
    ranks.length === 0 ? null : ranks.reduce((sum, rank) => sum + rank, 0) / ranks.length;
  return {
    schema: "nsrl.solomon_attention_raw_rank_summary.v1",
    model: context.modelPath,
    textPrefix: context.textPrefix,
    prompts: rows.length,
    ranked: ranked.length,
    top1: ranks.filter((rank) => rank <= 1).length,
    top5: ranks.filter((rank) => rank <= 5).length,
    top10: ranks.filter((rank) => rank <= 10).length,
    medianRank: percentile(ranks, 0.5),
    meanRank: meanRank === null ? null : Number(meanRank.toFixed(2)),
    worstRank: ranks.at(-1) ?? null,
    medianMarginQ8: percentile(margins, 0.5),
    worstMarginQ8: margins[0] ?? null,
    misses: ranked
      .filter((row) => row.expectedRank > 10)
      .slice(0, 12)
      .map((row) => `${row.primaryName}:${row.expectedRank}`),
  };
}

function percentile(values, ratio) {
  if (values.length === 0) {
    return null;
  }
  const index = Math.min(values.length - 1, Math.max(0, Math.floor(values.length * ratio)));
  return values[index];
}
