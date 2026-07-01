#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { SolomonAttentionSampler } from "../web/attention-sampler.js";

let modelPath = "web/assets/solomon-attention.nsrllmm";
let topN = 10;
let summary = false;
let minTop1 = null;
let minTop10 = null;
let minTop5 = null;

const args = process.argv.slice(2);
for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === "--model") {
    modelPath = requiredValue(args, ++index, arg);
  } else if (arg === "--top-n") {
    topN = Number(requiredValue(args, ++index, arg));
  } else if (arg === "--summary") {
    summary = true;
  } else if (arg === "--min-top1") {
    minTop1 = Number(requiredValue(args, ++index, arg));
  } else if (arg === "--min-top10") {
    minTop10 = Number(requiredValue(args, ++index, arg));
  } else if (arg === "--min-top5") {
    minTop5 = Number(requiredValue(args, ++index, arg));
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}

const sampler = new SolomonAttentionSampler(readFileSync(modelPath));
const rows = [];
for (const example of sampler.textMemory?.examples || []) {
  const primaryName = normalizeNameForText(example.primaryName);
  rows.push(
    sampler.diagnoseMemoryContinuation(`seal of ${example.primaryName}`, {
      seed: 13,
      textPrefix: `Solomon selects ${primaryName}: `,
      topN,
    }),
  );
}

if (summary) {
  const result = rankSummary(rows, { modelPath });
  console.log(JSON.stringify(result));
  if (minTop1 !== null && result.top1 < minTop1) {
    console.error(`body-start top1 ${result.top1} < ${minTop1}`);
    process.exit(1);
  }
  if (minTop5 !== null && result.top5 < minTop5) {
    console.error(`body-start top5 ${result.top5} < ${minTop5}`);
    process.exit(1);
  }
  if (minTop10 !== null && result.top10 < minTop10) {
    console.error(`body-start top10 ${result.top10} < ${minTop10}`);
    process.exit(1);
  }
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
    schema: "nsrl.solomon_attention_body_start_rank_summary.v1",
    model: context.modelPath,
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
    top1Misses: ranked
      .filter((row) => row.expectedRank > 1)
      .slice(0, 12)
      .map((row) => `${row.primaryName}:${row.expectedToken}:${row.expectedRank}`),
    misses: ranked
      .filter((row) => row.expectedRank > 10)
      .slice(0, 12)
      .map((row) => `${row.primaryName}:${row.expectedToken}:${row.expectedRank}`),
  };
}

function percentile(values, ratio) {
  if (values.length === 0) {
    return null;
  }
  const index = Math.min(values.length - 1, Math.max(0, Math.floor(values.length * ratio)));
  return values[index];
}

function normalizeNameForText(name) {
  return String(name || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "");
}
