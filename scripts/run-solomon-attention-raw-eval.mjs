#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const defaults = {
  model: "web/assets/solomon-attention.nsrllmm",
  textIndex: "web/assets/solomon-spirit-text-signatures.tsv",
  bin: "target/release/nsrl-solomon-attention",
  outDir: "data/processed/key-solomon-goetia-multimodal-v1/attention-raw-eval",
  out: "docs/solomon-attention-eval.tsv",
  maxPrompts: 72,
  label: "raw-no-memory",
};

const schema = "nsrl.solomon_attention_raw_eval.v1";
const config = parseArgs(process.argv.slice(2));

if (!existsSync(config.bin)) {
  throw new Error(`missing attention binary: ${config.bin}`);
}

const rows = allNameRows(config.textIndex).slice(0, config.maxPrompts);
mkdirSync(config.outDir, { recursive: true });

let generated = 0;
let promptNameMatches = 0;
let scaffoldOutputs = 0;
let charTotal = 0;
let scoreTotal = 0;
let minScore = null;
let modelHash = "";
let samplePrompt = "";
let sampleOutput = "";
const distinctTexts = new Set();

for (const row of rows) {
  const prompt = `seal of ${row.primaryName}`;
  const sampleDir = path.join(config.outDir, safePathPart(row.primaryName));
  mkdirSync(sampleDir, { recursive: true });
  const run = spawnSync(
    config.bin,
    [
      "sample",
      "--model",
      config.model,
      "--out-dir",
      sampleDir,
      "--prompt",
      prompt,
      "--min-text-tokens",
      "24",
      "--max-text-tokens",
      "80",
      "--repeat-run-cap",
      "3",
      "--no-repeat-ngram",
      "3",
      "--conditioning-examples",
      "none",
      "--text-prior-examples",
      "none",
      "--no-embedded-text-memory",
      "--top-k",
      "1",
      "--sample-seed",
      "1",
    ],
    { encoding: "utf8" },
  );
  if (run.status !== 0) {
    throw new Error(`sample failed for ${row.primaryName}: ${run.stderr || run.stdout}`);
  }
  const sample = JSON.parse(readFileSync(path.join(sampleDir, "sample.json"), "utf8"));
  modelHash ||= sample.model_hash || "";
  const text = readFileSync(path.join(sampleDir, "text.txt"), "utf8").trimEnd();
  if (!samplePrompt) {
    samplePrompt = prompt;
    sampleOutput = text;
  }
  generated += text ? 1 : 0;
  distinctTexts.add(text);
  charTotal += [...text].length;
  if (startsWithExpectedName(text, row.primaryName)) {
    promptNameMatches += 1;
  }
  if (/^---?He is of the Goetia and teacheth with his ART in LINE\.$/.test(text)) {
    scaffoldOutputs += 1;
  }
  const quality = spawnSync(
    process.execPath,
    [
      "scripts/check-solomon-attention-raw-quality.mjs",
      "--text",
      path.join(sampleDir, "text.txt"),
      "--prompt",
      prompt,
      "--label",
      "raw",
    ],
    { encoding: "utf8" },
  );
  if (quality.status !== 0) {
    throw new Error(`quality check failed for ${row.primaryName}: ${quality.stderr || quality.stdout}`);
  }
  const score = Number(metricFromLine(quality.stdout.trim(), "raw_score"));
  if (!Number.isFinite(score)) {
    throw new Error(`quality output missing raw_score for ${row.primaryName}`);
  }
  scoreTotal += score;
  minScore = minScore === null ? score : Math.min(minScore, score);
}

const row = {
  model: config.label,
  model_path: config.model,
  eval_scope: "native_raw_no_memory_no_conditioning",
  prompts: rows.length,
  generated,
  prompt_name_matches: promptNameMatches,
  prompt_name_match_per_mille: perMille(promptNameMatches, rows.length),
  mean_raw_quality_score: rows.length ? Math.round(scoreTotal / rows.length) : 0,
  min_raw_quality_score: minScore ?? 0,
  distinct_texts: distinctTexts.size,
  scaffold_outputs: scaffoldOutputs,
  scaffold_output_per_mille: perMille(scaffoldOutputs, rows.length),
  mean_chars: rows.length ? Math.round(charTotal / rows.length) : 0,
  model_hash: modelHash,
  sample_prompt: samplePrompt,
  sample_output: sampleOutput,
};

const header = [
  "model",
  "model_path",
  "eval_scope",
  "prompts",
  "generated",
  "prompt_name_matches",
  "prompt_name_match_per_mille",
  "mean_raw_quality_score",
  "min_raw_quality_score",
  "distinct_texts",
  "scaffold_outputs",
  "scaffold_output_per_mille",
  "mean_chars",
  "model_hash",
  "sample_prompt",
  "sample_output",
];
const output = `${header.join("\t")}\n${header.map((column) => tsvCell(row[column])).join("\t")}\n`;
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
    } else if (arg === "--bin") {
      parsed.bin = requiredValue(args, ++index, arg);
    } else if (arg === "--out-dir") {
      parsed.outDir = requiredValue(args, ++index, arg);
    } else if (arg === "--out") {
      parsed.out = requiredValue(args, ++index, arg);
    } else if (arg === "--label") {
      parsed.label = sanitizeLabel(requiredValue(args, ++index, arg));
    } else if (arg === "--max-prompts") {
      parsed.maxPrompts = parsePositive(requiredValue(args, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return parsed;
}

function usage() {
  console.log(
    [
      "Usage: run-solomon-attention-raw-eval.mjs [--model PATH] [--text-index PATH]",
      "       [--bin PATH] [--out PATH] [--out-dir PATH]",
      "",
      "Samples NSRLLMM1 with conditioning, text prior, and embedded text memory disabled,",
      "then scores prompt-name binding and raw text quality. This is a negative-control",
      "free-running check, not the browser-path artifact probe.",
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

function allNameRows(tsvPath) {
  const lines = readFileSync(tsvPath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  const primaryNameIndex = header.indexOf("primary_name");
  if (primaryNameIndex < 0) {
    throw new Error(`${tsvPath} is missing primary_name`);
  }
  return lines
    .filter(Boolean)
    .map((line) => ({ primaryName: line.split("\t")[primaryNameIndex] }))
    .filter((row) => row.primaryName);
}

function startsWithExpectedName(text, name) {
  return normalizeText(text).startsWith(`Solomon selects ${normalizeText(name)}:`);
}

function normalizeText(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x20-\x7e]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function safePathPart(value) {
  return normalizeText(value).replace(/[^A-Za-z0-9_-]/g, "_");
}

function metricFromLine(line, key) {
  const match = line.match(new RegExp(`(?:^|\\s)${key}=([^\\s]+)`));
  return match?.[1] ?? "";
}

function perMille(value, total) {
  return total > 0 ? Math.floor((value * 1000) / total) : 0;
}

function tsvCell(value) {
  return String(value ?? "").replace(/[\t\r\n]+/g, " ");
}
