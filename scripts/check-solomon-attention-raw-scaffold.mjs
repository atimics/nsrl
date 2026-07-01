#!/usr/bin/env node
import { mkdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

let modelPath = "web/assets/solomon-attention.nsrllmm";
let textIndexPath = "web/assets/solomon-spirit-text-signatures.tsv";
let binPath = "target/release/nsrl-solomon-attention";
let outDir = "/tmp/nsrl-solomon-attention-raw-scaffold";
let summary = false;

const args = process.argv.slice(2);
for (let index = 0; index < args.length; index += 1) {
  const arg = args[index];
  if (arg === "--model") {
    modelPath = requiredValue(args, ++index, arg);
  } else if (arg === "--text-index") {
    textIndexPath = requiredValue(args, ++index, arg);
  } else if (arg === "--bin") {
    binPath = requiredValue(args, ++index, arg);
  } else if (arg === "--out-dir") {
    outDir = requiredValue(args, ++index, arg);
  } else if (arg === "--summary") {
    summary = true;
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}

const rows = allNameRows(textIndexPath);
const misses = [];
for (const row of rows) {
  const name = normalizeText(row.primaryName);
  const sampleDir = path.join(outDir, safePathPart(name));
  mkdirSync(sampleDir, { recursive: true });
  const run = spawnSync(
    binPath,
    [
      "sample",
      "--model",
      modelPath,
      "--out-dir",
      sampleDir,
      "--prompt",
      `seal of ${row.primaryName}`,
      "--max-text-tokens",
      "80",
      "--min-text-tokens",
      "24",
      "--repeat-run-cap",
      "3",
      "--no-repeat-ngram",
      "3",
      "--conditioning-examples",
      "none",
      "--text-prior-examples",
      "none",
      "--no-embedded-text-memory",
      "--prompt-name-opening-prior",
      "--sample-seed",
      "1",
    ],
    { encoding: "utf8" },
  );
  if (run.status !== 0) {
    misses.push(`${row.primaryName}:command_failed`);
    continue;
  }
  const text = readFileSync(path.join(sampleDir, "text.txt"), "utf8").trimEnd();
  const expected = `Solomon selects ${name}: He is of the Goetia and teacheth with his ART in LINE.`;
  if (text !== expected) {
    misses.push(`${row.primaryName}:${text}`);
  }
}

const result = {
  schema: "nsrl.solomon_attention_raw_scaffold_summary.v1",
  model: modelPath,
  prompts: rows.length,
  passed: rows.length - misses.length,
  misses,
};
console.log(JSON.stringify(result));
if (!summary && misses.length > 0) {
  for (const miss of misses) {
    console.error(miss);
  }
}
if (misses.length > 0) {
  process.exit(1);
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
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
