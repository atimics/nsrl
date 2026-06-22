#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  textIndex: "data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv",
  outDir: "data/processed/key-solomon-goetia-latent-v1",
  splitSeed: "solomon-prompt-split-v1",
  goldPerSpirit: 2,
};

const schema = "nsrl.solomon_prompt.v1";

function usage() {
  console.log(
    [
      "Usage: build-solomon-prompt-corpus.mjs [--text-index PATH] [--out-dir PATH]",
      "       [--split-seed TEXT] [--gold-per-spirit N]",
      "",
      "Builds the seed Solomon prompt JSONL and frozen gold hash TSV from the",
      "existing PG72679 spirit text/signature index.",
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
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--split-seed") {
      config.splitSeed = requireValue(argv, ++index, arg);
    } else if (arg === "--gold-per-spirit") {
      config.goldPerSpirit = parsePositive(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
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
  if (!/^[0-9]+$/.test(value) || Number(value) === 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function readRows(tsvPath) {
  const text = fs.readFileSync(tsvPath, "utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error(`${tsvPath} has no data rows`);
  }
  const header = lines[0].split("\t");
  const required = ["number", "primary_name", "aliases", "text"];
  for (const column of required) {
    if (!header.includes(column)) {
      throw new Error(`${tsvPath} missing required column ${column}`);
    }
  }
  return lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.number = Number(row.number);
    if (!Number.isInteger(row.number) || row.number < 1 || row.number > 72) {
      throw new Error(`${tsvPath} row ${rowIndex + 2} has invalid number`);
    }
    row.aliases_text = String(row.aliases || "").replace(/\|/g, " ");
    return row;
  });
}

function promptsForRow(row, splitSeed) {
  const facts = factsForText(row.text);
  const keywordText = keywords(row.text, row).slice(0, 10).join(" ");
  const rank = facts.rank || "Goetic spirit";
  const office = facts.office || `${row.primary_name} has an office described in the Goetia`;
  const appearance = facts.appearance || `${row.primary_name} has a distinct Goetic appearance`;
  return [
    prompt(row, splitSeed, {
      source: "canonical",
      tier: "tier-paraphrase",
      cluster: `${row.number}:canonical`,
      text: `${row.primary_name}. ${row.text}`,
    }),
    prompt(row, splitSeed, {
      source: "canonical",
      tier: "tier-paraphrase",
      cluster: `${row.number}:office`,
      text: `${row.primary_name} ${row.aliases_text}. ${office}`,
    }),
    prompt(row, splitSeed, {
      source: "epithet",
      tier: "tier-novel-vocab",
      cluster: `${row.number}:keywords`,
      text: `${row.primary_name}: ${rank}. Keywords: ${keywordText}.`,
    }),
    prompt(row, splitSeed, {
      source: "epithet",
      tier: "tier-cluster-holdout",
      cluster: `${row.number}:appearance`,
      text: `${row.primary_name}: ${appearance}`,
    }),
  ];
}

function prompt(row, splitSeed, fields) {
  const text = compactText(fields.text);
  const hash = promptHash(splitSeed, text);
  return {
    schema,
    spirit_id: row.number,
    text,
    source: fields.source,
    bucket: hash.value % 1000,
    tier: fields.tier,
    cluster: fields.cluster,
    prompt_hash: hash.hex,
  };
}

function factsForText(text) {
  const sentences = splitSentences(text);
  return {
    rank: cleanFact(firstMatch(text, [
      /\b(?:is|called)\s+(?:a|an)\s+([^.;,]{0,80}?\b(?:king|duke|prince|marquis|president|earl|count|knight)\b)/i,
      /\b(?:is|called)\s+([^.;,]{0,80}?\b(?:king|duke|prince|marquis|president|earl|count|knight)\b)/i,
    ])),
    office: selectSentence(sentences, [
      "office",
      "maketh",
      "teaches",
      "teacheth",
      "giveth",
      "gives",
      "declare",
      "discover",
      "causeth",
      "bringeth",
      "heal",
      "languages",
      "knowledge",
      "invisible",
    ]),
    appearance: selectSentence(sentences, [
      "appeareth",
      "appears",
      "form",
      "shape",
      "head",
      "voice",
      "riding",
      "carrying",
    ]),
  };
}

function splitSentences(text) {
  return compactText(text)
    .split(/(?<=[.!?])\s+|;\s+|\s+--\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 20 && sentence.length < 360);
}

function firstMatch(text, regexes) {
  for (const regex of regexes) {
    const match = text.match(regex);
    if (match) {
      return match[1] || match[0];
    }
  }
  return "";
}

function cleanFact(text) {
  return compactText(text)
    .replace(/^a\s+/i, "")
    .replace(/[.,;:]+$/g, "")
    .slice(0, 160);
}

function selectSentence(sentences, needles) {
  const scored = [];
  for (const sentence of sentences) {
    const folded = foldText(sentence).toLowerCase();
    let score = 0;
    for (const needle of needles) {
      if (folded.includes(foldText(needle).toLowerCase())) {
        score += 1;
      }
    }
    if (score > 0) {
      scored.push({ score, sentence: compactText(sentence) });
    }
  }
  scored.sort((left, right) => right.score - left.score || left.sentence.length - right.sentence.length);
  return scored[0]?.sentence ?? "";
}

function keywords(text, row) {
  const stop = new Set([
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "shall",
    "which",
    "spirit",
    "spirits",
    "called",
    "named",
    "unto",
    "upon",
    "before",
    "after",
    "their",
    "there",
    "them",
    "they",
    "seal",
    "character",
    "goetia",
  ]);
  const aliasFolded = new Set(
    [row.primary_name, ...String(row.aliases || "").split("|")]
      .map((name) => foldText(name).trim().toLowerCase())
      .filter(Boolean),
  );
  const counts = new Map();
  for (const token of foldText(text).toLowerCase().split(/\s+/)) {
    if (token.length < 4 || stop.has(token) || aliasFolded.has(token)) {
      continue;
    }
    counts.set(token, (counts.get(token) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([token]) => token);
}

function foldText(text) {
  return String(text)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/æ/g, "ae")
    .replace(/Æ/g, "Ae")
    .replace(/œ/g, "oe")
    .replace(/Œ/g, "Oe")
    .replace(/[^A-Za-z0-9]+/g, " ");
}

function compactText(text) {
  return String(text)
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function promptHash(seed, text) {
  const value = stableHash([seed, text]);
  return { value, hex: value.toString(16).padStart(8, "0") };
}

function stableHash(parts) {
  let hash = 2166136261 >>> 0;
  for (const part of parts) {
    for (const byte of Buffer.from(part, "utf8")) {
      hash ^= byte;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    hash ^= 255;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return (hash | 1) >>> 0;
}

function escapeTsv(value) {
  return String(value ?? "")
    .replace(/\t/g, " ")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const rows = readRows(config.textIndex);
  const prompts = rows.flatMap((row) => promptsForRow(row, config.splitSeed));
  fs.mkdirSync(config.outDir, { recursive: true });
  const promptsPath = path.join(config.outDir, "prompts.jsonl");
  fs.writeFileSync(promptsPath, `${prompts.map((item) => JSON.stringify(item)).join("\n")}\n`, "utf8");

  const goldRows = [];
  for (const row of rows) {
    for (const item of prompts.filter((prompt) => prompt.spirit_id === row.number).slice(0, config.goldPerSpirit)) {
      goldRows.push(item);
    }
  }
  const goldPath = path.join(config.outDir, "gold.tsv");
  const goldTsv = [
    "prompt_hash\tspirit_id\tsource\ttier\ttext",
    ...goldRows.map((item) =>
      [
        item.prompt_hash,
        item.spirit_id,
        item.source,
        item.tier,
        item.text,
      ].map(escapeTsv).join("\t"),
    ),
  ].join("\n");
  fs.writeFileSync(goldPath, `${goldTsv}\n`, "utf8");

  console.log(JSON.stringify({
    schema,
    prompts: promptsPath,
    gold: goldPath,
    rows: rows.length,
    prompts_count: prompts.length,
    gold_count: goldRows.length,
    split_seed: config.splitSeed,
  }));
}

main();
