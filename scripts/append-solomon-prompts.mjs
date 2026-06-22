#!/usr/bin/env node
import fs from "node:fs";

const defaults = {
  prompts: "data/processed/key-solomon-goetia-latent-v1/prompts.jsonl",
  out: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  fromGrounded: "",
  splitSeed: "solomon-prompt-split-v1",
  inPlace: false,
};

const schema = "nsrl.solomon_prompt.v1";

function usage() {
  console.log(
    [
      "Usage: append-solomon-prompts.mjs --from-grounded PATH [--prompts PATH] [--out PATH]",
      "       [--in-place] [--split-seed TEXT]",
      "",
      "Appends source-grounded text/signature TSV rows to a Solomon prompt JSONL",
      "without rewriting existing prompt rows or changing their stable hashes.",
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
    } else if (arg === "--prompts") {
      config.prompts = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.out = requireValue(argv, ++index, arg);
    } else if (arg === "--from-grounded") {
      config.fromGrounded = requireValue(argv, ++index, arg);
    } else if (arg === "--split-seed") {
      config.splitSeed = requireValue(argv, ++index, arg);
    } else if (arg === "--in-place") {
      config.inPlace = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.fromGrounded) {
    throw new Error("--from-grounded is required");
  }
  if (config.inPlace) {
    config.out = config.prompts;
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readPrompts(filePath, splitSeed) {
  const text = fs.readFileSync(filePath, "utf8");
  return text.split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    if (row.schema !== schema) {
      throw new Error(`${filePath} line ${index + 1} has unsupported schema`);
    }
    const hash = promptHash(splitSeed, row.text);
    if (row.prompt_hash !== hash.hex || row.bucket !== hash.value % 1000) {
      throw new Error(`${filePath} line ${index + 1} has unstable prompt hash or bucket`);
    }
    return row;
  });
}

function readGroundedRows(tsvPath) {
  const text = fs.readFileSync(tsvPath, "utf8");
  const lines = text.trimEnd().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error(`${tsvPath} has no data rows`);
  }
  const header = lines[0].split("\t");
  const required = ["number", "text", "variant_id", "source_lanes", "prompt_kind"];
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
    return row;
  });
}

function promptFromGrounded(row, splitSeed) {
  const text = compactText(row.text);
  const hash = promptHash(splitSeed, text);
  const tier = tierForPromptKind(row.prompt_kind);
  return {
    schema,
    spirit_id: row.number,
    text,
    source: "generated",
    bucket: hash.value % 1000,
    tier,
    cluster: clusterForGrounded(row, tier),
    prompt_hash: hash.hex,
    grounded_variant_id: row.variant_id,
    source_lanes: row.source_lanes,
    prompt_kind: row.prompt_kind,
    support_terms: row.support_terms ?? "",
  };
}

function tierForPromptKind(kind) {
  if (kind === "canonical" || kind === "alias-prompt") {
    return "tier-paraphrase";
  }
  if (kind.startsWith("source-") || kind.endsWith("-variant")) {
    return "tier-cluster-holdout";
  }
  return "tier-novel-vocab";
}

function clusterForGrounded(row, tier) {
  if (tier === "tier-cluster-holdout") {
    return `${row.number}:${row.prompt_kind}:${row.source_lanes || "goetia"}`;
  }
  return `${row.number}:${row.prompt_kind}`;
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

function compactText(text) {
  return String(text)
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const existing = readPrompts(config.prompts, config.splitSeed);
  const seen = new Set(existing.map((row) => row.prompt_hash));
  const appended = [];
  for (const row of readGroundedRows(config.fromGrounded)) {
    const prompt = promptFromGrounded(row, config.splitSeed);
    if (seen.has(prompt.prompt_hash)) {
      continue;
    }
    seen.add(prompt.prompt_hash);
    appended.push(prompt);
  }
  const merged = [...existing, ...appended];
  fs.mkdirSync(dirname(config.out), { recursive: true });
  fs.writeFileSync(config.out, `${merged.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  console.log(JSON.stringify({
    schema,
    prompts_in: config.prompts,
    grounded_in: config.fromGrounded,
    prompts_out: config.out,
    existing: existing.length,
    appended: appended.length,
    total: merged.length,
  }));
}

function dirname(filePath) {
  const index = filePath.lastIndexOf("/");
  return index === -1 ? "." : filePath.slice(0, index);
}

main();
