#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = {
  input: "data/processed/signal-replay-corpus/training-pairs.jsonl",
  outDir: "",
  model: "gemma4:latest",
  domain: "auto",
  variantsPerState: 1,
  limit: 24,
  offset: 0,
  attempts: 3,
  temperature: 0.72,
  topP: 0.88,
  seed: "ollama-state-output-v1",
  endpoint: "http://127.0.0.1:11434/api/generate",
  timeoutMs: 120000,
  keepExisting: "true",
};

const domainProfiles = {
  signal: {
    label: "Signal radio",
    maxChars: 110,
    numPredict: 36,
    system: [
      "Write raw in-world Signal pilot radio chatter.",
      "Return one short radio line only.",
      "No labels, no JSON, no markdown, no quotes, no explanations.",
      "Never mention AI, model, prompt, training, private_state, expected_output, or ranking.",
      "Keep it clipped, grounded, and operational.",
      "Use callsigns, station names, and vector names exactly as they appear; do not expand N0 into words or invent units.",
    ].join(" "),
    task: [
      "State:",
      "{state}",
      "",
      "Reference tone:",
      "{reference}",
      "",
      "Write one fresh radio line, 5 to 16 words. Mention only facts implied by the state.",
    ].join("\n"),
  },
  cosyworld: {
    label: "CosyWorld narration",
    maxChars: 190,
    numPredict: 56,
    system: [
      "Write raw in-world CosyWorld narration.",
      "Return one warm sentence only.",
      "No labels, no JSON, no markdown, no quotes, no explanations.",
      "Never mention AI, model, prompt, training, private_state, expected_output, or ranking.",
      "Keep it whimsical, grounded in the state, and physically specific.",
      "Do not use second person unless the state itself uses second person.",
    ].join(" "),
    task: [
      "State:",
      "{state}",
      "",
      "Reference tone:",
      "{reference}",
      "",
      "Write one fresh cosy narration sentence, 8 to 24 words. Use only characters, places, items, and actions implied by the state.",
    ].join("\n"),
  },
};

function usage() {
  console.log(`Usage: node scripts/generate-state-outputs-ollama.mjs [options]

Options:
  --input PATH              Source training-pairs JSONL [${defaults.input}]
  --out-dir PATH            Output directory [auto from input/model]
  --model NAME              Ollama model [${defaults.model}]
  --domain NAME             auto, signal, cosyworld [${defaults.domain}]
  --variants-per-state N    Drafts to keep per input row [${defaults.variantsPerState}]
  --limit N                 Max input rows to read; 0 means all [${defaults.limit}]
  --offset N                Input rows to skip first [${defaults.offset}]
  --attempts N              Retry attempts per variant [${defaults.attempts}]
  --temperature N           Ollama temperature [${defaults.temperature}]
  --top-p N                 Ollama top_p [${defaults.topP}]
  --seed TEXT               Stable run seed [${defaults.seed}]
  --endpoint URL            Ollama generate endpoint [${defaults.endpoint}]
  --timeout-ms N            Per-request timeout [${defaults.timeoutMs}]
  --keep-existing BOOL      Reuse cache rows when present [${defaults.keepExisting}]
`);
}

function parseArgs(argv) {
  const options = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (!(key in options)) {
      throw new Error(`unknown option: ${arg}`);
    }
    const value = argv[++index];
    if (value === undefined) {
      throw new Error(`${arg} requires a value`);
    }
    if (["variantsPerState", "limit", "offset", "attempts", "timeoutMs"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 0) {
        throw new Error(`${arg} requires a non-negative integer`);
      }
    } else if (["temperature", "topP"].includes(key)) {
      options[key] = Number.parseFloat(value);
      if (!Number.isFinite(options[key]) || options[key] < 0) {
        throw new Error(`${arg} requires a non-negative number`);
      }
    } else {
      options[key] = value;
    }
  }
  if (options.variantsPerState < 1) {
    throw new Error("--variants-per-state must be at least 1");
  }
  if (options.attempts < 1) {
    throw new Error("--attempts must be at least 1");
  }
  return options;
}

function parseBool(value, optionName) {
  const text = String(value).toLowerCase();
  if (["1", "true", "yes", "on"].includes(text)) return true;
  if (["0", "false", "no", "off"].includes(text)) return false;
  throw new Error(`${optionName} must be true or false`);
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function sanitizePathPart(text) {
  return String(text)
    .toLowerCase()
    .replace(/:latest$/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "model";
}

function defaultOutDir(options) {
  const input = resolveRepoPath(options.input);
  const parent = path.basename(path.dirname(input)).replace(/-corpus$/, "");
  return path.join("data/processed/ollama-state-outputs", `${parent}-${sanitizePathPart(options.model)}`);
}

function cleanAscii(text) {
  return String(text ?? "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanOneLine(text) {
  return String(text ?? "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function readJsonl(filePath) {
  const resolved = resolveRepoPath(filePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`missing JSONL file: ${resolved}`);
  }
  return fs.readFileSync(resolved, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${resolved}:${index + 1}: ${error.message}`);
      }
    });
}

function appendJsonl(filePath, row) {
  fs.appendFileSync(filePath, `${JSON.stringify(row)}\n`, "utf8");
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), "utf8");
}

function inferDomain(row, options) {
  if (options.domain !== "auto") return options.domain;
  const domain = cleanAscii(row.domain || "").toLowerCase();
  if (domainProfiles[domain]) return domain;
  const state = cleanAscii(row.private_state || row.simulated_private_state || "").toLowerCase();
  if (state.includes("pilot ") || state.includes("control vector") || state.includes("near prospect")) {
    return "signal";
  }
  return "cosyworld";
}

function promptFor(profile, row) {
  const state = cleanAscii(row.private_state || row.simulated_private_state || "");
  const reference = cleanAscii(row.expected_output || row.line || row.target || "(none)");
  return profile.task
    .replace("{state}", state)
    .replace("{reference}", reference);
}

function stripWrappers(text) {
  const lines = String(text ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/^```[a-z]*\s*/i, "")
    .replace(/```$/i, "")
    .split("\n")
    .map(cleanOneLine)
    .filter(Boolean);
  let out = (lines[0] || "")
    .replace(/^```[a-z]*\s*/i, "")
    .replace(/```$/i, "")
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/^(radio|output|line|narration|expected_output|response)\s*:\s*/i, "")
    .trim();
  out = out.split(/\n/)[0] || out;
  out = out.replace(/\s+([,.;:!?])/g, "$1").trim();
  return out;
}

function sentenceTrim(text, maxChars) {
  if (text.length <= maxChars) return text;
  const clipped = text.slice(0, maxChars + 1);
  const sentenceEnd = Math.max(clipped.lastIndexOf("."), clipped.lastIndexOf("!"), clipped.lastIndexOf("?"), clipped.lastIndexOf(";"));
  if (sentenceEnd >= 32) return clipped.slice(0, sentenceEnd + 1).trim();
  const space = clipped.lastIndexOf(" ");
  return clipped.slice(0, space >= 32 ? space : maxChars).trim();
}

function normalizeGenerated(text, profile) {
  return sentenceTrim(stripWrappers(text), profile.maxChars);
}

function rejectionReason(text, profile) {
  if (!text) return "empty";
  if (text.length < 10) return "too_short";
  if (text.length > profile.maxChars) return "too_long";
  if (/[{}[\]]/.test(text)) return "structured_wrapper";
  if (/https?:\/\//i.test(text)) return "url";
  if (/\b(ranked|private_state|expected_output|simulated_private_state|assistant|chatbot|model|training|prompt|json)\b/i.test(text)) {
    return "meta_term";
  }
  if (/\bAI\b/.test(text)) return "ai_term";
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 3) return "too_few_words";
  if (words.length > 32) return "too_many_words";
  return "";
}

async function ollamaGenerate(options, profile, prompt, variantSeed) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs);
  try {
    const response = await fetch(options.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: options.model,
        system: profile.system,
        prompt,
        stream: false,
        options: {
          temperature: options.temperature,
          top_p: options.topP,
          num_predict: profile.numPredict,
          seed: variantSeed,
        },
      }),
    });
    if (!response.ok) {
      const body = await response.text();
      throw new Error(`ollama returned HTTP ${response.status}: ${body.slice(0, 300)}`);
    }
    const payload = await response.json();
    return String(payload.response || "");
  } finally {
    clearTimeout(timeout);
  }
}

function cacheKey(row, domain, variant) {
  return `${row.id || idFor([row.private_state || "", row.expected_output || ""])}:${domain}:${variant}`;
}

function loadCache(cachePath) {
  const cache = new Map();
  if (!fs.existsSync(cachePath)) return cache;
  for (const row of readJsonl(cachePath)) {
    if (row.cache_key && row.accepted) {
      cache.set(row.cache_key, row);
    }
  }
  return cache;
}

function buildOutputRow(sourceRow, cacheRow) {
  const baseId = sourceRow.id || idFor([sourceRow.private_state || "", sourceRow.expected_output || ""]);
  return {
    id: idFor([baseId, cacheRow.cache_key, cacheRow.output]),
    source_id: baseId,
    speaker: sourceRow.speaker,
    kind: `${sourceRow.kind || "state"}_ollama`,
    domain: cacheRow.domain,
    generator_model: cacheRow.model,
    variant: cacheRow.variant,
    private_state: sourceRow.private_state || sourceRow.simulated_private_state || "",
    source_expected_output: sourceRow.expected_output || "",
    expected_output: cacheRow.output,
  };
}

async function draftRow(options, row, domain, variant, cachePath) {
  const profile = domainProfiles[domain];
  if (!profile) {
    throw new Error(`unsupported domain: ${domain}`);
  }
  const key = cacheKey(row, domain, variant);
  const prompt = promptFor(profile, row);
  const baseSeed = Number.parseInt(idFor([options.seed, key]).slice(0, 8), 16);

  for (let attempt = 1; attempt <= options.attempts; attempt += 1) {
    const raw = await ollamaGenerate(options, profile, prompt, baseSeed + attempt);
    const output = normalizeGenerated(raw, profile);
    const rejected = rejectionReason(output, profile);
    const cacheRow = {
      schema: "nsrl.ollama_state_output_cache.v1",
      cache_key: key,
      created_at: new Date().toISOString(),
      accepted: !rejected,
      rejection_reason: rejected || null,
      model: options.model,
      domain,
      variant,
      attempt,
      prompt_hash: idFor([profile.system, prompt]),
      output,
      raw_response: cleanAscii(raw),
    };
    appendJsonl(cachePath, cacheRow);
    if (!rejected) return cacheRow;
  }
  return null;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir || defaultOutDir(options));
  fs.mkdirSync(outDir, { recursive: true });

  const cachePath = path.join(outDir, "cache.jsonl");
  const pairsPath = path.join(outDir, "training-pairs.jsonl");
  const expectedOutputPath = path.join(outDir, "expected-output.txt");
  const rejectsPath = path.join(outDir, "rejects.jsonl");
  const manifestPath = path.join(outDir, "manifest.json");

  const inputRows = readJsonl(options.input);
  const selectedRows = inputRows
    .slice(options.offset, options.limit === 0 ? undefined : options.offset + options.limit)
    .filter((row) => cleanAscii(row.private_state || row.simulated_private_state || ""));
  const keepExisting = parseBool(options.keepExisting, "--keep-existing");
  const cache = keepExisting ? loadCache(cachePath) : new Map();
  const outputRows = [];
  const rejects = [];

  for (const [rowIndex, row] of selectedRows.entries()) {
    const domain = inferDomain(row, options);
    for (let variant = 0; variant < options.variantsPerState; variant += 1) {
      const key = cacheKey(row, domain, variant);
      let cacheRow = cache.get(key);
      if (!cacheRow) {
        cacheRow = await draftRow(options, row, domain, variant, cachePath);
        if (cacheRow) cache.set(key, cacheRow);
      }
      if (cacheRow) {
        outputRows.push(buildOutputRow(row, cacheRow));
      } else {
        rejects.push({
          source_id: row.id || null,
          domain,
          variant,
          reason: "all_attempts_rejected",
        });
      }
    }
    console.error(`generated ${rowIndex + 1}/${selectedRows.length} rows`);
  }

  writeJsonl(pairsPath, outputRows);
  writeJsonl(rejectsPath, rejects);
  fs.writeFileSync(expectedOutputPath, `${outputRows.map((row) => row.expected_output).join("\n")}\n`, "utf8");

  const byDomain = outputRows.reduce((acc, row) => {
    acc[row.domain] = (acc[row.domain] || 0) + 1;
    return acc;
  }, {});
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.ollama_state_outputs.v1",
    created_at: new Date().toISOString(),
    model: options.model,
    endpoint: options.endpoint,
    input_path: resolveRepoPath(options.input),
    out_dir: outDir,
    training_pairs_path: pairsPath,
    expected_output_path: expectedOutputPath,
    cache_path: cachePath,
    rejects_path: rejectsPath,
    input_rows: inputRows.length,
    selected_rows: selectedRows.length,
    variants_per_state: options.variantsPerState,
    output_rows: outputRows.length,
    rejected_variants: rejects.length,
    domains: byDomain,
    flattening: "private_state newline expected_output, no control labels",
    notes: [
      "Generated locally through Ollama from simulator private_state rows.",
      "expected_output is replaced by the accepted in-world line; source_expected_output keeps the deterministic baseline.",
      "cache.jsonl is append-only so interrupted runs can resume with --keep-existing true.",
    ],
  }, null, 2)}\n`, "utf8");

  console.log(`rows=${outputRows.length}`);
  console.log(`rejects=${rejects.length}`);
  console.log(`training_pairs=${pairsPath}`);
  console.log(`manifest=${manifestPath}`);
}

main().catch((error) => {
  console.error(`generate-state-outputs-ollama: ${error.message}`);
  process.exit(1);
});
