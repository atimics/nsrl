#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "data/processed/contextual-reading-pairs",
  domains: "signal,cosyworld",
  styles: "shakespeare,blake,crowley",
  maxPairsPerLane: 24,
  chunkChars: 720,
  minChunkChars: 120,
  seed: "contextual-reading-v1",
  signalPairs: "data/processed/signal-replay-corpus/training-pairs.jsonl",
  cosyworldPairs: "data/processed/cosyworld-kernel-corpus/training-pairs.jsonl",
  shakespeareSource: "data/processed/visionary-balanced-prose/shakespeare.clean.txt",
  shakespeareFallback: "data/raw/shakespeare-gutenberg-100.txt",
  blakeSources: "data/processed/blake-poems.clean.txt,data/processed/blake-marriage-heaven-hell.clean.txt",
  crowleySources: "data/processed/crowley-household-gods.clean.txt,data/processed/crowley-tannhauser.clean.txt",
};

const domainPairOptions = {
  signal: "signalPairs",
  cosyworld: "cosyworldPairs",
};

function usage() {
  console.log(`Usage: node scripts/build-contextual-reading-pairs.mjs [options]

Options:
  --out-dir PATH             Output directory [${defaults.outDir}]
  --domains LIST             Comma-separated domains: signal,cosyworld [${defaults.domains}]
  --styles LIST              Comma-separated styles: shakespeare,blake,crowley [${defaults.styles}]
  --max-pairs-per-lane N     Rows per domain/style lane [${defaults.maxPairsPerLane}]
  --chunk-chars N            Target maximum literary chunk size [${defaults.chunkChars}]
  --min-chunk-chars N        Minimum literary chunk size [${defaults.minChunkChars}]
  --seed TEXT                Stable id seed [${defaults.seed}]
  --signal-pairs PATH        Signal private-state pairs [${defaults.signalPairs}]
  --cosyworld-pairs PATH     CosyWorld private-state pairs [${defaults.cosyworldPairs}]
  --shakespeare-source PATH  Shakespeare source [${defaults.shakespeareSource}]
  --shakespeare-fallback PATH
                             Fallback Shakespeare source [${defaults.shakespeareFallback}]
  --blake-sources LIST       Comma-separated Blake sources [${defaults.blakeSources}]
  --crowley-sources LIST     Comma-separated Crowley sources [${defaults.crowleySources}]
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
    if (["maxPairsPerLane", "chunkChars", "minChunkChars"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 1) {
        throw new Error(`${arg} requires a positive integer`);
      }
    } else {
      options[key] = value;
    }
  }
  if (options.minChunkChars > options.chunkChars) {
    throw new Error("--min-chunk-chars cannot be larger than --chunk-chars");
  }
  return options;
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function splitList(value) {
  return String(value || "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function cleanLine(text) {
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

function cleanBlock(text) {
  return String(text ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trimEnd())
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
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

function firstExisting(paths) {
  for (const filePath of paths) {
    const resolved = resolveRepoPath(filePath);
    if (fs.existsSync(resolved)) {
      return resolved;
    }
  }
  throw new Error(`none of these source files exist: ${paths.map(resolveRepoPath).join(", ")}`);
}

function sourceTextForStyle(style, options) {
  if (style === "shakespeare") {
    const sourcePath = firstExisting([options.shakespeareSource, options.shakespeareFallback]);
    return {
      sourcePaths: [sourcePath],
      text: fs.readFileSync(sourcePath, "utf8"),
    };
  }
  if (style === "blake") {
    const sourcePaths = splitList(options.blakeSources)
      .map(resolveRepoPath)
      .filter((sourcePath) => fs.existsSync(sourcePath));
    if (sourcePaths.length === 0) {
      throw new Error(`no Blake source files found from --blake-sources ${options.blakeSources}`);
    }
    return {
      sourcePaths,
      text: sourcePaths.map((sourcePath) => fs.readFileSync(sourcePath, "utf8")).join("\n\n"),
    };
  }
  if (style === "crowley") {
    const sourcePaths = splitList(options.crowleySources)
      .map(resolveRepoPath)
      .filter((sourcePath) => fs.existsSync(sourcePath));
    if (sourcePaths.length === 0) {
      throw new Error(`no Crowley source files found from --crowley-sources ${options.crowleySources}`);
    }
    return {
      sourcePaths,
      text: sourcePaths.map((sourcePath) => fs.readFileSync(sourcePath, "utf8")).join("\n\n"),
    };
  }
  throw new Error(`unknown style: ${style}`);
}

function skipFrontMatter(style, text) {
  if (style === "shakespeare") {
    const marker = text.indexOf("\nFrom fairest creatures");
    return marker >= 0 ? text.slice(marker + 1) : text;
  }
  if (style === "blake") {
    const marker = text.indexOf("\nPiping down the valleys wild");
    if (marker >= 0) return text.slice(marker + 1);
    const alternate = text.indexOf("\nRintrah roars");
    return alternate >= 0 ? text.slice(alternate + 1) : text;
  }
  if (style === "crowley") {
    const markers = [
      "\nSmoke without fire!",
      "\nI shall not tell thee",
      "\nOne is incisive",
    ];
    for (const marker of markers) {
      const offset = text.indexOf(marker);
      if (offset >= 0) return text.slice(offset + 1);
    }
  }
  return text;
}

function isUsableBlock(block) {
  const compact = cleanLine(block);
  if (compact.length < 20) return false;
  const lower = compact.toLowerCase();
  const banned = [
    "project gutenberg",
    "complete works",
    "contents",
    "william shakespeare",
    "william blake",
    "aleister crowley",
    "john w. luce",
    "boston",
    "transcriber's notes",
    "graphics and textual content",
    "illustration",
    "all rights reserved",
    "society for the propagation",
    "privately printed",
  ];
  if (banned.some((term) => lower.includes(term))) return false;
  const letters = compact.replace(/[^a-z]/gi, "");
  if (letters.length < 16) return false;
  const upperLetters = compact.replace(/[^A-Z]/g, "");
  return upperLetters.length / letters.length < 0.72;
}

function sentencePieces(block, maxChars) {
  const lines = cleanBlock(block).split("\n").map(cleanLine).filter(Boolean);
  if (lines.length > 1) {
    const chunks = [];
    let current = "";
    for (const line of lines) {
      const next = current ? `${current}\n${line}` : line;
      if (next.length > maxChars && current) {
        chunks.push(current);
        current = line;
      } else {
        current = next;
      }
    }
    if (current) chunks.push(current);
    return chunks;
  }
  const compact = cleanBlock(block).replace(/\n+/g, " ");
  const pieces = compact.match(/[^.!?]+[.!?]+["']?|[^.!?]+$/g) || [compact];
  const chunks = [];
  let current = "";
  for (const piece of pieces.map(cleanLine).filter(Boolean)) {
    const next = current ? `${current} ${piece}` : piece;
    if (next.length > maxChars && current) {
      chunks.push(current);
      current = piece;
    } else if (piece.length > maxChars) {
      for (let offset = 0; offset < piece.length; offset += maxChars) {
        chunks.push(piece.slice(offset, offset + maxChars).trim());
      }
      current = "";
    } else {
      current = next;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

function literaryChunks(style, rawText, options) {
  const text = cleanBlock(skipFrontMatter(style, rawText));
  const paragraphs = text.split(/\n\s*\n+/).map(cleanBlock).filter(isUsableBlock);
  const chunks = [];
  let current = "";
  for (const paragraph of paragraphs) {
    const paragraphChunks = paragraph.length > options.chunkChars
      ? sentencePieces(paragraph, options.chunkChars)
      : [paragraph];
    for (const piece of paragraphChunks) {
      if (!isUsableBlock(piece)) continue;
      const next = current ? `${current}\n\n${piece}` : piece;
      if (next.length > options.chunkChars && current.length >= options.minChunkChars) {
        chunks.push(current);
        current = piece;
      } else if (piece.length >= options.minChunkChars) {
        if (current && next.length <= options.chunkChars) {
          current = next;
        } else {
          if (current.length >= options.minChunkChars) chunks.push(current);
          current = piece;
        }
      } else {
        current = next;
      }
      if (chunks.length >= options.maxPairsPerLane * 4) break;
    }
    if (chunks.length >= options.maxPairsPerLane * 4) break;
  }
  if (current.length >= options.minChunkChars) chunks.push(current);
  return chunks
    .map(cleanBlock)
    .filter((chunk) => chunk.length >= options.minChunkChars && isUsableBlock(chunk));
}

function conciseSourceIntent(row) {
  const state = cleanLine(row.private_state);
  const output = cleanLine(row.expected_output);
  return state || output;
}

function signalPrivateState(row, style) {
  const speaker = cleanLine(row.speaker || "PILOT");
  const intent = conciseSourceIntent(row);
  if (style === "shakespeare") {
    return cleanLine(`${speaker} carries the route concern as formal danger, keeping station, cargo, and consequence under a measured old cadence. ${intent}`);
  }
  if (style === "blake") {
    return cleanLine(`${speaker} feels the lane as a bright industrial omen, compressing cargo, hazard, and trust into a small prophetic image. ${intent}`);
  }
  if (style === "crowley") {
    return cleanLine(`${speaker} feels the route as a ritual threshold, turning cargo, danger, and desire into theatrical invocation. ${intent}`);
  }
  return cleanLine(`${speaker} keeps the Signal fact active while changing only the surface cadence. ${intent}`);
}

function cosyworldPrivateState(row, style) {
  const speaker = cosyworldAnchor(row);
  const intent = conciseSourceIntent(row);
  if (style === "shakespeare") {
    return cleanLine(`${speaker} holds the room's welcome beneath courtly weather, letting kindness and worry speak in an older measured cadence. ${intent}`);
  }
  if (style === "blake") {
    return cleanLine(`${speaker} feels small mercy as a luminous household sign, turning care, threshold, and object into compact vision. ${intent}`);
  }
  if (style === "crowley") {
    return cleanLine(`${speaker} feels the hearth as a small stage of charm and appetite, letting welcome, longing, and household magic speak ceremonially. ${intent}`);
  }
  return cleanLine(`${speaker} keeps the cosy fact active while changing only the surface cadence. ${intent}`);
}

function cosyworldAnchor(row) {
  const state = cleanLine(row.private_state || "");
  const locatedMatch = state.match(/^(.+?)\s+is at\s+(.+?)\s+with\b/);
  if (locatedMatch) {
    return `${locatedMatch[1]} at ${locatedMatch[2]}`;
  }
  const stateMatch = state.match(/^(.+?)\s+(?:wants|notices|holds|makes|waits|answers)\b/);
  if (stateMatch) return stateMatch[1];
  const speaker = cleanLine(row.speaker || "");
  if (speaker && speaker !== "COSYWORLD") return speaker;
  const output = cleanLine(row.expected_output || "");
  const outputMatch = output.match(/^(?:Using\s+)?(.+?)\s+(?:is|has|offers|waits|keeps|points|finds|turns|stores|glows|marks|rings|makes|and)\b/);
  if (outputMatch) return outputMatch[1];
  return "The room";
}

function simulatedPrivateState(domain, style, row) {
  if (domain === "signal") return signalPrivateState(row, style);
  if (domain === "cosyworld") return cosyworldPrivateState(row, style);
  throw new Error(`unknown domain: ${domain}`);
}

function trainingRow({ seed, domain, style, row, chunk, chunkIndex, sourcePaths }) {
  const privateState = simulatedPrivateState(domain, style, row);
  const expectedOutput = cleanBlock(chunk);
  const id = idFor([seed, domain, style, row.id || row.private_state || "", chunkIndex, expectedOutput]);
  return {
    id,
    domain,
    style,
    speaker: cleanLine(row.speaker || ""),
    kind: cleanLine(row.kind || ""),
    private_state: privateState,
    simulated_private_state: privateState,
    source_private_state: cleanLine(row.private_state || ""),
    source_expected_output: cleanLine(row.expected_output || ""),
    expected_output: expectedOutput,
    source_paths: sourcePaths,
    chunk_index: chunkIndex,
  };
}

function writeJsonl(filePath, records) {
  fs.writeFileSync(filePath, records.map((record) => `${JSON.stringify(record)}\n`).join(""), "utf8");
}

function writeLaneFiles(outDir, lanes) {
  for (const [lane, rows] of lanes.entries()) {
    writeJsonl(path.join(outDir, `${lane}.training-pairs.jsonl`), rows);
    fs.writeFileSync(
      path.join(outDir, `${lane}.expected-output.txt`),
      `${rows.map((row) => row.expected_output).join("\n\n")}\n`,
      "utf8"
    );
  }
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const domains = splitList(options.domains);
  const styles = splitList(options.styles);
  const domainPairs = new Map();
  for (const domain of domains) {
    const optionKey = domainPairOptions[domain];
    if (!optionKey) throw new Error(`unknown domain: ${domain}`);
    const pairs = readJsonl(options[optionKey]).filter((row) => row.private_state && row.expected_output);
    if (pairs.length === 0) throw new Error(`no usable pairs for domain: ${domain}`);
    domainPairs.set(domain, pairs);
  }

  const styleChunks = new Map();
  const styleSources = new Map();
  for (const style of styles) {
    const source = sourceTextForStyle(style, options);
    const chunks = literaryChunks(style, source.text, options);
    if (chunks.length === 0) throw new Error(`no usable literary chunks for style: ${style}`);
    styleChunks.set(style, chunks);
    styleSources.set(style, source.sourcePaths);
  }

  const rows = [];
  const lanes = new Map();
  for (const domain of domains) {
    const pairs = domainPairs.get(domain);
    for (const style of styles) {
      const chunks = styleChunks.get(style);
      const sourcePaths = styleSources.get(style);
      const laneRows = [];
      const laneLimit = Math.min(options.maxPairsPerLane, chunks.length);
      for (let index = 0; index < laneLimit; index += 1) {
        const row = pairs[index % pairs.length];
        const chunk = chunks[index];
        const built = trainingRow({
          seed: options.seed,
          domain,
          style,
          row,
          chunk,
          chunkIndex: index,
          sourcePaths,
        });
        rows.push(built);
        laneRows.push(built);
      }
      lanes.set(`${domain}-${style}`, laneRows);
    }
  }

  const trainingPairsPath = path.join(outDir, "training-pairs.jsonl");
  const expectedOutputPath = path.join(outDir, "expected-output.txt");
  const manifestPath = path.join(outDir, "manifest.json");
  writeJsonl(trainingPairsPath, rows);
  fs.writeFileSync(expectedOutputPath, `${rows.map((row) => row.expected_output).join("\n\n")}\n`, "utf8");
  writeLaneFiles(outDir, lanes);
  fs.writeFileSync(manifestPath, `${JSON.stringify({
    schema: "nsrl.contextual_reading_pairs.v1",
    created_at: new Date().toISOString(),
    training_pairs_path: trainingPairsPath,
    expected_output_path: expectedOutputPath,
    out_dir: outDir,
    domains,
    styles,
    max_pairs_per_lane: options.maxPairsPerLane,
    chunk_chars: options.chunkChars,
    min_chunk_chars: options.minChunkChars,
    rows: rows.length,
    lanes: Object.fromEntries([...lanes.entries()].map(([lane, laneRows]) => [lane, laneRows.length])),
    source_paths: Object.fromEntries([...styleSources.entries()]),
    domain_pair_paths: Object.fromEntries(domains.map((domain) => [domain, resolveRepoPath(options[domainPairOptions[domain]])])),
    notes: [
      "Rows use private_state and expected_output so future trainers can consume them like world-state pairs.",
      "For reading data, private_state is simulated from an in-world Signal or CosyWorld state; expected_output is a real literary source chunk.",
      "Do not train tiny raw-output models on these JSONL field names. Use expected-output.txt only if you intentionally want output-only literary continuation.",
    ],
  }, null, 2)}\n`, "utf8");

  console.log(`rows=${rows.length}`);
  console.log(`training_pairs=${trainingPairsPath}`);
  console.log(`expected_output=${expectedOutputPath}`);
  console.log(`manifest=${manifestPath}`);
}

try {
  main();
} catch (error) {
  console.error(`build-contextual-reading-pairs: ${error.message}`);
  process.exit(1);
}
