#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "data/processed/signal-romance-smoke",
  signalSft: "/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-sft.jsonl",
  signalVoice: "/Users/ratimics/develop/signal/corpus/ship-radio/ship-radio-voice.txt",
  signalRepeat: 6,
  evalCount: 24,
  styleBytes: 8000,
  styleChunkBytes: 256,
  styleEveryFrames: 16,
  styleSourceDir: "data/processed/signal-romance-sources",
  seed: "signal-romance-v1",
};

const knownStations = ["Prospect Ref", "Kepler Yard", "Helios Works", "Freeport"];
const reservedUpperTokens = new Set(["YOU", "LINE"]);

function usage() {
  console.log(`Usage: node scripts/build-signal-romance-corpus.mjs [options]

Options:
  --out-dir PATH             Output directory [${defaults.outDir}]
  --signal-sft PATH          Signal SFT JSONL [${defaults.signalSft}]
  --signal-voice PATH        Signal voice seed text [${defaults.signalVoice}]
  --signal-repeat N          Extra repeats after the first training pass [${defaults.signalRepeat}]
  --eval-count N             Held-out prompt count [${defaults.evalCount}]
  --style-bytes N            Bytes sampled per style source [${defaults.styleBytes}]
  --style-chunk-bytes N      Bytes per interleaved style chunk [${defaults.styleChunkBytes}]
  --style-every-frames N     Insert one style chunk per N signal frames [${defaults.styleEveryFrames}]
  --style-source-dir PATH    Optional extra style source directory [${defaults.styleSourceDir}]
  --seed TEXT                Deterministic split/sampling seed [${defaults.seed}]
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
    if (["signalRepeat", "evalCount", "styleBytes", "styleChunkBytes", "styleEveryFrames"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 0) {
        throw new Error(`${arg} requires a non-negative integer`);
      }
    } else {
      options[key] = value;
    }
  }
  if (options.styleChunkBytes <= 0) {
    throw new Error("--style-chunk-bytes must be positive");
  }
  if (options.styleEveryFrames <= 0) {
    throw new Error("--style-every-frames must be positive");
  }
  return options;
}

function resolveRepoPath(filePath) {
  if (path.isAbsolute(filePath)) {
    return filePath;
  }
  return path.join(repoRoot, filePath);
}

function readText(filePath, label) {
  const resolved = resolveRepoPath(filePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`missing ${label}: ${resolved}`);
  }
  return fs.readFileSync(resolved, "utf8");
}

function maybeReadText(filePath) {
  const resolved = resolveRepoPath(filePath);
  if (!fs.existsSync(resolved)) {
    return null;
  }
  return fs.readFileSync(resolved, "utf8");
}

function cleanAscii(text) {
  return text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[–—]/g, "-")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function compact(text) {
  return cleanAscii(String(text ?? "")).replace(/\s+/g, " ").trim();
}

function idFor(parts) {
  return crypto.createHash("sha1").update(parts.join("\0")).digest("hex").slice(0, 16);
}

function hashScore(seed, text) {
  const digest = crypto.createHash("sha1").update(`${seed}\0${text}`).digest();
  return digest.readUInt32BE(0);
}

function parseJsonl(text, filePath) {
  const records = [];
  for (const [index, line] of text.split(/\n/).entries()) {
    if (!line.trim()) {
      continue;
    }
    try {
      records.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`${filePath}:${index + 1}: ${error.message}`);
    }
  }
  return records;
}

function parseFields(text) {
  const fields = {};
  for (const line of text.split(/\n/)) {
    const match = line.match(/^([^=\s]+)=(.+)$/);
    if (match) {
      fields[match[1]] = compact(match[2]);
    }
  }
  return fields;
}

function speakerFromHeader(header) {
  if (header === "YOU") {
    return { speaker: "YOU", role: "player" };
  }
  const match = header.match(/^([A-Z]+)\s+([A-Z]\d+)$/);
  if (!match) {
    return { speaker: header, role: "ship" };
  }
  return { speaker: match[2], role: match[1].toLowerCase() };
}

function extractChoiceFrames(record) {
  const user = record.messages.find((message) => message.role === "user")?.content ?? "";
  const assistant = record.messages.find((message) => message.role === "assistant")?.content ?? "";
  const blocks = new Map();
  let current = null;
  for (const rawLine of user.split(/\n/)) {
    const line = rawLine.trim();
    const header = line.match(/^([A-Z]+(?:\s+[A-Z]\d+)?|YOU):$/);
    if (header) {
      const speaker = speakerFromHeader(header[1]);
      current = speaker.speaker;
      blocks.set(current, { ...speaker, choices: new Map() });
      continue;
    }
    const choice = line.match(/^(\d+)\s+(.+)$/);
    if (choice && current && blocks.has(current)) {
      blocks.get(current).choices.set(choice[1], compact(choice[2]));
    }
  }

  const frames = [];
  for (const part of assistant.split(",")) {
    const assignment = part.trim().match(/^([^=]+)=(\d+)$/);
    if (!assignment) {
      continue;
    }
    const block = blocks.get(assignment[1]);
    const line = block?.choices.get(assignment[2]);
    if (!block || !line) {
      continue;
    }
    frames.push(makeFrame({
      source: "choice_batch",
      sourceId: record.id,
      speaker: block.speaker,
      role: block.role,
      state: "ranked",
      line,
      fields: {
        speaker: block.speaker,
        role: block.role,
        choice: assignment[2],
        context: record.metadata?.context,
      },
    }));
  }
  return frames;
}

function extractLineFrame(record) {
  const user = record.messages.find((message) => message.role === "user")?.content ?? "";
  const assistant = record.messages.find((message) => message.role === "assistant")?.content ?? "";
  const fields = parseFields(user);
  if (!assistant.trim()) {
    return null;
  }
  return makeFrame({
    source: "ship_radio_line",
    sourceId: record.id,
    speaker: fields.speaker ?? "SHIP",
    role: fields.role ?? "ship",
    state: fields.state ?? "",
    line: assistant,
    fields,
  });
}

function extractStations(line, fields) {
  const terms = new Set();
  for (const station of knownStations) {
    if (line.includes(station)) {
      terms.add(station);
    }
  }
  for (const value of Object.values(fields)) {
    const text = String(value ?? "");
    for (const station of knownStations) {
      if (text.includes(station)) {
        terms.add(station);
      }
    }
  }
  return [...terms];
}

function extractCommodities(line, fields) {
  const text = `${line} ${Object.values(fields).join(" ")}`;
  const terms = new Set();
  for (const match of text.matchAll(/\b[A-Z]{2,3}\b/g)) {
    if (!reservedUpperTokens.has(match[0]) && !/^N\d+$/.test(match[0])) {
      terms.add(match[0]);
    }
  }
  return [...terms];
}

function makeFrame({ source, sourceId, speaker, role, state, line, fields }) {
  const cleanLine = compact(line);
  const cleanFields = Object.fromEntries(
    Object.entries(fields ?? {})
      .filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "")
      .map(([key, value]) => [key, compact(value)])
  );
  const stations = extractStations(cleanLine, cleanFields);
  const commodities = extractCommodities(cleanLine, cleanFields);
  const route = cleanFields.route ?? routeFromLine(cleanLine);
  const memory = cleanFields.memory ?? "";
  const module = cleanFields.module ?? "";
  const groundingTerms = [...new Set([...stations, ...commodities, route, memory, module].filter(Boolean))];
  const compactGrounding = [
    `${speaker}/${role || "ship"}`,
    state,
    ...commodities,
    ...stations,
    route,
    memory,
    module,
  ].filter(Boolean).join(" ");
  const prompt = `RANKED: ${cleanLine}\nVOICE: `;
  const id = idFor([source, sourceId, speaker, role, state, cleanLine, compactGrounding]);
  return {
    id,
    source,
    source_id: sourceId,
    speaker,
    role,
    state,
    fields: cleanFields,
    line: cleanLine,
    prompt,
    target: cleanLine,
    grounding_terms: groundingTerms.length > 0 ? groundingTerms : fallbackGroundingTerms(cleanLine),
    stations,
    commodities,
  };
}

function routeFromLine(line) {
  const match = line.match(/\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)>([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)\b/);
  return match ? `${match[1]}>${match[2]}` : "";
}

function fallbackGroundingTerms(line) {
  return [...new Set(
    line
      .split(/[^A-Za-z0-9>]+/)
      .filter((token) => token.length >= 4)
      .slice(0, 4)
  )];
}

function trainingText(frame) {
  return `${frame.prompt}${frame.target}\nEND\n`;
}

function parseSignalFrames(signalSftPath) {
  const text = readText(signalSftPath, "Signal SFT JSONL");
  const records = parseJsonl(text, signalSftPath);
  const frames = [];
  for (const record of records) {
    if (record.task === "ship_radio_line") {
      const frame = extractLineFrame(record);
      if (frame) {
        frames.push(frame);
      }
    } else if (record.task === "choice_batch") {
      frames.push(...extractChoiceFrames(record));
    }
  }
  const unique = new Map();
  for (const frame of frames) {
    const key = [frame.speaker, frame.role, frame.state, frame.prompt, frame.target].join("\0");
    if (!unique.has(key)) {
      unique.set(key, frame);
    }
  }
  return { rawFrames: frames, frames: [...unique.values()] };
}

function voiceDoctrine(signalVoicePath) {
  const text = readText(signalVoicePath, "Signal voice seed");
  const beforeCanonical = text.split(/Canonical radio lines:/i)[0] ?? text;
  return cleanAscii(beforeCanonical.replace(/^#.*$/gm, "")).trim();
}

function styleSources() {
  const builtIns = [
    {
      label: "shakespeare",
      paths: [
        "data/processed/visionary-balanced-prose-balanced-prose-literary-v1/shakespeare.source.txt",
        "data/processed/visionary-balanced-prose/shakespeare.clean.txt",
        "data/raw/shakespeare-gutenberg-100.txt",
      ],
    },
    {
      label: "blake",
      paths: [
        "data/processed/visionary-balanced-prose-balanced-prose-literary-v1/blake.source.txt",
        "data/processed/blake-poems.clean.txt",
        "data/processed/blake-marriage-heaven-hell.clean.txt",
      ],
    },
    {
      label: "crowley",
      paths: [
        "data/processed/visionary-balanced-prose-balanced-prose-literary-v1/crowley.source.txt",
        "data/processed/crowley-household-gods.clean.txt",
        "data/processed/crowley-tannhauser.clean.txt",
      ],
    },
    {
      label: "simplewiki",
      paths: [
        "data/processed/simplewiki-expository-v1/simplewiki.clean.txt",
        "data/processed/visionary-balanced-prose-balanced-prose-literary-v1/simplewiki-synthetic.source.txt",
      ],
    },
  ];
  return builtIns;
}

function externalStyleSources(styleSourceDir) {
  const resolved = resolveRepoPath(styleSourceDir);
  if (!fs.existsSync(resolved)) {
    return [];
  }
  const files = fs.readdirSync(resolved)
    .filter((file) => file.endsWith(".clean.txt"))
    .sort();
  return files.map((file) => ({
    label: path.basename(file, ".clean.txt"),
    paths: [path.join(resolved, file)],
  }));
}

function readStyleBundle(source) {
  const chunks = [];
  const usedPaths = [];
  for (const candidate of source.paths) {
    const text = maybeReadText(candidate);
    if (text) {
      chunks.push(text);
      usedPaths.push(resolveRepoPath(candidate));
      if (candidate.includes("source.txt") || candidate.includes("simplewiki.clean.txt")) {
        break;
      }
    }
  }
  if (chunks.length === 0) {
    throw new Error(`missing style source for ${source.label}: ${source.paths.join(", ")}`);
  }
  return { text: cleanAscii(chunks.join("\n\n")), usedPaths };
}

function buildStyleFrames(options) {
  const frames = [];
  const manifest = [];
  const sources = [...styleSources(), ...externalStyleSources(options.styleSourceDir)];
  for (const source of sources) {
    const bundle = readStyleBundle(source);
    const text = bundle.text;
    const targetBytes = Math.min(options.styleBytes, Buffer.byteLength(text));
    const chunkCount = Math.max(1, Math.ceil(targetBytes / options.styleChunkBytes));
    const step = Math.max(1, Math.floor(text.length / (chunkCount + 1)));
    let emitted = 0;
    for (let index = 0; index < chunkCount && emitted < targetBytes; index += 1) {
      const offset = Math.min(text.length - 1, step * (index + 1));
      const remaining = targetBytes - emitted;
      const size = Math.min(options.styleChunkBytes, remaining);
      const chunk = paragraphWindow(text, offset, size);
      emitted += Buffer.byteLength(chunk);
      frames.push(`${chunk}\n\n`);
    }
    manifest.push({
      label: source.label,
      paths: bundle.usedPaths,
      source_bytes: Buffer.byteLength(text),
      emitted_bytes: emitted,
      chunks: chunkCount,
    });
  }
  return { frames, manifest };
}

function paragraphWindow(text, offset, size) {
  let start = text.lastIndexOf("\n\n", offset);
  if (start === -1 || offset - start > size) {
    start = Math.max(0, offset - Math.floor(size / 3));
  }
  let chunk = text.slice(start, start + size * 2);
  const end = chunk.indexOf("\n\n", size);
  if (end !== -1) {
    chunk = chunk.slice(0, end);
  } else {
    chunk = chunk.slice(0, size);
  }
  return cleanAscii(chunk);
}

function writeJsonl(filePath, records) {
  fs.writeFileSync(filePath, records.map((record) => `${JSON.stringify(record)}\n`).join(""), "utf8");
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = resolveRepoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const parsed = parseSignalFrames(options.signalSft);
  const frames = parsed.frames.sort((a, b) => a.id.localeCompare(b.id));
  const evalCount = Math.min(options.evalCount, Math.max(0, frames.length - 1));
  const evalIds = new Set(
    frames
      .map((frame) => ({ frame, score: hashScore(options.seed, frame.id) }))
      .sort((a, b) => a.score - b.score)
      .slice(0, evalCount)
      .map(({ frame }) => frame.id)
  );
  const trainFrames = frames.filter((frame) => !evalIds.has(frame.id));
  const evalFrames = frames.filter((frame) => evalIds.has(frame.id));
  const doctrine = voiceDoctrine(options.signalVoice);
  const style = buildStyleFrames(options);

  const corpusParts = [
    "SIGNAL_ROMANCE_CORPUS_V1\n",
    "DIVISION ranker=crlplrimes voice=nsrl fallback=deterministic\n",
    `SIGNAL_DOCTRINE\n${doctrine}\nEND_DOCTRINE\n`,
  ];
  let styleIndex = 0;
  const appendSignalPass = (pass) => {
    corpusParts.push(`SIGNAL_TRUTH_PASS ${pass}\n`);
    for (const [index, frame] of trainFrames.entries()) {
      corpusParts.push(trainingText(frame));
      if ((index + 1) % options.styleEveryFrames === 0 && style.frames.length > 0) {
        corpusParts.push(style.frames[styleIndex % style.frames.length]);
        styleIndex += 1;
      }
    }
  };
  appendSignalPass(0);
  for (let pass = 1; pass <= options.signalRepeat; pass += 1) {
    appendSignalPass(pass);
  }

  const corpus = corpusParts.join("\n");
  const corpusPath = path.join(outDir, "corpus.txt");
  const framesPath = path.join(outDir, "frames.jsonl");
  const trainPath = path.join(outDir, "train-frames.jsonl");
  const evalPath = path.join(outDir, "eval-prompts.jsonl");
  const manifestPath = path.join(outDir, "manifest.json");

  fs.writeFileSync(corpusPath, corpus, "utf8");
  writeJsonl(framesPath, frames);
  writeJsonl(trainPath, trainFrames);
  writeJsonl(evalPath, evalFrames);

  const allStations = [...new Set(frames.flatMap((frame) => frame.stations))].sort();
  const allCommodities = [...new Set(frames.flatMap((frame) => frame.commodities))].sort();
  const manifest = {
    schema: "nsrl.signal_romance_corpus.v1",
    created_at: new Date().toISOString(),
    corpus_path: corpusPath,
    frames_path: framesPath,
    train_frames_path: trainPath,
    eval_prompts_path: evalPath,
    signal_sft_path: resolveRepoPath(options.signalSft),
    signal_voice_path: resolveRepoPath(options.signalVoice),
    raw_signal_frames: parsed.rawFrames.length,
    unique_signal_frames: frames.length,
    train_frames: trainFrames.length,
    eval_frames: evalFrames.length,
    signal_repeat: options.signalRepeat,
    style_sources: style.manifest,
    style_source_dir: resolveRepoPath(options.styleSourceDir),
    corpus_bytes: Buffer.byteLength(corpus),
    known_stations: allStations,
    known_commodities: allCommodities,
    notes: [
      "crlplrimes remains the ranker/authority",
      "nsrl frames train the voice layer only",
      "eval prompts are held out from the flattened training frames",
      "Signal voice seed contributes doctrine only; canonical bullets are not copied into the corpus",
    ],
  };
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

  console.log(`corpus=${corpusPath}`);
  console.log(`manifest=${manifestPath}`);
  console.log(`frames=${frames.length}`);
  console.log(`train_frames=${trainFrames.length}`);
  console.log(`eval_frames=${evalFrames.length}`);
  console.log(`corpus_bytes=${manifest.corpus_bytes}`);
}

try {
  main();
} catch (error) {
  console.error(`build-signal-romance-corpus: ${error.message}`);
  process.exit(1);
}
