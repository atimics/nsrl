#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  textIndex: "web/assets/solomon-spirit-text-signatures.tsv",
  outDir: "data/processed/key-solomon-goetia-multimodal-v1",
  maxTextChars: 220,
  promptProfile: "all",
  padContext: 0,
  sequenceProfile: "joint",
  textOnlyRepeats: 0,
  nameInitialRepeats: 0,
  nameOpeningRepeats: 0,
  textTokenProfile: "char",
};

const schema = "nsrl.solomon_multimodal_corpus.v1";
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
const TEXT_CHUNK_BASE = 160;
const TEXT_CHUNKS = [
  "Solomon selects ",
  ": ",
  "He ",
  "is ",
  "appeareth ",
  "maketh ",
  "teacheth ",
  "giveth ",
  "causeth ",
  "knoweth ",
  "healeth ",
  "teaches ",
  "and ",
  "the ",
  "of ",
  "to ",
  "in ",
  "a ",
  "his ",
  "with ",
  "upon ",
  "unto ",
  "This ",
  "His ",
  "Bael",
  "Agares",
  "Vassago",
  "Samigina",
  "Marbas",
  "Valefor",
  "Amon",
  "Barbatos",
  "Paimon",
  "Buer",
  "Gusion",
  "Sitri",
  "Beleth",
  "Leraje",
  "Eligos",
  "Zepar",
  "Botis",
  "Bathin",
  "Sallos",
  "Purson",
  "Marax",
  "Ipos",
  "Aim",
  "Naberius",
  "Glasya-Labolas",
  "Bune",
  "Ronove",
  "Berith",
  "Astaroth",
  "Forneus",
  "Foras",
  "Asmoday",
  "Gaap",
  "Furfur",
  "Marchosias",
  "Stolas",
  "Phenex",
  "Halphas",
  "Malphas",
  "Raum",
  "Focalor",
  "Vepar",
  "Sabnock",
  "Shax",
  "Vine",
  "Bifrons",
  "Uvall",
  "Haagenti",
  "Crocell",
  "Furcas",
  "Balam",
  "Alloces",
  "Camio",
  "Murmur",
  "Orobas",
  "Gremory",
  "Ose",
  "Amy",
  "Oriax",
  "Vapula",
  "Zagan",
  "Volac",
  "Andras",
  "Haures",
  "Andrealphus",
  "Cimejes",
  "Amdusias",
  "Belial",
  "Decarabia",
  "Seere",
  "Dantalion",
  "Andromalius",
];
const SIGNATURE_GRID = 16;
const SIGNATURE_BINS = SIGNATURE_GRID * SIGNATURE_GRID;
const VOCAB_SIZE = 256;

function usage() {
  console.log(
    [
      "Usage: build-solomon-multimodal-corpus.mjs [--text-index PATH] [--out-dir PATH]",
      "       [--max-text-chars N] [--prompt-profile generic|names|seal-names|all]",
      "       [--pad-context N] [--sequence-profile joint|text-only|name-opening|joint-and-text]",
      "       [--text-only-repeats N] [--name-initial-repeats N]",
      "       [--name-opening-repeats N]",
      "       [--text-token-profile char|chunked]",
      "",
      "Builds a discrete joint Solomon corpus:",
      "<BOS> <PROMPT> prompt bytes <TEXT> text bytes <IMAGE> 16x16 image-bin tokens <EOS>.",
      "Optional --pad-context inserts PAD tokens before each example so fixed-window",
      "attention training sees early record positions.",
      "Optional --text-only-repeats adds prompt/text/EOS training-only sequences",
      "to rebalance text targets without changing joint examples.",
      "Optional --name-initial-repeats adds prompt/name-initial sequences",
      "to train the first token after Solomon selects.",
      "Optional --name-opening-repeats adds short prompt/name-opening sequences",
      "to directly train prompt-conditioned continuation after Solomon selects.",
      "Optional --text-token-profile chunked uses reserved byte-vocab IDs",
      "160..255 for fixed common Solomon text chunks.",
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
    } else if (arg === "--max-text-chars") {
      config.maxTextChars = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--prompt-profile") {
      config.promptProfile = requireValue(argv, ++index, arg);
      if (!["generic", "names", "seal-names", "all"].includes(config.promptProfile)) {
        throw new Error("--prompt-profile requires generic, names, seal-names, or all");
      }
    } else if (arg === "--pad-context") {
      config.padContext = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--sequence-profile") {
      config.sequenceProfile = requireValue(argv, ++index, arg);
      if (!["joint", "text-only", "name-opening", "joint-and-text"].includes(config.sequenceProfile)) {
        throw new Error("--sequence-profile requires joint, text-only, name-opening, or joint-and-text");
      }
    } else if (arg === "--text-only-repeats") {
      config.textOnlyRepeats = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--name-initial-repeats") {
      config.nameInitialRepeats = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--name-opening-repeats") {
      config.nameOpeningRepeats = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--text-token-profile") {
      config.textTokenProfile = requireValue(argv, ++index, arg);
      if (!["char", "chunked"].includes(config.textTokenProfile)) {
        throw new Error("--text-token-profile requires char or chunked");
      }
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (TEXT_CHUNKS.length > VOCAB_SIZE - TEXT_CHUNK_BASE) {
    throw new Error(`TEXT_CHUNKS has ${TEXT_CHUNKS.length} entries, max ${VOCAB_SIZE - TEXT_CHUNK_BASE}`);
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePositiveInteger(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parseNonNegativeInteger(value, flag) {
  if (!/^(0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
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
  for (const column of ["number", "primary_name", "aliases", "text"]) {
    if (!header.includes(column)) {
      throw new Error(`${tsvPath} is missing required column ${column}`);
    }
  }
  const signatureColumn = header.includes("signature_16x16")
    ? "signature_16x16"
    : header.find((column) => /^signature_16x16$/.test(column));
  if (!signatureColumn) {
    throw new Error(`${tsvPath} must contain signature_16x16`);
  }
  return lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.number = Number(row.number);
    if (!Number.isInteger(row.number) || row.number < 1 || row.number > 72) {
      throw new Error(`${tsvPath} row ${rowIndex + 2} has invalid spirit number`);
    }
    row.signature = parseSignature(row[signatureColumn], tsvPath, rowIndex + 2);
    return row;
  });
}

function parseSignature(value, source, lineNumber) {
  const bins = String(value)
    .split(",")
    .map((part) => {
      if (!/^[0-9]+$/.test(part.trim())) {
        throw new Error(`${source}:${lineNumber} has a non-integer signature bin`);
      }
      const parsed = Number(part.trim());
      if (parsed < 0 || parsed > 255) {
        throw new Error(`${source}:${lineNumber} has a signature bin outside u8`);
      }
      return parsed;
    });
  if (bins.length !== SIGNATURE_BINS) {
    throw new Error(`${source}:${lineNumber} has ${bins.length} signature bins, expected ${SIGNATURE_BINS}`);
  }
  return bins;
}

function promptsForRow(row, profile) {
  const aliases = String(row.aliases || "")
    .split("|")
    .map((alias) => normalizeText(alias))
    .filter(Boolean);
  const name = normalizeText(row.primary_name);
  const generic = ["king solomon seal"];
  const names = unique([
    name,
    `seal of ${name}`,
    `${name} goetic seal`,
    ...aliases.map((alias) => `seal of ${alias}`),
  ]);
  if (profile === "generic") {
    return generic;
  }
  if (profile === "names") {
    return names;
  }
  if (profile === "seal-names") {
    return [`seal of ${name}`];
  }
  return unique([...generic, ...names]);
}

function textForRow(row, maxTextChars) {
  const name = normalizeText(row.primary_name);
  const selected = selectSentence(row.text) || normalizeText(row.text);
  return truncateText(`Solomon selects ${name}: ${selected}`, maxTextChars);
}

function nameOpeningForRow(row) {
  const name = normalizeText(row.primary_name);
  return `Solomon selects ${name}: He `;
}

function nameInitialForRow(row) {
  const name = normalizeText(row.primary_name);
  return `Solomon selects ${name.slice(0, 1)}`;
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
  const scored = sentences.map((sentence, index) => {
    const folded = sentence.toLowerCase();
    let score = 0;
    for (const needle of needles) {
      if (folded.includes(needle)) {
        score += 1;
      }
    }
    return { score, index, sentence };
  });
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

function encodeTextTokens(text, profile) {
  const normalized = normalizeText(text);
  const tokens = [];
  for (let index = 0; index < normalized.length;) {
    if (profile === "chunked") {
      const match = matchTextChunk(normalized, index);
      if (match) {
        tokens.push(TEXT_CHUNK_BASE + match.chunkIndex);
        index += match.chunk.length;
        continue;
      }
    }
    const code = normalized.charCodeAt(index);
    if (code >= TEXT_COUNT) {
      tokens.push(TEXT_BASE + "?".charCodeAt(0));
    } else {
      tokens.push(TEXT_BASE + code);
    }
    index += 1;
  }
  return tokens;
}

function matchTextChunk(text, index) {
  let best = null;
  for (let chunkIndex = 0; chunkIndex < TEXT_CHUNKS.length; chunkIndex += 1) {
    const chunk = TEXT_CHUNKS[chunkIndex];
    if (!text.startsWith(chunk, index)) {
      continue;
    }
    if (!best || chunk.length > best.chunk.length) {
      best = { chunkIndex, chunk };
    }
  }
  return best;
}

function imageTokens(signature) {
  return signature.map((value) => IMAGE_BASE + Math.min(IMAGE_BINS - 1, Math.floor((value * IMAGE_BINS) / 256)));
}

function buildExamples(rows, config) {
  const examples = [];
  let tokenOffset = 0;
  for (const row of rows) {
    const text = textForRow(row, config.maxTextChars);
    const image = imageTokens(row.signature);
    for (const prompt of promptsForRow(row, config.promptProfile)) {
      const tokens = [
        BOS,
        PROMPT,
        ...encodeTextTokens(prompt, config.textTokenProfile),
        TEXT,
        ...encodeTextTokens(text, config.textTokenProfile),
        IMAGE,
        ...image,
        EOS,
      ];
      if (config.sequenceProfile === "joint" || config.sequenceProfile === "joint-and-text") {
        const padding = Array(config.padContext).fill(PAD);
        tokenOffset += padding.length;
        const example = {
          schema: "nsrl.solomon_multimodal_example.v1",
          spirit_id: row.number,
          primary_name: normalizeText(row.primary_name),
          prompt,
          text,
          image_grid: SIGNATURE_GRID,
          image_bins: IMAGE_BINS,
          token_offset: tokenOffset,
          token_count: tokens.length,
          token_hash: fnv64Hex(tokens),
          padding_before: padding.length,
        };
        examples.push({ example, tokens, corpusTokens: [...padding, ...tokens] });
        tokenOffset += tokens.length;
      }
      const textOnlyCount =
        (config.sequenceProfile === "text-only" || config.sequenceProfile === "joint-and-text" ? 1 : 0) +
        config.textOnlyRepeats;
      for (let repeatIndex = 0; repeatIndex < textOnlyCount; repeatIndex += 1) {
        const textOnlyTokens = textOnlySequence(prompt, text, config.textTokenProfile);
        const textOnlyPadding = Array(config.padContext).fill(PAD);
        tokenOffset += textOnlyPadding.length;
        examples.push({
          example: null,
          tokens: textOnlyTokens,
          corpusTokens: [...textOnlyPadding, ...textOnlyTokens],
        });
        tokenOffset += textOnlyTokens.length;
      }
      for (let repeatIndex = 0; repeatIndex < config.nameInitialRepeats; repeatIndex += 1) {
        const initialTokens = textOnlySequence(prompt, nameInitialForRow(row), config.textTokenProfile);
        const initialPadding = Array(config.padContext).fill(PAD);
        tokenOffset += initialPadding.length;
        examples.push({
          example: null,
          tokens: initialTokens,
          corpusTokens: [...initialPadding, ...initialTokens],
        });
        tokenOffset += initialTokens.length;
      }
      const nameOpeningCount =
        (config.sequenceProfile === "name-opening" ? 1 : 0) + config.nameOpeningRepeats;
      for (let repeatIndex = 0; repeatIndex < nameOpeningCount; repeatIndex += 1) {
        const openingTokens = textOnlySequence(prompt, nameOpeningForRow(row), config.textTokenProfile);
        const openingPadding = Array(config.padContext).fill(PAD);
        tokenOffset += openingPadding.length;
        examples.push({
          example: null,
          tokens: openingTokens,
          corpusTokens: [...openingPadding, ...openingTokens],
        });
        tokenOffset += openingTokens.length;
      }
    }
  }
  return examples;
}

function textOnlySequence(prompt, text, profile) {
  return [
    BOS,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    TEXT,
    ...encodeTextTokens(text, profile),
    EOS,
  ];
}

function writeTokens(tokens, outPath) {
  const bytes = Buffer.alloc(tokens.length * 2);
  for (let index = 0; index < tokens.length; index += 1) {
    bytes.writeUInt16LE(tokens[index], index * 2);
  }
  fs.writeFileSync(outPath, bytes);
}

function writeByteTokens(tokens, outPath) {
  const bytes = Buffer.alloc(tokens.length);
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token < 0 || token > 255) {
      throw new Error(`token ${token} does not fit in u8`);
    }
    bytes[index] = token;
  }
  fs.writeFileSync(outPath, bytes);
}

function writeVocab(outPath) {
  const lines = ["token_id\ttype\tvalue"];
  const specials = new Map([
    [PAD, "PAD"],
    [BOS, "BOS"],
    [PROMPT, "PROMPT"],
    [TEXT, "TEXT"],
    [IMAGE, "IMAGE"],
    [EOS, "EOS"],
  ]);
  for (let token = 0; token < VOCAB_SIZE; token += 1) {
    if (specials.has(token)) {
      lines.push(`${token}\tspecial\t${specials.get(token)}`);
    } else if (token >= TEXT_BASE && token < TEXT_BASE + TEXT_COUNT) {
      const code = token - TEXT_BASE;
      const printable = code >= 32 && code <= 126 ? String.fromCharCode(code) : `byte_${code}`;
      lines.push(`${token}\ttext\t${escapeTsv(printable)}`);
    } else if (token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS) {
      lines.push(`${token}\timage_bin\t${token - IMAGE_BASE}`);
    } else if (token >= TEXT_CHUNK_BASE && token < TEXT_CHUNK_BASE + TEXT_CHUNKS.length) {
      lines.push(`${token}\ttext_chunk\t${escapeTsv(TEXT_CHUNKS[token - TEXT_CHUNK_BASE])}`);
    } else {
      lines.push(`${token}\treserved\t`);
    }
  }
  fs.writeFileSync(outPath, `${lines.join("\n")}\n`, "utf8");
}

function escapeTsv(value) {
  return String(value ?? "").replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function fnv64Hex(tokens) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const token of tokens) {
    hash ^= BigInt(token & 0xff);
    hash = (hash * prime) & mask;
    hash ^= BigInt((token >> 8) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const rows = readRows(config.textIndex);
  if (rows.length !== 72) {
    throw new Error(`expected 72 Solomon rows, found ${rows.length}`);
  }
  const examples = buildExamples(rows, config);
  const tokens = examples.flatMap((entry) => entry.corpusTokens);

  fs.mkdirSync(config.outDir, { recursive: true });
  const tokenPath = path.join(config.outDir, "corpus.tokens.u16");
  const byteTokenPath = path.join(config.outDir, "corpus.tokens.u8");
  const examplePath = path.join(config.outDir, "examples.jsonl");
  const vocabPath = path.join(config.outDir, "vocab.tsv");
  const manifestPath = path.join(config.outDir, "manifest.json");

  writeTokens(tokens, tokenPath);
  writeByteTokens(tokens, byteTokenPath);
  const jointExamples = examples.filter((entry) => entry.example);
  fs.writeFileSync(examplePath, `${jointExamples.map((entry) => JSON.stringify(entry.example)).join("\n")}\n`, "utf8");
  writeVocab(vocabPath);
  fs.writeFileSync(
    manifestPath,
    `${JSON.stringify(
      {
        schema,
        source_text_index: config.textIndex,
        rows: rows.length,
        examples: jointExamples.length,
        training_sequences: examples.length,
        prompt_profile: config.promptProfile,
        sequence_profile: config.sequenceProfile,
        text_only_repeats: config.textOnlyRepeats,
        name_initial_repeats: config.nameInitialRepeats,
        name_opening_repeats: config.nameOpeningRepeats,
        text_token_profile: config.textTokenProfile,
        text_chunk_base: TEXT_CHUNK_BASE,
        text_chunks: TEXT_CHUNKS.length,
        token_count: tokens.length,
        token_hash: fnv64Hex(tokens),
        pad_context_tokens: config.padContext,
        signature_grid: SIGNATURE_GRID,
        signature_bins: SIGNATURE_BINS,
        image_bins: IMAGE_BINS,
        vocab_size: VOCAB_SIZE,
        token_layout: {
          pad: PAD,
          bos: BOS,
          prompt: PROMPT,
          text: TEXT,
          image: IMAGE,
          eos: EOS,
          text_base: TEXT_BASE,
          text_count: TEXT_COUNT,
          image_base: IMAGE_BASE,
          image_bins: IMAGE_BINS,
          text_chunk_base: TEXT_CHUNK_BASE,
          text_chunks: TEXT_CHUNKS.length,
        },
        corpus_tokens_u16: path.relative(config.outDir, tokenPath),
        corpus_tokens_u8: path.relative(config.outDir, byteTokenPath),
        examples_jsonl: path.relative(config.outDir, examplePath),
        vocab_tsv: path.relative(config.outDir, vocabPath),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  console.log(
    JSON.stringify({
      schema,
      out_dir: config.outDir,
      examples: jointExamples.length,
      training_sequences: examples.length,
      token_count: tokens.length,
      token_hash: fnv64Hex(tokens),
    }),
  );
}

main();
