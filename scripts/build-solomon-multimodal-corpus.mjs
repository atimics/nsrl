#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

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
  corpusVersion: "v1",
  imageTokenProfile: "ink16",
};

const schema = "nsrl.solomon_multimodal_corpus.v1";
const PAD = 0;
const BOS = 1;
const PROMPT = 2;
const TEXT = 3;
const IMAGE = 4;
const EOS = 5;
const TASK_TEXT_TO_IMAGE = 6;
const TASK_IMAGE_TO_TEXT = 7;
const TASK_MATCH = 8;
const TASK_EXPLAIN = 9;
const TASK_IDENTIFY = 10;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
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
      "       [--corpus-version v1|v2] [--image-token-profile ink16|ink-edge16|symbolic16]",
      "",
      "Builds a discrete joint Solomon corpus:",
      "<BOS> <PROMPT> prompt bytes <TEXT> text bytes <IMAGE> 16x16 image-bin tokens <EOS>.",
      "With --corpus-version v2, also emits task-marked binding records for",
      "text-to-image, image-to-text, image-to-explain, text-image-explain, image-to-attributes, match, explain, and identify training.",
      "V2 also adds explicit primary-name, alias, and seal-ID identity bindings.",
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
  let imageTokenProfileExplicit = false;
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
    } else if (arg === "--corpus-version") {
      config.corpusVersion = requireValue(argv, ++index, arg);
      if (!["v1", "v2"].includes(config.corpusVersion)) {
        throw new Error("--corpus-version requires v1 or v2");
      }
    } else if (arg === "--image-token-profile") {
      config.imageTokenProfile = requireValue(argv, ++index, arg);
      imageTokenProfileExplicit = true;
      if (!["ink16", "ink-edge16", "symbolic16"].includes(config.imageTokenProfile)) {
        throw new Error("--image-token-profile requires ink16, ink-edge16, or symbolic16");
      }
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (TEXT_CHUNKS.length > VOCAB_SIZE - TEXT_CHUNK_BASE) {
    throw new Error(`TEXT_CHUNKS has ${TEXT_CHUNKS.length} entries, max ${VOCAB_SIZE - TEXT_CHUNK_BASE}`);
  }
  if (config.corpusVersion === "v2" && !imageTokenProfileExplicit) {
    config.imageTokenProfile = "symbolic16";
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
  const selected = descriptionForRow(row);
  return truncateText(`Solomon selects ${name}: ${selected}`, maxTextChars);
}

function descriptionForRow(row) {
  return selectSentence(row.text) || normalizeText(row.text);
}

function attributesForRow(row, maxTextChars) {
  const name = normalizeText(row.primary_name);
  const source = normalizeText(row.text);
  const rank = compactFact(rankForRow(row, source) || "not stated in source", 32);
  const legions = compactFact(cleanFact(firstMatch(source, [
    /\b[^.;]*\b(?:[0-9]+|sixty-six|thirty-one|twenty-six|thirty|forty|fifty|twenty|six)\s+legions?\b[^.;]*/i,
  ])) || "legions recorded in source", 40);
  const appearance = compactFact(selectNeedleSentence(source, [
    "appeareth",
    "appears",
    "form",
    "shape",
    "voice",
    "riding",
    "carrying",
  ]) || `${name} has an appearance described in the source`, 44);
  const office = compactFact(selectNeedleSentence(source, [
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
  ]) || descriptionForRow(row), 48);
  const render = (budgets) => {
    const [rankMax, legionsMax, appearanceMax, officeMax] = budgets;
    return `${name} attributes: rank ${compactFact(rank, rankMax)}; legions ${compactFact(legions, legionsMax)}; appearance ${compactFact(appearance, appearanceMax)}; office ${compactFact(office, officeMax)}`;
  };
  const budgets = [32, 40, 44, 48];
  let rendered = render(budgets);
  while (rendered.length > maxTextChars && Math.max(...budgets) > 24) {
    const index = budgets.indexOf(Math.max(...budgets));
    budgets[index] -= 4;
    rendered = render(budgets);
  }
  return truncateText(rendered, maxTextChars);
}

function sourceEvidenceForRow(row, maxTextChars) {
  const sourceText = normalizeText(row.text);
  const excerpt = truncateText(descriptionForRow(row), maxTextChars);
  return {
    source_spirit_id: row.number,
    source_text_hash: fnv64TextHex(sourceText),
    source_excerpt: excerpt,
    source_excerpt_hash: fnv64TextHex(excerpt),
  };
}

function rankForRow(row, source) {
  const name = normalizeText(row.primary_name);
  const rankWord = "(?:king|duke|prince|marquis|president|earl|count|knight)";
  const namePattern = escapeRegExp(name).replace(/\s+/g, "\\s+");
  return cleanFact(firstMatch(source, [
    new RegExp(`\\bis\\s+${namePattern}\\s*,\\s*((?:a|an)\\s+[^.;,]*?\\b${rankWord}\\b[^.;,]*)`, "i"),
    new RegExp(`\\b(?:he|it|this\\s+spirit)?\\s*(?:is|was)\\s+([^.;]*?\\b${rankWord}\\b)`, "i"),
    new RegExp(`\\bbeing\\s+himself\\s+([^.;]*?\\b${rankWord}\\b)`, "i"),
    new RegExp(`\\b(?:is|called)\\s+((?:a|an)\\s+[^.;,]*?\\b${rankWord}\\b[^.;,]*)`, "i"),
    new RegExp(`\\b(?:is|called)\\s+([^.;,]*?\\b${rankWord}\\b[^.;,]*)`, "i"),
    new RegExp(`\\border\\s+is\\s+((?:a|an)?\\s*[^.;,]*?\\b${rankWord}\\b[^.;,]*)`, "i"),
  ]));
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

function selectNeedleSentence(text, needles) {
  const sentences = normalizeText(text)
    .split(/(?<=[.!?])\s+|;\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 16);
  for (const sentence of sentences) {
    const folded = sentence.toLowerCase();
    if (needles.some((needle) => folded.includes(needle))) {
      return sentence;
    }
  }
  return "";
}

function firstMatch(text, regexes) {
  for (const regex of regexes) {
    const match = regex.exec(text);
    if (match) {
      return match[1] || match[0];
    }
  }
  return "";
}

function cleanFact(text) {
  return normalizeText(text)
    .replace(/^[,;:. -]+|[,;:. -]+$/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function compactFact(text, maxChars) {
  const compact = cleanFact(text);
  if (compact.length <= maxChars) {
    return compact;
  }
  const clipped = compact.slice(0, maxChars);
  const lastSpace = clipped.lastIndexOf(" ");
  return (lastSpace > 16 ? clipped.slice(0, lastSpace) : clipped).replace(/[,:;]+$/g, "").trim();
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

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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
  return solomonImage.imageTokens(signature, symbolicImageOptions());
}

function imageTaskTokens(signature, profile) {
  return solomonImage.imageTaskTokens(signature, profile, symbolicImageOptions());
}

function imageTokenChannels(profile) {
  return solomonImage.imageTokenChannels(profile);
}

function imageTokenChannelStats(rows, profile) {
  return solomonImage.imageTokenChannelStats(rows.map((row) => row.signature), profile, symbolicImageOptions());
}

function imageChannelTokens(signature, channel) {
  return solomonImage.imageChannelTokens(signature, channel, symbolicImageOptions());
}

function symbolicImageOptions() {
  return {
    grid: SIGNATURE_GRID,
    imageBase: IMAGE_BASE,
    imageBins: IMAGE_BINS,
    channelTokens: {
      ink: IMAGE_CHANNEL_INK,
      edge: IMAGE_CHANNEL_EDGE,
      component: IMAGE_CHANNEL_COMPONENT,
      radial: IMAGE_CHANNEL_RADIAL,
      direction: IMAGE_CHANNEL_DIRECTION,
    },
  };
}

function buildExamples(rows, config) {
  const examples = [];
  let tokenOffset = 0;
  const addSequence = (tokens, example) => {
    const padding = Array(config.padContext).fill(PAD);
    tokenOffset += padding.length;
    const entry = {
      example: example
        ? {
            ...example,
            token_offset: tokenOffset,
            token_count: tokens.length,
            token_hash: fnv64Hex(tokens),
            padding_before: padding.length,
          }
        : null,
      tokens,
      corpusTokens: [...padding, ...tokens],
    };
    examples.push(entry);
    tokenOffset += tokens.length;
  };

  for (const row of rows) {
    const text = textForRow(row, config.maxTextChars);
    const image = imageTokens(row.signature);
    if (config.corpusVersion === "v2") {
      for (const task of identityBindingSequencesForRow(row, config)) {
        addSequence(task.tokens, task.example);
      }
    }
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
        addSequence(tokens, {
          schema: "nsrl.solomon_multimodal_example.v1",
          task: "canonical-joint",
          spirit_id: row.number,
          primary_name: normalizeText(row.primary_name),
          prompt,
          text,
          image_grid: SIGNATURE_GRID,
          image_bins: IMAGE_BINS,
        });
      }
      const textOnlyCount =
        (config.sequenceProfile === "text-only" || config.sequenceProfile === "joint-and-text" ? 1 : 0) +
        config.textOnlyRepeats;
      for (let repeatIndex = 0; repeatIndex < textOnlyCount; repeatIndex += 1) {
        const textOnlyTokens = textOnlySequence(prompt, text, config.textTokenProfile);
        addSequence(textOnlyTokens, null);
      }
      for (let repeatIndex = 0; repeatIndex < config.nameInitialRepeats; repeatIndex += 1) {
        const initialTokens = textOnlySequence(prompt, nameInitialForRow(row), config.textTokenProfile);
        addSequence(initialTokens, null);
      }
      const nameOpeningCount =
        (config.sequenceProfile === "name-opening" ? 1 : 0) + config.nameOpeningRepeats;
      for (let repeatIndex = 0; repeatIndex < nameOpeningCount; repeatIndex += 1) {
        const openingTokens = textOnlySequence(prompt, nameOpeningForRow(row), config.textTokenProfile);
        addSequence(openingTokens, null);
      }
      if (config.corpusVersion === "v2") {
        for (const task of taskSequencesForRow(rows, row, prompt, config)) {
          addSequence(task.tokens, task.example);
        }
      }
    }
  }
  return examples;
}

function identityBindingSequencesForRow(row, config) {
  const name = normalizeText(row.primary_name);
  const description = truncateText(descriptionForRow(row), config.maxTextChars);
  const sourceEvidence = sourceEvidenceForRow(row, config.maxTextChars);
  const image = imageTaskTokens(row.signature, config.imageTokenProfile);
  const base = {
    schema: "nsrl.solomon_multimodal_example.v2",
    spirit_id: row.number,
    primary_name: name,
    identity_binding: true,
    image_grid: SIGNATURE_GRID,
    image_bins: IMAGE_BINS,
    image_token_profile: config.imageTokenProfile,
    image_token_channels: imageTokenChannels(config.imageTokenProfile),
  };
  const sequences = [];
  for (const binding of identityBindingPromptsForRow(row)) {
    sequences.push(
      {
        example: {
          ...base,
          task: "identify",
          binding_kind: binding.kind,
          prompt: binding.prompt,
          text: name,
        },
        tokens: taskTextSequence(TASK_IDENTIFY, binding.prompt, name, config.textTokenProfile),
      },
      {
        example: {
          ...base,
          ...sourceEvidence,
          task: "text-to-image",
          binding_kind: binding.kind,
          prompt: binding.prompt,
          source_query_kind: "identity-to-image",
          text: description,
        },
        tokens: taskImageSequence(TASK_TEXT_TO_IMAGE, binding.prompt, image, config.textTokenProfile),
      },
    );
  }
  return sequences;
}

function identityBindingPromptsForRow(row) {
  const prompts = [];
  const seen = new Set();
  const add = (kind, prompt) => {
    const normalized = normalizeText(prompt);
    const key = normalized.toLowerCase();
    if (!normalized || seen.has(key)) {
      return;
    }
    seen.add(key);
    prompts.push({ kind, prompt: normalized });
  };
  const name = normalizeText(row.primary_name);
  add("primary-name", name);
  add("primary-seal", `seal of ${name}`);
  for (const alias of String(row.aliases || "").split("|").map((value) => normalizeText(value)).filter(Boolean)) {
    add("alias", alias);
    add("alias-seal", `seal of ${alias}`);
  }
  add("seal-id", `seal id ${row.number}`);
  add("seal-id", `spirit ${row.number}`);
  add("seal-id", `goetic spirit ${row.number}`);
  return prompts;
}

function taskSequencesForRow(rows, row, prompt, config) {
  const name = normalizeText(row.primary_name);
  const taskPrompt = identityTaskPromptForRow(row, prompt);
  const text = truncateText(textForRow(row, config.maxTextChars), config.maxTextChars);
  const description = truncateText(descriptionForRow(row), config.maxTextChars);
  const attributes = attributesForRow(row, config.maxTextChars);
  const sourceEvidence = sourceEvidenceForRow(row, config.maxTextChars);
  const image = imageTaskTokens(row.signature, config.imageTokenProfile);
  const negative = hardNegativeRow(rows, row, taskPrompt, config.imageTokenProfile);
  const wrong = negative.row;
  const wrongImage = imageTaskTokens(wrong.signature, config.imageTokenProfile);
  const wrongPrompt = hardNegativePrompt(wrong, taskPrompt);
  const negativeEvidence = {
    negative_selection: negative.selection,
    negative_image_token_distance: negative.image_token_distance,
    negative_image_token_rank: negative.image_token_rank,
  };
  const attributePrompt = "seal attributes";
  const base = {
    schema: "nsrl.solomon_multimodal_example.v2",
    spirit_id: row.number,
    primary_name: name,
    prompt: taskPrompt,
    image_grid: SIGNATURE_GRID,
    image_bins: IMAGE_BINS,
    image_token_profile: config.imageTokenProfile,
    image_token_channels: imageTokenChannels(config.imageTokenProfile),
  };
  return [
    {
      example: { ...base, task: "identify", text: name },
      tokens: taskTextSequence(TASK_IDENTIFY, taskPrompt, name, config.textTokenProfile),
    },
    {
      example: { ...base, ...sourceEvidence, task: "text-to-image", source_query_kind: "identity-to-image", text: description },
      tokens: taskImageSequence(TASK_TEXT_TO_IMAGE, taskPrompt, image, config.textTokenProfile),
    },
    {
      example: { ...base, ...sourceEvidence, task: "image-to-text", source_query_kind: "image-identity", text: name },
      tokens: taskImageToTextSequence(TASK_IMAGE_TO_TEXT, image, name, config.textTokenProfile),
    },
    {
      example: { ...base, ...sourceEvidence, task: "image-to-explain", source_query_kind: "image-source", text },
      tokens: taskImageToTextSequence(TASK_EXPLAIN, image, text, config.textTokenProfile),
    },
    {
      example: { ...base, ...sourceEvidence, task: "text-image-explain", source_query_kind: "text-image-source", text },
      tokens: taskPromptImageToTextSequence(TASK_EXPLAIN, taskPrompt, image, text, config.textTokenProfile),
    },
    {
      example: {
        ...base,
        ...sourceEvidence,
        task: "image-to-attributes",
        prompt: attributePrompt,
        source_query_kind: "image-attributes",
        text: attributes,
      },
      tokens: taskImagePromptToTextSequence(TASK_EXPLAIN, image, attributePrompt, attributes, config.textTokenProfile),
    },
    {
      example: { ...base, ...sourceEvidence, task: "explain", prompt: name, source_query_kind: "primary-name", text },
      tokens: taskTextSequence(TASK_EXPLAIN, name, text, config.textTokenProfile),
    },
    {
      example: {
        ...base,
        ...sourceEvidence,
        task: "description-to-image",
        prompt: description,
        source_query_kind: "source-description",
        text: description,
      },
      tokens: taskImageSequence(TASK_TEXT_TO_IMAGE, description, image, config.textTokenProfile),
    },
    {
      example: { ...base, task: "match", text: "yes", match_label: "yes" },
      tokens: taskMatchSequence(taskPrompt, image, "yes", config.textTokenProfile),
    },
    {
      example: {
        ...base,
        task: "match",
        text: "no",
        match_label: "no",
        negative_role: "image",
        negative_spirit_id: wrong.number,
        negative_primary_name: normalizeText(wrong.primary_name),
        ...negativeEvidence,
      },
      tokens: taskMatchSequence(taskPrompt, wrongImage, "no", config.textTokenProfile),
    },
    {
      example: {
        ...base,
        task: "match",
        prompt: wrongPrompt,
        text: "no",
        match_label: "no",
        negative_role: "prompt",
        negative_spirit_id: wrong.number,
        negative_primary_name: normalizeText(wrong.primary_name),
        ...negativeEvidence,
      },
      tokens: taskMatchSequence(wrongPrompt, image, "no", config.textTokenProfile),
    },
  ];
}

function identityTaskPromptForRow(row, prompt) {
  if (promptMentionsRowIdentity(row, prompt)) {
    return normalizeText(prompt);
  }
  return `seal of ${normalizeText(row.primary_name)}`;
}

function promptMentionsRowIdentity(row, prompt) {
  const query = phraseKey(prompt);
  if (!query) {
    return false;
  }
  const names = unique([
    row.primary_name,
    ...String(row.aliases || "").split("|"),
  ]);
  return names.some((name) => containsPhraseKey(query, phraseKey(name)));
}

function containsPhraseKey(haystack, needle) {
  return Boolean(needle) && (` ${haystack} `).includes(` ${needle} `);
}

function phraseKey(value) {
  return normalizeText(value).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function taskTextSequence(taskToken, prompt, text, profile) {
  return [
    BOS,
    taskToken,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    TEXT,
    ...encodeTextTokens(text, profile),
    EOS,
  ];
}

function taskImageSequence(taskToken, prompt, image, profile) {
  return [
    BOS,
    taskToken,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    IMAGE,
    ...image,
    EOS,
  ];
}

function taskImageToTextSequence(taskToken, image, text, profile) {
  return [
    BOS,
    taskToken,
    IMAGE,
    ...image,
    TEXT,
    ...encodeTextTokens(text, profile),
    EOS,
  ];
}

function taskPromptImageToTextSequence(taskToken, prompt, image, text, profile) {
  return [
    BOS,
    taskToken,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    IMAGE,
    ...image,
    TEXT,
    ...encodeTextTokens(text, profile),
    EOS,
  ];
}

function taskImagePromptToTextSequence(taskToken, image, prompt, text, profile) {
  return [
    BOS,
    taskToken,
    IMAGE,
    ...image,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    TEXT,
    ...encodeTextTokens(text, profile),
    EOS,
  ];
}

function taskMatchSequence(prompt, image, label, profile) {
  return [
    BOS,
    TASK_MATCH,
    PROMPT,
    ...encodeTextTokens(prompt, profile),
    IMAGE,
    ...image,
    TEXT,
    ...encodeTextTokens(label, profile),
    EOS,
  ];
}

function hardNegativeRow(rows, row, prompt, imageTokenProfile) {
  const sourceImage = imageTaskTokens(row.signature, imageTokenProfile);
  const salt = hashString(`${row.number}:${prompt}:hard-negative`);
  const candidates = rows
    .filter((candidate) => candidate.number !== row.number)
    .map((candidate) => ({
      row: candidate,
      image_token_distance: imageTokenDistance(sourceImage, imageTaskTokens(candidate.signature, imageTokenProfile)),
      tie_breaker: hashString(`${salt}:${candidate.number}:nearest-image-token`),
    }))
    .sort((left, right) =>
      left.image_token_distance - right.image_token_distance ||
      left.tie_breaker - right.tie_breaker ||
      left.row.number - right.row.number,
    );
  if (candidates.length === 0) {
    throw new Error(`no hard-negative candidate for spirit ${row.number}`);
  }
  return {
    row: candidates[0].row,
    selection: "nearest-image-token",
    image_token_distance: candidates[0].image_token_distance,
    image_token_rank: 1,
  };
}

function imageTokenDistance(left, right) {
  const count = Math.min(left.length, right.length);
  let distance = Math.abs(left.length - right.length) * IMAGE_BINS;
  for (let index = 0; index < count; index += 1) {
    distance += Math.abs((left[index] || 0) - (right[index] || 0));
  }
  return distance;
}

function hardNegativePrompt(row, sourcePrompt) {
  const name = normalizeText(row.primary_name);
  const options = unique([
    name,
    `seal of ${name}`,
    `${name} goetic seal`,
  ]);
  const index = hashString(`${row.number}:${sourcePrompt}:hard-negative-prompt`) % options.length;
  return options[index];
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
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
    [TASK_TEXT_TO_IMAGE, "TASK_TEXT_TO_IMAGE"],
    [TASK_IMAGE_TO_TEXT, "TASK_IMAGE_TO_TEXT"],
    [TASK_MATCH, "TASK_MATCH"],
    [TASK_EXPLAIN, "TASK_EXPLAIN"],
    [TASK_IDENTIFY, "TASK_IDENTIFY"],
    [IMAGE_CHANNEL_INK, "IMAGE_CHANNEL_INK"],
    [IMAGE_CHANNEL_EDGE, "IMAGE_CHANNEL_EDGE"],
    [IMAGE_CHANNEL_COMPONENT, "IMAGE_CHANNEL_COMPONENT"],
    [IMAGE_CHANNEL_RADIAL, "IMAGE_CHANNEL_RADIAL"],
    [IMAGE_CHANNEL_DIRECTION, "IMAGE_CHANNEL_DIRECTION"],
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
    if (Number(token) < 0 || Number(token) > 255) {
      throw new Error(`token ${token} is outside byte range`);
    }
    hash ^= BigInt(token & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64TextHex(value) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of Buffer.from(String(value), "utf8")) {
    hash ^= BigInt(Number(byte) & 0xff);
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
        corpus_version: config.corpusVersion,
        image_token_profile: config.imageTokenProfile,
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
        image_token_channels: imageTokenChannels(config.imageTokenProfile),
        image_token_channel_stats: imageTokenChannelStats(rows, config.imageTokenProfile),
        vocab_size: VOCAB_SIZE,
        token_layout: {
          pad: PAD,
          bos: BOS,
          prompt: PROMPT,
          text: TEXT,
          image: IMAGE,
          eos: EOS,
          task_text_to_image: TASK_TEXT_TO_IMAGE,
          task_image_to_text: TASK_IMAGE_TO_TEXT,
          task_match: TASK_MATCH,
          task_explain: TASK_EXPLAIN,
          task_identify: TASK_IDENTIFY,
          image_channel_ink: IMAGE_CHANNEL_INK,
          image_channel_edge: IMAGE_CHANNEL_EDGE,
          image_channel_component: IMAGE_CHANNEL_COMPONENT,
          image_channel_radial: IMAGE_CHANNEL_RADIAL,
          image_channel_direction: IMAGE_CHANNEL_DIRECTION,
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
      corpus_version: config.corpusVersion,
      examples: jointExamples.length,
      training_sequences: examples.length,
      token_count: tokens.length,
      token_hash: fnv64Hex(tokens),
    }),
  );
}

main();
