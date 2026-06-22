#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const defaults = {
  outDir: "data/processed/crowley-bard-aphorism-v2",
  shakespeareSource: "data/processed/crowley-bard-focused-v1/shakespeare.body.txt",
  minChars: 60,
  maxChars: 240,
  crowleyBytes: 420000,
  blakeBytes: 160000,
  shakespeareBytes: 100000,
};

const sourceOrder = ["crowley", "crowley", "crowley", "crowley", "blake", "shakespeare"];
const flavorTerms = new Set([
  "abyss",
  "angel",
  "beast",
  "child",
  "delight",
  "dream",
  "ecstasy",
  "fire",
  "flame",
  "god",
  "heaven",
  "hell",
  "holy",
  "joy",
  "kiss",
  "law",
  "light",
  "love",
  "moon",
  "night",
  "nuit",
  "pan",
  "secret",
  "serpent",
  "soul",
  "star",
  "sun",
  "will",
]);
const bannedSubstrings = [
  "project gutenberg",
  "public domain",
  "distributed proofreaders",
  "chapter ",
  "contents",
  "footnote",
  "references",
  "wikisource",
  "william blake",
  "songs of innocence",
  "songs of experience",
  "book of thel",
  "the divine image",
  "holy thursday",
  "nurse's song",
  "infant joy",
  "john w. luce",
  "r. brimley",
  "by the revd",
  "c. verey",
  "mr crowley",
  "middle-class",
  "second-hand",
  "classes of literature",
  "http",
  "www",
  "class=",
  "align=",
  "bgcolor",
  "nbsp",
  "act i",
  "act ii",
  "act iii",
  "act iv",
  "act v",
  "scene i",
  "scene ii",
  "scene iii",
  "scene iv",
  "dramatis",
  "enter ",
  "exeunt",
  "chorus.",
  "william shakespeare",
  "aleister crowley",
  "parolles",
  "helena",
  "bertram",
  "lafeu",
  "hamlet",
  "horatio",
  "ophelia",
  "polonius",
  "romeo",
  "juliet",
  "othello",
  "iago",
  "falstaff",
  "prospero",
  "caliban",
  "macbeth",
  "banquo",
  "gloucester",
  "cassio",
];
const danglingEndWords = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "for",
  "from",
  "if",
  "in",
  "is",
  "nor",
  "of",
  "on",
  "or",
  "so",
  "than",
  "that",
  "the",
  "these",
  "this",
  "to",
  "was",
  "were",
  "when",
  "where",
  "while",
  "who",
  "with",
]);
const stopWords = new Set([
  "a",
  "all",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "for",
  "from",
  "has",
  "hath",
  "have",
  "he",
  "her",
  "him",
  "his",
  "i",
  "in",
  "is",
  "it",
  "me",
  "my",
  "not",
  "of",
  "on",
  "or",
  "our",
  "she",
  "so",
  "that",
  "the",
  "thee",
  "their",
  "them",
  "thou",
  "thy",
  "to",
  "we",
  "with",
  "ye",
  "you",
  "your",
]);

function usage() {
  console.log(`Usage: node scripts/build-crowley-bard-aphorism-corpus.mjs [options]

Options:
  --out-dir PATH             Output directory [${defaults.outDir}]
  --shakespeare-source PATH  Shakespeare source text [${defaults.shakespeareSource}]
  --crowley-bytes N          Target selected Crowley bytes [${defaults.crowleyBytes}]
  --blake-bytes N            Target selected Blake bytes [${defaults.blakeBytes}]
  --shakespeare-bytes N      Target selected Shakespeare bytes [${defaults.shakespeareBytes}]
  --min-chars N              Minimum aphorism chars [${defaults.minChars}]
  --max-chars N              Maximum aphorism chars [${defaults.maxChars}]
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
    const key = arg.slice(2).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    if (!(key in options)) {
      throw new Error(`unknown option: ${arg}`);
    }
    const value = argv[++index];
    if (value === undefined) {
      throw new Error(`${arg} requires a value`);
    }
    if (["minChars", "maxChars", "crowleyBytes", "blakeBytes", "shakespeareBytes"].includes(key)) {
      const parsed = Number.parseInt(value, 10);
      if (!Number.isFinite(parsed) || parsed < 1) {
        throw new Error(`${arg} requires a positive integer`);
      }
      options[key] = parsed;
    } else {
      options[key] = value;
    }
  }
  if (options.minChars > options.maxChars) {
    throw new Error("--min-chars cannot be larger than --max-chars");
  }
  return options;
}

function repoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function existing(paths) {
  return paths.map(repoPath).filter((filePath) => fs.existsSync(filePath));
}

function listCleanFiles(dir, prefix) {
  const abs = repoPath(dir);
  if (!fs.existsSync(abs)) return [];
  return fs
    .readdirSync(abs)
    .filter((name) => name.startsWith(prefix) && name.endsWith(".clean.txt"))
    .sort()
    .map((name) => path.join(abs, name));
}

function readSource(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function normalizeText(text) {
  return String(text ?? "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .replace(/&/g, " and ")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .toLowerCase();
}

function cleanUnit(text) {
  let unit = normalizeText(text)
    .replace(/^\s*(?:\d+|[ivxlcdm]+)[.)]?\s+/i, "")
    .replace(/["()[\]{}]/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/[-:;,]\s*$/g, "")
    .trim();
  if (unit && !/[.!?]$/.test(unit)) {
    unit += ".";
  }
  return unit;
}

function wordsOf(text) {
  return text.match(/[a-z][a-z']*/g) || [];
}

function repeatedTrigramCount(words) {
  const seen = new Set();
  let repeats = 0;
  for (let index = 0; index + 2 < words.length; index += 1) {
    const key = `${words[index]} ${words[index + 1]} ${words[index + 2]}`;
    if (seen.has(key)) repeats += 1;
    seen.add(key);
  }
  return repeats;
}

function rejectionReason(unit, options) {
  if (unit.length < options.minChars) return "too_short";
  if (unit.length > options.maxChars) return "too_long";
  if (bannedSubstrings.some((term) => unit.includes(term))) return "banned_term";
  if (/^(?:by|preface|poems of|songs of|the marriage of heaven and hell)\b/.test(unit)) return "title_line";
  if (/^[a-z][a-z'-]{1,24}\.\s+[a-z]/.test(unit)) return "speaker_label";
  if (/\b(?:liber|sub figura)\b/.test(unit) && unit.length < 90) return "title_line";
  if (/\b(?:act|scene)\s+[ivxlcdm]+\b/.test(unit)) return "stage_line";
  if (/[{}<>_=\\]/.test(unit)) return "markup_punct";
  if ((unit.match(/\d/g) || []).length > 3) return "too_many_digits";
  if ((unit.match(/[;:!?.,]/g) || []).length > 8) return "punctuation_heavy";

  const words = wordsOf(unit);
  if (words.length < 8) return "too_few_words";
  const counts = new Map();
  for (const word of words) counts.set(word, (counts.get(word) || 0) + 1);
  if (Math.max(...counts.values()) > 3) return "word_repeat";
  if (counts.size / words.length < 0.62) return "low_distinct_ratio";
  if (repeatedTrigramCount(words) > 0) return "repeated_trigram";
  if (danglingEndWords.has(words[words.length - 1])) return "dangling_end";
  const stopRatio = words.filter((word) => stopWords.has(word)).length / words.length;
  if (stopRatio > 0.68) return "stopword_heavy";
  return "";
}

function scoreUnit(unit) {
  const words = wordsOf(unit);
  const unique = new Set(words);
  const flavor = words.filter((word) => flavorTerms.has(word)).length;
  const stopRatio = words.filter((word) => stopWords.has(word)).length / Math.max(1, words.length);
  let score = 100;
  if (unit.length >= 90 && unit.length <= 180) score += 24;
  score += Math.min(36, flavor * 6);
  score += Math.min(18, Math.round((unique.size / Math.max(1, words.length)) * 18));
  if (/[.!?]$/.test(unit)) score += 8;
  score -= Math.max(0, Math.round((stopRatio - 0.48) * 80));
  return score;
}

function pushCandidate(rawCandidates, raw, source, sourceFile, sourceIndex, options, rejectedCounts) {
  const text = cleanUnit(raw);
  const reason = rejectionReason(text, options);
  if (reason) {
    rejectedCounts.set(reason, (rejectedCounts.get(reason) || 0) + 1);
    return;
  }
  rawCandidates.push({
    source,
    source_file: path.relative(repoRoot, sourceFile),
    source_index: sourceIndex,
    score: scoreUnit(text),
    chars: text.length,
    text,
  });
}

function extractCandidates(source, sourceFiles, options) {
  const candidates = [];
  const rejectedCounts = new Map();
  for (const sourceFile of sourceFiles) {
    const normalized = normalizeText(readSource(sourceFile));
    const lines = normalized
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .filter((line) => !/^(?:preface|contents|the end|chapter|book [ivxlcdm]+)$/i.test(line));

    let buffer = "";
    let sourceIndex = 0;
    for (const line of lines) {
      if (line.length > options.maxChars * 2) {
        for (const sentence of line.split(/(?<=[.!?])\s+/)) {
          pushCandidate(candidates, sentence, source, sourceFile, sourceIndex++, options, rejectedCounts);
        }
        continue;
      }
      buffer = buffer ? `${buffer} ${line}` : line;
      if (/[.!?]$/.test(line) || buffer.length >= options.minChars) {
        pushCandidate(candidates, buffer, source, sourceFile, sourceIndex++, options, rejectedCounts);
        buffer = "";
      }
    }
    if (buffer) pushCandidate(candidates, buffer, source, sourceFile, sourceIndex++, options, rejectedCounts);

    const paragraphText = normalized.replace(/\n{2,}/g, "\n\n");
    for (const paragraph of paragraphText.split(/\n\n+/)) {
      const flat = paragraph.replace(/\s+/g, " ").trim();
      if (!flat) continue;
      let sentenceBuffer = "";
      for (const sentence of flat.split(/(?<=[.!?])\s+/)) {
        sentenceBuffer = sentenceBuffer ? `${sentenceBuffer} ${sentence}` : sentence;
        if (sentenceBuffer.length >= options.minChars) {
          pushCandidate(candidates, sentenceBuffer, source, sourceFile, sourceIndex++, options, rejectedCounts);
          sentenceBuffer = "";
        }
      }
    }
  }
  return { candidates, rejectedCounts };
}

function selectByBytes(candidates, targetBytes) {
  const selected = [];
  const seen = new Set();
  const seenOpeners = new Set();
  let bytes = 0;
  const sorted = [...candidates].sort((left, right) => {
    if (right.score !== left.score) return right.score - left.score;
    return left.source_index - right.source_index;
  });
  for (const candidate of sorted) {
    const key = candidate.text;
    const opener = wordsOf(candidate.text).slice(0, 5).join(" ");
    if (seen.has(key) || seenOpeners.has(opener)) continue;
    selected.push(candidate);
    seen.add(key);
    seenOpeners.add(opener);
    bytes += Buffer.byteLength(candidate.text) + 2;
    if (bytes >= targetBytes) break;
  }
  return selected.sort((left, right) => left.source_index - right.source_index);
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), "utf8");
}

function interleave(selectedBySource) {
  const indexes = Object.fromEntries(Object.keys(selectedBySource).map((source) => [source, 0]));
  const rows = [];
  let moved = true;
  while (moved) {
    moved = false;
    for (const source of sourceOrder) {
      const index = indexes[source] || 0;
      const items = selectedBySource[source] || [];
      if (index < items.length) {
        rows.push(items[index]);
        indexes[source] = index + 1;
        moved = true;
      }
    }
  }
  return rows;
}

function sourceStats(rows) {
  const bytes = rows.reduce((total, row) => total + Buffer.byteLength(row.text) + 2, 0);
  return { count: rows.length, bytes };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir = repoPath(options.outDir);
  fs.mkdirSync(outDir, { recursive: true });

  const shakespeareFiles = existing([options.shakespeareSource, "data/raw/shakespeare-gutenberg-100.txt"]).slice(0, 1);
  const blakeFiles = [
    ...existing(["data/processed/blake-poems.clean.txt", "data/processed/blake-marriage-heaven-hell.clean.txt"]),
    ...listCleanFiles("data/processed/crowley-bard-sources", "blake-"),
  ];
  const crowleyFiles = [
    ...existing(["data/processed/crowley-household-gods.clean.txt", "data/processed/crowley-tannhauser.clean.txt"]),
    ...listCleanFiles("data/processed/crowley-bard-sources", "crowley-"),
  ];

  const sourceFiles = {
    shakespeare: [...new Set(shakespeareFiles)],
    blake: [...new Set(blakeFiles)],
    crowley: [...new Set(crowleyFiles)],
  };
  for (const [source, files] of Object.entries(sourceFiles)) {
    if (files.length === 0) throw new Error(`no ${source} source files found`);
  }

  const extracted = Object.fromEntries(
    Object.entries(sourceFiles).map(([source, files]) => [source, extractCandidates(source, files, options)]),
  );
  const selectedBySource = {
    crowley: selectByBytes(extracted.crowley.candidates, options.crowleyBytes),
    blake: selectByBytes(extracted.blake.candidates, options.blakeBytes),
    shakespeare: selectByBytes(extracted.shakespeare.candidates, options.shakespeareBytes),
  };
  const corpusRows = interleave(selectedBySource);
  const corpusText = `${corpusRows.map((row) => row.text).join("\n\n")}\n`;

  const paths = {
    corpus: path.join(outDir, "corpus.txt"),
    shakespeare: path.join(outDir, "shakespeare.aphorisms.txt"),
    blake: path.join(outDir, "blake.aphorisms.txt"),
    crowley: path.join(outDir, "crowley.aphorisms.txt"),
    selected: path.join(outDir, "selected-aphorisms.jsonl"),
    manifest: path.join(outDir, "aphorism-manifest.json"),
    metrics: path.join(outDir, "aphorism-metrics.tsv"),
  };

  fs.writeFileSync(paths.corpus, corpusText, "utf8");
  for (const [source, rows] of Object.entries(selectedBySource)) {
    fs.writeFileSync(paths[source], `${rows.map((row) => row.text).join("\n\n")}\n`, "utf8");
  }
  writeJsonl(paths.selected, corpusRows);

  const stats = Object.fromEntries(Object.entries(selectedBySource).map(([source, rows]) => [source, sourceStats(rows)]));
  fs.writeFileSync(
    paths.metrics,
    [
      "source\tcandidate_count\tselected_count\tselected_bytes\trejected",
      ...Object.keys(sourceFiles).map((source) => {
        const rejected = Object.fromEntries(extracted[source].rejectedCounts);
        return [
          source,
          extracted[source].candidates.length,
          stats[source].count,
          stats[source].bytes,
          JSON.stringify(rejected),
        ].join("\t");
      }),
    ].join("\n") + "\n",
    "utf8",
  );
  fs.writeFileSync(
    paths.manifest,
    JSON.stringify(
      {
        schema: "nsrl.crowley_bard_aphorism_corpus.v1",
        out_dir: path.relative(repoRoot, outDir),
        source_layout: "weighted_unit_interleave",
        source_order: sourceOrder,
        min_chars: options.minChars,
        max_chars: options.maxChars,
        source_targets: {
          crowley: options.crowleyBytes,
          blake: options.blakeBytes,
          shakespeare: options.shakespeareBytes,
        },
        source_stats: stats,
        source_files: Object.fromEntries(
          Object.entries(sourceFiles).map(([source, files]) => [
            source,
            files.map((filePath) => path.relative(repoRoot, filePath)),
          ]),
        ),
        corpus: path.relative(repoRoot, paths.corpus),
        selected_aphorisms: path.relative(repoRoot, paths.selected),
        metrics: path.relative(repoRoot, paths.metrics),
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );

  console.log(`corpus=${path.relative(repoRoot, paths.corpus)}`);
  console.log(`selected=${path.relative(repoRoot, paths.selected)}`);
  console.log(`metrics=${path.relative(repoRoot, paths.metrics)}`);
  console.log(`manifest=${path.relative(repoRoot, paths.manifest)}`);
  console.log(`corpus_bytes=${Buffer.byteLength(corpusText)}`);
}

try {
  main();
} catch (error) {
  console.error(`build-crowley-bard-aphorism-corpus: ${error.message}`);
  process.exit(1);
}
