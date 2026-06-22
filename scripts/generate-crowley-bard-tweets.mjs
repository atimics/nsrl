#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const prompts = [
  "my soul is",
  "the world is",
  "the angel says",
  "love is",
  "night is",
  "i have seen",
  "the child",
  "o soul",
  "tannhauser",
  "the little god",
  "bright form",
  "silent delight",
];

const topKCycle = [6, 8, 10, 12];
const flavorTerms = [
  "soul",
  "angel",
  "child",
  "love",
  "night",
  "flame",
  "heaven",
  "hell",
  "tannhauser",
  "god",
  "dream",
  "delight",
];
const bannedTerms = [
  "ai",
  "assistant",
  "chatbot",
  "model",
  "training",
  "prompt",
  "json",
  "ranked",
  "http",
  "www",
  "class",
  "align",
  "bgcolor",
  "nbsp",
  "project gutenberg",
  "scene ii",
  "scene iii",
  "scene iv",
  "act iii",
  "act iv",
  "act v",
  "enter",
  "exeunt",
  "dramatis",
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
  "alicia",
  "crassus",
  "act i",
  "act ii",
];
const leakageSubstrings = [
  "ai",
  "assistant",
  "chatbot",
  "model",
  "training",
  "prompt",
  "json",
  "ranked",
  "project gutenberg",
  "class",
  "align",
  "bgcolor",
  "http",
  "www",
  "parolles",
  "helena",
  "bertram",
  "lafeu",
  "alicia",
  "crassus",
  "enter",
  "exeunt",
];
const decodeBannedTokens = [
  "assistant",
  "chatbot",
  "model",
  "training",
  "prompt",
  "json",
  "http",
  "www",
  "class",
  "align",
  "bgcolor",
  "nbsp",
  "enter",
  "exeunt",
  "dramatis",
  "alicia",
  "crassus",
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
const glueWords = new Set([
  ...stopWords,
  "also",
  "am",
  "away",
  "can",
  "come",
  "could",
  "did",
  "do",
  "doth",
  "down",
  "even",
  "ever",
  "go",
  "give",
  "had",
  "hath",
  "have",
  "has",
  "hear",
  "here",
  "if",
  "into",
  "know",
  "let",
  "like",
  "made",
  "make",
  "may",
  "might",
  "more",
  "most",
  "much",
  "must",
  "never",
  "no",
  "not",
  "now",
  "one",
  "out",
  "own",
  "say",
  "see",
  "shall",
  "should",
  "still",
  "take",
  "than",
  "then",
  "there",
  "these",
  "thing",
  "things",
  "those",
  "this",
  "unto",
  "up",
  "upon",
  "was",
  "were",
  "when",
  "will",
  "would",
  "yet",
]);

const defaults = {
  runDir: "data/processed/crowley-bard-focused-v1",
  outDir: "",
  model: "",
  vocab: "",
  tokens: "",
  rawCount: 96,
  keepCount: 16,
  minChars: 60,
  maxChars: 240,
  maxNewTokens: 36,
};

function usage() {
  console.log(`Usage: node scripts/generate-crowley-bard-tweets.mjs [options]

Options:
  --run-dir PATH       Balanced-prose run directory [${defaults.runDir}]
  --out-dir PATH       Output directory [RUN_DIR/tweets-strict]
  --model PATH         Override model path instead of manifest model
  --vocab PATH         Override vocab path instead of manifest vocab
  --tokens PATH        Override token path instead of manifest tokens
  --raw-count N        Number of raw candidates to generate [${defaults.rawCount}]
  --keep-count N       Number of accepted tweets to keep [${defaults.keepCount}]
  --min-chars N        Minimum accepted text length [${defaults.minChars}]
  --max-chars N        Maximum accepted text length [${defaults.maxChars}]
  --max-new-tokens N   Decode budget per raw candidate [${defaults.maxNewTokens}]
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
    if (["rawCount", "keepCount", "minChars", "maxChars", "maxNewTokens"].includes(key)) {
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
  if ((options.model || options.vocab || options.tokens) && !(options.model && options.vocab && options.tokens)) {
    throw new Error("--model, --vocab, and --tokens must be provided together");
  }
  if (!options.outDir) {
    options.outDir = path.join(options.runDir, "tweets-strict");
  }
  return options;
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.join(repoRoot, filePath);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function cleanAsciiLower(text) {
  return String(text ?? "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x20-\x7e]/g, " ")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}

function trimTweet(text, minChars, maxChars) {
  const clean = cleanAsciiLower(text);
  for (let index = minChars; index < clean.length; index += 1) {
    if (/[.!?]/.test(clean[index])) {
      return clean.slice(0, index + 1).trim();
    }
  }
  if (clean.length <= maxChars) return clean;
  const cut = clean.slice(0, maxChars + 1);
  const lastSpace = cut.lastIndexOf(" ");
  return cut.slice(0, lastSpace >= minChars ? lastSpace : maxChars).trim();
}

function wordsOf(text) {
  return text.match(/[a-z][a-z']*/g) || [];
}

function repeatedTrigramCount(words) {
  const seen = new Map();
  let repeats = 0;
  for (let index = 0; index + 2 < words.length; index += 1) {
    const trigram = `${words[index]} ${words[index + 1]} ${words[index + 2]}`;
    const count = seen.get(trigram) || 0;
    if (count === 1) repeats += 1;
    seen.set(trigram, count + 1);
  }
  return repeats;
}

function maxConsecutiveStopwords(words) {
  let current = 0;
  let max = 0;
  for (const word of words) {
    if (stopWords.has(word)) {
      current += 1;
      max = Math.max(max, current);
    } else {
      current = 0;
    }
  }
  return max;
}

function maxConsecutiveWordsInSet(words, wordSet) {
  let current = 0;
  let max = 0;
  for (const word of words) {
    if (wordSet.has(word)) {
      current += 1;
      max = Math.max(max, current);
    } else {
      current = 0;
    }
  }
  return max;
}

function countTerms(text, terms) {
  let count = 0;
  const hits = [];
  for (const term of terms) {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`\\b${escaped}\\b`, "g");
    const matches = text.match(regex);
    if (matches) {
      count += matches.length;
      hits.push(term);
    }
  }
  return { count, hits };
}

function leakageHitsFor(text) {
  const hits = [];
  for (const term of leakageSubstrings) {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const matchesWordLikeTerm = /^[a-z][a-z ]*[a-z]$/.test(term);
    const regex = matchesWordLikeTerm
      ? new RegExp(`\\b${escaped}\\b`)
      : null;
    if ((regex && regex.test(text)) || (!regex && text.includes(term))) {
      hits.push(term);
    }
  }
  return hits;
}

function analyze(rawText, options) {
  const text = trimTweet(rawText, options.minChars, options.maxChars);
  const words = wordsOf(text);
  const counts = new Map();
  for (const word of words) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  const maxWordCount = Math.max(0, ...counts.values());
  const distinctWords = counts.size;
  const distinctRatio = words.length ? distinctWords / words.length : 0;
  const repeatedTrigrams = repeatedTrigramCount(words);
  const cruft = countTerms(text, bannedTerms);
  const flavor = countTerms(text, flavorTerms);
  const stopwordRatio = words.length
    ? words.filter((word) => stopWords.has(word)).length / words.length
    : 0;
  const glueRatio = words.length
    ? words.filter((word) => glueWords.has(word)).length / words.length
    : 0;
  const stopwordRun = maxConsecutiveStopwords(words);
  const glueRun = maxConsecutiveWordsInSet(words, glueWords);
  const leakageHits = leakageHitsFor(text);
  const punctuationRuns = (text.match(/[!?.,;:]{2,}/g) || []).length;
  const sentenceTerminalCount = (text.match(/[.!?]/g) || []).length;
  const expressiveTerminalCount = (text.match(/[!?]/g) || []).length;
  const lastWord = words[words.length - 1] || "";
  const rejectionReasons = [];

  if (text.length < options.minChars) rejectionReasons.push("too_short");
  if (text.length > options.maxChars) rejectionReasons.push("too_long");
  if (maxWordCount > 4) rejectionReasons.push("word_repeat_gt4");
  if (repeatedTrigrams > 0) rejectionReasons.push("repeated_trigram");
  if (distinctRatio < 0.55) rejectionReasons.push("low_distinct_ratio");
  if (stopwordRatio > 0.55) rejectionReasons.push("stopword_heavy");
  if (stopwordRun > 4) rejectionReasons.push("function_word_run");
  if (stopwordRatio > 0.4 && stopwordRun >= 3) rejectionReasons.push("function_word_soup");
  if (glueRatio > 0.72) rejectionReasons.push("glue_word_heavy");
  if (glueRun > 8) rejectionReasons.push("glue_word_run");
  if (cruft.count > 0) rejectionReasons.push("cruft");
  if (leakageHits.length > 0) rejectionReasons.push("leakage_substring");
  if (punctuationRuns > 0) rejectionReasons.push("punctuation_run");
  if (sentenceTerminalCount > 4) rejectionReasons.push("terminal_heavy");
  if (expressiveTerminalCount > 3) rejectionReasons.push("expressive_punctuation_heavy");
  if (danglingEndWords.has(lastWord)) rejectionReasons.push("dangling_end_word");
  if (words.length >= 12 && !/[.!?]$/.test(text)) rejectionReasons.push("no_sentence_terminal");

  let score = 100;
  if (text.length < 90) score -= 90 - text.length;
  if (text.length > 180) score -= Math.ceil((text.length - 180) * 1.5);
  if (text.length >= 90 && text.length <= 180) score += 30;
  score += Math.min(40, flavor.count * 8);
  score -= Math.max(0, maxWordCount - 2) * 8;
  score -= repeatedTrigrams * 40;
  if (stopwordRatio > 0.58) score -= Math.ceil((stopwordRatio - 0.58) * 200);
  score -= Math.max(0, stopwordRun - 3) * 16;
  if (glueRatio > 0.42) score -= Math.ceil((glueRatio - 0.42) * 220);
  score -= Math.max(0, glueRun - 4) * 10;
  score -= punctuationRuns * 10;
  score -= Math.max(0, sentenceTerminalCount - 2) * 12;
  score -= Math.max(0, expressiveTerminalCount - 1) * 8;
  score -= cruft.count * 100;
  if (danglingEndWords.has(lastWord)) score -= 40;
  if (words.length >= 12 && !/[.!?]$/.test(text)) score -= 30;

  return {
    text,
    chars: text.length,
    words: words.length,
    distinct_words: distinctWords,
    distinct_ratio_q1000: Math.round(distinctRatio * 1000),
    stopword_ratio_q1000: Math.round(stopwordRatio * 1000),
    glue_word_ratio_q1000: Math.round(glueRatio * 1000),
    max_stopword_run: stopwordRun,
    max_glue_word_run: glueRun,
    sentence_terminal_count: sentenceTerminalCount,
    expressive_terminal_count: expressiveTerminalCount,
    max_word_count: maxWordCount,
    repeated_trigram_count: repeatedTrigrams,
    cruft_count: cruft.count,
    cruft_hits: cruft.hits,
    leakage_hits: leakageHits,
    flavor_hits: flavor.hits,
    score,
    accepted: rejectionReasons.length === 0,
    rejection_reasons: rejectionReasons,
  };
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (result.status !== 0) {
    const stderr = result.stderr ? `\n${result.stderr.trim()}` : "";
    throw new Error(`${command} failed with status ${result.status}${stderr}`);
  }
  return result;
}

function generateCandidate(paths, candidate) {
  const decodeBanArgs = decodeBannedTokens.flatMap((token) => ["--decode-ban-token", token]);
  run(paths.trainBin, [
    "--mode", "lexeme-generate",
    "--model", paths.model,
    "--vocab", paths.vocab,
    "--tokens", paths.tokens,
    "--prompt", candidate.prompt,
    "--max-new-tokens", String(candidate.max_new_tokens),
    "--decode", "sample",
    "--sample-seed", String(candidate.sample_seed),
    "--top-k", String(candidate.top_k),
    "--decode-function-word-run-cap", "4",
    ...decodeBanArgs,
    "--corpus-prior",
    "--corpus-prior-order", "3",
    "--corpus-prior-logit-shift", "9",
    "--decode-frequency-cap", "600",
    "--decode-frequency-min-q15", "6144",
    "--decode-frequency-logit-shift", "5",
    "--decode-local-frequency-cap", "2",
    "--decode-local-frequency-min-q15", "8192",
    "--decode-local-frequency-logit-shift", "4",
    "--decode-local-frequency-hard-cap", "2",
    "--repeat-window", "64",
    "--repeat-penalty-shift", "4",
    "--max-repeat-run", "2",
    "--no-repeat-ngram", "3",
    "--strict-adjacency",
    "--quality-weight-profile", "prose-aware",
    "--text-out", candidate.raw_path,
    "--trace", candidate.trace_path,
  ]);
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), "utf8");
}

function cleanOutputDirectory(outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const generatedFilePattern =
    /^(?:tweet-\d+\.(?:raw|trace)(?: \d+)?\.(?:txt|jsonl)|tweets(?: \d+)?\.(?:jsonl|md)|metrics(?: \d+)?\.tsv|candidates(?: \d+)?\.jsonl)$/;
  for (const entry of fs.readdirSync(outDir)) {
    if (generatedFilePattern.test(entry)) {
      fs.rmSync(path.join(outDir, entry), { force: true });
    }
  }
  fs.rmSync(path.join(outDir, "candidates"), { recursive: true, force: true });
}

function cleanCandidateDirectory(candidateDir) {
  fs.rmSync(candidateDir, { recursive: true, force: true });
  fs.mkdirSync(candidateDir, { recursive: true });
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const runDir = resolveRepoPath(options.runDir);
  const outDir = resolveRepoPath(options.outDir);
  const candidateDir = path.join(path.dirname(outDir), "tweets-candidates");
  const manifestPath = path.join(runDir, "manifest.json");
  if (!fs.existsSync(manifestPath) && !(options.model && options.vocab && options.tokens)) {
    throw new Error(`missing run manifest: ${manifestPath}`);
  }
  const manifest = fs.existsSync(manifestPath) ? readJson(manifestPath) : {};
  const trainBin = resolveRepoPath("target/release/nsrl-train");
  if (!fs.existsSync(trainBin)) {
    throw new Error(`missing release binary: ${trainBin}`);
  }
  const paths = {
    trainBin,
    model: resolveRepoPath(options.model || manifest.model),
    vocab: resolveRepoPath(options.vocab || manifest.vocab),
    tokens: resolveRepoPath(options.tokens || manifest.tokens),
  };
  for (const [name, filePath] of Object.entries(paths)) {
    if (name !== "trainBin" && !fs.existsSync(filePath)) {
      throw new Error(`missing ${name}: ${filePath}`);
    }
  }

  cleanOutputDirectory(outDir);
  cleanCandidateDirectory(candidateDir);
  const candidates = [];
  for (let index = 0; index < options.rawCount; index += 1) {
    const number = String(index + 1).padStart(3, "0");
    const candidate = {
      id: `candidate-${number}`,
      prompt: prompts[index % prompts.length],
      sample_seed: 29 + index * 17,
      top_k: topKCycle[index % topKCycle.length],
      max_new_tokens: options.maxNewTokens,
      raw_path: path.join(candidateDir, `candidate-${number}.raw.txt`),
      trace_path: path.join(candidateDir, `candidate-${number}.trace.jsonl`),
    };
    generateCandidate(paths, candidate);
    const raw = fs.readFileSync(candidate.raw_path, "utf8");
    const analysis = analyze(raw, options);
    const row = {
      ...candidate,
      raw_path: path.relative(repoRoot, candidate.raw_path),
      trace_path: path.relative(repoRoot, candidate.trace_path),
      raw: cleanAsciiLower(raw),
      ...analysis,
    };
    candidates.push(row);
    console.error(`${candidate.id} accepted=${row.accepted ? "yes" : "no"} score=${row.score}`);
  }

  const accepted = candidates
    .filter((candidate) => candidate.accepted)
    .sort((left, right) => right.score - left.score || left.chars - right.chars)
    .slice(0, options.keepCount);

  const tweetRows = accepted.map((candidate, index) => {
    const tweetNumber = String(index + 1).padStart(2, "0");
    const rawPath = path.join(outDir, `tweet-${tweetNumber}.raw.txt`);
    const tracePath = path.join(outDir, `tweet-${tweetNumber}.trace.jsonl`);
    fs.writeFileSync(rawPath, `${candidate.text}\n`, "utf8");
    fs.writeFileSync(tracePath, `${JSON.stringify({
      schema: "nsrl.crowley_bard_tweet_trace_pointer.v1",
      source_candidate_id: candidate.id,
      seed_text: candidate.prompt,
      sample_seed: candidate.sample_seed,
      top_k: candidate.top_k,
      full_trace_ref: candidate.id,
      full_trace_dir: "tweets-candidates",
    })}\n`, "utf8");
    return {
      id: `tweet-${tweetNumber}`,
      source_candidate_id: candidate.id,
      seed_text: candidate.prompt,
      sample_seed: candidate.sample_seed,
      top_k: candidate.top_k,
      chars: candidate.chars,
      score: candidate.score,
      text: candidate.text,
      trace_ref: `tweet-${tweetNumber}`,
    };
  });

  writeJsonl(path.join(candidateDir, "candidates.jsonl"), candidates);
  writeJsonl(path.join(outDir, "tweets.jsonl"), tweetRows);
  fs.writeFileSync(
    path.join(outDir, "tweets.md"),
    `${tweetRows.map((tweet) => `### ${tweet.id} (${tweet.chars})\n${tweet.text}`).join("\n\n")}\n`,
    "utf8",
  );
  fs.writeFileSync(
    path.join(outDir, "metrics.tsv"),
    [
      "id\tsource_candidate_id\tscore\tchars\twords\tdistinct_words\tdistinct_ratio_q1000\tstopword_ratio_q1000\tglue_word_ratio_q1000\tmax_stopword_run\tmax_glue_word_run\tsentence_terminal_count\texpressive_terminal_count\tmax_word_count\trepeated_trigram_count\tcruft_count\tseed_text\ttop_k\tsample_seed\ttext",
      ...tweetRows.map((tweet) => [
        tweet.id,
        tweet.source_candidate_id,
        tweet.score,
        tweet.chars,
        wordsOf(tweet.text).length,
        new Set(wordsOf(tweet.text)).size,
        Math.round((new Set(wordsOf(tweet.text)).size / Math.max(1, wordsOf(tweet.text).length)) * 1000),
        Math.round((wordsOf(tweet.text).filter((word) => stopWords.has(word)).length / Math.max(1, wordsOf(tweet.text).length)) * 1000),
        Math.round((wordsOf(tweet.text).filter((word) => glueWords.has(word)).length / Math.max(1, wordsOf(tweet.text).length)) * 1000),
        maxConsecutiveStopwords(wordsOf(tweet.text)),
        maxConsecutiveWordsInSet(wordsOf(tweet.text), glueWords),
        (tweet.text.match(/[.!?]/g) || []).length,
        (tweet.text.match(/[!?]/g) || []).length,
        Math.max(0, ...Array.from(wordsOf(tweet.text).reduce((counts, word) => counts.set(word, (counts.get(word) || 0) + 1), new Map()).values())),
        repeatedTrigramCount(wordsOf(tweet.text)),
        countTerms(tweet.text, bannedTerms).count,
        tweet.seed_text,
        tweet.top_k,
        tweet.sample_seed,
        tweet.text,
      ].join("\t")),
    ].join("\n") + "\n",
    "utf8",
  );

  console.log(`raw_candidates=${candidates.length}`);
  console.log(`accepted_candidates=${candidates.filter((candidate) => candidate.accepted).length}`);
  console.log(`kept_tweets=${tweetRows.length}`);
  console.log(`tweets=${path.join(outDir, "tweets.md")}`);
  console.log(`candidate_audit=${candidateDir}`);
}

try {
  main();
} catch (error) {
  console.error(`generate-crowley-bard-tweets: ${error.message}`);
  process.exit(1);
}
