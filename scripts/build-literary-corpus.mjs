#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const AUTHORS = ["crowley", "shakespeare", "blake"];

function usage() {
  console.log(`Usage:
  node scripts/build-literary-corpus.mjs \\
    --crowley PATH [--crowley PATH ...] \\
    --shakespeare PATH [--shakespeare PATH ...] \\
    --blake PATH [--blake PATH ...] \\
    --out-dir PATH [--bytes-per-author N] [--holdout-bytes-per-author N]
    [--chunk-bytes N]

Builds a deterministic, author-balanced UTF-8 corpus. Inputs may be raw
Project Gutenberg text or already-cleaned text. No author is repeated to fill
the requested byte budget; the shortest available author sets the balance.`);
}

function parseArgs(argv) {
  const options = {
    sources: Object.fromEntries(AUTHORS.map((author) => [author, []])),
    outDir: null,
    bytesPerAuthor: null,
    holdoutBytesPerAuthor: 0,
    chunkBytes: 4096,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (AUTHORS.some((author) => arg === `--${author}`)) {
      const author = arg.slice(2);
      options.sources[author].push(argv[++index]);
    } else if (arg === "--out-dir") {
      options.outDir = argv[++index];
    } else if (arg === "--bytes-per-author") {
      options.bytesPerAuthor = Number.parseInt(argv[++index], 10);
    } else if (arg === "--holdout-bytes-per-author") {
      options.holdoutBytesPerAuthor = Number.parseInt(argv[++index], 10);
    } else if (arg === "--chunk-bytes") {
      options.chunkBytes = Number.parseInt(argv[++index], 10);
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }

  if (!options.outDir) throw new Error("--out-dir is required");
  for (const author of AUTHORS) {
    if (options.sources[author].length === 0) {
      throw new Error(`at least one --${author} input is required`);
    }
  }
  if (options.bytesPerAuthor !== null && options.bytesPerAuthor < 1) {
    throw new Error("--bytes-per-author must be a positive integer");
  }
  if (!Number.isInteger(options.chunkBytes) || options.chunkBytes < 1) {
    throw new Error("--chunk-bytes must be a positive integer");
  }
  if (!Number.isInteger(options.holdoutBytesPerAuthor) || options.holdoutBytesPerAuthor < 0) {
    throw new Error("--holdout-bytes-per-author must be a non-negative integer");
  }
  return options;
}

function cleanText(input) {
  let text = input.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const start = text.search(/^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (start >= 0) text = text.slice(text.indexOf("\n", start) + 1);
  const end = text.search(/^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (end >= 0) text = text.slice(0, end);
  return text
    .normalize("NFC")
    .replace(/[\t ]+$/gm, "")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function truncateUtf8(text, maxBytes) {
  const bytes = Buffer.from(text, "utf8");
  if (bytes.length <= maxBytes) return text;
  let end = maxBytes;
  while (end > 0 && (bytes[end] & 0xc0) === 0x80) end -= 1;
  return bytes.subarray(0, end).toString("utf8").trimEnd();
}

function splitUtf8(text, chunkBytes) {
  const chunks = [];
  let remainder = text;
  while (Buffer.byteLength(remainder, "utf8") > chunkBytes) {
    let chunk = truncateUtf8(remainder, chunkBytes);
    const paragraphBreak = chunk.lastIndexOf("\n\n");
    const lineBreak = chunk.lastIndexOf("\n");
    const wordBreak = chunk.lastIndexOf(" ");
    const splitAt = paragraphBreak > chunk.length / 2
      ? paragraphBreak + 2
      : lineBreak > chunk.length / 2
        ? lineBreak + 1
        : wordBreak > chunk.length / 2
          ? wordBreak + 1
          : chunk.length;
    chunks.push(chunk.slice(0, splitAt).trim());
    remainder = remainder.slice(splitAt).trimStart();
  }
  if (remainder.trim()) chunks.push(remainder.trim());
  return chunks;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function interleaveAuthorChunks(texts, chunkBytes, label) {
  const chunks = Object.fromEntries(
    AUTHORS.map((author) => [author, splitUtf8(texts[author], chunkBytes)]),
  );
  const outputParts = [`<|literary-${label}:v1|>`];
  const chunkCounts = Object.fromEntries(AUTHORS.map((author) => [author, 0]));
  for (let index = 0; ; index += 1) {
    let wrote = false;
    for (const author of AUTHORS) {
      if (index >= chunks[author].length) continue;
      outputParts.push(`<|source:${author}|>\n${chunks[author][index]}`);
      chunkCounts[author] += 1;
      wrote = true;
    }
    if (!wrote) break;
  }
  return { text: `${outputParts.join("\n\n")}\n`, chunkCounts };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const texts = {};
  const sourceManifest = {};

  for (const author of AUTHORS) {
    const parts = [];
    sourceManifest[author] = [];
    for (const sourcePath of options.sources[author]) {
      const raw = await readFile(sourcePath, "utf8");
      const cleaned = cleanText(raw);
      parts.push(cleaned);
      sourceManifest[author].push({
        path: sourcePath,
        input_bytes: Buffer.byteLength(raw),
        cleaned_bytes: Buffer.byteLength(cleaned),
        input_sha256: sha256(raw),
      });
    }
    texts[author] = parts.join("\n\n");
  }

  const availableBytes = Object.fromEntries(
    AUTHORS.map((author) => [author, Buffer.byteLength(texts[author])]),
  );
  const shortest = Math.min(...Object.values(availableBytes));
  const bytesPerAuthor = Math.min(options.bytesPerAuthor ?? shortest, shortest);
  if (options.holdoutBytesPerAuthor >= bytesPerAuthor) {
    throw new Error("holdout bytes must be smaller than the balanced bytes per author");
  }
  const trainBytesPerAuthor = bytesPerAuthor - options.holdoutBytesPerAuthor;
  const trainTexts = {};
  const holdoutTexts = {};
  for (const author of AUTHORS) {
    const balanced = truncateUtf8(texts[author], bytesPerAuthor);
    trainTexts[author] = truncateUtf8(balanced, trainBytesPerAuthor);
    holdoutTexts[author] = balanced.slice(trainTexts[author].length).trimStart();
  }
  const corpus = interleaveAuthorChunks(trainTexts, options.chunkBytes, "corpus");
  const holdout = options.holdoutBytesPerAuthor > 0
    ? interleaveAuthorChunks(holdoutTexts, options.chunkBytes, "holdout")
    : null;

  const outDir = path.resolve(options.outDir);
  await mkdir(outDir, { recursive: true });
  const corpusPath = path.join(outDir, "corpus.txt");
  const holdoutPath = path.join(outDir, "holdout.txt");
  const manifestPath = path.join(outDir, "corpus.manifest.json");
  await writeFile(corpusPath, corpus.text);
  if (holdout) await writeFile(holdoutPath, holdout.text);
  await writeFile(manifestPath, `${JSON.stringify({
    schema: "nsrl.literary_corpus.v1",
    authors: AUTHORS,
    layout: "round_robin_author_chunks",
    bytes_per_author_limit: bytesPerAuthor,
    train_bytes_per_author_limit: trainBytesPerAuthor,
    holdout_bytes_per_author_limit: options.holdoutBytesPerAuthor,
    chunk_bytes: options.chunkBytes,
    available_bytes: availableBytes,
    chunk_counts: corpus.chunkCounts,
    holdout_chunk_counts: holdout?.chunkCounts ?? null,
    sources: sourceManifest,
    corpus: {
      path: corpusPath,
      bytes: Buffer.byteLength(corpus.text),
      sha256: sha256(corpus.text),
    },
    holdout: holdout ? {
      path: holdoutPath,
      bytes: Buffer.byteLength(holdout.text),
      sha256: sha256(holdout.text),
    } : null,
  }, null, 2)}\n`);

  console.log(manifestPath);
}

main().catch((error) => {
  console.error(`build-literary-corpus: ${error.message}`);
  process.exit(1);
});
