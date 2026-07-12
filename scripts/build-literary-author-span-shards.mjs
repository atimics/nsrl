#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const AUTHORS = ["crowley", "shakespeare", "blake"];
const sourceDir = process.argv[2] ?? "data/local-runs/literary-scale-24k-seq32";
const outDir = process.argv[3] ?? "data/experiments/literary-h8-author-block-swarm-v1/shards";
const leafChunks = Number.parseInt(process.argv[4] ?? "24", 10);
const routerTrainChunks = Number.parseInt(process.argv[5] ?? "4", 10);
const routerCalibrationChunks = Number.parseInt(process.argv[6] ?? "4", 10);

for (const value of [leafChunks, routerTrainChunks, routerCalibrationChunks]) {
  if (!Number.isInteger(value) || value < 1) throw new Error("chunk budgets must be positive integers");
}

const corpusPath = path.join(sourceDir, "corpus.txt");
const holdoutPath = path.join(sourceDir, "holdout.txt");
const sourceManifestPath = path.join(sourceDir, "corpus.manifest.json");
const sourceTokensPath = path.join(sourceDir, "tokens.u8");
const sourceHoldoutTokensPath = path.join(sourceDir, "holdout.tokens.u8");
const sourceManifest = JSON.parse(fs.readFileSync(sourceManifestPath, "utf8"));
if (sourceManifest.schema !== "nsrl.literary_corpus.v1"
  || sourceManifest.layout !== "round_robin_author_chunks") {
  throw new Error("source manifest is not a round-robin literary corpus");
}

const corpusBytes = fs.readFileSync(corpusPath);
const holdoutBytes = fs.readFileSync(holdoutPath);
const sourceTokens = fs.readFileSync(sourceTokensPath);
const sourceHoldoutTokens = fs.readFileSync(sourceHoldoutTokensPath);
assertHash(corpusBytes, sourceManifest.corpus.sha256, "corpus");
assertHash(holdoutBytes, sourceManifest.holdout.sha256, "holdout");
if (!asciiLowerTokens(corpusBytes).equals(sourceTokens)) {
  throw new Error("ASCII-lower tokenizer does not reproduce source corpus tokens");
}
if (!asciiLowerTokens(holdoutBytes).equals(sourceHoldoutTokens)) {
  throw new Error("ASCII-lower tokenizer does not reproduce source holdout tokens");
}
const corpusChunks = parseSourceChunks(corpusBytes.toString("utf8"), "corpus");
const holdoutChunks = parseSourceChunks(holdoutBytes.toString("utf8"), "holdout");

for (const author of AUTHORS) {
  if (corpusChunks[author].length !== sourceManifest.chunk_counts[author]) {
    throw new Error(`${author} corpus chunk count mismatch`);
  }
  if (holdoutChunks[author].length !== sourceManifest.holdout_chunk_counts[author]) {
    throw new Error(`${author} holdout chunk count mismatch`);
  }
  const required = leafChunks + routerTrainChunks + routerCalibrationChunks;
  if (corpusChunks[author].length < required) {
    throw new Error(`${author} has ${corpusChunks[author].length} chunks; ${required} required`);
  }
}

fs.mkdirSync(outDir, { recursive: true });
for (const split of ["leaf-train", "router-train", "router-calibration", "final-test"]) {
  fs.mkdirSync(path.join(outDir, split), { recursive: true });
}

const splitChunks = Object.fromEntries(AUTHORS.map((author) => {
  const chunks = corpusChunks[author];
  const leafEnd = leafChunks;
  const routerEnd = leafEnd + routerTrainChunks;
  const calibrationEnd = routerEnd + routerCalibrationChunks;
  return [author, {
    leaf_train: chunks.slice(0, leafEnd),
    router_train: chunks.slice(leafEnd, routerEnd),
    router_calibration: chunks.slice(routerEnd, calibrationEnd),
    final_test: holdoutChunks[author],
    unused_corpus: chunks.slice(calibrationEnd),
  }];
}));

const manifest = {
  schema: "nsrl.literary_author_span_shards.v1",
  authors: AUTHORS,
  source: {
    manifest: path.resolve(sourceManifestPath),
    manifest_sha256: sha256(fs.readFileSync(sourceManifestPath)),
    corpus: fileBinding(corpusPath, corpusBytes),
    holdout: fileBinding(holdoutPath, holdoutBytes),
    corpus_tokens: fileBinding(sourceTokensPath, sourceTokens),
    holdout_tokens: fileBinding(sourceHoldoutTokensPath, sourceHoldoutTokens),
    chunk_bytes: sourceManifest.chunk_bytes,
  },
  policy: {
    author_labels_used_for_training_provenance_only: true,
    inference_router_must_be_target_blind: true,
    leaf_chunks_per_author: leafChunks,
    router_train_chunks_per_author: routerTrainChunks,
    router_calibration_chunks_per_author: routerCalibrationChunks,
    final_test_source: "original untouched holdout chunks",
    overlap_between_splits: false,
    tokenizer: "byte_ascii_lower_text_u8_v1",
  },
  splits: {},
  unused_corpus_chunks: {},
};

for (const split of ["leaf_train", "router_train", "router_calibration", "final_test"]) {
  const directory = split.replaceAll("_", "-");
  manifest.splits[split] = {};
  for (const author of AUTHORS) {
    const chunks = splitChunks[author][split];
    const text = `${chunks.join("\n\n")}\n`;
    const tokens = asciiLowerTokens(Buffer.from(text));
    const textPath = path.join(outDir, directory, `${author}.txt`);
    const tokenPath = path.join(outDir, directory, `${author}.tokens.u8`);
    fs.writeFileSync(textPath, text);
    fs.writeFileSync(tokenPath, tokens);
    manifest.splits[split][author] = {
      chunks: chunks.length,
      text: fileBinding(textPath, Buffer.from(text)),
      tokens: fileBinding(tokenPath, tokens),
    };
  }
  if (split !== "leaf_train") {
    const scoreLines = ["sample_id\tprompt_hex"];
    for (const author of AUTHORS) {
      splitChunks[author][split].forEach((chunk, index) => {
        const tokens = asciiLowerTokens(Buffer.from(chunk));
        if (tokens.length <= 64) throw new Error(`${split}/${author}/${index} is too short`);
        scoreLines.push(`${split}-${author}-${index}\t${tokens.toString("hex")}`);
      });
    }
    const scorePath = path.join(outDir, `${directory}.score-input.tsv`);
    const scoreBytes = Buffer.from(`${scoreLines.join("\n")}\n`);
    fs.writeFileSync(scorePath, scoreBytes);
    manifest.splits[split].score_input = fileBinding(scorePath, scoreBytes);
  }
}

for (const author of AUTHORS) {
  manifest.unused_corpus_chunks[author] = splitChunks[author].unused_corpus.map((chunk) => ({
    bytes: Buffer.byteLength(chunk),
    sha256: sha256(Buffer.from(chunk)),
  }));
}

const manifestPath = path.join(outDir, "manifest.json");
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(manifestPath);

function parseSourceChunks(text, label) {
  const header = `<|literary-${label}:v1|>`;
  if (!text.startsWith(header)) throw new Error(`missing ${label} header`);
  const marker = /<\|source:(crowley|shakespeare|blake)\|>\n/g;
  const found = [...text.matchAll(marker)];
  const chunks = Object.fromEntries(AUTHORS.map((author) => [author, []]));
  if (found.length === 0) throw new Error(`no ${label} source markers`);
  for (let index = 0; index < found.length; index += 1) {
    const author = found[index][1];
    const start = found[index].index + found[index][0].length;
    const end = index + 1 < found.length ? found[index + 1].index : text.length;
    const chunk = text.slice(start, end).trim();
    if (!chunk) throw new Error(`empty ${label} ${author} chunk`);
    chunks[author].push(chunk);
  }
  return chunks;
}

function asciiLowerTokens(input) {
  const out = [];
  let pendingSpace = false;
  for (const byte of input) {
    let token = null;
    if (byte >= 65 && byte <= 90) token = byte + 32;
    else if ((byte >= 97 && byte <= 122) || (byte >= 48 && byte <= 57)
      || [46, 44, 59, 58, 63, 33, 39, 45].includes(byte)) token = byte;
    if (token === null) {
      pendingSpace = true;
    } else {
      if (pendingSpace && out.length > 0) out.push(32);
      pendingSpace = false;
      out.push(token);
    }
  }
  return Buffer.from(out);
}

function fileBinding(file, bytes) {
  return { path: path.resolve(file), bytes: bytes.length, sha256: sha256(bytes) };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function assertHash(bytes, expected, label) {
  if (sha256(bytes) !== expected) throw new Error(`${label} hash mismatch`);
}
