#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outRoot: "data/processed/cheap-trained/focused-pair-v1024-d16-seq16",
  lanes: "signal-radio,signal-replay,cosyworld,cosyworld-kernel,contextual-reading",
  vocabSize: 1024,
  embeddingDim: 16,
  seqLen: 16,
  cosyworldRepeat: 16,
  epochs: 1,
};

const laneDefs = {
  "signal-radio": {
    label: "Signal radio",
    pairsPath: "data/processed/cheap-trained/signal-radio/training-pairs.jsonl",
    repeat: 1,
    sampleCount: 3,
  },
  "signal-replay": {
    label: "Signal replay",
    pairsPath: "data/processed/signal-replay-corpus/training-pairs.jsonl",
    repeat: 1,
    sampleCount: 4,
  },
  "signal-replay-gemma4": {
    label: "Signal replay Gemma4",
    pairsPath: "data/processed/ollama-state-outputs/signal-replay-gemma4/training-pairs.jsonl",
    repeat: 1,
    sampleCount: 4,
  },
  cosyworld: {
    label: "CosyWorld",
    pairsPath: "data/processed/cheap-trained/cosyworld/training-pairs.jsonl",
    repeat: "cosyworldRepeat",
    sampleCount: 3,
  },
  "cosyworld-kernel": {
    label: "CosyWorld kernel",
    pairsPath: "data/processed/cosyworld-kernel-corpus/training-pairs.jsonl",
    repeat: "cosyworldRepeat",
    sampleCount: 4,
  },
  "cosyworld-kernel-gemma4": {
    label: "CosyWorld kernel Gemma4",
    pairsPath: "data/processed/ollama-state-outputs/cosyworld-kernel-gemma4/training-pairs.jsonl",
    repeat: "cosyworldRepeat",
    sampleCount: 4,
  },
  "contextual-reading": {
    label: "Contextual reading",
    pairsPath: "data/processed/contextual-reading-pairs/training-pairs.jsonl",
    repeat: 1,
    sampleCount: 4,
  },
  "cosyworld-shared-literary": {
    label: "CosyWorld shared literary",
    pairsPath: "data/processed/cosyworld-shared-literary-corpus/training-pairs.jsonl",
    repeat: 1,
    sampleCount: 3,
  },
};

function usage() {
  console.log(`Usage: node scripts/train-focused-pair-models.mjs [options]

Options:
  --out-root PATH          Output root [${defaults.outRoot}]
  --lanes LIST             Comma-separated lanes [${defaults.lanes}]
  --vocab-size N           Lexeme vocab cap [${defaults.vocabSize}]
  --embedding-dim N        Embedding dimension [${defaults.embeddingDim}]
  --seq-len N              Softmax sequence length [${defaults.seqLen}]
  --cosyworld-repeat N     Repeat small CosyWorld pairs [${defaults.cosyworldRepeat}]
  --epochs N               Training epochs [${defaults.epochs}]
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
    if (["vocabSize", "embeddingDim", "seqLen", "cosyworldRepeat", "epochs"].includes(key)) {
      options[key] = Number.parseInt(value, 10);
      if (!Number.isFinite(options[key]) || options[key] < 1) {
        throw new Error(`${arg} requires a positive integer`);
      }
    } else {
      options[key] = value;
    }
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

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, rows.map((row) => `${JSON.stringify(row)}\n`).join(""), "utf8");
}

function cleanText(text) {
  return String(text ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, "--")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x09\x0a\x0d\x20-\x7e]/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function run(command, args) {
  console.log([path.relative(repoRoot, command), ...args].join(" "));
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: ["ignore", "inherit", "inherit"],
  });
  if (result.status !== 0) {
    throw new Error(`${command} failed with status ${result.status}`);
  }
}

function readFirstJson(filePath) {
  const line = fs.readFileSync(filePath, "utf8").split(/\r?\n/).find(Boolean);
  if (!line) {
    throw new Error(`empty trace: ${filePath}`);
  }
  return JSON.parse(line);
}

function buildCorpus(rows, repeat) {
  const pieces = [];
  const expected = [];
  for (let pass = 0; pass < repeat; pass += 1) {
    for (const row of rows) {
      const state = cleanText(row.private_state || row.simulated_private_state || "");
      const output = cleanText(row.expected_output || row.line || row.target || "");
      if (!state || !output) continue;
      pieces.push(`${state}\n${output}`);
      expected.push(output);
    }
  }
  if (pieces.length === 0) {
    throw new Error("no usable private_state/expected_output rows");
  }
  return {
    corpus: `${pieces.join("\n\n")}\n`,
    expected: `${expected.join("\n\n")}\n`,
    usableRows: pieces.length,
  };
}

function sampleRows(rows, count) {
  const byDomainStyle = [];
  const domainStyleSeen = new Set();
  for (const row of rows) {
    const key = [row.domain || "", row.style || ""].join("\0");
    if (!row.style) break;
    if (domainStyleSeen.has(key)) continue;
    domainStyleSeen.add(key);
    byDomainStyle.push(row);
    if (byDomainStyle.length >= count) return byDomainStyle;
  }

  const seen = new Set();
  const out = [];
  for (const row of rows) {
    const output = cleanText(row.expected_output || row.line || row.target || "").toLowerCase();
    const outputFamily = output
      .replace(/\d+/g, "0")
      .split(/[.;]/, 1)[0]
      .slice(0, 48);
    const key = [
      row.domain || "",
      row.style || "",
      row.speaker || "",
      row.kind || "",
      row.event_type || "",
      outputFamily,
    ].join("\0");
    if (seen.has(key) && out.length < count) continue;
    seen.add(key);
    out.push(row);
    if (out.length >= count) break;
  }
  return out.length ? out : rows.slice(0, count);
}

function laneRepeat(laneDef, options) {
  if (typeof laneDef.repeat === "string") return options[laneDef.repeat];
  return laneDef.repeat;
}

function isSignalLane(lane) {
  return lane.startsWith("signal-");
}

function sampleMaxNewTokens(lane) {
  if (lane === "contextual-reading") return 100;
  return 24;
}

function trainLane(lane, options, binaries) {
  const laneDef = laneDefs[lane];
  if (!laneDef) throw new Error(`unknown lane: ${lane}`);
  const outDir = resolveRepoPath(path.join(options.outRoot, lane));
  fs.mkdirSync(outDir, { recursive: true });

  const rows = readJsonl(laneDef.pairsPath);
  const repeat = laneRepeat(laneDef, options);
  const built = buildCorpus(rows, repeat);
  const corpusPath = path.join(outDir, "corpus.txt");
  const expectedPath = path.join(outDir, "expected-output.txt");
  const pairsPath = path.join(outDir, "training-pairs.jsonl");
  fs.writeFileSync(corpusPath, built.corpus, "utf8");
  fs.writeFileSync(expectedPath, built.expected, "utf8");
  writeJsonl(pairsPath, rows);

  const prefix = `v${options.vocabSize}`;
  const tokensPath = path.join(outDir, `${prefix}.tokens.u16`);
  const vocabPath = path.join(outDir, `${prefix}.vocab.tsv`);
  const tokenTracePath = path.join(outDir, `${prefix}.tokens.trace.jsonl`);
  const priorTokensPath = path.join(outDir, `${prefix}.decode-prior.tokens.u16`);
  const priorTracePath = path.join(outDir, `${prefix}.decode-prior.tokens.trace.jsonl`);

  run(binaries.corpus, [
    "lexeme-tokenize",
    "--corpus", corpusPath,
    "--tokens-out", tokensPath,
    "--vocab-out", vocabPath,
    "--trace", tokenTracePath,
    "--seq-len", "32",
    "--stride", "1",
    "--max-vocab", String(options.vocabSize),
    "--lexeme-vocab-profile", "balanced",
    "--lexeme-frequency-cap", String(options.vocabSize),
    "--preview-tokens", "32",
  ]);

  run(binaries.corpus, [
    "lexeme-tokenize-fixed-vocab",
    "--corpus", expectedPath,
    "--vocab", vocabPath,
    "--tokens-out", priorTokensPath,
    "--trace", priorTracePath,
    "--seq-len", "32",
    "--stride", "1",
    "--preview-tokens", "32",
  ]);

  const tokenTrace = readFirstJson(tokenTracePath);
  const actualVocabSize = tokenTrace.vocab.size;
  const maxWindows = tokenTrace.windows.count;
  const embeddingPath = path.join(outDir, `v${actualVocabSize}-d${options.embeddingDim}.nsrllex`);
  const embeddingTracePath = path.join(outDir, `v${actualVocabSize}-d${options.embeddingDim}.embedding.trace.jsonl`);
  const modelPath = path.join(outDir, `v${actualVocabSize}-d${options.embeddingDim}-seq${options.seqLen}.nsrllm`);
  const softmaxTracePath = path.join(outDir, `v${actualVocabSize}-d${options.embeddingDim}-seq${options.seqLen}.softmax.trace.jsonl`);

  run(binaries.train, [
    "--mode", "lexeme-embedding",
    "--tokens", tokensPath,
    "--vocab", vocabPath,
    "--model-out", embeddingPath,
    "--trace", embeddingTracePath,
    "--vocab-size", String(actualVocabSize),
    "--embedding-dim", String(options.embeddingDim),
    "--context-radius", "2",
    "--stride", "1",
    "--max-windows", String(maxWindows),
    "--epochs", String(options.epochs),
    "--lr-shift", "8",
    "--concept-frequency-cap", String(options.vocabSize),
    "--frequency-weight-min-q15", "4096",
    "--quality-weight-profile", "cruft-aware",
  ]);

  run(binaries.train, [
    "--mode", "lexeme-softmax",
    "--tokens", tokensPath,
    "--vocab", vocabPath,
    "--model", embeddingPath,
    "--model-out", modelPath,
    "--trace", softmaxTracePath,
    "--seq-len", String(options.seqLen),
    "--lexeme-context-features", "ordered",
    "--stride", "1",
    "--max-windows", String(maxWindows),
    "--epochs", String(options.epochs),
    "--lr-shift", "18",
    "--lr-shift-decay-windows", String(Math.max(1, Math.floor(maxWindows / 2))),
    "--lr-shift-decay-step", "1",
    "--max-lr-shift", "23",
    "--max-weight-delta", "1",
    "--target-frequency-cap", String(options.vocabSize),
    "--frequency-weight-min-q15", "4096",
    "--quality-weight-profile", "cruft-aware",
  ]);

  const samples = [];
  for (const [index, row] of sampleRows(rows, laneDef.sampleCount).entries()) {
    const prompt = `${cleanText(row.private_state || row.simulated_private_state || "")}\n`;
    const sampleLabel = `sample-${String(index + 1).padStart(2, "0")}`;
    const sampleTextPath = path.join(outDir, `${sampleLabel}.txt`);
    const sampleTracePath = path.join(outDir, `${sampleLabel}.trace.jsonl`);
    run(binaries.train, [
      "--mode", "lexeme-generate",
      "--model", modelPath,
      "--vocab", vocabPath,
      "--tokens", priorTokensPath,
      "--prompt", prompt,
      "--max-new-tokens", String(sampleMaxNewTokens(lane)),
      "--decode", "sample",
      "--sample-seed", String(7 + index * 11),
      "--top-k", isSignalLane(lane) ? "6" : "8",
      "--decode-profile", "coherent-prose",
      "--corpus-prior",
      "--corpus-prior-order", "3",
      "--corpus-prior-logit-shift", isSignalLane(lane) ? "8" : "9",
      "--repeat-window", "64",
      "--repeat-penalty-shift", "4",
      "--max-repeat-run", "2",
      "--no-repeat-ngram", "3",
      "--strict-adjacency",
      "--generated-only",
      "--text-out", sampleTextPath,
      "--trace", sampleTracePath,
    ]);
    samples.push({
      label: sampleLabel,
      prompt,
      expected_output: cleanText(row.expected_output || ""),
      text_path: sampleTextPath,
      trace_path: sampleTracePath,
      text: fs.readFileSync(sampleTextPath, "utf8").trim(),
    });
  }

  const softmaxTrace = readFirstJson(softmaxTracePath);
  const manifest = {
    schema: "nsrl.focused_pair_model.v1",
    created_at: new Date().toISOString(),
    lane,
    label: laneDef.label,
    source_pairs_path: resolveRepoPath(laneDef.pairsPath),
    out_dir: outDir,
    corpus_path: corpusPath,
    expected_output_path: expectedPath,
    training_pairs_path: pairsPath,
    tokens_path: tokensPath,
    decode_prior_tokens_path: priorTokensPath,
    vocab_path: vocabPath,
    embedding_path: embeddingPath,
    model_path: modelPath,
    token_trace_path: tokenTracePath,
    embedding_trace_path: embeddingTracePath,
    softmax_trace_path: softmaxTracePath,
    source_rows: rows.length,
    training_rows: built.usableRows,
    repeat,
    token_count: tokenTrace.tokens.count,
    windows: maxWindows,
    vocab_cap: options.vocabSize,
    vocab_size: actualVocabSize,
    embedding_dim: options.embeddingDim,
    seq_len: options.seqLen,
    epochs: options.epochs,
    final_accuracy_per_mille: softmaxTrace.metrics.final_accuracy_per_mille,
    final_mistakes: softmaxTrace.metrics.final_mistakes,
    flattening: "private_state newline expected_output, no control labels",
    samples,
  };
  fs.writeFileSync(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return manifest;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const binaries = {
    corpus: resolveRepoPath("target/release/nsrl-corpus"),
    train: resolveRepoPath("target/release/nsrl-train"),
  };
  for (const binary of Object.values(binaries)) {
    if (!fs.existsSync(binary)) {
      throw new Error(`missing release binary: ${binary}`);
    }
  }

  const lanes = splitList(options.lanes);
  const manifests = lanes.map((lane) => trainLane(lane, options, binaries));
  const indexPath = resolveRepoPath(path.join(options.outRoot, "manifest.json"));
  fs.mkdirSync(path.dirname(indexPath), { recursive: true });
  fs.writeFileSync(indexPath, `${JSON.stringify({
    schema: "nsrl.focused_pair_model_index.v1",
    created_at: new Date().toISOString(),
    out_root: resolveRepoPath(options.outRoot),
    lanes: manifests.map((manifest) => ({
      lane: manifest.lane,
      model_path: manifest.model_path,
      manifest_path: path.join(manifest.out_dir, "manifest.json"),
      vocab_size: manifest.vocab_size,
      token_count: manifest.token_count,
      windows: manifest.windows,
      final_accuracy_per_mille: manifest.final_accuracy_per_mille,
    })),
  }, null, 2)}\n`, "utf8");
  console.log(`index_manifest=${indexPath}`);
}

try {
  main();
} catch (error) {
  console.error(`train-focused-pair-models: ${error.message}`);
  process.exit(1);
}
