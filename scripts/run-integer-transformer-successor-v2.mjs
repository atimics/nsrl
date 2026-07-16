#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const manifestSchema = "nsrl.integer_transformer_successor_manifest.v2";
const contract = "integer-transformer-successor-v2";
const frozenDatasetHash = "0x8fe7b86378f81951";
const frozenTargets = 5896;
const tokenizer = "byte_identity_u8_v1";
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;
const manifestHeader = "schema\tcontract\ttrain\teval\tcandidate\tcontext\tstride\ttargets\tdataset_hash\ttokenizer\ttokenizer_hash\tcandidate_model_hash\tcandidate_artifact_hash\tevaluator_hash\trunner_hash\tmatrix_hash\tevidence_hash\ttransformer_replay_hash\tuniform_replay_hash\tretrieval_replay_hash\tbyte_ngram_replay_hash\tfloat_transformer_replay_hash";
const evaluatorSources = [
  "crates/nsrl-core/Cargo.toml",
  "crates/nsrl-core/src/attention.rs",
  "crates/nsrl-core/src/lib.rs",
  "crates/nsrl-core/src/numeric.rs",
  "crates/nsrl-eval/Cargo.toml",
  "crates/nsrl-eval/src/contract.rs",
  "crates/nsrl-eval/src/successor.rs",
  "crates/nsrl-train-core/Cargo.toml",
  "crates/nsrl-train-core/src/lib.rs",
  "crates/nsrl-train/Cargo.toml",
  "crates/nsrl-train/src/artifact_contract.rs",
  "crates/nsrl-train/src/lib.rs",
  "crates/nsrl-train/src/mini_transformer/block_expert.rs",
  "crates/nsrl-train/src/mini_transformer/decoding.rs",
  "crates/nsrl-train/src/mini_transformer/generation.rs",
  "crates/nsrl-train/src/mini_transformer/gradients.rs",
  "crates/nsrl-train/src/mini_transformer/model.rs",
  "crates/nsrl-train/src/mini_transformer/trace.rs",
  "crates/nsrl-train/src/mini_transformer/training.rs",
  "crates/nsrl-train/src/bin/nsrl-successor-eval.rs",
  "scripts/run-float-transformer-successor-v2.py",
];

export function fnv64(bytes, initial = fnvOffset) {
  let hash = initial;
  for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  return hash;
}

export function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

export function sourceSetHash() {
  let hash = fnvOffset;
  for (const relative of evaluatorSources) {
    const absolute = path.join(root, relative);
    if (!fs.existsSync(absolute)) throw new Error(`missing evaluator source ${relative}`);
    hash = fnv64(Buffer.from(relative), hash);
    hash = fnv64(Buffer.from([0]), hash);
    hash = fnv64(fs.readFileSync(absolute), hash);
    hash = fnv64(Buffer.from([255]), hash);
  }
  return hex64(hash);
}

export function runnerHash() {
  return hex64(fnv64(fs.readFileSync(fileURLToPath(import.meta.url))));
}

function simpleRelative(value, field) {
  if (!value || path.isAbsolute(value) || value.split(/[\\/]/).some((part) => part === ".." || part === "." || part === "")) {
    throw new Error(`${field} must be a simple relative path`);
  }
  return value;
}

export function loadManifest(manifestPath) {
  const absolute = path.resolve(manifestPath);
  const lines = fs.readFileSync(absolute, "utf8").trimEnd().split("\n");
  if (lines.length !== 2 || lines[0] !== manifestHeader) {
    throw new Error(`successor manifest must contain ${manifestHeader} and one row`);
  }
  const fields = lines[1].split("\t");
  if (fields.length !== 22 || fields[0] !== manifestSchema || fields[1] !== contract) {
    throw new Error("successor manifest schema or contract mismatch");
  }
  const directory = path.dirname(absolute);
  const trainPath = path.join(directory, simpleRelative(fields[2], "train"));
  const evalPath = path.join(directory, simpleRelative(fields[3], "eval"));
  const candidatePath = path.join(directory, simpleRelative(fields[4], "candidate"));
  const context = strictPositive(fields[5], "context");
  const stride = strictPositive(fields[6], "stride");
  const targets = strictPositive(fields[7], "targets");
  if (context !== 64 || stride !== 1 || targets !== frozenTargets) {
    throw new Error("successor manifest evaluation geometry is not the frozen 64/1/5,896 surface");
  }
  const trainBytes = fs.readFileSync(trainPath);
  const evalBytes = fs.readFileSync(evalPath);
  const datasetHash = hex64(fnv64(evalBytes, fnv64(Buffer.from([255]), fnv64(trainBytes))));
  if (fields[8] !== frozenDatasetHash || datasetHash !== frozenDatasetHash) {
    throw new Error(`successor dataset hash mismatch: manifest=${fields[8]} actual=${datasetHash}`);
  }
  if (evalBytes.length - context !== targets) {
    throw new Error(`successor target count mismatch: ${evalBytes.length - context} != ${targets}`);
  }
  const tokenizerHash = hex64(fnv64(Buffer.from(tokenizer)));
  if (fields[9] !== tokenizer || fields[10] !== tokenizerHash) {
    throw new Error("successor tokenizer binding mismatch");
  }
  const candidateArtifactHash = hex64(fnv64(fs.readFileSync(candidatePath)));
  if (fields[12] !== candidateArtifactHash) {
    throw new Error(`successor candidate artifact hash mismatch: ${fields[12]} != ${candidateArtifactHash}`);
  }
  for (const [index, name] of [
    [11, "candidate_model_hash"], [13, "evaluator_hash"], [14, "runner_hash"],
    [15, "matrix_hash"], [16, "evidence_hash"], [17, "transformer_replay_hash"],
    [18, "uniform_replay_hash"], [19, "retrieval_replay_hash"],
    [20, "byte_ngram_replay_hash"], [21, "float_transformer_replay_hash"],
  ]) {
    if (!/^0x[0-9a-f]{16}$/.test(fields[index])) throw new Error(`invalid ${name}`);
  }
  return {
    absolute,
    directory,
    trainPath,
    evalPath,
    candidatePath,
    context,
    stride,
    targets,
    datasetHash,
    tokenizerHash,
    candidateModelHash: fields[11],
    candidateArtifactHash,
    evaluatorHash: fields[13],
    runnerHash: fields[14],
    matrixHash: fields[15],
    evidenceHash: fields[16],
    replayHashes: fields.slice(17, 22),
  };
}

function command(program, args, options = {}) {
  const result = spawnSync(program, args, {
    cwd: root,
    encoding: "utf8",
    stdio: options.capture ? "pipe" : "inherit",
    env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1" },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = options.capture ? `\n${result.stdout}\n${result.stderr}` : "";
    throw new Error(`${program} exited ${result.status}${detail}`);
  }
  return result.stdout?.trim() ?? "";
}

function parseArgs(argv) {
  const config = {
    manifest: "benchmarks/integer-transformer-proof-v1/successor-v2-manifest.tsv",
    mode: "check",
    outDir: "",
    allowUnfrozen: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") config.manifest = argv[++index] ?? "";
    else if (arg === "--out-dir") config.outDir = argv[++index] ?? "";
    else if (arg === "--check") config.mode = "check";
    else if (arg === "--freeze") config.mode = "freeze";
    else if (arg === "--allow-unfrozen") config.allowUnfrozen = true;
    else throw new Error(`unknown argument ${arg}`);
  }
  if (!config.manifest) throw new Error("--manifest PATH is required");
  return config;
}

function validateFloatTrace(trace, manifest) {
  if (trace.schema !== "nsrl.float_transformer_successor.v2"
      || trace.contract !== contract
      || trace.bindings?.dataset_hash !== manifest.datasetHash
      || trace.bindings?.targets !== frozenTargets
      || trace.bindings?.tokenizer !== tokenizer
      || trace.bindings?.tokenizer_hash !== manifest.tokenizerHash) {
    throw new Error("float transformer trace bindings are invalid");
  }
  if (trace.architecture?.kind !== "genuine_float_transformer"
      || trace.architecture?.dtype !== "float32"
      || !String(trace.architecture?.attention).startsWith("causal_")
      || Number(trace.architecture?.heads) < 1
      || Number(trace.architecture?.layers) < 1
      || Number(trace.architecture?.residual_connections) < 1) {
    throw new Error("float baseline is not a genuine causal float transformer");
  }
  const trained = trace.training?.trained_parameter_groups;
  for (const group of ["embeddings", "q", "k", "v", "o", "up", "gate", "down", "output"]) {
    if (!Array.isArray(trained) || !trained.includes(group)) {
      throw new Error(`float transformer did not train ${group}`);
    }
  }
  if (trace.evaluation?.partition !== "eval"
      || trace.evaluation?.stride !== 1
      || trace.evaluation?.windows !== frozenTargets) {
    throw new Error("float transformer did not evaluate the identical frozen partition");
  }
  if (Object.values(trace.assistance ?? {}).some(Boolean)) {
    throw new Error("float transformer declares forbidden assistance");
  }
}

function publishedPaths(manifest) {
  return {
    matrix: path.join(manifest.directory, "successor-v2-matrix.tsv"),
    evidence: path.join(manifest.directory, "successor-v2-evidence.json"),
    floatTrace: path.join(manifest.directory, "successor-v2-float-transformer.json"),
  };
}

function byteEqual(actual, expected, label) {
  if (!fs.existsSync(expected) || !fs.readFileSync(actual).equals(fs.readFileSync(expected))) {
    throw new Error(`${label} does not match the replayed successor-v2 artifact`);
  }
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const manifest = loadManifest(config.manifest);
  const actualEvaluatorHash = sourceSetHash();
  const actualRunnerHash = runnerHash();
  if (!config.allowUnfrozen && manifest.evaluatorHash !== actualEvaluatorHash) {
    throw new Error(`evaluator hash mismatch: ${manifest.evaluatorHash} != ${actualEvaluatorHash}`);
  }
  if (!config.allowUnfrozen && manifest.runnerHash !== actualRunnerHash) {
    throw new Error(`runner hash mismatch: ${manifest.runnerHash} != ${actualRunnerHash}`);
  }

  const temporary = config.outDir
    ? path.resolve(config.outDir)
    : fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-successor-v2-"));
  fs.mkdirSync(temporary, { recursive: true });
  const floatLogits = path.join(temporary, "float-transformer.logits");
  const floatTrace = path.join(temporary, "float-transformer.json");
  const matrix = path.join(temporary, "matrix.tsv");
  const evidence = path.join(temporary, "evidence.json");

  command("python3", [
    "scripts/run-float-transformer-successor-v2.py",
    "--train", path.relative(root, manifest.trainPath),
    "--eval", path.relative(root, manifest.evalPath),
    "--out-logits", floatLogits,
    "--out-trace", floatTrace,
  ]);
  const floatTraceValue = JSON.parse(fs.readFileSync(floatTrace, "utf8"));
  validateFloatTrace(floatTraceValue, manifest);
  const floatTraceHash = hex64(fnv64(fs.readFileSync(floatTrace)));

  command("cargo", [
    "build", "--release", "-p", "nsrl-train", "--bin", "nsrl-successor-eval",
    "--features", "mini-heads-8,mini-calibrated",
  ]);
  command(path.join(root, "target/release/nsrl-successor-eval"), [
    "--train", manifest.trainPath,
    "--eval", manifest.evalPath,
    "--candidate", manifest.candidatePath,
    "--float-logits", floatLogits,
    "--out-matrix", matrix,
    "--out-evidence", evidence,
    "--dataset-hash", manifest.datasetHash,
    "--tokenizer-hash", manifest.tokenizerHash,
    "--candidate-model-hash", manifest.candidateModelHash,
    "--candidate-artifact-hash", manifest.candidateArtifactHash,
    "--evaluator-hash", actualEvaluatorHash,
    "--runner-hash", actualRunnerHash,
    "--float-trace-hash", floatTraceHash,
  ]);

  const matrixHash = hex64(fnv64(fs.readFileSync(matrix)));
  const evidenceHash = hex64(fnv64(fs.readFileSync(evidence)));
  const resultLines = fs.readFileSync(matrix, "utf8").trimEnd().split("\n").slice(1);
  const replayHashes = resultLines.map((line) => line.split("\t")[17]);
  if (resultLines.length !== 5) throw new Error("successor evaluator did not emit five systems");
  const freezeValues = {
    evaluator_hash: actualEvaluatorHash,
    runner_hash: actualRunnerHash,
    matrix_hash: matrixHash,
    evidence_hash: evidenceHash,
    replay_hashes: replayHashes,
    float_transformer_model_hash: floatTraceValue.model.final_hash,
  };
  console.log(JSON.stringify(freezeValues));

  if (!config.allowUnfrozen) {
    if (matrixHash !== manifest.matrixHash || evidenceHash !== manifest.evidenceHash) {
      throw new Error("successor matrix/evidence hashes do not match the frozen manifest");
    }
    if (JSON.stringify(replayHashes) !== JSON.stringify(manifest.replayHashes)) {
      throw new Error("successor replay hashes do not match the frozen manifest");
    }
  }

  const published = publishedPaths(manifest);
  if (config.mode === "freeze") {
    fs.copyFileSync(matrix, published.matrix);
    fs.copyFileSync(evidence, published.evidence);
    fs.copyFileSync(floatTrace, published.floatTrace);
  } else if (!config.allowUnfrozen) {
    byteEqual(matrix, published.matrix, "published matrix");
    byteEqual(evidence, published.evidence, "published evidence");
    byteEqual(floatTrace, published.floatTrace, "published float-transformer trace");
  }

  if (!config.allowUnfrozen) {
    const check = command("cargo", [
      "run", "--quiet", "-p", "nsrl-eval", "--", "successor-check",
      "--manifest", manifest.absolute,
      "--results", published.matrix,
      "--evidence", published.evidence,
      "--allow-falsification",
    ], { capture: true });
    const result = JSON.parse(check);
    if (typeof result.passed !== "boolean") throw new Error("successor checker omitted promotion decision");
    console.log(JSON.stringify({ valid: true, promoted: result.passed, targets: result.targets }));
  }
}

function strictPositive(value, name) {
  if (!/^[1-9][0-9]*$/.test(value)) throw new Error(`${name} must be a positive integer`);
  return Number(value);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
