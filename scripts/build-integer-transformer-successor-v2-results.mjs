#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const manifestSchema = "nsrl.integer_transformer_successor_manifest.v2";
const resultSchema = "nsrl.integer_transformer_successor_result.v2";
const contractId = "integer-transformer-successor-v2";
const datasetHash = "0x8fe7b86378f81951";
const targetCount = 5896;
const assistance = "suffix-memory=off,retrieval=off,routing-oracle=off";
const manifestHeader = "schema\tcontract\ttrain\teval\tcontext\tstride\ttargets\tdataset_hash\tcandidate\tcandidate_artifact_hash\tcandidate_hash\tmodel_hash\trunner\trunner_hash\tassistance\tfloat_model\tfloat_model_hash\tfloat_runner\tfloat_runner_hash";
const resultHeader = "schema\tcontract\tsuite\tpartition\tdataset_hash\tcandidate_hash\tmodel_hash\trunner_hash\tassistance_hash\tsystem\ttargets\tmistakes\ttotal_nll_millibits\tzero_probability_windows\treplay_hash";
const byteClasses = 256;
const zeroProbabilityFloorMillibits = 32000;
const fnvOffset = 0xcbf29ce484222325n;
const fnvPrime = 0x100000001b3n;
const fnvMask = 0xffffffffffffffffn;
const exp2NegFracQ15 = [
  32767, 32679, 32591, 32503, 32415, 32327, 32240, 32153, 32066, 31979, 31893, 31806, 31720,
  31635, 31549, 31464, 31379, 31294, 31209, 31125, 31041, 30957, 30873, 30790, 30706, 30623,
  30541, 30458, 30376, 30293, 30212, 30130, 30048, 29967, 29886, 29805, 29725, 29644, 29564,
  29484, 29405, 29325, 29246, 29167, 29088, 29009, 28931, 28852, 28774, 28697, 28619, 28542,
  28464, 28388, 28311, 28234, 28158, 28082, 28006, 27930, 27855, 27779, 27704, 27629, 27554,
  27480, 27406, 27332, 27258, 27184, 27110, 27037, 26964, 26891, 26818, 26746, 26674, 26601,
  26530, 26458, 26386, 26315, 26244, 26173, 26102, 26031, 25961, 25891, 25821, 25751, 25681,
  25612, 25543, 25474, 25405, 25336, 25268, 25199, 25131, 25063, 24995, 24928, 24860, 24793,
  24726, 24659, 24593, 24526, 24460, 24394, 24328, 24262, 24196, 24131, 24066, 24001, 23936,
  23871, 23806, 23742, 23678, 23614, 23550, 23486, 23423, 23359, 23296, 23233, 23170, 23108,
  23045, 22983, 22921, 22859, 22797, 22735, 22674, 22613, 22552, 22491, 22430, 22369, 22309,
  22248, 22188, 22128, 22068, 22009, 21949, 21890, 21831, 21772, 21713, 21654, 21595, 21537,
  21479, 21421, 21363, 21305, 21247, 21190, 21133, 21076, 21019, 20962, 20905, 20849, 20792,
  20736, 20680, 20624, 20568, 20513, 20457, 20402, 20347, 20292, 20237, 20182, 20127, 20073,
  20019, 19965, 19911, 19857, 19803, 19750, 19696, 19643, 19590, 19537, 19484, 19431, 19379,
  19326, 19274, 19222, 19170, 19118, 19066, 19015, 18963, 18912, 18861, 18810, 18759, 18708,
  18658, 18607, 18557, 18507, 18457, 18407, 18357, 18308, 18258, 18209, 18160, 18110, 18061,
  18013, 17964, 17915, 17867, 17819, 17770, 17722, 17674, 17627, 17579, 17531, 17484, 17437,
  17390, 17343, 17296, 17249, 17202, 17156, 17109, 17063, 17017, 16971, 16925, 16879, 16834,
  16788, 16743, 16697, 16652, 16607, 16562, 16518, 16473, 16428,
];

function parseArgs(argv) {
  const config = { manifest: "", candidateTrace: "", candidateLogits: "", floatTrace: "", floatLogits: "", out: "" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--manifest") config.manifest = argv[++index] ?? "";
    else if (arg === "--candidate-trace") config.candidateTrace = argv[++index] ?? "";
    else if (arg === "--candidate-logits") config.candidateLogits = argv[++index] ?? "";
    else if (arg === "--float-trace") config.floatTrace = argv[++index] ?? "";
    else if (arg === "--float-logits") config.floatLogits = argv[++index] ?? "";
    else if (arg === "--out") config.out = argv[++index] ?? "";
    else throw new Error(`unknown argument: ${arg}`);
  }
  for (const [name, value] of Object.entries(config)) {
    if (!value) throw new Error(`--${name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)} PATH is required`);
  }
  return config;
}

export function stableHash(bytes) {
  let hash = fnvOffset;
  for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function hashParts(parts) {
  let hash = fnvOffset;
  for (const part of parts) {
    for (const byte of part) hash = ((hash ^ BigInt(byte)) * fnvPrime) & fnvMask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function stableU8SliceHash(bytes) {
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(bytes.length));
  return hashParts([length, bytes]);
}

function resolveManifestPath(directory, value) {
  if (!value || path.isAbsolute(value)) throw new Error(`manifest path must be relative: ${value}`);
  return path.resolve(directory, value);
}

function loadManifest(manifestPath) {
  const absolute = path.resolve(manifestPath);
  const lines = fs.readFileSync(absolute, "utf8").trimEnd().split("\n");
  if (lines.length !== 2 || lines[0] !== manifestHeader) {
    throw new Error("successor manifest header or row count is invalid");
  }
  const fields = lines[1].split("\t");
  if (fields.length !== 19 || fields[0] !== manifestSchema || fields[1] !== contractId) {
    throw new Error("successor manifest schema or contract is invalid");
  }
  const directory = path.dirname(absolute);
  const manifest = {
    absolute,
    trainPath: resolveManifestPath(directory, fields[2]),
    evalPath: resolveManifestPath(directory, fields[3]),
    context: Number(fields[4]),
    stride: Number(fields[5]),
    targets: Number(fields[6]),
    datasetHash: fields[7],
    candidatePath: resolveManifestPath(directory, fields[8]),
    candidateArtifactHash: fields[9],
    candidateHash: fields[10],
    modelHash: fields[11],
    runnerPath: resolveManifestPath(directory, fields[12]),
    runnerHash: fields[13],
    assistance: fields[14],
    assistanceHash: stableHash(Buffer.from(fields[14], "utf8")),
    floatModelPath: resolveManifestPath(directory, fields[15]),
    floatModelHash: fields[16],
    floatRunnerPath: resolveManifestPath(directory, fields[17]),
    floatRunnerHash: fields[18],
  };
  const train = fs.readFileSync(manifest.trainPath);
  const evaluation = fs.readFileSync(manifest.evalPath);
  const actualDatasetHash = hashParts([train, Buffer.from([255]), evaluation]);
  const actualTargets = evaluation.length > manifest.context
    ? Math.ceil((evaluation.length - manifest.context) / manifest.stride)
    : 0;
  if (manifest.datasetHash !== datasetHash || actualDatasetHash !== datasetHash) {
    throw new Error(`successor dataset binding mismatch: ${manifest.datasetHash}, ${actualDatasetHash}`);
  }
  if (manifest.targets !== targetCount || actualTargets !== targetCount) {
    throw new Error(`successor target binding mismatch: ${manifest.targets}, ${actualTargets}`);
  }
  if (manifest.assistance !== assistance) throw new Error("successor assistance binding mismatch");
  for (const [name, file, expected] of [
    ["candidate artifact", manifest.candidatePath, manifest.candidateArtifactHash],
    ["runner", manifest.runnerPath, manifest.runnerHash],
    ["float model", manifest.floatModelPath, manifest.floatModelHash],
    ["float runner", manifest.floatRunnerPath, manifest.floatRunnerHash],
  ]) {
    const actual = stableHash(fs.readFileSync(file));
    if (actual !== expected) throw new Error(`${name} hash ${actual} does not match ${expected}`);
  }
  return { ...manifest, train, evaluation };
}

function validateCandidateTrace(trace, manifest) {
  const ablation = trace.ablation ?? {};
  if (trace.schema !== "nsrl.mini_transformer_eval.v1"
    || trace.data?.token_count !== manifest.evaluation.length
    || trace.data?.token_hash !== stableU8SliceHash(manifest.evaluation)
    || trace.data?.windows !== manifest.targets
    || trace.model?.seq_len !== manifest.context
    || trace.evaluation?.stride !== manifest.stride
    || trace.evaluation?.invalid_forward_count !== 0
    || ablation.mode !== "transformer-only"
    || ablation.source_model_hash !== manifest.candidateHash
    || ablation.evaluated_model_hash !== manifest.modelHash
    || ablation.source_suffix_memory_present !== true
    || ablation.suffix_memory_enabled !== false
    || ablation.retrieval_enabled !== false
    || ablation.routing_oracle_enabled !== false) {
    throw new Error("candidate trace does not match the manifest-bound transformer-only trial");
  }
}

function validateFloatTrace(trace, manifest) {
  if (trace.schema !== "nsrl.float_transformer_eval.v1"
    || trace.contract !== contractId
    || trace.dataset_hash !== manifest.datasetHash
    || trace.targets !== manifest.targets
    || trace.context !== manifest.context
    || trace.stride !== manifest.stride
    || trace.model_hash !== manifest.floatModelHash
    || trace.runner_hash !== manifest.floatRunnerHash
    || trace.architecture?.kind !== "causal-float-transformer"
    || trace.architecture?.attention !== "scaled-dot-product-softmax"
    || trace.training?.trained_parameters !== "all") {
    throw new Error("float trace is not the manifest-bound trained float transformer");
  }
}

function readLogits(file, targets) {
  const bytes = fs.readFileSync(file);
  const expected = targets * byteClasses * 4;
  if (bytes.length !== expected) {
    throw new Error(`${file} has ${bytes.length} bytes, expected ${expected}`);
  }
  return new Int32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
}

function integerLog2Q20(value) {
  if (value <= 0n) throw new Error("integerLog2Q20 requires a positive value");
  const integer = BigInt(value.toString(2).length - 1);
  let normalized = value << (63n - integer);
  let fractional = 0n;
  for (let bit = 19n; bit >= 0n; bit -= 1n) {
    normalized = (normalized * normalized) >> 63n;
    if (normalized >= (1n << 64n)) {
      normalized >>= 1n;
      fractional |= 1n << bit;
    }
  }
  return (integer << 20n) | fractional;
}

function countLogits(counts) {
  const logits = new Int32Array(byteClasses);
  for (let token = 0; token < byteClasses; token += 1) {
    const q20 = integerLog2Q20(BigInt(counts[token] + 1));
    logits[token] = Number((q20 + (1n << 11n)) >> 12n);
  }
  return logits;
}

function base2WeightQ15(deltaQ8) {
  if (deltaQ8 >= 0) return 32767;
  const magnitude = -deltaQ8;
  const integerShift = Math.floor(magnitude / 256);
  if (integerShift >= 15) return 0;
  return exp2NegFracQ15[magnitude & 255] >> integerShift;
}

function scoreWindow(logits, target) {
  let predicted = 0;
  let maxLogit = logits[0];
  for (let token = 1; token < byteClasses; token += 1) {
    if (logits[token] > maxLogit) {
      maxLogit = logits[token];
      predicted = token;
    }
  }
  let sum = 0n;
  let targetWeight = 0n;
  for (let token = 0; token < byteClasses; token += 1) {
    const weight = BigInt(base2WeightQ15(logits[token] - maxLogit));
    sum += weight;
    if (token === target) targetWeight = weight;
  }
  if (targetWeight === 0n) {
    return { predicted, nll: zeroProbabilityFloorMillibits, zero: true, targetWeight };
  }
  const lossQ20 = integerLog2Q20(sum) - integerLog2Q20(targetWeight);
  const nll = Number((lossQ20 * 1000n + (1n << 19n)) >> 20n);
  return { predicted, nll, zero: false, targetWeight };
}

function replayUpdateU64(hash, value) {
  let current = BigInt(value);
  let out = hash;
  for (let index = 0; index < 8; index += 1) {
    out = ((out ^ (current & 255n)) * fnvPrime) & fnvMask;
    current >>= 8n;
  }
  return out;
}

function scoreSystem(manifest, system, logitsForWindow) {
  let mistakes = 0;
  let totalNllMillibits = 0;
  let zeroProbabilityWindows = 0;
  let replay = fnvOffset;
  for (const byte of Buffer.from(`${contractId}\0${system}\0`, "utf8")) {
    replay = ((replay ^ BigInt(byte)) * fnvPrime) & fnvMask;
  }
  let window = 0;
  for (let start = 0; start + manifest.context < manifest.evaluation.length; start += manifest.stride) {
    const target = manifest.evaluation[start + manifest.context];
    const score = scoreWindow(logitsForWindow(window, start), target);
    mistakes += Number(score.predicted !== target);
    totalNllMillibits += score.nll;
    zeroProbabilityWindows += Number(score.zero);
    replay = replayUpdateU64(replay, BigInt(start));
    replay = replayUpdateU64(replay, BigInt(target));
    replay = replayUpdateU64(replay, BigInt(score.predicted));
    replay = replayUpdateU64(replay, BigInt(score.nll));
    replay = replayUpdateU64(replay, score.targetWeight);
    window += 1;
  }
  if (window !== manifest.targets) throw new Error(`${system} scored ${window} windows`);
  return {
    system,
    targets: window,
    mistakes,
    totalNllMillibits,
    zeroProbabilityWindows,
    replayHash: `0x${replay.toString(16).padStart(16, "0")}`,
  };
}

function buildTables(bytes, orders) {
  const tables = new Map(orders.map((order) => [order, new Map()]));
  const unigram = new Uint32Array(byteClasses);
  for (let target = 0; target < bytes.length; target += 1) {
    unigram[bytes[target]] += 1;
    for (const order of orders) {
      if (order === 0 || target < order) continue;
      const key = bytes.subarray(target - order, target).toString("hex");
      const table = tables.get(order);
      let counts = table.get(key);
      if (!counts) {
        counts = new Uint32Array(byteClasses);
        table.set(key, counts);
      }
      counts[bytes[target]] += 1;
    }
  }
  tables.set(0, new Map([["", unigram]]));
  return tables;
}

function lookup(tables, context, order) {
  return tables.get(order)?.get(context.subarray(context.length - order).toString("hex"));
}

function resultRow(manifest, row) {
  return [
    resultSchema,
    contractId,
    "substrate",
    "eval",
    manifest.datasetHash,
    manifest.candidateHash,
    manifest.modelHash,
    manifest.runnerHash,
    manifest.assistanceHash,
    row.system,
    row.targets,
    row.mistakes,
    row.totalNllMillibits,
    row.zeroProbabilityWindows,
    row.replayHash,
  ].join("\t");
}

export function buildResults(config) {
  const manifest = loadManifest(config.manifest);
  const candidateTrace = JSON.parse(fs.readFileSync(config.candidateTrace, "utf8"));
  const floatTrace = JSON.parse(fs.readFileSync(config.floatTrace, "utf8"));
  validateCandidateTrace(candidateTrace, manifest);
  validateFloatTrace(floatTrace, manifest);
  const candidateLogits = readLogits(config.candidateLogits, manifest.targets);
  const floatLogits = readLogits(config.floatLogits, manifest.targets);
  const orders = [1, 2, 4, 8, 16, 32, 64];
  const tables = buildTables(manifest.train, orders);
  const unigramLogits = countLogits(tables.get(0).get(""));
  const uniformLogits = new Int32Array(byteClasses);
  const candidate = scoreSystem(manifest, "transformer-only", (window) =>
    candidateLogits.subarray(window * byteClasses, (window + 1) * byteClasses));
  const uniform = scoreSystem(manifest, "uniform", () => uniformLogits);
  const retrieval = scoreSystem(manifest, "retrieval", (_window, start) => {
    const context = manifest.evaluation.subarray(start, start + manifest.context);
    for (const order of [64, 32, 16, 8, 4, 2, 1]) {
      const counts = lookup(tables, context, order);
      if (counts) return countLogits(counts);
    }
    return unigramLogits;
  });
  const byteNgram = scoreSystem(manifest, "byte-ngram", (_window, start) => {
    const context = manifest.evaluation.subarray(start, start + manifest.context);
    for (const order of [4, 2, 1]) {
      const counts = lookup(tables, context, order);
      if (counts) return countLogits(counts);
    }
    return unigramLogits;
  });
  const floatTransformer = scoreSystem(manifest, "float-transformer", (window) =>
    floatLogits.subarray(window * byteClasses, (window + 1) * byteClasses));
  if (candidate.mistakes !== candidateTrace.evaluation.mistakes) {
    throw new Error("candidate logits do not reproduce the candidate trace mistake count");
  }
  if (floatTransformer.mistakes !== floatTrace.evaluation?.q8_mistakes) {
    throw new Error("float logits do not reproduce the float trace mistake count");
  }
  const rows = [candidate, uniform, retrieval, byteNgram, floatTransformer];
  return {
    output: `${resultHeader}\n${rows.map((row) => resultRow(manifest, row)).join("\n")}\n`,
    summary: {
      dataset_hash: manifest.datasetHash,
      targets: manifest.targets,
      candidate_hash: manifest.candidateHash,
      model_hash: manifest.modelHash,
      runner_hash: manifest.runnerHash,
      assistance_hash: manifest.assistanceHash,
      rows,
    },
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const built = buildResults(config);
  fs.mkdirSync(path.dirname(path.resolve(config.out)), { recursive: true });
  fs.writeFileSync(config.out, built.output);
  process.stdout.write(`${JSON.stringify(built.summary)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
