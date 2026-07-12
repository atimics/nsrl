#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.integer_transformer_proof_freeze.v1";
const contract = "integer-transformer-proof-v1";
const defaultCandidate = "data/experiments/integer-transformer-proof-v1/candidate-default";
const defaultFreeze = "benchmarks/integer-transformer-proof-v1/promoted-candidate.json";
const frozenFiles = [
  "candidate.nsrlmt",
  "candidate.eval.json",
  "candidate-health.json",
  "manifest.json",
  "proof-check.json",
  "proof-results.tsv",
  "train.trace.jsonl",
];

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function buildFreeze(candidateRelative) {
  const candidate = path.resolve(root, candidateRelative);
  const proof = readJson(path.join(candidate, "proof-check.json"));
  const health = readJson(path.join(candidate, "candidate-health.json"));
  const evaluation = readJson(path.join(candidate, "candidate.eval.json"));
  const training = readJson(path.join(candidate, "train.trace.jsonl").trim());
  if (proof.passed !== true) throw new Error("candidate proof did not pass");
  if (health.ok !== true) throw new Error("candidate health did not pass");
  if (proof.dataset_hash !== "0x8fe7b86378f81951") {
    throw new Error(`unexpected frozen dataset hash ${proof.dataset_hash}`);
  }
  if (evaluation.model.hash !== training.hashes.final_model) {
    throw new Error("evaluation and training model hashes do not match");
  }
  const files = Object.fromEntries(
    frozenFiles.map((name) => {
      const bytes = fs.readFileSync(path.join(candidate, name));
      return [name, { bytes: bytes.length, sha256: sha256(bytes) }];
    }),
  );
  return {
    schema,
    contract,
    status: "promoted",
    candidate_directory: candidateRelative,
    regeneration_command:
      "scripts/run-integer-transformer-proof-candidate.sh data/experiments/integer-transformer-proof-v1/candidate-default",
    dataset_hash: proof.dataset_hash,
    model_hash: evaluation.model.hash,
    quantization_profile: training.model.quantization_profile,
    architecture_profile: training.model.architecture_profile,
    metrics: {
      targets: proof.targets,
      mistakes: proof.candidate.mistakes,
      accuracy_per_mille: evaluation.evaluation.accuracy_per_mille,
      probability_error_q15: proof.candidate.probability_error_q15,
      unique_predicted_tokens: evaluation.evaluation.unique_predicted_tokens,
      most_predicted_token_share_per_mille:
        evaluation.evaluation.most_predicted_token_share_per_mille,
    },
    baselines: proof.baselines,
    files,
  };
}

function parseArgs(argv) {
  const config = { candidate: defaultCandidate, out: defaultFreeze, check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--candidate") config.candidate = argv[++index] ?? "";
    else if (arg === "--out") config.out = argv[++index] ?? "";
    else if (arg === "--check") config.check = true;
    else throw new Error(`unknown argument: ${arg}`);
  }
  return config;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const output = stableJson(buildFreeze(config.candidate));
  const out = path.resolve(root, config.out);
  if (config.check) {
    if (fs.readFileSync(out, "utf8") !== output) {
      throw new Error(`${config.out} is stale; regenerate the frozen candidate record`);
    }
    process.stdout.write(JSON.stringify({ checked: true, out: config.out }) + "\n");
    return;
  }
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, output);
  process.stdout.write(JSON.stringify({ frozen: true, out: config.out }) + "\n");
}

main();
