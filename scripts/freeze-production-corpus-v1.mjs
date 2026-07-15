#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let manifestPath = "data/processed/production-corpus-v1/manifest.json";
let checkpointPath = "benchmarks/production-corpus-v1/checkpoint.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--manifest") manifestPath = process.argv[++index];
  else if (arg === "--out") checkpointPath = process.argv[++index];
  else if (arg === "--check") checkOnly = true;
  else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/freeze-production-corpus-v1.mjs [--manifest PATH] [--out PATH] [--check]");
    process.exit(0);
  } else throw new Error(`unknown argument: ${arg}`);
}

function artifactSnapshot(artifact) {
  return {
    file: path.basename(artifact.path),
    bytes: artifact.bytes,
    sha256: artifact.sha256,
    ...(artifact.documents === undefined ? {} : { documents: artifact.documents }),
    ...(artifact.records === undefined ? {} : { records: artifact.records }),
    ...(artifact.quarantined === undefined ? {} : { quarantined: artifact.quarantined }),
    ...(artifact.source_split === undefined ? {} : { source_split: artifact.source_split }),
  };
}

function snapshot(manifest) {
  const artifacts = {};
  for (const name of [
    "train", "train_index", "dev", "dev_index", "test", "test_index",
    "records", "tokenizer_training", "contamination", "tokenizer", "tokenizer_trace",
    "train_tokens", "train_token_trace", "dev_tokens", "dev_token_trace",
    "test_tokens", "test_token_trace",
  ]) artifacts[name] = artifactSnapshot(manifest.artifacts[name]);
  return {
    schema: "nsrl.production_corpus_checkpoint.v1",
    contract_id: manifest.contract_id,
    config: manifest.config,
    sources: manifest.sources.map((source) => ({
      id: source.id,
      sha256: source.sha256,
      format: source.format,
      source_url: source.source_url,
      license_id: source.license_id,
      rights_basis_url: source.rights_basis_url,
      rights_scope: source.rights_scope,
      attribution: source.attribution,
      documents_loaded: source.documents_loaded,
      upstream: source.upstream,
      provenance_files: source.provenance_files,
    })),
    policy: manifest.policy,
    counts: manifest.counts,
    artifacts,
    tokenizer: manifest.tokenizer,
    encodings: manifest.encodings,
    gates: manifest.gates,
    known_non_claims: [
      "simplewiki_source_is_capped_at_20000_documents",
      "project_gutenberg_inputs_require_deployment_jurisdiction_review",
      "corpus_freeze_does_not_claim_model_quality",
      "ten_to_thirty_million_parameter_models_not_trained_yet",
      "same_shape_float_twins_not_trained_yet"
    ]
  };
}

function validate(checkpoint) {
  if (checkpoint.schema !== "nsrl.production_corpus_checkpoint.v1" || checkpoint.contract_id !== "production-corpus-v1") {
    throw new Error("invalid production corpus checkpoint schema or contract");
  }
  const requiredGates = [
    "no_cross_split_documents", "tokenizer_training_is_train_only",
    "evaluation_panels_excluded", "tokenizer_bound", "all_splits_encoded",
  ];
  for (const gate of requiredGates) if (checkpoint.gates[gate] !== true) throw new Error(`checkpoint gate failed: ${gate}`);
  if (checkpoint.gates.rights_approved_sources !== checkpoint.sources.length) throw new Error("rights-approved source count mismatch");
  if (checkpoint.tokenizer.target_vocab_size !== 8192 || checkpoint.tokenizer.actual_vocab_size !== 8192) {
    throw new Error("production tokenizer must reach 8192 tokens");
  }
  const accepted = ["train", "dev", "test"].reduce((sum, split) => sum + checkpoint.artifacts[split].documents, 0);
  if (accepted !== checkpoint.counts.accepted) throw new Error("accepted document count does not match splits");
  for (const split of ["train", "dev", "test"]) {
    if (checkpoint.encodings[split].documents !== checkpoint.artifacts[split].documents) {
      throw new Error(`encoding document count mismatch: ${split}`);
    }
    if (checkpoint.encodings[split].bos_tokens !== checkpoint.encodings[split].documents
      || checkpoint.encodings[split].eos_tokens !== checkpoint.encodings[split].documents) {
      throw new Error(`document boundary token mismatch: ${split}`);
    }
  }
  if (checkpoint.counts.contaminated_quarantined !== checkpoint.artifacts.contamination.quarantined) {
    throw new Error("contamination count mismatch");
  }
}

let checkpoint;
try {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  checkpoint = snapshot(manifest);
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = JSON.parse(await readFile(checkpointPath, "utf8"));
}
validate(checkpoint);
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  const frozen = await readFile(checkpointPath, "utf8");
  if (frozen !== rendered) throw new Error("production corpus checkpoint is stale");
  console.log(JSON.stringify({ schema: "nsrl.production_corpus_checkpoint_check.v1", ok: true, checkpoint: checkpointPath }));
} else {
  await writeFile(checkpointPath, rendered);
  console.log(checkpointPath);
}
