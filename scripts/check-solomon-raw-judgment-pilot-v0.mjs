#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {sha256Bytes} from "./lib/solomon-council-v0.mjs";
import {imageTaskTokens} from "./lib/solomon-symbolic-image.mjs";
import {SolomonAttentionSampler} from "../web/attention-sampler.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runnerPath = fileURLToPath(import.meta.url);
const modelPath = path.join(root, "web/assets/solomon-attention.nsrllmm");
const sourcePath = path.join(root, "web/assets/solomon-spirit-text-signatures.tsv");
const resultPath = path.join(
  root,
  "benchmarks/solomon-council-v0/raw-judgment-pilot-v0.json",
);
const freeze = process.argv.slice(2).includes("--freeze");
if (process.argv.length > (freeze ? 3 : 2)) {
  throw new Error("Usage: node scripts/check-solomon-raw-judgment-pilot-v0.mjs [--freeze]");
}

const sourceBytes = fs.readFileSync(sourcePath);
const rows = sourceBytes.toString("utf8").trimEnd().split("\n").slice(1).map((line) => {
  const columns = line.split("\t");
  return {
    name: columns[1],
    signature: columns[7].split(",").map(Number),
  };
});
if (rows.length !== 72 || rows.some((row) => row.signature.length !== 256)) {
  throw new Error("raw judgment pilot requires 72 canonical 16x16 signatures");
}

const modelBytes = fs.readFileSync(modelPath);
const sampler = new SolomonAttentionSampler(modelBytes);
sampler.textMemory = new Proxy({}, {
  get() {
    throw new Error("raw judgment pilot touched embedded text memory");
  },
});
sampler.selectTextExample = () => {
  throw new Error("raw judgment pilot selected an embedded example");
};
const candidates = [
  {candidate_id: "yes", text: "yes"},
  {candidate_id: "no", text: "no"},
];
const conditions = [];
for (const profile of ["ink16", "symbolic16"]) {
  let correct = 0;
  let yes = 0;
  let promptVisible = 0;
  let imageMarkerVisible = 0;
  const traceHashes = [];
  for (let index = 0; index < rows.length; index += 1) {
    for (const [imageRow, expected] of [
      [rows[index], true],
      [rows[(index + 1) % rows.length], false],
    ]) {
      const score = sampler.scoreRawContinuations(
        `seal of ${rows[index].name}`,
        candidates,
        {task: "match", imageTokens: imageTaskTokens(imageRow.signature, profile)},
      );
      const predicted = score.selected_candidate_id === "yes";
      correct += Number(predicted === expected);
      yes += Number(predicted);
      promptVisible += Number(score.conditioning.prompt_marker_visible);
      imageMarkerVisible += Number(score.conditioning.image_marker_visible);
      traceHashes.push(sha256Bytes(Buffer.from(JSON.stringify(score))));
      if (score.provenance.embedded_text_memory_used
          || score.provenance.retrieval_used
          || score.provenance.oracle_or_target_lookup_used
          || !score.provenance.raw_transformer_only) {
        throw new Error("raw judgment pilot scorer declared a forbidden path");
      }
    }
  }
  const total = rows.length * 2;
  conditions.push({
    image_token_profile: profile,
    balanced_trials: total,
    positive_trials: rows.length,
    negative_trials: rows.length,
    correct,
    accuracy_per_mille: Math.floor(correct * 1000 / total),
    yes_predictions: yes,
    no_predictions: total - yes,
    prompt_marker_visible_trials: promptVisible,
    image_marker_visible_trials: imageMarkerVisible,
    trace_set_sha256: sha256Bytes(Buffer.from(traceHashes.join("\n"))),
  });
}

const result = {
  schema: "nsrl.solomon_raw_judgment_pilot.v0",
  analysis_role: "substrate_falsification",
  model: {
    path: path.relative(root, modelPath),
    sha256: sha256Bytes(modelBytes),
    outer_model_hash: `0x${sampler.modelHash.toString(16).padStart(16, "0")}`,
    inner_model_hash: `0x${sampler.innerModelHash.toString(16).padStart(16, "0")}`,
    context_tokens: sampler.contextSeqLen,
    embedded_memory_examples_present_but_disabled: 72,
  },
  source: {
    path: path.relative(root, sourcePath),
    sha256: sha256Bytes(sourceBytes),
    canonical_seals: rows.length,
  },
  runner: {
    path: path.relative(root, runnerPath),
    sha256: sha256Bytes(fs.readFileSync(runnerPath)),
  },
  protocol: {
    task: "match",
    candidates: candidates.map(({candidate_id, text}) => ({candidate_id, text})),
    negative_rule: "cyclic-next-seal",
    raw_transformer_only: true,
    hidden_memory_used: false,
    retrieval_used: false,
    oracle_or_target_lookup_used: false,
  },
  conditions,
  conclusion: {
    suitable_for_frozen_wisdom_ceremony: false,
    reason: "balanced seal-match accuracy is exactly chance because every prediction is no and the 32-token decision context excludes both prompt and image markers",
    production_casebook_freeze_authorized: false,
  },
};
const bytes = Buffer.from(`${JSON.stringify(result, null, 2)}\n`);
if (freeze) {
  fs.mkdirSync(path.dirname(resultPath), {recursive: true});
  fs.writeFileSync(resultPath, bytes);
} else {
  if (!fs.existsSync(resultPath) || !fs.readFileSync(resultPath).equals(bytes)) {
    throw new Error("frozen raw judgment pilot does not byte-replay; run with --freeze to inspect");
  }
}
process.stdout.write(bytes);
