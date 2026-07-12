#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let experimentDir = "data/experiments/literary-recursive-swarm-v1";
let outPath = null;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--experiment-dir") experimentDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/summarize-recursive-literary-leaves.mjs [--experiment-dir PATH] [--out PATH]");
    process.exit(0);
  } else throw new Error(`unknown argument: ${arg}`);
}

const manifest = JSON.parse(await readFile(path.join(experimentDir, "experiment.manifest.json"), "utf8"));
const authors = manifest.authors;
const rows = [];
for (const job of manifest.leaf_jobs) {
  const expertDir = path.dirname(job.model_path);
  const train = JSON.parse(await readFile(job.train_trace_path, "utf8"));
  const evaluations = {};
  for (const author of authors) {
    const trace = JSON.parse(await readFile(path.join(expertDir, `eval-${author}.jsonl`), "utf8"));
    evaluations[author] = {
      accuracy_per_mille: trace.evaluation.accuracy_per_mille,
      mean_probability_error_q15: trace.evaluation.mean_probability_error_q15,
      invalid_forward_count: trace.evaluation.invalid_forward_count,
      logits_hash: trace.evaluation.logits_hash,
    };
  }
  const accuracyValues = authors.map((author) => evaluations[author].accuracy_per_mille);
  const ownAccuracy = evaluations[job.author].accuracy_per_mille;
  const otherAccuracy = Math.trunc(
    authors.filter((author) => author !== job.author)
      .map((author) => evaluations[author].accuracy_per_mille)
      .reduce((sum, value) => sum + value, 0) / 2,
  );
  rows.push({
    expert_id: job.expert_id,
    author: job.author,
    variant: job.variant,
    model_hash: train.final_model_hash,
    updates: train.training.updates,
    train_accuracy_per_mille: train.metrics.final_accuracy_per_mille,
    train_mean_probability_error_q15: Math.trunc(train.metrics.final_probability_error_q15 / train.data.windows),
    attention_delta_l1: train.metrics.attention_delta_l1,
    invalid_forwards: train.metrics.final_invalid_forward_count,
    evaluations,
    mean_cross_author_accuracy_per_mille: Math.trunc(accuracyValues.reduce((sum, value) => sum + value, 0) / accuracyValues.length),
    own_author_advantage_per_mille: ownAccuracy - otherAccuracy,
  });
}

const bestByFinalSplit = {};
for (const author of authors) {
  const ranked = [...rows].sort((left, right) => {
    const leftEval = left.evaluations[author];
    const rightEval = right.evaluations[author];
    return leftEval.mean_probability_error_q15 - rightEval.mean_probability_error_q15 ||
      rightEval.accuracy_per_mille - leftEval.accuracy_per_mille;
  });
  bestByFinalSplit[author] = {
    expert_id: ranked[0].expert_id,
    trained_author: ranked[0].author,
    ...ranked[0].evaluations[author],
  };
}

const bestGeneralist = [...rows].sort((left, right) =>
  right.mean_cross_author_accuracy_per_mille - left.mean_cross_author_accuracy_per_mille,
)[0];
const report = {
  schema: "nsrl.recursive_literary_leaf_comparison.v1",
  experiment_manifest: path.resolve(experimentDir, "experiment.manifest.json"),
  leaf_count: rows.length,
  unique_model_hashes: new Set(rows.map((row) => row.model_hash)).size,
  all_invalid_forward_counts_zero: rows.every((row) => row.invalid_forwards === 0 && authors.every((author) => row.evaluations[author].invalid_forward_count === 0)),
  best_generalist: {
    expert_id: bestGeneralist.expert_id,
    trained_author: bestGeneralist.author,
    mean_cross_author_accuracy_per_mille: bestGeneralist.mean_cross_author_accuracy_per_mille,
  },
  best_by_final_split: bestByFinalSplit,
  leaves: rows,
};
const output = `${JSON.stringify(report, null, 2)}\n`;
if (outPath) await writeFile(outPath, output);
else process.stdout.write(output);
