#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const runs = [];
let outPath = null;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--run") {
    const value = process.argv[++index] ?? "";
    const separator = value.indexOf("=");
    if (separator < 1) throw new Error("--run requires LABEL=DIR");
    runs.push({ label: value.slice(0, separator), dir: value.slice(separator + 1) });
  } else if (arg === "--out") {
    outPath = process.argv[++index];
  } else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/summarize-literary-runs.mjs --run LABEL=DIR [--run ...] [--out PATH]");
    process.exit(0);
  } else {
    throw new Error(`unknown argument: ${arg}`);
  }
}
if (runs.length === 0) throw new Error("at least one --run is required");

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function optionalJson(file) {
  try {
    return await readJson(file);
  } catch (error) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

const rows = [];
for (const run of runs) {
  const dir = path.resolve(run.dir);
  const corpus = await optionalJson(path.join(dir, "corpus.manifest.json"));
  const train = await optionalJson(path.join(dir, "train.trace.jsonl"));
  const heldout = await optionalJson(path.join(dir, "holdout.eval.jsonl"));
  const progress = await optionalJson(path.join(dir, "progress.jsonl"));
  const active = train ?? progress;
  rows.push({
    label: run.label,
    directory: dir,
    status: train && heldout ? "complete" : progress ? "failed_during_training" : "incomplete",
    corpus_sha256: corpus?.corpus?.sha256 ?? null,
    holdout_sha256: corpus?.holdout?.sha256 ?? null,
    balanced_bytes_per_author: corpus?.bytes_per_author_limit ?? null,
    holdout_bytes_per_author: corpus?.holdout_bytes_per_author_limit ?? null,
    seq_len: active?.training?.seq_len ?? active?.model?.seq_len ?? null,
    stride: active?.training?.stride ?? null,
    windows: train?.data?.windows ?? progress?.data?.windows ?? null,
    completed_updates: train?.training?.updates ?? progress?.training?.updates ?? 0,
    adaptive_shift_updates: train?.metrics?.adaptive_rule_update_count ??
      progress?.metrics?.adaptive_rule_shift_adjustment_count ?? 0,
    train_accuracy_per_mille: train?.metrics?.final_accuracy_per_mille ?? null,
    train_mean_probability_error_q15: train
      ? Math.trunc(train.metrics.final_probability_error_q15 / train.data.windows)
      : null,
    heldout_accuracy_per_mille: heldout?.evaluation?.accuracy_per_mille ?? null,
    heldout_mean_probability_error_q15: heldout?.evaluation?.mean_probability_error_q15 ?? null,
    invalid_forwards: heldout?.evaluation?.invalid_forward_count ??
      train?.metrics?.final_invalid_forward_count ?? null,
    attention_delta_l1: train?.metrics?.attention_delta_l1 ??
      progress?.metrics?.attention_delta_l1 ?? null,
    model_hash: train?.final_model_hash ?? progress?.model_hash ?? null,
    final_attention_qk_learning_rate_shift:
      train?.metrics?.final_attention_qk_learning_rate_shift ??
      progress?.metrics?.current_attention_qk_learning_rate_shift ?? null,
  });
}

const report = {
  schema: "nsrl.literary_scale_comparison.v1",
  generated_at: new Date().toISOString(),
  runs: rows,
};
const output = `${JSON.stringify(report, null, 2)}\n`;
if (outPath) await writeFile(outPath, output);
else process.stdout.write(output);
