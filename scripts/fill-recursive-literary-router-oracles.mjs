#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let experimentDir = "data/experiments/literary-recursive-swarm-v1";
let scorer = "target/release/nsrl-mini-transformer-oracle-score";
let reuseScores = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--experiment-dir") experimentDir = process.argv[++index];
  else if (arg === "--scorer") scorer = process.argv[++index];
  else if (arg === "--reuse-scores") reuseScores = true;
  else if (arg === "--help" || arg === "-h") {
    console.log("Usage: node scripts/fill-recursive-literary-router-oracles.mjs [--experiment-dir PATH] [--scorer PATH]");
    process.exit(0);
  } else throw new Error(`unknown argument: ${arg}`);
}

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const manifest = JSON.parse(await readFile(path.join(experimentDir, "experiment.manifest.json"), "utf8"));
const outDir = path.join(experimentDir, "router-oracles");
await mkdir(outDir, { recursive: true });
const report = {
  schema: "nsrl.recursive_literary_local_router_oracles.v1",
  experiment_manifest: path.resolve(experimentDir, "experiment.manifest.json"),
  scorer: path.resolve(scorer),
  authors: {},
};

function parseScoreTsv(content) {
  const lines = content.trim().split("\n");
  const header = lines.shift()?.split("\t");
  const expected = ["sample_id", "oracle_target", "child_mean_probability_error_q15", "child_accuracy_per_mille", "child_ids", "child_model_hashes"];
  if (JSON.stringify(header) !== JSON.stringify(expected)) throw new Error("oracle scorer returned an unexpected TSV header");
  const rows = new Map();
  for (const line of lines) {
    const [sampleId, target, losses, accuracies, childIds, modelHashes] = line.split("\t");
    rows.set(sampleId, {
      oracle_target: Number.parseInt(target, 10),
      oracle_child_losses_q15: losses.split(",").map(Number),
      oracle_child_accuracies_per_mille: accuracies.split(",").map(Number),
      oracle_child_ids: childIds.split(","),
      oracle_child_model_hashes: modelHashes.split(","),
    });
  }
  return rows;
}

for (const author of manifest.authors) {
  const jobs = manifest.leaf_jobs
    .filter((job) => job.author === author)
    .sort((left, right) => left.variant - right.variant);
  if (jobs.length !== 3 || jobs.some((job, index) => job.variant !== index)) {
    throw new Error(`${author} does not have variants 0, 1, 2`);
  }
  report.authors[author] = {};
  for (const split of ["router_train", "router_calibration", "final_test"]) {
    const descriptor = manifest.router_datasets.local[author][split];
    const sourceContent = await readFile(descriptor.path, "utf8");
    if (sha256(sourceContent) !== descriptor.sha256) throw new Error(`${author}/${split} source hash mismatch`);
    const rows = sourceContent.trim().split("\n").map((line) => JSON.parse(line));
    const scoreInput = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.score-input.tsv`);
    const scoreOutput = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.scores.tsv`);
    const oracleOutput = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.jsonl`);
    const routerTsvOutput = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.router.tsv`);
    const scoreInputContent = `sample_id\tprompt_hex\n${rows.map((row) => `${row.sample_id}\t${Buffer.from(row.prompt, "utf8").toString("hex")}`).join("\n")}\n`;
    await writeFile(scoreInput, scoreInputContent);

    const args = ["--input", scoreInput, "--out", scoreOutput, "--stride", "8"];
    for (const job of jobs) args.push("--model", `${job.expert_id}=${job.model_path}`);
    if (!reuseScores) {
      const result = spawnSync(scorer, args, { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
      if (result.status !== 0) {
        throw new Error(`oracle scorer failed for ${author}/${split}: ${result.stderr || result.stdout}`);
      }
    }
    const scores = parseScoreTsv(await readFile(scoreOutput, "utf8"));
    const labelCounts = [0, 0, 0];
    let marginSum = 0;
    let tieCount = 0;
    const filled = rows.map((row) => {
      const score = scores.get(row.sample_id);
      if (!score) throw new Error(`missing score for ${row.sample_id}`);
      if (JSON.stringify(row.candidate_ids) !== JSON.stringify(score.oracle_child_ids)) {
        throw new Error(`candidate order mismatch for ${row.sample_id}`);
      }
      labelCounts[score.oracle_target] += 1;
      const orderedLosses = [...score.oracle_child_losses_q15].sort((a, b) => a - b);
      const margin = orderedLosses[1] - orderedLosses[0];
      marginSum += margin;
      if (margin === 0) tieCount += 1;
      const maxLoss = Math.max(...score.oracle_child_losses_q15);
      const probeFeaturesQ15 = [
        ...score.oracle_child_losses_q15.map((loss) => Math.max(0, Math.min(32767, 65535 - loss))),
        ...score.oracle_child_accuracies_per_mille.map((accuracy) => Math.round((accuracy * 32767) / 1000)),
        ...score.oracle_child_losses_q15.map((loss) => Math.max(0, Math.min(32767, (maxLoss - loss) * 8))),
      ];
      return { ...row, ...score, router_features_q15: [...row.features_q15, ...probeFeaturesQ15] };
    });
    if (scores.size !== filled.length) throw new Error(`${author}/${split} score count mismatch`);
    const oracleContent = `${filled.map((row) => JSON.stringify(row)).join("\n")}\n`;
    await writeFile(oracleOutput, oracleContent);
    const routerTsvContent = `sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15\n${filled.map((row) => `${row.sample_id}\t${row.oracle_target}\t${row.router_features_q15.join(",")}\t${row.oracle_child_losses_q15.join(",")}`).join("\n")}\n`;
    await writeFile(routerTsvOutput, routerTsvContent);
    report.authors[author][split] = {
      source_path: descriptor.path,
      source_sha256: descriptor.sha256,
      score_input_path: path.resolve(scoreInput),
      score_input_sha256: sha256(scoreInputContent),
      scores_path: path.resolve(scoreOutput),
      scores_sha256: sha256(await readFile(scoreOutput)),
      oracle_path: path.resolve(oracleOutput),
      oracle_sha256: sha256(oracleContent),
      router_tsv_path: path.resolve(routerTsvOutput),
      router_tsv_sha256: sha256(routerTsvContent),
      rows: filled.length,
      label_counts: labelCounts,
      mean_oracle_margin_q15: Math.trunc(marginSum / Math.max(1, filled.length)),
      tie_count: tieCount,
      scorer_stride: 8,
      candidate_ids: jobs.map((job) => job.expert_id),
    };
  }
}

const reportPath = path.join(outDir, "oracle-report.json");
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(reportPath);
