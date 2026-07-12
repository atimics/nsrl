#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let experimentDir = "data/experiments/literary-recursive-swarm-v1";
let scorer = "target/release/nsrl-mini-transformer-oracle-score";
let routerBinary = "target/release/nsrl-router";
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--experiment-dir") experimentDir = process.argv[++index];
  else if (arg === "--scorer") scorer = process.argv[++index];
  else if (arg === "--router") routerBinary = process.argv[++index];
  else throw new Error(`unknown argument: ${arg}`);
}

const AUTHORS = ["crowley", "shakespeare", "blake"];
const VIEWS = ["semantic", "structural", "full"];
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const manifest = JSON.parse(await readFile(path.join(experimentDir, "experiment.manifest.json"), "utf8"));
const localReport = JSON.parse(await readFile(path.join(experimentDir, "local-router-report.json"), "utf8"));
const outDir = path.join(experimentDir, "root-oracles");
await mkdir(outDir, { recursive: true });

function spawn(binary, args, label) {
  const result = spawnSync(binary, args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${label} failed: ${result.stderr || result.stdout}`);
}

function parseScoreTsv(content) {
  const lines = content.trim().split("\n");
  lines.shift();
  return lines.map((line) => {
    const [sampleId, target, losses, accuracies, childIds, hashes] = line.split("\t");
    return { sample_id: sampleId, target: Number(target), losses: losses.split(",").map(Number), accuracies: accuracies.split(",").map(Number), child_ids: childIds.split(","), model_hashes: hashes.split(",") };
  });
}

function parseRouterPredictions(content) {
  const lines = content.trim().split("\n");
  lines.shift();
  return lines.map((line) => {
    const [sampleId, target, predicted, top2, probabilities] = line.split("\t");
    return { sample_id: sampleId, target: Number(target), predicted: Number(predicted), top2: top2.split(",").map(Number), probabilities: probabilities.split(",").map(Number) };
  });
}

function probeFeatures(losses, accuracies) {
  const maxLoss = Math.max(...losses);
  return [
    ...losses.map((loss) => Math.max(0, Math.min(32767, 65535 - loss))),
    ...accuracies.map((accuracy) => Math.round((accuracy * 32767) / 1000)),
    ...losses.map((loss) => Math.max(0, Math.min(32767, (maxLoss - loss) * 8))),
  ];
}

function consensus(rowsByView, weights) {
  const output = [];
  for (let rowIndex = 0; rowIndex < rowsByView.semantic.length; rowIndex += 1) {
    const scores = [0, 0, 0];
    for (const view of VIEWS) {
      const weight = weights[view];
      const row = rowsByView[view][rowIndex];
      for (let child = 0; child < 3; child += 1) scores[child] += row.probabilities[child] * weight;
      scores[row.top2[0]] += 32767 * weight;
      scores[row.top2[1]] += 16384 * weight;
    }
    const ranked = [0, 1, 2].sort((left, right) => scores[right] - scores[left] || left - right);
    output.push({ predicted: ranked[0], top2: ranked.slice(0, 2) });
  }
  return output;
}

const report = { schema: "nsrl.recursive_literary_root_oracles.v1", experiment_manifest: path.resolve(experimentDir, "experiment.manifest.json"), splits: {} };
for (const split of ["router_train", "router_calibration", "final_test"]) {
  const descriptor = manifest.router_datasets.root[split];
  const sourceContent = await readFile(descriptor.path, "utf8");
  if (sha256(sourceContent) !== descriptor.sha256) throw new Error(`${split} root source hash mismatch`);
  const sourceRows = sourceContent.trim().split("\n").map((line) => JSON.parse(line));
  const scoreInput = path.join(outDir, `${split.replaceAll("_", "-")}.score-input.tsv`);
  await writeFile(scoreInput, `sample_id\tprompt_hex\n${sourceRows.map((row) => `${row.sample_id}\t${Buffer.from(row.prompt, "utf8").toString("hex")}`).join("\n")}\n`);
  const pods = {};

  for (const author of AUTHORS) {
    const jobs = manifest.leaf_jobs.filter((job) => job.author === author).sort((left, right) => left.variant - right.variant);
    const scoresPath = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.scores.tsv`);
    const scoreArgs = ["--input", scoreInput, "--out", scoresPath, "--stride", "8"];
    for (const job of jobs) scoreArgs.push("--model", `${job.expert_id}=${job.model_path}`);
    spawn(scorer, scoreArgs, `${author}/${split} expert scoring`);
    const childScores = parseScoreTsv(await readFile(scoresPath, "utf8"));
    if (childScores.length !== sourceRows.length) throw new Error(`${author}/${split} score row mismatch`);
    const routerTsv = path.join(outDir, `${author}-${split.replaceAll("_", "-")}.router.tsv`);
    const routerRows = sourceRows.map((row, index) => ({
      sample_id: row.sample_id,
      target: childScores[index].target,
      features: [...row.features_q15, ...probeFeatures(childScores[index].losses, childScores[index].accuracies)],
      losses: childScores[index].losses,
    }));
    await writeFile(routerTsv, `sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15\n${routerRows.map((row) => `${row.sample_id}\t${row.target}\t${row.features.join(",")}\t${row.losses.join(",")}`).join("\n")}\n`);
    const predictions = {};
    for (const view of VIEWS) {
      const routerDir = path.join(experimentDir, "routers", `${author}-router-${view}`);
      const predictionsPath = path.join(outDir, `${author}-${view}-${split.replaceAll("_", "-")}.predictions.tsv`);
      const tracePath = path.join(outDir, `${author}-${view}-${split.replaceAll("_", "-")}.eval.jsonl`);
      spawn(routerBinary, ["eval", "--data", routerTsv, "--model", path.join(routerDir, "router.nsrlrt"), "--trace", tracePath, "--predictions-out", predictionsPath], `${author}/${view}/${split} router eval`);
      predictions[view] = parseRouterPredictions(await readFile(predictionsPath, "utf8"));
    }
    const weights = localReport.authors[author].calibration.calibrated_consensus_weights;
    const selected = consensus(predictions, weights);
    pods[author] = sourceRows.map((row, index) => {
      const child = selected[index].predicted;
      return {
        sample_id: row.sample_id,
        selected_child: child,
        top2_children: selected[index].top2,
        selected_loss_q15: childScores[index].losses[child],
        selected_accuracy_per_mille: childScores[index].accuracies[child],
        oracle_child: childScores[index].target,
        oracle_child_loss_q15: Math.min(...childScores[index].losses),
        local_regret_q15: childScores[index].losses[child] - Math.min(...childScores[index].losses),
      };
    });
  }

  const rootRows = sourceRows.map((row, index) => {
    const podLosses = AUTHORS.map((author) => pods[author][index].selected_loss_q15);
    const podAccuracies = AUTHORS.map((author) => pods[author][index].selected_accuracy_per_mille);
    const target = [0, 1, 2].sort((left, right) => podLosses[left] - podLosses[right] || podAccuracies[right] - podAccuracies[left] || left - right)[0];
    return {
      ...row,
      candidate_ids: AUTHORS.map((author) => `author-${author}-pod`),
      bootstrap_target: row.bootstrap_target,
      oracle_target: target,
      oracle_child_losses_q15: podLosses,
      oracle_child_accuracies_per_mille: podAccuracies,
      router_features_q15: [...row.features_q15, ...probeFeatures(podLosses, podAccuracies)],
      selected_pod_children: Object.fromEntries(AUTHORS.map((author) => [author, pods[author][index]])),
    };
  });
  const rootJsonPath = path.join(outDir, `root-${split.replaceAll("_", "-")}.jsonl`);
  const rootTsvPath = path.join(outDir, `root-${split.replaceAll("_", "-")}.router.tsv`);
  const rootJson = `${rootRows.map((row) => JSON.stringify(row)).join("\n")}\n`;
  const rootTsv = `sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15\n${rootRows.map((row) => `${row.sample_id}\t${row.oracle_target}\t${row.router_features_q15.join(",")}\t${row.oracle_child_losses_q15.join(",")}`).join("\n")}\n`;
  await writeFile(rootJsonPath, rootJson);
  await writeFile(rootTsvPath, rootTsv);
  const labelCounts = [0, 0, 0];
  rootRows.forEach((row) => { labelCounts[row.oracle_target] += 1; });
  report.splits[split] = {
    rows: rootRows.length,
    label_counts: labelCounts,
    oracle_path: path.resolve(rootJsonPath),
    oracle_sha256: sha256(rootJson),
    router_tsv_path: path.resolve(rootTsvPath),
    router_tsv_sha256: sha256(rootTsv),
    local_pod_mean_regret_q15: Object.fromEntries(AUTHORS.map((author) => [author, Math.trunc(pods[author].reduce((sum, row) => sum + row.local_regret_q15, 0) / pods[author].length)])),
  };
}

const reportPath = path.join(outDir, "root-oracle-report.json");
await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(reportPath);
