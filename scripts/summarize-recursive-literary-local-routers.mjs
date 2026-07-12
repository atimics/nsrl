#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let experimentDir = "data/experiments/literary-recursive-swarm-v1";
let outPath = null;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--experiment-dir") experimentDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else throw new Error(`unknown argument: ${arg}`);
}

const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const AUTHORS = ["crowley", "shakespeare", "blake"];
const VIEWS = ["semantic", "structural", "full"];

function parsePredictions(content) {
  const lines = content.trim().split("\n");
  const header = lines.shift()?.split("\t");
  if (header?.[0] !== "sample_id" || header?.[4] !== "probabilities_q15") throw new Error("unexpected router prediction header");
  return lines.map((line) => {
    const [sampleId, target, predicted, top2, probabilities, losses] = line.split("\t");
    return {
      sample_id: sampleId,
      target: Number(target),
      predicted: Number(predicted),
      top2: top2.split(",").map(Number),
      probabilities: probabilities.split(",").map(Number),
      losses: losses.split(",").map(Number),
    };
  });
}

function metrics(rows, predictions, top2Predictions) {
  let correct = 0;
  let top2 = 0;
  let regret = 0;
  const predictedCounts = [0, 0, 0];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const predicted = predictions[index];
    correct += Number(predicted === row.target);
    top2 += Number(top2Predictions[index].includes(row.target));
    predictedCounts[predicted] += 1;
    regret += row.losses[predicted] - Math.min(...row.losses);
  }
  return {
    rows: rows.length,
    accuracy_per_mille: Math.trunc((correct * 1000) / rows.length),
    top2_oracle_coverage_per_mille: Math.trunc((top2 * 1000) / rows.length),
    mean_regret_q15: Math.trunc(regret / rows.length),
    predicted_counts: predictedCounts,
  };
}

function consensusPredictions(baseRows, byView, weights) {
  const predictions = [];
  const top2 = [];
  let disagreementRows = 0;
  for (let rowIndex = 0; rowIndex < baseRows.length; rowIndex += 1) {
    const scores = [0, 0, 0];
    const replicaPredictions = [];
    for (let viewIndex = 0; viewIndex < VIEWS.length; viewIndex += 1) {
      const view = VIEWS[viewIndex];
      const weight = weights[viewIndex];
      const row = byView[view].rows[rowIndex];
      replicaPredictions.push(row.predicted);
      for (let child = 0; child < 3; child += 1) scores[child] += row.probabilities[child] * weight;
      scores[row.top2[0]] += 32767 * weight;
      scores[row.top2[1]] += 16384 * weight;
    }
    if (new Set(replicaPredictions).size > 1) disagreementRows += 1;
    const ranked = [0, 1, 2].sort((left, right) => scores[right] - scores[left] || left - right);
    predictions.push(ranked[0]);
    top2.push(ranked.slice(0, 2));
  }
  return { predictions, top2, disagreementRows };
}

function calibrationWeights(baseRows, byView) {
  let best = null;
  for (let semantic = 1; semantic <= 4; semantic += 1) {
    for (let structural = 1; structural <= 4; structural += 1) {
      for (let full = 1; full <= 4; full += 1) {
        const weights = [semantic, structural, full];
        const consensus = consensusPredictions(baseRows, byView, weights);
        const result = metrics(baseRows, consensus.predictions, consensus.top2);
        const key = [result.mean_regret_q15, 1000 - result.top2_oracle_coverage_per_mille, -result.accuracy_per_mille, weights.reduce((sum, value) => sum + value, 0), semantic, structural, full];
        if (!best || key.some((value, index) => value < best.key[index] && key.slice(0, index).every((prefix, prefixIndex) => prefix === best.key[prefixIndex]))) {
          best = { weights, result, key };
        }
      }
    }
  }
  return best;
}

const report = { schema: "nsrl.recursive_literary_local_router_report.v1", authors: {} };
for (const author of AUTHORS) {
  report.authors[author] = {};
  let selectedWeights = null;
  for (const split of ["calibration", "final"]) {
    const byView = {};
    for (const view of VIEWS) {
      const file = path.join(experimentDir, "routers", `${author}-router-${view}`, `${split}.predictions.tsv`);
      const content = await readFile(file, "utf8");
      byView[view] = { rows: parsePredictions(content), path: path.resolve(file), sha256: sha256(content) };
    }
    const baseRows = byView.semantic.rows;
    for (const view of VIEWS) {
      if (byView[view].rows.length !== baseRows.length || byView[view].rows.some((row, index) => row.sample_id !== baseRows[index].sample_id)) {
        throw new Error(`${author}/${split} router prediction rows do not align`);
      }
    }
    const replicaMetrics = {};
    for (const view of VIEWS) {
      replicaMetrics[view] = metrics(
        baseRows,
        byView[view].rows.map((row) => row.predicted),
        byView[view].rows.map((row) => row.top2),
      );
    }
    const unweighted = consensusPredictions(baseRows, byView, [1, 1, 1]);
    if (split === "calibration") selectedWeights = calibrationWeights(baseRows, byView).weights;
    const calibrated = consensusPredictions(baseRows, byView, selectedWeights);
    report.authors[author][split] = {
      replicas: replicaMetrics,
      unweighted_consensus: metrics(baseRows, unweighted.predictions, unweighted.top2),
      calibrated_consensus_weights: Object.fromEntries(VIEWS.map((view, index) => [view, selectedWeights[index]])),
      calibrated_consensus: metrics(baseRows, calibrated.predictions, calibrated.top2),
      replica_disagreement_rows: calibrated.disagreementRows,
      replica_disagreement_per_mille: Math.trunc((calibrated.disagreementRows * 1000) / baseRows.length),
      prediction_artifacts: Object.fromEntries(VIEWS.map((view) => [view, { path: byView[view].path, sha256: byView[view].sha256 }])),
    };
  }
}

const output = `${JSON.stringify(report, null, 2)}\n`;
if (outPath) await writeFile(outPath, output);
else process.stdout.write(output);
