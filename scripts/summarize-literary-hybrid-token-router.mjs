#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-shared-trunk-hybrid-moe-v1";
const out = process.argv[3] ?? path.join(root, "learned-token-router-report.json");
const schema = process.argv[4] ?? "nsrl.literary_learned_token_consensus.v1";
const granularity = process.argv[5] ?? "token";
if (!["token", "span"].includes(granularity)) throw new Error("granularity must be token or span");
const views = [`${granularity}-hidden-a`, `${granularity}-hidden-b`, `${granularity}-full`];
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

function triplet(value) {
  const values = value.split(",").map((item) => Number.parseInt(item, 10));
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item))) {
    throw new Error("invalid triplet");
  }
  return values;
}

function parseDetails(content) {
  const groups = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const fields = line.split("\t");
    const group = groups.get(fields[0]) ?? new Map();
    group.set(Number.parseInt(fields[1], 10), {
      losses: triplet(fields[3]),
      mistakes: triplet(fields[4]),
    });
    groups.set(fields[0], group);
  }
  return groups;
}

function parseMap(content) {
  return content
    .trimEnd()
    .split("\n")
    .slice(1)
    .map((line) => {
      const [routeId, sampleId, start, end] = line.split("\t");
      return {
        routeId,
        sampleId,
        start: Number.parseInt(start, 10),
        end: Number.parseInt(end, 10),
      };
    });
}

function parsePredictions(content) {
  const rows = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const fields = line.split("\t");
    rows.set(fields[0], {
      target: Number.parseInt(fields[1], 10),
      predicted: Number.parseInt(fields[2], 10),
      probabilities: triplet(fields[4]),
    });
  }
  return rows;
}

function rawConsensus(predictions, weights, routeId) {
  const rows = predictions.map((view) => view.get(routeId));
  if (rows.some((row) => !row)) throw new Error(`missing prediction for ${routeId}`);
  const scores = [0, 1, 2].map((expert) =>
    rows.reduce((total, row, index) => total + weights[index] * row.probabilities[expert], 0),
  );
  return {
    predicted: [0, 1, 2].sort((left, right) => scores[right] - scores[left] || left - right)[0],
    target: rows[0].target,
    scores,
  };
}

function consensus(state, weights, margin) {
  const output = new Map();
  const previous = new Map();
  for (const mapping of state.mappings) {
    const row = rawConsensus(state.predictions, weights, mapping.routeId);
    const prior = previous.get(mapping.sampleId);
    let predicted = row.predicted;
    if (prior !== undefined && row.scores[predicted] - row.scores[prior] < margin) {
      predicted = prior;
    }
    previous.set(mapping.sampleId, predicted);
    output.set(mapping.routeId, { predicted, target: row.target });
  }
  return output;
}

function evaluate(state, predictionForRoute) {
  let windows = 0;
  let mistakes = 0;
  let error = 0;
  let oracleError = 0;
  let correct = 0;
  let switches = 0;
  const utilization = [0, 0, 0];
  const previous = new Map();
  for (const mapping of state.mappings) {
    const { predicted, target } = predictionForRoute(mapping.routeId);
    correct += Number(predicted === target);
    if (previous.has(mapping.sampleId) && previous.get(mapping.sampleId) !== predicted) switches += 1;
    previous.set(mapping.sampleId, predicted);
    const sample = state.details.get(mapping.sampleId);
    if (!sample) throw new Error(`missing details for ${mapping.sampleId}`);
    const measuredRows = [...sample.entries()]
      .filter(([offset]) => offset >= mapping.start && offset < mapping.end)
      .map(([, row]) => row);
    if (measuredRows.length === 0) {
      throw new Error(`span ${mapping.routeId} contains no measured targets`);
    }
    for (const row of measuredRows) {
      windows += 1;
      mistakes += row.mistakes[predicted];
      error += row.losses[predicted];
      oracleError += Math.min(...row.losses);
      utilization[predicted] += 1;
    }
  }
  const decisions = state.mappings.length;
  return {
    windows,
    route_correct: correct,
    route_accuracy_per_mille: Math.floor((correct * 1000) / decisions),
    mistakes,
    accuracy_per_mille: Math.floor(((windows - mistakes) * 1000) / windows),
    probability_error_q15: error,
    mean_probability_error_q15: Math.floor(error / windows),
    mean_routing_regret_q15: Math.floor((error - oracleError) / windows),
    route_switches: switches,
    utilization_tokens: utilization,
    utilization_per_mille: utilization.map((count) => Math.floor((count * 1000) / windows)),
  };
}

function better(left, right) {
  return left.probability_error_q15 < right.probability_error_q15 ||
    (left.probability_error_q15 === right.probability_error_q15 &&
      (left.accuracy_per_mille > right.accuracy_per_mille ||
        (left.accuracy_per_mille === right.accuracy_per_mille &&
          left.route_switches < right.route_switches)));
}

const states = {};
const sourceSha256 = {};
for (const split of ["calibration", "final"]) {
  const detailBytes = await readFile(path.join(root, "oracles", `${split}-details.tsv`));
  const mapBytes = await readFile(path.join(root, "data-hidden", granularity, `${split}-map.tsv`));
  const predictionBytes = await Promise.all(
    views.map((view) => readFile(path.join(root, "routers", view, `${split}.predictions.tsv`))),
  );
  states[split] = {
    details: parseDetails(detailBytes.toString("utf8")),
    mappings: parseMap(mapBytes.toString("utf8")),
    predictions: predictionBytes.map((bytes) => parsePredictions(bytes.toString("utf8"))),
  };
  sourceSha256[split] = {
    details: sha256(detailBytes),
    mappings: sha256(mapBytes),
    predictions: Object.fromEntries(views.map((view, index) => [view, sha256(predictionBytes[index])])),
  };
}

const replicas = {};
for (let index = 0; index < views.length; index += 1) {
  replicas[views[index]] = {};
  for (const split of ["calibration", "final"]) {
    const state = states[split];
    replicas[views[index]][split] = evaluate(state, (routeId) => state.predictions[index].get(routeId));
  }
}

let selected = null;
for (let a = 1; a <= 3; a += 1) {
  for (let b = 1; b <= 3; b += 1) {
    for (let c = 1; c <= 3; c += 1) {
      const weights = [a, b, c];
      for (const margin of [0, 256, 512, 1024, 2048, 4096, 8192]) {
        const predictions = consensus(states.calibration, weights, margin);
        const metrics = evaluate(states.calibration, (routeId) => predictions.get(routeId));
        if (selected === null || better(metrics, selected.calibration)) {
          selected = { weights, hysteresis_margin_q15: margin, calibration: metrics };
        }
      }
    }
  }
}
const finalPredictions = consensus(states.final, selected.weights, selected.hysteresis_margin_q15);
selected.final = evaluate(states.final, (routeId) => finalPredictions.get(routeId));
const finalOracle = JSON.parse(await readFile(path.join(root, "oracles", "final-report.json"), "utf8"));
const fixed = finalOracle.fixed_experts[finalOracle.best_fixed_expert];
const report = {
  schema,
  granularity,
  selection_split: "router_calibration_only",
  final_split_used_for_selection: false,
  views,
  expert_ids: finalOracle.models?.ids ?? finalOracle.experts?.ids ?? ["expert-0", "expert-1", "expert-2"],
  replicas,
  selected_consensus: selected,
  frozen_final_baselines: { fixed, oracle: finalOracle.oracle_routes[granularity] },
  delta_vs_fixed: {
    accuracy_per_mille: selected.final.accuracy_per_mille - fixed.accuracy_per_mille,
    mistakes: selected.final.mistakes - fixed.mistakes,
    probability_error_q15: selected.final.probability_error_q15 - fixed.probability_error_q15,
    mean_probability_error_q15:
      selected.final.mean_probability_error_q15 - fixed.mean_probability_error_q15,
  },
  source_sha256: sourceSha256,
  known_non_claims: [
    "router_inputs_exclude_current_target",
    "rolling_probe_features_use_prior_observed_tokens",
    "does_not_claim_language_model_quality",
  ],
};
await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, selected, delta_vs_fixed: report.delta_vs_fixed }));
