#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const granularity = process.argv[3] ?? "token";
if (!["token", "span"].includes(granularity)) throw new Error("granularity must be token or span");

const states = Object.fromEntries(["calibration", "final"].map((split) => [split, {
  details: parseDetails(fs.readFileSync(path.join(root, "oracles", `${split}-details.tsv`), "utf8")),
  mappings: parseMap(fs.readFileSync(
    path.join(root, "data-hidden", granularity, `${split}-map.tsv`),
    "utf8",
  )),
}]));

const candidates = [1, 2, 4, 8].map((epochs) => {
  const directory = path.join(root, "recursive-sweeps", `${granularity}-e${epochs}`);
  const predictions = parsePredictions(
    fs.readFileSync(path.join(directory, "calibration.predictions.tsv"), "utf8"),
  );
  return {
    epochs,
    calibration: evaluate(states.calibration, predictions),
    training_trace: JSON.parse(fs.readFileSync(path.join(directory, "train.trace.jsonl"), "utf8")),
  };
});
candidates.sort((left, right) =>
  left.calibration.probability_error_q15 - right.calibration.probability_error_q15
  || left.calibration.mistakes - right.calibration.mistakes
  || left.calibration.route_switches - right.calibration.route_switches
  || left.epochs - right.epochs
);
const selected = candidates[0];
const selectedDir = path.join(root, "recursive-routers", granularity);
const finalPredictionPath = path.join(selectedDir, "final.predictions.tsv");
const final = fs.existsSync(finalPredictionPath)
  ? evaluate(states.final, parsePredictions(fs.readFileSync(finalPredictionPath, "utf8")))
  : null;
const oracle = JSON.parse(fs.readFileSync(path.join(root, "oracles", "final-report.json"), "utf8"));
const fixed = oracle.fixed_experts[oracle.best_fixed_expert];
const report = {
  schema: "nsrl.literary_recursive_neural_router.v1",
  granularity,
  architecture: {
    child_neural_routers: ["hidden-a", "hidden-b", "full"],
    root_neural_router: "NSRLRT1 41x16x3 integer MLP",
    root_features: "nine child probability channels plus 32 shared-trunk hidden channels",
    current_target_excluded: true,
  },
  selection_split: "router_calibration_only",
  final_split_used_for_selection: false,
  candidates,
  selected_epochs: selected.epochs,
  selected_calibration: selected.calibration,
  final,
  frozen_final_baselines: { fixed, oracle: oracle.oracle_routes[granularity] },
  delta_vs_fixed: final ? {
    probability_error_q15: final.probability_error_q15 - fixed.probability_error_q15,
    mean_probability_error_q15:
      final.mean_probability_error_q15 - fixed.mean_probability_error_q15,
    mistakes: final.mistakes - fixed.mistakes,
    accuracy_per_mille: final.accuracy_per_mille - fixed.accuracy_per_mille,
  } : null,
};
const out = path.join(root, `recursive-${granularity}-router-report.json`);
fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, selected_epochs: selected.epochs, selected_calibration: selected.calibration, final }));

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
  return content.trimEnd().split("\n").slice(1).map((line) => {
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
    rows.set(fields[0], Number.parseInt(fields[2], 10));
  }
  return rows;
}

function evaluate(state, predictions) {
  let windows = 0;
  let mistakes = 0;
  let probabilityError = 0;
  let oracleError = 0;
  let routeCorrect = 0;
  let switches = 0;
  const utilization = [0, 0, 0];
  const previous = new Map();
  for (const mapping of state.mappings) {
    const predicted = predictions.get(mapping.routeId);
    if (![0, 1, 2].includes(predicted)) throw new Error(`missing prediction ${mapping.routeId}`);
    const sample = state.details.get(mapping.sampleId);
    if (!sample) throw new Error(`missing sample ${mapping.sampleId}`);
    const measured = [...sample.entries()]
      .filter(([offset]) => offset >= mapping.start && offset < mapping.end)
      .map(([, row]) => row);
    if (measured.length === 0) throw new Error(`empty mapped span ${mapping.routeId}`);
    const targetLosses = [0, 1, 2].map((expert) =>
      measured.reduce((total, row) => total + row.losses[expert], 0),
    );
    const target = [0, 1, 2].sort((left, right) => targetLosses[left] - targetLosses[right] || left - right)[0];
    routeCorrect += Number(predicted === target);
    if (previous.has(mapping.sampleId) && previous.get(mapping.sampleId) !== predicted) switches += 1;
    previous.set(mapping.sampleId, predicted);
    for (const row of measured) {
      windows += 1;
      mistakes += row.mistakes[predicted];
      probabilityError += row.losses[predicted];
      oracleError += Math.min(...row.losses);
      utilization[predicted] += 1;
    }
  }
  return {
    windows,
    decisions: state.mappings.length,
    route_correct: routeCorrect,
    route_accuracy_per_mille: Math.floor(routeCorrect * 1000 / state.mappings.length),
    mistakes,
    accuracy_per_mille: Math.floor((windows - mistakes) * 1000 / windows),
    probability_error_q15: probabilityError,
    mean_probability_error_q15: Math.floor(probabilityError / windows),
    mean_routing_regret_q15: Math.floor((probabilityError - oracleError) / windows),
    route_switches: switches,
    utilization_tokens: utilization,
    utilization_per_mille: utilization.map((count) => Math.floor(count * 1000 / windows)),
  };
}

function triplet(value) {
  const values = value.split(",").map((item) => Number.parseInt(item, 10));
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item))) {
    throw new Error("invalid triplet");
  }
  return values;
}
