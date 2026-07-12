#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2]
  ?? "data/experiments/literary-h8-gradient-block-curriculum-v1";
const dataDirectory = process.argv[3]
  ?? (fs.existsSync(path.join(root, "data-full-hidden"))
    ? "data-full-hidden"
    : fs.existsSync(path.join(root, "data-projected"))
      ? "data-projected"
      : "data-hidden");
const sweepDirectoryName = process.argv[4] ?? "regret-sweeps";
const sweepDir = path.join(root, sweepDirectoryName);
const candidates = fs.readdirSync(sweepDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => {
    const match = /^(?:(token|span)-(hidden-a|hidden-b|full)|recursive-(token|span))-s(\d+)-e(\d+)(?:-r(\d+))?$/.exec(entry.name);
    return match ? {
      directory: entry.name,
      granularity: match[1] ?? match[3],
      view: match[2] ?? "recursive",
      regret_gradient_shift: Number.parseInt(match[4], 10),
      epochs: Number.parseInt(match[5], 10),
      max_train_rows: match[6] ? Number.parseInt(match[6], 10) : null,
    } : null;
  })
  .filter(Boolean);
if (candidates.length === 0) throw new Error("no regret sweep candidates found");

const availableSplits = ["calibration", "final"].filter((split) =>
  fs.existsSync(path.join(root, "oracles", `${split}-details.tsv`))
  && fs.existsSync(path.join(root, dataDirectory, "token", `${split}-map.tsv`))
  && fs.existsSync(path.join(root, dataDirectory, "span", `${split}-map.tsv`)));
const states = Object.fromEntries(availableSplits.map((split) => [split,
  Object.fromEntries(["token", "span"].map((granularity) => [granularity, {
    details: parseDetails(fs.readFileSync(
      path.join(root, "oracles", `${split}-details.tsv`),
      "utf8",
    )),
    mappings: parseMap(fs.readFileSync(
      path.join(root, dataDirectory, granularity, `${split}-map.tsv`),
      "utf8",
    )),
  }])),
]));

for (const candidate of candidates) {
  const directory = path.join(sweepDir, candidate.directory);
  candidate.metrics = evaluate(
    states.calibration[candidate.granularity],
    parsePredictions(fs.readFileSync(
      path.join(directory, "calibration.predictions.tsv"),
      "utf8",
    )),
  );
  candidate.training_trace = JSON.parse(fs.readFileSync(
    path.join(directory, "train.trace.jsonl"),
    "utf8",
  ));
}

const groups = {};
for (const candidate of candidates) {
  const key = `${candidate.granularity}-${candidate.view}`;
  (groups[key] ??= []).push(candidate);
}
const selected = {};
for (const [key, values] of Object.entries(groups)) {
  values.sort((left, right) =>
    left.metrics.probability_error_q15 - right.metrics.probability_error_q15
    || left.metrics.mistakes - right.metrics.mistakes
    || left.metrics.route_switches - right.metrics.route_switches
    || left.regret_gradient_shift - right.regret_gradient_shift
    || left.epochs - right.epochs
    || (left.max_train_rows ?? Number.MAX_SAFE_INTEGER)
      - (right.max_train_rows ?? Number.MAX_SAFE_INTEGER));
  selected[key] = values[0];
  const finalPredictionPath = path.join(
    sweepDir,
    values[0].directory,
    "final.predictions.tsv",
  );
  if (states.final && fs.existsSync(finalPredictionPath)) {
    selected[key].final_metrics = evaluate(
      states.final[values[0].granularity],
      parsePredictions(fs.readFileSync(finalPredictionPath, "utf8")),
    );
  }
}

const report = {
  schema: "nsrl.literary_expected_regret_router_sweep.v1",
  objective: "expected_regret",
  feature_data_directory: dataDirectory,
  sweep_directory: sweepDirectoryName,
  selection_split: "router_calibration_only",
  final_split_used_for_selection: false,
  selected,
  candidates,
};
const output = path.join(root, "regret-router-sweep-selection.json");
fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output, selected: Object.fromEntries(
  Object.entries(selected).map(([key, value]) => [key, {
    shift: value.regret_gradient_shift,
    epochs: value.epochs,
    max_train_rows: value.max_train_rows,
    metrics: value.metrics,
    final_metrics: value.final_metrics ?? null,
  }]),
) }));

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
    return { routeId, sampleId, start: Number(start), end: Number(end) };
  });
}

function parsePredictions(content) {
  return new Map(content.trimEnd().split("\n").slice(1).map((line) => {
    const fields = line.split("\t");
    return [fields[0], Number.parseInt(fields[2], 10)];
  }));
}

function evaluate(state, predictions) {
  let windows = 0;
  let mistakes = 0;
  let error = 0;
  let switches = 0;
  const utilization = [0, 0, 0];
  const previous = new Map();
  for (const mapping of state.mappings) {
    const predicted = predictions.get(mapping.routeId);
    if (![0, 1, 2].includes(predicted)) throw new Error(`missing prediction ${mapping.routeId}`);
    const sample = state.details.get(mapping.sampleId);
    const measured = [...sample.entries()]
      .filter(([offset]) => offset >= mapping.start && offset < mapping.end)
      .map(([, row]) => row);
    if (measured.length === 0) throw new Error(`empty mapping ${mapping.routeId}`);
    if (previous.has(mapping.sampleId) && previous.get(mapping.sampleId) !== predicted) switches += 1;
    previous.set(mapping.sampleId, predicted);
    for (const row of measured) {
      windows += 1;
      mistakes += row.mistakes[predicted];
      error += row.losses[predicted];
      utilization[predicted] += 1;
    }
  }
  return {
    windows,
    mistakes,
    accuracy_per_mille: Math.floor((windows - mistakes) * 1_000 / windows),
    probability_error_q15: error,
    mean_probability_error_q15: Math.floor(error / windows),
    route_switches: switches,
    utilization_tokens: utilization,
  };
}

function triplet(value) {
  const values = value.split(",").map(Number);
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item))) {
    throw new Error("invalid triplet");
  }
  return values;
}
