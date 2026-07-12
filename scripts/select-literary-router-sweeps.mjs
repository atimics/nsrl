#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2];
if (!root) throw new Error("usage: select-literary-router-sweeps.mjs ROOT");
const dataDirectory = process.argv[3]
  ?? (fs.existsSync(path.join(root, "data-projected")) ? "data-projected" : "data-hidden");
const views = ["hidden-a", "hidden-b", "full"];
const report = {
  schema: "nsrl.literary_router_sweep_selection.v1",
  selection_split: "router_calibration_only",
  final_split_used_for_selection: false,
  feature_data_directory: dataDirectory,
  granularities: {},
};
for (const granularity of ["token", "span"]) {
  const details = parseDetails(fs.readFileSync(path.join(root, "oracles", "calibration-details.tsv"), "utf8"));
  const mappings = parseMap(fs.readFileSync(
    path.join(root, dataDirectory, granularity, "calibration-map.tsv"),
    "utf8",
  ));
  report.granularities[granularity] = {};
  for (const view of views) {
    const candidates = [1, 2, 4, 8].map((epochs) => {
      const directory = path.join(root, "sweeps", `${granularity}-${view}-e${epochs}`);
      const predictions = parsePredictions(fs.readFileSync(
        path.join(directory, "calibration.predictions.tsv"),
        "utf8",
      ));
      return {
        epochs,
        metrics: evaluate(details, mappings, predictions),
        trace: JSON.parse(fs.readFileSync(path.join(directory, "train.trace.jsonl"), "utf8")),
      };
    });
    candidates.sort((left, right) =>
      left.metrics.probability_error_q15 - right.metrics.probability_error_q15
      || left.metrics.mistakes - right.metrics.mistakes
      || left.metrics.route_switches - right.metrics.route_switches
      || left.epochs - right.epochs
    );
    report.granularities[granularity][view] = {
      selected_epochs: candidates[0].epochs,
      selected_metrics: candidates[0].metrics,
      candidates,
    };
  }
}
const out = path.join(root, "router-sweep-selection.json");
fs.writeFileSync(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, selected: Object.fromEntries(
  Object.entries(report.granularities).map(([granularity, views]) => [
    granularity,
    Object.fromEntries(Object.entries(views).map(([view, value]) => [view, value.selected_epochs])),
  ]),
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

function evaluate(details, mappings, predictions) {
  let windows = 0;
  let mistakes = 0;
  let error = 0;
  let switches = 0;
  const utilization = [0, 0, 0];
  const previous = new Map();
  for (const mapping of mappings) {
    const predicted = predictions.get(mapping.routeId);
    if (![0, 1, 2].includes(predicted)) throw new Error(`missing prediction ${mapping.routeId}`);
    const sample = details.get(mapping.sampleId);
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
    accuracy_per_mille: Math.floor((windows - mistakes) * 1000 / windows),
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
