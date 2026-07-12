#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const views = ["hidden-a", "hidden-b", "full"];
const outDir = path.join(root, "data-recursive");
const manifest = {
  schema: "nsrl.literary_recursive_router_datasets.v1",
  root,
  child_views: views,
  feature_layout: {
    child_router_probabilities_q15: { start: 0, end: 9, routers: 3, classes_each: 3 },
    shared_trunk_hidden_q15: { start: 9, end: 41, channels: 32 },
    current_target_excluded: true,
  },
  granularities: {},
};

for (const granularity of ["token", "span"]) {
  const directory = path.join(outDir, granularity);
  fs.mkdirSync(directory, { recursive: true });
  manifest.granularities[granularity] = {};
  for (const split of ["train", "calibration", "final"]) {
    const sourcePath = path.join(root, "data-hidden", granularity, `${split}.tsv`);
    const sourceBytes = fs.readFileSync(sourcePath);
    const source = parseDataset(sourceBytes.toString("utf8"));
    const predictions = views.map((view) => {
      const predictionPath = path.join(
        root,
        "routers",
        `${granularity}-${view}`,
        `${split}.predictions.tsv`,
      );
      return {
        path: predictionPath,
        bytes: fs.readFileSync(predictionPath),
      };
    });
    const predictionMaps = predictions.map(({ bytes }) => parsePredictions(bytes.toString("utf8")));
    const lines = ["sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15"];
    for (const row of source) {
      const child = predictionMaps.map((rows, index) => {
        const prediction = rows.get(row.sampleId);
        if (!prediction) throw new Error(`missing ${views[index]} prediction for ${row.sampleId}`);
        if (prediction.target !== row.target) throw new Error(`target mismatch for ${row.sampleId}`);
        return prediction.probabilities;
      });
      const features = [...child.flat(), ...row.features.slice(0, 32)];
      if (features.length !== 41) throw new Error("recursive feature count mismatch");
      lines.push(
        `${row.sampleId}\t${row.target}\t${features.join(",")}\t${row.losses.join(",")}`,
      );
    }
    const outputPath = path.join(directory, `${split}.tsv`);
    const outputBytes = Buffer.from(`${lines.join("\n")}\n`);
    fs.writeFileSync(outputPath, outputBytes);
    manifest.granularities[granularity][split] = {
      rows: source.length,
      source: binding(sourcePath, sourceBytes),
      child_predictions: Object.fromEntries(
        predictions.map(({ path: predictionPath, bytes }, index) => [
          views[index],
          binding(predictionPath, bytes),
        ]),
      ),
      output: binding(outputPath, outputBytes),
    };
  }
}

const manifestPath = path.join(outDir, "manifest.json");
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(manifestPath);

function parseDataset(content) {
  return content.trimEnd().split("\n").slice(1).map((line) => {
    const [sampleId, target, features, losses] = line.split("\t");
    const row = {
      sampleId,
      target: Number.parseInt(target, 10),
      features: tripletOrVector(features),
      losses: tripletOrVector(losses),
    };
    if (row.features.length !== 41 || row.losses.length !== 3) {
      throw new Error(`invalid source row ${sampleId}`);
    }
    return row;
  });
}

function parsePredictions(content) {
  const rows = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const fields = line.split("\t");
    const probabilities = tripletOrVector(fields[4]);
    if (probabilities.length !== 3) throw new Error("invalid child probabilities");
    rows.set(fields[0], { target: Number.parseInt(fields[1], 10), probabilities });
  }
  return rows;
}

function tripletOrVector(value) {
  return value.split(",").map((item) => {
    const parsed = Number.parseInt(item, 10);
    if (!Number.isInteger(parsed)) throw new Error("invalid integer feature");
    return parsed;
  });
}

function binding(file, bytes) {
  return { path: path.resolve(file), bytes: bytes.length, sha256: sha256(bytes) };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
