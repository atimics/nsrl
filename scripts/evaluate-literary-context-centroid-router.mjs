#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-context-block-swarm-v1";
const sourceRoot = process.argv[3] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const split = process.argv[4] ?? "final-test";
const spanLength = Number.parseInt(process.argv[5] ?? "16", 10);
if (spanLength < 1) throw new Error("span length must be positive");

const clusterManifest = JSON.parse(fs.readFileSync(path.join(root, "clusters", "manifest.json")));
const centroids = clusterManifest.clusters.map((cluster) => cluster.centroid_q15);
const prompts = parseScoreInput(fs.readFileSync(
  path.join(sourceRoot, "shards", `${split}.score-input.tsv`),
  "utf8",
));
const details = parseDetails(fs.readFileSync(
  path.join(root, "oracles", `${split}-details.tsv`),
  "utf8",
));

const token = aggregate();
const span = aggregate();
for (const [sampleId, rows] of details) {
  const prompt = prompts.get(sampleId);
  if (!prompt) throw new Error(`missing prompt ${sampleId}`);
  let previousToken = null;
  for (const row of rows) {
    const context = prompt.subarray(Math.max(0, row.offset - 512), row.offset);
    const choice = closest(contextFeaturesQ15(context), centroids);
    add(token, row, choice);
    if (previousToken !== null && previousToken !== choice) token.route_switches += 1;
    previousToken = choice;
    token.decisions += 1;
  }
  let previousSpan = null;
  for (let start = 0; start < rows.length; start += spanLength) {
    const group = rows.slice(start, start + spanLength);
    const first = group[0];
    const context = prompt.subarray(Math.max(0, first.offset - 512), first.offset);
    const choice = closest(contextFeaturesQ15(context), centroids);
    for (const row of group) add(span, row, choice);
    if (previousSpan !== null && previousSpan !== choice) span.route_switches += 1;
    previousSpan = choice;
    span.decisions += 1;
  }
}

const oracle = JSON.parse(fs.readFileSync(path.join(root, "oracles", `${split}.json`)));
const fixed = oracle.fixed_experts[oracle.best_fixed_expert];
const report = {
  schema: "nsrl.literary_context_centroid_router.v1",
  split,
  span_length: spanLength,
  target_blind: true,
  feature_context_tokens: 512,
  fixed,
  oracle: oracle.oracle_routes,
  routes: {
    token: finalize(token),
    span: finalize(span),
  },
};
for (const route of Object.values(report.routes)) {
  route.delta_vs_fixed = {
    probability_error_q15: route.probability_error_q15 - fixed.probability_error_q15,
    mistakes: route.mistakes - fixed.mistakes,
  };
}
const output = path.join(root, "oracles", `${split}-centroid-router.json`);
fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output, routes: report.routes }));

function aggregate() {
  return {
    windows: 0,
    decisions: 0,
    mistakes: 0,
    probability_error_q15: 0,
    route_switches: 0,
    utilization_tokens: [0, 0, 0],
  };
}

function add(result, row, choice) {
  result.windows += 1;
  result.mistakes += row.mistakes[choice];
  result.probability_error_q15 += row.losses[choice];
  result.utilization_tokens[choice] += 1;
}

function finalize(result) {
  return {
    ...result,
    accuracy_per_mille: Math.floor((result.windows - result.mistakes) * 1000 / result.windows),
    mean_probability_error_q15: Math.floor(result.probability_error_q15 / result.windows),
    utilization_per_mille: result.utilization_tokens.map((count) =>
      Math.floor(count * 1000 / result.windows)),
  };
}

function parseScoreInput(content) {
  const rows = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const [sampleId, promptHex] = line.split("\t");
    rows.set(sampleId, Buffer.from(promptHex, "hex"));
  }
  return rows;
}

function parseDetails(content) {
  const groups = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const fields = line.split("\t");
    const group = groups.get(fields[0]) ?? [];
    group.push({
      offset: Number.parseInt(fields[1], 10),
      losses: triplet(fields[3]),
      mistakes: triplet(fields[4]),
    });
    groups.set(fields[0], group);
  }
  for (const rows of groups.values()) rows.sort((left, right) => left.offset - right.offset);
  return groups;
}

function contextFeaturesQ15(bytes) {
  const buckets = Array(24).fill(0);
  for (let index = 1; index < bytes.length; index += 1) {
    buckets[((bytes[index - 1] * 257) + bytes[index]) % buckets.length] += 1;
  }
  const normalize = (count, total) => Math.max(
    0,
    Math.min(32767, Math.round(count * 32767 / Math.max(1, total))),
  );
  const predicates = [
    (byte) => byte >= 97 && byte <= 122,
    (byte) => byte >= 48 && byte <= 57,
    (byte) => byte === 32 || byte === 9,
    (byte) => byte === 10,
    (byte) => ",.;:!?".includes(String.fromCharCode(byte)),
    (byte) => "aeiouy".includes(String.fromCharCode(byte)),
    (byte) => byte === 39 || byte === 34,
    (byte) => byte < 32 || byte > 126,
  ];
  return [
    ...buckets.map((count) => normalize(count, bytes.length - 1)),
    ...predicates.map((predicate) => normalize([...bytes].filter(predicate).length, bytes.length)),
  ];
}

function closest(features, centroids) {
  return centroids.map((centroid, index) => ({
    index,
    distance: features.reduce((total, value, feature) => {
      const difference = value - centroid[feature];
      return total + difference * difference;
    }, 0),
  })).sort((left, right) => left.distance - right.distance || left.index - right.index)[0].index;
}

function triplet(value) {
  const values = value.split(",").map((item) => Number.parseInt(item, 10));
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item))) {
    throw new Error("invalid triplet");
  }
  return values;
}
