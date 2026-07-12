#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const scorePath = process.argv[2]
  ?? "data/experiments/literary-h8-context-block-swarm-v1/clusters/leaf-spans.score-input.tsv";
const signaturePath = process.argv[3]
  ?? "data/experiments/literary-h8-gradient-block-swarm-v1/signatures/leaf.tsv";
const outDir = process.argv[4]
  ?? "data/experiments/literary-h8-gradient-block-swarm-v1/clusters";
const clusterCount = Number.parseInt(process.argv[5] ?? "3", 10);
const iterations = Number.parseInt(process.argv[6] ?? "20", 10);
if (clusterCount < 2 || iterations < 1) throw new Error("invalid clustering bounds");

const scoreBytes = fs.readFileSync(scorePath);
const signatureBytes = fs.readFileSync(signaturePath);
const prompts = parseScoreInput(scoreBytes.toString("utf8"));
const signatures = parseSignatures(signatureBytes.toString("utf8"));
if (prompts.size !== signatures.size) throw new Error("score/signature sample count mismatch");

const points = [];
for (const [id, tokens] of prompts) {
  const signature = signatures.get(id);
  if (!signature) throw new Error(`missing signature ${id}`);
  const [author, range] = id.split("@");
  const [start, end] = range.split("-").map((value) => Number.parseInt(value, 10));
  points.push({ id, author, start, end, tokens, ...signature, cluster: -1 });
}
const channels = points[0].signature_q15.length;
const normalization = Array.from({ length: channels }, (_, channel) => {
  const mean = points.reduce((total, point) => total + point.signature_q15[channel], 0)
    / points.length;
  const variance = points.reduce((total, point) => {
    const difference = point.signature_q15[channel] - mean;
    return total + difference * difference;
  }, 0) / points.length;
  return { mean, standard_deviation: Math.sqrt(variance) };
});
for (const point of points) {
  point.normalized_q12 = point.signature_q15.map((value, channel) => {
    const stats = normalization[channel];
    if (stats.standard_deviation === 0) return 0;
    return Math.max(-32768, Math.min(32767, Math.round(
      (value - stats.mean) * 4096 / stats.standard_deviation,
    )));
  });
}

let centroids = initialCentroids(points, clusterCount);
let changed = 0;
let completedIterations = 0;
for (let iteration = 0; iteration < iterations; iteration += 1) {
  changed = 0;
  for (const point of points) {
    const next = closest(point.normalized_q12, centroids);
    if (point.cluster !== next) changed += 1;
    point.cluster = next;
  }
  centroids = recompute(points, clusterCount);
  completedIterations = iteration + 1;
  if (changed === 0) break;
}

fs.mkdirSync(outDir, { recursive: true });
const clusters = [];
for (let cluster = 0; cluster < clusterCount; cluster += 1) {
  const members = points.filter((point) => point.cluster === cluster);
  if (members.length === 0) throw new Error(`empty gradient cluster ${cluster}`);
  const parts = [];
  for (const member of members) {
    if (parts.length > 0) parts.push(Buffer.from(" "));
    parts.push(member.tokens);
  }
  const tokens = Buffer.concat(parts);
  const tokensPath = path.join(outDir, `cluster-${cluster}.tokens.u8`);
  const spansPath = path.join(outDir, `cluster-${cluster}.spans.jsonl`);
  fs.writeFileSync(tokensPath, tokens);
  fs.writeFileSync(
    spansPath,
    `${members.map((point) => JSON.stringify({
      id: point.id,
      author: point.author,
      source_start: point.start,
      source_end: point.end,
      tokens_sha256: sha256(point.tokens),
      windows: point.windows,
      mistakes: point.mistakes,
      mean_probability_error_q15: point.mean_error,
      hidden_gradient_signature_q15: point.signature_q15,
      normalized_signature_q12: point.normalized_q12,
      squared_distance: squaredDistance(point.normalized_q12, centroids[cluster]),
    })).join("\n")}\n`,
  );
  clusters.push({
    id: cluster,
    spans: members.length,
    token_bytes: tokens.length,
    author_spans: Object.fromEntries(["crowley", "shakespeare", "blake"].map((author) => [
      author,
      members.filter((point) => point.author === author).length,
    ])),
    mean_probability_error_q15: Math.floor(
      members.reduce((total, point) => total + point.mean_error, 0) / members.length,
    ),
    mean_mistakes_per_span_milli: Math.floor(
      members.reduce((total, point) => total + point.mistakes, 0) * 1000 / members.length,
    ),
    centroid_normalized_q12: centroids[cluster],
    tokens: binding(tokensPath, tokens),
    provenance: binding(spansPath, fs.readFileSync(spansPath)),
  });
}

const minimumSpans = Math.min(...clusters.map((cluster) => cluster.spans));
const manifest = {
  schema: "nsrl.literary_hidden_gradient_cluster_shards.v1",
  source: {
    span_score_input: binding(scorePath, scoreBytes),
    hidden_gradient_signatures: binding(signaturePath, signatureBytes),
  },
  policy: {
    cluster_count: clusterCount,
    algorithm: "deterministic_farthest_seed_kmeans_standardized_hidden_gradient",
    signature_channels: channels,
    signed_channels: 16,
    magnitude_channels: 16,
    normalization_scale_q: 12,
    training_labels_use_current_targets: true,
    inference_router_must_be_target_blind: true,
    cross_author: true,
    overlapping_spans: false,
    iterations_requested: iterations,
    iterations_completed: completedIterations,
    final_assignment_changes: changed,
  },
  normalization,
  spans: points.length,
  minimum_cluster_spans: minimumSpans,
  clusters,
};
const manifestPath = path.join(outDir, "manifest.json");
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(manifestPath);

function parseScoreInput(content) {
  const rows = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const [id, promptHex] = line.split("\t");
    rows.set(id, Buffer.from(promptHex, "hex"));
  }
  return rows;
}

function parseSignatures(content) {
  const rows = new Map();
  for (const line of content.trimEnd().split("\n").slice(1)) {
    const fields = line.split("\t");
    const signature = fields[5].split(",").map((value) => Number.parseInt(value, 10));
    if (signature.length !== 32 || signature.some((value) => !Number.isInteger(value))) {
      throw new Error(`invalid signature ${fields[0]}`);
    }
    rows.set(fields[0], {
      windows: Number.parseInt(fields[1], 10),
      mistakes: Number.parseInt(fields[2], 10),
      probability_error_q15: Number.parseInt(fields[3], 10),
      mean_error: Number.parseInt(fields[4], 10),
      signature_q15: signature,
    });
  }
  return rows;
}

function initialCentroids(points, count) {
  const dimensions = points[0].normalized_q12.length;
  const mean = Array(dimensions).fill(0).map((_, feature) => Math.round(
    points.reduce((total, point) => total + point.normalized_q12[feature], 0) / points.length,
  ));
  const first = [...points].sort((left, right) =>
    squaredDistance(left.normalized_q12, mean) - squaredDistance(right.normalized_q12, mean)
    || left.id.localeCompare(right.id)
  )[0];
  const selected = [[...first.normalized_q12]];
  while (selected.length < count) {
    const next = [...points].sort((left, right) => {
      const leftDistance = Math.min(...selected.map((centroid) =>
        squaredDistance(left.normalized_q12, centroid)));
      const rightDistance = Math.min(...selected.map((centroid) =>
        squaredDistance(right.normalized_q12, centroid)));
      return rightDistance - leftDistance || left.id.localeCompare(right.id);
    })[0];
    selected.push([...next.normalized_q12]);
  }
  return selected;
}

function closest(features, centroids) {
  return centroids.map((centroid, index) => ({
    index,
    distance: squaredDistance(features, centroid),
  })).sort((left, right) => left.distance - right.distance || left.index - right.index)[0].index;
}

function recompute(points, count) {
  return Array.from({ length: count }, (_, cluster) => {
    const members = points.filter((point) => point.cluster === cluster);
    if (members.length === 0) throw new Error(`empty gradient cluster ${cluster}`);
    return Array(points[0].normalized_q12.length).fill(0).map((_, feature) => Math.round(
      members.reduce((total, point) => total + point.normalized_q12[feature], 0) / members.length,
    ));
  });
}

function squaredDistance(left, right) {
  return left.reduce((total, value, index) => {
    const difference = value - right[index];
    return total + difference * difference;
  }, 0);
}

function binding(file, bytes) {
  return { path: path.resolve(file), bytes: bytes.length, sha256: sha256(bytes) };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
