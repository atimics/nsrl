#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const AUTHORS = ["crowley", "shakespeare", "blake"];
const sourceRoot = process.argv[2]
  ?? "data/experiments/literary-h8-author-block-swarm-v1/shards";
const outDir = process.argv[3]
  ?? "data/experiments/literary-h8-context-block-swarm-v1/clusters";
const spanTokens = Number.parseInt(process.argv[4] ?? "512", 10);
const clusterCount = Number.parseInt(process.argv[5] ?? "3", 10);
const iterations = Number.parseInt(process.argv[6] ?? "20", 10);
if (spanTokens < 128 || clusterCount < 2 || iterations < 1) {
  throw new Error("span tokens >=128, clusters >=2, and iterations >=1 are required");
}

const sourceManifestPath = path.join(sourceRoot, "manifest.json");
const sourceManifestBytes = fs.readFileSync(sourceManifestPath);
const sourceManifest = JSON.parse(sourceManifestBytes);
if (sourceManifest.schema !== "nsrl.literary_author_span_shards.v1") {
  throw new Error("unexpected source shard manifest");
}

const spans = [];
const sourceBindings = {};
for (const author of AUTHORS) {
  const binding = sourceManifest.splits.leaf_train[author].tokens;
  const bytes = fs.readFileSync(binding.path);
  if (bytes.length !== binding.bytes || sha256(bytes) !== binding.sha256) {
    throw new Error(`${author} leaf token binding mismatch`);
  }
  sourceBindings[author] = binding;
  for (let start = 0; start + spanTokens <= bytes.length; start += spanTokens) {
    const tokens = bytes.subarray(start, start + spanTokens);
    spans.push({
      id: `${author}@${start}-${start + spanTokens}`,
      author,
      start,
      end: start + spanTokens,
      tokens,
      token_sha256: sha256(tokens),
      features_q15: contextFeaturesQ15(tokens),
      cluster: -1,
    });
  }
}
if (spans.length < clusterCount * 2) throw new Error("too few spans for clustering");

let centroids = initialCentroids(spans, clusterCount);
let changed = 0;
let completedIterations = 0;
for (let iteration = 0; iteration < iterations; iteration += 1) {
  changed = 0;
  for (const span of spans) {
    const next = closestCentroid(span.features_q15, centroids);
    if (span.cluster !== next) changed += 1;
    span.cluster = next;
  }
  const nextCentroids = recomputeCentroids(spans, clusterCount);
  completedIterations = iteration + 1;
  centroids = nextCentroids;
  if (changed === 0) break;
}

fs.mkdirSync(outDir, { recursive: true });
const scoreInputPath = path.join(outDir, "leaf-spans.score-input.tsv");
const scoreInputBytes = Buffer.from(
  `sample_id\tprompt_hex\n${spans.map((span) => `${span.id}\t${span.tokens.toString("hex")}`).join("\n")}\n`,
);
fs.writeFileSync(scoreInputPath, scoreInputBytes);
const clusters = [];
for (let cluster = 0; cluster < clusterCount; cluster += 1) {
  const members = spans.filter((span) => span.cluster === cluster);
  if (members.length === 0) throw new Error(`empty cluster ${cluster}`);
  const parts = [];
  for (const member of members) {
    if (parts.length > 0) parts.push(Buffer.from(" "));
    parts.push(member.tokens);
  }
  const tokens = Buffer.concat(parts);
  const tokenPath = path.join(outDir, `cluster-${cluster}.tokens.u8`);
  const provenancePath = path.join(outDir, `cluster-${cluster}.spans.jsonl`);
  fs.writeFileSync(tokenPath, tokens);
  fs.writeFileSync(
    provenancePath,
    `${members.map((span) => JSON.stringify({
      id: span.id,
      author: span.author,
      source_start: span.start,
      source_end: span.end,
      token_sha256: span.token_sha256,
      features_q15: span.features_q15,
      squared_distance: squaredDistance(span.features_q15, centroids[cluster]),
    })).join("\n")}\n`,
  );
  clusters.push({
    id: cluster,
    centroid_q15: centroids[cluster],
    spans: members.length,
    token_bytes: tokens.length,
    author_spans: Object.fromEntries(AUTHORS.map((author) => [
      author,
      members.filter((span) => span.author === author).length,
    ])),
    mean_squared_distance: Math.floor(
      members.reduce(
        (total, span) => total + squaredDistance(span.features_q15, centroids[cluster]),
        0,
      ) / members.length,
    ),
    tokens: binding(tokenPath, tokens),
    provenance: binding(provenancePath, fs.readFileSync(provenancePath)),
  });
}

const minimumSpans = Math.min(...clusters.map((cluster) => cluster.spans));
const manifest = {
  schema: "nsrl.literary_context_cluster_shards.v1",
  source: {
    manifest: binding(sourceManifestPath, sourceManifestBytes),
    leaf_tokens: sourceBindings,
  },
  policy: {
    span_tokens: spanTokens,
    cluster_count: clusterCount,
    algorithm: "deterministic_farthest_seed_kmeans_q15",
    feature_count: 32,
    features: "24 byte-bigram hash buckets plus 8 structural ratios",
    target_blind: true,
    cross_author: true,
    overlapping_spans: false,
    remainder_tokens_excluded: true,
    iterations_requested: iterations,
    iterations_completed: completedIterations,
    final_assignment_changes: changed,
  },
  spans: spans.length,
  leaf_span_score_input: binding(scoreInputPath, scoreInputBytes),
  minimum_cluster_spans: minimumSpans,
  minimum_cluster_training_windows_stride46_seq64:
    Math.floor((minimumSpans * spanTokens + minimumSpans - 1 - 65) / 46) + 1,
  clusters,
};
const manifestPath = path.join(outDir, "manifest.json");
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(manifestPath);

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

function initialCentroids(points, count) {
  const mean = Array(32).fill(0).map((_, feature) => Math.round(
    points.reduce((total, point) => total + point.features_q15[feature], 0) / points.length,
  ));
  const first = [...points].sort((left, right) =>
    squaredDistance(left.features_q15, mean) - squaredDistance(right.features_q15, mean)
    || left.id.localeCompare(right.id)
  )[0];
  const selected = [first.features_q15];
  while (selected.length < count) {
    const next = [...points].sort((left, right) => {
      const leftDistance = Math.min(...selected.map((centroid) =>
        squaredDistance(left.features_q15, centroid)));
      const rightDistance = Math.min(...selected.map((centroid) =>
        squaredDistance(right.features_q15, centroid)));
      return rightDistance - leftDistance || left.id.localeCompare(right.id);
    })[0];
    selected.push([...next.features_q15]);
  }
  return selected;
}

function closestCentroid(features, centroids) {
  return centroids
    .map((centroid, index) => ({ index, distance: squaredDistance(features, centroid) }))
    .sort((left, right) => left.distance - right.distance || left.index - right.index)[0].index;
}

function recomputeCentroids(points, count) {
  return Array.from({ length: count }, (_, cluster) => {
    const members = points.filter((point) => point.cluster === cluster);
    if (members.length === 0) throw new Error(`empty cluster ${cluster}`);
    return Array(32).fill(0).map((_, feature) => Math.round(
      members.reduce((total, point) => total + point.features_q15[feature], 0) / members.length,
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
