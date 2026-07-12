#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const experimentDir = process.argv[2] ?? "data/experiments/literary-token-routing-v1/learned-router";
const recursiveDir = process.argv[3] ?? "data/experiments/literary-recursive-swarm-v1";
const spanLen = Number.parseInt(process.argv[4] ?? "16", 10);
const historyLen = Number.parseInt(process.argv[5] ?? "16", 10);
const featureSource = process.argv[6] ?? "context";
if (!Number.isInteger(spanLen) || spanLen < 1 || !Number.isInteger(historyLen) || historyLen < 1) {
  throw new Error("span and history lengths must be positive integers");
}
if (!["context", "hidden", "projected"].includes(featureSource)) {
  throw new Error("feature source must be context, hidden, or projected");
}

const splits = {
  train: "router-train.score-input.tsv",
  calibration: "router-calibration.score-input.tsv",
  final: "final-test.score-input.tsv",
};
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const clampQ15 = (value) => Math.max(0, Math.min(32_767, Math.round(value)));

function parseScoreInput(content) {
  const rows = new Map();
  for (const [index, line] of content.trimEnd().split("\n").entries()) {
    if (index === 0 && line === "sample_id\tprompt_hex") continue;
    const [sampleId, promptHex] = line.split("\t");
    if (!sampleId || !promptHex) throw new Error(`bad score-input row ${index + 1}`);
    rows.set(sampleId, Buffer.from(promptHex, "hex"));
  }
  return rows;
}

function parseTriplet(value, label) {
  const values = value.split(",").map((item) => Number.parseInt(item, 10));
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item) || item < 0)) {
    throw new Error(`bad ${label} triplet`);
  }
  return values;
}

function parseDetails(content) {
  const groups = new Map();
  for (const [index, line] of content.trimEnd().split("\n").entries()) {
    if (index === 0 && line.startsWith("sample_id\ttarget_offset\t")) continue;
    const fields = line.split("\t");
    if (fields.length !== 9) throw new Error(`bad details row ${index + 1}`);
    const row = {
      sampleId: fields[0],
      offset: Number.parseInt(fields[1], 10),
      targetHex: fields[2],
      losses: parseTriplet(fields[3], "loss"),
      mistakes: parseTriplet(fields[4], "mistake"),
      hiddenFeaturesQ15: fields[5].split(",").map((item) => Number.parseInt(item, 10)),
    };
    if (
      row.hiddenFeaturesQ15.length !== 32 ||
      row.hiddenFeaturesQ15.some((item) => !Number.isInteger(item) || item < -32_768 || item > 32_767)
    ) {
      throw new Error("bad hidden feature shape");
    }
    if (!Number.isInteger(row.offset) || row.offset < 0) throw new Error("bad target offset");
    const group = groups.get(row.sampleId) ?? [];
    group.push(row);
    groups.set(row.sampleId, group);
  }
  for (const rows of groups.values()) rows.sort((left, right) => left.offset - right.offset);
  return groups;
}

function contextFeaturesQ15(bytes) {
  const lower = Buffer.from(bytes.map((byte) => (byte >= 65 && byte <= 90 ? byte + 32 : byte)));
  const buckets = Array(24).fill(0);
  for (let index = 1; index < lower.length; index += 1) {
    buckets[((lower[index - 1] * 257) + lower[index]) % buckets.length] += 1;
  }
  const normalize = (count, total) => clampQ15((count * 32_767) / Math.max(1, total));
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
    ...buckets.map((count) => normalize(count, Math.max(1, lower.length - 1))),
    ...predicates.map((predicate) => normalize([...lower].filter(predicate).length, lower.length)),
  ];
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / Math.max(1, values.length);
}

function globalPriors(groups) {
  const all = [...groups.values()].flat();
  return {
    losses: [0, 1, 2].map((expert) => mean(all.map((row) => row.losses[expert]))),
    accuracies: [0, 1, 2].map(
      (expert) => 1 - mean(all.map((row) => row.mistakes[expert])),
    ),
  };
}

function probeFeaturesQ15(history, priors) {
  const losses = [0, 1, 2].map((expert) =>
    history.length > 0 ? mean(history.map((row) => row.losses[expert])) : priors.losses[expert],
  );
  const accuracies = [0, 1, 2].map((expert) =>
    history.length > 0
      ? 1 - mean(history.map((row) => row.mistakes[expert]))
      : priors.accuracies[expert],
  );
  const maxLoss = Math.max(...losses);
  return [
    ...losses.map((loss) => clampQ15(65_535 - loss)),
    ...accuracies.map((accuracy) => clampQ15(accuracy * 32_767)),
    ...losses.map((loss) => clampQ15((maxLoss - loss) * 8)),
  ];
}

function oracleTarget(losses) {
  return [0, 1, 2].sort((left, right) => losses[left] - losses[right] || left - right)[0];
}

function routerLine(routeId, losses, features) {
  return `${routeId}\t${oracleTarget(losses)}\t${features.join(",")}\t${losses.join(",")}`;
}

const oracleDir = path.join(experimentDir, "oracles");
const dataDir = path.join(
  experimentDir,
  featureSource === "hidden"
    ? "data-hidden"
    : featureSource === "projected"
      ? "data-projected"
      : "data",
);
await mkdir(path.join(dataDir, "token"), { recursive: true });
await mkdir(path.join(dataDir, "span"), { recursive: true });

const loaded = {};
for (const [split, scoreName] of Object.entries(splits)) {
  const scorePath = path.join(recursiveDir, "root-oracles", scoreName);
  const detailsPath = path.join(oracleDir, `${split}-details.tsv`);
  const scoreContent = await readFile(scorePath, "utf8");
  const detailsContent = await readFile(detailsPath, "utf8");
  loaded[split] = {
    prompts: parseScoreInput(scoreContent),
    groups: parseDetails(detailsContent),
    sourceHashes: { score_input: sha256(scoreContent), details: sha256(detailsContent) },
  };
}
const priors = globalPriors(loaded.train.groups);
const manifest = {
  schema: "nsrl.literary_token_router_datasets.v1",
  span_len: spanLen,
  history_len: historyLen,
  context_bytes: 32,
  feature_source: featureSource,
  features: featureSource === "hidden"
    ? {
        count: 41,
        pooled_contextual_hidden_channels: 32,
        hidden_feature_model_index: 2,
        rolling_probe_features: 9,
        current_target_excluded: true,
        rolling_history_uses_prior_observed_tokens_only: true,
      }
    : featureSource === "projected"
      ? {
          count: 41,
          signed_projected_contextual_hidden_channels: 32,
          rolling_probe_features: 9,
          current_target_excluded: true,
          rolling_history_uses_prior_observed_tokens_only: true,
        }
    : {
        count: 41,
        context_bigram_buckets: 24,
        structural_ratios: 8,
        rolling_probe_features: 9,
        current_target_excluded: true,
        rolling_history_uses_prior_observed_tokens_only: true,
      },
  train_only_global_priors: priors,
  splits: {},
};

for (const split of Object.keys(splits)) {
  const { prompts, groups, sourceHashes } = loaded[split];
  const tokenLines = ["sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15"];
  const spanLines = ["sample_id\ttarget\tfeatures_q15\toracle_child_losses_q15"];
  const tokenMap = ["route_id\tsample_id\tstart_offset\tend_offset"];
  const spanMap = ["route_id\tsample_id\tstart_offset\tend_offset"];
  const tokenLabels = [0, 0, 0];
  const spanLabels = [0, 0, 0];

  for (const [sampleId, rows] of groups) {
    const prompt = prompts.get(sampleId);
    if (!prompt) throw new Error(`missing prompt for ${sampleId}`);
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      if (row.offset >= prompt.length || prompt[row.offset].toString(16).padStart(2, "0") !== row.targetHex) {
        throw new Error(`target binding mismatch for ${sampleId}@${row.offset}`);
      }
      const history = rows.slice(Math.max(0, index - historyLen), index);
      const context = prompt.subarray(Math.max(0, row.offset - 32), row.offset);
      const baseFeatures = featureSource === "hidden" || featureSource === "projected"
        ? row.hiddenFeaturesQ15
        : contextFeaturesQ15(context);
      const features = [...baseFeatures, ...probeFeaturesQ15(history, priors)];
      if (features.length !== 41) throw new Error("feature shape mismatch");
      const routeId = `${sampleId}@${row.offset}`;
      tokenLines.push(routerLine(routeId, row.losses, features));
      tokenMap.push(`${routeId}\t${sampleId}\t${row.offset}\t${row.offset + 1}`);
      tokenLabels[oracleTarget(row.losses)] += 1;
    }

    for (let start = 0; start < rows.length; start += spanLen) {
      const end = Math.min(rows.length, start + spanLen);
      const first = rows[start];
      const last = rows[end - 1];
      const history = rows.slice(Math.max(0, start - historyLen), start);
      const context = prompt.subarray(Math.max(0, first.offset - 32), first.offset);
      const baseFeatures = featureSource === "hidden" || featureSource === "projected"
        ? first.hiddenFeaturesQ15
        : contextFeaturesQ15(context);
      const features = [...baseFeatures, ...probeFeaturesQ15(history, priors)];
      const losses = [0, 1, 2].map((expert) =>
        Math.round(mean(rows.slice(start, end).map((row) => row.losses[expert]))),
      );
      const routeId = `${sampleId}@${first.offset}-${last.offset + 1}`;
      spanLines.push(routerLine(routeId, losses, features));
      spanMap.push(`${routeId}\t${sampleId}\t${first.offset}\t${last.offset + 1}`);
      spanLabels[oracleTarget(losses)] += 1;
    }
  }

  const outputs = {};
  for (const [granularity, lines, mappings, labels] of [
    ["token", tokenLines, tokenMap, tokenLabels],
    ["span", spanLines, spanMap, spanLabels],
  ]) {
    const dataContent = `${lines.join("\n")}\n`;
    const mapContent = `${mappings.join("\n")}\n`;
    const dataPath = path.join(dataDir, granularity, `${split}.tsv`);
    const mapPath = path.join(dataDir, granularity, `${split}-map.tsv`);
    await writeFile(dataPath, dataContent);
    await writeFile(mapPath, mapContent);
    outputs[granularity] = {
      data_path: path.resolve(dataPath),
      data_sha256: sha256(dataContent),
      map_path: path.resolve(mapPath),
      map_sha256: sha256(mapContent),
      rows: lines.length - 1,
      label_counts: labels,
    };
  }
  manifest.splits[split] = { source_sha256: sourceHashes, ...outputs };
}

const manifestPath = path.join(
  experimentDir,
  featureSource === "hidden"
    ? "dataset-manifest-hidden.json"
    : featureSource === "projected"
      ? "dataset-manifest-projected.json"
      : "dataset-manifest.json",
);
await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ manifest: manifestPath, splits: manifest.splits }));
