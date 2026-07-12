#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-token-routing-v1/learned-router";
const out = process.argv[3] ?? path.join(root, "learned-router-report.json");
const profile = process.argv[4] ?? "context";
const granularities = ["token", "span"];
const profileConfig = profile === "hidden"
  ? {
      dataDir: "data-hidden",
      views: {
        token: [
          ["hidden-a", "routers-hidden/token-hidden-a"],
          ["hidden-b", "sweeps-hidden/token-hidden-b-e2"],
          ["full", "routers-hidden/token-full"],
        ],
        span: [
          ["hidden-a", "sweeps-hidden/span-hidden-a-e1"],
          ["hidden-b", "routers-hidden/span-hidden-b"],
          ["full", "routers-hidden/span-full"],
        ],
      },
    }
  : {
      dataDir: "data",
      views: Object.fromEntries(
        granularities.map((granularity) => [
          granularity,
          ["semantic", "structural", "full"].map((view) => [
            view,
            `routers/${granularity}-${view}`,
          ]),
        ]),
      ),
    };
if (!["context", "hidden"].includes(profile)) throw new Error("profile must be context or hidden");
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

function parseTriplet(value) {
  const values = value.split(",").map((item) => Number.parseInt(item, 10));
  if (values.length !== 3 || values.some((item) => !Number.isInteger(item))) {
    throw new Error("invalid triplet");
  }
  return values;
}

function parseDetails(content) {
  const groups = new Map();
  for (const [index, line] of content.trimEnd().split("\n").entries()) {
    if (index === 0) continue;
    const fields = line.split("\t");
    const row = {
      offset: Number.parseInt(fields[1], 10),
      losses: parseTriplet(fields[3]),
      mistakes: parseTriplet(fields[4]),
    };
    const group = groups.get(fields[0]) ?? new Map();
    group.set(row.offset, row);
    groups.set(fields[0], group);
  }
  return groups;
}

function parseMappings(content) {
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
  for (const [index, line] of content.trimEnd().split("\n").entries()) {
    if (index === 0) continue;
    const fields = line.split("\t");
    rows.set(fields[0], {
      target: Number.parseInt(fields[1], 10),
      predicted: Number.parseInt(fields[2], 10),
      probabilities: parseTriplet(fields[4]),
    });
  }
  return rows;
}

function consensusPrediction(predictions, weights, routeId) {
  const rows = predictions.map((view) => view.get(routeId));
  if (rows.some((row) => !row)) throw new Error(`missing prediction for ${routeId}`);
  const scores = [0, 1, 2].map((expert) =>
    rows.reduce(
      (total, row, index) => total + (weights[index] * row.probabilities[expert]),
      0,
    ),
  );
  const predicted = [0, 1, 2].sort(
    (left, right) => scores[right] - scores[left] || left - right,
  )[0];
  return { predicted, target: rows[0].target, scores };
}

function consensusPredictionsWithHysteresis(state, weights, margin) {
  const output = new Map();
  const previousBySample = new Map();
  for (const mapping of state.mappings) {
    const row = consensusPrediction(state.predictions, weights, mapping.routeId);
    const previous = previousBySample.get(mapping.sampleId);
    let predicted = row.predicted;
    if (
      previous !== undefined &&
      row.scores[predicted] - row.scores[previous] < margin
    ) {
      predicted = previous;
    }
    previousBySample.set(mapping.sampleId, predicted);
    output.set(mapping.routeId, { predicted, target: row.target });
  }
  return output;
}

function evaluate(mappings, details, predictionForRoute) {
  let windows = 0;
  let mistakes = 0;
  let probabilityError = 0;
  let oracleError = 0;
  let routeCorrect = 0;
  let routeSwitches = 0;
  const utilization = [0, 0, 0];
  const previousBySample = new Map();
  for (const mapping of mappings) {
    const { predicted, target } = predictionForRoute(mapping.routeId);
    if (![0, 1, 2].includes(predicted)) throw new Error("prediction out of range");
    routeCorrect += Number(predicted === target);
    if (previousBySample.has(mapping.sampleId) && previousBySample.get(mapping.sampleId) !== predicted) {
      routeSwitches += 1;
    }
    previousBySample.set(mapping.sampleId, predicted);
    const sample = details.get(mapping.sampleId);
    if (!sample) throw new Error(`missing details for ${mapping.sampleId}`);
    for (let offset = mapping.start; offset < mapping.end; offset += 1) {
      const row = sample.get(offset);
      if (!row) throw new Error(`missing target ${mapping.sampleId}@${offset}`);
      windows += 1;
      mistakes += row.mistakes[predicted];
      probabilityError += row.losses[predicted];
      oracleError += Math.min(...row.losses);
      utilization[predicted] += 1;
    }
  }
  return {
    windows,
    route_decisions: mappings.length,
    route_correct: routeCorrect,
    route_accuracy_per_mille: Math.floor((routeCorrect * 1000) / mappings.length),
    mistakes,
    accuracy_per_mille: Math.floor(((windows - mistakes) * 1000) / windows),
    probability_error_q15: probabilityError,
    mean_probability_error_q15: Math.floor(probabilityError / windows),
    mean_routing_regret_q15: Math.floor((probabilityError - oracleError) / windows),
    route_switches: routeSwitches,
    utilization_tokens: utilization,
    utilization_per_mille: utilization.map((count) => Math.floor((count * 1000) / windows)),
  };
}

function better(left, right) {
  return (
    left.metrics.mean_probability_error_q15 < right.metrics.mean_probability_error_q15 ||
    (left.metrics.mean_probability_error_q15 === right.metrics.mean_probability_error_q15 &&
      (left.metrics.accuracy_per_mille > right.metrics.accuracy_per_mille ||
        (left.metrics.accuracy_per_mille === right.metrics.accuracy_per_mille &&
          left.metrics.route_switches < right.metrics.route_switches)))
  );
}

const report = {
  schema: "nsrl.literary_learned_token_routing.v1",
  feature_profile: profile,
  selection_split: "router_calibration_only",
  final_split_used_for_selection: false,
  granularities: {},
  known_non_claims: [
    "routes_outputs_of_three_whole_models_not_shared_trunk_ffn_experts_yet",
    "teacher_forced_previous_token_probe_features",
    "does_not_claim_language_model_quality",
  ],
};

for (const granularity of granularities) {
  const viewConfigs = profileConfig.views[granularity];
  const views = viewConfigs.map(([view]) => view);
  const splitState = {};
  for (const split of ["calibration", "final"]) {
    const detailBytes = await readFile(path.join(root, "oracles", `${split}-details.tsv`));
    const mapBytes = await readFile(
      path.join(root, profileConfig.dataDir, granularity, `${split}-map.tsv`),
    );
    const predictions = [];
    const predictionHashes = {};
    for (const [view, routerDir] of viewConfigs) {
      const bytes = await readFile(
        path.join(root, routerDir, `${split}.predictions.tsv`),
      );
      predictions.push(parsePredictions(bytes.toString("utf8")));
      predictionHashes[view] = sha256(bytes);
    }
    splitState[split] = {
      details: parseDetails(detailBytes.toString("utf8")),
      mappings: parseMappings(mapBytes.toString("utf8")),
      predictions,
      hashes: { details: sha256(detailBytes), mappings: sha256(mapBytes), predictions: predictionHashes },
    };
  }

  const replicas = {};
  for (let viewIndex = 0; viewIndex < views.length; viewIndex += 1) {
    const view = views[viewIndex];
    replicas[view] = {};
    for (const split of ["calibration", "final"]) {
      const state = splitState[split];
      replicas[view][split] = evaluate(state.mappings, state.details, (routeId) => {
        const row = state.predictions[viewIndex].get(routeId);
        if (!row) throw new Error(`missing ${view} prediction for ${routeId}`);
        return row;
      });
    }
  }

  let selected = null;
  const hysteresisMargins = [0, 256, 512, 1024, 2048, 4096, 8192];
  for (let semantic = 1; semantic <= 3; semantic += 1) {
    for (let structural = 1; structural <= 3; structural += 1) {
      for (let full = 1; full <= 3; full += 1) {
        const weights = [semantic, structural, full];
        for (const hysteresisMarginQ15 of hysteresisMargins) {
          const state = splitState.calibration;
          const predictions = consensusPredictionsWithHysteresis(
            state,
            weights,
            hysteresisMarginQ15,
          );
          const metrics = evaluate(state.mappings, state.details, (routeId) =>
            predictions.get(routeId),
          );
          const candidate = { weights, hysteresisMarginQ15, metrics };
          if (selected === null || better(candidate, selected)) selected = candidate;
        }
      }
    }
  }
  const finalState = splitState.final;
  const finalPredictions = consensusPredictionsWithHysteresis(
    finalState,
    selected.weights,
    selected.hysteresisMarginQ15,
  );
  const finalConsensus = evaluate(finalState.mappings, finalState.details, (routeId) =>
    finalPredictions.get(routeId),
  );

  report.granularities[granularity] = {
    replicas,
    calibrated_consensus: {
      weights: Object.fromEntries(views.map((view, index) => [view, selected.weights[index]])),
      hysteresis_margin_q15: selected.hysteresisMarginQ15,
      calibration: selected.metrics,
      final: finalConsensus,
    },
    source_hashes: {
      calibration: splitState.calibration.hashes,
      final: splitState.final.hashes,
    },
  };
}

for (const granularity of granularities) {
  for (const [view, routerDir] of profileConfig.views[granularity]) {
    const modelBytes = await readFile(
      path.join(root, routerDir, "router.nsrlrt"),
    );
    report.granularities[granularity].replicas[view].model_sha256 = sha256(modelBytes);
  }
}

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    out,
    token: report.granularities.token.calibrated_consensus,
    span: report.granularities.span.calibrated_consensus,
  }),
);
