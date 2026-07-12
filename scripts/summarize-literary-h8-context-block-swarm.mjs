#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-context-block-swarm-v1";
const authorRoot = process.argv[3] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const clusters = read("clusters/manifest.json");
const calibration = read("oracles/router-calibration.json");
const final = read("oracles/final-test.json");
const calibrationCentroid = read("oracles/router-calibration-centroid-router.json");
const finalCentroid = read("oracles/final-test-centroid-router.json");
const trunk = JSON.parse(fs.readFileSync(path.join(authorRoot, "baseline", "final.json"), "utf8"))
  .fixed_experts[0];
const authorCalibration = JSON.parse(fs.readFileSync(
  path.join(authorRoot, "oracles", "router-calibration.json"),
  "utf8",
));
const authorFixed = authorCalibration.fixed_experts[authorCalibration.best_fixed_expert];
const authorTokenOracleGain =
  authorFixed.probability_error_q15 - authorCalibration.oracle_routes.token.probability_error_q15;
const calibrationFixed = calibration.fixed_experts[calibration.best_fixed_expert];
const contextTokenOracleGain =
  calibrationFixed.probability_error_q15 - calibration.oracle_routes.token.probability_error_q15;
const gatePassed = contextTokenOracleGain > authorTokenOracleGain;

const generation = ["love", "law", "soul"].map((id) => {
  const bytes = fs.readFileSync(path.join(root, "generation", `${id}-top8.txt`));
  const nonSpace = [...bytes].filter((byte) => byte !== 32).length;
  const alphabetic = [...bytes].filter(
    (byte) => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122),
  ).length;
  let run = 0;
  let longest = 0;
  for (const byte of bytes) {
    if (byte === 32) {
      run += 1;
      longest = Math.max(longest, run);
    } else run = 0;
  }
  const metrics = {
    id,
    bytes: bytes.length,
    non_space_per_mille: Math.floor(nonSpace * 1000 / bytes.length),
    alphabetic_per_mille: Math.floor(alphabetic * 1000 / bytes.length),
    distinct_bytes: new Set(bytes).size,
    longest_space_run: longest,
  };
  metrics.pass = metrics.non_space_per_mille >= 500
    && metrics.alphabetic_per_mille >= 300
    && metrics.distinct_bytes >= 15
    && metrics.longest_space_run <= 16;
  return metrics;
});

const fixed = final.fixed_experts.map((metrics, cluster) => ({
  cluster,
  ...metrics,
  delta_vs_trunk: delta(metrics, trunk),
}));
const report = {
  schema: "nsrl.literary_h8_context_cluster_block_swarm.v1",
  architecture: {
    trunk: "frozen H8 d128 ff256 two-block integer transformer",
    expert_count: 3,
    expert: "rank-8 per-block i16 Q15 residual",
    decomposition: "deterministic target-blind k-means over non-overlapping 512-token spans",
    cross_author: true,
  },
  clustering: {
    policy: clusters.policy,
    spans: clusters.spans,
    minimum_cluster_spans: clusters.minimum_cluster_spans,
    clusters: clusters.clusters.map(({ id, spans, token_bytes, author_spans, mean_squared_distance }) => ({
      id,
      spans,
      token_bytes,
      author_spans,
      mean_squared_distance,
    })),
  },
  selected_experts: [0, 1, 2].map((cluster) => ({
    cluster,
    training: read(`experts/cluster-${cluster}/train.json`),
  })),
  router_training_gate: {
    metric: "calibration token-oracle total Q15 gain over best fixed expert",
    reference_author_swarm_gain_q15: authorTokenOracleGain,
    context_cluster_gain_q15: contextTokenOracleGain,
    requires_strictly_larger_than_reference: true,
    passed: gatePassed,
    neural_router_training_skipped: !gatePassed,
  },
  frozen_final: {
    trunk,
    fixed_cluster_experts: fixed,
    target_aware_oracles: Object.fromEntries(
      Object.entries(final.oracle_routes).map(([name, metrics]) => [
        name,
        { ...metrics, delta_vs_trunk: delta(metrics, trunk) },
      ]),
    ),
    target_blind_centroid_router: {
      token: {
        ...finalCentroid.routes.token,
        delta_vs_trunk: delta(finalCentroid.routes.token, trunk),
      },
      span: {
        ...finalCentroid.routes.span,
        delta_vs_trunk: delta(finalCentroid.routes.span, trunk),
      },
      calibration: calibrationCentroid.routes,
    },
  },
  generation_gate: {
    expert: "cluster-1 (best fixed final cluster expert)",
    samples: generation,
    pass: generation.every((sample) => sample.pass),
  },
  decision: {
    expert_swarm_promoted: false,
    centroid_router_promoted: false,
    neural_router_trained: gatePassed,
    prose_promoted: false,
    evidence: {
      best_fixed_delta_vs_trunk_q15: Math.min(...fixed.map((row) =>
        row.delta_vs_trunk.probability_error_q15)),
      token_oracle_delta_vs_trunk_q15:
        final.oracle_routes.token.probability_error_q15 - trunk.probability_error_q15,
      centroid_span_delta_vs_trunk_q15:
        finalCentroid.routes.span.probability_error_q15 - trunk.probability_error_q15,
    },
    interpretation:
      "surface-context clusters are below author level but still do not align training gradients with deployable expert utility; their oracle gap is smaller than the rejected author swarm",
    next_experiment:
      "cluster spans by measured frozen-trunk residual/gradient signatures rather than surface text, while keeping inference routing target-blind through a separately learned hidden-state predictor",
  },
};
fs.writeFileSync(path.join(root, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.decision));

function read(relative) {
  return JSON.parse(fs.readFileSync(path.join(root, relative), "utf8"));
}

function delta(metrics, reference) {
  return {
    probability_error_q15: metrics.probability_error_q15 - reference.probability_error_q15,
    mean_probability_error_q15:
      metrics.mean_probability_error_q15 - reference.mean_probability_error_q15,
    mistakes: metrics.mistakes - reference.mistakes,
    accuracy_per_mille: metrics.accuracy_per_mille - reference.accuracy_per_mille,
  };
}
