#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-gradient-block-swarm-v1";
const baselineRoot = process.argv[3] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const signatures = read("signatures/leaf.json");
const clusters = read("clusters/manifest.json");
const final = read("oracles/final-test.json");
const tokenRouter = read("learned-token-router-report.json");
const spanRouter = read("learned-span-router-report.json");
const recursiveToken = read("recursive-token-router-report.json");
const recursiveSpan = read("recursive-span-router-report.json");
const trunk = JSON.parse(fs.readFileSync(path.join(baselineRoot, "baseline", "final.json"), "utf8"))
  .fixed_experts[0];
const fixed = final.fixed_experts.map((metrics, cluster) => ({
  cluster,
  ...metrics,
  delta_vs_trunk: delta(metrics, trunk),
}));
const selectedTraining = [0, 1, 2].map((cluster) => ({
  cluster,
  trace: read(`experts/cluster-${cluster}/train.json`),
}));
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

const report = {
  schema: "nsrl.literary_h8_hidden_gradient_block_swarm.v1",
  architecture: {
    trunk: "frozen H8 d128 ff256 two-block integer transformer",
    signatures: "16 signed plus 16 magnitude final-hidden gradient channels",
    leaves: "three rank-8 per-block i16 Q15 residual experts",
    child_router_swarm: "three 41x16x3 integer neural routers per granularity",
    recursive_root: "41x16x3 integer neural router over child probabilities and trunk hidden state",
  },
  optimizer_correction: {
    objective: "probability_error",
    old_order: "round each Q30 outer product before learning-rate application",
    corrected_order: "accumulate raw Q30 products, multiply learning rate, divide once with error-feedback carry",
    bidirectional_loss_guard_available: true,
    result: "all three selected experts decrease their own exact leaf probability error without saturation",
  },
  signature_evidence: signatures,
  clustering: {
    policy: clusters.policy,
    spans: clusters.spans,
    clusters: clusters.clusters.map(({ id, spans, token_bytes, author_spans, mean_probability_error_q15 }) => ({
      id,
      spans,
      token_bytes,
      author_spans,
      mean_probability_error_q15,
    })),
  },
  selected_training: selectedTraining,
  frozen_final: {
    trunk,
    fixed_gradient_experts: fixed,
    target_aware_oracles: Object.fromEntries(
      Object.entries(final.oracle_routes).map(([name, metrics]) => [
        name,
        { ...metrics, delta_vs_trunk: delta(metrics, trunk) },
      ]),
    ),
    child_neural_router_consensus: {
      token: {
        ...tokenRouter.selected_consensus.final,
        delta_vs_trunk: delta(tokenRouter.selected_consensus.final, trunk),
      },
      span: {
        ...spanRouter.selected_consensus.final,
        delta_vs_trunk: delta(spanRouter.selected_consensus.final, trunk),
      },
    },
    recursive_neural_router: {
      token: { ...recursiveToken.final, delta_vs_trunk: delta(recursiveToken.final, trunk) },
      span: { ...recursiveSpan.final, delta_vs_trunk: delta(recursiveSpan.final, trunk) },
    },
  },
  generation_gate: {
    expert: "gradient cluster 0 (best fixed final leaf)",
    samples: generation,
    pass: generation.every((sample) => sample.pass),
  },
  decision: {
    gradient_expert_mechanism_promoted: true,
    selected_checkpoint: "gradient cluster 0",
    learned_router_promoted_over_fixed: false,
    recursive_router_promoted_over_fixed: false,
    prose_promoted: false,
    evidence: {
      fixed_leaf_deltas_vs_trunk_q15: fixed.map((row) =>
        row.delta_vs_trunk.probability_error_q15),
      best_fixed_delta_vs_trunk_q15: Math.min(...fixed.map((row) =>
        row.delta_vs_trunk.probability_error_q15)),
      token_oracle_delta_vs_trunk_q15:
        final.oracle_routes.token.probability_error_q15 - trunk.probability_error_q15,
      token_oracle_gain_beyond_best_fixed_q15:
        Math.min(...fixed.map((row) => row.probability_error_q15))
          - final.oracle_routes.token.probability_error_q15,
      best_recursive_delta_vs_trunk_q15: Math.min(
        recursiveToken.final.probability_error_q15,
        recursiveSpan.final.probability_error_q15,
      ) - trunk.probability_error_q15,
    },
    interpretation:
      "model-native failure clustering plus corrected fractional gradient accumulation produces the first internal experts that all beat the trunk; current routers cannot capture the remaining small conditional gap",
    next_experiment:
      "increase expert utility separation with resumed multi-stage metric-aligned training, then retrain target-blind routers only after fixed leaves preserve their gains",
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
