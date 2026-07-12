#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-author-block-swarm-v1";
const authors = ["crowley", "shakespeare", "blake"];
const shardManifest = read("shards/manifest.json");
const baselineReport = read("baseline/final.json");
const oracle = read("oracles/final-test.json");
const learnedToken = read("learned-token-router-report.json");
const learnedSpan = read("learned-span-router-report.json");
const recursiveToken = read("recursive-token-router-report.json");
const recursiveSpan = read("recursive-span-router-report.json");
const baseline = baselineReport.fixed_experts[0];
const fixed = Object.fromEntries(authors.map((author, index) => [author, {
  ...oracle.fixed_experts[index],
  delta_vs_trunk: delta(oracle.fixed_experts[index], baseline),
}]));

const generation = ["love", "law", "soul"].map((id) => {
  const bytes = fs.readFileSync(path.join(root, "generation", `${id}-top8.txt`));
  const nonSpace = [...bytes].filter((byte) => byte !== 32).length;
  const alphabetic = [...bytes].filter(
    (byte) => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122),
  ).length;
  let longest = 0;
  let run = 0;
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
  schema: "nsrl.literary_h8_author_block_expert_recursive_swarm.v1",
  data_contract: {
    source_manifest: shardManifest.source,
    policy: shardManifest.policy,
    leaf_token_bytes: Object.fromEntries(authors.map((author) => [
      author,
      shardManifest.splits.leaf_train[author].tokens.bytes,
    ])),
    router_train_chunks_per_author: shardManifest.policy.router_train_chunks_per_author,
    router_calibration_chunks_per_author:
      shardManifest.policy.router_calibration_chunks_per_author,
    final_test_chunks_per_author: Object.fromEntries(authors.map((author) => [
      author,
      shardManifest.splits.final_test[author].chunks,
    ])),
  },
  architecture: {
    trunk: "frozen H8 d128 ff256 two-block integer transformer",
    leaves: "three provenance-labelled rank-8 per-block i16 Q15 residual experts",
    leaf_parameters_each: oracle.experts.parameter_count_each,
    child_router_swarm: "three 41x16x3 NSRLRT1 integer neural routers per granularity",
    recursive_root: "one 41x16x3 NSRLRT1 over nine child probabilities plus 32 trunk features",
    granularities: ["token", "span-16"],
  },
  selected_leaf_training: Object.fromEntries(authors.map((author) => [
    author,
    read(`experts/${author}/train.json`),
  ])),
  frozen_final: {
    trunk: baseline,
    fixed_author_experts: fixed,
    target_aware_oracles: Object.fromEntries(
      Object.entries(oracle.oracle_routes).map(([granularity, metrics]) => [
        granularity,
        { ...metrics, delta_vs_trunk: delta(metrics, baseline) },
      ]),
    ),
    child_neural_router_consensus: {
      token: {
        ...learnedToken.selected_consensus.final,
        delta_vs_trunk: delta(learnedToken.selected_consensus.final, baseline),
      },
      span: {
        ...learnedSpan.selected_consensus.final,
        delta_vs_trunk: delta(learnedSpan.selected_consensus.final, baseline),
      },
    },
    recursive_neural_router: {
      token: { ...recursiveToken.final, delta_vs_trunk: delta(recursiveToken.final, baseline) },
      span: { ...recursiveSpan.final, delta_vs_trunk: delta(recursiveSpan.final, baseline) },
    },
  },
  generation_gate: {
    expert: "shakespeare (best fixed author leaf)",
    requirements: {
      non_space_per_mille_min: 500,
      alphabetic_per_mille_min: 300,
      distinct_bytes_min: 15,
      longest_space_run_max: 16,
    },
    samples: generation,
    pass: generation.every((sample) => sample.pass),
  },
  decision: {
    author_swarm_promoted: false,
    child_router_promoted: false,
    recursive_router_promoted: false,
    prose_promoted: false,
    evidence: {
      best_fixed_delta_vs_trunk_q15: fixed.shakespeare.delta_vs_trunk.probability_error_q15,
      token_oracle_delta_vs_trunk_q15:
        oracle.oracle_routes.token.probability_error_q15 - baseline.probability_error_q15,
      best_learned_delta_vs_trunk_q15:
        recursiveSpan.final.probability_error_q15 - baseline.probability_error_q15,
    },
    interpretation:
      "author labels create a small conditional utility ceiling, but all fixed leaves and learned recursive routes lose to the trunk; author-level decomposition is too coarse",
    next_experiment:
      "subdivide leaf experts by contiguous spans or token-context clusters, require a larger calibration oracle gap before router training, and expand authentic Blake coverage",
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
