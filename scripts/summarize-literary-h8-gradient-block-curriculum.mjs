#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const root = process.argv[2]
  ?? "data/experiments/literary-h8-gradient-block-curriculum-v1";
const parentRoot = process.argv[3]
  ?? "data/experiments/literary-h8-gradient-block-swarm-v1";

const parent = readParent("report.json");
const parentCalibration = readParent("oracles/router-calibration.json");
const parentFinal = readParent("oracles/final-test.json");
const stage1 = read("scores/stage1-combined.json");
const stage2 = read("scores/stage2-combined.json");
const final = read("oracles/final-test.json");
const childToken = read("learned-token-router-report.json");
const childSpan = read("learned-span-router-report.json");
const recursiveToken = read("recursive-token-router-report.json");
const recursiveSpan = read("recursive-span-router-report.json");
const regretSweep = read("regret-router-sweep-selection.json");
const trunk = parent.frozen_final.trunk;
const parentBestFixed = parentFinal.fixed_experts[parentFinal.best_fixed_expert];
const finalBestFixed = final.fixed_experts[final.best_fixed_expert];

const selectedStages = [
  {
    stage: 1,
    token_offset: 23_552,
    selected_learning_rates: [1_024, 1_024, 64],
    selected_layers: ["all", "final", "all"],
    calibration: routeSummary(stage1),
    candidate_substitutions: candidateScores(1, ["c0-1024", "c0-256", "c1-1024", "c1-256", "c2-256", "c2-64"]),
  },
  {
    stage: 2,
    token_offset: 47_168,
    selected_learning_rates: [1_024, 1_024, 64],
    selected_layers: ["all", "final", "all"],
    calibration: routeSummary(stage2),
    candidate_substitutions: candidateScores(2, ["c0-1024", "c0-256", "c1-1024", "c1-256", "c2-64", "c2-16"]),
  },
];

const artifacts = [0, 1, 2].map((cluster) => {
  const relative = `experts/cluster-${cluster}/expert.nsrlbe`;
  const bytes = fs.readFileSync(path.join(root, relative));
  return {
    cluster,
    path: path.resolve(root, relative),
    bytes: bytes.length,
    sha256: sha256(bytes),
  };
});

const generation = ["love", "law", "soul"].map((id) => generationMetrics(id));
const bestChild = bestByError([
  { id: "child-token", metrics: childToken.selected_consensus.final },
  { id: "child-span", metrics: childSpan.selected_consensus.final },
]);
const bestRecursive = bestByError([
  { id: "recursive-token", metrics: recursiveToken.final },
  { id: "recursive-span", metrics: recursiveSpan.final },
]);
const bestRegretRouter = regretSweep.selected["span-hidden-b"];

const report = {
  schema: "nsrl.literary_h8_gradient_block_curriculum.v1",
  parent_experiment: path.resolve(parentRoot),
  policy: {
    unit_of_scale: "many short resumable per-block expert runs",
    training_shards: "successive non-overlapping bands of gradient-cluster leaf tokens",
    selection_split: "router calibration only",
    final_split_used_for_selection: false,
    candidate_gate: "whole-triad token-oracle probability error, then frozen final audit",
    router_gate: "retrain only after calibration conditional utility widens",
  },
  architecture: parent.architecture,
  selected_artifacts: artifacts,
  stages: selectedStages,
  calibration_progression: [
    { id: "parent", ...routeSummary(parentCalibration) },
    { id: "stage-1", ...routeSummary(stage1) },
    { id: "stage-2", ...routeSummary(stage2) },
  ],
  frozen_final: {
    trunk,
    parent_best_fixed: parentBestFixed,
    fixed_experts: final.fixed_experts.map((metrics, cluster) => ({
      cluster,
      ...metrics,
      delta_vs_trunk: delta(metrics, trunk),
    })),
    best_fixed_expert: final.best_fixed_expert,
    best_fixed: {
      ...finalBestFixed,
      delta_vs_trunk: delta(finalBestFixed, trunk),
      delta_vs_parent_best_fixed: delta(finalBestFixed, parentBestFixed),
    },
    target_aware_oracles: Object.fromEntries(
      Object.entries(final.oracle_routes).map(([granularity, metrics]) => [
        granularity,
        {
          ...metrics,
          delta_vs_trunk: delta(metrics, trunk),
          gain_beyond_best_fixed_q15:
            finalBestFixed.probability_error_q15 - metrics.probability_error_q15,
        },
      ]),
    ),
    learned_child_routers: {
      token: childToken.selected_consensus.final,
      span: childSpan.selected_consensus.final,
      best: bestChild,
    },
    recursive_router_of_routers: {
      token: recursiveToken.final,
      span: recursiveSpan.final,
      best: bestRecursive,
    },
    expected_regret_router: {
      objective: "direct gradient of expected child loss",
      selected_view: "span-hidden-b",
      regret_gradient_shift: bestRegretRouter.regret_gradient_shift,
      epochs: bestRegretRouter.epochs,
      calibration: bestRegretRouter.metrics,
      final: bestRegretRouter.final_metrics,
      delta_vs_fixed_q15:
        bestRegretRouter.final_metrics.probability_error_q15
          - finalBestFixed.probability_error_q15,
    },
  },
  generation_gate: {
    expert: `cluster-${final.best_fixed_expert}`,
    samples: generation,
    pass: generation.every((sample) => sample.pass),
  },
  decision: {
    short_run_expert_curriculum_promoted:
      finalBestFixed.probability_error_q15 < parentBestFixed.probability_error_q15,
    selected_checkpoint: `stage-2 cluster-${final.best_fixed_expert}`,
    learned_router_promoted_over_fixed:
      bestChild.metrics.probability_error_q15 < finalBestFixed.probability_error_q15,
    recursive_router_promoted_over_fixed:
      bestRecursive.metrics.probability_error_q15 < finalBestFixed.probability_error_q15,
    expected_regret_router_promoted_over_fixed:
      bestRegretRouter.final_metrics.probability_error_q15
        < finalBestFixed.probability_error_q15,
    prose_promoted: generation.every((sample) => sample.pass),
    evidence: {
      parent_best_fixed_delta_vs_trunk_q15:
        parentBestFixed.probability_error_q15 - trunk.probability_error_q15,
      curriculum_best_fixed_delta_vs_trunk_q15:
        finalBestFixed.probability_error_q15 - trunk.probability_error_q15,
      fixed_gain_vs_parent_q15:
        parentBestFixed.probability_error_q15 - finalBestFixed.probability_error_q15,
      parent_token_oracle_delta_vs_trunk_q15:
        parentFinal.oracle_routes.token.probability_error_q15 - trunk.probability_error_q15,
      curriculum_token_oracle_delta_vs_trunk_q15:
        final.oracle_routes.token.probability_error_q15 - trunk.probability_error_q15,
      curriculum_token_oracle_gain_beyond_best_fixed_q15:
        finalBestFixed.probability_error_q15 - final.oracle_routes.token.probability_error_q15,
      best_child_delta_vs_fixed_q15:
        bestChild.metrics.probability_error_q15 - finalBestFixed.probability_error_q15,
      best_recursive_delta_vs_fixed_q15:
        bestRecursive.metrics.probability_error_q15 - finalBestFixed.probability_error_q15,
      expected_regret_delta_vs_fixed_q15:
        bestRegretRouter.final_metrics.probability_error_q15
          - finalBestFixed.probability_error_q15,
      expected_regret_improvement_vs_classification_router_q15:
        bestChild.metrics.probability_error_q15
          - bestRegretRouter.final_metrics.probability_error_q15,
    },
    interpretation:
      "small independently gated expert continuations generalize and more than double the calibration routing ceiling; direct expected-regret training preserves one-unit loss differences and closes most of the learned-router gap, but does not beat the fixed expert on final data",
    next_experiment:
      "replace lossy four-channel hidden averaging with target-blind signed projections of the full contextual state, retain direct expected-regret training, then distill decisions into a block-local top-one or top-two dispatcher",
  },
  known_non_claims: [
    "unchanged_next_byte_mistake_count",
    "target_aware_oracles_are_ceiling_measurements_only",
    "generation_still_fails_prose_gate",
    "does_not_claim_llm_quality",
  ],
};

fs.writeFileSync(path.join(root, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.decision));

function candidateScores(stage, ids) {
  return ids.map((id) => {
    const score = read(`scores/stage${stage}-${id}.json`);
    return { id, ...routeSummary(score) };
  });
}

function routeSummary(score) {
  const bestFixed = score.fixed_experts[score.best_fixed_expert];
  return {
    fixed_probability_error_q15: score.fixed_experts.map((row) => row.probability_error_q15),
    best_fixed_expert: score.best_fixed_expert,
    best_fixed_probability_error_q15: bestFixed.probability_error_q15,
    prompt_oracle_probability_error_q15: score.oracle_routes.prompt.probability_error_q15,
    span_oracle_probability_error_q15: score.oracle_routes.span.probability_error_q15,
    token_oracle_probability_error_q15: score.oracle_routes.token.probability_error_q15,
    token_oracle_gain_beyond_best_fixed_q15:
      bestFixed.probability_error_q15 - score.oracle_routes.token.probability_error_q15,
    token_oracle_utilization: score.oracle_routes.token.utilization_tokens,
  };
}

function generationMetrics(id) {
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
    } else {
      run = 0;
    }
  }
  const metrics = {
    id,
    bytes: bytes.length,
    non_space_per_mille: Math.floor(nonSpace * 1_000 / bytes.length),
    alphabetic_per_mille: Math.floor(alphabetic * 1_000 / bytes.length),
    distinct_bytes: new Set(bytes).size,
    longest_space_run: longest,
  };
  metrics.pass = metrics.non_space_per_mille >= 500
    && metrics.alphabetic_per_mille >= 300
    && metrics.distinct_bytes >= 15
    && metrics.longest_space_run <= 16;
  return metrics;
}

function bestByError(candidates) {
  return [...candidates].sort((left, right) =>
    left.metrics.probability_error_q15 - right.metrics.probability_error_q15
    || left.id.localeCompare(right.id))[0];
}

function delta(metrics, reference) {
  return {
    probability_error_q15:
      metrics.probability_error_q15 - reference.probability_error_q15,
    mean_probability_error_q15:
      metrics.mean_probability_error_q15 - reference.mean_probability_error_q15,
    mistakes: metrics.mistakes - reference.mistakes,
    accuracy_per_mille: metrics.accuracy_per_mille - reference.accuracy_per_mille,
  };
}

function read(relative) {
  return JSON.parse(fs.readFileSync(path.join(root, relative), "utf8"));
}

function readParent(relative) {
  return JSON.parse(fs.readFileSync(path.join(parentRoot, relative), "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
