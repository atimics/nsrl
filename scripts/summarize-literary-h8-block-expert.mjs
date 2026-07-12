#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-block-expert-v1";
const baselinePath = process.argv[3]
  ?? "data/experiments/literary-h8-curriculum-v1/stage1-offset0/holdout.json";

const specs = [
  ["scaled-s4-lr1", "stable"],
  ["scaled-s4-lr4", "stable"],
  ["scaled-s4-lr16", "stable"],
  ["scaled-s8-lr256", "stable"],
  ["scaled-s8-lr1024", "stable"],
  ["scaled-s8-lr4096", "stable"],
  ["rank32-s8-lr256", "capacity_probe"],
  ["rank32-s8-lr1024", "capacity_probe"],
];

const baselineTrace = readJson(baselinePath);
const baseline = {
  model_hash: baselineTrace.model.hash,
  windows: baselineTrace.data.windows,
  mistakes: baselineTrace.evaluation.mistakes,
  accuracy_per_mille: baselineTrace.evaluation.accuracy_per_mille,
  probability_error_q15: baselineTrace.evaluation.probability_error_q15,
  mean_probability_error_q15: baselineTrace.evaluation.mean_probability_error_q15,
};

const trials = specs.map(([directory, role]) => {
  const train = readJson(path.join(root, directory, "train.json"));
  const holdout = readJson(path.join(root, directory, "holdout.json"));
  return {
    id: directory,
    role,
    rank: train.config.rank,
    residual_shift: train.config.residual_shift,
    learning_rate: train.config.learning_rate,
    parameter_count: train.parameter_count,
    train_initial: train.initial,
    train_final: train.final,
    updates: train.updates,
    holdout: holdout.metrics,
    holdout_probability_delta_vs_trunk:
      holdout.metrics.probability_error_q15 - baseline.probability_error_q15,
    holdout_mistake_delta_vs_trunk: holdout.metrics.mistakes - baseline.mistakes,
    promoted:
      holdout.metrics.probability_error_q15 < baseline.probability_error_q15
      && holdout.metrics.mistakes <= baseline.mistakes,
  };
});

const generation = ["love", "law", "soul"].map((id) => {
  const bytes = fs.readFileSync(path.join(root, "generation", `${id}-top8.txt`));
  const nonSpace = [...bytes].filter((byte) => byte !== 32).length;
  const alphabetic = [...bytes].filter(
    (byte) => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122),
  ).length;
  let longestSpaceRun = 0;
  let spaceRun = 0;
  for (const byte of bytes) {
    if (byte === 32) {
      spaceRun += 1;
      longestSpaceRun = Math.max(longestSpaceRun, spaceRun);
    } else {
      spaceRun = 0;
    }
  }
  const metrics = {
    id,
    bytes: bytes.length,
    non_space_per_mille: Math.floor(nonSpace * 1000 / bytes.length),
    alphabetic_per_mille: Math.floor(alphabetic * 1000 / bytes.length),
    distinct_bytes: new Set(bytes).size,
    longest_space_run: longestSpaceRun,
  };
  metrics.pass = metrics.non_space_per_mille >= 500
    && metrics.alphabetic_per_mille >= 300
    && metrics.distinct_bytes >= 15
    && metrics.longest_space_run <= 16;
  return metrics;
});

const closest = trials.toSorted((left, right) =>
  left.holdout_probability_delta_vs_trunk - right.holdout_probability_delta_vs_trunk
  || left.holdout_mistake_delta_vs_trunk - right.holdout_mistake_delta_vs_trunk
)[0];

const report = {
  schema: "nsrl.literary_h8_block_low_rank_expert_experiment.v1",
  architecture: {
    trunk: "frozen H8 d128 ff256 two-block integer transformer",
    expert: "per-block fixed-sign down projection plus learned i16 Q15 expansion",
    inference: "integer trunk plus small residual artifact",
    routing_fit: "one artifact per author/span/token cluster; router work remains",
  },
  baseline,
  trials,
  closest_non_promoted_trial: closest.id,
  generation_gate: {
    trial: "scaled-s8-lr256",
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
    mechanism_validated: true,
    checkpoint_promoted: trials.some((trial) => trial.promoted),
    reason: "stable internal experts learned the training slice, but no rank/rate point improved exact untouched holdout and sampled prose still failed",
    next_experiment: "train many provenance-labelled author/span experts and a target-blind hierarchical router; expand authentic Blake data before increasing the shared trunk",
  },
};

fs.writeFileSync(path.join(root, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report));

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}
