#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2]
  ?? "data/experiments/literary-h8-gradient-block-projected-router-shift7-v1";
const clippedRoot = process.argv[3]
  ?? "data/experiments/literary-h8-gradient-block-projected-router-v1";
const seed2Root = process.argv[4]
  ?? "data/experiments/literary-h8-gradient-block-projected-router-seed2-v1";
const parentRoot = process.argv[5]
  ?? "data/experiments/literary-h8-gradient-block-curriculum-v1";

const parent = readAt(parentRoot, "report.json");
const calibration = read("oracles/router-calibration.json");
const final = read("oracles/final-test.json");
const selected = read("regret-router-sweep-selection.json");
const clippedSelection = readAt(clippedRoot, "regret-router-sweep-selection.json");
const seed2Selection = readAt(seed2Root, "regret-router-sweep-selection.json");
const fixedCalibration = calibration.fixed_experts[calibration.best_fixed_expert];
const fixedFinal = final.fixed_experts[final.best_fixed_expert];
const projected = selected.selected["span-hidden-a"];
const recursiveToken = selected.selected["token-recursive"];
const recursiveSpan = selected.selected["span-recursive"];
const seed2Best = bestSelection(seed2Selection.selected);
const clippedBest = bestSelection(clippedSelection.selected);

const report = {
  schema: "nsrl.literary_h8_signed_projected_router.v1",
  frozen_experts: {
    source: path.resolve(parentRoot, "experts"),
    unchanged_from_parent: true,
    best_fixed_final: fixedFinal,
    token_oracle_final: final.oracle_routes.token,
    token_oracle_gain_beyond_fixed_q15:
      fixedFinal.probability_error_q15 - final.oracle_routes.token.probability_error_q15,
  },
  feature_experiment: {
    prior_kind: "contiguous means over each four adjacent hidden channels",
    candidate_kind: "32 deterministic signed projections over all 128 contextual channels",
    projection_seed: calibration.router_features.projection_seed,
    selected_projection_shift: calibration.router_features.projection_shift,
    scale_gate: "lowest tested shift with no i16 feature saturation on router calibration",
    clipped_shift_4: {
      saturated_calibration_rows: countSaturatedRows(
        path.join(clippedRoot, "oracles", "router-calibration-details.tsv"),
      ),
      calibration_rows: clippedBest.metrics.windows,
      best_route: clippedBest,
    },
    selected_shift_7: {
      saturated_calibration_rows: countSaturatedRows(
        path.join(root, "oracles", "router-calibration-details.tsv"),
      ),
      calibration_rows: projected.metrics.windows,
    },
    independent_seed_2: {
      final_split_scored: false,
      best_calibration_route: seed2Best,
    },
  },
  selected_child: {
    id: "seed-1 shift-7 span-hidden-a expected-regret",
    regret_gradient_shift: projected.regret_gradient_shift,
    epochs: projected.epochs,
    calibration: projected.metrics,
    final: projected.final_metrics,
    delta_vs_fixed: {
      calibration_q15:
        projected.metrics.probability_error_q15 - fixedCalibration.probability_error_q15,
      final_q15:
        projected.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
    },
  },
  recursive_router_of_routers: {
    architecture: "three expected-regret child routers feeding a 41x16x3 expected-regret root",
    token: recursiveToken,
    span: recursiveSpan,
  },
  comparison_to_pooled: {
    pooled_expected_regret_delta_vs_fixed_q15:
      parent.decision.evidence.expected_regret_delta_vs_fixed_q15,
    projected_expected_regret_delta_vs_fixed_q15:
      projected.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
    improvement_q15:
      parent.decision.evidence.expected_regret_delta_vs_fixed_q15
        - (projected.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15),
  },
  generation_gate: {
    inherited_from_unchanged_expert: true,
    parent: parent.generation_gate,
    pass: parent.generation_gate.pass,
  },
  decision: {
    signed_projection_runtime_validated: true,
    signed_projection_checkpoint_promoted_over_fixed: false,
    recursive_router_promoted_over_fixed: false,
    prose_promoted: false,
    evidence: {
      clipped_calibration_rows: countSaturatedRows(
        path.join(clippedRoot, "oracles", "router-calibration-details.tsv"),
      ),
      selected_calibration_saturated_rows: countSaturatedRows(
        path.join(root, "oracles", "router-calibration-details.tsv"),
      ),
      selected_calibration_delta_vs_fixed_q15:
        projected.metrics.probability_error_q15 - fixedCalibration.probability_error_q15,
      selected_final_delta_vs_fixed_q15:
        projected.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      recursive_token_final_delta_vs_fixed_q15:
        recursiveToken.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      recursive_span_final_delta_vs_fixed_q15:
        recursiveSpan.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      token_oracle_gain_beyond_fixed_q15:
        fixedFinal.probability_error_q15 - final.oracle_routes.token.probability_error_q15,
    },
    interpretation:
      "saturation-free signed projections remove the pooled router's small final regression, but both child and recursive routes fall back to the best fixed expert; compressing 128 channels into 32 remains insufficient",
    next_experiment:
      "add a versioned wider integer router that consumes all 128 contextual channels plus nine prior-token probes, increase hidden width from 16 to 32, and retain calibration-selected expected-regret training",
  },
  known_non_claims: [
    "target_aware_oracle_is_a_ceiling_only",
    "projected_router_does_not_beat_fixed_expert",
    "unchanged_experts_mean_generation_is_not_retrained",
    "does_not_claim_llm_quality",
  ],
};

fs.writeFileSync(path.join(root, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.decision));

function bestSelection(values) {
  return Object.entries(values)
    .filter(([, value]) => value?.metrics)
    .map(([id, value]) => ({ id, ...value }))
    .sort((left, right) =>
      left.metrics.probability_error_q15 - right.metrics.probability_error_q15
      || left.metrics.route_switches - right.metrics.route_switches
      || left.id.localeCompare(right.id))[0];
}

function countSaturatedRows(file) {
  return fs.readFileSync(file, "utf8").trimEnd().split("\n").slice(1)
    .filter((line) => line.split("\t")[5].split(",")
      .some((value) => value === "-32768" || value === "32767"))
    .length;
}

function read(relative) {
  return readAt(root, relative);
}

function readAt(directory, relative) {
  return JSON.parse(fs.readFileSync(path.join(directory, relative), "utf8"));
}
