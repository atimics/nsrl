#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2]
  ?? "data/experiments/literary-h8-gradient-block-wide-router-v1";
const parentRoot = process.argv[3]
  ?? "data/experiments/literary-h8-gradient-block-curriculum-v1";
const projectedRoot = process.argv[4]
  ?? "data/experiments/literary-h8-gradient-block-projected-router-shift7-v1";

const parent = readAt(parentRoot, "report.json");
const projected = readAt(projectedRoot, "report.json");
const calibration = read("oracles/router-calibration.json");
const final = read("oracles/final-test.json");
const childSweep = read("child-regret-router-sweep-selection.json");
const rootSweep = read("root-regret-router-sweep-selection.json");
const fixedCalibration = calibration.fixed_experts[calibration.best_fixed_expert];
const fixedFinal = final.fixed_experts[final.best_fixed_expert];
const childToken = childSweep.selected["token-hidden-a"];
const childSpan = childSweep.selected["span-hidden-b"];
const rootToken = rootSweep.selected["token-recursive"];
const rootSpan = rootSweep.selected["span-recursive"];

const report = {
  schema: "nsrl.literary_h8_wide_router.v1",
  architecture: {
    artifact: "NSRLRT2",
    input_features: 137,
    contextual_hidden_channels: 128,
    prior_token_probe_features: 9,
    hidden_width: 32,
    outputs: 3,
    integer_weights: "i8",
    integer_activations: "Q15",
    objective: "expected_regret",
    deterministic_epoch_shuffle: true,
    backward_compatibility:
      "NSRLRT1 41x16 artifacts, traces, and predictions replay byte-for-byte",
  },
  frozen_experts: {
    unchanged_from_parent: true,
    best_fixed_final: fixedFinal,
    token_oracle_final: final.oracle_routes.token,
    token_oracle_gain_beyond_fixed_q15:
      fixedFinal.probability_error_q15 - final.oracle_routes.token.probability_error_q15,
  },
  training_policy: {
    full_epoch_result: "all expected-regret replicas collapse to fixed expert 1",
    hard_label_result: "all class-balanced replicas collapse to expert 0",
    early_stop_sweep_rows_per_epoch: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    selection_split: "router calibration only",
    final_split_used_for_selection: false,
  },
  selected_children: {
    token: routeEvidence(childToken, fixedCalibration, fixedFinal),
    span: routeEvidence(childSpan, fixedCalibration, fixedFinal),
  },
  recursive_router_of_routers: {
    architecture:
      "three 137x32x3 NSRLRT2 child routers feed a 137x32x3 NSRLRT2 root",
    token: routeEvidence(rootToken, fixedCalibration, fixedFinal),
    span: routeEvidence(rootSpan, fixedCalibration, fixedFinal),
  },
  comparisons: {
    pooled_expected_regret_final_delta_vs_fixed_q15:
      parent.decision.evidence.expected_regret_delta_vs_fixed_q15,
    projected_expected_regret_final_delta_vs_fixed_q15:
      projected.decision.evidence.selected_final_delta_vs_fixed_q15,
    wide_child_span_final_delta_vs_fixed_q15:
      childSpan.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
    wide_recursive_final_delta_vs_fixed_q15:
      rootSpan.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
  },
  generation_gate: {
    inherited_from_unchanged_expert: true,
    parent: parent.generation_gate,
    pass: parent.generation_gate.pass,
  },
  decision: {
    nsrlrt2_runtime_validated: true,
    wide_child_router_promoted_over_fixed: false,
    wide_recursive_router_promoted_over_fixed: false,
    prose_promoted: false,
    evidence: {
      child_token_calibration_delta_vs_fixed_q15:
        childToken.metrics.probability_error_q15 - fixedCalibration.probability_error_q15,
      child_token_final_delta_vs_fixed_q15:
        childToken.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      child_span_calibration_delta_vs_fixed_q15:
        childSpan.metrics.probability_error_q15 - fixedCalibration.probability_error_q15,
      child_span_final_delta_vs_fixed_q15:
        childSpan.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      recursive_token_final_delta_vs_fixed_q15:
        rootToken.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      recursive_span_final_delta_vs_fixed_q15:
        rootSpan.final_metrics.probability_error_q15 - fixedFinal.probability_error_q15,
      token_oracle_gain_beyond_fixed_q15:
        fixedFinal.probability_error_q15 - final.oracle_routes.token.probability_error_q15,
    },
    interpretation:
      "uncompressed hidden state and doubled router width expose a fragile early-stopping signal but do not generalize; the recursive root safely rejects child switches and ties fixed",
    next_experiment:
      "replace post-hoc route classification with a block-local top-two gate trained jointly with experts on next-token loss, using deterministic load balance, switching cost, and frozen-trunk ablations",
  },
  known_non_claims: [
    "wide_router_does_not_beat_fixed_expert",
    "target_aware_oracle_is_a_ceiling_only",
    "generation_still_fails_prose_gate",
    "does_not_claim_llm_quality",
  ],
};

fs.writeFileSync(path.join(root, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report.decision));

function routeEvidence(route, fixedCalibrationMetrics, fixedFinalMetrics) {
  return {
    regret_gradient_shift: route.regret_gradient_shift,
    epochs: route.epochs,
    max_train_rows: route.max_train_rows,
    calibration: route.metrics,
    final: route.final_metrics,
    delta_vs_fixed: {
      calibration_q15:
        route.metrics.probability_error_q15 - fixedCalibrationMetrics.probability_error_q15,
      final_q15:
        route.final_metrics.probability_error_q15 - fixedFinalMetrics.probability_error_q15,
    },
  };
}

function read(relative) {
  return readAt(root, relative);
}

function readAt(directory, relative) {
  return JSON.parse(fs.readFileSync(path.join(directory, relative), "utf8"));
}
