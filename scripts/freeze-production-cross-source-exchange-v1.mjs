#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const framePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const predictorPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-predictor.json";
const rawStructureContractPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-calibration-evaluation-structure-contract.json";
const outputPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json";
const analyzerPath = new URL("./analyze-production-cross-source-exchange-v1.mjs", import.meta.url);
const checkerPath = new URL("./check-production-cross-source-exchange-v1.mjs", import.meta.url);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const read = (file) => fs.readFileSync(file);

const frameBytes = read(framePath);
const predictorBytes = read(predictorPath);
const rawContractBytes = read(rawStructureContractPath);
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
const rawContract = JSON.parse(rawContractBytes);
assert(frame.schema === "nsrl.production_cross_source_exchange_source_frame.v1"
  && frame.analysis_role === "prospective_pre_outcome_source_frame"
  && frame.outcome_firewall.action_cube_outcomes_read === false,
"source frame is not prospectively frozen");
assert(predictor.schema === "nsrl.production_cross_source_exchange_predictor.v1"
  && predictor.analysis_role === "fitting_only_frozen_before_calibration_evaluation"
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"predictor crossed the calibration/evaluation firewall");
assert(rawContract.schema === "nsrl.production_atomic_structure_contract.v1"
  && rawContract.surface.document_start === 8
  && rawContract.surface.documents === 64
  && rawContract.surface.windows_per_document === 2,
"raw calibration/evaluation structure contract changed");
const fitting = frame.sources.filter((source) => source.role === "fitting");
const calibration = frame.sources.filter((source) => source.role === "calibration");
const evaluation = frame.sources.filter((source) => source.role === "evaluation");
assert(fitting.length === 16 && calibration.length === 39 && evaluation.length === 16,
  "source role counts changed");
assert(new Set(frame.sources.map((source) => source.author_key)).size === 71,
  "source authors are not distinct");
assert(predictor.training_population.independent_source_panels === fitting.length
  && predictor.fitted_rows.every((row) => fitting.some(
    (source) => source.source_id === row.source_id)), "predictor has non-fitting data");
assert(rawContract.bindings.token_stream_hash === JSON.parse(read(
  "data/processed/production-cross-source-exchange-v1/calibration-evaluation.tokens.json"
)).token_hash, "raw contract token stream changed");

const contract = {
  schema: "nsrl.production_cross_source_exchange_contract.v1",
  analysis_role: "prospective_pre_calibration_evaluation_outcome",
  theory_binding: {
    journal_entry: "MJ-2026-07-15-16",
    propositions: ["16.1", "16.2", "16.4", "16.5"],
    guarantee:
      "marginal simultaneous source-panel residual coverage and marginal unsafe-action control conditional on exchangeability",
    conditional_correctness_among_firings_claimed: false,
  },
  bindings: {
    source_frame: {path: framePath, sha256: sha256(frameBytes)},
    predictor: {path: predictorPath, sha256: sha256(predictorBytes)},
    raw_structure_contract: {
      path: rawStructureContractPath,
      sha256: sha256(rawContractBytes),
      manifest_hash: rawContract.manifest_hash,
      token_stream_hash: rawContract.bindings.token_stream_hash,
      source_index_hash: rawContract.bindings.source_index_hash,
    },
    analyzer: {
      path: "scripts/analyze-production-cross-source-exchange-v1.mjs",
      sha256: sha256(read(analyzerPath)),
    },
    checker: {
      path: "scripts/check-production-cross-source-exchange-v1.mjs",
      sha256: sha256(read(checkerPath)),
    },
  },
  population: {
    source_unit: frame.source_definition.unit,
    intended_population: frame.source_definition.intended_population,
    independence_design:
      "one ebook per distinct normalized author; entire sources assigned to exactly one role",
    exchangeability_assumption:
      "calibration and evaluation panels are exchangeable within the frozen distinct-author English Project Gutenberg frame conditional on the fitting sources",
    fitting_source_panels: fitting.length,
    calibration_source_panels: calibration.length,
    evaluation_source_panels: evaluation.length,
    fitting_source_ids: fitting.map((source) => source.source_id),
    calibration_source_ids: calibration.map((source) => source.source_id),
    evaluation_source_ids: evaluation.map((source) => source.source_id),
    documents_200_212_are_not_sources: true,
  },
  panel_sampling: {
    ...frame.panel_sampling,
    score_scope: "maximum over every frozen exchange and panel document",
    source_failure_policy: "abort; do not replace or drop a frozen source after this contract",
  },
  exchange_set: [{
    id: "base43_atom2_to_atom4",
    base_mask: 43,
    outgoing_atom: 2,
    incoming_atom: 4,
    control_mask: 47,
    candidate_mask: 59,
    singleton_margin_q32: "loss(mask16)-loss(mask0) minus loss(mask4)-loss(mask0)",
    exact_contrast_q32: "loss(mask59)-loss(mask47)",
    interaction_residual_q32: "exact_contrast minus singleton_margin",
  }],
  probe_features: {
    ordered_features: predictor.probe_features.ordered_features,
    representation: predictor.probe_features.representation,
    candidate_multi_atom_outcomes_excluded: true,
    allowed_router_inputs: [
      "six Q32 singleton effects", "frozen fitted predictor", "frozen conformal correction",
    ],
  },
  predictor: {
    name: predictor.algorithm.name,
    fitting_source_panels: predictor.training_population.independent_source_panels,
    neighbors: predictor.algorithm.neighbors,
    distance: predictor.algorithm.distance,
    tie_break: predictor.algorithm.distance_tie_break,
    output: predictor.algorithm.prediction,
    q32_conversion: predictor.algorithm.output_conversion,
    fitted_parameters_sha256: sha256(predictorBytes),
  },
  conformal: {
    alpha_numerator: 1,
    alpha_denominator: 20,
    calibration_units: "independent source panels",
    calibration_source_panels: calibration.length,
    simultaneous_score:
      "A_u=max over panel documents d and frozen exchanges e of rho_d(e)-q_e(phi_d)",
    ties: "included on covered side",
    order_statistic_rank: 38,
    insufficient_resolution_rule: "positive_infinity_if_rank_exceeds_calibration_count",
    evaluation_may_not_recalibrate_threshold: true,
  },
  router: {
    selection: "minimum certified upper contrast among the frozen exchange set; exchange-id tie break",
    strict_firing_rule: "lambda_e+q_e(phi)+Q_alpha < 0",
    abstention_rule: "retain control mask 47 on equality or positive certified upper contrast",
    candidate_multi_atom_outcomes_hidden_until_after_action: true,
  },
  falsifiers: {
    coverage: {
      test: "one-sided exact binomial test of source-panel failure probability greater than 1/20",
      evaluation_source_panels: evaluation.length,
      exact_binomial_rejection_failures: 3,
      exact_binomial_alpha: "1/20",
      maximum_failures_for_support: 1,
      two_failures_decision: "inconclusive_no_promotion",
    },
    unsafe_action: {
      falsifier: "one or more fired evaluation source panels with nonnegative exact exchange contrast",
      maximum_unsafe_firings_for_support: 0,
    },
    nonvacuity: {
      falsifier: "no evaluation source panel fires",
      minimum_fired_source_panels: 1,
    },
    incremental_value: {
      falsifier: "aggregate Q32 contrast over fired evaluation source panels is nonnegative",
      strict_negative_aggregate_required: true,
    },
  },
  decision_rule: {
    support_requires:
      "at most one envelope failure, zero unsafe firings, at least one firing, and negative aggregate fired contrast",
    scope_if_supported:
      "bounded prospective source-transfer evidence on the frozen distinct-author English Project Gutenberg frame",
    scope_if_falsified: "record failure without changing thresholds or opening same-source documents 200--212",
  },
  authorization: {
    execute_calibration_evaluation_cube: true,
    alter_source_roles_after_outcomes: false,
    alter_exchange_features_predictor_score_or_falsifiers_after_outcomes: false,
    read_documents_200_212: false,
    optimizer_change: false,
    paid_scaling: false,
  },
};
const bytes = `${JSON.stringify(contract, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: contract.schema,
  output: outputPath,
  contract_sha256: sha256(Buffer.from(bytes)),
  fitting_source_panels: fitting.length,
  calibration_source_panels: calibration.length,
  untouched_evaluation_source_panels: evaluation.length,
  conformal_order_statistic_rank: contract.conformal.order_statistic_rank,
  calibration_evaluation_outcomes_read: false,
  documents_200_212_read: false,
}, null, 2)}\n`);
