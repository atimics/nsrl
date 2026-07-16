#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const framePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json";
const predictorPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-predictor.json";
const rawStructureContractPaths = process.argv.slice(4, 12);
const outputPath = process.argv[12]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json";
const defaultRawPaths = Array.from({length: 4}, (_, passageOrdinal) =>
  Array.from({length: 2}, (_, shard) =>
    `benchmarks/production-model-v1/p10m-multifamily-exchange-v1-confirmation-passage-${passageOrdinal}-shard-${shard}-structure-contract.json`)).flat();
const rawPaths = rawStructureContractPaths.length === 8 ? rawStructureContractPaths : defaultRawPaths;
const analyzerPath = new URL("./analyze-production-multifamily-exchange-v1.mjs", import.meta.url);
const checkerPath = new URL("./check-production-multifamily-exchange-v1.mjs", import.meta.url);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const read = (file) => fs.readFileSync(file);

const frameBytes = read(framePath);
const predictorBytes = read(predictorPath);
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
assert(frame.schema === "nsrl.production_multifamily_exchange_source_frame.v1"
  && frame.analysis_role === "prospective_pre_outcome_source_frame"
  && frame.outcome_firewall.action_cube_outcomes_read === false,
"source frame is not prospectively frozen");
assert(predictor.schema === "nsrl.production_multifamily_exchange_predictor.v1"
  && predictor.analysis_role === "fitting_only_frozen_before_calibration_evaluation"
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"predictor crossed the calibration/evaluation firewall");
const rawContracts = rawPaths.map((rawPath, bindingIndex) => {
  const passageOrdinal = Math.floor(bindingIndex / 2);
  const shard = bindingIndex % 2;
  const bytes = read(rawPath);
  const value = JSON.parse(bytes);
  assert(value.schema === "nsrl.production_atomic_structure_contract.v1"
    && value.surface.document_start === 8 && value.surface.documents === 64
    && value.surface.windows_per_document === 2,
  `passage ${passageOrdinal} raw structure contract changed`);
  const tokenTracePath = `data/processed/production-multifamily-exchange-v1/confirmation-passage-${passageOrdinal}-shard-${shard}.tokens.json`;
  assert(value.bindings.token_stream_hash === JSON.parse(read(tokenTracePath)).token_hash,
    `passage ${passageOrdinal} token stream changed`);
  return {bytes, value, path: rawPath, token_trace_path: tokenTracePath};
});
const fitting = frame.sources.filter((source) => source.role === "fitting");
const calibration = frame.sources.filter((source) => source.role === "calibration");
const evaluation = frame.sources.filter((source) => source.role === "evaluation");
const families = frame.source_definition.family_order;
assert(fitting.length === 12 && calibration.length === 76 && evaluation.length === 16,
  "source role counts changed");
for (const family of families) {
  const familySources = frame.sources.filter((source) => source.family === family);
  assert(familySources.length === 26
    && familySources.filter((source) => source.role === "fitting").length === 3
    && familySources.filter((source) => source.role === "calibration").length === 19
    && familySources.filter((source) => source.role === "evaluation").length === 4
    && new Set(familySources.map((source) => source.independence_key)).size === 26,
  `${family} stratified source frame changed`);
}
assert(predictor.training_population.independent_source_panels === fitting.length
  && predictor.training_population.passage_documents === 48
  && predictor.fitted_rows.every((row) => fitting.some(
    (source) => source.source_id === row.source_id)), "predictor has non-fitting data");

const contract = {
  schema: "nsrl.production_multifamily_exchange_contract.v1",
  analysis_role: "prospective_pre_calibration_evaluation_outcome",
  theory_binding: {
    journal_entry: "MJ-2026-07-15-16",
    propositions: ["16.1", "16.2", "16.4", "16.5"],
    guarantee: "marginal simultaneous source-panel residual coverage and marginal unsafe-action control conditional on source-panel exchangeability",
    conditional_correctness_among_firings_claimed: false,
  },
  bindings: {
    source_frame: {path: framePath, sha256: sha256(frameBytes)},
    predictor: {path: predictorPath, sha256: sha256(predictorBytes)},
    raw_structure_contracts: rawContracts.map((raw, bindingIndex) => ({
      passage_ordinal: Math.floor(bindingIndex / 2), shard: bindingIndex % 2,
      path: raw.path, sha256: sha256(raw.bytes),
      manifest_hash: raw.value.manifest_hash,
      token_stream_hash: raw.value.bindings.token_stream_hash,
      source_index_hash: raw.value.bindings.source_index_hash,
      result_path:
        `benchmarks/production-model-v1/p10m-multifamily-exchange-v1-confirmation-passage-${Math.floor(bindingIndex / 2)}-shard-${bindingIndex % 2}-structure.json`,
    })),
    analyzer: {
      path: "scripts/analyze-production-multifamily-exchange-v1.mjs",
      sha256: sha256(read(analyzerPath)),
    },
    checker: {
      path: "scripts/check-production-multifamily-exchange-v1.mjs",
      sha256: sha256(read(checkerPath)),
    },
  },
  population: {
    source_unit: frame.source_definition.unit,
    intended_population: frame.source_definition.intended_population,
    families,
    independence_design:
      "within-family unique author, first-listed author, most-specific agency, or first-author/journal key; whole publications assigned to exactly one role",
    exchangeability_assumption:
      "calibration and evaluation source panels are exchangeable within each stratum of the frozen four-family acquisition frame conditional on fitting sources",
    stratified_role_counts_per_family: {fitting: 3, calibration: 19, evaluation: 4},
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
    score_scope: "maximum over all four frozen nonoverlapping passage documents and every frozen exchange",
    source_failure_policy: "abort; do not replace or drop a frozen source or passage after this contract",
  },
  exchange_set: [{
    id: "base43_atom2_to_atom4", base_mask: 43, outgoing_atom: 2, incoming_atom: 4,
    control_mask: 47, candidate_mask: 59,
    singleton_margin_q32: "loss(mask16)-loss(mask0) minus loss(mask4)-loss(mask0)",
    exact_contrast_q32: "loss(mask59)-loss(mask47)",
    interaction_residual_q32: "exact_contrast minus singleton_margin",
  }],
  probe_features: {
    ordered_features: predictor.probe_features.ordered_features,
    representation: predictor.probe_features.representation,
    candidate_multi_atom_outcomes_excluded: true,
    allowed_router_inputs: [
      "six Q32 singleton effects", "frozen fitting-source predictor", "frozen conformal correction",
    ],
  },
  predictor: {
    name: predictor.algorithm.name,
    fitting_source_panels: predictor.training_population.independent_source_panels,
    fitting_passage_documents: predictor.training_population.passage_documents,
    neighbors: predictor.algorithm.neighbors,
    within_source_match: predictor.algorithm.within_source_match,
    distance: predictor.algorithm.distance,
    within_source_tie_break: predictor.algorithm.within_source_tie_break,
    source_tie_break: predictor.algorithm.source_tie_break,
    output: predictor.algorithm.prediction,
    q32_conversion: predictor.algorithm.output_conversion,
    fitted_parameters_sha256: sha256(predictorBytes),
  },
  conformal: {
    alpha_numerator: 1, alpha_denominator: 20,
    calibration_units: "independent whole-source panels calibrated separately within each of four families",
    calibration_source_panels: calibration.length,
    simultaneous_score:
      "A_u=max over the four passage documents d and frozen exchanges e of rho_d(e)-q_e(phi_d)",
    ties: "included on covered side",
    order_statistic_rank_per_family: 19,
    insufficient_resolution_rule: "positive_infinity_if_rank_exceeds_calibration_count",
    evaluation_may_not_recalibrate_threshold: true,
  },
  router: {
    selection: "minimum certified upper contrast among the frozen exchange set; exchange-id tie break",
    strict_firing_rule: "for each passage, lambda_e+q_e(phi)+Q_alpha,family < 0",
    abstention_rule: "retain control mask 47 on equality or positive certified upper contrast",
    candidate_multi_atom_outcomes_hidden_until_after_action: true,
  },
  sequence: {
    unit: "one fired conditional parameter exchange on one untouched evaluation passage",
    proposal_order_seed: "nsrl-m4-heldout-exchange-sequence-2026-07-15-v1",
    proposal_order:
      "ascending SHA256(seed, family, source_id, passage_ordinal), fixed before outcomes",
    state_scope:
      "each proposal compares candidate mask 59 with control mask 47 on its passage-local frozen model state; the next proposal cannot use prior held-out outcomes",
    minimum_fired_exchanges_for_support: 8,
    cumulative_net_improvement:
      "negative cumulative sum of exact fired candidate-minus-control Q32 contrasts",
  },
  falsifiers: {
    coverage: {
      test: "one-sided exact binomial test of source-panel failure probability greater than 1/20",
      evaluation_source_panels: evaluation.length,
      evaluation_source_panels_per_family: 4,
      exact_binomial_rejection_failures: 3,
      exact_binomial_alpha: "1/20",
      maximum_failures_for_support: 1,
      maximum_failures_per_family_for_support: 1,
      two_failures_decision: "inconclusive_no_promotion",
    },
    unsafe_action: {
      falsifier: "one or more fired evaluation passages with nonnegative exact exchange contrast",
      maximum_unsafe_firings_for_support: 0,
    },
    nonvacuity: {
      falsifier: "fewer than eight evaluation passages fire or firings occur in fewer than two source families",
      minimum_fired_passages: 8,
      minimum_firing_families: 2,
    },
    incremental_value: {
      falsifier: "aggregate fired contrast is nonnegative overall or within any family containing a firing",
      strict_negative_aggregate_required: true,
      strict_negative_aggregate_in_every_firing_family_required: true,
    },
    family_promotion: {
      rule:
        "a source family may be promoted only when it fires, has no unsafe firing, has negative aggregate fired contrast, and remains within its frozen coverage gate",
      falsifier: "one or more source families with an unsafe firing is marked promoted",
      maximum_unsafe_promoted_families: 0,
    },
  },
  decision_rule: {
    support_requires:
      "at most one total envelope failure and at most one per family, zero unsafe passage firings, at least eight ordered conditional exchanges across at least two families, no unsafe family promotion, and strictly positive net held-out improvement overall and within every firing family",
    scope_if_supported:
      "bounded prospective multi-family, four-passage source-transfer evidence on the frozen acquisition frame",
    scope_if_falsified:
      "record failure without changing thresholds, families, passages, or opening same-source documents 200--212",
  },
  authorization: {
    execute_calibration_evaluation_cubes: true,
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
  schema: contract.schema, output: outputPath,
  contract_sha256: sha256(Buffer.from(bytes)), families,
  fitting_source_panels: fitting.length, calibration_source_panels: calibration.length,
  untouched_evaluation_source_panels: evaluation.length,
  passages_per_source_panel: frame.panel_sampling.passage_documents_per_source,
  conformal_order_statistic_rank_per_family: contract.conformal.order_statistic_rank_per_family,
  calibration_evaluation_outcomes_read: false, documents_200_212_read: false,
}, null, 2)}\n`);
