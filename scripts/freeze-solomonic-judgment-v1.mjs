#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const framePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-source-frame.json";
const structureContractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-structure-contract.json";
const predictorPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-predictor.json";
const parentResultPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json";
const outputPath = process.argv[6]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json";

const analyzerPath = new URL("./analyze-solomonic-judgment-v1.mjs", import.meta.url);
const checkerPath = new URL("./check-solomonic-judgment-v1.mjs", import.meta.url);
const publisherPath = new URL("./publish-solomonic-judgment-v1.mjs", import.meta.url);
const publicationCheckerPath = new URL(
  "./check-solomonic-judgment-publication-v1.mjs", import.meta.url);
const preparerPath = new URL("./prepare-solomonic-judgment-v1.mjs", import.meta.url);
const freezerPath = new URL(import.meta.url);
const judgmentSchemaPath = "protocol/judgment-record-v1.schema.json";

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const entropy = (probability) => probability === 0 || probability === 1 ? 0
  : -probability * Math.log2(probability) - (1 - probability) * Math.log2(1 - probability);
const binding = (file) => ({path: file, sha256: sha256(fs.readFileSync(file))});

const frameBytes = fs.readFileSync(framePath);
const structureContractBytes = fs.readFileSync(structureContractPath);
const predictorBytes = fs.readFileSync(predictorPath);
const parentResultBytes = fs.readFileSync(parentResultPath);
const frame = JSON.parse(frameBytes);
const structureContract = JSON.parse(structureContractBytes);
const predictor = JSON.parse(predictorBytes);
const parentResult = JSON.parse(parentResultBytes);
assert(frame.schema === "nsrl.solomonic_judgment_source_frame.v1"
  && frame.analysis_role === "prospective_pre_outcome_source_frame"
  && frame.outcome_firewall.action_cube_outcomes_read === false,
"source frame is not prospective");
assert(structureContract.schema === "nsrl.production_atomic_structure_contract.v1"
  && structureContract.surface.document_start === 8
  && structureContract.surface.documents === 64
  && structureContract.surface.hard_stop_before_document === 72,
"wrong raw action-cube contract");
assert(predictor.schema === "nsrl.production_multifamily_exchange_predictor.v1"
  && predictor.analysis_role === "fitting_only_frozen_before_calibration_evaluation",
"wrong frozen predictor");
assert(parentResult.schema === "nsrl.production_multifamily_exchange_result.v1"
  && parentResult.analysis_role === "prospective_untouched_evaluation",
"wrong parent calibration/result artifact");

const occultSeed = "nsrl-m4-occult-hash-parity-2026-07-15-v1";
const occultRows = parentResult.calibration.rows.flatMap((panel) => panel.passages.map((passage) => ({
  bit: crypto.createHash("sha256").update(
    `${occultSeed}\0${panel.family}\0${panel.source_id}\0${passage.passage_ordinal}`).digest()[0] & 1,
  favorable: BigInt(passage.exchange_contrast_q32) < 0n ? 1 : 0,
})));
const favorable = occultRows.reduce((sum, row) => sum + row.favorable, 0);
const nullBits = occultRows.length * entropy(favorable / occultRows.length);
const groups = [0, 1].map((bit) => {
  const rows = occultRows.filter((row) => row.bit === bit);
  const successes = rows.reduce((sum, row) => sum + row.favorable, 0);
  return {bit, rows: rows.length, favorable: successes,
    empirical_code_bits: rows.length * entropy(successes / rows.length)};
});
const splitBits = groups.reduce((sum, group) => sum + group.empirical_code_bits, 0);
const bicPenaltyBits = 0.5 * Math.log2(occultRows.length);
const netGainBits = nullBits - splitBits - bicPenaltyBits;
const compressionThresholdBits = Math.log2(20);
const occultActivated = netGainBits >= compressionThresholdBits;

const corrections = Object.fromEntries(frame.population.families.map((family) => [
  family, parentResult.conformal.by_family[family].correction_q32,
]));
assert(Object.keys(corrections).length === 3
  && Object.values(corrections).every((value) => /^-?[0-9]+$/.test(value)),
"missing inherited family correction");
const faculties = [
  {
    id: "federal_register_faculty", kind: "domain_exchange",
    action_id: "federal_register_base43_atom2_to_atom4",
    action_family: "federal_register_exchange", source_family: "federal_register",
  },
  {
    id: "rfc_faculty", kind: "domain_exchange",
    action_id: "rfc_base43_atom2_to_atom4",
    action_family: "rfc_exchange", source_family: "rfc",
  },
  {
    id: "science_faculty", kind: "domain_exchange",
    action_id: "science_base43_atom2_to_atom4",
    action_family: "science_exchange", source_family: "science",
  },
  {
    id: "occult_hash_parity", kind: "symbolic_hypothesis",
    action_id: "occult_hash_parity_exchange", action_family: "occult_correspondence",
    source_family: "all",
  },
  {
    id: "abstention", kind: "abstention",
    action_id: "abstain", action_family: "abstention", source_family: "all",
  },
];
const contract = {
  schema: "nsrl.solomonic_judgment_contract.v1",
  analysis_role: "prospective_pre_outcome",
  frozen_at: "2026-07-15",
  question:
    "Can an adaptive evidence-bound controller select among three source-specialist faculties or abstain while preserving a 95% unsafe-regret boundary?",
  bindings: {
    source_frame: {path: framePath, sha256: sha256(frameBytes)},
    structure_contract: {path: structureContractPath, sha256: sha256(structureContractBytes)},
    predictor: {path: predictorPath, sha256: sha256(predictorBytes)},
    parent_result: {path: parentResultPath, sha256: sha256(parentResultBytes)},
    judgment_record_schema: binding(judgmentSchemaPath),
    preparer: {path: "scripts/prepare-solomonic-judgment-v1.mjs", sha256: sha256(fs.readFileSync(preparerPath))},
    freezer: {path: "scripts/freeze-solomonic-judgment-v1.mjs", sha256: sha256(fs.readFileSync(freezerPath))},
    analyzer: {path: "scripts/analyze-solomonic-judgment-v1.mjs", sha256: sha256(fs.readFileSync(analyzerPath))},
    checker: {path: "scripts/check-solomonic-judgment-v1.mjs", sha256: sha256(fs.readFileSync(checkerPath))},
    publisher: {path: "scripts/publish-solomonic-judgment-v1.mjs", sha256: sha256(fs.readFileSync(publisherPath))},
    publication_checker: {
      path: "scripts/check-solomonic-judgment-publication-v1.mjs",
      sha256: sha256(fs.readFileSync(publicationCheckerPath)),
    },
  },
  population: {
    families: frame.population.families,
    source_panels: frame.sources.length,
    passages_per_source: frame.panel_sampling.passages_per_source,
    sources_per_family: frame.population.sources_per_family,
    source_ids: frame.sources.map((source) => source.source_id),
    original_gutenberg_fitting_frame_excluded: true,
    intended_population: frame.population.intended_population,
    sequential_assumption:
      "for the e-process null, each predictably selected source-panel unsafe indicator has conditional mean at most 1/20 given the controller filtration",
    conformal_assumption:
      "each new source panel is exchangeable with the 19 inherited calibration source panels inside its declared family, conditional on fitting",
  },
  panel: {
    passages_per_source: frame.panel_sampling.passages_per_source,
    windows_per_passage: frame.panel_sampling.model_windows_per_passage,
    context_tokens: frame.panel_sampling.context_tokens,
  },
  action: {
    exchange_id: "base43_atom2_to_atom4", control_mask: 47, candidate_mask: 59,
    base_mask: 43, outgoing_atom: 2, incoming_atom: 4,
    intervention_cost: {coordinate_writes: 2, objective_penalty_q32: "0"},
    reversibility: {
      class: "exact", inverse_coordinate_writes: 2,
      condition: "before any intervening parameter update",
    },
  },
  faculties,
  conformal: {
    alpha: "1/20", calibration_source_panels_per_family: 19,
    simultaneous_scope: "maximum over four passages and the frozen exchange",
    corrections_q32: corrections,
    strict_action_rule: "lambda + predicted interaction residual + family correction < 0",
    abstention_on_equality: true,
  },
  controller: {
    source_order_seed: "nsrl-m4-solomonic-source-order-2026-07-15-v1",
    filtration:
      "source metadata, singleton probes, frozen predictor/correction, and prior judgment/e-process state; current candidate multi-atom outcome excluded",
    adaptive_rule:
      "after each complete source panel update the unsafe e-process; all later actions abstain once E reaches 20",
    e_process: {
      null: "Pr(source panel contains a fired non-improving action | prior controller filtration) <= 1/20",
      alternative: "1/4",
      safe_factor: "15/19", unsafe_factor: "5",
      update: "E_t=E_(t-1)*(5 if unsafe else 15/19)",
      ville_threshold: 20,
      type_i_error_bound: "1/20 anytime, including predictable action choice and stopping",
    },
  },
  numeric_contract: {
    q32_fractional_bits: 32,
    vocab_size: 8192,
    windows_per_passage: 2,
    q47_weight_max_exponent_bits: 47,
    maximum_nll_bits_per_window: 60,
    maximum_absolute_contrast_q32: (120n * (1n << 32n)).toString(),
    derivation:
      "target weight is at least 1 and 8192 Q47 weights sum to at most 2^60; two nonnegative window losses therefore differ by at most 120*2^32",
  },
  occult_feature: {
    id: "sha256_parity_correspondence", seed: occultSeed,
    hypothesis_role: "hypothesis_generation_only_until_compression_gate_passes",
    target: "sign of exact candidate-minus-control Q32 contrast",
    compression: {
      method: "binary empirical codelength reduction minus one-parameter BIC penalty",
      untouched_evaluation_outcomes_used: false,
      rows: occultRows.length, favorable: favorable, groups,
      null_code_bits: nullBits.toFixed(12), split_code_bits: splitBits.toFixed(12),
      bic_penalty_bits: bicPenaltyBits.toFixed(12), net_gain_bits: netGainBits.toFixed(12),
      activation_threshold_bits: compressionThresholdBits.toFixed(12),
    },
    activation: {
      activated: occultActivated,
      rule: "net compression gain must be at least log2(20) bits before untouched evaluation",
      status_if_inactive: "falsified",
      ordinary_falsifiers_apply_if_active: true,
    },
  },
  pass_conditions: {
    minimum_fired_passages: 2,
    minimum_transfer_families: 2,
    maximum_coverage_failures_for_support: 0,
    coverage_rejection_failures: 2,
    signed_regret_must_be_strictly_negative: true,
    signed_regret_in_every_firing_family_must_be_strictly_negative: true,
    positive_regret_must_remain_inside_e_process_95_bound: true,
    symbolic_features_receive_no_exemption: true,
  },
  publication_contract: {
    allowed_statuses: ["supported", "falsified", "inconclusive"],
    supported:
      "all pass conditions hold and no frozen falsifier fires",
    falsified:
      "the e-process or positive-regret boundary is crossed, two source envelopes fail, signed regret is nonnegative after firing, or a feature-specific falsifier fires",
    inconclusive:
      "neither supported nor falsified; includes insufficient firing or source-family breadth and exactly one envelope failure",
    fail_closed_on_unknown_status: true,
  },
  authorization: {
    execute_frozen_action_cube: true,
    alter_sources_features_faculties_thresholds_or_falsifiers_after_outcome: false,
    claim_universal_wisdom: false,
    promote_optimizer: false,
    authorize_paid_scaling: false,
    read_documents_200_212: false,
  },
};
const bytes = `${JSON.stringify(contract, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: contract.schema, output: outputPath, contract_sha256: sha256(Buffer.from(bytes)),
  source_panels: contract.population.source_panels,
  action_families: contract.faculties.map((faculty) => faculty.action_family),
  corrections_q32: corrections,
  occult_compression_net_gain_bits: contract.occult_feature.compression.net_gain_bits,
  occult_activated: occultActivated,
  action_cube_outcomes_read: false,
}, null, 2)}\n`);
