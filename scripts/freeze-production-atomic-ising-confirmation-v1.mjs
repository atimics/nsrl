#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const proposalSourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const proposalIsingPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json";
const structureContractPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1-contract.json";
const outputPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json";
const analyzerPath = new URL(
  "./analyze-production-atomic-ising-confirmation-v1.mjs", import.meta.url);
const checkerPath = new URL(
  "./check-production-atomic-ising-confirmation-v1.mjs", import.meta.url);
const structureCheckerPath = new URL(
  "./check-production-atomic-structure-v1.mjs", import.meta.url);
const proposalAnalyzerPath = new URL(
  "./analyze-production-document-ising-v1.mjs", import.meta.url);
const proposalCheckerPath = new URL(
  "./check-production-document-ising-proposal-v1.mjs", import.meta.url);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const read = (file) => fs.readFileSync(file);

const proposalSourceBytes = read(proposalSourcePath);
const proposalIsingBytes = read(proposalIsingPath);
const structureContractBytes = read(structureContractPath);
const proposalSource = JSON.parse(proposalSourceBytes.toString("utf8"));
const proposalIsing = JSON.parse(proposalIsingBytes.toString("utf8"));
const structureContract = JSON.parse(structureContractBytes.toString("utf8"));
assert(proposalSource.schema === "nsrl.production_atomic_structure.v1"
  && proposalSource.analysis_role === "proposal_only_calibration", "wrong proposal source");
assert(proposalSource.transfer_documents_read === 0
  && proposalSource.reserved_documents_read === 0, "proposal source crossed firewall");
assert(proposalIsing.schema === "nsrl.production_atomic_ising_proposal.v1"
  && proposalIsing.analysis_role === "proposal_only_calibration", "wrong proposal Ising result");
assert(proposalIsing.source_result_sha256 === sha256(proposalSourceBytes),
  "proposal Ising source hash mismatch");
assert(structureContract.schema === "nsrl.production_atomic_structure_contract.v1"
  && structureContract.analysis_role === "untouched_confirmation",
"wrong confirmation structure contract");
assert(structureContract.surface.document_start === 136
  && structureContract.surface.documents === 64
  && structureContract.surface.windows_per_document === 2
  && structureContract.surface.hard_stop_before_document === 200,
"confirmation structure surface changed");
assert(structureContract.authorization.optimizer_change === false
  && structureContract.authorization.paid_scaling === false,
"structure contract authorized promotion");
const frozen = proposalIsing.frozen_confirmation_candidates;
assert(frozen.pairwise_ising_map_mask === 59
  && frozen.gibbs_magnetization_mask === 61
  && frozen.global_directional_control_mask === 47,
"proposal candidate masks changed");
assert(JSON.stringify(frozen.cluster_candidate_masks) === JSON.stringify([47, 59])
  && JSON.stringify(frozen.cluster_medoid_feature_vectors) === JSON.stringify([
    ["0", "0", "0", "0", "1977", "-4068"],
    ["0", "0", "0", "0", "-6398", "-4020"],
  ]), "proposal cluster router changed");
assert(JSON.stringify(proposalIsing.stable_low_order_characters) === JSON.stringify([32]),
  "stable low-order character changed");

const contract = {
  schema: "nsrl.production_atomic_ising_confirmation_contract.v1",
  analysis_role: "prospective_confirmation_pre_outcome",
  proposal_bindings: {
    structure_result_sha256: sha256(proposalSourceBytes),
    document_ising_result_sha256: sha256(proposalIsingBytes),
    document_ising_analyzer_sha256: sha256(read(proposalAnalyzerPath)),
    document_ising_checker_sha256: sha256(read(proposalCheckerPath)),
    proposal_document_start: 8,
    proposal_documents: 64,
    proposal_source_clusters: 1,
    transfer_documents_72_135_read: false,
    reserved_documents_136_212_read: false,
  },
  execution: {
    structure_contract_sha256: sha256(structureContractBytes),
    structure_manifest_hash: structureContract.manifest_hash,
    source_fnv64: structureContract.bindings.source_fnv64,
    binary_fnv64: structureContract.bindings.binary_fnv64,
    model_hash: structureContract.bindings.model_hash,
    tokenizer_hash: structureContract.bindings.tokenizer_hash,
    token_stream_hash: structureContract.bindings.token_stream_hash,
    source_index_hash: structureContract.bindings.source_index_hash,
    move_fingerprint: structureContract.move_fingerprint,
  },
  implementation: {
    structure_checker_sha256: sha256(read(structureCheckerPath)),
    analyzer_sha256: sha256(read(analyzerPath)),
    checker_sha256: sha256(read(checkerPath)),
  },
  replay_scope: {
    frozen_structure_cube_reexecuted: false,
    frozen_structure_cube_independently_reconstructed: true,
    derived_confirmation_byte_replayed: true,
    rationale:
      "clean checkout verification treats the hash-bound production cube as frozen input; model data are not repository artifacts",
  },
  surface: {
    document_start: 136,
    documents: 64,
    windows_per_document: 2,
    hard_stop_before_document: 200,
    still_sealed_documents: "200--212",
    expected_source_clusters: 1,
    cross_source_generalization_identified: false,
  },
  candidates: {
    pairwise_ising_map_mask: frozen.pairwise_ising_map_mask,
    gibbs_magnetization_mask: frozen.gibbs_magnetization_mask,
    global_directional_control_mask: frozen.global_directional_control_mask,
    cluster_medoid_feature_vectors: frozen.cluster_medoid_feature_vectors,
    cluster_candidate_masks: frozen.cluster_candidate_masks,
    cluster_distance: "L1",
    cluster_tie_break: "cluster_zero",
    cluster_features: "six_Q32_singleton_contrasts_against_baseline",
  },
  stable_low_order_rule: {
    character: 32,
    standard_parameter: "field_h",
    expected_direction: "negative",
    minimum_visible_documents: 32,
    minimum_directional_fraction: "3/4",
    aggregate_q20_q32_sign_agreement_required: true,
    inferential_role: "descriptive_replication_only",
  },
  gibbs: {
    frozen_action_source: "proposal_Q20_quenched_document_average",
    frozen_fugacity: "1/2",
    descriptive_confirmation_grid: ["1/4", "1/2", "3/4"],
    confirmation_reestimation_may_not_change_frozen_action: true,
  },
  inference: {
    primary_endpoints: [
      "pairwise_ising_map_vs_baseline_q32_document_direction",
      "gibbs_magnetization_vs_baseline_q32_document_direction",
      "cluster_routed_vs_global_directional_q32_document_direction",
    ],
    direction: "negative_contrast_is_improvement",
    per_endpoint_test: "one_sided_exact_binomial_sign_test_conditional_on_non_ties",
    multiplicity: "Holm_step_down",
    familywise_alpha: "1/20",
    support_rule: "Holm_reject_and_aggregate_Q32_contrast_negative",
    assumptions: [
      "document_units_independent_or_exchangeable_for_the_sign_test",
      "conditional_sign_symmetry_under_each_null",
      "router_and_candidates_frozen_before_candidate_outcomes",
    ],
  },
  authorization: {
    compute_documents_136_199: true,
    compute_document_200_or_later: false,
    change_frozen_candidates_after_outcomes: false,
    optimizer_change: false,
    paid_scaling: false,
  },
};
const bytes = `${JSON.stringify(contract, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, bytes);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_confirmation_contract_freeze.v1",
  contract_sha256: sha256(Buffer.from(bytes)),
  structure_contract_sha256: contract.execution.structure_contract_sha256,
  candidates: contract.candidates,
  primary_endpoints: contract.inference.primary_endpoints,
  hard_stop_before_document: contract.surface.hard_stop_before_document,
  documents_200_212_read: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
