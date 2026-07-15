#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {
  encodeCanonicalJson,
  invariant,
  optimizerControlBinding,
  sha256,
} from "./lib/production-atomic-ising-v1.mjs";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const sourceContractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json";
const kernelPath = new URL("./lib/production-atomic-ising-v1.mjs", import.meta.url);
const analyzerPath = new URL("./analyze-production-atomic-ising-v1.mjs", import.meta.url);
const checkerPath = new URL("./check-production-atomic-ising-v1.mjs", import.meta.url);
const optimizerPath = new URL(
  "../crates/nsrl-train/src/production/training.rs",
  import.meta.url,
);
const optimizerControl = optimizerControlBinding(fs.readFileSync(optimizerPath));

const sourceBytes = fs.readFileSync(sourcePath);
const sourceContractBytes = fs.readFileSync(sourceContractPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const sourceContract = JSON.parse(sourceContractBytes.toString("utf8"));
invariant(source.schema === "nsrl.production_atomic_structure.v1", "wrong source schema");
invariant(sourceContract.schema === "nsrl.production_atomic_structure_contract.v1",
  "wrong source contract schema");
invariant(source.bindings.manifest_hash === sourceContract.manifest_hash,
  "source result does not match its contract");
invariant(source.analysis_role === "proposal_only_calibration"
  && source.transfer_documents_read === 0
  && source.reserved_documents_read === 0,
"source result crossed the proposal-only firewall");
invariant(source.surface.document_start === 8
  && source.surface.documents === 64
  && source.surface.hard_stop_before_document === 72,
"source proposal surface changed");
invariant(source.decision.optimizer_change_authorized === false,
  "source result authorized an optimizer change");
invariant(sourceContract.authorization.optimizer_change === false
  && sourceContract.authorization.paid_scaling === false,
"source contract authorized promotion");

const q20Temperatures = [1, 2, 4, 8, 16, 32, 64, 128];
const contract = {
  schema: "nsrl.production_atomic_ising_audit_contract.v1",
  analysis_role: "proposal_only_calibration",
  source: {
    schema: source.schema,
    result_sha256: sha256(sourceBytes),
    contract_sha256: sha256(sourceContractBytes),
    manifest_hash: source.bindings.manifest_hash,
    proposal_document_start: source.surface.document_start,
    proposal_documents: source.surface.documents,
    hard_stop_before_document: source.surface.hard_stop_before_document,
  },
  implementation: {
    kernel_sha256: sha256(fs.readFileSync(kernelPath)),
    analyzer_sha256: sha256(fs.readFileSync(analyzerPath)),
    checker_sha256: sha256(fs.readFileSync(checkerPath)),
  },
  arithmetic: {
    objective_integer_type: "arbitrary_precision_signed",
    objective_floating_point_operations: 0,
    walsh_normalization_denominator: 64,
    dyadic_weight_shift: 128,
    dyadic_weight_formula:
      "delta=q*T+r; weight=(2*T-r)*2^(127-q); require 0<=q<128",
    rational_encoding: "reduced_decimal_numerator_denominator",
  },
  temperature_sweep: {
    ensemble: "piecewise_linear_dyadic_boltzmann",
    q20_temperature_units: q20Temperatures.map(String),
    q32_temperature_units: q20Temperatures.map((value) => String(value * 4096)),
    selected_ground_state_tie_break: "lowest_vertex_mask",
    metrics: [
      "ground_state_probability",
      "expected_energy_above_ground",
      "selected_ground_state_overlap",
      "edwards_anderson_replica_overlap",
      "magnetic_susceptibility_per_spin",
      "spin_glass_susceptibility",
      "magnetization_by_atom",
    ],
  },
  sigma_delta: {
    retained_walsh_degree: 3,
    vertex_order: "binary_reflected_gray_code",
    quantizer_denominator: 64,
    rounding: "nearest_ties_away_from_zero",
    conservation_identity:
      "sum(input_residual_numerator)=64*sum(emitted_integer_residual)+final_accumulator",
  },
  required_gates: [
    "source_and_implementation_hashes_match",
    "proposal_only_firewall_verified",
    "integer_walsh_inversion_and_parseval_verified",
    "all_temperature_weights_positive",
    "all_metric_fractions_canonical",
    "sigma_delta_conservation_and_bounded_carry_verified",
    "byte_replay_verified",
    "optimizer_control_tuple_matches",
  ],
  control: {
    default_optimizer: optimizerControl.default_optimizer,
    optimizer_source_path: "crates/nsrl-train/src/production/training.rs",
    optimizer_state_magic_literal: optimizerControl.optimizer_state_magic_literal,
    optimizer_state_version: optimizerControl.optimizer_state_version,
    optimizer_control_semantic_sha256: optimizerControl.semantic_sha256,
    structure_certificate_selected: false,
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
  },
};

const bytes = encodeCanonicalJson(contract);
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, bytes);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_audit_contract_freeze.v1",
  contract_sha256: sha256(Buffer.from(bytes)),
  source_result_sha256: contract.source.result_sha256,
  optimizer_control_semantic_sha256: contract.control.optimizer_control_semantic_sha256,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
