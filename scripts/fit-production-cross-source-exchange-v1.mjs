#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const framePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const structureContractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-fitting-structure-contract.json";
const structurePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-fitting-structure.json";
const outputPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-predictor.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const reconstruct = (coefficients, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
};
const median = (values) => [...values].sort((left, right) => left < right ? -1 : left > right ? 1 : 0)[
  Math.floor((values.length - 1) / 2)];
const rowFromDocument = (sourceId, document) => {
  const coefficients = document.coefficients.map(BigInt);
  const features = Array.from({length: 6}, (_, atom) => coefficients[1 << atom]);
  const lambda = features[4] - features[2];
  const delta = reconstruct(coefficients, 59) - reconstruct(coefficients, 47);
  return {
    source_id: sourceId,
    singleton_features_q32: features.map(String),
    lambda_q32: lambda.toString(),
    interaction_residual_q32: (delta - lambda).toString(),
  };
};

const frameBytes = fs.readFileSync(framePath);
const contractBytes = fs.readFileSync(structureContractPath);
const structureBytes = fs.readFileSync(structurePath);
const frame = JSON.parse(frameBytes);
const contract = JSON.parse(contractBytes);
const structure = JSON.parse(structureBytes);
assert(frame.schema === "nsrl.production_cross_source_exchange_source_frame.v1"
  && frame.outcome_firewall.action_cube_outcomes_read === false, "wrong prospective source frame");
assert(contract.schema === "nsrl.production_atomic_structure_contract.v1"
  && structure.schema === "nsrl.production_atomic_structure.v1", "wrong fitting cube inputs");
assert(structure.bindings.manifest_hash === contract.manifest_hash
  && structure.bindings.token_stream_hash === contract.bindings.token_stream_hash,
"fitting cube contract mismatch");
const fittingSources = frame.sources.filter((source) => source.role === "fitting");
assert(fittingSources.length === frame.role_partition.fitting_sources, "fitting source count changed");
assert(structure.q32.documents.length === 64, "fitting cube raw surface changed");
const rows = fittingSources.map((source, index) => {
  const document = structure.q32.documents[index];
  assert(document.document === frame.phases.fitting.document_start + index,
    "fitting document order changed");
  const binding = frame.phases.fitting.document_bindings[document.document];
  assert(binding.source_id === source.source_id && binding.analysis_role === "fitting",
    "fitting source/document binding changed");
  return rowFromDocument(source.source_id, document);
});
const featureMedians = Array.from({length: 6}, (_, atom) => median(
  rows.map((row) => BigInt(row.singleton_features_q32[atom]))));
const featureScales = featureMedians.map((center, atom) => {
  const scale = median(rows.map(
    (row) => (BigInt(row.singleton_features_q32[atom]) - center < 0n
      ? center - BigInt(row.singleton_features_q32[atom])
      : BigInt(row.singleton_features_q32[atom]) - center)));
  return scale > 0n ? scale : 1n;
});
const predictor = {
  schema: "nsrl.production_cross_source_exchange_predictor.v1",
  analysis_role: "fitting_only_frozen_before_calibration_evaluation",
  source_sha256: {
    source_frame: sha256(frameBytes),
    fitting_structure_contract: sha256(contractBytes),
    fitting_structure: sha256(structureBytes),
    fitter: sha256(fs.readFileSync(new URL(import.meta.url))),
  },
  training_population: {
    independent_source_panels: rows.length,
    source_ids: rows.map((row) => row.source_id),
    documents_per_panel: 1,
  },
  exchange: {
    id: "base43_atom2_to_atom4",
    base_mask: 43,
    outgoing_atom: 2,
    incoming_atom: 4,
    control_mask: 47,
    candidate_mask: 59,
  },
  probe_features: {
    representation: "Q32 signed integers",
    ordered_features: Array.from({length: 6}, (_, atom) =>
      `singleton_effect_atom_${atom}_loss_mask_${1 << atom}_minus_mask_0`),
    available_before_candidate_multi_atom_outcome: true,
    medians_q32: featureMedians.map(String),
    median_absolute_deviation_scales_q32: featureScales.map(String),
  },
  algorithm: {
    name: "three_nearest_fitting_source_median_residual_v1",
    neighbors: 3,
    distance:
      "sum over six features of floor(abs(x_j-train_j)*2^20/max(1,fitting_MAD_j))",
    distance_tie_break: "ascending source_id",
    prediction: "lower median Q32 interaction residual of the three nearest fitting sources",
    internal_fractional_bits: 20,
    output_conversion: "already exact Q32 integer; no floating-point conversion",
    hyperparameters_selected_before_fitting_outcomes: true,
  },
  fitted_rows: rows,
  firewall: {
    calibration_outcomes_read: false,
    evaluation_outcomes_read: false,
    documents_200_212_read: false,
  },
};
const bytes = `${JSON.stringify(predictor, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: predictor.schema,
  output: outputPath,
  predictor_sha256: sha256(Buffer.from(bytes)),
  fitting_source_panels: rows.length,
  calibration_outcomes_read: false,
  evaluation_outcomes_read: false,
  documents_200_212_read: false,
}, null, 2)}\n`);
