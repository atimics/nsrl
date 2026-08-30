#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const auditBytes = fs.readFileSync(config.audit);
const audit = JSON.parse(auditBytes);
const replayAuditBytes = fs.readFileSync(config.replayAudit);
const replayAudit = JSON.parse(replayAuditBytes);

assert(contract.schema === "nsrl.production_direct_head_cross_document_stability_contract.v1",
  "cross-document stability contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}

checkTrace(audit, contract);
checkTrace(replayAudit, contract);
const exactTraceRerunReplay = sha256(auditBytes) === sha256(replayAuditBytes)
  && JSON.stringify(audit) === JSON.stringify(replayAudit);

const thresholds = contract.classification;
const directionMeasurements = audit.directions.map((direction) => {
  let classification = "broad_descent_with_nonpositive_aggregate";
  if (direction.summary.descent_documents
      >= thresholds.stable_direction_descent_documents_minimum
    && direction.summary.total_nll_improvement_q20
      > thresholds.stable_direction_total_nll_improvement_q20_minimum_exclusive) {
    classification = "stable";
  } else if (direction.summary.descent_documents
      <= thresholds.consistent_regression_descent_documents_maximum) {
    classification = "consistent_regression";
  } else if (direction.summary.descent_documents
      >= thresholds.mixed_direction_descent_documents_minimum
    && direction.summary.descent_documents
      <= thresholds.mixed_direction_descent_documents_maximum) {
    classification = "mixed";
  }
  return {
    global_coordinate: direction.global_coordinate,
    parameter_group: direction.parameter_group,
    local_coordinate: direction.local_coordinate,
    delta: direction.delta,
    descent_documents: direction.summary.descent_documents,
    regression_documents: direction.summary.regression_documents,
    unchanged_documents: direction.summary.unchanged_documents,
    total_nll_improvement_q20: direction.summary.total_nll_improvement_q20,
    minimum_nll_improvement_q20: direction.summary.minimum_nll_improvement_q20,
    maximum_nll_improvement_q20: direction.summary.maximum_nll_improvement_q20,
    classification,
  };
});
const stableDirections = directionMeasurements
  .filter((direction) => direction.classification === "stable");
const expectedMatrixCells = contract.directions.length * contract.surface.documents;
const observedMatrixCells = audit.directions.reduce(
  (total, direction) => total + direction.samples.length, 0);
const gates = {
  exact_trace_rerun_replay: exactTraceRerunReplay,
  complete_direction_document_matrix: observedMatrixCells === expectedMatrixCells,
  source_model_unchanged: audit.hashes.initial_model === audit.hashes.final_model
    && audit.gates.source_model_unchanged === true,
  frozen_parameters_unchanged:
    audit.hashes.initial_frozen_parameters === audit.hashes.final_frozen_parameters
      && audit.gates.frozen_parameters_unchanged === true,
  same_coordinate_family_followup_requires_stable_direction:
    stableDirections.length
      >= thresholds.stable_directions_minimum_for_same_coordinate_family_followup,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_cross_document_stability_gate.v1",
  contract: binding(config.contract, contractBytes),
  audit: binding(config.audit, auditBytes),
  replay_audit: binding(config.replayAudit, replayAuditBytes),
  measurements: {
    documents: audit.surfaces.map((surface) => surface.document),
    windows_per_document: audit.surface.windows_per_document,
    directions: directionMeasurements,
    stable_directions: stableDirections,
    stable_direction_count: stableDirections.length,
    mixed_direction_count: directionMeasurements
      .filter((direction) => direction.classification === "mixed").length,
    consistent_regression_direction_count: directionMeasurements
      .filter((direction) => direction.classification === "consistent_regression").length,
    expected_matrix_cells: expectedMatrixCells,
    observed_matrix_cells: observedMatrixCells,
  },
  gates,
  same_coordinate_family_followup_supported: passed,
  public_development_opened: false,
  public_test_opened: false,
  open_generation_opened: false,
  hidden_panel_opened: false,
  paid_scaling_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "cross-document stability gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  same_coordinate_family_followup_supported: passed,
  stable_direction_count: stableDirections.length,
  gates,
  out: config.out,
})}\n`);
if (!passed) process.exitCode = 1;

function checkTrace(trace, expected) {
  assert(trace.schema === "nsrl.direct_head_cross_document_audit.v1",
    "cross-document audit trace schema is invalid");
  assert(trace.objective === expected.implementation.objective
    && trace.method === expected.implementation.method
    && trace.profile === expected.profile
    && trace.parameter_count === expected.parameter_count,
  "cross-document audit implementation binding is invalid");
  assert(trace.bindings.tokenizer_hash === expected.bindings.tokenizer_hash
    && trace.bindings.token_stream_hash === expected.bindings.train_token_stream_hash,
  "cross-document audit corpus binding is invalid");
  assert(trace.surface.context_tokens === expected.surface.context_tokens
    && trace.surface.document_start === expected.surface.document_start
    && trace.surface.documents === expected.surface.documents
    && trace.surface.windows_per_document === expected.surface.windows_per_document
    && trace.window_selection === expected.implementation.window_selection,
  "cross-document audit surface is invalid");
  assert(trace.hashes.initial_model === expected.source.model_hash
    && trace.hashes.final_model === expected.source.model_hash
    && trace.hashes.initial_frozen_parameters === expected.source.frozen_parameter_hash
    && trace.hashes.final_frozen_parameters === expected.source.frozen_parameter_hash,
  "cross-document audit model binding is invalid");
  assert(trace.gates.public_development_opened === false
    && trace.gates.public_test_opened === false
    && trace.gates.open_generation_opened === false
    && trace.gates.hidden_panel_opened === false,
  "cross-document audit opened an unauthorized stage");

  const expectedDocuments = Array.from(
    {length: expected.surface.documents},
    (_, index) => expected.surface.document_start + index,
  );
  assert(trace.surfaces.length === expectedDocuments.length,
    "cross-document surface count is invalid");
  for (const [index, surface] of trace.surfaces.entries()) {
    assert(surface.document === expectedDocuments[index]
      && surface.windows === expected.surface.windows_per_document,
    `cross-document surface ${index} is invalid`);
  }

  assert(trace.directions.length === expected.directions.length,
    "cross-document direction count is invalid");
  for (const [index, direction] of trace.directions.entries()) {
    const frozen = expected.directions[index];
    assert(direction.global_coordinate === frozen.global_coordinate
      && direction.parameter_group === "output_weight"
      && direction.local_coordinate === frozen.global_coordinate
      && direction.delta === frozen.delta,
    `cross-document direction ${index} is invalid`);
    assert(direction.samples.length === expectedDocuments.length,
      `cross-document direction ${index} sample count is invalid`);
    const improvements = [];
    for (const [sampleIndex, sample] of direction.samples.entries()) {
      const surface = trace.surfaces[sampleIndex];
      assert(sample.document === expectedDocuments[sampleIndex]
        && sample.baseline_nll_q20 === surface.baseline_nll_q20
        && sample.nll_improvement_q20
          === sample.baseline_nll_q20 - sample.candidate_nll_q20
        && sample.strict_descent === (sample.nll_improvement_q20 > 0),
      `cross-document direction ${index} sample ${sampleIndex} is inconsistent`);
      improvements.push(sample.nll_improvement_q20);
    }
    const descent = improvements.filter((value) => value > 0).length;
    const regression = improvements.filter((value) => value < 0).length;
    const unchanged = improvements.length - descent - regression;
    assert(direction.summary.descent_documents === descent
      && direction.summary.regression_documents === regression
      && direction.summary.unchanged_documents === unchanged
      && direction.summary.total_nll_improvement_q20
        === improvements.reduce((total, value) => total + value, 0)
      && direction.summary.minimum_nll_improvement_q20 === Math.min(...improvements)
      && direction.summary.maximum_nll_improvement_q20 === Math.max(...improvements),
    `cross-document direction ${index} summary is inconsistent`);
  }

  const expectedBindings = expected.surface.documents
    * expected.surface.windows_per_document;
  assert(trace.window_bindings.length === expectedBindings,
    "cross-document window binding count is invalid");
  for (const document of expectedDocuments) {
    const bindings = trace.window_bindings
      .filter((binding) => binding.document === document);
    assert(bindings.length === expected.surface.windows_per_document
      && bindings.every((binding) =>
        binding.context_tokens === expected.surface.context_tokens
        && binding.target_offset === binding.context_start + binding.context_tokens),
    `cross-document window bindings for document ${document} are invalid`);
  }
}

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, sha256: sha256(bytes)};
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(args) {
  const config = {check: false};
  const names = {
    "--contract": "contract", "--audit": "audit",
    "--replay-audit": "replayAudit", "--out": "out",
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--check") config.check = true;
    else {
      const name = names[args[index]];
      if (!name) throw new Error(`unknown argument ${args[index]}`);
      config[name] = args[++index] || "";
    }
  }
  for (const name of Object.values(names)) assert(config[name], `missing ${name}`);
  return config;
}
