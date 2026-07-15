#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {
  buildObjectiveAudit,
  encodeCanonicalJson,
  invariant,
  optimizerControlBinding,
  sha256,
} from "./lib/production-atomic-ising-v1.mjs";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1.json";
const kernelPath = new URL("./lib/production-atomic-ising-v1.mjs", import.meta.url);
const checkerPath = new URL("./check-production-atomic-ising-v1.mjs", import.meta.url);
const optimizerPath = new URL(
  "../crates/nsrl-train/src/production/training.rs",
  import.meta.url,
);

const sourceBytes = fs.readFileSync(sourcePath);
const contractBytes = fs.readFileSync(contractPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const contract = JSON.parse(contractBytes.toString("utf8"));

invariant(source.schema === "nsrl.production_atomic_structure.v1", "wrong source schema");
invariant(contract.schema === "nsrl.production_atomic_ising_audit_contract.v1",
  "wrong audit contract schema");
invariant(source.analysis_role === "proposal_only_calibration"
  && contract.analysis_role === "proposal_only_calibration",
"audit source is not proposal-only");
invariant(source.transfer_documents_read === 0 && source.reserved_documents_read === 0,
  "audit source crossed proposal firewall");
invariant(sha256(sourceBytes) === contract.source.result_sha256,
  "source result hash mismatch");
invariant(source.bindings.manifest_hash === contract.source.manifest_hash,
  "source manifest binding changed");
invariant(sha256(fs.readFileSync(kernelPath)) === contract.implementation.kernel_sha256,
  "kernel hash mismatch");
invariant(sha256(fs.readFileSync(new URL(import.meta.url)))
  === contract.implementation.analyzer_sha256, "analyzer hash mismatch");
invariant(sha256(fs.readFileSync(checkerPath)) === contract.implementation.checker_sha256,
  "checker hash mismatch");
const optimizerControl = optimizerControlBinding(fs.readFileSync(optimizerPath));
invariant(optimizerControl.semantic_sha256
  === contract.control.optimizer_control_semantic_sha256,
"optimizer control tuple changed");
invariant(contract.control.optimizer_change_authorized === false
  && contract.control.paid_scaling_authorized === false,
"audit contract authorized promotion");

const q20 = buildObjectiveAudit(
  source.q20,
  contract.temperature_sweep.q20_temperature_units,
  contract,
);
const q32 = buildObjectiveAudit(
  source.q32,
  contract.temperature_sweep.q32_temperature_units,
  contract,
);
const result = {
  schema: "nsrl.production_atomic_ising_audit.v1",
  analysis_role: "proposal_only_calibration",
  source_result_sha256: contract.source.result_sha256,
  source_contract_sha256: contract.source.contract_sha256,
  audit_contract_sha256: sha256(contractBytes),
  rank: 6,
  vertices: 64,
  bindings: source.bindings,
  implementation: contract.implementation,
  arithmetic: contract.arithmetic,
  q20,
  q32,
  gates: {
    proposal_only_firewall_verified: true,
    integer_walsh_reconstruction_verified: true,
    temperature_sweeps_verified: true,
    overlap_and_susceptibility_metrics_verified: true,
    sigma_delta_conservation_verified: true,
    optimizer_control_tuple_matches: true,
  },
  decision: {
    audit_contract_passed: true,
    structure_certificate_selected: false,
    default_optimizer: contract.control.default_optimizer,
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
  },
};

const bytes = encodeCanonicalJson(result);
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, bytes);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_audit_run.v1",
  result_sha256: sha256(Buffer.from(bytes)),
  q20_ground_state_masks: q20.ising_walsh.ground_state_masks,
  q32_ground_state_masks: q32.ising_walsh.ground_state_masks,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
