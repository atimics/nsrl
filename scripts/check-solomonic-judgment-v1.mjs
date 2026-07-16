#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {execFileSync} from "node:child_process";

const resultPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json";
const structurePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-structure.json";
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const gcd = (left, right) => {
  let a = left;
  let b = right;
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};
const reduce = (numerator, denominator) => {
  const divisor = gcd(numerator, denominator);
  return {numerator: numerator / divisor, denominator: denominator / divisor};
};
const boundary = (rounds) => {
  for (let unsafe = 0; unsafe <= rounds; unsafe += 1) {
    const numerator = 5n ** BigInt(unsafe) * 15n ** BigInt(rounds - unsafe);
    const denominator = 19n ** BigInt(rounds - unsafe);
    if (numerator >= 20n * denominator) return unsafe;
  }
  return rounds + 1;
};

const resultBytes = fs.readFileSync(resultPath);
const contractBytes = fs.readFileSync(contractPath);
const structureBytes = fs.readFileSync(structurePath);
const result = JSON.parse(resultBytes);
const contract = JSON.parse(contractBytes);
assert(result.schema === "nsrl.solomonic_judgment_result.v1"
  && result.analysis_role === "prospective_untouched_evaluation", "wrong result schema");
assert(contract.schema === "nsrl.solomonic_judgment_contract.v1"
  && contract.analysis_role === "prospective_pre_outcome", "wrong contract schema");
assert(result.source_sha256.contract === sha256(contractBytes)
  && result.source_sha256.structure_result === sha256(structureBytes),
"result top-level hash binding changed");
for (const key of ["source_frame", "structure_contract", "predictor", "parent_result",
  "judgment_record_schema", "preparer", "freezer", "analyzer", "checker", "publisher",
  "publication_checker"]) {
  const binding = contract.bindings[key];
  assert(binding && sha256(fs.readFileSync(binding.path)) === binding.sha256,
    `contract binding changed: ${key}`);
}
assert(result.source_sha256.analyzer === contract.bindings.analyzer.sha256,
  "result analyzer hash changed");

const replayPath = path.join(os.tmpdir(), `solomonic-judgment-replay-${process.pid}.json`);
try {
  execFileSync(process.execPath, [contract.bindings.analyzer.path, contractPath, structurePath, replayPath],
    {stdio: "pipe"});
  assert(fs.readFileSync(replayPath).equals(resultBytes), "analyzer byte replay changed");
} finally {
  fs.rmSync(replayPath, {force: true});
}

const expectedFamilies = new Set([
  "federal_register_exchange", "rfc_exchange", "science_exchange",
  "occult_correspondence", "abstention",
]);
assert(result.judgments.length === contract.population.source_panels * contract.panel.passages_per_source,
  "judgment count changed");
assert(new Set(result.judgments.map((record) => record.judgment_id)).size === result.judgments.length,
  "judgment ids repeat");
for (const [index, record] of result.judgments.entries()) {
  assert(record.schema === "nsrl.judgment_record.v1" && record.sequence_index === index + 1,
    "formal judgment identity changed");
  assert(record.context && record.candidate_actions && record.selected && record.falsifier
    && record.provenance && record.realized_outcome, "formal judgment field missing");
  assert(record.candidate_actions.length === 5
    && new Set(record.candidate_actions.map((candidate) => candidate.action_family)).size === 5
    && record.candidate_actions.every((candidate) => expectedFamilies.has(candidate.action_family)),
  "judgment does not contain three domain faculties, occult hypothesis, and abstention");
  const selected = record.candidate_actions.find(
    (candidate) => candidate.action_id === record.selected.action_id);
  assert(selected && selected.eligible, "selected action was not eligible");
  if (record.selected.kind === "action") {
    assert(record.context.history_guard_open
      && BigInt(record.audit.certified_upper_contrast_q32) < 0n
      && selected.symbolic_feature === false,
    "action selection used an invalid or symbolic faculty");
  } else assert(record.selected.action_id === "abstain", "abstention record changed");
  assert(record.candidate_actions.filter((candidate) => candidate.symbolic_feature).every(
    (candidate) => candidate.falsifier.status_if_triggered === "falsified"
      && candidate.uncertainty_envelope_q32.coverage === "19/20 marginal source-panel"),
  "symbolic candidate received an exemption");
}

let eValue = {numerator: 1n, denominator: 1n};
let unsafe = 0;
let signed = 0n;
let positive = 0n;
for (const [index, row] of result.sequential_controller.source_ledger.entries()) {
  signed += BigInt(row.signed_regret_q32);
  positive += BigInt(row.positive_regret_q32);
  if (row.unsafe_source_panel) {
    unsafe += 1;
    eValue = reduce(eValue.numerator * 5n, eValue.denominator);
  } else eValue = reduce(eValue.numerator * 15n, eValue.denominator * 19n);
  assert(BigInt(row.unsafe_e_value_after.numerator) === eValue.numerator
    && BigInt(row.unsafe_e_value_after.denominator) === eValue.denominator,
  `e-process ledger changed at source ${index + 1}`);
}
assert(BigInt(result.heldout_regret.signed_regret_q32) === signed
  && BigInt(result.heldout_regret.positive_regret_q32) === positive
  && result.sequential_controller.unsafe_source_panels === unsafe,
"regret or unsafe ledger changed");
const finalBoundary = boundary(result.sequential_controller.source_panel_rounds);
const maxContrast = BigInt(contract.numeric_contract.maximum_absolute_contrast_q32);
const expectedPositiveBoundary = BigInt(finalBoundary - 1)
  * BigInt(contract.panel.passages_per_source) * maxContrast;
assert(result.sequential_controller.unsafe_count_boundary_at_final_round === finalBoundary
  && BigInt(result.sequential_controller.cumulative_positive_regret_95_bound_q32)
    === expectedPositiveBoundary,
"95% unsafe/positive-regret boundary changed");

// At p=1/20, the predictable likelihood-ratio update has conditional mean one:
// (1/20)*5 + (19/20)*(15/19) = 1. It is no larger for any p <= 1/20.
assert(1n * 5n * 19n + 19n * 15n === 20n * 19n,
  "e-process supermartingale identity changed");
assert(contract.controller.e_process.ville_threshold === 20
  && contract.controller.e_process.type_i_error_bound === "1/20 anytime, including predictable action choice and stopping",
"Ville boundary changed");
assert(contract.faculties.filter((faculty) => faculty.kind === "domain_exchange").length === 3
  && contract.faculties.some((faculty) => faculty.kind === "abstention"),
"faculty surface changed");
assert(contract.occult_feature.activation.ordinary_falsifiers_apply_if_active === true
  && result.pass_conditions.symbolic_features_receive_no_exemption === true,
"occult feature escaped ordinary falsification");
assert(result.authorization.universal_wisdom_claimed === false
  && result.authorization.optimizer_promotion_authorized === false
  && result.authorization.paid_scaling_authorized === false
  && result.authorization.original_gutenberg_fitting_frame_used === false,
"authorization boundary changed");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomonic_judgment_check.v1",
  source_panels: result.sequential_controller.source_panel_rounds,
  judgments: result.judgments.length,
  action_families: [...expectedFamilies],
  fired_passages: result.heldout_regret.fired_passages,
  signed_regret_q32: result.heldout_regret.signed_regret_q32,
  positive_regret_q32: result.heldout_regret.positive_regret_q32,
  positive_regret_95_bound_q32: result.sequential_controller.cumulative_positive_regret_95_bound_q32,
  transfer_families: result.transfer.qualifying_source_families,
  pass_conditions: result.pass_conditions,
  analyzer_byte_replay: true,
}, null, 2)}\n`);
