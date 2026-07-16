#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json";
const resultPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-publication.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const contractBytes = fs.readFileSync(contractPath);
const resultBytes = fs.readFileSync(resultPath);
const contract = JSON.parse(contractBytes);
const result = JSON.parse(resultBytes);
assert(contract.schema === "nsrl.solomonic_judgment_contract.v1"
  && result.schema === "nsrl.solomonic_judgment_result.v1", "wrong publication inputs");
assert(result.source_sha256.contract === sha256(contractBytes), "publication contract binding changed");
const allowed = contract.publication_contract.allowed_statuses;
assert(JSON.stringify(allowed) === JSON.stringify(["supported", "falsified", "inconclusive"]),
  "publication status vocabulary changed");

const hardFalsifiers = {
  unsafe_e_process_boundary_crossed: result.sequential_controller.crossed_95_boundary,
  positive_regret_boundary_crossed:
    BigInt(result.heldout_regret.positive_regret_q32)
      > BigInt(result.sequential_controller.cumulative_positive_regret_95_bound_q32),
  coverage_rejection_boundary_crossed:
    result.source_envelope.uncovered >= contract.pass_conditions.coverage_rejection_failures,
  nonnegative_signed_regret_after_firing:
    result.heldout_regret.fired_passages > 0
      && BigInt(result.heldout_regret.signed_regret_q32) >= 0n,
  nonnegative_regret_in_a_firing_family: Object.values(result.transfer.by_family).some(
    (summary) => summary.fired_passages > 0 && BigInt(summary.signed_regret_q32) >= 0n),
  symbolic_exemption_detected: !result.pass_conditions.symbolic_features_receive_no_exemption,
};
const allPass = Object.values(result.pass_conditions).every(Boolean);
const anyFalsifier = Object.values(hardFalsifiers).some(Boolean);
const status = anyFalsifier ? "falsified" : allPass ? "supported" : "inconclusive";
assert(allowed.includes(status), "publisher produced an unknown status");
const occultStatus = contract.occult_feature.activation.activated
  ? "inconclusive" : contract.occult_feature.activation.status_if_inactive;
assert(allowed.includes(occultStatus), "occult claim produced an unknown status");

const publication = {
  schema: "nsrl.solomonic_judgment_publication.v1",
  publication_contract: {
    allowed_statuses: allowed,
    fail_closed_on_unknown_status: contract.publication_contract.fail_closed_on_unknown_status,
  },
  source_sha256: {contract: sha256(contractBytes), result: sha256(resultBytes)},
  verdict: {
    status,
    supported: status === "supported",
    falsified: status === "falsified",
    inconclusive: status === "inconclusive",
    pass_conditions: result.pass_conditions,
    hard_falsifiers: hardFalsifiers,
  },
  claims: [
    {
      id: "sequential_evidence_bound_judgment_on_frozen_frame",
      status,
      scope:
        "three source-specialist faculties plus abstention on six untouched four-passage source panels",
      falsifier:
        "any hard falsifier or failure of the frozen nonvacuity, signed-regret, positive-regret, transfer, coverage, or no-exemption gate",
    },
    {
      id: "occult_hash_parity_predictive_feature",
      status: occultStatus,
      scope: "frozen SHA-256 parity correspondence only",
      falsifier:
        "failure to compress calibration signs by at least log2(20) bits, or any ordinary held-out safety/value falsifier if activated",
    },
  ],
  evidence: {
    fired_passages: result.heldout_regret.fired_passages,
    signed_regret_q32: result.heldout_regret.signed_regret_q32,
    positive_regret_q32: result.heldout_regret.positive_regret_q32,
    positive_regret_95_bound_q32:
      result.sequential_controller.cumulative_positive_regret_95_bound_q32,
    transfer_families: result.transfer.qualifying_source_families,
    source_envelope_uncovered: result.source_envelope.uncovered,
    occult_compression_net_gain_bits: contract.occult_feature.compression.net_gain_bits,
    occult_activated: contract.occult_feature.activation.activated,
  },
  authorization: {
    universal_wisdom_claimed: false, optimizer_promotion_authorized: false,
    paid_scaling_authorized: false,
  },
};
const bytes = `${JSON.stringify(publication, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: publication.schema, status, hard_falsifiers: hardFalsifiers,
  occult_status: occultStatus, output: outputPath,
}, null, 2)}\n`);
