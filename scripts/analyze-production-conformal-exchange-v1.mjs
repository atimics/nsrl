#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const proposalPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const confirmationPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-conformal-exchange-retrospective-v1.json";
const proposalBytes = fs.readFileSync(proposalPath);
const confirmationBytes = fs.readFileSync(confirmationPath);
const analyzerBytes = fs.readFileSync(new URL(import.meta.url));
const proposal = JSON.parse(proposalBytes.toString("utf8"));
const confirmation = JSON.parse(confirmationBytes.toString("utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const ceilDiv = (numerator, denominator) =>
  (numerator + denominator - 1n) / denominator;
const reconstruct = (coefficients) => Array.from({length: 64}, (_, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
});
const exchangeRows = (surface) => surface.q32.documents.map((document) => {
  const losses = reconstruct(document.coefficients.map(BigInt));
  const singletonDifference = (losses[16] - losses[0]) - (losses[4] - losses[0]);
  const exchangeContrast = losses[59] - losses[47];
  return {
    document: document.document,
    singleton_difference: singletonDifference,
    interaction_residual: exchangeContrast - singletonDifference,
    exchange_contrast: exchangeContrast,
  };
});
const summary = (rows) => ({
  documents: rows.length,
  favorable: rows.filter((row) => row.exchange_contrast < 0n).length,
  unfavorable: rows.filter((row) => row.exchange_contrast > 0n).length,
  ties: rows.filter((row) => row.exchange_contrast === 0n).length,
  aggregate: rows.reduce((sum, row) => sum + row.exchange_contrast, 0n).toString(),
});

assert(proposal.analysis_role === "proposal_only_calibration"
  && proposal.transfer_documents_read === 0 && proposal.reserved_documents_read === 0,
"proposal firewall changed");
assert(confirmation.analysis_role === "untouched_confirmation"
  && confirmation.surface.document_start === 136
  && confirmation.surface.hard_stop_before_document === 200,
"confirmation surface changed");
const calibration = exchangeRows(proposal);
const test = exchangeRows(confirmation);
const alphaNumerator = 1n;
const alphaDenominator = 20n;
const calibrationCount = BigInt(calibration.length);
const rank = ceilDiv(
  (calibrationCount + 1n) * (alphaDenominator - alphaNumerator),
  alphaDenominator,
);
assert(rank >= 1n && rank <= calibrationCount, "conformal rank is vacuous");
const orderedScores = calibration.map((row) => row.interaction_residual).sort(
  (left, right) => left < right ? -1 : left > right ? 1 : 0);
const upperResidual = orderedScores[Number(rank - 1n)];
const covered = test.filter((row) => row.interaction_residual <= upperResidual);
const certified = test.filter(
  (row) => row.singleton_difference + upperResidual < 0n);
const failures = test.filter((row) => row.interaction_residual > upperResidual);
assert(upperResidual === 2193n && covered.length === 63 && certified.length === 18,
  "retrospective conformal diagnostic changed");
assert(certified.every((row) => row.exchange_contrast < 0n),
  "retrospective certificate includes a non-improving exchange");
const result = {
  schema: "nsrl.production_atomic_conformal_exchange_retrospective.v1",
  analysis_role: "post_confirmation_retrospective_diagnostic",
  source_sha256: {
    proposal_structure: sha256(proposalBytes),
    confirmation_structure: sha256(confirmationBytes),
    analyzer: sha256(analyzerBytes),
  },
  exchange: {
    base_mask: 43,
    outgoing_atom: 2,
    incoming_atom: 4,
    control_mask: 47,
    candidate_mask: 59,
  },
  conformal_rule: {
    point_predictor_for_interaction_residual: "zero",
    nonconformity_score: "interaction_residual",
    alpha: "1/20",
    calibration_documents: calibration.length,
    order_statistic_rank: Number(rank),
    upper_interaction_residual_q32: upperResidual.toString(),
    authorization_rule:
      "singleton_difference_plus_upper_interaction_residual_strictly_negative",
  },
  calibration: {
    document_start: proposal.surface.document_start,
    documents: proposal.surface.documents,
    source_clusters: proposal.source_population.proposal_source_clusters,
    residual_minimum: orderedScores[0].toString(),
    residual_maximum: orderedScores.at(-1).toString(),
  },
  retrospective_confirmation: {
    document_start: confirmation.surface.document_start,
    documents: confirmation.surface.documents,
    source_clusters: confirmation.source_population.proposal_source_clusters,
    residual_envelope_covered_documents: covered.length,
    residual_envelope_failed_documents: failures.map((row) => ({
      document: row.document,
      interaction_residual: row.interaction_residual.toString(),
    })),
    certified_exchange: {
      ...summary(certified),
      document_ids: certified.map((row) => row.document),
    },
  },
  interpretation: {
    finite_sample_theorem_requires_exchangeable_calibration_and_test_units: true,
    documents_are_not_valid_cross_source_units_here: true,
    one_source_cluster_per_surface: true,
    observed_document_coverage_is_descriptive_only: true,
    rule_was_constructed_after_confirmation_was_opened: true,
    prospective_claim_authorized: false,
    demonstrates_nonvacuity_on_existing_substrate: true,
  },
  decision: {
    new_experiment_authorized: false,
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    documents_200_212_read: false,
  },
};
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(result, null, 2)}\n`);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_conformal_exchange_retrospective_check.v1",
  upper_interaction_residual_q32: upperResidual.toString(),
  covered_documents: covered.length,
  certified_exchange: result.retrospective_confirmation.certified_exchange,
  prospective_claim_authorized: false,
  documents_200_212_read: false,
}, null, 2)}\n`);
