#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const artifactPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-conformal-exchange-retrospective-v1.json";
const proposalPath = "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const confirmationPath =
  "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const analyzerPath = new URL("./analyze-production-conformal-exchange-v1.mjs", import.meta.url);
const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const ceilDiv = (numerator, denominator) =>
  (numerator + denominator - 1n) / denominator;
const conformalRank = (calibrationUnits, alphaNumerator, alphaDenominator) => ceilDiv(
  BigInt(calibrationUnits + 1) * BigInt(alphaDenominator - alphaNumerator),
  BigInt(alphaDenominator),
);
const quantile = (values, rank) => [...values].sort((left, right) => left - right)[rank - 1];

// Exhaust every possible test rank for several finite calibration sizes.
let scalarRankCases = 0;
for (const [n, alphaNumerator, alphaDenominator] of [
  [4, 1, 5],
  [5, 1, 3],
  [9, 1, 5],
  [19, 1, 20],
]) {
  const rank = Number(conformalRank(n, alphaNumerator, alphaDenominator));
  assert(rank <= n, `test case n=${n} unexpectedly vacuous`);
  const population = Array.from({length: n + 1}, (_, index) => index);
  let covered = 0;
  for (let testIndex = 0; testIndex <= n; testIndex += 1) {
    const calibration = population.filter((_, index) => index !== testIndex);
    const upper = quantile(calibration, rank);
    if (population[testIndex] <= upper) covered += 1;
    scalarRankCases += 1;
  }
  assert(covered * alphaDenominator
    >= (n + 1) * (alphaDenominator - alphaNumerator),
  `finite conformal coverage failed for n=${n}`);
}

// Verify max-score simultaneous coverage and safe adaptive selection.
const n = 4;
const rank = Number(conformalRank(n, 1, 5));
let simultaneousCovered = 0;
let unsafeSelections = 0;
for (let testIndex = 0; testIndex <= n; testIndex += 1) {
  const units = Array.from({length: n + 1}, (_, unit) => ({
    features: [unit % 2, unit],
    residuals: [unit - 2, unit],
    predictors: [-2, 0],
  }));
  const scores = units.map((unit) => Math.max(
    unit.residuals[0] - unit.predictors[0],
    unit.residuals[1] - unit.predictors[1],
  ));
  const upper = quantile(scores.filter((_, index) => index !== testIndex), rank);
  const test = units[testIndex];
  const covered = test.residuals.every(
    (residual, exchange) => residual <= test.predictors[exchange] + upper);
  if (covered) simultaneousCovered += 1;
  const selectedExchange = test.features[0];
  const singletonDifference = -(test.predictors[selectedExchange] + upper + 1);
  const fires = singletonDifference + test.predictors[selectedExchange] + upper < 0;
  const exactContrast = singletonDifference + test.residuals[selectedExchange];
  if (fires && exactContrast >= 0) unsafeSelections += 1;
  assert(!covered || !fires || exactContrast < 0,
    "covered adaptive exchange violated its margin certificate");
}
assert(simultaneousCovered === 4 && unsafeSelections <= 1,
  "simultaneous conformal selection frequency changed");

let minimumNonvacuous = 0;
for (let calibrationUnits = 1; calibrationUnits <= 100; calibrationUnits += 1) {
  if (conformalRank(calibrationUnits, 1, 20) <= BigInt(calibrationUnits)) {
    minimumNonvacuous = calibrationUnits;
    break;
  }
}
assert(minimumNonvacuous === 19, "95% conformal resolution calculation changed");

assert(artifact.schema === "nsrl.production_atomic_conformal_exchange_retrospective.v1",
  "wrong retrospective artifact schema");
assert(artifact.source_sha256.proposal_structure
  === sha256(fs.readFileSync(proposalPath)), "proposal source hash mismatch");
assert(artifact.source_sha256.confirmation_structure
  === sha256(fs.readFileSync(confirmationPath)), "confirmation source hash mismatch");
assert(artifact.source_sha256.analyzer === sha256(fs.readFileSync(analyzerPath)),
  "retrospective analyzer hash mismatch");
assert(artifact.conformal_rule.order_statistic_rank === 62
  && artifact.conformal_rule.upper_interaction_residual_q32 === "2193",
"retrospective conformal order statistic changed");
assert(artifact.retrospective_confirmation.residual_envelope_covered_documents === 63
  && artifact.retrospective_confirmation.certified_exchange.documents === 18
  && artifact.retrospective_confirmation.certified_exchange.favorable === 18
  && artifact.retrospective_confirmation.certified_exchange.unfavorable === 0,
"retrospective nonvacuity diagnostic changed");
assert(artifact.interpretation.prospective_claim_authorized === false
  && artifact.decision.documents_200_212_read === false,
"retrospective diagnostic escaped its evidence boundary");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.conformal_exchange_theory_check.v1",
  scalar_rank_cases: scalarRankCases,
  exact_finite_sample_rank_coverage_verified: true,
  simultaneous_max_score_adaptive_selection_verified: true,
  minimum_calibration_units_for_nonvacuous_95_percent_envelope: minimumNonvacuous,
  retrospective_document_diagnostic: {
    upper_interaction_residual_q32: artifact.conformal_rule.upper_interaction_residual_q32,
    envelope_covered_documents: artifact.retrospective_confirmation
      .residual_envelope_covered_documents,
    certified_favorable: artifact.retrospective_confirmation.certified_exchange.favorable,
    certified_unfavorable: artifact.retrospective_confirmation.certified_exchange.unfavorable,
    cross_source_claim: false,
  },
  documents_200_212_read: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
