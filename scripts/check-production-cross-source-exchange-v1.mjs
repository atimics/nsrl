#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const resultPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json";
const result = JSON.parse(fs.readFileSync(resultPath, "utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const choose = (n, k) => {
  let value = 1;
  for (let index = 1; index <= k; index += 1) value = value * (n - index + 1) / index;
  return value;
};
const binomialUpperTail = (n, p, threshold) => {
  let value = 0;
  for (let k = threshold; k <= n; k += 1) {
    value += choose(n, k) * p ** k * (1 - p) ** (n - k);
  }
  return value;
};
const ceilDiv = (numerator, denominator) => (numerator + denominator - 1n) / denominator;

assert(result.schema === "nsrl.production_cross_source_exchange_result.v1"
  && result.analysis_role === "prospective_untouched_evaluation", "wrong result schema");
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json";
const framePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const predictorPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-predictor.json";
const rawContractPath = process.argv[6]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-calibration-evaluation-structure-contract.json";
const rawResultPath = process.argv[7]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-calibration-evaluation-structure.json";
const analyzerPath = new URL("./analyze-production-cross-source-exchange-v1.mjs", import.meta.url);
const contractBytes = fs.readFileSync(contractPath);
const frameBytes = fs.readFileSync(framePath);
const predictorBytes = fs.readFileSync(predictorPath);
const rawContractBytes = fs.readFileSync(rawContractPath);
const rawResultBytes = fs.readFileSync(rawResultPath);
assert(result.source_sha256.contract === sha256(contractBytes)
  && result.source_sha256.source_frame === sha256(frameBytes)
  && result.source_sha256.predictor === sha256(predictorBytes)
  && result.source_sha256.raw_structure_contract === sha256(rawContractBytes)
  && result.source_sha256.raw_structure_result === sha256(rawResultBytes)
  && result.source_sha256.analyzer === sha256(fs.readFileSync(analyzerPath)),
"result replay binding changed");
const contract = JSON.parse(contractBytes);
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
const rawContract = JSON.parse(rawContractBytes);
const rawResult = JSON.parse(rawResultBytes);
assert(contract.schema === "nsrl.production_cross_source_exchange_contract.v1"
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"contract is not prospective");
assert(frame.outcome_firewall.action_cube_outcomes_read === false
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"prospective outcome firewall changed");
assert(rawResult.bindings.manifest_hash === rawContract.manifest_hash
  && rawResult.bindings.token_stream_hash === rawContract.bindings.token_stream_hash,
"raw cube binding changed");

const roles = Object.fromEntries(["fitting", "calibration", "evaluation"].map((role) => [
  role, new Set(frame.sources.filter((source) => source.role === role).map((source) => source.source_id)),
]));
assert(roles.fitting.size === 16 && roles.calibration.size === 39 && roles.evaluation.size === 16,
  "source-panel role counts changed");
assert([...roles.fitting].every((source) => !roles.calibration.has(source) && !roles.evaluation.has(source))
  && [...roles.calibration].every((source) => !roles.evaluation.has(source)),
"a source crossed fitting/calibration/evaluation roles");
assert(new Set(frame.sources.map((source) => source.author_key)).size === 71
  && frame.sources.every((source) => source.source_id.startsWith("gutenberg-")),
"source independence frame changed");
assert(new Set(predictor.fitted_rows.map((row) => row.source_id)).size === 16
  && predictor.fitted_rows.every((row) => roles.fitting.has(row.source_id)),
"predictor contains a non-fitting source");
assert(contract.population.calibration_source_panels === 39
  && contract.population.evaluation_source_panels === 16
  && contract.population.fitting_source_panels === 16,
"contract population counts changed");
assert(contract.panel_sampling.panel_documents_per_source === 1
  && contract.panel_sampling.model_windows_per_panel_document === 2,
"panel sampling changed");
assert(contract.exchange_set.length === 1
  && contract.exchange_set[0].control_mask === 47
  && contract.exchange_set[0].candidate_mask === 59,
"frozen exchange set changed");

const rank = Number(ceilDiv(40n * 19n, 20n));
assert(rank === 38 && result.conformal.order_statistic_rank === rank,
  "39-source 95% conformal rank changed");
const calibrationScores = result.calibration.rows.map((row) => BigInt(row.simultaneous_score_q32))
  .sort((left, right) => left < right ? -1 : left > right ? 1 : 0);
const correction = calibrationScores[rank - 1];
assert(BigInt(result.conformal.correction_q32) === correction,
  "calibration correction is not the frozen order statistic");
const checkRow = (row) => {
  const lambda = BigInt(row.lambda_q32);
  const predicted = BigInt(row.predicted_interaction_residual_q32);
  const residual = BigInt(row.interaction_residual_q32);
  const delta = BigInt(row.exchange_contrast_q32);
  assert(BigInt(row.simultaneous_score_q32) === residual - predicted,
    "source-panel simultaneous score changed");
  assert(BigInt(row.upper_interaction_residual_q32) === predicted + correction,
    "source-panel upper residual changed");
  assert(row.covered === (residual <= predicted + correction), "coverage flag changed");
  assert(row.fires === (lambda + predicted + correction < 0n), "strict abstention rule changed");
  assert(row.unsafe === (row.fires && delta >= 0n), "unsafe firing flag changed");
  assert(row.neighbors.length === 3, "predictor neighbor count changed");
};
result.calibration.rows.forEach(checkRow);
result.untouched_evaluation.rows.forEach(checkRow);
const evaluation = result.untouched_evaluation.rows;
const uncovered = evaluation.filter((row) => !row.covered);
const fired = evaluation.filter((row) => row.fires);
const unsafe = fired.filter((row) => row.unsafe);
const aggregate = fired.reduce((sum, row) => sum + BigInt(row.exchange_contrast_q32), 0n);
assert(evaluation.length === 16
  && result.untouched_evaluation.envelope_uncovered === uncovered.length
  && result.untouched_evaluation.fired_source_panels === fired.length
  && result.untouched_evaluation.unsafe_firings === unsafe.length
  && BigInt(result.untouched_evaluation.aggregate_fired_exchange_contrast_q32) === aggregate,
"evaluation summary changed");
assert(binomialUpperTail(16, 0.05, 3) <= 0.05
  && binomialUpperTail(16, 0.05, 2) > 0.05
  && contract.falsifiers.coverage.exact_binomial_rejection_failures === 3,
"coverage falsifier is not the exact one-sided 5% threshold");
const expectedGates = {
  source_envelope_not_rejected: uncovered.length < 3,
  source_envelope_promotion_gate:
    uncovered.length <= contract.falsifiers.coverage.maximum_failures_for_support,
  unsafe_action_gate: unsafe.length === 0,
  nonvacuity_gate: fired.length >= contract.falsifiers.nonvacuity.minimum_fired_source_panels,
  incremental_value_gate: fired.length > 0 && aggregate < 0n,
};
assert(Object.keys(expectedGates).every(
  (key) => result.falsifier_gates[key] === expectedGates[key]), "falsifier gate changed");
assert(result.decision.optimizer_change_authorized === false
  && result.decision.paid_scaling_authorized === false
  && result.decision.documents_200_212_read === false
  && contract.authorization.read_documents_200_212 === false,
"cross-source experiment escaped its authorization boundary");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_cross_source_exchange_check.v1",
  distinct_author_sources: 71,
  fitting_source_panels: 16,
  calibration_source_panels: 39,
  evaluation_source_panels: 16,
  conformal_order_statistic_rank: rank,
  correction_q32: correction.toString(),
  envelope_uncovered: uncovered.length,
  fired_source_panels: fired.length,
  unsafe_firings: unsafe.length,
  aggregate_fired_exchange_contrast_q32: aggregate.toString(),
  gates: expectedGates,
  decision: result.decision.status,
  documents_200_212_read: false,
}, null, 2)}\n`);
