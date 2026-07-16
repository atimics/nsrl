#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json";
const structurePath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-calibration-evaluation-structure.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const ceilDiv = (numerator, denominator) => (numerator + denominator - 1n) / denominator;
const absolute = (value) => value < 0n ? -value : value;
const reconstruct = (coefficients, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
};
const lowerMedian = (values) => [...values].sort(
  (left, right) => left < right ? -1 : left > right ? 1 : 0)[Math.floor((values.length - 1) / 2)];
const rowFromDocument = (sourceId, document) => {
  const coefficients = document.coefficients.map(BigInt);
  const features = Array.from({length: 6}, (_, atom) => coefficients[1 << atom]);
  const lambda = features[4] - features[2];
  const delta = reconstruct(coefficients, 59) - reconstruct(coefficients, 47);
  return {source_id: sourceId, features, lambda, delta, residual: delta - lambda};
};
const predict = (features, predictor) => {
  const scales = predictor.probe_features.median_absolute_deviation_scales_q32.map(BigInt);
  const nearest = predictor.fitted_rows.map((row) => {
    const training = row.singleton_features_q32.map(BigInt);
    const distance = training.reduce((sum, value, atom) =>
      sum + ((absolute(features[atom] - value) << 20n) / scales[atom]), 0n);
    return {source_id: row.source_id, distance, residual: BigInt(row.interaction_residual_q32)};
  }).sort((left, right) => left.distance < right.distance ? -1
    : left.distance > right.distance ? 1 : left.source_id.localeCompare(right.source_id));
  const neighbors = nearest.slice(0, predictor.algorithm.neighbors);
  return {
    q: lowerMedian(neighbors.map((neighbor) => neighbor.residual)),
    neighbors: neighbors.map((neighbor) => ({
      source_id: neighbor.source_id,
      normalized_l1_q20: neighbor.distance.toString(),
      interaction_residual_q32: neighbor.residual.toString(),
    })),
  };
};

const contractBytes = fs.readFileSync(contractPath);
const structureBytes = fs.readFileSync(structurePath);
const contract = JSON.parse(contractBytes);
const structure = JSON.parse(structureBytes);
assert(contract.schema === "nsrl.production_cross_source_exchange_contract.v1"
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"wrong prospective contract");
assert(structure.schema === "nsrl.production_atomic_structure.v1", "wrong raw structure result");
const frameBytes = fs.readFileSync(contract.bindings.source_frame.path);
const predictorBytes = fs.readFileSync(contract.bindings.predictor.path);
const rawContractBytes = fs.readFileSync(contract.bindings.raw_structure_contract.path);
assert(sha256(frameBytes) === contract.bindings.source_frame.sha256
  && sha256(predictorBytes) === contract.bindings.predictor.sha256
  && sha256(rawContractBytes) === contract.bindings.raw_structure_contract.sha256,
"prospective binding hash mismatch");
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
const rawContract = JSON.parse(rawContractBytes);
assert(structure.bindings.manifest_hash === rawContract.manifest_hash
  && structure.bindings.token_stream_hash === rawContract.bindings.token_stream_hash,
"raw calibration/evaluation cube contract mismatch");
assert(predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"predictor crossed outcome firewall");
const calibrationSources = frame.sources.filter((source) => source.role === "calibration");
const evaluationSources = frame.sources.filter((source) => source.role === "evaluation");
assert(calibrationSources.length === contract.population.calibration_source_panels
  && evaluationSources.length === contract.population.evaluation_source_panels,
"source role count changed");
const roleSources = [...calibrationSources, ...evaluationSources];
assert(structure.q32.documents.length === 64 && roleSources.length <= 64,
  "raw calibration/evaluation surface changed");
const rows = roleSources.map((source, index) => {
  const document = structure.q32.documents[index];
  const binding = frame.phases.calibration_evaluation.document_bindings[document.document];
  assert(document.document === frame.phases.calibration_evaluation.document_start + index
    && binding.source_id === source.source_id && binding.analysis_role === source.role,
  "calibration/evaluation document binding changed");
  const row = rowFromDocument(source.source_id, document);
  const prediction = predict(row.features, predictor);
  return {...row, prediction, score: row.residual - prediction.q, role: source.role};
});
const calibration = rows.filter((row) => row.role === "calibration");
const evaluation = rows.filter((row) => row.role === "evaluation");
const n = calibration.length;
const rank = Number(ceilDiv(
  BigInt(n + 1) * BigInt(contract.conformal.alpha_denominator - contract.conformal.alpha_numerator),
  BigInt(contract.conformal.alpha_denominator)));
assert(rank === contract.conformal.order_statistic_rank && rank <= n,
  "conformal threshold is vacuous or rank changed");
const orderedScores = calibration.map((row) => row.score).sort(
  (left, right) => left < right ? -1 : left > right ? 1 : 0);
const correction = orderedScores[rank - 1];
const evaluateRow = (row) => {
  const covered = row.residual <= row.prediction.q + correction;
  const fires = row.lambda + row.prediction.q + correction < 0n;
  const unsafe = fires && row.delta >= 0n;
  return {
    source_id: row.source_id,
    singleton_features_q32: row.features.map(String),
    lambda_q32: row.lambda.toString(),
    predicted_interaction_residual_q32: row.prediction.q.toString(),
    interaction_residual_q32: row.residual.toString(),
    simultaneous_score_q32: row.score.toString(),
    upper_interaction_residual_q32: (row.prediction.q + correction).toString(),
    exchange_contrast_q32: row.delta.toString(),
    covered,
    fires,
    unsafe,
    neighbors: row.prediction.neighbors,
  };
};
const calibrationRows = calibration.map(evaluateRow);
const evaluationRows = evaluation.map(evaluateRow);
const uncovered = evaluationRows.filter((row) => !row.covered);
const fired = evaluationRows.filter((row) => row.fires);
const unsafe = fired.filter((row) => row.unsafe);
const aggregateFired = fired.reduce((sum, row) => sum + BigInt(row.exchange_contrast_q32), 0n);
const gates = {
  source_envelope_not_rejected: uncovered.length < contract.falsifiers.coverage.exact_binomial_rejection_failures,
  source_envelope_promotion_gate: uncovered.length <= contract.falsifiers.coverage.maximum_failures_for_support,
  unsafe_action_gate: unsafe.length === 0,
  nonvacuity_gate: fired.length >= contract.falsifiers.nonvacuity.minimum_fired_source_panels,
  incremental_value_gate: fired.length > 0 && aggregateFired < 0n,
};
const decision = Object.values(gates).every(Boolean)
  ? "supported_on_frozen_distinct_author_gutenberg_frame"
  : gates.source_envelope_not_rejected && !gates.source_envelope_promotion_gate
    ? "coverage_inconclusive_no_promotion"
    : "prospective_cross_source_certificate_falsified_or_vacuous";
const result = {
  schema: "nsrl.production_cross_source_exchange_result.v1",
  analysis_role: "prospective_untouched_evaluation",
  source_sha256: {
    contract: sha256(contractBytes),
    source_frame: sha256(frameBytes),
    predictor: sha256(predictorBytes),
    raw_structure_contract: sha256(rawContractBytes),
    raw_structure_result: sha256(structureBytes),
    analyzer: sha256(fs.readFileSync(new URL(import.meta.url))),
  },
  population: contract.population,
  exchange: contract.exchange_set[0],
  conformal: {
    alpha: `${contract.conformal.alpha_numerator}/${contract.conformal.alpha_denominator}`,
    calibration_source_panels: n,
    order_statistic_rank: rank,
    simultaneous_score: contract.conformal.simultaneous_score,
    correction_q32: correction.toString(),
    calibration_score_minimum_q32: orderedScores[0].toString(),
    calibration_score_maximum_q32: orderedScores.at(-1).toString(),
  },
  calibration: {
    source_panels: calibrationRows.length,
    rows: calibrationRows,
  },
  untouched_evaluation: {
    source_panels: evaluationRows.length,
    envelope_covered: evaluationRows.length - uncovered.length,
    envelope_uncovered: uncovered.length,
    uncovered_source_ids: uncovered.map((row) => row.source_id),
    fired_source_panels: fired.length,
    firing_rate: `${fired.length}/${evaluationRows.length}`,
    favorable_firings: fired.filter((row) => BigInt(row.exchange_contrast_q32) < 0n).length,
    tied_firings: fired.filter((row) => BigInt(row.exchange_contrast_q32) === 0n).length,
    unfavorable_firings: fired.filter((row) => BigInt(row.exchange_contrast_q32) > 0n).length,
    unsafe_firings: unsafe.length,
    aggregate_fired_exchange_contrast_q32: aggregateFired.toString(),
    rows: evaluationRows,
  },
  falsifier_gates: gates,
  decision: {
    status: decision,
    all_preregistered_support_gates_pass: Object.values(gates).every(Boolean),
    claim_scope:
      "marginal source-panel safety and nonvacuity on the frozen distinct-author English Project Gutenberg frame only",
    arbitrary_web_or_simplewiki_transfer_claimed: false,
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    documents_200_212_read: false,
  },
};
const bytes = `${JSON.stringify(result, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  correction_q32: result.conformal.correction_q32,
  untouched_evaluation: {
    source_panels: result.untouched_evaluation.source_panels,
    envelope_covered: result.untouched_evaluation.envelope_covered,
    fired_source_panels: result.untouched_evaluation.fired_source_panels,
    unsafe_firings: result.untouched_evaluation.unsafe_firings,
    aggregate_fired_exchange_contrast_q32:
      result.untouched_evaluation.aggregate_fired_exchange_contrast_q32,
  },
  gates,
  decision: result.decision.status,
  documents_200_212_read: false,
}, null, 2)}\n`);
