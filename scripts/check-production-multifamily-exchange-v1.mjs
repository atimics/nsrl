#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const resultPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json";
const framePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json";
const predictorPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-predictor.json";
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const maximum = (values) => values.reduce((result, value) => value > result ? value : result);
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

const resultBytes = fs.readFileSync(resultPath);
const contractBytes = fs.readFileSync(contractPath);
const frameBytes = fs.readFileSync(framePath);
const predictorBytes = fs.readFileSync(predictorPath);
const result = JSON.parse(resultBytes);
const contract = JSON.parse(contractBytes);
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
assert(result.schema === "nsrl.production_multifamily_exchange_result.v1"
  && result.analysis_role === "prospective_untouched_evaluation", "wrong result schema");
assert(contract.schema === "nsrl.production_multifamily_exchange_contract.v1"
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"contract is not prospective");
assert(result.source_sha256.contract === sha256(contractBytes)
  && result.source_sha256.source_frame === sha256(frameBytes)
  && result.source_sha256.predictor === sha256(predictorBytes)
  && contract.bindings.source_frame.sha256 === sha256(frameBytes)
  && contract.bindings.predictor.sha256 === sha256(predictorBytes),
"top-level replay binding changed");
const analyzerPath = new URL("./analyze-production-multifamily-exchange-v1.mjs", import.meta.url);
const checkerPath = new URL(import.meta.url);
assert(result.source_sha256.analyzer === sha256(fs.readFileSync(analyzerPath))
  && contract.bindings.analyzer.sha256 === sha256(fs.readFileSync(analyzerPath))
  && contract.bindings.checker.sha256 === sha256(fs.readFileSync(checkerPath)),
"analyzer/checker hash binding changed");
assert(frame.outcome_firewall.action_cube_outcomes_read === false
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"prospective outcome firewall changed");
for (const [ordinal, binding] of contract.bindings.raw_structure_contracts.entries()) {
  const rawContractBytes = fs.readFileSync(binding.path);
  const rawResultBytes = fs.readFileSync(binding.result_path);
  const rawContract = JSON.parse(rawContractBytes);
  const rawResult = JSON.parse(rawResultBytes);
  assert(binding.passage_ordinal === Math.floor(ordinal / 2) && binding.shard === ordinal % 2
    && sha256(rawContractBytes) === binding.sha256
    && result.source_sha256.raw_structure_contracts[ordinal] === sha256(rawContractBytes)
    && result.source_sha256.raw_structure_results[ordinal] === sha256(rawResultBytes)
    && rawResult.bindings.manifest_hash === rawContract.manifest_hash
    && rawResult.bindings.token_stream_hash === rawContract.bindings.token_stream_hash,
  `raw passage ${ordinal} replay binding changed`);
}

const roles = Object.fromEntries(["fitting", "calibration", "evaluation"].map((role) => [
  role, new Set(frame.sources.filter((source) => source.role === role).map((source) => source.source_id)),
]));
assert(roles.fitting.size === 12 && roles.calibration.size === 76 && roles.evaluation.size === 16,
  "source-panel role counts changed");
assert(new Set(frame.sources.map((source) => source.source_id)).size === 104
  && new Set(frame.sources.map((source) => source.sha256)).size === 104
  && new Set(frame.sources.flatMap((source) => source.passages.map(
    (passage) => passage.sha256))).size === 416,
"source publications or sampled passages are not globally distinct");
assert([...roles.fitting].every((source) => !roles.calibration.has(source) && !roles.evaluation.has(source))
  && [...roles.calibration].every((source) => !roles.evaluation.has(source)),
"a source crossed fitting/calibration/evaluation roles");
for (const family of contract.population.families) {
  const sources = frame.sources.filter((source) => source.family === family);
  assert(sources.length === 26 && new Set(sources.map((source) => source.independence_key)).size === 26
    && sources.filter((source) => source.role === "fitting").length === 3
    && sources.filter((source) => source.role === "calibration").length === 19
    && sources.filter((source) => source.role === "evaluation").length === 4
    && sources.every((source) => source.passages.length === 4),
  `${family} source/passage design changed`);
  for (const source of sources) {
    for (let ordinal = 1; ordinal < source.passages.length; ordinal += 1) {
      assert(source.passages[ordinal - 1].byte_offset + source.passages[ordinal - 1].bytes
        <= source.passages[ordinal].byte_offset, `${source.source_id} passages overlap`);
    }
  }
}
assert(new Set(predictor.fitted_rows.map((row) => row.source_id)).size === 12
  && predictor.fitted_rows.length === 48
  && predictor.fitted_rows.every((row) => roles.fitting.has(row.source_id)),
"predictor contains non-fitting data or the wrong passage count");
assert(contract.panel_sampling.passage_documents_per_source === 4
  && contract.panel_sampling.model_windows_per_passage_document === 2
  && contract.exchange_set.length === 1
  && contract.exchange_set[0].control_mask === 47
  && contract.exchange_set[0].candidate_mask === 59
  && contract.sequence.minimum_fired_exchanges_for_support === 8
  && contract.falsifiers.nonvacuity.minimum_fired_passages === 8,
"panel or exchange design changed");

const rank = Number(ceilDiv(20n * 19n, 20n));
assert(rank === 19 && result.conformal.order_statistic_rank_per_family === rank,
  "19-source per-family 95% conformal rank changed");
const corrections = Object.fromEntries(contract.population.families.map((family) => {
  const calibrationScores = result.calibration.rows.filter((panel) => panel.family === family).map(
    (panel) => BigInt(panel.simultaneous_source_panel_score_q32)).sort(
    (left, right) => left < right ? -1 : left > right ? 1 : 0);
  assert(calibrationScores.length === 19, `${family} calibration source-panel count changed`);
  const correction = calibrationScores[rank - 1];
  assert(BigInt(result.conformal.by_family[family].correction_q32) === correction,
    `${family} correction is not the frozen source-panel order statistic`);
  return [family, correction];
}));
const checkPanel = (panel) => {
  const correction = corrections[panel.family];
  assert(panel.passages.length === 4, "source panel passage count changed");
  const scores = [];
  for (const passage of panel.passages) {
    const lambda = BigInt(passage.lambda_q32);
    const predicted = BigInt(passage.predicted_interaction_residual_q32);
    const residual = BigInt(passage.interaction_residual_q32);
    const delta = BigInt(passage.exchange_contrast_q32);
    const score = BigInt(passage.simultaneous_component_score_q32);
    scores.push(score);
    assert(score === residual - predicted, "passage simultaneous component changed");
    assert(BigInt(passage.upper_interaction_residual_q32) === predicted + correction,
      "passage upper residual changed");
    assert(passage.covered === (residual <= predicted + correction), "passage coverage flag changed");
    assert(passage.fires === (lambda + predicted + correction < 0n), "strict passage firing rule changed");
    assert(passage.unsafe === (passage.fires && delta >= 0n), "unsafe passage flag changed");
    assert(passage.neighbors.length === 3
      && new Set(passage.neighbors.map((neighbor) => neighbor.source_id)).size === 3,
    "predictor neighbors are not three distinct fitting source panels");
  }
  const panelScore = maximum(scores);
  assert(BigInt(panel.simultaneous_source_panel_score_q32) === panelScore
    && panel.covered === (panelScore <= correction)
    && panel.fires === panel.passages.some((passage) => passage.fires)
    && panel.unsafe === panel.passages.some((passage) => passage.unsafe),
  "source-panel maximum or summary changed");
};
result.calibration.rows.forEach(checkPanel);
result.untouched_evaluation.rows.forEach(checkPanel);
const evaluation = result.untouched_evaluation.rows;
const uncovered = evaluation.filter((panel) => !panel.covered);
const orderedProposals = evaluation.flatMap((panel) => panel.passages.map((passage) => ({
  ...passage, source_id: panel.source_id, family: panel.family,
  proposal_order_key: sha256(`${contract.sequence.proposal_order_seed}\0${panel.family}\0${panel.source_id}\0${passage.passage_ordinal}`),
}))).sort((left, right) => left.proposal_order_key.localeCompare(right.proposal_order_key));
const firedPassages = orderedProposals.filter((passage) => passage.fires);
const unsafe = firedPassages.filter((passage) => passage.unsafe);
const aggregate = firedPassages.reduce(
  (sum, passage) => sum + BigInt(passage.exchange_contrast_q32), 0n);
assert(evaluation.length === 16
  && result.untouched_evaluation.envelope_uncovered === uncovered.length
  && result.untouched_evaluation.fired_source_panels === evaluation.filter((panel) => panel.fires).length
  && result.untouched_evaluation.fired_passages === firedPassages.length
  && result.untouched_evaluation.unsafe_firings === unsafe.length
  && BigInt(result.untouched_evaluation.aggregate_fired_exchange_contrast_q32) === aggregate,
"evaluation summary changed");
assert(result.conditional_exchange_sequence.ordered_heldout_proposals === orderedProposals.length
  && result.conditional_exchange_sequence.fired_exchanges === firedPassages.length
  && BigInt(result.conditional_exchange_sequence.net_heldout_improvement_q32) === -aggregate
  && result.conditional_exchange_sequence.actions.length === firedPassages.length,
"conditional exchange sequence summary changed");
let cumulativeNetImprovement = 0n;
for (const [index, passage] of firedPassages.entries()) {
  const action = result.conditional_exchange_sequence.actions[index];
  cumulativeNetImprovement -= BigInt(passage.exchange_contrast_q32);
  assert(action.exchange_index === index + 1
    && action.proposal_index === orderedProposals.indexOf(passage) + 1
    && action.proposal_order_key === passage.proposal_order_key
    && action.source_id === passage.source_id && action.family === passage.family
    && action.passage_ordinal === passage.passage_ordinal
    && BigInt(action.exchange_contrast_q32) === BigInt(passage.exchange_contrast_q32)
    && BigInt(action.net_improvement_q32) === -BigInt(passage.exchange_contrast_q32)
    && BigInt(action.cumulative_net_improvement_q32) === cumulativeNetImprovement
    && action.unsafe === passage.unsafe,
  "ordered conditional exchange action changed");
}
const familySummaries = result.untouched_evaluation.by_family;
for (const family of contract.population.families) {
  const panels = evaluation.filter((panel) => panel.family === family);
  const passages = firedPassages.filter((passage) => passage.family === family);
  const familyAggregate = passages.reduce(
    (sum, passage) => sum + BigInt(passage.exchange_contrast_q32), 0n);
  assert(panels.length === 4
    && familySummaries[family].envelope_uncovered === panels.filter((panel) => !panel.covered).length
    && familySummaries[family].fired_passages === passages.length
    && familySummaries[family].unsafe_firings === passages.filter((passage) => passage.unsafe).length
    && BigInt(familySummaries[family].aggregate_fired_exchange_contrast_q32) === familyAggregate
    && BigInt(familySummaries[family].net_heldout_improvement_q32) === -familyAggregate,
  `${family} evaluation summary changed`);
  const expectedPromotion = passages.length > 0
    && passages.filter((passage) => passage.unsafe).length === 0 && familyAggregate < 0n
    && familySummaries[family].envelope_uncovered
      <= contract.falsifiers.coverage.maximum_failures_per_family_for_support;
  assert(result.untouched_evaluation.family_promotions[family].promoted === expectedPromotion
    && result.untouched_evaluation.family_promotions[family].unsafe_promotion
      === (expectedPromotion && passages.some((passage) => passage.unsafe)),
  `${family} frozen promotion rule changed`);
}
assert(binomialUpperTail(16, 0.05, 3) <= 0.05
  && binomialUpperTail(16, 0.05, 2) > 0.05
  && contract.falsifiers.coverage.exact_binomial_rejection_failures === 3,
"coverage falsifier is not the exact one-sided 5% threshold");
const firingFamilies = contract.population.families.filter(
  (family) => familySummaries[family].fired_passages > 0);
const expectedGates = {
  source_envelope_not_rejected: uncovered.length < 3,
  source_envelope_promotion_gate:
    uncovered.length <= contract.falsifiers.coverage.maximum_failures_for_support,
  per_family_envelope_gate: contract.population.families.every((family) =>
    familySummaries[family].envelope_uncovered
      <= contract.falsifiers.coverage.maximum_failures_per_family_for_support),
  unsafe_action_gate: unsafe.length === 0,
  nonvacuity_gate:
    firedPassages.length >= contract.falsifiers.nonvacuity.minimum_fired_passages,
  source_family_breadth_gate:
    firingFamilies.length >= contract.falsifiers.nonvacuity.minimum_firing_families,
  incremental_value_gate: firedPassages.length > 0 && aggregate < 0n
    && firingFamilies.every((family) =>
      BigInt(familySummaries[family].aggregate_fired_exchange_contrast_q32) < 0n),
  no_unsafe_family_promotion_gate: contract.population.families.every((family) =>
    !result.untouched_evaluation.family_promotions[family].unsafe_promotion),
};
assert(Object.keys(expectedGates).every(
  (key) => result.falsifier_gates[key] === expectedGates[key]), "falsifier gate changed");
assert(result.decision.optimizer_change_authorized === false
  && result.decision.paid_scaling_authorized === false
  && result.decision.documents_200_212_read === false
  && contract.authorization.read_documents_200_212 === false,
"experiment escaped its authorization boundary");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_multifamily_exchange_check.v1",
  families: contract.population.families,
  total_source_panels: 104, fitting_source_panels: 12,
  calibration_source_panels: 76, evaluation_source_panels: 16,
  passages_per_source_panel: 4, conformal_order_statistic_rank: rank,
  corrections_q32: Object.fromEntries(Object.entries(corrections).map(
    ([family, correction]) => [family, correction.toString()])),
  envelope_uncovered: uncovered.length,
  fired_source_panels: evaluation.filter((panel) => panel.fires).length,
  fired_passages: firedPassages.length, firing_families: firingFamilies,
  unsafe_firings: unsafe.length,
  aggregate_fired_exchange_contrast_q32: aggregate.toString(),
  net_heldout_improvement_q32: (-aggregate).toString(),
  unsafe_promoted_families: contract.population.families.filter(
    (family) => result.untouched_evaluation.family_promotions[family].unsafe_promotion),
  gates: expectedGates, decision: result.decision.status,
  documents_200_212_read: false,
}, null, 2)}\n`);
