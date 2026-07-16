#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json";
const outputPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json";
const providedStructurePaths = process.argv.slice(4);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const ceilDiv = (numerator, denominator) => (numerator + denominator - 1n) / denominator;
const absolute = (value) => value < 0n ? -value : value;
const maximum = (values) => values.reduce((result, value) => value > result ? value : result);
const reconstruct = (coefficients, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
};
const lowerMedian = (values) => [...values].sort(
  (left, right) => left < right ? -1 : left > right ? 1 : 0)[Math.floor((values.length - 1) / 2)];
const rowFromDocument = (source, passageOrdinal, document) => {
  const coefficients = document.coefficients.map(BigInt);
  const features = Array.from({length: 6}, (_, atom) => coefficients[1 << atom]);
  const lambda = features[4] - features[2];
  const delta = reconstruct(coefficients, 59) - reconstruct(coefficients, 47);
  return {
    source_id: source.source_id, family: source.family, role: source.role,
    passage_ordinal: passageOrdinal, features, lambda, delta, residual: delta - lambda,
  };
};
const predict = (features, predictor) => {
  const scales = predictor.probe_features.median_absolute_deviation_scales_q32.map(BigInt);
  const bySource = new Map();
  for (const row of predictor.fitted_rows) {
    if (!bySource.has(row.source_id)) bySource.set(row.source_id, []);
    const training = row.singleton_features_q32.map(BigInt);
    const distance = training.reduce((sum, value, atom) =>
      sum + ((absolute(features[atom] - value) << 20n) / scales[atom]), 0n);
    bySource.get(row.source_id).push({
      source_id: row.source_id, family: row.family, passage_ordinal: row.passage_ordinal,
      distance, residual: BigInt(row.interaction_residual_q32),
    });
  }
  const sourceMatches = [...bySource.values()].map((rows) => rows.sort(
    (left, right) => left.distance < right.distance ? -1 : left.distance > right.distance ? 1
      : left.passage_ordinal - right.passage_ordinal)[0]).sort(
    (left, right) => left.distance < right.distance ? -1 : left.distance > right.distance ? 1
      : left.source_id.localeCompare(right.source_id));
  const neighbors = sourceMatches.slice(0, predictor.algorithm.neighbors);
  assert(neighbors.length === 3 && new Set(neighbors.map((row) => row.source_id)).size === 3,
    "predictor did not select three distinct fitting source panels");
  return {
    q: lowerMedian(neighbors.map((neighbor) => neighbor.residual)),
    neighbors: neighbors.map((neighbor) => ({
      source_id: neighbor.source_id, family: neighbor.family,
      passage_ordinal: neighbor.passage_ordinal,
      normalized_l1_q20: neighbor.distance.toString(),
      interaction_residual_q32: neighbor.residual.toString(),
    })),
  };
};

const contractBytes = fs.readFileSync(contractPath);
const contract = JSON.parse(contractBytes);
assert(contract.schema === "nsrl.production_multifamily_exchange_contract.v1"
  && contract.analysis_role === "prospective_pre_calibration_evaluation_outcome",
"wrong prospective contract");
const frameBytes = fs.readFileSync(contract.bindings.source_frame.path);
const predictorBytes = fs.readFileSync(contract.bindings.predictor.path);
assert(sha256(frameBytes) === contract.bindings.source_frame.sha256
  && sha256(predictorBytes) === contract.bindings.predictor.sha256,
"source frame or predictor binding changed");
const frame = JSON.parse(frameBytes);
const predictor = JSON.parse(predictorBytes);
assert(frame.outcome_firewall.action_cube_outcomes_read === false
  && predictor.firewall.calibration_outcomes_read === false
  && predictor.firewall.evaluation_outcomes_read === false,
"prospective outcome firewall changed");
const structurePaths = providedStructurePaths.length > 0 ? providedStructurePaths
  : contract.bindings.raw_structure_contracts.map((binding) => binding.result_path);
assert(structurePaths.length === 8, "need eight passage-and-shard structure results");
const structures = [];
const rawContractBytes = [];
const structureBytes = [];
for (let bindingIndex = 0; bindingIndex < 8; bindingIndex += 1) {
  const binding = contract.bindings.raw_structure_contracts[bindingIndex];
  const rawBytes = fs.readFileSync(binding.path);
  const resultBytes = fs.readFileSync(structurePaths[bindingIndex]);
  assert(sha256(rawBytes) === binding.sha256, `raw structure contract ${bindingIndex} changed`);
  const rawContract = JSON.parse(rawBytes);
  const structure = JSON.parse(resultBytes);
  assert(structure.schema === "nsrl.production_atomic_structure.v1"
    && structure.bindings.manifest_hash === rawContract.manifest_hash
    && structure.bindings.token_stream_hash === rawContract.bindings.token_stream_hash,
  `raw passage/shard ${bindingIndex} result binding changed`);
  rawContractBytes.push(rawBytes);
  structureBytes.push(resultBytes);
  structures.push({binding, structure});
}
const calibrationSources = frame.sources.filter((source) => source.role === "calibration")
  .sort((left, right) => left.family.localeCompare(right.family)
    || left.source_id.localeCompare(right.source_id));
const evaluationSources = frame.sources.filter((source) => source.role === "evaluation")
  .sort((left, right) => left.family.localeCompare(right.family)
    || left.source_id.localeCompare(right.source_id));
assert(calibrationSources.length === 76 && evaluationSources.length === 16,
  "source-panel counts changed");
const roleSources = [...calibrationSources, ...evaluationSources];
const panelRows = new Map(roleSources.map((source) => [source.source_id, {
  source_id: source.source_id, family: source.family, role: source.role, passages: [],
}]));
const sourceById = new Map(roleSources.map((source) => [source.source_id, source]));
for (let passageOrdinal = 0; passageOrdinal < 4; passageOrdinal += 1) {
  let processed = 0;
  const phaseGroup = frame.phases.confirmation_passages[passageOrdinal];
  assert(phaseGroup.passage_ordinal === passageOrdinal && phaseGroup.shards.length === 2,
    `passage ${passageOrdinal} phase group changed`);
  for (let shard = 0; shard < 2; shard += 1) {
    const {binding: rawBinding, structure} = structures[passageOrdinal * 2 + shard];
    const phase = phaseGroup.shards[shard];
    assert(rawBinding.passage_ordinal === passageOrdinal && rawBinding.shard === shard
      && structure.q32.documents.length === 64,
    `passage ${passageOrdinal} shard ${shard} surface changed`);
    for (let index = 0; index < phase.role_documents; index += 1) {
      const document = structure.q32.documents[index];
      const documentBinding = phase.document_bindings[document.document];
      const source = sourceById.get(documentBinding.source_id);
      assert(document.document === phase.document_start + index && source
        && documentBinding.family === source.family
        && documentBinding.passage_ordinal === passageOrdinal
        && documentBinding.analysis_role === source.role,
      `passage ${passageOrdinal} shard ${shard} source/document binding changed`);
      const row = rowFromDocument(source, passageOrdinal, document);
      const prediction = predict(row.features, predictor);
      panelRows.get(source.source_id).passages.push({
        ...row, prediction, score: row.residual - prediction.q,
      });
      processed += 1;
    }
  }
  assert(processed === roleSources.length,
    `passage ${passageOrdinal} did not cover every calibration/evaluation source`);
}
for (const panel of panelRows.values()) assert(panel.passages.length === 4,
  `source panel does not have four passages: ${panel.source_id}`);
const calibrationPanels = calibrationSources.map((source) => panelRows.get(source.source_id));
const evaluationPanels = evaluationSources.map((source) => panelRows.get(source.source_id));
const familyConformal = Object.fromEntries(contract.population.families.map((family) => {
  const scores = calibrationPanels.filter((panel) => panel.family === family).map(
    (panel) => maximum(panel.passages.map((row) => row.score)));
  const n = scores.length;
  const rank = Number(ceilDiv(
    BigInt(n + 1) * BigInt(contract.conformal.alpha_denominator - contract.conformal.alpha_numerator),
    BigInt(contract.conformal.alpha_denominator)));
  assert(n === 19 && rank === contract.conformal.order_statistic_rank_per_family && rank <= n,
    `${family} conformal threshold is vacuous or rank changed`);
  const ordered = [...scores].sort(
    (left, right) => left < right ? -1 : left > right ? 1 : 0);
  return [family, {
    calibration_source_panels: n, order_statistic_rank: rank,
    correction_q32: ordered[rank - 1], minimum_q32: ordered[0], maximum_q32: ordered.at(-1),
  }];
}));
const evaluatePassage = (row) => {
  const correction = familyConformal[row.family].correction_q32;
  const upperResidual = row.prediction.q + correction;
  const covered = row.residual <= upperResidual;
  const fires = row.lambda + upperResidual < 0n;
  const unsafe = fires && row.delta >= 0n;
  return {
    passage_ordinal: row.passage_ordinal,
    singleton_features_q32: row.features.map(String),
    lambda_q32: row.lambda.toString(),
    predicted_interaction_residual_q32: row.prediction.q.toString(),
    interaction_residual_q32: row.residual.toString(),
    simultaneous_component_score_q32: row.score.toString(),
    upper_interaction_residual_q32: upperResidual.toString(),
    exchange_contrast_q32: row.delta.toString(), covered, fires, unsafe,
    neighbors: row.prediction.neighbors,
  };
};
const evaluatePanel = (panel) => {
  const correction = familyConformal[panel.family].correction_q32;
  const passages = panel.passages.map(evaluatePassage);
  const score = maximum(panel.passages.map((row) => row.score));
  return {
    source_id: panel.source_id, family: panel.family,
    simultaneous_source_panel_score_q32: score.toString(),
    covered: score <= correction, fires: passages.some((row) => row.fires),
    unsafe: passages.some((row) => row.unsafe), passages,
  };
};
const calibrationRows = calibrationPanels.map(evaluatePanel);
const evaluationRows = evaluationPanels.map(evaluatePanel);
const uncovered = evaluationRows.filter((panel) => !panel.covered);
const orderedProposals = evaluationRows.flatMap((panel) => panel.passages.map((passage) => ({
  ...passage, source_id: panel.source_id, family: panel.family,
  proposal_order_key: sha256(`${contract.sequence.proposal_order_seed}\0${panel.family}\0${panel.source_id}\0${passage.passage_ordinal}`),
}))).sort((left, right) => left.proposal_order_key.localeCompare(right.proposal_order_key));
const firedPassages = orderedProposals.filter((passage) => passage.fires);
const unsafe = firedPassages.filter((passage) => passage.unsafe);
const aggregateFired = firedPassages.reduce(
  (sum, passage) => sum + BigInt(passage.exchange_contrast_q32), 0n);
let cumulativeNetImprovement = 0n;
const exchangeSequence = firedPassages.map((passage, exchangeIndex) => {
  cumulativeNetImprovement -= BigInt(passage.exchange_contrast_q32);
  return {
    exchange_index: exchangeIndex + 1,
    proposal_index: orderedProposals.indexOf(passage) + 1,
    proposal_order_key: passage.proposal_order_key,
    source_id: passage.source_id,
    family: passage.family,
    passage_ordinal: passage.passage_ordinal,
    exchange_id: contract.exchange_set[0].id,
    exchange_contrast_q32: passage.exchange_contrast_q32,
    net_improvement_q32: (-BigInt(passage.exchange_contrast_q32)).toString(),
    cumulative_net_improvement_q32: cumulativeNetImprovement.toString(),
    unsafe: passage.unsafe,
  };
});
const familySummaries = Object.fromEntries(contract.population.families.map((family) => {
  const panels = evaluationRows.filter((panel) => panel.family === family);
  const passages = firedPassages.filter((passage) => passage.family === family);
  const aggregate = passages.reduce(
    (sum, passage) => sum + BigInt(passage.exchange_contrast_q32), 0n);
  return [family, {
    evaluation_source_panels: panels.length,
    envelope_covered: panels.filter((panel) => panel.covered).length,
    envelope_uncovered: panels.filter((panel) => !panel.covered).length,
    fired_source_panels: panels.filter((panel) => panel.fires).length,
    fired_passages: passages.length,
    unsafe_firings: passages.filter((passage) => passage.unsafe).length,
    aggregate_fired_exchange_contrast_q32: aggregate.toString(),
    net_heldout_improvement_q32: (-aggregate).toString(),
  }];
}));
const firingFamilies = Object.entries(familySummaries).filter(([, summary]) =>
  summary.fired_passages > 0).map(([family]) => family);
const familyPromotions = Object.fromEntries(contract.population.families.map((family) => {
  const summary = familySummaries[family];
  const promoted = summary.fired_passages > 0 && summary.unsafe_firings === 0
    && BigInt(summary.aggregate_fired_exchange_contrast_q32) < 0n
    && summary.envelope_uncovered
      <= contract.falsifiers.coverage.maximum_failures_per_family_for_support;
  return [family, {
    status: promoted ? "promoted" : summary.fired_passages === 0 ? "abstained"
      : "withheld_by_frozen_family_gate",
    promoted,
    unsafe_promotion: promoted && summary.unsafe_firings > 0,
  }];
}));
const gates = {
  source_envelope_not_rejected:
    uncovered.length < contract.falsifiers.coverage.exact_binomial_rejection_failures,
  source_envelope_promotion_gate:
    uncovered.length <= contract.falsifiers.coverage.maximum_failures_for_support,
  per_family_envelope_gate: Object.values(familySummaries).every((summary) =>
    summary.envelope_uncovered <= contract.falsifiers.coverage.maximum_failures_per_family_for_support),
  unsafe_action_gate: unsafe.length === 0,
  nonvacuity_gate:
    firedPassages.length >= contract.falsifiers.nonvacuity.minimum_fired_passages,
  source_family_breadth_gate:
    firingFamilies.length >= contract.falsifiers.nonvacuity.minimum_firing_families,
  incremental_value_gate: firedPassages.length > 0 && aggregateFired < 0n
    && firingFamilies.every((family) =>
      BigInt(familySummaries[family].aggregate_fired_exchange_contrast_q32) < 0n),
  no_unsafe_family_promotion_gate: Object.values(familyPromotions).every(
    (promotion) => !promotion.unsafe_promotion),
};
const allGatesPass = Object.values(gates).every(Boolean);
const decision = allGatesPass ? "supported_on_frozen_four_family_multipassage_frame"
  : gates.source_envelope_not_rejected && !gates.source_envelope_promotion_gate
    ? "coverage_inconclusive_no_promotion"
    : "prospective_multifamily_certificate_falsified_or_vacuous";
const result = {
  schema: "nsrl.production_multifamily_exchange_result.v1",
  analysis_role: "prospective_untouched_evaluation",
  source_sha256: {
    contract: sha256(contractBytes), source_frame: sha256(frameBytes),
    predictor: sha256(predictorBytes),
    raw_structure_contracts: rawContractBytes.map(sha256),
    raw_structure_results: structureBytes.map(sha256),
    analyzer: sha256(fs.readFileSync(new URL(import.meta.url))),
  },
  population: contract.population,
  exchange: contract.exchange_set[0],
  conformal: {
    alpha: `${contract.conformal.alpha_numerator}/${contract.conformal.alpha_denominator}`,
    calibration_source_panels: calibrationPanels.length,
    order_statistic_rank_per_family: contract.conformal.order_statistic_rank_per_family,
    simultaneous_score: contract.conformal.simultaneous_score,
    by_family: Object.fromEntries(Object.entries(familyConformal).map(([family, values]) => [family, {
      calibration_source_panels: values.calibration_source_panels,
      order_statistic_rank: values.order_statistic_rank,
      correction_q32: values.correction_q32.toString(),
      calibration_score_minimum_q32: values.minimum_q32.toString(),
      calibration_score_maximum_q32: values.maximum_q32.toString(),
    }])),
  },
  calibration: {source_panels: calibrationRows.length, rows: calibrationRows},
  conditional_exchange_sequence: {
    proposal_order_seed: contract.sequence.proposal_order_seed,
    ordered_heldout_proposals: orderedProposals.length,
    fired_exchanges: exchangeSequence.length,
    minimum_fired_exchanges_for_support: contract.sequence.minimum_fired_exchanges_for_support,
    aggregate_candidate_minus_control_q32: aggregateFired.toString(),
    net_heldout_improvement_q32: (-aggregateFired).toString(),
    actions: exchangeSequence,
  },
  untouched_evaluation: {
    source_panels: evaluationRows.length,
    passage_documents: evaluationRows.length * 4,
    envelope_covered: evaluationRows.length - uncovered.length,
    envelope_uncovered: uncovered.length,
    uncovered_source_ids: uncovered.map((panel) => panel.source_id),
    fired_source_panels: evaluationRows.filter((panel) => panel.fires).length,
    fired_passages: firedPassages.length,
    firing_families: firingFamilies,
    favorable_firings: firedPassages.filter(
      (row) => BigInt(row.exchange_contrast_q32) < 0n).length,
    tied_firings: firedPassages.filter(
      (row) => BigInt(row.exchange_contrast_q32) === 0n).length,
    unfavorable_firings: firedPassages.filter(
      (row) => BigInt(row.exchange_contrast_q32) > 0n).length,
    unsafe_firings: unsafe.length,
    aggregate_fired_exchange_contrast_q32: aggregateFired.toString(),
    net_heldout_improvement_q32: (-aggregateFired).toString(),
    by_family: familySummaries,
    family_promotions: familyPromotions,
    rows: evaluationRows,
  },
  falsifier_gates: gates,
  decision: {
    status: decision, all_preregistered_support_gates_pass: allGatesPass,
    claim_scope: "marginal source-panel safety and multipassage nonvacuity on the frozen four-family acquisition frame only",
    arbitrary_text_or_future_source_transfer_claimed: false,
    optimizer_change_authorized: false, paid_scaling_authorized: false,
    documents_200_212_read: false,
  },
};
const bytes = `${JSON.stringify(result, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  corrections_q32: Object.fromEntries(Object.entries(result.conformal.by_family).map(
    ([family, values]) => [family, values.correction_q32])),
  untouched_evaluation: {
    source_panels: result.untouched_evaluation.source_panels,
    envelope_covered: result.untouched_evaluation.envelope_covered,
    fired_source_panels: result.untouched_evaluation.fired_source_panels,
    fired_passages: result.untouched_evaluation.fired_passages,
    firing_families: result.untouched_evaluation.firing_families,
    unsafe_firings: result.untouched_evaluation.unsafe_firings,
    aggregate_fired_exchange_contrast_q32:
      result.untouched_evaluation.aggregate_fired_exchange_contrast_q32,
    net_heldout_improvement_q32: result.untouched_evaluation.net_heldout_improvement_q32,
  },
  gates, decision, documents_200_212_read: false,
}, null, 2)}\n`);
