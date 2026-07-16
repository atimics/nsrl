#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-contract.json";
const structurePath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-structure.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json";

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const absolute = (value) => value < 0n ? -value : value;
const maximum = (values) => values.reduce((result, value) => value > result ? value : result);
const lowerMedian = (values) => [...values].sort(
  (left, right) => left < right ? -1 : left > right ? 1 : 0)[Math.floor((values.length - 1) / 2)];
const reconstruct = (coefficients, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
};
const gcd = (left, right) => {
  let a = absolute(left);
  let b = absolute(right);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};
const reduceFraction = (numerator, denominator) => {
  const divisor = gcd(numerator, denominator);
  return {numerator: numerator / divisor, denominator: denominator / divisor};
};
const fractionAt = (rounds, unsafe) => ({
  numerator: 5n ** BigInt(unsafe) * 15n ** BigInt(rounds - unsafe),
  denominator: 19n ** BigInt(rounds - unsafe),
});
const unsafeBoundary = (rounds, threshold) => {
  for (let unsafe = 0; unsafe <= rounds; unsafe += 1) {
    const value = fractionAt(rounds, unsafe);
    if (value.numerator >= BigInt(threshold) * value.denominator) return unsafe;
  }
  return rounds + 1;
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
  const matches = [...bySource.values()].map((rows) => rows.sort(
    (left, right) => left.distance < right.distance ? -1 : left.distance > right.distance ? 1
      : left.passage_ordinal - right.passage_ordinal)[0]).sort(
    (left, right) => left.distance < right.distance ? -1 : left.distance > right.distance ? 1
      : left.source_id.localeCompare(right.source_id));
  const neighbors = matches.slice(0, predictor.algorithm.neighbors);
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
const structureBytes = fs.readFileSync(structurePath);
const contract = JSON.parse(contractBytes);
const structure = JSON.parse(structureBytes);
assert(contract.schema === "nsrl.solomonic_judgment_contract.v1"
  && contract.analysis_role === "prospective_pre_outcome", "wrong prospective contract");
assert(structure.schema === "nsrl.production_atomic_structure.v1", "wrong structure result");
const readBound = (binding) => {
  const bytes = fs.readFileSync(binding.path);
  assert(sha256(bytes) === binding.sha256, `binding changed: ${binding.path}`);
  return {bytes, value: JSON.parse(bytes)};
};
const frameBound = readBound(contract.bindings.source_frame);
const predictorBound = readBound(contract.bindings.predictor);
const parentBound = readBound(contract.bindings.parent_result);
const rawContractBound = readBound(contract.bindings.structure_contract);
const frame = frameBound.value;
const predictor = predictorBound.value;
const parent = parentBound.value;
const rawContract = rawContractBound.value;
assert(frame.outcome_firewall.action_cube_outcomes_read === false
  && frame.outcome_firewall.original_gutenberg_fitting_frame_used === false,
"prospective source firewall changed");
assert(structure.bindings.manifest_hash === rawContract.manifest_hash
  && structure.bindings.token_stream_hash === rawContract.bindings.token_stream_hash,
"structure result does not match frozen raw contract");
assert(structure.q32.documents.length === 64, "rank-six structure surface changed");

const sourceById = new Map(frame.sources.map((source) => [source.source_id, source]));
const rowsBySource = new Map(frame.sources.map((source) => [source.source_id, []]));
const documentByNumber = new Map(structure.q32.documents.map((document) => [document.document, document]));
for (const binding of frame.execution.document_bindings.filter(
  (entry) => entry.analysis_role === "untouched_evaluation")) {
  const source = sourceById.get(binding.source_id);
  const document = documentByNumber.get(binding.document);
  assert(source && document && binding.family === source.family,
    `missing source/document binding: ${binding.source_id}:${binding.passage_ordinal}`);
  const coefficients = document.coefficients.map(BigInt);
  const features = Array.from({length: 6}, (_, atom) => coefficients[1 << atom]);
  const lambda = features[4] - features[2];
  const delta = reconstruct(coefficients, 59) - reconstruct(coefficients, 47);
  const residual = delta - lambda;
  const prediction = predict(features, predictor);
  const correction = BigInt(contract.conformal.corrections_q32[source.family]);
  rowsBySource.get(source.source_id).push({
    source, binding, features, lambda, delta, residual, prediction, correction,
    score: residual - prediction.q,
    certifiedUpperContrast: lambda + prediction.q + correction,
  });
}
for (const [sourceId, rows] of rowsBySource) {
  rows.sort((left, right) => left.binding.passage_ordinal - right.binding.passage_ordinal);
  assert(rows.length === 4, `source does not have four evaluation passages: ${sourceId}`);
}

const orderedSources = [...frame.sources].sort((left, right) => sha256(
  `${contract.controller.source_order_seed}\0${left.family}\0${left.source_id}`).localeCompare(sha256(
  `${contract.controller.source_order_seed}\0${right.family}\0${right.source_id}`)));
const facultyForFamily = Object.fromEntries(contract.faculties.filter(
  (faculty) => faculty.kind === "domain_exchange").map((faculty) => [faculty.source_family, faculty]));
const maxContrast = BigInt(contract.numeric_contract.maximum_absolute_contrast_q32);
const publicationThreshold = contract.controller.e_process.ville_threshold;
let eValue = {numerator: 1n, denominator: 1n};
let maximumEValue = {numerator: 1n, denominator: 1n};
let unsafeSourcePanels = 0;
let cumulativeSignedRegret = 0n;
let cumulativePositiveRegret = 0n;
let sequenceIndex = 0;
const judgments = [];
const sourceLedger = [];

const provenance = (role, binding) => ({path: binding.path, sha256: binding.sha256, role});
const candidateFor = ({faculty, row, historyGuardOpen}) => {
  const matches = faculty.source_family === row.source.family;
  const eligible = matches && historyGuardOpen && row.certifiedUpperContrast < 0n;
  const strength = -row.certifiedUpperContrast;
  return {
    action_id: faculty.action_id,
    faculty_id: faculty.id,
    action_family: faculty.action_family,
    eligible,
    control_mask: 47,
    candidate_mask: 59,
    evidence: {
      strength_q32: strength.toString(),
      conditions: [
        `context source family equals ${faculty.source_family}`,
        "singleton probes and predictor are available before candidate multi-atom outcome",
        "family-specific simultaneous conformal upper contrast is strictly negative",
        "unsafe e-process history guard remains below its Ville threshold",
      ],
      provenance_sha256: contract.bindings.parent_result.sha256,
    },
    predicted_benefit_q32: strength.toString(),
    possible_harm_q32: (row.certifiedUpperContrast > 0n
      ? row.certifiedUpperContrast : 0n).toString(),
    uncertainty_envelope_q32: {
      lower: (-maxContrast).toString(), upper: row.certifiedUpperContrast.toString(),
      coverage: "19/20 marginal source-panel", unit: "candidate_minus_control_q32",
    },
    intervention_cost: {coordinate_writes: 2, objective_penalty_q32: "0"},
    reversibility: {
      class: "exact", inverse_coordinate_writes: 2,
      condition: "exact only before any intervening parameter update",
    },
    falsifier: {
      condition: "opened candidate-minus-control Q32 contrast is nonnegative, the source envelope fails, or the unsafe e-process reaches 20",
      status_if_triggered: "falsified",
    },
    provenance: [
      provenance("frozen residual predictor", contract.bindings.predictor),
      provenance("frozen calibration and correction", contract.bindings.parent_result),
      provenance("untouched source and passage", contract.bindings.source_frame),
    ],
    symbolic_feature: false,
  };
};
const occultCandidate = ({row, historyGuardOpen}) => {
  const bit = crypto.createHash("sha256").update(
    `${contract.occult_feature.seed}\0${row.source.family}\0${row.source.source_id}\0${row.binding.passage_ordinal}`)
    .digest()[0] & 1;
  const activated = contract.occult_feature.activation.activated;
  return {
    action_id: "occult_hash_parity_exchange",
    faculty_id: "occult_hash_parity",
    action_family: "occult_correspondence",
    eligible: activated && bit === 1 && historyGuardOpen && row.certifiedUpperContrast < 0n,
    control_mask: 47,
    candidate_mask: 59,
    evidence: {
      strength_q32: (-row.certifiedUpperContrast).toString(),
      conditions: [
        `pre-outcome hash parity equals ${bit}`,
        `calibration compression gate activated equals ${activated}`,
        "if activated, the same conformal envelope, e-process guard, regret ledger, and falsifier apply",
      ],
      provenance_sha256: contract.bindings.parent_result.sha256,
    },
    predicted_benefit_q32: (-row.certifiedUpperContrast).toString(),
    possible_harm_q32: (row.certifiedUpperContrast > 0n
      ? row.certifiedUpperContrast : 0n).toString(),
    uncertainty_envelope_q32: {
      lower: (-maxContrast).toString(), upper: row.certifiedUpperContrast.toString(),
      coverage: "19/20 marginal source-panel", unit: "candidate_minus_control_q32",
    },
    intervention_cost: {coordinate_writes: 2, objective_penalty_q32: "0"},
    reversibility: {
      class: "exact", inverse_coordinate_writes: 2,
      condition: "exact only before any intervening parameter update",
    },
    falsifier: {
      condition: "fails the frozen compression gate or, if activated, fails any ordinary exchange safety/value gate",
      status_if_triggered: "falsified",
    },
    provenance: [
      provenance("compression and ordinary calibration evidence", contract.bindings.parent_result),
      provenance("untouched source and passage", contract.bindings.source_frame),
    ],
    symbolic_feature: true,
  };
};
const abstentionCandidate = () => ({
  action_id: "abstain", faculty_id: "abstention", action_family: "abstention", eligible: true,
  evidence: {
    strength_q32: "0",
    conditions: ["always available", "selected whenever no eligible action has a strictly negative certified upper contrast"],
    provenance_sha256: contract.bindings.source_frame.sha256,
  },
  predicted_benefit_q32: "0", possible_harm_q32: "0",
  uncertainty_envelope_q32: {
    lower: "0", upper: "0", coverage: "19/20 marginal source-panel",
    unit: "candidate_minus_control_q32",
  },
  intervention_cost: {coordinate_writes: 0, objective_penalty_q32: "0"},
  reversibility: {class: "not_applicable", inverse_coordinate_writes: 0, condition: "no intervention"},
  falsifier: {
    condition: "a preregistered eligible action has a strictly negative certified upper contrast",
    status_if_triggered: "inconclusive",
  },
  provenance: [provenance("untouched context", contract.bindings.source_frame)],
  symbolic_feature: false,
});

for (const source of orderedSources) {
  const historyGuardOpen = eValue.numerator
    < BigInt(publicationThreshold) * eValue.denominator;
  const history = {
    completed_source_panels: sourceLedger.length,
    cumulative_signed_regret_q32: cumulativeSignedRegret.toString(),
    unsafe_source_panels: unsafeSourcePanels,
    unsafe_e_value_numerator: eValue.numerator.toString(),
    unsafe_e_value_denominator: eValue.denominator.toString(),
  };
  const panelJudgments = [];
  for (const row of rowsBySource.get(source.source_id)) {
    sequenceIndex += 1;
    const domainCandidates = contract.faculties.filter(
      (faculty) => faculty.kind === "domain_exchange").map((faculty) =>
      candidateFor({faculty, row, historyGuardOpen}));
    const occult = occultCandidate({row, historyGuardOpen});
    const abstention = abstentionCandidate();
    const matching = domainCandidates.find((candidate) =>
      candidate.faculty_id === facultyForFamily[source.family].id);
    const selectedAction = matching.eligible ? matching : abstention;
    const selectedIsAction = selectedAction.action_id !== "abstain";
    const signedRegret = selectedIsAction ? row.delta : 0n;
    const positiveRegret = signedRegret > 0n ? signedRegret : 0n;
    const judgment = {
      schema: "nsrl.judgment_record.v1",
      judgment_id: `${source.source_id}:passage:${row.binding.passage_ordinal}`,
      sequence_index: sequenceIndex,
      context: {
        source_id: source.source_id, source_family: source.family,
        passage_ordinal: row.binding.passage_ordinal,
        history_available_before_action: history,
        history_guard_open: historyGuardOpen,
      },
      candidate_actions: [...domainCandidates, occult, abstention],
      selected: selectedIsAction ? {
        kind: "action", action_id: selectedAction.action_id,
        rationale: "matching domain faculty has a strictly negative conformal upper contrast and the sequential guard is open",
      } : {
        kind: "abstention", action_id: "abstain",
        rationale: historyGuardOpen
          ? "no eligible matching faculty has a strictly negative conformal upper contrast"
          : "unsafe e-process history guard is closed",
      },
      realized_outcome: {
        opened_after_selection: true,
        signed_regret_q32: signedRegret.toString(),
        positive_regret_q32: positiveRegret.toString(),
        unsafe: selectedIsAction && row.delta >= 0n,
      },
      falsifier: {
        condition: "selected action has nonnegative exact contrast, cumulative positive regret crosses its 95% boundary, or transfer/value gates fail",
        status_if_triggered: "falsified",
      },
      provenance: [
        provenance("prospective judgment contract", {path: contractPath, sha256: sha256(contractBytes)}),
        provenance("untouched action cube", {path: structurePath, sha256: sha256(structureBytes)}),
        provenance("untouched source frame", contract.bindings.source_frame),
      ],
      audit: {
        singleton_features_q32: row.features.map(String),
        lambda_q32: row.lambda.toString(),
        predicted_interaction_residual_q32: row.prediction.q.toString(),
        conformal_correction_q32: row.correction.toString(),
        certified_upper_contrast_q32: row.certifiedUpperContrast.toString(),
        interaction_residual_q32: row.residual.toString(),
        simultaneous_component_score_q32: row.score.toString(),
        exact_candidate_minus_control_q32: row.delta.toString(),
        neighbors: row.prediction.neighbors,
      },
    };
    panelJudgments.push(judgment);
    judgments.push(judgment);
  }
  const selected = panelJudgments.filter((record) => record.selected.kind === "action");
  const panelUnsafe = selected.some((record) => record.realized_outcome.unsafe);
  const panelSigned = selected.reduce(
    (sum, record) => sum + BigInt(record.realized_outcome.signed_regret_q32), 0n);
  const panelPositive = selected.reduce(
    (sum, record) => sum + BigInt(record.realized_outcome.positive_regret_q32), 0n);
  cumulativeSignedRegret += panelSigned;
  cumulativePositiveRegret += panelPositive;
  if (panelUnsafe) {
    unsafeSourcePanels += 1;
    eValue = reduceFraction(eValue.numerator * 5n, eValue.denominator);
  } else {
    eValue = reduceFraction(eValue.numerator * 15n, eValue.denominator * 19n);
  }
  if (eValue.numerator * maximumEValue.denominator
    > maximumEValue.numerator * eValue.denominator) maximumEValue = eValue;
  const panelScore = maximum(rowsBySource.get(source.source_id).map((row) => row.score));
  sourceLedger.push({
    source_index: sourceLedger.length + 1,
    source_id: source.source_id, source_family: source.family,
    order_key: sha256(`${contract.controller.source_order_seed}\0${source.family}\0${source.source_id}`),
    conformal_source_panel_score_q32: panelScore.toString(),
    conformal_correction_q32: contract.conformal.corrections_q32[source.family],
    covered: panelScore <= BigInt(contract.conformal.corrections_q32[source.family]),
    history_guard_open_before_source: historyGuardOpen,
    fired_passages: selected.length,
    unsafe_source_panel: panelUnsafe,
    signed_regret_q32: panelSigned.toString(),
    positive_regret_q32: panelPositive.toString(),
    cumulative_signed_regret_q32: cumulativeSignedRegret.toString(),
    unsafe_e_value_after: {
      numerator: eValue.numerator.toString(), denominator: eValue.denominator.toString(),
    },
  });
}

const firedJudgments = judgments.filter((record) => record.selected.kind === "action");
const familySummaries = Object.fromEntries(contract.population.families.map((family) => {
  const familyRecords = judgments.filter((record) => record.context.source_family === family);
  const fired = familyRecords.filter((record) => record.selected.kind === "action");
  const signed = fired.reduce(
    (sum, record) => sum + BigInt(record.realized_outcome.signed_regret_q32), 0n);
  const positive = fired.reduce(
    (sum, record) => sum + BigInt(record.realized_outcome.positive_regret_q32), 0n);
  return [family, {
    source_panels: sourceLedger.filter((row) => row.source_family === family).length,
    passage_judgments: familyRecords.length, fired_passages: fired.length,
    unsafe_firings: fired.filter((record) => record.realized_outcome.unsafe).length,
    signed_regret_q32: signed.toString(), positive_regret_q32: positive.toString(),
  }];
}));
const transferringFamilies = Object.entries(familySummaries).filter(([, summary]) =>
  summary.fired_passages > 0 && BigInt(summary.signed_regret_q32) < 0n).map(([family]) => family);
const coverageFailures = sourceLedger.filter((row) => !row.covered).length;
const boundary = unsafeBoundary(sourceLedger.length, publicationThreshold);
const positiveRegretBoundary = BigInt(Math.max(0, boundary - 1))
  * BigInt(contract.panel.passages_per_source) * maxContrast;
const eProcessCrossed = sourceLedger.some((row) => BigInt(row.unsafe_e_value_after.numerator)
  >= BigInt(publicationThreshold) * BigInt(row.unsafe_e_value_after.denominator));
const passConditions = {
  controller_fires_nonvacuously:
    firedJudgments.length >= contract.pass_conditions.minimum_fired_passages,
  heldout_signed_regret_beats_always_abstain: cumulativeSignedRegret < 0n,
  positive_regret_inside_preregistered_95_bound:
    cumulativePositiveRegret <= positiveRegretBoundary && !eProcessCrossed,
  transfers_across_more_than_one_source_family:
    transferringFamilies.length >= contract.pass_conditions.minimum_transfer_families,
  conformal_source_envelope_support:
    coverageFailures <= contract.pass_conditions.maximum_coverage_failures_for_support,
  negative_regret_in_every_firing_family: Object.values(familySummaries).every(
    (summary) => summary.fired_passages === 0 || BigInt(summary.signed_regret_q32) < 0n),
  symbolic_features_receive_no_exemption:
    judgments.every((record) => record.candidate_actions.filter(
      (candidate) => candidate.symbolic_feature).every((candidate) =>
      candidate.falsifier.status_if_triggered === "falsified"
      && candidate.uncertainty_envelope_q32.coverage === "19/20 marginal source-panel")),
};

const result = {
  schema: "nsrl.solomonic_judgment_result.v1",
  analysis_role: "prospective_untouched_evaluation",
  source_sha256: {
    contract: sha256(contractBytes), structure_result: sha256(structureBytes),
    source_frame: sha256(frameBound.bytes), predictor: sha256(predictorBound.bytes),
    parent_result: sha256(parentBound.bytes), structure_contract: sha256(rawContractBound.bytes),
    analyzer: sha256(fs.readFileSync(new URL(import.meta.url))),
  },
  population: contract.population,
  faculties: contract.faculties,
  occult_feature: contract.occult_feature,
  conformal: contract.conformal,
  sequential_controller: {
    source_panel_rounds: sourceLedger.length,
    adaptive_history_rule:
      "later source panels abstain if the prior unsafe e-process reaches its Ville threshold",
    unsafe_null: contract.controller.e_process.null,
    alternative: contract.controller.e_process.alternative,
    ville_threshold: publicationThreshold,
    final_e_value: {numerator: eValue.numerator.toString(), denominator: eValue.denominator.toString()},
    maximum_e_value: {
      numerator: maximumEValue.numerator.toString(), denominator: maximumEValue.denominator.toString(),
    },
    crossed_95_boundary: eProcessCrossed,
    unsafe_source_panels: unsafeSourcePanels,
    unsafe_count_boundary_at_final_round: boundary,
    maximum_unsafe_source_panels_inside_boundary: Math.max(0, boundary - 1),
    maximum_absolute_contrast_q32: maxContrast.toString(),
    cumulative_positive_regret_95_bound_q32: positiveRegretBoundary.toString(),
    source_ledger: sourceLedger,
  },
  heldout_regret: {
    comparator: "always_abstain_retain_control_mask_47",
    fired_passages: firedJudgments.length,
    abstained_passages: judgments.length - firedJudgments.length,
    signed_regret_q32: cumulativeSignedRegret.toString(),
    net_improvement_q32: (-cumulativeSignedRegret).toString(),
    positive_regret_q32: cumulativePositiveRegret.toString(),
    favorable_firings: firedJudgments.filter(
      (record) => BigInt(record.realized_outcome.signed_regret_q32) < 0n).length,
    unsafe_firings: firedJudgments.filter((record) => record.realized_outcome.unsafe).length,
  },
  transfer: {qualifying_source_families: transferringFamilies, by_family: familySummaries},
  source_envelope: {
    covered: sourceLedger.length - coverageFailures, uncovered: coverageFailures,
    exact_binomial_rejection_failures: contract.pass_conditions.coverage_rejection_failures,
  },
  pass_conditions: passConditions,
  judgments,
  authorization: {
    universal_wisdom_claimed: false, optimizer_promotion_authorized: false,
    paid_scaling_authorized: false, original_gutenberg_fitting_frame_used: false,
    documents_200_212_read: false,
  },
};
const bytes = `${JSON.stringify(result, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: result.schema, source_panels: sourceLedger.length,
  fired_passages: result.heldout_regret.fired_passages,
  signed_regret_q32: result.heldout_regret.signed_regret_q32,
  positive_regret_q32: result.heldout_regret.positive_regret_q32,
  positive_regret_95_bound_q32: result.sequential_controller.cumulative_positive_regret_95_bound_q32,
  transfer_families: transferringFamilies, coverage_failures: coverageFailures,
  pass_conditions: passConditions,
}, null, 2)}\n`);
