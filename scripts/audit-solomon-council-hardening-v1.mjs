#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {
  sha256Bytes,
  sha256Json,
  stableJson,
  verifyReceiptIdentity,
  verifyReceiptRevision,
} from "./lib/solomon-council-v0.mjs";
import {verifyWisdomCeremonyCompilation} from "./lib/solomon-wisdom-ceremony-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const auditorPath = path.relative(root, fileURLToPath(import.meta.url));
const defaults = {
  output: "benchmarks/solomon-council-v1/hardening-result.json",
  check: false,
};
const paths = {
  contract: "benchmarks/solomon-council-v1/hardening-contract.json",
  casebook: "benchmarks/solomon-council-v0/production-v0/casebook.json",
  solo_bundle: "benchmarks/solomon-council-v0/production-v0/solo-bundle.json",
  council_bundle: "benchmarks/solomon-council-v0/production-v0/council-bundle.json",
  council_invocations: "benchmarks/solomon-council-v0/production-v0/council-invocations.jsonl",
  solo_invocations: "benchmarks/solomon-council-v0/production-v0/solo-invocations.jsonl",
  council_receipts: "benchmarks/solomon-council-v0/production-v0/council-requests-receipts.jsonl",
  eval_input: "benchmarks/solomon-council-v0/production-v0/eval-input.json",
  eval_result: "benchmarks/solomon-council-v0/wisdom-eval-result.json",
  generation_integrity: "benchmarks/solomon-council-v0/production-v0/generation-integrity.json",
  provenance: "benchmarks/solomon-council-v0/production-v0/provenance.json",
  fixture_prior_receipt: "benchmarks/solomon-council-v0/fixtures/select-receipt.json",
  fixture_observation: "benchmarks/solomon-council-v0/fixtures/select-observation.json",
  fixture_revised_receipt: "benchmarks/solomon-council-v0/fixtures/select-revised-receipt.json",
};

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (relative) => path.join(root, relative);
const readJson = (relative) => JSON.parse(fs.readFileSync(absolute(relative), "utf8"));
const readJsonl = (relative) => fs.readFileSync(absolute(relative), "utf8").trimEnd()
  .split("\n").filter(Boolean).map((line) => JSON.parse(line));
const binding = (relative) => {
  const bytes = fs.readFileSync(absolute(relative));
  return {path: relative, sha256: sha256Bytes(bytes), bytes: bytes.length};
};

const config = parseArgs(process.argv.slice(2));
const contract = readJson(paths.contract);
assert(contract.schema === "nsrl.solomon_council_hardening_contract.v1",
  "wrong Council hardening contract schema");
assert(contract.frozen_before_hardening_audit === true
  && contract.v0_lane_outcomes_already_opened === true,
"hardening contract must disclose its retrospective boundary");

const casebook = readJson(paths.casebook);
const soloBundle = readJson(paths.solo_bundle);
const councilBundle = readJson(paths.council_bundle);
const evalInput = readJson(paths.eval_input);
const evalResult = readJson(paths.eval_result);
const generationIntegrity = readJson(paths.generation_integrity);
const provenance = readJson(paths.provenance);
verifyWisdomCeremonyCompilation(evalInput, {baseDir: root});
assert(evalResult.source_sha256 === sha256Json(evalInput)
  && evalResult.verdict?.promotion_gate_passed === true,
"historical v0 result is missing or no longer byte-bound to its input");
assert(generationIntegrity.ok === true && provenance.ok === true,
  "historical generation-integrity or provenance report is not green");

const fixturePriorReceipt = readJson(paths.fixture_prior_receipt);
const fixtureObservation = readJson(paths.fixture_observation);
const fixtureRevisedReceipt = readJson(paths.fixture_revised_receipt);
verifyReceiptRevision(fixtureRevisedReceipt, fixturePriorReceipt, fixtureObservation);

const evidenceCache = new Map();
const publicEvidence = casebook.cases.map((caseEntry) => ({
  caseEntry,
  evidence: selectedRecord(caseEntry.evidence[0]),
}));
const goldById = new Map(evalInput.episodes.map((episode) => [episode.episode_id, episode.gold]));
const councilById = new Map(councilBundle.episodes.map(
  (episode) => [episode.episode_id, episode.prediction]));
const parityPredictions = new Map();
for (const {caseEntry, evidence} of publicEvidence) {
  const decisionId = deriveDecision(evidence);
  assert(caseEntry.decision_ids.includes(decisionId),
    `${caseEntry.episode_id} tool-parity decision is outside the frozen decision set`);
  parityPredictions.set(caseEntry.episode_id, predictionForDecision(decisionId));
}

const dimensions = {};
for (const dimension of [...new Set(casebook.cases.map((entry) => entry.dimension))].sort()) {
  const cases = casebook.cases.filter((entry) => entry.dimension === dimension);
  const paritySolo = scoreDimension(cases, dimension,
    (entry) => parityPredictions.get(entry.episode_id), goldById);
  const council = scoreDimension(cases, dimension,
    (entry) => councilById.get(entry.episode_id), goldById);
  dimensions[dimension] = {
    cases: cases.length,
    historical_unassisted_solo: evalResult.dimensions[dimension].solo,
    tool_parity_solo: paritySolo,
    council,
    council_strictly_outperforms_tool_parity_solo:
      strictlyOutperforms(dimension, council, paritySolo),
    exact_tie: stableJson(council) === stableJson(paritySolo),
  };
}

const councilInvocationRecords = readJsonl(paths.council_invocations);
const soloInvocationRecords = readJsonl(paths.solo_invocations);
const councilToolObservations = councilInvocationRecords.filter(
  (record) => record.value?.tool_observation).map((record) => record.value.tool_observation);
const soloToolObservations = soloInvocationRecords.filter(
  (record) => record.value?.tool_observation).map((record) => record.value.tool_observation);
const receiptRecords = readJsonl(paths.council_receipts).filter(
  (record) => record.value?.schema === "nsrl.wisdom_receipt.v0");
for (const record of receiptRecords) verifyReceiptIdentity(record.value);
const episodeIds = casebook.cases.map((entry) => entry.episode_id);
const soloToolProfiles = toolObservationProfiles(soloInvocationRecords);
const councilToolProfiles = toolObservationProfiles(councilInvocationRecords);
const soloPermissionBudgetDeclarations = permissionBudgetDeclarations(soloInvocationRecords);
const councilPermissionBudgetDeclarations = permissionBudgetDeclarations(receiptRecords);
const equivalentToolObservations = equivalentLaneProfiles(
  soloToolProfiles, councilToolProfiles, episodeIds);
const equivalentToolPermissions = equivalentLaneProfiles(
  permissionProfiles(soloPermissionBudgetDeclarations),
  permissionProfiles(councilPermissionBudgetDeclarations),
  episodeIds,
);
const equivalentToolBudgets = equivalentLaneProfiles(
  budgetProfiles(soloPermissionBudgetDeclarations),
  budgetProfiles(councilPermissionBudgetDeclarations),
  episodeIds,
);
const actualSameModel = soloBundle.underlying_model_sha256
  === councilBundle.underlying_model_sha256;
const actualSamePublicCasebook = soloBundle.casebook_sha256 === councilBundle.casebook_sha256;

const adversarialEvidence = {
  misleading: publicEvidence.filter(
    ({caseEntry}) => caseEntry.dimension === "hard_negative_rejection").length,
  incomplete: publicEvidence.filter(
    ({evidence}) => evidence.kind === "incomplete_source_record").length,
  stale: publicEvidence.filter(
    ({evidence}) => evidence.adversarial_condition === "stale").length,
  conflicting: publicEvidence.filter(({evidence}) => evidence.kind === "claim_set").length,
};
const toolBoundaries = {
  successful_observations: councilToolObservations.length,
  tool_failures: councilToolObservations.filter(
    (observation) => observation.status === "failure").length,
  permission_denials: councilToolObservations.filter(
    (observation) => observation.status === "permission_denied").length,
  actual_solo_tool_observations: soloToolObservations.length,
  actual_council_tool_observations: councilToolObservations.length,
  actual_solo_permission_budget_declarations: soloPermissionBudgetDeclarations.length,
  actual_council_permission_budget_declarations: councilPermissionBudgetDeclarations.length,
  actual_equivalent_tool_observations: equivalentToolObservations,
  actual_equivalent_tool_permissions: equivalentToolPermissions,
  actual_equivalent_tool_budgets: equivalentToolBudgets,
  counterfactual_parity_solo_decisions: parityPredictions.size,
};
const humanAmbiguity = {
  human_authored_ambiguous_cases: publicEvidence.filter(
    ({evidence}) => evidence.human_authored === true && evidence.ambiguous === true).length,
  ask_user_or_abstain_cases: councilBundle.episodes.filter(
    (episode) => ["ask_user", "abstain"].includes(episode.prediction.decision_id)).length,
  deterministic_public_verifier_cases: publicEvidence.filter(
    ({evidence}) => evidence.verification_contract === "deterministic-public-evidence-v0").length,
};
const productionObserved = receiptRecords.filter(
  (record) => record.value.outcome?.status === "observed");
const productionRevised = receiptRecords.filter((record) => record.value.revisions?.length > 0);
const outcomes = {
  production_receipts: receiptRecords.length,
  production_receipts_with_observed_outcomes: productionObserved.length,
  production_receipts_with_calibration_revisions: productionRevised.length,
  distinct_observation_sources: new Set(productionRevised.flatMap(
    (record) => record.value.revisions.map((revision) => revision.observer))).size,
  fixture_revision_replays: 1,
  fixture_revision_counts_toward_production_minimum: false,
};
const unfamiliarCases = casebook.cases.filter((entry) => entry.unfamiliar_source);
const crossModalCases = casebook.cases.filter(
  (entry) => entry.dimension === "cross_modal_agreement");
const transfer = {
  unfamiliar_source_cases: unfamiliarCases.length,
  unfamiliar_source_families: new Set(unfamiliarCases.map((entry) => entry.source_family)).size,
  cross_modal_cases: crossModalCases.length,
  cross_modal_source_families: new Set(crossModalCases.map((entry) => entry.source_family)).size,
};

const gates = {
  actual_tool_parity_baseline: actualSameModel
    && actualSamePublicCasebook
    && equivalentToolObservations
    && equivalentToolPermissions
    && equivalentToolBudgets,
  strict_council_outperformance: Object.values(dimensions).every(
    (dimension) => dimension.council_strictly_outperforms_tool_parity_solo),
  adversarial_evidence_coverage: meetsMinimums(
    adversarialEvidence, contract.adversarial_evidence_minimums),
  tool_boundary_coverage: meetsMinimums(toolBoundaries, contract.tool_boundary_minimums),
  human_ambiguity_coverage: meetsMinimums(
    humanAmbiguity, contract.human_ambiguity_minimums),
  outcome_revision_coverage: meetsMinimums(outcomes, contract.outcome_minimums),
  long_transfer_coverage: meetsMinimums(transfer, contract.transfer_minimums),
  generation_integrity_and_provenance: generationIntegrity.ok === true && provenance.ok === true,
  exact_v0_ceremony_replay: true,
  all_receipts_shadow_only: receiptRecords.length === casebook.cases.length
    && receiptRecords.every((record) => record.value.mode === "shadow"
      && record.value.shadow_execution?.action_execution_allowed === false
      && record.value.shadow_execution?.action_executed === false),
};
const allPassed = Object.values(gates).every(Boolean);
assert(allPassed === false, "historical v0 unexpectedly passes the frozen hardening contract");
assert(Object.values(dimensions).every((dimension) => dimension.exact_tie),
  "tool-parity counterfactual no longer ties Council on every historical dimension");

const sourcePaths = [paths.contract, paths.casebook, paths.solo_bundle, paths.council_bundle,
  paths.council_invocations, paths.solo_invocations, paths.council_receipts, paths.eval_input,
  paths.eval_result, paths.generation_integrity, paths.provenance, paths.fixture_prior_receipt,
  paths.fixture_observation, paths.fixture_revised_receipt, auditorPath];
const result = {
  schema: "nsrl.solomon_council_hardening_result.v1",
  audit_id: contract.audit_id,
  analysis_role: contract.analysis_role,
  historical_ceremony: {
    ceremony_id: casebook.ceremony_id,
    cases: casebook.cases.length,
    underlying_model: casebook.underlying_model,
    v0_promotion_gate_passed: true,
    v0_result_preserved_as_historical_record: true,
  },
  sources: Object.fromEntries(sourcePaths.map((relative) => [sourceKey(relative), binding(relative)])),
  baseline_fairness: {
    actual_same_model: actualSameModel,
    actual_same_public_casebook: actualSamePublicCasebook,
    actual_solo_tool_observations: toolBoundaries.actual_solo_tool_observations,
    actual_council_tool_observations: toolBoundaries.actual_council_tool_observations,
    actual_solo_permission_budget_declarations:
      toolBoundaries.actual_solo_permission_budget_declarations,
    actual_council_permission_budget_declarations:
      toolBoundaries.actual_council_permission_budget_declarations,
    actual_equivalent_tool_observations: equivalentToolObservations,
    actual_equivalent_tool_permissions: equivalentToolPermissions,
    actual_equivalent_tool_budgets: equivalentToolBudgets,
    actual_equivalent_tool_access: gates.actual_tool_parity_baseline,
    counterfactual_tool_parity_cases: parityPredictions.size,
    counterfactual_role: "diagnostic_only_because_it_was_computed_after_v0_gold_opening",
  },
  dimensions,
  coverage: {
    adversarial_evidence: adversarialEvidence,
    tool_boundaries: toolBoundaries,
    human_ambiguity: humanAmbiguity,
    outcomes,
    transfer,
  },
  integrity: {
    exact_v0_ceremony_replay: true,
    generation_integrity_green: generationIntegrity.ok === true,
    provenance_green: provenance.ok === true,
    no_oracle_target_lookup: evalInput.integrity.no_oracle_target_lookup,
    no_hidden_memory: evalInput.integrity.no_hidden_memory,
    no_retrieval_target_leakage: evalInput.integrity.no_retrieval_target_leakage,
    all_receipts_shadow_only: gates.all_receipts_shadow_only,
  },
  gates: {...gates, all_passed: allPassed},
  verdict: {
    status: "falsified",
    claim: "Council v0 strictly outperforms the same underlying model under the Council-v1 hardening contract",
    reason: "The historical solo lane lacked tool parity; an explicitly diagnostic parity baseline ties the Council on every dimension, and the stale, tool-failure, permission-denial, human-ambiguity, production-outcome, and longer-transfer requirements are not met.",
  },
  authorization: {
    historical_v0_result_rewritten: false,
    effective_council_promotion_authorized: false,
    operational_action_execution_authorized: false,
    product_release_authorized: false,
    remain_shadow_only: true,
  },
  next_required_evidence: [
    "Generate prospective solo and Council lanes with identical tool observations, permissions, budgets, questions, evidence, and unopened gold.",
    "Add stale evidence, actual tool failures, permission denials, and human-authored ambiguous decisions.",
    "Record production outcomes and deterministic receipt calibration revisions from at least three observation sources.",
    "At least double both unfamiliar-source and cross-modal sets before any new promotion claim.",
  ],
};
const bytes = Buffer.from(`${JSON.stringify(result, null, 2)}\n`);
const outputPath = absolute(config.output);
if (config.check) {
  assert(fs.existsSync(outputPath), `hardening result is missing: ${config.output}`);
  assert(fs.readFileSync(outputPath).equals(bytes),
    "hardening result does not byte-replay from its frozen sources");
} else {
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  fs.writeFileSync(outputPath, bytes);
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_council_hardening_audit.v1",
  checked: config.check,
  verdict: result.verdict.status,
  historical_cases: casebook.cases.length,
  actual_solo_tool_observations: toolBoundaries.actual_solo_tool_observations,
  actual_council_tool_observations: toolBoundaries.actual_council_tool_observations,
  parity_dimensions_tied: Object.values(dimensions).filter((dimension) => dimension.exact_tie).length,
  hardening_gates_passed: Object.values(gates).filter(Boolean).length,
  hardening_gates_total: Object.keys(gates).length,
  remain_shadow_only: true,
  output: config.output,
}, null, 2)}\n`);

function parseArgs(args) {
  const value = {...defaults};
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--out") value.output = args[++index] || "";
    else if (args[index] === "--check") value.check = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  assert(value.output && !path.isAbsolute(value.output)
    && !value.output.split(/[\\/]/).includes(".."),
  "--out must be a repository-relative path");
  return value;
}

function selectedRecord(recordBinding) {
  const cacheKey = `${recordBinding.path}\0${recordBinding.sha256}`;
  if (!evidenceCache.has(cacheKey)) {
    const bytes = fs.readFileSync(absolute(recordBinding.path));
    assert(sha256Bytes(bytes) === recordBinding.sha256,
      `evidence artifact changed: ${recordBinding.path}`);
    evidenceCache.set(cacheKey, new Map(bytes.toString("utf8").trimEnd().split("\n")
      .filter(Boolean).map((line) => JSON.parse(line)).map(
        (record) => [record.artifact_id, record.value])));
  }
  const value = evidenceCache.get(cacheKey).get(recordBinding.record_id);
  assert(value && sha256Json(value) === recordBinding.record_sha256,
    `evidence record changed: ${recordBinding.record_id}`);
  return value;
}

function deriveDecision(evidence) {
  if (evidence.kind === "sealed_metadata_claim") {
    return stableJson(evidence.observed_value) === stableJson(evidence.claimed_value)
      ? "accept" : "reject";
  }
  if (evidence.kind === "claim_set") {
    const values = new Map();
    for (const claim of evidence.claims) {
      const key = `${claim.subject}\0${claim.predicate}`;
      if (!values.has(key)) values.set(key, new Set());
      values.get(key).add(stableJson(claim.value));
    }
    return [...values.values()].every((set) => set.size === 1) ? "accept" : "reject";
  }
  if (evidence.kind === "consequence_ledger") {
    return evidence.actions.map((action) => ({
      action_id: action.action_id,
      cost: action.fixed_cost_milli
        + Math.floor(action.event_probability_milli * action.event_impact_milli / 1000),
    })).sort((left, right) => left.cost - right.cost
      || left.action_id.localeCompare(right.action_id))[0].action_id;
  }
  if (evidence.kind === "incomplete_source_record") {
    return Object.hasOwn(evidence.present_fields, evidence.requested_field) ? "accept" : "abstain";
  }
  if (evidence.kind === "text_image_binding") {
    const observed = sha256Bytes(Buffer.from(evidence.observed_signature_u8_16x16));
    assert(observed === evidence.observed_signature_sha256,
      `observed signature bytes changed: ${evidence.episode_id}`);
    return evidence.claimed_signature_sha256 === observed ? "accept" : "reject";
  }
  throw new Error(`unsupported public evidence kind ${evidence.kind}`);
}

function predictionForDecision(decisionId) {
  return {
    prediction_label: decisionId === "accept",
    confidence_milli: decisionId === "abstain" ? 1000 : 900,
    abstained: decisionId === "abstain",
    decision_id: decisionId,
  };
}

function scoreDimension(cases, dimension, prediction, goldByEpisode) {
  if (dimension === "calibration") {
    const squared = cases.reduce((sum, entry) => {
      const lane = prediction(entry);
      const gold = goldByEpisode.get(entry.episode_id);
      const probability = lane.prediction_label
        ? lane.confidence_milli : 1000 - lane.confidence_milli;
      const target = gold.expected_label ? 1000 : 0;
      return sum + (probability - target) ** 2;
    }, 0);
    const mean = Math.floor(squared / cases.length);
    return {mean_brier_millionths: mean, score_per_mille: 1000 - Math.floor(mean / 1000)};
  }
  if (dimension === "decision_regret") {
    const total = cases.reduce((sum, entry) => {
      const costs = goldByEpisode.get(entry.episode_id).decision_costs_milli;
      return sum + costs[prediction(entry).decision_id] - Math.min(...Object.values(costs));
    }, 0);
    return {total_regret_milli: total, mean_regret_milli: Math.floor(total / cases.length)};
  }
  if (dimension === "appropriate_abstention") {
    const correct = cases.filter((entry) => prediction(entry).abstained
      === goldByEpisode.get(entry.episode_id).should_abstain).length;
    return {correct, score_per_mille: Math.floor(correct * 1000 / cases.length)};
  }
  const correct = cases.filter((entry) => prediction(entry).prediction_label
    === goldByEpisode.get(entry.episode_id).expected_label).length;
  return {correct, score_per_mille: Math.floor(correct * 1000 / cases.length)};
}

function strictlyOutperforms(dimension, council, baseline) {
  if (dimension === "calibration") {
    return council.mean_brier_millionths < baseline.mean_brier_millionths;
  }
  if (dimension === "decision_regret") {
    return council.mean_regret_milli < baseline.mean_regret_milli;
  }
  return council.score_per_mille > baseline.score_per_mille;
}

function meetsMinimums(observed, minimums) {
  return Object.entries(minimums).every(([key, minimum]) => observed[key] >= minimum);
}

function recordEpisodeId(record) {
  return record.value?.episode_id
    ?? record.value?.request?.request_id
    ?? record.value?.request_id
    ?? null;
}

function toolObservationProfiles(records) {
  return records.flatMap((record) => {
    const episodeId = recordEpisodeId(record);
    const observation = record.value?.tool_observation;
    return episodeId && observation ? [{episode_id: episodeId, observation}] : [];
  });
}

function permissionBudgetDeclarations(records) {
  return records.flatMap((record) => {
    const episodeId = recordEpisodeId(record);
    if (!episodeId) return [];
    const declarations = Array.isArray(record.value?.permissions_and_budget)
      ? record.value.permissions_and_budget
      : record.value?.circle ? [record.value.circle] : [];
    return declarations.map((declaration) => ({episode_id: episodeId, declaration}));
  });
}

function permissionProfiles(declarations) {
  return declarations.map(({episode_id: episodeId, declaration}) => ({
    episode_id: episodeId,
    permissions: [...(declaration.permissions ?? [])].sort(),
    tools: [...(declaration.tools ?? [])].sort(),
  }));
}

function budgetProfiles(declarations) {
  return declarations.map(({episode_id: episodeId, declaration}) => ({
    episode_id: episodeId,
    budget: declaration.budget ?? null,
  }));
}

function equivalentLaneProfiles(left, right, requiredEpisodeIds) {
  const leftEpisodes = new Set(left.map((profile) => profile.episode_id));
  const rightEpisodes = new Set(right.map((profile) => profile.episode_id));
  if (!requiredEpisodeIds.every(
    (episodeId) => leftEpisodes.has(episodeId) && rightEpisodes.has(episodeId))) {
    return false;
  }
  const canonical = (profiles) => profiles.map(stableJson).sort();
  return stableJson(canonical(left)) === stableJson(canonical(right));
}

function sourceKey(relative) {
  if (relative === auditorPath) return "auditor";
  return Object.entries(paths).find(([, value]) => value === relative)?.[0]
    ?? path.basename(relative).replace(/[^a-z0-9]+/gi, "_").replace(/^_|_$/g, "").toLowerCase();
}
