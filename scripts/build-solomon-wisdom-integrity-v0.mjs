#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {
  RECOMMENDING_FACULTIES,
  loadCouncilAuthority,
  sha256Bytes,
  sha256Json,
  stableJson,
} from "./lib/solomon-council-v0.mjs";
import {verifyWisdomLanesForOpening} from "./lib/solomon-wisdom-ceremony-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const checkerPath = path.relative(root, fileURLToPath(import.meta.url));
const defaults = {
  casebook: "benchmarks/solomon-council-v0/production-v0/casebook.json",
  solo: "benchmarks/solomon-council-v0/production-v0/solo-bundle.json",
  council: "benchmarks/solomon-council-v0/production-v0/council-bundle.json",
  opening: "benchmarks/solomon-council-v0/production-v0/gold-opening.json",
  sourceReport: "benchmarks/solomon-council-v0/production-v0/source-quality.json",
  generation: "benchmarks/solomon-council-v0/production-v0/generation-integrity.json",
  provenance: "benchmarks/solomon-council-v0/production-v0/provenance.json",
};
const forbiddenPublicFields = new Set([
  "gold", "expected_label", "should_abstain", "decision_costs_milli", "nonce",
]);
const toolByFaculty = {
  mathematician: "metric_checker",
  engineer: "artifact_inspector",
  historian: "source_catalog",
  skeptic: "contradiction_checker",
  consequence_planner: "consequence_ledger",
};
const config = parseArgs(process.argv.slice(2));
const readJson = (relative) => JSON.parse(fs.readFileSync(resolveRelative(relative), "utf8"));
const casebook = readJson(config.casebook);
const soloBundle = readJson(config.solo);
const councilBundle = readJson(config.council);
const opening = readJson(config.opening);

// This is the protocol-level byte replay. The independent checks below add
// source semantics, compact raw-model bindings, and leakage assertions.
verifyWisdomLanesForOpening({casebook, soloBundle, councilBundle}, {baseDir: root});
assert(casebook.analysis_role === "frozen_same_model_comparison",
  "integrity reports are only valid for a frozen same-model comparison");
assert(casebook.cases.length === 576, "production casebook must contain 576 cases");
assert(casebook.minimum_cases_per_dimension === 72,
  "production casebook must require 72 cases per dimension");

const modelHash = casebook.underlying_model.artifact.sha256;
const modelBytes = fs.readFileSync(resolveRelative(casebook.underlying_model.artifact.path));
assert(sha256Bytes(modelBytes) === modelHash, "underlying model bytes changed");
assert(soloBundle.underlying_model_sha256 === modelHash
  && councilBundle.underlying_model_sha256 === modelHash,
"solo and council are not bound to the same underlying model");

const casebookHash = sha256Json(casebook);
const soloHash = sha256Json(soloBundle);
const councilHash = sha256Json(councilBundle);
assert(opening.schema === "nsrl.solomon_wisdom_gold_opening.v0",
  "wrong gold opening schema");
assert(opening.casebook_sha256 === casebookHash
  && opening.solo_bundle_sha256 === soloHash
  && opening.council_bundle_sha256 === councilHash,
"gold opening does not bind the frozen casebook and both lane bundles");
assert(opening.opened_after_both_lane_bundles === true,
  "gold opening is not marked as occurring after both lanes");
assert(opening.gold.length === casebook.cases.length,
  "gold opening case population changed");

const authority = loadCouncilAuthority(root);
const recommendingInputCeiling = Math.min(...RECOMMENDING_FACULTIES.map(
  (facultyId) => authority.manifests.get(facultyId).manifest.resource_ceiling.input_bytes));
const bundleCache = new Map();
const parentCache = new Map();
const evidenceRecords = [];
const verificationContracts = new Set();
const parentSources = new Map();
const casesPerDimension = new Map();
let selectedRecordMaxBytes = 0;
let selectedRecordPlusQuestionMaxBytes = 0;

for (const caseEntry of casebook.cases) {
  assert(caseEntry.evidence.length === 1,
    `${caseEntry.episode_id} must select exactly one public evidence record`);
  const binding = caseEntry.evidence[0];
  assert(binding.record_id === caseEntry.episode_id,
    `${caseEntry.episode_id} evidence selector changed`);
  const evidence = selectedRecord(binding, `${caseEntry.episode_id} evidence`);
  assert(evidence.episode_id === caseEntry.episode_id
    && evidence.dimension === caseEntry.dimension,
  `${caseEntry.episode_id} public evidence identity changed`);
  assert(evidence.schema === "nsrl.solomon_wisdom_evidence.v0"
    && evidence.analysis_role === "public_evidence_no_gold",
  `${caseEntry.episode_id} is not public no-gold evidence`);
  rejectForbiddenFields(evidence, caseEntry.episode_id);
  validateEvidenceSemantics(evidence, caseEntry);
  verificationContracts.add(evidence.verification_rule);
  evidenceRecords.push({caseEntry, binding, evidence});
  casesPerDimension.set(caseEntry.dimension, (casesPerDimension.get(caseEntry.dimension) || 0) + 1);
  const recordBytes = Buffer.byteLength(JSON.stringify(evidence));
  selectedRecordMaxBytes = Math.max(selectedRecordMaxBytes, recordBytes);
  selectedRecordPlusQuestionMaxBytes = Math.max(
    selectedRecordPlusQuestionMaxBytes,
    recordBytes + Buffer.byteLength(caseEntry.question),
  );
}
assert([...casesPerDimension.values()].every((count) => count === 72),
  "each wisdom dimension must contain exactly 72 cases");
assert(new Set(evidenceRecords.map((entry) => entry.binding.record_sha256)).size === 576,
  "selected public evidence record hashes must be unique");
assert(selectedRecordPlusQuestionMaxBytes <= recommendingInputCeiling,
  "selected public evidence exceeds the smallest recommending circle input ceiling");

const soloById = new Map(soloBundle.episodes.map((entry) => [entry.episode_id, entry]));
const councilById = new Map(councilBundle.episodes.map((entry) => [entry.episode_id, entry]));
const traceHashes = [];
let rawModelScoreBindings = 0;
let toolObservationBindings = 0;
let receiptRecommendationBindings = 0;
let dissentingReceipts = 0;
let shadowOnlyReceipts = 0;

for (const {caseEntry, binding: evidenceBinding, evidence} of evidenceRecords) {
  const episodeId = caseEntry.episode_id;
  const soloEpisode = soloById.get(episodeId);
  const councilEpisode = councilById.get(episodeId);
  assert(soloEpisode && councilEpisode, `${episodeId} is missing a lane episode`);
  const derivedDecision = deriveDecision(evidence);

  const soloTrace = selectedRecord(soloEpisode.trace, `${episodeId} solo trace`);
  traceHashes.push(soloEpisode.trace.record_sha256 || soloEpisode.trace.sha256);
  assert(soloTrace.gold_accessed === false && soloTrace.hidden_memory_used === false
    && soloTrace.retrieval_target_accessed === false,
  `${episodeId} solo trace violates the leakage policy`);
  assert(soloTrace.invocations.length === 1 && soloTrace.invocations[0].role === "solo",
    `${episodeId} solo invocation shape changed`);
  const soloInvocation = auditInvocation(
    soloTrace.invocations[0], caseEntry, "solo", evidenceBinding, derivedDecision);
  rawModelScoreBindings += 1;
  assert(stableJson(soloInvocation.output.prediction) === stableJson(soloEpisode.prediction),
    `${episodeId} solo output and lane prediction differ`);

  const councilTrace = selectedRecord(councilEpisode.trace, `${episodeId} council trace`);
  traceHashes.push(councilEpisode.trace.record_sha256 || councilEpisode.trace.sha256);
  assert(councilTrace.gold_accessed === false && councilTrace.hidden_memory_used === false
    && councilTrace.retrieval_target_accessed === false,
  `${episodeId} council trace violates the leakage policy`);
  assert(councilTrace.invocations.length === RECOMMENDING_FACULTIES.length,
    `${episodeId} council invocation count changed`);
  const outputs = new Map();
  for (const invocation of councilTrace.invocations) {
    assert(RECOMMENDING_FACULTIES.includes(invocation.role),
      `${episodeId} has an unknown recommending faculty`);
    const audited = auditInvocation(
      invocation, caseEntry, invocation.role, evidenceBinding, derivedDecision);
    outputs.set(invocation.role, audited.output);
    rawModelScoreBindings += 1;
    toolObservationBindings += 1;
  }
  assert(outputs.size === RECOMMENDING_FACULTIES.length,
    `${episodeId} repeats or omits a recommending faculty`);

  const request = selectedRecord(councilEpisode.request, `${episodeId} council request`);
  const receipt = selectedRecord({
    path: councilEpisode.receipt.path,
    sha256: councilEpisode.receipt.artifact_sha256,
    record_id: councilEpisode.receipt.record_id,
    record_sha256: councilEpisode.receipt.record_sha256,
  }, `${episodeId} council receipt`);
  assert(receipt.identity.receipt_sha256 === councilEpisode.receipt.receipt_sha256,
    `${episodeId} council receipt identity changed`);
  assert(receipt.request.request_sha256 === sha256Json(request),
    `${episodeId} council request binding changed`);
  assert(receipt.bindings.models.length === 1
    && receipt.bindings.models[0].artifact_sha256 === modelHash,
  `${episodeId} receipt is not bound to exactly the shared model`);
  for (const facultyId of RECOMMENDING_FACULTIES) {
    const receiptInvocation = receipt.faculty_invocations.find(
      (entry) => entry.faculty_id === facultyId);
    assert(receiptInvocation
      && stableJson(receiptInvocation.recommendation) === stableJson(outputs.get(facultyId).recommendation),
    `${episodeId} ${facultyId} recommendation is not receipt-bound`);
    receiptRecommendationBindings += 1;
  }
  assert(receipt.deliberation.dissent.length > 0,
    `${episodeId} receipt did not preserve dissent`);
  dissentingReceipts += 1;
  assert(receipt.mode === "shadow"
    && receipt.shadow_execution.action_execution_allowed === false
    && receipt.shadow_execution.action_executed === false,
  `${episodeId} receipt escaped shadow mode`);
  shadowOnlyReceipts += 1;
  const receiptDecision = receipt.decision.kind === "select"
    ? receipt.decision.selected_action_id : receipt.decision.kind;
  assert(receiptDecision === derivedDecision
    && councilEpisode.prediction.decision_id === derivedDecision,
  `${episodeId} council decision differs from the deterministic public verifier`);
}

assert(rawModelScoreBindings === 576 * 6,
  "raw native score binding population changed");
assert(toolObservationBindings === 576 * 5,
  "faculty tool-observation population changed");
assert(receiptRecommendationBindings === 576 * 5,
  "faculty recommendation receipt-binding population changed");
assert(dissentingReceipts === 576 && shadowOnlyReceipts === 576,
  "every council receipt must preserve dissent and remain shadow-only");
assert(new Set(traceHashes).size === 1152,
  "selected trace record hashes must be unique across both lanes");

const sourceHashes = evidenceRecords.map((entry) => entry.binding.record_sha256).sort();
const sortedTraceHashes = [...traceHashes].sort();
const checker = {path: checkerPath, sha256: sha256Bytes(fs.readFileSync(fileURLToPath(import.meta.url)))};
const sourceReport = {
  schema: "nsrl.solomon_wisdom_source_quality.v0",
  ok: true,
  ceremony_id: casebook.ceremony_id,
  casebook_sha256: casebookHash,
  underlying_model: {
    model_id: casebook.underlying_model.model_id,
    artifact_sha256: modelHash,
  },
  checker,
  cases: casebook.cases.length,
  cases_per_dimension: Object.fromEntries([...casesPerDimension].sort()),
  selected_evidence_records: sourceHashes.length,
  selected_evidence_record_hashes: sourceHashes,
  parent_sources: [...parentSources.values()].sort((left, right) => left.path.localeCompare(right.path)),
  source_families: [...new Set(casebook.cases.map((entry) => entry.source_family))].sort(),
  unfamiliar_source_families: [...new Set(casebook.cases.filter(
    (entry) => entry.unfamiliar_source).map((entry) => entry.source_family))].sort(),
  deterministic_verification_rules: [...verificationContracts].sort(),
  public_gold_fields_absent: true,
  cross_modal_cases: casesPerDimension.get("cross_modal_agreement"),
  selected_record_max_bytes: selectedRecordMaxBytes,
  selected_record_plus_question_max_bytes: selectedRecordPlusQuestionMaxBytes,
  minimum_recommending_circle_input_ceiling_bytes: recommendingInputCeiling,
  selected_evidence_inside_circle_ceiling: true,
};
const sourceReportBytes = jsonBytes(sourceReport);
const sourceReportBinding = {
  path: config.sourceReport,
  sha256: sha256Bytes(sourceReportBytes),
};
const generation = {
  schema: "nsrl.wisdom_generation_integrity.v0",
  ok: true,
  model_artifact_sha256: modelHash,
  source_report: sourceReportBinding,
  gates: {
    quality_report_green: true,
    generation_integrity_green: true,
    source_grounding_green: true,
    cross_modal_agreement_green: true,
    same_model_invocation_green: true,
    trace_replay_green: true,
    faculty_output_binding_green: true,
  },
};
const provenance = {
  schema: "nsrl.wisdom_provenance_gate.v0",
  ok: true,
  model_artifact_sha256: modelHash,
  source_hashes: sourceHashes,
  trace_hashes: sortedTraceHashes,
  gates: {
    no_oracle_target_lookup: true,
    no_hidden_memory: true,
    no_retrieval_target_leakage: true,
    gold_sealed_until_both_predictions: true,
  },
};

emit(config.sourceReport, sourceReportBytes);
emit(config.generation, jsonBytes(generation));
emit(config.provenance, jsonBytes(provenance));
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_integrity_build.v0",
  checked: config.check,
  ceremony_id: casebook.ceremony_id,
  cases: casebook.cases.length,
  model_artifact_sha256: modelHash,
  selected_evidence_records: sourceHashes.length,
  selected_trace_records: sortedTraceHashes.length,
  raw_model_score_bindings: rawModelScoreBindings,
  faculty_tool_observation_bindings: toolObservationBindings,
  faculty_recommendation_receipt_bindings: receiptRecommendationBindings,
  all_gates_green: true,
}, null, 2)}\n`);

function parseArgs(args) {
  const result = {...defaults, check: false, freeze: false};
  const keys = new Map([
    ["--casebook", "casebook"], ["--solo", "solo"], ["--council", "council"],
    ["--opening", "opening"], ["--source-report", "sourceReport"],
    ["--generation", "generation"], ["--provenance", "provenance"],
  ]);
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--check") result.check = true;
    else if (args[index] === "--freeze") result.freeze = true;
    else if (keys.has(args[index])) result[keys.get(args[index])] = args[++index] || "";
    else throw new Error(`unknown argument ${args[index]}`);
  }
  assert(result.check !== result.freeze,
    "integrity builder requires exactly one of --freeze or --check");
  for (const [key, value] of Object.entries(result)) {
    if (["check", "freeze"].includes(key)) continue;
    assert(value && !path.isAbsolute(value) && !value.split(/[\\/]/).includes(".."),
      `--${key} must be a repository-relative path`);
  }
  return result;
}

function resolveRelative(relative) {
  return path.join(root, relative);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function emit(relative, bytes) {
  const absolute = resolveRelative(relative);
  if (config.check) {
    assert(fs.existsSync(absolute) && fs.readFileSync(absolute).equals(bytes),
      `wisdom integrity artifact does not byte-replay: ${relative}`);
    return;
  }
  assert(!fs.existsSync(absolute), `refusing to overwrite wisdom integrity artifact: ${relative}`);
  fs.mkdirSync(path.dirname(absolute), {recursive: true});
  fs.writeFileSync(absolute, bytes, {flag: "wx"});
}

function loadBundle(binding, label) {
  const key = `${binding.path}:${binding.sha256}`;
  if (bundleCache.has(key)) return bundleCache.get(key);
  const bytes = fs.readFileSync(resolveRelative(binding.path));
  assert(sha256Bytes(bytes) === binding.sha256, `${label} bundle byte hash changed`);
  const records = new Map();
  for (const line of bytes.toString("utf8").trimEnd().split("\n")) {
    const record = JSON.parse(line);
    assert(typeof record.artifact_id === "string" && !records.has(record.artifact_id),
      `${label} bundle has an invalid or repeated artifact id`);
    records.set(record.artifact_id, record.value);
  }
  bundleCache.set(key, records);
  return records;
}

function selectedRecord(binding, label) {
  assert(binding && typeof binding.path === "string" && typeof binding.sha256 === "string",
    `${label} binding is incomplete`);
  assert(typeof binding.record_id === "string" && typeof binding.record_sha256 === "string",
    `${label} requires a record selector`);
  const value = loadBundle(binding, label).get(binding.record_id);
  assert(value !== undefined, `${label} selected record is missing`);
  assert(sha256Json(value) === binding.record_sha256, `${label} selected record hash changed`);
  return value;
}

function rejectForbiddenFields(value, label) {
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    assert(!forbiddenPublicFields.has(key), `${label} exposes forbidden public field ${key}`);
    rejectForbiddenFields(child, label);
  }
}

function loadParent(binding, label) {
  assert(binding && typeof binding.path === "string" && typeof binding.sha256 === "string",
    `${label} parent-source binding is incomplete`);
  const current = parentSources.get(binding.path);
  assert(!current || current.sha256 === binding.sha256,
    `${label} parent source has conflicting hashes`);
  parentSources.set(binding.path, {path: binding.path, sha256: binding.sha256});
  if (parentCache.has(binding.path)) return parentCache.get(binding.path);
  const bytes = fs.readFileSync(resolveRelative(binding.path));
  assert(sha256Bytes(bytes) === binding.sha256, `${label} parent source bytes changed`);
  let value;
  if (binding.path.endsWith(".tsv")) {
    value = bytes.toString("utf8").trimEnd().split("\n").slice(1).map((line) => {
      const columns = line.split("\t");
      return {name: columns[1], signature: columns[7].split(",").map(Number)};
    });
  } else {
    value = JSON.parse(bytes).sources;
  }
  parentCache.set(binding.path, value);
  return value;
}

function validateEvidenceSemantics(evidence, caseEntry) {
  assert(evidence.verification_contract === "deterministic-public-evidence-v0",
    `${caseEntry.episode_id} verification contract changed`);
  if (evidence.kind === "sealed_metadata_claim") {
    const sources = loadParent(evidence.parent_source, caseEntry.episode_id);
    const source = sources.find((entry) => entry.source_id === evidence.source_id);
    assert(source && Object.hasOwn(source, evidence.field),
      `${caseEntry.episode_id} parent source record or field is missing`);
    assert(stableJson(source[evidence.field]) === stableJson(evidence.observed_value),
      `${caseEntry.episode_id} observed value differs from its parent source`);
    assert(evidence.source_family === (source.family || "gutenberg"),
      `${caseEntry.episode_id} source family differs from its parent source`);
  } else if (evidence.kind === "claim_set") {
    const sources = loadParent(evidence.parent_source, caseEntry.episode_id);
    assert(evidence.claims.length === 2, `${caseEntry.episode_id} claim-set size changed`);
    const source = sources.find((entry) => entry.source_id === evidence.claims[0].subject);
    assert(source && evidence.claims.every((claim) => claim.subject === source.source_id
      && claim.predicate === "title"), `${caseEntry.episode_id} claim-set subject changed`);
    assert(evidence.claims[0].value === source.title,
      `${caseEntry.episode_id} first claim differs from its parent source`);
    assert(sources.some((entry) => entry.title === evidence.claims[1].value)
      && evidence.claims[1].value !== source.title,
    `${caseEntry.episode_id} contradictory claim lacks a parent-source witness`);
  } else if (evidence.kind === "incomplete_source_record") {
    const sources = loadParent(evidence.parent_source, caseEntry.episode_id);
    const source = sources.find((entry) => entry.source_id === evidence.source_id);
    assert(source && stableJson(evidence.present_fields)
      === stableJson({title: source.title, author: source.author}),
    `${caseEntry.episode_id} incomplete record differs from its parent source`);
    assert(!Object.hasOwn(evidence.present_fields, evidence.requested_field),
      `${caseEntry.episode_id} abstention field is unexpectedly present`);
  } else if (evidence.kind === "text_image_binding") {
    const seals = loadParent(evidence.parent_source, caseEntry.episode_id);
    const claimed = seals.find((entry) => entry.name === evidence.claimed_name);
    assert(claimed && sha256Bytes(Buffer.from(claimed.signature))
      === evidence.claimed_signature_sha256,
    `${caseEntry.episode_id} claimed seal differs from its parent source`);
    assert(evidence.observed_signature_u8_16x16.length === 256
      && sha256Bytes(Buffer.from(evidence.observed_signature_u8_16x16))
        === evidence.observed_signature_sha256,
    `${caseEntry.episode_id} observed seal hash changed`);
    assert(seals.some((entry) => stableJson(entry.signature)
      === stableJson(evidence.observed_signature_u8_16x16)),
    `${caseEntry.episode_id} observed seal has no parent-source witness`);
  } else if (evidence.kind === "consequence_ledger") {
    assert(evidence.actions.length === 3
      && new Set(evidence.actions.map((entry) => entry.action_id)).size === 3,
    `${caseEntry.episode_id} consequence action set changed`);
    for (const action of evidence.actions) {
      assert([action.fixed_cost_milli, action.event_probability_milli,
        action.event_impact_milli].every(Number.isSafeInteger),
      `${caseEntry.episode_id} consequence ledger contains a non-integer`);
    }
  } else {
    throw new Error(`${caseEntry.episode_id} has unsupported public evidence kind ${evidence.kind}`);
  }
}

function deriveDecision(evidence) {
  if (evidence.kind === "sealed_metadata_claim") {
    return stableJson(evidence.observed_value) === stableJson(evidence.claimed_value)
      ? "accept" : "reject";
  }
  if (evidence.kind === "claim_set") {
    const values = new Map();
    for (const claim of evidence.claims) {
      const key = `${claim.subject}\u0000${claim.predicate}`;
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
    return evidence.claimed_signature_sha256 === evidence.observed_signature_sha256
      ? "accept" : "reject";
  }
  throw new Error(`unsupported evidence kind ${evidence.kind}`);
}

function auditInvocation(invocation, caseEntry, role, evidenceBinding, derivedDecision) {
  assert(invocation.role === role && invocation.model_sha256 === modelHash,
    `${caseEntry.episode_id} ${role} invocation model or role changed`);
  const input = selectedRecord(invocation.input, `${caseEntry.episode_id} ${role} input`);
  const output = selectedRecord(invocation.output, `${caseEntry.episode_id} ${role} output`);
  assert(input.model_sha256 === modelHash && output.model_sha256 === modelHash,
    `${caseEntry.episode_id} ${role} did not use the shared model`);
  assert(input.role === role && input.question === caseEntry.question
    && stableJson(input.evidence) === stableJson(caseEntry.evidence)
    && input.gold_accessed === false,
  `${caseEntry.episode_id} ${role} received a different task or gold`);
  auditModelScore(output.model_score, caseEntry, role);
  if (role === "solo") {
    assert(output.schema === "nsrl.solomon_solo_model_output.v0",
      `${caseEntry.episode_id} solo output schema changed`);
  } else {
    assert(output.schema === "nsrl.solomon_faculty_model_output.v0"
      && output.faculty_id === role,
    `${caseEntry.episode_id} ${role} output identity changed`);
    const observation = output.tool_observation;
    assert(observation.schema === "nsrl.solomon_faculty_tool_observation.v0"
      && observation.tool === toolByFaculty[role]
      && observation.evidence_sha256 === (evidenceBinding.record_sha256 || evidenceBinding.sha256)
      && observation.derived_decision_id === derivedDecision
      && observation.derived_without_gold === true,
    `${caseEntry.episode_id} ${role} tool observation is not public-evidence bound`);
  }
  return {input, output};
}

function auditModelScore(score, caseEntry, role) {
  assert(score && score.schema === "nsrl.solomon_native_judgment_score_binding.v0",
    `${caseEntry.episode_id} ${role} omitted the raw native model score`);
  assert(caseEntry.decision_ids.includes(score.selected_candidate_id)
    && Number.isSafeInteger(score.confidence_milli)
    && Number.isSafeInteger(score.margin_microunits)
    && /^[0-9a-f]{64}$/.test(score.score_sha256)
    && typeof score.visible_context_hash === "string"
    && score.zero_probability_tokens_q15 === 0
    && score.raw_transformer_only === true
    && score.context_independence_proven_from_weights === true
    && score.forbidden_assistance_absent === true,
  `${caseEntry.episode_id} ${role} raw model score failed its compact provenance contract`);
}
