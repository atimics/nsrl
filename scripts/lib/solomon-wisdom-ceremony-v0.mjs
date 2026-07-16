import fs from "node:fs";
import path from "node:path";

import {
  RECOMMENDING_FACULTIES,
  sha256Bytes,
  sha256Json,
  stableJson,
  verifyReceipt,
} from "./solomon-council-v0.mjs";
import {WISDOM_DIMENSIONS} from "./solomon-wisdom-constants-v0.mjs";

const shaPattern = /^[0-9a-f]{64}$/;
const laneNames = ["solo", "council"];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, required, optional, label) {
  assert(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const allowed = new Set([...required, ...optional]);
  for (const key of required) assert(Object.hasOwn(value, key), `${label} missing ${key}`);
  for (const key of Object.keys(value)) assert(allowed.has(key), `${label} has unknown field ${key}`);
}

function nonempty(value, label) {
  assert(typeof value === "string" && value.length > 0, `${label} must be a nonempty string`);
}

function hash(value, label) {
  assert(typeof value === "string" && shaPattern.test(value), `${label} must be lowercase SHA-256`);
}

function integer(value, minimum, maximum, label) {
  assert(Number.isSafeInteger(value) && value >= minimum && value <= maximum,
    `${label} must be an integer in [${minimum}, ${maximum}]`);
}

function unique(values, label) {
  assert(Array.isArray(values), `${label} must be an array`);
  assert(new Set(values).size === values.length, `${label} must not repeat values`);
}

function resolveArtifact(baseDir, artifactPath) {
  return path.isAbsolute(artifactPath) ? artifactPath : path.resolve(baseDir, artifactPath);
}

function verifyArtifact(binding, baseDir, label) {
  exactKeys(binding, ["path", "sha256"], [], label);
  nonempty(binding.path, `${label} path`);
  hash(binding.sha256, `${label} hash`);
  const resolved = resolveArtifact(baseDir, binding.path);
  assert(fs.existsSync(resolved), `${label} is missing: ${binding.path}`);
  const bytes = fs.readFileSync(resolved);
  assert(sha256Bytes(bytes) === binding.sha256, `${label} byte hash changed`);
  return {bytes, resolved};
}

function parseJsonArtifact(binding, baseDir, label) {
  const artifact = verifyArtifact(binding, baseDir, label);
  try {
    return {...artifact, value: JSON.parse(artifact.bytes)};
  } catch (error) {
    throw new Error(`${label} is not JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function validatePrediction(prediction, decisionIds, label) {
  exactKeys(prediction, [
    "prediction_label", "confidence_milli", "abstained", "decision_id",
  ], [], label);
  assert(typeof prediction.prediction_label === "boolean", `${label} label must be boolean`);
  integer(prediction.confidence_milli, 0, 1000, `${label} confidence`);
  assert(typeof prediction.abstained === "boolean", `${label} abstention must be boolean`);
  nonempty(prediction.decision_id, `${label} decision id`);
  assert(decisionIds.includes(prediction.decision_id), `${label} decision is outside the frozen casebook`);
}

function validateCasebook(casebook, baseDir) {
  exactKeys(casebook, [
    "schema", "analysis_role", "ceremony_id", "frozen_before_lane_generation",
    "minimum_cases_per_dimension", "underlying_model", "integrity_policy", "cases",
  ], [], "wisdom casebook");
  assert(casebook.schema === "nsrl.solomon_wisdom_casebook.v0", "wrong wisdom casebook schema");
  assert(["frozen_same_model_comparison", "self_test_only"].includes(casebook.analysis_role),
    "unknown wisdom casebook analysis role");
  nonempty(casebook.ceremony_id, "wisdom ceremony id");
  assert(casebook.frozen_before_lane_generation === true,
    "wisdom casebook was not frozen before lane generation");
  integer(casebook.minimum_cases_per_dimension,
    casebook.analysis_role === "frozen_same_model_comparison" ? 72 : 1,
    100000, "minimum wisdom cases");
  exactKeys(casebook.underlying_model, ["model_id", "artifact"], [], "casebook model");
  nonempty(casebook.underlying_model.model_id, "casebook model id");
  verifyArtifact(casebook.underlying_model.artifact, baseDir, "casebook model artifact");
  exactKeys(casebook.integrity_policy, [
    "no_oracle_target_lookup", "no_hidden_memory", "no_retrieval_target_leakage",
    "gold_commitment_algorithm",
  ], [], "casebook integrity policy");
  assert(casebook.integrity_policy.no_oracle_target_lookup === true
    && casebook.integrity_policy.no_hidden_memory === true
    && casebook.integrity_policy.no_retrieval_target_leakage === true,
  "casebook integrity policy permits a forbidden information path");
  assert(casebook.integrity_policy.gold_commitment_algorithm === "sha256-canonical-json-v0",
    "casebook gold commitment algorithm changed");
  assert(Array.isArray(casebook.cases), "casebook cases must be an array");
  const ids = casebook.cases.map((entry) => entry.episode_id);
  unique(ids, "casebook episode ids");
  const byId = new Map();
  for (const entry of casebook.cases) {
    exactKeys(entry, [
      "episode_id", "dimension", "source_family", "unfamiliar_source", "evidence",
      "decision_ids", "gold_commitment_sha256",
    ], [], `casebook case ${entry?.episode_id ?? "unknown"}`);
    nonempty(entry.episode_id, "casebook episode id");
    assert(WISDOM_DIMENSIONS.includes(entry.dimension), `unknown wisdom dimension ${entry.dimension}`);
    nonempty(entry.source_family, `casebook ${entry.episode_id} source family`);
    assert(typeof entry.unfamiliar_source === "boolean",
      `casebook ${entry.episode_id} unfamiliar-source flag must be boolean`);
    if (entry.dimension === "unfamiliar_source_transfer") {
      assert(entry.unfamiliar_source === true,
        `casebook ${entry.episode_id} transfer source is not unfamiliar`);
    }
    assert(Array.isArray(entry.evidence) && entry.evidence.length > 0,
      `casebook ${entry.episode_id} has no evidence`);
    const evidenceHashes = [];
    for (const [index, binding] of entry.evidence.entries()) {
      verifyArtifact(binding, baseDir, `casebook ${entry.episode_id} evidence ${index}`);
      evidenceHashes.push(binding.sha256);
    }
    unique(evidenceHashes, `casebook ${entry.episode_id} evidence hashes`);
    unique(entry.decision_ids, `casebook ${entry.episode_id} decision ids`);
    assert(entry.decision_ids.length > 0, `casebook ${entry.episode_id} has no decisions`);
    entry.decision_ids.forEach((value) => nonempty(value, `casebook ${entry.episode_id} decision id`));
    hash(entry.gold_commitment_sha256, `casebook ${entry.episode_id} gold commitment`);
    byId.set(entry.episode_id, entry);
  }
  for (const dimension of WISDOM_DIMENSIONS) {
    const count = casebook.cases.filter((entry) => entry.dimension === dimension).length;
    assert(count >= casebook.minimum_cases_per_dimension,
      `casebook ${dimension} has ${count} cases below ${casebook.minimum_cases_per_dimension}`);
  }
  return byId;
}

function validateTrace(trace, lane, caseEntry, modelHash, baseDir, label) {
  exactKeys(trace, [
    "schema", "ceremony_id", "episode_id", "lane", "underlying_model_sha256",
    "runner", "gold_accessed", "hidden_memory_used", "retrieval_target_accessed",
    "invocations", "prediction",
  ], [], label);
  assert(trace.schema === "nsrl.solomon_wisdom_lane_trace.v0", `${label} has wrong schema`);
  assert(trace.episode_id === caseEntry.episode_id, `${label} episode binding changed`);
  assert(trace.lane === lane, `${label} lane binding changed`);
  assert(trace.underlying_model_sha256 === modelHash, `${label} model binding changed`);
  verifyArtifact(trace.runner, baseDir, `${label} runner`);
  assert(trace.gold_accessed === false, `${label} accessed opened gold`);
  assert(trace.hidden_memory_used === false, `${label} used hidden memory`);
  assert(trace.retrieval_target_accessed === false, `${label} used retrieval target leakage`);
  validatePrediction(trace.prediction, caseEntry.decision_ids, `${label} prediction`);
  assert(Array.isArray(trace.invocations), `${label} invocations must be an array`);
  const expectedRoles = lane === "solo" ? ["solo"] : RECOMMENDING_FACULTIES;
  assert(trace.invocations.length === expectedRoles.length,
    `${label} invocation count ${trace.invocations.length} != ${expectedRoles.length}`);
  const roles = trace.invocations.map((invocation) => invocation.role);
  assert(stableJson([...roles].sort()) === stableJson([...expectedRoles].sort()),
    `${label} invocation roles changed`);
  const invocationIds = [];
  const outputs = new Map();
  for (const [index, invocation] of trace.invocations.entries()) {
    exactKeys(invocation, [
      "invocation_id", "role", "model_sha256", "input", "output",
    ], [], `${label} invocation ${index}`);
    nonempty(invocation.invocation_id, `${label} invocation id`);
    invocationIds.push(invocation.invocation_id);
    assert(expectedRoles.includes(invocation.role), `${label} has unauthorized role ${invocation.role}`);
    assert(invocation.model_sha256 === modelHash, `${label} invocation used another model`);
    verifyArtifact(invocation.input, baseDir, `${label} ${invocation.role} input`);
    const output = parseJsonArtifact(invocation.output, baseDir, `${label} ${invocation.role} output`);
    outputs.set(invocation.role, output.value);
  }
  unique(invocationIds, `${label} invocation ids`);
  if (lane === "solo") {
    const output = outputs.get("solo");
    exactKeys(output, [
      "schema", "episode_id", "model_sha256", "prediction",
    ], [], `${label} solo model output`);
    assert(output.schema === "nsrl.solomon_solo_model_output.v0", `${label} solo output schema changed`);
    assert(output.episode_id === caseEntry.episode_id, `${label} solo output episode changed`);
    assert(output.model_sha256 === modelHash, `${label} solo output model changed`);
    assert(stableJson(output.prediction) === stableJson(trace.prediction),
      `${label} solo prediction differs from model output`);
  }
  return outputs;
}

function validateBundle(bundle, lane, casebook, casebookHash, casesById, baseDir) {
  exactKeys(bundle, [
    "schema", "lane", "ceremony_id", "casebook_sha256", "underlying_model_sha256",
    "generated_without_opened_gold", "episodes",
  ], [], `${lane} lane bundle`);
  assert(bundle.schema === "nsrl.solomon_wisdom_lane_bundle.v0", `wrong ${lane} bundle schema`);
  assert(bundle.lane === lane, `${lane} bundle lane changed`);
  assert(bundle.ceremony_id === casebook.ceremony_id, `${lane} bundle ceremony changed`);
  assert(bundle.casebook_sha256 === casebookHash, `${lane} bundle casebook hash changed`);
  const modelHash = casebook.underlying_model.artifact.sha256;
  assert(bundle.underlying_model_sha256 === modelHash, `${lane} bundle model changed`);
  assert(bundle.generated_without_opened_gold === true, `${lane} bundle was generated after gold access`);
  assert(Array.isArray(bundle.episodes), `${lane} bundle episodes must be an array`);
  assert(bundle.episodes.length === casebook.cases.length, `${lane} bundle case count changed`);
  unique(bundle.episodes.map((entry) => entry.episode_id), `${lane} bundle episode ids`);
  const byId = new Map();
  for (const entry of bundle.episodes) {
    const required = ["episode_id", "prediction", "trace"];
    if (lane === "council") required.push("request", "receipt");
    exactKeys(entry, required, [], `${lane} bundle episode ${entry?.episode_id ?? "unknown"}`);
    const caseEntry = casesById.get(entry.episode_id);
    assert(caseEntry, `${lane} bundle has unknown episode ${entry.episode_id}`);
    validatePrediction(entry.prediction, caseEntry.decision_ids,
      `${lane} bundle ${entry.episode_id} prediction`);
    const traceArtifact = parseJsonArtifact(entry.trace, baseDir,
      `${lane} bundle ${entry.episode_id} trace`);
    const outputs = validateTrace(traceArtifact.value, lane, caseEntry, modelHash, baseDir,
      `${lane} trace ${entry.episode_id}`);
    assert(traceArtifact.value.ceremony_id === casebook.ceremony_id,
      `${lane} trace ${entry.episode_id} ceremony changed`);
    assert(stableJson(traceArtifact.value.prediction) === stableJson(entry.prediction),
      `${lane} bundle ${entry.episode_id} prediction differs from trace`);
    let receiptIdentity = "";
    if (lane === "council") {
      const requestArtifact = parseJsonArtifact(entry.request, baseDir,
        `council bundle ${entry.episode_id} request`);
      const receiptArtifact = parseJsonArtifact({path: entry.receipt.path, sha256: entry.receipt.artifact_sha256},
        baseDir, `council bundle ${entry.episode_id} receipt`);
      exactKeys(entry.receipt, ["path", "artifact_sha256", "receipt_sha256"], [],
        `council bundle ${entry.episode_id} receipt binding`);
      hash(entry.receipt.receipt_sha256, `council bundle ${entry.episode_id} receipt identity`);
      verifyReceipt(receiptArtifact.value, requestArtifact.value);
      receiptIdentity = receiptArtifact.value.identity.receipt_sha256;
      assert(receiptIdentity === entry.receipt.receipt_sha256,
        `council bundle ${entry.episode_id} receipt identity changed`);
      assert(receiptArtifact.value.request.request_sha256 === sha256Json(requestArtifact.value),
        `council bundle ${entry.episode_id} request binding changed`);
      const models = receiptArtifact.value.bindings.models;
      assert(Array.isArray(models) && models.length === 1 && models[0].artifact_sha256 === modelHash,
        `council bundle ${entry.episode_id} receipt is not bound to exactly the frozen model`);
      const receiptDecisionId = receiptArtifact.value.decision.kind === "select"
        ? receiptArtifact.value.decision.selected_action_id
        : receiptArtifact.value.decision.kind;
      assert(entry.prediction.decision_id === receiptDecisionId,
        `council bundle ${entry.episode_id} decision differs from receipt`);
      assert(entry.prediction.abstained === (receiptArtifact.value.decision.kind === "abstain"),
        `council bundle ${entry.episode_id} abstention differs from receipt`);
      assert(entry.prediction.confidence_milli === receiptArtifact.value.decision.confidence_milli,
        `council bundle ${entry.episode_id} confidence differs from receipt`);
      const sourceHashes = new Set(caseEntry.evidence.map((binding) => binding.sha256));
      const receiptSourceHashes = receiptArtifact.value.bindings.sources.map(
        (source) => source.content_sha256);
      for (const source of receiptArtifact.value.bindings.sources) {
        assert(sourceHashes.has(source.content_sha256),
          `council bundle ${entry.episode_id} receipt cites evidence outside its case`);
      }
      assert(stableJson([...sourceHashes].sort()) === stableJson([...receiptSourceHashes].sort()),
        `council bundle ${entry.episode_id} receipt evidence differs from its case`);
      for (const facultyId of RECOMMENDING_FACULTIES) {
        const output = outputs.get(facultyId);
        exactKeys(output, [
          "schema", "episode_id", "faculty_id", "model_sha256", "recommendation",
        ], [], `council ${entry.episode_id} ${facultyId} model output`);
        assert(output.schema === "nsrl.solomon_faculty_model_output.v0",
          `council ${entry.episode_id} ${facultyId} output schema changed`);
        assert(output.episode_id === entry.episode_id && output.faculty_id === facultyId,
          `council ${entry.episode_id} ${facultyId} output binding changed`);
        assert(output.model_sha256 === modelHash,
          `council ${entry.episode_id} ${facultyId} output model changed`);
        const receiptInvocation = receiptArtifact.value.faculty_invocations.find(
          (invocation) => invocation.faculty_id === facultyId);
        assert(receiptInvocation && stableJson(receiptInvocation.recommendation) === stableJson(output.recommendation),
          `council ${entry.episode_id} ${facultyId} recommendation differs from model output`);
      }
    }
    byId.set(entry.episode_id, {
      prediction: entry.prediction,
      trace_sha256: entry.trace.sha256,
      receipt_sha256: receiptIdentity,
    });
  }
  for (const caseEntry of casebook.cases) {
    assert(byId.has(caseEntry.episode_id), `${lane} bundle omits ${caseEntry.episode_id}`);
  }
  return byId;
}

function validateOpening(opening, casebook, casebookHash, soloHash, councilHash, casesById) {
  exactKeys(opening, [
    "schema", "ceremony_id", "casebook_sha256", "solo_bundle_sha256",
    "council_bundle_sha256", "opened_after_both_lane_bundles", "gold",
  ], [], "wisdom gold opening");
  assert(opening.schema === "nsrl.solomon_wisdom_gold_opening.v0",
    "wrong wisdom gold-opening schema");
  assert(opening.ceremony_id === casebook.ceremony_id, "gold opening ceremony changed");
  assert(opening.casebook_sha256 === casebookHash, "gold opening casebook changed");
  assert(opening.solo_bundle_sha256 === soloHash, "gold opening solo bundle changed");
  assert(opening.council_bundle_sha256 === councilHash, "gold opening council bundle changed");
  assert(opening.opened_after_both_lane_bundles === true,
    "gold was not opened after both lane bundles");
  assert(Array.isArray(opening.gold) && opening.gold.length === casebook.cases.length,
    "gold opening case count changed");
  unique(opening.gold.map((entry) => entry.episode_id), "gold opening episode ids");
  const byId = new Map();
  for (const entry of opening.gold) {
    exactKeys(entry, ["episode_id", "nonce", "gold"], [],
      `gold opening ${entry?.episode_id ?? "unknown"}`);
    const caseEntry = casesById.get(entry.episode_id);
    assert(caseEntry, `gold opening has unknown episode ${entry.episode_id}`);
    hash(entry.nonce, `gold opening ${entry.episode_id} nonce`);
    validateGold(entry.gold, caseEntry, `gold opening ${entry.episode_id}`);
    const commitment = sha256Json({
      episode_id: entry.episode_id,
      nonce: entry.nonce,
      gold: entry.gold,
    });
    assert(commitment === caseEntry.gold_commitment_sha256,
      `gold opening ${entry.episode_id} does not match frozen commitment`);
    byId.set(entry.episode_id, entry.gold);
  }
  return byId;
}

function verifyIntegrityReports(bindings, modelHash, expectedSources, expectedTraces, baseDir) {
  exactKeys(bindings, ["generation_integrity_report", "provenance_report"], [],
    "wisdom ceremony integrity bindings");
  const generation = parseJsonArtifact(bindings.generation_integrity_report, baseDir,
    "wisdom generation-integrity report");
  assert(generation.value.schema === "nsrl.wisdom_generation_integrity.v0"
    && generation.value.ok === true, "wisdom generation-integrity report is not green");
  assert(generation.value.model_artifact_sha256 === modelHash,
    "wisdom generation-integrity model changed");
  exactKeys(generation.value.gates, [
    "quality_report_green", "generation_integrity_green", "source_grounding_green",
    "cross_modal_agreement_green", "same_model_invocation_green", "trace_replay_green",
    "faculty_output_binding_green",
  ], [], "wisdom generation-integrity gates");
  assert(Object.values(generation.value.gates).every((value) => value === true),
    "wisdom generation-integrity report has a red gate");
  verifyArtifact(generation.value.source_report, baseDir,
    "wisdom generation-integrity source report");
  const provenance = parseJsonArtifact(bindings.provenance_report, baseDir,
    "wisdom provenance report");
  assert(provenance.value.schema === "nsrl.wisdom_provenance_gate.v0"
    && provenance.value.ok === true, "wisdom provenance report is not green");
  assert(provenance.value.model_artifact_sha256 === modelHash,
    "wisdom provenance model changed");
  exactKeys(provenance.value.gates, [
    "no_oracle_target_lookup", "no_hidden_memory", "no_retrieval_target_leakage",
    "gold_sealed_until_both_predictions",
  ], [], "wisdom provenance gates");
  assert(Object.values(provenance.value.gates).every((value) => value === true),
    "wisdom provenance report has a red gate");
  assert(stableJson([...provenance.value.source_hashes].sort())
    === stableJson([...expectedSources].sort()),
  "wisdom provenance source set differs from the casebook");
  assert(stableJson([...provenance.value.trace_hashes].sort())
    === stableJson([...expectedTraces].sort()),
  "wisdom provenance trace set differs from the sealed lane bundles");
  return {
    generation_integrity_report: {
      ...bindings.generation_integrity_report,
      schema: "nsrl.wisdom_generation_integrity.v0",
    },
    provenance_report: {
      ...bindings.provenance_report,
      schema: "nsrl.wisdom_provenance_gate.v0",
    },
  };
}

function verifyCeremonyArtifacts(bindings, values, baseDir) {
  exactKeys(bindings, [
    "casebook", "solo_bundle", "council_bundle", "gold_opening",
  ], [], "wisdom ceremony artifact bindings");
  const result = {};
  for (const [key, label] of [
    ["casebook", "casebook"],
    ["solo_bundle", "solo lane bundle"],
    ["council_bundle", "council lane bundle"],
    ["gold_opening", "gold opening"],
  ]) {
    const parsed = parseJsonArtifact(bindings[key], baseDir, `wisdom ceremony ${label}`);
    assert(stableJson(parsed.value) === stableJson(values[key]),
      `wisdom ceremony ${label} bytes differ from compiled value`);
    result[key] = bindings[key];
  }
  return result;
}

export function goldCommitment(episodeId, nonce, gold) {
  nonempty(episodeId, "gold commitment episode id");
  hash(nonce, "gold commitment nonce");
  return sha256Json({episode_id: episodeId, nonce, gold});
}

function validateGold(gold, caseEntry, label) {
  exactKeys(gold, [
    "expected_label", "should_abstain", "decision_costs_milli",
  ], [], `${label} values`);
  assert(typeof gold.expected_label === "boolean", `${label} label must be boolean`);
  assert(typeof gold.should_abstain === "boolean", `${label} abstention must be boolean`);
  const costs = gold.decision_costs_milli;
  assert(costs && typeof costs === "object" && !Array.isArray(costs),
    `${label} costs must be an object`);
  assert(stableJson(Object.keys(costs).sort()) === stableJson([...caseEntry.decision_ids].sort()),
    `${label} decisions differ from casebook`);
  for (const [decisionId, cost] of Object.entries(costs)) {
    integer(cost, 0, Number.MAX_SAFE_INTEGER, `${label} cost ${decisionId}`);
  }
  if (caseEntry.dimension === "hard_negative_rejection") {
    assert(gold.expected_label === false, `${label} hard negative is not negative`);
  }
}

export function freezeWisdomCasebook(draft, {baseDir = process.cwd()} = {}) {
  exactKeys(draft, [
    "schema", "analysis_role", "ceremony_id", "minimum_cases_per_dimension",
    "underlying_model", "integrity_policy", "cases",
  ], [], "wisdom casebook draft");
  assert(draft.schema === "nsrl.solomon_wisdom_casebook_draft.v0",
    "wrong wisdom casebook draft schema");
  assert(Array.isArray(draft.cases), "wisdom casebook draft cases must be an array");
  const cases = [];
  const gold = [];
  for (const entry of draft.cases) {
    exactKeys(entry, [
      "episode_id", "dimension", "source_family", "unfamiliar_source", "evidence",
      "decision_ids", "nonce", "gold",
    ], [], `wisdom casebook draft ${entry?.episode_id ?? "unknown"}`);
    hash(entry.nonce, `wisdom casebook draft ${entry.episode_id} nonce`);
    const publicEntry = {
      episode_id: entry.episode_id,
      dimension: entry.dimension,
      source_family: entry.source_family,
      unfamiliar_source: entry.unfamiliar_source,
      evidence: entry.evidence,
      decision_ids: entry.decision_ids,
      gold_commitment_sha256: goldCommitment(entry.episode_id, entry.nonce, entry.gold),
    };
    validateGold(entry.gold, publicEntry, `wisdom casebook draft ${entry.episode_id}`);
    cases.push(publicEntry);
    gold.push({episode_id: entry.episode_id, nonce: entry.nonce, gold: entry.gold});
  }
  const casebook = {
    schema: "nsrl.solomon_wisdom_casebook.v0",
    analysis_role: draft.analysis_role,
    ceremony_id: draft.ceremony_id,
    frozen_before_lane_generation: true,
    minimum_cases_per_dimension: draft.minimum_cases_per_dimension,
    underlying_model: draft.underlying_model,
    integrity_policy: draft.integrity_policy,
    cases,
  };
  validateCasebook(casebook, baseDir);
  return {
    casebook,
    vault: {
      schema: "nsrl.solomon_wisdom_gold_vault.v0",
      ceremony_id: casebook.ceremony_id,
      casebook_sha256: sha256Json(casebook),
      sealed_before_lane_generation: true,
      gold,
    },
  };
}

function validateVault(vault, casebook, casesById) {
  exactKeys(vault, [
    "schema", "ceremony_id", "casebook_sha256", "sealed_before_lane_generation", "gold",
  ], [], "wisdom gold vault");
  assert(vault.schema === "nsrl.solomon_wisdom_gold_vault.v0", "wrong wisdom gold-vault schema");
  assert(vault.ceremony_id === casebook.ceremony_id, "wisdom gold vault ceremony changed");
  assert(vault.casebook_sha256 === sha256Json(casebook), "wisdom gold vault casebook changed");
  assert(vault.sealed_before_lane_generation === true,
    "wisdom gold vault was not sealed before lane generation");
  assert(Array.isArray(vault.gold) && vault.gold.length === casebook.cases.length,
    "wisdom gold vault case count changed");
  unique(vault.gold.map((entry) => entry.episode_id), "wisdom gold vault episode ids");
  for (const entry of vault.gold) {
    exactKeys(entry, ["episode_id", "nonce", "gold"], [],
      `wisdom gold vault ${entry?.episode_id ?? "unknown"}`);
    const caseEntry = casesById.get(entry.episode_id);
    assert(caseEntry, `wisdom gold vault has unknown episode ${entry.episode_id}`);
    hash(entry.nonce, `wisdom gold vault ${entry.episode_id} nonce`);
    validateGold(entry.gold, caseEntry, `wisdom gold vault ${entry.episode_id}`);
    assert(goldCommitment(entry.episode_id, entry.nonce, entry.gold)
      === caseEntry.gold_commitment_sha256,
    `wisdom gold vault ${entry.episode_id} commitment changed`);
  }
}

export function verifyWisdomLanesForOpening({casebook, soloBundle, councilBundle},
  {baseDir = process.cwd()} = {}) {
  const casesById = validateCasebook(casebook, baseDir);
  const casebookHash = sha256Json(casebook);
  validateBundle(soloBundle, "solo", casebook, casebookHash, casesById, baseDir);
  validateBundle(councilBundle, "council", casebook, casebookHash, casesById, baseDir);
  return {casesById, casebookHash};
}

export function openWisdomGold({casebook, soloBundle, councilBundle, vault},
  {baseDir = process.cwd()} = {}) {
  const {casesById, casebookHash} = verifyWisdomLanesForOpening(
    {casebook, soloBundle, councilBundle}, {baseDir});
  validateVault(vault, casebook, casesById);
  return {
    schema: "nsrl.solomon_wisdom_gold_opening.v0",
    ceremony_id: casebook.ceremony_id,
    casebook_sha256: casebookHash,
    solo_bundle_sha256: sha256Json(soloBundle),
    council_bundle_sha256: sha256Json(councilBundle),
    opened_after_both_lane_bundles: true,
    gold: structuredClone(vault.gold),
  };
}

export function compileWisdomCeremony({
  casebook,
  soloBundle,
  councilBundle,
  opening,
  integrityBindings,
  ceremonyBindings = null,
}, {baseDir = process.cwd()} = {}) {
  const casesById = validateCasebook(casebook, baseDir);
  const casebookHash = sha256Json(casebook);
  const soloHash = sha256Json(soloBundle);
  const councilHash = sha256Json(councilBundle);
  const soloById = validateBundle(soloBundle, "solo", casebook, casebookHash, casesById, baseDir);
  const councilById = validateBundle(
    councilBundle, "council", casebook, casebookHash, casesById, baseDir);
  const goldById = validateOpening(
    opening, casebook, casebookHash, soloHash, councilHash, casesById);
  const sourceHashes = new Set(casebook.cases.flatMap(
    (entry) => entry.evidence.map((binding) => binding.sha256)));
  const traceHashes = new Set([
    ...[...soloById.values()].map((entry) => entry.trace_sha256),
    ...[...councilById.values()].map((entry) => entry.trace_sha256),
  ]);
  const reports = verifyIntegrityReports(
    integrityBindings, casebook.underlying_model.artifact.sha256,
    sourceHashes, traceHashes, baseDir);
  const episodes = casebook.cases.map((caseEntry) => {
    const solo = soloById.get(caseEntry.episode_id);
    const council = councilById.get(caseEntry.episode_id);
    return {
      schema: "nsrl.solomon_wisdom_episode.v0",
      episode_id: caseEntry.episode_id,
      dimension: caseEntry.dimension,
      source_family: caseEntry.source_family,
      unfamiliar_source: caseEntry.unfamiliar_source,
      evidence_sha256: caseEntry.evidence.map((binding) => binding.sha256),
      gold_opened_after_both_predictions: true,
      gold: goldById.get(caseEntry.episode_id),
      solo: {
        model_sha256: casebook.underlying_model.artifact.sha256,
        trace_sha256: solo.trace_sha256,
        ...solo.prediction,
      },
      council: {
        model_sha256: casebook.underlying_model.artifact.sha256,
        trace_sha256: council.trace_sha256,
        receipt_sha256: council.receipt_sha256,
        ...council.prediction,
      },
    };
  });
  if (casebook.analysis_role === "frozen_same_model_comparison") {
    assert(ceremonyBindings, "production wisdom evaluation requires byte-bound ceremony artifacts");
  }
  const ceremony = ceremonyBindings
    ? verifyCeremonyArtifacts(ceremonyBindings, {
      casebook,
      solo_bundle: soloBundle,
      council_bundle: councilBundle,
      gold_opening: opening,
    }, baseDir)
    : null;
  return {
    schema: "nsrl.solomon_wisdom_eval.v0",
    analysis_role: casebook.analysis_role,
    frozen_before_outcomes: casebook.frozen_before_lane_generation,
    minimum_cases_per_dimension: casebook.minimum_cases_per_dimension,
    underlying_model: {
      model_id: casebook.underlying_model.model_id,
      artifact_sha256: casebook.underlying_model.artifact.sha256,
    },
    integrity: {
      no_oracle_target_lookup: casebook.integrity_policy.no_oracle_target_lookup,
      no_hidden_memory: casebook.integrity_policy.no_hidden_memory,
      no_retrieval_target_leakage: casebook.integrity_policy.no_retrieval_target_leakage,
      generation_integrity_passed: true,
      ...reports,
    },
    ...(ceremony ? {ceremony} : {}),
    episodes,
  };
}

export function verifyWisdomCeremonyCompilation(input, {baseDir = process.cwd()} = {}) {
  assert(input && typeof input === "object" && input.ceremony,
    "production wisdom evaluation has no ceremony binding");
  const parsed = Object.fromEntries(Object.entries(input.ceremony).map(([key, binding]) => [
    key, parseJsonArtifact(binding, baseDir, `wisdom ceremony ${key}`).value,
  ]));
  const stripSchema = (binding) => ({path: binding.path, sha256: binding.sha256});
  const replay = compileWisdomCeremony({
    casebook: parsed.casebook,
    soloBundle: parsed.solo_bundle,
    councilBundle: parsed.council_bundle,
    opening: parsed.gold_opening,
    integrityBindings: {
      generation_integrity_report: stripSchema(input.integrity.generation_integrity_report),
      provenance_report: stripSchema(input.integrity.provenance_report),
    },
    ceremonyBindings: input.ceremony,
  }, {baseDir});
  assert(stableJson(replay) === stableJson(input),
    "wisdom evaluation input differs from deterministic ceremony compilation");
  return true;
}
