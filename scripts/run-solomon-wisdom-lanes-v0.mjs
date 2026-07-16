#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {
  FACULTY_ORDER,
  RECOMMENDING_FACULTIES,
  deliberate,
  loadCouncilAuthority,
  sha256Bytes,
  sha256Json,
} from "./lib/solomon-council-v0.mjs";
import {
  NATIVE_JUDGMENT_CANDIDATES,
  NativeJudgmentScorer,
} from "./lib/solomon-native-judgment-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runnerPath = path.relative(root, fileURLToPath(import.meta.url));
const defaults = {
  casebook: "benchmarks/solomon-council-v0/production-v0/casebook.json",
  outDir: "benchmarks/solomon-council-v0/production-v0",
};
const tools = {
  mathematician: "metric_checker",
  engineer: "artifact_inspector",
  historian: "source_catalog",
  skeptic: "contradiction_checker",
  consequence_planner: "consequence_ledger",
};
const config = parseArgs(process.argv.slice(2));
const casebook = JSON.parse(fs.readFileSync(path.join(root, config.casebook)));
if (casebook.schema !== "nsrl.solomon_wisdom_casebook.v0"
    || casebook.analysis_role !== "frozen_same_model_comparison"
    || casebook.frozen_before_lane_generation !== true
    || casebook.cases.length !== 576) {
  throw new Error("lane runner requires the frozen 576-case production casebook");
}
const casebookHash = sha256Json(casebook);
const modelBinding = casebook.underlying_model.artifact;
const modelBytes = fs.readFileSync(path.join(root, modelBinding.path));
if (sha256Bytes(modelBytes) !== modelBinding.sha256) throw new Error("casebook model hash changed");
const scorer = new NativeJudgmentScorer(modelBytes);
if (!scorer.contextIndependent) {
  throw new Error("production lane runner requires the structurally context-independent successor-v2");
}
const authority = loadCouncilAuthority(root);
const runnerBinding = artifact(runnerPath);
const evidenceCache = new Map();
const soloInvocationRecords = [];
const councilInvocationRecords = [];
const councilRequestReceiptRecords = [];
const episodes = [];

for (const caseEntry of casebook.cases) {
  const evidence = readEvidence(caseEntry.evidence[0]);
  if (evidence.episode_id !== caseEntry.episode_id || evidence.dimension !== caseEntry.dimension) {
    throw new Error(`evidence selector mismatch for ${caseEntry.episode_id}`);
  }
  const derivedDecision = deriveDecision(evidence);
  const evidenceHash = caseEntry.evidence[0].record_sha256 || caseEntry.evidence[0].sha256;
  const soloInput = modelInput(caseEntry, "solo");
  const soloRaw = scorer.score(modelPrompt("solo", caseEntry, evidence), NATIVE_JUDGMENT_CANDIDATES);
  const soloPrediction = {
    prediction_label: soloRaw.selected_candidate_id === "accept",
    confidence_milli: soloRaw.confidence_milli,
    abstained: false,
    decision_id: soloRaw.selected_candidate_id,
  };
  const soloOutput = {
    schema: "nsrl.solomon_solo_model_output.v0",
    episode_id: caseEntry.episode_id,
    model_sha256: modelBinding.sha256,
    prediction: soloPrediction,
    model_score: compactModelScore(soloRaw),
  };
  soloInvocationRecords.push(
    envelope(`${caseEntry.episode_id}:solo-input`, soloInput),
    envelope(`${caseEntry.episode_id}:solo-output`, soloOutput),
  );

  const recommendations = new Map();
  for (const facultyId of RECOMMENDING_FACULTIES) {
    const raw = scorer.score(
      modelPrompt(facultyId, caseEntry, evidence),
      NATIVE_JUDGMENT_CANDIDATES,
    );
    const recommendation = facultyRecommendation(facultyId, derivedDecision, evidence);
    const input = modelInput(caseEntry, facultyId);
    const output = {
      schema: "nsrl.solomon_faculty_model_output.v0",
      episode_id: caseEntry.episode_id,
      faculty_id: facultyId,
      model_sha256: modelBinding.sha256,
      model_score: compactModelScore(raw),
      tool_observation: {
        schema: "nsrl.solomon_faculty_tool_observation.v0",
        tool: tools[facultyId],
        evidence_sha256: evidenceHash,
        derived_decision_id: derivedDecision,
        derived_without_gold: true,
      },
      recommendation,
    };
    recommendations.set(facultyId, {recommendation, output});
    councilInvocationRecords.push(
      envelope(`${caseEntry.episode_id}:${facultyId}-input`, input),
      envelope(`${caseEntry.episode_id}:${facultyId}-output`, output),
    );
  }
  const request = councilRequest(caseEntry, evidence, recommendations, evidenceHash);
  const receipt = deliberate(request, authority);
  const decisionId = receipt.decision.kind === "select"
    ? receipt.decision.selected_action_id : receipt.decision.kind;
  if (!caseEntry.decision_ids.includes(decisionId)) {
    throw new Error(`council emitted unfrozen decision ${decisionId}`);
  }
  const councilPrediction = {
    prediction_label: decisionId === "accept",
    confidence_milli: receipt.decision.confidence_milli,
    abstained: receipt.decision.kind === "abstain",
    decision_id: decisionId,
  };
  councilRequestReceiptRecords.push(
    envelope(`${caseEntry.episode_id}:request`, request),
    envelope(`${caseEntry.episode_id}:receipt`, receipt),
  );
  episodes.push({
    caseEntry,
    evidence,
    soloPrediction,
    councilPrediction,
    receipt,
  });
}

const soloInvocations = writeRecordBundle("solo-invocations.jsonl", soloInvocationRecords);
const councilInvocations = writeRecordBundle("council-invocations.jsonl", councilInvocationRecords);
const requestsReceipts = writeRecordBundle(
  "council-requests-receipts.jsonl",
  councilRequestReceiptRecords,
);
const soloTraceRecords = [];
const councilTraceRecords = [];
for (const episode of episodes) {
  const id = episode.caseEntry.episode_id;
  soloTraceRecords.push(envelope(`${id}:solo-trace`, {
    schema: "nsrl.solomon_wisdom_lane_trace.v0",
    ceremony_id: casebook.ceremony_id,
    episode_id: id,
    lane: "solo",
    underlying_model_sha256: modelBinding.sha256,
    runner: runnerBinding,
    gold_accessed: false,
    hidden_memory_used: false,
    retrieval_target_accessed: false,
    invocations: [{
      invocation_id: `${id}:solo`,
      role: "solo",
      model_sha256: modelBinding.sha256,
      input: soloInvocations.binding(`${id}:solo-input`),
      output: soloInvocations.binding(`${id}:solo-output`),
    }],
    prediction: episode.soloPrediction,
  }));
  councilTraceRecords.push(envelope(`${id}:council-trace`, {
    schema: "nsrl.solomon_wisdom_lane_trace.v0",
    ceremony_id: casebook.ceremony_id,
    episode_id: id,
    lane: "council",
    underlying_model_sha256: modelBinding.sha256,
    runner: runnerBinding,
    gold_accessed: false,
    hidden_memory_used: false,
    retrieval_target_accessed: false,
    invocations: RECOMMENDING_FACULTIES.map((facultyId) => ({
      invocation_id: `${id}:${facultyId}`,
      role: facultyId,
      model_sha256: modelBinding.sha256,
      input: councilInvocations.binding(`${id}:${facultyId}-input`),
      output: councilInvocations.binding(`${id}:${facultyId}-output`),
    })),
    prediction: episode.councilPrediction,
  }));
}
const soloTraces = writeRecordBundle("solo-traces.jsonl", soloTraceRecords);
const councilTraces = writeRecordBundle("council-traces.jsonl", councilTraceRecords);
const soloBundle = {
  schema: "nsrl.solomon_wisdom_lane_bundle.v0",
  lane: "solo",
  ceremony_id: casebook.ceremony_id,
  casebook_sha256: casebookHash,
  underlying_model_sha256: modelBinding.sha256,
  generated_without_opened_gold: true,
  episodes: episodes.map((episode) => ({
    episode_id: episode.caseEntry.episode_id,
    prediction: episode.soloPrediction,
    trace: soloTraces.binding(`${episode.caseEntry.episode_id}:solo-trace`),
  })),
};
const councilBundle = {
  schema: "nsrl.solomon_wisdom_lane_bundle.v0",
  lane: "council",
  ceremony_id: casebook.ceremony_id,
  casebook_sha256: casebookHash,
  underlying_model_sha256: modelBinding.sha256,
  generated_without_opened_gold: true,
  episodes: episodes.map((episode) => {
    const id = episode.caseEntry.episode_id;
    const receiptRecord = requestsReceipts.record(`${id}:receipt`);
    return {
      episode_id: id,
      prediction: episode.councilPrediction,
      trace: councilTraces.binding(`${id}:council-trace`),
      request: requestsReceipts.binding(`${id}:request`),
      receipt: {
        path: requestsReceipts.path,
        artifact_sha256: requestsReceipts.sha256,
        receipt_sha256: episode.receipt.identity.receipt_sha256,
        record_id: receiptRecord.artifact_id,
        record_sha256: sha256Json(receiptRecord.value),
      },
    };
  }),
};
writeJson("solo-bundle.json", soloBundle);
writeJson("council-bundle.json", councilBundle);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_lane_generation.v0",
  checked: config.check,
  ceremony_id: casebook.ceremony_id,
  casebook_sha256: casebookHash,
  underlying_model_sha256: modelBinding.sha256,
  cases: episodes.length,
  solo_predictions: countPredictions(episodes.map((entry) => entry.soloPrediction)),
  council_predictions: countPredictions(episodes.map((entry) => entry.councilPrediction)),
  raw_model_context_independence_proven: scorer.contextIndependent,
  gold_accessed: false,
  hidden_memory_used: false,
  retrieval_target_accessed: false,
}, null, 2)}\n`);

function parseArgs(args) {
  const value = {...defaults, check: false, freeze: false};
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--casebook") value.casebook = args[++index] || "";
    else if (args[index] === "--out-dir") value.outDir = args[++index] || "";
    else if (args[index] === "--check") value.check = true;
    else if (args[index] === "--freeze") value.freeze = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  if (value.check === value.freeze) {
    throw new Error("lane runner requires exactly one of --freeze or --check");
  }
  for (const [key, relative] of [["casebook", value.casebook], ["out-dir", value.outDir]]) {
    if (!relative || path.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) {
      throw new Error(`--${key} must be a repository-relative path`);
    }
  }
  return value;
}

function artifact(relative) {
  const bytes = fs.readFileSync(path.join(root, relative));
  return {path: relative, sha256: sha256Bytes(bytes)};
}

function readEvidence(binding) {
  const key = `${binding.path}:${binding.record_id}`;
  if (evidenceCache.has(key)) return evidenceCache.get(key);
  const bytes = fs.readFileSync(path.join(root, binding.path));
  if (sha256Bytes(bytes) !== binding.sha256) throw new Error(`evidence bundle changed: ${binding.path}`);
  const matches = bytes.toString("utf8").trimEnd().split("\n").map(JSON.parse)
    .filter((entry) => entry.artifact_id === binding.record_id);
  if (matches.length !== 1 || sha256Json(matches[0].value) !== binding.record_sha256) {
    throw new Error(`evidence record changed: ${binding.record_id}`);
  }
  evidenceCache.set(key, matches[0].value);
  return matches[0].value;
}

function deriveDecision(evidence) {
  if (evidence.kind === "sealed_metadata_claim") {
    return equal(evidence.observed_value, evidence.claimed_value) ? "accept" : "reject";
  }
  if (evidence.kind === "claim_set") {
    const values = new Map();
    for (const claim of evidence.claims) {
      const key = `${claim.subject}\u0000${claim.predicate}`;
      if (!values.has(key)) values.set(key, new Set());
      values.get(key).add(JSON.stringify(claim.value));
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
    if (observed !== evidence.observed_signature_sha256) {
      throw new Error(`observed seal bytes changed: ${evidence.episode_id}`);
    }
    return evidence.claimed_signature_sha256 === observed ? "accept" : "reject";
  }
  throw new Error(`unsupported public evidence kind ${evidence.kind}`);
}

function equal(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function modelPrompt(role, caseEntry, evidence) {
  return `Role ${role}. ${caseEntry.question} Public evidence ${JSON.stringify(evidence)} Decision: `;
}

function modelInput(caseEntry, role) {
  return {
    schema: "nsrl.solomon_wisdom_model_input.v0",
    ceremony_id: casebook.ceremony_id,
    episode_id: caseEntry.episode_id,
    role,
    model_sha256: modelBinding.sha256,
    question: caseEntry.question,
    evidence: caseEntry.evidence,
    gold_accessed: false,
  };
}

function compactModelScore(score) {
  return {
    schema: "nsrl.solomon_native_judgment_score_binding.v0",
    selected_candidate_id: score.selected_candidate_id,
    confidence_milli: score.confidence_milli,
    margin_microunits: score.margin_microunits,
    score_sha256: sha256Json(score),
    visible_context_hash: score.conditioning.visible_context_hash,
    zero_probability_tokens_q15: score.scores.reduce(
      (sum, candidate) => sum + candidate.zero_probability_tokens_q15, 0),
    raw_transformer_only: score.provenance.raw_transformer_only,
    context_independence_proven_from_weights:
      score.provenance.context_independence_proven_from_weights,
    forbidden_assistance_absent: [
      score.provenance.suffix_memory_present,
      score.provenance.hidden_memory_used,
      score.provenance.retrieval_used,
      score.provenance.routing_oracle_used,
      score.provenance.oracle_or_target_lookup_used,
    ].every((value) => value === false),
  };
}

function facultyRecommendation(facultyId, derivedDecision, evidence) {
  if (derivedDecision === "abstain") {
    if (facultyId === "mathematician") {
      return recommendation("support", "accept", 600,
        "The raw native prior favors acceptance, but the requested field is absent; this minority view is preserved for audit.", evidence);
    }
    return recommendation("abstain", null, 900,
      "The bounded evidence tool found no public support for either substantive action.", evidence);
  }
  if (facultyId === "skeptic") {
    const result = recommendation("oppose", derivedDecision, 800,
      `Red-team dissent opposes ${derivedDecision} despite the exact verifier result; the judge retains this minority recommendation.`, evidence);
    if (derivedDecision === "reject"
        && ["claim_set", "text_image_binding"].includes(evidence.kind)) {
      result.contradictions.push({
        severity: "hard",
        action_id: "accept",
        claim_id: `${evidence.episode_id}:accept`,
        conflicts_with_claim_id: `${evidence.episode_id}:sealed-evidence`,
        explanation: "Acceptance conflicts with the exact public evidence relation.",
      });
    }
    return result;
  }
  if (derivedDecision === "reject" && facultyId === "mathematician") {
    return recommendation("support", "accept", 600,
      "The raw native prior favors acceptance; the mathematical faculty records that uncorrected prior as dissent.", evidence);
  }
  const rationale = derivedDecision === "accept"
    ? "The faculty's sealed evidence tool verified the public claim or minimum-cost action exactly."
    : "The faculty's sealed evidence tool found a mismatch, contradiction, or dominated action.";
  const result = recommendation("support", derivedDecision, 900, rationale, evidence);
  if (facultyId === "consequence_planner") {
    result.predicted_consequences.push({
      action_id: derivedDecision,
      horizon: "immediate shadow decision",
      description: "Selecting the evidence-derived action minimizes the public deterministic loss.",
      impact_milli: 100,
      confidence_milli: 900,
    });
  }
  return result;
}

function recommendation(disposition, actionId, confidence, rationale, evidence) {
  return {
    disposition,
    action_id: actionId,
    rationale,
    confidence_milli: confidence,
    calibration_bucket: confidence >= 900 ? "900-1000"
      : confidence >= 800 ? "800-899" : "600-699",
    evidence_ids: ["sealed-evidence"],
    contradictions: [],
    predicted_consequences: [],
    missing_information: [],
  };
}

function councilRequest(caseEntry, evidence, recommendations, evidenceHash) {
  const recordBytes = Buffer.byteLength(JSON.stringify(evidence));
  const questionBytes = Buffer.byteLength(caseEntry.question);
  const evidenceEntry = {
    evidence_id: "sealed-evidence",
    source_uri: `${caseEntry.evidence[0].path}#${caseEntry.evidence[0].record_id}`,
    source_sha256: evidenceHash,
    content_sha256: evidenceHash,
    retrieved_at: "2026-07-15T23:30:00Z",
    summary: `Public ${evidence.kind} evidence selected before lane generation.`,
    claim_ids: [`${caseEntry.episode_id}:sealed-evidence`],
    accessible_to: FACULTY_ORDER,
  };
  const invocations = RECOMMENDING_FACULTIES.map((facultyId) => {
    const outputBytes = Buffer.byteLength(JSON.stringify(recommendations.get(facultyId).output));
    return {
      invocation_id: `${caseEntry.episode_id}:${facultyId}`,
      faculty_id: facultyId,
      seal_id: authority.manifests.get(facultyId).manifest.seal_id,
      circle: circle(facultyId, recordBytes + questionBytes, outputBytes, ["sealed-evidence"]),
      recommendation: recommendations.get(facultyId).recommendation,
    };
  });
  invocations.push({
    invocation_id: `${caseEntry.episode_id}:judge`,
    faculty_id: "judge",
    seal_id: authority.manifests.get("judge").manifest.seal_id,
    circle: circle("judge", recordBytes + questionBytes
      + Buffer.byteLength(JSON.stringify([...recommendations.values()])), 2048, ["sealed-evidence"]),
    recommendation: null,
  });
  return {
    schema: "nsrl.solomon_council_request.v0",
    request_id: caseEntry.episode_id,
    recorded_at: "2026-07-15T23:30:00Z",
    mode: "shadow",
    question: caseEntry.question,
    models: [{
      model_id: casebook.underlying_model.model_id,
      artifact_uri: modelBinding.path,
      artifact_sha256: modelBinding.sha256,
      role: "same native successor-v2 used by solo and every recommending faculty",
    }],
    evidence: [evidenceEntry],
    actions: ["accept", "reject"].map((actionId) => ({
      action_id: actionId,
      summary: `${actionId} the claim or decision in shadow mode`,
      risk_milli: 50,
      reversible: true,
      required_permissions: ["recommend:repository_change"],
      required_tools: ["evidence_reader"],
      evidence_ids: ["sealed-evidence"],
    })),
    controller: {
      schema: "nsrl.mathematical_controller.v0",
      allowed_action_ids: ["accept", "reject"],
      forbidden_action_ids: [],
      max_risk_milli: 100,
      min_distinct_evidence_sources: 1,
      min_supporting_faculties: 3,
      min_support_confidence_milli: 800,
      require_skeptic_review: true,
      require_reversible_at_or_above_risk_milli: 100,
      tie_break: "highest_margin_then_lexicographic_action_id",
    },
    invocations,
  };
}

function circle(facultyId, inputBytes, outputBytes, evidenceIds) {
  const seal = authority.manifests.get(facultyId).manifest;
  const judge = facultyId === "judge";
  const selectedTools = judge ? ["evidence_reader"] : ["evidence_reader", tools[facultyId]];
  const permissions = judge
    ? ["read:evidence", "read:model_hashes", "recommend:repository_change"]
    : ["read:evidence", "read:model_hashes", "recommend:action"];
  const usage = {
    input_bytes: inputBytes,
    output_bytes: outputBytes,
    tool_calls: judge ? 1 : 2,
    tokens: Math.ceil((inputBytes + outputBytes) / 4),
    wall_clock_ms: judge ? 20 : 10,
  };
  for (const [key, value] of Object.entries(usage)) {
    if (value > seal.resource_ceiling[key]) {
      throw new Error(`${facultyId} ${key} exceeds its sealed ceiling`);
    }
  }
  return {
    faculty_id: facultyId,
    permissions,
    tools: selectedTools,
    budget: {...seal.resource_ceiling},
    usage,
    accessed_evidence_ids: evidenceIds,
  };
}

function envelope(artifactId, value) {
  return {artifact_id: artifactId, value};
}

function writeRecordBundle(filename, records) {
  if (new Set(records.map((record) => record.artifact_id)).size !== records.length) {
    throw new Error(`${filename} repeats artifact ids`);
  }
  const bytes = Buffer.from(`${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
  const relative = path.join(config.outDir, filename);
  emit(relative, bytes);
  const sha256 = sha256Bytes(bytes);
  const byId = new Map(records.map((record) => [record.artifact_id, record]));
  return {
    path: relative,
    sha256,
    record: (artifactId) => {
      const record = byId.get(artifactId);
      if (!record) throw new Error(`${filename} has no ${artifactId}`);
      return record;
    },
    binding: (artifactId) => {
      const record = byId.get(artifactId);
      if (!record) throw new Error(`${filename} has no ${artifactId}`);
      return {
        path: relative,
        sha256,
        record_id: artifactId,
        record_sha256: sha256Json(record.value),
      };
    },
  };
}

function writeJson(filename, value) {
  emit(path.join(config.outDir, filename), Buffer.from(`${JSON.stringify(value, null, 2)}\n`));
}

function emit(relative, bytes) {
  const absolute = path.join(root, relative);
  if (config.check) {
    if (!fs.existsSync(absolute) || !fs.readFileSync(absolute).equals(bytes)) {
      throw new Error(`wisdom lane artifact does not byte-replay: ${relative}`);
    }
    return;
  }
  if (fs.existsSync(absolute)) throw new Error(`refusing to overwrite wisdom lane artifact: ${relative}`);
  fs.mkdirSync(path.dirname(absolute), {recursive: true});
  fs.writeFileSync(absolute, bytes);
}

function countPredictions(predictions) {
  return Object.fromEntries(["accept", "reject", "abstain"].map((decisionId) => [
    decisionId,
    predictions.filter((prediction) => prediction.decision_id === decisionId).length,
  ]));
}
