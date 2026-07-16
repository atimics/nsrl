#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {
  RECOMMENDING_FACULTIES,
  deliberate,
  sha256Bytes,
  sha256Json,
} from "./lib/solomon-council-v0.mjs";
import {
  compileWisdomCeremony,
  freezeWisdomCasebook,
  goldCommitment,
  openWisdomGold,
} from "./lib/solomon-wisdom-ceremony-v0.mjs";
import {evaluateWisdom, WISDOM_DIMENSIONS} from "./lib/solomon-wisdom-eval-v0.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-wisdom-ceremony-"));
const fixtureRequest = JSON.parse(fs.readFileSync(
  path.join(repoRoot, "benchmarks/solomon-council-v0/fixtures/select-request.json"), "utf8"));
const modelSource = path.join(repoRoot, fixtureRequest.models[0].artifact_uri);
const modelPath = path.join(temp, "model.nsrlmt");
fs.copyFileSync(modelSource, modelPath);
const modelHash = sha256Bytes(fs.readFileSync(modelPath));
const runnerPath = path.join(repoRoot, "scripts/run-solomon-council-v0.mjs");

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const expectFailure = (operation, pattern) => {
  try {
    operation();
  } catch (error) {
    assert(pattern.test(String(error.message)), `wrong ceremony failure: ${error.message}`);
    return;
  }
  throw new Error(`expected ceremony failure matching ${pattern}`);
};
const writeJson = (name, value) => {
  const artifactPath = path.join(temp, name);
  fs.writeFileSync(artifactPath, `${JSON.stringify(value, null, 2)}\n`);
  return {path: artifactPath, sha256: sha256Bytes(fs.readFileSync(artifactPath))};
};
const bindFile = (artifactPath) => ({
  path: artifactPath,
  sha256: sha256Bytes(fs.readFileSync(artifactPath)),
});
const clone = (value) => structuredClone(value);
const abstainRequest = (request) => {
  for (const invocation of request.invocations) {
    if (invocation.faculty_id === "judge") continue;
    invocation.recommendation = {
      disposition: "abstain",
      action_id: null,
      rationale: "The self-check deliberately exercises the sealed abstention route.",
      confidence_milli: 700,
      calibration_bucket: "700-799",
      evidence_ids: [],
      contradictions: [],
      predicted_consequences: [],
      missing_information: [],
    };
  }
};

try {
  assert(modelHash === fixtureRequest.models[0].artifact_sha256,
    "self-check model fixture hash changed");
  const evidence = fixtureRequest.evidence.map((source, index) => {
    const sourcePath = path.join(repoRoot, source.source_uri);
    const copiedPath = path.join(temp, `evidence-${index}.json`);
    fs.copyFileSync(sourcePath, copiedPath);
    const binding = bindFile(copiedPath);
    assert(binding.sha256 === source.content_sha256, "self-check evidence fixture hash changed");
    return binding;
  });
  const cases = [];
  const goldOpenings = [];
  const soloEpisodes = [];
  const councilEpisodes = [];
  const traces = [];
  for (const [index, dimension] of WISDOM_DIMENSIONS.entries()) {
    const episodeId = `ceremony-self-test-${String(index + 1).padStart(2, "0")}-${dimension}`;
    const expected = dimension !== "hard_negative_rejection";
    const shouldAbstain = dimension === "appropriate_abstention";
    const request = clone(fixtureRequest);
    request.request_id = episodeId;
    request.recorded_at = `2026-07-15T22:${String(index).padStart(2, "0")}:00Z`;
    request.models[0].artifact_uri = modelPath;
    for (const invocation of request.invocations) {
      invocation.invocation_id = `${episodeId}:${invocation.faculty_id}`;
    }
    if (shouldAbstain) abstainRequest(request);
    const receipt = deliberate(request);
    const councilDecisionId = receipt.decision.kind === "select"
      ? receipt.decision.selected_action_id : receipt.decision.kind;
    const decisionIds = ["solo-bad", councilDecisionId].filter(
      (value, position, values) => values.indexOf(value) === position);
    const gold = {
      expected_label: expected,
      should_abstain: shouldAbstain,
      decision_costs_milli: Object.fromEntries(decisionIds.map(
        (decisionId) => [decisionId, decisionId === councilDecisionId ? 100 : 900])),
    };
    const nonce = sha256Json({self_test_nonce: index + 1});
    cases.push({
      episode_id: episodeId,
      dimension,
      source_family: dimension === "unfamiliar_source_transfer" ? "unseen-self-test" : "self-test",
      unfamiliar_source: dimension === "unfamiliar_source_transfer",
      evidence,
      decision_ids: decisionIds,
      gold_commitment_sha256: goldCommitment(episodeId, nonce, gold),
    });
    goldOpenings.push({episode_id: episodeId, nonce, gold});

    const soloPrediction = {
      prediction_label: !expected,
      confidence_milli: 900,
      abstained: !shouldAbstain,
      decision_id: "solo-bad",
    };
    const soloInput = writeJson(`${index}-solo-input.json`, {
      schema: "nsrl.solomon_solo_model_input.v0", episode_id: episodeId,
      evidence_sha256: evidence.map((binding) => binding.sha256),
    });
    const soloOutput = writeJson(`${index}-solo-output.json`, {
      schema: "nsrl.solomon_solo_model_output.v0",
      episode_id: episodeId,
      model_sha256: modelHash,
      prediction: soloPrediction,
    });
    const soloTrace = writeJson(`${index}-solo-trace.json`, {
      schema: "nsrl.solomon_wisdom_lane_trace.v0",
      ceremony_id: "solomon-wisdom-ceremony-self-test-v0",
      episode_id: episodeId,
      lane: "solo",
      underlying_model_sha256: modelHash,
      runner: bindFile(runnerPath),
      gold_accessed: false,
      hidden_memory_used: false,
      retrieval_target_accessed: false,
      invocations: [{
        invocation_id: `${episodeId}:solo`, role: "solo", model_sha256: modelHash,
        input: soloInput, output: soloOutput,
      }],
      prediction: soloPrediction,
    });
    traces.push(soloTrace.sha256);
    soloEpisodes.push({episode_id: episodeId, prediction: soloPrediction, trace: soloTrace});

    const requestBinding = writeJson(`${index}-council-request.json`, request);
    const receiptBinding = writeJson(`${index}-council-receipt.json`, receipt);
    const invocations = [];
    for (const facultyId of RECOMMENDING_FACULTIES) {
      const facultyInvocation = request.invocations.find(
        (invocation) => invocation.faculty_id === facultyId);
      const output = writeJson(`${index}-${facultyId}-output.json`, {
        schema: "nsrl.solomon_faculty_model_output.v0",
        episode_id: episodeId,
        faculty_id: facultyId,
        model_sha256: modelHash,
        recommendation: facultyInvocation.recommendation,
      });
      invocations.push({
        invocation_id: `${episodeId}:${facultyId}`,
        role: facultyId,
        model_sha256: modelHash,
        input: requestBinding,
        output,
      });
    }
    const councilPrediction = {
      prediction_label: expected,
      confidence_milli: receipt.decision.confidence_milli,
      abstained: receipt.decision.kind === "abstain",
      decision_id: councilDecisionId,
    };
    const councilTrace = writeJson(`${index}-council-trace.json`, {
      schema: "nsrl.solomon_wisdom_lane_trace.v0",
      ceremony_id: "solomon-wisdom-ceremony-self-test-v0",
      episode_id: episodeId,
      lane: "council",
      underlying_model_sha256: modelHash,
      runner: bindFile(runnerPath),
      gold_accessed: false,
      hidden_memory_used: false,
      retrieval_target_accessed: false,
      invocations,
      prediction: councilPrediction,
    });
    traces.push(councilTrace.sha256);
    councilEpisodes.push({
      episode_id: episodeId,
      prediction: councilPrediction,
      trace: councilTrace,
      request: requestBinding,
      receipt: {
        path: receiptBinding.path,
        artifact_sha256: receiptBinding.sha256,
        receipt_sha256: receipt.identity.receipt_sha256,
      },
    });
  }

  const casebook = {
    schema: "nsrl.solomon_wisdom_casebook.v0",
    analysis_role: "self_test_only",
    ceremony_id: "solomon-wisdom-ceremony-self-test-v0",
    frozen_before_lane_generation: true,
    minimum_cases_per_dimension: 1,
    underlying_model: {
      model_id: fixtureRequest.models[0].model_id,
      artifact: bindFile(modelPath),
    },
    integrity_policy: {
      no_oracle_target_lookup: true,
      no_hidden_memory: true,
      no_retrieval_target_leakage: true,
      gold_commitment_algorithm: "sha256-canonical-json-v0",
    },
    cases,
  };
  const casebookHash = sha256Json(casebook);
  const soloBundle = {
    schema: "nsrl.solomon_wisdom_lane_bundle.v0",
    lane: "solo",
    ceremony_id: casebook.ceremony_id,
    casebook_sha256: casebookHash,
    underlying_model_sha256: modelHash,
    generated_without_opened_gold: true,
    episodes: soloEpisodes,
  };
  const councilBundle = {
    schema: "nsrl.solomon_wisdom_lane_bundle.v0",
    lane: "council",
    ceremony_id: casebook.ceremony_id,
    casebook_sha256: casebookHash,
    underlying_model_sha256: modelHash,
    generated_without_opened_gold: true,
    episodes: councilEpisodes,
  };
  const opening = {
    schema: "nsrl.solomon_wisdom_gold_opening.v0",
    ceremony_id: casebook.ceremony_id,
    casebook_sha256: casebookHash,
    solo_bundle_sha256: sha256Json(soloBundle),
    council_bundle_sha256: sha256Json(councilBundle),
    opened_after_both_lane_bundles: true,
    gold: goldOpenings,
  };
  const draft = {
    schema: "nsrl.solomon_wisdom_casebook_draft.v0",
    analysis_role: casebook.analysis_role,
    ceremony_id: casebook.ceremony_id,
    minimum_cases_per_dimension: casebook.minimum_cases_per_dimension,
    underlying_model: casebook.underlying_model,
    integrity_policy: casebook.integrity_policy,
    cases: cases.map((entry) => {
      const secret = goldOpenings.find((candidate) => candidate.episode_id === entry.episode_id);
      return {
        episode_id: entry.episode_id,
        dimension: entry.dimension,
        source_family: entry.source_family,
        unfamiliar_source: entry.unfamiliar_source,
        evidence: entry.evidence,
        decision_ids: entry.decision_ids,
        nonce: secret.nonce,
        gold: secret.gold,
      };
    }),
  };
  const frozen = freezeWisdomCasebook(draft, {baseDir: repoRoot});
  assert(sha256Json(frozen.casebook) === sha256Json(casebook),
    "casebook freezer did not reproduce the public casebook");
  const replayedOpening = openWisdomGold({
    casebook: frozen.casebook,
    soloBundle,
    councilBundle,
    vault: frozen.vault,
  }, {baseDir: repoRoot});
  assert(sha256Json(replayedOpening) === sha256Json(opening),
    "gold-opening command did not reproduce the bound opening");
  const sourceReport = writeJson("source-quality-report.json", {
    schema: "nsrl.solomon_wisdom_source_quality_self_test.v0", ok: true,
  });
  const generationReport = writeJson("generation-integrity.json", {
    schema: "nsrl.wisdom_generation_integrity.v0",
    ok: true,
    model_artifact_sha256: modelHash,
    source_report: sourceReport,
    gates: {
      quality_report_green: true,
      generation_integrity_green: true,
      source_grounding_green: true,
      cross_modal_agreement_green: true,
      same_model_invocation_green: true,
      trace_replay_green: true,
      faculty_output_binding_green: true,
    },
  });
  const provenanceReport = writeJson("provenance.json", {
    schema: "nsrl.wisdom_provenance_gate.v0",
    ok: true,
    model_artifact_sha256: modelHash,
    source_hashes: [...new Set(evidence.map((binding) => binding.sha256))].sort(),
    trace_hashes: [...new Set(traces)].sort(),
    gates: {
      no_oracle_target_lookup: true,
      no_hidden_memory: true,
      no_retrieval_target_leakage: true,
      gold_sealed_until_both_predictions: true,
    },
  });
  const casebookArtifact = writeJson("casebook.json", casebook);
  const soloBundleArtifact = writeJson("solo-bundle.json", soloBundle);
  const councilBundleArtifact = writeJson("council-bundle.json", councilBundle);
  const openingArtifact = writeJson("gold-opening.json", opening);
  const ceremony = {
    casebook,
    soloBundle,
    councilBundle,
    opening,
    integrityBindings: {
      generation_integrity_report: generationReport,
      provenance_report: provenanceReport,
    },
    ceremonyBindings: {
      casebook: casebookArtifact,
      solo_bundle: soloBundleArtifact,
      council_bundle: councilBundleArtifact,
      gold_opening: openingArtifact,
    },
  };
  const compiled = compileWisdomCeremony(ceremony, {baseDir: repoRoot});
  const result = evaluateWisdom(compiled);
  assert(result.verdict.all_dimensions_outperform === true,
    "compiled self-check did not exercise all scoring dimensions");
  assert(result.verdict.promotion_gate_passed === false,
    "compiled self-check improperly authorized promotion");

  const tamperedGold = clone(ceremony);
  delete tamperedGold.ceremonyBindings;
  tamperedGold.opening.gold[0].gold.expected_label = !tamperedGold.opening.gold[0].gold.expected_label;
  expectFailure(() => compileWisdomCeremony(tamperedGold, {baseDir: repoRoot}), /commitment/);
  const earlyGold = clone(ceremony);
  delete earlyGold.ceremonyBindings;
  earlyGold.opening.opened_after_both_lane_bundles = false;
  expectFailure(() => compileWisdomCeremony(earlyGold, {baseDir: repoRoot}), /after both/);
  const wrongModel = clone(ceremony);
  delete wrongModel.ceremonyBindings;
  wrongModel.soloBundle.underlying_model_sha256 = "0".repeat(64);
  expectFailure(() => compileWisdomCeremony(wrongModel, {baseDir: repoRoot}), /model changed/);
  const tamperedTrace = clone(ceremony);
  delete tamperedTrace.ceremonyBindings;
  tamperedTrace.soloBundle.episodes[0].trace.sha256 = "0".repeat(64);
  expectFailure(() => compileWisdomCeremony(tamperedTrace, {baseDir: repoRoot}), /byte hash changed/);
  const wrongFacultyModel = clone(ceremony);
  delete wrongFacultyModel.ceremonyBindings;
  const councilTracePath = wrongFacultyModel.councilBundle.episodes[0].trace.path;
  const councilTrace = JSON.parse(fs.readFileSync(councilTracePath, "utf8"));
  councilTrace.invocations[0].model_sha256 = "0".repeat(64);
  const wrongTrace = writeJson("wrong-faculty-model-trace.json", councilTrace);
  wrongFacultyModel.councilBundle.episodes[0].trace = wrongTrace;
  wrongFacultyModel.councilBundle = {
    ...wrongFacultyModel.councilBundle,
  };
  wrongFacultyModel.opening.council_bundle_sha256 = sha256Json(wrongFacultyModel.councilBundle);
  expectFailure(() => compileWisdomCeremony(wrongFacultyModel, {baseDir: repoRoot}),
    /invocation used another model/);
  const provenanceMismatch = clone(ceremony);
  const badProvenance = clone(JSON.parse(fs.readFileSync(provenanceReport.path, "utf8")));
  badProvenance.trace_hashes = ["f".repeat(64)];
  provenanceMismatch.integrityBindings.provenance_report = writeJson(
    "bad-provenance.json", badProvenance);
  expectFailure(() => compileWisdomCeremony(provenanceMismatch, {baseDir: repoRoot}),
    /trace set differs/);

  process.stdout.write(`${JSON.stringify({
    schema: "nsrl.solomon_wisdom_ceremony_self_check.v0",
    dimensions: WISDOM_DIMENSIONS,
    byte_verified_model_evidence_traces_and_receipts: true,
    exact_same_model_invocations_enforced: true,
    five_faculty_roles_enforced: true,
    faculty_outputs_match_receipt: true,
    gold_commitments_enforced: true,
    gold_opening_binds_both_lane_bundles: true,
    provenance_sets_exact: true,
    casebook_freeze_and_gold_opening_replay: true,
    self_test_cannot_authorize_promotion: true,
  }, null, 2)}\n`);
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}
