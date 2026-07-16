#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {sha256Bytes} from "./lib/solomon-council-v0.mjs";
import {
  evaluateWisdom,
  WISDOM_DIMENSIONS,
} from "./lib/solomon-wisdom-eval-v0.mjs";

const modelHash = "1".repeat(64);
const evidenceHash = "2".repeat(64);
const soloTraceHash = "3".repeat(64);
const councilTraceHash = "4".repeat(64);
const receiptHash = "5".repeat(64);
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const expectFailure = (operation, pattern) => {
  try {
    operation();
  } catch (error) {
    assert(pattern.test(String(error.message)), `wrong wisdom-eval failure: ${error.message}`);
    return;
  }
  throw new Error(`expected wisdom-eval failure matching ${pattern}`);
};
const lane = (prediction, confidence, abstained, decision, council = false) => ({
  model_sha256: modelHash,
  trace_sha256: council ? councilTraceHash : soloTraceHash,
  prediction_label: prediction,
  confidence_milli: confidence,
  abstained,
  decision_id: decision,
  ...(council ? {receipt_sha256: receiptHash} : {}),
});
const episode = (dimension, index) => {
  const negative = dimension === "hard_negative_rejection";
  const abstention = dimension === "appropriate_abstention";
  const expected = !negative;
  return {
    schema: "nsrl.solomon_wisdom_episode.v0",
    episode_id: `self-test-${index}-${dimension}`,
    dimension,
    source_family: dimension === "unfamiliar_source_transfer" ? "unseen_fixture" : "fixture",
    unfamiliar_source: dimension === "unfamiliar_source_transfer",
    evidence_sha256: [evidenceHash],
    gold_opened_after_both_predictions: true,
    gold: {
      expected_label: expected,
      should_abstain: abstention,
      decision_costs_milli: {bad: 900, good: 100},
    },
    solo: lane(!expected, 900, !abstention, "bad"),
    council: lane(expected, 900, abstention, "good", true),
  };
};
const input = {
  schema: "nsrl.solomon_wisdom_eval.v0",
  analysis_role: "self_test_only",
  frozen_before_outcomes: true,
  minimum_cases_per_dimension: 1,
  underlying_model: {model_id: "fixture-model", artifact_sha256: modelHash},
  integrity: {
    no_oracle_target_lookup: true,
    no_hidden_memory: true,
    no_retrieval_target_leakage: true,
    generation_integrity_passed: true,
    generation_integrity_report: {
      path: "self-test-generation-integrity.json",
      sha256: "8".repeat(64),
      schema: "nsrl.wisdom_generation_integrity.v0",
    },
    provenance_report: {
      path: "self-test-provenance.json",
      sha256: "9".repeat(64),
      schema: "nsrl.wisdom_provenance_gate.v0",
    },
  },
  episodes: WISDOM_DIMENSIONS.map(episode),
};

const result = evaluateWisdom(input, {evaluatorSha256: "6".repeat(64)});
assert(result.verdict.all_dimensions_outperform === true,
  "self-test council did not outperform on every dimension");
assert(result.verdict.promotion_gate_passed === false
  && result.authorization.council_promotion_authorized === false
  && result.verdict.self_test_only === true,
"self-test data improperly authorized promotion");

const wrongModel = structuredClone(input);
wrongModel.episodes[0].council.model_sha256 = "7".repeat(64);
expectFailure(() => evaluateWisdom(wrongModel), /underlying model/);
const oracle = structuredClone(input);
oracle.integrity.no_oracle_target_lookup = false;
expectFailure(() => evaluateWisdom(oracle), /integrity gate failed/);
const earlyGold = structuredClone(input);
earlyGold.episodes[0].gold_opened_after_both_predictions = false;
expectFailure(() => evaluateWisdom(earlyGold), /Gold|gold/);
const missingDimension = structuredClone(input);
missingDimension.episodes = missingDimension.episodes.slice(1);
expectFailure(() => evaluateWisdom(missingDimension), /has 0 cases/);
const undersizedProduction = structuredClone(input);
undersizedProduction.analysis_role = "frozen_same_model_comparison";
expectFailure(() => evaluateWisdom(undersizedProduction), /at least 72 cases/);
const missingIntegrityArtifact = structuredClone(input);
missingIntegrityArtifact.analysis_role = "frozen_same_model_comparison";
missingIntegrityArtifact.minimum_cases_per_dimension = 72;
expectFailure(() => evaluateWisdom(missingIntegrityArtifact), /report is missing/);

const temp = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-wisdom-eval-"));
try {
  const writeReport = (name, value) => {
    const reportPath = path.join(temp, name);
    fs.writeFileSync(reportPath, `${JSON.stringify(value, null, 2)}\n`);
    return {path: reportPath, sha256: sha256Bytes(fs.readFileSync(reportPath))};
  };
  const sourceReport = writeReport("source-report.json", {schema: "self-test", ok: true});
  const generationReport = writeReport("generation-integrity.json", {
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
  const provenanceReport = writeReport("provenance.json", {
    schema: "nsrl.wisdom_provenance_gate.v0",
    ok: true,
    model_artifact_sha256: modelHash,
    source_hashes: [evidenceHash],
    trace_hashes: [soloTraceHash, councilTraceHash],
    gates: {
      no_oracle_target_lookup: true,
      no_hidden_memory: true,
      no_retrieval_target_leakage: true,
      gold_sealed_until_both_predictions: true,
    },
  });
  const noCeremony = structuredClone(input);
  noCeremony.analysis_role = "frozen_same_model_comparison";
  noCeremony.minimum_cases_per_dimension = 72;
  noCeremony.episodes = WISDOM_DIMENSIONS.flatMap((dimension, dimensionIndex) =>
    Array.from({length: 72}, (_, index) => ({
      ...episode(dimension, dimensionIndex * 72 + index),
    })));
  noCeremony.integrity.generation_integrity_report = {
    ...generationReport, schema: "nsrl.wisdom_generation_integrity.v0",
  };
  noCeremony.integrity.provenance_report = {
    ...provenanceReport, schema: "nsrl.wisdom_provenance_gate.v0",
  };
  expectFailure(() => evaluateWisdom(noCeremony), /byte-bound ceremony/);
} finally {
  fs.rmSync(temp, {recursive: true, force: true});
}

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_eval_self_check.v0",
  dimensions: WISDOM_DIMENSIONS,
  all_dimension_scoring_exercised: true,
  same_model_binding_enforced: true,
  post_prediction_gold_enforced: true,
  oracle_and_hidden_path_gates_enforced: true,
  missing_dimension_rejected: true,
  undersized_production_eval_rejected: true,
  missing_integrity_artifact_rejected: true,
  production_ceremony_required: true,
  self_test_cannot_authorize_promotion: true,
}, null, 2)}\n`);
