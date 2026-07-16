import fs from "node:fs";
import path from "node:path";

import {
  sha256Bytes,
  sha256Json,
} from "./solomon-council-v0.mjs";
import {verifyWisdomCeremonyCompilation} from "./solomon-wisdom-ceremony-v0.mjs";
export {WISDOM_DIMENSIONS} from "./solomon-wisdom-constants-v0.mjs";
import {WISDOM_DIMENSIONS} from "./solomon-wisdom-constants-v0.mjs";

const shaPattern = /^[0-9a-f]{64}$/;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, required, label) {
  assert(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  for (const key of required) assert(Object.hasOwn(value, key), `${label} missing ${key}`);
  for (const key of Object.keys(value)) assert(required.includes(key), `${label} has unknown field ${key}`);
}

function hash(value, label) {
  assert(typeof value === "string" && shaPattern.test(value), `${label} must be lowercase SHA-256`);
}

function integer(value, minimum, maximum, label) {
  assert(Number.isSafeInteger(value) && value >= minimum && value <= maximum,
    `${label} must be an integer in [${minimum}, ${maximum}]`);
}

function validateLane(lane, modelHash, costs, label, council) {
  const keys = [
    "model_sha256", "trace_sha256", "prediction_label", "confidence_milli", "abstained",
    "decision_id",
  ];
  if (council) keys.push("receipt_sha256");
  exactKeys(lane, keys, label);
  assert(lane.model_sha256 === modelHash, `${label} did not use the frozen underlying model`);
  hash(lane.trace_sha256, `${label} trace hash`);
  if (council) hash(lane.receipt_sha256, `${label} receipt hash`);
  assert(typeof lane.prediction_label === "boolean", `${label} prediction must be boolean`);
  integer(lane.confidence_milli, 0, 1000, `${label} confidence`);
  assert(typeof lane.abstained === "boolean", `${label} abstention must be boolean`);
  assert(typeof lane.decision_id === "string" && Object.hasOwn(costs, lane.decision_id),
    `${label} decision has no frozen cost`);
}

function validateEpisode(episode, modelHash) {
  exactKeys(episode, [
    "schema", "episode_id", "dimension", "source_family", "unfamiliar_source",
    "evidence_sha256", "gold_opened_after_both_predictions", "gold", "solo", "council",
  ], `episode ${episode?.episode_id ?? "unknown"}`);
  assert(episode.schema === "nsrl.solomon_wisdom_episode.v0", "wrong wisdom episode schema");
  assert(typeof episode.episode_id === "string" && episode.episode_id.length > 0,
    "wisdom episode id is empty");
  assert(WISDOM_DIMENSIONS.includes(episode.dimension), "unknown wisdom dimension");
  assert(typeof episode.source_family === "string" && episode.source_family.length > 0,
    "wisdom source family is empty");
  assert(typeof episode.unfamiliar_source === "boolean", "unfamiliar-source flag must be boolean");
  assert(Array.isArray(episode.evidence_sha256) && episode.evidence_sha256.length > 0,
    "wisdom episode needs evidence hashes");
  episode.evidence_sha256.forEach((value) => hash(value, "episode evidence hash"));
  assert(episode.gold_opened_after_both_predictions === true,
    "gold must be opened only after both frozen predictions");
  exactKeys(episode.gold, [
    "expected_label", "should_abstain", "decision_costs_milli",
  ], "wisdom gold");
  assert(typeof episode.gold.expected_label === "boolean", "gold label must be boolean");
  assert(typeof episode.gold.should_abstain === "boolean", "gold abstention must be boolean");
  const costs = episode.gold.decision_costs_milli;
  assert(costs && typeof costs === "object" && !Array.isArray(costs)
    && Object.keys(costs).length > 0, "wisdom gold needs decision costs");
  for (const [decisionId, cost] of Object.entries(costs)) {
    assert(decisionId.length > 0, "empty decision id in gold costs");
    integer(cost, 0, Number.MAX_SAFE_INTEGER, `decision cost ${decisionId}`);
  }
  validateLane(episode.solo, modelHash, costs, "solo lane", false);
  validateLane(episode.council, modelHash, costs, "council lane", true);
  if (episode.dimension === "hard_negative_rejection") {
    assert(episode.gold.expected_label === false, "hard-negative episode must have a negative gold label");
  }
  if (episode.dimension === "unfamiliar_source_transfer") {
    assert(episode.unfamiliar_source === true, "transfer episode must use an unfamiliar source");
  }
}

function validateIntegrityBinding(binding, schema, modelHash, production, label, artifactBase) {
  exactKeys(binding, ["path", "sha256", "schema"], label);
  assert(typeof binding.path === "string" && binding.path.length > 0, `${label} path is empty`);
  hash(binding.sha256, `${label} hash`);
  assert(binding.schema === schema, `${label} schema binding changed`);
  if (!production) return;
  const resolved = path.isAbsolute(binding.path) ? binding.path : path.resolve(artifactBase, binding.path);
  assert(fs.existsSync(resolved), `${label} is missing: ${binding.path}`);
  const bytes = fs.readFileSync(resolved);
  assert(sha256Bytes(bytes) === binding.sha256, `${label} artifact hash changed`);
  const report = JSON.parse(bytes);
  assert(report.schema === schema && report.ok === true,
    `${label} artifact is not a green ${schema} report`);
  assert(report.model_artifact_sha256 === modelHash, `${label} model binding changed`);
  const requiredGates = schema === "nsrl.wisdom_provenance_gate.v0"
    ? [
      "no_oracle_target_lookup", "no_hidden_memory", "no_retrieval_target_leakage",
      "gold_sealed_until_both_predictions",
    ]
    : [
      "quality_report_green", "generation_integrity_green", "source_grounding_green",
      "cross_modal_agreement_green", "same_model_invocation_green", "trace_replay_green",
      "faculty_output_binding_green",
    ];
  exactKeys(report.gates, requiredGates, `${label} gates`);
  assert(Object.values(report.gates).every((value) => value === true),
    `${label} contains a non-green required gate`);
  if (schema === "nsrl.wisdom_provenance_gate.v0") {
    assert(Array.isArray(report.source_hashes) && report.source_hashes.length > 0,
      "provenance report has no source hashes");
    report.source_hashes.forEach((value) => hash(value, "provenance source hash"));
    assert(Array.isArray(report.trace_hashes) && report.trace_hashes.length > 0,
      "provenance report has no trace hashes");
    report.trace_hashes.forEach((value) => hash(value, "provenance trace hash"));
  } else {
    exactKeys(report.source_report, ["path", "sha256"], "generation-integrity source report");
    hash(report.source_report.sha256, "generation-integrity source report hash");
    const sourcePath = path.isAbsolute(report.source_report.path)
      ? report.source_report.path : path.resolve(artifactBase, report.source_report.path);
    assert(fs.existsSync(sourcePath),
      `generation-integrity source report is missing: ${report.source_report.path}`);
    assert(sha256Bytes(fs.readFileSync(sourcePath)) === report.source_report.sha256,
      "generation-integrity source report byte hash changed");
  }
}

function correctness(rows, lane) {
  const correct = rows.filter((row) => row[lane].prediction_label === row.gold.expected_label).length;
  return {correct, score_per_mille: Math.floor(correct * 1000 / rows.length)};
}

function calibration(rows, lane) {
  const squared = rows.reduce((sum, row) => {
    const prediction = row[lane].prediction_label
      ? row[lane].confidence_milli : 1000 - row[lane].confidence_milli;
    const target = row.gold.expected_label ? 1000 : 0;
    return sum + (prediction - target) ** 2;
  }, 0);
  const meanSquaredMillionths = Math.floor(squared / rows.length);
  return {
    mean_brier_millionths: meanSquaredMillionths,
    score_per_mille: 1000 - Math.floor(meanSquaredMillionths / 1000),
  };
}

function abstention(rows, lane) {
  const correct = rows.filter((row) => row[lane].abstained === row.gold.should_abstain).length;
  return {correct, score_per_mille: Math.floor(correct * 1000 / rows.length)};
}

function regret(rows, lane) {
  const total = rows.reduce((sum, row) => {
    const values = Object.values(row.gold.decision_costs_milli);
    const best = Math.min(...values);
    return sum + row.gold.decision_costs_milli[row[lane].decision_id] - best;
  }, 0);
  return {total_regret_milli: total, mean_regret_milli: Math.floor(total / rows.length)};
}

export function evaluateWisdom(input, {evaluatorSha256 = "", artifactBase = process.cwd()} = {}) {
  const inputKeys = [
    "schema", "analysis_role", "frozen_before_outcomes", "minimum_cases_per_dimension",
    "underlying_model", "integrity", "episodes",
  ];
  if (Object.hasOwn(input, "ceremony")) inputKeys.push("ceremony");
  exactKeys(input, inputKeys, "wisdom evaluation");
  assert(input.schema === "nsrl.solomon_wisdom_eval.v0", "wrong wisdom evaluation schema");
  assert(["frozen_same_model_comparison", "self_test_only"].includes(input.analysis_role),
    "unknown wisdom evaluation role");
  assert(input.frozen_before_outcomes === true, "wisdom evaluation was not frozen before outcomes");
  integer(input.minimum_cases_per_dimension, 1, 100000, "minimum wisdom cases");
  if (input.analysis_role === "frozen_same_model_comparison") {
    assert(input.minimum_cases_per_dimension >= 72,
      "production wisdom evaluation requires at least 72 cases per dimension");
  }
  exactKeys(input.underlying_model, ["model_id", "artifact_sha256"], "wisdom model");
  assert(typeof input.underlying_model.model_id === "string"
    && input.underlying_model.model_id.length > 0, "wisdom model id is empty");
  hash(input.underlying_model.artifact_sha256, "wisdom model hash");
  exactKeys(input.integrity, [
    "no_oracle_target_lookup", "no_hidden_memory", "no_retrieval_target_leakage",
    "generation_integrity_passed", "generation_integrity_report", "provenance_report",
  ], "wisdom integrity");
  assert([
    input.integrity.no_oracle_target_lookup,
    input.integrity.no_hidden_memory,
    input.integrity.no_retrieval_target_leakage,
    input.integrity.generation_integrity_passed,
  ].every((value) => value === true),
    "wisdom evaluation integrity gate failed");
  const production = input.analysis_role === "frozen_same_model_comparison";
  validateIntegrityBinding(input.integrity.generation_integrity_report,
    "nsrl.wisdom_generation_integrity.v0", input.underlying_model.artifact_sha256,
    production, "generation integrity report", artifactBase);
  validateIntegrityBinding(input.integrity.provenance_report,
    "nsrl.wisdom_provenance_gate.v0", input.underlying_model.artifact_sha256,
    production, "provenance report", artifactBase);
  if (production) {
    assert(Object.hasOwn(input, "ceremony"),
      "production wisdom evaluation requires a byte-bound ceremony");
    verifyWisdomCeremonyCompilation(input, {baseDir: artifactBase});
  }
  assert(Array.isArray(input.episodes), "wisdom episodes must be an array");
  input.episodes.forEach((episode) => validateEpisode(
    episode, input.underlying_model.artifact_sha256));
  assert(new Set(input.episodes.map((episode) => episode.episode_id)).size === input.episodes.length,
    "wisdom episode ids repeat");
  const dimensions = {};
  for (const dimension of WISDOM_DIMENSIONS) {
    const rows = input.episodes.filter((episode) => episode.dimension === dimension);
    assert(rows.length >= input.minimum_cases_per_dimension,
      `${dimension} has ${rows.length} cases below ${input.minimum_cases_per_dimension}`);
    let solo;
    let council;
    let councilOutperforms;
    if (dimension === "calibration") {
      solo = calibration(rows, "solo");
      council = calibration(rows, "council");
      councilOutperforms = council.mean_brier_millionths < solo.mean_brier_millionths;
    } else if (dimension === "decision_regret") {
      solo = regret(rows, "solo");
      council = regret(rows, "council");
      councilOutperforms = council.mean_regret_milli < solo.mean_regret_milli;
    } else if (dimension === "appropriate_abstention") {
      solo = abstention(rows, "solo");
      council = abstention(rows, "council");
      councilOutperforms = council.score_per_mille > solo.score_per_mille;
    } else {
      solo = correctness(rows, "solo");
      council = correctness(rows, "council");
      councilOutperforms = council.score_per_mille > solo.score_per_mille;
    }
    dimensions[dimension] = {cases: rows.length, solo, council, council_outperforms: councilOutperforms};
  }
  const allDimensionsOutperform = Object.values(dimensions).every(
    (dimension) => dimension.council_outperforms);
  return {
    schema: "nsrl.solomon_wisdom_eval_result.v0",
    analysis_role: input.analysis_role,
    source_sha256: sha256Json(input),
    evaluator_sha256: evaluatorSha256,
    underlying_model: input.underlying_model,
    integrity: input.integrity,
    dimensions,
    verdict: {
      all_dimensions_outperform: allDimensionsOutperform,
      promotion_gate_passed: production && allDimensionsOutperform,
      self_test_only: !production,
    },
    authorization: {
      council_promotion_authorized: production && allDimensionsOutperform,
      product_release_authorized: false,
    },
  };
}
