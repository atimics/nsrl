#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json";
const resultPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-result.json";
const outputDirectory = process.argv[4]
  ?? "data/experiments/production-model-v1/p10m-adaptive-composition-v1/execution";
const replayDirectory = process.argv[5] ?? null;
const fail = (message) => { throw new Error(`adaptive composition execution: ${message}`); };
const assert = (condition, message) => { if (!condition) fail(message); };
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const read = (file) => fs.readFileSync(file);
const parseTsv = (file) => {
  const lines = fs.readFileSync(file, "utf8").trimEnd().split("\n");
  const header = lines.shift().split("\t");
  return lines.filter(Boolean).map((line) => Object.fromEntries(
    line.split("\t").map((value, index) => [header[index], value])));
};
const contract = JSON.parse(read(contractPath));
const result = JSON.parse(read(resultPath));
assert(contract.schema === "nsrl.adaptive_composition_execution_contract.v1"
  && contract.analysis_role === "frozen_after_calibration_before_adaptive_endpoint"
  && contract.calibration.artifacts_frozen_before_adaptive_endpoint
  && !contract.calibration.adaptive_outcomes_read_at_freeze
  && !contract.calibration.endpoint_outcomes_read_at_freeze,
"wrong execution contract");
assert(result.schema === "nsrl.adaptive_composition_result.v1"
  && result.analysis_role === "preregistered_fresh_source_execution",
"wrong result schema");

const verifyBinding = (binding, label) => {
  const bytes = read(binding.path);
  assert(bytes.length === binding.bytes && sha256(bytes) === binding.sha256,
    `${label} binding changed`);
};
verifyBinding(contract.bindings.preregistration, "preregistration");
verifyBinding(contract.bindings.source_frame, "source frame");
verifyBinding(contract.bindings.action_manifest, "action manifest");
verifyBinding(contract.bindings.actions, "actions");
verifyBinding(contract.bindings.predictor, "predictor");
verifyBinding(contract.bindings.fitting_cube, "fitting cube");
verifyBinding(contract.bindings.runner_source, "runner source");
verifyBinding(contract.bindings.runner_binary, "runner binary");
verifyBinding(contract.bindings.checker, "checker");
verifyBinding(contract.bindings.freezer, "freezer");
verifyBinding(contract.bindings.sealer, "calibration sealer");
verifyBinding(contract.bindings.precalibration_contract, "pre-calibration contract");
for (const [kind, binding] of Object.entries(contract.bindings.calibration)) {
  verifyBinding(binding, `calibration ${kind}`);
}
for (const [state, binding] of Object.entries(contract.bindings.state_models)) {
  verifyBinding(binding, `state model ${state}`);
}
for (const [role, artifacts] of Object.entries(contract.bindings.roles)) {
  for (const [kind, binding] of Object.entries(artifacts)) verifyBinding(binding, `${role} ${kind}`);
}

const frame = JSON.parse(read(contract.bindings.source_frame.path));
assert(frame.schema === "nsrl.adaptive_composition_source_frame.v1"
  && frame.outcome_firewall.all_m18_m19_source_ids_and_independence_keys_excluded,
"fresh source frame changed");
for (const family of ["federal_register", "rfc", "science"]) {
  const sources = frame.sources.filter((source) => source.family === family);
  assert(sources.length === 152
    && sources.filter((source) => source.role === "fitting").length === 12
    && sources.filter((source) => source.role === "calibration").length === 119
    && sources.filter((source) => source.role === "adaptive").length === 2
    && sources.filter((source) => source.role === "endpoint").length === 19
    && new Set(sources.map((source) => source.source_id)).size === 152
    && new Set(sources.map((source) => source.independence_key)).size === 152,
  `${family} source role partition changed`);
}
const priorSources = frame.exclusions.flatMap((binding) => JSON.parse(read(binding.path)).sources);
const priorIds = new Set(priorSources.map((source) => source.source_id));
const priorKeys = new Set(priorSources.map((source) => `${source.family}\0${source.independence_key}`));
assert(frame.sources.every((source) => !priorIds.has(source.source_id)
  && !priorKeys.has(`${source.family}\0${source.independence_key}`)),
"M18/M19 source identity leaked into M5");

const actions = parseTsv(contract.bindings.actions.path);
assert(actions.length === 28, "action manifest must contain 14 two-write actions");
for (const state of ["empty", "H", "T", "HH", "HT", "TH", "TT"]) {
  const head = actions.filter((row) => row.state === state && row.action === "H");
  const trunk = actions.filter((row) => row.state === state && row.action === "T");
  assert(head.length === 2 && trunk.length === 2
    && [...head, ...trunk].every((row) => ["-1", "1"].includes(row.delta))
    && head.every((row) => Number(row.group) >= 11)
    && trunk.every((row) => Number(row.group) < 11)
    && head.map((row) => `${row.group}:${row.coordinate}:${row.delta}`).join(",")
      !== trunk.map((row) => `${row.group}:${row.coordinate}:${row.delta}`).join(","),
  `${state} physical actions are not distinct`);
}
assert(contract.noncommutativity.passed
  && contract.noncommutativity.ht_model_hash !== contract.noncommutativity.th_model_hash
  && contract.noncommutativity.ht_function_hash !== contract.noncommutativity.th_function_hash
  && contract.bindings.state_models.HT.sha256 !== contract.bindings.state_models.TH.sha256,
"noncommutativity gate failed");

const cube = parseTsv(path.join(outputDirectory, "calibration-cube.tsv"));
const scores = parseTsv(path.join(outputDirectory, "calibration-scores.tsv"));
const corrections = parseTsv(path.join(outputDirectory, "corrections.tsv"));
assert(cube.length === 1428 * 7 * 2, "full calibration cube row count changed");
assert(scores.length === 357 && corrections.length === 3, "calibration summary shape changed");
const calibrationManifest = JSON.parse(read(contract.bindings.calibration.manifest.path));
assert(calibrationManifest.schema === "nsrl.adaptive_composition_calibration.v1"
  && calibrationManifest.cube_rows === cube.length
  && calibrationManifest.source_scores === scores.length
  && !calibrationManifest.adaptive_outcomes_read
  && !calibrationManifest.endpoint_outcomes_read,
"calibration firewall manifest changed");
const scoreBySource = new Map();
for (const row of cube) {
  const contrast = BigInt(row.contrast_q32);
  const predicted = BigInt(row.predicted_q32);
  const residual = BigInt(row.residual_q32);
  assert(contrast - predicted === residual, "calibration residual arithmetic failed");
  const key = `${row.family}\0${row.source_id}`;
  if (!scoreBySource.has(key) || residual > scoreBySource.get(key)) scoreBySource.set(key, residual);
}
for (const row of scores) {
  assert(scoreBySource.get(`${row.family}\0${row.source_id}`) === BigInt(row.simultaneous_score_q32),
    "simultaneous source score changed");
}
for (const family of ["federal_register", "rfc", "science"]) {
  const familyScores = scores.filter((row) => row.family === family)
    .map((row) => BigInt(row.simultaneous_score_q32)).sort((left, right) => left < right ? -1 : 1);
  const correction = corrections.find((row) => row.family === family);
  assert(familyScores.length === 119 && correction.rank === "119"
    && BigInt(correction.correction_q32) === familyScores[118]
    && BigInt(result.corrections_q32[family]) === familyScores[118],
  `${family} rank-119 correction changed`);
}

const decisions = parseTsv(path.join(outputDirectory, "decisions.tsv"));
assert(decisions.length === 96, "four policy replays must each contain 24 decisions");
const predictor = new Map(parseTsv(contract.bindings.predictor.path).map((row) =>
  [`${row.family}\0${row.state}\0${row.action}`, BigInt(row.lower_median_contrast_q32)]));
const correctionByFamily = new Map(corrections.map((row) =>
  [row.family, BigInt(row.correction_q32)]));
const adaptivePanels = parseTsv(contract.bindings.roles.adaptive.panels.path);
const nextState = (state, action) => state === "empty" ? action : `${state}${action}`;
for (const policy of ["adaptive", "always_abstain", "head_only", "trunk_only"]) {
  const rows = decisions.filter((row) => row.policy === policy);
  assert(rows.length === 24, `${policy} decision count changed`);
  let state = "empty";
  let accepted = 0;
  for (const [document, row] of rows.entries()) {
    const panel = adaptivePanels[document];
    assert(Number(row.document) === document && row.state_before === state
      && row.family === panel.family && row.source_id === panel.source_id
      && row.passage === panel.passage_ordinal,
      `${policy} state trace changed`);
    const allowed = policy === "adaptive" ? ["H", "T"]
      : policy === "head_only" ? ["H"] : policy === "trunk_only" ? ["T"] : [];
    const certified = accepted >= 2 ? [] : allowed.map((action) => ({action,
      upper: predictor.get(`${row.family}\0${state}\0${action}`)
        + correctionByFamily.get(row.family)})).filter(({upper}) => upper < 0n)
      .sort((left, right) => left.upper < right.upper ? -1
        : left.upper > right.upper ? 1 : left.action.localeCompare(right.action));
    const expected = certified[0] ?? null;
    if (row.action === "abstain") {
      assert(row.certified_upper_q32 === "" && row.exact_contrast_q32 === "0"
        && row.state_after === state && expected === null,
      `${policy} abstention changed state or skipped a certified action`);
      continue;
    }
    assert(accepted < 2 && ["H", "T"].includes(row.action) && expected
      && row.action === expected.action
      && BigInt(row.certified_upper_q32) === expected.upper,
    `${policy} violated the frozen certificate or tie-break rule`);
    if (policy === "head_only") assert(row.action === "H", "head-only policy used trunk");
    if (policy === "trunk_only") assert(row.action === "T", "trunk-only policy used head");
    state = nextState(state, row.action);
    accepted += 1;
    assert(row.state_after === state && ["H", "T", "HH", "HT", "TH", "TT"].includes(state),
      `${policy} escaped reachable state set`);
  }
  const endpoint = result.endpoints[policy];
  assert(endpoint.final_state === state && endpoint.accepted_actions === accepted,
    `${policy} endpoint state changed`);
}
const adaptiveRows = decisions.filter((row) => row.policy === "adaptive" && row.action !== "abstain");
const signedRegret = adaptiveRows.reduce((sum, row) => sum + BigInt(row.exact_contrast_q32), 0n);
const positiveRegret = adaptiveRows.reduce((sum, row) => {
  const value = BigInt(row.exact_contrast_q32); return sum + (value > 0n ? value : 0n);
}, 0n);
const allFiredStrictlyNegative = adaptiveRows.every((row) => BigInt(row.exact_contrast_q32) < 0n);
assert(signedRegret === BigInt(result.adaptive_trajectory.signed_regret_q32)
  && positiveRegret === BigInt(result.adaptive_trajectory.positive_regret_q32)
  && result.adaptive_trajectory.head_fires
    === adaptiveRows.filter((row) => row.action === "H").length
  && result.adaptive_trajectory.trunk_fires
    === adaptiveRows.filter((row) => row.action === "T").length,
"adaptive regret or firing summary changed");

const endpointNll = (name) => BigInt(result.endpoints[name].total_nll_millibits);
const beatsAbstain = endpointNll("adaptive") < endpointNll("always_abstain");
const beatsBestFixed = endpointNll("adaptive")
  < (endpointNll("head_only") < endpointNll("trunk_only")
    ? endpointNll("head_only") : endpointNll("trunk_only"));
const zeroNonincrease = result.endpoints.adaptive.zero_probability_windows
  <= result.endpoints.always_abstain.zero_probability_windows;
const bothFamilies = result.adaptive_trajectory.head_fires > 0
  && result.adaptive_trajectory.trunk_fires > 0;
const zeroRegret = positiveRegret === 0n && allFiredStrictlyNegative;
const supported = beatsAbstain && beatsBestFixed && zeroNonincrease && bothFamilies && zeroRegret;
assert(result.support_gates.beats_always_abstain === beatsAbstain
  && result.support_gates.beats_best_fixed_policy === beatsBestFixed
  && result.support_gates.zero_probability_nonincrease === zeroNonincrease
  && result.support_gates.both_physical_families_fire === bothFamilies
  && result.support_gates.zero_positive_regret === zeroRegret
  && result.support_gates.all_fired_exact_contrasts_strictly_negative
    === allFiredStrictlyNegative
  && result.support_gates.all_passed === supported
  && result.verdict === (supported ? "supported" : "falsified"),
"preregistered verdict arithmetic changed");

for (const [policy, artifact] of [
  ["adaptive", "adaptive-final.nsrlpm"], ["always_abstain", "always-abstain-final.nsrlpm"],
  ["head_only", "head-only-final.nsrlpm"], ["trunk_only", "trunk-only-final.nsrlpm"],
]) {
  const finalState = result.endpoints[policy].final_state;
  assert(read(path.join(outputDirectory, artifact)).equals(read(
    contract.bindings.state_models[finalState].path)), `${policy} retained model is not persisted state`);
}
if (replayDirectory) {
  for (const artifact of ["calibration-cube.tsv", "calibration-scores.tsv", "corrections.tsv",
    "decisions.tsv", "adaptive-final.nsrlpm", "always-abstain-final.nsrlpm",
    "head-only-final.nsrlpm", "trunk-only-final.nsrlpm"]) {
    assert(read(path.join(outputDirectory, artifact)).equals(read(path.join(replayDirectory, artifact))),
      `replay mismatch: ${artifact}`);
  }
  assert(read(resultPath).equals(read(path.join(replayDirectory, "result.json"))),
    "result JSON replay mismatch");
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.adaptive_composition_execution_check.v1", verdict: result.verdict,
  calibration_cube_rows: cube.length, calibration_source_panels: scores.length,
  adaptive_fires: adaptiveRows.length, head_fires: result.adaptive_trajectory.head_fires,
  trunk_fires: result.adaptive_trajectory.trunk_fires,
  positive_regret_q32: result.adaptive_trajectory.positive_regret_q32,
  endpoints: result.endpoints, byte_replay: Boolean(replayDirectory), ok: true,
}, null, 2)}\n`);
