#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const preregPath = process.argv[2]
  ?? "protocol/examples/p10m-adaptive-composition-v1-preregistration.json";
const framePath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-source-frame.json";
const manifestDirectory = process.argv[4]
  ?? "data/experiments/production-model-v1/p10m-adaptive-composition-v1/manifest";
const processedDirectory = process.argv[5]
  ?? "data/processed/p10m-adaptive-composition-v1";
const binaryPath = process.argv[6]
  ?? "target/release/nsrl-adaptive-composition";
const outputPath = process.argv[7]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json";

const fail = (message) => { throw new Error(`adaptive composition freeze: ${message}`); };
const assert = (condition, message) => { if (!condition) fail(message); };
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const bind = (artifactPath) => {
  const bytes = fs.readFileSync(artifactPath);
  return {path: artifactPath, sha256: sha256(bytes), bytes: bytes.length};
};
const prereg = JSON.parse(fs.readFileSync(preregPath));
const frame = JSON.parse(fs.readFileSync(framePath));
const actionTracePath = path.join(manifestDirectory, "action-manifest.json");
const actionTrace = JSON.parse(fs.readFileSync(actionTracePath));

assert(prereg.schema === "nsrl.adaptive_composition_preregistration.v1"
  && prereg.analysis_role === "prospective_pre_source_acquisition",
"wrong preregistration");
assert(frame.schema === "nsrl.adaptive_composition_source_frame.v1"
  && frame.analysis_role === "prospective_pre_outcome_source_frame",
"source frame is not prospective");
assert(Object.values(frame.outcome_firewall).every((value) => value === false || value === true)
  && !frame.outcome_firewall.fitting_outcomes_read
  && !frame.outcome_firewall.calibration_outcomes_read
  && !frame.outcome_firewall.adaptive_outcomes_read
  && !frame.outcome_firewall.endpoint_outcomes_read
  && frame.outcome_firewall.all_m18_m19_source_ids_and_independence_keys_excluded,
"source outcome firewall changed");
assert(actionTrace.schema === "nsrl.adaptive_composition_action_manifest.v1"
  && actionTrace.analysis_role === "fitting_only_before_calibration"
  && actionTrace.noncommutativity.passed
  && !actionTrace.calibration_outcomes_read
  && !actionTrace.adaptive_outcomes_read
  && !actionTrace.endpoint_outcomes_read,
"action manifest crossed an evaluation firewall");

const roleArtifacts = Object.fromEntries(Object.keys(prereg.source_design
  ? {fitting: 1, calibration: 1, adaptive: 1, endpoint: 1}
  : {}).map((role) => [role, {
    corpus: bind(path.join(processedDirectory, `${role}.txt`)),
    index: bind(path.join(processedDirectory, `${role}.index.tsv`)),
    panels: bind(path.join(processedDirectory, `${role}.panels.tsv`)),
    tokens: bind(path.join(processedDirectory, `${role}.nsrltok`)),
    token_trace: bind(path.join(processedDirectory, `${role}.tokens.json`)),
  }]));
assert(Object.keys(roleArtifacts).length === 4, "role artifact set changed");

const stateModels = Object.fromEntries(
  ["empty", "H", "T", "HH", "HT", "TH", "TT"].map((state) =>
    [state, bind(path.join(manifestDirectory, `state-${state}.nsrlpm`))]));
const runnerSource = "crates/nsrl-train/src/bin/nsrl-adaptive-composition.rs";
const checkerSource = "scripts/check-adaptive-composition-execution-v1.mjs";
const freezerSource = "scripts/freeze-adaptive-composition-v1.mjs";
const sealerSource = "scripts/seal-adaptive-composition-calibration-v1.mjs";
const contract = {
  schema: "nsrl.adaptive_composition_execution_contract.v1",
  analysis_role: "frozen_after_fitting_before_calibration",
  experiment_id: prereg.experiment_id,
  theory: prereg.theory,
  bindings: {
    preregistration: bind(preregPath), source_frame: bind(framePath),
    action_manifest: bind(actionTracePath),
    actions: bind(path.join(manifestDirectory, "actions.tsv")),
    predictor: bind(path.join(manifestDirectory, "predictor.tsv")),
    fitting_cube: bind(path.join(manifestDirectory, "fitting-cube.tsv")),
    state_models: stateModels, roles: roleArtifacts,
    runner_source: bind(runnerSource), runner_binary: bind(binaryPath),
    checker: bind(checkerSource), freezer: bind(freezerSource), sealer: bind(sealerSource),
  },
  controller: prereg.controller,
  actions: prereg.actions,
  reachable_state_contract: prereg.reachable_state_contract,
  noncommutativity: actionTrace.noncommutativity,
  predictor: {
    algorithm: "within-family lower median exact fitting contrast",
    observations_per_family_state_action: 48,
    fitting_panels_per_family: 12,
    current_candidate_outcomes_used: false,
  },
  calibration: {
    score: "max over seven states, two physical actions, and four passages of exact contrast minus fitting predictor",
    sources_per_family: 119, order_statistic_rank: 119,
    per_panel_error_spend: "1/120", global_error: "1/20",
  },
  endpoint: prereg.objectives,
  comparators: prereg.comparators,
  support_requires: prereg.support_requires,
  hard_falsifiers: prereg.hard_falsifiers,
  authorization: {
    calibration_evaluation: true, adaptive_evaluation: true, endpoint_evaluation: true,
    granted_by_user_message: "run it", optimizer_promotion: false, paid_scaling: false,
  },
};
const bytes = Buffer.from(`${JSON.stringify(contract, null, 2)}\n`);
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
fs.writeFileSync(outputPath, bytes);
process.stdout.write(`${JSON.stringify({
  schema: contract.schema, output: outputPath, sha256: sha256(bytes),
  source_frame_frozen: true, actions_frozen: true, runner_checker_sealer_frozen: true,
  evaluation_authorized: true,
}, null, 2)}\n`);
