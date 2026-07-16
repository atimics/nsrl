#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const check = args[0] === "--check";
if (check) args.shift();
const contractPath = args[0]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json";
const resultPath = args[1]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-result.json";
const replayPath = args[2]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-replay-receipt.json";
const outputPath = args[3]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-publication.json";
if (args.length > 4) throw new Error("too many adaptive publication arguments");

const resolve = (value) => path.isAbsolute(value) ? value : path.join(root, value);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const bind = (value) => {
  const bytes = fs.readFileSync(resolve(value));
  return {path: value, sha256: sha256(bytes), bytes: bytes.length};
};
const assert = (condition, message) => {
  if (!condition) throw new Error(`adaptive publication: ${message}`);
};
const contractBinding = bind(contractPath);
const resultBinding = bind(resultPath);
const replayBinding = bind(replayPath);
const contract = JSON.parse(fs.readFileSync(resolve(contractPath)));
const result = JSON.parse(fs.readFileSync(resolve(resultPath)));
const replay = JSON.parse(fs.readFileSync(resolve(replayPath)));
assert(contract.schema === "nsrl.adaptive_composition_execution_contract.v1"
  && result.schema === "nsrl.adaptive_composition_result.v1"
  && replay.schema === "nsrl.adaptive_composition_replay_receipt.v1",
"wrong publication input schema");
assert(replay.sources.contract.sha256 === contractBinding.sha256
  && replay.sources.result.sha256 === resultBinding.sha256
  && replay.verdict === result.verdict,
"replay receipt is not bound to the publication inputs");
assert(Object.values(replay.guarantees).every((value) => value === true || value === false)
  && replay.guarantees.calibration_byte_replay
  && replay.guarantees.decision_trace_byte_replay
  && replay.guarantees.retained_model_byte_replay
  && replay.guarantees.result_json_byte_replay
  && replay.guarantees.post_outcome_threshold_change === false,
"replay receipt does not prove the frozen execution");

const allowedStatuses = ["supported", "falsified", "inconclusive"];
assert(allowedStatuses.includes(result.verdict), "result has an unknown verdict");
const noActions = result.adaptive_trajectory.accepted_actions === 0;
const hardFalsifiers = {
  no_adaptive_action_fired: noActions,
  adaptive_did_not_beat_always_abstain: !result.support_gates.beats_always_abstain,
  adaptive_did_not_beat_best_fixed_policy: !result.support_gates.beats_best_fixed_policy,
  both_physical_action_families_did_not_fire: !result.support_gates.both_physical_families_fire,
  zero_probability_windows_increased: !result.support_gates.zero_probability_nonincrease,
  exact_replay_failed: false,
};
assert(result.verdict === "falsified"
  && Object.values(hardFalsifiers).some(Boolean),
"publication is not the frozen falsification");
const corrections = Object.values(result.corrections_q32).map(BigInt);
const publication = {
  schema: "nsrl.adaptive_composition_publication.v1",
  publication_contract: {
    allowed_statuses: allowedStatuses,
    fail_closed_on_unknown_status: true,
  },
  sources: {
    contract: contractBinding,
    result: resultBinding,
    replay_receipt: replayBinding,
    publisher: bind(path.relative(root, fileURLToPath(import.meta.url))),
  },
  verdict: {
    status: result.verdict,
    supported: result.verdict === "supported",
    falsified: result.verdict === "falsified",
    inconclusive: result.verdict === "inconclusive",
    hard_falsifiers: hardFalsifiers,
  },
  claims: [
    {
      id: "nonvacuous_persistent_adaptive_composition_improves_canonical_nll",
      status: "falsified",
      scope: "six fresh adaptive source panels, 24 ordered passages, two frozen noncommuting physical action families, maximum two persistent accepts",
      reason: "the simultaneous rank-119 corrections admitted no action, so every policy retained the empty state and tied always-abstain",
    },
    {
      id: "zero_positive_regret_for_fired_actions",
      status: "inconclusive",
      scope: "the same frozen six-panel adaptive trajectory",
      reason: "zero positive regret is vacuous because no action fired",
    },
  ],
  evidence: {
    corrections_q32: result.corrections_q32,
    minimum_correction_q32: corrections.reduce((left, right) => left < right ? left : right).toString(),
    maximum_correction_q32: corrections.reduce((left, right) => left > right ? left : right).toString(),
    adaptive_trajectory: result.adaptive_trajectory,
    endpoints: result.endpoints,
    support_gates: result.support_gates,
    calibration_cube_rows: replay.execution_summary.calibration_cube_rows,
    calibration_source_panels: replay.execution_summary.calibration_source_panels,
    exact_byte_replay: true,
  },
  interpretation: {
    threshold_retuning_after_outcome_authorized: false,
    frozen_successor_modification_authorized: false,
    default_optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    product_release_authorized: false,
    next_admissible_optimizer_experiment: "E21-A momentum-embedded quantization error on the small transformer",
    next_product_facing_experiment: "E21-D native Q32 pairwise-sigmoid multimodal alignment",
  },
};
const bytes = Buffer.from(`${JSON.stringify(publication, null, 2)}\n`);
const absoluteOutput = resolve(outputPath);
if (check) {
  assert(fs.existsSync(absoluteOutput) && fs.readFileSync(absoluteOutput).equals(bytes),
    "publication does not byte-replay");
} else {
  assert(!fs.existsSync(absoluteOutput), "refusing to overwrite adaptive publication");
  fs.mkdirSync(path.dirname(absoluteOutput), {recursive: true});
  fs.writeFileSync(absoluteOutput, bytes, {flag: "wx"});
}
process.stdout.write(`${JSON.stringify({
  schema: publication.schema,
  checked: check,
  verdict: publication.verdict.status,
  adaptive_fires: publication.evidence.adaptive_trajectory.accepted_actions,
  endpoint_nll_millibits: publication.evidence.endpoints.adaptive.total_nll_millibits,
  byte_replay: publication.evidence.exact_byte_replay,
  output: outputPath,
}, null, 2)}\n`);
