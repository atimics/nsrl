#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
assert(contract.schema === "nsrl.production_target_margin_trust_region_contract.v1",
  "target-margin trust-region contract schema is invalid");
for (const artifact of contract.implementation.artifacts) {
  assert(sha256(fs.readFileSync(artifact.path)) === artifact.sha256,
    `${artifact.path} SHA-256 mismatch`);
}

const expected = contract.preflight;
assert(config.traces.length === expected.feature_shifts.length,
  "one preflight trace is required for every frozen feature shift");
const candidates = config.traces.map(({shift, file}) => {
  const bytes = fs.readFileSync(file);
  const trace = JSON.parse(bytes);
  assert(expected.feature_shifts.includes(shift), `unexpected feature shift ${shift}`);
  assert(trace.schema === "nsrl.production_target_margin_train.v1",
    `${file} has the wrong trace schema`);
  const training = trace.training;
  const guard = trace.descent_guard;
  assert(training.feature_shift === shift
    && training.context_tokens === expected.context_tokens
    && training.windows === expected.windows
    && training.window_schedule_windows === expected.window_schedule_windows
    && training.evaluation_windows === expected.evaluation_windows
    && training.targets_per_window === expected.targets_per_window
    && training.epochs === expected.epochs
    && training.batch_windows === expected.batch_windows
    && training.optimizer_steps === expected.optimizer_steps
    && training.margin_q8 === expected.margin_q8
    && guard.windows === expected.descent_guard_windows,
  `${file} does not match the frozen preflight geometry`);
  const gates = {
    schedule_complete: trace.cursor.schedule_complete === true,
    output_matrix_movement_minimum:
      training.movement_l1 >= expected.candidate_gates.output_matrix_movement_minimum,
    accepted_guard_batches_minimum:
      guard.batches_accepted >= expected.candidate_gates.accepted_guard_batches_minimum,
    frozen_parameters_unchanged: trace.gates.frozen_parameters_unchanged === true,
    output_bias_unchanged: trace.gates.output_bias_unchanged === true,
    weight_saturation_maximum:
      trace.health.weight_saturation_count
        <= expected.candidate_gates.weight_saturation_maximum,
    guard_nonworsening_invariant: trace.gates.descent_guard_nonworsening === true,
    guard_disjoint_from_window_schedule:
      trace.gates.descent_guard_disjoint_from_window_schedule === true,
    guard_nll_strictly_improves:
      guard.final_nll_millibits < guard.initial_nll_millibits,
    guard_mean_target_rank_must_not_worsen:
      guard.final_evaluation.mean_target_rank_x1000
        <= guard.initial_evaluation.mean_target_rank_x1000,
    guard_top10_hits_must_not_decrease:
      guard.final_evaluation.top10_hits >= guard.initial_evaluation.top10_hits,
  };
  return {
    feature_shift: shift,
    trace: binding(file, bytes),
    window_schedule_rank_hash: training.window_schedule_rank_hash,
    descent_guard_window_rank_hash: guard.window_rank_hash,
    guard_initial_nll_millibits: guard.initial_nll_millibits,
    guard_final_nll_millibits: guard.final_nll_millibits,
    guard_initial_evaluation: guard.initial_evaluation,
    guard_final_evaluation: guard.final_evaluation,
    gates,
    passed: Object.values(gates).every(Boolean),
  };
});

const first = candidates[0];
for (const candidate of candidates.slice(1)) {
  assert(candidate.window_schedule_rank_hash === first.window_schedule_rank_hash,
    "preflight candidates do not share one update schedule");
  assert(candidate.descent_guard_window_rank_hash === first.descent_guard_window_rank_hash,
    "preflight candidates do not share one descent guard");
  assert(candidate.guard_initial_nll_millibits === first.guard_initial_nll_millibits,
    "preflight candidates do not share one source guard NLL");
  assert(JSON.stringify(candidate.guard_initial_evaluation)
    === JSON.stringify(first.guard_initial_evaluation),
  "preflight candidates do not share one source guard evaluation");
}

const passing = candidates.filter(candidate => candidate.passed);
passing.sort((left, right) =>
  left.guard_final_nll_millibits - right.guard_final_nll_millibits
  || left.guard_final_evaluation.mean_target_rank_x1000
    - right.guard_final_evaluation.mean_target_rank_x1000
  || right.guard_final_evaluation.top10_hits - left.guard_final_evaluation.top10_hits
  || right.feature_shift - left.feature_shift);
const selected = passing[0] ?? null;
const result = {
  schema: "nsrl.production_target_margin_trust_region_preflight_selection.v1",
  contract: binding(config.contract, contractBytes),
  candidates,
  selected_feature_shift: selected?.feature_shift ?? null,
  selected_window_schedule_rank_hash: selected?.window_schedule_rank_hash ?? null,
  selected_descent_guard_window_rank_hash:
    selected?.descent_guard_window_rank_hash ?? null,
  selected_guard_initial_nll_millibits: selected?.guard_initial_nll_millibits ?? null,
  selected_guard_initial_evaluation: selected?.guard_initial_evaluation ?? null,
  passing_feature_shifts: passing.map(candidate => candidate.feature_shift),
  selection_order: expected.selection_order,
  preflight_passed: selected !== null,
  public_test_opened: false,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "target-margin trust-region preflight selection does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  selected_feature_shift: result.selected_feature_shift,
  passing_feature_shifts: result.passing_feature_shifts,
  out: config.out,
})}\n`);
if (!result.preflight_passed) process.exitCode = 1;

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, sha256: sha256(bytes)};
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(args) {
  const config = {
    contract:
      "benchmarks/production-model-v1/p10m-target-margin-trust-region-v1-contract.json",
    traces: [],
    out: "benchmarks/production-model-v1/p10m-target-margin-trust-region-v1-preflight.json",
    check: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--contract") config.contract = args[++index] || "";
    else if (args[index] === "--trace") {
      const value = args[++index] || "";
      const separator = value.indexOf(":");
      assert(separator > 0, "--trace must be SHIFT:PATH");
      config.traces.push({
        shift: Number.parseInt(value.slice(0, separator), 10),
        file: value.slice(separator + 1),
      });
    } else if (args[index] === "--out") config.out = args[++index] || "";
    else if (args[index] === "--check") config.check = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  return config;
}
