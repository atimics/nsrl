#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
assert(contract.schema === "nsrl.production_target_margin_contract.v1",
  "target-margin contract schema is invalid");

const expectedShifts = contract.preflight.feature_shifts;
assert(config.traces.length === expectedShifts.length,
  "one preflight trace is required for every frozen feature shift");
const candidates = config.traces.map(({shift, file}) => {
  const bytes = fs.readFileSync(file);
  const trace = JSON.parse(bytes);
  assert(expectedShifts.includes(shift), `unexpected feature shift ${shift}`);
  assert(trace.schema === "nsrl.production_target_margin_train.v1",
    `${file} has the wrong trace schema`);
  const training = trace.training;
  const initial = trace.evaluation.initial;
  const final = trace.evaluation.final;
  assert(training.feature_shift === shift
    && training.context_tokens === contract.preflight.context_tokens
    && training.windows === contract.preflight.windows
    && training.evaluation_windows === contract.preflight.evaluation_windows
    && training.targets_per_window === contract.preflight.targets_per_window
    && training.epochs === contract.preflight.epochs
    && training.batch_windows === contract.preflight.batch_windows
    && training.optimizer_steps === contract.preflight.optimizer_steps
    && training.margin_q8 === contract.preflight.margin_q8,
  `${file} does not match the frozen preflight geometry`);
  const gates = {
    schedule_complete: trace.cursor.schedule_complete === true,
    output_matrix_movement_minimum:
      training.movement_l1 >= contract.preflight.candidate_gates.output_matrix_movement_minimum,
    frozen_parameters_unchanged: trace.gates.frozen_parameters_unchanged === true,
    output_bias_unchanged: trace.gates.output_bias_unchanged === true,
    weight_saturation_maximum:
      trace.health.weight_saturation_count
        <= contract.preflight.candidate_gates.weight_saturation_maximum,
    mean_target_rank_strictly_improves:
      final.mean_target_rank_x1000 < initial.mean_target_rank_x1000,
    mistakes_must_not_increase: final.mistakes <= initial.mistakes,
    margin_satisfied_strictly_increases:
      final.margin_satisfied > initial.margin_satisfied,
  };
  return {
    feature_shift: shift,
    trace: binding(file, bytes),
    initial,
    final,
    gates,
    passed: Object.values(gates).every(Boolean),
  };
});

const passing = candidates.filter(candidate => candidate.passed);
assert(passing.length > 0, "no feature-shift preflight passed the frozen gates");
passing.sort((left, right) =>
  left.final.mean_target_rank_x1000 - right.final.mean_target_rank_x1000
  || left.final.mistakes - right.final.mistakes
  || right.final.margin_satisfied - left.final.margin_satisfied
  || right.feature_shift - left.feature_shift);
const selected = passing[0];
const result = {
  schema: "nsrl.production_target_margin_preflight_selection.v1",
  contract: binding(config.contract, contractBytes),
  candidates,
  selected_feature_shift: selected.feature_shift,
  passing_feature_shifts: passing.map(candidate => candidate.feature_shift),
  selection_order: contract.preflight.selection_order,
  preflight_passed: true,
  hidden_panel_opened: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "target-margin preflight selection does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  selected_feature_shift: result.selected_feature_shift,
  passing_feature_shifts: result.passing_feature_shifts,
  out: config.out,
})}\n`);

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
    contract: "benchmarks/production-model-v1/p10m-target-margin-head-v1-contract.json",
    traces: [],
    out: "benchmarks/production-model-v1/p10m-target-margin-head-v1-preflight.json",
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
