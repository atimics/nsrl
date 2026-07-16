#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const options = parseArgs(process.argv.slice(2));
for (const name of ["input-dir", "model", "tokens", "out"]) {
  if (!options[name]) throw new Error(`--${name} is required`);
}

const selectedRanks = parseIntegerSet(options.ranks ?? "8,16,32", "ranks");
const selectedShifts = parseIntegerSet(options.shifts ?? "0,1,2,3,4", "shifts");
const traces = fs
  .readdirSync(options["input-dir"])
  .filter((name) => name.endsWith(".train.json"))
  .sort()
  .map((name) => {
    const trace = JSON.parse(
      fs.readFileSync(path.join(options["input-dir"], name), "utf8"),
    );
    if (trace.schema !== "nsrl.mini_transformer_low_rank_expert_train.v4") {
      throw new Error(`${name} has unsupported schema ${trace.schema}`);
    }
    return {
      id: name.slice(0, -".train.json".length),
      rank: trace.config.rank,
      learning_rate_shift: trace.config.learning_rate_shift,
      error_feedback: trace.config.error_feedback,
      optimizer_steps: trace.updates.optimizer_steps,
      active_rank: trace.updates.active_rank,
      residual_carry_event_count: trace.updates.residual_carry_event_count,
      final_residual_carry_nonzero_count:
        trace.updates.final_residual_carry_nonzero_count,
      parameter_update: trace.updates.parameter_update,
      functional_update: trace.updates.functional_update,
      initial: trace.initial,
      final: trace.final,
    };
  })
  .filter(
    (trace) =>
      selectedRanks.has(trace.rank) &&
      selectedShifts.has(trace.learning_rate_shift),
  );

if (traces.length === 0) throw new Error("no matrix traces found");
const expectedRuns = selectedRanks.size * selectedShifts.size * 2;
if (traces.length !== expectedRuns) {
  throw new Error(`expected ${expectedRuns} matrix traces, found ${traces.length}`);
}
const functionalClasses = groupByHash(traces, "functional_update");
const parameterClasses = groupByHash(traces, "parameter_update");

const report = {
  schema: "nsrl.integer_reachable_capacity_matrix.v1",
  claim_status: "bounded_observation_not_capacity_proof",
  inputs: {
    model: options.model,
    model_sha256: sha256(options.model),
    tokens: options.tokens,
    tokens_sha256: sha256(options.tokens),
  },
  matrix: traces,
  observed_capacity: {
    runs: traces.length,
    unique_functional_update_hashes: functionalClasses.length,
    unique_parameter_update_hashes: parameterClasses.length,
    zero_functional_updates: traces.filter(
      (trace) => trace.functional_update.nonzero_count === 0,
    ).length,
    functional_equivalence_classes: functionalClasses,
    parameter_equivalence_classes: parameterClasses,
  },
  paired_effects: {
    carry: carryEffects(traces),
    rank: rankEffects(traces),
  },
};

fs.writeFileSync(options.out, `${JSON.stringify(report, null, 2)}\n`);

function groupByHash(rows, field) {
  const classes = new Map();
  for (const row of rows) {
    const hash = row[field].hash;
    const ids = classes.get(hash) ?? [];
    ids.push(row.id);
    classes.set(hash, ids);
  }
  return [...classes.entries()]
    .map(([hash, run_ids]) => ({ hash, run_ids }))
    .sort((left, right) => left.hash.localeCompare(right.hash));
}

function carryEffects(rows) {
  const effects = [];
  for (const withoutCarry of rows.filter((row) => !row.error_feedback)) {
    const withCarry = rows.find(
      (row) =>
        row.error_feedback &&
        row.rank === withoutCarry.rank &&
        row.learning_rate_shift === withoutCarry.learning_rate_shift,
    );
    if (!withCarry) continue;
    effects.push({
      rank: withCarry.rank,
      learning_rate_shift: withCarry.learning_rate_shift,
      distinct_functional_update:
        withCarry.functional_update.hash !== withoutCarry.functional_update.hash,
      functional_nonzero_gain:
        withCarry.functional_update.nonzero_count -
        withoutCarry.functional_update.nonzero_count,
      probability_error_reduction_q15:
        withoutCarry.final.probability_error_q15 -
        withCarry.final.probability_error_q15,
    });
  }
  return effects;
}

function rankEffects(rows) {
  const effects = [];
  for (const shift of [...new Set(rows.map((row) => row.learning_rate_shift))]) {
    for (const errorFeedback of [false, true]) {
      const group = rows
        .filter(
          (row) =>
            row.learning_rate_shift === shift &&
            row.error_feedback === errorFeedback,
        )
        .sort((left, right) => left.rank - right.rank);
      for (let index = 1; index < group.length; index += 1) {
        effects.push({
          learning_rate_shift: shift,
          error_feedback: errorFeedback,
          lower_rank: group[index - 1].rank,
          higher_rank: group[index].rank,
          distinct_functional_update:
            group[index - 1].functional_update.hash !==
            group[index].functional_update.hash,
          active_rank_gain:
            group[index].active_rank - group[index - 1].active_rank,
          probability_error_reduction_q15:
            group[index - 1].final.probability_error_q15 -
            group[index].final.probability_error_q15,
        });
      }
    }
  }
  return effects;
}

function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function parseIntegerSet(value, name) {
  const values = value.split(",").map((part) => Number.parseInt(part, 10));
  if (values.some((item) => !Number.isSafeInteger(item) || item < 0)) {
    throw new Error(`--${name} requires comma-separated nonnegative integers`);
  }
  return new Set(values);
}

function parseArgs(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`invalid argument near ${key ?? "end"}`);
    }
    parsed[key.slice(2)] = value;
  }
  return parsed;
}
