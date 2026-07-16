#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const args = parseArgs(process.argv.slice(2));
if (!args["input-dir"] || !args.out) throw new Error("--input-dir and --out are required");
const contractPath = "benchmarks/integer-reachable-capacity-v1/longitudinal-contract.json";
const contractBytes = fs.readFileSync(contractPath);
const contract = JSON.parse(contractBytes);
const earlyBytes = fs.readFileSync(contract.source_matrix.path);
if (sha256(earlyBytes) !== contract.source_matrix.sha256) {
  throw new Error("source reachable-capacity matrix hash mismatch");
}
for (const input of Object.values(contract.inputs)) {
  if (sha256(fs.readFileSync(input.path)) !== input.sha256) {
    throw new Error(`input hash mismatch: ${input.path}`);
  }
}
const early = JSON.parse(earlyBytes);
const baseline = JSON.parse(fs.readFileSync(
  path.join(args["input-dir"], "zero-heldout.eval.json"), "utf8",
));
if (baseline.schema !== "nsrl.mini_transformer_low_rank_expert_eval.v2"
  || baseline.metrics.windows !== contract.matrix.heldout_windows) {
  throw new Error("zero-expert heldout evaluation does not match the contract");
}

const rows = early.matrix.map((earlyRow) => {
  const train = JSON.parse(fs.readFileSync(
    path.join(args["input-dir"], `${earlyRow.id}.train.json`), "utf8",
  ));
  const evaluation = JSON.parse(fs.readFileSync(
    path.join(args["input-dir"], `${earlyRow.id}.eval.json`), "utf8",
  ));
  if (train.schema !== "nsrl.mini_transformer_low_rank_expert_train.v4"
    || evaluation.schema !== "nsrl.mini_transformer_low_rank_expert_eval.v2"
    || train.config.rank !== earlyRow.rank
    || train.config.learning_rate_shift !== earlyRow.learning_rate_shift
    || train.config.error_feedback !== earlyRow.error_feedback
    || train.updates.optimizer_steps !== contract.matrix.long_optimizer_steps
    || train.initial.windows !== contract.matrix.long_train_windows
    || evaluation.metrics.windows !== contract.matrix.heldout_windows) {
    throw new Error(`${earlyRow.id} does not match the longitudinal contract`);
  }
  const gain = baseline.metrics.probability_error_q15
    - evaluation.metrics.probability_error_q15;
  const mistakeGain = baseline.metrics.mistakes - evaluation.metrics.mistakes;
  return {
    id: earlyRow.id,
    rank: earlyRow.rank,
    learning_rate_shift: earlyRow.learning_rate_shift,
    error_feedback: earlyRow.error_feedback,
    early: {
      optimizer_steps: earlyRow.optimizer_steps,
      active_rank: earlyRow.active_rank,
      functional_update: earlyRow.functional_update,
      parameter_update: earlyRow.parameter_update,
      reachable: earlyRow.functional_update.nonzero_count > 0,
    },
    long_run: {
      optimizer_steps: train.updates.optimizer_steps,
      active_rank: train.updates.active_rank,
      functional_update: train.updates.functional_update,
      parameter_update: train.updates.parameter_update,
      weight_saturation_count: train.updates.weight_saturation_count,
      hidden_saturation_count: train.updates.hidden_saturation_count,
    },
    heldout: {
      mistakes: evaluation.metrics.mistakes,
      probability_error_q15: evaluation.metrics.probability_error_q15,
      probability_error_gain_q15: gain,
      mistake_gain: mistakeGain,
      improved: gain > 0,
    },
  };
});
if (rows.length !== contract.matrix.cells) {
  throw new Error(`expected ${contract.matrix.cells} longitudinal rows, found ${rows.length}`);
}

const classification = binaryClassification(rows);
const functionalL1 = rows.map((row) => row.early.functional_update.delta_l1);
const functionalNonzero = rows.map((row) => row.early.functional_update.nonzero_count);
const activeRank = rows.map((row) => row.early.active_rank);
const gains = rows.map((row) => row.heldout.probability_error_gain_q15);
const correlations = {
  functional_delta_l1_spearman: spearman(functionalL1, gains),
  functional_nonzero_count_spearman: spearman(functionalNonzero, gains),
  active_rank_spearman: spearman(activeRank, gains),
  reachable_point_biserial: pearson(
    rows.map((row) => Number(row.early.reachable)),
    gains,
  ),
};
const permutation = permutationTest(
  functionalL1,
  gains,
  contract.analysis.permutation_test.permutations,
  contract.analysis.permutation_test.seed,
);
const reachableRows = rows.filter((row) => row.early.reachable);
const noOpRows = rows.filter((row) => !row.early.reachable);
const groupComparison = {
  early_reachable_cells: reachableRows.length,
  early_noop_cells: noOpRows.length,
  early_reachable_mean_gain_q15: mean(reachableRows.map((row) => row.heldout.probability_error_gain_q15)),
  early_noop_mean_gain_q15: mean(noOpRows.map((row) => row.heldout.probability_error_gain_q15)),
  mean_gain_difference_q15: mean(reachableRows.map((row) => row.heldout.probability_error_gain_q15))
    - mean(noOpRows.map((row) => row.heldout.probability_error_gain_q15)),
};
const saturatedRows = rows.filter((row) => row.long_run.weight_saturation_count > 0
  || row.long_run.hidden_saturation_count > 0);
const saturation = {
  cells_with_any_saturation: saturatedRows.length,
  cells_without_saturation: rows.length - saturatedRows.length,
  early_reachable_cells_with_any_saturation:
    saturatedRows.filter((row) => row.early.reachable).length,
  early_reachable_cells: reachableRows.length,
  total_weight_saturations: rows.reduce(
    (sum, row) => sum + row.long_run.weight_saturation_count, 0,
  ),
  total_hidden_saturations: rows.reduce(
    (sum, row) => sum + row.long_run.hidden_saturation_count, 0,
  ),
};
const equivalenceClasses = [...Map.groupBy(rows, (row) => row.early.functional_update.hash)]
  .map(([hash, members]) => ({
    hash,
    early_reachable: members[0].early.reachable,
    run_ids: members.map((row) => row.id),
    distinct_long_functional_hashes:
      new Set(members.map((row) => row.long_run.functional_update.hash)).size,
    heldout_gain_min_q15: Math.min(...members.map((row) => row.heldout.probability_error_gain_q15)),
    heldout_gain_max_q15: Math.max(...members.map((row) => row.heldout.probability_error_gain_q15)),
  }))
  .sort((left, right) => left.hash.localeCompare(right.hash));
const predictionSupported = classification.matthews_correlation > 0
  && correlations.functional_delta_l1_spearman > 0
  && permutation.p_value_one_sided_greater <= 0.05
  && groupComparison.mean_gain_difference_q15 > 0;

const report = {
  schema: "nsrl.integer_reachable_capacity_longitudinal.v1",
  contract: { path: contractPath, sha256: sha256(contractBytes) },
  source_matrix: contract.source_matrix,
  inputs: contract.inputs,
  baseline: baseline.metrics,
  matrix: rows,
  analysis: {
    binary_prediction: classification,
    correlations,
    permutation_test: permutation,
    group_comparison: groupComparison,
    saturation,
    early_functional_equivalence_classes: equivalenceClasses,
  },
  conclusion: {
    prediction_supported: predictionSupported,
    claim_status: predictionSupported
      ? "supported_in_bounded_longitudinal_matrix"
      : "not_supported_in_bounded_longitudinal_matrix",
    interpretation: predictionSupported
      ? "early reachable functional movement was positively associated with later disjoint heldout gain"
      : "early functional distinctness alone did not reliably predict later disjoint heldout gain",
    safety_status: saturation.cells_with_any_saturation === 0
      ? "all_long_runs_saturation_free"
      : "association_supported_but_long_run_saturation_requires_followup",
    known_non_claims: contract.known_non_claims,
  },
};
fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
fs.writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({
  claim_status: report.conclusion.claim_status,
  mcc: classification.matthews_correlation,
  spearman: correlations.functional_delta_l1_spearman,
  permutation_p: permutation.p_value_one_sided_greater,
  mean_gain_difference_q15: groupComparison.mean_gain_difference_q15,
}));

function binaryClassification(values) {
  const counts = { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 };
  for (const row of values) {
    if (row.early.reachable && row.heldout.improved) counts.true_positive += 1;
    else if (row.early.reachable) counts.false_positive += 1;
    else if (row.heldout.improved) counts.false_negative += 1;
    else counts.true_negative += 1;
  }
  const { true_positive: tp, false_positive: fp, true_negative: tn, false_negative: fn } = counts;
  const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return {
    ...counts,
    precision: ratio(tp, tp + fp),
    recall: ratio(tp, tp + fn),
    specificity: ratio(tn, tn + fp),
    accuracy: ratio(tp + tn, values.length),
    matthews_correlation: denominator === 0 ? 0 : ((tp * tn) - (fp * fn)) / denominator,
  };
}

function permutationTest(xs, ys, permutations, seed) {
  const observed = spearman(xs, ys);
  const shuffled = [...ys];
  let state = seed >>> 0;
  let atLeastObserved = 0;
  for (let iteration = 0; iteration < permutations; iteration += 1) {
    for (let index = shuffled.length - 1; index > 0; index -= 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      const target = state % (index + 1);
      [shuffled[index], shuffled[target]] = [shuffled[target], shuffled[index]];
    }
    if (spearman(xs, shuffled) >= observed) atLeastObserved += 1;
  }
  return {
    permutations,
    seed,
    observed_spearman: observed,
    p_value_one_sided_greater: (atLeastObserved + 1) / (permutations + 1),
  };
}

function spearman(xs, ys) {
  return pearson(ranks(xs), ranks(ys));
}

function ranks(values) {
  const sorted = values.map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value || left.index - right.index);
  const output = Array(values.length);
  for (let start = 0; start < sorted.length;) {
    let end = start + 1;
    while (end < sorted.length && sorted[end].value === sorted[start].value) end += 1;
    const rank = (start + end - 1) / 2 + 1;
    for (let index = start; index < end; index += 1) output[sorted[index].index] = rank;
    start = end;
  }
  return output;
}

function pearson(xs, ys) {
  const xMean = mean(xs);
  const yMean = mean(ys);
  let numerator = 0;
  let xSquares = 0;
  let ySquares = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const x = xs[index] - xMean;
    const y = ys[index] - yMean;
    numerator += x * y;
    xSquares += x * x;
    ySquares += y * y;
  }
  return xSquares === 0 || ySquares === 0 ? 0 : numerator / Math.sqrt(xSquares * ySquares);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function ratio(numerator, denominator) {
  return denominator === 0 ? 0 : numerator / denominator;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index += 2) {
    if (!values[index]?.startsWith("--") || values[index + 1] === undefined) {
      throw new Error(`invalid argument near ${values[index] ?? "end"}`);
    }
    parsed[values[index].slice(2)] = values[index + 1];
  }
  return parsed;
}
