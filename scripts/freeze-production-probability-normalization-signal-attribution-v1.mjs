#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir =
  "data/experiments/production-model-v1/p10m-probability-normalization-signal-attribution";
let outPath =
  "benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const difference = (left, right) => left.filter((value) => !right.includes(value));
const intersection = (left, right) => left.filter((value) => right.includes(value));
const sortedUnique = (values) => [...new Set(values)].sort((left, right) => left - right);

function outcomeFor(gatesPass, candidateGates) {
  if (!gatesPass) return "failed_contract";
  if (!candidateGates.every_exact_target_change_preserved) {
    return "newton_misses_exact_signal";
  }
  if (Object.values(candidateGates).every(Boolean)) {
    return "newton_normalization_attributed_and_ready";
  }
  return "newton_deviates_from_exact_ceiling";
}

function nextGateFor(outcome) {
  if (outcome === "newton_normalization_attributed_and_ready") {
    return "p10m_normalized_wide_gradient_preflight_contract";
  }
  if (outcome === "newton_misses_exact_signal") {
    return "p10m_probability_normalization_algorithm_review";
  }
  if (outcome === "newton_deviates_from_exact_ceiling") {
    return "p10m_probability_normalization_refinement_review";
  }
  return "p10m_probability_normalization_signal_attribution_contract_review";
}

function errorTraceComplete(error) {
  return Number.isSafeInteger(error.probability_changed_values)
    && Number.isSafeInteger(error.probability_error_l1)
    && Number.isSafeInteger(error.probability_error_max)
    && Number.isSafeInteger(error.target_error_windows)
    && Number.isSafeInteger(error.target_error_l1)
    && Number.isSafeInteger(error.target_error_max)
    && error.probability_changed_values >= 0
    && error.probability_error_l1 >= error.probability_error_max
    && error.target_error_l1 >= error.target_error_max;
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, sourceModel, candidateModel,
    tokenizerBytes, devBytes, audit] = await Promise.all([
    readFile(contract.source_normalization_review.contract_path),
    readFile(contract.source_normalization_review.checkpoint_path),
    readFile(contract.models.source.path),
    readFile(contract.models.candidate.path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.dev_tokens_path),
    readJson(path.join(runDir, "audit.json")),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const legacy = audit.methods.find(({ normalization }) =>
    normalization === "legacy_q31_lut");
  const newton = audit.methods.find(({ normalization }) =>
    normalization === contract.selection_policy.candidate_normalization);
  const exact = audit.methods.find(({ normalization }) =>
    normalization === contract.selection_policy.accuracy_ceiling);
  const legacyIndices = legacy.target_changed_window_indices;
  const newtonIndices = newton.target_changed_window_indices;
  const exactIndices = exact.target_changed_window_indices;
  const legacyOnly = difference(legacyIndices, exactIndices);
  const newtonOnly = difference(newtonIndices, exactIndices);
  const exactMissingFromLegacy = difference(exactIndices, legacyIndices);
  const exactMissingFromNewton = difference(exactIndices, newtonIndices);
  const attributionByWindow = new Map(audit.window_attributions.map((row) =>
    [row.window_index, row]));
  const unionIndices = sortedUnique(audit.methods.flatMap((method) =>
    method.target_changed_window_indices));
  const shifts = contract.evaluation.all_other_forward_shifts_frozen;
  const methodsComplete = audit.methods.length === contract.evaluation.methods.length
    && audit.methods.every((method, index) =>
      method.normalization === contract.evaluation.methods[index].id
      && method.reciprocal_fractional_bits
        === contract.evaluation.methods[index].reciprocal_fractional_bits
      && method.target_changed_window_indices.every((windowIndex, itemIndex, values) =>
        Number.isSafeInteger(windowIndex) && windowIndex >= 0
          && windowIndex < contract.evaluation.windows
          && (itemIndex === 0 || values[itemIndex - 1] < windowIndex))
      && errorTraceComplete(method.source_error_vs_exact)
      && errorTraceComplete(method.candidate_error_vs_exact));
  const exactErrorZero = [exact.source_error_vs_exact, exact.candidate_error_vs_exact]
    .every((error) => Object.values(error).every((value) => value === 0));
  const attributionRowsComplete = sameJson(
    audit.window_attributions.map(({ window_index }) => window_index),
    unionIndices,
  ) && audit.window_attributions.every((row) =>
    row.target_probabilities_q23.length === contract.evaluation.methods.length
      && row.target_probabilities_q23.every((pair, index) =>
        pair.normalization === contract.evaluation.methods[index].id
        && pair.delta === pair.candidate - pair.source));
  const sourceRows = sourceCheckpoint.normalization_rows;
  const gates = {
    source_normalization_review_matches:
      sha256(sourceContractBytes)
        === contract.source_normalization_review.contract_sha256
      && sha256(sourceCheckpointBytes)
        === contract.source_normalization_review.checkpoint_sha256
      && sourceCheckpoint.outcome
        === contract.source_normalization_review.required_outcome
      && sourceCheckpoint.next_gate
        === contract.source_normalization_review.required_next_gate
      && sourceCheckpoint.accuracy_effect.best_nondivision_meets_mass_threshold
        === contract.source_normalization_review.newton_mass_threshold_passed,
    artifact_and_data_hashes_match:
      sha256(sourceModel) === contract.models.source.sha256
      && sha256(candidateModel) === contract.models.candidate.sha256
      && sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256,
    only_signal_attribution_observation_changes:
      audit.schema
        === "nsrl.production_probability_normalization_signal_attribution.v1"
      && audit.profile === contract.profile
      && audit.parameter_count === contract.parameter_count
      && audit.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && audit.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && audit.evaluation.context_tokens === contract.evaluation.context_tokens
      && audit.evaluation.windows === contract.evaluation.windows
      && audit.evaluation.probability_fractional_bits
        === contract.evaluation.probability_fractional_bits
      && audit.models.source_hash === contract.models.source.common_forward_model_hash
      && audit.models.candidate_hash === contract.models.candidate.common_forward_model_hash
      && audit.forward_shifts.qkv === shifts.qkv
      && audit.forward_shifts.o === shifts.o
      && audit.forward_shifts.up === contract.models.common_up_forward_shift
      && audit.forward_shifts.gate === shifts.gate
      && audit.forward_shifts.down === shifts.down
      && audit.forward_shifts.output === shifts.output,
    three_predeclared_methods_complete: methodsComplete,
    aggregate_target_change_counts_reproduced:
      legacyIndices.length
        === contract.source_normalization_review.legacy_target_changed_windows
      && newtonIndices.length
        === contract.source_normalization_review.newton_target_changed_windows
      && exactIndices.length
        === contract.source_normalization_review.exact_target_changed_windows
      && legacyIndices.length === sourceRows[0].delta.target_probability_changed_windows
      && newtonIndices.length === sourceRows[2].delta.target_probability_changed_windows
      && exactIndices.length === sourceRows[3].delta.target_probability_changed_windows,
    exact_division_is_zero_error_ceiling: exactErrorZero,
    window_attribution_accounting_complete: attributionRowsComplete,
    zero_forward_saturation:
      audit.health.source_residual_saturation_count === 0
      && audit.health.candidate_residual_saturation_count === 0,
    test_split_not_accessed:
      contract.selection_policy.no_test_split_access === true
      && audit.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const legacyOnlyAttributed = legacyOnly.length > 0 && legacyOnly.every((windowIndex) => {
    const row = attributionByWindow.get(windowIndex);
    const exactPair = row?.target_probabilities_q23.find(({ normalization }) =>
      normalization === "q47_exact_division");
    const legacyPair = row?.target_probabilities_q23.find(({ normalization }) =>
      normalization === "legacy_q31_lut");
    return exactPair?.delta === 0 && legacyPair?.delta !== 0
      && (row.target_weight_q15.changed || row.normalization_sum.changed);
  });
  const candidateGates = {
    every_exact_target_change_preserved: exactMissingFromNewton.length === 0,
    candidate_only_target_changes_within_budget:
      newtonOnly.length
        <= contract.selection_policy.maximum_candidate_only_target_change_windows,
    source_probability_error_within_one_q23_unit:
      newton.source_error_vs_exact.probability_error_max
        <= contract.selection_policy.maximum_candidate_probability_error_q23_units,
    candidate_probability_error_within_one_q23_unit:
      newton.candidate_error_vs_exact.probability_error_max
        <= contract.selection_policy.maximum_candidate_probability_error_q23_units,
    source_target_error_within_one_q23_unit:
      newton.source_error_vs_exact.target_error_max
        <= contract.selection_policy.maximum_candidate_target_error_q23_units,
    candidate_target_error_within_one_q23_unit:
      newton.candidate_error_vs_exact.target_error_max
        <= contract.selection_policy.maximum_candidate_target_error_q23_units,
    legacy_only_target_changes_attributed:
      contract.selection_policy.require_at_least_one_legacy_only_target_change
      && contract.selection_policy
        .require_legacy_only_changes_have_weight_or_denominator_movement
      && legacyOnlyAttributed,
  };
  const gatesPass = Object.values(gates).every(Boolean);
  const candidateEligible = gatesPass && Object.values(candidateGates).every(Boolean);
  const outcome = outcomeFor(gatesPass, candidateGates);
  const legacyOnlyRows = legacyOnly.map((windowIndex) => attributionByWindow.get(windowIndex));
  const newtonOnlyRows = newtonOnly.map((windowIndex) => attributionByWindow.get(windowIndex));
  return {
    schema: "nsrl.production_probability_normalization_signal_attribution_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_normalization_review: contract.source_normalization_review,
    models: contract.models,
    bindings: contract.bindings,
    evaluation: {
      context_tokens: audit.evaluation.context_tokens,
      windows: audit.evaluation.windows,
      probability_fractional_bits: audit.evaluation.probability_fractional_bits,
      logit_changed_windows: audit.logit_signal.changed_windows,
      target_logit_changed_windows: audit.logit_signal.target_changed_windows,
    },
    methods: audit.methods,
    set_attribution: {
      legacy_exact_overlap_indices: intersection(legacyIndices, exactIndices),
      legacy_only_indices: legacyOnly,
      exact_missing_from_legacy_indices: exactMissingFromLegacy,
      newton_exact_overlap_indices: intersection(newtonIndices, exactIndices),
      newton_only_indices: newtonOnly,
      exact_missing_from_newton_indices: exactMissingFromNewton,
    },
    legacy_only_attribution: {
      windows: legacyOnly.length,
      target_logit_changed_windows:
        legacyOnlyRows.filter((row) => row.target_logit_q8.changed).length,
      target_weight_changed_windows:
        legacyOnlyRows.filter((row) => row.target_weight_q15.changed).length,
      normalization_sum_changed_windows:
        legacyOnlyRows.filter((row) => row.normalization_sum.changed).length,
      all_exact_q23_deltas_zero:
        legacyOnlyRows.every((row) => row.target_probabilities_q23
          .find(({ normalization }) => normalization === "q47_exact_division").delta === 0),
      classification:
        "legacy_reciprocal_amplifies_sub_q23_weight_or_denominator_movement",
    },
    newton_only_attribution: {
      windows: newtonOnly.length,
      target_logit_changed_windows:
        newtonOnlyRows.filter((row) => row.target_logit_q8.changed).length,
      target_weight_changed_windows:
        newtonOnlyRows.filter((row) => row.target_weight_q15.changed).length,
      normalization_sum_changed_windows:
        newtonOnlyRows.filter((row) => row.normalization_sum.changed).length,
      all_exact_q23_deltas_zero:
        newtonOnlyRows.every((row) => row.target_probabilities_q23
          .find(({ normalization }) => normalization === "q47_exact_division").delta === 0),
    },
    window_attributions: audit.window_attributions,
    health: audit.health,
    gates,
    candidate_gates: candidateGates,
    diagnostic_eligible: gatesPass,
    candidate_eligible: candidateEligible,
    selection: {
      policy: contract.selection_policy,
      selected_normalization: candidateEligible
        ? contract.selection_policy.candidate_normalization : null,
    },
    outcome,
    promotion_eligible: false,
    paid_scale_authorized: false,
    next_gate: nextGateFor(outcome),
    known_non_claims: contract.known_non_claims,
  };
}

let checkpoint;
try {
  checkpoint = await buildCheckpoint();
} catch (error) {
  if (!checkOnly || error.code !== "ENOENT") throw error;
  checkpoint = await readJson(outPath);
}
const currentContractBytes = await readFile(contractPath);
const expectedGates = Object.values(checkpoint.gates ?? {}).every(Boolean);
const expectedCandidateEligible = expectedGates
  && Object.values(checkpoint.candidate_gates ?? {}).every(Boolean);
const expectedOutcome = outcomeFor(expectedGates, checkpoint.candidate_gates ?? {});
if (checkpoint.schema
    !== "nsrl.production_probability_normalization_signal_attribution_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || !sameJson(checkpoint.methods?.map(({ normalization }) => normalization), [
    "legacy_q31_lut", "q47_newton1", "q47_exact_division",
  ])
  || checkpoint.diagnostic_eligible !== expectedGates
  || checkpoint.candidate_eligible !== expectedCandidateEligible
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.selection?.selected_normalization
    !== (expectedCandidateEligible ? "q47_newton1" : null)
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production normalization signal-attribution checkpoint is invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production normalization signal-attribution checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_probability_normalization_signal_attribution_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
    selected_normalization: checkpoint.selection.selected_normalization,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    selected_normalization: checkpoint.selection.selected_normalization,
  }));
}
