#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir =
  "data/experiments/production-model-v1/p10m-probability-normalization-accuracy";
let outPath =
  "benchmarks/production-model-v1/p10m-probability-normalization-accuracy.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-probability-normalization-accuracy-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);

function outcomeFor(gatesPass, selected) {
  if (!gatesPass) return "failed_contract";
  if (!selected) return "normalization_accuracy_not_recovered";
  if (selected.normalization === "q47_exact_division") {
    return "only_exact_division_meets_accuracy";
  }
  return "normalization_candidate_found";
}

function nextGateFor(outcome) {
  if (outcome === "normalization_candidate_found") {
    return "p10m_normalized_wide_gradient_preflight_contract";
  }
  if (outcome === "only_exact_division_meets_accuracy") {
    return "p10m_exact_normalization_implementation_review";
  }
  if (outcome === "normalization_accuracy_not_recovered") {
    return "p10m_probability_normalization_signal_attribution_review";
  }
  return "p10m_probability_normalization_contract_review";
}

function rowEligible(row, contract) {
  const scale = 2 ** contract.evaluation.probability_fractional_bits;
  const maximumPpm = contract.selection_policy.maximum_mass_error_ppm;
  const targetMinimum =
    contract.selection_policy.minimum_target_probability_changed_windows;
  return row.normalization !== contract.selection_policy.baseline_normalization
    && row.mass.source_error_max * 1_000_000 <= scale * maximumPpm
    && row.mass.candidate_error_max * 1_000_000 <= scale * maximumPpm
    && row.delta.target_probability_changed_windows >= targetMinimum;
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, referenceContractBytes,
    referenceCheckpointBytes, sourceModel, candidateModel, tokenizerBytes, devBytes,
    audit] = await Promise.all([
    readFile(contract.source_preflight.contract_path),
    readFile(contract.source_preflight.checkpoint_path),
    readFile(contract.reference_probability_resolution.contract_path),
    readFile(contract.reference_probability_resolution.checkpoint_path),
    readFile(contract.models.source.path),
    readFile(contract.models.candidate.path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.dev_tokens_path),
    readJson(path.join(runDir, "audit.json")),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const referenceCheckpoint = JSON.parse(referenceCheckpointBytes);
  const methodContracts = contract.evaluation.normalizations;
  const legacy = audit.normalizations[0];
  const reference = referenceCheckpoint.precision_rows.find(({ fractional_bits }) =>
    fractional_bits === contract.reference_probability_resolution.fractional_bits);
  const { normalization: _legacyName, reciprocal_fractional_bits: _legacyBits,
    ...legacyMetrics } = legacy;
  const { fractional_bits: _referenceBits, ...referenceMetrics } = reference;
  const eligibleRows = audit.normalizations.filter((row) => rowEligible(row, contract));
  const selected = eligibleRows[0] ?? null;
  const shifts = contract.evaluation.all_other_forward_shifts_frozen;
  const rowsComplete = audit.normalizations.length === methodContracts.length
    && audit.normalizations.every((row, index) =>
      row.normalization === methodContracts[index].id
      && row.reciprocal_fractional_bits
        === methodContracts[index].reciprocal_fractional_bits
      && row.uniform_probability_floor
        === Math.floor((2 ** contract.evaluation.probability_fractional_bits)
          / contract.evaluation.vocab_size));
  const accountingComplete = audit.normalizations.every((row) =>
    row.quality.total_microbits_delta
      === row.quality.candidate_total_microbits - row.quality.source_total_microbits
    && row.quality.improved_loss_windows + row.quality.worsened_loss_windows
      + row.quality.equal_loss_windows === contract.evaluation.windows
    && row.delta.probability_changed_windows <= contract.evaluation.windows
    && row.delta.target_probability_changed_windows <= contract.evaluation.windows
    && (row.delta.probability_changed_windows > 0)
      === (row.delta.probability_delta_l1 > 0)
    && (row.delta.target_probability_changed_windows > 0)
      === (row.delta.target_probability_delta_l1 > 0)
    && row.mass.source_error_max <= row.mass.source_error_l1
    && row.mass.candidate_error_max <= row.mass.candidate_error_l1);
  const gates = {
    source_preflight_matches:
      sha256(sourceContractBytes) === contract.source_preflight.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_preflight.checkpoint_sha256
      && sourceCheckpoint.outcome === contract.source_preflight.required_outcome
      && sourceCheckpoint.next_gate === contract.source_preflight.required_next_gate,
    reference_probability_resolution_matches:
      sha256(referenceContractBytes)
        === contract.reference_probability_resolution.contract_sha256
      && sha256(referenceCheckpointBytes)
        === contract.reference_probability_resolution.checkpoint_sha256
      && reference.delta.target_probability_changed_windows
        === contract.reference_probability_resolution.target_probability_changed_windows,
    artifact_and_data_hashes_match:
      sha256(sourceModel) === contract.models.source.sha256
      && sha256(candidateModel) === contract.models.candidate.sha256
      && sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256,
    four_predeclared_normalization_rows_complete:
      sameJson(methodContracts.map(({ id }) => id), [
        "legacy_q31_lut", "q47_lut", "q47_newton1", "q47_exact_division",
      ]) && rowsComplete,
    only_probability_normalization_changes:
      audit.schema === "nsrl.production_probability_normalization_audit.v1"
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
    legacy_q23_metrics_reproduced_exactly: sameJson(legacyMetrics, referenceMetrics),
    probability_mass_and_loss_accounting_complete: accountingComplete,
    selection_policy_applied_exactly:
      selected === (audit.normalizations.find((row) => rowEligible(row, contract)) ?? null),
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
  const gatesPass = Object.values(gates).every(Boolean);
  const outcome = outcomeFor(gatesPass, selected);
  const scale = 2 ** contract.evaluation.probability_fractional_bits;
  const newton = audit.normalizations.find(({ normalization }) =>
    normalization === "q47_newton1");
  const exact = audit.normalizations.find(({ normalization }) =>
    normalization === "q47_exact_division");
  return {
    schema: "nsrl.production_probability_normalization_accuracy_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_preflight: contract.source_preflight,
    reference_probability_resolution: contract.reference_probability_resolution,
    models: contract.models,
    bindings: contract.bindings,
    evaluation: {
      context_tokens: audit.evaluation.context_tokens,
      windows: audit.evaluation.windows,
      probability_fractional_bits: audit.evaluation.probability_fractional_bits,
      probability_scale: scale,
      logit_changed_windows: audit.logit_signal.changed_windows,
      target_logit_changed_windows: audit.logit_signal.target_changed_windows,
    },
    normalization_rows: audit.normalizations.map((row) => ({
      ...row,
      mass: {
        ...row.mass,
        source_error_max_ppm: Math.round(row.mass.source_error_max * 1_000_000 / scale),
        candidate_error_max_ppm:
          Math.round(row.mass.candidate_error_max * 1_000_000 / scale),
      },
      accuracy_eligible: rowEligible(row, contract),
    })),
    accuracy_effect: {
      best_nondivision_normalization: newton.normalization,
      best_nondivision_meets_mass_threshold:
        newton.mass.source_error_max * 1_000_000
          <= scale * contract.selection_policy.maximum_mass_error_ppm
        && newton.mass.candidate_error_max * 1_000_000
          <= scale * contract.selection_policy.maximum_mass_error_ppm,
      source_max_mass_error_reduction_vs_legacy:
        legacy.mass.source_error_max / newton.mass.source_error_max,
      candidate_max_mass_error_reduction_vs_legacy:
        legacy.mass.candidate_error_max / newton.mass.candidate_error_max,
      newton_target_probability_changed_windows:
        newton.delta.target_probability_changed_windows,
      exact_division_target_probability_changed_windows:
        exact.delta.target_probability_changed_windows,
      legacy_target_probability_changed_windows:
        legacy.delta.target_probability_changed_windows,
      newton_target_change_excess_vs_exact:
        newton.delta.target_probability_changed_windows
          - exact.delta.target_probability_changed_windows,
      legacy_target_change_excess_vs_exact:
        legacy.delta.target_probability_changed_windows
          - exact.delta.target_probability_changed_windows,
      classification:
        "mass_accuracy_recovered_but_legacy_target_signal_requires_attribution",
    },
    selection: {
      policy: contract.selection_policy,
      selected_normalization: selected?.normalization ?? null,
      selected_reciprocal_fractional_bits: selected?.reciprocal_fractional_bits ?? null,
      selected_source_mass_error_max: selected?.mass.source_error_max ?? null,
      selected_candidate_mass_error_max: selected?.mass.candidate_error_max ?? null,
      selected_target_probability_changed_windows:
        selected?.delta.target_probability_changed_windows ?? null,
    },
    health: audit.health,
    gates,
    diagnostic_eligible: gatesPass,
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
const selected = checkpoint.normalization_rows?.find(({ normalization, accuracy_eligible }) =>
  normalization !== "legacy_q31_lut" && accuracy_eligible) ?? null;
const expectedOutcome = outcomeFor(expectedGates, selected);
if (checkpoint.schema
    !== "nsrl.production_probability_normalization_accuracy_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || !sameJson(checkpoint.normalization_rows?.map(({ normalization }) => normalization), [
    "legacy_q31_lut", "q47_lut", "q47_newton1", "q47_exact_division",
  ])
  || checkpoint.diagnostic_eligible !== expectedGates
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.selection?.selected_normalization !== (selected?.normalization ?? null)
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production probability-normalization checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production probability-normalization checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_probability_normalization_accuracy_check.v1",
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
