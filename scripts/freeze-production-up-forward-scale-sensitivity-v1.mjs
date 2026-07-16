#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-up-forward-scale-sensitivity";
let outPath = "benchmarks/production-model-v1/p10m-up-forward-scale-sensitivity.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-up-forward-scale-sensitivity-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);

function outcomeFor(gatesPass, safeRows, functionalRows) {
  if (!gatesPass) return "failed_contract";
  if (safeRows.length > 0) return "safe_functional_boundary_found";
  if (functionalRows.length > 0) return "only_saturated_functional_boundary_found";
  return "no_functional_boundary_in_sweep";
}

function nextGateFor(outcome) {
  if (outcome === "safe_functional_boundary_found") {
    return "p10m_up_forward_scale_training_contract";
  }
  if (outcome === "only_saturated_functional_boundary_found") {
    return "p10m_up_forward_scale_safety_review";
  }
  return "p10m_integer_objective_quality_review";
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const shifts = contract.selection_policy.row_order;
  const [sourceDiagnosticContractBytes, sourceDiagnosticBytes, sourceModel,
    candidateModel, tokenizerBytes, devBytes, ...rows] = await Promise.all([
    readFile(contract.source_diagnostic.contract_path),
    readFile(contract.source_diagnostic.checkpoint_path),
    readFile(contract.models.source.path),
    readFile(contract.models.candidate.path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.dev_tokens_path),
    ...shifts.map((shift) => readJson(path.join(runDir, `up-forward-shift-${shift}.json`))),
  ]);
  const sourceDiagnostic = JSON.parse(sourceDiagnosticBytes);
  const frozenOtherShifts = contract.evaluation.all_other_forward_shifts_frozen;
  const rowShapeValid = rows.every((row, index) => {
    const shift = shifts[index];
    return row.schema === "nsrl.production_model_functional_comparison.v1"
      && row.profile === contract.profile
      && row.parameter_count === contract.parameter_count
      && row.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && row.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && row.evaluation.context_tokens === contract.evaluation.context_tokens
      && row.evaluation.windows === contract.evaluation.windows
      && row.forward_shifts.qkv === frozenOtherShifts.qkv
      && row.forward_shifts.o === frozenOtherShifts.o
      && row.forward_shifts.up === shift
      && row.forward_shifts.gate === frozenOtherShifts.gate
      && row.forward_shifts.down === frozenOtherShifts.down
      && row.forward_shifts.output === frozenOtherShifts.output;
  });
  const accountingComplete = rows.every((row) => {
    const quality = row.quality;
    const delta = row.functional_delta;
    return quality.total_millibits_delta
        === quality.candidate_total_millibits - quality.source_total_millibits
      && quality.improved_loss_windows + quality.worsened_loss_windows
        + quality.equal_loss_windows === contract.evaluation.windows
      && delta.feature_changed_windows <= contract.evaluation.windows
      && delta.logits_changed_windows <= contract.evaluation.windows
      && delta.probabilities_changed_windows <= contract.evaluation.windows
      && delta.target_probability_changed_windows <= contract.evaluation.windows
      && (delta.feature_changed_windows > 0) === (delta.feature_delta_l1 > 0)
      && (delta.logits_changed_windows > 0) === (delta.logit_delta_l1 > 0)
      && (delta.probabilities_changed_windows > 0) === (delta.probability_delta_l1 > 0);
  });
  const functionalRows = rows.filter((row) =>
    row.functional_delta.feature_changed_windows > 0);
  const safeRows = functionalRows.filter((row) =>
    row.health.source_residual_saturation_count === 0
      && row.health.candidate_residual_saturation_count === 0);
  const selected = safeRows[0] ?? null;
  const selectedShift = selected?.forward_shifts.up ?? null;
  const expectedSelectedShift = shifts.find((shift, index) => {
    const row = rows[index];
    return row.functional_delta.feature_changed_windows > 0
      && row.health.source_residual_saturation_count === 0
      && row.health.candidate_residual_saturation_count === 0;
  }) ?? null;
  const baseline = rows[0];
  const gates = {
    source_diagnostic_matches:
      sha256(sourceDiagnosticContractBytes)
        === contract.source_diagnostic.contract_sha256
      && sha256(sourceDiagnosticBytes) === contract.source_diagnostic.checkpoint_sha256
      && sourceDiagnostic.diagnostic_eligible === true
      && sourceDiagnostic.classification
        === contract.source_diagnostic.required_classification,
    artifact_and_data_hashes_match:
      sha256(sourceModel) === contract.models.source.sha256
      && sha256(candidateModel) === contract.models.candidate.sha256
      && sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256,
    four_predeclared_rows_complete:
      rows.length === 4
      && sameJson(shifts, [10, 9, 8, 7])
      && rowShapeValid,
    only_common_up_forward_shift_changes:
      rows.every((row) => sameJson(
        {
          qkv: row.forward_shifts.qkv,
          o: row.forward_shifts.o,
          gate: row.forward_shifts.gate,
          down: row.forward_shifts.down,
          output: row.forward_shifts.output,
        },
        frozenOtherShifts,
      )),
    baseline_reproduces_masked_functional_comparison:
      baseline.forward_shifts.up === contract.evaluation.baseline_up_forward_shift
      && baseline.models.source_hash === contract.models.source.base_model_hash
      && baseline.models.candidate_hash === contract.models.candidate.base_model_hash
      && baseline.quality.total_millibits_delta === 0
      && baseline.functional_delta.feature_changed_windows === 0
      && baseline.functional_delta.logits_changed_windows === 0
      && baseline.functional_delta.probabilities_changed_windows === 0,
    aggregate_loss_and_functional_delta_accounting_complete: accountingComplete,
    selection_policy_applied_exactly: selectedShift === expectedSelectedShift,
    test_split_not_accessed: contract.selection_policy.no_test_split_access === true
      && rows.every((row) => row.bindings.token_stream_hash
        === contract.bindings.dev_token_stream_hash),
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false
      && contract.authorization.paid_scale_requires_separate_contract === true,
  };
  const gatesPass = Object.values(gates).every(Boolean);
  const outcome = outcomeFor(gatesPass, safeRows, functionalRows);
  return {
    schema: "nsrl.production_forward_scale_sensitivity_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_diagnostic: contract.source_diagnostic,
    models: contract.models,
    bindings: contract.bindings,
    rows: rows.map((row) => ({
      up_forward_shift: row.forward_shifts.up,
      source_model_hash: row.models.source_hash,
      candidate_model_hash: row.models.candidate_hash,
      quality: row.quality,
      functional_delta: row.functional_delta,
      health: row.health,
      safe_functional_delta: row.functional_delta.feature_changed_windows > 0
        && row.health.source_residual_saturation_count === 0
        && row.health.candidate_residual_saturation_count === 0,
    })),
    selection: {
      policy: contract.selection_policy,
      selected_up_forward_shift: selectedShift,
      selected_row: selected
        ? {
            quality: selected.quality,
            functional_delta: selected.functional_delta,
            health: selected.health,
          }
        : null,
    },
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
const safeRows = (checkpoint.rows ?? []).filter(({ safe_functional_delta }) =>
  safe_functional_delta);
const functionalRows = (checkpoint.rows ?? []).filter(({ functional_delta }) =>
  functional_delta.feature_changed_windows > 0);
const expectedOutcome = outcomeFor(expectedGates, safeRows, functionalRows);
if (checkpoint.schema !== "nsrl.production_forward_scale_sensitivity_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || checkpoint.rows?.length !== 4
  || checkpoint.diagnostic_eligible !== expectedGates
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production up forward-scale sensitivity checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production up forward-scale sensitivity checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_forward_scale_sensitivity_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
    selected_up_forward_shift: checkpoint.selection.selected_up_forward_shift,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    selected_up_forward_shift: checkpoint.selection.selected_up_forward_shift,
  }));
}
