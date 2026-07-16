#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

let runDir = "data/experiments/production-model-v1/p10m-target-probability-resolution";
let outPath = "benchmarks/production-model-v1/p10m-target-probability-resolution.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--run-dir") runDir = process.argv[++index];
  else if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const contractPath =
  "benchmarks/production-model-v1/p10m-target-probability-resolution-contract.json";
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);

function outcomeFor(gatesPass, baseline, selected) {
  if (!gatesPass) return "failed_contract";
  if (baseline.delta.target_probability_changed_windows > 0) {
    return "q15_already_target_sensitive";
  }
  if (selected) return "wider_precision_recovers_target_signal";
  return "target_signal_absent_at_q31";
}

function nextGateFor(outcome) {
  if (outcome === "wider_precision_recovers_target_signal") {
    return "p10m_wide_probability_gradient_preflight_contract";
  }
  if (outcome === "target_signal_absent_at_q31") {
    return "p10m_output_logit_resolution_review";
  }
  return "p10m_integer_objective_quality_review";
}

async function buildCheckpoint() {
  const contractBytes = await readFile(contractPath);
  const contract = JSON.parse(contractBytes);
  const [sourceContractBytes, sourceCheckpointBytes, sourceModel, candidateModel,
    tokenizerBytes, devBytes, audit] = await Promise.all([
    readFile(contract.source_training.contract_path),
    readFile(contract.source_training.checkpoint_path),
    readFile(contract.models.source.path),
    readFile(contract.models.candidate.path),
    readFile(contract.bindings.tokenizer_path),
    readFile(contract.bindings.dev_tokens_path),
    readJson(path.join(runDir, "audit.json")),
  ]);
  const sourceCheckpoint = JSON.parse(sourceCheckpointBytes);
  const bits = contract.evaluation.probability_fractional_bits;
  const baseline = audit.precisions[0];
  const selected = audit.precisions.find((row, index) => index > 0
    && row.delta.target_probability_changed_windows
      > baseline.delta.target_probability_changed_windows) ?? null;
  const selectedBits = selected?.fractional_bits ?? null;
  const expectedSelectedBits = bits.find((fractionalBits, index) => index > 0
    && audit.precisions[index].delta.target_probability_changed_windows
      > baseline.delta.target_probability_changed_windows) ?? null;
  const otherShifts = contract.evaluation.all_other_forward_shifts_frozen;
  const rowsComplete = audit.precisions.length === bits.length
    && audit.precisions.every((row, index) => {
      const fractionalBits = bits[index];
      return row.fractional_bits === fractionalBits
        && row.uniform_probability_floor
          === Math.floor((2 ** fractionalBits) / contract.evaluation.vocab_size)
        && row.source_target.min <= row.source_target.max
        && row.candidate_target.min <= row.candidate_target.max
        && row.source_target.zero_windows <= contract.evaluation.windows
        && row.candidate_target.zero_windows <= contract.evaluation.windows;
    });
  const accountingComplete = audit.precisions.every((row) =>
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
    source_training_matches:
      sha256(sourceContractBytes) === contract.source_training.contract_sha256
      && sha256(sourceCheckpointBytes) === contract.source_training.checkpoint_sha256
      && sourceCheckpoint.outcome === contract.source_training.required_outcome
      && sourceCheckpoint.next_gate === contract.source_training.required_next_gate,
    artifact_and_data_hashes_match:
      sha256(sourceModel) === contract.models.source.sha256
      && sha256(candidateModel) === contract.models.candidate.sha256
      && sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256,
    five_predeclared_precision_rows_complete:
      sameJson(bits, [15, 19, 23, 27, 31]) && rowsComplete,
    only_probability_fractional_precision_changes:
      audit.schema === "nsrl.production_probability_resolution_audit.v1"
      && audit.profile === contract.profile
      && audit.parameter_count === contract.parameter_count
      && audit.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && audit.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && audit.evaluation.context_tokens === contract.evaluation.context_tokens
      && audit.evaluation.windows === contract.evaluation.windows
      && audit.models.source_hash === contract.models.source.common_forward_model_hash
      && audit.models.candidate_hash === contract.models.candidate.common_forward_model_hash
      && audit.forward_shifts.qkv === otherShifts.qkv
      && audit.forward_shifts.o === otherShifts.o
      && audit.forward_shifts.up === contract.models.common_up_forward_shift
      && audit.forward_shifts.gate === otherShifts.gate
      && audit.forward_shifts.down === otherShifts.down
      && audit.forward_shifts.output === otherShifts.output,
    q15_requantization_matches_production_exactly:
      audit.compatibility.q15_requantization_exact === true,
    probability_mass_and_loss_accounting_complete: accountingComplete,
    selection_policy_applied_exactly: selectedBits === expectedSelectedBits,
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
  const outcome = outcomeFor(gatesPass, baseline, selected);
  return {
    schema: "nsrl.production_target_probability_resolution_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    source_training: contract.source_training,
    models: contract.models,
    bindings: contract.bindings,
    evaluation: {
      context_tokens: audit.evaluation.context_tokens,
      windows: audit.evaluation.windows,
      vocab_size: contract.evaluation.vocab_size,
      logit_changed_windows: audit.logit_signal.changed_windows,
      target_logit_changed_windows: audit.logit_signal.target_changed_windows,
    },
    precision_rows: audit.precisions,
    selection: {
      policy: contract.selection_policy,
      baseline_fractional_bits: baseline.fractional_bits,
      baseline_uniform_probability_floor: baseline.uniform_probability_floor,
      baseline_target_probability_changed_windows:
        baseline.delta.target_probability_changed_windows,
      selected_fractional_bits: selectedBits,
      selected_uniform_probability_floor: selected?.uniform_probability_floor ?? null,
      selected_target_probability_changed_windows:
        selected?.delta.target_probability_changed_windows ?? null,
      selected_resolution_gain_vs_q15: selected
        ? selected.uniform_probability_floor / baseline.uniform_probability_floor
        : null,
    },
    compatibility: audit.compatibility,
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
const baseline = checkpoint.precision_rows?.[0];
const selected = checkpoint.precision_rows?.find((row, index) => index > 0
  && row.delta.target_probability_changed_windows
    > baseline.delta.target_probability_changed_windows) ?? null;
const expectedOutcome = outcomeFor(expectedGates, baseline, selected);
if (checkpoint.schema !== "nsrl.production_target_probability_resolution_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract?.path !== contractPath
  || checkpoint.contract?.sha256 !== sha256(currentContractBytes)
  || !sameJson(checkpoint.precision_rows?.map(({ fractional_bits }) => fractional_bits),
    [15, 19, 23, 27, 31])
  || checkpoint.diagnostic_eligible !== expectedGates
  || checkpoint.outcome !== expectedOutcome
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false
  || checkpoint.next_gate !== nextGateFor(expectedOutcome)) {
  throw new Error("production target-probability resolution checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production target-probability resolution checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_target_probability_resolution_check.v1",
    ok: true,
    outcome: checkpoint.outcome,
    selected_fractional_bits: checkpoint.selection.selected_fractional_bits,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    outcome: checkpoint.outcome,
    selected_fractional_bits: checkpoint.selection.selected_fractional_bits,
  }));
}
