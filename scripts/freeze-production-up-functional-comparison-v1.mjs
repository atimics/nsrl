#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

const contractPath = "benchmarks/production-model-v1/p10m-up-functional-comparison-contract.json";
const sourceTracePath = "data/experiments/production-model-v1/p10m-up-useful-update/train-3.json";
const candidateTracePath =
  "data/experiments/production-model-v1/p10m-up-shift22-breakthrough/train-3.json";
const comparisonPath =
  "data/experiments/production-model-v1/p10m-up-shift22-breakthrough/functional-comparison.json";
let outPath = "benchmarks/production-model-v1/p10m-up-functional-comparison.json";
let checkOnly = false;
for (let index = 2; index < process.argv.length; index += 1) {
  if (process.argv[index] === "--out") outPath = process.argv[++index];
  else if (process.argv[index] === "--check") checkOnly = true;
  else throw new Error(`unknown argument: ${process.argv[index]}`);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));

function classificationsFor(comparison) {
  const delta = comparison.functional_delta;
  const totalDelta = comparison.quality.total_millibits_delta;
  return [
    ["weight_updates_masked_before_final_features", delta.feature_changed_windows === 0],
    ["feature_changes_masked_by_output_projection",
      delta.feature_changed_windows > 0 && delta.logits_changed_windows === 0],
    ["logit_changes_masked_by_softmax_quantization",
      delta.logits_changed_windows > 0 && delta.probabilities_changed_windows === 0],
    ["target_probability_effects_cancel",
      delta.target_probability_changed_windows > 0 && totalDelta === 0],
    ["candidate_functionally_worse", totalDelta > 0],
    ["candidate_functionally_better", totalDelta < 0],
  ].filter(([, selected]) => selected).map(([id]) => id);
}

async function buildCheckpoint() {
  const [contractBytes, sourceModel, candidateModel, tokenizerBytes, devBytes,
    sourceTrace, candidateTrace, comparison] = await Promise.all([
    readFile(contractPath),
    readFile((await readJson(contractPath)).source_model.path),
    readFile((await readJson(contractPath)).candidate_model.path),
    readFile((await readJson(contractPath)).bindings.tokenizer_path),
    readFile((await readJson(contractPath)).bindings.dev_tokens_path),
    readJson(sourceTracePath),
    readJson(candidateTracePath),
    readJson(comparisonPath),
  ]);
  const contract = JSON.parse(contractBytes);
  const sourceShifts = sourceTrace.training.learning_rate_shifts;
  const candidateShifts = candidateTrace.training.learning_rate_shifts;
  const onlyUpDiffers = Object.keys(sourceShifts).every((group) =>
    candidateShifts[group] === (group === "up"
      ? contract.candidate_model.up_shift
      : sourceShifts[group]));
  const selectedClassifications = classificationsFor(comparison);
  const delta = comparison.functional_delta;
  const quality = comparison.quality;
  const gates = {
    artifact_and_data_hashes_match:
      sha256(sourceModel) === contract.source_model.sha256
      && sha256(candidateModel) === contract.candidate_model.sha256
      && sha256(tokenizerBytes) === contract.bindings.tokenizer_sha256
      && sha256(devBytes) === contract.bindings.dev_tokens_sha256,
    model_schedule_is_matched_except_up_shift:
      sourceTrace.training.optimizer_steps === candidateTrace.training.optimizer_steps
      && sourceTrace.training.total_optimizer_step === contract.source_model.optimizer_steps
      && candidateTrace.training.total_optimizer_step === contract.candidate_model.optimizer_steps
      && sourceTrace.cursor.next_window === contract.source_model.train_windows
      && candidateTrace.cursor.next_window === contract.candidate_model.train_windows
      && sourceTrace.cursor.next_epoch === 0
      && candidateTrace.cursor.next_epoch === 0
      && sourceTrace.hashes.final_model === contract.source_model.model_hash
      && candidateTrace.hashes.final_model === contract.candidate_model.model_hash
      && sourceShifts.up === contract.source_model.up_shift
      && onlyUpDiffers,
    all_256_windows_compared:
      comparison.schema === "nsrl.production_model_functional_comparison.v1"
      && comparison.profile === contract.profile
      && comparison.parameter_count === contract.parameter_count
      && comparison.evaluation.context_tokens === contract.evaluation.context_tokens
      && comparison.evaluation.windows === contract.evaluation.windows
      && comparison.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
      && comparison.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
      && comparison.models.source_hash === contract.source_model.model_hash
      && comparison.models.candidate_hash === contract.candidate_model.model_hash,
    source_and_candidate_aggregate_losses_reproduced:
      quality.source_total_millibits === contract.source_model.dev_total_millibits
      && quality.candidate_total_millibits === contract.candidate_model.dev_total_millibits
      && quality.total_millibits_delta
        === quality.candidate_total_millibits - quality.source_total_millibits,
    functional_delta_accounting_complete:
      quality.improved_loss_windows + quality.worsened_loss_windows
        + quality.equal_loss_windows === contract.evaluation.windows
      && delta.feature_changed_windows <= contract.evaluation.windows
      && delta.logits_changed_windows <= contract.evaluation.windows
      && delta.probabilities_changed_windows <= contract.evaluation.windows
      && delta.target_logit_changed_windows <= contract.evaluation.windows
      && delta.target_probability_changed_windows <= contract.evaluation.windows
      && delta.prediction_changed_windows <= contract.evaluation.windows
      && (delta.feature_changed_windows > 0) === (delta.feature_delta_l1 > 0)
      && (delta.logits_changed_windows > 0) === (delta.logit_delta_l1 > 0)
      && (delta.probabilities_changed_windows > 0) === (delta.probability_delta_l1 > 0),
    zero_forward_saturation:
      comparison.health.source_residual_saturation_count === 0
      && comparison.health.candidate_residual_saturation_count === 0,
    exactly_one_classification_selected: selectedClassifications.length === 1,
    paid_cloud_execution_not_authorized:
      contract.authorization.paid_cloud_execution === false,
  };
  const eligible = Object.values(gates).every(Boolean);
  const classification = selectedClassifications.length === 1
    ? selectedClassifications[0]
    : "ambiguous_functional_comparison";
  return {
    schema: "nsrl.production_functional_comparison_checkpoint.v1",
    profile: contract.profile,
    parameter_count: contract.parameter_count,
    contract: { path: contractPath, sha256: sha256(contractBytes) },
    models: {
      source: contract.source_model,
      candidate: contract.candidate_model,
    },
    bindings: contract.bindings,
    comparison,
    classification,
    selected_classifications: selectedClassifications,
    gates,
    diagnostic_eligible: eligible,
    promotion_eligible: false,
    paid_scale_authorized: false,
    next_gate: eligible && classification === "weight_updates_masked_before_final_features"
      ? "p10m_up_forward_scale_sensitivity_contract"
      : "p10m_integer_objective_quality_review",
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
if (checkpoint.schema !== "nsrl.production_functional_comparison_checkpoint.v1"
  || checkpoint.profile !== "p10m"
  || checkpoint.parameter_count !== 9317632
  || checkpoint.contract.path !== contractPath
  || checkpoint.contract.sha256 !== sha256(currentContractBytes)
  || checkpoint.selected_classifications.length !== 1
  || checkpoint.classification !== checkpoint.selected_classifications[0]
  || checkpoint.diagnostic_eligible !== Object.values(checkpoint.gates).every(Boolean)
  || checkpoint.promotion_eligible !== false
  || checkpoint.paid_scale_authorized !== false) {
  throw new Error("production functional comparison checkpoint is structurally invalid");
}
const rendered = `${JSON.stringify(checkpoint, null, 2)}\n`;
if (checkOnly) {
  if (await readFile(outPath, "utf8") !== rendered) {
    throw new Error("production functional comparison checkpoint is stale");
  }
  console.log(JSON.stringify({
    schema: "nsrl.production_functional_comparison_check.v1",
    ok: true,
    classification: checkpoint.classification,
  }));
} else {
  await writeFile(outPath, rendered);
  console.log(JSON.stringify({
    out: outPath,
    classification: checkpoint.classification,
    diagnostic_eligible: checkpoint.diagnostic_eligible,
  }));
}
