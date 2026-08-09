#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let contractPath = "";
let runDir = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !runDir || !outPath) {
  throw new Error("--contract, --run-dir, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const deltaValue = (group) => group.total ?? group;
const contract = await json(contractPath);
assert(
  contract.schema === "nsrl.production_representation_stability_contract.v1",
  "unexpected representation stability contract schema",
);
for (const artifact of [
  { path: contract.source.model_path, sha256: contract.source.model_sha256 },
  { path: contract.bindings.tokenizer_path, sha256: contract.bindings.tokenizer_sha256 },
  { path: contract.bindings.train_tokens_path, sha256: contract.bindings.train_tokens_sha256 },
  { path: contract.bindings.dev_tokens_path, sha256: contract.bindings.dev_tokens_sha256 },
  ...contract.derivation.artifacts,
]) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const files = {
  train: path.join(runDir, "train.json"),
  model: path.join(runDir, "candidate.nsrlpm"),
  optimizer: path.join(runDir, "candidate.nsrlpo"),
  development: path.join(runDir, "development.json"),
  saturation: path.join(runDir, "saturation.json"),
  delta: path.join(runDir, "delta.json"),
};
const [trace, modelBytes, optimizerBytes, development, saturation, delta] = await Promise.all([
  json(files.train),
  readFile(files.model),
  readFile(files.optimizer),
  json(files.development),
  json(files.saturation),
  json(files.delta),
]);
assert(trace.schema === "nsrl.production_full_train_smoke.v1", "unexpected training trace schema");
assert(
  trace.bindings.tokenizer_hash === contract.bindings.tokenizer_hash
    && trace.bindings.token_stream_hash === contract.bindings.train_token_stream_hash,
  "training binding mismatch",
);
assert(
  trace.training.context_tokens === contract.training.context_tokens
    && trace.training.windows === contract.training.windows
    && trace.training.targets_per_window === contract.training.targets_per_window
    && trace.training.batch_windows === contract.training.batch_windows
    && trace.training.output_backward_shift === contract.training.output_backward_shift
    && trace.training.probability_gradient_fractional_bits
      === contract.training.probability_gradient_fractional_bits
    && trace.training.probability_normalization === contract.training.probability_normalization
    && (contract.training.backward_quantization === undefined
      || (trace.training.backward_quantization === contract.training.backward_quantization
        && trace.training.backward_stochastic_seed
          === contract.training.backward_stochastic_seed))
    && (contract.training.embedding_residual_flush === undefined
      || trace.training.embedding_residual_flush
        === contract.training.embedding_residual_flush)
    && (contract.training.descent_guard_windows === undefined
      || (trace.training.descent_guard_windows === contract.training.descent_guard_windows
        && trace.training.descent_guard_policy
          === contract.training.descent_guard_policy))
    && (contract.training.signed_block_candidate_family === undefined
      || trace.training.descent_guard_candidate_family
        === contract.training.signed_block_candidate_family),
  "training geometry mismatch",
);
assert(
  same(trace.training.learning_rate_shifts, contract.training.learning_rate_shifts),
  "training schedule mismatch",
);
assert(
  trace.transaction?.saturation_policy === contract.training.atomic_saturation_policy
    && trace.gates.saturated_batch_rejection_enabled === true,
  "atomic saturation policy is not active",
);
assert(trace.hashes.initial_model === contract.source.model_hash, "source model hash mismatch");
assert(
  development.model_hash === trace.hashes.final_model
    && saturation.bindings.model_hash === trace.hashes.final_model
    && delta.bindings.source_model_hash === contract.source.model_hash
    && delta.bindings.candidate_model_hash === trace.hashes.final_model,
  "candidate evidence binding mismatch",
);
assert(
  development.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
  "development stream binding mismatch",
);

const movement = Object.fromEntries(
  Object.entries(delta.groups).map(([group, value]) => [group, deltaValue(value).l1]),
);
const requiredGroupsMoved = contract.gates.required_parameter_groups
  .every((group) => movement[group] > 0);
const frozenGroupsUnchanged = contract.gates.frozen_parameter_groups
  .every((group) => movement[group] === 0);
const committedTrainingSaturation = Object.values(trace.health)
  .reduce((sum, value) => sum + value, 0);
const rejected = trace.transaction.rejected_batch;
const rejectedSaturation = rejected === null ? 0 :
  rejected.gradient_saturation_count
    + rejected.residual_saturation_count
    + rejected.weight_saturation_count;
const scheduleComplete = trace.cursor.schedule_complete === true
  && trace.training.total_optimizer_step === contract.gates.required_total_optimizer_step
  && rejected === null;
const developmentTotal = development.evaluation.total_nll_millibits;
const developmentDelta = developmentTotal - contract.source.development_total_nll_millibits;
const developmentImproved = developmentDelta < 0;
const manifestSaturation = saturation.aggregate.residual_saturation_count;
const numericHealthPassed = committedTrainingSaturation === 0
  && rejectedSaturation === 0
  && manifestSaturation <= contract.gates.manifest_residual_saturation_max;
const stochasticQuantization = contract.training.backward_quantization ?? null;
const stochasticRoundUpCount = trace.diagnostics.backward_stochastic_round_up_count ?? 0;
const stochasticSignalPassed = stochasticQuantization === null
  || stochasticRoundUpCount >= contract.gates.backward_stochastic_round_up_count_min;
const embeddingResidualFlush = contract.training.embedding_residual_flush ?? null;
const embeddingResidualFlushPassed = embeddingResidualFlush === null
  || trace.gates.batched_embedding_residual_flush === true;
const descentGuardWindows = contract.training.descent_guard_windows ?? null;
const descentGuard = descentGuardWindows === null ? null : trace.descent_guard;
const descentGuardPassed = descentGuard === null
  ? descentGuardWindows === null
  : trace.gates.training_only_descent_guard_enabled === true
    && trace.gates.descent_guard_update_windows_disjoint === true
    && descentGuard.surface === contract.training.descent_guard_surface
    && descentGuard.window_rank_hash === contract.training.descent_guard_window_rank_hash
    && descentGuard.update_window_overlap_count === 0
    && descentGuard.final_nll_millibits <= descentGuard.initial_nll_millibits
    && descentGuard.accepted_batches + descentGuard.rejected_batches
      === descentGuard.evaluated_batches
    && descentGuard.evaluated_batches
      >= contract.gates.descent_guard_evaluated_batches_min
    && descentGuard.accepted_batches
      >= contract.gates.descent_guard_accepted_batches_min;
const signedBlockRequired = contract.training.signed_block_candidate_family !== undefined;
const signedBlock = signedBlockRequired ? trace.signed_block_trust_region : null;
const signedBlockPassed = !signedBlockRequired
  || trace.gates.signed_block_trust_region_enabled === true
    && trace.gates.signed_block_source_candidate_guarantees_nonworsening === true
    && signedBlock.source_always_candidate === true
    && signedBlock.evaluated_batches >= contract.gates.signed_block_evaluated_batches_min
    && signedBlock.selected_batches >= contract.gates.signed_block_selected_batches_min
    && signedBlock.last_selection.candidates_evaluated
      === contract.gates.signed_block_candidates_evaluated
    && signedBlock.last_selection.selected_nll_millibits
      < signedBlock.last_selection.before_nll_millibits;
const stabilityPassed = scheduleComplete
  && numericHealthPassed
  && frozenGroupsUnchanged
  && embeddingResidualFlushPassed
  && descentGuardPassed
  && signedBlockPassed;
const livenessPassed = requiredGroupsMoved;
const developmentAccepted = contract.gates.development_total_nll_non_regression === true
  ? developmentDelta <= 0
  : developmentImproved;
const allGatesPassed = stabilityPassed
  && stochasticSignalPassed
  && livenessPassed
  && developmentAccepted;
const outcome = allGatesPassed
  ? descentGuardWindows === null
    ? "stable_live_development_improved"
    : signedBlockRequired
      ? "stable_signed_block_development_improved"
      : "stable_guarded_development_nonregressing"
  : !scheduleComplete
    ? "atomic_guard_stopped_before_full_horizon"
    : !numericHealthPassed
      ? "full_horizon_numeric_health_failed"
      : !frozenGroupsUnchanged
        ? "isolation_failed"
        : !stochasticSignalPassed
          ? "full_horizon_stochastic_signal_missing"
          : !embeddingResidualFlushPassed
            ? "batch_complete_embedding_residual_flush_missing"
            : !descentGuardPassed
              ? "training_only_descent_guard_failed"
              : !signedBlockPassed
                ? "signed_block_trust_region_failed"
            : !livenessPassed
              ? "stable_but_required_representation_groups_not_live"
              : "stable_live_without_development_improvement";

const result = {
  schema: "nsrl.production_representation_stability.v1",
  checked: check,
  objective: contract.objective,
  outcome,
  source_model_hash: contract.source.model_hash,
  candidate: {
    model_hash: trace.hashes.final_model,
    model_sha256: sha256(modelBytes),
    optimizer_state_hash: trace.hashes.optimizer_state,
    optimizer_sha256: sha256(optimizerBytes),
    total_optimizer_step: trace.training.total_optimizer_step,
    next_epoch: trace.cursor.next_epoch,
    next_window: trace.cursor.next_window,
    schedule_complete: trace.cursor.schedule_complete,
  },
  transaction: trace.transaction,
  health: {
    committed_training: trace.health,
    rejected_batch_saturation_count: rejectedSaturation,
    development_residual_saturation_count: development.health.residual_saturation_count,
    manifest_residual_saturation_count: manifestSaturation,
  },
  development: {
    source_total_nll_millibits: contract.source.development_total_nll_millibits,
    candidate_total_nll_millibits: developmentTotal,
    delta_millibits: developmentDelta,
  },
  movement_l1: movement,
  ...(stochasticQuantization === null ? {} : {
    backward_quantization: {
      mode: stochasticQuantization,
      seed: contract.training.backward_stochastic_seed,
      stochastic_round_up_count: stochasticRoundUpCount,
      backward_quantization_count: trace.diagnostics.backward_quantization_count,
    },
  }),
  ...(descentGuard === null ? {} : {
    descent_guard: descentGuard,
  }),
  ...(signedBlock === null ? {} : {
    signed_block_trust_region: signedBlock,
  }),
  gates: {
    schedule_complete_at_required_step: scheduleComplete,
    atomic_saturation_guard_active: true,
    no_batch_rejected: rejected === null,
    committed_training_saturation_zero: committedTrainingSaturation === 0,
    rejected_batch_saturation_zero: rejectedSaturation === 0,
    development_residual_saturation_zero:
      development.health.residual_saturation_count === 0,
    manifest_residual_saturation_zero: manifestSaturation === 0,
    ...(stochasticQuantization === null ? {} : {
      stochastic_round_up_count_at_least_minimum: stochasticSignalPassed,
    }),
    ...(embeddingResidualFlush === null ? {} : {
      batch_complete_embedding_residual_flush_active: embeddingResidualFlushPassed,
    }),
    ...(descentGuardWindows === null ? {} : {
      training_only_descent_guard_passed: descentGuardPassed,
    }),
    ...(signedBlockRequired ? {
      signed_block_trust_region_passed: signedBlockPassed,
    } : {}),
    required_parameter_groups_moved: requiredGroupsMoved,
    frozen_parameter_groups_unchanged: frozenGroupsUnchanged,
    development_strictly_improved: developmentImproved,
    ...(contract.gates.development_total_nll_non_regression === true ? {
      development_did_not_regress: developmentDelta <= 0,
    } : {}),
    test_partition_not_read: true,
    all_stability_liveness_development_gates_passed: allGatesPassed,
  },
  authorization: {
    diagnostic_only: true,
    test_evaluation: false,
    quality_postflight: false,
    quality_promotion: false,
    open_generation_rerun: false,
  },
};
const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "representation stability checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  total_optimizer_step: result.candidate.total_optimizer_step,
  all_gates_passed: result.gates.all_stability_liveness_development_gates_passed,
  out: outPath,
})}\n`);
