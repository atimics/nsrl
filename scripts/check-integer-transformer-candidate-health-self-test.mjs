#!/usr/bin/env node

import assert from "node:assert/strict";

import { checkCandidateHealth } from "./check-integer-transformer-candidate-health.mjs";

const healthy = {
  schema: "nsrl.training_mini_transformer_integer_adam_trace.v1",
  model: {
    architecture_profile: "small-h8-d128-ff256",
    transformer_layers: 2,
    rms_norm_enabled: true,
  },
  training: { train_scope: "all" },
  data: { examined_windows: 512 },
  updates: { accepted_windows: 512, accepted_batches: 32, rejected_batches: 0 },
  loss: { initial_probability_error_q15: 1000, final_probability_error_q15: 900 },
  delta_l1: { attention_q: 1, attention_k: 2, attention_v: 3, attention_o: 4 },
  saturation: { mlp: 0, attention: 0, residual: 0 },
};
const policy = { expectedProfile: "small-h8-d128-ff256" };
const evaluation = {
  schema: "nsrl.mini_transformer_eval.v1",
  evaluation: {
    invalid_forward_count: 0,
    unique_predicted_tokens: 32,
    most_predicted_token: 32,
    most_predicted_token_count: 200,
    most_predicted_token_share_per_mille: 200,
  },
};

assert.equal(checkCandidateHealth(healthy, policy).ok, true);
assert.equal(checkCandidateHealth(
  { ...healthy, model: { ...healthy.model, quantization_profile: "calibrated-v1" } },
  { ...policy, expectedQuantizationProfile: "calibrated-v1" },
).ok, true);
assert.equal(checkCandidateHealth(healthy, { ...policy, expectedQuantizationProfile: "calibrated-v1" }).ok, false);
assert.equal(checkCandidateHealth(healthy, policy, evaluation).ok, true);
assert.equal(checkCandidateHealth({ ...healthy, delta_l1: { ...healthy.delta_l1, attention_q: 0 } }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, saturation: { ...healthy.saturation, residual: 1 } }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, model: { ...healthy.model, rms_norm_enabled: false } }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, loss: { initial_probability_error_q15: 1000, final_probability_error_q15: 1001 } }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, updates: { ...healthy.updates, rejected_batches: 1 } }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, saturation: { ...healthy.saturation, residual: 1 } }, { ...policy, maxResidualSaturations: 1 }).ok, true);
assert.equal(checkCandidateHealth({ ...healthy, saturation: {} }, policy).ok, false);
assert.equal(checkCandidateHealth({ ...healthy, updates: { accepted_windows: 1, accepted_batches: 1 } }, policy).ok, false);
assert.equal(checkCandidateHealth(healthy, policy, {
  ...evaluation,
  evaluation: { ...evaluation.evaluation, unique_predicted_tokens: 1, most_predicted_token_share_per_mille: 944 },
}).ok, false);
assert.equal(checkCandidateHealth(healthy, policy, {
  ...evaluation,
  evaluation: { ...evaluation.evaluation, most_predicted_token_share_per_mille: undefined },
}).ok, false);

console.log(JSON.stringify({ passed: true, checks: 14 }));
