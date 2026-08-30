#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const developmentBytes = fs.readFileSync(config.development);
const development = JSON.parse(developmentBytes);
const sourceTestBytes = fs.readFileSync(config.sourceTest);
const sourceTest = JSON.parse(sourceTestBytes);
const candidateTestBytes = fs.readFileSync(config.candidateTest);
const candidateTest = JSON.parse(candidateTestBytes);
const sourceRolloutBytes = fs.readFileSync(config.sourceRollout);
const sourceRollout = JSON.parse(sourceRolloutBytes);
const candidateRolloutBytes = fs.readFileSync(config.candidateRollout);
const candidateRollout = JSON.parse(candidateRolloutBytes);

assert(contract.schema === "nsrl.production_direct_head_nll_safe_set_contract.v1"
  && development.schema
    === "nsrl.production_direct_head_nll_safe_set_development_gate.v1"
  && development.contract.sha256 === sha256(contractBytes)
  && development.development_gate_passed === true
  && development.public_test_authorized === true,
"direct-head safe-set development gate is invalid");
assert(sourceTest.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateTest.schema === "nsrl.production_model_canonical_eval.v2"
  && sourceRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && candidateRollout.schema === "nsrl.production_rollout_divergence_audit.v1",
"direct-head safe-set confirmation schema is invalid");

const candidateModelHash = development.candidate_model_hash;
const sourceModelHash = contract.source.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
assert(sourceTest.model_hash === sourceModelHash
  && candidateTest.model_hash === candidateModelHash
  && sourceRollout.bindings.model_hash === sourceModelHash
  && candidateRollout.bindings.model_hash === candidateModelHash
  && sourceTest.bindings.tokenizer_hash === tokenizerHash
  && candidateTest.bindings.tokenizer_hash === tokenizerHash
  && sourceTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
  && candidateTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
  && sourceRollout.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
  && candidateRollout.bindings.token_stream_hash === contract.bindings.test_token_stream_hash,
"direct-head safe-set confirmation binding mismatch");

const expected = contract.confirmation_gates;
assert(sourceRollout.counts.windows === expected.rollout_windows
  && candidateRollout.counts.windows === expected.rollout_windows
  && sourceRollout.counts.rollout_tokens === expected.rollout_tokens
  && candidateRollout.counts.rollout_tokens === expected.rollout_tokens,
"direct-head safe-set confirmation rollout geometry is invalid");
const sourceTestNll = sourceTest.evaluation.total_nll_millibits;
const candidateTestNll = candidateTest.evaluation.total_nll_millibits;
const sourceRank = sourceRollout.teacher_forced.mean_target_rank;
const candidateRank = candidateRollout.teacher_forced.mean_target_rank;
const measurements = {
  test_source_total_nll_millibits: sourceTestNll,
  test_candidate_total_nll_millibits: candidateTestNll,
  test_nll_delta_millibits: candidateTestNll - sourceTestNll,
  test_nll_regression_per_mille:
    Math.max(0, Math.ceil((candidateTestNll - sourceTestNll) * 1000 / sourceTestNll)),
  source_teacher_forced_top1_matches: sourceRollout.teacher_forced.top1_matches,
  candidate_teacher_forced_top1_matches: candidateRollout.teacher_forced.top1_matches,
  source_teacher_forced_mean_target_rank: sourceRank,
  candidate_teacher_forced_mean_target_rank: candidateRank,
  teacher_forced_rank_improvement_per_mille:
    improvementPerMille(sourceRank, candidateRank),
  source_teacher_forced_mean_target_probability_q15:
    sourceRollout.teacher_forced.mean_target_probability_q15,
  candidate_teacher_forced_mean_target_probability_q15:
    candidateRollout.teacher_forced.mean_target_probability_q15,
  source_free_running_self_loop_transition_per_mille:
    sourceRollout.free_running.self_loop_transition_per_mille,
  candidate_free_running_self_loop_transition_per_mille:
    candidateRollout.free_running.self_loop_transition_per_mille,
  source_test_residual_saturation_count: sourceTest.health.residual_saturation_count,
  candidate_test_residual_saturation_count: candidateTest.health.residual_saturation_count,
};
const gates = {
  test_nll_regression_per_mille_maximum:
    measurements.test_nll_regression_per_mille
      <= expected.test_nll_regression_per_mille_maximum,
  test_residual_saturation_must_not_increase:
    candidateTest.health.residual_saturation_count <= sourceTest.health.residual_saturation_count,
  teacher_forced_mean_target_rank_improvement_per_mille_minimum:
    measurements.teacher_forced_rank_improvement_per_mille
      >= expected.teacher_forced_mean_target_rank_improvement_per_mille_minimum,
  teacher_forced_top1_must_not_decrease:
    !expected.teacher_forced_top1_must_not_decrease
      || candidateRollout.teacher_forced.top1_matches
        >= sourceRollout.teacher_forced.top1_matches,
  teacher_forced_mean_target_probability_must_not_decrease:
    !expected.teacher_forced_mean_target_probability_must_not_decrease
      || candidateRollout.teacher_forced.mean_target_probability_q15
        >= sourceRollout.teacher_forced.mean_target_probability_q15,
  free_running_self_loop_transition_per_mille_must_not_increase:
    candidateRollout.free_running.self_loop_transition_per_mille
      <= sourceRollout.free_running.self_loop_transition_per_mille,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_nll_safe_set_confirmation_gate.v1",
  contract: binding(config.contract, contractBytes),
  development: binding(config.development, developmentBytes),
  candidate_model_hash: candidateModelHash,
  measurements,
  gates,
  confirmation_gate_passed: passed,
  public_test_opened: true,
  open_generation_authorized: passed,
  open_generation_opened: false,
  hidden_panel_opened: false,
  evidence: {
    source_test: binding(config.sourceTest, sourceTestBytes),
    candidate_test: binding(config.candidateTest, candidateTestBytes),
    source_test_rollout: binding(config.sourceRollout, sourceRolloutBytes),
    candidate_test_rollout: binding(config.candidateRollout, candidateRolloutBytes),
  },
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "direct-head safe-set confirmation gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  confirmation_gate_passed: passed,
  open_generation_authorized: passed,
  gates,
  out: config.out,
})}\n`);
if (!passed) process.exitCode = 1;

function improvementPerMille(source, candidate) {
  return Math.floor((source - candidate) * 1000 / Math.max(1, source));
}

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
  const config = {check: false};
  const names = {
    "--contract": "contract", "--development": "development",
    "--source-test": "sourceTest", "--candidate-test": "candidateTest",
    "--source-rollout": "sourceRollout", "--candidate-rollout": "candidateRollout",
    "--out": "out",
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--check") config.check = true;
    else {
      const name = names[args[index]];
      if (!name) throw new Error(`unknown argument ${args[index]}`);
      config[name] = args[++index] || "";
    }
  }
  for (const name of Object.values(names)) assert(config[name], `missing ${name}`);
  return config;
}
