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
const sourceContextBytes = fs.readFileSync(config.sourceContext);
const sourceContext = JSON.parse(sourceContextBytes);
const candidateContextBytes = fs.readFileSync(config.candidateContext);
const candidateContext = JSON.parse(candidateContextBytes);
const sourceSaturationBytes = fs.readFileSync(config.sourceSaturation);
const sourceSaturation = JSON.parse(sourceSaturationBytes);
const candidateSaturationBytes = fs.readFileSync(config.candidateSaturation);
const candidateSaturation = JSON.parse(candidateSaturationBytes);

assert(contract.schema === "nsrl.production_target_margin_contract.v1"
  && development.schema === "nsrl.production_target_margin_development_gate.v1"
  && development.contract.sha256 === sha256(contractBytes)
  && development.development_gate_passed === true
  && development.public_test_authorized === true,
"target-margin development gate is invalid");
assert(sourceTest.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateTest.schema === "nsrl.production_model_canonical_eval.v2"
  && sourceRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && candidateRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && sourceContext.schema === "nsrl.production_context_sensitivity_audit.v1"
  && candidateContext.schema === "nsrl.production_context_sensitivity_audit.v1"
  && sourceSaturation.schema === "nsrl.production_residual_saturation_audit.v1"
  && candidateSaturation.schema === "nsrl.production_residual_saturation_audit.v1",
"target-margin confirmation schema is invalid");

const candidateModelHash = development.candidate_model_hash;
const sourceModelHash = contract.source.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
for (const artifact of [sourceTest, sourceRollout, sourceContext, sourceSaturation]) {
  assert((artifact.model_hash ?? artifact.bindings?.model_hash) === sourceModelHash,
    "source confirmation model binding mismatch");
  assert((artifact.bindings?.tokenizer_hash ?? tokenizerHash) === tokenizerHash,
    "source confirmation tokenizer binding mismatch");
}
for (const artifact of [candidateTest, candidateRollout, candidateContext, candidateSaturation]) {
  assert((artifact.model_hash ?? artifact.bindings?.model_hash) === candidateModelHash,
    "candidate confirmation model binding mismatch");
  assert((artifact.bindings?.tokenizer_hash ?? tokenizerHash) === tokenizerHash,
    "candidate confirmation tokenizer binding mismatch");
}
assert(sourceTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
  && candidateTest.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
  && sourceRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
"target-margin confirmation token-stream binding mismatch");

const expected = contract.confirmation_gates;
assert(sourceRollout.counts.windows === expected.rollout_windows
  && candidateRollout.counts.windows === expected.rollout_windows
  && sourceRollout.counts.rollout_tokens === expected.rollout_tokens
  && candidateRollout.counts.rollout_tokens === expected.rollout_tokens,
"target-margin rollout geometry is invalid");
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
    Math.floor((sourceRank - candidateRank) * 1000 / Math.max(1, sourceRank)),
  source_teacher_forced_mean_target_probability_q15:
    sourceRollout.teacher_forced.mean_target_probability_q15,
  candidate_teacher_forced_mean_target_probability_q15:
    candidateRollout.teacher_forced.mean_target_probability_q15,
  source_free_running_self_loop_transition_per_mille:
    sourceRollout.free_running.self_loop_transition_per_mille,
  candidate_free_running_self_loop_transition_per_mille:
    candidateRollout.free_running.self_loop_transition_per_mille,
  source_context_unique_greedy_tokens: sourceContext.aggregate.unique_greedy_tokens,
  candidate_context_unique_greedy_tokens: candidateContext.aggregate.unique_greedy_tokens,
  source_context_greedy_self_loops: sourceContext.aggregate.greedy_self_loops,
  candidate_context_greedy_self_loops: candidateContext.aggregate.greedy_self_loops,
  source_test_residual_saturation_count: sourceTest.health.residual_saturation_count,
  candidate_test_residual_saturation_count: candidateTest.health.residual_saturation_count,
  candidate_rollout_residual_saturation_count: candidateRollout.residual_saturation_count,
  candidate_context_residual_saturation_count:
    candidateContext.aggregate.residual_saturation_count,
  candidate_manifest_residual_saturation_count:
    candidateSaturation.aggregate.residual_saturation_count,
};
const gates = {
  test_nll_regression_per_mille_maximum:
    measurements.test_nll_regression_per_mille <= expected.test_nll_regression_per_mille_maximum,
  test_residual_saturation_must_not_increase:
    candidateTest.health.residual_saturation_count <= sourceTest.health.residual_saturation_count,
  teacher_forced_mean_target_rank_improvement_per_mille_minimum:
    measurements.teacher_forced_rank_improvement_per_mille
      >= expected.teacher_forced_mean_target_rank_improvement_per_mille_minimum,
  teacher_forced_top1_matches_minimum:
    candidateRollout.teacher_forced.top1_matches
      >= expected.teacher_forced_top1_matches_minimum,
  teacher_forced_mean_target_probability_must_not_decrease:
    candidateRollout.teacher_forced.mean_target_probability_q15
      >= sourceRollout.teacher_forced.mean_target_probability_q15,
  free_running_self_loop_transition_per_mille_must_not_increase:
    candidateRollout.free_running.self_loop_transition_per_mille
      <= sourceRollout.free_running.self_loop_transition_per_mille,
  context_unique_greedy_tokens_minimum:
    candidateContext.aggregate.unique_greedy_tokens
      >= expected.context_unique_greedy_tokens_minimum,
  context_greedy_self_loops_maximum:
    candidateContext.aggregate.greedy_self_loops <= expected.context_greedy_self_loops_maximum,
  rollout_context_and_manifest_residual_saturation_maximum:
    candidateRollout.residual_saturation_count
      <= expected.rollout_context_and_manifest_residual_saturation_maximum
      && candidateContext.aggregate.residual_saturation_count
        <= expected.rollout_context_and_manifest_residual_saturation_maximum
      && candidateSaturation.aggregate.residual_saturation_count
        <= expected.rollout_context_and_manifest_residual_saturation_maximum,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_target_margin_quality_gate.v1",
  contract: binding(config.contract, contractBytes),
  development: binding(config.development, developmentBytes),
  candidate_model_hash: candidateModelHash,
  measurements,
  gates,
  quality_gate_passed: passed,
  open_generation_rerun_authorized:
    passed && contract.authorization.open_generation_rerun_only_after_all_public_gates,
  evidence: {
    source_test: binding(config.sourceTest, sourceTestBytes),
    candidate_test: binding(config.candidateTest, candidateTestBytes),
    source_rollout: binding(config.sourceRollout, sourceRolloutBytes),
    candidate_rollout: binding(config.candidateRollout, candidateRolloutBytes),
    source_context: binding(config.sourceContext, sourceContextBytes),
    candidate_context: binding(config.candidateContext, candidateContextBytes),
    source_saturation: binding(config.sourceSaturation, sourceSaturationBytes),
    candidate_saturation: binding(config.candidateSaturation, candidateSaturationBytes),
  },
  hidden_panel_opened: false,
  paid_scaling_authorized: false,
  known_non_claims: contract.known_non_claims,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "target-margin quality gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  quality_gate_passed: passed,
  open_generation_rerun_authorized: result.open_generation_rerun_authorized,
  gates,
  out: config.out,
})}\n`);

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
    "--source-context": "sourceContext", "--candidate-context": "candidateContext",
    "--source-saturation": "sourceSaturation",
    "--candidate-saturation": "candidateSaturation", "--out": "out",
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
