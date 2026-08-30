#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const trainingBytes = fs.readFileSync(config.training);
const training = JSON.parse(trainingBytes);
const sourceDevBytes = fs.readFileSync(config.sourceDev);
const sourceDev = JSON.parse(sourceDevBytes);
const candidateDevBytes = fs.readFileSync(config.candidateDev);
const candidateDev = JSON.parse(candidateDevBytes);
const sourceRolloutBytes = fs.readFileSync(config.sourceRollout);
const sourceRollout = JSON.parse(sourceRolloutBytes);
const candidateRolloutBytes = fs.readFileSync(config.candidateRollout);
const candidateRollout = JSON.parse(candidateRolloutBytes);

assert(contract.schema === "nsrl.production_direct_head_nll_safe_set_contract.v1"
  && training.schema === "nsrl.production_direct_head_nll_safe_set_training_gate.v1"
  && training.contract.sha256 === sha256(contractBytes)
  && training.training_gate_passed === true
  && training.public_development_authorized === true,
"direct-head safe-set training gate is invalid");
assert(sourceDev.schema === "nsrl.production_model_canonical_eval.v2"
  && candidateDev.schema === "nsrl.production_model_canonical_eval.v2"
  && sourceRollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && candidateRollout.schema === "nsrl.production_rollout_divergence_audit.v1",
"direct-head safe-set development evaluation schema is invalid");

const candidateModelHash = training.candidate_model_hash;
const sourceModelHash = contract.source.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
assert(sourceDev.model_hash === sourceModelHash
  && candidateDev.model_hash === candidateModelHash
  && sourceRollout.bindings.model_hash === sourceModelHash
  && candidateRollout.bindings.model_hash === candidateModelHash
  && sourceDev.bindings.tokenizer_hash === tokenizerHash
  && candidateDev.bindings.tokenizer_hash === tokenizerHash
  && sourceDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateDev.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && sourceRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash
  && candidateRollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
"direct-head safe-set development binding mismatch");

const expected = contract.development_gates;
assert(sourceRollout.counts.windows === expected.rollout_windows
  && candidateRollout.counts.windows === expected.rollout_windows
  && sourceRollout.counts.rollout_tokens === expected.rollout_tokens
  && candidateRollout.counts.rollout_tokens === expected.rollout_tokens,
"direct-head safe-set development rollout geometry is invalid");
const sourceNll = sourceDev.evaluation.total_nll_millibits;
const candidateNll = candidateDev.evaluation.total_nll_millibits;
const sourceRank = sourceRollout.teacher_forced.mean_target_rank;
const candidateRank = candidateRollout.teacher_forced.mean_target_rank;
const measurements = {
  development_source_total_nll_millibits: sourceNll,
  development_candidate_total_nll_millibits: candidateNll,
  development_nll_delta_millibits: candidateNll - sourceNll,
  development_nll_regression_per_mille:
    Math.max(0, Math.ceil((candidateNll - sourceNll) * 1000 / sourceNll)),
  source_development_residual_saturation_count: sourceDev.health.residual_saturation_count,
  candidate_development_residual_saturation_count:
    candidateDev.health.residual_saturation_count,
  source_development_teacher_forced_top1_matches:
    sourceRollout.teacher_forced.top1_matches,
  candidate_development_teacher_forced_top1_matches:
    candidateRollout.teacher_forced.top1_matches,
  source_development_teacher_forced_mean_target_rank: sourceRank,
  candidate_development_teacher_forced_mean_target_rank: candidateRank,
  development_teacher_forced_rank_improvement_per_mille:
    improvementPerMille(sourceRank, candidateRank),
  source_development_teacher_forced_mean_target_probability_q15:
    sourceRollout.teacher_forced.mean_target_probability_q15,
  candidate_development_teacher_forced_mean_target_probability_q15:
    candidateRollout.teacher_forced.mean_target_probability_q15,
};
const gates = {
  development_nll_regression_per_mille_maximum:
    measurements.development_nll_regression_per_mille
      <= expected.development_nll_regression_per_mille_maximum,
  development_residual_saturation_must_not_increase:
    candidateDev.health.residual_saturation_count
      <= sourceDev.health.residual_saturation_count,
  development_teacher_forced_mean_target_rank_improvement_per_mille_minimum:
    measurements.development_teacher_forced_rank_improvement_per_mille
      >= expected.development_teacher_forced_mean_target_rank_improvement_per_mille_minimum,
  development_teacher_forced_top1_must_not_decrease:
    !expected.development_teacher_forced_top1_must_not_decrease
      || candidateRollout.teacher_forced.top1_matches
        >= sourceRollout.teacher_forced.top1_matches,
  development_teacher_forced_mean_target_probability_must_not_decrease:
    !expected.development_teacher_forced_mean_target_probability_must_not_decrease
      || candidateRollout.teacher_forced.mean_target_probability_q15
        >= sourceRollout.teacher_forced.mean_target_probability_q15,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_nll_safe_set_development_gate.v1",
  contract: binding(config.contract, contractBytes),
  training: binding(config.training, trainingBytes),
  candidate_model_hash: candidateModelHash,
  measurements,
  gates,
  development_gate_passed: passed,
  public_development_opened: true,
  public_test_authorized: passed,
  public_test_opened: false,
  open_generation_opened: false,
  hidden_panel_opened: false,
  evidence: {
    source_development: binding(config.sourceDev, sourceDevBytes),
    candidate_development: binding(config.candidateDev, candidateDevBytes),
    source_development_rollout: binding(config.sourceRollout, sourceRolloutBytes),
    candidate_development_rollout: binding(config.candidateRollout, candidateRolloutBytes),
  },
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "direct-head safe-set development gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  development_gate_passed: passed,
  public_test_authorized: passed,
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
    "--contract": "contract", "--training": "training",
    "--source-dev": "sourceDev", "--candidate-dev": "candidateDev",
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
