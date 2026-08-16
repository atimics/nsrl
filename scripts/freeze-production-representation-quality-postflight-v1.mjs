#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

let contractPath = "";
let runDir = "";
let openGenerationDir = "";
let outPath = "";
let check = false;
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (arg === "--contract") contractPath = process.argv[++index];
  else if (arg === "--run-dir") runDir = process.argv[++index];
  else if (arg === "--open-generation-dir") openGenerationDir = process.argv[++index];
  else if (arg === "--out") outPath = process.argv[++index];
  else if (arg === "--check") check = true;
  else throw new Error(`unknown argument: ${arg}`);
}
if (!contractPath || !runDir || !openGenerationDir || !outPath) {
  throw new Error("--contract, --run-dir, --open-generation-dir, and --out are required");
}

const json = (file) => readFile(file, "utf8").then(JSON.parse);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const binding = (file, bytes) => ({ path: file, bytes: bytes.length, sha256: sha256(bytes) });

const contractBytes = await readFile(contractPath);
const contract = JSON.parse(contractBytes);
assert(
  contract.schema === "nsrl.production_representation_quality_postflight_contract.v1",
  "unexpected representation quality-postflight contract schema",
);
for (const artifact of contract.bindings.artifacts) {
  assert(
    sha256(await readFile(artifact.path)) === artifact.sha256,
    `contracted artifact hash mismatch: ${artifact.path}`,
  );
}

const name = contract.name;
const testPath = path.join(runDir, "test.json");
const rolloutPath = path.join(openGenerationDir, `${name}-rollout-divergence.json`);
const contextPath = path.join(openGenerationDir, `${name}-context-sensitivity.json`);
const saturationPath = path.join(openGenerationDir, `${name}-residual-saturation.json`);
const [scaleBytes, testBytes, rolloutBytes, contextBytes, saturationBytes] = await Promise.all([
  readFile(contract.candidate.scale_evidence_path),
  readFile(testPath),
  readFile(rolloutPath),
  readFile(contextPath),
  readFile(saturationPath),
]);
const [scale, test, rollout, context, saturation] = [
  scaleBytes,
  testBytes,
  rolloutBytes,
  contextBytes,
  saturationBytes,
].map((bytes) => JSON.parse(bytes));

assert(
  scale.schema === "nsrl.production_representation_health_scale.v1"
    && scale.gates.all_broader_horizon_gates_passed === true
    && scale.candidate.model_hash === contract.candidate.model_hash
    && scale.candidate.model_sha256 === contract.candidate.model_sha256,
  "candidate scale evidence binding is invalid",
);
assert(
  test.schema === "nsrl.production_model_canonical_eval.v2"
    && rollout.schema === "nsrl.production_rollout_divergence_audit.v1"
    && context.schema === "nsrl.production_context_sensitivity_audit.v1"
    && saturation.schema === "nsrl.production_residual_saturation_audit.v1",
  "quality-postflight audit schema is invalid",
);
const candidateHash = contract.candidate.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
assert(
  test.model_hash === candidateHash
    && rollout.bindings.model_hash === candidateHash
    && context.bindings.model_hash === candidateHash
    && saturation.bindings.model_hash === candidateHash
    && test.bindings.tokenizer_hash === tokenizerHash
    && rollout.bindings.tokenizer_hash === tokenizerHash
    && context.bindings.tokenizer_hash === tokenizerHash
    && saturation.bindings.tokenizer_hash === tokenizerHash
    && test.bindings.token_stream_hash === contract.bindings.test_token_stream_hash
    && rollout.bindings.token_stream_hash === contract.bindings.dev_token_stream_hash,
  "quality-postflight audit bindings are invalid",
);

const expected = contract.postflight;
assert(
  test.evaluation.context_tokens === expected.test.context_tokens
    && test.evaluation.windows === expected.test.windows
    && rollout.counts.windows === expected.rollout.windows
    && rollout.counts.context_tokens === expected.rollout.context_tokens
    && rollout.counts.rollout_tokens === expected.rollout.rollout_tokens
    && rollout.counts.evaluated_positions === expected.rollout.evaluated_positions
    && context.counts.prompts === expected.context.prompts
    && context.counts.top_k === expected.context.top_k
    && saturation.counts.prompts === expected.saturation.prompts
    && saturation.counts.layers === expected.saturation.layers,
  "quality-postflight audit geometry is invalid",
);

const testNll = test.evaluation.total_nll_millibits;
const testDelta = testNll - contract.source.test_nll_millibits;
const gates = {
  broader_horizon_candidate_passed: scale.gates.all_broader_horizon_gates_passed === true,
  development_nll_strictly_improved: scale.development.delta_millibits < 0,
  public_test_nll_strictly_improved: testDelta < 0,
  teacher_forced_top1_minimum:
    rollout.teacher_forced.top1_matches >= expected.rollout.minimum_teacher_forced_top1_matches,
  teacher_forced_mean_target_rank_maximum:
    rollout.teacher_forced.mean_target_rank <= expected.rollout.maximum_mean_target_rank,
  teacher_forced_mean_target_probability_q15_minimum:
    rollout.teacher_forced.mean_target_probability_q15
      >= expected.rollout.minimum_mean_target_probability_q15,
  free_running_self_loop_per_mille_maximum:
    rollout.free_running.self_loop_transition_per_mille
      <= expected.rollout.maximum_self_loop_transition_per_mille,
  prefix_to_suffix_context_effect_per_mille_maximum:
    rollout.counterfactual_context.prefix_to_suffix_logit_l1_per_mille
      <= expected.rollout.maximum_prefix_to_suffix_logit_l1_per_mille,
  context_unique_greedy_tokens_minimum:
    context.aggregate.unique_greedy_tokens >= expected.context.minimum_unique_greedy_tokens,
  context_greedy_self_loops_maximum:
    context.aggregate.greedy_self_loops <= expected.context.maximum_greedy_self_loops,
  inference_residual_saturation_maximum:
    rollout.residual_saturation_count <= expected.saturation.maximum_residual_saturation_count
      && context.aggregate.residual_saturation_count
        <= expected.saturation.maximum_residual_saturation_count
      && saturation.aggregate.residual_saturation_count
        <= expected.saturation.maximum_residual_saturation_count,
};
const qualityGatePassed = Object.values(gates).every(Boolean);
const openGenerationRerunAuthorized = qualityGatePassed
  && contract.authorization.open_generation_rerun_on_pass === true;
const result = {
  schema: "nsrl.production_representation_quality_postflight.v1",
  checked: check,
  objective: contract.objective,
  outcome: qualityGatePassed
    ? "public_quality_postflight_passed"
    : "public_quality_postflight_failed",
  contract: binding(contractPath, contractBytes),
  candidate: {
    model_hash: candidateHash,
    model_sha256: contract.candidate.model_sha256,
    scale_evidence: binding(contract.candidate.scale_evidence_path, scaleBytes),
  },
  measurements: {
    development_total_nll_delta_millibits: scale.development.delta_millibits,
    source_test_total_nll_millibits: contract.source.test_nll_millibits,
    candidate_test_total_nll_millibits: testNll,
    test_total_nll_delta_millibits: testDelta,
    teacher_forced_top1_matches: rollout.teacher_forced.top1_matches,
    teacher_forced_mean_target_rank: rollout.teacher_forced.mean_target_rank,
    teacher_forced_mean_target_probability_q15:
      rollout.teacher_forced.mean_target_probability_q15,
    free_running_self_loop_transition_per_mille:
      rollout.free_running.self_loop_transition_per_mille,
    prefix_to_suffix_logit_l1_per_mille:
      rollout.counterfactual_context.prefix_to_suffix_logit_l1_per_mille,
    context_unique_greedy_tokens: context.aggregate.unique_greedy_tokens,
    context_greedy_self_loops: context.aggregate.greedy_self_loops,
    rollout_residual_saturation_count: rollout.residual_saturation_count,
    context_residual_saturation_count: context.aggregate.residual_saturation_count,
    manifest_residual_saturation_count: saturation.aggregate.residual_saturation_count,
  },
  gates,
  quality_gate_passed: qualityGatePassed,
  open_generation_rerun_authorized: openGenerationRerunAuthorized,
  evidence: {
    test: binding(testPath, testBytes),
    rollout: binding(rolloutPath, rolloutBytes),
    context: binding(contextPath, contextBytes),
    saturation: binding(saturationPath, saturationBytes),
  },
  authorization: {
    public_test_confirmation: true,
    public_generation_diagnostics: true,
    open_generation_rerun: openGenerationRerunAuthorized,
    hidden_panel_access: false,
    quality_promotion: false,
    paid_scaling: false,
  },
  known_non_claims: contract.known_non_claims,
};

const rendered = `${JSON.stringify(result, null, 2)}\n`;
if (check) {
  const existing = await readFile(outPath, "utf8");
  const unchecked = `${JSON.stringify({ ...result, checked: false }, null, 2)}\n`;
  assert(existing === unchecked || existing === rendered, "quality-postflight checkpoint differs");
} else {
  await writeFile(outPath, rendered);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: check,
  outcome: result.outcome,
  candidate: result.candidate.model_hash,
  test_delta_millibits: result.measurements.test_total_nll_delta_millibits,
  quality_gate_passed: result.quality_gate_passed,
  open_generation_rerun_authorized: result.open_generation_rerun_authorized,
  out: outPath,
})}\n`);
