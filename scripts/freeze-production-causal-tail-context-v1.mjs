#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const preflightBytes = fs.readFileSync(config.preflight);
const preflight = JSON.parse(preflightBytes);
const rolloutBytes = fs.readFileSync(config.rollout);
const rollout = JSON.parse(rolloutBytes);
const contextBytes = fs.readFileSync(config.context);
const context = JSON.parse(contextBytes);
const saturationBytes = fs.readFileSync(config.saturation);
const saturation = JSON.parse(saturationBytes);

assert([
  "nsrl.production_causal_tail_context_contract.v1",
  "nsrl.production_causal_tail_stability_contract.v1",
].includes(contract.schema),
  "causal tail-context contract schema is invalid");
assert(contract.authorization?.postflight_quality_gate_required === true
  && contract.authorization?.hidden_panel_access === false
  && contract.authorization?.paid_scaling === false,
"causal tail-context authorization is invalid");
for (const artifact of contract.derivation?.artifacts ?? []) {
  verifyInput(artifact.path, artifact.sha256);
}
assert(preflight.schema === "nsrl.production_causal_sequence_preflight.v1"
  && preflight.contract?.sha256 === sha256(contractBytes)
  && preflight.open_generation_rerun_authorized === false,
"causal tail-context preflight binding is invalid");
assert(rollout.schema === "nsrl.production_rollout_divergence_audit.v1"
  && context.schema === "nsrl.production_context_sensitivity_audit.v1"
  && saturation.schema === "nsrl.production_residual_saturation_audit.v1",
"causal tail-context audit schema is invalid");

const modelHash = preflight.candidate?.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
assert(modelHash
  && rollout.bindings?.model_hash === modelHash
  && context.bindings?.model_hash === modelHash
  && saturation.bindings?.model_hash === modelHash
  && rollout.bindings?.tokenizer_hash === tokenizerHash
  && context.bindings?.tokenizer_hash === tokenizerHash
  && saturation.bindings?.tokenizer_hash === tokenizerHash
  && rollout.bindings?.token_stream_hash === contract.bindings.dev_token_stream_hash,
"causal tail-context audit bindings are invalid");

const expected = contract.postflight;
assert(rollout.counts?.windows === expected.rollout.windows
  && rollout.counts?.context_tokens === expected.rollout.context_tokens
  && rollout.counts?.rollout_tokens === expected.rollout.rollout_tokens
  && rollout.counts?.evaluated_positions === expected.rollout.evaluated_positions
  && context.counts?.prompts === expected.context.prompts
  && context.counts?.top_k === expected.context.top_k
  && saturation.counts?.prompts === expected.saturation.prompts
  && saturation.counts?.layers === expected.saturation.layers,
"causal tail-context audit geometry is invalid");

const gates = {
  preflight_passed: preflight.preflight_passed === true,
  teacher_forced_top1_minimum:
    rollout.teacher_forced?.top1_matches >= expected.rollout.minimum_teacher_forced_top1_matches,
  teacher_forced_mean_target_rank_maximum:
    rollout.teacher_forced?.mean_target_rank <= expected.rollout.maximum_mean_target_rank,
  teacher_forced_mean_target_probability_q15_minimum:
    rollout.teacher_forced?.mean_target_probability_q15
      >= expected.rollout.minimum_mean_target_probability_q15,
  free_running_self_loop_per_mille_maximum:
    rollout.free_running?.self_loop_transition_per_mille
      <= expected.rollout.maximum_self_loop_transition_per_mille,
  prefix_to_suffix_context_effect_per_mille_maximum:
    rollout.counterfactual_context?.prefix_to_suffix_logit_l1_per_mille
      <= expected.rollout.maximum_prefix_to_suffix_logit_l1_per_mille,
  context_unique_greedy_tokens_minimum:
    context.aggregate?.unique_greedy_tokens >= expected.context.minimum_unique_greedy_tokens,
  context_greedy_self_loops_maximum:
    context.aggregate?.greedy_self_loops <= expected.context.maximum_greedy_self_loops,
  inference_residual_saturation_maximum:
    rollout.residual_saturation_count <= expected.saturation.maximum_residual_saturation_count
      && context.aggregate?.residual_saturation_count
        <= expected.saturation.maximum_residual_saturation_count
      && saturation.aggregate?.residual_saturation_count
        <= expected.saturation.maximum_residual_saturation_count,
};
const qualityGatePassed = Object.values(gates).every(Boolean);
const openGenerationRerunAuthorized = qualityGatePassed
  && contract.authorization?.open_generation_rerun === true;
const result = {
  schema: "nsrl.production_causal_tail_context_quality_gate.v1",
  contract: binding(config.contract, contractBytes),
  preflight: binding(config.preflight, preflightBytes),
  candidate_model_hash: modelHash,
  measurements: {
    development_total_nll_delta_millibits:
      preflight.deltas?.development_total_nll_millibits ?? null,
    test_total_nll_delta_millibits: preflight.deltas?.test_total_nll_millibits ?? null,
    teacher_forced_top1_matches: rollout.teacher_forced?.top1_matches ?? null,
    teacher_forced_mean_target_rank: rollout.teacher_forced?.mean_target_rank ?? null,
    teacher_forced_mean_target_probability_q15:
      rollout.teacher_forced?.mean_target_probability_q15 ?? null,
    free_running_self_loop_transition_per_mille:
      rollout.free_running?.self_loop_transition_per_mille ?? null,
    prefix_to_suffix_logit_l1_per_mille:
      rollout.counterfactual_context?.prefix_to_suffix_logit_l1_per_mille ?? null,
    context_unique_greedy_tokens: context.aggregate?.unique_greedy_tokens ?? null,
    context_greedy_self_loops: context.aggregate?.greedy_self_loops ?? null,
    rollout_residual_saturation_count: rollout.residual_saturation_count ?? null,
    context_residual_saturation_count: context.aggregate?.residual_saturation_count ?? null,
    manifest_residual_saturation_count: saturation.aggregate?.residual_saturation_count ?? null,
  },
  gates,
  quality_gate_passed: qualityGatePassed,
  open_generation_rerun_authorized: openGenerationRerunAuthorized,
  evidence: {
    rollout: binding(config.rollout, rolloutBytes),
    context: binding(config.context, contextBytes),
    saturation: binding(config.saturation, saturationBytes),
  },
  hidden_panel_opened: false,
  known_non_claims: contract.known_non_claims,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "causal tail-context quality gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  schema: result.schema,
  checked: config.check,
  candidate: modelHash,
  gates,
  quality_gate_passed: qualityGatePassed,
  open_generation_rerun_authorized: openGenerationRerunAuthorized,
  out: config.out,
})}\n`);

function verifyInput(file, expectedSha256) {
  assert(sha256(fs.readFileSync(file)) === expectedSha256, `${file} SHA-256 mismatch`);
}

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, fnv64: fnv64(bytes), sha256: sha256(bytes)};
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function fnv64(bytes) {
  let value = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    value = ((value ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function parseArgs(args) {
  const config = {
    contract: "benchmarks/production-model-v1/p10m-causal-tail-context-v1-contract.json",
    preflight: "benchmarks/production-model-v1/p10m-causal-tail-context-v1.json",
    rollout: "benchmarks/open-generation-v1/p10m-causal-tail-context-v1-rollout-divergence.json",
    context: "benchmarks/open-generation-v1/p10m-causal-tail-context-v1-context-sensitivity.json",
    saturation: "benchmarks/open-generation-v1/p10m-causal-tail-context-v1-residual-saturation.json",
    out: "benchmarks/production-model-v1/p10m-causal-tail-context-v1-quality-gate.json",
    check: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--contract") config.contract = args[++index] || "";
    else if (args[index] === "--preflight") config.preflight = args[++index] || "";
    else if (args[index] === "--rollout") config.rollout = args[++index] || "";
    else if (args[index] === "--context") config.context = args[++index] || "";
    else if (args[index] === "--saturation") config.saturation = args[++index] || "";
    else if (args[index] === "--out") config.out = args[++index] || "";
    else if (args[index] === "--check") config.check = true;
    else throw new Error(`unknown argument ${args[index]}`);
  }
  return config;
}
