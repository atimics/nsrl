#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const config = parseArgs(process.argv.slice(2));
const contractBytes = fs.readFileSync(config.contract);
const contract = JSON.parse(contractBytes);
const confirmationBytes = fs.readFileSync(config.confirmation);
const confirmation = JSON.parse(confirmationBytes);
const sourceContextBytes = fs.readFileSync(config.sourceContext);
const sourceContext = JSON.parse(sourceContextBytes);
const candidateContextBytes = fs.readFileSync(config.candidateContext);
const candidateContext = JSON.parse(candidateContextBytes);
const sourceSaturationBytes = fs.readFileSync(config.sourceSaturation);
const sourceSaturation = JSON.parse(sourceSaturationBytes);
const candidateSaturationBytes = fs.readFileSync(config.candidateSaturation);
const candidateSaturation = JSON.parse(candidateSaturationBytes);

assert(contract.schema === "nsrl.production_direct_head_nll_guard_contract.v1"
  && confirmation.schema
    === "nsrl.production_direct_head_nll_guard_confirmation_gate.v1"
  && confirmation.contract.sha256 === sha256(contractBytes)
  && confirmation.confirmation_gate_passed === true
  && confirmation.open_generation_authorized === true,
"direct-head confirmation gate is invalid");
assert(sourceContext.schema === "nsrl.production_context_sensitivity_audit.v1"
  && candidateContext.schema === "nsrl.production_context_sensitivity_audit.v1"
  && sourceSaturation.schema === "nsrl.production_residual_saturation_audit.v1"
  && candidateSaturation.schema === "nsrl.production_residual_saturation_audit.v1",
"direct-head open-generation schema is invalid");

const candidateModelHash = confirmation.candidate_model_hash;
const sourceModelHash = contract.source.model_hash;
const tokenizerHash = contract.bindings.tokenizer_hash;
for (const artifact of [sourceContext, sourceSaturation]) {
  assert((artifact.model_hash ?? artifact.bindings?.model_hash) === sourceModelHash,
    "source open-generation model binding mismatch");
  assert((artifact.bindings?.tokenizer_hash ?? tokenizerHash) === tokenizerHash,
    "source open-generation tokenizer binding mismatch");
}
for (const artifact of [candidateContext, candidateSaturation]) {
  assert((artifact.model_hash ?? artifact.bindings?.model_hash) === candidateModelHash,
    "candidate open-generation model binding mismatch");
  assert((artifact.bindings?.tokenizer_hash ?? tokenizerHash) === tokenizerHash,
    "candidate open-generation tokenizer binding mismatch");
}

const expected = contract.open_generation_gates;
const measurements = {
  source_context_unique_greedy_tokens: sourceContext.aggregate.unique_greedy_tokens,
  candidate_context_unique_greedy_tokens: candidateContext.aggregate.unique_greedy_tokens,
  source_context_greedy_self_loops: sourceContext.aggregate.greedy_self_loops,
  candidate_context_greedy_self_loops: candidateContext.aggregate.greedy_self_loops,
  source_context_residual_saturation_count:
    sourceContext.aggregate.residual_saturation_count,
  candidate_context_residual_saturation_count:
    candidateContext.aggregate.residual_saturation_count,
  source_manifest_residual_saturation_count:
    sourceSaturation.aggregate.residual_saturation_count,
  candidate_manifest_residual_saturation_count:
    candidateSaturation.aggregate.residual_saturation_count,
};
const gates = {
  context_unique_greedy_tokens_minimum:
    candidateContext.aggregate.unique_greedy_tokens
      >= expected.context_unique_greedy_tokens_minimum,
  context_greedy_self_loops_maximum:
    candidateContext.aggregate.greedy_self_loops
      <= expected.context_greedy_self_loops_maximum,
  context_and_manifest_residual_saturation_maximum:
    candidateContext.aggregate.residual_saturation_count
      <= expected.context_and_manifest_residual_saturation_maximum
      && candidateSaturation.aggregate.residual_saturation_count
        <= expected.context_and_manifest_residual_saturation_maximum,
};
const passed = Object.values(gates).every(Boolean);
const result = {
  schema: "nsrl.production_direct_head_nll_guard_quality_gate.v1",
  contract: binding(config.contract, contractBytes),
  confirmation: binding(config.confirmation, confirmationBytes),
  candidate_model_hash: candidateModelHash,
  measurements,
  gates,
  quality_gate_passed: passed,
  hidden_panel_opened: false,
  paid_scaling_authorized: false,
  evidence: {
    source_context: binding(config.sourceContext, sourceContextBytes),
    candidate_context: binding(config.candidateContext, candidateContextBytes),
    source_saturation: binding(config.sourceSaturation, sourceSaturationBytes),
    candidate_saturation: binding(config.candidateSaturation, candidateSaturationBytes),
  },
  known_non_claims: contract.known_non_claims,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.out, "utf8") === output,
    "direct-head quality gate does not byte-replay");
} else {
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  quality_gate_passed: passed,
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
    "--contract": "contract", "--confirmation": "confirmation",
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
