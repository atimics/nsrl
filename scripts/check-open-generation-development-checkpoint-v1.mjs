#!/usr/bin/env node

import fs from "node:fs";

const checkpointPath = process.argv[2]
  || "benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json";
const checkpoint = JSON.parse(fs.readFileSync(checkpointPath, "utf8"));
const hash = (value) => typeof value === "string" && /^0x[0-9a-f]{16}$/.test(value);
const sha256 = (value) => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);

assert(checkpoint.schema === "nsrl.open_generation_development_checkpoint.v1"
  && checkpoint.contract === "open-generation-v1"
  && checkpoint.partition === "development",
"open-generation development checkpoint schema is invalid");
assert(hash(checkpoint.candidate?.model_fnv64) && hash(checkpoint.candidate?.tokenizer_fnv64),
  "open-generation candidate bindings are invalid");
assert(checkpoint.execution?.decoder === "incremental_linear_attention_cache_v1"
  && checkpoint.execution.counts?.prompts === 12
  && checkpoint.execution.counts?.samples === 60
  && checkpoint.execution.counts?.generated_tokens === 30_720
  && checkpoint.execution.counts?.samples_beyond_training_context === 60
  && checkpoint.execution.cache?.maximum_state_bytes > 0
  && checkpoint.execution.cache?.maximum_workspace_bytes > 0
  && checkpoint.execution.residual_saturation_count === 0,
"open-generation execution evidence is incomplete");
assert(Object.values(checkpoint.execution.forbidden_assistance ?? {})
  .every((value) => value === false),
"open-generation baseline contains forbidden assistance");
assert(checkpoint.modeling?.original_utf8_bytes > 0
  && checkpoint.modeling?.candidate_tokens > 0
  && checkpoint.modeling?.total_nll_millibits > 0
  && checkpoint.modeling?.millibits_per_original_utf8_byte > 0
  && checkpoint.modeling?.required_baselines_measured === false,
"open-generation candidate modeling evidence is invalid");

const servingGates = [
  "complete_generation_matrix",
  "incremental_cached_decoding",
  "no_residual_saturation",
  "forbidden_assistance_absent",
];
const qualityGates = [
  "repeat_4gram_health",
  "unique_4gram_health",
  "entropy_health",
  "utf8_validity",
  "context_use",
  "distractor_resistance",
];
assert(servingGates.every((gate) => checkpoint.generation?.gates?.[gate] === true)
  && qualityGates.every((gate) => checkpoint.generation?.gates?.[gate] === false)
  && checkpoint.generation?.gates?.development_generation_passed === false,
"open-generation baseline gate classification is invalid");
assert(checkpoint.evidence_layers?.candidate_modeling_measured === true
  && checkpoint.evidence_layers?.modeling_measured === false
  && checkpoint.evidence_layers?.hidden_panel_measured === false
  && checkpoint.promotion_passed === false,
"open-generation non-promotion boundary is invalid");

for (const name of ["run", "modeling", "samples", "decoder_traces", "result"]) {
  const artifact = checkpoint.artifacts?.[name];
  assert(artifact?.path?.startsWith("data/experiments/open-generation-v1/")
    && Number.isSafeInteger(artifact.bytes) && artifact.bytes > 0
    && hash(artifact.fnv64) && sha256(artifact.sha256),
  `open-generation ${name} artifact binding is invalid`);
}
assert(Object.values(checkpoint.runner_bindings ?? {}).every(hash),
  "open-generation runner bindings are invalid");
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.open_generation_development_checkpoint_check.v1",
  checkpoint: checkpointPath,
  candidate: checkpoint.candidate,
  development_generation_passed: false,
  promotion_passed: false,
})}\n`);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
