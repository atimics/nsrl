#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const CONTRACT = "open-generation-v1";
const RESULT_SCHEMA = "nsrl.open_generation_development_result.v1";
const SAMPLE_SCHEMA = "nsrl.open_generation_sample.v1";
const RUN_SCHEMA = "nsrl.open_generation_run.v1";
const MANIFEST_HEADER = "schema\tcontract\ttokenizer\ttokenizer_hash\tdevelopment_panel\tdevelopment_panel_hash\thidden_test_sha256\tprompt_count\tmax_prompt_bytes\tgeneration_tokens\tsampling_seeds\tretained_improvement_per_mille\tmax_repeat_4gram_share_per_mille\tmin_unique_4gram_share_per_mille\tmin_entropy_q10\tmin_utf8_valid_per_mille\tmin_context_use_per_mille\tmin_distractor_resistance_per_mille\tmin_human_preference_delta_per_mille";
const PANEL_HEADER = "schema\tcontract\tpartition\tid\tcategory\tmax_new_tokens\trequired_phrase_hex\tprompt_hex";
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

const config = parseArgs(process.argv.slice(2));
const manifestBytes = fs.readFileSync(config.manifest);
const manifest = parseManifest(manifestBytes.toString("utf8"), config.manifest);
const runBytes = fs.readFileSync(config.run);
const run = JSON.parse(runBytes);
const sampleBytes = fs.readFileSync(config.samples);
const samples = sampleBytes.toString("utf8").trimEnd().split("\n")
  .filter(Boolean).map((line) => JSON.parse(line));
const decoderTraceBytes = fs.readFileSync(config.decoderTraces);
const decoderTraces = decoderTraceBytes.toString("utf8").trimEnd().split("\n")
  .filter(Boolean).map((line) => JSON.parse(line));
const modelingBytes = config.modeling ? fs.readFileSync(config.modeling) : null;
const modeling = modelingBytes ? JSON.parse(modelingBytes) : null;
const runnerBinaryBytes = fs.readFileSync(config.runnerBinary);
const candidateModelBytes = fs.readFileSync(config.candidateModel);
const candidateTokenizerBytes = fs.readFileSync(config.candidateTokenizer);
const modelingRunnerBinaryBytes = modeling ? fs.readFileSync(config.modelingRunnerBinary) : null;
const runnerSource = fs.readFileSync("crates/nsrl-train/src/bin/nsrl-open-generation-run.rs");
const modelingRunnerSource = modeling
  ? fs.readFileSync("crates/nsrl-train/src/bin/nsrl-open-generation-modeling-run.rs")
  : null;

assert(run.schema === RUN_SCHEMA && run.contract === CONTRACT,
  "open-generation run schema or contract is invalid");
assert(run.partition === "development"
  && run.execution === "incremental_linear_attention_cache_v1",
"open-generation run is not an incremental development run");
assert(run.bindings?.samples_fnv64 === hex64(fnv64(sampleBytes)),
  "open-generation sample ledger hash mismatch");
assert(run.bindings?.manifest_fnv64 === hex64(fnv64(manifestBytes)),
  "open-generation manifest hash mismatch");
assert(run.bindings?.decoder_traces_fnv64 === hex64(fnv64(decoderTraceBytes)),
  "open-generation decoder trace ledger hash mismatch");
assert(run.bindings?.runner_source_fnv64 === hex64(fnv64(runnerSource)),
  "open-generation runner source hash mismatch");
assert(run.bindings?.runner_binary_fnv64 === hex64(fnv64(runnerBinaryBytes)),
  "open-generation runner binary hash mismatch");
assert(run.bindings?.contract_tokenizer_fnv64 === manifest.tokenizerHash,
  "open-generation contract tokenizer binding mismatch");
assert(run.sampling?.greedy === true && Number.isSafeInteger(run.sampling.top_k)
  && run.sampling.top_k >= 2
  && JSON.stringify(run.sampling.seeds) === JSON.stringify(manifest.seeds),
"open-generation run sampling matrix does not match the manifest");

const expectedModes = new Map([["greedy", new Set([0])], ["sample", new Set(manifest.seeds)]]);
const expectedSamples = manifest.promptCount * (manifest.seeds.length + 1);
assert(samples.length === expectedSamples, `expected ${expectedSamples} samples, got ${samples.length}`);
assert(decoderTraces.length === expectedSamples,
  `expected ${expectedSamples} decoder traces, got ${decoderTraces.length}`);
const sampleKeys = new Set();
let totalFourGrams = 0;
let totalUniqueFourGrams = 0;
let maxRepeatFourGramSharePerMille = 0;
let minUniqueFourGramSharePerMille = 1000;
let minEntropyQ10 = Number.MAX_SAFE_INTEGER;
let validUtf8 = 0;
let requiredPhraseSamples = 0;
let requiredPhraseHits = 0;
let distractorSamples = 0;
let distractorHits = 0;
let samplesBeyondTrainingContext = 0;
let residualSaturationCount = 0;
let maximumCacheStateBytes = 0;
let maximumCacheWorkspaceBytes = 0;
let maximumCacheTokens = 0;
const categories = new Set();
const promptIds = new Set();
const modelHashes = new Set();
const tokenizerHashes = new Set();

for (const [sampleIndex, sample] of samples.entries()) {
  const decoderTrace = decoderTraces[sampleIndex];
  assert(sample.schema === SAMPLE_SCHEMA && sample.contract === CONTRACT
    && sample.partition === "development", "sample schema, contract, or partition is invalid");
  assert(expectedModes.has(sample.mode)
    && expectedModes.get(sample.mode).has(sample.seed), "sample mode or seed is invalid");
  const prompt = manifest.prompts.get(sample.prompt_id);
  assert(prompt && sample.category === prompt.category
    && sample.required_phrase_hex === prompt.requiredPhraseHex,
  `${sample.prompt_id} does not match the frozen development panel`);
  assert(sample.top_k === (sample.mode === "greedy" ? 1 : run.sampling.top_k),
    `${sample.prompt_id} ${sample.mode} has the wrong top-k`);
  const key = `${sample.prompt_id}\0${sample.mode}\0${sample.seed}`;
  assert(!sampleKeys.has(key), `duplicate sample ${key}`);
  sampleKeys.add(key);
  promptIds.add(sample.prompt_id);
  categories.add(sample.category);
  modelHashes.add(sample.bindings?.model_fnv64);
  tokenizerHashes.add(sample.bindings?.tokenizer_fnv64);
  assert(Array.isArray(sample.generated_tokens)
    && sample.generated_tokens.length === manifest.generationTokens
    && sample.generated_token_count === manifest.generationTokens,
  `${sample.prompt_id} ${sample.mode} has the wrong generated-token count`);
  assert(sample.execution?.decoder === "incremental_linear_attention_cache_v1",
    `${sample.prompt_id} ${sample.mode} did not use the incremental decoder`);
  assert(sample.generated_token_fnv64 === hex64(hashTokens(sample.generated_tokens)),
    `${sample.prompt_id} ${sample.mode} generated-token hash mismatch`);
  assert(decoderTrace.schema === "nsrl.production_generation.v2"
    && decoderTrace.execution === "incremental_linear_attention_cache_v1",
  `${sample.prompt_id} ${sample.mode} decoder trace is invalid`);
  assert(decoderTrace.bindings?.model_hash === sample.bindings.model_fnv64
    && decoderTrace.bindings?.tokenizer_hash === sample.bindings.tokenizer_fnv64,
  `${sample.prompt_id} ${sample.mode} decoder binding mismatch`);
  assert(JSON.stringify(decoderTrace.generation?.tokens) === JSON.stringify(sample.generated_tokens)
    && decoderTrace.generation?.token_hash === sample.generated_token_fnv64,
  `${sample.prompt_id} ${sample.mode} decoder tokens do not match the sample ledger`);
  assert(decoderTrace.prompt?.token_count === sample.prompt_token_count
    && decoderTrace.generation?.cache?.state_bytes === sample.execution.cache_state_bytes
    && decoderTrace.generation?.cache?.workspace_bytes === sample.execution.cache_workspace_bytes
    && decoderTrace.generation?.cache?.tokens_processed === sample.execution.cache_tokens_processed
    && decoderTrace.generation?.steps_beyond_training_context
      === sample.execution.steps_beyond_training_context
    && decoderTrace.generation?.residual_saturation_count
      === sample.execution.residual_saturation_count,
  `${sample.prompt_id} ${sample.mode} sample execution does not match its decoder trace`);
  assert(sample.execution.cache_state_bytes > 0 && sample.execution.cache_workspace_bytes > 0,
    `${sample.prompt_id} ${sample.mode} omitted cache accounting`);
  samplesBeyondTrainingContext += Number(sample.execution.steps_beyond_training_context > 0);
  residualSaturationCount += sample.execution.residual_saturation_count;
  maximumCacheStateBytes = Math.max(maximumCacheStateBytes, sample.execution.cache_state_bytes);
  maximumCacheWorkspaceBytes = Math.max(
    maximumCacheWorkspaceBytes, sample.execution.cache_workspace_bytes);
  maximumCacheTokens = Math.max(maximumCacheTokens, sample.execution.cache_tokens_processed);

  const fourGrams = sample.generated_tokens.length - 3;
  const unique = new Set();
  for (let index = 0; index < fourGrams; index += 1) {
    unique.add(sample.generated_tokens.slice(index, index + 4).join(","));
  }
  const uniqueShare = Math.floor(unique.size * 1000 / fourGrams);
  const repeatShare = 1000 - uniqueShare;
  totalFourGrams += fourGrams;
  totalUniqueFourGrams += unique.size;
  maxRepeatFourGramSharePerMille = Math.max(maxRepeatFourGramSharePerMille, repeatShare);
  minUniqueFourGramSharePerMille = Math.min(minUniqueFourGramSharePerMille, uniqueShare);
  minEntropyQ10 = Math.min(minEntropyQ10, entropyQ10(sample.generated_tokens));

  const generated = decodeHex(sample.generated_hex, "generated_hex");
  let utf8 = "";
  try {
    utf8 = new TextDecoder("utf-8", {fatal: true}).decode(generated);
    validUtf8 += 1;
  } catch {
    utf8 = "";
  }
  if (sample.required_phrase_hex) {
    requiredPhraseSamples += 1;
    const required = decodeHex(sample.required_phrase_hex, "required_phrase_hex")
      .toString("utf8").toLocaleLowerCase("en-US");
    const hit = utf8.toLocaleLowerCase("en-US").includes(required);
    requiredPhraseHits += Number(hit);
    if (sample.category === "long-context-reference") {
      distractorSamples += 1;
      distractorHits += Number(hit);
    }
  }
}

assert(promptIds.size === manifest.promptCount
  && [...promptIds].every((id) => manifest.prompts.has(id)),
"sample ledger is missing frozen prompt ids");
assert(modelHashes.size === 1 && !modelHashes.has(undefined), "samples do not bind one model");
assert(tokenizerHashes.size === 1 && !tokenizerHashes.has(undefined),
  "samples do not bind one candidate tokenizer");
assert([...modelHashes][0] === run.bindings?.candidate_model_fnv64
  && [...tokenizerHashes][0] === run.bindings?.candidate_tokenizer_fnv64,
"sample candidate bindings do not match the run");
if (modeling) {
  assert(modeling.schema === "nsrl.open_generation_modeling.v1"
    && modeling.contract === CONTRACT && modeling.partition === "development"
    && modeling.objective === "integer_base2_softmax_nll_per_original_utf8_byte"
    && modeling.sequence_policy
      === "reset_per_prompt_bos_then_score_candidate_tokens_no_eos",
  "candidate modeling trace schema or policy is invalid");
  assert(modeling.bindings?.manifest_fnv64 === run.bindings.manifest_fnv64
    && modeling.bindings?.development_panel_fnv64 === hex64(fnv64(manifest.panelBytes))
    && modeling.bindings?.contract_tokenizer_fnv64 === manifest.tokenizerHash
    && modeling.bindings?.candidate_model_hash === [...modelHashes][0]
    && modeling.bindings?.candidate_tokenizer_hash === [...tokenizerHashes][0]
    && modeling.bindings?.candidate_model_artifact_fnv64
      === hex64(fnv64(candidateModelBytes))
    && modeling.bindings?.candidate_tokenizer_artifact_fnv64
      === hex64(fnv64(candidateTokenizerBytes))
    && modeling.bindings?.runner_source_fnv64
      === hex64(fnv64(modelingRunnerSource))
    && modeling.bindings?.runner_binary_fnv64
      === hex64(fnv64(modelingRunnerBinaryBytes)),
  "candidate modeling trace bindings do not match the generation run");
  assert(modeling.counts?.prompts === manifest.promptCount
    && modeling.counts?.original_utf8_bytes > 0 && modeling.counts?.candidate_tokens > 0
    && modeling.metrics?.total_nll_millibits > 0
    && modeling.metrics?.millibits_per_original_utf8_byte > 0
    && modeling.residual_saturation_count === 0,
  "candidate modeling trace counts, metric, or health are invalid");
}
assert(run.counts?.prompts === promptIds.size && run.counts?.samples === samples.length
  && run.counts?.generated_tokens === samples.length * manifest.generationTokens,
"run counts do not match the sample ledger");
assert(run.counts?.samples_beyond_training_context === samplesBeyondTrainingContext
  && run.residual_saturation_count === residualSaturationCount
  && run.cache?.maximum_state_bytes === maximumCacheStateBytes
  && run.cache?.maximum_workspace_bytes === maximumCacheWorkspaceBytes
  && run.cache?.maximum_tokens_processed === maximumCacheTokens,
"run cache or health totals do not match the sample ledger");
assert(categories.size === 6, "sample ledger is missing a required category");
assert(requiredPhraseSamples > 0 && distractorSamples > 0,
  "sample ledger does not exercise context and distractor metrics");

const metrics = {
  max_repeat_4gram_share_per_mille: maxRepeatFourGramSharePerMille,
  min_unique_4gram_share_per_mille: minUniqueFourGramSharePerMille,
  aggregate_unique_4gram_share_per_mille:
    Math.floor(totalUniqueFourGrams * 1000 / totalFourGrams),
  min_entropy_q10: minEntropyQ10,
  utf8_valid_per_mille: Math.floor(validUtf8 * 1000 / samples.length),
  context_use_per_mille: Math.floor(requiredPhraseHits * 1000 / requiredPhraseSamples),
  distractor_resistance_per_mille: Math.floor(distractorHits * 1000 / distractorSamples),
};
const gates = {
  complete_generation_matrix: sampleKeys.size === expectedSamples,
  incremental_cached_decoding: samplesBeyondTrainingContext === samples.length,
  no_residual_saturation: residualSaturationCount === 0,
  repeat_4gram_health:
    metrics.max_repeat_4gram_share_per_mille <= manifest.maxRepeatFourGramSharePerMille,
  unique_4gram_health:
    metrics.min_unique_4gram_share_per_mille >= manifest.minUniqueFourGramSharePerMille,
  entropy_health: metrics.min_entropy_q10 >= manifest.minEntropyQ10,
  utf8_validity: metrics.utf8_valid_per_mille >= manifest.minUtf8ValidPerMille,
  context_use: metrics.context_use_per_mille >= manifest.minContextUsePerMille,
  distractor_resistance:
    metrics.distractor_resistance_per_mille >= manifest.minDistractorResistancePerMille,
  forbidden_assistance_absent: Object.values(run.forbidden_assistance ?? {}).every((value) => value === false),
};
const developmentGenerationPassed = Object.values(gates).every(Boolean);
const missingEvidence = [
  modeling
    ? "required_baseline_bits_per_original_utf8_byte"
    : "candidate_and_required_baseline_bits_per_original_utf8_byte",
  "retained_float_twin_improvement",
  "blinded_human_preference",
  "hidden_panel_generation",
  "candidate_integer_transformer_proof_v1",
];
const result = {
  schema: RESULT_SCHEMA,
  contract: CONTRACT,
  partition: "development",
  candidate: {
    model_fnv64: [...modelHashes][0],
    tokenizer_fnv64: [...tokenizerHashes][0],
  },
  sources: {
    manifest: binding(config.manifest, manifestBytes),
    development_panel: binding(manifest.panelPath, manifest.panelBytes),
    run: binding(config.run, runBytes),
    samples: binding(config.samples, sampleBytes),
    decoder_traces: binding(config.decoderTraces, decoderTraceBytes),
    generation_runner_binary: binding(config.runnerBinary, runnerBinaryBytes),
    candidate_model: binding(config.candidateModel, candidateModelBytes),
    candidate_tokenizer: binding(config.candidateTokenizer, candidateTokenizerBytes),
    ...(modelingRunnerBinaryBytes ? {
      modeling_runner_binary: binding(config.modelingRunnerBinary, modelingRunnerBinaryBytes),
    } : {}),
    ...(modelingBytes ? {modeling: binding(config.modeling, modelingBytes)} : {}),
  },
  counts: {
    prompts: promptIds.size,
    samples: samples.length,
    generated_tokens: samples.length * manifest.generationTokens,
    samples_beyond_training_context: samplesBeyondTrainingContext,
    required_phrase_samples: requiredPhraseSamples,
    required_phrase_hits: requiredPhraseHits,
    distractor_samples: distractorSamples,
    distractor_hits: distractorHits,
  },
  thresholds: manifest.thresholds,
  metrics,
  modeling: modeling ? {
    sequence_policy: modeling.sequence_policy,
    original_utf8_bytes: modeling.counts.original_utf8_bytes,
    candidate_tokens: modeling.counts.candidate_tokens,
    total_nll_millibits: modeling.metrics.total_nll_millibits,
    millibits_per_original_utf8_byte:
      modeling.metrics.millibits_per_original_utf8_byte,
    required_baselines_measured: false,
  } : null,
  gates: {...gates, development_generation_passed: developmentGenerationPassed},
  evidence_layers: {
    generation_health_measured: true,
    candidate_modeling_measured: Boolean(modeling),
    modeling_measured: false,
    blinded_human_product_quality_measured: false,
    hidden_panel_measured: false,
    integer_transformer_proof_v1_bound: false,
  },
  missing_evidence: missingEvidence,
  promotion_passed: false,
};
const output = `${JSON.stringify(result, null, 2)}\n`;
if (config.check) {
  assert(fs.readFileSync(config.check, "utf8") === output,
    "open-generation development result does not byte-replay");
} else {
  assert(config.out, "--out is required unless --check is used");
  fs.mkdirSync(path.dirname(config.out), {recursive: true});
  fs.writeFileSync(config.out, output);
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.open_generation_development_evaluation.v1",
  checked: Boolean(config.check),
  development_generation_passed: developmentGenerationPassed,
  promotion_passed: false,
  metrics,
  failed_gates: Object.entries(gates).filter(([, passed]) => !passed).map(([gate]) => gate),
  missing_evidence: missingEvidence,
  output: config.check || config.out,
})}\n`);

function parseArgs(args) {
  const value = {
    manifest: "benchmarks/open-generation-v1/manifest.tsv",
    run: "",
    samples: "",
    decoderTraces: "",
    modeling: "",
    runnerBinary: "",
    modelingRunnerBinary: "",
    candidateModel: "",
    candidateTokenizer: "",
    out: "",
    check: "",
  };
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--manifest") value.manifest = args[++index] || "";
    else if (args[index] === "--run") value.run = args[++index] || "";
    else if (args[index] === "--samples") value.samples = args[++index] || "";
    else if (args[index] === "--decoder-traces") value.decoderTraces = args[++index] || "";
    else if (args[index] === "--modeling") value.modeling = args[++index] || "";
    else if (args[index] === "--runner-binary") value.runnerBinary = args[++index] || "";
    else if (args[index] === "--modeling-runner-binary") {
      value.modelingRunnerBinary = args[++index] || "";
    } else if (args[index] === "--candidate-model") value.candidateModel = args[++index] || "";
    else if (args[index] === "--candidate-tokenizer") {
      value.candidateTokenizer = args[++index] || "";
    } else if (args[index] === "--out") value.out = args[++index] || "";
    else if (args[index] === "--check") value.check = args[++index] || "";
    else throw new Error(`unknown argument ${args[index]}`);
  }
  assert(value.manifest && value.run && value.samples && value.decoderTraces
    && value.runnerBinary && value.candidateModel && value.candidateTokenizer,
  "--manifest, --run, --samples, --decoder-traces, --runner-binary, --candidate-model, and --candidate-tokenizer are required");
  assert(!value.modeling || value.modelingRunnerBinary,
    "--modeling-runner-binary is required with --modeling");
  assert(Boolean(value.out) !== Boolean(value.check), "use exactly one of --out or --check RESULT");
  return value;
}

function parseManifest(text, manifestPath) {
  const lines = text.trimEnd().split("\n");
  assert(lines.length === 2 && lines[0] === MANIFEST_HEADER,
    "open-generation manifest header or row count is invalid");
  const fields = lines[1].split("\t");
  assert(fields.length === 19 && fields[1] === CONTRACT, "open-generation manifest is invalid");
  const integer = (index, name) => {
    assert(/^-?[0-9]+$/.test(fields[index]), `invalid ${name}`);
    return Number(fields[index]);
  };
  const thresholds = {
    retained_improvement_per_mille: integer(11, "retained improvement"),
    max_repeat_4gram_share_per_mille: integer(12, "repeat threshold"),
    min_unique_4gram_share_per_mille: integer(13, "unique threshold"),
    min_entropy_q10: integer(14, "entropy threshold"),
    min_utf8_valid_per_mille: integer(15, "UTF-8 threshold"),
    min_context_use_per_mille: integer(16, "context threshold"),
    min_distractor_resistance_per_mille: integer(17, "distractor threshold"),
    min_human_preference_delta_per_mille: integer(18, "human threshold"),
  };
  assert(fields[4] && !path.isAbsolute(fields[4]) && !fields[4].split(/[\\/]/).includes(".."),
    "development panel path must stay under the manifest directory");
  const panelPath = path.join(path.dirname(manifestPath), fields[4]);
  const panelBytes = fs.readFileSync(panelPath);
  assert(hex64(fnv64(panelBytes)) === fields[5], "development panel hash mismatch");
  const prompts = parsePanel(panelBytes.toString("utf8"), integer(7, "prompt count"),
    integer(9, "generation tokens"));
  return {
    promptCount: integer(7, "prompt count"),
    generationTokens: integer(9, "generation tokens"),
    seeds: fields[10].split(",").map((seed) => Number(seed)),
    tokenizerHash: fields[3],
    panelPath,
    panelBytes,
    prompts,
    maxRepeatFourGramSharePerMille: thresholds.max_repeat_4gram_share_per_mille,
    minUniqueFourGramSharePerMille: thresholds.min_unique_4gram_share_per_mille,
    minEntropyQ10: thresholds.min_entropy_q10,
    minUtf8ValidPerMille: thresholds.min_utf8_valid_per_mille,
    minContextUsePerMille: thresholds.min_context_use_per_mille,
    minDistractorResistancePerMille: thresholds.min_distractor_resistance_per_mille,
    thresholds,
  };
}

function parsePanel(text, expectedPrompts, generationTokens) {
  const lines = text.trimEnd().split("\n");
  assert(lines[0] === PANEL_HEADER && lines.length === expectedPrompts + 1,
    "development panel header or prompt count is invalid");
  const prompts = new Map();
  for (const line of lines.slice(1)) {
    const fields = line.split("\t");
    assert(fields.length === 8 && fields[0] === "nsrl.open_generation_prompt.v1"
      && fields[1] === CONTRACT && fields[2] === "development"
      && /^[a-z0-9-]+$/.test(fields[3]) && /^[a-z-]+$/.test(fields[4])
      && Number(fields[5]) === generationTokens,
    "development panel row is invalid");
    assert(!prompts.has(fields[3]), `duplicate development prompt ${fields[3]}`);
    const requiredPhraseHex = fields[6] === "-" ? "" : fields[6];
    decodeHex(requiredPhraseHex, "panel required_phrase_hex");
    assert(decodeHex(fields[7], "panel prompt_hex").length > 0,
      "development prompt must not be empty");
    prompts.set(fields[3], {category: fields[4], requiredPhraseHex});
  }
  return prompts;
}

function entropyQ10(tokens) {
  const counts = new Map();
  for (const token of tokens) counts.set(token, (counts.get(token) ?? 0) + 1);
  let entropy = 0;
  for (const count of counts.values()) {
    const probability = count / tokens.length;
    entropy -= probability * Math.log2(probability);
  }
  return Math.round(entropy * 1024);
}

function decodeHex(value, name) {
  assert(typeof value === "string" && value.length % 2 === 0
    && /^[0-9a-f]*$/.test(value), `${name} is not lowercase hexadecimal`);
  return Buffer.from(value, "hex");
}

function binding(file, bytes) {
  return {path: file, bytes: bytes.length, fnv64: hex64(fnv64(bytes))};
}

function fnv64(bytes) {
  let hash = FNV_OFFSET;
  for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * FNV_PRIME) & FNV_MASK;
  return hash;
}

function hashTokens(tokens) {
  const bytes = Buffer.alloc(tokens.length * 4);
  for (const [index, token] of tokens.entries()) bytes.writeUInt32LE(token, index * 4);
  return fnv64(bytes);
}

function hex64(value) {
  return `0x${value.toString(16).padStart(16, "0")}`;
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
