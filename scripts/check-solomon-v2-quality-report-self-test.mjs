#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_v2_quality_report_self_test.v1";
const SPIRIT_COUNT = 2;
const REQUIRED_TASKS = [
  "canonical-joint",
  "identify",
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
  "match",
];
const REVERSE_IMAGE_RETRIEVAL_TASKS = [
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
];
const FORWARD_IMAGE_PLAN_TASKS = [
  "text-to-image",
  "description-to-image",
];
const IMAGE_RETRIEVAL_TASKS = [
  ...FORWARD_IMAGE_PLAN_TASKS,
  ...REVERSE_IMAGE_RETRIEVAL_TASKS,
];
const IMAGE_RETRIEVAL_TASK_COUNTS = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const CHANNELS = ["ink", "edge", "component", "radial", "direction"];
const TOKEN_LAYOUT = {
  bos: 1,
  prompt: 2,
  text: 3,
  image: 4,
  eos: 5,
  task_text_to_image: 6,
  task_image_to_text: 7,
  task_match: 8,
  task_explain: 9,
  task_identify: 10,
  image_channel_ink: 11,
  image_channel_edge: 12,
  image_channel_component: 13,
  image_channel_radial: 14,
  image_channel_direction: 15,
  image_base: 144,
  image_bins: 16,
};
const IMAGE_CHANNEL_PAYLOAD_TOKENS = 256;
const RETRIEVAL_LABEL_COUNT = 72;
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

function usage() {
  console.log([
    "Usage: check-solomon-v2-quality-report-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds a tiny synthetic v2 Solomon product spine and checks that the",
    "quality-report gate accepts complete multimodal binding evidence while",
    "rejecting weak retrieval margins, broken symbolic image-token evidence,",
    "missing source grounding, bad grounded description prompts, and broken",
    "curriculum identity-binding preservation.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "", keep: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function writeText(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, text, "utf8");
}

function writeJson(filePath, value) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function fnv64BytesHex(bytes) {
  let hash = FNV64_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64StringHex(value) {
  let hash = FNV64_OFFSET;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64TokenHex(tokens) {
  return fnv64BytesHex(Buffer.from(tokens.map((token) => Number(token) & 0xff)));
}

function fileHash(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function buildFixture(root, name, mutate = () => {}) {
  const dir = path.join(root, name);
  const paths = {
    dir,
    evalPath: path.join(dir, "attention-eval.json"),
    examplesPath: path.join(dir, "examples.jsonl"),
    manifestPath: path.join(dir, "manifest.json"),
    tokensPath: path.join(dir, "corpus.tokens.u8"),
    retrievalHeadPath: path.join(dir, "retrieval-head.json"),
    retrievalEvalPath: path.join(dir, "retrieval-head-eval.json"),
    sampleBindingPath: path.join(dir, "sample-binding.json"),
    generationIntegrityPath: path.join(dir, "generation-integrity.json"),
    denoiseBridgePath: path.join(dir, "denoise-bridge.json"),
    denoiseModelPath: path.join(dir, "denoise", "model.nsrltch"),
    denoiseTracePath: path.join(dir, "denoise", "trace.json"),
    denoiseRawPath: path.join(dir, "denoise", "samples.ink128.u8"),
    generativeEvalDir: path.join(dir, "generative-eval"),
    generativeEvalSummaryPath: path.join(dir, "generative-eval", "summary.tsv"),
    generativeEvalSamplesPath: path.join(dir, "generative-eval", "samples.tsv"),
    generativeEvalConfigPath: path.join(dir, "generative-eval", "config.json"),
    generativeEvalLatentModelPath: path.join(dir, "generative-eval", "latent-decoded.nsrltch"),
    generativeEvalSamplerModelPath: path.join(dir, "generative-eval", "sampler.nsrltch"),
    identityInferencePath: path.join(dir, "identity-inference.json"),
    curriculumStagesPath: path.join(dir, "curriculum-stages.json"),
    groundedCorpusPath: path.join(dir, "grounded-corpus.json"),
    promptsPath: path.join(dir, "prompts.jsonl"),
    sourceTextPath: path.join(dir, "source.tsv"),
    qualityReportPath: path.join(dir, "quality-report.json"),
  };
  fs.mkdirSync(dir, { recursive: true });
  const state = buildState(paths);
  mutate(state, paths);
  materializeState(state, paths);
  return paths;
}

function buildState(paths) {
  const examples = [];
  const tokens = [];
  for (let spiritId = 1; spiritId <= SPIRIT_COUNT; spiritId += 1) {
    for (const task of REQUIRED_TASKS.filter((item) => item !== "match")) {
      addExample({ examples, tokens, spiritId, task });
    }
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "yes" });
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "no", negativeRole: "image" });
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "no", negativeRole: "prompt" });
  }
  const prompts = Array.from({ length: SPIRIT_COUNT }, (_unused, index) => ({
    spirit_id: index + 1,
    prompt_hash: `synthetic_prompt_${String(index + 1).padStart(3, "0")}`,
    tier: "holdout",
    source: "expanded",
    text: `holdout binding prompt ${index + 1}`,
  }));
  const sourceRows = Array.from({ length: SPIRIT_COUNT }, (_unused, index) => ({
    spirit_id: index + 1,
    primary_name: `Synthetic Solomon ${index + 1}`,
    source_text: `source passage for synthetic solomon ${index + 1} with seal attributes and binding notes`,
  }));

  return {
    examples,
    tokens,
    prompts,
    sourceRows,
    evalTrace: syntheticEvalTrace(),
    manifest: syntheticManifest(examples.length),
    retrievalHead: syntheticRetrievalHead(),
    retrievalEval: syntheticRetrievalEval(paths),
    sampleBinding: syntheticSampleBinding(),
    generationIntegrity: {
      schema: "nsrl.solomon_generation_integrity_check.v1",
      ok: true,
      trace_count: 1,
      violations: [],
    },
    denoiseBridge: null,
    generativeEval: null,
    identityInference: syntheticIdentityInference(paths),
    curriculumStages: syntheticCurriculumStages(paths),
    groundedCorpus: syntheticGroundedCorpus(paths),
  };
}

function addExample({ examples, tokens, spiritId, task, matchLabel = "", negativeRole = "" }) {
  const slice = tokenSliceForTask(task, spiritId, matchLabel);
  const offset = tokens.length;
  tokens.push(...slice);
  const row = {
    schema: "nsrl.solomon_multimodal_example.v2",
    spirit_id: spiritId,
    task,
    text: task === "match" ? matchLabel : `synthetic prompt ${spiritId}`,
    image_token_profile: "symbolic16",
    image_token_channels: CHANNELS,
    token_offset: offset,
    token_count: slice.length,
    token_hash: fnv64TokenHex(slice),
  };
  if (task === "match") {
    row.match_label = matchLabel;
    if (matchLabel === "no") {
      row.negative_spirit_id = spiritId === 1 ? 2 : 1;
      row.negative_role = negativeRole;
      row.negative_selection = "nearest-image-token";
      row.negative_image_token_rank = 1;
      row.negative_image_token_distance = 100 + spiritId;
    }
  }
  examples.push(row);
}

function tokenSliceForTask(task, spiritId, matchLabel) {
  const promptPayload = 32 + spiritId;
  const textPayload = matchLabel === "no" ? 82 + spiritId : 64 + spiritId;
  const image = imagePayload(spiritId);
  if (task === "canonical-joint") return [1, 2, promptPayload, 3, textPayload, 4, ...image, 5];
  if (task === "identify") return [1, 10, 2, promptPayload, 3, textPayload, 5];
  if (task === "text-to-image") return [1, 6, 2, promptPayload, 4, ...image, 5];
  if (task === "description-to-image") return [1, 6, 2, promptPayload, 4, ...image, 5];
  if (task === "image-to-text") return [1, 7, 4, ...image, 3, textPayload, 5];
  if (task === "image-to-explain") return [1, 9, 4, ...image, 3, textPayload, 5];
  if (task === "image-to-attributes") return [1, 9, 4, ...image, 2, promptPayload, 3, textPayload, 5];
  if (task === "text-image-explain") return [1, 9, 2, promptPayload, 4, ...image, 3, textPayload, 5];
  if (task === "explain") return [1, 9, 2, promptPayload, 3, textPayload, 5];
  if (task === "match") return [1, 8, 2, promptPayload, 4, ...image, 3, textPayload, 5];
  throw new Error(`unknown task ${task}`);
}

function imagePayload(spiritId) {
  const payload = [];
  for (const [channelIndex, channel] of CHANNELS.entries()) {
    payload.push(TOKEN_LAYOUT[`image_channel_${channel}`]);
    for (let index = 0; index < IMAGE_CHANNEL_PAYLOAD_TOKENS; index += 1) {
      payload.push(TOKEN_LAYOUT.image_base + ((index + spiritId + channelIndex) % TOKEN_LAYOUT.image_bins));
    }
  }
  return payload;
}

function syntheticManifest(exampleCount) {
  return {
    schema: "nsrl.solomon_multimodal_manifest.v2",
    corpus_version: "v2",
    rows: SPIRIT_COUNT,
    examples: exampleCount,
    training_sequences: exampleCount,
    image_token_profile: "symbolic16",
    image_token_channels: CHANNELS,
    signature_bins: IMAGE_CHANNEL_PAYLOAD_TOKENS,
    corpus_tokens_u8: "corpus.tokens.u8",
    source_text_index: "source.tsv",
    token_layout: TOKEN_LAYOUT,
    image_token_channel_stats: Object.fromEntries(
      CHANNELS.map((channel) => [
        channel,
        {
          records: SPIRIT_COUNT,
          tokens_per_record: IMAGE_CHANNEL_PAYLOAD_TOKENS,
          active_records: SPIRIT_COUNT,
          multi_bin_records: SPIRIT_COUNT,
          nonzero_tokens: SPIRIT_COUNT * IMAGE_CHANNEL_PAYLOAD_TOKENS,
          distinct_bins: TOKEN_LAYOUT.image_bins,
          max_bin: TOKEN_LAYOUT.image_bins - 1,
          unique_record_hashes: SPIRIT_COUNT,
          duplicate_record_hashes: 0,
        },
      ]),
    ),
    token_hash: "",
  };
}

function syntheticEvalTrace() {
  const taskPhases = {
    "canonical-joint": phaseStats(["prompt", "text", "image"]),
    identify: phaseStats(["prompt", "text"]),
    "text-to-image": phaseStats(["prompt", "image"]),
    "image-to-text": phaseStats(["image", "text"]),
    "image-to-explain": phaseStats(["image", "text"]),
    "text-image-explain": phaseStats(["prompt", "image", "text"]),
    "image-to-attributes": phaseStats(["image", "prompt", "text"]),
    explain: phaseStats(["prompt", "text"]),
    "description-to-image": phaseStats(["prompt", "image"]),
    match: phaseStats(["prompt", "image", "text"]),
  };
  return {
    schema: "nsrl.solomon_attention_eval_trace.v1",
    model: "synthetic-attention.nsrllmm",
    model_hash: "0xsyntheticattention",
    token_hash: "0xsynthetictokens",
    example_count: 24,
    skipped_examples: 0,
    d_model: 128,
    heads: 2,
    hidden_dim: 256,
    transformer_layers: 2,
    context_seq_len: 384,
    total: stat(32),
    special: stat(10),
    prompt: stat(20),
    text: stat(18),
    image: stat(14),
    tasks: Object.fromEntries(REQUIRED_TASKS.map((task) => [task, stat(task === "match" ? 6 : 2)])),
    task_phases: taskPhases,
    output_heads: {
      special_head: outputHead(["special"], [[1, 16]], 10),
      text_head: outputHead(["prompt", "text"], [[16, 144]], 38),
      image_head: outputHead(["image-channel", "image-bin"], [[11, 16], [144, 160]], 14),
    },
  };
}

function phaseStats(phases) {
  return Object.fromEntries(phases.map((phase) => [phase, stat(1)]));
}

function outputHead(tokenClasses, tokenRanges, targets) {
  return {
    source: "nsrllmm-output-token-head",
    token_classes: tokenClasses,
    token_ranges: tokenRanges,
    allowed_token_count: tokenRanges.reduce((sum, [start, end]) => sum + end - start, 0),
    stats: stat(targets),
  };
}

function stat(targets) {
  return {
    targets,
    correct: targets,
    invalid_contexts: 0,
    accuracy_per_mille: 1000,
    top5_accuracy_per_mille: 1000,
    top10_accuracy_per_mille: 1000,
    mean_target_rank_per_mille: 0,
    mean_target_margin_q8: 256,
  };
}

function syntheticRetrievalHead() {
  const labels = Array.from({ length: RETRIEVAL_LABEL_COUNT }, (_unused, index) => ({
    label: index,
    spirit_id: index + 1,
    primary_name: `Synthetic Solomon ${index + 1}`,
  }));
  const imageHead = sparseHead(labels.length, 1);
  imageHead.biases[0] = 10000;
  imageHead.weights = Array.from({ length: labels.length }, () => []);
  const head = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    feature_count: 4,
    labels,
    text_head: sparseHead(labels.length, 0),
    image_head: imageHead,
  };
  head.model_hash = fnv64StringHex(JSON.stringify(head));
  return head;
}

function sparseHead(labelCount, featureBase) {
  return {
    biases: Array.from({ length: labelCount }, () => 0),
    weights: Array.from({ length: labelCount }, (_unused, index) => [[featureBase + (index % 2), index + 1]]),
  };
}

function syntheticRetrievalEval(paths) {
  return {
    schema: "nsrl.solomon_v2_retrieval_head_eval.v1",
    ok: true,
    errors: [],
    model: paths.retrievalHeadPath,
    model_hash: "",
    feature_count: 4,
    examples: paths.examplesPath,
    examples_hash: "",
    tokens: paths.tokensPath,
    tokens_hash: "",
    prompts: paths.promptsPath,
    prompts_hash: "",
    prompt_rows_total: SPIRIT_COUNT,
    heldout_prompt_rows: SPIRIT_COUNT,
    heldout_prompt_unique_targets: SPIRIT_COUNT,
    known_prompts: retrievalMetric(SPIRIT_COUNT),
    identity_bindings: {
      required_kinds: REQUIRED_IDENTITY_BINDING_KINDS,
      total: retrievalMetric(SPIRIT_COUNT * REQUIRED_IDENTITY_BINDING_KINDS.length),
      by_kind: Object.fromEntries(REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [kind, retrievalMetric(SPIRIT_COUNT)])),
    },
    heldout_prompts: retrievalMetric(SPIRIT_COUNT),
    image_to_text: retrievalMetric(SPIRIT_COUNT * REVERSE_IMAGE_RETRIEVAL_TASKS.length),
    image_tasks: Object.fromEntries(
      IMAGE_RETRIEVAL_TASKS.map((task) => [task, retrievalMetric(IMAGE_RETRIEVAL_TASK_COUNTS[task])]),
    ),
    match: {
      yes: retrievalMetric(SPIRIT_COUNT),
      no: retrievalMetric(SPIRIT_COUNT * 2),
      no_by_role: {
        image: retrievalMetric(SPIRIT_COUNT),
        prompt: retrievalMetric(SPIRIT_COUNT),
      },
    },
  };
}

function retrievalMetric(count, margin = 10) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: margin,
    mean_margin: margin + 2,
  };
}

function syntheticSampleBinding() {
  return {
    schema: "nsrl.solomon_attention_sample_binding_check.v1",
    ok: true,
    errors: [],
    retrieval_head: null,
    retrieval_head_model_hash: "",
    samples: SPIRIT_COUNT,
    text_image_agreement: true,
    generated_text_image_agreement: true,
    signature_retrieval_agreement: true,
    image_to_text_identification: true,
    generated_text_identification: true,
    min_signature_margin: 10,
    min_retrieval_image_margin: 10,
    min_image_to_text_margin: 10,
    min_retrieval_text_margin: 10,
    min_generated_text_margin: 10,
    results: Array.from({ length: SPIRIT_COUNT }, (_entry, index) => {
      const spiritId = index + 1;
      return {
        sample_dir: `synthetic-sample-${spiritId}`,
        prompt: `Synthetic Solomon ${spiritId}`,
        image_ink16_u8: `synthetic-sample-${spiritId}/image.ink16.u8`,
        expected_spirit_id: spiritId,
        expected_primary_name: `Synthetic Solomon ${spiritId}`,
        generated_text: `Synthetic Solomon ${spiritId} source-bound answer`,
        image_to_text_identity: { spirit_id: spiritId, margin: 10 },
        generated_text_identity: { spirit_id: spiritId, margin: 10 },
        generated_text_image_agree: true,
        confidence: { margin: 10 },
      };
    }),
  };
}

function enableDenoiseBridge(state, paths) {
  state.denoiseBridge = syntheticDenoiseBridge(state, paths);
}

function syntheticDenoiseBridge(state, paths) {
  const plan = syntheticAttentionPlan();
  const inkRange = Math.max(...plan) - Math.min(...plan);
  const denoiseModelHash = fnv64BytesHex(syntheticDenoiseModelBytes());
  const results = state.sampleBinding.results.slice(0, 1).map((sample, index) => {
    const outputDetail = {
      index,
      signature_distance: 0,
      ink_range: inkRange,
      expected_spirit_id: sample.expected_spirit_id,
      expected_primary_name: sample.expected_primary_name,
      retrieval_image_rank: 1,
      retrieval_image_margin: 10000,
      retrieval_image_top1_spirit_id: sample.expected_spirit_id,
      retrieval_image_top1_primary_name: sample.expected_primary_name,
      image_to_text_identity: true,
    };
    return {
      ok: true,
      prompt: sample.prompt,
      expected_spirit_id: sample.expected_spirit_id,
      expected_primary_name: sample.expected_primary_name,
      attention_plan: sample.image_ink16_u8,
      denoise_trace: "denoise/trace.json",
      denoise_model: "denoise/model.nsrltch",
      denoise_model_hash: denoiseModelHash,
      denoise_raw_samples: "denoise/samples.ink128.u8",
      trace_integrity: { ok: true, violations: [] },
      output_signature: {
        samples: 1,
        min_signature_distance: 0,
        mean_signature_distance_q8: 0,
        min_ink_range: inkRange,
        output_image_to_text_identification: true,
        min_retrieval_image_margin: 10000,
        samples_detail: [outputDetail],
      },
    };
  });
  const expectedSpiritIds = results.map((result) => Number(result.expected_spirit_id));
  const uniqueExpectedSpiritIds = [...new Set(expectedSpiritIds)].sort((left, right) => left - right);
  return {
    schema: "nsrl.solomon_attention_denoise_bridge_check.v1",
    ok: true,
    errors: [],
    pairs: results.length,
    min_unique_targets: 1,
    expected_spirit_ids: expectedSpiritIds,
    unique_expected_spirit_ids: uniqueExpectedSpiritIds,
    expected_unique_targets: uniqueExpectedSpiritIds.length,
    missing_expected_spirit_ids: Array.from({ length: RETRIEVAL_LABEL_COUNT }, (_entry, index) => index + 1)
      .filter((spiritId) => !uniqueExpectedSpiritIds.includes(spiritId)),
    target_coverage_ok: true,
    denoise_model: "denoise/model.nsrltch",
    denoise_model_hash: denoiseModelHash,
    denoise_model_hashes: [denoiseModelHash],
    denoise_model_consistent: true,
    retrieval_head: "retrieval-head.json",
    retrieval_head_model_hash: "",
    trace_integrity_ok: true,
    min_output_signature_distance: 0,
    min_output_ink_range: inkRange,
    output_image_to_text_identification: true,
    min_output_retrieval_image_margin: 10000,
    results,
  };
}

function syntheticAttentionPlan() {
  return Array.from({ length: IMAGE_CHANNEL_PAYLOAD_TOKENS }, (_unused, index) =>
    index % 7 === 0 || index % 13 === 0 ? 128 : 0,
  );
}

function syntheticDenoiseModelBytes() {
  return Buffer.from("synthetic-solomon-denoise-model\n", "utf8");
}

function syntheticDenoiseTrace() {
  return {
    schema: "nsrl.bitmap_sample_trace.v1",
    latent_target_source: "decoded-latent",
    raw_samples: "samples.ink128.u8",
  };
}

function raw128FromPlan(plan) {
  const size = 128;
  const block = size / 16;
  const raw = Buffer.alloc(size * size);
  for (let y = 0; y < size; y += 1) {
    const binY = Math.floor(y / block);
    for (let x = 0; x < size; x += 1) {
      const binX = Math.floor(x / block);
      raw[y * size + x] = Number(plan[binY * 16 + binX] || 0) & 0xff;
    }
  }
  return raw;
}

function enableGenerativeEval(state, paths) {
  state.generativeEval = syntheticGenerativeEval(state, paths);
}

function weakenGeneratedEvalRetrievalMargin(state, paths) {
  enableGenerativeEval(state, paths);
  state.retrievalHead.image_head.biases[1] = state.retrievalHead.image_head.biases[0];
  state.generativeEval.summaryRows[0].min_generated_retrieval_margin = 0;
  for (const row of state.generativeEval.sampleRows) {
    row.generated_retrieval_margin = 0;
  }
}

function syntheticGenerativeEval(state, _paths) {
  const limit = 1;
  const model = "synthetic-decoded-prior";
  const selected = selectSyntheticGenerativeEvalPrompts(state.prompts, limit);
  return {
    model,
    limit,
    selected,
    summaryHeader: [
      "model",
      "latent_model",
      "latent_model_hash",
      "prompts",
      "top1",
      "top5",
      "top1_per_mille",
      "top5_per_mille",
      "top1_16",
      "top5_16",
      "top1_16_per_mille",
      "top5_16_per_mille",
      "top1_px",
      "top5_px",
      "top1_px_per_mille",
      "top5_px_per_mille",
      "latent_top1",
      "latent_top5",
      "latent_top1_per_mille",
      "latent_top5_per_mille",
      "generated_retrieval_top1",
      "generated_retrieval_top5",
      "generated_retrieval_top1_per_mille",
      "generated_retrieval_top5_per_mille",
      "mean_generated_retrieval_rank_q8",
      "min_generated_retrieval_margin",
      "mean_rank_q8",
      "mean_rank_16_q8",
      "mean_rank_px_q8",
      "mean_latent_rank_q8",
      "mean_generated_target_distance_q8",
      "mean_generated_target_distance_16_q8",
      "mean_generated_target_distance_px_q8",
      "mean_latent_decoded_target_distance_q8",
      "mean_latent_target_distance_q8",
      "mean_generated_ink_q8",
      "mean_generated_outside_ink_q8",
      "mean_generated_edge_ink_q8",
      "selected_mean_wash_penalty_q8",
      "text_weight",
    ],
    sampleHeader: [
      "model",
      "out_dir",
      "partition",
      "sampler_target_source",
      "spirit_id",
      "prompt_hash",
      "tier",
      "source",
      "text",
      "prompt",
      "generated_retrieval_rank",
      "generated_retrieval_margin",
      "generated_retrieval_top1_spirit_id",
      "generated_retrieval_top1_name",
      "generated_retrieval_identity",
    ],
    config: {
      partition: "eval",
      latentTarget: "decoded",
      samplerModel: "sampler.nsrltch",
      samplerModelHash: "",
      prompts: "../prompts.jsonl",
      promptsHash: "",
      promptRows: 0,
      selectedPromptRows: selected.length,
      selectedPromptEligibleRows: selected.length,
      selectedPromptUniqueTargets: syntheticUniquePromptTargets(selected),
      selectedPromptEligibleUniqueTargets: syntheticUniquePromptTargets(selected),
      selectedPromptHash: generativeEvalPromptSelectionHash(selected),
      runName: "synthetic-generative-eval",
      evalPermille: 180,
      limit,
      retrievalHead: "../retrieval-head.json",
      retrievalHeadModelHash: "",
      retrievalHeadFeatureCount: 0,
      latentModels: [{ label: model, path: "latent-decoded.nsrltch" }],
      latentModelProvenance: [],
      latentModelHashes: {},
    },
    summaryRows: [syntheticGenerativeEvalSummaryRow(model, selected.length)],
    sampleRows: syntheticGenerativeEvalSampleRows(model, selected),
  };
}

function syntheticGenerativeEvalSummaryRow(model, prompts) {
  return {
    model,
    latent_model: "latent-decoded.nsrltch",
    latent_model_hash: "",
    prompts,
    top1: prompts,
    top5: prompts,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    top1_16: prompts,
    top5_16: prompts,
    top1_16_per_mille: 1000,
    top5_16_per_mille: 1000,
    top1_px: prompts,
    top5_px: prompts,
    top1_px_per_mille: 1000,
    top5_px_per_mille: 1000,
    latent_top1: prompts,
    latent_top5: prompts,
    latent_top1_per_mille: 1000,
    latent_top5_per_mille: 1000,
    generated_retrieval_top1: prompts,
    generated_retrieval_top5: prompts,
    generated_retrieval_top1_per_mille: 1000,
    generated_retrieval_top5_per_mille: 1000,
    mean_generated_retrieval_rank_q8: 256,
    min_generated_retrieval_margin: 10000,
    mean_rank_q8: 256,
    mean_rank_16_q8: 256,
    mean_rank_px_q8: 256,
    mean_latent_rank_q8: 256,
    mean_generated_target_distance_q8: 0,
    mean_generated_target_distance_16_q8: 0,
    mean_generated_target_distance_px_q8: 0,
    mean_latent_decoded_target_distance_q8: 0,
    mean_latent_target_distance_q8: 0,
    mean_generated_ink_q8: 64,
    mean_generated_outside_ink_q8: 0,
    mean_generated_edge_ink_q8: 64,
    selected_mean_wash_penalty_q8: 0,
    text_weight: 256,
  };
}

function syntheticGenerativeEvalSampleRows(model, prompts) {
  return prompts.map((prompt, index) => ({
    model,
    out_dir: `sample-${String(index + 1).padStart(3, "0")}`,
    partition: "eval",
    sampler_target_source: "decoded-latent",
    spirit_id: prompt.spirit_id,
    prompt_hash: prompt.prompt_hash || "",
    tier: prompt.tier || "",
    source: prompt.source || "",
    text: prompt.text || prompt.prompt || "",
    prompt: prompt.text || prompt.prompt || "",
    generated_retrieval_rank: 1,
    generated_retrieval_margin: 10000,
    generated_retrieval_top1_spirit_id: prompt.spirit_id,
    generated_retrieval_top1_name: `Synthetic Solomon ${prompt.spirit_id}`,
    generated_retrieval_identity: 1,
  }));
}

function selectSyntheticGenerativeEvalPrompts(prompts, limit) {
  const candidates = prompts
    .filter(isSyntheticGenerativeEvalHeldoutPrompt)
    .map((prompt) => ({ ...prompt, partition: "eval" }))
    .sort((left, right) => `${left.tier}:${left.prompt_hash}`.localeCompare(`${right.tier}:${right.prompt_hash}`));
  const selected = [];
  const usedTargets = new Set();
  for (const prompt of candidates) {
    if (selected.length >= limit) break;
    if (usedTargets.has(prompt.spirit_id)) continue;
    selected.push(prompt);
    usedTargets.add(prompt.spirit_id);
  }
  return selected;
}

function isSyntheticGenerativeEvalHeldoutPrompt(prompt) {
  const tier = String(prompt.tier || "").toLowerCase();
  const source = String(prompt.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function syntheticUniquePromptTargets(prompts) {
  return new Set(
    prompts
      .map((prompt) => Number(prompt.spirit_id || 0))
      .filter((spiritId) => Number.isInteger(spiritId) && spiritId > 0),
  ).size;
}

function generativeEvalPromptSelectionHash(prompts) {
  const lines = prompts
    .map((prompt) => [
      prompt.prompt_hash || "",
      prompt.spirit_id || "",
      prompt.partition || "",
      prompt.tier || "",
      prompt.source || "",
      prompt.text || prompt.prompt || "",
    ].join("\t"))
    .sort()
    .join("\n");
  return fnv64BytesHex(Buffer.from(`${lines}\n`, "utf8"));
}

function syntheticGenerativeLatentModelBytes() {
  return Buffer.from("synthetic-solomon-latent-decoded-model\n", "utf8");
}

function syntheticGenerativeSamplerModelBytes() {
  return Buffer.from("synthetic-solomon-bitmap-sampler-model\n", "utf8");
}

function syntheticGenerativeTrace() {
  return {
    schema: "nsrl.bitmap_sampler_trace.v1",
    model: "../sampler.nsrltch",
    model_format: "NSRLTCH",
    latent_model: "../latent-decoded.nsrltch",
    latent_target_source: "decoded-latent",
    raw_samples: "samples.ink128.u8",
    image_size: 128,
  };
}

function syntheticIdentityInference(paths) {
  return {
    schema: "nsrl.solomon_v2_identity_inference.v1",
    ok: true,
    errors: [],
    retrieval_head: null,
    model_hash: "",
    text_index: paths.sourceTextPath,
    text_index_hash: "",
    query_count: 3,
    text_queries: [{ spirit_id: 1, source_text: "source evidence" }],
    image_queries: [{ spirit_id: 1, source_text: "source evidence" }],
    sample_queries: [{ spirit_id: 1, source_text: "source evidence" }],
    source_summary: {
      text_queries_have_source_text: true,
      image_queries_have_source_text: true,
      sample_queries_have_source_text: true,
    },
    sample_summary: {
      samples: 1,
      text_image_agreement: true,
      generated_text_image_agreement: true,
      signature_retrieval_agreement: true,
      expected_image_agreement: true,
      expected_generated_text_agreement: true,
      source_text_evidence: true,
      generated_text_source_evidence: true,
      min_source_text_chars: 32,
      min_prompt_text_margin: 10,
      min_generated_text_margin: 10,
      min_image_retrieval_margin: 10,
      min_signature_margin: 10,
    },
  };
}

function syntheticCurriculumStages(paths) {
  const stages = [
    curriculumStage(paths, 0, "identity", ["identify"], ["identify", "image-to-text", "explain"]),
    curriculumStage(paths, 1, "image", ["text-to-image"], ["text-to-image", "description-to-image", "image-to-text"]),
  ];
  return {
    schema: "nsrl.solomon_v2_curriculum_stage_check.v1",
    ok: true,
    errors: [],
    stage_count: stages.length,
    require_loss_non_increasing: true,
    required_stage_names: stages.map((stage) => stage.expected_stage_name),
    source_corpus_provenance: {
      source_examples: paths.examplesPath,
      source_examples_hash: "",
      source_examples_consistent: true,
      source_tokens: paths.tokensPath,
      source_tokens_hash: "",
      source_tokens_consistent: true,
    },
    stages,
  };
}

function curriculumStage(paths, index, stageName, identityTasks, evidenceTasks) {
  return {
    index,
    ok: true,
    stage_name: stageName,
    expected_stage_name: stageName,
    stage_dir: path.join(paths.dir, `stage-${stageName}`),
    filter: { stage_name: stageName },
    examples: SPIRIT_COUNT,
    token_count: 1,
    source_dir: paths.dir,
    source_manifest_schema: "nsrl.solomon_multimodal_manifest.v2",
    source_examples: paths.examplesPath,
    source_examples_hash: "",
    source_tokens: paths.tokensPath,
    source_tokens_hash: "",
    identity_bindings: identityBindingSummary(identityTasks),
    source_identity_bindings: identityBindingSummary(identityTasks),
    stage_evidence: stageEvidence(stageName, evidenceTasks),
    task_marker_integrity: taskMarkerIntegrity(paths),
    task_modality_integrity: taskModalityIntegrity(paths),
    image_channel_marker_integrity: imageChannelMarkerIntegrity(paths),
    train: {
      attention_kind: "integer-transformer",
      text_token_profile: "byte",
      batch_mode: "map-reduce",
      map_reduce_workers: 2,
      windows: 2,
      examined_windows: 2,
      updates: 1,
      accepted_batches: 1,
      rejected_batches: 0,
      probability_error_delta_i64: -1,
      d_model: 128,
      heads: 2,
      hidden_dim: 256,
      transformer_layers: 2,
      context_seq_len: 384,
      seq_len: 384,
    },
  };
}

function identityBindingSummary(tasks) {
  const byTask = Object.fromEntries(tasks.map((task) => [task, identityBindingTask(task)]));
  return {
    rows: tasks.length * SPIRIT_COUNT * REQUIRED_IDENTITY_BINDING_KINDS.length,
    binding_hash: "0xsyntheticidentitybinding",
    by_task: byTask,
    by_kind: Object.fromEntries(REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [kind, SPIRIT_COUNT])),
  };
}

function identityBindingTask(task) {
  return {
    task,
    rows: SPIRIT_COUNT * REQUIRED_IDENTITY_BINDING_KINDS.length,
    spirits: SPIRIT_COUNT,
    binding_hash: `0xsyntheticidentitybinding${task.replace(/[^a-z0-9]/g, "")}`,
    counts: Object.fromEntries(REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [kind, SPIRIT_COUNT])),
  };
}

function stageEvidence(stageName, requiredTasks) {
  return {
    stage_name: stageName,
    expected_spirits: SPIRIT_COUNT,
    records: requiredTasks.length * SPIRIT_COUNT,
    spirits: SPIRIT_COUNT,
    required: Object.fromEntries(
      requiredTasks.map((task) => [task, { records: SPIRIT_COUNT, spirits: SPIRIT_COUNT }]),
    ),
    image_plan: stageName === "image" ? { min_spirits: SPIRIT_COUNT } : {},
    image_classification: stageName === "image" ? { min_spirits: SPIRIT_COUNT } : {},
    image_grounding: {},
    match: {},
  };
}

function taskMarkerIntegrity(paths) {
  return {
    ok: true,
    examples: paths.examplesPath,
    tokens: paths.tokensPath,
    checked_records: 1,
    hash_mismatches: 0,
    marker_mismatches: 0,
    out_of_bounds: 0,
    missing_offsets: 0,
    by_task: {},
  };
}

function taskModalityIntegrity(paths) {
  return {
    ok: true,
    examples: paths.examplesPath,
    tokens: paths.tokensPath,
    checked_records: 1,
    missing_offsets: 0,
    out_of_bounds: 0,
    modality_mismatches: 0,
    by_task: {},
  };
}

function imageChannelMarkerIntegrity(paths) {
  return {
    ok: true,
    examples: paths.examplesPath,
    tokens: paths.tokensPath,
    required_channels: CHANNELS,
    checked_records: 1,
    missing_offsets: 0,
    out_of_bounds: 0,
    missing_image_markers: 0,
    missing_channel_markers: 0,
    short_channel_payloads: 0,
    bad_channel_payloads: 0,
    channel_order_mismatches: 0,
    by_task: {},
    by_channel: Object.fromEntries(
      CHANNELS.map((channel) => [
        channel,
        {
          checked_records: 1,
          found_markers: 1,
          missing_channel_markers: 0,
          short_channel_payloads: 0,
          bad_channel_payloads: 0,
          channel_order_mismatches: 0,
        },
      ]),
    ),
  };
}

function syntheticGroundedCorpus(paths) {
  const sourceTasks = ["explain", "image-to-explain", "text-image-explain", "description-to-image"];
  const attributeTasks = ["image-to-attributes"];
  return {
    schema: "nsrl.solomon_v2_grounded_corpus_check.v1",
    ok: true,
    errors: [],
    examples: paths.examplesPath,
    examples_hash: "",
    text_index: paths.sourceTextPath,
    text_index_hash: "",
    expect_spirits: SPIRIT_COUNT,
    source_text_tasks: sourceTasks,
    attribute_tasks: attributeTasks,
    require_source_provenance: true,
    require_name_source_explain: true,
    require_description_source_image: true,
    require_image_attribute_generic_prompt: true,
    min_source_overlap_tokens: 2,
    min_attribute_source_overlap_tokens: 8,
    max_source_placeholder_rows: 0,
    max_attribute_generic_rank_rows: 0,
    tasks: {
      ...Object.fromEntries(sourceTasks.map((task) => [task, groundedTask(2)])),
      ...Object.fromEntries(attributeTasks.map((task) => [task, groundedTask(8, true)])),
    },
  };
}

function groundedTask(minSourceOverlapTokens, attribute = false) {
  const row = {
    records: SPIRIT_COUNT,
    spirits: SPIRIT_COUNT,
    min_source_overlap_tokens: minSourceOverlapTokens,
    source_provenance_rows: SPIRIT_COUNT,
    source_provenance_hash_mismatches: 0,
    source_excerpt_hash_mismatches: 0,
    name_source_prompt_rows: SPIRIT_COUNT,
    name_source_prompt_ok_rows: SPIRIT_COUNT,
    description_source_prompt_rows: SPIRIT_COUNT,
    description_source_prompt_ok_rows: SPIRIT_COUNT,
    image_attribute_prompt_rows: SPIRIT_COUNT,
    image_attribute_prompt_ok_rows: SPIRIT_COUNT,
    placeholder_rows: 0,
  };
  if (attribute) {
    row.generic_attribute_rank_rows = 0;
  }
  return row;
}

function materializeState(state, paths) {
  writeText(paths.sourceTextPath, sourceTextTsv(state.sourceRows));
  writeText(paths.promptsPath, `${state.prompts.map((row) => JSON.stringify(row)).join("\n")}\n`);
  writeText(paths.examplesPath, `${state.examples.map((row) => JSON.stringify(row)).join("\n")}\n`);
  fs.writeFileSync(paths.tokensPath, Buffer.from(state.tokens));

  const examplesHash = fileHash(paths.examplesPath);
  const tokensHash = fileHash(paths.tokensPath);
  const promptsHash = fileHash(paths.promptsPath);
  const sourceHash = fileHash(paths.sourceTextPath);

  state.manifest.token_hash = tokensHash;
  state.retrievalHead.model_hash = fnv64StringHex(JSON.stringify(withoutModelHash(state.retrievalHead)));
  state.retrievalEval.model_hash = state.retrievalHead.model_hash;
  state.retrievalEval.examples_hash = examplesHash;
  state.retrievalEval.tokens_hash = tokensHash;
  state.retrievalEval.prompts_hash = promptsHash;
  state.sampleBinding.retrieval_head_model_hash = state.retrievalHead.model_hash;
  if (state.denoiseBridge) {
    state.denoiseBridge.retrieval_head_model_hash = state.retrievalHead.model_hash;
  }
  if (state.generativeEval) {
    state.generativeEval.config.retrievalHeadModelHash = state.retrievalHead.model_hash;
    state.generativeEval.config.retrievalHeadFeatureCount = state.retrievalHead.feature_count;
    state.generativeEval.config.promptsHash = promptsHash;
    state.generativeEval.config.promptRows = state.prompts.length;
  }
  state.identityInference.model_hash = state.retrievalHead.model_hash;
  state.identityInference.text_index_hash = sourceHash;
  state.curriculumStages.source_corpus_provenance.source_examples_hash = examplesHash;
  state.curriculumStages.source_corpus_provenance.source_tokens_hash = tokensHash;
  for (const stage of state.curriculumStages.stages) {
    stage.source_examples_hash = examplesHash;
    stage.source_tokens_hash = tokensHash;
  }
  state.groundedCorpus.examples_hash = examplesHash;
  state.groundedCorpus.text_index_hash = sourceHash;

  writeJson(paths.manifestPath, state.manifest);
  writeJson(paths.evalPath, state.evalTrace);
  writeJson(paths.retrievalHeadPath, state.retrievalHead);
  writeJson(paths.retrievalEvalPath, state.retrievalEval);
  writeJson(paths.sampleBindingPath, state.sampleBinding);
  writeJson(paths.generationIntegrityPath, state.generationIntegrity);
  materializeSampleBindingArtifacts(state, paths);
  if (state.denoiseBridge) {
    materializeDenoiseBridgeArtifacts(state, paths);
    writeJson(paths.denoiseBridgePath, state.denoiseBridge);
  }
  if (state.generativeEval) {
    materializeGenerativeEvalArtifacts(state, paths);
  }
  writeJson(paths.identityInferencePath, state.identityInference);
  writeJson(paths.curriculumStagesPath, state.curriculumStages);
  writeJson(paths.groundedCorpusPath, state.groundedCorpus);
}

function materializeSampleBindingArtifacts(state, paths) {
  for (const result of state.sampleBinding.results || []) {
    const planPath = path.join(paths.dir, result.image_ink16_u8);
    fs.mkdirSync(path.dirname(planPath), { recursive: true });
    fs.writeFileSync(planPath, Buffer.from(syntheticAttentionPlan()));
  }
}

function materializeDenoiseBridgeArtifacts(_state, paths) {
  fs.mkdirSync(path.dirname(paths.denoiseModelPath), { recursive: true });
  fs.writeFileSync(paths.denoiseModelPath, syntheticDenoiseModelBytes());
  writeJson(paths.denoiseTracePath, syntheticDenoiseTrace());
  fs.writeFileSync(paths.denoiseRawPath, raw128FromPlan(syntheticAttentionPlan()));
}

function materializeGenerativeEvalArtifacts(state, paths) {
  const generative = state.generativeEval;
  fs.mkdirSync(paths.generativeEvalDir, { recursive: true });
  fs.writeFileSync(paths.generativeEvalLatentModelPath, syntheticGenerativeLatentModelBytes());
  fs.writeFileSync(paths.generativeEvalSamplerModelPath, syntheticGenerativeSamplerModelBytes());
  const latentModelHash = fileHash(paths.generativeEvalLatentModelPath);
  const samplerModelHash = fileHash(paths.generativeEvalSamplerModelPath);
  const selectedPromptHash = generativeEvalPromptSelectionHash(generative.selected);
  const selectedUniqueTargets = syntheticUniquePromptTargets(generative.selected);

  generative.config.samplerModelHash = samplerModelHash;
  generative.config.selectedPromptRows = generative.selected.length;
  generative.config.selectedPromptEligibleRows = generative.selected.length;
  generative.config.selectedPromptUniqueTargets = selectedUniqueTargets;
  generative.config.selectedPromptEligibleUniqueTargets = selectedUniqueTargets;
  generative.config.selectedPromptHash = selectedPromptHash;
  generative.config.latentModelProvenance = [
    {
      label: generative.model,
      path: "latent-decoded.nsrltch",
      modelHash: latentModelHash,
    },
  ];
  generative.config.latentModelHashes = {
    [generative.model]: latentModelHash,
  };

  for (const row of generative.sampleRows) {
    const sampleDir = path.join(paths.generativeEvalDir, row.out_dir);
    fs.mkdirSync(sampleDir, { recursive: true });
    writeJson(path.join(sampleDir, "trace.json"), syntheticGenerativeTrace());
    fs.writeFileSync(path.join(sampleDir, "samples.ink128.u8"), raw128FromPlan(syntheticAttentionPlan()));
  }

  const summaryRows = generative.summaryRows.map((row) => ({
    ...row,
    latent_model_hash: latentModelHash,
  }));
  writeText(paths.generativeEvalSummaryPath, tsvTable(generative.summaryHeader, summaryRows));
  writeText(paths.generativeEvalSamplesPath, tsvTable(generative.sampleHeader, generative.sampleRows));
  writeJson(paths.generativeEvalConfigPath, generative.config);
}

function withoutModelHash(model) {
  const copy = { ...model };
  delete copy.model_hash;
  return copy;
}

function sourceTextTsv(rows) {
  const header = ["spirit_id", "primary_name", "source_text"];
  const body = rows.map((row) => header.map((key) => String(row[key] || "")).join("\t"));
  return `${header.join("\t")}\n${body.join("\n")}\n`;
}

function tsvTable(header, rows) {
  const body = rows.map((row) => header.map((key) => tsvValue(row[key])).join("\t"));
  return `${header.join("\t")}\n${body.join("\n")}\n`;
}

function tsvValue(value) {
  return String(value ?? "").replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function runQualityReport(paths) {
  const args = [
    "scripts/check-solomon-v2-quality-report.mjs",
    "--eval", paths.evalPath,
    "--retrieval-head", paths.retrievalHeadPath,
    "--retrieval-head-eval", paths.retrievalEvalPath,
    "--sample-binding", paths.sampleBindingPath,
    "--generation-integrity", paths.generationIntegrityPath,
    "--examples", paths.examplesPath,
    "--manifest", paths.manifestPath,
    "--tokens", paths.tokensPath,
    "--identity-inference", paths.identityInferencePath,
    "--curriculum-stages", paths.curriculumStagesPath,
    "--grounded-corpus", paths.groundedCorpusPath,
    "--out", paths.qualityReportPath,
    "--min-task-targets", "all=1",
    "--min-task-top5-per-mille", "all=1",
    "--min-phase-targets", "all=1",
    "--require-heldout-prompts",
    "--min-heldout-prompt-rows", String(SPIRIT_COUNT),
    "--min-match-yes-top1", String(SPIRIT_COUNT),
    "--min-match-no-top1", String(SPIRIT_COUNT * 2),
    "--min-match-no-image-top1", String(SPIRIT_COUNT),
    "--min-match-no-prompt-top1", String(SPIRIT_COUNT),
    "--min-retrieval-margin", "2",
    "--require-architecture-profile",
    "--require-promoted-small-profile",
    "--min-d-model", "128",
    "--min-heads", "2",
    "--min-hidden-dim", "256",
    "--min-transformer-layers", "2",
    "--min-context-seq-len", "384",
    "--require-corpus-version", "v2",
    "--require-image-token-profile", "symbolic16",
    "--require-image-token-channels", CHANNELS.join(","),
    "--require-image-channel-token-stats",
    "--min-image-channel-distinct-bins", "2",
    "--require-identity-inference",
    "--require-curriculum-stages",
    "--require-curriculum-stage-names", "identity,image",
    "--require-grounded-corpus",
    "--min-grounded-source-overlap-tokens", "2",
    "--min-grounded-attribute-source-overlap-tokens", "8",
    "--max-grounded-source-placeholder-rows", "0",
    "--max-grounded-attribute-generic-rank-rows", "0",
    "--require-confidence-trace",
  ];
  if (fs.existsSync(paths.denoiseBridgePath)) {
    args.push(
      "--denoise-bridge", paths.denoiseBridgePath,
      "--require-denoise-bridge",
      "--require-denoise-output-identity",
    );
  }
  if (fs.existsSync(paths.generativeEvalSummaryPath)) {
    args.push(
      "--generative-eval", paths.generativeEvalDir,
      "--require-generative-eval",
      "--require-generative-output-identity",
      "--min-generated-top5-per-mille", "1",
      "--min-generated-top5-16-per-mille", "1",
      "--min-generated-top5-px-per-mille", "1",
      "--min-generated-retrieval-top1-per-mille", "1000",
      "--min-generated-retrieval-top5-per-mille", "1000",
      "--min-generated-retrieval-margin", "1",
      "--min-generated-prompt-rows", "1",
    );
  }
  const result = childProcess.spawnSync(process.execPath, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  const report = parseQualityReport(result.stdout, paths.qualityReportPath);
  return { result, report };
}

function parseQualityReport(stdout, outPath) {
  if (fs.existsSync(outPath)) {
    return JSON.parse(fs.readFileSync(outPath, "utf8"));
  }
  const objects = extractJsonObjects(stdout);
  return objects[objects.length - 1] || null;
}

function extractJsonObjects(text) {
  const objects = [];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] !== "{") continue;
    const end = matchingJsonObjectEnd(text, index);
    if (end < 0) continue;
    try {
      objects.push(JSON.parse(text.slice(index, end + 1)));
      index = end;
    } catch {
      // Ignore log fragments that happen to contain braces.
    }
  }
  return objects;
}

function matchingJsonObjectEnd(text, start) {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      continue;
    }
    if (char === "\"") {
      inString = true;
    } else if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function runCase(root, name, mutate, expectOk, expectedSubstrings = []) {
  const paths = buildFixture(root, name, mutate);
  const { result, report } = runQualityReport(paths);
  const errors = Array.isArray(report?.errors) ? report.errors.map(String) : [];
  const ok = Boolean(report?.ok);
  const statusOk = expectOk ? result.status === 0 : result.status !== 0;
  const reportOk = ok === expectOk;
  const substringsOk = expectedSubstrings.every((substring) =>
    errors.some((error) => error.includes(substring)),
  );
  return {
    name,
    ok: statusOk && reportOk && substringsOk,
    expected_ok: expectOk,
    status: result.status,
    schema: report?.schema || "",
    report_ok: ok,
    confidence_label: report?.confidence_trace?.label || "",
    confidence_trace_ready: report?.confidence_trace_ready === true,
    model_only_quality_floor_met: report?.model_only_quality_floor?.met === true,
    expected_error_substrings: expectedSubstrings,
    matched_error_substrings: expectedSubstrings.filter((substring) =>
      errors.some((error) => error.includes(substring)),
    ),
    errors: errors.slice(0, 20),
    stdout_tail: statusOk && reportOk ? "" : tailLines(result.stdout || "", 40),
    stderr_tail: result.stderr ? tailLines(result.stderr, 40) : "",
  };
}

function tailLines(text, maxLines) {
  const lines = String(text).split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function writeReport(outPath, report) {
  if (!outPath) return;
  const resolved = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-quality-report-self-test-"));
  const cases = [];
  try {
    cases.push(runCase(root, "good", () => {}, true));
    cases.push(runCase(
      root,
      "bad-retrieval-margin",
      (state) => {
        state.retrievalEval.heldout_prompts.min_margin = 0;
      },
      false,
      ["retrieval head held-out prompts min_margin 0 < 2"],
    ));
    cases.push(runCase(
      root,
      "bad-symbolic-channel-evidence",
      (state) => {
        state.manifest.image_token_channel_stats.edge.distinct_bins = 1;
        state.curriculumStages.stages[0].image_channel_marker_integrity.by_channel.edge.found_markers = 0;
      },
      false,
      [
        "corpus manifest image_token_channel_stats edge distinct_bins 1 < 2",
        "confidence trace: required symbolic image-token byte evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-symbolic-channel-duplicates",
      (state) => {
        state.manifest.image_token_channel_stats.edge.unique_record_hashes = 1;
        state.manifest.image_token_channel_stats.edge.duplicate_record_hashes = SPIRIT_COUNT - 1;
      },
      false,
      [
        "corpus manifest image_token_channel_stats edge unique_record_hashes 1 != records 2",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-source-grounding",
      (state) => {
        state.identityInference.source_summary.image_queries_have_source_text = false;
      },
      false,
      [
        "identity inference image queries are missing source text evidence",
        "confidence trace: source-grounded identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-source-grounding-missing-sample",
      (state) => {
        delete state.identityInference.source_summary.sample_queries_have_source_text;
      },
      false,
      [
        "identity inference sample queries are missing source text evidence",
        "confidence trace: source-grounded identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-grounded-description-source",
      (state) => {
        state.groundedCorpus.tasks["description-to-image"].description_source_prompt_ok_rows = 1;
      },
      false,
      [
        "grounded corpus task description-to-image description-source prompt rows 1 != records 2",
        "confidence trace: required grounded corpus evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-grounded-attribute-prompt",
      (state) => {
        state.groundedCorpus.tasks["image-to-attributes"].image_attribute_prompt_ok_rows = 1;
      },
      false,
      [
        "grounded corpus task image-to-attributes generic attribute prompt rows 1 != records 2",
        "confidence trace: required grounded corpus evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-curriculum-binding",
      (state) => {
        state.curriculumStages.stages[0].identity_bindings.by_task.identify.binding_hash = "0xbroken";
      },
      false,
      [
        "curriculum stage 0 identify identity binding hash 0xbroken != source",
        "confidence trace: required curriculum identity-binding evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-native-task-confidence",
      (state) => {
        state.evalTrace.tasks["image-to-text"].top5_accuracy_per_mille = 0;
      },
      false,
      [
        "attention eval task image-to-text top5 0 < 1",
        "confidence trace: image-to-text native task eval top5 0 < 1",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-sample-generated-text-agreement",
      (state) => {
        state.sampleBinding.generated_text_image_agreement = false;
        state.sampleBinding.results[0].generated_text_image_agree = false;
      },
      false,
      [
        "sample binding generated text/image agreement is not true",
        "sample binding does not provide complete text/image/signature agreement",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-generated-text-source-evidence",
      (state) => {
        state.identityInference.sample_summary.generated_text_source_evidence = false;
      },
      false,
      [
        "identity inference generated_text_source_evidence is not true",
        "source-grounded identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-identity-prompt-text-margin",
      (state) => {
        state.identityInference.sample_summary.min_prompt_text_margin = 0;
      },
      false,
      [
        "identity inference min_prompt_text_margin 0 <= 0",
        "source-grounded identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-identity-generated-text-margin",
      (state) => {
        state.identityInference.sample_summary.min_generated_text_margin = 0;
      },
      false,
      [
        "identity inference min_generated_text_margin 0 <= 0",
        "source-grounded identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "good-denoise-bridge",
      (state, paths) => {
        enableDenoiseBridge(state, paths);
      },
      true,
    ));
    cases.push(runCase(
      root,
      "bad-denoise-output-identity",
      (state, paths) => {
        enableDenoiseBridge(state, paths);
        state.denoiseBridge.output_image_to_text_identification = false;
        state.denoiseBridge.results[0].output_signature.output_image_to_text_identification = false;
        state.denoiseBridge.results[0].output_signature.samples_detail[0].image_to_text_identity = false;
      },
      false,
      [
        "denoise bridge output image-to-text identification is not true",
        "confidence trace: required denoise bridge evidence is not complete",
        "confidence trace: required denoised-output image-to-text identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-denoise-target-coverage",
      (state, paths) => {
        enableDenoiseBridge(state, paths);
        state.denoiseBridge.expected_unique_targets = 0;
        state.denoiseBridge.unique_expected_spirit_ids = [];
        state.denoiseBridge.target_coverage_ok = false;
      },
      false,
      [
        "denoise bridge expected_unique_targets 0 != recomputed 1",
        "denoise bridge target_coverage_ok is false",
      ],
    ));
    cases.push(runCase(
      root,
      "good-generative-eval",
      (state, paths) => {
        enableGenerativeEval(state, paths);
      },
      true,
    ));
    cases.push(runCase(
      root,
      "bad-generative-output-identity",
      (state, paths) => {
        enableGenerativeEval(state, paths);
        state.generativeEval.sampleRows[0].generated_retrieval_identity = 0;
      },
      false,
      [
        "generative eval sample row 2 generated_retrieval_identity 0 != recomputed 1",
        "generative eval output identity is incomplete for matching model synthetic-decoded-prior",
        "confidence trace: required product-generation output identity evidence is not complete",
      ],
    ));
    cases.push(runCase(
      root,
      "bad-generative-output-margin",
      (state, paths) => {
        weakenGeneratedEvalRetrievalMargin(state, paths);
      },
      false,
      [
        "generative eval no model met the configured product-generation floor",
        "generative eval output identity requires a matching product-floor model",
        "confidence trace: required product-generation output identity evidence is not complete",
      ],
    ));
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
  const report = {
    schema,
    ok: cases.every((item) => item.ok),
    scratch_root: config.keep ? root : "",
    kept: config.keep,
    cases,
    errors: cases.filter((item) => !item.ok).map((item) => `${item.name} failed`),
  };
  writeReport(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
