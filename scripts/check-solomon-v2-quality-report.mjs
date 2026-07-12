#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

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
const EVAL_PHASES = ["special", "prompt", "text", "image"];
const REQUIRED_OUTPUT_HEADS = ["special_head", "text_head", "image_head"];
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
const IMAGE_RETRIEVAL_TASK_MIN_COUNTS = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};
const PRODUCT_DIRECTIONAL_GROUPS = [
  {
    key: "text_prompt_to_image_plan",
    label: "text prompt -> 16x16 image plan",
    tasks: ["text-to-image", "description-to-image"],
    required_phases: {
      "text-to-image": ["image"],
      "description-to-image": ["image"],
    },
  },
  {
    key: "seal_image_to_text",
    label: "seal image -> identity / attributes / source text",
    tasks: ["image-to-text", "image-to-explain", "image-to-attributes"],
    required_phases: {
      "image-to-text": ["text"],
      "image-to-explain": ["text"],
      "image-to-attributes": ["text"],
    },
  },
  {
    key: "text_and_seal_to_explanation",
    label: "text + seal -> explanation / retrieval",
    tasks: ["text-image-explain", "match"],
    required_phases: {
      "text-image-explain": ["text"],
      match: ["text"],
    },
  },
  {
    key: "identity_source_binding",
    label: "prompt/name -> identity / source text",
    tasks: ["canonical-joint", "identify", "explain"],
    required_phases: {
      "canonical-joint": ["text", "image"],
      identify: ["text"],
      explain: ["text"],
    },
  },
];
const CURRICULUM_IDENTITY_BINDING_TASKS = {
  identity: ["identify"],
  image: ["text-to-image"],
  "text-to-image": ["text-to-image"],
  all: ["identify", "text-to-image"],
};
const CURRICULUM_STAGE_EVIDENCE_TASKS = {
  identity: ["identify", "image-to-text", "explain"],
  image: ["text-to-image", "description-to-image", "image-to-text"],
  "text-to-image": ["text-to-image", "description-to-image"],
  "description-to-image": ["description-to-image"],
  "image-to-text": ["image-to-text", "image-to-explain", "text-image-explain", "image-to-attributes"],
  explain: ["explain", "image-to-explain", "text-image-explain", "image-to-attributes"],
  match: ["match"],
  "hard-negative": ["match"],
  all: REQUIRED_TASKS,
};
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const TASK_TOKEN_LAYOUT_FALLBACK = {
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
const IMAGE_CHANNEL_PAYLOAD_TOKENS_FALLBACK = 256;
const GENERATED_RETRIEVAL_GRID = 16;
const GENERATED_RETRIEVAL_BINS = GENERATED_RETRIEVAL_GRID * GENERATED_RETRIEVAL_GRID;
const GENERATED_RETRIEVAL_IMAGE_SIZE = 128;
const GENERATED_RETRIEVAL_IMAGE_BYTES = GENERATED_RETRIEVAL_IMAGE_SIZE * GENERATED_RETRIEVAL_IMAGE_SIZE;
const IMAGE_BEARING_TASKS = new Set([
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "match",
]);
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;
const GENERATION_TRACE_ALLOWED_TARGET_KEYS = new Set([
  "latent_target_source",
  "latent_target_number",
  "latent_target_name",
  "latent_target_score",
  "latent_target_latent_score",
  "latent_target_lexical_score",
  "latent_target_signature",
]);
const GENERATION_TRACE_FREE_TEXT_VALUE_KEYS = new Set([
  "latent_prompt",
  "latent_target_name",
]);
const GENERATION_TRACE_FORBIDDEN_KEY_PATTERNS = [
  /display[_-]?cleanup/i,
  /cleanup/i,
  /post[_-]?process/i,
  /postprocess/i,
  /oracle/i,
  /ground[_-]?truth/i,
  /guidance/i,
  /target[_-]?(pixel|pixels|bitmap|image|ink|seal|lookup|source|guidance)/i,
  /(pixel|pixels|bitmap|image|ink|seal)[_-]?target/i,
];
const GENERATION_TRACE_BROAD_FORBIDDEN_VALUE =
  /target[-_\s]*(pixel|pixels|bitmap|image|ink|seal|signature)|ground[-_\s]*truth|oracle|retrieval[-_\s]*hybrid|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess|targetctx/i;
const GENERATION_TRACE_SOURCE_FORBIDDEN_VALUE =
  /\btarget\b|target[-_\s]*(lookup|guidance|source)|retrieval[-_\s]*hybrid|ground[-_\s]*truth|oracle|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess/i;

const defaults = {
  evalPath: "",
  examplesPath: "",
  manifestPath: "",
  tokensPath: "",
  retrievalHeadPath: "",
  retrievalHeadEvalPath: "",
  sampleBindingPath: "",
  generationIntegrityPath: "",
  identityInferencePath: "",
  curriculumStagesPath: "",
  denoiseBridgePath: "",
  groundedCorpusPath: "",
  generativeEvalPath: "",
  outPath: "",
  minTotalTop5PerMille: 0,
  minTextTop5PerMille: 0,
  minImageTop5PerMille: 0,
  minTaskTargets: {},
  minTaskTop5PerMille: {},
  minPhaseTargets: {},
  minGeneratedTop5PerMille: 0,
  minGeneratedTop516PerMille: 0,
  minGeneratedTop5PxPerMille: 0,
  minGeneratedRetrievalTop1PerMille: 0,
  minGeneratedRetrievalTop5PerMille: 0,
  minGeneratedRetrievalMargin: 0,
  minGeneratedPromptRows: 0,
  minLatentTop5PerMille: 0,
  maxGeneratedMeanRankQ8: 0,
  maxGeneratedMeanRank16Q8: 0,
  maxGeneratedMeanRankPxQ8: 0,
  maxGeneratedMeanTargetDistanceQ8: 0,
  maxGeneratedMeanTargetDistance16Q8: 0,
  maxGeneratedMeanTargetDistancePxQ8: 0,
  requireHeldoutPrompts: false,
  minHeldoutPromptRows: 0,
  minMatchYesTop1: 0,
  minMatchNoTop1: 0,
  minMatchNoImageTop1: 0,
  minMatchNoPromptTop1: 0,
  minRetrievalMargin: 0,
  requireArchitectureProfile: false,
  minDModel: 0,
  minHeads: 0,
  minHiddenDim: 0,
  minTransformerLayers: 0,
  minContextSeqLen: 0,
  requireValidHeadGeometry: true,
  requirePromotedSmallProfile: false,
  requireCorpusVersion: "",
  requireImageTokenProfile: "",
  requireImageTokenChannels: [],
  requireImageChannelTokenStats: false,
  minImageChannelDistinctBins: 2,
  requireCurriculumStageNames: [],
  requireIdentityInference: false,
  requireCurriculumStages: false,
  requireDenoiseBridge: false,
  requireDenoiseOutputIdentity: false,
  minDenoiseBridgeUniqueTargets: 0,
  requireGroundedCorpus: false,
  minGroundedSourceOverlapTokens: 0,
  minGroundedAttributeSourceOverlapTokens: 0,
  maxGroundedSourcePlaceholderRows: 0,
  maxGroundedAttributeGenericRankRows: 0,
  requireConfidenceTrace: false,
  requireGenerativeEval: false,
  requireGenerativeOutputIdentity: false,
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-v2-quality-report.mjs --eval PATH --retrieval-head-eval PATH",
      "       --sample-binding PATH --generation-integrity PATH [--out PATH]",
      "",
      "Builds a single v2 Solomon quality report from model eval, retrieval-head",
      "eval, generated sample binding, and generation integrity traces.",
      "",
      "Options:",
      "  --retrieval-head PATH",
      "  --examples PATH",
      "  --manifest PATH",
      "  --tokens PATH",
      "  --min-total-top5-per-mille N",
      "  --min-text-top5-per-mille N",
      "  --min-image-top5-per-mille N",
      "  --min-task-targets all=N[,TASK=N...]",
      "  --min-task-top5-per-mille all=N[,TASK=N...]",
      "  --require-heldout-prompts",
      "  --min-heldout-prompt-rows N",
      "  --min-match-yes-top1 N",
      "  --min-match-no-top1 N",
      "  --min-match-no-image-top1 N",
      "  --min-match-no-prompt-top1 N",
      "  --min-retrieval-margin N",
      "  --identity-inference PATH",
      "  --curriculum-stages PATH",
      "  --denoise-bridge PATH",
      "  --grounded-corpus PATH",
      "  --generative-eval PATH   (run dir or summary.tsv)",
      "  --require-identity-inference",
      "  --require-curriculum-stages",
      "  --require-denoise-bridge",
      "  --require-denoise-output-identity",
      "  --min-denoise-bridge-unique-targets N",
      "  --require-grounded-corpus",
      "  --min-grounded-source-overlap-tokens N",
      "  --min-grounded-attribute-source-overlap-tokens N",
      "  --max-grounded-source-placeholder-rows N",
      "  --max-grounded-attribute-generic-rank-rows N",
      "  --require-confidence-trace",
      "  --require-generative-eval",
      "  --require-generative-output-identity",
      "  --min-generated-top5-per-mille N",
      "  --min-generated-top5-16-per-mille N",
      "  --min-generated-top5-px-per-mille N",
      "  --min-generated-retrieval-top1-per-mille N",
      "  --min-generated-retrieval-top5-per-mille N",
      "  --min-generated-retrieval-margin N",
      "  --min-generated-prompt-rows N",
      "  --min-latent-top5-per-mille N",
      "  --max-generated-mean-rank-q8 N",
      "  --max-generated-mean-rank-16-q8 N",
      "  --max-generated-mean-rank-px-q8 N",
      "  --max-generated-mean-target-distance-q8 N",
      "  --max-generated-mean-target-distance-16-q8 N",
      "  --max-generated-mean-target-distance-px-q8 N",
      "  --require-architecture-profile",
      "  --min-d-model N",
      "  --min-heads N",
      "  --min-hidden-dim N",
      "  --min-transformer-layers N",
      "  --min-context-seq-len N",
      "  --require-promoted-small-profile",
      "  --require-corpus-version VALUE",
      "  --require-image-token-profile PROFILE",
      "  --require-image-token-channels LIST",
      "  --require-image-channel-token-stats",
      "  --min-image-channel-distinct-bins N",
      "  --require-curriculum-stage-names LIST",
      "  --allow-invalid-head-geometry",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--eval") {
      config.evalPath = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = requireValue(argv, ++index, arg);
    } else if (arg === "--examples") {
      config.examplesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--manifest") {
      config.manifestPath = requireValue(argv, ++index, arg);
    } else if (arg === "--tokens") {
      config.tokensPath = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head-eval") {
      config.retrievalHeadEvalPath = requireValue(argv, ++index, arg);
    } else if (arg === "--sample-binding") {
      config.sampleBindingPath = requireValue(argv, ++index, arg);
    } else if (arg === "--generation-integrity") {
      config.generationIntegrityPath = requireValue(argv, ++index, arg);
    } else if (arg === "--identity-inference") {
      config.identityInferencePath = requireValue(argv, ++index, arg);
    } else if (arg === "--curriculum-stages") {
      config.curriculumStagesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--denoise-bridge") {
      config.denoiseBridgePath = requireValue(argv, ++index, arg);
    } else if (arg === "--grounded-corpus") {
      config.groundedCorpusPath = requireValue(argv, ++index, arg);
    } else if (arg === "--generative-eval") {
      config.generativeEvalPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-identity-inference") {
      config.requireIdentityInference = true;
    } else if (arg === "--require-curriculum-stages") {
      config.requireCurriculumStages = true;
    } else if (arg === "--require-denoise-bridge") {
      config.requireDenoiseBridge = true;
    } else if (arg === "--require-denoise-output-identity") {
      config.requireDenoiseOutputIdentity = true;
      config.requireDenoiseBridge = true;
    } else if (arg === "--min-denoise-bridge-unique-targets") {
      config.minDenoiseBridgeUniqueTargets = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-grounded-corpus") {
      config.requireGroundedCorpus = true;
    } else if (arg === "--min-grounded-source-overlap-tokens") {
      config.minGroundedSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-grounded-attribute-source-overlap-tokens") {
      config.minGroundedAttributeSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-grounded-source-placeholder-rows") {
      config.maxGroundedSourcePlaceholderRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-grounded-attribute-generic-rank-rows") {
      config.maxGroundedAttributeGenericRankRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-confidence-trace") {
      config.requireConfidenceTrace = true;
    } else if (arg === "--require-generative-eval") {
      config.requireGenerativeEval = true;
    } else if (arg === "--require-generative-output-identity") {
      config.requireGenerativeOutputIdentity = true;
      config.requireGenerativeEval = true;
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--min-total-top5-per-mille") {
      config.minTotalTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-text-top5-per-mille") {
      config.minTextTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-image-top5-per-mille") {
      config.minImageTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-task-targets") {
      config.minTaskTargets = parseTaskThresholdMap(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-task-top5-per-mille") {
      config.minTaskTop5PerMille = parseTaskThresholdMap(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-phase-targets") {
      config.minPhaseTargets = parsePhaseThresholdMap(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-top5-per-mille") {
      config.minGeneratedTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-top5-16-per-mille") {
      config.minGeneratedTop516PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-top5-px-per-mille") {
      config.minGeneratedTop5PxPerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-top1-per-mille") {
      config.minGeneratedRetrievalTop1PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-top5-per-mille") {
      config.minGeneratedRetrievalTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-margin") {
      config.minGeneratedRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-prompt-rows") {
      config.minGeneratedPromptRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-latent-top5-per-mille") {
      config.minLatentTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-rank-q8") {
      config.maxGeneratedMeanRankQ8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-rank-16-q8") {
      config.maxGeneratedMeanRank16Q8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-rank-px-q8") {
      config.maxGeneratedMeanRankPxQ8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-target-distance-q8") {
      config.maxGeneratedMeanTargetDistanceQ8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-target-distance-16-q8") {
      config.maxGeneratedMeanTargetDistance16Q8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-target-distance-px-q8") {
      config.maxGeneratedMeanTargetDistancePxQ8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-heldout-prompts") {
      config.requireHeldoutPrompts = true;
    } else if (arg === "--min-heldout-prompt-rows") {
      config.minHeldoutPromptRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-yes-top1") {
      config.minMatchYesTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-top1") {
      config.minMatchNoTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-image-top1") {
      config.minMatchNoImageTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-match-no-prompt-top1") {
      config.minMatchNoPromptTop1 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-retrieval-margin") {
      config.minRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-architecture-profile") {
      config.requireArchitectureProfile = true;
    } else if (arg === "--min-d-model") {
      config.minDModel = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heads") {
      config.minHeads = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-hidden-dim") {
      config.minHiddenDim = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-transformer-layers") {
      config.minTransformerLayers = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-context-seq-len") {
      config.minContextSeqLen = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-promoted-small-profile") {
      config.requirePromotedSmallProfile = true;
      config.requireArchitectureProfile = true;
    } else if (arg === "--require-corpus-version") {
      config.requireCorpusVersion = requireValue(argv, ++index, arg);
    } else if (arg === "--require-image-token-profile") {
      config.requireImageTokenProfile = requireValue(argv, ++index, arg);
    } else if (arg === "--require-image-token-channels") {
      config.requireImageTokenChannels = parseList(requireValue(argv, ++index, arg));
    } else if (arg === "--require-image-channel-token-stats") {
      config.requireImageChannelTokenStats = true;
    } else if (arg === "--min-image-channel-distinct-bins") {
      config.minImageChannelDistinctBins = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-curriculum-stage-names") {
      config.requireCurriculumStageNames = parseList(requireValue(argv, ++index, arg)).map(canonicalCurriculumStageName);
      config.requireCurriculumStages = true;
    } else if (arg === "--allow-invalid-head-geometry") {
      config.requireValidHeadGeometry = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  for (const [key, flag] of [
    ["evalPath", "--eval"],
    ["retrievalHeadEvalPath", "--retrieval-head-eval"],
    ["sampleBindingPath", "--sample-binding"],
    ["generationIntegrityPath", "--generation-integrity"],
  ]) {
    if (!config[key]) {
      throw new Error(`${flag} is required`);
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

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function parseList(value) {
  return String(value)
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function canonicalCurriculumStageName(stageName) {
  return stageName === "hard-negatives" ? "hard-negative" : stageName;
}

function parseTaskThresholdMap(value, flag) {
  const thresholds = {};
  const entries = String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) {
    throw new Error(`${flag} selected no task thresholds`);
  }
  for (const entry of entries) {
    const split = entry.indexOf("=");
    if (split <= 0) {
      throw new Error(`${flag} entry ${JSON.stringify(entry)} must be TASK=N`);
    }
    const task = entry.slice(0, split).trim();
    const rawValue = entry.slice(split + 1).trim();
    if (task !== "all" && !REQUIRED_TASKS.includes(task)) {
      throw new Error(`${flag} unknown task ${JSON.stringify(task)}`);
    }
    thresholds[task] = parseNonNegative(rawValue, flag);
  }
  return thresholds;
}

function parsePhaseThresholdMap(value, flag) {
  const thresholds = {};
  const entries = String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) {
    throw new Error(`${flag} selected no phase thresholds`);
  }
  for (const entry of entries) {
    const split = entry.indexOf("=");
    if (split <= 0) {
      throw new Error(`${flag} entry ${JSON.stringify(entry)} must be PHASE=N`);
    }
    const phase = entry.slice(0, split).trim();
    const rawValue = entry.slice(split + 1).trim();
    if (phase !== "all" && !EVAL_PHASES.includes(phase)) {
      throw new Error(`${flag} unknown phase ${JSON.stringify(phase)}`);
    }
    thresholds[phase] = parseNonNegative(rawValue, flag);
  }
  return thresholds;
}

function taskThreshold(thresholds, task) {
  return Number(thresholds?.[task] ?? thresholds?.all ?? 0);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function fnv64BytesHex(bytes) {
  let hash = FNV64_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function readJsonl(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  return text.split(/\r?\n/).filter(Boolean).map((line, rowIndex) => {
    const row = JSON.parse(line);
    row.__line = rowIndex + 1;
    return row;
  });
}

function tryReadJson(filePath, errors, label) {
  try {
    return readJson(filePath);
  } catch (error) {
    errors.push(`${label} ${filePath}: ${error.message}`);
    return null;
  }
}

function readTsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    throw new Error(`${filePath} is empty`);
  }
  const lines = text.split(/\r?\n/);
  const header = lines[0].split("\t");
  return lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.row_index = rowIndex + 2;
    return row;
  });
}

function readTokens(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (filePath.endsWith(".u16")) {
    if (bytes.length % 2 !== 0) {
      throw new Error(`${filePath} byte length ${bytes.length} is not divisible by 2`);
    }
    const tokens = [];
    for (let index = 0; index < bytes.length; index += 2) {
      tokens.push(bytes.readUInt16LE(index));
    }
    return tokens;
  }
  return Array.from(bytes);
}

function expectSchema(row, filePath, schema, errors) {
  if (row?.schema !== schema) {
    errors.push(`${filePath} schema ${JSON.stringify(row?.schema)} != ${schema}`);
  }
}

function checkEvalTrace(trace, filePath, config) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_attention_eval_trace.v1", errors);
  if (Number(trace.skipped_examples || 0) !== 0) {
    errors.push(`attention eval skipped_examples ${trace.skipped_examples} != 0`);
  }
  if (Number(trace.total?.invalid_contexts || 0) !== 0) {
    errors.push(`attention eval total invalid_contexts ${trace.total?.invalid_contexts} != 0`);
  }
  const taskSummary = {};
  for (const task of REQUIRED_TASKS) {
    const stats = trace.tasks?.[task];
    if (!stats) {
      errors.push(`attention eval is missing task ${task}`);
      continue;
    }
    const minTop5 = taskThreshold(config.minTaskTop5PerMille, task);
    const minTargets = taskThreshold(config.minTaskTargets, task);
    const targets = Number(stats.targets || 0);
    const top5 = Number(stats.top5_accuracy_per_mille || 0);
    taskSummary[task] = {
      ...metricSummary(stats),
      min_targets: minTargets,
      min_top5_per_mille: minTop5,
    };
    if (targets <= 0) {
      errors.push(`attention eval task ${task} has no targets`);
    }
    if (targets < minTargets) {
      errors.push(`attention eval task ${task} targets ${targets} < ${minTargets}`);
    }
    if (Number(stats.invalid_contexts || 0) !== 0) {
      errors.push(`attention eval task ${task} invalid_contexts ${stats.invalid_contexts} != 0`);
    }
    if (top5 < minTop5) {
      errors.push(`attention eval task ${task} top5 ${top5} < ${minTop5}`);
    }
  }
  const totalTop5 = Number(trace.total?.top5_accuracy_per_mille || 0);
  const textTop5 = Number(trace.text?.top5_accuracy_per_mille || 0);
  const imageTop5 = Number(trace.image?.top5_accuracy_per_mille || 0);
  const phaseSummary = {};
  for (const phase of EVAL_PHASES) {
    const stats = trace[phase] || {};
    const minTargets = taskThreshold(config.minPhaseTargets, phase);
    const targets = Number(stats.targets || 0);
    phaseSummary[phase] = {
      ...metricSummary(stats),
      min_targets: minTargets,
    };
    if (targets < minTargets) {
      errors.push(`attention eval ${phase} targets ${targets} < ${minTargets}`);
    }
  }
  const outputHeads = checkEvalOutputHeads(trace, errors);
  const architecture = architectureProfile(trace, config);
  errors.push(...architecture.errors);
  if (totalTop5 < config.minTotalTop5PerMille) {
    errors.push(`attention eval total top5 ${totalTop5} < ${config.minTotalTop5PerMille}`);
  }
  if (textTop5 < config.minTextTop5PerMille) {
    errors.push(`attention eval text top5 ${textTop5} < ${config.minTextTop5PerMille}`);
  }
  if (imageTop5 < config.minImageTop5PerMille) {
    errors.push(`attention eval image top5 ${imageTop5} < ${config.minImageTop5PerMille}`);
  }
  const taskPhaseReport = checkEvalTaskPhases(trace, errors);
  return {
    ok: errors.length === 0,
    errors,
    model: trace.model || "",
    model_hash: trace.model_hash || "",
    token_hash: trace.token_hash || "",
    example_count: Number(trace.example_count || 0),
    skipped_examples: Number(trace.skipped_examples || 0),
    architecture: dropErrors(architecture),
    total: metricSummary(trace.total || {}),
    phases: phaseSummary,
    output_heads: outputHeads,
    text: metricSummary(trace.text || {}),
    image: metricSummary(trace.image || {}),
    tasks: taskSummary,
    task_phases: taskPhaseReport.tasks,
    directional_groups: taskPhaseReport.directional_groups,
  };
}

function checkEvalTaskPhases(trace, errors) {
  const raw = trace.task_phases;
  const tasks = {};
  const directionalErrors = [];
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    const error = "attention eval is missing task_phases";
    errors.push(error);
    directionalErrors.push(error);
    return {
      present: false,
      tasks,
      directional_groups: directionalGroupReport({}, trace, directionalErrors),
    };
  }
  for (const task of REQUIRED_TASKS) {
    const phases = raw[task];
    if (!phases || typeof phases !== "object" || Array.isArray(phases)) {
      const error = `attention eval task_phases missing ${task}`;
      errors.push(error);
      directionalErrors.push(error);
      continue;
    }
    tasks[task] = {};
    let targets = 0;
    for (const phase of EVAL_PHASES) {
      if (!phases[phase]) {
        continue;
      }
      const summary = metricSummary(phases[phase]);
      tasks[task][phase] = summary;
      targets += summary.targets;
      if (summary.invalid_contexts !== 0) {
        errors.push(`attention eval task_phases ${task}:${phase} invalid_contexts ${summary.invalid_contexts} != 0`);
      }
    }
    if (targets <= 0) {
      const error = `attention eval task_phases ${task} has no phase targets`;
      errors.push(error);
      directionalErrors.push(error);
    }
  }
  const directional = directionalGroupReport(raw, trace, directionalErrors);
  errors.push(...directional.errors.filter((error) => !directionalErrors.includes(error)));
  return {
    present: true,
    tasks,
    directional_groups: directional,
  };
}

function directionalGroupReport(taskPhases, trace, initialErrors = []) {
  const errors = [...initialErrors];
  const groups = {};
  for (const group of PRODUCT_DIRECTIONAL_GROUPS) {
    const taskTargets = {};
    const phaseTargets = {};
    const taskStats = [];
    let targets = 0;
    const groupErrors = [];
    for (const task of group.tasks) {
      const stats = trace.tasks?.[task] || {};
      taskStats.push(stats);
      const taskTargetCount = Number(stats.targets || 0);
      taskTargets[task] = taskTargetCount;
      targets += taskTargetCount;
      if (taskTargetCount <= 0) {
        groupErrors.push(`attention eval directional group ${group.key} task ${task} has no targets`);
      }
      for (const phase of group.required_phases[task] || []) {
        const phaseTargetCount = Number(taskPhases?.[task]?.[phase]?.targets || 0);
        phaseTargets[`${task}:${phase}`] = phaseTargetCount;
        if (phaseTargetCount <= 0) {
          groupErrors.push(`attention eval directional group ${group.key} task ${task} phase ${phase} has no targets`);
        }
      }
    }
    groups[group.key] = {
      label: group.label,
      tasks: group.tasks,
      required_phases: group.required_phases,
      targets,
      stats: aggregateMetricSummaries(taskStats),
      task_targets: taskTargets,
      phase_targets: phaseTargets,
      ok: groupErrors.length === 0,
      errors: groupErrors,
    };
    errors.push(...groupErrors);
  }
  return {
    required: true,
    ok: errors.length === 0,
    groups,
    errors,
  };
}

function checkEvalOutputHeads(trace, errors) {
  const heads = trace.output_heads;
  const summary = {};
  if (!heads || typeof heads !== "object" || Array.isArray(heads)) {
    errors.push("attention eval is missing output_heads");
    return summary;
  }
  for (const headName of REQUIRED_OUTPUT_HEADS) {
    const head = heads[headName];
    const headSummary = {
      source: String(head?.source || ""),
      token_classes: Array.isArray(head?.token_classes) ? head.token_classes.map(String) : [],
      token_ranges: Array.isArray(head?.token_ranges) ? head.token_ranges : [],
      allowed_token_count: Number(head?.allowed_token_count || 0),
      stats: metricSummary(head?.stats || {}),
    };
    summary[headName] = headSummary;
    if (!head || typeof head !== "object" || Array.isArray(head)) {
      errors.push(`attention eval output_heads missing ${headName}`);
      continue;
    }
    if (headSummary.source !== "nsrllmm-output-token-head") {
      errors.push(`attention eval output_heads.${headName}.source ${JSON.stringify(head.source)} != nsrllmm-output-token-head`);
    }
    if (headSummary.token_classes.length === 0) {
      errors.push(`attention eval output_heads.${headName} has no token_classes`);
    }
    if (headSummary.token_ranges.length === 0) {
      errors.push(`attention eval output_heads.${headName} has no token_ranges`);
    }
    if (headSummary.allowed_token_count <= 0) {
      errors.push(`attention eval output_heads.${headName}.allowed_token_count ${headSummary.allowed_token_count} must be > 0`);
    }
    if (headSummary.stats.targets <= 0) {
      errors.push(`attention eval output_heads.${headName} has no targets`);
    }
    if (headSummary.stats.invalid_contexts !== 0) {
      errors.push(`attention eval output_heads.${headName} invalid_contexts ${headSummary.stats.invalid_contexts} != 0`);
    }
  }
  const specialTargets = Number(trace.special?.targets || 0);
  const textTargets = Number(trace.prompt?.targets || 0) + Number(trace.text?.targets || 0);
  const imageTargets = Number(trace.image?.targets || 0);
  if (summary.special_head && summary.special_head.stats.targets !== specialTargets) {
    errors.push(`attention eval output_heads.special_head targets ${summary.special_head.stats.targets} != special targets ${specialTargets}`);
  }
  if (summary.text_head && summary.text_head.stats.targets !== textTargets) {
    errors.push(`attention eval output_heads.text_head targets ${summary.text_head.stats.targets} != prompt+text targets ${textTargets}`);
  }
  if (summary.image_head && summary.image_head.stats.targets !== imageTargets) {
    errors.push(`attention eval output_heads.image_head targets ${summary.image_head.stats.targets} != image targets ${imageTargets}`);
  }
  return summary;
}

function checkCorpusContract(config) {
  const errors = [];
  const required =
    Boolean(config.requireCorpusVersion) ||
    Boolean(config.requireImageTokenProfile) ||
    config.requireImageTokenChannels.length > 0 ||
    config.requireImageChannelTokenStats;
  let manifest = null;
  let examples = [];
  if (config.manifestPath) {
    manifest = tryReadJson(config.manifestPath, errors, "corpus manifest");
  } else if (required) {
    errors.push("corpus manifest is required for corpus contract gates");
  }
  if (config.examplesPath) {
    try {
      examples = readJsonl(config.examplesPath);
    } catch (error) {
      errors.push(`corpus examples ${config.examplesPath} could not be read: ${error.message}`);
    }
  } else if (required) {
    errors.push("corpus examples are required for corpus contract gates");
  }

  const manifestSummary = summarizeCorpusManifest(manifest);
  if (manifest) {
    if (config.requireCorpusVersion && manifest.corpus_version !== config.requireCorpusVersion) {
      errors.push(
        `corpus manifest corpus_version ${JSON.stringify(manifest.corpus_version || "")} != ${JSON.stringify(config.requireCorpusVersion)}`,
      );
    }
    if (config.requireImageTokenProfile && manifest.image_token_profile !== config.requireImageTokenProfile) {
      errors.push(
        `corpus manifest image_token_profile ${JSON.stringify(manifest.image_token_profile || "")} != ${JSON.stringify(config.requireImageTokenProfile)}`,
      );
    }
    const manifestChannels = Array.isArray(manifest.image_token_channels)
      ? manifest.image_token_channels.map((channel) => String(channel))
      : [];
    if (config.requireImageTokenChannels.length > 0 && manifestChannels.length === 0) {
      errors.push("corpus manifest is missing image_token_channels");
    }
    for (const channel of config.requireImageTokenChannels) {
      if (!manifestChannels.includes(channel)) {
        errors.push(`corpus manifest image_token_channels missing ${channel}`);
      }
    }
    if (config.requireImageChannelTokenStats) {
      errors.push(...checkImageChannelTokenStats(config, manifest, manifestChannels));
    }
  }

  const examplesSummary = summarizeCorpusExamples(examples, config);
  if (required && config.examplesPath && examplesSummary.v2_records <= 0) {
    errors.push("corpus examples have no v2 records");
  }
  if (
    config.requireImageTokenProfile &&
    examplesSummary.v2_records > 0 &&
    Number(examplesSummary.image_token_profiles[config.requireImageTokenProfile] || 0) !== examplesSummary.v2_records
  ) {
    errors.push(
      `corpus examples image_token_profile ${config.requireImageTokenProfile} covers ${Number(
        examplesSummary.image_token_profiles[config.requireImageTokenProfile] || 0,
      )}/${examplesSummary.v2_records} v2 records`,
    );
  }
  if (config.requireImageTokenChannels.length > 0 && examplesSummary.missing_image_token_channels > 0) {
    errors.push(`corpus examples have ${examplesSummary.missing_image_token_channels} v2 records without image_token_channels`);
  }
  for (const channel of config.requireImageTokenChannels) {
    const rows = Number(examplesSummary.required_channel_rows[channel] || 0);
    if (examplesSummary.v2_records > 0 && rows !== examplesSummary.v2_records) {
      errors.push(`corpus examples image_token_channels ${channel} covers ${rows}/${examplesSummary.v2_records} v2 records`);
    }
  }
  if (required && examples.length > 0) {
    errors.push(...checkCorpusTaskCoverage(examplesSummary, Number(manifest?.rows || 72)));
  }
  const taskMarkerIntegrity = checkCorpusTaskMarkerIntegrity(config, manifest, examples, required);
  errors.push(...taskMarkerIntegrity.errors);
  const taskModalityIntegrity = checkCorpusTaskModalityIntegrity(config, manifest, examples, required);
  errors.push(...taskModalityIntegrity.errors);
  const imageChannelMarkerIntegrity = checkCorpusImageChannelMarkerIntegrity(config, manifest, examples, required);
  errors.push(...imageChannelMarkerIntegrity.errors);

  return {
    ok: errors.length === 0,
    present: Boolean(config.manifestPath || config.examplesPath),
    errors,
    manifest: config.manifestPath,
    examples: config.examplesPath,
    required_corpus_version: config.requireCorpusVersion || null,
    required_image_token_profile: config.requireImageTokenProfile || null,
    required_image_token_channels: config.requireImageTokenChannels,
    require_image_channel_token_stats: config.requireImageChannelTokenStats,
    min_image_channel_distinct_bins: config.minImageChannelDistinctBins,
    manifest_summary: manifestSummary,
    examples_summary: examplesSummary,
    task_marker_integrity: taskMarkerIntegrity,
    task_modality_integrity: taskModalityIntegrity,
    image_channel_marker_integrity: imageChannelMarkerIntegrity,
  };
}

function checkImageChannelTokenStats(config, manifest, channels) {
  const errors = [];
  const stats = manifest.image_token_channel_stats;
  if (!stats || typeof stats !== "object" || Array.isArray(stats)) {
    return ["corpus manifest is missing image_token_channel_stats"];
  }
  const requiredChannels = config.requireImageTokenChannels.length > 0 ? config.requireImageTokenChannels : channels;
  const expectedRecords = Number(manifest.rows || 0);
  const expectedTokensPerRecord = Number(manifest.signature_bins || 0);
  for (const channel of requiredChannels) {
    const row = stats[channel];
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      errors.push(`corpus manifest image_token_channel_stats missing ${channel}`);
      continue;
    }
    const records = Number(row.records || 0);
    const tokensPerRecord = Number(row.tokens_per_record || 0);
    const activeRecords = Number(row.active_records || 0);
    const multiBinRecords = Number(row.multi_bin_records || 0);
    const nonzeroTokens = Number(row.nonzero_tokens || 0);
    const distinctBins = Number(row.distinct_bins || 0);
    const maxBin = Number(row.max_bin || 0);
    const uniqueRecordHashes = Number(row.unique_record_hashes || 0);
    const duplicateRecordHashes = Number(row.duplicate_record_hashes || 0);
    if (expectedRecords > 0 && records !== expectedRecords) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} records ${records} != ${expectedRecords}`);
    }
    if (expectedTokensPerRecord > 0 && tokensPerRecord !== expectedTokensPerRecord) {
      errors.push(
        `corpus manifest image_token_channel_stats ${channel} tokens_per_record ${tokensPerRecord} != ${expectedTokensPerRecord}`,
      );
    }
    if (records > 0 && activeRecords !== records) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} active_records ${activeRecords} != ${records}`);
    }
    if (records > 0 && multiBinRecords !== records) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} multi_bin_records ${multiBinRecords} != ${records}`);
    }
    if (nonzeroTokens <= 0) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} nonzero_tokens ${nonzeroTokens} <= 0`);
    }
    if (distinctBins < config.minImageChannelDistinctBins) {
      errors.push(
        `corpus manifest image_token_channel_stats ${channel} distinct_bins ${distinctBins} < ${config.minImageChannelDistinctBins}`,
      );
    }
    if (maxBin <= 0) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} max_bin ${maxBin} <= 0`);
    }
    if (records > 0 && uniqueRecordHashes !== records) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} unique_record_hashes ${uniqueRecordHashes} != records ${records}`);
    }
    if (duplicateRecordHashes !== 0) {
      errors.push(`corpus manifest image_token_channel_stats ${channel} duplicate_record_hashes ${duplicateRecordHashes} != 0`);
    }
  }
  return errors;
}

function summarizeCorpusManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return {
      schema: "",
      corpus_version: "",
      image_token_profile: "",
      image_token_channels: [],
      image_token_channel_stats: {},
      examples: 0,
      training_sequences: 0,
      token_hash: "",
    };
  }
  return {
    schema: manifest.schema || "",
    corpus_version: manifest.corpus_version || "",
    image_token_profile: manifest.image_token_profile || "",
    image_token_channels: Array.isArray(manifest.image_token_channels) ? manifest.image_token_channels.map(String) : [],
    image_token_channel_stats:
      manifest.image_token_channel_stats && typeof manifest.image_token_channel_stats === "object"
        ? manifest.image_token_channel_stats
        : {},
    examples: Number(manifest.examples || 0),
    training_sequences: Number(manifest.training_sequences || 0),
    token_hash: manifest.token_hash || "",
  };
}

function summarizeCorpusExamples(examples, config) {
  const imageTokenProfiles = new Map();
  const requiredChannelRows = Object.fromEntries(config.requireImageTokenChannels.map((channel) => [channel, 0]));
  const allSpirits = new Set();
  const taskGroups = new Map();
  const errors = [];
  let v2Records = 0;
  let missingImageTokenProfile = 0;
  let missingImageTokenChannels = 0;
  for (const row of examples) {
    const task = row.task || "canonical-joint";
    const spiritId = normalizedCorpusSpiritId(row.spirit_id);
    if (spiritId !== null) {
      allSpirits.add(spiritId);
    }
    const taskGroup = ensureCorpusCoverageGroup(taskGroups, task);
    taskGroup.records += 1;
    if (spiritId !== null) {
      taskGroup.spirits.add(spiritId);
    }
    if (row?.schema !== "nsrl.solomon_multimodal_example.v2") {
      continue;
    }
    v2Records += 1;
    const profile = String(row.image_token_profile || "");
    if (!profile) {
      missingImageTokenProfile += 1;
    } else {
      imageTokenProfiles.set(profile, (imageTokenProfiles.get(profile) || 0) + 1);
    }
    const channels = Array.isArray(row.image_token_channels) ? row.image_token_channels.map((channel) => String(channel)) : [];
    if (channels.length === 0) {
      missingImageTokenChannels += 1;
    }
    for (const channel of config.requireImageTokenChannels) {
      if (channels.includes(channel)) {
        requiredChannelRows[channel] += 1;
      }
    }
    if (task === "match") {
      const label = String(row.match_label || row.text || "").toLowerCase();
      if (label !== "yes" && label !== "no") {
        errors.push(`corpus examples line ${row.__line}: match row has invalid label ${JSON.stringify(label)}`);
      } else {
        const labelGroup = ensureCorpusCoverageGroup(taskGroup.labels, label);
        labelGroup.records += 1;
        if (spiritId !== null) {
          labelGroup.spirits.add(spiritId);
        }
        if (label === "no") {
          const negativeSpiritId = normalizedCorpusSpiritId(row.negative_spirit_id);
          if (negativeSpiritId === null) {
            errors.push(`corpus examples line ${row.__line}: negative match row is missing negative_spirit_id`);
          } else if (negativeSpiritId === spiritId) {
            errors.push(`corpus examples line ${row.__line}: negative match row points at its own spirit_id`);
          }
          const role = corpusMatchNegativeRole(row);
          if (role !== "image" && role !== "prompt") {
            errors.push(`corpus examples line ${row.__line}: negative match row has invalid negative_role ${JSON.stringify(row.negative_role)}`);
          } else {
            const roleGroup = ensureCorpusCoverageGroup(labelGroup.roles, role);
            roleGroup.records += 1;
            if (spiritId !== null) {
              roleGroup.spirits.add(spiritId);
            }
          }
          if (String(row.negative_selection || "") !== "nearest-image-token") {
            errors.push(`corpus examples line ${row.__line}: negative match row negative_selection ${JSON.stringify(row.negative_selection || "")} != nearest-image-token`);
          }
          if (Number(row.negative_image_token_rank) !== 1) {
            errors.push(`corpus examples line ${row.__line}: negative match row negative_image_token_rank ${JSON.stringify(row.negative_image_token_rank || "")} != 1`);
          }
          const distance = Number(row.negative_image_token_distance);
          if (!Number.isInteger(distance) || distance <= 0) {
            errors.push(`corpus examples line ${row.__line}: negative match row has invalid negative_image_token_distance ${JSON.stringify(row.negative_image_token_distance || "")}`);
          }
        }
      }
    }
  }
  return {
    records: examples.length,
    distinct_spirits: allSpirits.size,
    v2_records: v2Records,
    missing_image_token_profile: missingImageTokenProfile,
    missing_image_token_channels: missingImageTokenChannels,
    image_token_profiles: Object.fromEntries([...imageTokenProfiles.entries()].sort(([left], [right]) => left.localeCompare(right))),
    required_channel_rows: requiredChannelRows,
    tasks: Object.fromEntries(
      [...taskGroups.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([task, group]) => [
        task,
        corpusCoverageGroupSummary(group),
      ]),
    ),
    coverage_errors: errors,
  };
}

function checkCorpusTaskCoverage(examplesSummary, expectSpirits) {
  const errors = [...(examplesSummary.coverage_errors || [])];
  const expected = Number(expectSpirits || 0);
  if (expected <= 0) {
    return errors;
  }
  if (Number(examplesSummary.distinct_spirits || 0) !== expected) {
    errors.push(`corpus examples distinct spirits ${Number(examplesSummary.distinct_spirits || 0)} != ${expected}`);
  }
  for (const task of REQUIRED_TASKS) {
    const taskCoverage = examplesSummary.tasks?.[task];
    if (!taskCoverage) {
      errors.push(`corpus examples are missing required task: ${task}`);
      continue;
    }
    if (Number(taskCoverage.spirits || 0) !== expected) {
      errors.push(`corpus examples task ${task} covers ${Number(taskCoverage.spirits || 0)} spirits, expected ${expected}`);
    }
    if (task === "match") {
      const labels = taskCoverage.labels || {};
      for (const label of ["yes", "no"]) {
        if (!labels[label]) {
          errors.push(`corpus examples match task is missing ${label} rows`);
        } else if (Number(labels[label].spirits || 0) !== expected) {
          errors.push(`corpus examples match ${label} rows cover ${Number(labels[label].spirits || 0)} spirits, expected ${expected}`);
        }
      }
      const negativeRoles = labels.no?.roles || {};
      for (const role of ["image", "prompt"]) {
        if (!negativeRoles[role]) {
          errors.push(`corpus examples match no rows are missing ${role} negative_role rows`);
        } else if (Number(negativeRoles[role].spirits || 0) !== expected) {
          errors.push(
            `corpus examples match no ${role} rows cover ${Number(negativeRoles[role].spirits || 0)} spirits, expected ${expected}`,
          );
        }
      }
    }
  }
  return errors;
}

function ensureCorpusCoverageGroup(map, key) {
  if (!map.has(key)) {
    map.set(key, { records: 0, spirits: new Set(), labels: new Map(), roles: new Map() });
  }
  return map.get(key);
}

function corpusCoverageGroupSummary(group) {
  const summary = {
    records: group.records,
    spirits: group.spirits.size,
  };
  if (group.labels.size > 0) {
    summary.labels = Object.fromEntries(
      [...group.labels.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([label, labelGroup]) => [
        label,
        corpusCoverageGroupSummary(labelGroup),
      ]),
    );
  }
  if (group.roles.size > 0) {
    summary.roles = Object.fromEntries(
      [...group.roles.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([role, roleGroup]) => [
        role,
        corpusCoverageGroupSummary(roleGroup),
      ]),
    );
  }
  return summary;
}

function corpusMatchNegativeRole(row) {
  const role = String(row.negative_role || "image").toLowerCase();
  if (role === "prompt" || role === "text" || role === "name") {
    return "prompt";
  }
  if (role === "image" || role === "seal") {
    return "image";
  }
  return role;
}

function normalizedCorpusSpiritId(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function absentCorpusTaskMarkerIntegrity() {
  return {
    ok: true,
    present: false,
    tokens: "",
    checked_records: 0,
    hash_mismatches: 0,
    marker_mismatches: 0,
    out_of_bounds: 0,
    missing_offsets: 0,
    by_task: {},
    errors: [],
  };
}

function checkCorpusTaskMarkerIntegrity(config, manifest, examples, required) {
  const v2Examples = examples.filter((row) => row?.schema === "nsrl.solomon_multimodal_example.v2");
  const tokensPath = resolveCorpusTokensPath(config, manifest);
  if (v2Examples.length === 0 && !tokensPath) {
    return absentCorpusTaskMarkerIntegrity();
  }
  if (!tokensPath) {
    if (!required) {
      return absentCorpusTaskMarkerIntegrity();
    }
    const report = absentCorpusTaskMarkerIntegrity();
    report.ok = false;
    report.errors = required
      ? ["corpus task marker integrity requires manifest corpus_tokens_u8/corpus_tokens_u16 or --tokens"]
      : [];
    return report;
  }
  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ...absentCorpusTaskMarkerIntegrity(),
      ok: false,
      present: true,
      tokens: tokensPath,
      errors: [`corpus token file ${tokensPath}: ${error.message}`],
    };
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const errors = [];
  const byTask = new Map();
  let checkedRecords = 0;
  let hashMismatches = 0;
  let markerMismatches = 0;
  let outOfBounds = 0;
  let missingOffsets = 0;
  for (const row of v2Examples) {
    const task = row.task || "";
    const expected = expectedCorpusTaskMarker(task, layout);
    if (!expected) {
      continue;
    }
    const taskSummary = ensureCorpusTaskMarkerGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`corpus examples line ${row.__line}: ${task} missing valid token_offset/token_count`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `corpus examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length}`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const actualMarker = slice.slice(0, expected.length);
    if (!sameCorpusTaskMarker(actualMarker, expected)) {
      markerMismatches += 1;
      taskSummary.marker_mismatches += 1;
      errors.push(
        `corpus examples line ${row.__line}: ${task} token marker ${JSON.stringify(actualMarker)} != ${JSON.stringify(expected)}`,
      );
    }
    if (row.token_hash) {
      const actualHash = fnv64TokenHex(slice);
      if (actualHash !== row.token_hash) {
        hashMismatches += 1;
        taskSummary.hash_mismatches += 1;
        errors.push(`corpus examples line ${row.__line}: ${task} token_hash ${actualHash} != ${row.token_hash}`);
      }
    }
  }
  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
    checked_records: checkedRecords,
    hash_mismatches: hashMismatches,
    marker_mismatches: markerMismatches,
    out_of_bounds: outOfBounds,
    missing_offsets: missingOffsets,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    errors,
  };
}

function absentCorpusTaskModalityIntegrity() {
  return {
    ok: true,
    present: false,
    tokens: "",
    checked_records: 0,
    missing_offsets: 0,
    out_of_bounds: 0,
    modality_mismatches: 0,
    by_task: {},
    errors: [],
  };
}

function checkCorpusTaskModalityIntegrity(config, manifest, examples, required) {
  const checkedExamples = examples.filter((row) => expectedCorpusTaskModalities(row.task || "canonical-joint"));
  const tokensPath = resolveCorpusTokensPath(config, manifest);
  if (checkedExamples.length === 0 && !tokensPath) {
    return absentCorpusTaskModalityIntegrity();
  }
  if (!tokensPath) {
    if (!required) {
      return absentCorpusTaskModalityIntegrity();
    }
    const report = absentCorpusTaskModalityIntegrity();
    report.ok = false;
    report.errors = ["corpus task modality integrity requires manifest corpus_tokens_u8/corpus_tokens_u16 or --tokens"];
    return report;
  }
  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ...absentCorpusTaskModalityIntegrity(),
      ok: false,
      present: true,
      tokens: tokensPath,
      errors: [`corpus token file ${tokensPath}: ${error.message}`],
    };
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const errors = [];
  const byTask = new Map();
  let checkedRecords = 0;
  let missingOffsets = 0;
  let outOfBounds = 0;
  let modalityMismatches = 0;
  for (const row of checkedExamples) {
    const task = row.task || "canonical-joint";
    const taskSummary = ensureCorpusTaskModalityGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`corpus examples line ${row.__line}: ${task} missing valid token_offset/token_count for modality order`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `corpus examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for modality order`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const rowErrors = checkCorpusTaskModalityOrder(task, slice, layout);
    if (rowErrors.length > 0) {
      modalityMismatches += 1;
      taskSummary.modality_mismatches += 1;
      for (const error of rowErrors) {
        errors.push(`corpus examples line ${row.__line}: ${error}`);
      }
    }
  }
  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
    checked_records: checkedRecords,
    missing_offsets: missingOffsets,
    out_of_bounds: outOfBounds,
    modality_mismatches: modalityMismatches,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    errors,
  };
}

function absentCorpusImageChannelMarkerIntegrity() {
  return {
    ok: true,
    present: false,
    tokens: "",
    required_channels: [],
    checked_records: 0,
    missing_offsets: 0,
    out_of_bounds: 0,
    missing_image_markers: 0,
    missing_channel_markers: 0,
    short_channel_payloads: 0,
    bad_channel_payloads: 0,
    channel_order_mismatches: 0,
    by_task: {},
    by_channel: {},
    errors: [],
  };
}

function checkCorpusImageChannelMarkerIntegrity(config, manifest, examples, required) {
  const requiredChannels = config.requireImageTokenChannels.map((channel) => String(channel));
  if (requiredChannels.length === 0) {
    return absentCorpusImageChannelMarkerIntegrity();
  }
  const v2ImageExamples = examples.filter(
    (row) => row?.schema === "nsrl.solomon_multimodal_example.v2" && IMAGE_BEARING_TASKS.has(String(row.task || "")),
  );
  const tokensPath = resolveCorpusTokensPath(config, manifest);
  if (!tokensPath) {
    if (!required) {
      return absentCorpusImageChannelMarkerIntegrity();
    }
    const report = absentCorpusImageChannelMarkerIntegrity();
    report.ok = false;
    report.required_channels = requiredChannels;
    report.errors = ["corpus image channel marker integrity requires manifest corpus_tokens_u8/corpus_tokens_u16 or --tokens"];
    return report;
  }
  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ...absentCorpusImageChannelMarkerIntegrity(),
      ok: false,
      present: true,
      tokens: tokensPath,
      required_channels: requiredChannels,
      errors: [`corpus token file ${tokensPath}: ${error.message}`],
    };
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const imageToken = Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image);
  const imageBase = Number(layout.image_base ?? TASK_TOKEN_LAYOUT_FALLBACK.image_base);
  const imageBins = Number(layout.image_bins ?? TASK_TOKEN_LAYOUT_FALLBACK.image_bins);
  const payloadTokens = Number(manifest?.signature_bins || IMAGE_CHANNEL_PAYLOAD_TOKENS_FALLBACK);
  const errors = [];
  const byTask = new Map();
  const byChannel = new Map();
  let checkedRecords = 0;
  let missingOffsets = 0;
  let outOfBounds = 0;
  let missingImageMarkers = 0;
  let missingChannelMarkers = 0;
  let shortChannelPayloads = 0;
  let badChannelPayloads = 0;
  let channelOrderMismatches = 0;
  if (v2ImageExamples.length === 0) {
    errors.push("corpus image channel marker integrity found no image-bearing v2 records");
  }
  for (const row of v2ImageExamples) {
    const task = row.task || "";
    const taskSummary = ensureCorpusImageChannelMarkerGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`corpus examples line ${row.__line}: ${task} missing valid token_offset/token_count for image channel markers`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `corpus examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for image channel markers`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const imageIndex = slice.indexOf(imageToken);
    if (imageIndex < 0) {
      missingImageMarkers += 1;
      taskSummary.missing_image_markers += 1;
      errors.push(`corpus examples line ${row.__line}: ${task} token slice is missing IMAGE marker ${imageToken}`);
      continue;
    }
    let previousChannelPosition = imageIndex;
    for (const channel of requiredChannels) {
      const channelSummary = ensureCorpusImageChannelMarkerGroup(byChannel, channel);
      channelSummary.checked_records += 1;
      const marker = expectedImageChannelMarker(channel, layout);
      if (!Number.isInteger(marker)) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`corpus image channel ${channel} has no token_layout image_channel_${channel} marker`);
        continue;
      }
      const markerCheck = findImageChannelPayload(slice, imageIndex + 1, marker, imageBase, imageBins, payloadTokens);
      if (!markerCheck.found) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`corpus examples line ${row.__line}: ${task} missing image channel marker ${channel}:${marker}`);
        continue;
      }
      if (markerCheck.shortPayload) {
        shortChannelPayloads += 1;
        taskSummary.short_channel_payloads += 1;
        channelSummary.short_channel_payloads += 1;
        errors.push(
          `corpus examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has fewer than ${payloadTokens} tokens`,
        );
        continue;
      }
      if (markerCheck.badPayload) {
        badChannelPayloads += 1;
        taskSummary.bad_channel_payloads += 1;
        channelSummary.bad_channel_payloads += 1;
        errors.push(
          `corpus examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has token outside ${imageBase}..${
            imageBase + imageBins - 1
          }`,
        );
        continue;
      }
      if (markerCheck.position <= previousChannelPosition) {
        channelOrderMismatches += 1;
        taskSummary.channel_order_mismatches += 1;
        channelSummary.channel_order_mismatches += 1;
        errors.push(`corpus examples line ${row.__line}: ${task} image channel ${channel}:${marker} is out of order`);
      }
      previousChannelPosition = markerCheck.position;
      taskSummary.found_markers += 1;
      channelSummary.found_markers += 1;
    }
  }
  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
    required_channels: requiredChannels,
    checked_records: checkedRecords,
    missing_offsets: missingOffsets,
    out_of_bounds: outOfBounds,
    missing_image_markers: missingImageMarkers,
    missing_channel_markers: missingChannelMarkers,
    short_channel_payloads: shortChannelPayloads,
    bad_channel_payloads: badChannelPayloads,
    channel_order_mismatches: channelOrderMismatches,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    by_channel: Object.fromEntries([...byChannel.entries()].sort(([left], [right]) => left.localeCompare(right))),
    errors,
  };
}

function ensureCorpusImageChannelMarkerGroup(map, key) {
  if (!map.has(key)) {
    map.set(key, {
      checked_records: 0,
      found_markers: 0,
      missing_offsets: 0,
      out_of_bounds: 0,
      missing_image_markers: 0,
      missing_channel_markers: 0,
      short_channel_payloads: 0,
      bad_channel_payloads: 0,
      channel_order_mismatches: 0,
    });
  }
  return map.get(key);
}

function expectedImageChannelMarker(channel, layout) {
  const key = `image_channel_${String(channel).replace(/-/g, "_")}`;
  return Number(layout[key]);
}

function findImageChannelPayload(slice, startIndex, marker, imageBase, imageBins, payloadTokens) {
  let position = slice.indexOf(marker, startIndex);
  let sawBadPayload = false;
  while (position >= 0) {
    const payloadStart = position + 1;
    const payloadEnd = payloadStart + payloadTokens;
    if (payloadEnd > slice.length) {
      return { found: true, position, shortPayload: true, badPayload: false };
    }
    let badPayload = false;
    for (let index = payloadStart; index < payloadEnd; index += 1) {
      const token = Number(slice[index]);
      if (token < imageBase || token >= imageBase + imageBins) {
        badPayload = true;
        break;
      }
    }
    if (!badPayload) {
      return { found: true, position, shortPayload: false, badPayload: false };
    }
    sawBadPayload = true;
    position = slice.indexOf(marker, position + 1);
  }
  if (sawBadPayload) {
    return { found: true, position: -1, shortPayload: false, badPayload: true };
  }
  return { found: false, position: -1, shortPayload: false, badPayload: false };
}

function resolveCorpusTokensPath(config, manifest) {
  if (config.tokensPath) {
    return config.tokensPath;
  }
  const tokenRef = manifest?.corpus_tokens_u8 || manifest?.corpus_tokens_u16 || "";
  if (!tokenRef) {
    return "";
  }
  if (path.isAbsolute(tokenRef)) {
    return tokenRef;
  }
  return config.manifestPath ? path.resolve(path.dirname(config.manifestPath), tokenRef) : tokenRef;
}

function ensureCorpusTaskMarkerGroup(map, task) {
  if (!map.has(task)) {
    map.set(task, {
      checked_records: 0,
      hash_mismatches: 0,
      marker_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
    });
  }
  return map.get(task);
}

function ensureCorpusTaskModalityGroup(map, task) {
  if (!map.has(task)) {
    map.set(task, {
      checked_records: 0,
      missing_offsets: 0,
      out_of_bounds: 0,
      modality_mismatches: 0,
    });
  }
  return map.get(task);
}

function expectedCorpusTaskModalities(task) {
  if (task === "canonical-joint") return ["prompt", "text", "image"];
  if (task === "identify" || task === "explain") return ["prompt", "text"];
  if (task === "text-to-image" || task === "description-to-image") return ["prompt", "image"];
  if (task === "image-to-text" || task === "image-to-explain") return ["image", "text"];
  if (task === "text-image-explain" || task === "match") return ["prompt", "image", "text"];
  if (task === "image-to-attributes") return ["image", "prompt", "text"];
  return null;
}

function checkCorpusTaskModalityOrder(task, slice, layout) {
  const expected = expectedCorpusTaskModalities(task);
  if (!expected) {
    return [];
  }
  const markerTokens = {
    prompt: Number(layout.prompt ?? TASK_TOKEN_LAYOUT_FALLBACK.prompt),
    text: Number(layout.text ?? TASK_TOKEN_LAYOUT_FALLBACK.text),
    image: Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image),
  };
  const eosToken = Number(layout.eos ?? TASK_TOKEN_LAYOUT_FALLBACK.eos);
  const eosIndex = slice.indexOf(eosToken, 1);
  const searchEnd = eosIndex >= 0 ? eosIndex : slice.length;
  const positions = Object.fromEntries(
    Object.entries(markerTokens).map(([name, token]) => [
      name,
      corpusMarkerPositions(slice, token).filter((position) => position > 0 && position < searchEnd),
    ]),
  );
  const errors = [];
  if (eosIndex < 0) {
    errors.push(`${task} modality order is missing EOS marker ${eosToken}`);
  }
  for (const name of expected) {
    const found = positions[name] || [];
    if (found.length !== 1) {
      errors.push(`${task} modality order expected exactly one ${name.toUpperCase()} marker, found ${found.length}`);
    }
  }
  for (const name of Object.keys(markerTokens)) {
    if (!expected.includes(name) && (positions[name] || []).length > 0) {
      errors.push(`${task} modality order has unexpected ${name.toUpperCase()} marker before EOS`);
    }
  }
  let previousName = "";
  let previousPosition = -1;
  for (const name of expected) {
    const found = positions[name] || [];
    if (found.length !== 1) {
      continue;
    }
    const position = found[0];
    if (position <= previousPosition) {
      errors.push(
        `${task} modality order ${expected.join("->")} has ${name.toUpperCase()} at ${position} after ${previousName.toUpperCase()} at ${previousPosition}`,
      );
    }
    previousName = name;
    previousPosition = position;
  }
  return errors;
}

function corpusMarkerPositions(tokens, marker) {
  const positions = [];
  for (let index = 0; index < tokens.length; index += 1) {
    if (Number(tokens[index]) === marker) {
      positions.push(index);
    }
  }
  return positions;
}

function expectedCorpusTaskMarker(task, layout) {
  const bos = Number(layout.bos ?? TASK_TOKEN_LAYOUT_FALLBACK.bos);
  const prompt = Number(layout.prompt ?? TASK_TOKEN_LAYOUT_FALLBACK.prompt);
  const image = Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image);
  if (task === "identify") return [bos, Number(layout.task_identify), prompt];
  if (task === "text-to-image" || task === "description-to-image") {
    return [bos, Number(layout.task_text_to_image), prompt];
  }
  if (task === "image-to-text") return [bos, Number(layout.task_image_to_text), image];
  if (task === "image-to-explain" || task === "image-to-attributes") {
    return [bos, Number(layout.task_explain), image];
  }
  if (task === "text-image-explain" || task === "explain") {
    return [bos, Number(layout.task_explain), prompt];
  }
  if (task === "match") return [bos, Number(layout.task_match), prompt];
  return null;
}

function sameCorpusTaskMarker(actual, expected) {
  if (actual.length !== expected.length) {
    return false;
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (Number(actual[index]) !== Number(expected[index])) {
      return false;
    }
  }
  return true;
}

function fnv64TokenHex(tokens) {
  let hash = FNV64_OFFSET;
  for (const token of tokens) {
    hash ^= BigInt(Number(token) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function architectureProfile(trace, config) {
  const dModel = Number(trace.d_model || 0);
  const heads = Number(trace.heads || 0);
  const hiddenDim = Number(trace.hidden_dim || 0);
  const transformerLayers = Number(trace.transformer_layers || 0);
  const contextSeqLen = Number(trace.context_seq_len || 0);
  const headDim = heads > 0 && dModel > 0 && dModel % heads === 0 ? dModel / heads : 0;
  const headDimPowerOfFour = isPowerOfFour(headDim);
  const hasProfile = dModel > 0 && heads > 0 && hiddenDim > 0 && transformerLayers > 0 && contextSeqLen > 0;
  const promotedSmallProfile = promotedSmallProfileStatus({
    dModel,
    heads,
    headDim,
    headDimPowerOfFour,
    hiddenDim,
    transformerLayers,
    contextSeqLen,
  });
  const errors = [];
  if (config.requireArchitectureProfile && !hasProfile) {
    errors.push("attention eval is missing architecture profile fields");
  }
  if (hasProfile && dModel % heads !== 0) {
    errors.push(`attention d_model ${dModel} is not divisible by heads ${heads}`);
  }
  if (hasProfile && config.requireValidHeadGeometry && !headDimPowerOfFour) {
    errors.push(`attention head_dim ${headDim} is not a power of four`);
  }
  if (dModel < config.minDModel) {
    errors.push(`attention d_model ${dModel} < ${config.minDModel}`);
  }
  if (heads < config.minHeads) {
    errors.push(`attention heads ${heads} < ${config.minHeads}`);
  }
  if (hiddenDim < config.minHiddenDim) {
    errors.push(`attention hidden_dim ${hiddenDim} < ${config.minHiddenDim}`);
  }
  if (transformerLayers < config.minTransformerLayers) {
    errors.push(`attention transformer_layers ${transformerLayers} < ${config.minTransformerLayers}`);
  }
  if (contextSeqLen < config.minContextSeqLen) {
    errors.push(`attention context_seq_len ${contextSeqLen} < ${config.minContextSeqLen}`);
  }
  if (config.requirePromotedSmallProfile) {
    errors.push(...promotedSmallProfile.errors);
  }
  return {
    ok: errors.length === 0,
    errors,
    has_profile: hasProfile,
    d_model: dModel,
    heads,
    head_dim: headDim,
    head_dim_power_of_four: headDimPowerOfFour,
    hidden_dim: hiddenDim,
    transformer_layers: transformerLayers,
    context_seq_len: contextSeqLen,
    promoted_small_profile: dropErrors(promotedSmallProfile),
    token_heads: {
      text_char: { source: "nsrllmm-output-token-head", token_min: 16, token_max: 143 },
      image_plan: { source: "nsrllmm-output-token-head", token_min: 144, token_max: 159 },
      text_chunk: { source: "nsrllmm-output-token-head", token_min: 160, token_max: 255 },
    },
  };
}

function promotedSmallProfileStatus({
  dModel,
  heads,
  headDim,
  headDimPowerOfFour,
  hiddenDim,
  transformerLayers,
  contextSeqLen,
}) {
  const target = {
    d_model: 128,
    heads: 2,
    head_dim: 64,
    hidden_dim_min: 256,
    hidden_dim_max: 512,
    transformer_layers_min: 2,
    transformer_layers_max: 4,
    context_seq_len_min: 384,
    context_seq_len_max: 768,
  };
  const errors = [];
  if (dModel !== target.d_model) {
    errors.push(`promoted small profile d_model ${dModel} != ${target.d_model}`);
  }
  if (heads !== target.heads) {
    errors.push(`promoted small profile heads ${heads} != ${target.heads}`);
  }
  if (headDim !== target.head_dim) {
    errors.push(`promoted small profile head_dim ${headDim} != ${target.head_dim}`);
  }
  if (!headDimPowerOfFour) {
    errors.push(`promoted small profile head_dim ${headDim} is not a power of four`);
  }
  if (hiddenDim < target.hidden_dim_min || hiddenDim > target.hidden_dim_max) {
    errors.push(
      `promoted small profile hidden_dim ${hiddenDim} outside ${target.hidden_dim_min}-${target.hidden_dim_max}`,
    );
  }
  if (
    transformerLayers < target.transformer_layers_min ||
    transformerLayers > target.transformer_layers_max
  ) {
    errors.push(
      `promoted small profile transformer_layers ${transformerLayers} outside ${target.transformer_layers_min}-${target.transformer_layers_max}`,
    );
  }
  if (contextSeqLen < target.context_seq_len_min || contextSeqLen > target.context_seq_len_max) {
    errors.push(
      `promoted small profile context_seq_len ${contextSeqLen} outside ${target.context_seq_len_min}-${target.context_seq_len_max}`,
    );
  }
  return {
    required_shape: target,
    ok: errors.length === 0,
    errors,
  };
}

function isPowerOfFour(value) {
  if (!Number.isInteger(value) || value <= 0) {
    return false;
  }
  let current = value;
  while (current > 1) {
    if (current % 4 !== 0) {
      return false;
    }
    current /= 4;
  }
  return true;
}

function metricSummary(stats) {
  return {
    targets: Number(stats.targets || 0),
    correct: Number(stats.correct || 0),
    invalid_contexts: Number(stats.invalid_contexts || 0),
    accuracy_per_mille: Number(stats.accuracy_per_mille || 0),
    top5_accuracy_per_mille: Number(stats.top5_accuracy_per_mille || 0),
    top10_accuracy_per_mille: Number(stats.top10_accuracy_per_mille || 0),
    mean_target_rank_per_mille: Number(stats.mean_target_rank_per_mille || 0),
    mean_target_margin_q8: Number(stats.mean_target_margin_q8 || 0),
  };
}

function aggregateMetricSummaries(items) {
  let targets = 0;
  let correct = 0;
  let invalidContexts = 0;
  let top5Numerator = 0;
  let top10Numerator = 0;
  let rankNumerator = 0;
  let marginNumerator = 0;
  let top5Targets = 0;
  let top10Targets = 0;
  let rankTargets = 0;
  let marginTargets = 0;
  for (const item of items) {
    const summary = metricSummary(item || {});
    if (summary.targets <= 0) {
      continue;
    }
    targets += summary.targets;
    correct += summary.correct;
    invalidContexts += summary.invalid_contexts;
    top5Numerator += summary.top5_accuracy_per_mille * summary.targets;
    top10Numerator += summary.top10_accuracy_per_mille * summary.targets;
    rankNumerator += summary.mean_target_rank_per_mille * summary.targets;
    marginNumerator += summary.mean_target_margin_q8 * summary.targets;
    top5Targets += summary.targets;
    top10Targets += summary.targets;
    rankTargets += summary.targets;
    marginTargets += summary.targets;
  }
  return {
    targets,
    correct,
    invalid_contexts: invalidContexts,
    accuracy_per_mille: targets > 0 ? Math.floor((correct * 1000) / targets) : 0,
    top5_accuracy_per_mille: top5Targets > 0 ? Math.floor(top5Numerator / top5Targets) : 0,
    top10_accuracy_per_mille: top10Targets > 0 ? Math.floor(top10Numerator / top10Targets) : 0,
    mean_target_rank_per_mille: rankTargets > 0 ? Math.floor(rankNumerator / rankTargets) : 0,
    mean_target_margin_q8: marginTargets > 0 ? Math.floor(marginNumerator / marginTargets) : 0,
  };
}

function checkRetrievalHeadEval(trace, filePath, config) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_v2_retrieval_head_eval.v1", errors);
  const classRetrievalHead = checkRetrievalHeadArtifact(trace, filePath, config, errors);
  const corpusProvenance = checkRetrievalHeadCorpusProvenance(trace, filePath, config, errors);
  if (trace.ok !== true) {
    errors.push("retrieval head eval ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `retrieval head eval: ${error}`));
  }
  requireAllTop1(trace.known_prompts, "known prompts", errors);
  requireMarginFloor(trace.known_prompts, "known prompts", config.minRetrievalMargin, errors);
  requireAllTop1(trace.identity_bindings?.total, "identity bindings", errors);
  requireMarginFloor(trace.identity_bindings?.total, "identity bindings", config.minRetrievalMargin, errors);
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    requireAllTop1(trace.identity_bindings?.by_kind?.[kind], `identity binding ${kind}`, errors);
    requireMarginFloor(trace.identity_bindings?.by_kind?.[kind], `identity binding ${kind}`, config.minRetrievalMargin, errors);
  }
  if (Number(trace.heldout_prompts?.count || 0) > 0) {
    requireAllTop1(trace.heldout_prompts, "held-out prompts", errors);
    requireMarginFloor(trace.heldout_prompts, "held-out prompts", config.minRetrievalMargin, errors);
  }
  const heldoutRows = Number(trace.heldout_prompt_rows || trace.heldout_prompts?.count || 0);
  if (config.requireHeldoutPrompts && heldoutRows <= 0) {
    errors.push("retrieval head held-out prompts are required but no prompt rows were evaluated");
  }
  if (heldoutRows < config.minHeldoutPromptRows) {
    errors.push(`retrieval head held-out prompt rows ${heldoutRows} < ${config.minHeldoutPromptRows}`);
  }
  const heldoutPromptProvenance = checkRetrievalHeadHeldoutPromptProvenance(
    trace,
    filePath,
    config,
    heldoutRows,
    errors,
  );
  requireAllTop1(trace.image_to_text, "image-to-text/source", errors);
  requireMarginFloor(trace.image_to_text, "image-to-text/source", config.minRetrievalMargin, errors);
  for (const task of IMAGE_RETRIEVAL_TASKS) {
    requireAllTop1(trace.image_tasks?.[task], task, errors);
    requireCountFloor(trace.image_tasks?.[task], task, IMAGE_RETRIEVAL_TASK_MIN_COUNTS[task], errors);
    requireMarginFloor(trace.image_tasks?.[task], task, config.minRetrievalMargin, errors);
  }
  requireAllTop1(trace.match?.yes, "match yes", errors);
  requireAllTop1(trace.match?.no, "match no", errors);
  requireAllTop1(trace.match?.no_by_role?.image, "match no image", errors);
  requireAllTop1(trace.match?.no_by_role?.prompt, "match no prompt", errors);
  requireTop1Floor(trace.match?.yes, "match yes", config.minMatchYesTop1, errors);
  requireTop1Floor(trace.match?.no, "match no", config.minMatchNoTop1, errors);
  requireTop1Floor(trace.match?.no_by_role?.image, "match no image", config.minMatchNoImageTop1, errors);
  requireTop1Floor(trace.match?.no_by_role?.prompt, "match no prompt", config.minMatchNoPromptTop1, errors);
  requireMarginFloor(trace.match?.yes, "match yes", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no, "match no", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no_by_role?.image, "match no image", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no_by_role?.prompt, "match no prompt", config.minRetrievalMargin, errors);
  return {
    ok: errors.length === 0,
    errors,
    model: trace.model || "",
    model_hash: trace.model_hash || "",
    examples: trace.examples || "",
    examples_hash: trace.examples_hash || "",
    tokens: trace.tokens || "",
    tokens_hash: trace.tokens_hash || "",
    prompts: trace.prompts || "",
    prompts_hash: trace.prompts_hash || "",
    corpus_provenance: corpusProvenance,
    heldout_prompt_provenance: heldoutPromptProvenance,
    class_retrieval_head: classRetrievalHead,
    feature_count: Number(trace.feature_count || 0),
    known_prompts: retrievalMetricSummary(trace.known_prompts),
    identity_bindings: {
      required_kinds: Array.isArray(trace.identity_bindings?.required_kinds)
        ? trace.identity_bindings.required_kinds
        : REQUIRED_IDENTITY_BINDING_KINDS,
      total: retrievalMetricSummary(trace.identity_bindings?.total),
      by_kind: Object.fromEntries(
        REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [
          kind,
          retrievalMetricSummary(trace.identity_bindings?.by_kind?.[kind]),
        ]),
      ),
    },
    heldout_prompts: retrievalMetricSummary(trace.heldout_prompts),
    heldout_prompt_rows: heldoutRows,
    image_to_text: retrievalMetricSummary(trace.image_to_text),
    image_tasks: Object.fromEntries(
      IMAGE_RETRIEVAL_TASKS.map((task) => [task, retrievalMetricSummary(trace.image_tasks?.[task])]),
    ),
    match: {
      yes: retrievalMetricSummary(trace.match?.yes),
      no: retrievalMetricSummary(trace.match?.no),
      no_by_role: {
        image: retrievalMetricSummary(trace.match?.no_by_role?.image),
        prompt: retrievalMetricSummary(trace.match?.no_by_role?.prompt),
      },
    },
  };
}

function checkRetrievalHeadCorpusProvenance(trace, evalFilePath, config, errors) {
  const expectedExamples = config.examplesPath || "";
  const expectedTokens = expectedRetrievalCorpusTokensPath(config);
  const evalDir = path.dirname(evalFilePath);
  const summary = {
    examples: trace.examples || "",
    expected_examples: expectedExamples,
    examples_match: null,
    examples_hash: trace.examples_hash || "",
    expected_examples_hash: "",
    examples_hash_match: null,
    tokens: trace.tokens || "",
    expected_tokens: expectedTokens,
    tokens_match: null,
    tokens_hash: trace.tokens_hash || "",
    expected_tokens_hash: "",
    tokens_hash_match: null,
  };

  if (expectedExamples) {
    if (!summary.examples) {
      errors.push("retrieval head eval examples path is missing");
    } else {
      summary.examples_match = sameReferencedPath(summary.examples, expectedExamples, evalDir);
      if (summary.examples_match === false) {
        errors.push(
          `retrieval head eval examples ${summary.examples} does not match corpus examples ${expectedExamples}`,
        );
      }
    }
  }
  if (expectedTokens) {
    if (!summary.tokens) {
      errors.push("retrieval head eval tokens path is missing");
    } else {
      summary.tokens_match = sameReferencedPath(summary.tokens, expectedTokens, evalDir);
      if (summary.tokens_match === false) {
        errors.push(`retrieval head eval tokens ${summary.tokens} does not match corpus tokens ${expectedTokens}`);
      }
    }
  }

  const examplesHash = compareRetrievalCorpusHash(
    "examples",
    summary.examples_hash,
    expectedExamples,
    errors,
  );
  summary.expected_examples_hash = examplesHash.expected_hash;
  summary.examples_hash_match = examplesHash.hash_match;
  const tokensHash = compareRetrievalCorpusHash("tokens", summary.tokens_hash, expectedTokens, errors);
  summary.expected_tokens_hash = tokensHash.expected_hash;
  summary.tokens_hash_match = tokensHash.hash_match;

  return summary;
}

function checkRetrievalHeadHeldoutPromptProvenance(trace, evalFilePath, config, heldoutRows, errors) {
  const evalDir = path.dirname(evalFilePath);
  const prompts = trace.prompts || "";
  const promptsHash = trace.prompts_hash || "";
  const required = Boolean(config.requireHeldoutPrompts || heldoutRows > 0);
  const summary = {
    required,
    prompts,
    resolved_prompts: "",
    prompts_present: false,
    prompts_hash: promptsHash,
    expected_prompts_hash: "",
    prompts_hash_match: null,
    heldout_prompt_rows: heldoutRows,
    prompt_rows_total: Number(trace.prompt_rows_total || 0),
    prompt_rows_counted: 0,
    unique_targets: Number(trace.heldout_prompt_unique_targets || 0),
    unique_targets_counted: 0,
    row_count_match: null,
    unique_targets_match: null,
  };

  if (!required) {
    return summary;
  }
  if (!prompts) {
    errors.push("retrieval head held-out prompts path is missing");
    return summary;
  }

  const candidates = referencedPathCandidates(prompts, evalDir);
  const resolvedPrompts = candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0] || "";
  summary.resolved_prompts = resolvedPrompts;
  summary.prompts_present = Boolean(resolvedPrompts && fs.existsSync(resolvedPrompts));
  if (!summary.prompts_present) {
    errors.push(`retrieval head held-out prompts ${prompts} could not be resolved`);
    return summary;
  }

  if (!promptsHash) {
    errors.push("retrieval head eval prompts_hash is missing");
  } else {
    try {
      summary.expected_prompts_hash = fnv64FileHex(resolvedPrompts);
      summary.prompts_hash_match = String(promptsHash) === summary.expected_prompts_hash;
      if (!summary.prompts_hash_match) {
        errors.push(
          `retrieval head eval prompts_hash ${promptsHash} does not match held-out prompts hash ${summary.expected_prompts_hash}`,
        );
      }
    } catch (error) {
      errors.push(`retrieval head eval prompts_hash could not read held-out prompts ${resolvedPrompts}: ${error.message}`);
    }
  }

  try {
    const counted = countRetrievalHeadPromptRows(resolvedPrompts);
    summary.prompt_rows_counted = counted.eligible_rows;
    summary.unique_targets_counted = counted.unique_targets;
    summary.row_count_match = summary.prompt_rows_counted === heldoutRows;
    if (!summary.row_count_match) {
      errors.push(
        `retrieval head held-out prompt rows ${heldoutRows} != eligible prompt file rows ${summary.prompt_rows_counted}`,
      );
    }
    summary.unique_targets_match = summary.unique_targets === summary.unique_targets_counted;
    if (!summary.unique_targets_match) {
      errors.push(
        `retrieval head held-out prompt unique targets ${summary.unique_targets} != eligible prompt file unique targets ${summary.unique_targets_counted}`,
      );
    }
    const expectedSpirits = expectedGroundedCorpusSpirits(config) || 72;
    if (summary.unique_targets_counted < expectedSpirits) {
      errors.push(
        `retrieval head eligible held-out prompt unique targets ${summary.unique_targets_counted} < ${expectedSpirits}`,
      );
    }
  } catch (error) {
    errors.push(`retrieval head held-out prompt rows could not read ${resolvedPrompts}: ${error.message}`);
  }

  return summary;
}

function countRetrievalHeadPromptRows(filePath) {
  const rows = readJsonl(filePath).filter((row) => {
    const spiritId = normalizedCorpusSpiritId(row.spirit_id);
    const text = String(row.text || row.prompt || "");
    return spiritId !== null && spiritId >= 1 && text;
  });
  const eligible = rows.filter(isRetrievalHeadHeldoutPromptRow);
  return {
    total_rows: rows.length,
    eligible_rows: eligible.length,
    unique_targets: new Set(eligible.map((row) => normalizedCorpusSpiritId(row.spirit_id))).size,
  };
}

function isRetrievalHeadHeldoutPromptRow(row) {
  const tier = String(row.tier || "").toLowerCase();
  const source = String(row.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function expectedRetrievalCorpusTokensPath(config) {
  if (config.tokensPath) {
    return config.tokensPath;
  }
  if (!config.manifestPath) {
    return "";
  }
  try {
    return resolveCorpusTokensPath(config, readJson(config.manifestPath));
  } catch (_error) {
    return "";
  }
}

function sameReferencedPath(ref, expected, refBaseDir) {
  if (!ref || !expected) {
    return null;
  }
  const expectedPath = normalizeReferencedPath(expected);
  return referencedPathCandidates(ref, refBaseDir).some((candidate) => candidate === expectedPath);
}

function referencedPathCandidates(ref, baseDir) {
  const candidates = path.isAbsolute(ref)
    ? [path.resolve(ref)]
    : [
        path.resolve(ref),
        path.resolve(baseDir, ref),
      ];
  return [...new Set(candidates.map(normalizeReferencedPath))];
}

function normalizeReferencedPath(filePath) {
  const resolved = path.resolve(filePath);
  try {
    return fs.realpathSync.native(resolved);
  } catch (_error) {
    return resolved;
  }
}

function compareRetrievalCorpusHash(label, traceHash, expectedPath, errors) {
  const summary = {
    expected_hash: "",
    hash_match: null,
  };
  if (!traceHash || !expectedPath) {
    return summary;
  }
  try {
    summary.expected_hash = fnv64FileHex(path.resolve(expectedPath));
  } catch (error) {
    errors.push(`retrieval head eval ${label}_hash could not read corpus ${label} ${expectedPath}: ${error.message}`);
    return summary;
  }
  summary.hash_match = String(traceHash) === summary.expected_hash;
  if (!summary.hash_match) {
    errors.push(
      `retrieval head eval ${label}_hash ${traceHash} does not match corpus ${label} hash ${summary.expected_hash}`,
    );
  }
  return summary;
}

function checkRetrievalHeadArtifact(trace, evalFilePath, config, errors) {
  const resolved = resolveRetrievalHeadPath(trace, evalFilePath, config);
  const summary = {
    source: resolved.source,
    path: resolved.path,
    present: false,
    schema: "",
    model_hash: "",
    hash_matches_eval: false,
    hash_verified: false,
    feature_count: 0,
    labels: 0,
    text_head: false,
    image_head: false,
    text_nonzero_weights: 0,
    image_nonzero_weights: 0,
  };
  if (!resolved.source) {
    errors.push("retrieval head artifact path is missing");
    return summary;
  }
  if (!resolved.exists) {
    errors.push(`retrieval head artifact ${resolved.source} was not found`);
    return summary;
  }
  const model = tryReadJson(resolved.path, errors, "retrieval head artifact");
  if (!model) {
    return summary;
  }
  summary.present = true;
  summary.schema = String(model.schema || "");
  summary.model_hash = String(model.model_hash || "");
  summary.feature_count = Number(model.feature_count || 0);
  summary.labels = Array.isArray(model.labels) ? model.labels.length : 0;
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    errors.push(`retrieval head artifact schema ${JSON.stringify(model.schema)} != nsrl.solomon_v2_retrieval_head.v1`);
  }
  if (summary.labels !== 72) {
    errors.push(`retrieval head artifact labels ${summary.labels} != 72`);
  }
  if (Number(trace.feature_count || 0) > 0 && summary.feature_count !== Number(trace.feature_count || 0)) {
    errors.push(
      `retrieval head artifact feature_count ${summary.feature_count} != eval feature_count ${Number(trace.feature_count || 0)}`,
    );
  }
  const textHead = retrievalHeadComponentSummary(model.text_head, summary.labels);
  const imageHead = retrievalHeadComponentSummary(model.image_head, summary.labels);
  summary.text_head = textHead.ok;
  summary.image_head = imageHead.ok;
  summary.text_nonzero_weights = textHead.nonzero_weights;
  summary.image_nonzero_weights = imageHead.nonzero_weights;
  if (!textHead.ok) {
    errors.push(
      `retrieval head artifact text_head invalid: biases=${textHead.biases} weights=${textHead.weights} malformed_rows=${textHead.malformed_rows}`,
    );
  }
  if (!imageHead.ok) {
    errors.push(
      `retrieval head artifact image_head invalid: biases=${imageHead.biases} weights=${imageHead.weights} malformed_rows=${imageHead.malformed_rows}`,
    );
  }
  const recomputedHash = recomputeRetrievalHeadHash(model);
  summary.hash_verified = Boolean(summary.model_hash) && summary.model_hash === recomputedHash;
  if (!summary.hash_verified) {
    errors.push(`retrieval head artifact model_hash ${summary.model_hash || ""} != recomputed ${recomputedHash}`);
  }
  summary.hash_matches_eval =
    Boolean(summary.model_hash) &&
    Boolean(trace.model_hash) &&
    summary.model_hash === String(trace.model_hash);
  if (!summary.hash_matches_eval) {
    errors.push(`retrieval head artifact model_hash ${summary.model_hash || ""} != eval model_hash ${trace.model_hash || ""}`);
  }
  return summary;
}

function resolveRetrievalHeadPath(trace, evalFilePath, config) {
  const source = config.retrievalHeadPath || trace.model || "";
  if (!source) {
    return { source: "", path: "", exists: false };
  }
  const candidates = path.isAbsolute(source)
    ? [source]
    : [
        path.resolve(source),
        path.resolve(path.dirname(evalFilePath), source),
      ];
  const seen = new Set();
  for (const candidate of candidates) {
    if (seen.has(candidate)) {
      continue;
    }
    seen.add(candidate);
    if (fs.existsSync(candidate)) {
      return { source, path: candidate, exists: true };
    }
  }
  return { source, path: candidates[0], exists: false };
}

function retrievalHeadComponentSummary(head, labelCount) {
  const biases = Array.isArray(head?.biases) ? head.biases.length : 0;
  const weights = Array.isArray(head?.weights) ? head.weights.length : 0;
  let malformedRows = 0;
  let nonzeroWeights = 0;
  if (Array.isArray(head?.weights)) {
    for (const row of head.weights) {
      if (!Array.isArray(row)) {
        malformedRows += 1;
        continue;
      }
      nonzeroWeights += row.length;
      for (const entry of row) {
        if (!Array.isArray(entry) || entry.length !== 2 || !Number.isInteger(Number(entry[0]))) {
          malformedRows += 1;
          break;
        }
      }
    }
  }
  return {
    ok: labelCount > 0 && biases === labelCount && weights === labelCount && malformedRows === 0,
    biases,
    weights,
    malformed_rows: malformedRows,
    nonzero_weights: nonzeroWeights,
  };
}

function recomputeRetrievalHeadHash(model) {
  const copy = { ...model };
  delete copy.model_hash;
  return fnv64Hex(JSON.stringify(copy));
}

function fnv64Hex(value) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function requireAllTop1(metric, label, errors) {
  const count = Number(metric?.count || 0);
  const top1 = Number(metric?.top1 || 0);
  if (count <= 0) {
    errors.push(`retrieval head ${label} has no rows`);
  } else if (top1 !== count) {
    errors.push(`retrieval head ${label} top1 ${top1} != count ${count}`);
  }
}

function requireTop1Floor(metric, label, floor, errors) {
  const minimum = Number(floor || 0);
  if (minimum <= 0) {
    return;
  }
  const top1 = Number(metric?.top1 || 0);
  if (top1 < minimum) {
    errors.push(`retrieval head ${label} top1 ${top1} < ${minimum}`);
  }
}

function requireCountFloor(metric, label, floor, errors) {
  const minimum = Number(floor || 0);
  if (minimum <= 0) {
    return;
  }
  const count = Number(metric?.count || 0);
  if (count < minimum) {
    errors.push(`retrieval head ${label} count ${count} < ${minimum}`);
  }
}

function requireMarginFloor(metric, label, floor, errors) {
  const minimum = Number(floor || 0);
  if (minimum <= 0) {
    return;
  }
  const count = Number(metric?.count || 0);
  if (count <= 0) {
    return;
  }
  const margin = Number(metric?.min_margin ?? Number.MIN_SAFE_INTEGER);
  if (margin < minimum) {
    errors.push(`retrieval head ${label} min_margin ${margin} < ${minimum}`);
  }
}

function retrievalMetricSummary(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    top1_per_mille: Number(metric?.top1_per_mille || 0),
    top5_per_mille: Number(metric?.top5_per_mille || 0),
    min_margin: finiteNumberOrNull(metric?.min_margin),
    mean_margin: finiteNumberOrNull(metric?.mean_margin),
  };
}

function checkSampleBinding(trace, filePath, retrievalReport) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_attention_sample_binding_check.v1", errors);
  const retrievalHeadProvenance = checkRetrievalEvidenceHeadProvenance(
    "sample binding",
    trace.retrieval_head_model_hash,
    retrievalReport,
    errors,
  );
  if (trace.ok !== true) {
    errors.push("sample binding ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `sample binding: ${error}`));
  }
  if (Number(trace.samples || 0) <= 0) {
    errors.push("sample binding has no samples");
  }
  if (trace.text_image_agreement !== true) {
    errors.push("sample binding text/image agreement is not true");
  }
  if (trace.signature_retrieval_agreement !== true) {
    errors.push("sample binding signature/retrieval agreement is not true");
  }
  if (trace.image_to_text_identification !== true) {
    errors.push("sample binding image-to-text identification is not true");
  }
  if (Number(trace.min_signature_margin || 0) <= 0) {
    errors.push(`sample binding min_signature_margin ${trace.min_signature_margin || 0} <= 0`);
  }
  if (Number(trace.min_retrieval_image_margin || 0) <= 0) {
    errors.push(`sample binding min_retrieval_image_margin ${trace.min_retrieval_image_margin || 0} <= 0`);
  }
  if (Number(trace.min_image_to_text_margin || 0) <= 0) {
    errors.push(`sample binding min_image_to_text_margin ${trace.min_image_to_text_margin || 0} <= 0`);
  }
  if (Number(trace.min_retrieval_text_margin || 0) <= 0) {
    errors.push(`sample binding min_retrieval_text_margin ${trace.min_retrieval_text_margin || 0} <= 0`);
  }
  if (trace.generated_text_identification !== true) {
    errors.push("sample binding generated text identification is not true");
  }
  if (trace.generated_text_image_agreement !== true) {
    errors.push("sample binding generated text/image agreement is not true");
  }
  if (Number(trace.min_generated_text_margin || 0) <= 0) {
    errors.push(`sample binding min_generated_text_margin ${trace.min_generated_text_margin || 0} <= 0`);
  }
  const results = Array.isArray(trace.results) ? trace.results : [];
  return {
    ok: errors.length === 0,
    errors,
    file_path: filePath,
    retrieval_head: trace.retrieval_head || null,
    retrieval_head_model_hash: trace.retrieval_head_model_hash || "",
    retrieval_head_provenance: retrievalHeadProvenance,
    samples: Number(trace.samples || 0),
    min_signature_margin: Number(trace.min_signature_margin || 0),
    min_retrieval_image_margin: Number(trace.min_retrieval_image_margin || 0),
    image_to_text_identification: trace.image_to_text_identification === true,
    min_image_to_text_margin: Number(trace.min_image_to_text_margin || 0),
    min_retrieval_text_margin: Number(trace.min_retrieval_text_margin || 0),
    generated_text_identification: trace.generated_text_identification === true,
    min_generated_text_margin: Number(trace.min_generated_text_margin || 0),
    text_image_agreement: trace.text_image_agreement === true,
    generated_text_image_agreement: trace.generated_text_image_agreement === true,
    signature_retrieval_agreement: trace.signature_retrieval_agreement === true,
    results: results.map((result) => ({
      sample_dir: result.sample_dir || "",
      prompt: result.prompt || "",
      image_ink16_u8: result.image_ink16_u8 || "",
      expected_spirit_id: Number(result.expected_spirit_id || 0),
      expected_primary_name: result.expected_primary_name || "",
      generated_text: result.generated_text || "",
      image_to_text_identity: result.image_to_text_identity || null,
      generated_text_identity: result.generated_text_identity || null,
      generated_text_image_agree: result.generated_text_image_agree === true,
      confidence: result.confidence || null,
    })),
  };
}

function checkRetrievalEvidenceHeadProvenance(label, actualModelHash, retrievalReport, errors) {
  const expectedModelHash = expectedRetrievalHeadModelHash(retrievalReport);
  const summary = {
    expected_model_hash: expectedModelHash,
    model_hash: actualModelHash || "",
    hash_match: null,
  };
  if (!expectedModelHash || !summary.model_hash) {
    return summary;
  }
  summary.hash_match = summary.model_hash === expectedModelHash;
  if (!summary.hash_match) {
    errors.push(
      `${label} retrieval head model_hash ${summary.model_hash} != retrieval head eval model_hash ${expectedModelHash}`,
    );
  }
  return summary;
}

function expectedRetrievalHeadModelHash(retrievalReport) {
  return String(retrievalReport?.model_hash || retrievalReport?.class_retrieval_head?.model_hash || "");
}

function checkGenerationIntegrity(trace, filePath) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_generation_integrity_check.v1", errors);
  if (trace.ok !== true) {
    errors.push("generation integrity ok is not true");
  }
  if (Array.isArray(trace.violations) && trace.violations.length > 0) {
    errors.push(...trace.violations.map((violation) => `generation integrity: ${violation.reason || JSON.stringify(violation)}`));
  }
  if (Number(trace.trace_count || 0) <= 0) {
    errors.push("generation integrity inspected no traces");
  }
  return {
    ok: errors.length === 0,
    errors,
    trace_count: Number(trace.trace_count || 0),
    violations: Array.isArray(trace.violations) ? trace.violations.length : 0,
  };
}

function checkIdentityInference(trace, filePath, retrievalReport, config) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_v2_identity_inference.v1", errors);
  const retrievalHeadProvenance = checkRetrievalEvidenceHeadProvenance(
    "identity inference",
    trace.model_hash,
    retrievalReport,
    errors,
  );
  const textIndexProvenance = checkSourceTextIndexProvenance(
    "identity inference",
    trace.text_index,
    trace.text_index_hash,
    filePath,
    config,
    errors,
  );
  if (trace.ok !== true) {
    errors.push("identity inference ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `identity inference: ${error}`));
  }
  if (Number(trace.query_count || 0) <= 0) {
    errors.push("identity inference has no queries");
  }
  const textQueryCount = Array.isArray(trace.text_queries) ? trace.text_queries.length : 0;
  const imageQueryCount = Array.isArray(trace.image_queries) ? trace.image_queries.length : 0;
  const sampleQueryCount = Array.isArray(trace.sample_queries) ? trace.sample_queries.length : 0;
  if (textQueryCount <= 0) {
    errors.push("identity inference has no text queries");
  }
  if (imageQueryCount <= 0) {
    errors.push("identity inference has no image queries");
  }
  if (sampleQueryCount <= 0) {
    errors.push("identity inference has no sample queries");
  }
  if (textQueryCount > 0 && trace.source_summary?.text_queries_have_source_text !== true) {
    errors.push("identity inference text queries are missing source text evidence");
  }
  if (imageQueryCount > 0 && trace.source_summary?.image_queries_have_source_text !== true) {
    errors.push("identity inference image queries are missing source text evidence");
  }
  if (sampleQueryCount > 0 && trace.source_summary?.sample_queries_have_source_text !== true) {
    errors.push("identity inference sample queries are missing source text evidence");
  }
  const sampleCount = Number(trace.sample_summary?.samples || 0);
  if (sampleCount > 0) {
    if (trace.sample_summary?.text_image_agreement !== true) {
      errors.push("identity inference text/image agreement is not true");
    }
    if (trace.sample_summary?.signature_retrieval_agreement !== true) {
      errors.push("identity inference signature/retrieval agreement is not true");
    }
    if (trace.sample_summary?.expected_image_agreement !== true) {
      errors.push("identity inference expected/image agreement is not true");
    }
    if (Number(trace.sample_summary?.min_image_retrieval_margin || 0) <= 0) {
      errors.push(
        `identity inference min_image_retrieval_margin ${trace.sample_summary?.min_image_retrieval_margin || 0} <= 0`,
      );
    }
    if (Number(trace.sample_summary?.min_signature_margin || 0) <= 0) {
      errors.push(`identity inference min_signature_margin ${trace.sample_summary?.min_signature_margin || 0} <= 0`);
    }
    if (trace.sample_summary?.source_text_evidence !== true) {
      errors.push("identity inference sample source_text_evidence is not true");
    }
    if (trace.sample_summary?.generated_text_source_evidence !== true) {
      errors.push("identity inference generated_text_source_evidence is not true");
    }
    if (trace.sample_summary?.generated_text_image_agreement !== true) {
      errors.push("identity inference generated_text_image_agreement is not true");
    }
    if (trace.sample_summary?.expected_generated_text_agreement !== true) {
      errors.push("identity inference expected_generated_text_agreement is not true");
    }
    if (Number(trace.sample_summary?.min_source_text_chars || 0) <= 0) {
      errors.push(`identity inference min_source_text_chars ${trace.sample_summary?.min_source_text_chars || 0} <= 0`);
    }
    if (Number(trace.sample_summary?.min_prompt_text_margin || 0) <= 0) {
      errors.push(
        `identity inference min_prompt_text_margin ${trace.sample_summary?.min_prompt_text_margin || 0} <= 0`,
      );
    }
    if (Number(trace.sample_summary?.min_generated_text_margin || 0) <= 0) {
      errors.push(
        `identity inference min_generated_text_margin ${trace.sample_summary?.min_generated_text_margin || 0} <= 0`,
      );
    }
  }
  return {
    ok: errors.length === 0,
    errors,
    present: true,
    text_index: trace.text_index || "",
    text_index_hash: trace.text_index_hash || "",
    text_index_provenance: textIndexProvenance,
    retrieval_head: trace.retrieval_head || null,
    model_hash: trace.model_hash || "",
    retrieval_head_provenance: retrievalHeadProvenance,
    query_count: Number(trace.query_count || 0),
    text_queries: textQueryCount,
    image_queries: imageQueryCount,
    sample_queries: sampleQueryCount,
    source_summary: trace.source_summary || {},
    sample_summary: trace.sample_summary || {},
  };
}

function absentIdentityInference() {
  return {
    ok: true,
    errors: [],
    present: false,
    query_count: 0,
    text_index: "",
    text_index_hash: "",
    text_index_provenance: absentSourceTextIndexProvenance(),
    retrieval_head: null,
    model_hash: "",
    retrieval_head_provenance: {
      expected_model_hash: "",
      model_hash: "",
      hash_match: null,
    },
    text_queries: 0,
    image_queries: 0,
    sample_queries: 0,
    source_summary: {},
    sample_summary: {},
  };
}

function checkSourceTextIndexProvenance(label, textIndex, textIndexHash, filePath, config, errors) {
  const expectedTextIndex = expectedSourceTextIndexPath(config);
  const summary = {
    text_index: textIndex || "",
    expected_text_index: expectedTextIndex,
    text_index_match: null,
    text_index_hash: textIndexHash || "",
    expected_text_index_hash: "",
    text_index_hash_match: null,
  };
  if (!expectedTextIndex) {
    return summary;
  }
  if (!summary.text_index) {
    errors.push(`${label} text_index path is missing`);
  } else {
    summary.text_index_match = sameReferencedPath(summary.text_index, expectedTextIndex, path.dirname(filePath));
    if (summary.text_index_match === false) {
      errors.push(`${label} text_index ${summary.text_index} does not match manifest source_text_index ${expectedTextIndex}`);
    }
  }
  if (!summary.text_index_hash) {
    errors.push(`${label} text_index_hash is missing`);
  } else {
    try {
      summary.expected_text_index_hash = fnv64FileHex(expectedTextIndex);
      summary.text_index_hash_match = summary.text_index_hash === summary.expected_text_index_hash;
      if (!summary.text_index_hash_match) {
        errors.push(
          `${label} text_index_hash ${summary.text_index_hash} does not match manifest source_text_index hash ${summary.expected_text_index_hash}`,
        );
      }
    } catch (error) {
      errors.push(`${label} text_index_hash could not read manifest source_text_index ${expectedTextIndex}: ${error.message}`);
    }
  }
  return summary;
}

function expectedSourceTextIndexPath(config) {
  if (!config.manifestPath) {
    return "";
  }
  try {
    const manifest = readJson(config.manifestPath);
    const sourceTextIndex = manifest.source_text_index || "";
    if (!sourceTextIndex) {
      return "";
    }
    const candidates = path.isAbsolute(sourceTextIndex)
      ? [sourceTextIndex]
      : [
          path.resolve(sourceTextIndex),
          path.resolve(path.dirname(config.manifestPath), sourceTextIndex),
        ];
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
    return candidates[0] || sourceTextIndex;
  } catch (_error) {
    return "";
  }
}

function absentSourceTextIndexProvenance() {
  return {
    text_index: "",
    expected_text_index: "",
    text_index_match: null,
    text_index_hash: "",
    expected_text_index_hash: "",
    text_index_hash_match: null,
  };
}

function checkCurriculumStages(trace, filePath, config) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_v2_curriculum_stage_check.v1", errors);
  const sourceCorpusProvenance = checkCurriculumSourceCorpusProvenance(trace, filePath, config, errors);
  if (trace.ok !== true) {
    errors.push("curriculum stages ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `curriculum stages: ${error}`));
  }
  if (Number(trace.stage_count || 0) <= 0) {
    errors.push("curriculum stages checked no stages");
  }
  const stages = Array.isArray(trace.stages) ? trace.stages : [];
  const requiredStageNames = Array.isArray(trace.required_stage_names)
    ? trace.required_stage_names.map(canonicalCurriculumStageName)
    : [];
  if (config.requireCurriculumStageNames.length > 0) {
    if (requiredStageNames.length !== config.requireCurriculumStageNames.length) {
      errors.push(
        `curriculum required stage count ${requiredStageNames.length} != ${config.requireCurriculumStageNames.length}`,
      );
    }
    for (const [index, expected] of config.requireCurriculumStageNames.entries()) {
      const actual = requiredStageNames[index] || "";
      if (actual !== expected) {
        errors.push(`curriculum required stage ${index} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
      }
    }
    if (stages.length !== config.requireCurriculumStageNames.length) {
      errors.push(`curriculum stage rows ${stages.length} != required stage count ${config.requireCurriculumStageNames.length}`);
    }
  }
  for (const [index, stage] of stages.entries()) {
    const expectedStageName = config.requireCurriculumStageNames[index] || "";
    if (expectedStageName) {
      const actualStageName = canonicalCurriculumStageName(stage.expected_stage_name || stage.stage_name || "");
      if (actualStageName !== expectedStageName) {
        errors.push(`curriculum stage ${index} name ${JSON.stringify(actualStageName)} != ${JSON.stringify(expectedStageName)}`);
      }
    }
    if (stage.ok !== true) {
      errors.push(`curriculum stage ${index} ok is not true`);
    }
    if (Number(stage.examples || 0) <= 0) {
      errors.push(`curriculum stage ${index} examples ${stage.examples || 0} <= 0`);
    }
    if (Number(stage.train?.accepted_batches || 0) <= 0) {
      errors.push(`curriculum stage ${index} accepted_batches ${stage.train?.accepted_batches || 0} <= 0`);
    }
    if (Number(stage.train?.updates || 0) <= 0) {
      errors.push(`curriculum stage ${index} updates ${stage.train?.updates || 0} <= 0`);
    }
    if (trace.require_loss_non_increasing !== false && Number(stage.train?.probability_error_delta_i64 || 0) > 0) {
      errors.push(
        `curriculum stage ${index} probability_error_delta_i64 ${stage.train?.probability_error_delta_i64 || 0} > 0`,
      );
    }
    const trainArchitecture = curriculumTrainArchitectureProfile(stage.train, config);
    errors.push(...trainArchitecture.errors.map((error) => `curriculum stage ${index} train ${error}`));
    errors.push(...checkCurriculumStageIdentityBindings(stage, index));
    errors.push(...checkCurriculumStageEvidence(stage, index));
    errors.push(...checkCurriculumStageTaskMarkerIntegrity(stage, index));
    errors.push(...checkCurriculumStageTaskModalityIntegrity(stage, index));
    errors.push(...checkCurriculumStageImageChannelMarkerIntegrity(stage, index, config));
  }
  const identityBinding = curriculumIdentityBindingSummary(stages);
  return {
    ok: errors.length === 0,
    errors,
    present: true,
    stage_count: Number(trace.stage_count || 0),
    required_stage_names: requiredStageNames,
    required_stage_names_floor: config.requireCurriculumStageNames,
    require_loss_non_increasing: trace.require_loss_non_increasing !== false,
    source_corpus_provenance: sourceCorpusProvenance,
    identity_binding: identityBinding,
    stages: stages.map((stage) => ({
      index: Number(stage.index || 0),
      stage_name: stage.stage_name || "",
      expected_stage_name: stage.expected_stage_name || "",
      stage_dir: stage.stage_dir || "",
      filter: stage.filter || {},
      examples: Number(stage.examples || 0),
      token_count: Number(stage.token_count || 0),
      source_dir: stage.source_dir || "",
      source_manifest_schema: stage.source_manifest_schema || "",
      source_examples: stage.source_examples || "",
      source_examples_hash: stage.source_examples_hash || "",
      source_tokens: stage.source_tokens || "",
      source_tokens_hash: stage.source_tokens_hash || "",
      identity_bindings: identityBindingStageJson(stage.identity_bindings),
      source_identity_bindings: identityBindingStageJson(stage.source_identity_bindings),
      stage_evidence: curriculumStageEvidenceJson(stage.stage_evidence),
      task_marker_integrity: curriculumTaskMarkerIntegrityJson(stage.task_marker_integrity),
      task_modality_integrity: curriculumTaskModalityIntegrityJson(stage.task_modality_integrity),
      image_channel_marker_integrity: curriculumImageChannelMarkerIntegrityJson(stage.image_channel_marker_integrity),
      train: {
        attention_kind: stage.train?.attention_kind || "",
        text_token_profile: stage.train?.text_token_profile || "",
        architecture: dropErrors(curriculumTrainArchitectureProfile(stage.train, config)),
        batch_mode: stage.train?.batch_mode || "",
        map_reduce_workers: Number(stage.train?.map_reduce_workers || 0),
        windows: Number(stage.train?.windows || 0),
        examined_windows: Number(stage.train?.examined_windows || 0),
        updates: Number(stage.train?.updates || 0),
        accepted_batches: Number(stage.train?.accepted_batches || 0),
        rejected_batches: Number(stage.train?.rejected_batches || 0),
        probability_error_delta_i64: Number(stage.train?.probability_error_delta_i64 || 0),
      },
    })),
  };
}

function curriculumTrainArchitectureProfile(train, config) {
  const trace = {
    d_model: train?.d_model,
    heads: train?.heads,
    hidden_dim: train?.hidden_dim,
    transformer_layers: train?.transformer_layers,
    context_seq_len: train?.context_seq_len || train?.seq_len,
  };
  const profile = architectureProfile(trace, config);
  delete profile.token_heads;
  return {
    ...profile,
    seq_len: Number(train?.seq_len || 0),
  };
}

function uniqueStrings(values) {
  return [...new Set(values.map((value) => String(value || "")).filter(Boolean))];
}

function checkCurriculumSourceCorpusProvenance(trace, filePath, config, errors) {
  const expectedExamples = config.examplesPath || "";
  const expectedTokens = expectedRetrievalCorpusTokensPath(config);
  const stages = Array.isArray(trace.stages) ? trace.stages : [];
  const traceProvenance = trace.source_corpus_provenance || {};
  const sourceExamples = uniqueStrings([
    traceProvenance.source_examples,
    ...stages.map((stage) => stage.source_examples),
  ]);
  const sourceExamplesHashes = uniqueStrings([
    traceProvenance.source_examples_hash,
    ...stages.map((stage) => stage.source_examples_hash),
  ]);
  const sourceTokens = uniqueStrings([
    traceProvenance.source_tokens,
    ...stages.map((stage) => stage.source_tokens),
  ]);
  const sourceTokensHashes = uniqueStrings([
    traceProvenance.source_tokens_hash,
    ...stages.map((stage) => stage.source_tokens_hash),
  ]);
  const summary = {
    source_examples: sourceExamples[0] || "",
    expected_examples: expectedExamples,
    examples_match: null,
    source_examples_hash: sourceExamplesHashes[0] || "",
    expected_examples_hash: "",
    examples_hash_match: null,
    source_examples_consistent:
      traceProvenance.source_examples_consistent !== false &&
      sourceExamples.length <= 1 &&
      sourceExamplesHashes.length <= 1,
    source_tokens: sourceTokens[0] || "",
    expected_tokens: expectedTokens,
    tokens_match: null,
    source_tokens_hash: sourceTokensHashes[0] || "",
    expected_tokens_hash: "",
    tokens_hash_match: null,
    source_tokens_consistent:
      traceProvenance.source_tokens_consistent !== false &&
      sourceTokens.length <= 1 &&
      sourceTokensHashes.length <= 1,
  };
  if (!summary.source_examples_consistent) {
    errors.push("curriculum source examples provenance is not consistent across stages");
  }
  if (!summary.source_tokens_consistent) {
    errors.push("curriculum source tokens provenance is not consistent across stages");
  }
  if (expectedExamples) {
    if (!summary.source_examples) {
      errors.push("curriculum source examples path is missing");
    } else {
      summary.examples_match = sameReferencedPath(summary.source_examples, expectedExamples, path.dirname(filePath));
      if (summary.examples_match === false) {
        errors.push(`curriculum source examples ${summary.source_examples} does not match corpus examples ${expectedExamples}`);
      }
    }
    if (!summary.source_examples_hash) {
      errors.push("curriculum source examples_hash is missing");
    } else {
      try {
        summary.expected_examples_hash = fnv64FileHex(path.resolve(expectedExamples));
        summary.examples_hash_match = summary.source_examples_hash === summary.expected_examples_hash;
        if (!summary.examples_hash_match) {
          errors.push(
            `curriculum source examples_hash ${summary.source_examples_hash} does not match corpus examples hash ${summary.expected_examples_hash}`,
          );
        }
      } catch (error) {
        errors.push(`curriculum source examples_hash could not read corpus examples ${expectedExamples}: ${error.message}`);
      }
    }
  }
  if (expectedTokens) {
    if (!summary.source_tokens) {
      errors.push("curriculum source tokens path is missing");
    } else {
      summary.tokens_match = sameReferencedPath(summary.source_tokens, expectedTokens, path.dirname(filePath));
      if (summary.tokens_match === false) {
        errors.push(`curriculum source tokens ${summary.source_tokens} does not match corpus tokens ${expectedTokens}`);
      }
    }
    if (!summary.source_tokens_hash) {
      errors.push("curriculum source tokens_hash is missing");
    } else {
      try {
        summary.expected_tokens_hash = fnv64FileHex(path.resolve(expectedTokens));
        summary.tokens_hash_match = summary.source_tokens_hash === summary.expected_tokens_hash;
        if (!summary.tokens_hash_match) {
          errors.push(
            `curriculum source tokens_hash ${summary.source_tokens_hash} does not match corpus tokens hash ${summary.expected_tokens_hash}`,
          );
        }
      } catch (error) {
        errors.push(`curriculum source tokens_hash could not read corpus tokens ${expectedTokens}: ${error.message}`);
      }
    }
  }
  return summary;
}

function checkCurriculumStageTaskMarkerIntegrity(stage, index) {
  const errors = [];
  const integrity = stage.task_marker_integrity;
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    errors.push(`curriculum stage ${index} is missing task_marker_integrity`);
    return errors;
  }
  if (integrity.ok !== true) {
    errors.push(`curriculum stage ${index} task_marker_integrity ok is not true`);
  }
  if (Number(integrity.checked_records || 0) <= 0) {
    errors.push(`curriculum stage ${index} task_marker_integrity checked no records`);
  }
  for (const field of ["hash_mismatches", "marker_mismatches", "out_of_bounds", "missing_offsets"]) {
    if (Number(integrity[field] || 0) > 0) {
      errors.push(`curriculum stage ${index} task_marker_integrity ${field} ${Number(integrity[field] || 0)} > 0`);
    }
  }
  return errors;
}

function curriculumTaskMarkerIntegrityJson(integrity) {
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    return null;
  }
  return {
    ok: integrity.ok === true,
    examples: integrity.examples || "",
    tokens: integrity.tokens || "",
    checked_records: Number(integrity.checked_records || 0),
    hash_mismatches: Number(integrity.hash_mismatches || 0),
    marker_mismatches: Number(integrity.marker_mismatches || 0),
    out_of_bounds: Number(integrity.out_of_bounds || 0),
    missing_offsets: Number(integrity.missing_offsets || 0),
    by_task: integrity.by_task || {},
  };
}

function checkCurriculumStageTaskModalityIntegrity(stage, index) {
  const errors = [];
  const integrity = stage.task_modality_integrity;
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    errors.push(`curriculum stage ${index} is missing task_modality_integrity`);
    return errors;
  }
  if (integrity.ok !== true) {
    errors.push(`curriculum stage ${index} task_modality_integrity ok is not true`);
  }
  if (Number(integrity.checked_records || 0) <= 0) {
    errors.push(`curriculum stage ${index} task_modality_integrity checked no records`);
  }
  for (const field of ["modality_mismatches", "out_of_bounds", "missing_offsets"]) {
    if (Number(integrity[field] || 0) > 0) {
      errors.push(`curriculum stage ${index} task_modality_integrity ${field} ${Number(integrity[field] || 0)} > 0`);
    }
  }
  return errors;
}

function curriculumTaskModalityIntegrityJson(integrity) {
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    return null;
  }
  return {
    ok: integrity.ok === true,
    examples: integrity.examples || "",
    tokens: integrity.tokens || "",
    checked_records: Number(integrity.checked_records || 0),
    missing_offsets: Number(integrity.missing_offsets || 0),
    out_of_bounds: Number(integrity.out_of_bounds || 0),
    modality_mismatches: Number(integrity.modality_mismatches || 0),
    by_task: integrity.by_task || {},
  };
}

function checkCurriculumStageImageChannelMarkerIntegrity(stage, index, config) {
  const errors = [];
  if (config.requireImageTokenChannels.length === 0) {
    return errors;
  }
  const integrity = stage.image_channel_marker_integrity;
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    errors.push(`curriculum stage ${index} is missing image_channel_marker_integrity`);
    return errors;
  }
  if (integrity.ok !== true) {
    errors.push(`curriculum stage ${index} image_channel_marker_integrity ok is not true`);
  }
  if (Number(integrity.checked_records || 0) <= 0) {
    errors.push(`curriculum stage ${index} image_channel_marker_integrity checked no records`);
  }
  const stageChannels = Array.isArray(integrity.required_channels)
    ? integrity.required_channels.map((channel) => String(channel))
    : [];
  for (const channel of config.requireImageTokenChannels) {
    if (!stageChannels.includes(channel)) {
      errors.push(`curriculum stage ${index} image_channel_marker_integrity missing required channel ${channel}`);
    }
  }
  for (const field of [
    "missing_offsets",
    "out_of_bounds",
    "missing_image_markers",
    "missing_channel_markers",
    "short_channel_payloads",
    "bad_channel_payloads",
    "channel_order_mismatches",
  ]) {
    if (Number(integrity[field] || 0) > 0) {
      errors.push(`curriculum stage ${index} image_channel_marker_integrity ${field} ${Number(integrity[field] || 0)} > 0`);
    }
  }
  return errors;
}

function curriculumImageChannelMarkerIntegrityJson(integrity) {
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    return null;
  }
  return {
    ok: integrity.ok === true,
    examples: integrity.examples || "",
    tokens: integrity.tokens || "",
    required_channels: Array.isArray(integrity.required_channels)
      ? integrity.required_channels.map((channel) => String(channel))
      : [],
    checked_records: Number(integrity.checked_records || 0),
    missing_offsets: Number(integrity.missing_offsets || 0),
    out_of_bounds: Number(integrity.out_of_bounds || 0),
    missing_image_markers: Number(integrity.missing_image_markers || 0),
    missing_channel_markers: Number(integrity.missing_channel_markers || 0),
    short_channel_payloads: Number(integrity.short_channel_payloads || 0),
    bad_channel_payloads: Number(integrity.bad_channel_payloads || 0),
    channel_order_mismatches: Number(integrity.channel_order_mismatches || 0),
    by_task: integrity.by_task || {},
    by_channel: integrity.by_channel || {},
  };
}

function checkCurriculumStageIdentityBindings(stage, index) {
  const errors = [];
  const stageName = stage.expected_stage_name || stage.stage_name || "";
  const requiredTasks = CURRICULUM_IDENTITY_BINDING_TASKS[stageName] || [];
  if (requiredTasks.length === 0) {
    return errors;
  }
  const selected = stage.identity_bindings;
  const source = stage.source_identity_bindings;
  if (!selected || !source) {
    errors.push(`curriculum stage ${index} ${stageName} is missing identity binding summaries`);
    return errors;
  }
  for (const task of requiredTasks) {
    const selectedTask = selected.by_task?.[task];
    const sourceTask = source.by_task?.[task];
    if (!sourceTask || Number(sourceTask.rows || 0) <= 0) {
      errors.push(`curriculum stage ${index} source identity bindings have no ${task} rows`);
      continue;
    }
    if (!selectedTask || Number(selectedTask.rows || 0) <= 0) {
      errors.push(`curriculum stage ${index} selected no ${task} identity bindings`);
      continue;
    }
    if (Number(selectedTask.rows || 0) !== Number(sourceTask.rows || 0)) {
      errors.push(
        `curriculum stage ${index} ${task} identity bindings ${selectedTask.rows || 0} != source ${sourceTask.rows || 0}`,
      );
    }
    if (selectedTask.binding_hash !== sourceTask.binding_hash) {
      errors.push(
        `curriculum stage ${index} ${task} identity binding hash ${selectedTask.binding_hash || ""} != source ${sourceTask.binding_hash || ""}`,
      );
    }
    if (Number(selectedTask.spirits || 0) !== Number(sourceTask.spirits || 0)) {
      errors.push(
        `curriculum stage ${index} ${task} identity spirits ${selectedTask.spirits || 0} != source ${sourceTask.spirits || 0}`,
      );
    }
    for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
      const selectedCount = Number(selectedTask.counts?.[kind] || 0);
      const sourceCount = Number(sourceTask.counts?.[kind] || 0);
      if (sourceCount <= 0) {
        errors.push(`curriculum stage ${index} source ${task} identity bindings are missing kind ${kind}`);
      } else if (selectedCount !== sourceCount) {
        errors.push(
          `curriculum stage ${index} ${task} identity kind ${kind} count ${selectedCount} != source ${sourceCount}`,
        );
      }
    }
  }
  return errors;
}

function checkCurriculumStageEvidence(stage, index) {
  const errors = [];
  const stageName = canonicalCurriculumStageName(stage.expected_stage_name || stage.stage_name || "");
  const requiredTasks = CURRICULUM_STAGE_EVIDENCE_TASKS[stageName] || [];
  if (requiredTasks.length === 0) {
    return errors;
  }
  const evidence = stage.stage_evidence;
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) {
    errors.push(`curriculum stage ${index} ${stageName} is missing stage_evidence`);
    return errors;
  }
  const expectedSpirits = Number(evidence.expected_spirits || 0);
  const required = evidence.required && typeof evidence.required === "object" ? evidence.required : {};
  for (const task of requiredTasks) {
    const taskEvidenceRow = required[task];
    if (!taskEvidenceRow || Number(taskEvidenceRow.records || 0) <= 0) {
      errors.push(`curriculum stage ${index} ${stageName} evidence has no ${task} rows`);
      continue;
    }
    if (expectedSpirits > 0 && Number(taskEvidenceRow.spirits || 0) !== expectedSpirits) {
      errors.push(
        `curriculum stage ${index} ${stageName} evidence ${task} spirits ${Number(taskEvidenceRow.spirits || 0)} != ${expectedSpirits}`,
      );
    }
  }
  if (stageName === "image") {
    const plan = evidence.image_plan || {};
    const classification = evidence.image_classification || {};
    if (expectedSpirits > 0 && Number(plan.min_spirits || 0) !== expectedSpirits) {
      errors.push(`curriculum stage ${index} image plan min_spirits ${Number(plan.min_spirits || 0)} != ${expectedSpirits}`);
    }
    if (expectedSpirits > 0 && Number(classification.min_spirits || 0) !== expectedSpirits) {
      errors.push(
        `curriculum stage ${index} image classification min_spirits ${Number(classification.min_spirits || 0)} != ${expectedSpirits}`,
      );
    }
  }
  return errors;
}

function curriculumStageEvidenceJson(evidence) {
  if (!evidence || typeof evidence !== "object" || Array.isArray(evidence)) {
    return null;
  }
  return {
    stage_name: evidence.stage_name || "",
    expected_spirits: Number(evidence.expected_spirits || 0),
    records: Number(evidence.records || 0),
    spirits: Number(evidence.spirits || 0),
    required: evidence.required || {},
    image_plan: evidence.image_plan || {},
    image_classification: evidence.image_classification || {},
    image_grounding: evidence.image_grounding || {},
    match: evidence.match || {},
  };
}

function curriculumIdentityBindingSummary(stages) {
  const out = {};
  for (const stage of stages) {
    const stageName = stage.expected_stage_name || stage.stage_name || "";
    const requiredTasks = CURRICULUM_IDENTITY_BINDING_TASKS[stageName] || [];
    if (requiredTasks.length === 0) {
      continue;
    }
    out[stageName] = {
      required_tasks: requiredTasks,
      selected_rows: Number(stage.identity_bindings?.rows || 0),
      source_rows: Number(stage.source_identity_bindings?.rows || 0),
      tasks: Object.fromEntries(
        requiredTasks.map((task) => [
          task,
          {
            selected: identityBindingTaskJson(stage.identity_bindings?.by_task?.[task]),
            source: identityBindingTaskJson(stage.source_identity_bindings?.by_task?.[task]),
            preserved:
              Boolean(stage.identity_bindings?.by_task?.[task]?.binding_hash) &&
              stage.identity_bindings?.by_task?.[task]?.binding_hash ===
                stage.source_identity_bindings?.by_task?.[task]?.binding_hash,
          },
        ]),
      ),
    };
  }
  return out;
}

function identityBindingStageJson(summary) {
  if (!summary) {
    return null;
  }
  return {
    rows: Number(summary.rows || 0),
    binding_hash: summary.binding_hash || "",
    by_task: summary.by_task || {},
    by_kind: summary.by_kind || {},
  };
}

function identityBindingTaskJson(summary) {
  if (!summary) {
    return {
      rows: 0,
      spirits: 0,
      binding_hash: "",
      counts: {},
    };
  }
  return {
    rows: Number(summary.rows || 0),
    spirits: Number(summary.spirits || 0),
    binding_hash: summary.binding_hash || "",
    counts: summary.counts || {},
  };
}

function absentCurriculumStages() {
  return {
    ok: true,
    errors: [],
    present: false,
    stage_count: 0,
    require_loss_non_increasing: true,
    source_corpus_provenance: {
      source_examples: "",
      expected_examples: "",
      examples_match: null,
      source_examples_hash: "",
      expected_examples_hash: "",
      examples_hash_match: null,
      source_examples_consistent: null,
      source_tokens: "",
      expected_tokens: "",
      tokens_match: null,
      source_tokens_hash: "",
      expected_tokens_hash: "",
      tokens_hash_match: null,
      source_tokens_consistent: null,
    },
    identity_binding: {},
    stages: [],
  };
}

function checkDenoiseBridgeSampleBindingProvenance(trace, filePath, sampleReport, errors) {
  const sampleResults = Array.isArray(sampleReport?.results) ? sampleReport.results : [];
  const bridgeResults = Array.isArray(trace.results) ? trace.results : [];
  const summary = {
    sample_binding: sampleReport?.file_path || "",
    sample_count: sampleResults.length,
    bridge_result_count: bridgeResults.length,
    matched_attention_plans: 0,
    missing_attention_plans: 0,
    prompt_mismatches: 0,
    identity_mismatches: 0,
    output_identity_mismatches: 0,
    matches: [],
  };
  if (bridgeResults.length === 0) {
    return summary;
  }
  if (sampleResults.length === 0) {
    errors.push("denoise bridge sample binding provenance has no sample binding results");
    summary.missing_attention_plans = bridgeResults.length;
    return summary;
  }
  for (const [index, result] of bridgeResults.entries()) {
    const attentionPlan = result.attention_plan || "";
    const matched = sampleResults.find((sample) =>
      sample.image_ink16_u8 && sameReferencedPath(attentionPlan, sample.image_ink16_u8, path.dirname(filePath)),
    );
    const match = {
      index,
      attention_plan: attentionPlan,
      matched_sample_dir: matched?.sample_dir || "",
      matched_image_ink16_u8: matched?.image_ink16_u8 || "",
      plan_match: Boolean(matched),
      prompt_match: matched ? String(result.prompt || "") === String(matched.prompt || "") : null,
      bridge_expected_spirit_id: result.expected_spirit_id ?? null,
      sample_expected_spirit_id: matched?.expected_spirit_id ?? null,
      bridge_expected_primary_name: result.expected_primary_name || "",
      sample_expected_primary_name: matched?.expected_primary_name || "",
      identity_match: matched ? bridgeSampleIdentityMatch(result, matched) : null,
      output_identity_match: matched ? bridgeOutputIdentityMatch(result, matched) : null,
    };
    if (!matched) {
      summary.missing_attention_plans += 1;
      errors.push(`denoise bridge result ${index} attention_plan ${attentionPlan || "<missing>"} is not in sample binding results`);
    } else {
      summary.matched_attention_plans += 1;
      if (match.prompt_match === false) {
        summary.prompt_mismatches += 1;
        errors.push(
          `denoise bridge result ${index} prompt ${JSON.stringify(result.prompt || "")} != sample binding prompt ${JSON.stringify(matched.prompt || "")}`,
        );
      }
      if (match.identity_match === false) {
        summary.identity_mismatches += 1;
        errors.push(
          `denoise bridge result ${index} expected identity ${bridgeIdentityLabel(result)} != sample binding expected identity ${sampleIdentityLabel(matched)}`,
        );
      }
      if (match.output_identity_match === false) {
        summary.output_identity_mismatches += 1;
        errors.push(
          `denoise bridge result ${index} output expected identity does not match sample binding expected identity ${sampleIdentityLabel(matched)}`,
        );
      }
    }
    summary.matches.push(match);
  }
  return summary;
}

function bridgeSampleIdentityMatch(bridgeResult, sampleResult) {
  const bridgeId = finiteNumberOrNull(bridgeResult.expected_spirit_id);
  const sampleId = finiteNumberOrNull(sampleResult.expected_spirit_id);
  const bridgeName = String(bridgeResult.expected_primary_name || "").trim();
  const sampleName = String(sampleResult.expected_primary_name || "").trim();
  if (bridgeId === null || sampleId === null) {
    return false;
  }
  if (bridgeId !== null && sampleId !== null && bridgeId !== sampleId) {
    return false;
  }
  if (bridgeName && sampleName && bridgeName !== sampleName) {
    return false;
  }
  return true;
}

function bridgeOutputIdentityMatch(bridgeResult, sampleResult) {
  const sampleId = finiteNumberOrNull(sampleResult.expected_spirit_id);
  const sampleName = String(sampleResult.expected_primary_name || "").trim();
  const details = Array.isArray(bridgeResult.output_signature?.samples_detail)
    ? bridgeResult.output_signature.samples_detail
    : [];
  if (details.length === 0) {
    return null;
  }
  if (sampleId === null) {
    return false;
  }
  for (const detail of details) {
    const detailId = finiteNumberOrNull(detail.expected_spirit_id);
    const detailName = String(detail.expected_primary_name || "").trim();
    if (detailId === null) {
      return false;
    }
    if (sampleId !== null && detailId !== null && detailId !== sampleId) {
      return false;
    }
    if (sampleName && detailName && detailName !== sampleName) {
      return false;
    }
  }
  return true;
}

function bridgeIdentityLabel(result) {
  return `${result.expected_spirit_id ?? "<missing>"}:${result.expected_primary_name || "<missing>"}`;
}

function sampleIdentityLabel(result) {
  return `${result.expected_spirit_id ?? "<missing>"}:${result.expected_primary_name || "<missing>"}`;
}

function checkDenoiseBridgeDenoiserModelProvenance(trace, filePath, errors) {
  const bridgeDir = path.dirname(filePath);
  const results = Array.isArray(trace.results) ? trace.results : [];
  const summary = {
    denoise_model: trace.denoise_model || "",
    resolved_denoise_model: trace.resolved_denoise_model || "",
    denoise_model_hash: trace.denoise_model_hash || "",
    denoise_model_hashes: Array.isArray(trace.denoise_model_hashes)
      ? trace.denoise_model_hashes.map((value) => String(value || "")).filter(Boolean)
      : [],
    denoise_model_consistent: trace.denoise_model_consistent === true,
    result_count: results.length,
    resolved_result_model_count: 0,
    missing_model_refs: 0,
    missing_model_hashes: 0,
    unresolved_models: 0,
    hash_mismatches: 0,
    unique_recomputed_hashes: [],
    results: [],
  };
  const recomputedHashes = [];
  for (const [index, result] of results.entries()) {
    const modelRef = result.denoise_model || "";
    const recordedHash = result.denoise_model_hash || "";
    const resolvedModel = resolveDenoiseModelReference(modelRef, result.denoise_trace || "", bridgeDir);
    const expectedHash = resolvedModel ? fnv64FileHex(resolvedModel) : "";
    const row = {
      index,
      denoise_model: modelRef,
      resolved_denoise_model: resolvedModel,
      denoise_model_hash: recordedHash,
      expected_denoise_model_hash: expectedHash,
      hash_match: recordedHash && expectedHash ? recordedHash === expectedHash : null,
    };
    if (!modelRef) {
      summary.missing_model_refs += 1;
      errors.push(`denoise bridge result ${index} denoise_model is missing`);
    }
    if (!recordedHash) {
      summary.missing_model_hashes += 1;
      errors.push(`denoise bridge result ${index} denoise_model_hash is missing`);
    }
    if (!resolvedModel) {
      summary.unresolved_models += 1;
      errors.push(`denoise bridge result ${index} denoise_model ${JSON.stringify(modelRef)} could not be resolved`);
    } else {
      summary.resolved_result_model_count += 1;
      recomputedHashes.push(expectedHash);
    }
    if (recordedHash && expectedHash && recordedHash !== expectedHash) {
      summary.hash_mismatches += 1;
      errors.push(`denoise bridge result ${index} denoise_model_hash ${recordedHash} != recomputed ${expectedHash}`);
    }
    summary.results.push(row);
  }
  summary.unique_recomputed_hashes = [...new Set(recomputedHashes)].sort();
  if (!summary.denoise_model_hash) {
    errors.push("denoise bridge denoise_model_hash is missing");
  }
  if (summary.unique_recomputed_hashes.length !== 1) {
    errors.push(`denoise bridge expected exactly one recomputed denoiser model hash, found ${summary.unique_recomputed_hashes.length}`);
  }
  if (
    summary.denoise_model_hash &&
    summary.unique_recomputed_hashes.length === 1 &&
    summary.denoise_model_hash !== summary.unique_recomputed_hashes[0]
  ) {
    errors.push(
      `denoise bridge denoise_model_hash ${summary.denoise_model_hash} != recomputed ${summary.unique_recomputed_hashes[0]}`,
    );
  }
  if (summary.denoise_model_hashes.length > 0) {
    const recordedHashes = [...new Set(summary.denoise_model_hashes)].sort();
    const expectedHashes = summary.unique_recomputed_hashes;
    const same =
      recordedHashes.length === expectedHashes.length &&
      recordedHashes.every((hash, index) => hash === expectedHashes[index]);
    if (!same) {
      errors.push(
        `denoise bridge denoise_model_hashes ${recordedHashes.join(",") || "<empty>"} != recomputed ${expectedHashes.join(",") || "<empty>"}`,
      );
    }
  }
  if (trace.denoise_model_consistent !== true) {
    errors.push("denoise bridge denoise_model_consistent is not true");
  }
  summary.ok =
    summary.missing_model_refs === 0 &&
    summary.missing_model_hashes === 0 &&
    summary.unresolved_models === 0 &&
    summary.hash_mismatches === 0 &&
    summary.unique_recomputed_hashes.length === 1 &&
    summary.denoise_model_hash === summary.unique_recomputed_hashes[0] &&
    trace.denoise_model_consistent === true;
  return summary;
}

function resolveDenoiseModelReference(modelRef, traceRef, bridgeDir) {
  if (!modelRef) {
    return "";
  }
  const baseDirs = [bridgeDir];
  if (traceRef) {
    for (const candidate of referencedPathCandidates(traceRef, bridgeDir)) {
      baseDirs.push(path.dirname(candidate));
    }
  }
  const candidates = path.isAbsolute(modelRef)
    ? [path.resolve(modelRef)]
    : [path.resolve(modelRef), ...baseDirs.map((baseDir) => path.resolve(baseDir, modelRef))];
  for (const candidate of [...new Set(candidates.map(normalizeReferencedPath))]) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return "";
}

function checkDenoiseBridgeOutputProvenance(trace, filePath, sampleReport, retrievalReport, config, errors) {
  const results = Array.isArray(trace.results) ? trace.results : [];
  const summary = absentDenoiseBridgeOutputProvenance();
  summary.required = results.length > 0;
  summary.result_count = results.length;
  summary.retrieval_required =
    config.requireDenoiseOutputIdentity ||
    (trace.output_image_to_text_identification !== null && trace.output_image_to_text_identification !== undefined) ||
    (trace.min_output_retrieval_image_margin !== null && trace.min_output_retrieval_image_margin !== undefined) ||
    Boolean(trace.retrieval_head_model_hash);

  let retrievalHead = null;
  if (summary.retrieval_required) {
    summary.expected_retrieval_head_model_hash = expectedRetrievalHeadModelHash(retrievalReport);
    summary.config_retrieval_head_model_hash = trace.retrieval_head_model_hash || "";
    summary.config_hash_match =
      summary.config_retrieval_head_model_hash && summary.expected_retrieval_head_model_hash
        ? summary.config_retrieval_head_model_hash === summary.expected_retrieval_head_model_hash
        : null;
    if (!summary.config_retrieval_head_model_hash) {
      errors.push("denoise bridge output provenance requires retrieval_head_model_hash");
    } else if (summary.config_hash_match === false) {
      errors.push(
        `denoise bridge output retrieval_head_model_hash ${summary.config_retrieval_head_model_hash} != retrieval head eval model_hash ${summary.expected_retrieval_head_model_hash}`,
      );
    }
    const headPath = resolveDenoiseBridgeRetrievalHeadPath(trace, filePath, retrievalReport, config);
    summary.retrieval_head = headPath.source;
    summary.resolved_retrieval_head = headPath.path;
    summary.retrieval_head_present = headPath.exists;
    if (!headPath.exists) {
      errors.push(`denoise bridge output retrieval head ${headPath.source || "<missing>"} could not be resolved`);
    } else {
      try {
        retrievalHead = readGenerativeRetrievalHead(headPath.path);
        summary.retrieval_head_model_hash = retrievalHead.model_hash || "";
        summary.retrieval_head_feature_count = Number(retrievalHead.feature_count || 0);
        summary.retrieval_head_label_count = Array.isArray(retrievalHead.labels) ? retrievalHead.labels.length : 0;
        const recomputedHeadHash = recomputeRetrievalHeadHash(retrievalHead.raw);
        summary.recomputed_retrieval_head_model_hash = recomputedHeadHash;
        summary.retrieval_head_hash_verified =
          Boolean(summary.retrieval_head_model_hash) && summary.retrieval_head_model_hash === recomputedHeadHash;
        if (!summary.retrieval_head_hash_verified) {
          errors.push(`denoise bridge output retrieval head model_hash ${summary.retrieval_head_model_hash || ""} != recomputed ${recomputedHeadHash}`);
        }
        summary.retrieval_head_hash_match =
          summary.retrieval_head_model_hash && summary.expected_retrieval_head_model_hash
            ? summary.retrieval_head_model_hash === summary.expected_retrieval_head_model_hash
            : null;
        if (summary.retrieval_head_hash_match === false) {
          errors.push(
            `denoise bridge output retrieval head model_hash ${summary.retrieval_head_model_hash} != retrieval head eval model_hash ${summary.expected_retrieval_head_model_hash}`,
          );
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        summary.invalid_retrieval_head = true;
        errors.push(`denoise bridge output retrieval head could not be read: ${message}`);
      }
    }
  }

  for (const [index, result] of results.entries()) {
    const record = recomputeDenoiseBridgeOutputResult(
      result,
      index,
      filePath,
      sampleReport,
      retrievalHead,
      summary.retrieval_required,
      errors,
    );
    summary.results.push(record);
    if (!record.attention_plan_present) summary.missing_attention_plans += 1;
    if (!record.raw_samples_present) summary.missing_raw_samples += 1;
    if (record.invalid_raw_samples) summary.invalid_raw_samples += 1;
    if (record.scored) summary.scored_results += 1;
    summary.result_mismatches += record.mismatches.length;
    summary.detail_mismatches += record.detail_mismatches.length;
  }

  summary.recomputed = summarizeDenoiseBridgeOutputAggregate(summary.results);
  compareDenoiseBridgeAggregateField(summary, "min_output_signature_distance", trace.min_output_signature_distance, summary.recomputed.min_output_signature_distance);
  compareDenoiseBridgeAggregateField(summary, "min_output_ink_range", trace.min_output_ink_range, summary.recomputed.min_output_ink_range);
  if (summary.retrieval_required) {
    compareDenoiseBridgeAggregateField(
      summary,
      "output_image_to_text_identification",
      trace.output_image_to_text_identification,
      summary.recomputed.output_image_to_text_identification,
    );
    compareDenoiseBridgeAggregateField(
      summary,
      "min_output_retrieval_image_margin",
      trace.min_output_retrieval_image_margin,
      summary.recomputed.min_output_retrieval_image_margin,
    );
  }
  if (summary.aggregate_mismatches.length > 0) {
    errors.push(`denoise bridge output aggregate mismatches: ${summary.aggregate_mismatches.join(",")}`);
  }

  summary.ok =
    summary.required === true &&
    summary.result_count > 0 &&
    summary.scored_results === summary.result_count &&
    summary.missing_attention_plans === 0 &&
    summary.missing_raw_samples === 0 &&
    summary.invalid_raw_samples === 0 &&
    summary.result_mismatches === 0 &&
    summary.detail_mismatches === 0 &&
    summary.aggregate_mismatches.length === 0 &&
    (!summary.retrieval_required ||
      (summary.retrieval_head_present === true &&
        summary.invalid_retrieval_head === false &&
        summary.retrieval_head_hash_verified === true &&
        summary.retrieval_head_hash_match !== false &&
        summary.config_hash_match !== false &&
        Boolean(summary.config_retrieval_head_model_hash)));
  return summary;
}

function absentDenoiseBridgeOutputProvenance() {
  return {
    required: false,
    ok: false,
    result_count: 0,
    retrieval_required: false,
    config_retrieval_head_model_hash: "",
    expected_retrieval_head_model_hash: "",
    config_hash_match: null,
    retrieval_head: "",
    resolved_retrieval_head: "",
    retrieval_head_present: false,
    invalid_retrieval_head: false,
    retrieval_head_model_hash: "",
    recomputed_retrieval_head_model_hash: "",
    retrieval_head_hash_verified: false,
    retrieval_head_hash_match: null,
    retrieval_head_feature_count: 0,
    retrieval_head_label_count: 0,
    scored_results: 0,
    missing_attention_plans: 0,
    missing_raw_samples: 0,
    invalid_raw_samples: 0,
    result_mismatches: 0,
    detail_mismatches: 0,
    aggregate_mismatches: [],
    recomputed: {
      min_output_signature_distance: null,
      min_output_ink_range: null,
      output_image_to_text_identification: null,
      min_output_retrieval_image_margin: null,
    },
    results: [],
  };
}

function resolveDenoiseBridgeRetrievalHeadPath(trace, filePath, retrievalReport, config) {
  const sources = [
    config.retrievalHeadPath || "",
    retrievalReport?.class_retrieval_head?.path || "",
    retrievalReport?.model || "",
    trace.retrieval_head || "",
  ].filter(Boolean);
  const bridgeDir = path.dirname(filePath);
  for (const source of sources) {
    const candidates = path.isAbsolute(source)
      ? [source]
      : [
          path.resolve(source),
          path.resolve(bridgeDir, source),
        ];
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return { source, path: candidate, exists: true };
      }
    }
  }
  const fallback = sources[0] || "";
  return {
    source: fallback,
    path: fallback ? (path.isAbsolute(fallback) ? fallback : path.resolve(fallback)) : "",
    exists: false,
  };
}

function recomputeDenoiseBridgeOutputResult(result, index, filePath, sampleReport, retrievalHead, retrievalRequired, errors) {
  const bridgeDir = path.dirname(filePath);
  const sampleResults = Array.isArray(sampleReport?.results) ? sampleReport.results : [];
  const attentionPlanRef = result.attention_plan || "";
  const attentionPlan = resolveBridgeReferencedPath(attentionPlanRef, bridgeDir, result.denoise_trace || "");
  const rawSamples = resolveBridgeReferencedPath(result.denoise_raw_samples || "", bridgeDir, result.denoise_trace || "");
  const matchedSample = sampleResults.find((sample) =>
    sample.image_ink16_u8 && sameReferencedPath(attentionPlanRef, sample.image_ink16_u8, bridgeDir),
  );
  const expectedSpiritId = finiteNumberOrNull(matchedSample?.expected_spirit_id);
  const expectedPrimaryName = matchedSample?.expected_primary_name || "";
  const outputSignature = result.output_signature || {};
  const record = {
    index,
    attention_plan: attentionPlanRef,
    resolved_attention_plan: attentionPlan,
    attention_plan_present: Boolean(attentionPlan && fs.existsSync(attentionPlan)),
    denoise_raw_samples: result.denoise_raw_samples || "",
    resolved_raw_samples: rawSamples,
    raw_samples_present: Boolean(rawSamples && fs.existsSync(rawSamples)),
    matched_sample_dir: matchedSample?.sample_dir || "",
    expected_spirit_id: expectedSpiritId,
    expected_primary_name: expectedPrimaryName,
    invalid_raw_samples: false,
    scored: false,
    reported: {
      samples: finiteNumberOrNull(outputSignature.samples),
      min_signature_distance: finiteNumberOrNull(outputSignature.min_signature_distance),
      mean_signature_distance_q8: finiteNumberOrNull(outputSignature.mean_signature_distance_q8),
      min_ink_range: finiteNumberOrNull(outputSignature.min_ink_range),
      output_image_to_text_identification:
        outputSignature.output_image_to_text_identification === null ||
        outputSignature.output_image_to_text_identification === undefined
          ? null
          : outputSignature.output_image_to_text_identification === true,
      min_retrieval_image_margin: finiteNumberOrNull(outputSignature.min_retrieval_image_margin),
    },
    recomputed: {
      samples: 0,
      min_signature_distance: null,
      mean_signature_distance_q8: null,
      min_ink_range: null,
      output_image_to_text_identification: null,
      min_retrieval_image_margin: null,
    },
    sample_details: [],
    mismatches: [],
    detail_mismatches: [],
    ok: false,
  };
  if (!record.attention_plan_present) {
    errors.push(`denoise bridge result ${index} attention_plan ${attentionPlanRef || "<missing>"} could not be resolved for output recompute`);
    return record;
  }
  let plan = null;
  try {
    plan = Array.from(fs.readFileSync(attentionPlan));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    errors.push(`denoise bridge result ${index} attention_plan could not be read: ${message}`);
    return record;
  }
  if (plan.length !== GENERATED_RETRIEVAL_BINS) {
    errors.push(`denoise bridge result ${index} attention_plan length ${plan.length} != ${GENERATED_RETRIEVAL_BINS}`);
    return record;
  }
  if (!record.raw_samples_present) {
    errors.push(`denoise bridge result ${index} denoise_raw_samples ${record.denoise_raw_samples || "<missing>"} could not be resolved`);
    return record;
  }
  let raw = null;
  try {
    raw = fs.readFileSync(rawSamples);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record.invalid_raw_samples = true;
    errors.push(`denoise bridge result ${index} raw samples could not be read: ${message}`);
    return record;
  }
  if (raw.length === 0 || raw.length % GENERATED_RETRIEVAL_IMAGE_BYTES !== 0) {
    record.invalid_raw_samples = true;
    errors.push(
      `denoise bridge result ${index} raw sample byte count ${raw.length} is not a positive multiple of ${GENERATED_RETRIEVAL_IMAGE_BYTES}`,
    );
    return record;
  }
  if (retrievalRequired && !retrievalHead) {
    errors.push(`denoise bridge result ${index} output identity recompute requires a retrieval head`);
  }
  const stats = recomputeDenoiseBridgeOutputStats(raw, plan, expectedSpiritId, expectedPrimaryName, retrievalHead);
  record.scored = true;
  record.recomputed = stats.summary;
  record.sample_details = stats.details;
  compareDenoiseBridgeResultField(record, "samples");
  compareDenoiseBridgeResultField(record, "min_signature_distance");
  compareDenoiseBridgeResultField(record, "mean_signature_distance_q8");
  compareDenoiseBridgeResultField(record, "min_ink_range");
  if (retrievalRequired) {
    compareDenoiseBridgeResultField(record, "output_image_to_text_identification");
    compareDenoiseBridgeResultField(record, "min_retrieval_image_margin");
  }
  compareDenoiseBridgeSampleDetails(record, outputSignature.samples_detail || [], retrievalRequired);
  if (record.mismatches.length > 0) {
    errors.push(`denoise bridge result ${index} output recompute mismatches: ${record.mismatches.join(",")}`);
  }
  if (record.detail_mismatches.length > 0) {
    errors.push(`denoise bridge result ${index} output sample-detail mismatches: ${record.detail_mismatches.join(",")}`);
  }
  record.ok = record.scored && record.mismatches.length === 0 && record.detail_mismatches.length === 0;
  return record;
}

function resolveBridgeReferencedPath(ref, bridgeDir, traceRef = "") {
  if (!ref) {
    return "";
  }
  const baseDirs = [bridgeDir];
  if (traceRef) {
    for (const candidate of referencedPathCandidates(traceRef, bridgeDir)) {
      baseDirs.push(path.dirname(candidate));
    }
  }
  const candidates = path.isAbsolute(ref)
    ? [path.resolve(ref)]
    : [path.resolve(ref), ...baseDirs.map((baseDir) => path.resolve(baseDir, ref))];
  return [...new Set(candidates.map(normalizeReferencedPath))].find((candidate) => fs.existsSync(candidate)) || "";
}

function recomputeDenoiseBridgeOutputStats(raw, plan, expectedSpiritId, expectedPrimaryName, retrievalHead) {
  const details = [];
  for (let offset = 0; offset < raw.length; offset += GENERATED_RETRIEVAL_IMAGE_BYTES) {
    const index = offset / GENERATED_RETRIEVAL_IMAGE_BYTES;
    const image = raw.subarray(offset, offset + GENERATED_RETRIEVAL_IMAGE_BYTES);
    const signature = generatedRetrievalSampleSignature(image);
    const detail = {
      index,
      signature_distance: signatureDistance256(signature, plan),
      ink_range: denoiseBridgeImageInkRange(image),
    };
    if (expectedSpiritId !== null) {
      detail.expected_spirit_id = expectedSpiritId;
      detail.expected_primary_name = expectedPrimaryName;
    }
    if (retrievalHead && expectedSpiritId !== null) {
      const ranked = rankGeneratedRetrievalImage(retrievalHead, signature, retrievalHead.labels.length);
      const rank = generatedRetrievalTargetRank(ranked, expectedSpiritId, retrievalHead.labels.length);
      const stats = generatedRetrievalRankStats(ranked, expectedSpiritId);
      detail.retrieval_image_rank = rank;
      detail.retrieval_image_margin = stats.margin;
      detail.retrieval_image_top1_spirit_id = ranked[0]?.spirit_id ?? null;
      detail.retrieval_image_top1_primary_name = ranked[0]?.primary_name ?? "";
      detail.image_to_text_identity = detail.retrieval_image_rank === 1 && ranked[0]?.spirit_id === expectedSpiritId;
    }
    details.push(detail);
  }
  const retrievalDetails = details.filter((detail) => detail.image_to_text_identity !== undefined);
  return {
    summary: {
      samples: details.length,
      min_signature_distance: details.length === 0 ? null : Math.min(...details.map((detail) => detail.signature_distance)),
      mean_signature_distance_q8:
        details.length === 0
          ? null
          : Math.round((details.reduce((sum, detail) => sum + detail.signature_distance, 0) * 256) / details.length),
      min_ink_range: details.length === 0 ? null : Math.min(...details.map((detail) => detail.ink_range)),
      output_image_to_text_identification:
        retrievalDetails.length === 0 ? null : retrievalDetails.every((detail) => detail.image_to_text_identity === true),
      min_retrieval_image_margin:
        retrievalDetails.length === 0
          ? null
          : Math.min(...retrievalDetails.map((detail) => detail.retrieval_image_margin ?? 0)),
    },
    details: details.slice(0, 8),
  };
}

function signatureDistance256(left, right) {
  let distance = 0;
  for (let index = 0; index < GENERATED_RETRIEVAL_BINS; index += 1) {
    distance += Math.abs(Number(left[index] || 0) - Number(right[index] || 0));
  }
  return distance;
}

function denoiseBridgeImageInkRange(image) {
  let min = 255;
  let max = 0;
  for (const value of image) {
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return max - min;
}

function compareDenoiseBridgeResultField(record, key) {
  const reported = record.reported[key];
  const recomputed = record.recomputed[key];
  if (!sameNullableValue(reported, recomputed)) {
    record.mismatches.push(`${key}=${reported} != recomputed ${recomputed}`);
  }
}

function compareDenoiseBridgeAggregateField(summary, key, reported, recomputed) {
  const normalized = typeof recomputed === "boolean" ? Boolean(reported) : finiteNumberOrNull(reported);
  const left = reported === null || reported === undefined ? null : normalized;
  if (!sameNullableValue(left, recomputed)) {
    summary.aggregate_mismatches.push(`${key}=${reported ?? null} != recomputed ${recomputed}`);
  }
}

function compareDenoiseBridgeSampleDetails(record, reportedDetails, retrievalRequired) {
  const reported = Array.isArray(reportedDetails) ? reportedDetails : [];
  if (reported.length !== record.sample_details.length) {
    record.detail_mismatches.push(`samples_detail length ${reported.length} != recomputed ${record.sample_details.length}`);
    return;
  }
  const fields = [
    "index",
    "signature_distance",
    "ink_range",
    "expected_spirit_id",
    "expected_primary_name",
  ];
  if (retrievalRequired) {
    fields.push(
      "retrieval_image_rank",
      "retrieval_image_margin",
      "retrieval_image_top1_spirit_id",
      "retrieval_image_top1_primary_name",
      "image_to_text_identity",
    );
  }
  for (let index = 0; index < record.sample_details.length; index += 1) {
    const actual = reported[index] || {};
    const expected = record.sample_details[index];
    for (const field of fields) {
      const actualValue = actual[field] === undefined ? null : actual[field];
      const expectedValue = expected[field] === undefined ? null : expected[field];
      if (!sameNullableValue(actualValue, expectedValue)) {
        record.detail_mismatches.push(`samples_detail[${index}].${field}=${actualValue} != recomputed ${expectedValue}`);
      }
    }
  }
}

function summarizeDenoiseBridgeOutputAggregate(results) {
  const scored = results.filter((result) => result.scored);
  const retrieval = scored.filter((result) => result.recomputed.output_image_to_text_identification !== null);
  return {
    min_output_signature_distance:
      scored.length === 0 ? null : Math.min(...scored.map((result) => result.recomputed.min_signature_distance ?? Number.POSITIVE_INFINITY)),
    min_output_ink_range:
      scored.length === 0 ? null : Math.min(...scored.map((result) => result.recomputed.min_ink_range ?? Number.POSITIVE_INFINITY)),
    output_image_to_text_identification:
      retrieval.length === 0 ? null : retrieval.every((result) => result.recomputed.output_image_to_text_identification === true),
    min_output_retrieval_image_margin:
      retrieval.length === 0
        ? null
        : Math.min(...retrieval.map((result) => result.recomputed.min_retrieval_image_margin ?? Number.POSITIVE_INFINITY)),
  };
}

function sameNullableValue(left, right) {
  const normalizedLeft = left === undefined ? null : left;
  const normalizedRight = right === undefined ? null : right;
  if (normalizedLeft === null || normalizedRight === null) {
    return normalizedLeft === null && normalizedRight === null;
  }
  if (typeof normalizedLeft === "boolean" || typeof normalizedRight === "boolean") {
    return typeof normalizedLeft === "boolean" && typeof normalizedRight === "boolean" && normalizedLeft === normalizedRight;
  }
  if (Number.isFinite(Number(normalizedLeft)) && Number.isFinite(Number(normalizedRight))) {
    return Number(normalizedLeft) === Number(normalizedRight);
  }
  return String(normalizedLeft) === String(normalizedRight);
}

function denoiseBridgeTargetCoverage(trace, results, config, errors) {
  const ids = results
    .map((result) => Number(result?.expected_spirit_id || 0))
    .filter((id) => Number.isInteger(id) && id >= 1 && id <= 72);
  const uniqueIds = Array.from(new Set(ids)).sort((left, right) => left - right);
  const missingIds = Array.from({ length: 72 }, (_unused, index) => index + 1).filter(
    (id) => !uniqueIds.includes(id),
  );
  const reportedUnique = finiteNumberOrNull(trace.expected_unique_targets);
  if (reportedUnique !== null && reportedUnique !== uniqueIds.length) {
    errors.push(`denoise bridge expected_unique_targets ${reportedUnique} != recomputed ${uniqueIds.length}`);
  }
  const reportedIds = Array.isArray(trace.unique_expected_spirit_ids)
    ? trace.unique_expected_spirit_ids.map((id) => Number(id)).filter((id) => Number.isInteger(id))
    : null;
  if (reportedIds && !sameNumberArray(reportedIds, uniqueIds)) {
    errors.push("denoise bridge unique_expected_spirit_ids do not match recompute");
  }
  if (trace.target_coverage_ok === false) {
    errors.push("denoise bridge target_coverage_ok is false");
  }
  if (config.minDenoiseBridgeUniqueTargets > 0 && uniqueIds.length < config.minDenoiseBridgeUniqueTargets) {
    errors.push(
      `denoise bridge unique targets ${uniqueIds.length} < ${config.minDenoiseBridgeUniqueTargets}`,
    );
  }
  return {
    min_unique_targets: config.minDenoiseBridgeUniqueTargets,
    expected_spirit_ids: ids,
    unique_expected_spirit_ids: uniqueIds,
    expected_unique_targets: uniqueIds.length,
    missing_expected_spirit_ids: missingIds,
    target_coverage_ok: uniqueIds.length >= config.minDenoiseBridgeUniqueTargets,
  };
}

function sameNumberArray(left, right) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((value, index) => Number(value) === Number(right[index]));
}

function checkDenoiseBridge(trace, filePath, config, retrievalReport, sampleReport) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_attention_denoise_bridge_check.v1", errors);
  const retrievalHeadProvenance = checkRetrievalEvidenceHeadProvenance(
    "denoise bridge",
    trace.retrieval_head_model_hash,
    retrievalReport,
    errors,
  );
  const sampleBindingProvenance = checkDenoiseBridgeSampleBindingProvenance(trace, filePath, sampleReport, errors);
  const denoiserModelProvenance = checkDenoiseBridgeDenoiserModelProvenance(trace, filePath, errors);
  const outputProvenance = checkDenoiseBridgeOutputProvenance(trace, filePath, sampleReport, retrievalReport, config, errors);
  if (trace.ok !== true) {
    errors.push("denoise bridge ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `denoise bridge: ${error}`));
  }
  if (Number(trace.pairs || 0) <= 0) {
    errors.push("denoise bridge checked no pairs");
  }
  if (trace.trace_integrity_ok !== true) {
    errors.push("denoise bridge trace_integrity_ok is not true");
  }
  const minOutputSignatureDistance = finiteNumberOrNull(trace.min_output_signature_distance);
  const minOutputInkRange = finiteNumberOrNull(trace.min_output_ink_range);
  const minOutputRetrievalImageMargin = finiteNumberOrNull(trace.min_output_retrieval_image_margin);
  if (minOutputSignatureDistance === null) {
    errors.push("denoise bridge missing min_output_signature_distance");
  }
  if (minOutputInkRange === null) {
    errors.push("denoise bridge missing min_output_ink_range");
  } else if (minOutputInkRange <= 0) {
    errors.push(`denoise bridge min_output_ink_range ${minOutputInkRange} <= 0`);
  }
  if (trace.output_image_to_text_identification !== null && trace.output_image_to_text_identification !== undefined) {
    if (trace.output_image_to_text_identification !== true) {
      errors.push("denoise bridge output image-to-text identification is not true");
    }
    if (minOutputRetrievalImageMargin === null) {
      errors.push("denoise bridge missing min_output_retrieval_image_margin");
    }
  }
  if (config.requireDenoiseOutputIdentity) {
    if (!trace.retrieval_head) {
      errors.push("denoise bridge output identity requires retrieval_head");
    }
    if (trace.output_image_to_text_identification !== true) {
      errors.push("denoise bridge output image-to-text identification is required but not true");
    }
    if (minOutputRetrievalImageMargin === null) {
      errors.push("denoise bridge output identity requires min_output_retrieval_image_margin");
    } else if (minOutputRetrievalImageMargin <= 0) {
      errors.push(`denoise bridge min_output_retrieval_image_margin ${minOutputRetrievalImageMargin} <= 0`);
    }
  }
  const results = Array.isArray(trace.results) ? trace.results : [];
  if (results.length !== Number(trace.pairs || 0)) {
    errors.push(`denoise bridge results ${results.length} != pairs ${Number(trace.pairs || 0)}`);
  }
  const targetCoverage = denoiseBridgeTargetCoverage(trace, results, config, errors);
  for (const [index, result] of results.entries()) {
    if (result.ok !== true) {
      errors.push(`denoise bridge result ${index} ok is not true`);
    }
    const traceIntegrity = result.trace_integrity;
    if (!traceIntegrity || typeof traceIntegrity !== "object" || Array.isArray(traceIntegrity)) {
      errors.push(`denoise bridge result ${index} missing trace_integrity`);
    } else {
      if (traceIntegrity.ok !== true) {
        errors.push(`denoise bridge result ${index} trace_integrity ok is not true`);
      }
      if (Array.isArray(traceIntegrity.violations) && traceIntegrity.violations.length > 0) {
        for (const violation of traceIntegrity.violations) {
          const field = violation?.field ? ` ${violation.field}` : "";
          const reason = violation?.reason || JSON.stringify(violation);
          errors.push(`denoise bridge result ${index} trace integrity violation${field}: ${reason}`);
        }
      }
    }
    const outputSignature = result.output_signature;
    if (!outputSignature) {
      errors.push(`denoise bridge result ${index} missing output_signature`);
      continue;
    }
    if (Number(outputSignature.samples || 0) <= 0) {
      errors.push(`denoise bridge result ${index} output_signature samples ${outputSignature.samples || 0} <= 0`);
    }
    const resultDistance = finiteNumberOrNull(outputSignature.min_signature_distance);
    if (resultDistance === null) {
      errors.push(`denoise bridge result ${index} missing output min_signature_distance`);
    }
    const resultInkRange = finiteNumberOrNull(outputSignature.min_ink_range);
    if (resultInkRange === null) {
      errors.push(`denoise bridge result ${index} missing output min_ink_range`);
    } else if (resultInkRange <= 0) {
      errors.push(`denoise bridge result ${index} output min_ink_range ${resultInkRange} <= 0`);
    }
    if (
      outputSignature.output_image_to_text_identification !== null &&
      outputSignature.output_image_to_text_identification !== undefined &&
      outputSignature.output_image_to_text_identification !== true
    ) {
      errors.push(`denoise bridge result ${index} output image-to-text identification is not true`);
    }
    if (config.requireDenoiseOutputIdentity) {
      if (outputSignature.output_image_to_text_identification !== true) {
        errors.push(`denoise bridge result ${index} output image-to-text identification is required but not true`);
      }
      const resultRetrievalMargin = finiteNumberOrNull(outputSignature.min_retrieval_image_margin);
      if (resultRetrievalMargin === null) {
        errors.push(`denoise bridge result ${index} missing output min_retrieval_image_margin`);
      } else if (resultRetrievalMargin <= 0) {
        errors.push(`denoise bridge result ${index} output min_retrieval_image_margin ${resultRetrievalMargin} <= 0`);
      }
    }
  }
  return {
    ok: errors.length === 0,
    errors,
    present: true,
    pairs: Number(trace.pairs || 0),
    min_unique_targets: targetCoverage.min_unique_targets,
    expected_spirit_ids: targetCoverage.expected_spirit_ids,
    unique_expected_spirit_ids: targetCoverage.unique_expected_spirit_ids,
    expected_unique_targets: targetCoverage.expected_unique_targets,
    missing_expected_spirit_ids: targetCoverage.missing_expected_spirit_ids,
    target_coverage_ok: targetCoverage.target_coverage_ok,
    denoise_model: trace.denoise_model || "",
    resolved_denoise_model: trace.resolved_denoise_model || "",
    denoise_model_hash: trace.denoise_model_hash || "",
    denoise_model_provenance: denoiserModelProvenance,
    retrieval_head: trace.retrieval_head || null,
    retrieval_head_model_hash: trace.retrieval_head_model_hash || "",
    retrieval_head_provenance: retrievalHeadProvenance,
    sample_binding_provenance: sampleBindingProvenance,
    output_provenance: outputProvenance,
    min_output_signature_distance: minOutputSignatureDistance,
    min_output_ink_range: minOutputInkRange,
    trace_integrity_ok: trace.trace_integrity_ok === true,
    require_output_image_to_text_identification: config.requireDenoiseOutputIdentity,
    output_image_to_text_identification:
      trace.output_image_to_text_identification === null ||
      trace.output_image_to_text_identification === undefined
        ? null
        : trace.output_image_to_text_identification === true,
    min_output_retrieval_image_margin: minOutputRetrievalImageMargin,
    results: results.map((result) => ({
      ok: result.ok === true,
      prompt: result.prompt || "",
      attention_plan: result.attention_plan || "",
      denoise_trace: result.denoise_trace || "",
      denoise_model: result.denoise_model || "",
      resolved_denoise_model: result.resolved_denoise_model || "",
      denoise_model_hash: result.denoise_model_hash || "",
      denoise_raw_samples: result.denoise_raw_samples || "",
      trace_integrity: result.trace_integrity || null,
      output_signature: result.output_signature || null,
    })),
  };
}

function absentDenoiseBridge() {
  return {
    ok: true,
    errors: [],
    present: false,
    pairs: 0,
    min_unique_targets: 0,
    expected_spirit_ids: [],
    unique_expected_spirit_ids: [],
    expected_unique_targets: 0,
    missing_expected_spirit_ids: [],
    target_coverage_ok: false,
    denoise_model: "",
    resolved_denoise_model: "",
    denoise_model_hash: "",
    denoise_model_provenance: {
      ok: false,
      denoise_model: "",
      resolved_denoise_model: "",
      denoise_model_hash: "",
      denoise_model_hashes: [],
      denoise_model_consistent: false,
      result_count: 0,
      resolved_result_model_count: 0,
      missing_model_refs: 0,
      missing_model_hashes: 0,
      unresolved_models: 0,
      hash_mismatches: 0,
      unique_recomputed_hashes: [],
      results: [],
    },
    retrieval_head: null,
    retrieval_head_model_hash: "",
    retrieval_head_provenance: {
      expected_model_hash: "",
      model_hash: "",
      hash_match: null,
    },
    output_provenance: absentDenoiseBridgeOutputProvenance(),
    sample_binding_provenance: {
      sample_binding: "",
      sample_count: 0,
      bridge_result_count: 0,
      matched_attention_plans: 0,
      missing_attention_plans: 0,
      prompt_mismatches: 0,
      identity_mismatches: 0,
      output_identity_mismatches: 0,
      matches: [],
    },
    min_output_signature_distance: null,
    min_output_ink_range: null,
    trace_integrity_ok: null,
    require_output_image_to_text_identification: false,
    output_image_to_text_identification: null,
    min_output_retrieval_image_margin: null,
    results: [],
  };
}

function checkGroundedCorpus(trace, filePath, config) {
  const errors = [];
  expectSchema(trace, filePath, "nsrl.solomon_v2_grounded_corpus_check.v1", errors);
  const examplesProvenance = checkGroundedCorpusExamplesProvenance(trace, filePath, config, errors);
  const textIndexProvenance = checkSourceTextIndexProvenance(
    "grounded corpus",
    trace.text_index,
    trace.text_index_hash,
    filePath,
    config,
    errors,
  );
  if (trace.ok !== true) {
    errors.push("grounded corpus ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    errors.push(...trace.errors.map((error) => `grounded corpus: ${error}`));
  }
  if (trace.require_source_provenance !== true) {
    errors.push("grounded corpus source provenance is not required");
  }
  if (trace.require_name_source_explain !== true) {
    errors.push("grounded corpus name-source explain prompt is not required");
  }
  if (trace.require_description_source_image !== true) {
    errors.push("grounded corpus description-source image prompt is not required");
  }
  if (trace.require_image_attribute_generic_prompt !== true) {
    errors.push("grounded corpus image-attribute generic prompt is not required");
  }
  const sourceTasks = Array.isArray(trace.source_text_tasks) ? trace.source_text_tasks : [];
  const attributeTasks = Array.isArray(trace.attribute_tasks) ? trace.attribute_tasks : [];
  const expectedSpirits = expectedGroundedCorpusSpirits(config);
  if (expectedSpirits > 0 && Number(trace.expect_spirits || 0) !== expectedSpirits) {
    errors.push(`grounded corpus expect_spirits ${Number(trace.expect_spirits || 0)} != promoted corpus spirits ${expectedSpirits}`);
  }
  const sourceFloor = Number(config.minGroundedSourceOverlapTokens || 0);
  const attributeFloor = Number(config.minGroundedAttributeSourceOverlapTokens || 0);
  if (Number(trace.min_source_overlap_tokens || 0) < sourceFloor) {
    errors.push(`grounded corpus source overlap floor ${Number(trace.min_source_overlap_tokens || 0)} < ${sourceFloor}`);
  }
  if (Number(trace.min_attribute_source_overlap_tokens || 0) < attributeFloor) {
    errors.push(
      `grounded corpus attribute source overlap floor ${Number(trace.min_attribute_source_overlap_tokens || 0)} < ${attributeFloor}`,
    );
  }
  const sourcePlaceholderCeiling = Number(config.maxGroundedSourcePlaceholderRows || 0);
  const attributeGenericRankCeiling = Number(config.maxGroundedAttributeGenericRankRows || 0);
  const recordedPlaceholderCeiling = trace.max_source_placeholder_rows;
  if (
    recordedPlaceholderCeiling !== undefined &&
    recordedPlaceholderCeiling !== null &&
    Number(recordedPlaceholderCeiling) > sourcePlaceholderCeiling
  ) {
    errors.push(
      `grounded corpus source placeholder ceiling ${Number(recordedPlaceholderCeiling)} > ${sourcePlaceholderCeiling}`,
    );
  }
  const recordedAttributeGenericRankCeiling = trace.max_attribute_generic_rank_rows;
  if (
    recordedAttributeGenericRankCeiling !== undefined &&
    recordedAttributeGenericRankCeiling !== null &&
    Number(recordedAttributeGenericRankCeiling) > attributeGenericRankCeiling
  ) {
    errors.push(
      `grounded corpus attribute generic rank ceiling ${Number(recordedAttributeGenericRankCeiling)} > ${attributeGenericRankCeiling}`,
    );
  }
  const tasks = trace.tasks && typeof trace.tasks === "object" ? trace.tasks : {};
  for (const task of [...sourceTasks, ...attributeTasks]) {
    const stats = tasks[task] || {};
    const taskFloor = attributeTasks.includes(task) ? attributeFloor : sourceFloor;
    if (Number(stats.records || 0) <= 0) {
      errors.push(`grounded corpus task ${task} has no rows`);
    }
    if (Number(trace.expect_spirits || 0) > 0 && Number(stats.spirits || 0) !== Number(trace.expect_spirits || 0)) {
      errors.push(`grounded corpus task ${task} spirits ${Number(stats.spirits || 0)} != ${Number(trace.expect_spirits || 0)}`);
    }
    if (Number(stats.min_source_overlap_tokens || 0) <= 0) {
      errors.push(`grounded corpus task ${task} has no source token overlap`);
    }
    if (Number(stats.min_source_overlap_tokens || 0) < taskFloor) {
      errors.push(`grounded corpus task ${task} source overlap ${Number(stats.min_source_overlap_tokens || 0)} < ${taskFloor}`);
    }
    if (stats.source_provenance_rows === undefined) {
      errors.push(`grounded corpus task ${task} missing source_provenance_rows`);
    } else if (Number(stats.source_provenance_rows || 0) !== Number(stats.records || 0)) {
      errors.push(
        `grounded corpus task ${task} source provenance rows ${Number(stats.source_provenance_rows || 0)} != records ${Number(stats.records || 0)}`,
      );
    }
    if (Number(stats.source_provenance_hash_mismatches || 0) > 0) {
      errors.push(
        `grounded corpus task ${task} source provenance hash mismatches ${Number(stats.source_provenance_hash_mismatches || 0)} > 0`,
      );
    }
    if (Number(stats.source_excerpt_hash_mismatches || 0) > 0) {
      errors.push(
        `grounded corpus task ${task} source excerpt hash mismatches ${Number(stats.source_excerpt_hash_mismatches || 0)} > 0`,
      );
    }
    if (sourceTasks.includes(task) && Number(stats.placeholder_rows || 0) > sourcePlaceholderCeiling) {
      errors.push(
        `grounded corpus task ${task} source placeholder rows ${Number(stats.placeholder_rows || 0)} > ${sourcePlaceholderCeiling}`,
      );
    }
    if (task === "explain" && trace.require_name_source_explain === true) {
      if (stats.name_source_prompt_ok_rows === undefined) {
        errors.push("grounded corpus task explain missing name_source_prompt_ok_rows");
      } else if (Number(stats.name_source_prompt_ok_rows || 0) !== Number(stats.records || 0)) {
        errors.push(
          `grounded corpus task explain name-source prompt rows ${Number(stats.name_source_prompt_ok_rows || 0)} != records ${Number(stats.records || 0)}`,
        );
      }
    }
    if (task === "description-to-image" && trace.require_description_source_image === true) {
      if (stats.description_source_prompt_ok_rows === undefined) {
        errors.push("grounded corpus task description-to-image missing description_source_prompt_ok_rows");
      } else if (Number(stats.description_source_prompt_ok_rows || 0) !== Number(stats.records || 0)) {
        errors.push(
          `grounded corpus task description-to-image description-source prompt rows ${Number(stats.description_source_prompt_ok_rows || 0)} != records ${Number(stats.records || 0)}`,
        );
      }
    }
    if (task === "image-to-attributes" && trace.require_image_attribute_generic_prompt === true) {
      if (stats.image_attribute_prompt_ok_rows === undefined) {
        errors.push("grounded corpus task image-to-attributes missing image_attribute_prompt_ok_rows");
      } else if (Number(stats.image_attribute_prompt_ok_rows || 0) !== Number(stats.records || 0)) {
        errors.push(
          `grounded corpus task image-to-attributes generic attribute prompt rows ${Number(stats.image_attribute_prompt_ok_rows || 0)} != records ${Number(stats.records || 0)}`,
        );
      }
    }
    if (attributeTasks.includes(task)) {
      if (stats.generic_attribute_rank_rows === undefined && attributeGenericRankCeiling === 0) {
        errors.push(`grounded corpus task ${task} missing generic_attribute_rank_rows`);
      } else if (Number(stats.generic_attribute_rank_rows || 0) > attributeGenericRankCeiling) {
        errors.push(
          `grounded corpus task ${task} generic rank rows ${Number(stats.generic_attribute_rank_rows || 0)} > ${attributeGenericRankCeiling}`,
        );
      }
    }
  }
  return {
    ok: errors.length === 0,
    errors,
    present: true,
    examples: trace.examples || "",
    examples_hash: trace.examples_hash || "",
    examples_provenance: examplesProvenance,
    text_index: trace.text_index || "",
    text_index_hash: trace.text_index_hash || "",
    text_index_provenance: textIndexProvenance,
    expect_spirits: Number(trace.expect_spirits || 0),
    source_text_tasks: sourceTasks,
    attribute_tasks: attributeTasks,
    require_source_provenance: trace.require_source_provenance === true,
    require_name_source_explain: trace.require_name_source_explain === true,
    require_description_source_image: trace.require_description_source_image === true,
    require_image_attribute_generic_prompt: trace.require_image_attribute_generic_prompt === true,
    min_source_overlap_tokens: Number(trace.min_source_overlap_tokens || 0),
    min_attribute_source_overlap_tokens: Number(trace.min_attribute_source_overlap_tokens || 0),
    max_source_placeholder_rows: Number(trace.max_source_placeholder_rows ?? sourcePlaceholderCeiling),
    max_attribute_generic_rank_rows: Number(trace.max_attribute_generic_rank_rows ?? attributeGenericRankCeiling),
    tasks,
  };
}

function checkGroundedCorpusExamplesProvenance(trace, filePath, config, errors) {
  const expectedExamples = config.examplesPath || "";
  const summary = {
    examples: trace.examples || "",
    expected_examples: expectedExamples,
    examples_match: null,
    examples_hash: trace.examples_hash || "",
    expected_examples_hash: "",
    examples_hash_match: null,
  };
  if (expectedExamples) {
    if (!summary.examples) {
      errors.push("grounded corpus examples path is missing");
    } else {
      summary.examples_match = sameReferencedPath(summary.examples, expectedExamples, path.dirname(filePath));
      if (summary.examples_match === false) {
        errors.push(`grounded corpus examples ${summary.examples} does not match corpus examples ${expectedExamples}`);
      }
    }
  }
  if (summary.examples_hash && expectedExamples) {
    try {
      summary.expected_examples_hash = fnv64FileHex(path.resolve(expectedExamples));
      summary.examples_hash_match = summary.examples_hash === summary.expected_examples_hash;
      if (!summary.examples_hash_match) {
        errors.push(
          `grounded corpus examples_hash ${summary.examples_hash} does not match corpus examples hash ${summary.expected_examples_hash}`,
        );
      }
    } catch (error) {
      errors.push(`grounded corpus examples_hash could not read corpus examples ${expectedExamples}: ${error.message}`);
    }
  }
  return summary;
}

function expectedGroundedCorpusSpirits(config) {
  if (config.manifestPath) {
    try {
      const manifest = readJson(config.manifestPath);
      const rows = Number(manifest.rows || 0);
      if (Number.isInteger(rows) && rows > 0) {
        return rows;
      }
    } catch (_error) {
      return 0;
    }
  }
  return config.examplesPath ? 72 : 0;
}

function absentGroundedCorpus() {
  return {
    ok: true,
    errors: [],
    present: false,
    examples: "",
    examples_hash: "",
    examples_provenance: {
      examples: "",
      expected_examples: "",
      examples_match: null,
      examples_hash: "",
      expected_examples_hash: "",
      examples_hash_match: null,
    },
    text_index: "",
    text_index_hash: "",
    text_index_provenance: absentSourceTextIndexProvenance(),
    expect_spirits: 0,
    source_text_tasks: [],
    attribute_tasks: [],
    min_source_overlap_tokens: 0,
    min_attribute_source_overlap_tokens: 0,
    max_source_placeholder_rows: 0,
    max_attribute_generic_rank_rows: 0,
    tasks: {},
  };
}

function checkGenerativeEval(inputPath, config, retrievalReport) {
  const errors = [];
  const summaryPath = resolveGenerativeEvalSummaryPath(inputPath);
  const runDir = resolveGenerativeEvalRunDir(inputPath, summaryPath);
  if (!fs.existsSync(summaryPath)) {
    errors.push(`generative eval summary not found: ${summaryPath}`);
    return generativeEvalReport({
      ok: false,
      errors,
      inputPath,
      summaryPath,
      runDir,
      rows: [],
      config,
    });
  }
  let rows = [];
  try {
    rows = readTsv(summaryPath);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  if (rows.length === 0) {
    errors.push(`generative eval summary has no rows: ${summaryPath}`);
  }
  const models = rows.map((row) => summarizeGenerativeEvalRow(row, errors));
  const floor = generativeEvalFloor(models, config);
  if (!floor.ok) {
    errors.push(...floor.errors);
  }
  const evidence = checkGenerativeEvalEvidence(runDir, models, floor, config, errors, retrievalReport);
  return generativeEvalReport({
    ok: errors.length === 0,
    errors,
    inputPath,
    summaryPath,
    runDir,
    rows: models,
    floor,
    evidence,
    config,
  });
}

function absentGenerativeEval() {
  return {
    ok: true,
    errors: [],
    present: false,
    input: "",
    summary: "",
    run_dir: "",
    model_count: 0,
    evidence: absentGenerativeEvalEvidence(),
    product_floor: generativeEvalFloor([], defaults),
    best: null,
    models: [],
  };
}

function resolveGenerativeEvalSummaryPath(inputPath) {
  if (!inputPath) {
    return "";
  }
  if (fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory()) {
    return path.join(inputPath, "summary.tsv");
  }
  if (path.basename(inputPath) === "summary.tsv") {
    return inputPath;
  }
  return path.join(inputPath, "summary.tsv");
}

function resolveGenerativeEvalRunDir(inputPath, summaryPath) {
  if (!inputPath) {
    return "";
  }
  if (fs.existsSync(inputPath) && fs.statSync(inputPath).isDirectory()) {
    return inputPath;
  }
  if (summaryPath && path.basename(summaryPath) === "summary.tsv") {
    return path.dirname(summaryPath);
  }
  return "";
}

function checkGenerativeEvalEvidence(runDir, models, floor, config, errors, retrievalReport) {
  const report = absentGenerativeEvalEvidence();
  report.required = config.requireGenerativeEval === true;
  report.output_identity.required = config.requireGenerativeOutputIdentity === true;
  report.run_dir = runDir || "";
  report.config_path = runDir ? path.join(runDir, "config.json") : "";
  report.samples_path = runDir ? path.join(runDir, "samples.tsv") : "";

  let runConfig = null;
  if (report.config_path && fs.existsSync(report.config_path)) {
    report.config_present = true;
    try {
      runConfig = readJson(report.config_path);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  } else {
    errors.push(`generative eval config not found: ${report.config_path || "<unknown>"}`);
  }
  if (runConfig) {
    report.config = {
      partition: runConfig.partition || "",
      latent_target: runConfig.latentTarget || "",
      sampler_model: runConfig.samplerModel || "",
      sampler_model_hash: runConfig.samplerModelHash || "",
      prompts: runConfig.prompts || "",
      run_name: runConfig.runName || "",
      eval_permille: Number(runConfig.evalPermille || 0),
      limit: Number(runConfig.limit || 0),
      retrieval_head: runConfig.retrievalHead || "",
      retrieval_head_model_hash: runConfig.retrievalHeadModelHash || "",
      retrieval_head_feature_count: Number(runConfig.retrievalHeadFeatureCount || 0),
      latent_model_count: Array.isArray(runConfig.latentModels) ? runConfig.latentModels.length : 0,
      prompts_hash: runConfig.promptsHash || "",
      prompt_rows: Number(runConfig.promptRows || 0),
      selected_prompt_rows: Number(runConfig.selectedPromptRows || 0),
      selected_prompt_eligible_rows: Number(runConfig.selectedPromptEligibleRows || 0),
      selected_prompt_unique_targets: Number(runConfig.selectedPromptUniqueTargets || 0),
      selected_prompt_eligible_unique_targets: Number(runConfig.selectedPromptEligibleUniqueTargets || 0),
      selected_prompt_hash: runConfig.selectedPromptHash || "",
    };
    report.retrieval_head_provenance = checkRetrievalEvidenceHeadProvenance(
      "generative eval",
      report.config.retrieval_head_model_hash,
      retrievalReport,
      errors,
    );
    if (report.config.partition !== "eval") {
      errors.push(`generative eval partition ${JSON.stringify(report.config.partition)} != "eval"`);
    }
    if (report.config.latent_target !== "decoded") {
      errors.push(`generative eval latent_target ${JSON.stringify(report.config.latent_target)} != "decoded"`);
    }
    if (config.minGeneratedPromptRows > 0 && report.config.limit < config.minGeneratedPromptRows) {
      errors.push(`generative eval limit ${report.config.limit} < ${config.minGeneratedPromptRows}`);
    }
  }

  let sampleRows = [];
  if (report.samples_path && fs.existsSync(report.samples_path)) {
    report.samples_present = true;
    try {
      sampleRows = readTsv(report.samples_path);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  } else {
    errors.push(`generative eval samples not found: ${report.samples_path || "<unknown>"}`);
  }
  report.sample_count = sampleRows.length;
  if (runConfig) {
    report.latent_model_provenance = checkGenerativeEvalLatentModelProvenance(runConfig, runDir, models, sampleRows, errors);
    report.prompt_provenance = checkGenerativeEvalPromptProvenance(runConfig, runDir, sampleRows, retrievalReport, config, errors);
    report.sampler_model_provenance = checkGenerativeEvalSamplerModelProvenance(runConfig, runDir, sampleRows, errors);
    report.generated_retrieval_provenance = checkGenerativeEvalGeneratedRetrievalProvenance(
      runConfig,
      runDir,
      models,
      sampleRows,
      retrievalReport,
      config,
      errors,
    );
  }
  for (const row of sampleRows) {
    incrementCount(report.sample_partitions, row.partition || "");
    incrementCount(report.sampler_target_sources, row.sampler_target_source || "");
    incrementCount(report.model_sample_counts, row.model || "");
  }
  report.trace_integrity = summarizeGenerativeTraceIntegrity(runDir, sampleRows, errors);
  report.output_identity = summarizeGenerativeOutputIdentity(sampleRows);
  report.output_identity.required = config.requireGenerativeOutputIdentity === true;
  if (generativeRetrievalEvidenceRequired(config) && !report.config.retrieval_head_model_hash) {
    errors.push("generative eval retrieval evidence requires retrievalHeadModelHash in config.json");
  }
  if (sampleRows.length === 0) {
    errors.push("generative eval samples.tsv has no rows");
  }
  const nonEvalPartitions = Object.keys(report.sample_partitions).filter((partition) => partition !== "eval");
  if (nonEvalPartitions.length > 0) {
    errors.push(`generative eval samples include non-eval partitions: ${nonEvalPartitions.join(",")}`);
  }
  const nonDecodedSources = Object.keys(report.sampler_target_sources).filter((source) => source !== "decoded-latent");
  if (nonDecodedSources.length > 0) {
    errors.push(`generative eval samples include non-decoded latent targets: ${nonDecodedSources.join(",")}`);
  }
  for (const row of sampleRows) {
    if (!row.prompt) {
      errors.push(`generative eval sample row ${row.row_index || "?"} missing prompt`);
      break;
    }
    const spiritId = Number(row.spirit_id || 0);
    if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72) {
      errors.push(`generative eval sample row ${row.row_index || "?"} invalid spirit_id ${JSON.stringify(row.spirit_id)}`);
      break;
    }
  }
  for (const model of models) {
    const count = Number(report.model_sample_counts[model.model] || 0);
    if (count !== Number(model.prompts || 0)) {
      errors.push(`generative eval samples for model ${model.model || "<missing-model>"} ${count} != summary prompts ${model.prompts}`);
    }
  }
  report.heldout_partition_ready =
    report.config_present &&
    report.samples_present &&
    report.config.partition === "eval" &&
    report.sample_count > 0 &&
    Object.keys(report.sample_partitions).every((partition) => partition === "eval") &&
    Object.keys(report.sampler_target_sources).every((source) => source === "decoded-latent") &&
    report.trace_integrity.ok === true;
  if (config.requireGenerativeOutputIdentity) {
    const matchingModel = floor?.matching_model || "";
    const matchingIdentity = report.output_identity.by_model[matchingModel];
    if (!matchingModel) {
      errors.push("generative eval output identity requires a matching product-floor model");
    } else if (!matchingIdentity || matchingIdentity.ok !== true) {
      errors.push(`generative eval output identity is incomplete for matching model ${matchingModel}`);
    }
  }
  return report;
}

function absentGenerativeEvalEvidence() {
  return {
    required: false,
    run_dir: "",
    config_path: "",
    samples_path: "",
    config_present: false,
    samples_present: false,
    sample_count: 0,
    heldout_partition_ready: false,
    config: {
      partition: "",
      latent_target: "",
      sampler_model: "",
      sampler_model_hash: "",
      prompts: "",
      prompts_hash: "",
      prompt_rows: 0,
      selected_prompt_rows: 0,
      selected_prompt_hash: "",
      run_name: "",
      eval_permille: 0,
      limit: 0,
      retrieval_head: "",
      retrieval_head_model_hash: "",
      retrieval_head_feature_count: 0,
      latent_model_count: 0,
    },
    latent_model_provenance: absentGenerativeEvalLatentModelProvenance(),
    sampler_model_provenance: absentGenerativeEvalSamplerModelProvenance(),
    generated_retrieval_provenance: absentGenerativeEvalGeneratedRetrievalProvenance(),
    retrieval_head_provenance: {
      expected_model_hash: "",
      model_hash: "",
      hash_match: null,
    },
    prompt_provenance: absentGenerativeEvalPromptProvenance(),
    sample_partitions: {},
    sampler_target_sources: {},
    model_sample_counts: {},
    trace_integrity: absentGenerativeTraceIntegrity(),
    output_identity: {
      required: false,
      rows: 0,
      scored_rows: 0,
      identity_rows: 0,
      positive_margin_rows: 0,
      min_margin: null,
      ok: false,
      by_model: {},
    },
  };
}

function generativeRetrievalEvidenceRequired(config) {
  return (
    config.requireGenerativeOutputIdentity === true ||
    Number(config.minGeneratedRetrievalTop1PerMille || 0) > 0 ||
    Number(config.minGeneratedRetrievalTop5PerMille || 0) > 0
  );
}

function checkGenerativeEvalLatentModelProvenance(runConfig, runDir, models, sampleRows, errors) {
  const configModels = Array.isArray(runConfig.latentModels) ? runConfig.latentModels : [];
  const configProvenance = Array.isArray(runConfig.latentModelProvenance) ? runConfig.latentModelProvenance : [];
  const configHashes =
    runConfig.latentModelHashes && typeof runConfig.latentModelHashes === "object" && !Array.isArray(runConfig.latentModelHashes)
      ? runConfig.latentModelHashes
      : {};
  const byLabel = new Map();
  for (const spec of configModels) {
    if (spec && spec.label) {
      byLabel.set(String(spec.label), {
        label: String(spec.label),
        path: String(spec.path || ""),
        config_hash: "",
      });
    }
  }
  for (const row of configProvenance) {
    if (!row?.label) continue;
    const label = String(row.label);
    const existing = byLabel.get(label) || { label, path: "", config_hash: "" };
    existing.path = existing.path || String(row.path || "");
    existing.config_hash = existing.config_hash || String(row.modelHash || row.model_hash || "");
    byLabel.set(label, existing);
  }
  for (const [label, hash] of Object.entries(configHashes)) {
    const existing = byLabel.get(label) || { label, path: "", config_hash: "" };
    existing.config_hash = existing.config_hash || String(hash || "");
    byLabel.set(label, existing);
  }

  const summary = absentGenerativeEvalLatentModelProvenance();
  summary.config_model_count = configModels.length;
  summary.config_provenance_count = configProvenance.length;
  summary.summary_model_count = models.length;
  summary.sample_count = sampleRows.length;
  const expectedByLabel = new Map();

  for (const model of models) {
    const label = model.model || "";
    const configured = byLabel.get(label) || null;
    const modelPath = configured?.path || model.latent_model || "";
    const configHash = configured?.config_hash || "";
    const summaryHash = model.latent_model_hash || "";
    const candidateModel = resolveGenerativeReferencedPath(modelPath, runDir);
    const resolvedModel = candidateModel && fs.existsSync(candidateModel) ? candidateModel : "";
    const expectedHash = resolvedModel ? fnv64FileHex(resolvedModel) : "";
    const row = {
      label,
      latent_model: modelPath,
      summary_latent_model: model.latent_model || "",
      candidate_latent_model: candidateModel,
      resolved_latent_model: resolvedModel,
      config_hash: configHash,
      summary_hash: summaryHash,
      expected_hash: expectedHash,
      config_hash_match: configHash && expectedHash ? configHash === expectedHash : null,
      summary_hash_match: summaryHash && expectedHash ? summaryHash === expectedHash : null,
      config_summary_hash_match: configHash && summaryHash ? configHash === summaryHash : null,
    };
    if (!label) {
      errors.push("generative eval latent model row missing label");
    }
    if (!modelPath) {
      summary.missing_model_paths += 1;
      errors.push(`generative eval latent model ${label || "<missing-label>"} path is missing`);
    }
    if (!configured) {
      summary.missing_config_models += 1;
      errors.push(`generative eval latent model ${label || "<missing-label>"} missing from config latentModels`);
    }
    if (!configHash) {
      summary.missing_config_hashes += 1;
      errors.push(`generative eval latent model ${label || "<missing-label>"} config hash is missing`);
    }
    if (!summaryHash) {
      summary.missing_summary_hashes += 1;
      errors.push(`generative eval latent model ${label || "<missing-label>"} summary latent_model_hash is missing`);
    }
    if (!expectedHash) {
      summary.unresolved_models += 1;
      errors.push(`generative eval latent model ${label || "<missing-label>"} ${JSON.stringify(modelPath)} could not be resolved`);
    } else {
      summary.resolved_model_count += 1;
      summary.unique_recomputed_hashes.push(expectedHash);
    }
    if (configHash && expectedHash && configHash !== expectedHash) {
      summary.config_hash_mismatches += 1;
      errors.push(`generative eval latent model ${label} config hash ${configHash} != recomputed ${expectedHash}`);
    }
    if (summaryHash && expectedHash && summaryHash !== expectedHash) {
      summary.summary_hash_mismatches += 1;
      errors.push(`generative eval latent model ${label} summary hash ${summaryHash} != recomputed ${expectedHash}`);
    }
    if (configHash && summaryHash && configHash !== summaryHash) {
      summary.config_summary_hash_mismatches += 1;
      errors.push(`generative eval latent model ${label} config hash ${configHash} != summary hash ${summaryHash}`);
    }
    if (label) {
      expectedByLabel.set(label, {
        label,
        latent_model: modelPath,
        resolved_latent_model: resolvedModel,
        expected_hash: expectedHash,
      });
    }
    summary.models.push(row);
  }

  for (const label of byLabel.keys()) {
    if (!models.some((model) => model.model === label)) {
      summary.unused_config_models += 1;
      errors.push(`generative eval latent model ${label} is in config but missing from summary.tsv`);
    }
  }

  for (const row of sampleRows) {
    const label = row.model || "";
    const expected = expectedByLabel.get(label) || null;
    const outDir = resolveGenerativeSampleOutDir(runDir, row.out_dir || "");
    const tracePath = outDir ? path.join(outDir, "trace.json") : "";
    const traceRecord = {
      row_index: row.row_index || null,
      model: label,
      prompt_hash: row.prompt_hash || "",
      trace: tracePath,
      trace_present: false,
      latent_model: "",
      candidate_latent_model: "",
      resolved_latent_model: "",
      latent_model_hash: "",
      expected_hash: expected?.expected_hash || "",
      hash_match: null,
      ok: false,
    };
    if (!expected) {
      summary.trace_missing_summary_models += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} model ${label || "<missing-model>"} is not in summary.tsv`);
    }
    if (!tracePath || !fs.existsSync(tracePath)) {
      summary.trace_missing_traces += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} missing trace for latent model provenance`);
      summary.trace_models.push(traceRecord);
      continue;
    }
    traceRecord.trace_present = true;
    let trace = null;
    try {
      trace = readJson(tracePath);
    } catch (error) {
      summary.trace_invalid_traces += 1;
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`generative eval sample row ${row.row_index || "?"} latent trace could not be read: ${message}`);
      summary.trace_models.push(traceRecord);
      continue;
    }
    traceRecord.latent_model = trace.latent_model || "";
    traceRecord.candidate_latent_model = traceRecord.latent_model
      ? resolveGenerativeTraceModelReference(traceRecord.latent_model, outDir, runDir)
      : "";
    traceRecord.resolved_latent_model =
      traceRecord.candidate_latent_model && fs.existsSync(traceRecord.candidate_latent_model)
        ? traceRecord.candidate_latent_model
        : "";
    traceRecord.latent_model_hash = traceRecord.resolved_latent_model ? fnv64FileHex(traceRecord.resolved_latent_model) : "";
    traceRecord.hash_match =
      traceRecord.latent_model_hash && expected?.expected_hash
        ? traceRecord.latent_model_hash === expected.expected_hash
        : null;
    if (!traceRecord.latent_model) {
      summary.trace_missing_model_refs += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} trace latent_model is missing`);
    }
    if (!traceRecord.resolved_latent_model) {
      summary.trace_unresolved_models += 1;
      errors.push(
        `generative eval sample row ${row.row_index || "?"} trace latent_model ${JSON.stringify(traceRecord.latent_model)} could not be resolved`,
      );
    } else {
      summary.trace_resolved_model_count += 1;
      summary.trace_latent_model_hashes.push(traceRecord.latent_model_hash);
    }
    if (traceRecord.hash_match === false) {
      summary.trace_hash_mismatches += 1;
      errors.push(
        `generative eval sample row ${row.row_index || "?"} trace latent_model hash ${traceRecord.latent_model_hash} != expected ${expected.expected_hash}`,
      );
    }
    traceRecord.ok = Boolean(
      traceRecord.trace_present &&
        traceRecord.resolved_latent_model &&
        traceRecord.hash_match === true &&
        expected,
    );
    summary.trace_models.push(traceRecord);
  }

  summary.unique_recomputed_hashes = [...new Set(summary.unique_recomputed_hashes)].sort();
  summary.trace_latent_model_hashes = [...new Set(summary.trace_latent_model_hashes)].sort();
  summary.ok =
    summary.summary_model_count > 0 &&
    summary.missing_model_paths === 0 &&
    summary.missing_config_models === 0 &&
    summary.missing_config_hashes === 0 &&
    summary.missing_summary_hashes === 0 &&
    summary.unresolved_models === 0 &&
    summary.config_hash_mismatches === 0 &&
    summary.summary_hash_mismatches === 0 &&
    summary.config_summary_hash_mismatches === 0 &&
    summary.unused_config_models === 0 &&
    summary.trace_missing_summary_models === 0 &&
    summary.trace_missing_traces === 0 &&
    summary.trace_invalid_traces === 0 &&
    summary.trace_missing_model_refs === 0 &&
    summary.trace_unresolved_models === 0 &&
    summary.trace_hash_mismatches === 0 &&
    summary.trace_models.length === sampleRows.length &&
    summary.trace_models.every((trace) => trace.ok === true);
  return summary;
}

function absentGenerativeEvalLatentModelProvenance() {
  return {
    ok: false,
    config_model_count: 0,
    config_provenance_count: 0,
    summary_model_count: 0,
    sample_count: 0,
    resolved_model_count: 0,
    missing_model_paths: 0,
    missing_config_models: 0,
    missing_config_hashes: 0,
    missing_summary_hashes: 0,
    unresolved_models: 0,
    config_hash_mismatches: 0,
    summary_hash_mismatches: 0,
    config_summary_hash_mismatches: 0,
    unused_config_models: 0,
    trace_missing_summary_models: 0,
    trace_missing_traces: 0,
    trace_invalid_traces: 0,
    trace_missing_model_refs: 0,
    trace_unresolved_models: 0,
    trace_resolved_model_count: 0,
    trace_hash_mismatches: 0,
    unique_recomputed_hashes: [],
    trace_latent_model_hashes: [],
    models: [],
    trace_models: [],
  };
}

function checkGenerativeEvalSamplerModelProvenance(runConfig, runDir, sampleRows, errors) {
  const samplerModel = runConfig.samplerModel || "";
  const samplerModelHash = runConfig.samplerModelHash || "";
  const candidateSamplerModel = samplerModel ? resolveGenerativeReferencedPath(samplerModel, runDir) : "";
  const resolvedSamplerModel = candidateSamplerModel && fs.existsSync(candidateSamplerModel) ? candidateSamplerModel : "";
  const expectedSamplerModelHash = resolvedSamplerModel ? fnv64FileHex(resolvedSamplerModel) : "";
  const summary = absentGenerativeEvalSamplerModelProvenance();
  summary.sampler_model = samplerModel;
  summary.candidate_sampler_model = candidateSamplerModel;
  summary.resolved_sampler_model = resolvedSamplerModel;
  summary.sampler_model_hash = samplerModelHash;
  summary.expected_sampler_model_hash = expectedSamplerModelHash;
  summary.sample_count = sampleRows.length;

  if (!samplerModel) {
    errors.push("generative eval samplerModel is missing from config.json");
  }
  if (!samplerModelHash) {
    errors.push("generative eval samplerModelHash is missing from config.json");
  }
  if (!resolvedSamplerModel) {
    errors.push(`generative eval samplerModel ${JSON.stringify(samplerModel)} could not be resolved`);
  }
  if (samplerModelHash && expectedSamplerModelHash) {
    summary.sampler_model_hash_match = samplerModelHash === expectedSamplerModelHash;
    if (!summary.sampler_model_hash_match) {
      errors.push(`generative eval samplerModelHash ${samplerModelHash} != recomputed ${expectedSamplerModelHash}`);
    }
  }

  for (const row of sampleRows) {
    const outDir = resolveGenerativeSampleOutDir(runDir, row.out_dir || "");
    const tracePath = outDir ? path.join(outDir, "trace.json") : "";
    const traceRecord = {
      row_index: row.row_index || null,
      model: row.model || "",
      prompt_hash: row.prompt_hash || "",
      trace: tracePath,
      trace_present: false,
      sampler_model: "",
      candidate_sampler_model: "",
      resolved_sampler_model: "",
      sampler_model_hash: "",
      expected_sampler_model_hash: expectedSamplerModelHash,
      hash_match: null,
      config_hash_match: null,
      model_format: "",
      ok: false,
    };
    if (!tracePath || !fs.existsSync(tracePath)) {
      summary.missing_traces += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} missing trace for sampler model provenance`);
      summary.traces.push(traceRecord);
      continue;
    }
    traceRecord.trace_present = true;
    let trace = null;
    try {
      trace = readJson(tracePath);
    } catch (error) {
      summary.invalid_traces += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} sampler trace could not be read: ${error.message}`);
      summary.traces.push(traceRecord);
      continue;
    }
    traceRecord.model_format = trace.model_format || "";
    traceRecord.sampler_model = trace.model || "";
    traceRecord.candidate_sampler_model = traceRecord.sampler_model
      ? resolveGenerativeTraceModelReference(traceRecord.sampler_model, outDir, runDir)
      : "";
    traceRecord.resolved_sampler_model =
      traceRecord.candidate_sampler_model && fs.existsSync(traceRecord.candidate_sampler_model)
        ? traceRecord.candidate_sampler_model
        : "";
    traceRecord.sampler_model_hash = traceRecord.resolved_sampler_model ? fnv64FileHex(traceRecord.resolved_sampler_model) : "";
    traceRecord.hash_match =
      traceRecord.sampler_model_hash && expectedSamplerModelHash
        ? traceRecord.sampler_model_hash === expectedSamplerModelHash
        : null;
    traceRecord.config_hash_match =
      traceRecord.sampler_model_hash && samplerModelHash ? traceRecord.sampler_model_hash === samplerModelHash : null;
    if (!traceRecord.sampler_model) {
      summary.missing_trace_model_refs += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} trace model is missing`);
    }
    if (!traceRecord.resolved_sampler_model) {
      summary.unresolved_trace_models += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} trace model ${JSON.stringify(traceRecord.sampler_model)} could not be resolved`);
    } else {
      summary.resolved_trace_model_count += 1;
      summary.trace_sampler_model_hashes.push(traceRecord.sampler_model_hash);
    }
    if (traceRecord.model_format !== "NSRLTCH") {
      summary.model_format_mismatches += 1;
      errors.push(`generative eval sample row ${row.row_index || "?"} trace model_format ${JSON.stringify(traceRecord.model_format)} != "NSRLTCH"`);
    }
    if (traceRecord.hash_match === false) {
      summary.trace_hash_mismatches += 1;
      errors.push(
        `generative eval sample row ${row.row_index || "?"} sampler model hash ${traceRecord.sampler_model_hash} != config model hash ${expectedSamplerModelHash}`,
      );
    }
    if (traceRecord.config_hash_match === false) {
      summary.trace_config_hash_mismatches += 1;
      errors.push(
        `generative eval sample row ${row.row_index || "?"} sampler model hash ${traceRecord.sampler_model_hash} != samplerModelHash ${samplerModelHash}`,
      );
    }
    traceRecord.ok = Boolean(
      traceRecord.trace_present &&
      traceRecord.resolved_sampler_model &&
      traceRecord.hash_match === true &&
      traceRecord.config_hash_match === true &&
      traceRecord.model_format === "NSRLTCH",
    );
    summary.traces.push(traceRecord);
  }

  summary.trace_sampler_model_hashes = [...new Set(summary.trace_sampler_model_hashes)].sort();
  if (summary.trace_sampler_model_hashes.length !== (sampleRows.length > 0 ? 1 : 0)) {
    errors.push(`generative eval expected exactly one trace sampler model hash, found ${summary.trace_sampler_model_hashes.length}`);
  }
  summary.ok =
    sampleRows.length > 0 &&
    Boolean(resolvedSamplerModel) &&
    Boolean(samplerModelHash) &&
    summary.sampler_model_hash_match === true &&
    summary.missing_traces === 0 &&
    summary.invalid_traces === 0 &&
    summary.missing_trace_model_refs === 0 &&
    summary.unresolved_trace_models === 0 &&
    summary.model_format_mismatches === 0 &&
    summary.trace_hash_mismatches === 0 &&
    summary.trace_config_hash_mismatches === 0 &&
    summary.trace_sampler_model_hashes.length === 1 &&
    summary.trace_sampler_model_hashes[0] === expectedSamplerModelHash;
  return summary;
}

function absentGenerativeEvalSamplerModelProvenance() {
  return {
    ok: false,
    sampler_model: "",
    candidate_sampler_model: "",
    resolved_sampler_model: "",
    sampler_model_hash: "",
    expected_sampler_model_hash: "",
    sampler_model_hash_match: null,
    sample_count: 0,
    resolved_trace_model_count: 0,
    missing_traces: 0,
    invalid_traces: 0,
    missing_trace_model_refs: 0,
    unresolved_trace_models: 0,
    model_format_mismatches: 0,
    trace_hash_mismatches: 0,
    trace_config_hash_mismatches: 0,
    trace_sampler_model_hashes: [],
    traces: [],
  };
}

function resolveGenerativeTraceModelReference(modelRef, outDir, runDir) {
  if (!modelRef) {
    return "";
  }
  const candidates = path.isAbsolute(modelRef)
    ? [path.resolve(modelRef)]
    : [
        path.resolve(modelRef),
        outDir ? path.resolve(outDir, modelRef) : "",
        runDir ? path.resolve(runDir, modelRef) : "",
      ].filter(Boolean);
  return [...new Set(candidates.map(normalizeReferencedPath))].find((candidate) => fs.existsSync(candidate)) || candidates[0] || "";
}

function checkGenerativeEvalPromptProvenance(runConfig, runDir, sampleRows, retrievalReport, config, errors) {
  const prompts = runConfig.prompts || "";
  const summary = absentGenerativeEvalPromptProvenance();
  summary.min_generated_prompt_rows = Number(config.minGeneratedPromptRows || 0);
  summary.prompts = prompts;
  summary.prompts_hash = runConfig.promptsHash || "";
  summary.prompt_rows = Number(runConfig.promptRows || 0);
  summary.selected_prompt_rows = Number(runConfig.selectedPromptRows || 0);
  summary.selected_prompt_hash = runConfig.selectedPromptHash || "";
  summary.retrieval_prompts_hash = retrievalReport.heldout_prompt_provenance?.prompts_hash || "";

  if (!prompts) {
    errors.push("generative eval config prompts path is missing");
    return summary;
  }

  const resolvedPrompts = resolveGenerativeReferencedPath(prompts, runDir);
  summary.resolved_prompts = resolvedPrompts;
  summary.prompts_present = Boolean(resolvedPrompts && fs.existsSync(resolvedPrompts));
  if (!summary.prompts_present) {
    errors.push(`generative eval prompts ${prompts} could not be resolved`);
    return summary;
  }

  try {
    const promptRows = readGenerativeEvalPromptRows(resolvedPrompts);
    summary.expected_prompts_hash = fnv64FileHex(resolvedPrompts);
    summary.counted_prompt_rows = promptRows.length;
    if (!summary.prompts_hash) {
      errors.push("generative eval config promptsHash is missing");
    } else {
      summary.prompts_hash_match = summary.prompts_hash === summary.expected_prompts_hash;
      if (!summary.prompts_hash_match) {
        errors.push(
          `generative eval promptsHash ${summary.prompts_hash} does not match prompts hash ${summary.expected_prompts_hash}`,
        );
      }
    }
    summary.prompt_rows_match = summary.prompt_rows === summary.counted_prompt_rows;
    if (!summary.prompt_rows_match) {
      errors.push(`generative eval promptRows ${summary.prompt_rows} != prompt file rows ${summary.counted_prompt_rows}`);
    }

    const selected = selectGenerativeEvalPrompts(runConfig, promptRows, readGenerativeEvalGoldHashes(runConfig, runDir));
    const selectedTargets = new Set(selected.map((row) => Number(row.spirit_id || 0)).filter((id) => Number.isInteger(id) && id > 0));
    summary.expected_selected_prompt_rows = selected.length;
    summary.expected_selected_unique_targets = selectedTargets.size;
    const selectedEligible = selected.filter(isGenerativeEvalHeldoutPrompt);
    const selectedEligibleTargets = new Set(
      selectedEligible.map((row) => Number(row.spirit_id || 0)).filter((id) => Number.isInteger(id) && id > 0),
    );
    summary.selected_prompt_eligible_rows_recorded = Object.prototype.hasOwnProperty.call(
      runConfig,
      "selectedPromptEligibleRows",
    );
    summary.selected_prompt_eligible_rows = Number(runConfig.selectedPromptEligibleRows || 0);
    summary.expected_selected_prompt_eligible_rows = selectedEligible.length;
    summary.selected_prompt_eligible_rows_match =
      !summary.selected_prompt_eligible_rows_recorded
        ? null
        : summary.selected_prompt_eligible_rows === summary.expected_selected_prompt_eligible_rows;
    if (summary.selected_prompt_eligible_rows_match === false) {
      errors.push(
        `generative eval selectedPromptEligibleRows ${summary.selected_prompt_eligible_rows} != recomputed ${summary.expected_selected_prompt_eligible_rows}`,
      );
    }
    summary.selected_prompt_unique_targets_recorded = Object.prototype.hasOwnProperty.call(
      runConfig,
      "selectedPromptUniqueTargets",
    );
    summary.selected_prompt_unique_targets = Number(runConfig.selectedPromptUniqueTargets || 0);
    summary.selected_prompt_unique_targets_match =
      !summary.selected_prompt_unique_targets_recorded
        ? null
        : summary.selected_prompt_unique_targets === summary.expected_selected_unique_targets;
    if (summary.selected_prompt_unique_targets_match === false) {
      errors.push(
        `generative eval selectedPromptUniqueTargets ${summary.selected_prompt_unique_targets} != recomputed ${summary.expected_selected_unique_targets}`,
      );
    }
    summary.selected_prompt_eligible_unique_targets_recorded = Object.prototype.hasOwnProperty.call(
      runConfig,
      "selectedPromptEligibleUniqueTargets",
    );
    summary.selected_prompt_eligible_unique_targets = Number(runConfig.selectedPromptEligibleUniqueTargets || 0);
    summary.expected_selected_prompt_eligible_unique_targets = selectedEligibleTargets.size;
    summary.selected_prompt_eligible_unique_targets_match =
      !summary.selected_prompt_eligible_unique_targets_recorded
        ? null
        : summary.selected_prompt_eligible_unique_targets === summary.expected_selected_prompt_eligible_unique_targets;
    if (summary.selected_prompt_eligible_unique_targets_match === false) {
      errors.push(
        `generative eval selectedPromptEligibleUniqueTargets ${summary.selected_prompt_eligible_unique_targets} != recomputed ${summary.expected_selected_prompt_eligible_unique_targets}`,
      );
    }
    summary.missing_selected_targets = missingSolomonTargets(selectedTargets);
    summary.expected_selected_prompt_hash = generativeEvalPromptSelectionHash(selected);
    summary.selected_prompt_rows_match = summary.selected_prompt_rows === summary.expected_selected_prompt_rows;
    if (!summary.selected_prompt_rows_match) {
      errors.push(
        `generative eval selectedPromptRows ${summary.selected_prompt_rows} != recomputed ${summary.expected_selected_prompt_rows}`,
      );
    }
    if (summary.min_generated_prompt_rows > 0) {
      if (summary.selected_prompt_rows < summary.min_generated_prompt_rows) {
        errors.push(
          `generative eval selectedPromptRows ${summary.selected_prompt_rows} < ${summary.min_generated_prompt_rows}`,
        );
      }
      if (summary.expected_selected_prompt_rows < summary.min_generated_prompt_rows) {
        errors.push(
          `generative eval recomputed selected prompts ${summary.expected_selected_prompt_rows} < ${summary.min_generated_prompt_rows}`,
        );
      }
      if (summary.expected_selected_unique_targets < summary.min_generated_prompt_rows) {
        errors.push(
          `generative eval recomputed selected unique targets ${summary.expected_selected_unique_targets} < ${summary.min_generated_prompt_rows}`,
        );
      }
      if (summary.expected_selected_prompt_eligible_rows < summary.min_generated_prompt_rows) {
        errors.push(
          `generative eval recomputed selected eligible prompts ${summary.expected_selected_prompt_eligible_rows} < ${summary.min_generated_prompt_rows}`,
        );
      }
      if (summary.expected_selected_prompt_eligible_unique_targets < summary.min_generated_prompt_rows) {
        errors.push(
          `generative eval recomputed selected eligible unique targets ${summary.expected_selected_prompt_eligible_unique_targets} < ${summary.min_generated_prompt_rows}`,
        );
      }
      if (!summary.selected_prompt_eligible_rows_recorded) {
        errors.push("generative eval config selectedPromptEligibleRows is missing");
      }
      if (!summary.selected_prompt_unique_targets_recorded) {
        errors.push("generative eval config selectedPromptUniqueTargets is missing");
      }
      if (!summary.selected_prompt_eligible_unique_targets_recorded) {
        errors.push("generative eval config selectedPromptEligibleUniqueTargets is missing");
      }
    }
    if (!summary.selected_prompt_hash) {
      errors.push("generative eval config selectedPromptHash is missing");
    } else {
      summary.selected_prompt_hash_match = summary.selected_prompt_hash === summary.expected_selected_prompt_hash;
      if (!summary.selected_prompt_hash_match) {
        errors.push(
          `generative eval selectedPromptHash ${summary.selected_prompt_hash} != recomputed ${summary.expected_selected_prompt_hash}`,
        );
      }
    }

    summary.sample_prompt_hashes_by_model = generativeEvalSamplePromptHashesByModel(sampleRows);
    summary.sample_unique_targets_by_model = generativeEvalSampleUniqueTargetsByModel(sampleRows);
    summary.sample_missing_targets_by_model = Object.fromEntries(
      Object.entries(summary.sample_unique_targets_by_model).map(([model, uniqueTargets]) => [
        model,
        uniqueTargets >= summary.min_generated_prompt_rows
          ? []
          : missingSolomonTargets(
              new Set(
                sampleRows
                  .filter((row) => (row.model || "") === model)
                  .map((row) => Number(row.spirit_id || 0))
                  .filter((id) => Number.isInteger(id) && id > 0),
              ),
            ),
      ]),
    );
    if (summary.min_generated_prompt_rows > 0) {
      for (const [model, uniqueTargets] of Object.entries(summary.sample_unique_targets_by_model)) {
        if (uniqueTargets < summary.min_generated_prompt_rows) {
          errors.push(
            `generative eval samples for model ${model || "<missing-model>"} cover ${uniqueTargets} unique targets < ${summary.min_generated_prompt_rows}`,
          );
        }
      }
    }
    summary.sample_prompt_sets_match = Object.values(summary.sample_prompt_hashes_by_model).every(
      (hash) => hash === summary.expected_selected_prompt_hash,
    );
    if (sampleRows.length > 0 && !summary.sample_prompt_sets_match) {
      errors.push("generative eval samples.tsv prompt set does not match selected prompts");
    }
  } catch (error) {
    errors.push(`generative eval prompt provenance could not be verified: ${error.message}`);
  }

  if (summary.retrieval_prompts_hash) {
    summary.retrieval_prompts_hash_match = summary.prompts_hash === summary.retrieval_prompts_hash;
    if (!summary.retrieval_prompts_hash_match) {
      errors.push(
        `generative eval promptsHash ${summary.prompts_hash || ""} != retrieval held-out prompts hash ${summary.retrieval_prompts_hash}`,
      );
    }
  }

  return summary;
}

function absentGenerativeEvalPromptProvenance() {
  return {
    prompts: "",
    resolved_prompts: "",
    prompts_present: false,
    prompts_hash: "",
    expected_prompts_hash: "",
    prompts_hash_match: null,
    prompt_rows: 0,
    counted_prompt_rows: 0,
    prompt_rows_match: null,
    min_generated_prompt_rows: 0,
    selected_prompt_rows: 0,
    expected_selected_prompt_rows: 0,
    expected_selected_unique_targets: 0,
    selected_prompt_eligible_rows_recorded: false,
    selected_prompt_eligible_rows: 0,
    expected_selected_prompt_eligible_rows: 0,
    selected_prompt_eligible_rows_match: null,
    selected_prompt_unique_targets_recorded: false,
    selected_prompt_unique_targets: 0,
    selected_prompt_unique_targets_match: null,
    selected_prompt_eligible_unique_targets_recorded: false,
    selected_prompt_eligible_unique_targets: 0,
    expected_selected_prompt_eligible_unique_targets: 0,
    selected_prompt_eligible_unique_targets_match: null,
    missing_selected_targets: [],
    selected_prompt_rows_match: null,
    selected_prompt_hash: "",
    expected_selected_prompt_hash: "",
    selected_prompt_hash_match: null,
    retrieval_prompts_hash: "",
    retrieval_prompts_hash_match: null,
    sample_prompt_hashes_by_model: {},
    sample_unique_targets_by_model: {},
    sample_missing_targets_by_model: {},
    sample_prompt_sets_match: null,
  };
}

function resolveGenerativeReferencedPath(ref, runDir) {
  const candidates = referencedPathCandidates(ref, runDir || process.cwd());
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0] || "";
}

function readGenerativeEvalPromptRows(filePath) {
  return readJsonl(filePath).map((row, index) => ({ ...row, index }));
}

function readGenerativeEvalGoldHashes(runConfig, runDir) {
  const gold = runConfig.gold || "";
  if (!gold) {
    return new Set();
  }
  const resolvedGold = resolveGenerativeReferencedPath(gold, runDir);
  if (!resolvedGold || !fs.existsSync(resolvedGold)) {
    return new Set();
  }
  const hashes = new Set();
  for (const line of fs.readFileSync(resolvedGold, "utf8").split(/\r?\n/)) {
    const first = line.trim().split("\t")[0];
    if (!first || first === "prompt_hash" || first.startsWith("#")) {
      continue;
    }
    hashes.add(first.toLowerCase());
  }
  return hashes;
}

function selectGenerativeEvalPrompts(runConfig, promptRows, goldHashes) {
  const selectionConfig = {
    partition: runConfig.partition || "eval",
    limit: Number(runConfig.limit || 0),
    splitSeed: runConfig.splitSeed || "solomon-prompt-split-v1",
    evalPermille: Number(runConfig.evalPermille || 180),
  };
  const candidates = generativeEvalSelectionCandidates(selectionConfig, promptRows, goldHashes);
  return balancedGenerativeEvalPromptSelection(candidates, selectionConfig.limit);
}

function generativeEvalSelectionCandidates(config, promptRows, goldHashes) {
  const mapped = promptRows.map((prompt) => ({
    ...prompt,
    partition: generativeEvalPromptPartition(prompt, config, goldHashes),
  }));
  const candidates = mapped.filter((prompt) => config.partition === "all" || prompt.partition === config.partition);
  if (config.partition === "eval") {
    const eligible = mapped.filter((prompt) => prompt.partition === "eval" && isGenerativeEvalHeldoutPrompt(prompt));
    const requiredTargets = Math.min(Number(config.limit || 0), 72);
    if (generativeEvalUniquePromptTargets(eligible) >= requiredTargets) {
      return sortGenerativeEvalPromptsForSelection(eligible);
    }
  }
  return sortGenerativeEvalPromptsForSelection(candidates);
}

function sortGenerativeEvalPromptsForSelection(prompts) {
  return [...prompts].sort((left, right) => {
    const leftKey = `${left.tier}:${left.prompt_hash}`;
    const rightKey = `${right.tier}:${right.prompt_hash}`;
    return leftKey.localeCompare(rightKey);
  });
}

function balancedGenerativeEvalPromptSelection(candidates, limit) {
  const byTier = new Map();
  for (const prompt of candidates) {
    const tier = prompt.tier || "";
    if (!byTier.has(tier)) {
      byTier.set(tier, []);
    }
    byTier.get(tier).push(prompt);
  }
  const tiers = [...byTier.keys()].sort();
  const offsets = new Map(tiers.map((tier) => [tier, 0]));
  const selected = [];
  const usedTargets = new Set();
  while (selected.length < limit) {
    let advanced = false;
    for (const tier of tiers) {
      const group = byTier.get(tier);
      let offset = offsets.get(tier);
      while (offset < group.length && usedTargets.has(group[offset].spirit_id)) {
        offset += 1;
      }
      offsets.set(tier, offset);
      if (offset >= group.length) {
        continue;
      }
      const prompt = group[offset];
      selected.push(prompt);
      usedTargets.add(prompt.spirit_id);
      offsets.set(tier, offset + 1);
      advanced = true;
      if (selected.length >= limit) {
        break;
      }
    }
    if (!advanced) {
      break;
    }
  }
  return selected;
}

function isGenerativeEvalHeldoutPrompt(prompt) {
  const tier = String(prompt.tier || "").toLowerCase();
  const source = String(prompt.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function generativeEvalUniquePromptTargets(prompts) {
  return new Set(
    prompts
      .map((prompt) => Number(prompt.spirit_id || 0))
      .filter((id) => Number.isInteger(id) && id > 0),
  ).size;
}

function generativeEvalPromptPartition(prompt, config, goldHashes) {
  if (goldHashes.has(String(prompt.prompt_hash).toLowerCase())) {
    return "gold";
  }
  if (isGenerativeEvalHeldoutPrompt(prompt)) {
    return "eval";
  }
  const bucket = prompt.tier === "tier-cluster-holdout"
    ? hashParts32([config.splitSeed, "cluster", prompt.cluster]) % 1000
    : Number(prompt.bucket);
  return bucket < config.evalPermille ? "eval" : "train";
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

function generativeEvalSamplePromptHashesByModel(sampleRows) {
  const byModel = new Map();
  for (const row of sampleRows) {
    const model = row.model || "";
    if (!byModel.has(model)) {
      byModel.set(model, []);
    }
    byModel.get(model).push(row);
  }
  return Object.fromEntries(
    [...byModel.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([model, rows]) => [
      model,
      generativeEvalPromptSelectionHash(rows),
    ]),
  );
}

function generativeEvalSampleUniqueTargetsByModel(sampleRows) {
  const byModel = new Map();
  for (const row of sampleRows) {
    const model = row.model || "";
    if (!byModel.has(model)) {
      byModel.set(model, new Set());
    }
    const spiritId = Number(row.spirit_id || 0);
    if (Number.isInteger(spiritId) && spiritId > 0) {
      byModel.get(model).add(spiritId);
    }
  }
  return Object.fromEntries(
    [...byModel.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([model, targets]) => [
      model,
      targets.size,
    ]),
  );
}

function missingSolomonTargets(targets) {
  return Array.from({ length: 72 }, (_value, index) => index + 1).filter((id) => !targets.has(id));
}

function hashParts32(parts) {
  let hash = 2166136261 >>> 0;
  for (const part of parts) {
    for (const byte of Buffer.from(String(part))) {
      hash ^= byte;
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    hash ^= 255;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return (hash | 1) >>> 0;
}

function absentGenerativeTraceIntegrity() {
  return {
    ok: false,
    expected_latent_target_source: "decoded-latent",
    sample_count: 0,
    trace_count: 0,
    missing_traces: 0,
    invalid_traces: 0,
    missing_raw_sample_refs: 0,
    raw_sample_path_mismatches: 0,
    missing_raw_samples: 0,
    nondecoded_traces: 0,
    violation_count: 0,
    traces: [],
  };
}

function summarizeGenerativeTraceIntegrity(runDir, sampleRows, errors) {
  const summary = absentGenerativeTraceIntegrity();
  summary.sample_count = sampleRows.length;
  for (const row of sampleRows) {
    const record = checkGenerativeSampleTrace(runDir, row);
    summary.traces.push(record);
    if (record.trace_present) summary.trace_count += 1;
    if (record.missing_trace) summary.missing_traces += 1;
    if (record.invalid_trace) summary.invalid_traces += 1;
    if (record.trace_present && !record.invalid_trace && record.raw_samples_ref_present !== true) {
      summary.missing_raw_sample_refs += 1;
    }
    if (
      record.trace_present &&
      !record.invalid_trace &&
      record.raw_samples_ref_present === true &&
      record.raw_samples_path_match !== true
    ) {
      summary.raw_sample_path_mismatches += 1;
    }
    if (record.trace_present && !record.invalid_trace && record.raw_samples_present !== true) {
      summary.missing_raw_samples += 1;
    }
    if (record.trace_present && !record.invalid_trace && record.latent_target_source !== "decoded-latent") {
      summary.nondecoded_traces += 1;
    }
    summary.violation_count += record.violations.length;
    if (record.violations.length > 0) {
      for (const violation of record.violations) {
        errors.push(
          `generative eval sample row ${record.row_index || "?"} trace ${violation.field}: ${violation.reason}`,
        );
      }
    }
  }
  summary.ok =
    sampleRows.length > 0 &&
    summary.trace_count === sampleRows.length &&
    summary.missing_traces === 0 &&
    summary.invalid_traces === 0 &&
    summary.missing_raw_sample_refs === 0 &&
    summary.raw_sample_path_mismatches === 0 &&
    summary.missing_raw_samples === 0 &&
    summary.nondecoded_traces === 0 &&
    summary.violation_count === 0;
  return summary;
}

function checkGenerativeSampleTrace(runDir, row) {
  const outDir = resolveGenerativeSampleOutDir(runDir, row.out_dir || "");
  const tracePath = outDir ? path.join(outDir, "trace.json") : "";
  const record = {
    row_index: row.row_index || null,
    model: row.model || "",
    prompt_hash: row.prompt_hash || "",
    spirit_id: Number(row.spirit_id || 0),
    out_dir: outDir,
    trace: tracePath,
    trace_present: false,
    missing_trace: false,
    invalid_trace: false,
    latent_target_source: "",
    raw_samples_ref: "",
    raw_samples_ref_present: false,
    raw_samples: "",
    expected_raw_samples: "",
    raw_samples_path_match: false,
    raw_samples_present: false,
    violations: [],
    ok: false,
  };
  if (!outDir) {
    record.missing_trace = true;
    record.violations.push({
      field: "out_dir",
      reason: "missing generative eval sample out_dir",
    });
    return finishGenerativeTraceRecord(record);
  }
  if (!fs.existsSync(tracePath)) {
    record.missing_trace = true;
    record.violations.push({
      field: "trace.json",
      reason: "missing generative eval sample trace",
    });
    return finishGenerativeTraceRecord(record);
  }
  record.trace_present = true;
  let trace = null;
  try {
    trace = readJson(tracePath);
  } catch (error) {
    record.invalid_trace = true;
    record.violations.push({
      field: "trace.json",
      reason: error instanceof Error ? error.message : String(error),
    });
    return finishGenerativeTraceRecord(record);
  }
  if (!trace || typeof trace !== "object" || Array.isArray(trace)) {
    record.invalid_trace = true;
    record.violations.push({
      field: "trace.json",
      reason: "trace is not a JSON object",
    });
    return finishGenerativeTraceRecord(record);
  }
  record.latent_target_source = typeof trace.latent_target_source === "string" ? trace.latent_target_source : "";
  if (trace.schema !== "nsrl.bitmap_sampler_trace.v1") {
    record.violations.push({
      field: "schema",
      reason: `unexpected trace schema ${JSON.stringify(trace.schema || "")}`,
    });
  }
  if (record.latent_target_source !== "decoded-latent") {
    record.violations.push({
      field: "latent_target_source",
      reason: `expected "decoded-latent", got ${JSON.stringify(record.latent_target_source)}`,
    });
  }
  const rawSamples = resolveGenerativeRawSamples(outDir, trace);
  record.raw_samples_ref = rawSamples.source;
  record.raw_samples_ref_present = rawSamples.source_present;
  record.raw_samples = rawSamples.path;
  record.expected_raw_samples = rawSamples.expected_path;
  record.raw_samples_path_match = rawSamples.path_match;
  record.raw_samples_present = rawSamples.present;
  if (!record.raw_samples_ref_present) {
    record.violations.push({
      field: "raw_samples",
      reason: "missing generated raw sample reference",
    });
  }
  if (record.raw_samples_ref_present && !record.raw_samples_path_match) {
    record.violations.push({
      field: "raw_samples",
      reason: `raw_samples must resolve to ${rawSamples.expected_name} in the sample out_dir`,
    });
  }
  if (!record.raw_samples_present) {
    record.violations.push({
      field: "raw_samples",
      reason: "missing generated raw sample bytes",
    });
  }
  scanGenerativeTraceObject(trace, [], record);
  return finishGenerativeTraceRecord(record);
}

function finishGenerativeTraceRecord(record) {
  record.ok = record.violations.length === 0;
  return record;
}

function resolveGenerativeSampleOutDir(runDir, outDir) {
  if (!outDir) {
    return "";
  }
  if (path.isAbsolute(outDir)) {
    return outDir;
  }
  const candidates = [
    path.resolve(outDir),
    runDir ? path.resolve(runDir, outDir) : "",
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (fs.existsSync(path.join(candidate, "trace.json"))) {
      return candidate;
    }
  }
  return candidates[0] || "";
}

function resolveGenerativeRawSamples(outDir, trace) {
  const imageSize = positiveInteger(trace.image_size) || GENERATED_RETRIEVAL_IMAGE_SIZE;
  const expectedName = `samples.ink${imageSize}.u8`;
  const expectedPath = path.resolve(outDir, expectedName);
  const rawSamples = typeof trace.raw_samples === "string" ? trace.raw_samples : "";
  const candidates = rawSamples ? rawSampleReferenceCandidates(rawSamples, outDir) : [];
  const matched = candidates.find((candidate) => sameResolvedPath(candidate, expectedPath)) || "";
  const resolvedPath = matched || candidates[0] || expectedPath;
  return {
    source: rawSamples,
    source_present: rawSamples.length > 0,
    path: resolvedPath,
    expected_path: expectedPath,
    expected_name: expectedName,
    path_match: Boolean(matched),
    present: Boolean(matched && fs.existsSync(matched)),
  };
}

function rawSampleReferenceCandidates(reference, sampleDir) {
  const candidates = [];
  if (path.isAbsolute(reference)) {
    candidates.push(path.resolve(reference));
  } else {
    candidates.push(path.resolve(sampleDir, reference));
    candidates.push(path.resolve(reference));
  }
  return [...new Set(candidates)];
}

function sameResolvedPath(left, right) {
  return path.resolve(left) === path.resolve(right);
}

function positiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : 0;
}

function scanGenerativeTraceObject(value, keyPath, record) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      scanGenerativeTraceObject(value[index], keyPath.concat(String(index)), record);
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    const field = nextPath.join(".");
    if (isForbiddenGenerativeTraceKey(key) && !GENERATION_TRACE_ALLOWED_TARGET_KEYS.has(key)) {
      record.violations.push({
        field,
        reason: "forbidden target-pixel, oracle, guidance, or cleanup field",
      });
    }
    if (typeof child === "string") {
      const reason = forbiddenGenerativeTraceValueReason(key, child);
      if (reason) {
        record.violations.push({
          field,
          reason,
        });
      }
    }
    scanGenerativeTraceObject(child, nextPath, record);
  }
}

function isForbiddenGenerativeTraceKey(key) {
  return GENERATION_TRACE_FORBIDDEN_KEY_PATTERNS.some((pattern) => pattern.test(key));
}

function forbiddenGenerativeTraceValueReason(key, value) {
  if (isGenerativePathLikeKey(key) || isGenerativeFreeTextValueKey(key)) {
    return "";
  }
  if (GENERATION_TRACE_BROAD_FORBIDDEN_VALUE.test(value)) {
    return "forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value";
  }
  if (isGenerativeSourceLikeKey(key) && GENERATION_TRACE_SOURCE_FORBIDDEN_VALUE.test(value)) {
    return "forbidden generation source value";
  }
  return "";
}

function isGenerativeFreeTextValueKey(key) {
  return GENERATION_TRACE_FREE_TEXT_VALUE_KEYS.has(key);
}

function isGenerativePathLikeKey(key) {
  return /(path|file|dir|raw[_-]?samples|preview|pgm|model)$/i.test(key);
}

function isGenerativeSourceLikeKey(key) {
  return /(source|mode|policy|method|strategy|guidance|cleanup|post[_-]?process|postprocess)$/i.test(key);
}

function summarizeGenerativeOutputIdentity(sampleRows) {
  const summary = {
    required: false,
    rows: sampleRows.length,
    scored_rows: 0,
    identity_rows: 0,
    positive_margin_rows: 0,
    min_margin: null,
    ok: false,
    by_model: {},
  };
  for (const row of sampleRows) {
    const model = row.model || "";
    if (!summary.by_model[model]) {
      summary.by_model[model] = {
        rows: 0,
        scored_rows: 0,
        identity_rows: 0,
        positive_margin_rows: 0,
        min_margin: null,
        ok: false,
      };
    }
    const modelSummary = summary.by_model[model];
    modelSummary.rows += 1;
    const identity = Number(row.generated_retrieval_identity);
    const rank = Number(row.generated_retrieval_rank);
    const margin = finiteNumberOrNull(row.generated_retrieval_margin);
    const hasScore =
      row.generated_retrieval_identity !== undefined &&
      row.generated_retrieval_identity !== "" &&
      Number.isFinite(identity);
    if (hasScore) {
      summary.scored_rows += 1;
      modelSummary.scored_rows += 1;
    }
    if (identity === 1) {
      summary.identity_rows += 1;
      modelSummary.identity_rows += 1;
    }
    if (margin !== null && margin > 0) {
      summary.positive_margin_rows += 1;
      modelSummary.positive_margin_rows += 1;
      summary.min_margin = summary.min_margin === null ? margin : Math.min(summary.min_margin, margin);
      modelSummary.min_margin = modelSummary.min_margin === null ? margin : Math.min(modelSummary.min_margin, margin);
    }
  }
  for (const modelSummary of Object.values(summary.by_model)) {
    modelSummary.ok =
      modelSummary.rows > 0 &&
      modelSummary.scored_rows === modelSummary.rows &&
      modelSummary.identity_rows === modelSummary.rows &&
      modelSummary.positive_margin_rows === modelSummary.rows;
  }
  summary.ok =
    summary.rows > 0 &&
    summary.scored_rows === summary.rows &&
    summary.identity_rows === summary.rows &&
    summary.positive_margin_rows === summary.rows;
  return summary;
}

function checkGenerativeEvalGeneratedRetrievalProvenance(
  runConfig,
  runDir,
  models,
  sampleRows,
  retrievalReport,
  config,
  errors,
) {
  const summary = absentGenerativeEvalGeneratedRetrievalProvenance();
  summary.sample_count = sampleRows.length;
  summary.summary_model_count = models.length;
  summary.required =
    generativeRetrievalEvidenceRequired(config) ||
    models.some((model) => model.generated_retrieval?.present === true) ||
    sampleRows.some(generativeSampleRowHasRetrievalScore) ||
    Boolean(runConfig.retrievalHeadModelHash);
  if (!summary.required) {
    summary.ok = true;
    return summary;
  }

  summary.config_retrieval_head = runConfig.retrievalHead || "";
  summary.config_retrieval_head_model_hash = runConfig.retrievalHeadModelHash || "";
  summary.expected_retrieval_head_model_hash = expectedRetrievalHeadModelHash(retrievalReport);
  summary.config_hash_match =
    summary.config_retrieval_head_model_hash && summary.expected_retrieval_head_model_hash
      ? summary.config_retrieval_head_model_hash === summary.expected_retrieval_head_model_hash
      : null;
  if (!summary.config_retrieval_head_model_hash) {
    errors.push("generative eval generated retrieval provenance requires retrievalHeadModelHash in config.json");
  } else if (summary.config_hash_match === false) {
    errors.push(
      `generative eval retrievalHeadModelHash ${summary.config_retrieval_head_model_hash} != retrieval head eval model_hash ${summary.expected_retrieval_head_model_hash}`,
    );
  }

  const headPath = resolveGenerativeRetrievalHeadPath(runConfig, runDir, retrievalReport, config);
  summary.retrieval_head = headPath.source;
  summary.resolved_retrieval_head = headPath.path;
  summary.retrieval_head_present = headPath.exists;
  if (!headPath.exists) {
    errors.push(`generative eval generated retrieval head ${headPath.source || "<missing>"} could not be resolved`);
    return summary;
  }

  let retrievalHead = null;
  try {
    retrievalHead = readGenerativeRetrievalHead(headPath.path);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    summary.invalid_retrieval_head = true;
    errors.push(`generative eval generated retrieval head could not be read: ${message}`);
    return summary;
  }
  summary.retrieval_head_model_hash = retrievalHead.model_hash || "";
  summary.retrieval_head_feature_count = Number(retrievalHead.feature_count || 0);
  summary.retrieval_head_label_count = Array.isArray(retrievalHead.labels) ? retrievalHead.labels.length : 0;
  const recomputedHeadHash = recomputeRetrievalHeadHash(retrievalHead.raw);
  summary.recomputed_retrieval_head_model_hash = recomputedHeadHash;
  summary.retrieval_head_hash_verified = Boolean(summary.retrieval_head_model_hash) && summary.retrieval_head_model_hash === recomputedHeadHash;
  if (!summary.retrieval_head_hash_verified) {
    errors.push(`generative eval retrieval head model_hash ${summary.retrieval_head_model_hash || ""} != recomputed ${recomputedHeadHash}`);
  }
  summary.retrieval_head_hash_match =
    summary.retrieval_head_model_hash && summary.expected_retrieval_head_model_hash
      ? summary.retrieval_head_model_hash === summary.expected_retrieval_head_model_hash
      : null;
  if (summary.retrieval_head_hash_match === false) {
    errors.push(
      `generative eval retrieval head model_hash ${summary.retrieval_head_model_hash} != retrieval head eval model_hash ${summary.expected_retrieval_head_model_hash}`,
    );
  }

  const rowsByModel = new Map();
  for (const row of sampleRows) {
    const label = row.model || "";
    if (!rowsByModel.has(label)) rowsByModel.set(label, []);
    rowsByModel.get(label).push(row);
  }

  for (const row of sampleRows) {
    const record = recomputeGenerativeSampleRetrieval(row, runDir, retrievalHead, errors);
    summary.samples.push(record);
    if (!record.raw_samples_present) summary.missing_raw_samples += 1;
    if (record.invalid_raw_samples) summary.invalid_raw_samples += 1;
    if (record.scored) summary.scored_rows += 1;
    if (record.missing_score_fields.length > 0) summary.rows_with_missing_scores += 1;
    if (record.rank_match === false) summary.rank_mismatches += 1;
    if (record.identity_match === false) summary.identity_mismatches += 1;
    if (record.margin_match === false) summary.margin_mismatches += 1;
    if (record.top1_spirit_match === false) summary.top1_spirit_mismatches += 1;
    if (record.top1_name_match === false) summary.top1_name_mismatches += 1;
  }

  for (const model of models) {
    const modelRows = rowsByModel.get(model.model || "") || [];
    const scoredRows = summary.samples.filter((sample) => sample.model === (model.model || "") && sample.scored);
    const recomputed = summarizeRecomputedGeneratedRetrievalRows(scoredRows, modelRows.length);
    const summaryRow = model.generated_retrieval || {};
    const modelRecord = {
      model: model.model || "",
      rows: modelRows.length,
      scored_rows: scoredRows.length,
      recomputed,
      summary: {
        present: summaryRow.present === true,
        top1: summaryRow.top1,
        top5: summaryRow.top5,
        top1_per_mille: summaryRow.top1_per_mille,
        top5_per_mille: summaryRow.top5_per_mille,
        mean_rank_q8: summaryRow.mean_rank_q8,
        min_margin: summaryRow.min_margin,
      },
      mismatches: [],
      ok: false,
    };
    compareGeneratedRetrievalSummaryField(modelRecord, "top1");
    compareGeneratedRetrievalSummaryField(modelRecord, "top5");
    compareGeneratedRetrievalSummaryField(modelRecord, "top1_per_mille");
    compareGeneratedRetrievalSummaryField(modelRecord, "top5_per_mille");
    compareGeneratedRetrievalSummaryField(modelRecord, "mean_rank_q8");
    compareGeneratedRetrievalSummaryField(modelRecord, "min_margin");
    if (modelRecord.mismatches.length > 0) {
      summary.summary_mismatches += modelRecord.mismatches.length;
      errors.push(
        `generative eval model ${modelRecord.model || "<missing-model>"} generated retrieval summary mismatches: ${modelRecord.mismatches.join(",")}`,
      );
    }
    modelRecord.ok = modelRecord.rows > 0 && modelRecord.scored_rows === modelRecord.rows && modelRecord.mismatches.length === 0;
    summary.by_model[modelRecord.model] = modelRecord;
  }

  summary.ok =
    summary.required === true &&
    summary.retrieval_head_present === true &&
    summary.invalid_retrieval_head === false &&
    summary.retrieval_head_hash_verified === true &&
    summary.retrieval_head_hash_match !== false &&
    summary.config_hash_match !== false &&
    Boolean(summary.config_retrieval_head_model_hash) &&
    summary.sample_count > 0 &&
    summary.scored_rows === summary.sample_count &&
    summary.missing_raw_samples === 0 &&
    summary.invalid_raw_samples === 0 &&
    summary.rows_with_missing_scores === 0 &&
    summary.rank_mismatches === 0 &&
    summary.identity_mismatches === 0 &&
    summary.margin_mismatches === 0 &&
    summary.top1_spirit_mismatches === 0 &&
    summary.top1_name_mismatches === 0 &&
    summary.summary_mismatches === 0;
  return summary;
}

function absentGenerativeEvalGeneratedRetrievalProvenance() {
  return {
    required: false,
    ok: false,
    sample_count: 0,
    summary_model_count: 0,
    config_retrieval_head: "",
    config_retrieval_head_model_hash: "",
    expected_retrieval_head_model_hash: "",
    config_hash_match: null,
    retrieval_head: "",
    resolved_retrieval_head: "",
    retrieval_head_present: false,
    invalid_retrieval_head: false,
    retrieval_head_model_hash: "",
    recomputed_retrieval_head_model_hash: "",
    retrieval_head_hash_verified: false,
    retrieval_head_hash_match: null,
    retrieval_head_feature_count: 0,
    retrieval_head_label_count: 0,
    scored_rows: 0,
    missing_raw_samples: 0,
    invalid_raw_samples: 0,
    rows_with_missing_scores: 0,
    rank_mismatches: 0,
    identity_mismatches: 0,
    margin_mismatches: 0,
    top1_spirit_mismatches: 0,
    top1_name_mismatches: 0,
    summary_mismatches: 0,
    by_model: {},
    samples: [],
  };
}

function generativeSampleRowHasRetrievalScore(row) {
  return [
    "generated_retrieval_rank",
    "generated_retrieval_margin",
    "generated_retrieval_top1_spirit_id",
    "generated_retrieval_top1_name",
    "generated_retrieval_identity",
  ].some((key) => row[key] !== undefined && row[key] !== "");
}

function resolveGenerativeRetrievalHeadPath(runConfig, runDir, retrievalReport, config) {
  const sources = [
    config.retrievalHeadPath || "",
    retrievalReport?.class_retrieval_head?.path || "",
    retrievalReport?.model || "",
    runConfig.retrievalHead || "",
  ].filter(Boolean);
  for (const source of sources) {
    const candidates = path.isAbsolute(source)
      ? [source]
      : [
          path.resolve(source),
          runDir ? path.resolve(runDir, source) : "",
        ].filter(Boolean);
    for (const candidate of candidates) {
      if (fs.existsSync(candidate)) {
        return { source, path: candidate, exists: true };
      }
    }
  }
  const fallback = sources[0] || "";
  return {
    source: fallback,
    path: fallback ? (path.isAbsolute(fallback) ? fallback : path.resolve(fallback)) : "",
    exists: false,
  };
}

function readGenerativeRetrievalHead(filePath) {
  const raw = readJson(filePath);
  if (raw.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(raw.schema)}`);
  }
  validateGenerativeRetrievalHeadLabels(filePath, raw.labels);
  return {
    raw,
    model_hash: String(raw.model_hash || ""),
    feature_count: Number(raw.feature_count || 0),
    labels: Array.isArray(raw.labels) ? raw.labels : [],
    image_head: hydrateGenerativeRetrievalHead(raw.image_head),
  };
}

function validateGenerativeRetrievalHeadLabels(filePath, labels) {
  if (!Array.isArray(labels) || labels.length !== 72) {
    throw new Error(`${filePath} retrieval head labels ${Array.isArray(labels) ? labels.length : 0} != 72`);
  }
  const ids = new Set();
  for (const label of labels) {
    const spiritId = Number(label?.spirit_id || 0);
    if (!Number.isInteger(spiritId) || spiritId < 1 || spiritId > 72) {
      throw new Error(`${filePath} retrieval head label has invalid spirit_id ${JSON.stringify(label?.spirit_id)}`);
    }
    ids.add(spiritId);
  }
  if (ids.size !== 72) {
    throw new Error(`${filePath} retrieval head labels must cover each spirit_id 1..72 exactly once`);
  }
}

function hydrateGenerativeRetrievalHead(head) {
  return {
    biases: Array.isArray(head?.biases) ? head.biases : [],
    weights: Array.isArray(head?.weights) ? head.weights.map((entries) => new Map(entries)) : [],
  };
}

function recomputeGenerativeSampleRetrieval(row, runDir, retrievalHead, errors) {
  const outDir = resolveGenerativeSampleOutDir(runDir, row.out_dir || "");
  const tracePath = outDir ? path.join(outDir, "trace.json") : "";
  const record = {
    row_index: row.row_index || null,
    model: row.model || "",
    prompt_hash: row.prompt_hash || "",
    spirit_id: Number(row.spirit_id || 0),
    out_dir: outDir,
    trace: tracePath,
    raw_samples_ref: "",
    raw_samples: "",
    expected_raw_samples: "",
    raw_samples_path_match: false,
    raw_samples_present: false,
    invalid_raw_samples: false,
    scored: false,
    recomputed_rank: null,
    recomputed_margin: null,
    recomputed_top1_spirit_id: null,
    recomputed_top1_name: "",
    recomputed_identity: null,
    reported_rank: numericOrNull(row.generated_retrieval_rank),
    reported_margin: numericOrNull(row.generated_retrieval_margin),
    reported_top1_spirit_id: numericOrNull(row.generated_retrieval_top1_spirit_id),
    reported_top1_name: row.generated_retrieval_top1_name || "",
    reported_identity: numericOrNull(row.generated_retrieval_identity),
    missing_score_fields: [],
    rank_match: null,
    margin_match: null,
    top1_spirit_match: null,
    top1_name_match: null,
    identity_match: null,
    ok: false,
  };
  for (const key of [
    "generated_retrieval_rank",
    "generated_retrieval_margin",
    "generated_retrieval_top1_spirit_id",
    "generated_retrieval_top1_name",
    "generated_retrieval_identity",
  ]) {
    if (row[key] === undefined || row[key] === "") {
      record.missing_score_fields.push(key);
    }
  }
  let trace = {};
  if (tracePath && fs.existsSync(tracePath)) {
    try {
      trace = readJson(tracePath);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      record.invalid_raw_samples = true;
      errors.push(`generative eval sample row ${row.row_index || "?"} retrieval trace could not be read: ${message}`);
      return record;
    }
  }
  const rawSamples = outDir ? resolveGenerativeRawSamples(outDir, trace) : null;
  record.raw_samples_ref = rawSamples?.source || "";
  record.raw_samples = rawSamples?.path || "";
  record.expected_raw_samples = rawSamples?.expected_path || "";
  record.raw_samples_path_match = rawSamples?.path_match === true;
  if (!rawSamples?.source_present) {
    record.invalid_raw_samples = true;
    errors.push(`generative eval sample row ${row.row_index || "?"} trace is missing raw_samples for generated retrieval recompute`);
    return record;
  }
  if (!record.raw_samples_path_match) {
    record.invalid_raw_samples = true;
    errors.push(
      `generative eval sample row ${row.row_index || "?"} raw_samples must resolve to ${rawSamples.expected_name} in the sample out_dir`,
    );
    return record;
  }
  if (!record.raw_samples || !rawSamples.present) {
    errors.push(`generative eval sample row ${row.row_index || "?"} missing raw samples for generated retrieval recompute`);
    return record;
  }
  record.raw_samples_present = true;
  let raw = null;
  try {
    raw = fs.readFileSync(record.raw_samples);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    record.invalid_raw_samples = true;
    errors.push(`generative eval sample row ${row.row_index || "?"} raw samples could not be read: ${message}`);
    return record;
  }
  if (raw.length === 0 || raw.length % GENERATED_RETRIEVAL_IMAGE_BYTES !== 0) {
    record.invalid_raw_samples = true;
    errors.push(
      `generative eval sample row ${row.row_index || "?"} raw sample byte count ${raw.length} is not a positive multiple of ${GENERATED_RETRIEVAL_IMAGE_BYTES}`,
    );
    return record;
  }
  const metrics = generatedRetrievalMetrics(raw, record.spirit_id, retrievalHead);
  record.scored = true;
  record.recomputed_rank = metrics.best_rank;
  record.recomputed_margin = metrics.best_margin;
  record.recomputed_top1_spirit_id = metrics.top1_spirit_id;
  record.recomputed_top1_name = metrics.top1_primary_name;
  record.recomputed_identity = metrics.best_rank === 1 ? 1 : 0;
  record.rank_match = record.reported_rank === metrics.best_rank;
  record.margin_match = record.reported_margin === metrics.best_margin;
  record.top1_spirit_match = record.reported_top1_spirit_id === metrics.top1_spirit_id;
  record.top1_name_match = record.reported_top1_name === metrics.top1_primary_name;
  record.identity_match = record.reported_identity === record.recomputed_identity;
  if (record.missing_score_fields.length > 0) {
    errors.push(
      `generative eval sample row ${row.row_index || "?"} missing generated retrieval score fields: ${record.missing_score_fields.join(",")}`,
    );
  }
  if (record.rank_match === false) {
    errors.push(`generative eval sample row ${row.row_index || "?"} generated_retrieval_rank ${record.reported_rank} != recomputed ${metrics.best_rank}`);
  }
  if (record.margin_match === false) {
    errors.push(
      `generative eval sample row ${row.row_index || "?"} generated_retrieval_margin ${record.reported_margin} != recomputed ${metrics.best_margin}`,
    );
  }
  if (record.top1_spirit_match === false) {
    errors.push(
      `generative eval sample row ${row.row_index || "?"} generated_retrieval_top1_spirit_id ${record.reported_top1_spirit_id} != recomputed ${metrics.top1_spirit_id}`,
    );
  }
  if (record.top1_name_match === false) {
    errors.push(
      `generative eval sample row ${row.row_index || "?"} generated_retrieval_top1_name ${JSON.stringify(record.reported_top1_name)} != recomputed ${JSON.stringify(metrics.top1_primary_name)}`,
    );
  }
  if (record.identity_match === false) {
    errors.push(
      `generative eval sample row ${row.row_index || "?"} generated_retrieval_identity ${record.reported_identity} != recomputed ${record.recomputed_identity}`,
    );
  }
  record.ok =
    record.scored &&
    record.missing_score_fields.length === 0 &&
    record.rank_match === true &&
    record.margin_match === true &&
    record.top1_spirit_match === true &&
    record.top1_name_match === true &&
    record.identity_match === true;
  return record;
}

function generatedRetrievalMetrics(raw, targetSpiritId, retrievalHead) {
  let best = null;
  for (let offset = 0; offset < raw.length; offset += GENERATED_RETRIEVAL_IMAGE_BYTES) {
    const image = raw.subarray(offset, offset + GENERATED_RETRIEVAL_IMAGE_BYTES);
    const signature = generatedRetrievalSampleSignature(image);
    const ranked = rankGeneratedRetrievalImage(retrievalHead, signature, retrievalHead.labels.length);
    const rank = generatedRetrievalTargetRank(ranked, targetSpiritId, retrievalHead.labels.length);
    const stats = generatedRetrievalRankStats(ranked, targetSpiritId);
    const metrics = {
      best_rank: rank,
      best_margin: stats.margin,
      top1_spirit_id: ranked[0]?.spirit_id ?? null,
      top1_primary_name: ranked[0]?.primary_name ?? "",
    };
    if (
      !best ||
      metrics.best_rank < best.best_rank ||
      (metrics.best_rank === best.best_rank &&
        metrics.best_margin !== null &&
        (best.best_margin === null || metrics.best_margin > best.best_margin))
    ) {
      best = metrics;
    }
  }
  return best || {
    best_rank: generatedRetrievalMissRank(retrievalHead.labels.length),
    best_margin: null,
    top1_spirit_id: null,
    top1_primary_name: "",
  };
}

function generatedRetrievalTargetRank(ranked, targetSpiritId, labelCount) {
  const index = ranked.findIndex((row) => row.spirit_id === targetSpiritId);
  return index >= 0 ? index + 1 : generatedRetrievalMissRank(labelCount, ranked.length);
}

function generatedRetrievalMissRank(labelCount, rankedCount = 0) {
  return Math.max(72, Number(labelCount || 0), Number(rankedCount || 0)) + 1;
}

function generatedRetrievalSampleSignature(image) {
  const sums = new Array(GENERATED_RETRIEVAL_BINS).fill(0);
  const counts = new Array(GENERATED_RETRIEVAL_BINS).fill(0);
  for (let y = 0; y < GENERATED_RETRIEVAL_IMAGE_SIZE; y += 1) {
    const binY = Math.floor((y * GENERATED_RETRIEVAL_GRID) / GENERATED_RETRIEVAL_IMAGE_SIZE);
    for (let x = 0; x < GENERATED_RETRIEVAL_IMAGE_SIZE; x += 1) {
      const binX = Math.floor((x * GENERATED_RETRIEVAL_GRID) / GENERATED_RETRIEVAL_IMAGE_SIZE);
      const bin = binY * GENERATED_RETRIEVAL_GRID + binX;
      sums[bin] += image[y * GENERATED_RETRIEVAL_IMAGE_SIZE + x];
      counts[bin] += 1;
    }
  }
  return sums.map((sum, index) => Math.floor((sum + Math.floor(counts[index] / 2)) / counts[index]));
}

function rankGeneratedRetrievalImage(model, signature, count) {
  const features = generatedRetrievalImageFeatures(generatedRetrievalSymbolicImageTokens(signature), model.feature_count);
  const ranked = model.labels.map((label) => ({
    label: label.label,
    spirit_id: Number(label.spirit_id || 0),
    primary_name: label.primary_name || "",
    score: scoreGeneratedRetrievalLabel(model.image_head, label.label, features),
  }));
  ranked.sort((left, right) => right.score - left.score || left.spirit_id - right.spirit_id);
  return ranked.slice(0, count);
}

function scoreGeneratedRetrievalLabel(head, label, features) {
  let score = Number(head.biases[label] || 0);
  const weights = head.weights[label] || new Map();
  for (const [feature, value] of features) {
    score += Number(weights.get(feature) || 0) * value;
  }
  return score;
}

function generatedRetrievalRankStats(ranked, targetSpiritId) {
  const target = ranked.find((row) => row.spirit_id === targetSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== targetSpiritId) || null;
  return {
    margin: target && runnerUp ? target.score - runnerUp.score : null,
  };
}

function generatedRetrievalSymbolicImageTokens(signature) {
  return solomonImage.symbolicImageTokens(signature, {
    grid: GENERATED_RETRIEVAL_GRID,
    imageBase: TASK_TOKEN_LAYOUT_FALLBACK.image_base,
    imageBins: TASK_TOKEN_LAYOUT_FALLBACK.image_bins,
    channelTokens: {
      ink: TASK_TOKEN_LAYOUT_FALLBACK.image_channel_ink,
      edge: TASK_TOKEN_LAYOUT_FALLBACK.image_channel_edge,
      component: TASK_TOKEN_LAYOUT_FALLBACK.image_channel_component,
      radial: TASK_TOKEN_LAYOUT_FALLBACK.image_channel_radial,
      direction: TASK_TOKEN_LAYOUT_FALLBACK.image_channel_direction,
    },
  });
}

function generatedRetrievalImageFeatures(image, featureCount) {
  const out = new Map();
  let channel = "ink";
  let position = 0;
  const channelNames = new Map([
    [TASK_TOKEN_LAYOUT_FALLBACK.image_channel_ink, "ink"],
    [TASK_TOKEN_LAYOUT_FALLBACK.image_channel_edge, "edge"],
    [TASK_TOKEN_LAYOUT_FALLBACK.image_channel_component, "component"],
    [TASK_TOKEN_LAYOUT_FALLBACK.image_channel_radial, "radial"],
    [TASK_TOKEN_LAYOUT_FALLBACK.image_channel_direction, "direction"],
  ]);
  for (const token of image) {
    if (channelNames.has(token)) {
      channel = channelNames.get(token);
      position = 0;
      addGeneratedRetrievalFeature(out, featureCount, "channel", channel, 32);
      continue;
    }
    const bin =
      token >= TASK_TOKEN_LAYOUT_FALLBACK.image_base &&
      token < TASK_TOKEN_LAYOUT_FALLBACK.image_base + TASK_TOKEN_LAYOUT_FALLBACK.image_bins
        ? token - TASK_TOKEN_LAYOUT_FALLBACK.image_base
        : token;
    addGeneratedRetrievalFeature(out, featureCount, "ipos", `${channel}:${position}:${bin}`, 64);
    addGeneratedRetrievalFeature(out, featureCount, "itok", `${channel}:${bin}`, 8);
    if (position % GENERATED_RETRIEVAL_GRID === 0) {
      addGeneratedRetrievalFeature(out, featureCount, "irow", `${channel}:${Math.floor(position / GENERATED_RETRIEVAL_GRID)}:${bin}`, 6);
    }
    position += 1;
  }
  return [...out.entries()];
}

function addGeneratedRetrievalFeature(out, featureCount, namespace, value, amount) {
  const hash = fnv32Text(`${namespace}\xff${value}`);
  const index = hash % featureCount;
  const sign = hash & 0x80000000 ? -1 : 1;
  out.set(index, Math.max(-127, Math.min(127, (out.get(index) || 0) + sign * amount)));
}

function fnv32Text(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function summarizeRecomputedGeneratedRetrievalRows(scoredRows, expectedRows) {
  const count = expectedRows;
  const top1 = scoredRows.filter((row) => Number(row.recomputed_rank || 0) === 1).length;
  const top5 = scoredRows.filter((row) => {
    const rank = Number(row.recomputed_rank || 0);
    return rank > 0 && rank <= 5;
  }).length;
  const rankTotal = scoredRows.reduce((sum, row) => sum + Number(row.recomputed_rank || 0), 0);
  const margins = scoredRows
    .map((row) => row.recomputed_margin)
    .filter((value) => Number.isFinite(value));
  return {
    top1,
    top5,
    top1_per_mille: count === 0 ? 0 : Math.floor((top1 * 1000) / count),
    top5_per_mille: count === 0 ? 0 : Math.floor((top5 * 1000) / count),
    mean_rank_q8: count === 0 ? 0 : Math.floor((rankTotal * 256) / count),
    min_margin: margins.length === 0 ? null : Math.min(...margins),
  };
}

function compareGeneratedRetrievalSummaryField(record, key) {
  const actual = record.summary[key];
  const expected = record.recomputed[key];
  if (expected === null) {
    if (actual !== null && actual !== undefined && actual !== "") {
      record.mismatches.push(`${key}=${actual} != recomputed ${expected}`);
    }
    return;
  }
  if (Number(actual) !== Number(expected)) {
    record.mismatches.push(`${key}=${actual} != recomputed ${expected}`);
  }
}

function numericOrNull(value) {
  if (value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function incrementCount(counts, key) {
  counts[key] = Number(counts[key] || 0) + 1;
}

function summarizeGenerativeEvalRow(row, errors) {
  const label = `generative eval row ${row.row_index || "?"} (${row.model || "<missing-model>"})`;
  const prompts = numericField(row, "prompts", label, errors);
  const top1 = numericField(row, "top1", label, errors);
  const top5 = numericField(row, "top5", label, errors);
  const top1Permille = numericField(row, "top1_per_mille", label, errors);
  const top5Permille = numericField(row, "top5_per_mille", label, errors);
  const top116 = numericField(row, "top1_16", label, errors);
  const top516 = numericField(row, "top5_16", label, errors);
  const top116Permille = numericField(row, "top1_16_per_mille", label, errors);
  const top516Permille = numericField(row, "top5_16_per_mille", label, errors);
  const top1Px = numericField(row, "top1_px", label, errors);
  const top5Px = numericField(row, "top5_px", label, errors);
  const top1PxPermille = numericField(row, "top1_px_per_mille", label, errors);
  const top5PxPermille = numericField(row, "top5_px_per_mille", label, errors);
  const latentTop1 = numericField(row, "latent_top1", label, errors);
  const latentTop5 = numericField(row, "latent_top5", label, errors);
  const latentTop1Permille = numericField(row, "latent_top1_per_mille", label, errors);
  const latentTop5Permille = numericField(row, "latent_top5_per_mille", label, errors);
  const retrievalTop1 = optionalNumericField(row, "generated_retrieval_top1", label, errors);
  const retrievalTop5 = optionalNumericField(row, "generated_retrieval_top5", label, errors);
  const retrievalTop1Permille = optionalNumericField(row, "generated_retrieval_top1_per_mille", label, errors);
  const retrievalTop5Permille = optionalNumericField(row, "generated_retrieval_top5_per_mille", label, errors);
  const retrievalMeanRankQ8 = optionalNumericField(row, "mean_generated_retrieval_rank_q8", label, errors);
  const retrievalMinMargin = optionalNumericField(row, "min_generated_retrieval_margin", label, errors);
  const meanRankQ8 = numericField(row, "mean_rank_q8", label, errors);
  const meanRank16Q8 = numericField(row, "mean_rank_16_q8", label, errors);
  const meanRankPxQ8 = numericField(row, "mean_rank_px_q8", label, errors);
  const meanLatentRankQ8 = numericField(row, "mean_latent_rank_q8", label, errors);
  const latentModelHash = String(row.latent_model_hash || "");
  if (!row.model) {
    errors.push(`${label} missing model`);
  }
  if (!row.latent_model) {
    errors.push(`${label} missing latent_model`);
  }
  if (!latentModelHash) {
    errors.push(`${label} missing latent_model_hash`);
  }
  if (prompts <= 0) {
    errors.push(`${label} prompts ${prompts} <= 0`);
  }
  validateTopK({ top1, top5, prompts, label: `${label} generated signature`, errors });
  validateTopK({ top1: top116, top5: top516, prompts, label: `${label} generated 16x16 signature`, errors });
  validateTopK({ top1: top1Px, top5: top5Px, prompts, label: `${label} generated pixel`, errors });
  validateTopK({ top1: latentTop1, top5: latentTop5, prompts, label: `${label} latent decoded`, errors });
  validatePerMille(top1Permille, `${label} top1_per_mille`, errors);
  validatePerMille(top5Permille, `${label} top5_per_mille`, errors);
  validatePerMille(top116Permille, `${label} top1_16_per_mille`, errors);
  validatePerMille(top516Permille, `${label} top5_16_per_mille`, errors);
  validatePerMille(top1PxPermille, `${label} top1_px_per_mille`, errors);
  validatePerMille(top5PxPermille, `${label} top5_px_per_mille`, errors);
  validatePerMille(latentTop1Permille, `${label} latent_top1_per_mille`, errors);
  validatePerMille(latentTop5Permille, `${label} latent_top5_per_mille`, errors);
  if (retrievalTop1 !== null || retrievalTop5 !== null) {
    validateTopK({
      top1: retrievalTop1 ?? 0,
      top5: retrievalTop5 ?? 0,
      prompts,
      label: `${label} generated retrieval`,
      errors,
    });
  }
  if (retrievalTop1Permille !== null) {
    validatePerMille(retrievalTop1Permille, `${label} generated_retrieval_top1_per_mille`, errors);
  }
  if (retrievalTop5Permille !== null) {
    validatePerMille(retrievalTop5Permille, `${label} generated_retrieval_top5_per_mille`, errors);
  }
  return {
    model: row.model || "",
    latent_model: row.latent_model || "",
    latent_model_hash: latentModelHash,
    prompts,
    generated_signature: {
      top1,
      top5,
      top1_per_mille: top1Permille,
      top5_per_mille: top5Permille,
      mean_rank_q8: meanRankQ8,
      mean_target_distance_q8: numericField(row, "mean_generated_target_distance_q8", label, errors),
    },
    generated_signature_16: {
      top1: top116,
      top5: top516,
      top1_per_mille: top116Permille,
      top5_per_mille: top516Permille,
      mean_rank_q8: meanRank16Q8,
      mean_target_distance_q8: numericField(row, "mean_generated_target_distance_16_q8", label, errors),
    },
    generated_pixel: {
      top1: top1Px,
      top5: top5Px,
      top1_per_mille: top1PxPermille,
      top5_per_mille: top5PxPermille,
      mean_rank_q8: meanRankPxQ8,
      mean_target_distance_q8: numericField(row, "mean_generated_target_distance_px_q8", label, errors),
    },
    latent_decoded: {
      top1: latentTop1,
      top5: latentTop5,
      top1_per_mille: latentTop1Permille,
      top5_per_mille: latentTop5Permille,
      mean_rank_q8: meanLatentRankQ8,
      mean_target_distance_q8: numericField(row, "mean_latent_decoded_target_distance_q8", label, errors),
      mean_selected_text_distance_q8: numericField(row, "mean_latent_target_distance_q8", label, errors),
    },
    generated_retrieval: {
      present: retrievalTop1 !== null || retrievalTop5 !== null || retrievalTop1Permille !== null || retrievalTop5Permille !== null,
      top1: retrievalTop1,
      top5: retrievalTop5,
      top1_per_mille: retrievalTop1Permille,
      top5_per_mille: retrievalTop5Permille,
      mean_rank_q8: retrievalMeanRankQ8,
      min_margin: retrievalMinMargin,
    },
    artifact: {
      mean_generated_ink_q8: numericField(row, "mean_generated_ink_q8", label, errors),
      mean_generated_outside_ink_q8: numericField(row, "mean_generated_outside_ink_q8", label, errors),
      mean_generated_edge_ink_q8: numericField(row, "mean_generated_edge_ink_q8", label, errors),
      selected_mean_wash_penalty_q8: numericField(row, "selected_mean_wash_penalty_q8", label, errors),
      text_weight: numericField(row, "text_weight", label, errors),
    },
  };
}

function generativeEvalReport({ ok, errors, inputPath, summaryPath, runDir, rows, floor, evidence, config }) {
  const best = bestGenerativeModel(rows);
  return {
    ok,
    errors,
    present: true,
    input: inputPath,
    summary: summaryPath,
    run_dir: runDir || "",
    model_count: rows.length,
    evidence: evidence || absentGenerativeEvalEvidence(),
    product_floor: floor || generativeEvalFloor(rows, config),
    best,
    models: rows,
  };
}

function generativeEvalFloor(rows, config) {
  const requirements = {
    min_generated_top5_per_mille: Number(config.minGeneratedTop5PerMille || 0),
    min_generated_top5_16_per_mille: effectiveMinGeneratedTop516PerMille(config),
    min_generated_top5_px_per_mille: Number(config.minGeneratedTop5PxPerMille || 0),
    min_generated_retrieval_top1_per_mille: Number(config.minGeneratedRetrievalTop1PerMille || 0),
    min_generated_retrieval_top5_per_mille: Number(config.minGeneratedRetrievalTop5PerMille || 0),
    min_generated_retrieval_margin: Number(config.minGeneratedRetrievalMargin || 0),
    min_generated_prompt_rows: Number(config.minGeneratedPromptRows || 0),
    require_generated_output_identity: config.requireGenerativeOutputIdentity === true,
    min_latent_top5_per_mille: Number(config.minLatentTop5PerMille || 0),
    max_generated_mean_rank_q8: Number(config.maxGeneratedMeanRankQ8 || 0),
    max_generated_mean_rank_16_q8: Number(config.maxGeneratedMeanRank16Q8 || 0),
    max_generated_mean_rank_px_q8: Number(config.maxGeneratedMeanRankPxQ8 || 0),
    max_generated_mean_target_distance_q8: Number(config.maxGeneratedMeanTargetDistanceQ8 || 0),
    max_generated_mean_target_distance_16_q8: Number(config.maxGeneratedMeanTargetDistance16Q8 || 0),
    max_generated_mean_target_distance_px_q8: Number(config.maxGeneratedMeanTargetDistancePxQ8 || 0),
  };
  const hasRequirements = Object.values(requirements).some((value) => value > 0);
  if (rows.length === 0) {
    return {
      ok: !hasRequirements,
      requirements,
      matching_model: null,
      errors: hasRequirements ? ["generative eval has no model rows for the configured product floor"] : [],
    };
  }
  const matchingModel = rows.find((row) => generativeModelMeetsFloor(row, requirements));
  const errors = [];
  if (hasRequirements && !matchingModel) {
    errors.push("generative eval no model met the configured product-generation floor");
  }
  return {
    ok: errors.length === 0,
    requirements,
    matching_model: matchingModel ? matchingModel.model : null,
    errors,
  };
}

function generativeModelMeetsFloor(row, requirements) {
  return (
    row.generated_signature.top5_per_mille >= requirements.min_generated_top5_per_mille &&
    row.generated_signature_16.top5_per_mille >= requirements.min_generated_top5_16_per_mille &&
    row.generated_pixel.top5_per_mille >= requirements.min_generated_top5_px_per_mille &&
    row.prompts >= requirements.min_generated_prompt_rows &&
    optionalFloorOk(
      row.generated_retrieval,
      "top1_per_mille",
      requirements.min_generated_retrieval_top1_per_mille,
    ) &&
    optionalFloorOk(
      row.generated_retrieval,
      "top5_per_mille",
      requirements.min_generated_retrieval_top5_per_mille,
    ) &&
    optionalFloorOk(
      row.generated_retrieval,
      "min_margin",
      requirements.min_generated_retrieval_margin,
    ) &&
    generatedOutputIdentityFloorOk(row, requirements) &&
    row.latent_decoded.top5_per_mille >= requirements.min_latent_top5_per_mille &&
    maxRankOk(row.generated_signature.mean_rank_q8, requirements.max_generated_mean_rank_q8) &&
    maxRankOk(row.generated_signature_16.mean_rank_q8, requirements.max_generated_mean_rank_16_q8) &&
    maxRankOk(row.generated_pixel.mean_rank_q8, requirements.max_generated_mean_rank_px_q8) &&
    maxRankOk(row.generated_signature.mean_target_distance_q8, requirements.max_generated_mean_target_distance_q8) &&
    maxRankOk(
      row.generated_signature_16.mean_target_distance_q8,
      requirements.max_generated_mean_target_distance_16_q8,
    ) &&
    maxRankOk(row.generated_pixel.mean_target_distance_q8, requirements.max_generated_mean_target_distance_px_q8)
  );
}

function generatedOutputIdentityFloorOk(row, requirements) {
  if (requirements.require_generated_output_identity !== true) {
    return true;
  }
  return (
    row.generated_retrieval?.present === true &&
    Number(row.generated_retrieval.top1 || 0) === Number(row.prompts || 0) &&
    Number(row.generated_retrieval.top1_per_mille || 0) === 1000 &&
    Number(row.generated_retrieval.min_margin || 0) > 0
  );
}

function effectiveMinGeneratedTop516PerMille(config) {
  const configured = Number(config.minGeneratedTop516PerMille || 0);
  const requiredEvidence = config.requireGenerativeEval || Boolean(config.generativeEvalPath);
  return Math.max(configured, requiredEvidence ? 1 : 0);
}

function optionalFloorOk(metric, key, minimum) {
  if (Number(minimum || 0) === 0) {
    return true;
  }
  return metric?.present === true && Number(metric?.[key] || 0) >= minimum;
}

function maxRankOk(value, maxValue) {
  return Number(maxValue || 0) === 0 || Number(value || 0) <= maxValue;
}

function bestGenerativeModel(rows) {
  if (rows.length === 0) {
    return null;
  }
  return [...rows].sort((left, right) => {
    const retrievalTop1Delta =
      Number(right.generated_retrieval?.top1_per_mille || 0) -
      Number(left.generated_retrieval?.top1_per_mille || 0);
    if (retrievalTop1Delta !== 0) return retrievalTop1Delta;
    const retrievalTop5Delta =
      Number(right.generated_retrieval?.top5_per_mille || 0) -
      Number(left.generated_retrieval?.top5_per_mille || 0);
    if (retrievalTop5Delta !== 0) return retrievalTop5Delta;
    const top516Delta = right.generated_signature_16.top5_per_mille - left.generated_signature_16.top5_per_mille;
    if (top516Delta !== 0) return top516Delta;
    const top5PxDelta = right.generated_pixel.top5_per_mille - left.generated_pixel.top5_per_mille;
    if (top5PxDelta !== 0) return top5PxDelta;
    const latentTop5Delta = right.latent_decoded.top5_per_mille - left.latent_decoded.top5_per_mille;
    if (latentTop5Delta !== 0) return latentTop5Delta;
    return left.generated_signature_16.mean_rank_q8 - right.generated_signature_16.mean_rank_q8;
  })[0];
}

function numericField(row, key, label, errors) {
  const value = row[key];
  if (value === undefined || value === "") {
    errors.push(`${label} missing ${key}`);
    return 0;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    errors.push(`${label} ${key} is not numeric: ${JSON.stringify(value)}`);
    return 0;
  }
  return number;
}

function optionalNumericField(row, key, label, errors) {
  const value = row[key];
  if (value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    errors.push(`${label} ${key} is not numeric: ${JSON.stringify(value)}`);
    return null;
  }
  return number;
}

function validateTopK({ top1, top5, prompts, label, errors }) {
  if (top1 < 0 || top5 < 0) {
    errors.push(`${label} top-k counts must be non-negative`);
  }
  if (top1 > top5) {
    errors.push(`${label} top1 ${top1} > top5 ${top5}`);
  }
  if (prompts > 0 && top5 > prompts) {
    errors.push(`${label} top5 ${top5} > prompts ${prompts}`);
  }
}

function validatePerMille(value, label, errors) {
  if (value < 0 || value > 1000) {
    errors.push(`${label} ${value} outside 0-1000`);
  }
}

function buildConfidenceTrace({
  evalReport,
  corpusContractReport,
  retrievalReport,
  sampleReport,
  identityInferenceReport,
  curriculumStagesReport,
  denoiseBridgeReport,
  groundedCorpusReport,
  generativeReport,
  config,
}) {
  const issues = [];
  const symbolicImageTokens = symbolicImageTokenConfidence(corpusContractReport, curriculumStagesReport, config);
  if (!symbolicImageTokens.ok) {
    issues.push(
      symbolicImageTokens.required
        ? "required symbolic image-token byte evidence is not complete"
        : "present symbolic image-token byte evidence is not complete",
    );
  }
  const knownPromptOk = requireCompleteMetric(
    retrievalReport.known_prompts,
    "known prompt text retrieval",
    issues,
  );
  let identityBindingOk = requireCompleteMetric(
    retrievalReport.identity_bindings?.total,
    "identity binding text retrieval",
    issues,
  );
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    identityBindingOk =
      requireCompleteMetric(
        retrievalReport.identity_bindings?.by_kind?.[kind],
        `identity binding ${kind} retrieval`,
        issues,
      ) && identityBindingOk;
  }
  const heldoutPromptOk = optionalCompleteMetric(
    retrievalReport.heldout_prompts,
    "held-out prompt text retrieval",
    config.requireHeldoutPrompts,
    issues,
  );
  const imageToTextOk = requireCompleteMetric(
    retrievalReport.image_to_text,
    "image-to-text retrieval",
    issues,
  );
  let imageTaskOk = true;
  for (const task of IMAGE_RETRIEVAL_TASKS) {
    imageTaskOk =
      requireCompleteMetric(retrievalReport.image_tasks?.[task], `${task} retrieval`, issues) &&
      imageTaskOk;
  }
  let forwardImagePlanOk = true;
  for (const task of FORWARD_IMAGE_PLAN_TASKS) {
    forwardImagePlanOk =
      requireNativeTaskMetric(evalReport.tasks?.[task], `${task} native image-plan eval`, issues) &&
      forwardImagePlanOk;
  }
  let nativeTaskEvalOk = true;
  for (const task of REQUIRED_TASKS) {
    nativeTaskEvalOk =
      requireNativeTaskMetric(evalReport.tasks?.[task], `${task} native task eval`, issues) &&
      nativeTaskEvalOk;
  }
  const directionalNativeEvalOk = evalReport.directional_groups?.ok === true;
  if (!directionalNativeEvalOk) {
    issues.push("native directional task-phase evidence is not complete");
  }
  const matchYesOk = requireCompleteMetric(retrievalReport.match?.yes, "match yes agreement", issues);
  const matchNoOk = requireCompleteMetric(retrievalReport.match?.no, "match no disagreement", issues);
  const matchNoImageOk = requireCompleteMetric(
    retrievalReport.match?.no_by_role?.image,
    "wrong-image hard negatives",
    issues,
  );
  const matchNoPromptOk = requireCompleteMetric(
    retrievalReport.match?.no_by_role?.prompt,
    "wrong-prompt hard negatives",
    issues,
  );
  const sampleOk =
    sampleReport.ok &&
    Number(sampleReport.samples || 0) > 0 &&
    sampleReport.text_image_agreement === true &&
    sampleReport.generated_text_image_agreement === true &&
    sampleReport.signature_retrieval_agreement === true &&
    sampleReport.image_to_text_identification === true &&
    sampleReport.generated_text_identification === true &&
    Number(sampleReport.min_signature_margin || 0) > 0 &&
    Number(sampleReport.min_retrieval_image_margin || 0) > 0 &&
    Number(sampleReport.min_image_to_text_margin || 0) > 0 &&
    Number(sampleReport.min_retrieval_text_margin || 0) > 0 &&
    Number(sampleReport.min_generated_text_margin || 0) > 0;
  if (!sampleOk) {
    issues.push("sample binding does not provide complete text/image/signature agreement");
  }
  const sourceOk =
    identityInferenceReport.present === true &&
    identityInferenceReport.ok === true &&
    sourceSummaryHasEvidence(identityInferenceReport.source_summary) &&
    sampleSummaryHasEvidence(identityInferenceReport.sample_summary);
  if (!sourceOk) {
    issues.push("source-grounded identity evidence is not complete");
  }
  const curriculumRequired = config.requireCurriculumStages;
  const curriculumIdentityOk = curriculumStagesReport.present
    ? curriculumStagesReport.ok && curriculumIdentityBindingReady(curriculumStagesReport.identity_binding)
    : !curriculumRequired;
  if (!curriculumIdentityOk) {
    issues.push(
      curriculumRequired
        ? "required curriculum identity-binding evidence is not complete"
        : "present curriculum identity-binding evidence is not complete",
    );
  }
  const groundedRequired = config.requireGroundedCorpus;
  const groundedOk = groundedCorpusReport.present ? groundedCorpusReport.ok : !groundedRequired;
  if (!groundedOk) {
    issues.push(
      groundedRequired
        ? "required grounded corpus evidence is not complete"
        : "present grounded corpus evidence is not complete",
    );
  }
  const denoiseRequired = config.requireDenoiseBridge;
  const denoiseOutputIdentityRequired = config.requireDenoiseOutputIdentity;
  const denoiseOutputIdentity =
    denoiseBridgeReport.output_image_to_text_identification === true ||
    (!denoiseOutputIdentityRequired &&
      (denoiseBridgeReport.output_image_to_text_identification === null ||
        denoiseBridgeReport.output_image_to_text_identification === undefined));
  const denoiseOutputIdentityMarginOk =
    !denoiseOutputIdentityRequired ||
    finiteNumberOrNull(denoiseBridgeReport.min_output_retrieval_image_margin) > 0;
  const denoiseTargetCoverageOk =
    Number(denoiseBridgeReport.expected_unique_targets || 0) >= Number(config.minDenoiseBridgeUniqueTargets || 0);
  const denoiseOk = denoiseBridgeReport.present
    ? denoiseBridgeReport.ok &&
      Number(denoiseBridgeReport.pairs || 0) > 0 &&
      finiteNumberOrNull(denoiseBridgeReport.min_output_signature_distance) !== null &&
      Number(denoiseBridgeReport.min_output_ink_range || 0) > 0 &&
      denoiseBridgeReport.trace_integrity_ok === true &&
      denoiseOutputIdentity &&
      denoiseOutputIdentityMarginOk &&
      denoiseTargetCoverageOk
    : !denoiseRequired;
  if (!denoiseOk) {
    issues.push(
      denoiseRequired
        ? "required denoise bridge evidence is not complete"
        : "present denoise bridge evidence is not complete",
    );
  }
  if (denoiseOutputIdentityRequired && !denoiseOutputIdentity) {
    issues.push("required denoised-output image-to-text identity evidence is not complete");
  }
  const generativeRequired = config.requireGenerativeEval;
  const generativeOutputIdentityRequired = config.requireGenerativeOutputIdentity;
  const generativeOutputIdentityOk = generativeReport.present
    ? !generativeOutputIdentityRequired ||
      generativeReport.evidence?.output_identity?.by_model?.[generativeReport.product_floor?.matching_model || ""]?.ok === true
    : !generativeOutputIdentityRequired;
  const generativeOk = generativeReport.present
    ? generativeReport.ok && generativeReport.product_floor?.ok === true && generativeOutputIdentityOk
    : !generativeRequired;
  if (!generativeOk) {
    issues.push(
      generativeRequired
        ? "required product-generation evidence is not complete"
        : "present product-generation evidence is not complete",
    );
  }
  if (!generativeOutputIdentityOk) {
    issues.push("required product-generation output identity evidence is not complete");
  }
  const allRequiredOk =
    knownPromptOk &&
    identityBindingOk &&
    heldoutPromptOk &&
    imageToTextOk &&
    imageTaskOk &&
    forwardImagePlanOk &&
    nativeTaskEvalOk &&
    directionalNativeEvalOk &&
    matchYesOk &&
    matchNoOk &&
    matchNoImageOk &&
    matchNoPromptOk &&
    sampleOk &&
    symbolicImageTokens.ok &&
    sourceOk &&
    curriculumIdentityOk &&
    groundedOk &&
    denoiseOk &&
    generativeOk;
  return {
    ok: allRequiredOk,
    errors: issues,
    issues,
    label: allRequiredOk
      ? generativeReport.present && denoiseBridgeReport.present
        ? "strong-bidirectional-product-generation"
        : generativeReport.present
          ? "strong-bidirectional-product"
          : denoiseBridgeReport.present
            ? "strong-bidirectional-generation"
            : "strong-bidirectional-binding"
      : "incomplete",
    text_binding: {
      known_prompts: confidenceMetric(retrievalReport.known_prompts),
      identity_bindings: {
        total: confidenceMetric(retrievalReport.identity_bindings?.total),
        by_kind: Object.fromEntries(
          REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [
            kind,
            confidenceMetric(retrievalReport.identity_bindings?.by_kind?.[kind]),
          ]),
        ),
      },
      heldout_prompts: confidenceMetric(retrievalReport.heldout_prompts),
    },
    image_binding: {
      image_to_text: confidenceMetric(retrievalReport.image_to_text),
      image_tasks: Object.fromEntries(
        IMAGE_RETRIEVAL_TASKS.map((task) => [task, confidenceMetric(retrievalReport.image_tasks?.[task])]),
      ),
      sample_image_to_text_identification: sampleReport.image_to_text_identification === true,
      min_image_to_text_margin: Number(sampleReport.min_image_to_text_margin || 0),
      min_retrieval_image_margin: Number(sampleReport.min_retrieval_image_margin || 0),
    },
    forward_image_plan: {
      tasks: Object.fromEntries(
        FORWARD_IMAGE_PLAN_TASKS.map((task) => [task, nativeTaskConfidenceMetric(evalReport.tasks?.[task])]),
      ),
    },
    native_task_eval: nativeTaskConfidenceEval(evalReport.tasks || {}),
    directional_native_eval: evalReport.directional_groups || null,
    cross_modal_agreement: {
      match_yes: confidenceMetric(retrievalReport.match?.yes),
      match_no: confidenceMetric(retrievalReport.match?.no),
      wrong_image_negatives: confidenceMetric(retrievalReport.match?.no_by_role?.image),
      wrong_prompt_negatives: confidenceMetric(retrievalReport.match?.no_by_role?.prompt),
      text_image_agreement: sampleReport.text_image_agreement === true,
      generated_text_image_agreement: sampleReport.generated_text_image_agreement === true,
      generated_text_identification: sampleReport.generated_text_identification === true,
      signature_retrieval_agreement: sampleReport.signature_retrieval_agreement === true,
      min_signature_margin: Number(sampleReport.min_signature_margin || 0),
      min_retrieval_text_margin: Number(sampleReport.min_retrieval_text_margin || 0),
      min_generated_text_margin: Number(sampleReport.min_generated_text_margin || 0),
    },
    symbolic_image_tokens: symbolicImageTokens,
    source_grounding: {
      present: identityInferenceReport.present === true,
      grounded_corpus_present: groundedCorpusReport.present === true,
      grounded_corpus_ok: groundedCorpusReport.ok === true,
      grounded_source_provenance: groundedCorpusReport.require_source_provenance === true,
      grounded_name_source_explain: groundedCorpusReport.require_name_source_explain === true,
      grounded_description_source_image: groundedCorpusReport.require_description_source_image === true,
      grounded_image_attribute_generic_prompt:
        groundedCorpusReport.require_image_attribute_generic_prompt === true,
      grounded_source_tasks: groundedCorpusReport.source_text_tasks || [],
      grounded_attribute_tasks: groundedCorpusReport.attribute_tasks || [],
      text_queries_have_source_text:
        identityInferenceReport.source_summary?.text_queries_have_source_text === true,
      image_queries_have_source_text:
        identityInferenceReport.source_summary?.image_queries_have_source_text === true,
      sample_queries_have_source_text:
        identityInferenceReport.source_summary?.sample_queries_have_source_text === true,
      sample_source_text_evidence:
        identityInferenceReport.sample_summary?.source_text_evidence === true,
      generated_text_source_evidence:
        identityInferenceReport.sample_summary?.generated_text_source_evidence === true,
      generated_text_image_agreement:
        identityInferenceReport.sample_summary?.generated_text_image_agreement === true,
      expected_generated_text_agreement:
        identityInferenceReport.sample_summary?.expected_generated_text_agreement === true,
      min_source_text_chars: Number(identityInferenceReport.sample_summary?.min_source_text_chars || 0),
      min_prompt_text_margin: Number(identityInferenceReport.sample_summary?.min_prompt_text_margin || 0),
      min_generated_text_margin: Number(identityInferenceReport.sample_summary?.min_generated_text_margin || 0),
    },
    identity_curriculum: {
      present: curriculumStagesReport.present === true,
      required: curriculumRequired,
      ok: curriculumIdentityOk,
      stage_count: Number(curriculumStagesReport.stage_count || 0),
      source_corpus_provenance: curriculumStagesReport.source_corpus_provenance || {},
      binding_stages: curriculumStagesReport.identity_binding || {},
    },
    generation_bridge: {
      present: denoiseBridgeReport.present === true,
      required: denoiseRequired,
      output_identity_required: denoiseOutputIdentityRequired,
      pairs: Number(denoiseBridgeReport.pairs || 0),
      min_unique_targets: Number(config.minDenoiseBridgeUniqueTargets || 0),
      expected_unique_targets: Number(denoiseBridgeReport.expected_unique_targets || 0),
      unique_expected_spirit_ids: denoiseBridgeReport.unique_expected_spirit_ids || [],
      missing_expected_spirit_ids: denoiseBridgeReport.missing_expected_spirit_ids || [],
      target_coverage_ok: denoiseTargetCoverageOk,
      denoise_model: denoiseBridgeReport.denoise_model || "",
      denoise_model_hash: denoiseBridgeReport.denoise_model_hash || "",
      denoise_model_provenance: denoiseBridgeReport.denoise_model_provenance || null,
      min_output_signature_distance: finiteNumberOrNull(denoiseBridgeReport.min_output_signature_distance),
      min_output_ink_range: finiteNumberOrNull(denoiseBridgeReport.min_output_ink_range),
      trace_integrity_ok:
        denoiseBridgeReport.trace_integrity_ok === null || denoiseBridgeReport.trace_integrity_ok === undefined
          ? null
          : denoiseBridgeReport.trace_integrity_ok === true,
      output_image_to_text_identification:
        denoiseBridgeReport.output_image_to_text_identification === null ||
        denoiseBridgeReport.output_image_to_text_identification === undefined
          ? null
          : denoiseBridgeReport.output_image_to_text_identification === true,
      min_output_retrieval_image_margin: finiteNumberOrNull(denoiseBridgeReport.min_output_retrieval_image_margin),
      sample_binding_provenance: denoiseBridgeReport.sample_binding_provenance || null,
      output_provenance: denoiseBridgeReport.output_provenance || null,
    },
    product_generation: {
      present: generativeReport.present === true,
      required: generativeRequired,
      output_identity_required: generativeOutputIdentityRequired,
      model_count: Number(generativeReport.model_count || 0),
      heldout_partition_ready: generativeReport.evidence?.heldout_partition_ready === true,
      sample_count: Number(generativeReport.evidence?.sample_count || 0),
      prompt_provenance: generativeReport.evidence?.prompt_provenance || null,
      sample_partitions: generativeReport.evidence?.sample_partitions || {},
      sampler_target_sources: generativeReport.evidence?.sampler_target_sources || {},
      trace_integrity_ok: generativeReport.evidence?.trace_integrity?.ok === true,
      trace_count: Number(generativeReport.evidence?.trace_integrity?.trace_count || 0),
      product_floor_ok: generativeReport.product_floor?.ok === true,
      matching_model: generativeReport.product_floor?.matching_model || null,
      matching_model_output_identity:
        generativeReport.evidence?.output_identity?.by_model?.[generativeReport.product_floor?.matching_model || ""] || null,
      output_identity: generativeReport.evidence?.output_identity || null,
      retrieval_head_provenance: generativeReport.evidence?.retrieval_head_provenance || null,
      latent_model_provenance: generativeReport.evidence?.latent_model_provenance || null,
      sampler_model_provenance: generativeReport.evidence?.sampler_model_provenance || null,
      generated_retrieval_provenance: generativeReport.evidence?.generated_retrieval_provenance || null,
      best_model: generativeReport.best?.model || null,
      best_retrieval_top1_per_mille: Number(generativeReport.best?.generated_retrieval?.top1_per_mille || 0),
      best_retrieval_top5_per_mille: Number(generativeReport.best?.generated_retrieval?.top5_per_mille || 0),
      best_retrieval_min_margin: finiteNumberOrNull(generativeReport.best?.generated_retrieval?.min_margin),
      best_top5_16_per_mille: Number(generativeReport.best?.generated_signature_16?.top5_per_mille || 0),
      best_top5_px_per_mille: Number(generativeReport.best?.generated_pixel?.top5_per_mille || 0),
      best_latent_top5_per_mille: Number(generativeReport.best?.latent_decoded?.top5_per_mille || 0),
    },
  };
}

function requireCompleteMetric(metric, label, issues) {
  const summary = confidenceMetric(metric);
  if (summary.count <= 0) {
    issues.push(`${label} has no rows`);
    return false;
  }
  if (summary.top1 !== summary.count) {
    issues.push(`${label} top1 ${summary.top1} != count ${summary.count}`);
    return false;
  }
  return true;
}

function optionalCompleteMetric(metric, label, required, issues) {
  const summary = confidenceMetric(metric);
  if (summary.count <= 0) {
    if (required) {
      issues.push(`${label} is required but has no rows`);
      return false;
    }
    return true;
  }
  if (summary.top1 !== summary.count) {
    issues.push(`${label} top1 ${summary.top1} != count ${summary.count}`);
    return false;
  }
  return true;
}

function requireNativeTaskMetric(metric, label, issues) {
  const summary = nativeTaskConfidenceMetric(metric);
  if (summary.targets <= 0) {
    issues.push(`${label} has no targets`);
    return false;
  }
  if (summary.invalid_contexts !== 0) {
    issues.push(`${label} invalid_contexts ${summary.invalid_contexts} != 0`);
    return false;
  }
  if (summary.targets < summary.min_targets) {
    issues.push(`${label} targets ${summary.targets} < ${summary.min_targets}`);
    return false;
  }
  if (summary.top5_accuracy_per_mille < summary.min_top5_per_mille) {
    issues.push(
      `${label} top5 ${summary.top5_accuracy_per_mille} < ${summary.min_top5_per_mille}`,
    );
    return false;
  }
  return true;
}

function confidenceMetric(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    top1_per_mille: Number(metric?.top1_per_mille || 0),
    top5_per_mille: Number(metric?.top5_per_mille || 0),
    min_margin: finiteNumberOrNull(metric?.min_margin),
    mean_margin: finiteNumberOrNull(metric?.mean_margin),
  };
}

function nativeTaskConfidenceMetric(metric) {
  return {
    targets: Number(metric?.targets || 0),
    correct: Number(metric?.correct || 0),
    invalid_contexts: Number(metric?.invalid_contexts || 0),
    accuracy_per_mille: Number(metric?.accuracy_per_mille || 0),
    top5_accuracy_per_mille: Number(metric?.top5_accuracy_per_mille || 0),
    top10_accuracy_per_mille: Number(metric?.top10_accuracy_per_mille || 0),
    mean_target_rank_per_mille: Number(metric?.mean_target_rank_per_mille || 0),
    mean_target_margin_q8: Number(metric?.mean_target_margin_q8 || 0),
    min_targets: Number(metric?.min_targets || 0),
    min_top5_per_mille: Number(metric?.min_top5_per_mille || 0),
  };
}

function nativeTaskConfidenceEval(tasks) {
  const taskMetrics = Object.fromEntries(
    REQUIRED_TASKS.map((task) => [task, nativeTaskConfidenceMetric(tasks?.[task])]),
  );
  return {
    tasks: taskMetrics,
    weakest_top5: weakestNativeTaskMetric(taskMetrics, "top5_accuracy_per_mille"),
    weakest_margin: weakestNativeTaskMetric(taskMetrics, "mean_target_margin_q8"),
  };
}

function weakestNativeTaskMetric(taskMetrics, field) {
  const entries = Object.entries(taskMetrics)
    .map(([task, metric]) => ({ task, value: Number(metric?.[field] || 0) }))
    .filter((item) => Number.isFinite(item.value));
  if (entries.length === 0) {
    return { task: "", [field]: null };
  }
  entries.sort((left, right) => left.value - right.value || left.task.localeCompare(right.task));
  const weakest = entries[0];
  return { task: weakest.task, [field]: weakest.value };
}

function symbolicImageTokenConfidence(corpusContractReport, curriculumStagesReport, config) {
  const requiredChannels = config.requireImageTokenChannels.map((channel) => String(channel));
  const required = requiredChannels.length > 0;
  const corpusIntegrity = corpusContractReport.image_channel_marker_integrity || {};
  const corpusOk =
    corpusContractReport.present === true &&
    corpusIntegrity.ok === true &&
    Number(corpusIntegrity.checked_records || 0) > 0 &&
    requiredChannels.every((channel) => Number(corpusIntegrity.by_channel?.[channel]?.found_markers || 0) > 0);
  const stageSummaries = Array.isArray(curriculumStagesReport.stages)
    ? curriculumStagesReport.stages.map((stage) => {
        const integrity = stage.image_channel_marker_integrity || {};
        return {
          index: Number(stage.index || 0),
          stage_name: stage.stage_name || "",
          expected_stage_name: stage.expected_stage_name || "",
          ok: integrity.ok === true,
          checked_records: Number(integrity.checked_records || 0),
          required_channels: Array.isArray(integrity.required_channels)
            ? integrity.required_channels.map((channel) => String(channel))
            : [],
          by_channel: compactImageChannelMarkerSummary(integrity.by_channel || {}),
        };
      })
    : [];
  const curriculumPresent = curriculumStagesReport.present === true;
  const curriculumRequired = config.requireCurriculumStages;
  const curriculumOk = curriculumPresent
    ? stageSummaries.length > 0 &&
      stageSummaries.every(
        (stage) =>
          stage.ok &&
          stage.checked_records > 0 &&
          requiredChannels.every((channel) => Number(stage.by_channel?.[channel]?.found_markers || 0) > 0),
      )
    : !curriculumRequired;
  return {
    required,
    ok: !required || (corpusOk && curriculumOk),
    required_channels: requiredChannels,
    corpus: {
      present: corpusContractReport.present === true,
      ok: corpusOk,
      checked_records: Number(corpusIntegrity.checked_records || 0),
      missing_image_markers: Number(corpusIntegrity.missing_image_markers || 0),
      missing_channel_markers: Number(corpusIntegrity.missing_channel_markers || 0),
      short_channel_payloads: Number(corpusIntegrity.short_channel_payloads || 0),
      bad_channel_payloads: Number(corpusIntegrity.bad_channel_payloads || 0),
      channel_order_mismatches: Number(corpusIntegrity.channel_order_mismatches || 0),
      by_channel: compactImageChannelMarkerSummary(corpusIntegrity.by_channel || {}),
    },
    curriculum: {
      present: curriculumPresent,
      required: curriculumRequired,
      ok: curriculumOk,
      stage_count: Number(curriculumStagesReport.stage_count || 0),
      stages: stageSummaries,
    },
  };
}

function compactImageChannelMarkerSummary(byChannel) {
  return Object.fromEntries(
    Object.entries(byChannel)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([channel, summary]) => [
        channel,
        {
          checked_records: Number(summary?.checked_records || 0),
          found_markers: Number(summary?.found_markers || 0),
          missing_channel_markers: Number(summary?.missing_channel_markers || 0),
          short_channel_payloads: Number(summary?.short_channel_payloads || 0),
          bad_channel_payloads: Number(summary?.bad_channel_payloads || 0),
          channel_order_mismatches: Number(summary?.channel_order_mismatches || 0),
        },
      ]),
  );
}

function sourceSummaryHasEvidence(sourceSummary) {
  if (!sourceSummary || typeof sourceSummary !== "object") {
    return false;
  }
  return sourceSummary.text_queries_have_source_text === true &&
    sourceSummary.image_queries_have_source_text === true &&
    sourceSummary.sample_queries_have_source_text === true;
}

function curriculumIdentityBindingReady(identityBinding) {
  const stages = Object.values(identityBinding || {});
  if (stages.length === 0) {
    return false;
  }
  return stages.every((stage) =>
    Object.values(stage.tasks || {}).every((task) =>
      task.preserved === true &&
      Number(task.selected?.rows || 0) > 0 &&
      Number(task.selected?.spirits || 0) > 0,
    ),
  );
}

function sampleSummaryHasEvidence(sampleSummary) {
  const sampleCount = Number(sampleSummary?.samples || 0);
  if (sampleCount <= 0) {
    return false;
  }
  return (
    sampleSummary?.source_text_evidence === true &&
    sampleSummary?.generated_text_source_evidence === true &&
    Number(sampleSummary?.min_source_text_chars || 0) > 0 &&
    sampleSummary?.text_image_agreement === true &&
    sampleSummary?.generated_text_image_agreement === true &&
    sampleSummary?.signature_retrieval_agreement === true &&
    sampleSummary?.expected_image_agreement === true &&
    sampleSummary?.expected_generated_text_agreement === true &&
    Number(sampleSummary?.min_prompt_text_margin || 0) > 0 &&
    Number(sampleSummary?.min_generated_text_margin || 0) > 0
  );
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function writeJson(filePath, row) {
  const dir = path.dirname(filePath);
  if (dir && dir !== ".") {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`);
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const evalReport = checkEvalTrace(readJson(config.evalPath), config.evalPath, config);
  const corpusContractReport = checkCorpusContract(config);
  const retrievalReport = checkRetrievalHeadEval(
    readJson(config.retrievalHeadEvalPath),
    config.retrievalHeadEvalPath,
    config,
  );
  const sampleReport = checkSampleBinding(readJson(config.sampleBindingPath), config.sampleBindingPath, retrievalReport);
  const integrityReport = checkGenerationIntegrity(readJson(config.generationIntegrityPath), config.generationIntegrityPath);
  const identityInferenceReport = config.identityInferencePath
    ? checkIdentityInference(readJson(config.identityInferencePath), config.identityInferencePath, retrievalReport, config)
    : absentIdentityInference();
  const curriculumStagesReport = config.curriculumStagesPath
    ? checkCurriculumStages(readJson(config.curriculumStagesPath), config.curriculumStagesPath, config)
    : absentCurriculumStages();
  const denoiseBridgeReport = config.denoiseBridgePath
    ? checkDenoiseBridge(readJson(config.denoiseBridgePath), config.denoiseBridgePath, config, retrievalReport, sampleReport)
    : absentDenoiseBridge();
  const groundedCorpusReport = config.groundedCorpusPath
    ? checkGroundedCorpus(readJson(config.groundedCorpusPath), config.groundedCorpusPath, config)
    : absentGroundedCorpus();
  const generativeReport = config.generativeEvalPath
    ? checkGenerativeEval(config.generativeEvalPath, config, retrievalReport)
    : absentGenerativeEval();
  const confidenceTrace = buildConfidenceTrace({
    evalReport,
    corpusContractReport,
    retrievalReport,
    sampleReport,
    identityInferenceReport,
    curriculumStagesReport,
    denoiseBridgeReport,
    groundedCorpusReport,
    generativeReport,
    config,
  });
  const errors = [
    ...evalReport.errors,
    ...corpusContractReport.errors,
    ...retrievalReport.errors,
    ...sampleReport.errors,
    ...integrityReport.errors,
    ...identityInferenceReport.errors,
    ...curriculumStagesReport.errors,
    ...denoiseBridgeReport.errors,
    ...groundedCorpusReport.errors,
    ...generativeReport.errors,
  ];
  if (config.requireIdentityInference && !identityInferenceReport.present) {
    errors.push("identity inference artifact is required");
  }
  if (config.requireCurriculumStages && !curriculumStagesReport.present) {
    errors.push("curriculum stages artifact is required");
  }
  if (config.requireDenoiseBridge && !denoiseBridgeReport.present) {
    errors.push("denoise bridge artifact is required");
  }
  if (config.requireGroundedCorpus && !groundedCorpusReport.present) {
    errors.push("grounded corpus artifact is required");
  }
  if (config.requireGenerativeEval && !generativeReport.present) {
    errors.push("generative eval artifact is required");
  }
  if (config.requireConfidenceTrace) {
    errors.push(...confidenceTrace.errors.map((error) => `confidence trace: ${error}`));
  }
  const result = {
    schema: "nsrl.solomon_v2_quality_report.v1",
    ok: errors.length === 0,
    binding_spine_ready:
      retrievalReport.ok &&
      sampleReport.ok &&
      integrityReport.ok &&
      identityInferenceReport.ok &&
      denoiseBridgeReport.ok,
    identity_inference_ready: identityInferenceReport.present && identityInferenceReport.ok,
    curriculum_ready: curriculumStagesReport.present && curriculumStagesReport.ok,
    denoise_bridge_ready: denoiseBridgeReport.present && denoiseBridgeReport.ok,
    grounded_corpus_ready: groundedCorpusReport.present && groundedCorpusReport.ok,
    product_generation_ready:
      generativeReport.present && generativeReport.ok && generativeReport.product_floor?.ok === true,
    confidence_trace_ready: confidenceTrace.ok,
    corpus_contract_ready: corpusContractReport.present && corpusContractReport.ok,
    task_eval_ready: evalReport.ok,
    architecture_profile_ready:
      evalReport.architecture.ok &&
      evalReport.architecture.has_profile &&
      retrievalReport.class_retrieval_head.text_head &&
      retrievalReport.class_retrieval_head.image_head,
    promoted_small_profile_ready:
      evalReport.architecture.promoted_small_profile.ok &&
      retrievalReport.class_retrieval_head.text_head &&
      retrievalReport.class_retrieval_head.image_head,
    model_only_quality_floor: {
      require_promoted_small_profile: config.requirePromotedSmallProfile,
      min_total_top5_per_mille: config.minTotalTop5PerMille,
      min_text_top5_per_mille: config.minTextTop5PerMille,
      min_image_top5_per_mille: config.minImageTop5PerMille,
      min_task_targets: config.minTaskTargets,
      min_task_top5_per_mille: config.minTaskTop5PerMille,
      min_phase_targets: config.minPhaseTargets,
      require_heldout_prompts: config.requireHeldoutPrompts,
      min_heldout_prompt_rows: config.minHeldoutPromptRows,
      min_match_yes_top1: config.minMatchYesTop1,
      min_match_no_top1: config.minMatchNoTop1,
      min_match_no_image_top1: config.minMatchNoImageTop1,
      min_match_no_prompt_top1: config.minMatchNoPromptTop1,
      min_retrieval_margin: config.minRetrievalMargin,
      min_d_model: config.minDModel,
      min_heads: config.minHeads,
      min_hidden_dim: config.minHiddenDim,
      min_transformer_layers: config.minTransformerLayers,
      min_context_seq_len: config.minContextSeqLen,
      require_corpus_version: config.requireCorpusVersion,
      require_image_token_profile: config.requireImageTokenProfile,
      require_image_token_channels: config.requireImageTokenChannels,
      require_image_channel_token_stats: config.requireImageChannelTokenStats,
      min_image_channel_distinct_bins: config.minImageChannelDistinctBins,
      require_identity_inference: config.requireIdentityInference,
      require_curriculum_stages: config.requireCurriculumStages,
      require_curriculum_stage_names: config.requireCurriculumStageNames,
      require_denoise_bridge: config.requireDenoiseBridge,
      require_denoise_output_identity: config.requireDenoiseOutputIdentity,
      min_denoise_bridge_unique_targets: config.minDenoiseBridgeUniqueTargets,
      require_grounded_corpus: config.requireGroundedCorpus,
      min_grounded_source_overlap_tokens: config.minGroundedSourceOverlapTokens,
      min_grounded_attribute_source_overlap_tokens: config.minGroundedAttributeSourceOverlapTokens,
      max_grounded_source_placeholder_rows: config.maxGroundedSourcePlaceholderRows,
      max_grounded_attribute_generic_rank_rows: config.maxGroundedAttributeGenericRankRows,
      require_confidence_trace: config.requireConfidenceTrace,
      require_generative_eval: config.requireGenerativeEval,
      require_generative_output_identity: config.requireGenerativeOutputIdentity,
      min_generated_top5_per_mille: config.minGeneratedTop5PerMille,
      min_generated_top5_16_per_mille: config.minGeneratedTop516PerMille,
      effective_min_generated_top5_16_per_mille: effectiveMinGeneratedTop516PerMille(config),
      min_generated_top5_px_per_mille: config.minGeneratedTop5PxPerMille,
      min_generated_retrieval_top1_per_mille: config.minGeneratedRetrievalTop1PerMille,
      min_generated_retrieval_top5_per_mille: config.minGeneratedRetrievalTop5PerMille,
      min_generated_retrieval_margin: config.minGeneratedRetrievalMargin,
      min_generated_prompt_rows: config.minGeneratedPromptRows,
      min_latent_top5_per_mille: config.minLatentTop5PerMille,
      max_generated_mean_rank_q8: config.maxGeneratedMeanRankQ8,
      max_generated_mean_rank_16_q8: config.maxGeneratedMeanRank16Q8,
      max_generated_mean_rank_px_q8: config.maxGeneratedMeanRankPxQ8,
      max_generated_mean_target_distance_q8: config.maxGeneratedMeanTargetDistanceQ8,
      max_generated_mean_target_distance_16_q8: config.maxGeneratedMeanTargetDistance16Q8,
      max_generated_mean_target_distance_px_q8: config.maxGeneratedMeanTargetDistancePxQ8,
      met:
        evalReport.ok &&
        corpusContractReport.ok &&
        retrievalReport.ok &&
        curriculumStagesReport.ok &&
        groundedCorpusReport.ok &&
        generativeReport.ok,
    },
    inputs: {
      eval: config.evalPath,
      examples: config.examplesPath,
      manifest: config.manifestPath,
      tokens: config.tokensPath,
      retrieval_head: config.retrievalHeadPath,
      retrieval_head_eval: config.retrievalHeadEvalPath,
      sample_binding: config.sampleBindingPath,
      generation_integrity: config.generationIntegrityPath,
      identity_inference: config.identityInferencePath,
      curriculum_stages: config.curriculumStagesPath,
      denoise_bridge: config.denoiseBridgePath,
      grounded_corpus: config.groundedCorpusPath,
      generative_eval: config.generativeEvalPath,
    },
    attention_eval: dropErrors(evalReport),
    corpus_contract: dropErrors(corpusContractReport),
    retrieval_head_eval: dropErrors(retrievalReport),
    sample_binding: dropErrors(sampleReport),
    generation_integrity: dropErrors(integrityReport),
    identity_inference: dropErrors(identityInferenceReport),
    curriculum_stages: dropErrors(curriculumStagesReport),
    denoise_bridge: dropErrors(denoiseBridgeReport),
    grounded_corpus: dropErrors(groundedCorpusReport),
    generative_eval: dropErrors(generativeReport),
    confidence_trace: dropErrors(confidenceTrace),
    errors,
  };
  if (config.outPath) {
    writeJson(config.outPath, result);
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    process.exit(1);
  }
}

function dropErrors(report) {
  const { errors: _errors, ...rest } = report;
  return rest;
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
