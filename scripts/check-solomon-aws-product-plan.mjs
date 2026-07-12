#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const DEFAULT_REQUIRED_STAGES = [
  "dataset",
  "denoiser",
  "prior",
  "generative-eval",
  "attention-curriculum",
];
const DEFAULT_CURRICULUM_STAGES = [
  "identity",
  "image",
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "explain",
  "hard-negative",
  "native-bind",
];
const DEFAULT_HELDOUT_PROMPTS = "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl";
const DEFAULT_GENERATIVE_GOLD = "data/processed/key-solomon-goetia-latent-v1/gold.tsv";
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

const defaults = {
  runDir: "",
  runEnvPath: "",
  planPath: "",
  promotionPath: "",
  outPath: "",
  requiredStages: DEFAULT_REQUIRED_STAGES,
  requiredCurriculumStages: DEFAULT_CURRICULUM_STAGES,
  requireDryRun: true,
  requireHeldoutPrompts: true,
  requireIdentityInference: true,
  requireGroundedCorpus: true,
  requireDenoiseBridge: true,
  requireDenoiseOutputIdentity: true,
  requireGenerativeEval: true,
  requireGenerativeOutputIdentity: true,
  requireGraviton: true,
  requireS3Artifacts: true,
  requirePromotionBundleCheck: true,
  requireArchitectureProfile: true,
  requirePromotedSmallProfile: true,
  requiredImageTokenProfile: "symbolic16",
  requiredImageTokenChannels: "ink,edge,component,radial,direction",
  requiredAttentionBatchMode: "map-reduce",
  requiredAttentionMapReduceWorkers: "0",
  requireAttentionCpuScaling: true,
  requiredAttentionCpuScalingPolicy: "auto-online-processors",
  minAttentionEffectiveWorkers: 1,
  requireImageChannelTokenStats: true,
  requireDirectionalGroups: true,
  minImageChannelDistinctBins: 2,
  requiredEvalMaxExamples: "none",
  requiredHeldoutPrompts: DEFAULT_HELDOUT_PROMPTS,
  minHeldoutPromptRows: 72,
  minMatchYesTop1: 72,
  minMatchNoTop1: 72,
  minMatchNoImageTop1: 72,
  minMatchNoPromptTop1: 72,
  minRetrievalMargin: 1,
  minGeneratedTop516PerMille: 1,
  minGeneratedRetrievalTop1PerMille: 1000,
  minGeneratedRetrievalTop5PerMille: 1000,
  minGeneratedRetrievalMargin: 1,
  minGenerativeEvalPermille: 190,
  minGenerativeEvalLimit: 72,
  minGeneratedPromptRows: 72,
  maxGeneratedMeanTargetDistance16Q8: 7000000,
  minTaskTargets: "all=72",
  minTaskTop5PerMille: "all=1",
  minPhaseTargets: "all=72",
  minDirectionAccuracyPerMille: "",
  minDirectionTop5PerMille: "all=1",
  minDirectionTop10PerMille: "",
  minSourceOverlapTokens: 2,
  minAttributeSourceOverlapTokens: 8,
  maxSourcePlaceholderRows: 0,
  maxAttributeGenericRankRows: 0,
  minDModel: 128,
  minHeads: 2,
  targetHeadDim: 64,
  minHiddenDim: 256,
  maxHiddenDim: 512,
  minTransformerLayers: 2,
  minContextSeqLen: 384,
  maxDenoiseOutputRetrievalRank: 1,
  minDenoiseOutputRetrievalMargin: 1,
  minDenoiseBridgeUniqueTargets: 2,
  minAttentionSeqLen: 384,
  maxAttentionSeqLen: 768,
  minNativeBindEpochs: 2,
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-aws-product-plan.mjs --run-dir PATH [options]",
      "   or: check-solomon-aws-product-plan.mjs --run-env PATH --plan PATH",
      "",
      "Checks that a dry-run AWS Solomon pipeline plan resolves to the narrow",
      "v2 product path: denoiser/prior/generative eval plus task-marked",
      "attention curriculum with symbolic image tokens and promotion gates.",
      "",
      "Options:",
      "  --out PATH",
      "  --promotion PATH",
      "  --required-stages LIST",
      "  --required-curriculum-stages LIST",
      "  --allow-non-dry-run",
      "  --allow-missing-heldout-prompts",
      "  --allow-missing-identity-inference",
      "  --allow-missing-grounded-corpus",
      "  --allow-missing-denoise-bridge",
      "  --allow-missing-denoise-output-identity",
      "  --allow-missing-generative-eval",
      "  --allow-missing-generative-output-identity",
      "  --allow-non-graviton-runner",
      "  --allow-missing-s3-artifacts",
      "  --allow-missing-promotion-bundle-check",
      "  --allow-missing-architecture-profile",
      "  --allow-missing-promoted-small-profile",
      "  --allow-missing-image-channel-token-stats",
      "  --allow-missing-attention-cpu-scaling",
      "  --required-image-token-profile NAME",
      "  --required-image-token-channels LIST",
      "  --required-attention-batch-mode MODE",
      "  --required-attention-map-reduce-workers N",
      "  --required-attention-cpu-scaling-policy NAME",
      "  --min-attention-effective-workers N",
      "  --min-native-bind-epochs N",
      "  --min-image-channel-distinct-bins N",
      "  --required-eval-max-examples VALUE",
      "  --required-heldout-prompts PATH",
      "  --min-heldout-prompt-rows N",
      "  --min-match-yes-top1 N",
      "  --min-match-no-top1 N",
      "  --min-match-no-image-top1 N",
      "  --min-match-no-prompt-top1 N",
      "  --min-retrieval-margin N",
      "  --min-generated-top5-16-per-mille N",
      "  --min-generated-retrieval-top1-per-mille N",
      "  --min-generated-retrieval-top5-per-mille N",
      "  --min-generated-retrieval-margin N",
      "  --min-generative-eval-permille N",
      "  --min-generative-eval-limit N",
      "  --min-generated-prompt-rows N",
      "  --max-generated-mean-target-distance-16-q8 N",
      "  --min-task-targets SPEC",
      "  --min-task-top5-per-mille SPEC",
      "  --min-phase-targets SPEC",
      "  --min-source-overlap-tokens N",
      "  --min-attribute-source-overlap-tokens N",
      "  --max-source-placeholder-rows N",
      "  --max-attribute-generic-rank-rows N",
      "  --min-d-model N",
      "  --min-heads N",
      "  --target-head-dim N",
      "  --min-hidden-dim N",
      "  --max-hidden-dim N",
      "  --min-transformer-layers N",
      "  --min-context-seq-len N",
      "  --max-denoise-output-retrieval-rank N",
      "  --min-denoise-output-retrieval-margin N",
      "  --min-attention-seq-len N",
      "  --max-attention-seq-len N",
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
    } else if (arg === "--run-dir") {
      config.runDir = requireValue(argv, ++index, arg);
    } else if (arg === "--run-env") {
      config.runEnvPath = requireValue(argv, ++index, arg);
    } else if (arg === "--plan") {
      config.planPath = requireValue(argv, ++index, arg);
    } else if (arg === "--promotion") {
      config.promotionPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--required-stages") {
      config.requiredStages = parseList(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--required-curriculum-stages") {
      config.requiredCurriculumStages = parseList(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--allow-non-dry-run") {
      config.requireDryRun = false;
    } else if (arg === "--allow-missing-heldout-prompts") {
      config.requireHeldoutPrompts = false;
    } else if (arg === "--allow-missing-identity-inference") {
      config.requireIdentityInference = false;
    } else if (arg === "--allow-missing-grounded-corpus") {
      config.requireGroundedCorpus = false;
    } else if (arg === "--allow-missing-denoise-bridge") {
      config.requireDenoiseBridge = false;
    } else if (arg === "--allow-missing-denoise-output-identity") {
      config.requireDenoiseOutputIdentity = false;
    } else if (arg === "--allow-missing-generative-eval") {
      config.requireGenerativeEval = false;
    } else if (arg === "--allow-missing-generative-output-identity") {
      config.requireGenerativeOutputIdentity = false;
    } else if (arg === "--allow-non-graviton-runner") {
      config.requireGraviton = false;
    } else if (arg === "--allow-missing-s3-artifacts") {
      config.requireS3Artifacts = false;
    } else if (arg === "--allow-missing-promotion-bundle-check") {
      config.requirePromotionBundleCheck = false;
    } else if (arg === "--allow-missing-architecture-profile") {
      config.requireArchitectureProfile = false;
    } else if (arg === "--allow-missing-promoted-small-profile") {
      config.requirePromotedSmallProfile = false;
    } else if (arg === "--allow-missing-image-channel-token-stats") {
      config.requireImageChannelTokenStats = false;
    } else if (arg === "--allow-missing-attention-cpu-scaling") {
      config.requireAttentionCpuScaling = false;
    } else if (arg === "--required-image-token-profile") {
      config.requiredImageTokenProfile = requireValue(argv, ++index, arg);
    } else if (arg === "--required-image-token-channels") {
      config.requiredImageTokenChannels = parseList(requireValue(argv, ++index, arg), arg).join(",");
    } else if (arg === "--required-attention-batch-mode") {
      config.requiredAttentionBatchMode = requireValue(argv, ++index, arg);
    } else if (arg === "--required-attention-map-reduce-workers") {
      config.requiredAttentionMapReduceWorkers = requireValue(argv, ++index, arg);
    } else if (arg === "--required-attention-cpu-scaling-policy") {
      config.requiredAttentionCpuScalingPolicy = requireValue(argv, ++index, arg);
    } else if (arg === "--min-attention-effective-workers") {
      config.minAttentionEffectiveWorkers = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-native-bind-epochs") {
      config.minNativeBindEpochs = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-image-channel-distinct-bins") {
      config.minImageChannelDistinctBins = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--required-eval-max-examples") {
      config.requiredEvalMaxExamples = requireValue(argv, ++index, arg);
    } else if (arg === "--required-heldout-prompts") {
      config.requiredHeldoutPrompts = requireValue(argv, ++index, arg);
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
    } else if (arg === "--min-generated-top5-16-per-mille") {
      config.minGeneratedTop516PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-top1-per-mille") {
      config.minGeneratedRetrievalTop1PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-top5-per-mille") {
      config.minGeneratedRetrievalTop5PerMille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-retrieval-margin") {
      config.minGeneratedRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generative-eval-permille") {
      config.minGenerativeEvalPermille = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generative-eval-limit") {
      config.minGenerativeEvalLimit = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-generated-prompt-rows") {
      config.minGeneratedPromptRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-generated-mean-target-distance-16-q8") {
      config.maxGeneratedMeanTargetDistance16Q8 = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-task-targets") {
      config.minTaskTargets = requireValue(argv, ++index, arg);
    } else if (arg === "--min-task-top5-per-mille") {
      config.minTaskTop5PerMille = requireValue(argv, ++index, arg);
    } else if (arg === "--min-phase-targets") {
      config.minPhaseTargets = requireValue(argv, ++index, arg);
    } else if (arg === "--min-source-overlap-tokens") {
      config.minSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-attribute-source-overlap-tokens") {
      config.minAttributeSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-source-placeholder-rows") {
      config.maxSourcePlaceholderRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-attribute-generic-rank-rows") {
      config.maxAttributeGenericRankRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-d-model") {
      config.minDModel = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heads") {
      config.minHeads = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--target-head-dim") {
      config.targetHeadDim = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-hidden-dim") {
      config.minHiddenDim = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-hidden-dim") {
      config.maxHiddenDim = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-transformer-layers") {
      config.minTransformerLayers = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-context-seq-len") {
      config.minContextSeqLen = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-denoise-output-retrieval-rank") {
      config.maxDenoiseOutputRetrievalRank = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-denoise-output-retrieval-margin") {
      config.minDenoiseOutputRetrievalMargin = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-attention-seq-len") {
      config.minAttentionSeqLen = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-attention-seq-len") {
      config.maxAttentionSeqLen = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.runDir) {
    config.runEnvPath ||= path.join(config.runDir, "run.env");
    config.planPath ||= path.join(config.runDir, "plan.tsv");
    config.promotionPath ||= path.join(config.runDir, "promotion.tsv");
  }
  if (!config.runEnvPath) {
    throw new Error("--run-dir or --run-env is required");
  }
  if (!config.planPath) {
    throw new Error("--run-dir or --plan is required");
  }
  config.promotionPath ||= path.join(path.dirname(config.runEnvPath), "promotion.tsv");
  if (config.minAttentionSeqLen > config.maxAttentionSeqLen) {
    throw new Error("--min-attention-seq-len cannot exceed --max-attention-seq-len");
  }
  if (config.minContextSeqLen > config.maxAttentionSeqLen) {
    throw new Error("--min-context-seq-len cannot exceed --max-attention-seq-len");
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

function parseList(value, flag) {
  const entries = String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (entries.length === 0) {
    throw new Error(`${flag} selected no entries`);
  }
  return entries;
}

function readKeyValueFile(filePath) {
  const entries = {};
  const lines = fs.readFileSync(filePath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }
    const splitIndex = line.indexOf("=");
    if (splitIndex < 0) {
      continue;
    }
    entries[line.slice(0, splitIndex)] = line.slice(splitIndex + 1);
  }
  return entries;
}

function readPlan(filePath) {
  const rows = [];
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  if (lines.length === 0 || lines[0] !== "stage\tcommand") {
    throw new Error(`${filePath} must start with stage\\tcommand`);
  }
  for (let index = 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }
    const tab = line.indexOf("\t");
    if (tab < 0) {
      throw new Error(`${filePath}:${index + 1}: expected tab-separated stage and command`);
    }
    rows.push({
      stage: line.slice(0, tab),
      command: line.slice(tab + 1),
    });
  }
  return rows;
}

function readPromotionManifest(filePath) {
  const rows = [];
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  if (lines.length === 0 || lines[0] !== "product\tstage\tartifact\tpath\trequired") {
    throw new Error(`${filePath} must start with product\\tstage\\tartifact\\tpath\\trequired`);
  }
  for (let index = 1; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim()) {
      continue;
    }
    const fields = line.split("\t");
    if (fields.length !== 5) {
      throw new Error(`${filePath}:${index + 1}: expected 5 tab-separated fields`);
    }
    rows.push({
      product: fields[0],
      stage: fields[1],
      artifact: fields[2],
      path: fields[3],
      required: fields[4],
    });
  }
  return rows;
}

function requireField(env, key, expected, errors) {
  const actual = env[key];
  if (actual !== expected) {
    errors.push(`${key} ${JSON.stringify(actual ?? "")} != ${JSON.stringify(expected)}`);
  }
}

function requireIntegerAtLeast(env, key, minimum, errors) {
  const actual = Number(env[key]);
  if (!Number.isInteger(actual) || actual < minimum) {
    errors.push(`${key} ${JSON.stringify(env[key] ?? "")} < ${minimum}`);
  }
}

function parseRustUsizeConst(source, name) {
  const pattern = new RegExp(`pub\\s+const\\s+${name}\\s*:\\s*usize\\s*=\\s*([0-9]+)\\s*;`);
  const match = source.match(pattern);
  return match ? Number(match[1]) : null;
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

function checkTrainCoreArchitecture(config, env, errors) {
  const sourcePath = path.resolve(env.repo_root || process.cwd(), "crates/nsrl-train-core/src/lib.rs");
  const summary = {
    path: sourcePath,
    present: false,
    d_model: 0,
    heads: 0,
    head_dim: 0,
    head_dim_power_of_four: false,
    hidden_dim: 0,
    target: {
      d_model: config.minDModel,
      heads: config.minHeads,
      head_dim: config.targetHeadDim,
      hidden_dim_min: config.minHiddenDim,
      hidden_dim_max: config.maxHiddenDim,
    },
    ok: false,
    errors: [],
  };

  let source = "";
  try {
    source = fs.readFileSync(sourcePath, "utf8");
    summary.present = true;
  } catch (error) {
    summary.errors.push(`train-core architecture source could not be read: ${error.message}`);
  }

  if (source) {
    summary.d_model = parseRustUsizeConst(source, "MINI_TRANSFORMER_D_MODEL") || 0;
    summary.heads = parseRustUsizeConst(source, "MINI_TRANSFORMER_HEADS") || 0;
    summary.hidden_dim = parseRustUsizeConst(source, "MINI_TRANSFORMER_HIDDEN_DIM") || 0;
    if (summary.d_model > 0 && summary.heads > 0 && summary.d_model % summary.heads === 0) {
      summary.head_dim = summary.d_model / summary.heads;
    }
    summary.head_dim_power_of_four = isPowerOfFour(summary.head_dim);

    if (summary.d_model !== config.minDModel) {
      summary.errors.push(`train-core d_model ${summary.d_model} != ${config.minDModel}`);
    }
    if (summary.heads !== config.minHeads) {
      summary.errors.push(`train-core heads ${summary.heads} != ${config.minHeads}`);
    }
    if (summary.head_dim !== config.targetHeadDim) {
      summary.errors.push(`train-core head_dim ${summary.head_dim} != ${config.targetHeadDim}`);
    }
    if (!summary.head_dim_power_of_four) {
      summary.errors.push(`train-core head_dim ${summary.head_dim} is not a power of four`);
    }
    if (summary.hidden_dim < config.minHiddenDim || summary.hidden_dim > config.maxHiddenDim) {
      summary.errors.push(
        `train-core hidden_dim ${summary.hidden_dim} outside ${config.minHiddenDim}-${config.maxHiddenDim}`,
      );
    }
  }

  summary.ok = summary.errors.length === 0;
  if (config.requirePromotedSmallProfile) {
    errors.push(...summary.errors);
  }
  return summary;
}

function checkCurriculumDenoiseRunner(config, env, errors) {
  const configuredFloor = Number(env.attention_denoise_min_unique_targets || 0);
  const requiredBridgePairCount = Math.max(
    config.minDenoiseBridgeUniqueTargets,
    Number.isInteger(configuredFloor) ? configuredFloor : 0,
  );
  const sourcePath = path.resolve(env.repo_root || process.cwd(), "scripts/run-solomon-attention-curriculum-smoke.sh");
  const summary = {
    path: sourcePath,
    present: false,
    min_unique_targets_arg: false,
    quality_min_unique_targets_arg: false,
    bridge_pair_count: 0,
    required_bridge_pair_count: requiredBridgePairCount,
    ok: false,
    errors: [],
  };

  let source = "";
  try {
    source = fs.readFileSync(sourcePath, "utf8");
    summary.present = true;
  } catch (error) {
    summary.errors.push(`curriculum runner source could not be read: ${error.message}`);
  }

  if (source) {
    summary.min_unique_targets_arg = source.includes('--min-unique-targets "$attention_denoise_min_unique_targets"');
    summary.quality_min_unique_targets_arg = source.includes(
      '--min-denoise-bridge-unique-targets "$quality_min_denoise_bridge_unique_targets"',
    );
    summary.bridge_pair_count = countLiteral(source, '--pair "$joint_out/');

    if (!summary.min_unique_targets_arg) {
      summary.errors.push("curriculum runner does not pass denoise --min-unique-targets");
    }
    if (!summary.quality_min_unique_targets_arg) {
      summary.errors.push("curriculum runner does not pass quality --min-denoise-bridge-unique-targets");
    }
    if (summary.bridge_pair_count < summary.required_bridge_pair_count) {
      summary.errors.push(
        `curriculum runner denoise bridge pairs ${summary.bridge_pair_count} < ${summary.required_bridge_pair_count}`,
      );
    }
  }

  summary.ok = summary.present && summary.errors.length === 0;
  if (!summary.ok) {
    errors.push(...summary.errors);
  }
  return summary;
}

function countLiteral(source, needle) {
  let count = 0;
  let offset = 0;
  while (true) {
    const index = source.indexOf(needle, offset);
    if (index < 0) {
      return count;
    }
    count += 1;
    offset = index + needle.length;
  }
}

function requireCommandContains(command, needle, stage, errors) {
  if (!command.includes(needle)) {
    errors.push(`${stage} command missing ${needle}`);
  }
}

function checkSequence(actual, expected, label, errors) {
  if (actual.join(",") !== expected.join(",")) {
    errors.push(`${label} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function referencedPathCandidates(ref, baseDirs) {
  const bases = baseDirs.filter(Boolean);
  const candidates = path.isAbsolute(ref)
    ? [path.resolve(ref)]
    : [
        path.resolve(ref),
        ...bases.map((baseDir) => path.resolve(baseDir, ref)),
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

function sameReferencedPath(ref, expected, baseDirs) {
  if (!ref || !expected) {
    return null;
  }
  const expectedCandidates = referencedPathCandidates(expected, baseDirs);
  const refCandidates = referencedPathCandidates(ref, baseDirs);
  return refCandidates.some((candidate) => expectedCandidates.includes(candidate));
}

function resolveReferencedPath(ref, baseDirs) {
  const candidates = referencedPathCandidates(ref, baseDirs);
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0] || "";
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

function readValidPromptRows(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  const rows = [];
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (!line.trim()) {
      continue;
    }
    const row = JSON.parse(line);
    const spiritId = Number(row.spirit_id);
    const prompt = String(row.text || row.prompt || "");
    if (Number.isInteger(spiritId) && spiritId >= 1 && prompt) {
      rows.push({ ...row, spirit_id: spiritId, text: prompt, index });
    }
  }
  return rows;
}

function countHeldoutPromptRows(filePath) {
  return readValidPromptRows(filePath).length;
}

function readGenerativeGoldHashes(filePath) {
  if (!filePath || !fs.existsSync(filePath)) {
    return new Set();
  }
  const hashes = new Set();
  for (const line of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const first = line.trim().split("\t")[0];
    if (!first || first === "prompt_hash" || first.startsWith("#")) {
      continue;
    }
    hashes.add(first.toLowerCase());
  }
  return hashes;
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

function generativePromptPartition(prompt, evalPermille, goldHashes) {
  if (goldHashes.has(String(prompt.prompt_hash || "").toLowerCase())) {
    return "gold";
  }
  if (isGenerativeHeldoutPrompt(prompt)) {
    return "eval";
  }
  const bucket = prompt.tier === "tier-cluster-holdout"
    ? hashParts32(["solomon-prompt-split-v1", "cluster", prompt.cluster]) % 1000
    : Number(prompt.bucket);
  return bucket < evalPermille ? "eval" : "train";
}

function summarizeGenerativeEvalPromptSelection(promptRows, evalPermille, limit, goldHashes) {
  const mapped = promptRows.map((prompt) => ({
    ...prompt,
    partition: generativePromptPartition(prompt, evalPermille, goldHashes),
  }));
  const eligible = mapped.filter((prompt) => prompt.partition === "eval" && isGenerativeHeldoutPrompt(prompt));
  const candidates = generativeEvalSelectionCandidates(mapped, eligible, limit);
  const selected = balancedGenerativePromptSelection(candidates, limit);
  const evalTargets = promptTargetSet(candidates);
  const eligibleTargets = promptTargetSet(eligible);
  const selectedTargets = promptTargetSet(selected);
  const selectedEligible = selected.filter(isGenerativeHeldoutPrompt);
  const selectedEligibleTargets = promptTargetSet(selectedEligible);
  return {
    eval_prompt_rows: candidates.length,
    eval_unique_targets: evalTargets.size,
    eligible_prompt_rows: eligible.length,
    eligible_unique_targets: eligibleTargets.size,
    selected_prompt_rows: selected.length,
    selected_unique_targets: selectedTargets.size,
    selected_prompt_eligible_rows: selectedEligible.length,
    selected_eligible_unique_targets: selectedEligibleTargets.size,
    selected_prompt_tiers: countBy(selected, (prompt) => prompt.tier || "unknown"),
    selected_prompt_sources: countBy(selected, (prompt) => prompt.source || "unknown"),
    missing_targets: Array.from({ length: 72 }, (_value, index) => index + 1).filter((id) => !selectedTargets.has(id)),
    missing_eligible_targets: Array.from({ length: 72 }, (_value, index) => index + 1).filter(
      (id) => !selectedEligibleTargets.has(id),
    ),
  };
}

function generativeEvalSelectionCandidates(mapped, eligible, limit) {
  const requiredTargets = Math.min(Number(limit || 0), 72);
  if (uniquePromptTargets(eligible) >= requiredTargets) {
    return sortGenerativePromptsForSelection(eligible);
  }
  return sortGenerativePromptsForSelection(mapped.filter((prompt) => prompt.partition === "eval"));
}

function sortGenerativePromptsForSelection(prompts) {
  return [...prompts].sort((left, right) => {
    const leftKey = `${left.tier}:${left.prompt_hash}`;
    const rightKey = `${right.tier}:${right.prompt_hash}`;
    return leftKey.localeCompare(rightKey);
  });
}

function balancedGenerativePromptSelection(candidates, limit) {
  const byTier = new Map();
  for (const prompt of candidates) {
    if (!byTier.has(prompt.tier)) {
      byTier.set(prompt.tier, []);
    }
    byTier.get(prompt.tier).push(prompt);
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

function isGenerativeHeldoutPrompt(prompt) {
  const tier = String(prompt.tier || "").toLowerCase();
  const source = String(prompt.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function uniquePromptTargets(prompts) {
  return promptTargetSet(prompts).size;
}

function promptTargetSet(prompts) {
  return new Set(
    prompts
      .map((prompt) => Number(prompt.spirit_id || 0))
      .filter((id) => Number.isInteger(id) && id > 0),
  );
}

function countBy(rows, keyFn) {
  const out = {};
  for (const row of rows) {
    const key = String(keyFn(row) || "unknown");
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}

function checkHeldoutPromptArtifact(config, env, baseDirs, errors) {
  const prompts = env.attention_heldout_prompts || "";
  const required = Boolean(config.requireHeldoutPrompts);
  const summary = {
    required,
    prompts,
    required_prompts: config.requiredHeldoutPrompts || "",
    prompts_match: null,
    resolved_prompts: "",
    prompts_present: false,
    prompts_hash: "",
    prompt_rows_counted: 0,
  };

  if (!required && !prompts) {
    return summary;
  }
  if (!prompts) {
    errors.push("attention_heldout_prompts is missing");
    return summary;
  }
  if (["none", "0", "false"].includes(String(prompts).toLowerCase())) {
    if (required) {
      errors.push(`attention_heldout_prompts ${JSON.stringify(prompts)} disables held-out prompt evidence`);
    }
    return summary;
  }
  if (required && config.requiredHeldoutPrompts) {
    summary.prompts_match = sameReferencedPath(prompts, config.requiredHeldoutPrompts, baseDirs);
    if (summary.prompts_match === false) {
      errors.push(
        `attention_heldout_prompts ${prompts} does not match required ${config.requiredHeldoutPrompts}`,
      );
    }
  }

  const resolvedPrompts = resolveReferencedPath(prompts, baseDirs);
  summary.resolved_prompts = resolvedPrompts;
  summary.prompts_present = Boolean(resolvedPrompts && fs.existsSync(resolvedPrompts));
  if (!summary.prompts_present) {
    if (required) {
      errors.push(`attention_heldout_prompts ${prompts} could not be resolved`);
    }
    return summary;
  }

  try {
    summary.prompts_hash = fnv64FileHex(resolvedPrompts);
    summary.prompt_rows_counted = countHeldoutPromptRows(resolvedPrompts);
    if (required && summary.prompt_rows_counted < config.minHeldoutPromptRows) {
      errors.push(
        `attention_heldout_prompts rows ${summary.prompt_rows_counted} < ${config.minHeldoutPromptRows}`,
      );
    }
  } catch (error) {
    if (required) {
      errors.push(`attention_heldout_prompts ${resolvedPrompts} could not be read: ${error.message}`);
    }
  }

  return summary;
}

function checkGenerativePromptArtifact(config, env, baseDirs, errors) {
  const prompts = env.generative_prompts || "";
  const heldoutPrompts = env.attention_heldout_prompts || "";
  const required = Boolean(config.requireGenerativeEval || config.minGeneratedPromptRows > 0);
  const summary = {
    required,
    prompts,
    heldout_prompts: heldoutPrompts,
    required_prompts: config.requiredHeldoutPrompts || "",
    prompts_match: null,
    heldout_prompts_match: null,
    resolved_prompts: "",
    prompts_present: false,
    prompts_hash: "",
    prompt_rows_counted: 0,
    eval_prompt_rows: 0,
    eval_unique_targets: 0,
    eligible_prompt_rows: 0,
    eligible_unique_targets: 0,
    selected_prompt_rows: 0,
    selected_unique_targets: 0,
    selected_prompt_eligible_rows: 0,
    selected_eligible_unique_targets: 0,
    selected_prompt_tiers: {},
    selected_prompt_sources: {},
    missing_targets: [],
    missing_eligible_targets: [],
  };

  if (!required && !prompts) {
    return summary;
  }
  if (!prompts) {
    errors.push("generative_prompts is missing");
    return summary;
  }
  if (["none", "0", "false"].includes(String(prompts).toLowerCase())) {
    if (required) {
      errors.push(`generative_prompts ${JSON.stringify(prompts)} disables held-out generation evidence`);
    }
    return summary;
  }
  if (config.requiredHeldoutPrompts) {
    summary.prompts_match = sameReferencedPath(prompts, config.requiredHeldoutPrompts, baseDirs);
    if (required && summary.prompts_match === false) {
      errors.push(`generative_prompts ${prompts} does not match required ${config.requiredHeldoutPrompts}`);
    }
  }
  if (heldoutPrompts) {
    summary.heldout_prompts_match = sameReferencedPath(prompts, heldoutPrompts, baseDirs);
    if (required && summary.heldout_prompts_match === false) {
      errors.push(`generative_prompts ${prompts} does not match attention_heldout_prompts ${heldoutPrompts}`);
    }
  }

  const resolvedPrompts = resolveReferencedPath(prompts, baseDirs);
  summary.resolved_prompts = resolvedPrompts;
  summary.prompts_present = Boolean(resolvedPrompts && fs.existsSync(resolvedPrompts));
  if (!summary.prompts_present) {
    if (required) {
      errors.push(`generative_prompts ${prompts} could not be resolved`);
    }
    return summary;
  }

  try {
    summary.prompts_hash = fnv64FileHex(resolvedPrompts);
    const promptRows = readValidPromptRows(resolvedPrompts);
    summary.prompt_rows_counted = promptRows.length;
    if (required && summary.prompt_rows_counted < config.minGeneratedPromptRows) {
      errors.push(`generative_prompts rows ${summary.prompt_rows_counted} < ${config.minGeneratedPromptRows}`);
    }
    const evalPermille = Number(env.generative_eval_permille || 0);
    const limit = Number(env.generative_limit || 0);
    if (Number.isInteger(evalPermille) && evalPermille > 0 && Number.isInteger(limit) && limit > 0) {
      const goldPath = resolveReferencedPath(DEFAULT_GENERATIVE_GOLD, baseDirs);
      const selection = summarizeGenerativeEvalPromptSelection(
        promptRows,
        evalPermille,
        limit,
        readGenerativeGoldHashes(goldPath),
      );
      Object.assign(summary, selection);
      if (required && summary.selected_prompt_rows < config.minGeneratedPromptRows) {
        errors.push(
          `generative eval selected prompt rows ${summary.selected_prompt_rows} < ${config.minGeneratedPromptRows}`,
        );
      }
      if (required && summary.selected_unique_targets < config.minGeneratedPromptRows) {
        errors.push(
          `generative eval selected unique targets ${summary.selected_unique_targets} < ${config.minGeneratedPromptRows}`,
        );
      }
      if (required && summary.selected_prompt_eligible_rows < config.minGeneratedPromptRows) {
        errors.push(
          `generative eval selected held-out prompt rows ${summary.selected_prompt_eligible_rows} < ${config.minGeneratedPromptRows}`,
        );
      }
      if (required && summary.selected_eligible_unique_targets < config.minGeneratedPromptRows) {
        errors.push(
          `generative eval selected held-out unique targets ${summary.selected_eligible_unique_targets} < ${config.minGeneratedPromptRows}`,
        );
      }
    }
  } catch (error) {
    if (required) {
      errors.push(`generative_prompts ${resolvedPrompts} could not be read: ${error.message}`);
    }
  }

  return summary;
}

function checkPromotionManifest(config, env, baseDirs, errors) {
  const summary = {
    path: config.promotionPath,
    env_path: env.promotion_manifest || "",
    present: false,
    rows: [],
    artifacts: {},
  };
  if (env.promotion_manifest) {
    const matchesEnvPath = sameReferencedPath(config.promotionPath, env.promotion_manifest, baseDirs);
    if (matchesEnvPath === false) {
      errors.push(`promotion manifest ${config.promotionPath} does not match run.env ${env.promotion_manifest}`);
    }
  }
  if (!config.promotionPath || !fs.existsSync(config.promotionPath)) {
    errors.push(`promotion manifest ${config.promotionPath || ""} is missing`);
    return summary;
  }
  let rows = [];
  try {
    rows = readPromotionManifest(config.promotionPath);
  } catch (error) {
    errors.push(error.message);
    return summary;
  }
  summary.present = true;
  summary.rows = rows;
  const byArtifact = new Map();
  for (const row of rows) {
    if (row.product !== "solomon-v1") {
      errors.push(`promotion artifact ${row.artifact} product ${JSON.stringify(row.product)} != "solomon-v1"`);
    }
    if (byArtifact.has(row.artifact)) {
      errors.push(`promotion artifact ${row.artifact} is duplicated`);
    }
    byArtifact.set(row.artifact, row);
    summary.artifacts[row.artifact] = {
      stage: row.stage,
      path: row.path,
      required: row.required,
    };
  }

  const runDir = env.run_dir || config.runDir || path.dirname(config.runEnvPath);
  const attentionOut = env.attention_curriculum_out_dir || path.join(runDir, "attention-curriculum");
  const generativeRun = env.generative_eval_run || path.join(runDir, "generative-eval", "current");
  const expected = [
    ["run_env", "pipeline", config.runEnvPath, "1"],
    ["plan", "pipeline", config.planPath, "1"],
    ["artifacts", "pipeline", path.join(runDir, "artifacts.tsv"), "1"],
    ["quality_report", "attention-curriculum", env.attention_curriculum_quality_report || path.join(attentionOut, "quality-report.json"), "1"],
    ["model", "attention-curriculum", path.join(attentionOut, "model.nsrllmm"), "1"],
    ["corpus_manifest", "attention-curriculum", path.join(attentionOut, "manifest.json"), "1"],
    ["attention_eval", "attention-curriculum", path.join(attentionOut, "attention-eval.json"), "1"],
    ["retrieval_head", "attention-curriculum", path.join(attentionOut, "retrieval-head.json"), "1"],
    ["retrieval_head_eval", "attention-curriculum", path.join(attentionOut, "retrieval-head-eval.json"), "1"],
    ["curriculum_stages", "attention-curriculum", path.join(attentionOut, "curriculum-stages.json"), "1"],
    ["sample_binding", "attention-curriculum", path.join(attentionOut, "prior-sample-binding.json"), "1"],
    ["identity_inference", "attention-curriculum", path.join(attentionOut, "identity-inference.json"), "1"],
    ["grounded_corpus", "attention-curriculum", path.join(attentionOut, "grounded-corpus.json"), "1"],
    ["generation_integrity", "attention-curriculum", path.join(attentionOut, "generation-integrity.json"), "1"],
    ["denoise_bridge", "attention-curriculum", path.join(attentionOut, "denoise-bridge.json"), config.requireDenoiseBridge ? "1" : String(env.attention_require_denoise_bridge || "0")],
    ["denoise_generation_integrity", "attention-curriculum", path.join(attentionOut, "denoise-generation-integrity.json"), config.requireDenoiseOutputIdentity ? "1" : String(env.attention_require_denoise_output_identity || "0")],
    ["run", "generative-eval", generativeRun, config.requireGenerativeEval ? "1" : String(env.attention_require_generative_eval || "0")],
    ["summary", "generative-eval", path.join(generativeRun, "summary.tsv"), config.requireGenerativeEval ? "1" : String(env.attention_require_generative_eval || "0")],
  ];
  for (const [artifact, stage, expectedPath, required] of expected) {
    const row = byArtifact.get(artifact);
    if (!row) {
      errors.push(`promotion manifest missing ${artifact}`);
      continue;
    }
    if (row.stage !== stage) {
      errors.push(`promotion ${artifact} stage ${JSON.stringify(row.stage)} != ${JSON.stringify(stage)}`);
    }
    if (row.required !== required) {
      errors.push(`promotion ${artifact} required ${JSON.stringify(row.required)} != ${JSON.stringify(required)}`);
    }
    if (sameReferencedPath(row.path, expectedPath, baseDirs) === false) {
      errors.push(`promotion ${artifact} path ${row.path} does not match expected ${expectedPath}`);
    }
  }
  return summary;
}

function checkProductPlan(config) {
  const errors = [];
  const env = readKeyValueFile(config.runEnvPath);
  const planRows = readPlan(config.planPath);
  const planStages = planRows.map((row) => row.stage);
  const planByStage = new Map(planRows.map((row) => [row.stage, row.command]));
  const expectedStages = config.requiredStages;
  const hasPromotionBundleCheckStage = planStages.includes("promotion-bundle-check");
  const expectedPlanStages = config.requirePromotionBundleCheck || hasPromotionBundleCheckStage
    ? [...expectedStages, "promotion-bundle-check"]
    : expectedStages;
  const expectedCurriculumStages = config.requiredCurriculumStages;
  const baseDirs = [env.repo_root, path.dirname(config.runEnvPath), process.cwd()];
  const runDir = env.run_dir || config.runDir || path.dirname(config.runEnvPath);
  const trainCoreArchitecture = checkTrainCoreArchitecture(config, env, errors);
  const curriculumDenoiseRunner = checkCurriculumDenoiseRunner(config, env, errors);

  requireField(env, "schema", "nsrl.solomon_aws_pipeline.v1", errors);
  if (config.requireDryRun) {
    requireField(env, "dry_run", "1", errors);
  }
  if (config.requireGraviton) {
    requireField(env, "require_graviton", "1", errors);
    if (env.dry_run === "0") {
      const kernel = env.runner_kernel || "";
      const arch = env.runner_arch || "";
      if (kernel !== "Linux" || !["aarch64", "arm64"].includes(arch)) {
        errors.push(`runner ${kernel}/${arch} is not Linux ARM64/Graviton`);
      }
    }
  }
  if (config.requireS3Artifacts) {
    requireField(env, "require_s3_artifacts", "1", errors);
    if (!String(env.s3_uri || "").startsWith("s3://")) {
      errors.push(`s3_uri ${JSON.stringify(env.s3_uri || "")} must start with s3://`);
    }
    if (!String(env.s3_pipeline_uri || "").startsWith("s3://")) {
      errors.push(`s3_pipeline_uri ${JSON.stringify(env.s3_pipeline_uri || "")} must start with s3://`);
    }
    if (
      env.s3_uri &&
      env.s3_pipeline_uri &&
      !String(env.s3_pipeline_uri).startsWith(`${String(env.s3_uri).replace(/\/+$/, "")}/pipelines/`)
    ) {
      errors.push(`s3_pipeline_uri ${env.s3_pipeline_uri} is not under ${env.s3_uri}/pipelines/`);
    }
  }
  checkSequence(String(env.stages || "").split(",").filter(Boolean), expectedStages, "run.env stages", errors);
  checkSequence(planStages, expectedPlanStages, "plan stages", errors);

  requireField(env, "attention_corpus_version", "v2", errors);
  requireField(env, "attention_joint_corpus_version", "v2", errors);
  requireField(env, "attention_text_token_profile", "chunked", errors);
  requireField(env, "attention_image_token_profile", "symbolic16", errors);
  requireField(env, "attention_joint_image_token_profile", "symbolic16", errors);
  requireField(env, "attention_batch_mode", config.requiredAttentionBatchMode, errors);
  requireField(env, "attention_map_reduce_workers", config.requiredAttentionMapReduceWorkers, errors);
  requireField(env, "attention_v2_stage_epochs", "1", errors);
  requireIntegerAtLeast(env, "attention_v2_native_bind_epochs", config.minNativeBindEpochs, errors);
  if (config.requireAttentionCpuScaling) {
    requireField(env, "attention_cpu_scaling_policy", config.requiredAttentionCpuScalingPolicy, errors);
    requireField(env, "attention_map_reduce_auto_workers", "1", errors);
  }
  requireField(env, "attention_eval_max_examples", config.requiredEvalMaxExamples, errors);
  requireField(env, "attention_require_image_token_profile", config.requiredImageTokenProfile, errors);
  requireField(env, "attention_require_image_token_channels", config.requiredImageTokenChannels, errors);
  if (config.requireImageChannelTokenStats) {
    requireField(env, "attention_require_image_channel_token_stats", "1", errors);
  }
  if (config.requireDirectionalGroups) {
    requireField(env, "attention_require_directional_groups", "1", errors);
  }
  requireField(env, "attention_min_image_channel_distinct_bins", String(config.minImageChannelDistinctBins), errors);
  requireField(env, "attention_require_confidence_trace", "1", errors);
  if (config.requireHeldoutPrompts) {
    requireField(env, "attention_require_heldout_prompts", "1", errors);
  }
  requireField(env, "attention_min_match_yes_top1", String(config.minMatchYesTop1), errors);
  requireField(env, "attention_min_match_no_top1", String(config.minMatchNoTop1), errors);
  requireField(env, "attention_min_match_no_image_top1", String(config.minMatchNoImageTop1), errors);
  requireField(env, "attention_min_match_no_prompt_top1", String(config.minMatchNoPromptTop1), errors);
  requireField(env, "attention_min_retrieval_margin", String(config.minRetrievalMargin), errors);
  if (config.requireIdentityInference) {
    requireField(env, "attention_require_identity_inference", "1", errors);
  }
  if (config.requireGroundedCorpus) {
    requireField(env, "attention_require_grounded_corpus", "1", errors);
  }
  requireField(env, "attention_min_source_overlap_tokens", String(config.minSourceOverlapTokens), errors);
  requireField(env, "attention_min_attribute_source_overlap_tokens", String(config.minAttributeSourceOverlapTokens), errors);
  requireField(env, "attention_max_source_placeholder_rows", String(config.maxSourcePlaceholderRows), errors);
  requireField(env, "attention_max_attribute_generic_rank_rows", String(config.maxAttributeGenericRankRows), errors);
  if (config.requireArchitectureProfile) {
    requireField(env, "attention_require_architecture_profile", "1", errors);
  }
  requireField(env, "attention_min_d_model", String(config.minDModel), errors);
  requireField(env, "attention_min_heads", String(config.minHeads), errors);
  requireField(env, "attention_min_hidden_dim", String(config.minHiddenDim), errors);
  requireField(env, "attention_min_transformer_layers", String(config.minTransformerLayers), errors);
  requireField(env, "attention_min_context_seq_len", String(config.minContextSeqLen), errors);
  if (config.requireDenoiseBridge) {
    requireField(env, "attention_require_denoise_bridge", "1", errors);
  }
  if (config.requireDenoiseOutputIdentity) {
    requireField(env, "attention_require_denoise_output_identity", "1", errors);
  }
  if (config.requireGenerativeEval) {
    requireField(env, "attention_require_generative_eval", "1", errors);
  }
  if (config.requireGenerativeOutputIdentity) {
    requireField(env, "attention_require_generative_output_identity", "1", errors);
  }
  if (config.requirePromotedSmallProfile) {
    requireField(env, "attention_require_promoted_small_profile", "1", errors);
  }
  if (config.requirePromotionBundleCheck) {
    requireField(env, "promotion_bundle_check", "1", errors);
  }
  requireField(env, "attention_min_task_targets", config.minTaskTargets, errors);
  requireField(env, "attention_min_task_top5_per_mille", config.minTaskTop5PerMille, errors);
  requireField(env, "attention_min_phase_targets", config.minPhaseTargets, errors);
  if (config.minDirectionAccuracyPerMille) {
    requireField(env, "attention_min_direction_accuracy_per_mille", config.minDirectionAccuracyPerMille, errors);
  }
  if (config.minDirectionTop5PerMille) {
    requireField(env, "attention_min_direction_top5_per_mille", config.minDirectionTop5PerMille, errors);
  }
  if (config.minDirectionTop10PerMille) {
    requireField(env, "attention_min_direction_top10_per_mille", config.minDirectionTop10PerMille, errors);
  }

  const seqLen = Number(env.attention_seq_len || 0);
  const processorCount = Number(env.processor_count || 0);
  const effectiveMapReduceWorkers = Number(env.attention_effective_map_reduce_workers || 0);
  if (config.requireAttentionCpuScaling) {
    if (!Number.isInteger(processorCount) || processorCount < 1) {
      errors.push(`processor_count ${JSON.stringify(env.processor_count || "")} is not a positive integer`);
    }
    if (!Number.isInteger(effectiveMapReduceWorkers) || effectiveMapReduceWorkers < config.minAttentionEffectiveWorkers) {
      errors.push(
        `attention_effective_map_reduce_workers ${JSON.stringify(env.attention_effective_map_reduce_workers || "")} < ${config.minAttentionEffectiveWorkers}`,
      );
    }
    if (
      env.attention_batch_mode === "map-reduce" &&
      env.attention_map_reduce_workers === "0" &&
      Number.isInteger(processorCount) &&
      processorCount > 0 &&
      effectiveMapReduceWorkers !== processorCount
    ) {
      errors.push(
        `attention_effective_map_reduce_workers ${effectiveMapReduceWorkers} != processor_count ${processorCount} for 0-auto workers`,
      );
    }
  }
  if (
    !Number.isInteger(seqLen) ||
    seqLen < config.minAttentionSeqLen ||
    seqLen > config.maxAttentionSeqLen
  ) {
    errors.push(
      `attention_seq_len ${JSON.stringify(env.attention_seq_len || "")} outside ${config.minAttentionSeqLen}-${config.maxAttentionSeqLen}`,
    );
  }
  const heldoutRows = Number(env.attention_min_heldout_prompt_rows || 0);
  if (!Number.isInteger(heldoutRows) || heldoutRows < config.minHeldoutPromptRows) {
    errors.push(
      `attention_min_heldout_prompt_rows ${JSON.stringify(env.attention_min_heldout_prompt_rows || "")} < ${config.minHeldoutPromptRows}`,
    );
  }
  const heldoutPromptArtifact = checkHeldoutPromptArtifact(config, env, baseDirs, errors);
  const generativePromptArtifact = checkGenerativePromptArtifact(config, env, baseDirs, errors);
  const promotionManifest = checkPromotionManifest(config, env, baseDirs, errors);
  const generativeEvalRun = env.generative_eval_run || "";
  const attentionGenerativeEval = env.attention_generative_eval || "";
  let attentionGenerativeEvalMatchesRun = null;
  if (config.requireGenerativeEval) {
    if (!generativeEvalRun) {
      errors.push("generative_eval_run is missing");
    }
    if (!attentionGenerativeEval) {
      errors.push("attention_generative_eval is missing");
    } else if (generativeEvalRun) {
      attentionGenerativeEvalMatchesRun = sameReferencedPath(attentionGenerativeEval, generativeEvalRun, baseDirs);
      if (attentionGenerativeEvalMatchesRun === false) {
        errors.push(`attention_generative_eval ${attentionGenerativeEval} does not match generative_eval_run ${generativeEvalRun}`);
      }
    }
  }
  const generativeEvalPermille = Number(env.generative_eval_permille || 0);
  if (!Number.isInteger(generativeEvalPermille) || generativeEvalPermille < config.minGenerativeEvalPermille) {
    errors.push(
      `generative_eval_permille ${JSON.stringify(env.generative_eval_permille || "")} < ${config.minGenerativeEvalPermille}`,
    );
  }
  const generativeLimit = Number(env.generative_limit || 0);
  if (!Number.isInteger(generativeLimit) || generativeLimit < config.minGenerativeEvalLimit) {
    errors.push(`generative_limit ${JSON.stringify(env.generative_limit || "")} < ${config.minGenerativeEvalLimit}`);
  }
  const generatedPromptRows = Number(env.attention_min_generated_prompt_rows || 0);
  if (!Number.isInteger(generatedPromptRows) || generatedPromptRows < config.minGeneratedPromptRows) {
    errors.push(
      `attention_min_generated_prompt_rows ${JSON.stringify(env.attention_min_generated_prompt_rows || "")} < ${config.minGeneratedPromptRows}`,
    );
  }
  const generatedTop516 = Number(env.attention_min_generated_top5_16_per_mille || 0);
  if (!Number.isInteger(generatedTop516) || generatedTop516 < config.minGeneratedTop516PerMille) {
    errors.push(
      `attention_min_generated_top5_16_per_mille ${JSON.stringify(env.attention_min_generated_top5_16_per_mille || "")} < ${config.minGeneratedTop516PerMille}`,
    );
  }
  const generatedRetrievalTop1 = Number(env.attention_min_generated_retrieval_top1_per_mille || 0);
  if (
    !Number.isInteger(generatedRetrievalTop1) ||
    generatedRetrievalTop1 < config.minGeneratedRetrievalTop1PerMille
  ) {
    errors.push(
      `attention_min_generated_retrieval_top1_per_mille ${JSON.stringify(env.attention_min_generated_retrieval_top1_per_mille || "")} < ${config.minGeneratedRetrievalTop1PerMille}`,
    );
  }
  const generatedRetrievalTop5 = Number(env.attention_min_generated_retrieval_top5_per_mille || 0);
  if (
    !Number.isInteger(generatedRetrievalTop5) ||
    generatedRetrievalTop5 < config.minGeneratedRetrievalTop5PerMille
  ) {
    errors.push(
      `attention_min_generated_retrieval_top5_per_mille ${JSON.stringify(env.attention_min_generated_retrieval_top5_per_mille || "")} < ${config.minGeneratedRetrievalTop5PerMille}`,
    );
  }
  const generatedRetrievalMargin = Number(env.attention_min_generated_retrieval_margin || 0);
  if (
    !Number.isInteger(generatedRetrievalMargin) ||
    generatedRetrievalMargin < config.minGeneratedRetrievalMargin
  ) {
    errors.push(
      `attention_min_generated_retrieval_margin ${JSON.stringify(env.attention_min_generated_retrieval_margin || "")} < ${config.minGeneratedRetrievalMargin}`,
    );
  }
  const generatedMeanTargetDistance16 = Number(env.attention_max_generated_mean_target_distance_16_q8 || 0);
  if (
    !Number.isInteger(generatedMeanTargetDistance16) ||
    generatedMeanTargetDistance16 < 1 ||
    generatedMeanTargetDistance16 > config.maxGeneratedMeanTargetDistance16Q8
  ) {
    errors.push(
      `attention_max_generated_mean_target_distance_16_q8 ${JSON.stringify(env.attention_max_generated_mean_target_distance_16_q8 || "")} > ${config.maxGeneratedMeanTargetDistance16Q8}`,
    );
  }
  const denoiseRetrievalRank = Number(env.attention_denoise_max_output_retrieval_rank || 0);
  if (
    !Number.isInteger(denoiseRetrievalRank) ||
    denoiseRetrievalRank < 1 ||
    denoiseRetrievalRank > config.maxDenoiseOutputRetrievalRank
  ) {
    errors.push(
      `attention_denoise_max_output_retrieval_rank ${JSON.stringify(env.attention_denoise_max_output_retrieval_rank || "")} > ${config.maxDenoiseOutputRetrievalRank}`,
    );
  }
  const denoiseRetrievalMargin = Number(env.attention_denoise_min_output_retrieval_margin || 0);
  if (
    !Number.isInteger(denoiseRetrievalMargin) ||
    denoiseRetrievalMargin < config.minDenoiseOutputRetrievalMargin
  ) {
    errors.push(
      `attention_denoise_min_output_retrieval_margin ${JSON.stringify(env.attention_denoise_min_output_retrieval_margin || "")} < ${config.minDenoiseOutputRetrievalMargin}`,
    );
  }
  const denoiseUniqueTargets = Number(env.attention_denoise_min_unique_targets || 0);
  if (
    !Number.isInteger(denoiseUniqueTargets) ||
    denoiseUniqueTargets < config.minDenoiseBridgeUniqueTargets
  ) {
    errors.push(
      `attention_denoise_min_unique_targets ${JSON.stringify(env.attention_denoise_min_unique_targets || "")} < ${config.minDenoiseBridgeUniqueTargets}`,
    );
  }
  checkSequence(
    String(env.attention_v2_curriculum_stages || "").split(",").filter(Boolean),
    expectedCurriculumStages,
    "attention_v2_curriculum_stages",
    errors,
  );
  checkSequence(
    String(env.attention_v2_curriculum_required_stages || "").split(",").filter(Boolean),
    expectedCurriculumStages,
    "attention_v2_curriculum_required_stages",
    errors,
  );

  if (planByStage.has("generative-eval")) {
    const command = planByStage.get("generative-eval");
    requireCommandContains(command, "node scripts/run-solomon-generative-eval.mjs", "generative-eval", errors);
    if (env.generative_prompts) {
      requireCommandContains(command, `--prompts ${env.generative_prompts}`, "generative-eval", errors);
    }
    requireCommandContains(command, "--partition eval", "generative-eval", errors);
    if (env.generative_eval_permille) {
      requireCommandContains(command, `--eval-permille ${env.generative_eval_permille}`, "generative-eval", errors);
    }
    if (env.generative_limit) {
      requireCommandContains(command, `--limit ${env.generative_limit}`, "generative-eval", errors);
    }
  }
  if (config.requirePromotionBundleCheck) {
    const command = planByStage.get("promotion-bundle-check");
    if (!command) {
      errors.push("plan missing promotion-bundle-check stage");
    } else {
      requireCommandContains(
        command,
        "node scripts/check-solomon-promotion-bundle.mjs",
        "promotion-bundle-check",
        errors,
      );
      requireCommandContains(
        command,
        `--promotion ${env.promotion_manifest || config.promotionPath}`,
        "promotion-bundle-check",
        errors,
      );
      requireCommandContains(
        command,
        `--out ${path.join(runDir, "promotion-bundle-check.json")}`,
        "promotion-bundle-check",
        errors,
      );
    }
  }
  if (planByStage.has("attention-curriculum")) {
    const command = planByStage.get("attention-curriculum");
    for (const needle of [
      "NSRL_SOLOMON_ATTENTION_CORPUS_VERSION=v2",
      "NSRL_SOLOMON_ATTENTION_JOINT_CORPUS_VERSION=v2",
      "NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE=chunked",
      "NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE=symbolic16",
      "NSRL_SOLOMON_ATTENTION_JOINT_IMAGE_TOKEN_PROFILE=symbolic16",
      `NSRL_SOLOMON_ATTENTION_BATCH_MODE=${config.requiredAttentionBatchMode}`,
      `NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=${config.requiredAttentionMapReduceWorkers}`,
      `NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=${config.requiredEvalMaxExamples}`,
      "NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES=identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind",
      "NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES=identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind",
      `NSRL_SOLOMON_ATTENTION_V2_STAGE_EPOCHS=${env.attention_v2_stage_epochs || ""}`,
      `NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS=${env.attention_v2_native_bind_epochs || ""}`,
      `NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE=${config.requiredImageTokenProfile}`,
      `NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS=${config.requiredImageTokenChannels}`,
      `NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS=${config.minImageChannelDistinctBins}`,
      "NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES=1",
      "NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE=1",
      "NSRL_SOLOMON_V2_REQUIRE_DIRECTIONAL_GROUPS=1",
      "bash scripts/run-solomon-attention-curriculum-smoke.sh",
    ]) {
      requireCommandContains(command, needle, "attention-curriculum", errors);
    }
    if (config.requireImageChannelTokenStats) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS=1", "attention-curriculum", errors);
    }
    if (config.requireHeldoutPrompts) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS=1", "attention-curriculum", errors);
      requireCommandContains(
        command,
        `NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS=${env.attention_heldout_prompts || ""}`,
        "attention-curriculum",
        errors,
      );
    }
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS=${env.attention_min_heldout_prompt_rows || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1=${env.attention_min_match_yes_top1 || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1=${env.attention_min_match_no_top1 || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1=${env.attention_min_match_no_image_top1 || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1=${env.attention_min_match_no_prompt_top1 || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN=${env.attention_min_retrieval_margin || ""}`,
      "attention-curriculum",
      errors,
    );
    if (config.requireIdentityInference) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE=1", "attention-curriculum", errors);
    }
    if (config.requireGroundedCorpus) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_GROUNDED_CORPUS=1", "attention-curriculum", errors);
    }
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS=${env.attention_min_source_overlap_tokens || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS=${env.attention_min_attribute_source_overlap_tokens || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS=${env.attention_max_source_placeholder_rows || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS=${env.attention_max_attribute_generic_rank_rows || ""}`,
      "attention-curriculum",
      errors,
    );
    if (config.requireArchitectureProfile) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=1", "attention-curriculum", errors);
    }
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_D_MODEL=${env.attention_min_d_model || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_HEADS=${env.attention_min_heads || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_HIDDEN_DIM=${env.attention_min_hidden_dim || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS=${env.attention_min_transformer_layers || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN=${env.attention_min_context_seq_len || ""}`,
      "attention-curriculum",
      errors,
    );
    if (config.requireDenoiseBridge) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE=1", "attention-curriculum", errors);
    }
    if (config.requireDenoiseOutputIdentity) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY=1", "attention-curriculum", errors);
    }
    requireCommandContains(
      command,
      `NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK=${env.attention_denoise_max_output_retrieval_rank || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN=${env.attention_denoise_min_output_retrieval_margin || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS=${env.attention_denoise_min_unique_targets || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS=${env.attention_denoise_min_unique_targets || ""}`,
      "attention-curriculum",
      errors,
    );
    if (config.requireGenerativeEval) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL=1", "attention-curriculum", errors);
      requireCommandContains(
        command,
        `NSRL_SOLOMON_V2_GENERATIVE_EVAL=${env.attention_generative_eval || ""}`,
        "attention-curriculum",
        errors,
      );
    }
    if (config.requireGenerativeOutputIdentity) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY=1", "attention-curriculum", errors);
    }
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE=${env.attention_min_generated_top5_16_per_mille || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE=${env.attention_min_generated_retrieval_top1_per_mille || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE=${env.attention_min_generated_retrieval_top5_per_mille || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN=${env.attention_min_generated_retrieval_margin || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS=${env.attention_min_generated_prompt_rows || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8=${env.attention_max_generated_mean_target_distance_16_q8 || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_TASK_TARGETS=${env.attention_min_task_targets || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=${env.attention_min_task_top5_per_mille || ""}`,
      "attention-curriculum",
      errors,
    );
    requireCommandContains(
      command,
      `NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=${env.attention_min_phase_targets || ""}`,
      "attention-curriculum",
      errors,
    );
    if (env.attention_min_direction_accuracy_per_mille) {
      requireCommandContains(
        command,
        `NSRL_SOLOMON_V2_MIN_DIRECTION_ACCURACY_PER_MILLE=${env.attention_min_direction_accuracy_per_mille}`,
        "attention-curriculum",
        errors,
      );
    }
    if (env.attention_min_direction_top5_per_mille) {
      requireCommandContains(
        command,
        `NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE=${env.attention_min_direction_top5_per_mille}`,
        "attention-curriculum",
        errors,
      );
    }
    if (env.attention_min_direction_top10_per_mille) {
      requireCommandContains(
        command,
        `NSRL_SOLOMON_V2_MIN_DIRECTION_TOP10_PER_MILLE=${env.attention_min_direction_top10_per_mille}`,
        "attention-curriculum",
        errors,
      );
    }
    if (config.requirePromotedSmallProfile) {
      requireCommandContains(command, "NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1", "attention-curriculum", errors);
    }
    const seqNeedle = `NSRL_SOLOMON_ATTENTION_SEQ_LEN=${env.attention_seq_len || ""}`;
    requireCommandContains(command, seqNeedle, "attention-curriculum", errors);
  }

  return {
    schema: "nsrl.solomon_aws_product_plan_check.v1",
    ok: errors.length === 0,
    run_env: config.runEnvPath,
    plan: config.planPath,
    stages: planStages,
    required_stages: expectedStages,
    required_plan_stages: expectedPlanStages,
    runner: {
      kernel: env.runner_kernel || "",
      arch: env.runner_arch || "",
      require_graviton: env.require_graviton === "1",
    },
    s3: {
      required: env.require_s3_artifacts === "1",
      uri: env.s3_uri || "",
      pipeline_uri: env.s3_pipeline_uri || "",
    },
    generative_eval: {
      prompts: env.generative_prompts || "",
      eval_permille: generativeEvalPermille,
      limit: generativeLimit,
      run: generativeEvalRun,
      attention_input: attentionGenerativeEval,
      attention_input_matches_run: attentionGenerativeEvalMatchesRun,
      prompt_artifact: generativePromptArtifact,
    },
    promotion: promotionManifest,
    promotion_bundle_check: env.promotion_bundle_check === "1",
    attention: {
      corpus_version: env.attention_corpus_version || "",
      joint_corpus_version: env.attention_joint_corpus_version || "",
      text_token_profile: env.attention_text_token_profile || "",
      image_token_profile: env.attention_image_token_profile || "",
      joint_image_token_profile: env.attention_joint_image_token_profile || "",
      batch_mode: env.attention_batch_mode || "",
      map_reduce_workers: Number(env.attention_map_reduce_workers || 0),
      cpu_scaling: {
        policy: env.attention_cpu_scaling_policy || "",
        auto_workers: env.attention_map_reduce_auto_workers === "1",
        processor_count: processorCount,
        effective_map_reduce_workers: effectiveMapReduceWorkers,
        min_effective_workers: config.minAttentionEffectiveWorkers,
      },
      seq_len: seqLen,
      eval_max_examples: env.attention_eval_max_examples || "",
      v2_stage_epochs: Number(env.attention_v2_stage_epochs || 0),
      native_bind_epochs: Number(env.attention_v2_native_bind_epochs || 0),
      curriculum_stages: String(env.attention_v2_curriculum_stages || "").split(",").filter(Boolean),
      curriculum_required_stages: String(env.attention_v2_curriculum_required_stages || "").split(",").filter(Boolean),
      required_image_token_profile: env.attention_require_image_token_profile || "",
      required_image_token_channels: env.attention_require_image_token_channels || "",
      require_image_channel_token_stats: env.attention_require_image_channel_token_stats === "1",
      require_directional_groups: env.attention_require_directional_groups === "1",
      min_direction_accuracy_per_mille: env.attention_min_direction_accuracy_per_mille || "",
      min_direction_top5_per_mille: env.attention_min_direction_top5_per_mille || "",
      min_direction_top10_per_mille: env.attention_min_direction_top10_per_mille || "",
      min_image_channel_distinct_bins: Number(env.attention_min_image_channel_distinct_bins || 0),
      require_heldout_prompts: env.attention_require_heldout_prompts === "1",
      heldout_prompt_artifact: heldoutPromptArtifact,
      min_heldout_prompt_rows: heldoutRows,
      min_match_yes_top1: Number(env.attention_min_match_yes_top1 || 0),
      min_match_no_top1: Number(env.attention_min_match_no_top1 || 0),
      min_match_no_image_top1: Number(env.attention_min_match_no_image_top1 || 0),
      min_match_no_prompt_top1: Number(env.attention_min_match_no_prompt_top1 || 0),
      min_retrieval_margin: Number(env.attention_min_retrieval_margin || 0),
      require_identity_inference: env.attention_require_identity_inference === "1",
      require_grounded_corpus: env.attention_require_grounded_corpus === "1",
      min_source_overlap_tokens: Number(env.attention_min_source_overlap_tokens || 0),
      min_attribute_source_overlap_tokens: Number(env.attention_min_attribute_source_overlap_tokens || 0),
      max_source_placeholder_rows: Number(env.attention_max_source_placeholder_rows || 0),
      max_attribute_generic_rank_rows: Number(env.attention_max_attribute_generic_rank_rows || 0),
      require_architecture_profile: env.attention_require_architecture_profile === "1",
      min_d_model: Number(env.attention_min_d_model || 0),
      min_heads: Number(env.attention_min_heads || 0),
      target_head_dim: config.targetHeadDim,
      min_hidden_dim: Number(env.attention_min_hidden_dim || 0),
      max_hidden_dim: config.maxHiddenDim,
      min_transformer_layers: Number(env.attention_min_transformer_layers || 0),
      min_context_seq_len: Number(env.attention_min_context_seq_len || 0),
      train_core_architecture: trainCoreArchitecture,
      curriculum_denoise_runner: curriculumDenoiseRunner,
      require_confidence_trace: env.attention_require_confidence_trace === "1",
      require_denoise_bridge: env.attention_require_denoise_bridge === "1",
      require_denoise_output_identity: env.attention_require_denoise_output_identity === "1",
      denoise_max_output_retrieval_rank: denoiseRetrievalRank,
      denoise_min_output_retrieval_margin: denoiseRetrievalMargin,
      denoise_min_unique_targets: denoiseUniqueTargets,
      require_generative_eval: env.attention_require_generative_eval === "1",
      require_generative_output_identity: env.attention_require_generative_output_identity === "1",
      min_generated_prompt_rows: generatedPromptRows,
      min_generated_top5_16_per_mille: generatedTop516,
      min_generated_retrieval_top1_per_mille: generatedRetrievalTop1,
      min_generated_retrieval_top5_per_mille: generatedRetrievalTop5,
      min_generated_retrieval_margin: generatedRetrievalMargin,
      max_generated_mean_target_distance_16_q8: generatedMeanTargetDistance16,
      min_task_targets: env.attention_min_task_targets || "",
      min_task_top5_per_mille: env.attention_min_task_top5_per_mille || "",
      min_phase_targets: env.attention_min_phase_targets || "",
      require_promoted_small_profile: env.attention_require_promoted_small_profile === "1",
    },
    errors,
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const report = checkProductPlan(config);
  const text = JSON.stringify(report, null, 2);
  if (config.outPath) {
    fs.writeFileSync(config.outPath, `${text}\n`);
  }
  console.log(text);
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
