#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const schema = "nsrl.solomon_objective_coverage_check.v1";
const REQUIRED_TASK_COUNTS = {
  "canonical-joint": 72,
  identify: 72,
  "text-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
  explain: 72,
  "description-to-image": 72,
  match: 216,
};
const REQUIRED_NATIVE_EVAL_TASKS = Object.keys(REQUIRED_TASK_COUNTS);
const REQUIRED_IMAGE_CHANNELS = ["ink", "edge", "component", "radial", "direction"];
const REQUIRED_SOURCE_PROVENANCE_TASKS = [
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
];
const REQUIRED_SOURCE_QUERY_KIND_BY_TASK = {
  "text-to-image": "identity-to-image",
  "image-to-text": "image-identity",
  "image-to-explain": "image-source",
  "text-image-explain": "text-image-source",
  "image-to-attributes": "image-attributes",
  explain: "primary-name",
  "description-to-image": "source-description",
};
const REQUIRED_CORPUS_NEGATIVE_CASES = [
  "weak-image-profile",
  "missing-prompt-hard-negatives",
  "bad-hard-negative-metadata",
  "missing-source-provenance",
  "bad-source-query-kind",
  "corrupt-task-marker",
  "bad-modality-order",
  "missing-output-heads",
  "missing-task-phases",
  "bad-promoted-small-profile",
  "bad-curriculum-train-profile",
  "missing-curriculum-train-task-coverage",
  "bad-curriculum-train-examples-provenance",
  "bad-curriculum-modality-order",
  "missing-curriculum-modality-integrity",
  "stale-grounded-corpus",
  "stale-identity-inference",
  "stale-retrieval-eval-hash",
  "tampered-retrieval-head",
];
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const REQUIRED_IDENTITY_BINDING_COUNTS = {
  "primary-name": 72,
  "primary-seal": 72,
  alias: 72,
  "alias-seal": 72,
  "seal-id": 216,
};
const REQUIRED_IMAGE_RETRIEVAL_TASKS = [
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
];
const REQUIRED_IMAGE_RETRIEVAL_TASK_COUNTS = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};
const REQUIRED_RETRIEVAL_COUNTS = {
  known_prompts: 72,
  heldout_prompts: 72,
  identity_total: 504,
  image_to_text: 288,
  match_yes: 72,
  match_no: 144,
  match_no_image: 72,
  match_no_prompt: 72,
};
const REQUIRED_CURRICULUM_STAGES = [
  "identity",
  "image",
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "explain",
  "hard-negative",
  "native-bind",
];
const REQUIRED_DIRECTIONAL_GROUPS = {
  text_prompt_to_image_plan: {
    min_targets: 4,
    phase_targets: {
      "text-to-image:prompt": 2,
      "text-to-image:image": 2,
      "description-to-image:prompt": 2,
      "description-to-image:image": 2,
    },
  },
  seal_image_to_text: {
    min_targets: 6,
    phase_targets: {
      "image-to-text:image": 2,
      "image-to-text:text": 2,
      "image-to-explain:image": 2,
      "image-to-explain:text": 2,
      "image-to-attributes:image": 2,
      "image-to-attributes:prompt": 2,
      "image-to-attributes:text": 2,
    },
  },
  text_and_seal_to_explanation: {
    min_targets: 4,
    phase_targets: {
      "text-image-explain:prompt": 2,
      "text-image-explain:image": 2,
      "text-image-explain:text": 2,
      "match:prompt": 2,
      "match:image": 2,
      "match:text": 2,
    },
  },
  identity_source_binding: {
    min_targets: 8,
    phase_targets: {
      "canonical-joint:prompt": 2,
      "canonical-joint:text": 2,
      "canonical-joint:image": 2,
      "identify:prompt": 2,
      "identify:text": 2,
      "explain:prompt": 2,
      "explain:text": 2,
    },
  },
};
const REQUIRED_AWS_MIN_TASK_TARGETS = "all=72";
const REQUIRED_AWS_MIN_TASK_TOP5_PER_MILLE = "all=1";
const REQUIRED_AWS_MIN_PHASE_TARGETS = "all=72";
const REQUIRED_AWS_NATIVE_BIND_EPOCHS = 2;
const REQUIRED_AWS_D_MODEL = 128;
const REQUIRED_AWS_HEADS = 2;
const REQUIRED_AWS_HEAD_DIM = 64;
const REQUIRED_AWS_MIN_HIDDEN_DIM = 256;
const REQUIRED_AWS_MAX_HIDDEN_DIM = 512;
const REQUIRED_AWS_MIN_TRANSFORMER_LAYERS = 2;
const REQUIRED_AWS_MIN_CONTEXT_SEQ_LEN = 384;
const REQUIRED_AWS_MIN_SEQ_LEN = 384;
const REQUIRED_AWS_MAX_SEQ_LEN = 768;
const REQUIRED_AWS_MIN_GENERATED_RETRIEVAL_MARGIN = 1;
const REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS = 2;
const REQUIRED_QUALITY_CASES = [
  "good",
  "bad-retrieval-margin",
  "bad-symbolic-channel-evidence",
  "bad-symbolic-channel-duplicates",
  "bad-source-grounding",
  "bad-source-grounding-missing-sample",
  "bad-grounded-description-source",
  "bad-grounded-attribute-prompt",
  "bad-curriculum-binding",
  "bad-native-task-confidence",
  "bad-sample-generated-text-agreement",
  "bad-generated-text-source-evidence",
  "bad-identity-prompt-text-margin",
  "bad-identity-generated-text-margin",
  "good-denoise-bridge",
  "bad-denoise-output-identity",
  "bad-denoise-target-coverage",
  "good-generative-eval",
  "bad-generative-output-identity",
  "bad-generative-output-margin",
];
const REQUIRED_TASK_EVAL_CASES = [
  "good",
  "bad-directional-phase",
  "bad-forward-direction-prompt-phase",
  "bad-seal-direction-image-phase",
  "bad-joint-direction-image-phase",
  "bad-identity-source-prompt-phase",
  "bad-directional-quality",
  "bad-task-coverage",
  "bad-match-negative-role-coverage",
  "bad-task-marker",
  "bad-modality-order",
  "bad-image-channel-marker",
  "bad-channel-stats",
  "bad-channel-duplicate-records",
  "bad-output-head",
  "bad-eval-provenance",
];
const REQUIRED_TOKEN_LAYOUT_CASES = [
  "good-js-corpus-builder-layout",
  "good-rust-native-attention-layout",
  "good-js-fallback-layouts",
  "good-js-retrieval-consumer-layouts",
  "good-shared-symbolic-image-defaults",
  "bad-js-layout-mismatch",
  "bad-rust-task-marker-mismatch",
  "bad-shared-marker-order",
];
const REQUIRED_PRIOR_SMOKE_CASES = [
  "good",
  "bad-target-source",
  "bad-missing-seed-variant",
  "bad-collapsed-interclass",
  "bad-eval-class-top1",
];
const REQUIRED_HELDOUT_RETRIEVAL_CASES = [
  "good",
  "bad-prompts-hash",
  "bad-heldout-row-count",
  "bad-heldout-top1",
  "bad-heldout-margin",
  "bad-missing-image-head",
  "bad-stale-model-hash",
];
const REQUIRED_GROUNDED_CORPUS_CASES = [
  "good",
  "bad-source-overlap",
  "bad-source-placeholder",
  "bad-attribute-generic-rank",
  "bad-source-provenance-hash",
  "bad-explain-prompt",
  "bad-description-prompt",
  "bad-attribute-name-prompt",
  "bad-missing-attribute-task",
];
const REQUIRED_GENERATIVE_CASES = [
  "good",
  "posthoc-score",
  "posthoc-bad-raw-path",
  "bad-raw-path",
  "bad-cleanup",
  "bad-missing-trace",
  "bad-empty-raw",
];
const REQUIRED_GENERATION_INTEGRITY_CASES = [
  "good-bitmap-decoded-latent",
  "good-attention-embedded",
  "bad-target-source",
  "bad-target-pixel-key",
  "bad-oracle-value",
  "bad-display-cleanup",
  "bad-raw-path",
  "bad-missing-raw",
  "bad-expected-latent-source",
];
const REQUIRED_SAMPLE_BINDING_CASES = [
  "good",
  "bad-generated-text",
  "bad-generated-image",
  "bad-cleanup",
];
const REQUIRED_DENOISE_BRIDGE_CASES = [
  "good",
  "bad-cleanup",
  "bad-source",
  "bad-signature",
  "bad-flat-output",
  "bad-retrieval-head-hash",
  "bad-output-retrieval-margin",
  "bad-unique-targets",
];
const REQUIRED_PROMOTION_CASES = [
  "good",
  "bad-generation",
  "bad-integrity",
  "bad-source-provenance",
  "bad-source-grounding",
  "bad-grounded-corpus-overlap",
  "bad-grounded-attribute-rank",
  "bad-grounded-name-source",
  "bad-grounded-description-source",
  "bad-grounded-attribute-prompt",
  "bad-prompt-provenance",
  "bad-generated-signature",
  "bad-generated-distance",
  "bad-confidence-spine",
  "bad-heldout-retrieval",
  "bad-hard-negative-retrieval",
  "bad-directional-group",
  "bad-native-task-confidence",
  "bad-head-dim",
  "bad-hidden-dim",
  "bad-retrieval-text-head",
  "bad-retrieval-head-hash",
  "bad-symbolic-tokens",
  "bad-corpus-channel-stats",
  "bad-corpus-channel-duplicates",
  "bad-corpus-task-marker",
  "bad-denoise-provenance",
  "bad-denoise-target-coverage",
];
const RELEASE_CANDIDATE_LIVE_READINESS_CASE = "good-local-live-readiness-next-action";
const REQUIRED_RELEASE_CANDIDATE_LIVE_READINESS_NEXT_ACTION = [
  "scripts/check-solomon-aws-live-launch-readiness.sh",
  "scripts/aws/launch-solomon-product-run.sh --execute",
  "scripts/aws/prove-solomon-product-run.sh",
  "--s3-pipeline-uri",
  "--launch-dir",
  "--require-launch-dir",
  "launch-result.json",
];
const REQUIRED_RELEASE_CANDIDATE_CASES = [
  "good-local",
  RELEASE_CANDIDATE_LIVE_READINESS_CASE,
  "good-release",
  "bad-skipped-local-proof",
  "bad-objective-evidence",
  "bad-native-task-handoff-ratchet",
  "bad-native-bind-handoff-ratchet",
  "bad-denoise-handoff-ratchet",
  "bad-denoise-runner-proof",
  "bad-architecture-handoff-ratchet",
  "bad-generated-handoff-floor",
  "bad-generated-handoff-heldout-coverage",
  "bad-generated-handoff-distance-cap",
  "bad-prelaunch-readiness",
  "bad-post-run-proof-command",
  "bad-quality-generated-text-agreement-evidence",
  "bad-sample-binding-generated-image-evidence",
  "bad-denoise-bridge-output-evidence",
  "bad-grounded-source-evidence",
  "bad-task-eval-channel-duplicate-evidence",
  "bad-task-eval-hard-negative-role-evidence",
  "bad-token-layout-contract-evidence",
  "bad-generation-integrity-evidence",
  "bad-live-launch-readiness-evidence",
  "bad-live-launch-readiness-ami-evidence",
  "bad-execute-guard-explicit-s3-evidence",
  "bad-execute-guard-evidence",
  "bad-release-proof-evidence",
  "bad-run-artifact-evidence",
  "bad-run-fetch-evidence",
  "bad-extra-release-gap",
  "bad-release-required",
];
const REQUIRED_AWS_RELEASE_PROOF_CASES = [
  "bad-mismatched-s3-pipeline",
  "bad-missing-stage-status",
  "bad-native-product-eval-scope",
  "bad-missing-required-launch-dir",
  "bad-dry-run-launch-dir",
  "bad-missing-launch-instance-id",
  "bad-launch-user-data-sha256",
  "bad-launch-command-image-id",
  "bad-launch-command-instance-type",
  "bad-launch-command-user-data",
  "bad-launch-command-tag-specification",
  "bad-launch-command-security-groups",
  "bad-launch-command-subnet",
  "bad-launch-command-key-name",
  "bad-launch-command-region",
  "bad-launch-command-profile",
  "bad-missing-launch-result",
  "bad-launch-result-sha256",
  "bad-launch-result-image-id",
  "bad-launch-result-instance-type",
  "bad-launch-result-subnet",
  "bad-launch-result-security-groups",
  "bad-launch-run-instance-mismatch",
  "bad-post-run-proof-command",
];
const REQUIRED_AWS_RUN_ARTIFACT_CASES = [
  "good",
  "bad-dry-run",
  "bad-non-graviton",
  "bad-missing-ec2-metadata",
  "bad-missing-stage-status",
  "bad-fixed-attention-workers",
  "bad-cpu-scaling-policy",
  "bad-image-token-profile",
  "bad-native-bind-epochs",
  "bad-native-product-eval-scope",
  "bad-generated-product-coverage",
  "bad-artifact-index-missing-curriculum-stages",
  "bad-completion-missing-curriculum-stages-artifact",
  "digest-changes-after-artifact-tamper",
  "bad-completion-env-config-mismatch",
  "bad-completion-generative-config-mismatch",
  "bad-s3",
  "bad-promotion-check",
  "bad-promotion-validation",
];
const REQUIRED_AWS_RUN_FETCH_CASES = [
  "good",
  "good-run-name",
  "bad-mismatched-s3-pipeline",
  "bad-missing-status",
  "bad-stale-promotion",
  "bad-native-product-eval-scope",
];
const REQUIRED_AWS_LIVE_LAUNCH_READINESS_CASES = [
  "good-explicit-s3-artifact",
  "bad-missing-explicit-s3-artifact",
  "bad-missing-explicit-ami",
];
const REQUIRED_AWS_LAUNCH_EXECUTE_GUARD_CASES = [
  "bad-execute-missing-explicit-s3-blocks-before-aws",
  "bad-execute-missing-explicit-artifact-blocks-before-aws",
  "bad-execute-prelaunch-blocks-before-aws",
  "good-execute-records-launch-result",
  "good-execute-command-matches-launch-manifest",
  "good-execute-command-matches-launch-manifest-with-profile",
];
const REQUIRED_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8 = 7000000;

function usage() {
  console.log([
    "Usage: check-solomon-objective-coverage.mjs --diagnostic PATH [--require-release] [--out PATH]",
    "",
    "Maps a Solomon product diagnostic report to the narrow, grounded,",
    "bidirectional multimodal objective. Local proof requires the real no-spend",
    "corpus/retrieval/native checks plus contract self-tests. --require-release",
    "also requires a synced real Graviton run in the diagnostic.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = {
    diagnosticPath: "",
    requireRelease: false,
    outPath: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--diagnostic") {
      config.diagnosticPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-release") {
      config.requireRelease = true;
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.diagnosticPath) {
    throw new Error("--diagnostic PATH is required");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function hasAll(values, required) {
  const set = new Set((values || []).map(String));
  return required.every((item) => set.has(item));
}

function sameSequence(values, required) {
  const actual = (values || []).map(String);
  return actual.length === required.length && required.every((item, index) => actual[index] === item);
}

function metricCovers(metric, minimum) {
  const count = Number(metric?.count || 0);
  return count >= minimum && Number(metric?.top1 || 0) === count;
}

function metricMatchesRows(metric, rows) {
  const count = Number(metric?.count || 0);
  return rows > 0 && count === rows && Number(metric?.top1 || 0) === count && Number(metric?.top5 || 0) === count;
}

function retrievalHeadHashProblems(corpus, retrievalHead) {
  const corpusHash = String(corpus?.retrieval_model_hash || "");
  const modelHash = String(retrievalHead?.model_hash || "");
  const evalHash = String(retrievalHead?.eval_model_hash || "");
  const problems = [];
  if (!corpusHash) {
    problems.push("corpus retrieval_model_hash is missing");
  }
  if (!modelHash) {
    problems.push("retrieval head model_hash is missing");
  }
  if (!evalHash) {
    problems.push("retrieval head eval_model_hash is missing");
  }
  if (corpusHash && modelHash && corpusHash !== modelHash) {
    problems.push(`corpus retrieval_model_hash ${corpusHash} != retrieval head model_hash ${modelHash}`);
  }
  if (evalHash && modelHash && evalHash !== modelHash) {
    problems.push(`retrieval head eval_model_hash ${evalHash} != model_hash ${modelHash}`);
  }
  return problems;
}

function imageChannelStatsComplete(stats) {
  return REQUIRED_IMAGE_CHANNELS.every((channel) => {
    const row = stats?.[channel] || {};
    const records = Number(row.records || 0);
    return records >= 72 &&
      Number(row.tokens_per_record || 0) === 256 &&
      Number(row.active_records || 0) === records &&
      Number(row.multi_bin_records || 0) === records &&
      Number(row.distinct_bins || 0) >= 2 &&
      Number(row.nonzero_tokens || 0) > 0 &&
      Number(row.max_bin || 0) > 0 &&
      Number(row.unique_record_hashes || 0) === records &&
      Number(row.duplicate_record_hashes || 0) === 0;
  });
}

function integrityComplete(integrity, zeroFields, minimumRecords = 1) {
  if (integrity?.ok !== true || Number(integrity?.checked_records || 0) < minimumRecords) {
    return false;
  }
  return zeroFields.every((field) => Number(integrity?.[field] || 0) === 0);
}

function hardNegativeRolesComplete(summary) {
  const noRows = Number(summary?.no_rows || 0);
  return noRows >= REQUIRED_RETRIEVAL_COUNTS.match_no &&
    Number(summary?.image_role_spirits || 0) >= 72 &&
    Number(summary?.prompt_role_spirits || 0) >= 72 &&
    Number(summary?.nearest_image_token_rows || 0) >= noRows &&
    Number(summary?.rank1_rows || 0) >= noRows &&
    Number(summary?.positive_distance_rows || 0) >= noRows;
}

function identityBindingCoverageComplete(summary) {
  return REQUIRED_IDENTITY_BINDING_KINDS.every((kind) => (
    Number(summary?.[kind]?.identify_spirits || 0) >= 72 &&
    Number(summary?.[kind]?.text_to_image_spirits || 0) >= 72
  ));
}

function sourceProvenanceComplete(summary) {
  return sourceProvenanceProblems(summary).length === 0;
}

function sourceProvenanceProblems(summary) {
  const problems = [];
  for (const task of REQUIRED_SOURCE_PROVENANCE_TASKS) {
    const row = summary?.[task] || {};
    const rows = Number(row.rows || 0);
    const expectedQueryKind = REQUIRED_SOURCE_QUERY_KIND_BY_TASK[task];
    if (rows < 72) {
      problems.push(`source provenance task ${task} has ${rows} rows, requires >= 72`);
      continue;
    }
    if (Number(row.spirits || 0) < 72) {
      problems.push(`source provenance task ${task} has ${Number(row.spirits || 0)} spirits, requires >= 72`);
    }
    if (Number(row.source_spirit_id_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source spirit ids`);
    }
    if (Number(row.source_text_hash_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source text hashes`);
    }
    if (Number(row.source_excerpt_hash_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source excerpt hashes`);
    }
    if (Number(row.source_excerpt_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source excerpts`);
    }
    if (String(row.expected_source_query_kind || "") !== expectedQueryKind) {
      problems.push(
        `source provenance task ${task} expected_source_query_kind ${JSON.stringify(row.expected_source_query_kind || "")} != ${JSON.stringify(expectedQueryKind)}`,
      );
    }
    if (Number(row.source_query_kind_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source query kind rows`);
    }
    if (Number(row.source_query_kind_ok_rows || 0) < rows) {
      problems.push(`source provenance task ${task} has incomplete source query kind ok rows`);
    }
    if (Number(row.source_query_kinds?.[expectedQueryKind] || 0) < rows) {
      problems.push(`source provenance task ${task} missing expected source query kind ${expectedQueryKind}`);
    }
  }
  return problems;
}

function generatedSampleIdentityComplete(sample) {
  return sample?.sampler_target_source === "decoded-latent" &&
    Number(sample?.generated_retrieval_rank || 0) === 1 &&
    Number(sample?.generated_retrieval_identity || 0) === 1 &&
    Number(sample?.mean_generated_retrieval_rank_q8 || 0) === 256 &&
    Number(sample?.generated_retrieval_top1_per_mille || 0) === 1000 &&
    Number(sample?.generated_retrieval_top5_per_mille || 0) === 1000 &&
    String(sample?.retrieval_head_model_hash || "").length > 0 &&
    Number(sample?.selected_prompt_rows || 0) > 0 &&
    Number(sample?.selected_prompt_eligible_rows || 0) > 0 &&
    Number(sample?.selected_prompt_unique_targets || 0) > 0 &&
    Number(sample?.selected_prompt_eligible_unique_targets || 0) > 0;
}

function generatedSamplePromptSelectionComplete(sample) {
  return Number(sample?.selected_prompt_sources?.generated || 0) > 0 &&
    Number(sample?.selected_prompt_tiers?.["tier-novel-vocab"] || 0) > 0;
}

function generatedSampleRetrievalHeadBindingProblems(cleanSample, posthocSample) {
  const cleanHash = String(cleanSample?.retrieval_head_model_hash || "");
  const posthocHash = String(posthocSample?.retrieval_head_model_hash || "");
  const problems = [];
  if (!cleanHash) {
    problems.push("clean generated sample retrieval head model hash is missing");
  }
  if (!posthocHash) {
    problems.push("post-hoc generated sample retrieval head model hash is missing");
  }
  if (cleanHash && posthocHash && cleanHash !== posthocHash) {
    problems.push(`generated clean/post-hoc retrieval head hashes differ: ${cleanHash} != ${posthocHash}`);
  }
  return problems;
}

function generatedSampleLatentProvenanceComplete(sample) {
  const latentModel = String(sample?.latent_model || "");
  const latentHash = String(sample?.latent_model_hash || "");
  const configHash = String(sample?.latent_model_config_hash || "");
  const provenanceHash = String(sample?.latent_model_provenance_hash || "");
  const provenancePath = String(sample?.latent_model_provenance_path || "");
  return latentModel.length > 0 &&
    latentHash.length > 0 &&
    configHash === latentHash &&
    provenanceHash === latentHash &&
    provenancePath === latentModel;
}

function imageChannelMarkerIntegrityComplete(integrity) {
  if (!integrityComplete(integrity, [
    "missing_offsets",
    "out_of_bounds",
    "missing_image_markers",
    "missing_channel_markers",
    "short_channel_payloads",
    "bad_channel_payloads",
    "channel_order_mismatches",
  ], 72)) {
    return false;
  }
  const requiredChannels = Array.isArray(integrity?.required_channels)
    ? integrity.required_channels.map(String)
    : [];
  return REQUIRED_IMAGE_CHANNELS.every(
    (channel) => requiredChannels.includes(channel) && Number(integrity?.by_channel?.[channel]?.found_markers || 0) > 0,
  );
}

function hasDirectionalMetricStats(stats) {
  if (!stats || typeof stats !== "object" || Array.isArray(stats)) {
    return false;
  }
  return ["targets", "accuracy_per_mille", "top5_accuracy_per_mille", "top10_accuracy_per_mille"].every((field) =>
    Number.isFinite(Number(stats[field])),
  );
}

function directionalTargetCount(group) {
  return Math.max(Number(group?.targets || 0), Number(group?.stats?.targets || 0));
}

function directionalPhaseMisses(directionalGroups) {
  const misses = [];
  for (const [key, requirement] of Object.entries(REQUIRED_DIRECTIONAL_GROUPS)) {
    const group = directionalGroups[key] || {};
    const phaseTargets = group.phase_targets || {};
    for (const [phase, minimum] of Object.entries(requirement.phase_targets)) {
      const actual = Number(phaseTargets[phase] || 0);
      if (actual < minimum) {
        misses.push({ key, phase, actual, minimum });
      }
    }
  }
  return misses;
}

function nativeIntegerTraceProblems(trace) {
  const problems = [];
  const qFormats = trace?.q_formats || {};
  const trainFields = trace?.train_required_fields || {};
  const requiredTrainFields = [
    "target_frequency_min_weight_q15",
    "argmax_margin_weight_q15",
    "initial_probability_error_q15",
    "final_probability_error_q15",
    "probability_error_delta_i64",
  ];
  const requiredEvalFields = [
    "mean_target_margin_q8",
    "min_target_margin_q8",
    "probability_error_q15",
    "mean_probability_error_q15",
  ];
  if (trace?.ok !== true) {
    problems.push("native integer trace is not ok");
  }
  if (trace?.train_schema !== "nsrl.solomon_attention_train_trace.v1") {
    problems.push(
      `native integer trace train schema ${JSON.stringify(trace?.train_schema || "")} is not nsrl.solomon_attention_train_trace.v1`,
    );
  }
  if (trace?.eval_schema !== "nsrl.solomon_attention_eval_trace.v1") {
    problems.push(
      `native integer trace eval schema ${JSON.stringify(trace?.eval_schema || "")} is not nsrl.solomon_attention_eval_trace.v1`,
    );
  }
  for (const [field, expected] of Object.entries({
    logits: "i32_q8",
    probabilities: "i16_q15",
    probability_error: "q15",
    target_margin: "q8",
    train_delta: "i64",
  })) {
    if (qFormats[field] !== expected) {
      problems.push(`native integer trace q_formats.${field} ${JSON.stringify(qFormats[field] || "")} is not ${expected}`);
    }
  }
  for (const field of requiredTrainFields) {
    if (!Number.isInteger(Number(trainFields[field]))) {
      problems.push(`native integer trace train field ${field} is missing or non-integer`);
    }
  }
  const evalFields = Array.isArray(trace?.eval_required_metric_fields)
    ? trace.eval_required_metric_fields.map(String)
    : [];
  for (const field of requiredEvalFields) {
    if (!evalFields.includes(field)) {
      problems.push(`native integer trace missing eval metric field ${field}`);
    }
  }
  if (Number(trace?.eval_metric_objects || 0) <= 0) {
    problems.push("native integer trace has no eval metric objects");
  }
  if (Number(trace?.numeric_leaves?.train || 0) <= 0) {
    problems.push("native integer trace has no train numeric leaves");
  }
  if (Number(trace?.numeric_leaves?.eval || 0) <= 0) {
    problems.push("native integer trace has no eval numeric leaves");
  }
  const nonIntegerPaths = Array.isArray(trace?.non_integer_numeric_paths)
    ? trace.non_integer_numeric_paths.map(String)
    : [];
  if (nonIntegerPaths.length > 0) {
    problems.push(`native integer trace has non-integer numeric leaves: ${nonIntegerPaths.slice(0, 5).join(", ")}`);
  }
  return problems;
}

function nativeTaskMetricMisses(tasks) {
  return REQUIRED_NATIVE_EVAL_TASKS
    .filter((task) => Number(tasks?.[task]?.targets || 0) <= 0)
    .map((task) => ({ task, targets: Number(tasks?.[task]?.targets || 0) }));
}

function nativeTaskInvalidContextMisses(tasks) {
  return REQUIRED_NATIVE_EVAL_TASKS
    .filter((task) => Number(tasks?.[task]?.invalid_contexts || 0) > 0)
    .map((task) => ({ task, invalid_contexts: Number(tasks?.[task]?.invalid_contexts || 0) }));
}

function nativeTaskPhaseMisses(taskPhases) {
  return REQUIRED_NATIVE_EVAL_TASKS
    .filter((task) => {
      const phases = taskPhases?.[task] || {};
      const targets = Object.values(phases).reduce((sum, phase) => sum + Number(phase?.targets || 0), 0);
      return targets <= 0;
    })
    .map((task) => ({
      task,
      targets: Object.values(taskPhases?.[task] || {}).reduce((sum, phase) => sum + Number(phase?.targets || 0), 0),
    }));
}

function nativeEvalScopeMisses(scope, awsAttention) {
  const misses = [];
  const proofScope = String(scope?.proof_scope || "");
  if (!["local-directional-smoke", "product-scale-native-eval"].includes(proofScope)) {
    misses.push("native eval scope is missing or unknown");
  }
  if (scope?.eval_max_examples !== "none") {
    misses.push("native eval scope does not use eval_max_examples=none");
  }
  const maxTargets = Number(scope?.eval_max_targets_per_task_phase || 0);
  if (!Number.isInteger(maxTargets) || maxTargets <= 0) {
    misses.push("native eval scope is missing eval_max_targets_per_task_phase");
  }
  if (scope?.smoke_min_task_targets !== "all=1") {
    misses.push("native eval scope does not declare smoke_min_task_targets=all=1");
  }
  if (scope?.smoke_min_phase_targets !== "special=1,prompt=1,text=1,image=1") {
    misses.push("native eval scope does not declare smoke phase target floors");
  }
  if (scope?.smoke_min_direction_top5_per_mille !== "all=1") {
    misses.push("native eval scope does not declare smoke directional top-5 floor");
  }
  if (scope?.product_min_task_targets !== REQUIRED_AWS_MIN_TASK_TARGETS) {
    misses.push(`native eval scope product_min_task_targets ${JSON.stringify(scope?.product_min_task_targets || "")} != ${JSON.stringify(REQUIRED_AWS_MIN_TASK_TARGETS)}`);
  }
  if (scope?.product_min_phase_targets !== REQUIRED_AWS_MIN_PHASE_TARGETS) {
    misses.push(`native eval scope product_min_phase_targets ${JSON.stringify(scope?.product_min_phase_targets || "")} != ${JSON.stringify(REQUIRED_AWS_MIN_PHASE_TARGETS)}`);
  }
  if (awsAttention?.eval_max_examples !== "none") {
    misses.push("AWS product plan does not require eval_max_examples=none");
  }
  if (awsAttention?.min_task_targets !== REQUIRED_AWS_MIN_TASK_TARGETS) {
    misses.push(`AWS product plan min_task_targets ${JSON.stringify(awsAttention?.min_task_targets || "")} != ${JSON.stringify(REQUIRED_AWS_MIN_TASK_TARGETS)}`);
  }
  if (awsAttention?.min_phase_targets !== REQUIRED_AWS_MIN_PHASE_TARGETS) {
    misses.push(`AWS product plan min_phase_targets ${JSON.stringify(awsAttention?.min_phase_targets || "")} != ${JSON.stringify(REQUIRED_AWS_MIN_PHASE_TARGETS)}`);
  }
  return misses;
}

function checkCaseList(evidence, checkName, requiredCases) {
  const cases = evidence?.[checkName]?.cases || [];
  return {
    cases,
    missing_cases: requiredCases.filter((name) => !cases.includes(name)),
  };
}

function checkReleaseCandidateNextAction(evidence, caseName, requiredIncludes) {
  const row = evidence?.["release-candidate-self-test"]?.next_action_cases?.[caseName] || {};
  const expected = Array.isArray(row.expected_next_action_includes)
    ? row.expected_next_action_includes.map(String)
    : [];
  const matched = Array.isArray(row.matched_next_action_includes)
    ? row.matched_next_action_includes.map(String)
    : [];
  const missingExpected = requiredIncludes.filter((needle) => !expected.includes(needle));
  const missingMatched = requiredIncludes.filter((needle) => !matched.includes(needle));
  return {
    case: caseName,
    expected_next_action_includes: expected,
    matched_next_action_includes: matched,
    missing_expected_includes: missingExpected,
    missing_matched_includes: missingMatched,
    ok: missingExpected.length === 0 && missingMatched.length === 0,
  };
}

function postRunProofCommandProblems(label, row) {
  const command = Array.isArray(row?.post_run_proof_command)
    ? row.post_run_proof_command.map(String)
    : [];
  const s3PipelineUri = String(row?.s3_pipeline_uri || "");
  const s3Index = command.indexOf("--s3-pipeline-uri");
  const launchDirIndex = command.indexOf("--launch-dir");
  return [
    command.includes("scripts/aws/prove-solomon-product-run.sh")
      ? ""
      : `${label} post-run proof command missing scripts/aws/prove-solomon-product-run.sh`,
    s3PipelineUri.startsWith("s3://")
      ? ""
      : `${label} s3_pipeline_uri is missing`,
    s3Index >= 0 && command[s3Index + 1] === s3PipelineUri
      ? ""
      : `${label} post-run proof command does not match s3_pipeline_uri`,
    launchDirIndex >= 0 && Boolean(command[launchDirIndex + 1])
      ? ""
      : `${label} post-run proof command missing --launch-dir`,
    command.includes("--require-launch-dir")
      ? ""
      : `${label} post-run proof command missing --require-launch-dir`,
  ].filter(Boolean);
}

function awsTrainCoreArchitectureMisses(awsAttention) {
  const architecture = awsAttention?.train_core_architecture || {};
  const requireArchitectureProfile = awsAttention?.require_architecture_profile === true;
  const minDModel = Number(awsAttention?.min_d_model || 0);
  const minHeads = Number(awsAttention?.min_heads || 0);
  const targetHeadDim = Number(awsAttention?.target_head_dim || 0);
  const minHiddenDim = Number(awsAttention?.min_hidden_dim || 0);
  const maxHiddenDim = Number(awsAttention?.max_hidden_dim || 0);
  const minTransformerLayers = Number(awsAttention?.min_transformer_layers || 0);
  const minContextSeqLen = Number(awsAttention?.min_context_seq_len || 0);
  const seqLen = Number(awsAttention?.seq_len || 0);
  const hasArchitecture =
    architecture.present === true ||
    Number(architecture.d_model || 0) > 0 ||
    Number(architecture.heads || 0) > 0 ||
    Number(architecture.hidden_dim || 0) > 0;
  if (!hasArchitecture) {
    return ["AWS train-core architecture proof is missing"];
  }
  return [
    requireArchitectureProfile ? "" : "AWS product plan does not require architecture profile",
    minDModel === REQUIRED_AWS_D_MODEL
      ? ""
      : `AWS product plan min_d_model ${minDModel} != ${REQUIRED_AWS_D_MODEL}`,
    minHeads === REQUIRED_AWS_HEADS
      ? ""
      : `AWS product plan min_heads ${minHeads} != ${REQUIRED_AWS_HEADS}`,
    targetHeadDim === REQUIRED_AWS_HEAD_DIM
      ? ""
      : `AWS product plan target_head_dim ${targetHeadDim} != ${REQUIRED_AWS_HEAD_DIM}`,
    minHiddenDim >= REQUIRED_AWS_MIN_HIDDEN_DIM
      ? ""
      : `AWS product plan min_hidden_dim ${minHiddenDim} < ${REQUIRED_AWS_MIN_HIDDEN_DIM}`,
    maxHiddenDim >= minHiddenDim && maxHiddenDim <= REQUIRED_AWS_MAX_HIDDEN_DIM
      ? ""
      : `AWS product plan max_hidden_dim ${maxHiddenDim} outside ${minHiddenDim}-${REQUIRED_AWS_MAX_HIDDEN_DIM}`,
    Number(architecture.d_model || 0) === REQUIRED_AWS_D_MODEL ? "" : "AWS train-core d_model is not 128",
    Number(architecture.heads || 0) === REQUIRED_AWS_HEADS ? "" : "AWS train-core heads is not 2",
    Number(architecture.head_dim || 0) === REQUIRED_AWS_HEAD_DIM ? "" : "AWS train-core head_dim is not 64",
    architecture.head_dim_power_of_four === true ? "" : "AWS train-core head_dim is not a power of four",
    Number(architecture.hidden_dim || 0) >= 256 &&
    Number(architecture.hidden_dim || 0) <= 512
      ? ""
      : "AWS train-core hidden_dim is outside 256-512",
    minTransformerLayers >= REQUIRED_AWS_MIN_TRANSFORMER_LAYERS
      ? ""
      : `AWS product plan min_transformer_layers ${minTransformerLayers} < ${REQUIRED_AWS_MIN_TRANSFORMER_LAYERS}`,
    minContextSeqLen >= REQUIRED_AWS_MIN_CONTEXT_SEQ_LEN
      ? ""
      : `AWS product plan min_context_seq_len ${minContextSeqLen} < ${REQUIRED_AWS_MIN_CONTEXT_SEQ_LEN}`,
    seqLen >= REQUIRED_AWS_MIN_SEQ_LEN && seqLen <= REQUIRED_AWS_MAX_SEQ_LEN
      ? ""
      : `AWS product plan seq_len ${seqLen} outside ${REQUIRED_AWS_MIN_SEQ_LEN}-${REQUIRED_AWS_MAX_SEQ_LEN}`,
  ].filter(Boolean);
}

function awsCurriculumDenoiseRunnerMisses(awsAttention, requiredFloor) {
  const runner = awsAttention?.curriculum_denoise_runner || {};
  const hasRunner =
    runner.present === true ||
    Number(runner.bridge_pair_count || 0) > 0 ||
    Number(runner.required_bridge_pair_count || 0) > 0;
  if (!hasRunner) {
    return ["AWS curriculum denoise runner proof is missing"];
  }
  const bridgePairCount = Number(runner.bridge_pair_count || 0);
  const requiredBridgePairCount = Math.max(Number(runner.required_bridge_pair_count || 0), requiredFloor);
  return [
    runner.ok === true ? "" : "AWS curriculum denoise runner proof is not ok",
    runner.present === true ? "" : "AWS curriculum denoise runner source is missing",
    runner.min_unique_targets_arg === true
      ? ""
      : "AWS curriculum denoise runner does not pass --min-unique-targets",
    runner.quality_min_unique_targets_arg === true
      ? ""
      : "AWS curriculum denoise runner does not pass --min-denoise-bridge-unique-targets",
    bridgePairCount >= requiredBridgePairCount
      ? ""
      : `AWS curriculum denoise runner bridge_pair_count ${bridgePairCount} < ${requiredBridgePairCount}`,
  ].filter(Boolean);
}

function requirement(key, label, level, ok, evidence = {}, missing = []) {
  return {
    key,
    label,
    level,
    ok: ok === true,
    evidence,
    missing: missing.filter(Boolean),
  };
}

function buildCoverage(diagnostic, config) {
  const evidence = diagnostic.evidence || {};
  const corpus = evidence["v2-corpus-contract"] || {};
  const retrievalHead = corpus.retrieval_head || {};
  const retrievalHeadHashMissing = retrievalHeadHashProblems(corpus, retrievalHead);
  const heldout = evidence["heldout-retrieval-proof"] || {};
  const native = evidence["native-directional-eval"] || {};
  const awsPlan = evidence["aws-product-plan"] || {};
  const awsLaunch = evidence["aws-launch-plan"] || {};
  const awsPrelaunch = evidence["aws-prelaunch-readiness"] || {};
  const qualityCases = checkCaseList(evidence, "quality-report-self-test", REQUIRED_QUALITY_CASES);
  const taskEvalCases = checkCaseList(evidence, "task-eval-self-test", REQUIRED_TASK_EVAL_CASES);
  const tokenLayoutCases = checkCaseList(evidence, "token-layout-self-test", REQUIRED_TOKEN_LAYOUT_CASES);
  const priorSmokeCases = checkCaseList(evidence, "prior-smoke-self-test", REQUIRED_PRIOR_SMOKE_CASES);
  const heldoutRetrievalCases = checkCaseList(
    evidence,
    "heldout-retrieval-proof-self-test",
    REQUIRED_HELDOUT_RETRIEVAL_CASES,
  );
  const groundedCorpusCases = checkCaseList(
    evidence,
    "grounded-corpus-self-test",
    REQUIRED_GROUNDED_CORPUS_CASES,
  );
  const generativeCases = checkCaseList(evidence, "generative-eval-provenance", REQUIRED_GENERATIVE_CASES);
  const generativeProvenance = evidence["generative-eval-provenance"] || {};
  const cleanGeneratedSample = generativeProvenance.clean_sample || {};
  const posthocGeneratedSample = generativeProvenance.posthoc_sample || {};
  const generatedRetrievalHeadBindingMissing = generatedSampleRetrievalHeadBindingProblems(
    cleanGeneratedSample,
    posthocGeneratedSample,
  );
  const generationIntegrityCases = checkCaseList(
    evidence,
    "generation-integrity-self-test",
    REQUIRED_GENERATION_INTEGRITY_CASES,
  );
  const sampleBindingCases = checkCaseList(evidence, "sample-binding-self-test", REQUIRED_SAMPLE_BINDING_CASES);
  const denoiseBridgeCases = checkCaseList(evidence, "denoise-bridge-self-test", REQUIRED_DENOISE_BRIDGE_CASES);
  const promotionCases = checkCaseList(evidence, "promotion-bundle-self-test", REQUIRED_PROMOTION_CASES);
  const releaseCandidateCases = checkCaseList(
    evidence,
    "release-candidate-self-test",
    REQUIRED_RELEASE_CANDIDATE_CASES,
  );
  const releaseCandidateLiveReadinessNextAction = checkReleaseCandidateNextAction(
    evidence,
    RELEASE_CANDIDATE_LIVE_READINESS_CASE,
    REQUIRED_RELEASE_CANDIDATE_LIVE_READINESS_NEXT_ACTION,
  );
  const awsReleaseProofCases = checkCaseList(
    evidence,
    "aws-release-proof-self-test",
    REQUIRED_AWS_RELEASE_PROOF_CASES,
  );
  const awsRunArtifactCases = checkCaseList(
    evidence,
    "aws-run-artifacts-self-test",
    REQUIRED_AWS_RUN_ARTIFACT_CASES,
  );
  const awsRunFetchCases = checkCaseList(
    evidence,
    "aws-run-fetch-self-test",
    REQUIRED_AWS_RUN_FETCH_CASES,
  );
  const awsLiveLaunchReadinessCases = checkCaseList(
    evidence,
    "aws-live-launch-readiness-self-test",
    REQUIRED_AWS_LIVE_LAUNCH_READINESS_CASES,
  );
  const awsLaunchExecuteGuardCases = checkCaseList(
    evidence,
    "aws-launch-execute-guard-self-test",
    REQUIRED_AWS_LAUNCH_EXECUTE_GUARD_CASES,
  );
  const awsAttention = awsPlan.attention || {};
  const awsGeneratedPromptRows = Number(awsAttention.min_generated_prompt_rows || 0);
  const awsGeneratedSelectedRows = Number(awsPlan.generated_prompt_rows || 0);
  const awsGeneratedUniqueTargets = Number(awsPlan.generated_unique_targets || 0);
  const awsGeneratedTop516 = Number(awsAttention.min_generated_top5_16_per_mille || 0);
  const awsGeneratedRetrievalTop1 = Number(awsAttention.min_generated_retrieval_top1_per_mille || 0);
  const awsGeneratedRetrievalTop5 = Number(awsAttention.min_generated_retrieval_top5_per_mille || 0);
  const awsGeneratedRetrievalMargin = Number(awsAttention.min_generated_retrieval_margin || 0);
  const awsGeneratedMeanTargetDistance16 = Number(awsAttention.max_generated_mean_target_distance_16_q8 || 0);
  const awsDenoiseMinUniqueTargets = Number(awsAttention.denoise_min_unique_targets || 0);
  const awsDenoiseRunnerMissing = awsCurriculumDenoiseRunnerMisses(
    awsAttention,
    Math.max(awsDenoiseMinUniqueTargets, REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS),
  );
  const minTaskTargets = awsAttention.min_task_targets || "";
  const minTaskTop5 = awsAttention.min_task_top5_per_mille || "";
  const minPhaseTargets = awsAttention.min_phase_targets || "";
  const awsTrainCoreArchitectureMissing = awsTrainCoreArchitectureMisses(awsAttention);
  const launchProofCommandProblems = postRunProofCommandProblems("launch plan", awsLaunch);
  const prelaunchProofCommandProblems = postRunProofCommandProblems("prelaunch readiness", awsPrelaunch);

  const taskCounts = corpus.task_counts || {};
  const taskMissing = Object.entries(REQUIRED_TASK_COUNTS).filter(
    ([task, minimum]) => Number(taskCounts[task] || 0) < minimum,
  );
  const corpusNegativeCases = Array.isArray(corpus.negative_cases)
    ? corpus.negative_cases.map(String)
    : [];
  const corpusNegativeMissing = REQUIRED_CORPUS_NEGATIVE_CASES.filter(
    (name) => !corpusNegativeCases.includes(name),
  );
  const identityBindingKindMissing = REQUIRED_IDENTITY_BINDING_KINDS.filter(
    (kind) => !metricCovers(retrievalHead.identity_bindings?.by_kind?.[kind], REQUIRED_IDENTITY_BINDING_COUNTS[kind]),
  );
  const imageRetrievalTaskMissing = REQUIRED_IMAGE_RETRIEVAL_TASKS.filter(
    (task) => !metricCovers(retrievalHead.image_tasks?.[task], REQUIRED_IMAGE_RETRIEVAL_TASK_COUNTS[task]),
  );
  const channels = corpus.image_token_channels || [];
  const channelMissing = REQUIRED_IMAGE_CHANNELS.filter((channel) => !channels.includes(channel));
  const imageChannelStats = corpus.image_token_channel_stats || {};
  const imageChannelStatsMissing = REQUIRED_IMAGE_CHANNELS.filter((channel) => {
    const row = imageChannelStats?.[channel] || {};
    const records = Number(row.records || 0);
    return !(
      records >= 72 &&
      Number(row.tokens_per_record || 0) === 256 &&
      Number(row.active_records || 0) === records &&
      Number(row.multi_bin_records || 0) === records &&
      Number(row.distinct_bins || 0) >= 2 &&
      Number(row.nonzero_tokens || 0) > 0 &&
      Number(row.max_bin || 0) > 0 &&
      Number(row.unique_record_hashes || 0) === records &&
      Number(row.duplicate_record_hashes || 0) === 0
    );
  });
  const taskMarkerIntegrity = corpus.task_marker_integrity || {};
  const taskModalityIntegrity = corpus.task_modality_integrity || {};
  const imageChannelMarkerIntegrity = corpus.image_channel_marker_integrity || {};
  const hardNegativeRoles = corpus.hard_negative_roles || {};
  const identityBindingCoverage = corpus.identity_binding_coverage || {};
  const sourceProvenance = corpus.source_provenance || {};
  const sourceProvenanceMissing = sourceProvenanceProblems(sourceProvenance);
  const architecture = native.architecture || {};
  const nativeIntegerTrace = native.integer_trace || {};
  const nativeIntegerTraceMissing = nativeIntegerTraceProblems(nativeIntegerTrace);
  const outputHeads = native.output_heads || {};
  const nativeTasks = native.tasks || {};
  const nativeTaskPhases = native.task_phases || {};
  const nativeEvalScope = native.eval_scope || {};
  const directionalGroups = native.directional_groups || {};
  const requiredDirectionalKeys = Object.keys(REQUIRED_DIRECTIONAL_GROUPS);
  const directionalGroupMissing = requiredDirectionalKeys.filter((key) => !directionalGroups[key]);
  const directionalMissing = requiredDirectionalKeys
    .filter((key) => directionalGroups[key] && directionalGroups[key]?.ok !== true);
  const directionalStatMissing = requiredDirectionalKeys
    .filter((key) => !hasDirectionalMetricStats(directionalGroups[key]?.stats));
  const directionalTop5FloorMissing = requiredDirectionalKeys
    .filter((key) => Number(directionalGroups[key]?.min_top5_accuracy_per_mille) !== 1);
  const directionalTargetMissing = Object.entries(REQUIRED_DIRECTIONAL_GROUPS)
    .filter(([key, requirement]) => directionalTargetCount(directionalGroups[key]) < requirement.min_targets)
    .map(([key, requirement]) => ({
      key,
      actual: directionalTargetCount(directionalGroups[key]),
      minimum: requirement.min_targets,
    }));
  const directionalPhaseTargetMissing = directionalPhaseMisses(directionalGroups);
  const nativeTaskMissing = nativeTaskMetricMisses(nativeTasks);
  const nativeTaskInvalidContextMissing = nativeTaskInvalidContextMisses(nativeTasks);
  const nativeTaskPhaseMissing = nativeTaskPhaseMisses(nativeTaskPhases);
  const minDirectionTop5 = awsPlan.attention?.min_direction_top5_per_mille || "";
  const nativeEvalScopeMissing = nativeEvalScopeMisses(nativeEvalScope, awsPlan.attention || {});
  const nativeBindEpochs = Number(awsPlan.attention?.native_bind_epochs || 0);
  const curriculumStages = awsPlan.attention?.curriculum_stages || [];
  const curriculumRequiredStages = awsPlan.attention?.curriculum_required_stages || [];
  const heldoutPrompts = heldout.heldout_prompts || {};
  const heldoutRows = Number(heldoutPrompts.rows || 0);
  const heldoutMetric = heldoutPrompts.metric || {};

  const requirements = [
    requirement(
      "v2_multimodal_task_corpus",
      "v2 corpus has explicit bidirectional task records for all 72 spirits plus hard negatives",
      "local-real",
      corpus.image_token_profile === "symbolic16" &&
        Number(corpus.examples || 0) > 0 &&
        taskMissing.length === 0 &&
        Number(taskCounts.match || 0) >= REQUIRED_TASK_COUNTS.match &&
        corpusNegativeMissing.length === 0 &&
        imageChannelStatsComplete(imageChannelStats) &&
        integrityComplete(taskMarkerIntegrity, ["hash_mismatches", "marker_mismatches", "out_of_bounds", "missing_offsets"], Number(corpus.examples || 1)) &&
        integrityComplete(taskModalityIntegrity, ["modality_mismatches", "out_of_bounds", "missing_offsets"], Number(corpus.examples || 1)) &&
        imageChannelMarkerIntegrityComplete(imageChannelMarkerIntegrity) &&
        hardNegativeRolesComplete(hardNegativeRoles) &&
        identityBindingCoverageComplete(identityBindingCoverage) &&
        sourceProvenanceMissing.length === 0,
      {
        examples: Number(corpus.examples || 0),
        task_counts: taskCounts,
        required_task_counts: REQUIRED_TASK_COUNTS,
        image_token_profile: corpus.image_token_profile || "",
        image_token_channel_stats: imageChannelStats,
        hard_negative_roles: hardNegativeRoles,
        identity_binding_coverage: identityBindingCoverage,
        source_provenance: sourceProvenance,
        task_marker_integrity: taskMarkerIntegrity,
        task_modality_integrity: taskModalityIntegrity,
        image_channel_marker_integrity: imageChannelMarkerIntegrity,
        negative_cases: corpusNegativeCases,
        required_negative_cases: REQUIRED_CORPUS_NEGATIVE_CASES,
      },
      [
        corpus.image_token_profile === "symbolic16" ? "" : "corpus image_token_profile is not symbolic16",
        Number(corpus.examples || 0) > 0 ? "" : "v2 corpus examples are missing",
        ...taskMissing.map(
          ([task, minimum]) => `task ${task} has ${Number(taskCounts[task] || 0)} records, requires >= ${minimum}`,
        ),
        ...corpusNegativeMissing.map((name) => `v2 corpus contract missing negative case ${name}`),
        ...imageChannelStatsMissing.map((channel) => `image channel ${channel} stats are incomplete`),
        integrityComplete(taskMarkerIntegrity, ["hash_mismatches", "marker_mismatches", "out_of_bounds", "missing_offsets"], Number(corpus.examples || 1))
          ? ""
          : "task marker integrity is incomplete",
        integrityComplete(taskModalityIntegrity, ["modality_mismatches", "out_of_bounds", "missing_offsets"], Number(corpus.examples || 1))
          ? ""
          : "task modality integrity is incomplete",
        imageChannelMarkerIntegrityComplete(imageChannelMarkerIntegrity)
          ? ""
          : "image channel marker integrity is incomplete",
        hardNegativeRolesComplete(hardNegativeRoles) ? "" : "hard-negative role coverage is incomplete",
        identityBindingCoverageComplete(identityBindingCoverage) ? "" : "identity binding coverage summary is incomplete",
        ...sourceProvenanceMissing,
      ],
    ),
    requirement(
      "symbolic_image_channels",
      "image tokens include deterministic ink/edge/component/radial/direction channels",
      "local-real",
      hasAll(channels, REQUIRED_IMAGE_CHANNELS),
      {
        image_token_channels: channels,
      },
      channelMissing.map((channel) => `missing image channel ${channel}`),
    ),
    requirement(
      "integer_token_layout_parity",
      "reserved integer task/image token layout is contract-checked across JS and Rust",
      "contract-self-test",
      tokenLayoutCases.missing_cases.length === 0,
      {
        cases: tokenLayoutCases.cases,
        canonical_layout: evidence["token-layout-self-test"]?.canonical_layout || {},
        required_cases: REQUIRED_TOKEN_LAYOUT_CASES,
      },
      tokenLayoutCases.missing_cases.map((name) => `token-layout self-test missing case ${name}`),
    ),
    requirement(
      "small_integer_architecture",
      "native attention smoke uses the promoted tiny integer transformer shape",
      "local-real",
      Number(architecture.d_model || 0) === 128 &&
        Number(architecture.heads || 0) === 2 &&
        Number(architecture.head_dim || 0) === 64 &&
        Number(architecture.hidden_dim || 0) >= 256 &&
        Number(architecture.hidden_dim || 0) <= 512 &&
        Number(architecture.transformer_layers || 0) >= 2 &&
        Number(architecture.transformer_layers || 0) <= 4 &&
        Number(architecture.context_seq_len || 0) >= 384 &&
        Number(architecture.context_seq_len || 0) <= 768,
      architecture,
      [
        Number(architecture.d_model || 0) === 128 ? "" : "d_model is not 128",
        Number(architecture.heads || 0) === 2 ? "" : "heads is not 2",
        Number(architecture.head_dim || 0) === 64 ? "" : "head_dim is not 64",
        Number(architecture.hidden_dim || 0) >= 256 &&
        Number(architecture.hidden_dim || 0) <= 512
          ? ""
          : "hidden_dim is outside 256-512",
        Number(architecture.transformer_layers || 0) >= 2 &&
        Number(architecture.transformer_layers || 0) <= 4
          ? ""
          : "transformer_layers is outside 2-4",
        Number(architecture.context_seq_len || 0) >= 384 &&
        Number(architecture.context_seq_len || 0) <= 768
          ? ""
          : "context_seq_len is outside 384-768",
      ],
    ),
    requirement(
      "native_integer_train_eval_trace",
      "native train/eval traces expose integer Q8/Q15/I64 metrics with integer-valued numeric leaves",
      "local-real",
      nativeIntegerTraceMissing.length === 0,
      nativeIntegerTrace,
      nativeIntegerTraceMissing,
    ),
    requirement(
      "aws_promoted_small_source_architecture",
      "AWS product plan verifies the checked-out integer training core plus layer/context ratchets before launch",
      "no-spend-plan",
      awsTrainCoreArchitectureMissing.length === 0,
      {
        train_core_architecture: awsAttention.train_core_architecture || {},
        require_architecture_profile: awsAttention.require_architecture_profile === true,
        min_d_model: Number(awsAttention.min_d_model || 0),
        min_heads: Number(awsAttention.min_heads || 0),
        min_hidden_dim: Number(awsAttention.min_hidden_dim || 0),
        min_transformer_layers: Number(awsAttention.min_transformer_layers || 0),
        min_context_seq_len: Number(awsAttention.min_context_seq_len || 0),
        seq_len: Number(awsAttention.seq_len || 0),
      },
      awsTrainCoreArchitectureMissing,
    ),
    requirement(
      "separate_output_heads",
      "eval exposes separate special/text/image token heads",
      "local-real",
      ["special", "text", "image"].every(
        (name) => outputHeads[name]?.source === "nsrllmm-output-token-head" && Number(outputHeads[name]?.targets || 0) > 0,
      ),
      outputHeads,
      ["special", "text", "image"]
        .filter((name) => !(outputHeads[name]?.source === "nsrllmm-output-token-head" && Number(outputHeads[name]?.targets || 0) > 0))
        .map((name) => `${name} output head missing or empty`),
    ),
    requirement(
      "class_retrieval_score_head",
      "72-way class/retrieval score head exposes separate text and image scorers with one verified model hash",
      "local-real",
      Number(retrievalHead.labels || 0) === 72 &&
        Number(retrievalHead.feature_count || 0) > 0 &&
        retrievalHeadHashMissing.length === 0 &&
        retrievalHead.text_head?.present === true &&
        Number(retrievalHead.text_head?.nonzero_weights || 0) > 0 &&
        retrievalHead.image_head?.present === true &&
        Number(retrievalHead.image_head?.nonzero_weights || 0) > 0,
      {
        schema: retrievalHead.schema || "",
        retrieval_model_hash: corpus.retrieval_model_hash || "",
        model_hash: retrievalHead.model_hash || "",
        eval_model_hash: retrievalHead.eval_model_hash || "",
        feature_count: Number(retrievalHead.feature_count || 0),
        labels: Number(retrievalHead.labels || 0),
        text_head: retrievalHead.text_head || {},
        image_head: retrievalHead.image_head || {},
      },
      [
        Number(retrievalHead.labels || 0) === 72 ? "" : "class/retrieval head does not have 72 labels",
        Number(retrievalHead.feature_count || 0) > 0 ? "" : "class/retrieval head has no features",
        ...retrievalHeadHashMissing,
        retrievalHead.text_head?.present === true ? "" : "class/retrieval text scorer is missing",
        Number(retrievalHead.text_head?.nonzero_weights || 0) > 0
          ? ""
          : "class/retrieval text scorer has no nonzero weights",
        retrievalHead.image_head?.present === true ? "" : "class/retrieval image scorer is missing",
        Number(retrievalHead.image_head?.nonzero_weights || 0) > 0
          ? ""
          : "class/retrieval image scorer has no nonzero weights",
      ],
    ),
    requirement(
      "retrieval_identity_spine",
      "72-way text/image retrieval head binds known prompts, identities, images, and hard negatives",
      "local-real",
      Number(retrievalHead.labels || 0) === 72 &&
        retrievalHead.text_head?.present === true &&
        retrievalHead.image_head?.present === true &&
        metricCovers(retrievalHead.known_prompts, REQUIRED_RETRIEVAL_COUNTS.known_prompts) &&
        metricCovers(retrievalHead.identity_bindings?.total, REQUIRED_RETRIEVAL_COUNTS.identity_total) &&
        identityBindingKindMissing.length === 0 &&
        metricCovers(retrievalHead.image_to_text, REQUIRED_RETRIEVAL_COUNTS.image_to_text) &&
        imageRetrievalTaskMissing.length === 0 &&
        metricCovers(retrievalHead.match?.yes, REQUIRED_RETRIEVAL_COUNTS.match_yes) &&
        metricCovers(retrievalHead.match?.no, REQUIRED_RETRIEVAL_COUNTS.match_no) &&
        metricCovers(retrievalHead.match?.no_by_role?.image, REQUIRED_RETRIEVAL_COUNTS.match_no_image) &&
        metricCovers(retrievalHead.match?.no_by_role?.prompt, REQUIRED_RETRIEVAL_COUNTS.match_no_prompt),
      {
        labels: Number(retrievalHead.labels || 0),
        text_head: retrievalHead.text_head || {},
        image_head: retrievalHead.image_head || {},
        known_prompts: retrievalHead.known_prompts || {},
        identity_bindings: retrievalHead.identity_bindings || {},
        image_to_text: retrievalHead.image_to_text || {},
        image_tasks: retrievalHead.image_tasks || {},
        match: retrievalHead.match || {},
        required_counts: {
          ...REQUIRED_RETRIEVAL_COUNTS,
          identity_by_kind: REQUIRED_IDENTITY_BINDING_COUNTS,
          image_tasks: REQUIRED_IMAGE_RETRIEVAL_TASK_COUNTS,
        },
      },
      [
        Number(retrievalHead.labels || 0) === 72 ? "" : "retrieval head does not have 72 labels",
        retrievalHead.text_head?.present === true ? "" : "retrieval text head missing",
        retrievalHead.image_head?.present === true ? "" : "retrieval image head missing",
        metricCovers(retrievalHead.known_prompts, REQUIRED_RETRIEVAL_COUNTS.known_prompts)
          ? ""
          : `known prompt retrieval has ${Number(retrievalHead.known_prompts?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.known_prompts}`,
        metricCovers(retrievalHead.identity_bindings?.total, REQUIRED_RETRIEVAL_COUNTS.identity_total)
          ? ""
          : `identity binding retrieval has ${Number(retrievalHead.identity_bindings?.total?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.identity_total}`,
        ...identityBindingKindMissing.map((kind) => {
          const metric = retrievalHead.identity_bindings?.by_kind?.[kind] || {};
          return `identity binding retrieval kind ${kind} has ${Number(metric.count || 0)} rows and ${Number(metric.top1 || 0)} top1, requires >= ${REQUIRED_IDENTITY_BINDING_COUNTS[kind]} rows with top1=count`;
        }),
        metricCovers(retrievalHead.image_to_text, REQUIRED_RETRIEVAL_COUNTS.image_to_text)
          ? ""
          : `image-to-text retrieval has ${Number(retrievalHead.image_to_text?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.image_to_text}`,
        ...imageRetrievalTaskMissing.map((task) => {
          const metric = retrievalHead.image_tasks?.[task] || {};
          return `image task retrieval ${task} has ${Number(metric.count || 0)} rows and ${Number(metric.top1 || 0)} top1, requires >= ${REQUIRED_IMAGE_RETRIEVAL_TASK_COUNTS[task]} rows with top1=count`;
        }),
        metricCovers(retrievalHead.match?.yes, REQUIRED_RETRIEVAL_COUNTS.match_yes)
          ? ""
          : `match yes retrieval has ${Number(retrievalHead.match?.yes?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.match_yes}`,
        metricCovers(retrievalHead.match?.no, REQUIRED_RETRIEVAL_COUNTS.match_no)
          ? ""
          : `match no retrieval has ${Number(retrievalHead.match?.no?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.match_no}`,
        metricCovers(retrievalHead.match?.no_by_role?.image, REQUIRED_RETRIEVAL_COUNTS.match_no_image)
          ? ""
          : `wrong-image hard negatives have ${Number(retrievalHead.match?.no_by_role?.image?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.match_no_image}`,
        metricCovers(retrievalHead.match?.no_by_role?.prompt, REQUIRED_RETRIEVAL_COUNTS.match_no_prompt)
          ? ""
          : `wrong-prompt hard negatives have ${Number(retrievalHead.match?.no_by_role?.prompt?.count || 0)} rows, requires >= ${REQUIRED_RETRIEVAL_COUNTS.match_no_prompt}`,
      ],
    ),
    requirement(
      "heldout_prompt_generalization",
      "held-out paraphrases retrieve the correct spirit top-1/top-5",
      "local-real",
      heldoutRows >= REQUIRED_RETRIEVAL_COUNTS.heldout_prompts &&
        Number(heldoutPrompts.unique_targets || 0) === 72 &&
        metricMatchesRows(heldoutMetric, heldoutRows),
      {
        ...heldoutPrompts,
        required_counts: {
          rows: REQUIRED_RETRIEVAL_COUNTS.heldout_prompts,
          metric_count: "rows",
        },
      },
      [
        heldoutRows >= REQUIRED_RETRIEVAL_COUNTS.heldout_prompts
          ? ""
          : `held-out prompt rows ${heldoutRows} < ${REQUIRED_RETRIEVAL_COUNTS.heldout_prompts}`,
        Number(heldoutPrompts.unique_targets || 0) === 72 ? "" : "held-out prompts do not cover 72 targets",
        Number(heldoutMetric.count || 0) === heldoutRows
          ? ""
          : `held-out retrieval metric has ${Number(heldoutMetric.count || 0)} rows, expected ${heldoutRows}`,
        metricMatchesRows(heldoutMetric, heldoutRows) ? "" : "held-out top-1/top-5 retrieval is incomplete for the full held-out row count",
      ],
    ),
    requirement(
      "heldout_retrieval_contract",
      "held-out retrieval self-test rejects stale prompt provenance, row-count drift, weak top-1/margin, missing image scorer, and stale model hash",
      "contract-self-test",
      heldoutRetrievalCases.missing_cases.length === 0,
      { cases: heldoutRetrievalCases.cases },
      heldoutRetrievalCases.missing_cases.map((name) => `held-out retrieval self-test missing case ${name}`),
    ),
    requirement(
      "grounded_source_contract",
      "grounded-corpus self-test rejects weak source overlap, placeholders, generic attribute ranks, bad source hashes, non-source prompts, and missing attribute task coverage",
      "contract-self-test",
      groundedCorpusCases.missing_cases.length === 0,
      { cases: groundedCorpusCases.cases },
      groundedCorpusCases.missing_cases.map((name) => `grounded-corpus self-test missing case ${name}`),
    ),
    requirement(
      "native_directional_multimodal_eval",
      "native eval proves text-to-image, image-to-text, text+seal, and identity/source directions with per-direction stats",
      "local-real",
      directionalGroupMissing.length === 0 &&
        directionalMissing.length === 0 &&
        directionalStatMissing.length === 0 &&
        directionalTop5FloorMissing.length === 0 &&
        directionalTargetMissing.length === 0 &&
        directionalPhaseTargetMissing.length === 0 &&
        nativeTaskMissing.length === 0 &&
        nativeTaskInvalidContextMissing.length === 0 &&
        nativeTaskPhaseMissing.length === 0,
      {
        directional_groups: directionalGroups,
        tasks: nativeTasks,
        task_phases: nativeTaskPhases,
      },
      [
        ...directionalGroupMissing.map((key) => `missing directional group ${key}`),
        ...directionalMissing.map((key) => `directional group ${key} is not ok`),
        ...directionalStatMissing.map((key) => `directional group ${key} is missing per-direction top-k stats`),
        ...directionalTop5FloorMissing.map((key) => `directional group ${key} is missing native top-5 floor all=1`),
        ...directionalTargetMissing.map(
          (item) => `directional group ${item.key} has ${item.actual} targets, requires >= ${item.minimum}`,
        ),
        ...directionalPhaseTargetMissing.map(
          (item) =>
            `directional group ${item.key} phase ${item.phase} has ${item.actual} targets, requires >= ${item.minimum}`,
        ),
        ...nativeTaskMissing.map((item) => `native eval task ${item.task} has ${item.targets} targets, requires > 0`),
        ...nativeTaskInvalidContextMissing.map(
          (item) => `native eval task ${item.task} has ${item.invalid_contexts} invalid contexts`,
        ),
        ...nativeTaskPhaseMissing.map(
          (item) => `native eval task ${item.task} phase targets total ${item.targets}, requires > 0`,
        ),
      ],
    ),
    requirement(
      "native_eval_scope_accounting",
      "native eval evidence declares local smoke scope while AWS release plan requires full 72-target task breadth",
      "local-real+no-spend-plan",
      nativeEvalScopeMissing.length === 0,
      {
        eval_scope: nativeEvalScope,
        aws_eval_max_examples: awsPlan.attention?.eval_max_examples || "",
        aws_min_task_targets: minTaskTargets,
        aws_min_phase_targets: minPhaseTargets,
      },
      nativeEvalScopeMissing,
    ),
    requirement(
      "directional_quality_ratchet",
      "product plan carries a per-direction top-k floor for bidirectional quality",
      "no-spend-plan",
      minDirectionTop5 === "all=1",
      {
        min_direction_accuracy_per_mille: awsPlan.attention?.min_direction_accuracy_per_mille || "",
        min_direction_top5_per_mille: minDirectionTop5,
        min_direction_top10_per_mille: awsPlan.attention?.min_direction_top10_per_mille || "",
      },
      [
        minDirectionTop5 === "all=1"
          ? ""
          : `min_direction_top5_per_mille ${JSON.stringify(minDirectionTop5)} != "all=1"`,
      ],
    ),
    requirement(
      "native_task_quality_ratchet",
      "product plan carries native per-task target/top-5 floors and per-phase target floors",
      "no-spend-plan",
      minTaskTargets === REQUIRED_AWS_MIN_TASK_TARGETS &&
        minTaskTop5 === REQUIRED_AWS_MIN_TASK_TOP5_PER_MILLE &&
        minPhaseTargets === REQUIRED_AWS_MIN_PHASE_TARGETS,
      {
        min_task_targets: minTaskTargets,
        required_min_task_targets: REQUIRED_AWS_MIN_TASK_TARGETS,
        min_task_top5_per_mille: minTaskTop5,
        required_min_task_top5_per_mille: REQUIRED_AWS_MIN_TASK_TOP5_PER_MILLE,
        min_phase_targets: minPhaseTargets,
        required_min_phase_targets: REQUIRED_AWS_MIN_PHASE_TARGETS,
      },
      [
        minTaskTargets === REQUIRED_AWS_MIN_TASK_TARGETS
          ? ""
          : `min_task_targets ${JSON.stringify(minTaskTargets)} != ${JSON.stringify(REQUIRED_AWS_MIN_TASK_TARGETS)}`,
        minTaskTop5 === REQUIRED_AWS_MIN_TASK_TOP5_PER_MILLE
          ? ""
          : `min_task_top5_per_mille ${JSON.stringify(minTaskTop5)} != ${JSON.stringify(REQUIRED_AWS_MIN_TASK_TOP5_PER_MILLE)}`,
        minPhaseTargets === REQUIRED_AWS_MIN_PHASE_TARGETS
          ? ""
          : `min_phase_targets ${JSON.stringify(minPhaseTargets)} != ${JSON.stringify(REQUIRED_AWS_MIN_PHASE_TARGETS)}`,
      ],
    ),
    requirement(
      "denoise_bridge_target_ratchet",
      "AWS product plan carries an executable distinct-target floor for the denoise-to-attention bridge",
      "no-spend-plan",
      awsDenoiseMinUniqueTargets >= REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS &&
        awsDenoiseRunnerMissing.length === 0,
      {
        denoise_min_unique_targets: awsDenoiseMinUniqueTargets,
        required_min_unique_targets: REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS,
        curriculum_denoise_runner: awsAttention.curriculum_denoise_runner || {},
      },
      [
        awsDenoiseMinUniqueTargets >= REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS
          ? ""
          : `denoise_min_unique_targets ${JSON.stringify(awsAttention.denoise_min_unique_targets || "")} < ${REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS}`,
        ...awsDenoiseRunnerMissing,
      ],
    ),
    requirement(
      "ordered_training_curriculum",
      "AWS product plan carries the ordered identity/image/text-to-image/image-to-text/explain/hard-negative/native-bind curriculum with a strengthened native-bind pass",
      "no-spend-plan",
      sameSequence(curriculumStages, REQUIRED_CURRICULUM_STAGES) &&
        sameSequence(curriculumRequiredStages, REQUIRED_CURRICULUM_STAGES) &&
        nativeBindEpochs >= REQUIRED_AWS_NATIVE_BIND_EPOCHS,
      {
        curriculum_stages: curriculumStages,
        curriculum_required_stages: curriculumRequiredStages,
        expected: REQUIRED_CURRICULUM_STAGES,
        native_bind_epochs: nativeBindEpochs,
        required_native_bind_epochs: REQUIRED_AWS_NATIVE_BIND_EPOCHS,
      },
      [
        sameSequence(curriculumStages, REQUIRED_CURRICULUM_STAGES)
          ? ""
          : `curriculum_stages ${JSON.stringify(curriculumStages)} != ${JSON.stringify(REQUIRED_CURRICULUM_STAGES)}`,
        sameSequence(curriculumRequiredStages, REQUIRED_CURRICULUM_STAGES)
          ? ""
          : `curriculum_required_stages ${JSON.stringify(curriculumRequiredStages)} != ${JSON.stringify(REQUIRED_CURRICULUM_STAGES)}`,
        nativeBindEpochs >= REQUIRED_AWS_NATIVE_BIND_EPOCHS
          ? ""
          : `native_bind_epochs ${JSON.stringify(nativeBindEpochs)} < ${REQUIRED_AWS_NATIVE_BIND_EPOCHS}`,
      ],
    ),
    requirement(
      "attention_task_eval_contract",
      "task eval self-test rejects stale eval/corpus provenance and broken multimodal task metrics, markers, modality order, and image-channel bytes",
      "contract-self-test",
      taskEvalCases.missing_cases.length === 0,
      { cases: taskEvalCases.cases },
      taskEvalCases.missing_cases.map((name) => `task-eval self-test missing case ${name}`),
    ),
    requirement(
      "prompt_to_layout_prior_contract",
      "prior-smoke self-test rejects wrong latent routing, missing seed variants, collapsed layouts, and weak held-out class eval",
      "contract-self-test",
      priorSmokeCases.missing_cases.length === 0,
      { cases: priorSmokeCases.cases },
      priorSmokeCases.missing_cases.map((name) => `prior-smoke self-test missing case ${name}`),
    ),
    requirement(
      "quality_report_contract",
      "quality report self-test rejects weak retrieval, symbolic-token, source, and curriculum failures",
      "contract-self-test",
      qualityCases.missing_cases.length === 0,
      { cases: qualityCases.cases },
      qualityCases.missing_cases.map((name) => `quality-report self-test missing case ${name}`),
    ),
    requirement(
      "generation_and_denoise_guardrails",
      "generation-integrity, sample-binding, denoise, and promotion self-tests reject target guidance, identity drift, cleanup, bad generated ranks, and stale confidence",
      "contract-self-test",
      generativeCases.missing_cases.length === 0 &&
        generatedSampleIdentityComplete(cleanGeneratedSample) &&
        generatedSampleIdentityComplete(posthocGeneratedSample) &&
        generatedSamplePromptSelectionComplete(cleanGeneratedSample) &&
        generatedSamplePromptSelectionComplete(posthocGeneratedSample) &&
        generatedRetrievalHeadBindingMissing.length === 0 &&
        generatedSampleLatentProvenanceComplete(cleanGeneratedSample) &&
        generatedSampleLatentProvenanceComplete(posthocGeneratedSample) &&
        generationIntegrityCases.missing_cases.length === 0 &&
        sampleBindingCases.missing_cases.length === 0 &&
        denoiseBridgeCases.missing_cases.length === 0 &&
        promotionCases.missing_cases.length === 0,
      {
        generative_eval_cases: generativeCases.cases,
        clean_generated_sample: cleanGeneratedSample,
        posthoc_generated_sample: posthocGeneratedSample,
        generation_integrity_cases: generationIntegrityCases.cases,
        sample_binding_cases: sampleBindingCases.cases,
        denoise_bridge_cases: denoiseBridgeCases.cases,
        promotion_bundle_cases: promotionCases.cases,
      },
      [
        ...generativeCases.missing_cases.map((name) => `generative provenance self-test missing case ${name}`),
        generatedSampleIdentityComplete(cleanGeneratedSample)
          ? ""
          : "clean generated sample retrieval identity evidence is incomplete",
        generatedSampleIdentityComplete(posthocGeneratedSample)
          ? ""
          : "post-hoc generated sample retrieval identity evidence is incomplete",
        generatedSamplePromptSelectionComplete(cleanGeneratedSample)
          ? ""
          : "clean generated sample prompt selection provenance is incomplete",
        generatedSamplePromptSelectionComplete(posthocGeneratedSample)
          ? ""
          : "post-hoc generated sample prompt selection provenance is incomplete",
        ...generatedRetrievalHeadBindingMissing,
        generatedSampleLatentProvenanceComplete(cleanGeneratedSample)
          ? ""
          : "clean generated sample latent model provenance is incomplete",
        generatedSampleLatentProvenanceComplete(posthocGeneratedSample)
          ? ""
          : "post-hoc generated sample latent model provenance is incomplete",
        ...generationIntegrityCases.missing_cases.map((name) => `generation-integrity self-test missing case ${name}`),
        ...sampleBindingCases.missing_cases.map((name) => `sample-binding self-test missing case ${name}`),
        ...denoiseBridgeCases.missing_cases.map((name) => `denoise bridge self-test missing case ${name}`),
        ...promotionCases.missing_cases.map((name) => `promotion bundle self-test missing case ${name}`),
      ],
    ),
    requirement(
      "release_candidate_handoff_contract",
      "release-candidate handoff self-test rejects skipped proof, weak objective evidence, broken prelaunch, missing live-readiness guidance, and hidden gaps",
      "contract-self-test",
      releaseCandidateCases.missing_cases.length === 0 && releaseCandidateLiveReadinessNextAction.ok,
      {
        cases: releaseCandidateCases.cases,
        live_readiness_next_action: releaseCandidateLiveReadinessNextAction,
      },
      [
        ...releaseCandidateCases.missing_cases.map((name) => `release-candidate self-test missing case ${name}`),
        releaseCandidateLiveReadinessNextAction.ok
          ? ""
          : "release-candidate self-test missing live-readiness next-action proof",
      ],
    ),
    requirement(
      "aws_graviton_cpu_scaling_plan",
      "AWS plan/prelaunch path targets Graviton with map-reduce auto CPU scaling",
      "no-spend-plan",
      awsPlan.runner?.require_graviton === true &&
        awsPlan.s3_required === true &&
        awsAttention.cpu_scaling?.policy === "auto-online-processors" &&
        awsAttention.cpu_scaling?.auto_workers === true &&
        awsAttention.require_promoted_small_profile === true &&
        awsAttention.require_generative_eval === true &&
        awsAttention.require_generative_output_identity === true &&
        awsGeneratedPromptRows >= 72 &&
        awsGeneratedSelectedRows >= 72 &&
        awsGeneratedUniqueTargets >= 72 &&
        awsGeneratedTop516 >= 1 &&
        awsGeneratedRetrievalTop1 >= 1000 &&
        awsGeneratedRetrievalTop5 >= 1000 &&
        awsGeneratedRetrievalMargin >= REQUIRED_AWS_MIN_GENERATED_RETRIEVAL_MARGIN &&
        Number.isInteger(awsGeneratedMeanTargetDistance16) &&
        awsGeneratedMeanTargetDistance16 >= 1 &&
        awsGeneratedMeanTargetDistance16 <= REQUIRED_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8 &&
        awsDenoiseMinUniqueTargets >= REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS &&
        awsDenoiseRunnerMissing.length === 0 &&
        awsReleaseProofCases.missing_cases.length === 0 &&
        awsRunArtifactCases.missing_cases.length === 0 &&
        awsRunFetchCases.missing_cases.length === 0 &&
        awsLiveLaunchReadinessCases.missing_cases.length === 0 &&
        awsLaunchExecuteGuardCases.missing_cases.length === 0 &&
        awsLaunch.graviton_instance === true &&
        awsLaunch.cpu_scaling?.policy === "auto-online-processors" &&
        launchProofCommandProblems.length === 0 &&
        awsPrelaunch.launch_ready === true &&
        awsPrelaunch.graviton_instance === true &&
        prelaunchProofCommandProblems.length === 0,
      {
        product_plan: awsPlan,
        launch_plan: awsLaunch,
        prelaunch_readiness: awsPrelaunch,
        generated_prompt_rows: awsGeneratedSelectedRows,
        generated_unique_targets: awsGeneratedUniqueTargets,
        required_generated_prompt_rows: 72,
        required_generated_unique_targets: 72,
        min_generated_retrieval_margin: awsGeneratedRetrievalMargin,
        required_min_generated_retrieval_margin:
          REQUIRED_AWS_MIN_GENERATED_RETRIEVAL_MARGIN,
        max_generated_mean_target_distance_16_q8: awsGeneratedMeanTargetDistance16,
        required_max_generated_mean_target_distance_16_q8:
          REQUIRED_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8,
        denoise_min_unique_targets: awsDenoiseMinUniqueTargets,
        required_denoise_min_unique_targets:
          REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS,
        curriculum_denoise_runner: awsAttention.curriculum_denoise_runner || {},
        aws_release_proof_cases: awsReleaseProofCases.cases,
        aws_run_artifact_cases: awsRunArtifactCases.cases,
        aws_run_fetch_cases: awsRunFetchCases.cases,
        aws_live_launch_readiness_cases: awsLiveLaunchReadinessCases.cases,
        aws_launch_execute_guard_cases: awsLaunchExecuteGuardCases.cases,
      },
      [
        awsPlan.runner?.require_graviton === true ? "" : "AWS product plan does not require Graviton",
        awsAttention.cpu_scaling?.policy === "auto-online-processors" ? "" : "attention CPU scaling policy is not auto-online-processors",
        awsAttention.cpu_scaling?.auto_workers === true ? "" : "attention auto workers are not enabled",
        awsAttention.require_promoted_small_profile === true ? "" : "AWS product plan does not require promoted small profile",
        awsAttention.require_generative_eval === true ? "" : "AWS product plan does not require generated eval",
        awsAttention.require_generative_output_identity === true ? "" : "AWS product plan does not require generated output identity",
        awsGeneratedPromptRows >= 72 ? "" : "AWS generated prompt-row floor is below 72",
        awsGeneratedSelectedRows >= 72
          ? ""
          : `AWS generated held-out prompt rows ${awsGeneratedSelectedRows} < 72`,
        awsGeneratedUniqueTargets >= 72
          ? ""
          : `AWS generated held-out unique targets ${awsGeneratedUniqueTargets} < 72`,
        awsGeneratedTop516 >= 1 ? "" : "AWS generated 16x16 top-5 floor is below 1",
        awsGeneratedRetrievalTop1 >= 1000 ? "" : "AWS generated retrieval top-1 floor is below 1000",
        awsGeneratedRetrievalTop5 >= 1000 ? "" : "AWS generated retrieval top-5 floor is below 1000",
        awsGeneratedRetrievalMargin >= REQUIRED_AWS_MIN_GENERATED_RETRIEVAL_MARGIN
          ? ""
          : `AWS generated retrieval margin floor ${JSON.stringify(awsAttention.min_generated_retrieval_margin || "")} < ${REQUIRED_AWS_MIN_GENERATED_RETRIEVAL_MARGIN}`,
        Number.isInteger(awsGeneratedMeanTargetDistance16) &&
        awsGeneratedMeanTargetDistance16 >= 1 &&
        awsGeneratedMeanTargetDistance16 <= REQUIRED_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8
          ? ""
          : `AWS generated 16x16 target-distance cap ${JSON.stringify(awsAttention.max_generated_mean_target_distance_16_q8 || "")} is outside 1-${REQUIRED_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8}`,
        awsDenoiseMinUniqueTargets >= REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS
          ? ""
          : `AWS denoise bridge unique-target floor ${JSON.stringify(awsAttention.denoise_min_unique_targets || "")} < ${REQUIRED_AWS_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS}`,
        ...awsDenoiseRunnerMissing,
        ...awsReleaseProofCases.missing_cases.map((name) => `AWS release proof self-test missing case ${name}`),
        ...awsRunArtifactCases.missing_cases.map((name) => `AWS run artifact self-test missing case ${name}`),
        ...awsRunFetchCases.missing_cases.map((name) => `AWS run fetch self-test missing case ${name}`),
        ...awsLiveLaunchReadinessCases.missing_cases.map((name) => `AWS live launch readiness self-test missing case ${name}`),
        ...awsLaunchExecuteGuardCases.missing_cases.map((name) => `AWS launch execute guard self-test missing case ${name}`),
        awsLaunch.graviton_instance === true ? "" : "launch plan is not Graviton",
        ...launchProofCommandProblems,
        awsPrelaunch.launch_ready === true ? "" : "prelaunch readiness is not green",
        ...prelaunchProofCommandProblems,
      ],
    ),
    requirement(
      "synced_graviton_release_run",
      "a real synced Graviton product run validates promotion bundle and quality report",
      "release-real",
      diagnostic.release_product_proof === true && diagnostic.live_product_evidence?.ok === true,
      {
        release_product_proof: diagnostic.release_product_proof === true,
        live_product_evidence: diagnostic.live_product_evidence || {},
        remaining_product_evidence: diagnostic.remaining_product_evidence || [],
      },
      diagnostic.release_product_proof === true
        ? []
        : diagnostic.remaining_product_evidence || ["release_product_proof is not true"],
    ),
  ];

  const localRequirements = requirements.filter((item) => item.level !== "release-real");
  const localObjectiveProof = localRequirements.every((item) => item.ok);
  const releaseObjectiveProof = requirements.every((item) => item.ok);
  const ok = localObjectiveProof && (!config.requireRelease || releaseObjectiveProof);
  return {
    schema,
    ok,
    require_release: config.requireRelease,
    diagnostic_schema: diagnostic.schema || "",
    diagnostic_ok: diagnostic.ok === true,
    diagnostic_path: path.resolve(config.diagnosticPath),
    local_product_proof: diagnostic.local_product_proof === true,
    release_product_proof: diagnostic.release_product_proof === true,
    local_objective_proof: localObjectiveProof,
    release_objective_proof: releaseObjectiveProof,
    requirements,
    missing: requirements
      .filter((item) => !item.ok && (config.requireRelease || item.level !== "release-real"))
      .flatMap((item) => item.missing.map((reason) => `${item.key}: ${reason}`)),
    remaining_release_evidence:
      diagnostic.release_product_proof === true ? [] : diagnostic.remaining_product_evidence || [],
  };
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  const resolved = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const diagnostic = readJson(config.diagnosticPath);
  const report = buildCoverage(diagnostic, config);
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
