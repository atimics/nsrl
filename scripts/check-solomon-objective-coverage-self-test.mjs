#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_objective_coverage_self_test.v1";
const QUALITY_CASES = [
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
const TASK_EVAL_CASES = [
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
const TOKEN_LAYOUT_CASES = [
  "good-js-corpus-builder-layout",
  "good-rust-native-attention-layout",
  "good-js-fallback-layouts",
  "good-js-retrieval-consumer-layouts",
  "good-shared-symbolic-image-defaults",
  "bad-js-layout-mismatch",
  "bad-rust-task-marker-mismatch",
  "bad-shared-marker-order",
];
const PRIOR_SMOKE_CASES = [
  "good",
  "bad-target-source",
  "bad-missing-seed-variant",
  "bad-collapsed-interclass",
  "bad-eval-class-top1",
];
const HELDOUT_RETRIEVAL_CASES = [
  "good",
  "bad-prompts-hash",
  "bad-heldout-row-count",
  "bad-heldout-top1",
  "bad-heldout-margin",
  "bad-missing-image-head",
  "bad-stale-model-hash",
];
const GROUNDED_CORPUS_CASES = [
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
const NATIVE_EVAL_TASK_PHASES = {
  "canonical-joint": { text: 2, image: 2 },
  identify: { text: 2 },
  "text-to-image": { image: 2 },
  "image-to-text": { text: 2 },
  "image-to-explain": { text: 2 },
  "text-image-explain": { text: 2 },
  "image-to-attributes": { text: 2 },
  explain: { text: 2 },
  "description-to-image": { image: 2 },
  match: { text: 6 },
};
const CORPUS_NEGATIVE_CASES = [
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
const SOURCE_PROVENANCE_TASKS = [
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
];
const SOURCE_QUERY_KIND_BY_TASK = {
  "text-to-image": "identity-to-image",
  "image-to-text": "image-identity",
  "image-to-explain": "image-source",
  "text-image-explain": "text-image-source",
  "image-to-attributes": "image-attributes",
  explain: "primary-name",
  "description-to-image": "source-description",
};
const GENERATIVE_CASES = [
  "good",
  "posthoc-score",
  "posthoc-bad-raw-path",
  "bad-raw-path",
  "bad-cleanup",
  "bad-missing-trace",
  "bad-empty-raw",
];
const GENERATION_INTEGRITY_CASES = [
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
const SAMPLE_BINDING_CASES = [
  "good",
  "bad-generated-text",
  "bad-generated-image",
  "bad-cleanup",
];
const DENOISE_BRIDGE_CASES = [
  "good",
  "bad-cleanup",
  "bad-source",
  "bad-signature",
  "bad-flat-output",
  "bad-retrieval-head-hash",
  "bad-output-retrieval-margin",
  "bad-unique-targets",
];
const PROMOTION_CASES = [
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
const RELEASE_CANDIDATE_CASES = [
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
const RELEASE_CANDIDATE_LIVE_READINESS_NEXT_ACTION = [
  "scripts/check-solomon-aws-live-launch-readiness.sh",
  "scripts/aws/launch-solomon-product-run.sh --execute",
  "scripts/aws/prove-solomon-product-run.sh",
  "--s3-pipeline-uri",
  "--launch-dir",
  "--require-launch-dir",
  "launch-result.json",
];
const AWS_RELEASE_PROOF_CASES = [
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
const AWS_RUN_ARTIFACT_CASES = [
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
const AWS_RUN_FETCH_CASES = [
  "good",
  "good-run-name",
  "bad-mismatched-s3-pipeline",
  "bad-missing-status",
  "bad-stale-promotion",
  "bad-native-product-eval-scope",
];
const AWS_LIVE_LAUNCH_READINESS_CASES = [
  "good-explicit-s3-artifact",
  "bad-missing-explicit-s3-artifact",
  "bad-missing-explicit-ami",
];
const AWS_LAUNCH_EXECUTE_GUARD_CASES = [
  "bad-execute-missing-explicit-s3-blocks-before-aws",
  "bad-execute-missing-explicit-artifact-blocks-before-aws",
  "bad-execute-prelaunch-blocks-before-aws",
  "good-execute-records-launch-result",
  "good-execute-command-matches-launch-manifest",
  "good-execute-command-matches-launch-manifest-with-profile",
];
const CURRICULUM_STAGES = [
  "identity",
  "image",
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "explain",
  "hard-negative",
  "native-bind",
];

function usage() {
  console.log([
    "Usage: check-solomon-objective-coverage-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic Solomon product diagnostics and checks that objective",
    "coverage accepts complete local/release evidence while rejecting stale",
    "quality self-test evidence and missing release-run proof.",
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

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function metric(count) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: 10,
    mean_margin: 12,
  };
}

function syntheticDiagnostic({ release = false, mutate = () => {} } = {}) {
  const diagnostic = {
    schema: "nsrl.solomon_product_diagnostic_check.v1",
    ok: true,
    full_product_proof: true,
    local_product_proof: true,
    release_product_proof: release,
    live_product_evidence: {
      required: release,
      provided: release,
      ok: release,
      run_dir: release ? "/synthetic/graviton-run" : "",
      check: release ? { name: "aws-run-artifacts", ok: true, status: 0 } : null,
    },
    remaining_product_evidence: release ? [] : ["no synced real Graviton product run was supplied with --aws-run-dir"],
    skipped: [],
    evidence: {
      "v2-corpus-contract": syntheticCorpusEvidence(),
      "heldout-retrieval-proof": syntheticHeldoutEvidence(),
      "heldout-retrieval-proof-self-test": { cases: [...HELDOUT_RETRIEVAL_CASES] },
      "grounded-corpus-self-test": { cases: [...GROUNDED_CORPUS_CASES] },
      "native-directional-eval": syntheticNativeEvidence(),
      "task-eval-self-test": { cases: [...TASK_EVAL_CASES] },
      "token-layout-self-test": {
        cases: [...TOKEN_LAYOUT_CASES],
        canonical_layout: {
          pad: 0,
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
          text_base: 16,
          text_count: 128,
          image_base: 144,
          image_bins: 16,
        },
      },
      "prior-smoke-self-test": { cases: [...PRIOR_SMOKE_CASES] },
      "quality-report-self-test": { cases: [...QUALITY_CASES] },
      "generative-eval-provenance": {
        cases: [...GENERATIVE_CASES],
        clean_sample: syntheticGeneratedSample(),
        posthoc_sample: syntheticGeneratedSample(),
      },
      "generation-integrity-self-test": { cases: [...GENERATION_INTEGRITY_CASES] },
      "sample-binding-self-test": { cases: [...SAMPLE_BINDING_CASES] },
      "denoise-bridge-self-test": { cases: [...DENOISE_BRIDGE_CASES] },
      "promotion-bundle-self-test": { cases: [...PROMOTION_CASES] },
      "release-candidate-self-test": {
        cases: [...RELEASE_CANDIDATE_CASES],
        next_action_cases: {
          [RELEASE_CANDIDATE_LIVE_READINESS_CASE]: {
            expected_next_action_includes: [...RELEASE_CANDIDATE_LIVE_READINESS_NEXT_ACTION],
            matched_next_action_includes: [...RELEASE_CANDIDATE_LIVE_READINESS_NEXT_ACTION],
          },
        },
      },
      "aws-release-proof-self-test": { cases: [...AWS_RELEASE_PROOF_CASES] },
      "aws-run-artifacts-self-test": { cases: [...AWS_RUN_ARTIFACT_CASES] },
      "aws-run-fetch-self-test": { cases: [...AWS_RUN_FETCH_CASES] },
      "aws-live-launch-readiness-self-test": { cases: [...AWS_LIVE_LAUNCH_READINESS_CASES] },
      "aws-launch-execute-guard-self-test": { cases: [...AWS_LAUNCH_EXECUTE_GUARD_CASES] },
      "aws-product-plan": syntheticAwsProductPlan(),
      "aws-launch-plan": syntheticAwsLaunchPlan(),
      "aws-prelaunch-readiness": syntheticAwsPrelaunchReadiness(),
    },
  };
  mutate(diagnostic);
  return diagnostic;
}

function syntheticGeneratedSample() {
  return {
    sampler_target_source: "decoded-latent",
    latent_model: "/synthetic/latent.nsrllat",
    latent_model_hash: "0x1234abcd",
    latent_model_config_hash: "0x1234abcd",
    latent_model_provenance_hash: "0x1234abcd",
    latent_model_provenance_path: "/synthetic/latent.nsrllat",
    generated_retrieval_rank: 1,
    generated_retrieval_identity: 1,
    mean_generated_retrieval_rank_q8: 256,
    generated_retrieval_top1_per_mille: 1000,
    generated_retrieval_top5_per_mille: 1000,
    retrieval_head_model_hash: "fixture-solomon-retrieval-head",
    selected_prompt_rows: 1,
    selected_prompt_eligible_rows: 1,
    selected_prompt_unique_targets: 1,
    selected_prompt_eligible_unique_targets: 1,
    selected_prompt_sources: { generated: 1 },
    selected_prompt_tiers: { "tier-novel-vocab": 1 },
  };
}

function syntheticCorpusEvidence() {
  return {
    examples: 1872,
    retrieval_model_hash: "0xsynthetic",
    task_counts: {
      identify: 576,
      "text-to-image": 576,
      "canonical-joint": 72,
      "image-to-text": 72,
      "image-to-explain": 72,
      "text-image-explain": 72,
      "image-to-attributes": 72,
      explain: 72,
      "description-to-image": 72,
      match: 216,
    },
    image_token_profile: "symbolic16",
    image_token_channels: ["ink", "edge", "component", "radial", "direction"],
    image_token_channel_stats: syntheticImageChannelStats(),
    hard_negative_roles: {
      no_rows: 144,
      image_role_spirits: 72,
      prompt_role_spirits: 72,
      nearest_image_token_rows: 144,
      rank1_rows: 144,
      positive_distance_rows: 144,
    },
    identity_binding_coverage: syntheticIdentityBindingCoverage(),
    source_provenance: syntheticSourceProvenance(),
    task_marker_integrity: {
      ok: true,
      checked_records: 1872,
      hash_mismatches: 0,
      marker_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
      by_task: {},
    },
    task_modality_integrity: {
      ok: true,
      checked_records: 1872,
      modality_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
      by_task: {},
    },
    image_channel_marker_integrity: {
      ok: true,
      checked_records: 1224,
      required_channels: ["ink", "edge", "component", "radial", "direction"],
      missing_offsets: 0,
      out_of_bounds: 0,
      missing_image_markers: 0,
      missing_channel_markers: 0,
      short_channel_payloads: 0,
      bad_channel_payloads: 0,
      channel_order_mismatches: 0,
      by_task: {},
      by_channel: Object.fromEntries(
        ["ink", "edge", "component", "radial", "direction"].map((channel) => [
          channel,
          { found_markers: 1224 },
        ]),
      ),
    },
    negative_cases: [...CORPUS_NEGATIVE_CASES],
    retrieval_head: {
      schema: "nsrl.solomon_v2_retrieval_head.v1",
      model_hash: "0xsynthetic",
      eval_model_hash: "0xsynthetic",
      feature_count: 4096,
      labels: 72,
      text_head: { present: true, nonzero_weights: 100 },
      image_head: { present: true, nonzero_weights: 100 },
      known_prompts: metric(504),
      identity_bindings: {
        total: metric(504),
        by_kind: {
          "primary-name": metric(72),
          "primary-seal": metric(72),
          alias: metric(72),
          "alias-seal": metric(72),
          "seal-id": metric(216),
        },
      },
      image_to_text: metric(288),
      image_tasks: {
        "text-to-image": metric(576),
        "description-to-image": metric(72),
        "image-to-text": metric(72),
        "image-to-explain": metric(72),
        "text-image-explain": metric(72),
        "image-to-attributes": metric(72),
      },
      match: {
        yes: metric(72),
        no: metric(144),
        no_by_role: {
          image: metric(72),
          prompt: metric(72),
        },
      },
    },
  };
}

function syntheticImageChannelStats() {
  return Object.fromEntries(
    ["ink", "edge", "component", "radial", "direction"].map((channel) => [
      channel,
      {
        records: 72,
        tokens_per_record: 256,
        active_records: 72,
        multi_bin_records: 72,
        nonzero_tokens: 4096,
        distinct_bins: 16,
        max_bin: 15,
        unique_record_hashes: 72,
        duplicate_record_hashes: 0,
      },
    ]),
  );
}

function syntheticIdentityBindingCoverage() {
  return Object.fromEntries(
    ["primary-name", "primary-seal", "alias", "alias-seal", "seal-id"].map((kind) => [
      kind,
      {
        identify_spirits: 72,
        text_to_image_spirits: 72,
      },
    ]),
  );
}

function syntheticSourceProvenance() {
  return Object.fromEntries(
    SOURCE_PROVENANCE_TASKS.map((task) => [
      task,
      {
        rows: 72,
        spirits: 72,
        source_spirit_id_rows: 72,
        source_text_hash_rows: 72,
        source_excerpt_hash_rows: 72,
        source_excerpt_rows: 72,
        expected_source_query_kind: SOURCE_QUERY_KIND_BY_TASK[task],
        source_query_kind_rows: 72,
        source_query_kind_ok_rows: 72,
        source_query_kinds: { [SOURCE_QUERY_KIND_BY_TASK[task]]: 72 },
      },
    ]),
  );
}

function syntheticHeldoutEvidence() {
  return {
    heldout_prompts: {
      rows: 1051,
      unique_targets: 72,
      prompts_hash: "0xsynthetic",
      metric: metric(1051),
    },
  };
}

function syntheticNativeEvidence() {
  return {
    eval_scope: syntheticNativeEvalScope(),
    architecture: {
      d_model: 128,
      heads: 2,
      head_dim: 64,
      hidden_dim: 256,
      transformer_layers: 2,
      context_seq_len: 384,
    },
    output_heads: {
      special: { source: "nsrllmm-output-token-head", targets: 20 },
      text: { source: "nsrllmm-output-token-head", targets: 32 },
      image: { source: "nsrllmm-output-token-head", targets: 16 },
    },
    integer_trace: syntheticNativeIntegerTrace(),
    tasks: syntheticNativeTasks(),
    task_phases: syntheticNativeTaskPhases(),
    directional_groups: {
      text_prompt_to_image_plan: directionalGroup(12, {
        "text-to-image:prompt": 2,
        "text-to-image:image": 2,
        "description-to-image:prompt": 2,
        "description-to-image:image": 2,
      }),
      seal_image_to_text: directionalGroup(20, {
        "image-to-text:image": 2,
        "image-to-text:text": 2,
        "image-to-explain:image": 2,
        "image-to-explain:text": 2,
        "image-to-attributes:image": 2,
        "image-to-attributes:prompt": 2,
        "image-to-attributes:text": 2,
      }),
      text_and_seal_to_explanation: directionalGroup(16, {
        "text-image-explain:prompt": 2,
        "text-image-explain:image": 2,
        "text-image-explain:text": 2,
        "match:prompt": 2,
        "match:image": 2,
        "match:text": 2,
      }),
      identity_source_binding: directionalGroup(20, {
        "canonical-joint:prompt": 2,
        "canonical-joint:text": 2,
        "canonical-joint:image": 2,
        "identify:prompt": 2,
        "identify:text": 2,
        "explain:prompt": 2,
        "explain:text": 2,
      }),
    },
  };
}

function syntheticNativeIntegerTrace() {
  return {
    ok: true,
    train_schema: "nsrl.solomon_attention_train_trace.v1",
    eval_schema: "nsrl.solomon_attention_eval_trace.v1",
    q_formats: {
      logits: "i32_q8",
      probabilities: "i16_q15",
      probability_error: "q15",
      target_margin: "q8",
      train_delta: "i64",
    },
    train_required_fields: {
      target_frequency_min_weight_q15: 2048,
      argmax_margin_weight_q15: 1024,
      initial_probability_error_q15: 65534,
      final_probability_error_q15: 32767,
      probability_error_delta_i64: -32767,
    },
    eval_required_metric_fields: [
      "mean_target_margin_q8",
      "min_target_margin_q8",
      "probability_error_q15",
      "mean_probability_error_q15",
    ],
    eval_metric_objects: 10,
    numeric_leaves: {
      train: 64,
      eval: 128,
    },
    non_integer_numeric_paths: [],
  };
}

function syntheticNativeEvalScope() {
  return {
    proof_scope: "local-directional-smoke",
    eval_max_examples: "none",
    eval_max_targets_per_task_phase: 2,
    smoke_min_task_targets: "all=1",
    smoke_min_phase_targets: "special=1,prompt=1,text=1,image=1",
    smoke_min_direction_top5_per_mille: "all=1",
    product_min_task_targets: "all=72",
    product_min_phase_targets: "all=72",
    product_scale: false,
  };
}

function syntheticNativeTasks() {
  return Object.fromEntries(
    Object.entries(NATIVE_EVAL_TASK_PHASES).map(([task, phases]) => [
      task,
      directionalStats(Object.values(phases).reduce((sum, count) => sum + count, 0)),
    ]),
  );
}

function syntheticNativeTaskPhases() {
  return Object.fromEntries(
    Object.entries(NATIVE_EVAL_TASK_PHASES).map(([task, phases]) => [
      task,
      Object.fromEntries(Object.entries(phases).map(([phase, targets]) => [phase, directionalStats(targets)])),
    ]),
  );
}

function directionalGroup(targets, phaseTargets) {
  return {
    ok: true,
    targets,
    stats: directionalStats(targets),
    min_top5_accuracy_per_mille: 1,
    phase_targets: phaseTargets,
  };
}

function directionalStats(targets) {
  return {
    targets,
    correct: targets,
    invalid_contexts: 0,
    accuracy_per_mille: 1000,
    top5_accuracy_per_mille: 1000,
    top10_accuracy_per_mille: 1000,
    mean_target_rank_per_mille: 1000,
    mean_target_margin_q8: 256,
  };
}

function syntheticAwsProductPlan() {
  return {
    runner: { require_graviton: true },
    s3_required: true,
    generated_prompt_rows: 72,
    generated_unique_targets: 72,
    attention: {
      curriculum_stages: [...CURRICULUM_STAGES],
      curriculum_required_stages: [...CURRICULUM_STAGES],
      native_bind_epochs: 2,
      min_direction_accuracy_per_mille: "",
      min_direction_top5_per_mille: "all=1",
      min_direction_top10_per_mille: "",
      eval_max_examples: "none",
      min_task_targets: "all=72",
      min_task_top5_per_mille: "all=1",
      min_phase_targets: "all=72",
      require_generative_eval: true,
      require_generative_output_identity: true,
      min_generated_prompt_rows: 72,
      min_generated_top5_16_per_mille: 1,
      min_generated_retrieval_top1_per_mille: 1000,
      min_generated_retrieval_top5_per_mille: 1000,
      min_generated_retrieval_margin: 1,
      max_generated_mean_target_distance_16_q8: 7000000,
      denoise_min_unique_targets: 2,
      curriculum_denoise_runner: {
        present: true,
        min_unique_targets_arg: true,
        quality_min_unique_targets_arg: true,
        bridge_pair_count: 2,
        required_bridge_pair_count: 2,
        ok: true,
        errors: [],
      },
      seq_len: 512,
      require_promoted_small_profile: true,
      require_architecture_profile: true,
      min_d_model: 128,
      min_heads: 2,
      target_head_dim: 64,
      min_hidden_dim: 256,
      max_hidden_dim: 512,
      min_transformer_layers: 2,
      min_context_seq_len: 384,
      train_core_architecture: {
        present: true,
        d_model: 128,
        heads: 2,
        head_dim: 64,
        head_dim_power_of_four: true,
        hidden_dim: 256,
        ok: true,
      },
      cpu_scaling: {
        policy: "auto-online-processors",
        auto_workers: true,
      },
    },
  };
}

function syntheticAwsLaunchPlan() {
  const s3PipelineUri = "s3://nsrl-product-plan-check/solomon/pipelines/synthetic-launch";
  return {
    dry_run: true,
    instance_type: "c8g.4xlarge",
    graviton_instance: true,
    s3_pipeline_uri: s3PipelineUri,
    post_run_proof_command: [
      "scripts/aws/prove-solomon-product-run.sh",
      "--s3-pipeline-uri",
      s3PipelineUri,
      "--launch-dir",
      "/synthetic/launch",
      "--require-launch-dir",
    ],
    cpu_scaling: {
      policy: "auto-online-processors",
    },
  };
}

function syntheticAwsPrelaunchReadiness() {
  const s3PipelineUri = "s3://nsrl-product-plan-check/solomon/pipelines/synthetic-prelaunch";
  return {
    launch_ready: true,
    dry_run: true,
    instance_type: "c8g.4xlarge",
    graviton_instance: true,
    s3_pipeline_uri: s3PipelineUri,
    post_run_proof_command: [
      "scripts/aws/prove-solomon-product-run.sh",
      "--s3-pipeline-uri",
      s3PipelineUri,
      "--launch-dir",
      "/synthetic/prelaunch",
      "--require-launch-dir",
    ],
  };
}

function runCoverage(root, name, diagnostic, { requireRelease = false } = {}) {
  const dir = path.join(root, name);
  const diagnosticPath = path.join(dir, "diagnostic.json");
  const outPath = path.join(dir, "objective-coverage.json");
  writeJson(diagnosticPath, diagnostic);
  const args = [
    "scripts/check-solomon-objective-coverage.mjs",
    "--diagnostic",
    diagnosticPath,
    "--out",
    outPath,
  ];
  if (requireRelease) {
    args.push("--require-release");
  }
  const result = childProcess.spawnSync(process.execPath, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  const report = fs.existsSync(outPath) ? JSON.parse(fs.readFileSync(outPath, "utf8")) : null;
  return { result, report };
}

function runCase(root, name, diagnostic, options) {
  const { expectOk, requireRelease = false, expectedMissing = [] } = options;
  const { result, report } = runCoverage(root, name, diagnostic, { requireRelease });
  const missing = Array.isArray(report?.missing) ? report.missing.map(String) : [];
  const statusOk = expectOk ? result.status === 0 : result.status !== 0;
  const reportOk = report?.ok === expectOk;
  const missingOk = expectedMissing.every((needle) => missing.some((item) => item.includes(needle)));
  return {
    name,
    ok: statusOk && reportOk && missingOk,
    expected_ok: expectOk,
    require_release: requireRelease,
    status: result.status,
    schema: report?.schema || "",
    report_ok: report?.ok === true,
    local_objective_proof: report?.local_objective_proof === true,
    release_objective_proof: report?.release_objective_proof === true,
    expected_missing: expectedMissing,
    matched_missing: expectedMissing.filter((needle) => missing.some((item) => item.includes(needle))),
    missing: missing.slice(0, 20),
    stderr_tail: result.stderr ? tailLines(result.stderr, 30) : "",
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
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-objective-coverage-self-test-"));
  const cases = [];
  try {
    cases.push(runCase(root, "good-local", syntheticDiagnostic(), { expectOk: true }));
    cases.push(runCase(root, "good-release", syntheticDiagnostic({ release: true }), {
      expectOk: true,
      requireRelease: true,
    }));
    cases.push(runCase(
      root,
      "bad-stale-quality-cases",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-symbolic-channel-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-symbolic-channel-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-quality-channel-duplicates",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-symbolic-channel-duplicates",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-symbolic-channel-duplicates"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-quality-native-task-confidence",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-native-task-confidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-native-task-confidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-quality-generated-text-agreement",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-sample-generated-text-agreement",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-sample-generated-text-agreement"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-quality-generated-output-margin",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-generative-output-margin",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-generative-output-margin"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-quality-identity-generated-text-margin",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["quality-report-self-test"].cases = QUALITY_CASES.filter(
            (name) => name !== "bad-identity-generated-text-margin",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["quality-report self-test missing case bad-identity-generated-text-margin"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-task-eval-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["task-eval-self-test"].cases = TASK_EVAL_CASES.filter(
            (name) => name !== "bad-eval-provenance",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["task-eval self-test missing case bad-eval-provenance"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-token-layout-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["token-layout-self-test"].cases = TOKEN_LAYOUT_CASES.filter(
            (name) => name !== "bad-rust-task-marker-mismatch",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["token-layout self-test missing case bad-rust-task-marker-mismatch"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-task-eval-channel-duplicates",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["task-eval-self-test"].cases = TASK_EVAL_CASES.filter(
            (name) => name !== "bad-channel-duplicate-records",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["task-eval self-test missing case bad-channel-duplicate-records"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-task-eval-hard-negative-role-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["task-eval-self-test"].cases = TASK_EVAL_CASES.filter(
            (name) => name !== "bad-match-negative-role-coverage",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["task-eval self-test missing case bad-match-negative-role-coverage"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-prior-smoke-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["prior-smoke-self-test"].cases = PRIOR_SMOKE_CASES.filter(
            (name) => name !== "bad-collapsed-interclass",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["prior-smoke self-test missing case bad-collapsed-interclass"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-heldout-retrieval-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["heldout-retrieval-proof-self-test"].cases = HELDOUT_RETRIEVAL_CASES.filter(
            (name) => name !== "bad-prompts-hash",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["held-out retrieval self-test missing case bad-prompts-hash"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-grounded-source-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["grounded-corpus-self-test"].cases = GROUNDED_CORPUS_CASES.filter(
            (name) => name !== "bad-attribute-generic-rank",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["grounded-corpus self-test missing case bad-attribute-generic-rank"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-missing-v2-task-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].task_counts["image-to-attributes"] = 0;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["task image-to-attributes has 0 records"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-corpus-task-marker-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].negative_cases = CORPUS_NEGATIVE_CASES.filter(
            (name) => name !== "corrupt-task-marker",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["v2 corpus contract missing negative case corrupt-task-marker"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-corpus-hard-negative-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].negative_cases = CORPUS_NEGATIVE_CASES.filter(
            (name) => name !== "missing-prompt-hard-negatives",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["v2 corpus contract missing negative case missing-prompt-hard-negatives"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-corpus-train-examples-provenance-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].negative_cases = CORPUS_NEGATIVE_CASES.filter(
            (name) => name !== "bad-curriculum-train-examples-provenance",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["v2 corpus contract missing negative case bad-curriculum-train-examples-provenance"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-channel-stats-summary",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].image_token_channel_stats.edge.distinct_bins = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["image channel edge stats are incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-marker-integrity-summary",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].task_marker_integrity.marker_mismatches = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["task marker integrity is incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-source-provenance-summary",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].source_provenance.explain.source_text_hash_rows = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["source provenance task explain has incomplete source text hashes"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-image-to-text-source-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].source_provenance["image-to-text"].source_text_hash_rows = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["source provenance task image-to-text has incomplete source text hashes"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-source-query-kind-summary",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].source_provenance["image-to-explain"].source_query_kind_ok_rows = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["source provenance task image-to-explain has incomplete source query kind ok rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-source-query-kind-binding",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const row = diagnostic.evidence["v2-corpus-contract"].source_provenance["image-to-explain"];
          row.expected_source_query_kind = "image-identity";
          row.source_query_kinds = { "image-identity": row.rows };
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "source provenance task image-to-explain expected_source_query_kind \"image-identity\" != \"image-source\"",
          "source provenance task image-to-explain missing expected source query kind image-source",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-heldout-row-floor",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["heldout-retrieval-proof"].heldout_prompts.rows = 1;
          diagnostic.evidence["heldout-retrieval-proof"].heldout_prompts.metric = metric(1);
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["held-out prompt rows 1 < 72"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-heldout-metric-count",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["heldout-retrieval-proof"].heldout_prompts.metric = metric(1);
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["held-out retrieval metric has 1 rows, expected 1051"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-generation-integrity-target-guidance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generation-integrity-self-test"].cases = GENERATION_INTEGRITY_CASES.filter(
            (name) => name !== "bad-target-pixel-key",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["generation-integrity self-test missing case bad-target-pixel-key"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-sample-binding-cases",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["sample-binding-self-test"].cases = SAMPLE_BINDING_CASES.filter(
            (name) => name !== "bad-generated-image",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["sample-binding self-test missing case bad-generated-image"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-denoise-bridge-output-margin",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["denoise-bridge-self-test"].cases = DENOISE_BRIDGE_CASES.filter(
            (name) => name !== "bad-output-retrieval-margin",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["denoise bridge self-test missing case bad-output-retrieval-margin"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-generative-posthoc-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].cases = GENERATIVE_CASES.filter(
            (name) => name !== "posthoc-score",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["generative provenance self-test missing case posthoc-score"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generative-clean-rank-summary",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].clean_sample.generated_retrieval_rank = 2;
          diagnostic.evidence["generative-eval-provenance"].clean_sample.generated_retrieval_identity = 0;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["clean generated sample retrieval identity evidence is incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generative-clean-prompt-selection-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].clean_sample.selected_prompt_sources = {};
          diagnostic.evidence["generative-eval-provenance"].clean_sample.selected_prompt_tiers = {};
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["clean generated sample prompt selection provenance is incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generative-posthoc-retrieval-head-binding",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].posthoc_sample.retrieval_head_model_hash = "0xstale";
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "generated clean/post-hoc retrieval head hashes differ: fixture-solomon-retrieval-head != 0xstale",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generative-clean-latent-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].clean_sample.latent_model_hash = "";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["clean generated sample latent model provenance is incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generative-posthoc-latent-provenance",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["generative-eval-provenance"].posthoc_sample.latent_model_provenance_hash = "0xdeadbeef";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["post-hoc generated sample latent model provenance is incomplete"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-promotion-grounding-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["promotion-bundle-self-test"].cases = PROMOTION_CASES.filter(
            (name) => name !== "bad-source-grounding",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["promotion bundle self-test missing case bad-source-grounding"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-promotion-confidence-spine",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["promotion-bundle-self-test"].cases = PROMOTION_CASES.filter(
            (name) => name !== "bad-confidence-spine",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["promotion bundle self-test missing case bad-confidence-spine"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-directional-stats",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["native-directional-eval"].directional_groups.seal_image_to_text.stats;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["missing per-direction top-k stats"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-integer-train-schema",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].integer_trace.train_schema = "nsrl.float_attention_train_trace.v1";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native integer trace train schema"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-integer-eval-q15-field",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].integer_trace.eval_required_metric_fields =
            diagnostic.evidence["native-directional-eval"].integer_trace.eval_required_metric_fields.filter(
              (field) => field !== "probability_error_q15",
            );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native integer trace missing eval metric field probability_error_q15"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-integer-numeric-leaf",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].integer_trace.non_integer_numeric_paths = [
            "train.learning_rate",
          ];
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native integer trace has non-integer numeric leaves"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-small-architecture-head-dim",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].architecture.head_dim = 32;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["head_dim is not 64"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-small-architecture-hidden-dim",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].architecture.hidden_dim = 1024;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["hidden_dim is outside 256-512"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-directional-group-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["native-directional-eval"].directional_groups.text_and_seal_to_explanation;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["missing directional group text_and_seal_to_explanation"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-text-seal-explain-phase-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const group = diagnostic.evidence["native-directional-eval"].directional_groups.text_and_seal_to_explanation;
          group.phase_targets["text-image-explain:image"] = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["directional group text_and_seal_to_explanation phase text-image-explain:image has 1 targets"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-directional-target-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const group = diagnostic.evidence["native-directional-eval"].directional_groups.identity_source_binding;
          group.targets = 4;
          group.stats.targets = 4;
          group.phase_targets["canonical-joint:image"] = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "directional group identity_source_binding has 4 targets",
          "phase canonical-joint:image has 1 targets",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-task-eval-task-metrics",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].tasks["text-to-image"].targets = 0;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native eval task text-to-image has 0 targets"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-task-eval-phase-metrics",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["native-directional-eval"].task_phases["image-to-text"] = {};
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native eval task image-to-text phase targets total 0"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-directional-floor",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["native-directional-eval"].directional_groups.seal_image_to_text.min_top5_accuracy_per_mille;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["missing native top-5 floor"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-eval-scope-accounting",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["native-directional-eval"].eval_scope;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native eval scope is missing or unknown"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-class-retrieval-score-head",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.image_head.present = false;
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.image_head.nonzero_weights = 0;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["class/retrieval image scorer is missing"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-corpus-retrieval-model-hash",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_model_hash = "0xstale";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["corpus retrieval_model_hash 0xstale != retrieval head model_hash 0xsynthetic"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-head-eval-model-hash",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.eval_model_hash = "0xstale";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["retrieval head eval_model_hash 0xstale != model_hash 0xsynthetic"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-identity-kind",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["v2-corpus-contract"].retrieval_head.identity_bindings.by_kind["alias-seal"];
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["identity binding retrieval kind alias-seal has 0 rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-identity-kind-count",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.identity_bindings.by_kind["alias-seal"] = metric(1);
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["identity binding retrieval kind alias-seal has 1 rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-forward-image-plan",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.image_tasks["text-to-image"].top1 = 575;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["image task retrieval text-to-image has 576 rows and 575 top1"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-image-task",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.image_tasks["image-to-attributes"].top1 = 71;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["image task retrieval image-to-attributes has 72 rows and 71 top1"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-image-task-count",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.image_tasks["image-to-attributes"] = metric(1);
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["image task retrieval image-to-attributes has 1 rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-retrieval-hard-negative-count",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["v2-corpus-contract"].retrieval_head.match.no_by_role.prompt = metric(1);
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["wrong-prompt hard negatives have 1 rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-directional-quality-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.min_direction_top5_per_mille = "";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["min_direction_top5_per_mille"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-task-quality-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.min_task_top5_per_mille = "";
          diagnostic.evidence["aws-product-plan"].attention.min_phase_targets = "all=1";
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["min_task_top5_per_mille", "min_phase_targets"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-denoise-handoff-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.denoise_min_unique_targets = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["denoise_min_unique_targets"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-denoise-runner-proof",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const runner = diagnostic.evidence["aws-product-plan"].attention.curriculum_denoise_runner;
          runner.bridge_pair_count = 1;
          runner.ok = false;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["curriculum denoise runner bridge_pair_count"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-aws-train-core-head-dim",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.train_core_architecture.head_dim = 32;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS train-core head_dim is not 64"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-aws-architecture-profile-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const attention = diagnostic.evidence["aws-product-plan"].attention;
          attention.require_promoted_small_profile = false;
          attention.require_architecture_profile = false;
          attention.min_d_model = 64;
          attention.min_heads = 1;
          attention.target_head_dim = 32;
          attention.min_hidden_dim = 128;
          attention.max_hidden_dim = 1024;
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "AWS product plan does not require promoted small profile",
          "AWS product plan does not require architecture profile",
          "AWS product plan min_d_model",
          "AWS product plan min_heads",
          "AWS product plan target_head_dim",
          "AWS product plan min_hidden_dim",
          "AWS product plan max_hidden_dim",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-aws-runtime-architecture-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          const attention = diagnostic.evidence["aws-product-plan"].attention;
          attention.min_transformer_layers = 1;
          attention.min_context_seq_len = 32;
          attention.seq_len = 32;
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "AWS product plan min_transformer_layers",
          "AWS product plan min_context_seq_len",
          "AWS product plan seq_len",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generated-retrieval-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.min_generated_retrieval_top1_per_mille = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS generated retrieval top-1 floor"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generated-retrieval-margin-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.min_generated_retrieval_margin = 0;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS generated retrieval margin floor"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generated-heldout-row-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].generated_prompt_rows = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS generated held-out prompt rows"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generated-heldout-target-coverage",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].generated_unique_targets = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS generated held-out unique targets"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-generated-signature-distance-ratchet",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.max_generated_mean_target_distance_16_q8 = 9000000;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS generated 16x16 target-distance cap"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-curriculum-stage-order",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.curriculum_stages = CURRICULUM_STAGES.filter(
            (stage) => stage !== "description-to-image",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["curriculum_stages"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-native-bind-product-plan",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-product-plan"].attention.native_bind_epochs = 1;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["native_bind_epochs"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-post-run-proof-command",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["aws-launch-plan"].post_run_proof_command;
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["launch plan post-run proof command"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-run-artifact-config-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-run-artifacts-self-test"].cases = AWS_RUN_ARTIFACT_CASES.filter(
            (name) => name !== "bad-artifact-index-missing-curriculum-stages",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS run artifact self-test missing case bad-artifact-index-missing-curriculum-stages"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-completion-artifact-map-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-run-artifacts-self-test"].cases = AWS_RUN_ARTIFACT_CASES.filter(
            (name) => name !== "bad-completion-missing-curriculum-stages-artifact",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS run artifact self-test missing case bad-completion-missing-curriculum-stages-artifact"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-synced-artifact-digest-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-run-artifacts-self-test"].cases = AWS_RUN_ARTIFACT_CASES.filter(
            (name) => name !== "digest-changes-after-artifact-tamper",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS run artifact self-test missing case digest-changes-after-artifact-tamper"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-run-artifact-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-run-artifact-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-run-artifact-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-dry-run-launch-dir",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-dry-run-launch-dir"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-user-data-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-user-data-sha256",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-user-data-sha256"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-user-data-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-user-data",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-user-data"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-instance-type-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-instance-type",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-instance-type"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-image-id-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-image-id",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-image-id"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-tag-specification-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-tag-specification",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-tag-specification"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-security-groups-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-security-groups",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-security-groups"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-subnet-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-subnet",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-subnet"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-key-name-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-key-name",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-key-name"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-region-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-region",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-region"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-command-profile-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-command-profile",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-command-profile"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-result-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-missing-launch-result",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-missing-launch-result"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-result-hash-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-result-sha256",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-result-sha256"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-result-field-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-result-instance-type",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-result-instance-type"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-result-subnet-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-result-subnet",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-result-subnet"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-result-security-groups-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-result-security-groups",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-result-security-groups"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-release-proof-launch-run-identity-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-release-proof-self-test"].cases = AWS_RELEASE_PROOF_CASES.filter(
            (name) => name !== "bad-launch-run-instance-mismatch",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS release proof self-test missing case bad-launch-run-instance-mismatch"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-run-fetch-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-run-fetch-self-test"].cases = AWS_RUN_FETCH_CASES.filter(
            (name) => name !== "bad-native-product-eval-scope",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS run fetch self-test missing case bad-native-product-eval-scope"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-live-launch-readiness-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-live-launch-readiness-self-test"].cases =
            AWS_LIVE_LAUNCH_READINESS_CASES.filter((name) => name !== "bad-missing-explicit-s3-artifact");
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS live launch readiness self-test missing case bad-missing-explicit-s3-artifact"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-live-launch-readiness-ami-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-live-launch-readiness-self-test"].cases =
            AWS_LIVE_LAUNCH_READINESS_CASES.filter((name) => name !== "bad-missing-explicit-ami");
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS live launch readiness self-test missing case bad-missing-explicit-ami"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-launch-execute-explicit-s3-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-launch-execute-guard-self-test"].cases =
            AWS_LAUNCH_EXECUTE_GUARD_CASES.filter(
              (name) => name !== "bad-execute-missing-explicit-s3-blocks-before-aws",
            );
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "AWS launch execute guard self-test missing case bad-execute-missing-explicit-s3-blocks-before-aws",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-launch-execute-guard-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-launch-execute-guard-self-test"].cases = [];
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["AWS launch execute guard self-test missing case bad-execute-prelaunch-blocks-before-aws"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-launch-execute-command-manifest-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-launch-execute-guard-self-test"].cases =
            AWS_LAUNCH_EXECUTE_GUARD_CASES.filter(
              (name) => name !== "good-execute-command-matches-launch-manifest",
            );
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "AWS launch execute guard self-test missing case good-execute-command-matches-launch-manifest",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-aws-launch-execute-command-manifest-profile-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["aws-launch-execute-guard-self-test"].cases =
            AWS_LAUNCH_EXECUTE_GUARD_CASES.filter(
              (name) => name !== "good-execute-command-matches-launch-manifest-with-profile",
            );
        },
      }),
      {
        expectOk: false,
        expectedMissing: [
          "AWS launch execute guard self-test missing case good-execute-command-matches-launch-manifest-with-profile",
        ],
      },
    ));
    cases.push(runCase(
      root,
      "bad-release-candidate-handoff-contract",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-extra-release-gap",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-extra-release-gap"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-hard-negative-role-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-task-eval-hard-negative-role-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-task-eval-hard-negative-role-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-token-layout-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-token-layout-contract-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-token-layout-contract-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-architecture-handoff-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-architecture-handoff-ratchet",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-architecture-handoff-ratchet"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-sample-binding-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-sample-binding-generated-image-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-sample-binding-generated-image-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-denoise-bridge-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-denoise-bridge-output-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-denoise-bridge-output-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-grounded-source-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-grounded-source-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-grounded-source-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-live-readiness-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== RELEASE_CANDIDATE_LIVE_READINESS_CASE,
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: [`release-candidate self-test missing case ${RELEASE_CANDIDATE_LIVE_READINESS_CASE}`],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-live-readiness-ami-case",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          diagnostic.evidence["release-candidate-self-test"].cases = RELEASE_CANDIDATE_CASES.filter(
            (name) => name !== "bad-live-launch-readiness-ami-evidence",
          );
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing case bad-live-launch-readiness-ami-evidence"],
      },
    ));
    cases.push(runCase(
      root,
      "bad-stale-release-candidate-live-readiness-action",
      syntheticDiagnostic({
        mutate: (diagnostic) => {
          delete diagnostic.evidence["release-candidate-self-test"].next_action_cases[
            RELEASE_CANDIDATE_LIVE_READINESS_CASE
          ];
        },
      }),
      {
        expectOk: false,
        expectedMissing: ["release-candidate self-test missing live-readiness next-action proof"],
      },
    ));
    cases.push(runCase(root, "bad-release-required", syntheticDiagnostic(), {
      expectOk: false,
      requireRelease: true,
      expectedMissing: ["no synced real Graviton product run was supplied with --aws-run-dir"],
    }));
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
