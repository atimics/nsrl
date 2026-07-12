#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_release_candidate_self_test.v1";
const expectedReleaseGap = "no synced real Graviton product run was supplied with --aws-run-dir";
const qualityCases = [
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
const taskEvalCases = [
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
const tokenLayoutCases = [
  "good-js-corpus-builder-layout",
  "good-rust-native-attention-layout",
  "good-js-fallback-layouts",
  "good-js-retrieval-consumer-layouts",
  "good-shared-symbolic-image-defaults",
  "bad-js-layout-mismatch",
  "bad-rust-task-marker-mismatch",
  "bad-shared-marker-order",
];
const canonicalTokenLayout = {
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
};
const priorSmokeCases = [
  "good",
  "bad-target-source",
  "bad-missing-seed-variant",
  "bad-collapsed-interclass",
  "bad-eval-class-top1",
];
const heldoutRetrievalCases = [
  "good",
  "bad-prompts-hash",
  "bad-heldout-row-count",
  "bad-heldout-top1",
  "bad-heldout-margin",
  "bad-missing-image-head",
  "bad-stale-model-hash",
];
const groundedCorpusCases = [
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
const nativeEvalTaskPhases = {
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
const corpusNegativeCases = [
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
const sourceProvenanceTasks = [
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
];
const sourceQueryKindByTask = {
  "text-to-image": "identity-to-image",
  "image-to-text": "image-identity",
  "image-to-explain": "image-source",
  "text-image-explain": "text-image-source",
  "image-to-attributes": "image-attributes",
  explain: "primary-name",
  "description-to-image": "source-description",
};
const generativeCases = [
  "good",
  "posthoc-score",
  "posthoc-bad-raw-path",
  "bad-raw-path",
  "bad-cleanup",
  "bad-missing-trace",
  "bad-empty-raw",
];
const generationIntegrityCases = [
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
const sampleBindingCases = [
  "good",
  "bad-generated-text",
  "bad-generated-image",
  "bad-cleanup",
];
const denoiseBridgeCases = [
  "good",
  "bad-cleanup",
  "bad-source",
  "bad-signature",
  "bad-flat-output",
  "bad-retrieval-head-hash",
  "bad-output-retrieval-margin",
  "bad-unique-targets",
];
const promotionCases = [
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
const liveReadinessNextActionCase = "good-local-live-readiness-next-action";
const releaseCandidateCases = [
  "good-local",
  liveReadinessNextActionCase,
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
const liveReadinessNextActionIncludes = [
  "scripts/check-solomon-aws-live-launch-readiness.sh",
  "scripts/aws/launch-solomon-product-run.sh --execute",
  "scripts/aws/prove-solomon-product-run.sh",
  "--s3-pipeline-uri",
  "--launch-dir",
  "--require-launch-dir",
  "launch-result.json",
];
const awsReleaseProofCases = [
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
const awsRunArtifactCases = [
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
const awsRunFetchCases = [
  "good",
  "good-run-name",
  "bad-mismatched-s3-pipeline",
  "bad-missing-status",
  "bad-stale-promotion",
  "bad-native-product-eval-scope",
];
const awsLiveLaunchReadinessCases = [
  "good-explicit-s3-artifact",
  "bad-missing-explicit-s3-artifact",
  "bad-missing-explicit-ami",
];
const awsLaunchExecuteGuardCases = [
  "bad-execute-missing-explicit-s3-blocks-before-aws",
  "bad-execute-missing-explicit-artifact-blocks-before-aws",
  "bad-execute-prelaunch-blocks-before-aws",
  "good-execute-records-launch-result",
  "good-execute-command-matches-launch-manifest",
  "good-execute-command-matches-launch-manifest-with-profile",
];
const curriculumStages = [
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
    "Usage: check-solomon-release-candidate-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic product diagnostics and checks that the release-candidate",
    "handoff accepts a complete no-spend proof while rejecting skipped checks,",
    "broken objective evidence, broken AWS prelaunch/release-proof evidence,",
    "and hidden extra release gaps.",
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

function readJsonMaybe(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
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

function directionalGroup(targets, phaseTargets) {
  return {
    ok: true,
    targets,
    stats: directionalStats(targets),
    min_top5_accuracy_per_mille: 1,
    phase_targets: phaseTargets,
  };
}

function syntheticNativeTasks() {
  return Object.fromEntries(
    Object.entries(nativeEvalTaskPhases).map(([task, phases]) => [
      task,
      directionalStats(Object.values(phases).reduce((sum, count) => sum + count, 0)),
    ]),
  );
}

function syntheticNativeTaskPhases() {
  return Object.fromEntries(
    Object.entries(nativeEvalTaskPhases).map(([task, phases]) => [
      task,
      Object.fromEntries(Object.entries(phases).map(([phase, targets]) => [phase, directionalStats(targets)])),
    ]),
  );
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
    negative_cases: [...corpusNegativeCases],
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
    sourceProvenanceTasks.map((task) => [
      task,
      {
        rows: 72,
        spirits: 72,
        source_spirit_id_rows: 72,
        source_text_hash_rows: 72,
        source_excerpt_hash_rows: 72,
        source_excerpt_rows: 72,
        expected_source_query_kind: sourceQueryKindByTask[task],
        source_query_kind_rows: 72,
        source_query_kind_ok_rows: 72,
        source_query_kinds: { [sourceQueryKindByTask[task]]: 72 },
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

function syntheticAwsProductPlan() {
  return {
    stages: ["dataset", "denoiser", "prior", "generative-eval", "attention-curriculum", "promotion-bundle-check"],
    required_plan_stages: ["dataset", "denoiser", "prior", "generative-eval", "attention-curriculum"],
    promotion_bundle_check: true,
    runner: { require_graviton: true },
    s3_required: true,
    generated_prompt_rows: 72,
    generated_unique_targets: 72,
    attention: {
      curriculum_stages: [...curriculumStages],
      curriculum_required_stages: [...curriculumStages],
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
        processor_count: 16,
        effective_map_reduce_workers: 16,
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
    ec2_metadata_required: true,
    product_stages: ["dataset", "denoiser", "prior", "generative-eval", "attention-curriculum"],
    cpu_scaling: {
      batch_mode: "map-reduce",
      map_reduce_workers: "0",
      policy: "auto-online-processors",
    },
    s3_uri: "s3://nsrl-product-plan-check/solomon",
    s3_pipeline_uri: s3PipelineUri,
    artifact_s3_uri: "s3://nsrl-product-plan-check/solomon/artifacts/product.tar.zst",
    post_run_proof_command: [
      "scripts/aws/prove-solomon-product-run.sh",
      "--s3-pipeline-uri",
      s3PipelineUri,
      "--launch-dir",
      "/synthetic/launch",
      "--require-launch-dir",
    ],
  };
}

function syntheticAwsPrelaunchReadiness() {
  const s3PipelineUri = "s3://nsrl-product-plan-check/solomon/pipelines/synthetic-prelaunch";
  return {
    launch_ready: true,
    dry_run: true,
    instance_type: "c8g.4xlarge",
    graviton_instance: true,
    ec2_metadata_required: true,
    product_stages: ["dataset", "denoiser", "prior", "generative-eval", "attention-curriculum"],
    cpu_scaling: {
      batch_mode: "map-reduce",
      map_reduce_workers: "0",
      policy: "auto-online-processors",
    },
    s3_uri: "s3://nsrl-product-plan-check/solomon",
    s3_pipeline_uri: s3PipelineUri,
    artifact_s3_uri: "s3://nsrl-product-plan-check/solomon/artifacts/product.tar.zst",
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
    remaining_product_evidence: release ? [] : [expectedReleaseGap],
    skipped: [],
    checks: [
      { name: "aws-product-plan", ok: true, status: 0 },
      { name: "aws-launch-plan", ok: true, status: 0 },
      { name: "aws-prelaunch-readiness", ok: true, status: 0 },
    ],
    evidence: {
      "v2-corpus-contract": syntheticCorpusEvidence(),
      "heldout-retrieval-proof": syntheticHeldoutEvidence(),
      "heldout-retrieval-proof-self-test": { cases: [...heldoutRetrievalCases] },
      "grounded-corpus-self-test": { cases: [...groundedCorpusCases] },
      "native-directional-eval": syntheticNativeEvidence(),
      "task-eval-self-test": { cases: [...taskEvalCases] },
      "token-layout-self-test": {
        cases: [...tokenLayoutCases],
        canonical_layout: { ...canonicalTokenLayout },
      },
      "prior-smoke-self-test": { cases: [...priorSmokeCases] },
      "quality-report-self-test": { cases: [...qualityCases] },
      "generative-eval-provenance": {
        cases: [...generativeCases],
        clean_sample: syntheticGeneratedSample(),
        posthoc_sample: syntheticGeneratedSample(),
      },
      "generation-integrity-self-test": { cases: [...generationIntegrityCases] },
      "sample-binding-self-test": { cases: [...sampleBindingCases] },
      "denoise-bridge-self-test": { cases: [...denoiseBridgeCases] },
      "promotion-bundle-self-test": { cases: [...promotionCases] },
      "release-candidate-self-test": {
        cases: [...releaseCandidateCases],
        next_action_cases: {
          [liveReadinessNextActionCase]: {
            expected_next_action_includes: [...liveReadinessNextActionIncludes],
            matched_next_action_includes: [...liveReadinessNextActionIncludes],
          },
        },
      },
      "aws-release-proof-self-test": { cases: [...awsReleaseProofCases] },
      "aws-run-artifacts-self-test": { cases: [...awsRunArtifactCases] },
      "aws-run-fetch-self-test": { cases: [...awsRunFetchCases] },
      "aws-live-launch-readiness-self-test": { cases: [...awsLiveLaunchReadinessCases] },
      "aws-launch-execute-guard-self-test": { cases: [...awsLaunchExecuteGuardCases] },
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

function runCandidate(root, name, diagnostic, { requireRelease = false } = {}) {
  const dir = path.join(root, name);
  const diagnosticPath = path.join(dir, "diagnostic.json");
  const outPath = path.join(dir, "release-candidate.json");
  writeJson(diagnosticPath, diagnostic);
  const args = [
    "scripts/check-solomon-release-candidate.mjs",
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
  return {
    result,
    report: readJsonMaybe(outPath),
  };
}

function runCase(root, name, diagnostic, options) {
  const { expectOk, requireRelease = false, expectedError = [], expectedNextActionIncludes = [] } = options;
  const { result, report } = runCandidate(root, name, diagnostic, { requireRelease });
  const errors = Array.isArray(report?.errors) ? report.errors.map(String) : [];
  const nextOperatorAction = String(report?.next_operator_action || "");
  const statusOk = expectOk ? result.status === 0 : result.status !== 0;
  const reportOk = report?.ok === expectOk;
  const errorOk = expectedError.every((needle) => errors.some((item) => item.includes(needle)));
  const nextActionOk = expectedNextActionIncludes.every((needle) => nextOperatorAction.includes(needle));
  return {
    name,
    ok: statusOk && reportOk && errorOk && nextActionOk,
    expected_ok: expectOk,
    require_release: requireRelease,
    status: result.status,
    schema: report?.schema || "",
    report_ok: report?.ok === true,
    candidate_state: report?.candidate_state || "",
    local_product_proof: report?.diagnostic?.local_product_proof === true,
    local_objective_proof: report?.objective_coverage?.local_objective_proof === true,
    release_product_proof: report?.diagnostic?.release_product_proof === true,
    release_objective_proof: report?.objective_coverage?.release_objective_proof === true,
    expected_error: expectedError,
    matched_error: expectedError.filter((needle) => errors.some((item) => item.includes(needle))),
    expected_next_action_includes: expectedNextActionIncludes,
    matched_next_action_includes: expectedNextActionIncludes.filter((needle) => nextOperatorAction.includes(needle)),
    next_operator_action: nextOperatorAction,
    errors: errors.slice(0, 20),
    stderr_tail: tailLines(result.stderr, 30),
  };
}

function tailLines(text, maxLines) {
  const lines = String(text || "").split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
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
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-release-candidate-self-test-"));
  const cases = [];
  try {
    cases.push(runCase(root, "good-local", syntheticDiagnostic(), {
      expectOk: true,
      expectedNextActionIncludes: liveReadinessNextActionIncludes,
    }));
    cases.push(runCase(root, liveReadinessNextActionCase, syntheticDiagnostic(), {
      expectOk: true,
      expectedNextActionIncludes: liveReadinessNextActionIncludes,
    }));
    cases.push(runCase(root, "good-release", syntheticDiagnostic({ release: true }), {
      expectOk: true,
      requireRelease: true,
      expectedNextActionIncludes: ["archive release-proof.json"],
    }));
    cases.push(runCase(root, "bad-skipped-local-proof", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.skipped = ["native-directional-eval"];
        diagnostic.full_product_proof = false;
        diagnostic.local_product_proof = false;
      },
    }), {
      expectOk: false,
      expectedError: ["diagnostic has skipped checks"],
    }));
    cases.push(runCase(root, "bad-objective-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.min_direction_top5_per_mille = "";
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "per-direction top-5 floor"],
    }));
    cases.push(runCase(root, "bad-native-task-handoff-ratchet", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.min_task_top5_per_mille = "";
        diagnostic.evidence["aws-product-plan"].attention.min_phase_targets = "all=1";
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "per-task top-5 floor", "per-phase target floor"],
    }));
    cases.push(runCase(root, "bad-native-bind-handoff-ratchet", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.native_bind_epochs = 1;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "native-bind epoch floor"],
    }));
    cases.push(runCase(root, "bad-denoise-handoff-ratchet", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.denoise_min_unique_targets = 1;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "denoise bridge unique-target floor"],
    }));
    cases.push(runCase(root, "bad-denoise-runner-proof", syntheticDiagnostic({
      mutate: (diagnostic) => {
        const runner = diagnostic.evidence["aws-product-plan"].attention.curriculum_denoise_runner;
        runner.bridge_pair_count = 1;
        runner.ok = false;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "denoise runner bridge-pair count"],
    }));
    cases.push(runCase(root, "bad-architecture-handoff-ratchet", syntheticDiagnostic({
      mutate: (diagnostic) => {
        const attention = diagnostic.evidence["aws-product-plan"].attention;
        attention.require_promoted_small_profile = false;
        attention.require_architecture_profile = false;
        attention.min_d_model = 64;
        attention.min_heads = 1;
        attention.target_head_dim = 32;
        attention.min_hidden_dim = 128;
        attention.max_hidden_dim = 1024;
        attention.min_transformer_layers = 1;
        attention.min_context_seq_len = 32;
        attention.seq_len = 32;
      },
    }), {
      expectOk: false,
      expectedError: [
        "local_objective_proof is not true",
        "promoted small profile",
        "architecture profile",
        "min_d_model",
        "min_transformer_layers",
        "seq_len",
      ],
    }));
    cases.push(runCase(root, "bad-generated-handoff-floor", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.min_generated_retrieval_top1_per_mille = 1;
        diagnostic.evidence["aws-product-plan"].attention.min_generated_retrieval_margin = 0;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "generated retrieval top-1 floor", "generated retrieval margin floor"],
    }));
    cases.push(runCase(root, "bad-generated-handoff-heldout-coverage", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].generated_prompt_rows = 1;
        diagnostic.evidence["aws-product-plan"].generated_unique_targets = 1;
      },
    }), {
      expectOk: false,
      expectedError: [
        "local_objective_proof is not true",
        "generated held-out prompt rows",
        "generated held-out unique targets",
      ],
    }));
    cases.push(runCase(root, "bad-generated-handoff-distance-cap", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-product-plan"].attention.max_generated_mean_target_distance_16_q8 = 9000000;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "generated 16x16 target-distance cap"],
    }));
    cases.push(runCase(root, "bad-prelaunch-readiness", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-prelaunch-readiness"].launch_ready = false;
        diagnostic.checks.find((item) => item.name === "aws-prelaunch-readiness").ok = false;
      },
    }), {
      expectOk: false,
      expectedError: ["aws-prelaunch-readiness check is not ok", "prelaunch readiness is not green"],
    }));
    cases.push(runCase(root, "bad-post-run-proof-command", syntheticDiagnostic({
      mutate: (diagnostic) => {
        delete diagnostic.evidence["aws-launch-plan"].post_run_proof_command;
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "launch plan post-run proof command"],
    }));
    cases.push(runCase(root, "bad-quality-generated-text-agreement-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["quality-report-self-test"].cases = qualityCases.filter(
          (name) => name !== "bad-sample-generated-text-agreement",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "quality-report self-test"],
    }));
    cases.push(runCase(root, "bad-sample-binding-generated-image-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["sample-binding-self-test"].cases = sampleBindingCases.filter(
          (name) => name !== "bad-generated-image",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "sample-binding self-test"],
    }));
    cases.push(runCase(root, "bad-denoise-bridge-output-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["denoise-bridge-self-test"].cases = denoiseBridgeCases.filter(
          (name) => name !== "bad-output-retrieval-margin",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "denoise bridge self-test"],
    }));
    cases.push(runCase(root, "bad-grounded-source-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["grounded-corpus-self-test"].cases = groundedCorpusCases.filter(
          (name) => name !== "bad-attribute-generic-rank",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "grounded-corpus self-test"],
    }));
    cases.push(runCase(root, "bad-task-eval-channel-duplicate-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["task-eval-self-test"].cases = taskEvalCases.filter(
          (name) => name !== "bad-channel-duplicate-records",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "task-eval self-test"],
    }));
    cases.push(runCase(root, "bad-task-eval-hard-negative-role-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["task-eval-self-test"].cases = taskEvalCases.filter(
          (name) => name !== "bad-match-negative-role-coverage",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "task-eval self-test"],
    }));
    cases.push(runCase(root, "bad-token-layout-contract-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["token-layout-self-test"].cases = tokenLayoutCases.filter(
          (name) => name !== "bad-rust-task-marker-mismatch",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "token-layout self-test"],
    }));
    cases.push(runCase(root, "bad-generation-integrity-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["generation-integrity-self-test"].cases = generationIntegrityCases.filter(
          (name) => name !== "bad-display-cleanup",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "generation-integrity self-test"],
    }));
    cases.push(runCase(root, "bad-execute-guard-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-launch-execute-guard-self-test"].cases = [];
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "launch execute guard"],
    }));
    cases.push(runCase(root, "bad-execute-guard-explicit-s3-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-launch-execute-guard-self-test"].cases = awsLaunchExecuteGuardCases.filter(
          (name) => name !== "bad-execute-missing-explicit-s3-blocks-before-aws",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "launch execute guard"],
    }));
    cases.push(runCase(root, "bad-live-launch-readiness-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-live-launch-readiness-self-test"].cases = awsLiveLaunchReadinessCases.filter(
          (name) => name !== "bad-missing-explicit-s3-artifact",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "live launch readiness self-test"],
    }));
    cases.push(runCase(root, "bad-live-launch-readiness-ami-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-live-launch-readiness-self-test"].cases = awsLiveLaunchReadinessCases.filter(
          (name) => name !== "bad-missing-explicit-ami",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "live launch readiness self-test"],
    }));
    cases.push(runCase(root, "bad-release-proof-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-release-proof-self-test"].cases = awsReleaseProofCases.filter(
          (name) => name !== "bad-dry-run-launch-dir",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "release proof self-test"],
    }));
    cases.push(runCase(root, "bad-run-artifact-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-run-artifacts-self-test"].cases = awsRunArtifactCases.filter(
          (name) => name !== "bad-artifact-index-missing-curriculum-stages",
        );
      },
    }), {
      expectOk: false,
      expectedError: [
        "local_objective_proof is not true",
        "run artifact self-test",
        "bad-artifact-index-missing-curriculum-stages",
      ],
    }));
    cases.push(runCase(root, "bad-run-fetch-evidence", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.evidence["aws-run-fetch-self-test"].cases = awsRunFetchCases.filter(
          (name) => name !== "bad-native-product-eval-scope",
        );
      },
    }), {
      expectOk: false,
      expectedError: ["local_objective_proof is not true", "run fetch self-test"],
    }));
    cases.push(runCase(root, "bad-extra-release-gap", syntheticDiagnostic({
      mutate: (diagnostic) => {
        diagnostic.remaining_product_evidence.push("manual audit evidence is missing");
      },
    }), {
      expectOk: false,
      expectedError: ["diagnostic release gaps are not limited"],
    }));
    cases.push(runCase(root, "bad-release-required", syntheticDiagnostic(), {
      expectOk: false,
      requireRelease: true,
      expectedError: ["release_product_proof is not true", "release_objective_proof is not true"],
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
