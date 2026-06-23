#!/usr/bin/env node
import fs from "node:fs";

const checks = [
  {
    path: "crates/nsrl-train/src/bin/nsrl-solomon-latent-train.rs",
    mustInclude: [
      "const SIGNATURE_GRID: usize = 16;",
      "fn binary_layout_error",
      "fn weighted_binary_layout_error",
      "LAYOUT_DISTINCTIVE_WEIGHT_BASE_Q8",
      "LAYOUT_DISTINCTIVE_WEIGHT_SPAN_Q8",
      "CLASS_CODE_LEARNING_SHIFT",
      "fn update_class_code",
      "LAYOUT_DECISION_HIGH",
      "LAYOUT_DECISION_LOW",
      "weighted_binary_layout_error(",
      "layout_distinctiveness.weight_q8(bin, target)",
      "fn signature_decision_error_count",
      "text_signature_decision_errors",
      "image_signature_decision_errors",
      "update_class_head(config, model, row)",
      "update_hardest_negative_rank",
      "MODEL_CLASS_EXTENSION_MAGIC",
    ],
    mustNotInclude: [
      "const SIGNATURE_GRID: usize = 8;",
      "signature_8x8",
      "text-feature-shape-probe",
      "tf512",
    ],
  },
  {
    path: "crates/nsrl-train/src/bin/nsrl-bitmap-sample.rs",
    mustInclude: [
      "const SIGNATURE_GRID: usize = 16;",
      "--prompt requires --latent-model",
      "--text-index target retrieval has been removed; use --latent-model",
      "--text-all target retrieval has been removed; use --latent-model --prompt",
      "enum LatentTargetMode",
      '"class-layout-code"',
      '"decoded-latent"',
      '"blended-class-text-code"',
      "blend_latents_q8",
      "latent_target_signature",
      "signature: sharpen_signature(",
      "self.decode_signature(class_latent)",
      "LATENT_CLASS_EXTENSION_MAGIC",
    ],
    mustNotInclude: [
      "prior_clean_path",
      "struct CleanPrior",
      "struct TextCondition",
      "struct TextTarget",
      "read_text_condition",
      "read_text_targets",
      "sample_text_conditioned_targets",
      "select_text_targets",
      "write_selected_text_targets",
      "text_match_score",
      "prompt_hint_tokens",
      "SealPrior",
      "LearnedPrior",
      "PatchPrior",
      "CoordinatePrior",
      "seal-prior",
      "learned-prior",
      "patch-prior",
      "coordinate-prior",
      "const SIGNATURE_GRID: usize = 8;",
    ],
  },
  {
    path: "scripts/run-solomon-coherence-panel.sh",
    mustInclude: [
      "NSRL_SOLOMON_LATENT_MODEL is required",
      "expected class-layout-code",
      "crocell|Crocell|Crocell",
      "stolas|Stolas|Stolas",
      "bael|Bael|Bael",
      "hidden-geometry-waters|hidden geometry and rushing waters|hidden geometry and rushing waters",
      "astronomy-herbs-teacher|astronomy and herbs teacher|astronomy and herbs teacher",
    ],
    mustNotInclude: [
      "NSRL_LATENT_TARGET",
      "text-feature-shape-probe",
      "tf512",
      "sample_cmd+=(--text-index",
    ],
  },
  {
    path: "scripts/run-solomon-generative-eval.mjs",
    mustInclude: [
      'latentTarget: "decoded"',
      'throw new Error("--latent-target retrieval was removed; use decoded learned latent targets")',
      'trace.latent_target_source !== "class-layout-code"',
      "expected class-layout-code",
      "return [];",
    ],
    mustNotInclude: [
      'latentTarget: "retrieval"',
      "--latent-target decoded|retrieval",
      "text-feature-shape-probe",
      "tf512",
      'args.push("--text-index", config.textIndex)',
    ],
  },
  {
    path: "scripts/aws/run-solomon-prior-smoke.sh",
    mustInclude: [
      "NSRL_SOLOMON_MAX_INTRA_PROMPT_DISTANCE",
      "NSRL_SOLOMON_MAX_TARGET_DISTANCE",
      "NSRL_SOLOMON_MIN_INTER_CLASS_DISTANCE",
      "NSRL_SOLOMON_MIN_TARGET_INK_CELLS",
      "prompt_slug\\tseed_variant\\tprompt",
      "check-solomon-prior-smoke.mjs",
      "--max-intra-prompt-distance",
      "--max-target-distance",
      "--min-inter-class-distance",
      "--min-target-ink-cells",
      "--min-eval-class-top1",
      "expected_target_source",
      "sampler used",
      "check-solomon-denoiser-model.mjs",
    ],
    mustNotInclude: [
      "NSRL_LATENT_TARGET",
      "text-feature-shape-probe",
      "tf512",
      "sample_cmd+=(--text-index",
    ],
  },
  {
    path: "scripts/check-solomon-prior-smoke.mjs",
    mustInclude: [
      "maxIntraPromptDistance",
      "maxTargetDistance",
      "minInterClassDistance",
      "minTargetInkCells",
      "intra_prompt_distances",
      "inter_class_distances",
      "target_distances",
      "class-layout-code",
      "expectedTargetSource",
      "model_format",
      "feature_channels",
      "Crocell",
      "Stolas",
      "Bael",
      "hidden geometry and rushing waters",
      "astronomy and herbs teacher",
    ],
    mustNotInclude: [
      'latentTarget: "retrieval"',
      "--latent-target decoded|retrieval",
    ],
  },
  {
    path: "scripts/check-solomon-denoiser-model.mjs",
    mustInclude: [
      "nsrl.solomon_denoiser_model_check.v1",
      "NSRLTCH",
      "minTextFeatureChannels",
      "feature_count",
      "text/layout channels are unreachable",
    ],
    mustNotInclude: [],
  },
];

function main() {
  const failures = [];
  for (const check of checks) {
    const text = fs.readFileSync(check.path, "utf8");
    for (const needle of check.mustInclude) {
      if (!text.includes(needle)) {
        failures.push(`${check.path}: missing required marker: ${needle}`);
      }
    }
    for (const needle of check.mustNotInclude) {
      if (text.includes(needle)) {
        failures.push(`${check.path}: banned old-path marker still present: ${needle}`);
      }
    }
  }

  const report = {
    schema: "nsrl.solomon_generation_honesty_check.v1",
    passed: failures.length === 0,
    checked_files: checks.map((check) => check.path),
    failures,
  };
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) {
    process.exit(1);
  }
}

main();
