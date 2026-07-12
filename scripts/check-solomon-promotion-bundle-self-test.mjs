#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const READY_FLAGS = [
  "binding_spine_ready",
  "identity_inference_ready",
  "curriculum_ready",
  "denoise_bridge_ready",
  "grounded_corpus_ready",
  "product_generation_ready",
  "confidence_trace_ready",
  "corpus_contract_ready",
  "task_eval_ready",
  "architecture_profile_ready",
  "promoted_small_profile_ready",
];

const ATTENTION_ARTIFACTS = [
  ["quality_report", "quality-report.json"],
  ["model", "model.nsrllmm"],
  ["corpus_manifest", "manifest.json"],
  ["attention_eval", "attention-eval.json"],
  ["retrieval_head", "retrieval-head.json"],
  ["retrieval_head_eval", "retrieval-head-eval.json"],
  ["curriculum_stages", "curriculum-stages.json"],
  ["sample_binding", "prior-sample-binding.json"],
  ["identity_inference", "identity-inference.json"],
  ["grounded_corpus", "grounded-corpus.json"],
  ["generation_integrity", "generation-integrity.json"],
  ["denoise_bridge", "denoise-bridge.json"],
  ["denoise_generation_integrity", "denoise-generation-integrity.json"],
];
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const REQUIRED_IMAGE_BINDING_TASK_COUNTS = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};
const REQUIRED_IMAGE_BINDING_TASKS = Object.keys(REQUIRED_IMAGE_BINDING_TASK_COUNTS);
const REQUIRED_CORPUS_TASKS = [
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
const REQUIRED_EVAL_PHASES = ["special", "prompt", "text", "image"];
const REQUIRED_IMAGE_TOKEN_CHANNELS = [
  "ink",
  "edge",
  "component",
  "radial",
  "direction",
];

function main() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-promotion-bundle-self-test-"));
  const good = buildFixture(path.join(root, "good"));
  runChecker("good", good.promotion, true);

  const badGeneration = buildFixture(path.join(root, "bad-generation"), (report) => {
    report.product_generation_ready = false;
    report.confidence_trace.product_generation.matching_model_output_identity.ok = false;
  });
  runChecker("bad generation", badGeneration.promotion, false, "product_generation_ready");

  const badIntegrity = buildFixture(path.join(root, "bad-integrity"), (report) => {
    report.generation_integrity.violations = 1;
    report.confidence_trace.product_generation.trace_integrity_ok = false;
  });
  runChecker("bad integrity", badIntegrity.promotion, false, "generation integrity violations");

  const badSourceProvenance = buildFixture(path.join(root, "bad-source-provenance"), (report) => {
    report.confidence_trace.source_grounding.grounded_source_provenance = false;
  });
  runChecker("bad source provenance", badSourceProvenance.promotion, false, "source grounding provenance");

  const badSourceGrounding = buildFixture(path.join(root, "bad-source-grounding"), (report) => {
    report.confidence_trace.source_grounding.image_queries_have_source_text = false;
  });
  runChecker("bad source grounding", badSourceGrounding.promotion, false, "image queries source text");

  const badGroundedCorpusOverlap = buildFixture(path.join(root, "bad-grounded-corpus-overlap"), (report) => {
    report.grounded_corpus.tasks.explain.min_source_overlap_tokens = 1;
  });
  runChecker(
    "bad grounded corpus overlap",
    badGroundedCorpusOverlap.promotion,
    false,
    "grounded corpus task explain source overlap 1 < 2",
  );

  const badGroundedAttributeRank = buildFixture(path.join(root, "bad-grounded-attribute-rank"), (report) => {
    report.grounded_corpus.tasks["image-to-attributes"].generic_attribute_rank_rows = 1;
  });
  runChecker(
    "bad grounded attribute rank",
    badGroundedAttributeRank.promotion,
    false,
    "grounded corpus task image-to-attributes generic rank rows 1 != 0",
  );

  const badGroundedNameSource = buildFixture(path.join(root, "bad-grounded-name-source"), (report) => {
    report.grounded_corpus.tasks.explain.name_source_prompt_ok_rows = 71;
    report.confidence_trace.source_grounding.grounded_name_source_explain = false;
  });
  runChecker(
    "bad grounded name source",
    badGroundedNameSource.promotion,
    false,
    "grounded corpus task explain name-source prompt rows 71 != records 72",
  );

  const badGroundedDescriptionSource = buildFixture(path.join(root, "bad-grounded-description-source"), (report) => {
    report.grounded_corpus.tasks["description-to-image"].description_source_prompt_ok_rows = 71;
    report.confidence_trace.source_grounding.grounded_description_source_image = false;
  });
  runChecker(
    "bad grounded description source",
    badGroundedDescriptionSource.promotion,
    false,
    "grounded corpus task description-to-image description-source prompt rows 71 != records 72",
  );

  const badGroundedAttributePrompt = buildFixture(path.join(root, "bad-grounded-attribute-prompt"), (report) => {
    report.grounded_corpus.tasks["image-to-attributes"].image_attribute_prompt_ok_rows = 71;
    report.confidence_trace.source_grounding.grounded_image_attribute_generic_prompt = false;
  });
  runChecker(
    "bad grounded attribute prompt",
    badGroundedAttributePrompt.promotion,
    false,
    "grounded corpus task image-to-attributes generic attribute prompt rows 71 != records 72",
  );

  const badPromptProvenance = buildFixture(path.join(root, "bad-prompt-provenance"), (report) => {
    report.generative_eval.evidence.prompt_provenance.selected_prompt_eligible_unique_targets = 12;
    report.confidence_trace.product_generation.prompt_provenance.selected_prompt_eligible_unique_targets = 12;
  });
  runChecker("bad prompt provenance", badPromptProvenance.promotion, false, "selected held-out unique targets");

  const badGeneratedSignature = buildFixture(path.join(root, "bad-generated-signature"), (report) => {
    report.generative_eval.models[0].generated_signature_16.top5_per_mille = 0;
    report.generative_eval.best.generated_signature_16.top5_per_mille = 0;
    report.confidence_trace.product_generation.best_top5_16_per_mille = 0;
  });
  runChecker("bad generated signature", badGeneratedSignature.promotion, false, "matching generated 16x16 top5 0 < 1");

  const badGeneratedDistance = buildFixture(path.join(root, "bad-generated-distance"), (report) => {
    report.generative_eval.models[0].generated_signature_16.mean_target_distance_q8 = 7000001;
    report.generative_eval.best.generated_signature_16.mean_target_distance_q8 = 7000001;
  });
  runChecker(
    "bad generated distance",
    badGeneratedDistance.promotion,
    false,
    "matching generated 16x16 target distance 7000001 > 7000000",
  );

  const badConfidenceSpine = buildFixture(path.join(root, "bad-confidence-spine"), (report) => {
    report.confidence_trace.text_binding.heldout_prompts.top1 = 12;
  });
  runChecker("bad confidence spine", badConfidenceSpine.promotion, false, "held-out prompt text retrieval top1");

  const badHeldoutRetrieval = buildFixture(path.join(root, "bad-heldout-retrieval"), (report) => {
    report.retrieval_head_eval.heldout_prompts.top1 = 71;
  });
  runChecker("bad heldout retrieval", badHeldoutRetrieval.promotion, false, "retrieval held-out prompts top1 71 != count 72");

  const badHardNegativeRetrieval = buildFixture(path.join(root, "bad-hard-negative-retrieval"), (report) => {
    report.retrieval_head_eval.match.no_by_role.prompt.min_margin = 0;
  });
  runChecker(
    "bad hard negative retrieval",
    badHardNegativeRetrieval.promotion,
    false,
    "retrieval wrong-prompt hard negatives min_margin 0 <= 0",
  );

  const badDirectionalGroup = buildFixture(path.join(root, "bad-directional-group"), (report) => {
    delete report.confidence_trace.directional_native_eval.groups.text_and_seal_to_explanation;
  });
  runChecker(
    "bad directional group",
    badDirectionalGroup.promotion,
    false,
    "directional native eval missing group text_and_seal_to_explanation",
  );

  const badNativeTaskConfidence = buildFixture(path.join(root, "bad-native-task-confidence"), (report) => {
    report.confidence_trace.native_task_eval.tasks["image-to-text"].top5_accuracy_per_mille = 0;
  });
  runChecker(
    "bad native task confidence",
    badNativeTaskConfidence.promotion,
    false,
    "image-to-text native task eval top5 0 < 1",
  );

  const badHeadDim = buildFixture(path.join(root, "bad-head-dim"), (report) => {
    report.attention_eval.architecture.head_dim = 32;
    report.attention_eval.architecture.promoted_small_profile.ok = true;
  });
  runChecker("bad head dim", badHeadDim.promotion, false, "quality report head_dim 32 != 64");

  const badHiddenDim = buildFixture(path.join(root, "bad-hidden-dim"), (report) => {
    report.attention_eval.architecture.hidden_dim = 1024;
    report.attention_eval.architecture.promoted_small_profile.ok = true;
  });
  runChecker("bad hidden dim", badHiddenDim.promotion, false, "quality report hidden_dim 1024 outside 256-512");

  const badRetrievalTextHead = buildFixture(path.join(root, "bad-retrieval-text-head"), (report) => {
    report.retrieval_head_eval.class_retrieval_head.text_head = false;
  });
  runChecker(
    "bad retrieval text head",
    badRetrievalTextHead.promotion,
    false,
    "quality report class retrieval text_head is not valid",
  );

  const badRetrievalHeadHash = buildFixture(path.join(root, "bad-retrieval-head-hash"), (report) => {
    report.retrieval_head_eval.class_retrieval_head.hash_verified = false;
  });
  runChecker(
    "bad retrieval head hash",
    badRetrievalHeadHash.promotion,
    false,
    "quality report class retrieval head hash is not verified",
  );

  const badSymbolicTokens = buildFixture(path.join(root, "bad-symbolic-tokens"), (report) => {
    report.confidence_trace.symbolic_image_tokens.corpus.by_channel.edge.found_markers = 0;
  });
  runChecker("bad symbolic tokens", badSymbolicTokens.promotion, false, "symbolic image-token corpus channel edge");

  const badCorpusChannelStats = buildFixture(path.join(root, "bad-corpus-channel-stats"), (report) => {
    report.corpus_contract.manifest_summary.image_token_channel_stats.edge.distinct_bins = 1;
  });
  runChecker(
    "bad corpus channel stats",
    badCorpusChannelStats.promotion,
    false,
    "corpus manifest image_token_channel_stats edge distinct_bins 1 < 2",
  );

  const badCorpusChannelDuplicates = buildFixture(path.join(root, "bad-corpus-channel-duplicates"), (report) => {
    report.corpus_contract.manifest_summary.image_token_channel_stats.edge.unique_record_hashes = 71;
    report.corpus_contract.manifest_summary.image_token_channel_stats.edge.duplicate_record_hashes = 1;
  });
  runChecker(
    "bad corpus channel duplicates",
    badCorpusChannelDuplicates.promotion,
    false,
    "corpus manifest image_token_channel_stats edge unique_record_hashes 71 != records 72",
  );

  const badCorpusTaskMarker = buildFixture(path.join(root, "bad-corpus-task-marker"), (report) => {
    report.corpus_contract.task_marker_integrity.hash_mismatches = 1;
  });
  runChecker(
    "bad corpus task marker",
    badCorpusTaskMarker.promotion,
    false,
    "corpus task_marker_integrity hash_mismatches 1 != 0",
  );

  const badDenoiseProvenance = buildFixture(path.join(root, "bad-denoise-provenance"), (report) => {
    report.denoise_bridge.output_provenance.retrieval_head_hash_verified = false;
    report.confidence_trace.generation_bridge.output_provenance.retrieval_head_hash_verified = false;
  });
  runChecker("bad denoise provenance", badDenoiseProvenance.promotion, false, "output provenance retrieval head hash");

  const badDenoiseTargetCoverage = buildFixture(path.join(root, "bad-denoise-target-coverage"), (report) => {
    report.denoise_bridge.expected_spirit_ids = [1, 1];
    report.denoise_bridge.unique_expected_spirit_ids = [1];
    report.denoise_bridge.expected_unique_targets = 1;
    report.denoise_bridge.target_coverage_ok = false;
    report.confidence_trace.generation_bridge.expected_spirit_ids = [1, 1];
    report.confidence_trace.generation_bridge.unique_expected_spirit_ids = [1];
    report.confidence_trace.generation_bridge.expected_unique_targets = 1;
    report.confidence_trace.generation_bridge.target_coverage_ok = false;
  });
  runChecker(
    "bad denoise target coverage",
    badDenoiseTargetCoverage.promotion,
    false,
    "quality report denoise bridge unique targets 1 < 2",
  );

  console.log(JSON.stringify({
    schema: "nsrl.solomon_promotion_bundle_self_test.v1",
    ok: true,
    cases: [
      { name: "good", ok: true },
      { name: "bad-generation", ok: true },
      { name: "bad-integrity", ok: true },
      { name: "bad-source-provenance", ok: true },
      { name: "bad-source-grounding", ok: true },
      { name: "bad-grounded-corpus-overlap", ok: true },
      { name: "bad-grounded-attribute-rank", ok: true },
      { name: "bad-grounded-name-source", ok: true },
      { name: "bad-grounded-description-source", ok: true },
      { name: "bad-grounded-attribute-prompt", ok: true },
      { name: "bad-prompt-provenance", ok: true },
      { name: "bad-generated-signature", ok: true },
      { name: "bad-generated-distance", ok: true },
      { name: "bad-confidence-spine", ok: true },
      { name: "bad-heldout-retrieval", ok: true },
      { name: "bad-hard-negative-retrieval", ok: true },
      { name: "bad-directional-group", ok: true },
      { name: "bad-native-task-confidence", ok: true },
      { name: "bad-head-dim", ok: true },
      { name: "bad-hidden-dim", ok: true },
      { name: "bad-retrieval-text-head", ok: true },
      { name: "bad-retrieval-head-hash", ok: true },
      { name: "bad-symbolic-tokens", ok: true },
      { name: "bad-corpus-channel-stats", ok: true },
      { name: "bad-corpus-channel-duplicates", ok: true },
      { name: "bad-corpus-task-marker", ok: true },
      { name: "bad-denoise-provenance", ok: true },
      { name: "bad-denoise-target-coverage", ok: true },
    ],
  }, null, 2));
}

export function buildFixture(root, mutateQualityReport = null) {
  const attentionDir = path.join(root, "attention-curriculum");
  const generativeDir = path.join(root, "generative-eval", "current");
  fs.mkdirSync(attentionDir, { recursive: true });
  fs.mkdirSync(generativeDir, { recursive: true });

  const qualityReport = syntheticQualityReport();
  if (mutateQualityReport) {
    mutateQualityReport(qualityReport);
  }
  fs.writeFileSync(
    path.join(attentionDir, "quality-report.json"),
    `${JSON.stringify(qualityReport, null, 2)}\n`,
  );
  for (const [, fileName] of ATTENTION_ARTIFACTS) {
    const filePath = path.join(attentionDir, fileName);
    if (!fs.existsSync(filePath)) {
      fs.writeFileSync(filePath, `${fileName}\n`);
    }
  }
  fs.writeFileSync(path.join(root, "run.env"), "schema=nsrl.solomon_aws_pipeline.v1\n");
  fs.writeFileSync(path.join(root, "plan.tsv"), "stage\tcommand\n");
  fs.writeFileSync(path.join(root, "artifacts.tsv"), "stage\tartifact\tpath\n");
  fs.writeFileSync(path.join(generativeDir, "summary.tsv"), "model\tprompts\n");

  const rows = [
    ["pipeline", "run_env", path.join(root, "run.env")],
    ["pipeline", "plan", path.join(root, "plan.tsv")],
    ["pipeline", "artifacts", path.join(root, "artifacts.tsv")],
    ...ATTENTION_ARTIFACTS.map(([artifact, fileName]) => [
      "attention-curriculum",
      artifact,
      path.join(attentionDir, fileName),
    ]),
    ["generative-eval", "run", generativeDir],
    ["generative-eval", "summary", path.join(generativeDir, "summary.tsv")],
  ];
  const promotion = path.join(root, "promotion.tsv");
  fs.writeFileSync(
    promotion,
    [
      "product\tstage\tartifact\tpath\trequired",
      ...rows.map(([stage, artifact, filePath]) => [
        "solomon-v1",
        stage,
        artifact,
        filePath,
        "1",
      ].join("\t")),
    ].join("\n") + "\n",
  );
  return { root, promotion };
}

function syntheticQualityReport() {
  const denoiseBridge = syntheticDenoiseBridge();
  const report = {
    schema: "nsrl.solomon_v2_quality_report.v1",
    ok: true,
    model_only_quality_floor: {
      require_promoted_small_profile: true,
      require_heldout_prompts: true,
      require_denoise_bridge: true,
      require_denoise_output_identity: true,
      require_generative_eval: true,
      require_generative_output_identity: true,
      min_task_targets: "all=72",
      min_task_top5_per_mille: "all=1",
      min_phase_targets: "all=72",
      min_generated_prompt_rows: 72,
      require_grounded_corpus: true,
      min_grounded_source_overlap_tokens: 2,
      min_grounded_attribute_source_overlap_tokens: 8,
      max_grounded_source_placeholder_rows: 0,
      max_grounded_attribute_generic_rank_rows: 0,
      min_heldout_prompt_rows: 72,
      min_match_yes_top1: 72,
      min_match_no_top1: 72,
      min_match_no_image_top1: 72,
      min_match_no_prompt_top1: 72,
      effective_min_generated_top5_16_per_mille: 1,
      max_generated_mean_target_distance_16_q8: 7000000,
      min_denoise_bridge_unique_targets: 2,
    },
    attention_eval: {
      architecture: {
        d_model: 128,
        heads: 2,
        head_dim: 64,
        hidden_dim: 256,
        transformer_layers: 2,
        context_seq_len: 512,
        promoted_small_profile: { ok: true },
      },
      phases: Object.fromEntries(REQUIRED_EVAL_PHASES.map((phase) => [phase, syntheticNativeMetric()])),
    },
    retrieval_head_eval: {
      ok: true,
      model_hash: "retrieval-head-hash",
      feature_count: 256,
      known_prompts: syntheticRetrievalMetric(72),
      identity_bindings: {
        required_kinds: REQUIRED_IDENTITY_BINDING_KINDS,
        total: syntheticRetrievalMetric(360),
        by_kind: Object.fromEntries(
          REQUIRED_IDENTITY_BINDING_KINDS.map((kind) => [kind, syntheticRetrievalMetric(72)]),
        ),
      },
      heldout_prompts: syntheticRetrievalMetric(72),
      heldout_prompt_rows: 72,
      image_to_text: syntheticRetrievalMetric(288),
      image_tasks: Object.fromEntries(
        REQUIRED_IMAGE_BINDING_TASKS.map((task) => [
          task,
          syntheticRetrievalMetric(REQUIRED_IMAGE_BINDING_TASK_COUNTS[task]),
        ]),
      ),
      match: {
        yes: syntheticRetrievalMetric(72),
        no: syntheticRetrievalMetric(144),
        no_by_role: {
          image: syntheticRetrievalMetric(72),
          prompt: syntheticRetrievalMetric(72),
        },
      },
      class_retrieval_head: {
        source: "retrieval-head.json",
        path: "/tmp/retrieval-head.json",
        present: true,
        schema: "nsrl.solomon_v2_retrieval_head.v1",
        model_hash: "retrieval-head-hash",
        hash_matches_eval: true,
        hash_verified: true,
        feature_count: 256,
        labels: 72,
        text_head: true,
        image_head: true,
        text_nonzero_weights: 720,
        image_nonzero_weights: 720,
      },
    },
    denoise_bridge: {
      ...denoiseBridge,
    },
    generation_integrity: {
      ok: true,
      trace_count: 2,
      violations: 0,
    },
    corpus_contract: syntheticCorpusContract(),
    grounded_corpus: syntheticGroundedCorpus(),
    generative_eval: {
      present: true,
      ok: true,
      product_floor: {
        ok: true,
        matching_model: "current",
        requirements: {
          min_generated_top5_16_per_mille: 1,
          min_generated_prompt_rows: 72,
          require_generated_output_identity: true,
          max_generated_mean_target_distance_16_q8: 7000000,
        },
      },
      evidence: {
        prompt_provenance: syntheticPromptProvenance(),
      },
      best: syntheticGenerativeModel(),
      models: [
        syntheticGenerativeModel(),
      ],
    },
    confidence_trace: {
      ok: true,
      label: "strong-bidirectional-product-generation",
      directional_native_eval: {
        ok: true,
        groups: {
          text_prompt_to_image_plan: { ok: true },
          seal_image_to_text: { ok: true },
          text_and_seal_to_explanation: { ok: true },
          identity_source_binding: { ok: true },
        },
      },
      source_grounding: {
        present: true,
        grounded_corpus_present: true,
        grounded_corpus_ok: true,
        grounded_source_provenance: true,
        grounded_name_source_explain: true,
        grounded_description_source_image: true,
        grounded_image_attribute_generic_prompt: true,
        grounded_source_tasks: ["explain", "image-to-explain", "text-image-explain", "description-to-image"],
        grounded_attribute_tasks: ["image-to-attributes"],
        text_queries_have_source_text: true,
        image_queries_have_source_text: true,
        sample_queries_have_source_text: true,
        sample_source_text_evidence: true,
        generated_text_source_evidence: true,
        generated_text_image_agreement: true,
        expected_generated_text_agreement: true,
        min_source_text_chars: 64,
      },
      text_binding: syntheticTextBinding(),
      image_binding: syntheticImageBinding(),
      forward_image_plan: syntheticForwardImagePlan(),
      native_task_eval: syntheticNativeTaskEval(),
      cross_modal_agreement: syntheticCrossModalAgreement(),
      symbolic_image_tokens: syntheticSymbolicImageTokens(),
      generation_bridge: syntheticGenerationBridge(),
      product_generation: {
        product_floor_ok: true,
        trace_integrity_ok: true,
        trace_count: 72,
        output_identity_required: true,
        matching_model_output_identity: { ok: true },
        prompt_provenance: syntheticPromptProvenance(),
        best_retrieval_top1_per_mille: 1000,
        best_retrieval_min_margin: 1,
        best_top5_16_per_mille: 1000,
      },
    },
  };
  for (const flag of READY_FLAGS) {
    report[flag] = true;
  }
  return report;
}

function syntheticDenoiseBridge() {
  const expectedSpiritIds = [1, 2];
  const pairCount = expectedSpiritIds.length;
  return {
    present: true,
    ok: true,
    pairs: pairCount,
    min_unique_targets: 2,
    expected_spirit_ids: expectedSpiritIds,
    unique_expected_spirit_ids: expectedSpiritIds,
    expected_unique_targets: expectedSpiritIds.length,
    missing_expected_spirit_ids: Array.from({ length: 72 - expectedSpiritIds.length }, (_entry, index) => index + expectedSpiritIds.length + 1),
    target_coverage_ok: true,
    denoise_model: "denoiser.nsrltch",
    denoise_model_hash: "denoise-hash",
    denoise_model_provenance: syntheticDenoiseModelProvenance(pairCount),
    sample_binding_provenance: syntheticDenoiseSampleBindingProvenance(pairCount),
    output_provenance: syntheticDenoiseOutputProvenance(pairCount),
    min_output_signature_distance: 2,
    min_output_ink_range: 8,
    trace_integrity_ok: true,
    require_output_image_to_text_identification: true,
    output_image_to_text_identification: true,
    min_output_retrieval_image_margin: 1,
  };
}

function syntheticGenerationBridge() {
  const denoise = syntheticDenoiseBridge();
  return {
    present: true,
    required: true,
    output_identity_required: true,
    pairs: denoise.pairs,
    denoise_model: denoise.denoise_model,
    denoise_model_hash: denoise.denoise_model_hash,
    denoise_model_provenance: denoise.denoise_model_provenance,
    sample_binding_provenance: denoise.sample_binding_provenance,
    output_provenance: denoise.output_provenance,
    min_output_signature_distance: denoise.min_output_signature_distance,
    min_output_ink_range: denoise.min_output_ink_range,
    trace_integrity_ok: denoise.trace_integrity_ok,
    min_unique_targets: denoise.min_unique_targets,
    expected_spirit_ids: denoise.expected_spirit_ids,
    unique_expected_spirit_ids: denoise.unique_expected_spirit_ids,
    expected_unique_targets: denoise.expected_unique_targets,
    missing_expected_spirit_ids: denoise.missing_expected_spirit_ids,
    target_coverage_ok: denoise.target_coverage_ok,
    output_image_to_text_identification: denoise.output_image_to_text_identification,
    min_output_retrieval_image_margin: denoise.min_output_retrieval_image_margin,
  };
}

function syntheticDenoiseModelProvenance(pairCount = 2) {
  return {
    ok: true,
    denoise_model: "denoiser.nsrltch",
    resolved_denoise_model: "/tmp/denoiser.nsrltch",
    denoise_model_hash: "denoise-hash",
    denoise_model_hashes: ["denoise-hash"],
    denoise_model_consistent: true,
    result_count: pairCount,
    resolved_result_model_count: pairCount,
    missing_model_refs: 0,
    missing_model_hashes: 0,
    unresolved_models: 0,
    hash_mismatches: 0,
    unique_recomputed_hashes: ["denoise-hash"],
    results: [],
  };
}

function syntheticDenoiseSampleBindingProvenance(pairCount = 2) {
  const names = ["Bael", "Agares"];
  return {
    sample_binding: "sample-binding.json",
    sample_count: pairCount,
    bridge_result_count: pairCount,
    matched_attention_plans: pairCount,
    missing_attention_plans: 0,
    prompt_mismatches: 0,
    identity_mismatches: 0,
    output_identity_mismatches: 0,
    matches: Array.from({ length: pairCount }, (_entry, index) => {
      const spiritId = index + 1;
      const name = names[index] || `Spirit ${spiritId}`;
      const sampleDir = `sample-${spiritId}`;
      return {
        index,
        attention_plan: `${sampleDir}/image.ink16.u8`,
        matched_sample_dir: sampleDir,
        matched_image_ink16_u8: `${sampleDir}/image.ink16.u8`,
        plan_match: true,
        prompt_match: true,
        bridge_expected_spirit_id: spiritId,
        sample_expected_spirit_id: spiritId,
        bridge_expected_primary_name: name,
        sample_expected_primary_name: name,
        identity_match: true,
        output_identity_match: true,
      };
    }),
  };
}

function syntheticDenoiseOutputProvenance(pairCount = 2) {
  return {
    required: true,
    ok: true,
    result_count: pairCount,
    retrieval_required: true,
    config_retrieval_head_model_hash: "retrieval-head-hash",
    expected_retrieval_head_model_hash: "retrieval-head-hash",
    config_hash_match: true,
    retrieval_head: "retrieval-head.json",
    resolved_retrieval_head: "/tmp/retrieval-head.json",
    retrieval_head_present: true,
    invalid_retrieval_head: false,
    retrieval_head_model_hash: "retrieval-head-hash",
    recomputed_retrieval_head_model_hash: "retrieval-head-hash",
    retrieval_head_hash_verified: true,
    retrieval_head_hash_match: true,
    retrieval_head_feature_count: 256,
    retrieval_head_label_count: 72,
    scored_results: pairCount,
    missing_attention_plans: 0,
    missing_raw_samples: 0,
    invalid_raw_samples: 0,
    result_mismatches: 0,
    detail_mismatches: 0,
    aggregate_mismatches: [],
    recomputed: {
      min_output_signature_distance: 2,
      min_output_ink_range: 8,
      output_image_to_text_identification: true,
      min_output_retrieval_image_margin: 1,
    },
    results: [],
  };
}

function syntheticSymbolicImageTokens() {
  const stage = (index, stageName) => ({
    index,
    stage_name: stageName,
    expected_stage_name: stageName,
    ok: true,
    checked_records: 72,
    required_channels: symbolicChannels(),
    by_channel: syntheticImageChannelSummaries(),
  });
  return {
    required: true,
    ok: true,
    required_channels: symbolicChannels(),
    corpus: {
      present: true,
      ok: true,
      checked_records: 1872,
      missing_image_markers: 0,
      missing_channel_markers: 0,
      short_channel_payloads: 0,
      bad_channel_payloads: 0,
      channel_order_mismatches: 0,
      by_channel: syntheticImageChannelSummaries(1872),
    },
    curriculum: {
      present: true,
      required: true,
      ok: true,
      stage_count: 2,
      stages: [
        stage(0, "identity"),
        stage(1, "image-to-text"),
      ],
    },
  };
}

function symbolicChannels() {
  return ["ink", "edge", "component", "radial", "direction"];
}

function syntheticImageChannelSummaries(records = 72) {
  return Object.fromEntries(
    symbolicChannels().map((channel) => [
      channel,
      {
        checked_records: records,
        found_markers: records,
        missing_channel_markers: 0,
        short_channel_payloads: 0,
        bad_channel_payloads: 0,
        channel_order_mismatches: 0,
      },
    ]),
  );
}

function syntheticTextBinding() {
  const byKind = {};
  for (const kind of ["primary-name", "primary-seal", "alias", "alias-seal", "seal-id"]) {
    byKind[kind] = syntheticConfidenceMetric(72);
  }
  return {
    known_prompts: syntheticConfidenceMetric(72),
    identity_bindings: {
      total: syntheticConfidenceMetric(360),
      by_kind: byKind,
    },
    heldout_prompts: syntheticConfidenceMetric(1051),
  };
}

function syntheticImageBinding() {
  const imageTasks = {};
  for (const task of REQUIRED_IMAGE_BINDING_TASKS) {
    imageTasks[task] = syntheticConfidenceMetric(REQUIRED_IMAGE_BINDING_TASK_COUNTS[task]);
  }
  return {
    image_to_text: syntheticConfidenceMetric(288),
    image_tasks: imageTasks,
    sample_image_to_text_identification: true,
    min_image_to_text_margin: 1,
    min_retrieval_image_margin: 1,
  };
}

function syntheticForwardImagePlan() {
  return {
    tasks: {
      "text-to-image": syntheticNativeMetric(),
      "description-to-image": syntheticNativeMetric(),
    },
  };
}

function syntheticNativeTaskEval() {
  return {
    tasks: Object.fromEntries(REQUIRED_CORPUS_TASKS.map((task) => [task, syntheticNativeMetric()])),
    weakest_top5: {
      task: "canonical-joint",
      top5_accuracy_per_mille: 1000,
    },
    weakest_margin: {
      task: "canonical-joint",
      mean_target_margin_q8: 256,
    },
  };
}

function syntheticCrossModalAgreement() {
  return {
    match_yes: syntheticConfidenceMetric(72),
    match_no: syntheticConfidenceMetric(144),
    wrong_image_negatives: syntheticConfidenceMetric(72),
    wrong_prompt_negatives: syntheticConfidenceMetric(72),
    text_image_agreement: true,
    generated_text_image_agreement: true,
    generated_text_identification: true,
    signature_retrieval_agreement: true,
    min_signature_margin: 1,
    min_retrieval_text_margin: 1,
    min_generated_text_margin: 1,
  };
}

function syntheticConfidenceMetric(count) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: 1,
    mean_margin: 1,
  };
}

function syntheticRetrievalMetric(count) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: 1,
    mean_margin: 1,
  };
}

function syntheticNativeMetric() {
  return {
    targets: 72,
    correct: 72,
    invalid_contexts: 0,
    accuracy_per_mille: 1000,
    top5_accuracy_per_mille: 1000,
    top10_accuracy_per_mille: 1000,
    mean_target_rank_per_mille: 1000,
  };
}

function syntheticPromptProvenance() {
  return {
    prompts_hash_match: true,
    prompt_rows_match: true,
    selected_prompt_rows_match: true,
    selected_prompt_hash_match: true,
    selected_prompt_eligible_rows_recorded: true,
    selected_prompt_eligible_rows: 72,
    expected_selected_prompt_eligible_rows: 72,
    selected_prompt_eligible_rows_match: true,
    selected_prompt_unique_targets_recorded: true,
    selected_prompt_unique_targets: 72,
    expected_selected_unique_targets: 72,
    selected_prompt_unique_targets_match: true,
    selected_prompt_eligible_unique_targets_recorded: true,
    selected_prompt_eligible_unique_targets: 72,
    expected_selected_prompt_eligible_unique_targets: 72,
    selected_prompt_eligible_unique_targets_match: true,
  };
}

function syntheticCorpusContract() {
  const v2Records = 720;
  return {
    present: true,
    ok: true,
    required_corpus_version: "v2",
    required_image_token_profile: "symbolic16",
    required_image_token_channels: symbolicChannels(),
    require_image_channel_token_stats: true,
    min_image_channel_distinct_bins: 2,
    manifest_summary: {
      corpus_version: "v2",
      image_token_profile: "symbolic16",
      image_token_channels: symbolicChannels(),
      image_token_channel_stats: Object.fromEntries(
        symbolicChannels().map((channel) => [channel, syntheticCorpusChannelStats()]),
      ),
      examples: 72,
      training_sequences: 720,
      token_hash: "0x1234",
    },
    examples_summary: {
      records: v2Records,
      distinct_spirits: 72,
      v2_records: v2Records,
      missing_image_token_profile: 0,
      missing_image_token_channels: 0,
      image_token_profiles: { symbolic16: v2Records },
      required_channel_rows: Object.fromEntries(symbolicChannels().map((channel) => [channel, v2Records])),
      tasks: syntheticCorpusTaskCoverage(),
      coverage_errors: [],
    },
    task_marker_integrity: syntheticCorpusTaskMarkerIntegrity(),
    task_modality_integrity: syntheticCorpusTaskModalityIntegrity(),
    image_channel_marker_integrity: syntheticCorpusImageChannelMarkerIntegrity(),
  };
}

function syntheticCorpusChannelStats() {
  return {
    records: 72,
    tokens_per_record: 256,
    active_records: 72,
    multi_bin_records: 72,
    nonzero_tokens: 4096,
    distinct_bins: 8,
    max_bin: 15,
    unique_record_hashes: 72,
    duplicate_record_hashes: 0,
  };
}

function syntheticCorpusTaskCoverage() {
  return Object.fromEntries(
    REQUIRED_CORPUS_TASKS.map((task) => {
      const summary = {
        records: task === "match" ? 216 : 72,
        spirits: 72,
      };
      if (task === "match") {
        summary.labels = {
          yes: { records: 72, spirits: 72 },
          no: {
            records: 144,
            spirits: 72,
            roles: {
              image: { records: 72, spirits: 72 },
              prompt: { records: 72, spirits: 72 },
            },
          },
        };
      }
      return [task, summary];
    }),
  );
}

function syntheticCorpusTaskMarkerIntegrity() {
  return {
    ok: true,
    present: true,
    tokens: "tokens.u16",
    checked_records: 720,
    hash_mismatches: 0,
    marker_mismatches: 0,
    out_of_bounds: 0,
    missing_offsets: 0,
    by_task: Object.fromEntries(
      REQUIRED_CORPUS_TASKS.map((task) => [task, { checked_records: task === "match" ? 216 : 72 }]),
    ),
  };
}

function syntheticCorpusTaskModalityIntegrity() {
  return {
    ok: true,
    present: true,
    tokens: "tokens.u16",
    checked_records: 720,
    missing_offsets: 0,
    out_of_bounds: 0,
    modality_mismatches: 0,
    by_task: Object.fromEntries(
      REQUIRED_CORPUS_TASKS.map((task) => [task, { checked_records: task === "match" ? 216 : 72 }]),
    ),
  };
}

function syntheticCorpusImageChannelMarkerIntegrity() {
  return {
    ok: true,
    present: true,
    tokens: "tokens.u16",
    required_channels: symbolicChannels(),
    checked_records: 504,
    missing_offsets: 0,
    out_of_bounds: 0,
    missing_image_markers: 0,
    missing_channel_markers: 0,
    short_channel_payloads: 0,
    bad_channel_payloads: 0,
    channel_order_mismatches: 0,
    by_channel: Object.fromEntries(
      symbolicChannels().map((channel) => [
        channel,
        {
          checked_records: 504,
          found_markers: 504,
          missing_offsets: 0,
          out_of_bounds: 0,
          missing_image_markers: 0,
          missing_channel_markers: 0,
          short_channel_payloads: 0,
          bad_channel_payloads: 0,
          channel_order_mismatches: 0,
        },
      ]),
    ),
  };
}

function syntheticGroundedCorpus() {
  return {
    present: true,
    ok: true,
    expect_spirits: 72,
    source_text_tasks: ["explain", "image-to-explain", "text-image-explain", "description-to-image"],
    attribute_tasks: ["image-to-attributes"],
    require_source_provenance: true,
    require_name_source_explain: true,
    require_description_source_image: true,
    require_image_attribute_generic_prompt: true,
    min_source_overlap_tokens: 2,
    min_attribute_source_overlap_tokens: 8,
    max_source_placeholder_rows: 0,
    max_attribute_generic_rank_rows: 0,
    tasks: {
      explain: syntheticGroundedTask(72, 2),
      "image-to-explain": syntheticGroundedTask(72, 2),
      "text-image-explain": syntheticGroundedTask(72, 2),
      "description-to-image": syntheticGroundedTask(72, 2),
      "image-to-attributes": {
        ...syntheticGroundedTask(72, 8),
        generic_attribute_rank_rows: 0,
      },
    },
  };
}

function syntheticGroundedTask(records, overlap) {
  return {
    records,
    spirits: 72,
    min_source_overlap_tokens: overlap,
    min_content_tokens: overlap,
    source_substring_rows: records,
    placeholder_rows: 0,
    placeholder_count: 0,
    source_provenance_rows: records,
    source_provenance_hash_mismatches: 0,
    source_excerpt_hash_mismatches: 0,
    name_source_prompt_rows: records,
    name_source_prompt_ok_rows: records,
    description_source_prompt_rows: records,
    description_source_prompt_ok_rows: records,
    image_attribute_prompt_rows: records,
    image_attribute_prompt_ok_rows: records,
  };
}

function syntheticGenerativeModel() {
  return {
    model: "current",
    prompts: 72,
    generated_signature_16: {
      top5_per_mille: 1000,
      mean_rank_q8: 256,
      mean_target_distance_q8: 1000,
    },
    generated_retrieval: {
      present: true,
      top1: 72,
      top5: 72,
      top1_per_mille: 1000,
      top5_per_mille: 1000,
      min_margin: 1,
    },
  };
}

function runChecker(label, promotion, shouldPass, expectedText = "") {
  const result = childProcess.spawnSync(
    process.execPath,
    ["scripts/check-solomon-promotion-bundle.mjs", "--promotion", promotion],
    { cwd: repoRoot, encoding: "utf8" },
  );
  if (shouldPass && result.status !== 0) {
    throw new Error(`${label} failed unexpectedly\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
  }
  if (!shouldPass && result.status === 0) {
    throw new Error(`${label} passed unexpectedly`);
  }
  if (!shouldPass && expectedText) {
    const output = `${result.stdout}\n${result.stderr}`;
    if (!output.includes(expectedText)) {
      throw new Error(`${label} expected ${JSON.stringify(expectedText)}, got:\n${output}`);
    }
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  }
}
