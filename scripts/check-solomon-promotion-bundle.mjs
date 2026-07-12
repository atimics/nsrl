#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const REQUIRED_ARTIFACTS = [
  "run_env",
  "plan",
  "artifacts",
  "quality_report",
  "model",
  "corpus_manifest",
  "attention_eval",
  "retrieval_head",
  "retrieval_head_eval",
  "curriculum_stages",
  "sample_binding",
  "identity_inference",
  "grounded_corpus",
  "generation_integrity",
  "denoise_bridge",
  "denoise_generation_integrity",
  "run",
  "summary",
];

const REQUIRED_READY_FLAGS = [
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
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const REQUIRED_FORWARD_IMAGE_PLAN_TASKS = [
  "text-to-image",
  "description-to-image",
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
const REQUIRED_GROUNDED_SOURCE_TASKS = [
  "explain",
  "image-to-explain",
  "text-image-explain",
  "description-to-image",
];
const REQUIRED_GROUNDED_ATTRIBUTE_TASKS = [
  "image-to-attributes",
];
const REQUIRED_IMAGE_TOKEN_CHANNELS = [
  "ink",
  "edge",
  "component",
  "radial",
  "direction",
];
const REQUIRED_DIRECTIONAL_GROUPS = [
  "text_prompt_to_image_plan",
  "seal_image_to_text",
  "text_and_seal_to_explanation",
  "identity_source_binding",
];
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
const REQUIRED_NATIVE_EVAL_TASKS = REQUIRED_CORPUS_TASKS;
const PROMOTED_SMALL_PROFILE = {
  dModel: 128,
  heads: 2,
  headDim: 64,
  hiddenDimMin: 256,
  hiddenDimMax: 512,
  transformerLayersMin: 2,
  transformerLayersMax: 4,
  contextSeqLenMin: 384,
  contextSeqLenMax: 768,
};

const defaults = {
  promotionPath: "",
  qualityReportPath: "",
  outPath: "",
  requireProduct: "solomon-v1",
  requireRequiredArtifacts: true,
  requireExistingArtifacts: true,
  requireQualityReady: true,
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-promotion-bundle.mjs --promotion PATH [options]",
      "",
      "Checks a completed Solomon promotion bundle. This is the post-run gate:",
      "promotion.tsv must point to required artifacts, and quality-report.json",
      "must say the narrow grounded multimodal product is ready.",
      "",
      "Options:",
      "  --quality-report PATH",
      "  --out PATH",
      "  --require-product NAME",
      "  --allow-missing-artifacts",
      "  --allow-not-ready",
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
    } else if (arg === "--promotion") {
      config.promotionPath = requireValue(argv, ++index, arg);
    } else if (arg === "--quality-report") {
      config.qualityReportPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-product") {
      config.requireProduct = requireValue(argv, ++index, arg);
    } else if (arg === "--allow-missing-artifacts") {
      config.requireRequiredArtifacts = false;
      config.requireExistingArtifacts = false;
    } else if (arg === "--allow-not-ready") {
      config.requireQualityReady = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.promotionPath) {
    throw new Error("--promotion is required");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readPromotionManifest(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  if (lines.length === 0 || lines[0] !== "product\tstage\tartifact\tpath\trequired") {
    throw new Error(`${filePath} must start with product\\tstage\\tartifact\\tpath\\trequired`);
  }
  const rows = [];
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
      line: index + 1,
    });
  }
  return rows;
}

function normalizeReferencedPath(filePath) {
  const resolved = path.resolve(filePath);
  try {
    return fs.realpathSync.native(resolved);
  } catch (_error) {
    return resolved;
  }
}

function resolveArtifactPath(ref, manifestDir) {
  if (!ref) {
    return "";
  }
  return normalizeReferencedPath(path.isAbsolute(ref) ? ref : path.join(manifestDir, ref));
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function checkQualityReport(report, reportPath, config, errors) {
  if (report.schema !== "nsrl.solomon_v2_quality_report.v1") {
    errors.push(`quality report schema ${JSON.stringify(report.schema || "")} != nsrl.solomon_v2_quality_report.v1`);
  }
  if (config.requireQualityReady && report.ok !== true) {
    errors.push("quality report ok is not true");
  }
  for (const flag of REQUIRED_READY_FLAGS) {
    if (config.requireQualityReady && report[flag] !== true) {
      errors.push(`quality report ${flag} is not true`);
    }
  }

  const floor = report.model_only_quality_floor || {};
  if (floor.require_promoted_small_profile !== true) {
    errors.push("quality report did not require promoted small profile");
  }
  if (floor.require_heldout_prompts !== true) {
    errors.push("quality report did not require held-out prompts");
  }
  if (floor.require_denoise_bridge !== true) {
    errors.push("quality report did not require denoise bridge");
  }
  if (floor.require_denoise_output_identity !== true) {
    errors.push("quality report did not require denoise output identity");
  }
  if (floor.require_generative_eval !== true) {
    errors.push("quality report did not require generative eval");
  }
  if (floor.require_generative_output_identity !== true) {
    errors.push("quality report did not require generative output identity");
  }
  if (Number(floor.min_generated_prompt_rows || 0) < 72) {
    errors.push(`quality report min_generated_prompt_rows ${floor.min_generated_prompt_rows || 0} < 72`);
  }
  const minDenoiseBridgeUniqueTargets = Number(floor.min_denoise_bridge_unique_targets || 0);
  if (minDenoiseBridgeUniqueTargets < 2) {
    errors.push(`quality report min_denoise_bridge_unique_targets ${floor.min_denoise_bridge_unique_targets || 0} < 2`);
  }

  const confidence = report.confidence_trace || {};
  if (confidence.label !== "strong-bidirectional-product-generation") {
    errors.push(`quality report confidence label ${JSON.stringify(confidence.label || "")} != "strong-bidirectional-product-generation"`);
  }
  if (confidence.directional_native_eval?.ok !== true) {
    errors.push("quality report directional native eval evidence is not ok");
  }
  checkDirectionalNativeEvalEvidence(confidence.directional_native_eval || {}, errors);
  checkConfidenceTraceSpine(confidence, errors);
  checkSymbolicImageTokenEvidence(confidence.symbolic_image_tokens || {}, errors);
  const corpusContract = checkCorpusContractEvidence(report.corpus_contract || {}, errors);
  const sourceGrounding = confidence.source_grounding || {};
  checkSourceGroundingEvidence(sourceGrounding, errors);
  const groundedCorpus = checkGroundedCorpusEvidence(report.grounded_corpus || {}, floor, errors);

  const integrity = report.generation_integrity || {};
  if (integrity.ok !== true) {
    errors.push("quality report generation integrity is not ok");
  }
  if (Number(integrity.trace_count || 0) <= 0) {
    errors.push(`quality report generation integrity trace_count ${integrity.trace_count || 0} <= 0`);
  }
  if (Number(integrity.violations || 0) !== 0) {
    errors.push(`quality report generation integrity violations ${integrity.violations || 0} != 0`);
  }

  const denoise = report.denoise_bridge || {};
  checkDenoiseBridgeEvidence(denoise, confidence.generation_bridge || {}, minDenoiseBridgeUniqueTargets, errors);

  const generative = report.generative_eval || {};
  const productGeneration = confidence.product_generation || {};
  if (generative.present !== true || generative.ok !== true) {
    errors.push("quality report generative eval artifact is not present and ok");
  }
  if (generative.product_floor?.ok !== true) {
    errors.push("quality report generative product floor is not ok");
  }
  if (!generative.product_floor?.matching_model) {
    errors.push("quality report generative product floor has no matching model");
  }
  if (productGeneration.product_floor_ok !== true) {
    errors.push("quality report confidence product-generation floor is not ok");
  }
  if (productGeneration.trace_integrity_ok !== true) {
    errors.push("quality report confidence product-generation trace integrity is not ok");
  }
  if (Number(productGeneration.trace_count || 0) < Number(floor.min_generated_prompt_rows || 0)) {
    errors.push(
      `quality report confidence product-generation trace_count ${productGeneration.trace_count || 0} < ${floor.min_generated_prompt_rows || 0}`,
    );
  }
  if (productGeneration.output_identity_required !== true) {
    errors.push("quality report confidence product-generation output identity is not required");
  }
  if (productGeneration.matching_model_output_identity?.ok !== true) {
    errors.push("quality report matching generative model output identity is not ok");
  }
  if (Number(productGeneration.best_retrieval_top1_per_mille || 0) < 1000) {
    errors.push(`quality report best generated retrieval top1 ${productGeneration.best_retrieval_top1_per_mille || 0} < 1000`);
  }
  if (Number(productGeneration.best_retrieval_min_margin || 0) <= 0) {
    errors.push(`quality report best generated retrieval min margin ${productGeneration.best_retrieval_min_margin || 0} <= 0`);
  }
  const generatedSealQuality = checkGeneratedSealQualityEvidence(generative, productGeneration, floor, errors);
  const promptProvenance =
    generative.evidence?.prompt_provenance ||
    generative.prompt_provenance ||
    productGeneration.prompt_provenance ||
    {};
  const minGeneratedPromptRows = Number(floor.min_generated_prompt_rows || 0);
  if (!promptProvenance || Object.keys(promptProvenance).length === 0) {
    errors.push("quality report generative prompt provenance is missing");
  } else {
    if (promptProvenance.prompts_hash_match !== true) {
      errors.push("quality report generative prompt promptsHash did not match");
    }
    if (promptProvenance.prompt_rows_match !== true) {
      errors.push("quality report generative prompt row count did not match");
    }
    if (promptProvenance.selected_prompt_rows_match !== true) {
      errors.push("quality report generative selected prompt rows did not match");
    }
    if (promptProvenance.selected_prompt_hash_match !== true) {
      errors.push("quality report generative selected prompt hash did not match");
    }
    if (promptProvenance.selected_prompt_eligible_rows_match !== true) {
      errors.push("quality report generative held-out selected prompt rows did not match");
    }
    if (promptProvenance.selected_prompt_unique_targets_match !== true) {
      errors.push("quality report generative selected prompt unique targets did not match");
    }
    if (promptProvenance.selected_prompt_eligible_unique_targets_match !== true) {
      errors.push("quality report generative held-out selected prompt unique targets did not match");
    }
    if (Number(promptProvenance.selected_prompt_eligible_rows || 0) < minGeneratedPromptRows) {
      errors.push(
        `quality report generative selected held-out prompt rows ${promptProvenance.selected_prompt_eligible_rows || 0} < ${minGeneratedPromptRows}`,
      );
    }
    if (Number(promptProvenance.selected_prompt_unique_targets || 0) < minGeneratedPromptRows) {
      errors.push(
        `quality report generative selected prompt unique targets ${promptProvenance.selected_prompt_unique_targets || 0} < ${minGeneratedPromptRows}`,
      );
    }
    if (Number(promptProvenance.selected_prompt_eligible_unique_targets || 0) < minGeneratedPromptRows) {
      errors.push(
        `quality report generative selected held-out unique targets ${promptProvenance.selected_prompt_eligible_unique_targets || 0} < ${minGeneratedPromptRows}`,
      );
    }
  }

  const architecture = report.attention_eval?.architecture || {};
  const promoted = architecture.promoted_small_profile || {};
  checkPromotedSmallArchitecture(architecture, promoted, errors);
  const classRetrievalHead = checkClassRetrievalHeadEvidence(report.retrieval_head_eval || {}, errors);
  const retrievalSpine = checkRetrievalSpineEvidence(report.retrieval_head_eval || {}, floor, errors);

  return {
    path: reportPath,
    ok: report.ok === true,
    ready_flags: Object.fromEntries(REQUIRED_READY_FLAGS.map((flag) => [flag, report[flag] === true])),
    architecture: {
      d_model: Number(architecture.d_model || 0),
      heads: Number(architecture.heads || 0),
      hidden_dim: Number(architecture.hidden_dim || 0),
      transformer_layers: Number(architecture.transformer_layers || 0),
      context_seq_len: Number(architecture.context_seq_len || 0),
      promoted_small_profile_ok: promoted.ok === true,
    },
    class_retrieval_head: classRetrievalHead,
    retrieval_spine: retrievalSpine,
    confidence_label: confidence.label || "",
    confidence_spine: summarizeConfidenceTraceSpine(confidence),
    symbolic_image_tokens: summarizeSymbolicImageTokenEvidence(confidence.symbolic_image_tokens || {}),
    corpus_contract: corpusContract,
    source_grounding: {
      grounded_source_provenance: sourceGrounding.grounded_source_provenance === true,
      grounded_source_tasks: Array.isArray(sourceGrounding.grounded_source_tasks)
        ? sourceGrounding.grounded_source_tasks.map(String)
        : [],
      grounded_attribute_tasks: Array.isArray(sourceGrounding.grounded_attribute_tasks)
        ? sourceGrounding.grounded_attribute_tasks.map(String)
        : [],
      min_source_text_chars: Number(sourceGrounding.min_source_text_chars || 0),
    },
    grounded_corpus: groundedCorpus,
    denoise: {
      pairs: Number(denoise.pairs || 0),
      trace_integrity_ok: denoise.trace_integrity_ok === true,
      denoise_model_provenance_ok: denoise.denoise_model_provenance?.ok === true,
      sample_binding_matched_attention_plans: Number(denoise.sample_binding_provenance?.matched_attention_plans || 0),
      min_output_signature_distance:
        finiteNumberOrNull(denoise.min_output_signature_distance) ?? null,
      min_output_ink_range: finiteNumberOrNull(denoise.min_output_ink_range) ?? null,
      output_image_to_text_identification: denoise.output_image_to_text_identification === true,
      min_output_retrieval_image_margin: Number(denoise.min_output_retrieval_image_margin || 0),
      min_unique_targets: Number(denoise.min_unique_targets || 0),
      expected_unique_targets: Number(denoise.expected_unique_targets || 0),
      target_coverage_ok: denoise.target_coverage_ok === true,
      output_provenance_ok: denoise.output_provenance?.ok === true,
      output_retrieval_head_hash_verified:
        denoise.output_provenance?.retrieval_head_hash_verified === true,
      generation_bridge_output_provenance_ok:
        confidence.generation_bridge?.output_provenance?.ok === true,
    },
    generation_integrity: {
      ok: integrity.ok === true,
      trace_count: Number(integrity.trace_count || 0),
      violations: Number(integrity.violations || 0),
    },
    product_generation: {
      matching_model: generative.product_floor?.matching_model || "",
      product_floor_ok: generative.product_floor?.ok === true,
      trace_integrity_ok: productGeneration.trace_integrity_ok === true,
      trace_count: Number(productGeneration.trace_count || 0),
      selected_prompt_eligible_rows: Number(promptProvenance.selected_prompt_eligible_rows || 0),
      selected_prompt_unique_targets: Number(promptProvenance.selected_prompt_unique_targets || 0),
      selected_prompt_eligible_unique_targets: Number(promptProvenance.selected_prompt_eligible_unique_targets || 0),
      output_identity_ok: productGeneration.matching_model_output_identity?.ok === true,
      best_retrieval_top1_per_mille: Number(productGeneration.best_retrieval_top1_per_mille || 0),
      best_retrieval_min_margin: Number(productGeneration.best_retrieval_min_margin || 0),
      generated_seal_quality: generatedSealQuality,
    },
  };
}

function checkRetrievalSpineEvidence(retrievalEval, floor, errors) {
  const minHeldoutRows = Math.max(Number(floor.min_heldout_prompt_rows || 0), 72);
  const metrics = {
    known_prompts: requireRetrievalMetric(retrievalEval.known_prompts, "known prompts", 72, errors),
    heldout_prompts: requireRetrievalMetric(retrievalEval.heldout_prompts, "held-out prompts", minHeldoutRows, errors),
    identity_bindings: requireRetrievalMetric(
      retrievalEval.identity_bindings?.total,
      "identity bindings",
      REQUIRED_IDENTITY_BINDING_KINDS.length * 72,
      errors,
    ),
    image_to_text: requireRetrievalMetric(retrievalEval.image_to_text, "image-to-text/source", 72, errors),
    match_yes: requireRetrievalMetric(retrievalEval.match?.yes, "match yes", 72, errors),
    match_no: requireRetrievalMetric(retrievalEval.match?.no, "match no", 144, errors),
    wrong_image_negatives: requireRetrievalMetric(
      retrievalEval.match?.no_by_role?.image,
      "wrong-image hard negatives",
      72,
      errors,
    ),
    wrong_prompt_negatives: requireRetrievalMetric(
      retrievalEval.match?.no_by_role?.prompt,
      "wrong-prompt hard negatives",
      72,
      errors,
    ),
  };
  metrics.identity_binding_kinds = {};
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    metrics.identity_binding_kinds[kind] = requireRetrievalMetric(
      retrievalEval.identity_bindings?.by_kind?.[kind],
      `identity binding ${kind}`,
      72,
      errors,
    );
  }
  metrics.image_tasks = {};
  for (const task of REQUIRED_IMAGE_BINDING_TASKS) {
    metrics.image_tasks[task] = requireRetrievalMetric(
      retrievalEval.image_tasks?.[task],
      `${task}`,
      REQUIRED_IMAGE_BINDING_TASK_COUNTS[task],
      errors,
    );
  }
  return metrics;
}

function requireRetrievalMetric(metric, label, minCount, errors) {
  const summary = retrievalMetricSummary(metric);
  if (summary.count < minCount) {
    errors.push(`quality report retrieval ${label} count ${summary.count} < ${minCount}`);
  }
  if (summary.count > 0 && summary.top1 !== summary.count) {
    errors.push(`quality report retrieval ${label} top1 ${summary.top1} != count ${summary.count}`);
  }
  if (summary.count > 0 && summary.top5 < summary.count) {
    errors.push(`quality report retrieval ${label} top5 ${summary.top5} < count ${summary.count}`);
  }
  if (summary.min_margin === null || summary.min_margin <= 0) {
    errors.push(`quality report retrieval ${label} min_margin ${summary.min_margin ?? 0} <= 0`);
  }
  return summary;
}

function retrievalMetricSummary(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    top1_per_mille: Number(metric?.top1_per_mille || 0),
    top5_per_mille: Number(metric?.top5_per_mille || 0),
    min_margin: finiteNumberOrNull(metric?.min_margin),
  };
}

function checkGeneratedSealQualityEvidence(generative, productGeneration, floor, errors) {
  const productFloor = generative.product_floor || {};
  const requirements = productFloor.requirements || {};
  const top516Floor = Math.max(
    Number(floor.effective_min_generated_top5_16_per_mille || 0),
    Number(requirements.min_generated_top5_16_per_mille || 0),
  );
  const targetDistance16Cap = Math.max(
    Number(floor.max_generated_mean_target_distance_16_q8 || 0),
    Number(requirements.max_generated_mean_target_distance_16_q8 || 0),
  );
  const matchingModelName = productFloor.matching_model || productGeneration.matching_model || "";
  const models = Array.isArray(generative.models) ? generative.models : [];
  const matchingModel = models.find((model) => model?.model === matchingModelName) || generative.best || {};
  const signature16 = matchingModel.generated_signature_16 || {};
  const matchingTop516 = Number(signature16.top5_per_mille || productGeneration.best_top5_16_per_mille || 0);
  const matchingMeanTargetDistance16 = finiteNumberOrNull(signature16.mean_target_distance_q8);
  if (top516Floor < 1) {
    errors.push(`quality report generated 16x16 top5 floor ${top516Floor} < 1`);
  }
  if (targetDistance16Cap <= 0) {
    errors.push("quality report did not require generated 16x16 target-distance cap");
  }
  if (!matchingModelName) {
    errors.push("quality report generated seal quality has no matching product-floor model");
  }
  if (matchingTop516 < Math.max(top516Floor, 1)) {
    errors.push(`quality report matching generated 16x16 top5 ${matchingTop516} < ${Math.max(top516Floor, 1)}`);
  }
  if (matchingMeanTargetDistance16 === null) {
    errors.push("quality report matching generated 16x16 target distance is missing");
  } else if (targetDistance16Cap > 0 && matchingMeanTargetDistance16 > targetDistance16Cap) {
    errors.push(
      `quality report matching generated 16x16 target distance ${matchingMeanTargetDistance16} > ${targetDistance16Cap}`,
    );
  }
  return {
    matching_model: matchingModelName,
    min_top5_16_per_mille: top516Floor,
    max_mean_target_distance_16_q8: targetDistance16Cap,
    matching_top5_16_per_mille: matchingTop516,
    matching_mean_target_distance_16_q8: matchingMeanTargetDistance16,
  };
}

function checkClassRetrievalHeadEvidence(retrievalEval, errors) {
  if (!retrievalEval || typeof retrievalEval !== "object" || Array.isArray(retrievalEval)) {
    errors.push("quality report retrieval head eval evidence is missing");
    return classRetrievalHeadSummary({});
  }
  if (retrievalEval.ok !== true) {
    errors.push("quality report retrieval head eval is not ok");
  }
  const head = retrievalEval.class_retrieval_head || {};
  const summary = classRetrievalHeadSummary(head);
  if (summary.present !== true) {
    errors.push("quality report class retrieval head is not present");
  }
  if (summary.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    errors.push(
      `quality report class retrieval head schema ${JSON.stringify(summary.schema)} != nsrl.solomon_v2_retrieval_head.v1`,
    );
  }
  if (summary.hash_verified !== true) {
    errors.push("quality report class retrieval head hash is not verified");
  }
  if (summary.hash_matches_eval !== true) {
    errors.push("quality report class retrieval head hash does not match retrieval eval");
  }
  if (summary.feature_count <= 0) {
    errors.push(`quality report class retrieval head feature_count ${summary.feature_count} <= 0`);
  }
  if (summary.labels !== 72) {
    errors.push(`quality report class retrieval head labels ${summary.labels} != 72`);
  }
  if (summary.text_head !== true) {
    errors.push("quality report class retrieval text_head is not valid");
  }
  if (summary.image_head !== true) {
    errors.push("quality report class retrieval image_head is not valid");
  }
  if (summary.text_nonzero_weights <= 0) {
    errors.push(`quality report class retrieval text_nonzero_weights ${summary.text_nonzero_weights} <= 0`);
  }
  if (summary.image_nonzero_weights <= 0) {
    errors.push(`quality report class retrieval image_nonzero_weights ${summary.image_nonzero_weights} <= 0`);
  }
  return summary;
}

function classRetrievalHeadSummary(head) {
  return {
    present: head.present === true,
    schema: String(head.schema || ""),
    model_hash: String(head.model_hash || ""),
    hash_verified: head.hash_verified === true,
    hash_matches_eval: head.hash_matches_eval === true,
    feature_count: Number(head.feature_count || 0),
    labels: Number(head.labels || 0),
    text_head: head.text_head === true,
    image_head: head.image_head === true,
    text_nonzero_weights: Number(head.text_nonzero_weights || 0),
    image_nonzero_weights: Number(head.image_nonzero_weights || 0),
  };
}

function checkPromotedSmallArchitecture(architecture, promoted, errors) {
  const dModel = Number(architecture.d_model || 0);
  const heads = Number(architecture.heads || 0);
  const headDim = Number(architecture.head_dim || 0);
  const hiddenDim = Number(architecture.hidden_dim || 0);
  const transformerLayers = Number(architecture.transformer_layers || 0);
  const contextSeqLen = Number(architecture.context_seq_len || 0);
  if (dModel !== PROMOTED_SMALL_PROFILE.dModel) {
    errors.push(`quality report d_model ${architecture.d_model || 0} != ${PROMOTED_SMALL_PROFILE.dModel}`);
  }
  if (heads !== PROMOTED_SMALL_PROFILE.heads) {
    errors.push(`quality report heads ${architecture.heads || 0} != ${PROMOTED_SMALL_PROFILE.heads}`);
  }
  if (headDim !== PROMOTED_SMALL_PROFILE.headDim) {
    errors.push(`quality report head_dim ${architecture.head_dim || 0} != ${PROMOTED_SMALL_PROFILE.headDim}`);
  }
  if (
    hiddenDim < PROMOTED_SMALL_PROFILE.hiddenDimMin ||
    hiddenDim > PROMOTED_SMALL_PROFILE.hiddenDimMax
  ) {
    errors.push(
      `quality report hidden_dim ${architecture.hidden_dim || 0} outside ${PROMOTED_SMALL_PROFILE.hiddenDimMin}-${PROMOTED_SMALL_PROFILE.hiddenDimMax}`,
    );
  }
  if (
    transformerLayers < PROMOTED_SMALL_PROFILE.transformerLayersMin ||
    transformerLayers > PROMOTED_SMALL_PROFILE.transformerLayersMax
  ) {
    errors.push(
      `quality report transformer_layers ${architecture.transformer_layers || 0} outside ${PROMOTED_SMALL_PROFILE.transformerLayersMin}-${PROMOTED_SMALL_PROFILE.transformerLayersMax}`,
    );
  }
  if (
    contextSeqLen < PROMOTED_SMALL_PROFILE.contextSeqLenMin ||
    contextSeqLen > PROMOTED_SMALL_PROFILE.contextSeqLenMax
  ) {
    errors.push(
      `quality report context_seq_len ${architecture.context_seq_len || 0} outside ${PROMOTED_SMALL_PROFILE.contextSeqLenMin}-${PROMOTED_SMALL_PROFILE.contextSeqLenMax}`,
    );
  }
  if (promoted.ok !== true) {
    errors.push("quality report promoted small profile is not ok");
  }
}

function checkGroundedCorpusEvidence(grounded, floor, errors) {
  const sourceFloor = Math.max(Number(floor.min_grounded_source_overlap_tokens || 0), 2);
  const attributeFloor = Math.max(Number(floor.min_grounded_attribute_source_overlap_tokens || 0), 8);
  const sourcePlaceholderCeiling = Number(floor.max_grounded_source_placeholder_rows || 0);
  const attributeGenericRankCeiling = Number(floor.max_grounded_attribute_generic_rank_rows || 0);
  if (!grounded || typeof grounded !== "object" || Array.isArray(grounded)) {
    errors.push("quality report grounded corpus evidence is missing");
    return groundedCorpusSummary({}, sourceFloor, attributeFloor);
  }
  if (grounded.present !== true) {
    errors.push("quality report grounded corpus is not present");
  }
  if (grounded.ok !== true) {
    errors.push("quality report grounded corpus is not ok");
  }
  if (grounded.require_source_provenance !== true) {
    errors.push("quality report grounded corpus source provenance is not required");
  }
  if (grounded.require_name_source_explain !== true) {
    errors.push("quality report grounded corpus name-source explain prompt is not required");
  }
  if (grounded.require_description_source_image !== true) {
    errors.push("quality report grounded corpus description-source image prompt is not required");
  }
  if (grounded.require_image_attribute_generic_prompt !== true) {
    errors.push("quality report grounded corpus image-attribute generic prompt is not required");
  }
  if (Number(grounded.expect_spirits || 0) !== 72) {
    errors.push(`quality report grounded corpus expect_spirits ${grounded.expect_spirits || 0} != 72`);
  }
  if (Number(grounded.min_source_overlap_tokens || 0) < sourceFloor) {
    errors.push(`quality report grounded corpus source overlap floor ${grounded.min_source_overlap_tokens || 0} < ${sourceFloor}`);
  }
  if (Number(grounded.min_attribute_source_overlap_tokens || 0) < attributeFloor) {
    errors.push(
      `quality report grounded corpus attribute source overlap floor ${grounded.min_attribute_source_overlap_tokens || 0} < ${attributeFloor}`,
    );
  }
  if (sourcePlaceholderCeiling !== 0) {
    errors.push(`quality report grounded source placeholder ceiling ${sourcePlaceholderCeiling} != 0`);
  }
  if (attributeGenericRankCeiling !== 0) {
    errors.push(`quality report grounded attribute generic rank ceiling ${attributeGenericRankCeiling} != 0`);
  }
  if (Number(grounded.max_source_placeholder_rows || 0) !== 0) {
    errors.push(`quality report grounded corpus max_source_placeholder_rows ${grounded.max_source_placeholder_rows || 0} != 0`);
  }
  if (Number(grounded.max_attribute_generic_rank_rows || 0) !== 0) {
    errors.push(
      `quality report grounded corpus max_attribute_generic_rank_rows ${grounded.max_attribute_generic_rank_rows || 0} != 0`,
    );
  }
  const sourceTasks = Array.isArray(grounded.source_text_tasks) ? grounded.source_text_tasks.map(String) : [];
  for (const task of REQUIRED_GROUNDED_SOURCE_TASKS) {
    if (!sourceTasks.includes(task)) {
      errors.push(`quality report grounded corpus missing source task ${task}`);
    }
    checkGroundedCorpusTask(grounded.tasks?.[task] || {}, task, sourceFloor, "source", errors);
  }
  const attributeTasks = Array.isArray(grounded.attribute_tasks) ? grounded.attribute_tasks.map(String) : [];
  for (const task of REQUIRED_GROUNDED_ATTRIBUTE_TASKS) {
    if (!attributeTasks.includes(task)) {
      errors.push(`quality report grounded corpus missing attribute task ${task}`);
    }
    checkGroundedCorpusTask(grounded.tasks?.[task] || {}, task, attributeFloor, "attribute", errors);
  }
  return groundedCorpusSummary(grounded, sourceFloor, attributeFloor);
}

function checkGroundedCorpusTask(stats, task, floor, kind, errors) {
  const records = Number(stats.records || 0);
  if (records <= 0) {
    errors.push(`quality report grounded corpus task ${task} has no rows`);
  }
  if (Number(stats.spirits || 0) !== 72) {
    errors.push(`quality report grounded corpus task ${task} spirits ${stats.spirits || 0} != 72`);
  }
  if (Number(stats.min_source_overlap_tokens || 0) < floor) {
    errors.push(`quality report grounded corpus task ${task} source overlap ${stats.min_source_overlap_tokens || 0} < ${floor}`);
  }
  if (Number(stats.source_provenance_rows || 0) !== records) {
    errors.push(
      `quality report grounded corpus task ${task} source provenance rows ${stats.source_provenance_rows || 0} != records ${records}`,
    );
  }
  if (Number(stats.source_provenance_hash_mismatches || 0) !== 0) {
    errors.push(
      `quality report grounded corpus task ${task} source provenance hash mismatches ${stats.source_provenance_hash_mismatches || 0} != 0`,
    );
  }
  if (Number(stats.source_excerpt_hash_mismatches || 0) !== 0) {
    errors.push(
      `quality report grounded corpus task ${task} source excerpt hash mismatches ${stats.source_excerpt_hash_mismatches || 0} != 0`,
    );
  }
  if (kind === "source" && Number(stats.placeholder_rows || 0) !== 0) {
    errors.push(`quality report grounded corpus task ${task} placeholder rows ${stats.placeholder_rows || 0} != 0`);
  }
  if (task === "explain" && Number(stats.name_source_prompt_ok_rows || 0) !== records) {
    errors.push(
      `quality report grounded corpus task explain name-source prompt rows ${stats.name_source_prompt_ok_rows || 0} != records ${records}`,
    );
  }
  if (task === "description-to-image" && Number(stats.description_source_prompt_ok_rows || 0) !== records) {
    errors.push(
      `quality report grounded corpus task description-to-image description-source prompt rows ${stats.description_source_prompt_ok_rows || 0} != records ${records}`,
    );
  }
  if (task === "image-to-attributes" && Number(stats.image_attribute_prompt_ok_rows || 0) !== records) {
    errors.push(
      `quality report grounded corpus task image-to-attributes generic attribute prompt rows ${stats.image_attribute_prompt_ok_rows || 0} != records ${records}`,
    );
  }
  if (kind === "attribute" && Number(stats.generic_attribute_rank_rows || 0) !== 0) {
    errors.push(`quality report grounded corpus task ${task} generic rank rows ${stats.generic_attribute_rank_rows || 0} != 0`);
  }
}

function groundedCorpusSummary(grounded, sourceFloor, attributeFloor) {
  return {
    present: grounded.present === true,
    ok: grounded.ok === true,
    expect_spirits: Number(grounded.expect_spirits || 0),
    min_source_overlap_tokens: Number(grounded.min_source_overlap_tokens || 0),
    min_attribute_source_overlap_tokens: Number(grounded.min_attribute_source_overlap_tokens || 0),
    required_min_source_overlap_tokens: sourceFloor,
    required_min_attribute_source_overlap_tokens: attributeFloor,
    max_source_placeholder_rows: Number(grounded.max_source_placeholder_rows || 0),
    max_attribute_generic_rank_rows: Number(grounded.max_attribute_generic_rank_rows || 0),
    require_name_source_explain: grounded.require_name_source_explain === true,
    require_description_source_image: grounded.require_description_source_image === true,
    require_image_attribute_generic_prompt: grounded.require_image_attribute_generic_prompt === true,
    source_text_tasks: Array.isArray(grounded.source_text_tasks) ? grounded.source_text_tasks.map(String) : [],
    attribute_tasks: Array.isArray(grounded.attribute_tasks) ? grounded.attribute_tasks.map(String) : [],
  };
}

function checkSourceGroundingEvidence(sourceGrounding, errors) {
  if (sourceGrounding.present !== true) {
    errors.push("quality report confidence source grounding identity evidence is not present");
  }
  if (sourceGrounding.grounded_corpus_present !== true) {
    errors.push("quality report confidence grounded corpus evidence is not present");
  }
  if (sourceGrounding.grounded_corpus_ok !== true) {
    errors.push("quality report confidence grounded corpus evidence is not ok");
  }
  if (sourceGrounding.grounded_source_provenance !== true) {
    errors.push("quality report confidence source grounding provenance is not ok");
  }
  if (sourceGrounding.grounded_name_source_explain !== true) {
    errors.push("quality report confidence source grounding name-source explain is not ok");
  }
  if (sourceGrounding.grounded_description_source_image !== true) {
    errors.push("quality report confidence source grounding description-source image is not ok");
  }
  if (sourceGrounding.grounded_image_attribute_generic_prompt !== true) {
    errors.push("quality report confidence source grounding image-attribute prompt is not ok");
  }
  const sourceTasks = Array.isArray(sourceGrounding.grounded_source_tasks)
    ? sourceGrounding.grounded_source_tasks.map(String)
    : [];
  for (const task of REQUIRED_GROUNDED_SOURCE_TASKS) {
    if (!sourceTasks.includes(task)) {
      errors.push(`quality report confidence source grounding missing source task ${task}`);
    }
  }
  const attributeTasks = Array.isArray(sourceGrounding.grounded_attribute_tasks)
    ? sourceGrounding.grounded_attribute_tasks.map(String)
    : [];
  for (const task of REQUIRED_GROUNDED_ATTRIBUTE_TASKS) {
    if (!attributeTasks.includes(task)) {
      errors.push(`quality report confidence source grounding missing attribute task ${task}`);
    }
  }
  for (const [key, label] of [
    ["text_queries_have_source_text", "text queries source text"],
    ["image_queries_have_source_text", "image queries source text"],
    ["sample_queries_have_source_text", "sample queries source text"],
    ["sample_source_text_evidence", "sample source text evidence"],
    ["generated_text_source_evidence", "generated text source evidence"],
    ["generated_text_image_agreement", "generated text/image source agreement"],
    ["expected_generated_text_agreement", "expected generated text agreement"],
  ]) {
    if (sourceGrounding[key] !== true) {
      errors.push(`quality report confidence ${label} is not true`);
    }
  }
  if (Number(sourceGrounding.min_source_text_chars || 0) <= 0) {
    errors.push(`quality report confidence min_source_text_chars ${sourceGrounding.min_source_text_chars || 0} <= 0`);
  }
}

function checkSymbolicImageTokenEvidence(symbolic, errors) {
  if (symbolic.required !== true) {
    errors.push("quality report symbolic image-token evidence is not required");
  }
  if (symbolic.ok !== true) {
    errors.push("quality report symbolic image-token evidence is not ok");
  }
  const requiredChannels = Array.isArray(symbolic.required_channels)
    ? symbolic.required_channels.map((channel) => String(channel))
    : [];
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    if (!requiredChannels.includes(channel)) {
      errors.push(`quality report symbolic image-token required_channels missing ${channel}`);
    }
  }

  const corpus = symbolic.corpus || {};
  if (corpus.present !== true) {
    errors.push("quality report symbolic image-token corpus evidence is not present");
  }
  if (corpus.ok !== true) {
    errors.push("quality report symbolic image-token corpus evidence is not ok");
  }
  if (Number(corpus.checked_records || 0) <= 0) {
    errors.push(`quality report symbolic image-token corpus checked_records ${corpus.checked_records || 0} <= 0`);
  }
  checkImageChannelMarkerSummary(corpus.by_channel || {}, "corpus", errors);

  const curriculum = symbolic.curriculum || {};
  if (curriculum.present !== true || curriculum.required !== true) {
    errors.push("quality report symbolic image-token curriculum evidence is not present and required");
  }
  if (curriculum.ok !== true) {
    errors.push("quality report symbolic image-token curriculum evidence is not ok");
  }
  if (Number(curriculum.stage_count || 0) <= 0) {
    errors.push(`quality report symbolic image-token curriculum stage_count ${curriculum.stage_count || 0} <= 0`);
  }
  const stages = Array.isArray(curriculum.stages) ? curriculum.stages : [];
  if (stages.length === 0) {
    errors.push("quality report symbolic image-token curriculum stages are missing");
  }
  for (const stage of stages) {
    const stageLabel = `curriculum stage ${stage.stage_name || stage.index || "unknown"}`;
    if (stage.ok !== true) {
      errors.push(`quality report symbolic image-token ${stageLabel} is not ok`);
    }
    if (Number(stage.checked_records || 0) <= 0) {
      errors.push(`quality report symbolic image-token ${stageLabel} checked_records ${stage.checked_records || 0} <= 0`);
    }
    checkImageChannelMarkerSummary(stage.by_channel || {}, stageLabel, errors);
  }
}

function checkImageChannelMarkerSummary(byChannel, label, errors) {
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    const summary = byChannel[channel] || {};
    const prefix = `quality report symbolic image-token ${label} channel ${channel}`;
    if (Number(summary.checked_records || 0) <= 0) {
      errors.push(`${prefix} checked_records ${summary.checked_records || 0} <= 0`);
    }
    if (Number(summary.found_markers || 0) <= 0) {
      errors.push(`${prefix} found_markers ${summary.found_markers || 0} <= 0`);
    }
    for (const field of [
      "missing_channel_markers",
      "short_channel_payloads",
      "bad_channel_payloads",
      "channel_order_mismatches",
    ]) {
      if (Number(summary[field] || 0) !== 0) {
        errors.push(`${prefix} ${field} ${summary[field] || 0} != 0`);
      }
    }
  }
}

function checkCorpusContractEvidence(corpus, errors) {
  if (!corpus || typeof corpus !== "object" || Array.isArray(corpus)) {
    errors.push("quality report corpus contract evidence is missing");
    return corpusContractSummary({});
  }
  if (corpus.present !== true) {
    errors.push("quality report corpus contract is not present");
  }
  if (corpus.ok !== true) {
    errors.push("quality report corpus contract is not ok");
  }
  if (corpus.required_corpus_version !== "v2") {
    errors.push(`quality report corpus contract required_corpus_version ${JSON.stringify(corpus.required_corpus_version || "")} != "v2"`);
  }
  if (corpus.required_image_token_profile !== "symbolic16") {
    errors.push(
      `quality report corpus contract required_image_token_profile ${JSON.stringify(corpus.required_image_token_profile || "")} != "symbolic16"`,
    );
  }
  if (corpus.require_image_channel_token_stats !== true) {
    errors.push("quality report corpus contract image channel token stats are not required");
  }
  if (Number(corpus.min_image_channel_distinct_bins || 0) < 2) {
    errors.push(`quality report corpus contract min_image_channel_distinct_bins ${corpus.min_image_channel_distinct_bins || 0} < 2`);
  }
  const requiredChannels = Array.isArray(corpus.required_image_token_channels)
    ? corpus.required_image_token_channels.map(String)
    : [];
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    if (!requiredChannels.includes(channel)) {
      errors.push(`quality report corpus contract required image channel missing ${channel}`);
    }
  }
  checkCorpusManifestSummary(corpus.manifest_summary || {}, errors);
  checkCorpusExamplesSummary(corpus.examples_summary || {}, errors);
  checkCorpusTaskMarkerIntegrity(corpus.task_marker_integrity || {}, errors);
  checkCorpusTaskModalityIntegrity(corpus.task_modality_integrity || {}, errors);
  checkCorpusImageChannelMarkerIntegrity(corpus.image_channel_marker_integrity || {}, errors);
  return corpusContractSummary(corpus);
}

function checkCorpusManifestSummary(manifest, errors) {
  if (manifest.corpus_version !== "v2") {
    errors.push(`quality report corpus manifest corpus_version ${JSON.stringify(manifest.corpus_version || "")} != "v2"`);
  }
  if (manifest.image_token_profile !== "symbolic16") {
    errors.push(`quality report corpus manifest image_token_profile ${JSON.stringify(manifest.image_token_profile || "")} != "symbolic16"`);
  }
  const channels = Array.isArray(manifest.image_token_channels) ? manifest.image_token_channels.map(String) : [];
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    if (!channels.includes(channel)) {
      errors.push(`quality report corpus manifest image_token_channels missing ${channel}`);
    }
  }
  const expectedRecords = Number(manifest.examples || manifest.rows || 72);
  const stats = manifest.image_token_channel_stats || {};
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    const row = stats[channel] || {};
    const prefix = `quality report corpus manifest image_token_channel_stats ${channel}`;
    const records = Number(row.records || 0);
    if (records < 72 || (expectedRecords > 0 && records !== expectedRecords)) {
      errors.push(`${prefix} records ${records} != ${expectedRecords > 0 ? expectedRecords : 72}`);
    }
    if (Number(row.tokens_per_record || 0) !== 256) {
      errors.push(`${prefix} tokens_per_record ${row.tokens_per_record || 0} != 256`);
    }
    if (Number(row.active_records || 0) !== records) {
      errors.push(`${prefix} active_records ${row.active_records || 0} != records ${records}`);
    }
    if (Number(row.multi_bin_records || 0) !== records) {
      errors.push(`${prefix} multi_bin_records ${row.multi_bin_records || 0} != records ${records}`);
    }
    if (Number(row.nonzero_tokens || 0) <= 0) {
      errors.push(`${prefix} nonzero_tokens ${row.nonzero_tokens || 0} <= 0`);
    }
    if (Number(row.distinct_bins || 0) < 2) {
      errors.push(`${prefix} distinct_bins ${row.distinct_bins || 0} < 2`);
    }
    if (Number(row.max_bin || 0) <= 0) {
      errors.push(`${prefix} max_bin ${row.max_bin || 0} <= 0`);
    }
    if (records > 0 && Number(row.unique_record_hashes || 0) !== records) {
      errors.push(`${prefix} unique_record_hashes ${row.unique_record_hashes || 0} != records ${records}`);
    }
    if (Number(row.duplicate_record_hashes || 0) !== 0) {
      errors.push(`${prefix} duplicate_record_hashes ${row.duplicate_record_hashes || 0} != 0`);
    }
  }
}

function checkCorpusExamplesSummary(examples, errors) {
  const v2Records = Number(examples.v2_records || 0);
  if (v2Records <= 0) {
    errors.push(`quality report corpus examples v2_records ${v2Records} <= 0`);
  }
  if (Number(examples.distinct_spirits || 0) !== 72) {
    errors.push(`quality report corpus examples distinct_spirits ${examples.distinct_spirits || 0} != 72`);
  }
  if (Number(examples.missing_image_token_profile || 0) !== 0) {
    errors.push(`quality report corpus examples missing_image_token_profile ${examples.missing_image_token_profile || 0} != 0`);
  }
  if (Number(examples.missing_image_token_channels || 0) !== 0) {
    errors.push(`quality report corpus examples missing_image_token_channels ${examples.missing_image_token_channels || 0} != 0`);
  }
  if (Number(examples.image_token_profiles?.symbolic16 || 0) !== v2Records) {
    errors.push(`quality report corpus examples symbolic16 rows ${examples.image_token_profiles?.symbolic16 || 0} != v2_records ${v2Records}`);
  }
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    const rows = Number(examples.required_channel_rows?.[channel] || 0);
    if (rows !== v2Records) {
      errors.push(`quality report corpus examples image channel ${channel} rows ${rows} != v2_records ${v2Records}`);
    }
  }
  if (Array.isArray(examples.coverage_errors) && examples.coverage_errors.length > 0) {
    errors.push("quality report corpus examples coverage errors are not empty");
  }
  for (const task of REQUIRED_CORPUS_TASKS) {
    const taskSummary = examples.tasks?.[task];
    if (!taskSummary) {
      errors.push(`quality report corpus examples missing task ${task}`);
      continue;
    }
    if (Number(taskSummary.spirits || 0) !== 72) {
      errors.push(`quality report corpus examples task ${task} spirits ${taskSummary.spirits || 0} != 72`);
    }
  }
  const match = examples.tasks?.match || {};
  for (const label of ["yes", "no"]) {
    if (Number(match.labels?.[label]?.spirits || 0) !== 72) {
      errors.push(`quality report corpus examples match ${label} spirits ${match.labels?.[label]?.spirits || 0} != 72`);
    }
  }
  for (const role of ["image", "prompt"]) {
    if (Number(match.labels?.no?.roles?.[role]?.spirits || 0) !== 72) {
      errors.push(`quality report corpus examples match no ${role} role spirits ${match.labels?.no?.roles?.[role]?.spirits || 0} != 72`);
    }
  }
}

function checkCorpusTaskMarkerIntegrity(integrity, errors) {
  checkCorpusIntegrityBase(integrity, "task_marker_integrity", errors);
  for (const field of ["hash_mismatches", "marker_mismatches", "out_of_bounds", "missing_offsets"]) {
    if (Number(integrity[field] || 0) !== 0) {
      errors.push(`quality report corpus task_marker_integrity ${field} ${integrity[field] || 0} != 0`);
    }
  }
  for (const task of REQUIRED_CORPUS_TASKS) {
    if (Number(integrity.by_task?.[task]?.checked_records || 0) <= 0) {
      errors.push(`quality report corpus task_marker_integrity task ${task} checked_records ${integrity.by_task?.[task]?.checked_records || 0} <= 0`);
    }
  }
}

function checkCorpusTaskModalityIntegrity(integrity, errors) {
  checkCorpusIntegrityBase(integrity, "task_modality_integrity", errors);
  for (const field of ["missing_offsets", "out_of_bounds", "modality_mismatches"]) {
    if (Number(integrity[field] || 0) !== 0) {
      errors.push(`quality report corpus task_modality_integrity ${field} ${integrity[field] || 0} != 0`);
    }
  }
  for (const task of REQUIRED_CORPUS_TASKS) {
    if (Number(integrity.by_task?.[task]?.checked_records || 0) <= 0) {
      errors.push(`quality report corpus task_modality_integrity task ${task} checked_records ${integrity.by_task?.[task]?.checked_records || 0} <= 0`);
    }
  }
}

function checkCorpusImageChannelMarkerIntegrity(integrity, errors) {
  checkCorpusIntegrityBase(integrity, "image_channel_marker_integrity", errors);
  const channels = Array.isArray(integrity.required_channels) ? integrity.required_channels.map(String) : [];
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    if (!channels.includes(channel)) {
      errors.push(`quality report corpus image_channel_marker_integrity required channel missing ${channel}`);
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
    if (Number(integrity[field] || 0) !== 0) {
      errors.push(`quality report corpus image_channel_marker_integrity ${field} ${integrity[field] || 0} != 0`);
    }
  }
  for (const channel of REQUIRED_IMAGE_TOKEN_CHANNELS) {
    const summary = integrity.by_channel?.[channel] || {};
    if (Number(summary.checked_records || 0) <= 0) {
      errors.push(`quality report corpus image_channel_marker_integrity channel ${channel} checked_records ${summary.checked_records || 0} <= 0`);
    }
    if (Number(summary.found_markers || 0) <= 0) {
      errors.push(`quality report corpus image_channel_marker_integrity channel ${channel} found_markers ${summary.found_markers || 0} <= 0`);
    }
  }
}

function checkCorpusIntegrityBase(integrity, label, errors) {
  if (!integrity || typeof integrity !== "object" || Array.isArray(integrity)) {
    errors.push(`quality report corpus ${label} is missing`);
    return;
  }
  if (integrity.present !== true) {
    errors.push(`quality report corpus ${label} is not present`);
  }
  if (integrity.ok !== true) {
    errors.push(`quality report corpus ${label} is not ok`);
  }
  if (Number(integrity.checked_records || 0) <= 0) {
    errors.push(`quality report corpus ${label} checked_records ${integrity.checked_records || 0} <= 0`);
  }
}

function corpusContractSummary(corpus) {
  const manifest = corpus.manifest_summary || {};
  const examples = corpus.examples_summary || {};
  return {
    present: corpus.present === true,
    ok: corpus.ok === true,
    corpus_version: manifest.corpus_version || "",
    image_token_profile: manifest.image_token_profile || "",
    image_token_channels: Array.isArray(manifest.image_token_channels) ? manifest.image_token_channels.map(String) : [],
    v2_records: Number(examples.v2_records || 0),
    distinct_spirits: Number(examples.distinct_spirits || 0),
    task_marker_checked_records: Number(corpus.task_marker_integrity?.checked_records || 0),
    image_channel_checked_records: Number(corpus.image_channel_marker_integrity?.checked_records || 0),
  };
}

function checkDenoiseBridgeEvidence(denoise, generationBridge, minUniqueTargets, errors) {
  if (denoise.present !== true || denoise.ok !== true) {
    errors.push("quality report denoise bridge artifact is not present and ok");
  }
  const denoisePairs = Number(denoise.pairs || 0);
  if (denoisePairs <= 0) {
    errors.push(`quality report denoise bridge pairs ${denoise.pairs || 0} <= 0`);
  }
  checkDenoiseBridgeAggregate(denoise, "denoise bridge", minUniqueTargets, denoisePairs, errors);
  checkDenoiseModelProvenance(
    denoise.denoise_model_provenance,
    "denoise bridge",
    denoisePairs,
    errors,
  );
  checkDenoiseSampleBindingProvenance(
    denoise.sample_binding_provenance,
    "denoise bridge",
    denoisePairs,
    errors,
  );
  checkDenoiseOutputProvenance(
    denoise.output_provenance,
    "denoise bridge",
    denoisePairs,
    errors,
  );

  if (generationBridge.present !== true) {
    errors.push("quality report confidence generation bridge evidence is not present");
  }
  if (generationBridge.required !== true) {
    errors.push("quality report confidence generation bridge is not required");
  }
  if (generationBridge.output_identity_required !== true) {
    errors.push("quality report confidence generation bridge output identity is not required");
  }
  const generationPairs = Number(generationBridge.pairs || 0);
  if (generationPairs <= 0) {
    errors.push(`quality report confidence generation bridge pairs ${generationBridge.pairs || 0} <= 0`);
  }
  if (denoisePairs > 0 && generationPairs > 0 && generationPairs !== denoisePairs) {
    errors.push(`quality report confidence generation bridge pairs ${generationPairs} != denoise bridge pairs ${denoisePairs}`);
  }
  checkDenoiseBridgeAggregate(generationBridge, "confidence generation bridge", minUniqueTargets, generationPairs, errors);
  checkDenoiseModelProvenance(
    generationBridge.denoise_model_provenance,
    "confidence generation bridge",
    generationPairs,
    errors,
  );
  checkDenoiseSampleBindingProvenance(
    generationBridge.sample_binding_provenance,
    "confidence generation bridge",
    generationPairs,
    errors,
  );
  checkDenoiseOutputProvenance(
    generationBridge.output_provenance,
    "confidence generation bridge",
    generationPairs,
    errors,
  );
  checkDenoiseGenerationBridgeParity(denoise, generationBridge, errors);
}

function checkDenoiseBridgeAggregate(record, label, minUniqueTargets, pairs, errors) {
  if (record.trace_integrity_ok !== true) {
    errors.push(`quality report ${label} trace integrity is not ok`);
  }
  if (finiteNumberOrNull(record.min_output_signature_distance) === null) {
    errors.push(`quality report ${label} missing min_output_signature_distance`);
  }
  const minOutputInkRange = finiteNumberOrNull(record.min_output_ink_range);
  if (minOutputInkRange === null) {
    errors.push(`quality report ${label} missing min_output_ink_range`);
  } else if (minOutputInkRange <= 0) {
    errors.push(`quality report ${label} min_output_ink_range ${minOutputInkRange} <= 0`);
  }
  if (record.output_image_to_text_identification !== true) {
    errors.push(`quality report ${label} output image-to-text identification is not true`);
  }
  if (record.target_coverage_ok !== true) {
    errors.push(`quality report ${label} target coverage is not ok`);
  }
  const configuredUniqueTargets = finiteNumberOrNull(record.min_unique_targets);
  if (configuredUniqueTargets === null) {
    errors.push(`quality report ${label} missing min_unique_targets`);
  } else if (configuredUniqueTargets < minUniqueTargets) {
    errors.push(`quality report ${label} min_unique_targets ${configuredUniqueTargets} < ${minUniqueTargets}`);
  }
  const expectedUniqueTargets = finiteNumberOrNull(record.expected_unique_targets);
  if (expectedUniqueTargets === null) {
    errors.push(`quality report ${label} missing expected_unique_targets`);
  } else if (expectedUniqueTargets < minUniqueTargets) {
    errors.push(`quality report ${label} unique targets ${expectedUniqueTargets} < ${minUniqueTargets}`);
  }
  const expectedIds = Array.isArray(record.expected_spirit_ids)
    ? record.expected_spirit_ids.map(Number).filter(Number.isFinite)
    : [];
  if (pairs > 0 && expectedIds.length !== pairs) {
    errors.push(`quality report ${label} expected_spirit_ids length ${expectedIds.length} != pairs ${pairs}`);
  }
  const uniqueIds = Array.isArray(record.unique_expected_spirit_ids)
    ? record.unique_expected_spirit_ids.map(Number).filter(Number.isFinite)
    : [];
  if (expectedUniqueTargets !== null && uniqueIds.length !== expectedUniqueTargets) {
    errors.push(`quality report ${label} unique_expected_spirit_ids length ${uniqueIds.length} != expected_unique_targets ${expectedUniqueTargets}`);
  }
  const minOutputRetrievalMargin = finiteNumberOrNull(record.min_output_retrieval_image_margin);
  if (minOutputRetrievalMargin === null) {
    errors.push(`quality report ${label} missing min_output_retrieval_image_margin`);
  } else if (minOutputRetrievalMargin <= 0) {
    errors.push(`quality report ${label} min_output_retrieval_image_margin ${minOutputRetrievalMargin} <= 0`);
  }
}

function checkDenoiseModelProvenance(provenance, label, expectedResults, errors) {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    errors.push(`quality report ${label} denoiser model provenance is missing`);
    return;
  }
  if (provenance.ok !== true) {
    errors.push(`quality report ${label} denoiser model provenance is not ok`);
  }
  const resultCount = Number(provenance.result_count || 0);
  if (expectedResults > 0 && resultCount !== expectedResults) {
    errors.push(`quality report ${label} denoiser model result_count ${resultCount} != pairs ${expectedResults}`);
  }
  if (Number(provenance.resolved_result_model_count || 0) !== resultCount) {
    errors.push(
      `quality report ${label} denoiser resolved_result_model_count ${provenance.resolved_result_model_count || 0} != result_count ${resultCount}`,
    );
  }
  for (const field of [
    "missing_model_refs",
    "missing_model_hashes",
    "unresolved_models",
    "hash_mismatches",
  ]) {
    if (Number(provenance[field] || 0) !== 0) {
      errors.push(`quality report ${label} denoiser ${field} ${provenance[field] || 0} != 0`);
    }
  }
  const recomputedHashes = Array.isArray(provenance.unique_recomputed_hashes)
    ? provenance.unique_recomputed_hashes.filter(Boolean)
    : [];
  if (recomputedHashes.length !== 1) {
    errors.push(`quality report ${label} denoiser unique_recomputed_hashes length ${recomputedHashes.length} != 1`);
  }
  if (provenance.denoise_model_consistent !== true) {
    errors.push(`quality report ${label} denoiser model is not consistent`);
  }
  if (
    provenance.denoise_model_hash &&
    recomputedHashes.length === 1 &&
    provenance.denoise_model_hash !== recomputedHashes[0]
  ) {
    errors.push(
      `quality report ${label} denoiser model hash ${provenance.denoise_model_hash} != recomputed ${recomputedHashes[0]}`,
    );
  }
}

function checkDenoiseSampleBindingProvenance(provenance, label, expectedResults, errors) {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    errors.push(`quality report ${label} sample binding provenance is missing`);
    return;
  }
  const sampleCount = Number(provenance.sample_count || 0);
  const bridgeResultCount = Number(provenance.bridge_result_count || 0);
  const matchedPlans = Number(provenance.matched_attention_plans || 0);
  if (sampleCount <= 0) {
    errors.push(`quality report ${label} sample binding sample_count ${sampleCount} <= 0`);
  }
  if (expectedResults > 0 && bridgeResultCount !== expectedResults) {
    errors.push(`quality report ${label} sample binding bridge_result_count ${bridgeResultCount} != pairs ${expectedResults}`);
  }
  if (bridgeResultCount <= 0) {
    errors.push(`quality report ${label} sample binding bridge_result_count ${bridgeResultCount} <= 0`);
  }
  if (matchedPlans !== bridgeResultCount || matchedPlans <= 0) {
    errors.push(`quality report ${label} sample binding matched_attention_plans ${matchedPlans} != bridge_result_count ${bridgeResultCount}`);
  }
  for (const field of [
    "missing_attention_plans",
    "prompt_mismatches",
    "identity_mismatches",
    "output_identity_mismatches",
  ]) {
    if (Number(provenance[field] || 0) !== 0) {
      errors.push(`quality report ${label} sample binding ${field} ${provenance[field] || 0} != 0`);
    }
  }
  const matches = Array.isArray(provenance.matches) ? provenance.matches : [];
  if (matches.length === 0) {
    errors.push(`quality report ${label} sample binding matches are missing`);
  }
  for (const [index, match] of matches.entries()) {
    for (const [field, description] of [
      ["plan_match", "plan"],
      ["prompt_match", "prompt"],
      ["identity_match", "identity"],
      ["output_identity_match", "output identity"],
    ]) {
      if (match?.[field] !== true) {
        errors.push(`quality report ${label} sample binding match ${index} ${description} is not true`);
      }
    }
  }
}

function checkDenoiseOutputProvenance(provenance, label, expectedResults, errors) {
  if (!provenance || typeof provenance !== "object" || Array.isArray(provenance)) {
    errors.push(`quality report ${label} output provenance is missing`);
    return;
  }
  if (provenance.ok !== true) {
    errors.push(`quality report ${label} output provenance is not ok`);
  }
  if (provenance.required !== true) {
    errors.push(`quality report ${label} output provenance is not required`);
  }
  const resultCount = Number(provenance.result_count || 0);
  if (expectedResults > 0 && resultCount !== expectedResults) {
    errors.push(`quality report ${label} output provenance result_count ${resultCount} != pairs ${expectedResults}`);
  }
  if (provenance.retrieval_required !== true) {
    errors.push(`quality report ${label} output provenance retrieval is not required`);
  }
  if (Number(provenance.scored_results || 0) !== resultCount || resultCount <= 0) {
    errors.push(`quality report ${label} output provenance scored_results ${provenance.scored_results || 0} != result_count ${resultCount}`);
  }
  for (const field of [
    "missing_attention_plans",
    "missing_raw_samples",
    "invalid_raw_samples",
    "result_mismatches",
    "detail_mismatches",
  ]) {
    if (Number(provenance[field] || 0) !== 0) {
      errors.push(`quality report ${label} output provenance ${field} ${provenance[field] || 0} != 0`);
    }
  }
  const aggregateMismatches = Array.isArray(provenance.aggregate_mismatches)
    ? provenance.aggregate_mismatches
    : [];
  if (aggregateMismatches.length !== 0) {
    errors.push(`quality report ${label} output provenance aggregate mismatches are not empty`);
  }
  if (provenance.retrieval_head_present !== true) {
    errors.push(`quality report ${label} output provenance retrieval head is not present`);
  }
  if (provenance.invalid_retrieval_head === true) {
    errors.push(`quality report ${label} output provenance retrieval head is invalid`);
  }
  if (provenance.retrieval_head_hash_verified !== true) {
    errors.push(`quality report ${label} output provenance retrieval head hash is not verified`);
  }
  if (provenance.retrieval_head_hash_match !== true) {
    errors.push(`quality report ${label} output provenance retrieval head hash does not match retrieval eval`);
  }
  if (provenance.config_hash_match !== true) {
    errors.push(`quality report ${label} output provenance config retrieval head hash does not match retrieval eval`);
  }
  if (Number(provenance.retrieval_head_feature_count || 0) <= 0) {
    errors.push(`quality report ${label} output provenance retrieval head feature_count ${provenance.retrieval_head_feature_count || 0} <= 0`);
  }
  if (Number(provenance.retrieval_head_label_count || 0) < 72) {
    errors.push(`quality report ${label} output provenance retrieval head label_count ${provenance.retrieval_head_label_count || 0} < 72`);
  }
  const recomputed = provenance.recomputed || {};
  if (finiteNumberOrNull(recomputed.min_output_signature_distance) === null) {
    errors.push(`quality report ${label} output provenance missing recomputed min_output_signature_distance`);
  }
  const recomputedInkRange = finiteNumberOrNull(recomputed.min_output_ink_range);
  if (recomputedInkRange === null) {
    errors.push(`quality report ${label} output provenance missing recomputed min_output_ink_range`);
  } else if (recomputedInkRange <= 0) {
    errors.push(`quality report ${label} output provenance recomputed min_output_ink_range ${recomputedInkRange} <= 0`);
  }
  if (recomputed.output_image_to_text_identification !== true) {
    errors.push(`quality report ${label} output provenance recomputed output image-to-text identification is not true`);
  }
  const recomputedRetrievalMargin = finiteNumberOrNull(recomputed.min_output_retrieval_image_margin);
  if (recomputedRetrievalMargin === null) {
    errors.push(`quality report ${label} output provenance missing recomputed min_output_retrieval_image_margin`);
  } else if (recomputedRetrievalMargin <= 0) {
    errors.push(`quality report ${label} output provenance recomputed min_output_retrieval_image_margin ${recomputedRetrievalMargin} <= 0`);
  }
}

function checkDenoiseGenerationBridgeParity(denoise, generationBridge, errors) {
  const numericFields = [
    "pairs",
    "min_output_signature_distance",
    "min_output_ink_range",
    "min_output_retrieval_image_margin",
    "min_unique_targets",
    "expected_unique_targets",
  ];
  for (const field of numericFields) {
    const denoiseValue = finiteNumberOrNull(denoise[field]);
    const generationValue = finiteNumberOrNull(generationBridge[field]);
    if (denoiseValue !== null && generationValue !== null && denoiseValue !== generationValue) {
      errors.push(`quality report confidence generation bridge ${field} ${generationValue} != denoise bridge ${denoiseValue}`);
    }
  }
  for (const field of ["trace_integrity_ok", "output_image_to_text_identification"]) {
    if (denoise[field] === true && generationBridge[field] !== true) {
      errors.push(`quality report confidence generation bridge ${field} does not mirror denoise bridge`);
    }
  }
  const denoiseHash = String(denoise.denoise_model_hash || "");
  const generationHash = String(generationBridge.denoise_model_hash || "");
  if (denoiseHash && generationHash && denoiseHash !== generationHash) {
    errors.push(`quality report confidence generation bridge denoise_model_hash ${generationHash} != denoise bridge ${denoiseHash}`);
  }
  for (const field of ["expected_spirit_ids", "unique_expected_spirit_ids"]) {
    const denoiseValues = Array.isArray(denoise[field]) ? denoise[field].map(Number) : [];
    const generationValues = Array.isArray(generationBridge[field]) ? generationBridge[field].map(Number) : [];
    if (!sameNumberArray(denoiseValues, generationValues)) {
      errors.push(`quality report confidence generation bridge ${field} does not mirror denoise bridge`);
    }
  }
}

function sameNumberArray(left, right) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((value, index) => value === right[index]);
}

function finiteNumberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function checkDirectionalNativeEvalEvidence(directional, errors) {
  const groups = directional.groups || {};
  for (const group of REQUIRED_DIRECTIONAL_GROUPS) {
    const item = groups[group];
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      errors.push(`quality report directional native eval missing group ${group}`);
      continue;
    }
    if (item.ok !== true) {
      errors.push(`quality report directional native eval group ${group} is not ok`);
    }
    if (Array.isArray(item.tasks) && item.tasks.length === 0) {
      errors.push(`quality report directional native eval group ${group} has no tasks`);
    }
    if (item.targets !== undefined && Number(item.targets || 0) <= 0) {
      errors.push(`quality report directional native eval group ${group} targets ${item.targets || 0} <= 0`);
    }
  }
}

function checkConfidenceTraceSpine(confidence, errors) {
  const text = confidence.text_binding || {};
  requireCompleteConfidenceMetric(text.known_prompts, "known prompt text retrieval", errors);
  requireCompleteConfidenceMetric(text.heldout_prompts, "held-out prompt text retrieval", errors);
  requireCompleteConfidenceMetric(text.identity_bindings?.total, "identity binding text retrieval", errors);
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    requireCompleteConfidenceMetric(
      text.identity_bindings?.by_kind?.[kind],
      `identity binding ${kind} retrieval`,
      errors,
    );
  }

  const image = confidence.image_binding || {};
  requireCompleteConfidenceMetric(image.image_to_text, "image-to-text retrieval", errors);
  for (const task of REQUIRED_IMAGE_BINDING_TASKS) {
    requireCompleteConfidenceMetric(image.image_tasks?.[task], `${task} retrieval`, errors);
  }
  if (image.sample_image_to_text_identification !== true) {
    errors.push("quality report confidence sample image-to-text identification is not true");
  }
  if (Number(image.min_image_to_text_margin || 0) <= 0) {
    errors.push(`quality report confidence min_image_to_text_margin ${image.min_image_to_text_margin || 0} <= 0`);
  }
  if (Number(image.min_retrieval_image_margin || 0) <= 0) {
    errors.push(`quality report confidence min_retrieval_image_margin ${image.min_retrieval_image_margin || 0} <= 0`);
  }

  const forward = confidence.forward_image_plan || {};
  for (const task of REQUIRED_FORWARD_IMAGE_PLAN_TASKS) {
    requireNativeConfidenceMetric(forward.tasks?.[task], `${task} native image-plan eval`, errors);
  }
  const nativeTaskEval = confidence.native_task_eval || {};
  for (const task of REQUIRED_NATIVE_EVAL_TASKS) {
    requireNativeConfidenceMetric(nativeTaskEval.tasks?.[task], `${task} native task eval`, errors);
  }
  if (!REQUIRED_NATIVE_EVAL_TASKS.includes(String(nativeTaskEval.weakest_top5?.task || ""))) {
    errors.push("quality report confidence native task weakest_top5 is missing or unknown");
  }
  if (!REQUIRED_NATIVE_EVAL_TASKS.includes(String(nativeTaskEval.weakest_margin?.task || ""))) {
    errors.push("quality report confidence native task weakest_margin is missing or unknown");
  }

  const cross = confidence.cross_modal_agreement || {};
  requireCompleteConfidenceMetric(cross.match_yes, "match yes agreement", errors);
  requireCompleteConfidenceMetric(cross.match_no, "match no disagreement", errors);
  requireCompleteConfidenceMetric(cross.wrong_image_negatives, "wrong-image hard negatives", errors);
  requireCompleteConfidenceMetric(cross.wrong_prompt_negatives, "wrong-prompt hard negatives", errors);
  for (const [key, label] of [
    ["text_image_agreement", "text/image agreement"],
    ["generated_text_image_agreement", "generated text/image agreement"],
    ["generated_text_identification", "generated text identification"],
    ["signature_retrieval_agreement", "signature/retrieval agreement"],
  ]) {
    if (cross[key] !== true) {
      errors.push(`quality report confidence ${label} is not true`);
    }
  }
  for (const [key, label] of [
    ["min_signature_margin", "min_signature_margin"],
    ["min_retrieval_text_margin", "min_retrieval_text_margin"],
    ["min_generated_text_margin", "min_generated_text_margin"],
  ]) {
    if (Number(cross[key] || 0) <= 0) {
      errors.push(`quality report confidence ${label} ${cross[key] || 0} <= 0`);
    }
  }
}

function requireCompleteConfidenceMetric(metric, label, errors) {
  const count = Number(metric?.count || 0);
  const top1 = Number(metric?.top1 || 0);
  const minMargin = Number(metric?.min_margin || 0);
  if (count < 72) {
    errors.push(`quality report confidence ${label} count ${count} < 72`);
  }
  if (top1 !== count) {
    errors.push(`quality report confidence ${label} top1 ${top1} != count ${count}`);
  }
  if (minMargin <= 0) {
    errors.push(`quality report confidence ${label} min_margin ${metric?.min_margin || 0} <= 0`);
  }
}

function requireNativeConfidenceMetric(metric, label, errors) {
  const targets = Number(metric?.targets || 0);
  const invalidContexts = Number(metric?.invalid_contexts || 0);
  const top5 = Number(metric?.top5_accuracy_per_mille || 0);
  if (targets < 72) {
    errors.push(`quality report confidence ${label} targets ${targets} < 72`);
  }
  if (invalidContexts !== 0) {
    errors.push(`quality report confidence ${label} invalid_contexts ${invalidContexts} != 0`);
  }
  if (top5 < 1) {
    errors.push(`quality report confidence ${label} top5 ${top5} < 1`);
  }
}

function summarizeConfidenceTraceSpine(confidence) {
  return {
    known_prompt_count: Number(confidence.text_binding?.known_prompts?.count || 0),
    heldout_prompt_count: Number(confidence.text_binding?.heldout_prompts?.count || 0),
    identity_binding_count: Number(confidence.text_binding?.identity_bindings?.total?.count || 0),
    image_to_text_count: Number(confidence.image_binding?.image_to_text?.count || 0),
    text_to_image_targets: Number(confidence.forward_image_plan?.tasks?.["text-to-image"]?.targets || 0),
    description_to_image_targets: Number(confidence.forward_image_plan?.tasks?.["description-to-image"]?.targets || 0),
    native_task_count: Object.keys(confidence.native_task_eval?.tasks || {}).length,
    native_weakest_top5_task: confidence.native_task_eval?.weakest_top5?.task || "",
    native_weakest_top5_per_mille:
      confidence.native_task_eval?.weakest_top5?.top5_accuracy_per_mille ?? null,
    native_weakest_margin_task: confidence.native_task_eval?.weakest_margin?.task || "",
    native_weakest_margin_q8:
      confidence.native_task_eval?.weakest_margin?.mean_target_margin_q8 ?? null,
    match_yes_count: Number(confidence.cross_modal_agreement?.match_yes?.count || 0),
    match_no_count: Number(confidence.cross_modal_agreement?.match_no?.count || 0),
    wrong_image_negative_count: Number(confidence.cross_modal_agreement?.wrong_image_negatives?.count || 0),
    wrong_prompt_negative_count: Number(confidence.cross_modal_agreement?.wrong_prompt_negatives?.count || 0),
  };
}

function summarizeSymbolicImageTokenEvidence(symbolic) {
  return {
    required: symbolic.required === true,
    ok: symbolic.ok === true,
    required_channels: Array.isArray(symbolic.required_channels)
      ? symbolic.required_channels.map((channel) => String(channel))
      : [],
    corpus_checked_records: Number(symbolic.corpus?.checked_records || 0),
    curriculum_stage_count: Number(symbolic.curriculum?.stage_count || 0),
  };
}

function checkPromotionBundle(config) {
  const errors = [];
  const manifestDir = path.dirname(path.resolve(config.promotionPath));
  const rows = readPromotionManifest(config.promotionPath);
  const byArtifact = new Map();
  const artifacts = {};

  for (const row of rows) {
    if (config.requireProduct && row.product !== config.requireProduct) {
      errors.push(`${config.promotionPath}:${row.line}: product ${JSON.stringify(row.product)} != ${JSON.stringify(config.requireProduct)}`);
    }
    if (!["0", "1"].includes(row.required)) {
      errors.push(`${config.promotionPath}:${row.line}: required ${JSON.stringify(row.required)} must be 0 or 1`);
    }
    if (byArtifact.has(row.artifact)) {
      errors.push(`${config.promotionPath}:${row.line}: duplicate artifact ${row.artifact}`);
    }
    const resolvedPath = resolveArtifactPath(row.path, manifestDir);
    const exists = Boolean(resolvedPath && fs.existsSync(resolvedPath));
    byArtifact.set(row.artifact, { ...row, resolved_path: resolvedPath, exists });
    artifacts[row.artifact] = {
      stage: row.stage,
      path: row.path,
      resolved_path: resolvedPath,
      required: row.required === "1",
      exists,
    };
    if (config.requireExistingArtifacts && row.required === "1" && !exists) {
      errors.push(`required promotion artifact ${row.artifact} is missing at ${row.path}`);
    }
  }

  if (config.requireRequiredArtifacts) {
    for (const artifact of REQUIRED_ARTIFACTS) {
      const row = byArtifact.get(artifact);
      if (!row) {
        errors.push(`promotion manifest missing ${artifact}`);
      } else if (row.required !== "1") {
        errors.push(`promotion artifact ${artifact} is not marked required`);
      }
    }
  }

  const qualityReportPath = config.qualityReportPath
    ? resolveArtifactPath(config.qualityReportPath, manifestDir)
    : byArtifact.get("quality_report")?.resolved_path || "";
  let quality = {
    path: qualityReportPath,
    ok: false,
    ready_flags: {},
    architecture: {},
  };
  if (!qualityReportPath) {
    errors.push("quality report path is missing");
  } else if (!fs.existsSync(qualityReportPath)) {
    errors.push(`quality report ${qualityReportPath} is missing`);
  } else {
    quality = checkQualityReport(readJson(qualityReportPath), qualityReportPath, config, errors);
  }

  return {
    schema: "nsrl.solomon_promotion_bundle_check.v1",
    ok: errors.length === 0,
    promotion: config.promotionPath,
    product: config.requireProduct,
    artifact_count: rows.length,
    artifacts,
    quality,
    errors,
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const report = checkPromotionBundle(config);
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
