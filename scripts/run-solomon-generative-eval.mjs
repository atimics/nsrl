#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const defaults = {
  prompts: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  textIndex: "data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv",
  samplerModel: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch",
  retrievalHead: "",
  requireRetrievalHead: false,
  gold: "data/processed/key-solomon-goetia-latent-v1/gold.tsv",
  outDir: "data/processed/key-solomon-goetia-latent-v1/generative-eval",
  runName: "",
  splitSeed: "solomon-prompt-split-v1",
  partition: "eval",
  evalPermille: 180,
  limit: 8,
  samples: 1,
  candidateMultiplier: 4,
  diversityWeight: 0,
  textWeight: 96,
  passes: 2,
  seed: "solomon-generative-eval-v1",
  latentTarget: "decoded",
  latentModels: [],
};
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

const signatureGrid = 16;
const signatureBins = signatureGrid * signatureGrid;
const fineSignatureGrid = 16;
const fineSignatureBins = fineSignatureGrid * fineSignatureGrid;
const imageSize = 128;
const imageBytes = imageSize * imageSize;
const inkThreshold = 64;
const contentWindow = 16;
const IMAGE_BASE = 144;
const IMAGE_BINS = 16;
const IMAGE_CHANNEL_INK = 11;
const IMAGE_CHANNEL_EDGE = 12;
const IMAGE_CHANNEL_COMPONENT = 13;
const IMAGE_CHANNEL_RADIAL = 14;
const IMAGE_CHANNEL_DIRECTION = 15;
const CHANNEL_NAMES = new Map([
  [IMAGE_CHANNEL_INK, "ink"],
  [IMAGE_CHANNEL_EDGE, "edge"],
  [IMAGE_CHANNEL_COMPONENT, "component"],
  [IMAGE_CHANNEL_RADIAL, "radial"],
  [IMAGE_CHANNEL_DIRECTION, "direction"],
]);
const BITMAP_TRACE_SCHEMA = "nsrl.bitmap_sampler_trace.v1";
const ALLOWED_TARGET_KEYS = new Set([
  "latent_target_source",
  "latent_target_number",
  "latent_target_name",
  "latent_target_score",
  "latent_target_latent_score",
  "latent_target_lexical_score",
  "latent_target_signature",
]);
const FREE_TEXT_TRACE_VALUE_KEYS = new Set([
  "latent_prompt",
  "latent_target_name",
]);
const FORBIDDEN_TRACE_KEY_PATTERNS = [
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
const BROAD_FORBIDDEN_TRACE_VALUE =
  /target[-_\s]*(pixel|pixels|bitmap|image|ink|seal|signature)|ground[-_\s]*truth|oracle|retrieval[-_\s]*hybrid|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess|targetctx/i;
const SOURCE_FORBIDDEN_TRACE_VALUE =
  /\btarget\b|target[-_\s]*(lookup|guidance|source)|retrieval[-_\s]*hybrid|ground[-_\s]*truth|oracle|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess/i;
const stopwords = new Set([
  "a", "about", "after", "again", "all", "also", "an", "and", "any", "are",
  "as", "at", "be", "before", "both", "but", "by", "can", "etc", "for",
  "from", "great", "have", "he", "her", "him", "his", "in", "is", "it",
  "man", "many", "men", "must", "of", "or", "order", "seal", "shall",
  "she", "spirit", "spirits", "the", "this", "thou", "to", "unto", "upon",
  "which", "who", "will", "with",
]);

function usage() {
  console.log(
    [
      "Usage: run-solomon-generative-eval.mjs [options]",
      "",
      "Samples labeled held-out prompts and scores generated image signatures",
      "against the known target signature, plus rank among all target signatures.",
      "",
      "Options:",
      "  --prompts PATH",
      "  --text-index PATH",
      "  --sampler-model PATH",
      "  --retrieval-head PATH",
      "  --require-retrieval-head",
      "  --gold PATH",
      "  --out-dir PATH",
      "  --run-name NAME",
      "  --partition eval|gold|train|all",
      "  --limit N",
      "  --latent-model LABEL=PATH   (repeatable; required class-head latent model)",
      "  --samples N",
      "  --candidate-multiplier N",
      "  --diversity-weight N",
      "  --text-weight N",
      "  --passes N",
      "  --seed TEXT",
      "  --latent-target decoded",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, latentModels: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--prompts") {
      config.prompts = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--sampler-model") {
      config.samplerModel = requireValue(argv, ++index, arg);
    } else if (arg === "--retrieval-head") {
      config.retrievalHead = requireValue(argv, ++index, arg);
    } else if (arg === "--require-retrieval-head") {
      config.requireRetrievalHead = true;
    } else if (arg === "--gold") {
      config.gold = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--run-name") {
      config.runName = requireValue(argv, ++index, arg);
    } else if (arg === "--split-seed") {
      config.splitSeed = requireValue(argv, ++index, arg);
    } else if (arg === "--partition") {
      config.partition = requireValue(argv, ++index, arg);
    } else if (arg === "--eval-permille") {
      config.evalPermille = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--limit") {
      config.limit = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--samples") {
      config.samples = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--candidate-multiplier") {
      config.candidateMultiplier = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--diversity-weight") {
      config.diversityWeight = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--text-weight") {
      config.textWeight = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--passes") {
      config.passes = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--seed") {
      config.seed = requireValue(argv, ++index, arg);
    } else if (arg === "--latent-target") {
      config.latentTarget = requireValue(argv, ++index, arg);
    } else if (arg === "--latent-model") {
      config.latentModels.push(parseLatentSpec(requireValue(argv, ++index, arg)));
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!["eval", "gold", "train", "all"].includes(config.partition)) {
    throw new Error("--partition must be eval, gold, train, or all");
  }
  if (config.latentTarget !== "decoded") {
    throw new Error("--latent-target retrieval was removed; use decoded learned latent targets");
  }
  if (config.evalPermille > 900) {
    throw new Error("--eval-permille must be <= 900");
  }
  if (config.latentModels.length === 0) {
    config.latentModels = defaultLatentModels();
  }
  if (config.latentModels.length === 0) {
    throw new Error("no latent models found; pass --latent-model LABEL=PATH");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePositive(value, flag) {
  if (!/^[0-9]+$/.test(value) || Number(value) === 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function parseLatentSpec(spec) {
  const equals = spec.indexOf("=");
  if (equals <= 0 || equals === spec.length - 1) {
    throw new Error(`invalid --latent-model ${spec}; expected LABEL=PATH`);
  }
  return {
    label: sanitizeSlug(spec.slice(0, equals)),
    path: spec.slice(equals + 1),
  };
}

function defaultLatentModels() {
  return [];
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  requireFile(config.prompts, "--prompts");
  requireFile(config.textIndex, "--text-index");
  requireFile(config.samplerModel, "--sampler-model");
  if (config.requireRetrievalHead && !config.retrievalHead) {
    throw new Error("--require-retrieval-head was set but --retrieval-head was not supplied");
  }
  if (config.retrievalHead) {
    requireFile(config.retrievalHead, "--retrieval-head");
  }
  for (const spec of config.latentModels) {
    requireFile(spec.path, `--latent-model ${spec.label}`);
  }

  const runName = config.runName || `run-${timestamp()}`;
  const runDir = path.join(config.outDir, runName);
  fs.rmSync(runDir, { recursive: true, force: true });
  fs.mkdirSync(runDir, { recursive: true });

  runCommand("build sampler", "cargo", [
    "build",
    "--release",
    "-q",
    "-p",
    "nsrl-train",
    "--bin",
    "nsrl-bitmap-sample",
  ], path.join(runDir, "build.log"));

  const textRows = readTextIndex(config.textIndex);
  const retrievalHead = config.retrievalHead ? readRetrievalHead(config.retrievalHead) : null;
  const promptRows = readPrompts(config.prompts);
  const prompts = selectPrompts(config, promptRows, readGoldHashes(config.gold));
  if (prompts.length === 0) {
    throw new Error(`no prompts selected for partition ${config.partition}`);
  }
  const promptProvenance = {
    promptsHash: fnv64FileHex(config.prompts),
    promptRows: promptRows.length,
    selectedPromptRows: prompts.length,
    selectedPromptEligibleRows: prompts.filter(isGenerativeHeldoutPrompt).length,
    selectedPromptUniqueTargets: uniquePromptTargets(prompts),
    selectedPromptEligibleUniqueTargets: uniquePromptTargets(prompts.filter(isGenerativeHeldoutPrompt)),
    selectedPromptTiers: countBy(prompts, (prompt) => prompt.tier || "unknown"),
    selectedPromptSources: countBy(prompts, (prompt) => prompt.source || "unknown"),
    selectedPromptHash: promptSelectionHash(prompts),
  };
  const latentModelProvenance = config.latentModels.map((latent) => ({
    label: latent.label,
    path: latent.path,
    modelHash: fnv64FileHex(latent.path),
  }));
  const latentModelHashes = Object.fromEntries(
    latentModelProvenance.map((latent) => [latent.label, latent.modelHash]),
  );
  const samplerModelHash = fnv64FileHex(config.samplerModel);

  const sampleRows = [];
  const summaryRows = [];
  for (const latent of config.latentModels) {
    const latentModel = readLatentModel(latent.path);
    const modelOutDir = path.join(runDir, latent.label);
    fs.mkdirSync(modelOutDir, { recursive: true });
    let top1 = 0;
    let top5 = 0;
    let fineTop1 = 0;
    let fineTop5 = 0;
    let pixelTop1 = 0;
    let pixelTop5 = 0;
    let latentTop1 = 0;
    let latentTop5 = 0;
    let retrievalTop1 = 0;
    let retrievalTop5 = 0;
    let rankTotal = 0;
    let fineRankTotal = 0;
    let pixelRankTotal = 0;
    let latentRankTotal = 0;
    let generatedMeanInkTotal = 0;
    let generatedOutsideInkTotal = 0;
    let generatedEdgeInkTotal = 0;
    let gtDistanceTotal = 0;
    let fineGtDistanceTotal = 0;
    let pixelGtDistanceTotal = 0;
    let latentDecodedDistanceTotal = 0;
    let latentDistanceTotal = 0;
    let retrievalRankTotal = 0;
    let minRetrievalMargin = null;
    for (const prompt of prompts) {
      const target = textRows.byNumber.get(prompt.spirit_id);
      if (!target) {
        throw new Error(`missing text-index row for spirit_id ${prompt.spirit_id}`);
      }
      const latentDecoded = latentDecodedMetrics(latentModel, prompt.text, target.signature, textRows.targets);
      const promptSlug = `${String(prompt.index).padStart(3, "0")}-${prompt.prompt_hash}`;
      const promptOutDir = path.join(modelOutDir, promptSlug);
      fs.mkdirSync(promptOutDir, { recursive: true });
      runSampler(config, latent, prompt, promptOutDir);
      const cleanTrace = readCleanBitmapTrace(promptOutDir, imageSize);
      const trace = cleanTrace.trace;
      const sampleBytes = readGeneratedSampleBytes(cleanTrace.rawSamplesPath, imageSize);
      const generated = generatedMetrics(sampleBytes, target.signature, textRows.targets);
      const generatedFine = generatedMetrics(
        sampleBytes,
        target.fineSignature,
        textRows.targets,
        "fineSignature",
        fineSignatureGrid,
      );
      const generatedPixel = generatedImageMetrics(sampleBytes, target.imageBytes, textRows.targets);
      const generatedArtifact = generatedArtifactMetrics(sampleBytes);
      const generatedRetrieval = retrievalHead
        ? generatedRetrievalMetrics(sampleBytes, prompt.spirit_id, retrievalHead)
        : null;
      if (generated.bestRank <= 1) {
        top1 += 1;
      }
      if (generated.bestRank <= 5) {
        top5 += 1;
      }
      if (generatedFine.bestRank <= 1) {
        fineTop1 += 1;
      }
      if (generatedFine.bestRank <= 5) {
        fineTop5 += 1;
      }
      if (generatedPixel.bestRank <= 1) {
        pixelTop1 += 1;
      }
      if (generatedPixel.bestRank <= 5) {
        pixelTop5 += 1;
      }
      if (latentDecoded.rank <= 1) {
        latentTop1 += 1;
      }
      if (latentDecoded.rank <= 5) {
        latentTop5 += 1;
      }
      if (generatedRetrieval) {
        if (generatedRetrieval.bestRank <= 1) {
          retrievalTop1 += 1;
        }
        if (generatedRetrieval.bestRank <= 5) {
          retrievalTop5 += 1;
        }
        retrievalRankTotal += generatedRetrieval.bestRank;
        if (generatedRetrieval.bestMargin !== null) {
          minRetrievalMargin =
            minRetrievalMargin === null
              ? generatedRetrieval.bestMargin
              : Math.min(minRetrievalMargin, generatedRetrieval.bestMargin);
        }
      }
      rankTotal += generated.bestRank;
      fineRankTotal += generatedFine.bestRank;
      pixelRankTotal += generatedPixel.bestRank;
      latentRankTotal += latentDecoded.rank;
      generatedMeanInkTotal += generatedArtifact.meanInkQ8;
      generatedOutsideInkTotal += generatedArtifact.outsideInkQ8;
      generatedEdgeInkTotal += generatedArtifact.edgeInkQ8;
      gtDistanceTotal += generated.bestDistance;
      fineGtDistanceTotal += generatedFine.bestDistance;
      pixelGtDistanceTotal += generatedPixel.bestDistance;
      latentDecodedDistanceTotal += latentDecoded.distance;
      latentDistanceTotal += trace.selected_min_text_distance ?? 0;
      sampleRows.push({
        model: latent.label,
        prompt_hash: prompt.prompt_hash,
        spirit_id: prompt.spirit_id,
        partition: prompt.partition,
        tier: prompt.tier,
        source: prompt.source,
        generated_rank: generated.bestRank,
        generated_target_distance: generated.bestDistance,
        generated_rank_16: generatedFine.bestRank,
        generated_target_distance_16: generatedFine.bestDistance,
        generated_rank_px: generatedPixel.bestRank,
        generated_target_distance_px: generatedPixel.bestDistance,
        generated_mean_ink_q8: generatedArtifact.meanInkQ8,
        generated_outside_ink_q8: generatedArtifact.outsideInkQ8,
        generated_edge_ink_q8: generatedArtifact.edgeInkQ8,
        latent_decoded_rank: latentDecoded.rank,
        latent_decoded_target_distance: latentDecoded.distance,
        generated_retrieval_rank: generatedRetrieval?.bestRank ?? "",
        generated_retrieval_margin: generatedRetrieval?.bestMargin ?? "",
        generated_retrieval_top1_spirit_id: generatedRetrieval?.top1SpiritId ?? "",
        generated_retrieval_top1_name: generatedRetrieval?.top1Name ?? "",
        generated_retrieval_identity: generatedRetrieval ? (generatedRetrieval.bestRank === 1 ? 1 : 0) : "",
        sampler_target_source: trace.latent_target_source ?? "",
        sampler_target_number: trace.latent_target_number ?? 0,
        sampler_target_name: trace.latent_target_name ?? "",
        latent_target_distance: trace.selected_min_text_distance ?? 0,
        samples: config.samples,
        candidate_multiplier: config.candidateMultiplier,
        text_weight: config.textWeight,
        selected_mean_wash_penalty_q8: trace.selected_mean_wash_penalty_q8 ?? 0,
        prompt: prompt.text,
        out_dir: promptOutDir,
      });
    }
    summaryRows.push({
      model: latent.label,
      latent_model: latent.path,
      latent_model_hash: latentModelHashes[latent.label] || "",
      prompts: prompts.length,
      top1,
      top5,
      top1_per_mille: Math.floor((top1 * 1000) / prompts.length),
      top5_per_mille: Math.floor((top5 * 1000) / prompts.length),
      top1_16: fineTop1,
      top5_16: fineTop5,
      top1_16_per_mille: Math.floor((fineTop1 * 1000) / prompts.length),
      top5_16_per_mille: Math.floor((fineTop5 * 1000) / prompts.length),
      top1_px: pixelTop1,
      top5_px: pixelTop5,
      top1_px_per_mille: Math.floor((pixelTop1 * 1000) / prompts.length),
      top5_px_per_mille: Math.floor((pixelTop5 * 1000) / prompts.length),
      latent_top1: latentTop1,
      latent_top5: latentTop5,
      latent_top1_per_mille: Math.floor((latentTop1 * 1000) / prompts.length),
      latent_top5_per_mille: Math.floor((latentTop5 * 1000) / prompts.length),
      generated_retrieval_top1: retrievalHead ? retrievalTop1 : "",
      generated_retrieval_top5: retrievalHead ? retrievalTop5 : "",
      generated_retrieval_top1_per_mille: retrievalHead ? Math.floor((retrievalTop1 * 1000) / prompts.length) : "",
      generated_retrieval_top5_per_mille: retrievalHead ? Math.floor((retrievalTop5 * 1000) / prompts.length) : "",
      mean_generated_retrieval_rank_q8: retrievalHead ? Math.floor((retrievalRankTotal * 256) / prompts.length) : "",
      min_generated_retrieval_margin: retrievalHead ? minRetrievalMargin ?? "" : "",
      mean_rank_q8: Math.floor((rankTotal * 256) / prompts.length),
      mean_rank_16_q8: Math.floor((fineRankTotal * 256) / prompts.length),
      mean_rank_px_q8: Math.floor((pixelRankTotal * 256) / prompts.length),
      mean_latent_rank_q8: Math.floor((latentRankTotal * 256) / prompts.length),
      mean_generated_ink_q8: Math.floor(generatedMeanInkTotal / prompts.length),
      mean_generated_outside_ink_q8: Math.floor(generatedOutsideInkTotal / prompts.length),
      mean_generated_edge_ink_q8: Math.floor(generatedEdgeInkTotal / prompts.length),
      mean_generated_target_distance_q8: Math.floor((gtDistanceTotal * 256) / prompts.length),
      mean_generated_target_distance_16_q8: Math.floor((fineGtDistanceTotal * 256) / prompts.length),
      mean_generated_target_distance_px_q8: Math.floor((pixelGtDistanceTotal * 256) / prompts.length),
      mean_latent_decoded_target_distance_q8: Math.floor((latentDecodedDistanceTotal * 256) / prompts.length),
      mean_latent_target_distance_q8: Math.floor((latentDistanceTotal * 256) / prompts.length),
      text_weight: config.textWeight,
      selected_mean_wash_penalty_q8: maxSampleValue(sampleRows, "selected_mean_wash_penalty_q8", latent.label),
    });
  }

  writeTsv(path.join(runDir, "samples.tsv"), sampleRows, [
    "model",
    "prompt_hash",
    "spirit_id",
    "partition",
    "tier",
    "source",
    "generated_rank",
    "generated_target_distance",
    "generated_rank_16",
    "generated_target_distance_16",
    "generated_rank_px",
    "generated_target_distance_px",
    "generated_mean_ink_q8",
    "generated_outside_ink_q8",
    "generated_edge_ink_q8",
    "latent_decoded_rank",
    "latent_decoded_target_distance",
    "generated_retrieval_rank",
    "generated_retrieval_margin",
    "generated_retrieval_top1_spirit_id",
    "generated_retrieval_top1_name",
    "generated_retrieval_identity",
    "sampler_target_source",
    "sampler_target_number",
    "sampler_target_name",
    "latent_target_distance",
    "samples",
    "candidate_multiplier",
    "text_weight",
    "selected_mean_wash_penalty_q8",
    "prompt",
    "out_dir",
  ]);
  writeTsv(path.join(runDir, "summary.tsv"), summaryRows, [
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
    "mean_generated_ink_q8",
    "mean_generated_outside_ink_q8",
    "mean_generated_edge_ink_q8",
    "mean_generated_target_distance_q8",
    "mean_generated_target_distance_16_q8",
    "mean_generated_target_distance_px_q8",
    "mean_latent_decoded_target_distance_q8",
    "mean_latent_target_distance_q8",
    "text_weight",
    "selected_mean_wash_penalty_q8",
  ]);
  fs.writeFileSync(
    path.join(runDir, "config.json"),
    `${JSON.stringify(
      {
        ...config,
        runName,
        runDir,
        ...promptProvenance,
        latentModelProvenance,
        latentModelHashes,
        samplerModelHash,
        retrievalHeadModelHash: retrievalHead?.model_hash || "",
        retrievalHeadFeatureCount: Number(retrievalHead?.feature_count || 0),
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  console.log(JSON.stringify({ run_dir: runDir, prompts: prompts.length, models: config.latentModels.length }));
}

function requireFile(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

function readLatentModel(filePath) {
  const bytes = fs.readFileSync(filePath);
  let offset = 0;
  function readBytes(count) {
    if (offset + count > bytes.length) {
      throw new Error(`${filePath} ended unexpectedly`);
    }
    const start = offset;
    offset += count;
    return bytes.subarray(start, offset);
  }
  function readU32() {
    const value = bytes.readUInt32LE(offset);
    offset += 4;
    return value;
  }
  function readI16() {
    const value = bytes.readInt16LE(offset);
    offset += 2;
    return value;
  }
  function readI8Vec(count) {
    const out = [];
    for (let index = 0; index < count; index += 1) {
      out.push(bytes.readInt8(offset));
      offset += 1;
    }
    return out;
  }
  function readI16Vec(count) {
    const out = [];
    for (let index = 0; index < count; index += 1) {
      out.push(readI16());
    }
    return out;
  }

  const magic = readBytes(8).toString("ascii");
  if (magic !== "NSRLLAT1") {
    throw new Error(`${filePath} is not an NSRLLAT1 model`);
  }
  const latentDim = readU32();
  const textFeatureCount = readU32();
  const modelSignatureBins = readU32();
  const textEncoderShift = readU32();
  readU32();
  const decoderShift = readU32();
  const modelSignatureGrid = readU32();
  if (
    latentDim <= 0 ||
    textFeatureCount <= 0 ||
    modelSignatureBins !== signatureBins ||
    modelSignatureGrid !== signatureGrid
  ) {
    throw new Error(`${filePath} has incompatible latent dimensions`);
  }
  const textWeights = readI8Vec(latentDim * textFeatureCount);
  const textBiases = readI16Vec(latentDim);
  readBytes(latentDim * signatureBins);
  readBytes(latentDim * 2);
  const decoderWeights = readI8Vec(signatureBins * latentDim);
  const decoderBiases = readI16Vec(signatureBins);
  if (offset !== bytes.length) {
    throw new Error(`${filePath} has ${bytes.length - offset} trailing bytes`);
  }
  return {
    latentDim,
    textFeatureCount,
    textEncoderShift,
    decoderShift,
    textWeights,
    textBiases,
    decoderWeights,
    decoderBiases,
  };
}

function latentDecodedMetrics(model, prompt, targetSignature, targetRows) {
  const features = textFeatures(prompt, model.textFeatureCount);
  const latent = encodeText(model, features);
  const signature = decodeSignature(model, latent);
  return {
    distance: signatureDistance(signature, targetSignature),
    rank: targetRank(signature, targetSignature, targetRows),
  };
}

function textFeatures(text, featureCount) {
  const features = new Array(featureCount).fill(0);
  const tokens = tokenizeText(text);
  if (tokens.length > 0 && tokens.length <= 4) {
    addTextFeature(features, "whole", tokens.join(" "), 0, 320);
  }
  tokens.forEach((token, position) => {
    addTextFeature(features, "tok", token, position, 72);
    if (tokens[position + 1]) {
      addTextFeature(features, "bi", `${token} ${tokens[position + 1]}`, position, 96);
    }
    if (tokens[position + 1] && tokens[position + 2]) {
      addTextFeature(
        features,
        "tri",
        `${token} ${tokens[position + 1]} ${tokens[position + 2]}`,
        position,
        112,
      );
    }
  });
  const content = contentTokens(tokens);
  if (content.length > 0 && content.length <= 5) {
    addTextFeature(features, "cwhole", content.join(" "), 0, 336);
    addTextFeature(features, "cset", sortedKey(content), 0, 336);
  }
  for (let index = 0; index < content.length; index += 1) {
    addTextFeature(features, "ctok", content[index], index, 128);
    if (content[index + 1]) {
      addTextFeature(features, "cbi", `${content[index]} ${content[index + 1]}`, index, 160);
    }
    if (content[index + 1] && content[index + 2]) {
      addTextFeature(
        features,
        "ctri",
        `${content[index]} ${content[index + 1]} ${content[index + 2]}`,
        index,
        176,
      );
    }
    const windowEnd = Math.min(content.length, index + contentWindow);
    for (let right = index + 1; right < windowEnd; right += 1) {
      addTextFeature(features, "skip2", `${content[index]} ${content[right]}`, index, 176);
      addTextFeature(features, "pair", sortedKey([content[index], content[right]]), index, 192);
      for (let third = right + 1; third < windowEnd; third += 1) {
        addTextFeature(
          features,
          "triple",
          sortedKey([content[index], content[right], content[third]]),
          index,
          192,
        );
      }
    }
  }
  return features;
}

function addTextFeature(features, namespace, text, position, base) {
  if (text.length < 2 || features.length === 0) return;
  const hash = hashParts([namespace, text]);
  const bin = hash % features.length;
  const value = Math.min(384, base + Math.min(28, text.length) * 6 + (position % 7) * 5);
  const signed = (hash >>> 31) === 0 ? value : -value;
  features[bin] = clamp(features[bin] + signed, -511, 511);
}

function encodeText(model, features) {
  const out = new Array(model.latentDim).fill(0);
  for (let dim = 0; dim < model.latentDim; dim += 1) {
    let acc = 0;
    const base = dim * model.textFeatureCount;
    for (let feature = 0; feature < model.textFeatureCount; feature += 1) {
      acc += model.textWeights[base + feature] * features[feature];
    }
    const value = signedRoundShift(acc, model.textEncoderShift) + model.textBiases[dim];
    out[dim] = clamp(value, -511, 511);
  }
  return out;
}

function decodeSignature(model, latent) {
  const out = new Array(signatureBins).fill(0);
  for (let bin = 0; bin < signatureBins; bin += 1) {
    let acc = model.decoderBiases[bin] * (2 ** model.decoderShift);
    const base = bin * model.latentDim;
    for (let dim = 0; dim < model.latentDim; dim += 1) {
      acc += model.decoderWeights[base + dim] * latent[dim];
    }
    out[bin] = clamp(signedRoundShift(acc, model.decoderShift), 0, 255);
  }
  return out;
}

function signedRoundShift(value, shift) {
  if (shift === 0) {
    return value;
  }
  const rounding = 2 ** (shift - 1);
  if (value >= 0) {
    return Math.trunc((value + rounding) / (2 ** shift));
  }
  return -Math.trunc((-value + rounding) / (2 ** shift));
}

function tokenizeText(text) {
  const tokens = [];
  let current = "";
  for (const ch of String(text)) {
    if (/^[A-Za-z0-9]$/.test(ch)) {
      current += ch.toLowerCase();
    } else if (current) {
      const normalized = normalizeToken(current);
      if (normalized.length >= 2) tokens.push(normalized);
      current = "";
    }
  }
  if (current) {
    const normalized = normalizeToken(current);
    if (normalized.length >= 2) tokens.push(normalized);
  }
  return tokens;
}

function normalizeToken(token) {
  if (["teach", "teacher", "teaches", "teacheth", "teaching"].includes(token)) return "teach";
  if (["know", "knows", "known", "knowing", "knoweth", "knowledge"].includes(token)) return "know";
  if (["make", "makes", "maketh", "making"].includes(token)) return "make";
  if (["discover", "discovers", "discovereth", "discovering"].includes(token)) return "discover";
  if (["produce", "produces", "produceth", "producing"].includes(token)) return "produce";
  if (["answer", "answers", "answereth", "answering"].includes(token)) return "answer";
  if (["virtue", "virtues"].includes(token)) return "virtue";
  if (["water", "waters"].includes(token)) return "water";
  if (["rush", "rushing", "rushings"].includes(token)) return "rush";
  if (["herb", "herbs"].includes(token)) return "herb";
  if (["stone", "stones"].includes(token)) return "stone";
  if (["science", "sciences"].includes(token)) return "science";
  if (token.length > 5 && token.endsWith("eth")) return token.slice(0, -3);
  if (token.length > 5 && token.endsWith("ing")) return token.slice(0, -3);
  if (token.length > 4 && token.endsWith("es")) return token.slice(0, -2);
  if (token.length > 3 && token.endsWith("s")) return token.slice(0, -1);
  return token;
}

function contentTokens(tokens) {
  return tokens.filter((token) => token.length >= 3 && !stopwords.has(token));
}

function sortedKey(tokens) {
  return [...tokens].sort().join("\x00");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function runSampler(config, latent, prompt, outDir) {
  const args = [
    "--model",
    config.samplerModel,
    "--latent-model",
    latent.path,
    "--prompt",
    prompt.text,
    "--out-dir",
    outDir,
    "--samples",
    String(config.samples),
    "--candidate-multiplier",
    String(config.candidateMultiplier),
    "--diversity-weight",
    String(config.diversityWeight),
    "--text-weight",
    String(config.textWeight),
    "--passes",
    String(config.passes),
    "--preview-columns",
    String(config.samples),
    "--seed",
    `${config.seed}-${latent.label}-${prompt.prompt_hash}`,
    "--init",
    "noise",
  ];
  runCommand(
    `sample ${latent.label} ${prompt.prompt_hash}`,
    cargoReleaseBin("nsrl-bitmap-sample"),
    args,
    path.join(outDir, "sample.log"),
  );
}

function cargoReleaseBin(name) {
  return path.join(process.env.CARGO_TARGET_DIR || "target", "release", name);
}

function runCommand(label, command, args, logPath) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  const log = [
    `$ ${command} ${args.join(" ")}`,
    result.stdout || "",
    result.stderr || "",
  ].join("\n");
  fs.writeFileSync(logPath, log, "utf8");
  if (result.status !== 0) {
    throw new Error(`${label} failed; see ${logPath}`);
  }
  return result.stdout.trim();
}

function readCleanBitmapTrace(outDir, expectedImageSize) {
  const tracePath = path.join(outDir, "trace.json");
  if (!fs.existsSync(tracePath)) {
    throw new Error(`${outDir} missing trace.json`);
  }
  let trace = null;
  try {
    trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`${tracePath} could not be parsed: ${message}`);
  }
  if (!trace || typeof trace !== "object" || Array.isArray(trace)) {
    throw new Error(`${tracePath} is not a JSON object`);
  }
  if (trace.schema !== BITMAP_TRACE_SCHEMA) {
    throw new Error(`${tracePath} schema ${JSON.stringify(trace.schema || "")} != ${BITMAP_TRACE_SCHEMA}`);
  }
  if (trace.latent_target_source !== "decoded-latent") {
    throw new Error(
      `${tracePath} latent_target_source ${JSON.stringify(trace.latent_target_source || "")} != decoded-latent`,
    );
  }
  const rawSamplesPath = validatedRawSamplesPath(trace, outDir, expectedImageSize, tracePath);
  const violations = [];
  scanTraceObject(trace, [], violations);
  if (violations.length > 0) {
    throw new Error(`${tracePath} ${violations[0].field}: ${violations[0].reason}`);
  }
  return { trace, rawSamplesPath };
}

function validatedRawSamplesPath(trace, outDir, expectedImageSize, tracePath) {
  if (typeof trace.raw_samples !== "string" || trace.raw_samples.length === 0) {
    throw new Error(`${tracePath} raw_samples is missing`);
  }
  const expectedName = `samples.ink${expectedImageSize}.u8`;
  const expectedPath = path.resolve(outDir, expectedName);
  const matched = rawSampleReferenceCandidates(trace.raw_samples, outDir)
    .find((candidate) => sameResolvedPath(candidate, expectedPath));
  if (!matched) {
    throw new Error(`${tracePath} raw_samples must resolve to ${expectedName} in out_dir`);
  }
  return matched;
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

function readGeneratedSampleBytes(filePath, expectedImageSize) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${filePath} missing generated raw sample bytes`);
  }
  const bytes = fs.readFileSync(filePath);
  const expectedImageBytes = expectedImageSize * expectedImageSize;
  if (bytes.length === 0 || bytes.length % expectedImageBytes !== 0) {
    throw new Error(
      `${filePath} byte count ${bytes.length} is not a positive multiple of ${expectedImageBytes}`,
    );
  }
  return bytes;
}

function scanTraceObject(value, keyPath, violations) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      scanTraceObject(value[index], keyPath.concat(String(index)), violations);
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    const field = nextPath.join(".");
    if (isForbiddenTraceKey(key) && !ALLOWED_TARGET_KEYS.has(key)) {
      violations.push({
        field,
        reason: "forbidden target-pixel, oracle, guidance, or cleanup field",
      });
    }
    if (typeof child === "string") {
      const reason = forbiddenTraceValueReason(key, child);
      if (reason) {
        violations.push({ field, reason });
      }
    }
    scanTraceObject(child, nextPath, violations);
  }
}

function isForbiddenTraceKey(key) {
  return FORBIDDEN_TRACE_KEY_PATTERNS.some((pattern) => pattern.test(key));
}

function forbiddenTraceValueReason(key, value) {
  if (isPathLikeTraceKey(key) || isFreeTextTraceValueKey(key)) {
    return "";
  }
  if (BROAD_FORBIDDEN_TRACE_VALUE.test(value)) {
    return "forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value";
  }
  if (isSourceLikeTraceKey(key) && SOURCE_FORBIDDEN_TRACE_VALUE.test(value)) {
    return "forbidden generation source value";
  }
  return "";
}

function isFreeTextTraceValueKey(key) {
  return FREE_TEXT_TRACE_VALUE_KEYS.has(key);
}

function isPathLikeTraceKey(key) {
  return /(path|file|dir|raw[_-]?samples|preview|pgm|model)$/i.test(key);
}

function isSourceLikeTraceKey(key) {
  return /(source|mode|policy|method|strategy|guidance|cleanup|post[_-]?process|postprocess)$/i.test(key);
}

function readTextIndex(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines[0].split("\t");
  const byName = new Map(header.map((name, index) => [name, index]));
  for (const column of ["number", "primary_name", "ink_128_u8", "signature_16x16"]) {
    if (!byName.has(column)) {
      throw new Error(`${filePath} missing ${column}`);
    }
  }
  const imageRoot = textIndexImageRoot(filePath);
  const byNumber = new Map();
  for (const line of lines.slice(1)) {
    if (!line.trim()) {
      continue;
    }
    const fields = line.split("\t");
    const number = Number(fields[byName.get("number")]);
    const imagePath = resolveIndexedImagePath(filePath, imageRoot, fields[byName.get("ink_128_u8")]);
    const imageBytesForRow = readTargetImageBytes(imagePath);
    const row = {
      number,
      name: fields[byName.get("primary_name")] || "",
      signature: parseSignature(fields[byName.get("signature_16x16")], filePath),
      fineSignature: sampleSignature(imageBytesForRow, fineSignatureGrid),
      imageBytes: imageBytesForRow,
      imagePath,
    };
    if (!byNumber.has(number)) {
      byNumber.set(number, row);
    }
  }
  const targets = [...byNumber.values()].sort((left, right) => left.number - right.number);
  return { byNumber, targets };
}

function readRetrievalHead(filePath) {
  const model = JSON.parse(fs.readFileSync(filePath, "utf8"));
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    throw new Error(`${filePath} has unexpected schema ${JSON.stringify(model.schema)}`);
  }
  validateRetrievalHeadLabels(filePath, model.labels);
  return {
    ...model,
    image_head: hydrateHead(model.image_head),
  };
}

function validateRetrievalHeadLabels(filePath, labels) {
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

function hydrateHead(head) {
  return {
    biases: head?.biases || [],
    weights: (head?.weights || []).map((entries) => new Map(entries)),
  };
}

function textIndexImageRoot(filePath) {
  const manifestPath = path.join(path.dirname(filePath), "manifest.json");
  if (!fs.existsSync(manifestPath)) {
    return "";
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!manifest.source_slices_manifest) {
    return "";
  }
  const candidates = [
    manifest.source_slices_manifest,
    path.join(path.dirname(filePath), manifest.source_slices_manifest),
  ].map((candidate) => path.resolve(candidate));
  const sourceManifest = candidates.find((candidate) => fs.existsSync(candidate));
  return sourceManifest ? path.dirname(path.dirname(sourceManifest)) : "";
}

function resolveIndexedImagePath(indexPath, imageRoot, imagePath) {
  const candidates = [
    imagePath,
    path.join(path.dirname(indexPath), imagePath),
    imageRoot ? path.join(imageRoot, imagePath) : "",
  ].filter(Boolean).map((candidate) => path.resolve(candidate));
  const resolved = candidates.find((candidate) => fs.existsSync(candidate));
  if (!resolved) {
    throw new Error(`${indexPath} references missing target image: ${imagePath}`);
  }
  return resolved;
}

function readTargetImageBytes(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length !== imageBytes) {
    throw new Error(`${filePath} has ${bytes.length} bytes, expected ${imageBytes}`);
  }
  return bytes;
}

function readPrompts(filePath) {
  return fs.readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => ({ ...JSON.parse(line), index }));
}

function readGoldHashes(filePath) {
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

function selectPrompts(config, prompts, goldHashes) {
  const candidates = generativeEvalSelectionCandidates(config, prompts, goldHashes);
  return balancedPromptSelection(candidates, config.limit);
}

function generativeEvalSelectionCandidates(config, prompts, goldHashes) {
  const mapped = prompts.map((prompt) => ({ ...prompt, partition: promptPartition(prompt, config, goldHashes) }));
  const candidates = mapped.filter((prompt) => config.partition === "all" || prompt.partition === config.partition);
  if (config.partition === "eval") {
    const eligible = mapped.filter((prompt) => prompt.partition === "eval" && isGenerativeHeldoutPrompt(prompt));
    const requiredTargets = Math.min(Number(config.limit || 0), 72);
    if (uniquePromptTargets(eligible) >= requiredTargets) {
      return sortPromptsForSelection(eligible);
    }
  }
  return sortPromptsForSelection(candidates);
}

function sortPromptsForSelection(prompts) {
  return [...prompts].sort((left, right) => {
    const leftKey = `${left.tier}:${left.prompt_hash}`;
    const rightKey = `${right.tier}:${right.prompt_hash}`;
    return leftKey.localeCompare(rightKey);
  });
}

function balancedPromptSelection(candidates, limit) {
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
  return new Set(
    prompts
      .map((prompt) => Number(prompt.spirit_id || 0))
      .filter((id) => Number.isInteger(id) && id > 0),
  ).size;
}

function countBy(rows, keyFn) {
  const out = {};
  for (const row of rows) {
    const key = String(keyFn(row) || "unknown");
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}

function promptSelectionHash(prompts) {
  const lines = prompts
    .map((prompt) => [
      prompt.prompt_hash || "",
      prompt.spirit_id || "",
      prompt.partition || "",
      prompt.tier || "",
      prompt.source || "",
      prompt.text || "",
    ].join("\t"))
    .sort()
    .join("\n");
  return fnv64TextHex(`${lines}\n`);
}

function promptPartition(prompt, config, goldHashes) {
  if (goldHashes.has(String(prompt.prompt_hash).toLowerCase())) {
    return "gold";
  }
  if (isGenerativeHeldoutPrompt(prompt)) {
    return "eval";
  }
  const bucket = prompt.tier === "tier-cluster-holdout"
    ? hashParts([config.splitSeed, "cluster", prompt.cluster]) % 1000
    : Number(prompt.bucket);
  return bucket < config.evalPermille ? "eval" : "train";
}

function generatedMetrics(
  sampleBytes,
  targetSignature,
  targetRows,
  signatureKey = "signature",
  grid = signatureGrid,
) {
  if (sampleBytes.length % imageBytes !== 0) {
    throw new Error(`sample byte count ${sampleBytes.length} is not a multiple of ${imageBytes}`);
  }
  let bestDistance = Number.MAX_SAFE_INTEGER;
  let bestRank = Number.MAX_SAFE_INTEGER;
  for (let offset = 0; offset < sampleBytes.length; offset += imageBytes) {
    const signature = sampleSignature(sampleBytes.subarray(offset, offset + imageBytes), grid);
    const distance = signatureDistance(signature, targetSignature, grid);
    const rank = targetRank(signature, targetSignature, targetRows, signatureKey, grid);
    if (rank < bestRank || (rank === bestRank && distance < bestDistance)) {
      bestRank = rank;
      bestDistance = distance;
    }
  }
  return { bestDistance, bestRank };
}

function generatedImageMetrics(sampleBytes, targetBytes, targetRows, imageKey = "imageBytes") {
  if (sampleBytes.length % imageBytes !== 0) {
    throw new Error(`sample byte count ${sampleBytes.length} is not a multiple of ${imageBytes}`);
  }
  let bestDistance = Number.MAX_SAFE_INTEGER;
  let bestRank = Number.MAX_SAFE_INTEGER;
  for (let offset = 0; offset < sampleBytes.length; offset += imageBytes) {
    const image = sampleBytes.subarray(offset, offset + imageBytes);
    const distance = imageDistance(image, targetBytes);
    const rank = imageTargetRank(image, targetBytes, targetRows, imageKey);
    if (rank < bestRank || (rank === bestRank && distance < bestDistance)) {
      bestRank = rank;
      bestDistance = distance;
    }
  }
  return { bestDistance, bestRank };
}

function generatedRetrievalMetrics(sampleBytes, targetSpiritId, retrievalHead) {
  if (sampleBytes.length % imageBytes !== 0) {
    throw new Error(`sample byte count ${sampleBytes.length} is not a multiple of ${imageBytes}`);
  }
  let bestRank = Number.MAX_SAFE_INTEGER;
  let bestMargin = null;
  let top1SpiritId = null;
  let top1Name = "";
  for (let offset = 0; offset < sampleBytes.length; offset += imageBytes) {
    const image = sampleBytes.subarray(offset, offset + imageBytes);
    const signature = sampleSignature(image, signatureGrid);
    const ranked = rankRetrievalImage(retrievalHead, signature, retrievalHead.labels.length);
    const rank = ranked.findIndex((row) => row.spirit_id === targetSpiritId) + 1;
    const stats = scoreRankStats(ranked, targetSpiritId);
    const margin = stats.margin;
    if (
      rank > 0 &&
      (rank < bestRank || (rank === bestRank && margin !== null && (bestMargin === null || margin > bestMargin)))
    ) {
      bestRank = rank;
      bestMargin = margin;
      top1SpiritId = ranked[0]?.spirit_id ?? null;
      top1Name = ranked[0]?.primary_name ?? "";
    }
  }
  return {
    bestRank,
    bestMargin,
    top1SpiritId,
    top1Name,
  };
}

function rankRetrievalImage(model, signature, count = 5) {
  const features = imageFeatures(symbolicImageTokens(signature), model.feature_count);
  return rankHead(model.image_head, model.labels, features, count);
}

function rankHead(head, labels, features, count) {
  const ranked = labels.map((label) => ({
    label: label.label,
    spirit_id: label.spirit_id,
    primary_name: label.primary_name,
    score: scoreLabel(head, label.label, features),
  }));
  ranked.sort((left, right) => right.score - left.score || left.spirit_id - right.spirit_id);
  return ranked.slice(0, count);
}

function scoreLabel(head, label, features) {
  let score = head.biases[label] || 0;
  const weights = head.weights[label] || new Map();
  for (const [feature, value] of features) {
    score += (weights.get(feature) || 0) * value;
  }
  return score;
}

function scoreRankStats(ranked, targetSpiritId) {
  const target = ranked.find((row) => row.spirit_id === targetSpiritId) || null;
  const runnerUp = ranked.find((row) => row.spirit_id !== targetSpiritId) || null;
  return {
    score: target?.score ?? null,
    runner_up_score: runnerUp?.score ?? null,
    margin: target && runnerUp ? target.score - runnerUp.score : null,
  };
}

function generatedArtifactMetrics(sampleBytes) {
  if (sampleBytes.length % imageBytes !== 0) {
    throw new Error(`sample byte count ${sampleBytes.length} is not a multiple of ${imageBytes}`);
  }
  const sampleCount = sampleBytes.length / imageBytes;
  const center = Math.floor(imageSize / 2);
  const sealRadius = Math.floor(imageSize / 2) - 2;
  const sealRadius2 = sealRadius * sealRadius;
  let totalInk = 0;
  let outsideInk = 0;
  let outsideCount = 0;
  let edgeInk = 0;
  let edgeCount = 0;
  for (let offset = 0; offset < sampleBytes.length; offset += imageBytes) {
    for (let y = 0; y < imageSize; y += 1) {
      for (let x = 0; x < imageSize; x += 1) {
        const value = sampleBytes[offset + y * imageSize + x];
        totalInk += value;
        const dx = x - center;
        const dy = y - center;
        if (dx * dx + dy * dy > sealRadius2) {
          outsideInk += value;
          outsideCount += 1;
        }
        if (x < 3 || y < 3 || x + 3 >= imageSize || y + 3 >= imageSize) {
          edgeInk += value;
          edgeCount += 1;
        }
      }
    }
  }
  return {
    meanInkQ8: Math.floor((totalInk * 256) / (imageBytes * sampleCount)),
    outsideInkQ8: Math.floor((outsideInk * 256) / outsideCount),
    edgeInkQ8: Math.floor((edgeInk * 256) / edgeCount),
  };
}

function imageTargetRank(image, targetBytes, targetRows, imageKey = "imageBytes") {
  const positiveDistance = imageDistance(image, targetBytes);
  let rank = 1;
  for (const row of targetRows) {
    const rowBytes = row[imageKey];
    if (rowBytes === targetBytes) {
      continue;
    }
    const distance = imageDistance(image, rowBytes);
    if (distance < positiveDistance || (distance === positiveDistance && rowBytes !== targetBytes)) {
      rank += 1;
    }
  }
  return rank;
}

function imageDistance(left, right) {
  let best = Number.MAX_SAFE_INTEGER;
  for (let variant = 0; variant < 8; variant += 1) {
    let distance = 0;
    for (let y = 0; y < imageSize; y += 1) {
      for (let x = 0; x < imageSize; x += 1) {
        const [sx, sy] = transformCoords(imageSize, x, y, variant);
        distance += Math.abs(left[y * imageSize + x] - right[sy * imageSize + sx]);
      }
    }
    best = Math.min(best, distance);
  }
  return best;
}

function sampleSignature(bytes, grid = signatureGrid) {
  const bins = grid * grid;
  const sums = new Array(bins).fill(0);
  const counts = new Array(bins).fill(0);
  for (let y = 0; y < imageSize; y += 1) {
    const binY = Math.floor((y * grid) / imageSize);
    for (let x = 0; x < imageSize; x += 1) {
      const binX = Math.floor((x * grid) / imageSize);
      const bin = binY * grid + binX;
      sums[bin] += bytes[y * imageSize + x];
      counts[bin] += 1;
    }
  }
  return sums.map((sum, index) => Math.floor((sum + Math.floor(counts[index] / 2)) / counts[index]));
}

function symbolicImageTokens(signature) {
  return solomonImage.symbolicImageTokens(signature, symbolicImageOptions());
}

function symbolicImageOptions() {
  return {
    grid: signatureGrid,
    imageBase: IMAGE_BASE,
    imageBins: IMAGE_BINS,
    channelTokens: {
      ink: IMAGE_CHANNEL_INK,
      edge: IMAGE_CHANNEL_EDGE,
      component: IMAGE_CHANNEL_COMPONENT,
      radial: IMAGE_CHANNEL_RADIAL,
      direction: IMAGE_CHANNEL_DIRECTION,
    },
  };
}

function imageFeatures(image, featureCount) {
  const out = new Map();
  let channel = "ink";
  let position = 0;
  for (const token of image) {
    if (CHANNEL_NAMES.has(token)) {
      channel = CHANNEL_NAMES.get(token);
      position = 0;
      addHashedImageFeature(out, featureCount, "channel", channel, 32);
      continue;
    }
    const bin = token >= IMAGE_BASE && token < IMAGE_BASE + IMAGE_BINS ? token - IMAGE_BASE : token;
    addHashedImageFeature(out, featureCount, "ipos", `${channel}:${position}:${bin}`, 64);
    addHashedImageFeature(out, featureCount, "itok", `${channel}:${bin}`, 8);
    if (position % signatureGrid === 0) {
      addHashedImageFeature(out, featureCount, "irow", `${channel}:${Math.floor(position / signatureGrid)}:${bin}`, 6);
    }
    position += 1;
  }
  return [...out.entries()];
}

function addHashedImageFeature(out, featureCount, namespace, value, amount) {
  const hash = fnv32(`${namespace}\xff${value}`);
  const index = hash % featureCount;
  const sign = hash & 0x80000000 ? -1 : 1;
  out.set(index, Math.max(-127, Math.min(127, (out.get(index) || 0) + sign * amount)));
}

function targetRank(
  signature,
  targetSignature,
  targetRows,
  signatureKey = "signature",
  grid = signatureGrid,
) {
  const positiveDistance = signatureDistance(signature, targetSignature, grid);
  let rank = 1;
  for (const row of targetRows) {
    const rowSignature = row[signatureKey];
    if (rowSignature === targetSignature) {
      continue;
    }
    const distance = signatureDistance(signature, rowSignature, grid);
    if (distance < positiveDistance || (distance === positiveDistance && rowSignature !== targetSignature)) {
      rank += 1;
    }
  }
  return rank;
}

function signatureDistance(left, right, grid = signatureGrid) {
  let best = Number.MAX_SAFE_INTEGER;
  for (let variant = 0; variant < 8; variant += 1) {
    let distance = 0;
    for (let y = 0; y < grid; y += 1) {
      for (let x = 0; x < grid; x += 1) {
        const [sx, sy] = transformCoords(grid, x, y, variant);
        distance += Math.abs(left[y * grid + x] - right[sy * grid + sx]);
      }
    }
    best = Math.min(best, distance);
  }
  return best;
}

function transformCoords(size, x, y, variant) {
  const last = size - 1;
  switch (variant % 8) {
    case 0:
      return [x, y];
    case 1:
      return [last - x, y];
    case 2:
      return [x, last - y];
    case 3:
      return [last - x, last - y];
    case 4:
      return [y, x];
    case 5:
      return [last - y, x];
    case 6:
      return [y, last - x];
    default:
      return [last - y, last - x];
  }
}

function parseSignature(text, filePath) {
  const parts = String(text).split(",").map((value) => Number(value));
  if (parts.length !== signatureBins || parts.some((value) => !Number.isFinite(value))) {
    throw new Error(`${filePath} has invalid ${signatureBins}-bin signature`);
  }
  return parts;
}

function writeTsv(filePath, rows, header) {
  const lines = [
    header.join("\t"),
    ...rows.map((row) => header.map((key) => tsvEscape(row[key] ?? "")).join("\t")),
  ];
  fs.writeFileSync(filePath, `${lines.join("\n")}\n`, "utf8");
}

function maxSampleValue(rows, key, model) {
  return rows
    .filter((row) => row.model === model)
    .reduce((best, row) => Math.max(best, Number(row[key] || 0)), 0);
}

function tsvEscape(value) {
  return String(value).replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function sanitizeSlug(value) {
  return String(value).replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "model";
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function fnv32(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index) & 0xff;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash >>> 0;
}

function hashParts(parts) {
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

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function fnv64TextHex(value) {
  return fnv64BytesHex(Buffer.from(String(value), "utf8"));
}

function fnv64BytesHex(bytes) {
  let hash = FNV64_OFFSET;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

main();
