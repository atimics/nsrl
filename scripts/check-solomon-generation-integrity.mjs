#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const defaults = {
  sampleDirs: [],
  traces: [],
  outPath: "",
  expectedLatentTargetSource: "",
};

const SUPPORTED_SCHEMAS = new Set([
  "nsrl.bitmap_sampler_trace.v1",
  "nsrl.solomon_attention_sample_trace.v1",
]);

const ATTENTION_TEXT_PRIOR_SOURCES = new Set(["", "none", "external", "embedded", "embedded_lm"]);
const ATTENTION_IMAGE_PRIOR_SOURCES = new Set(["", "none", "embedded"]);

const ALLOWED_TARGET_KEYS = new Set([
  "latent_target_source",
  "latent_target_number",
  "latent_target_name",
  "latent_target_score",
  "latent_target_latent_score",
  "latent_target_lexical_score",
  "latent_target_signature",
]);

const FREE_TEXT_VALUE_KEYS = new Set([
  "latent_prompt",
  "latent_target_name",
]);

const FORBIDDEN_KEY_PATTERNS = [
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

const BROAD_FORBIDDEN_VALUE = /target[-_\s]*(pixel|pixels|bitmap|image|ink|seal|signature)|ground[-_\s]*truth|oracle|retrieval[-_\s]*hybrid|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess|targetctx/i;
const SOURCE_FORBIDDEN_VALUE = /\btarget\b|target[-_\s]*(lookup|guidance|source)|retrieval[-_\s]*hybrid|ground[-_\s]*truth|oracle|display[-_\s]*cleanup|cleanup|post[-_\s]*process|postprocess/i;

function usage() {
  console.log(
    [
      "Usage: check-solomon-generation-integrity.mjs [--sample-dir PATH...] [--trace PATH...]",
      "",
      "Checks Solomon generation traces for forbidden target-pixel guidance,",
      "oracle target lookup, and display-time cleanup/postprocess sources.",
      "",
      "Options:",
      "  --sample-dir PATH",
      "  --trace PATH",
      "  --expected-latent-target-source VALUE",
      "  --out PATH",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, sampleDirs: [], traces: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--sample-dir") {
      config.sampleDirs.push(requireValue(argv, ++index, arg));
    } else if (arg === "--trace") {
      config.traces.push(requireValue(argv, ++index, arg));
    } else if (arg === "--expected-latent-target-source") {
      config.expectedLatentTargetSource = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.sampleDirs.length === 0 && config.traces.length === 0) {
    throw new Error("--sample-dir or --trace is required");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function collectTracePaths(config) {
  const tracePaths = config.traces.map((tracePath) => ({
    path: tracePath,
    source: "trace",
  }));
  for (const sampleDir of config.sampleDirs) {
    const candidates = ["sample.json", "trace.json"]
      .map((fileName) => path.join(sampleDir, fileName))
      .filter((filePath) => fs.existsSync(filePath));
    if (candidates.length === 0) {
      throw new Error(`${sampleDir} does not contain sample.json or trace.json`);
    }
    for (const candidate of candidates) {
      tracePaths.push({
        path: candidate,
        source: "sample-dir",
      });
    }
  }
  return tracePaths;
}

function readTrace(tracePath) {
  const row = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  if (!row || typeof row !== "object" || Array.isArray(row)) {
    throw new Error(`${tracePath} is not a JSON object`);
  }
  return row;
}

function checkTrace(tracePath, source, trace, config) {
  const record = {
    path: tracePath,
    source,
    schema: String(trace.schema || ""),
    generation_kind: generationKind(trace.schema),
    latent_target_source: stringField(trace.latent_target_source),
    text_prior_source: stringField(trace.text_prior_source),
    image_prior_source: stringField(trace.image_prior_source),
    init_mode: stringField(trace.init_mode),
    raw_samples: "",
    expected_raw_samples: "",
    raw_samples_present: null,
    raw_samples_path_match: null,
    violations: [],
  };

  if (!SUPPORTED_SCHEMAS.has(record.schema)) {
    record.violations.push({
      path: tracePath,
      field: "schema",
      reason: `unsupported generation trace schema ${JSON.stringify(record.schema)}`,
    });
  }

  if (record.schema === "nsrl.bitmap_sampler_trace.v1") {
    checkBitmapSamplerTrace(record, trace, config);
  } else if (record.schema === "nsrl.solomon_attention_sample_trace.v1") {
    checkAttentionSampleTrace(record, trace);
  }

  scanObject(trace, [], record);
  record.ok = record.violations.length === 0;
  return record;
}

function generationKind(schema) {
  if (schema === "nsrl.bitmap_sampler_trace.v1") {
    return "bitmap_sampler";
  }
  if (schema === "nsrl.solomon_attention_sample_trace.v1") {
    return "attention_sample";
  }
  return "unknown";
}

function stringField(value) {
  return typeof value === "string" ? value : "";
}

function resolveBitmapRawSamples(tracePath, trace) {
  const sampleDir = path.dirname(tracePath);
  const imageSize = positiveInteger(trace.image_size) || 128;
  const expectedName = `samples.ink${imageSize}.u8`;
  const expectedPath = path.resolve(sampleDir, expectedName);
  const source = typeof trace.raw_samples === "string" ? trace.raw_samples : "";
  const candidates = source ? rawSampleReferenceCandidates(source, sampleDir) : [];
  const matched = candidates.find((candidate) => sameResolvedPath(candidate, expectedPath)) || "";
  const resolvedPath = matched || candidates[0] || expectedPath;
  return {
    source,
    source_present: source.length > 0,
    path: resolvedPath,
    expected_path: expectedPath,
    expected_name: expectedName,
    path_match: Boolean(matched),
    present: Boolean(resolvedPath && fs.existsSync(resolvedPath)),
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

function checkBitmapSamplerTrace(record, trace, config) {
  if (config.expectedLatentTargetSource && trace.latent_target_source !== config.expectedLatentTargetSource) {
    record.violations.push({
      path: record.path,
      field: "latent_target_source",
      value: compactValue(trace.latent_target_source),
      reason: `expected ${JSON.stringify(config.expectedLatentTargetSource)}`,
    });
  }
  if (Object.hasOwn(trace, "target_source")) {
    record.violations.push({
      path: record.path,
      field: "target_source",
      value: compactValue(trace.target_source),
      reason: "generation traces must use latent_target_source, not target_source",
    });
  }
  const rawSamples = resolveBitmapRawSamples(record.path, trace);
  record.raw_samples = rawSamples.path;
  record.expected_raw_samples = rawSamples.expected_path;
  record.raw_samples_present = rawSamples.present;
  record.raw_samples_path_match = rawSamples.path_match;
  if (!rawSamples.source_present) {
    record.violations.push({
      path: record.path,
      field: "raw_samples",
      reason: "missing generated raw sample reference",
    });
  }
  if (rawSamples.source_present && !rawSamples.path_match) {
    record.violations.push({
      path: record.path,
      field: "raw_samples",
      value: compactValue(rawSamples.source),
      reason: `raw_samples must resolve to ${rawSamples.expected_name} in the sample directory`,
    });
  }
  if (!rawSamples.present) {
    record.violations.push({
      path: record.path,
      field: "raw_samples",
      value: compactValue(rawSamples.path),
      reason: "missing generated raw sample bytes",
    });
  }
}

function checkAttentionSampleTrace(record, trace) {
  if (!ATTENTION_TEXT_PRIOR_SOURCES.has(record.text_prior_source)) {
    record.violations.push({
      path: record.path,
      field: "text_prior_source",
      value: compactValue(record.text_prior_source),
      reason: "unknown attention text prior source",
    });
  }
  if (!ATTENTION_IMAGE_PRIOR_SOURCES.has(record.image_prior_source)) {
    record.violations.push({
      path: record.path,
      field: "image_prior_source",
      value: compactValue(record.image_prior_source),
      reason: "unknown attention image prior source",
    });
  }
}

function scanObject(value, keyPath, record) {
  if (!value || typeof value !== "object") {
    return;
  }
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      scanObject(value[index], keyPath.concat(String(index)), record);
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const nextPath = keyPath.concat(key);
    const field = nextPath.join(".");
    if (isForbiddenKey(key) && !isAllowedTargetKey(key)) {
      record.violations.push({
        path: record.path,
        field,
        value: compactValue(child),
        reason: "forbidden target-pixel, oracle, guidance, or cleanup field",
      });
    }
    if (typeof child === "string") {
      const valueViolation = forbiddenValueReason(key, child);
      if (valueViolation) {
        record.violations.push({
          path: record.path,
          field,
          value: compactValue(child),
          reason: valueViolation,
        });
      }
    }
    scanObject(child, nextPath, record);
  }
}

function isForbiddenKey(key) {
  return FORBIDDEN_KEY_PATTERNS.some((pattern) => pattern.test(key));
}

function isAllowedTargetKey(key) {
  return ALLOWED_TARGET_KEYS.has(key);
}

function forbiddenValueReason(key, value) {
  if (isFreeTextValueKey(key)) {
    return "";
  }
  if (BROAD_FORBIDDEN_VALUE.test(value)) {
    return "forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value";
  }
  if (isSourceLikeKey(key) && SOURCE_FORBIDDEN_VALUE.test(value)) {
    return "forbidden generation source value";
  }
  return "";
}

function isFreeTextValueKey(key) {
  return FREE_TEXT_VALUE_KEYS.has(key);
}

function isSourceLikeKey(key) {
  return /(source|mode|policy|method|strategy|guidance|cleanup|post[_-]?process|postprocess)$/i.test(key);
}

function compactValue(value) {
  if (Array.isArray(value)) {
    return `[array:${value.length}]`;
  }
  if (value && typeof value === "object") {
    return "{object}";
  }
  const text = String(value ?? "");
  return text.length > 96 ? `${text.slice(0, 93)}...` : text;
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
  const tracePaths = collectTracePaths(config);
  const records = tracePaths.map((entry) => checkTrace(entry.path, entry.source, readTrace(entry.path), config));
  const violations = records.flatMap((record) => record.violations);
  const result = {
    schema: "nsrl.solomon_generation_integrity_check.v1",
    ok: violations.length === 0,
    trace_count: records.length,
    expected_latent_target_source: config.expectedLatentTargetSource,
    traces: records,
    violations,
  };
  if (config.outPath) {
    writeJson(config.outPath, result);
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
