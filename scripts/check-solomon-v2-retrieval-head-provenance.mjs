#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const IMAGE_RETRIEVAL_TASKS = [
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
];
const REVERSE_IMAGE_RETRIEVAL_TASKS = [
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
];
const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];

const defaults = {
  evalPath: "",
  retrievalHeadPath: "",
  examplesPath: "",
  tokensPath: "",
  promptsPath: "",
  expectSpirits: 72,
  minFeatureCount: 1,
  minRetrievalMargin: 0,
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-v2-retrieval-head-provenance.mjs --eval PATH --examples PATH --tokens PATH [options]",
      "",
      "Verifies that a Solomon v2 integer retrieval/class head is tied to the",
      "exact promoted corpus bytes and that the serialized model still hashes",
      "to the eval trace that names it.",
      "",
      "Options:",
      "  --retrieval-head PATH      retrieval-head.json path (defaults to eval model)",
      "  --prompts PATH|none        verify held-out prompt file provenance or require none",
      "  --expect-spirits N         expected label count and spirit ids (default: 72)",
      "  --min-feature-count N      minimum sparse feature count (default: 1)",
      "  --min-retrieval-margin N   require every checked metric min_margin >= N",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = () => {
      index += 1;
      if (index >= argv.length) {
        throw new Error(`${arg} requires a value`);
      }
      return argv[index];
    };
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--eval") {
      config.evalPath = value();
    } else if (arg === "--retrieval-head") {
      config.retrievalHeadPath = value();
    } else if (arg === "--examples") {
      config.examplesPath = value();
    } else if (arg === "--tokens") {
      config.tokensPath = value();
    } else if (arg === "--prompts") {
      config.promptsPath = value();
    } else if (arg === "--expect-spirits") {
      config.expectSpirits = parsePositiveInteger(value(), arg);
    } else if (arg === "--min-feature-count") {
      config.minFeatureCount = parseNonNegativeInteger(value(), arg);
    } else if (arg === "--min-retrieval-margin") {
      config.minRetrievalMargin = parseNonNegativeInteger(value(), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  for (const [key, flag] of [
    ["evalPath", "--eval"],
    ["examplesPath", "--examples"],
    ["tokensPath", "--tokens"],
  ]) {
    if (!config[key]) {
      throw new Error(`${flag} is required`);
    }
  }
  return config;
}

function parsePositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const errors = [];
  const trace = readJson(config.evalPath, "retrieval head eval");
  const evalDir = path.dirname(path.resolve(config.evalPath));
  if (trace.schema !== "nsrl.solomon_v2_retrieval_head_eval.v1") {
    errors.push(`retrieval head eval schema ${JSON.stringify(trace.schema)} != nsrl.solomon_v2_retrieval_head_eval.v1`);
  }
  if (trace.ok !== true) {
    errors.push("retrieval head eval ok is not true");
  }
  if (Array.isArray(trace.errors) && trace.errors.length > 0) {
    for (const error of trace.errors) {
      errors.push(`retrieval head eval: ${error}`);
    }
  }

  const corpus = checkEvalCorpusProvenance(trace, config, evalDir, errors);
  const promptProvenance = checkPromptProvenance(trace, config, evalDir, errors);
  const retrievalHead = checkRetrievalHeadArtifact(trace, config, evalDir, corpus, errors);
  const expectedImageTasks = countImageRetrievalTasks(config.examplesPath);
  const metrics = checkRetrievalMetrics(trace, config, expectedImageTasks, errors);
  const report = {
    schema: "nsrl.solomon_v2_retrieval_head_provenance_check.v1",
    ok: errors.length === 0,
    errors,
    eval: path.resolve(config.evalPath),
    model_hash: String(trace.model_hash || ""),
    corpus_provenance: corpus,
    heldout_prompt_provenance: promptProvenance,
    retrieval_head: retrievalHead,
    expected_image_tasks: expectedImageTasks,
    metrics,
  };
  console.log(JSON.stringify(report, null, 2));
  if (errors.length > 0) {
    console.error(`Solomon v2 retrieval head provenance failed with ${errors.length} error(s).`);
    process.exit(1);
  }
}

function checkEvalCorpusProvenance(trace, config, evalDir, errors) {
  const summary = {
    examples: String(trace.examples || ""),
    expected_examples: path.resolve(config.examplesPath),
    examples_match: null,
    examples_hash: String(trace.examples_hash || ""),
    expected_examples_hash: "",
    examples_hash_match: null,
    tokens: String(trace.tokens || ""),
    expected_tokens: path.resolve(config.tokensPath),
    tokens_match: null,
    tokens_hash: String(trace.tokens_hash || ""),
    expected_tokens_hash: "",
    tokens_hash_match: null,
  };
  summary.examples_match = requireSameReferencedPath(
    "retrieval head eval examples",
    summary.examples,
    config.examplesPath,
    evalDir,
    errors,
  );
  summary.tokens_match = requireSameReferencedPath(
    "retrieval head eval tokens",
    summary.tokens,
    config.tokensPath,
    evalDir,
    errors,
  );
  summary.expected_examples_hash = fnv64FileHex(path.resolve(config.examplesPath));
  summary.examples_hash_match = summary.examples_hash === summary.expected_examples_hash;
  if (!summary.examples_hash_match) {
    errors.push(
      `retrieval head eval examples_hash ${summary.examples_hash || "<missing>"} != corpus examples hash ${summary.expected_examples_hash}`,
    );
  }
  summary.expected_tokens_hash = fnv64FileHex(path.resolve(config.tokensPath));
  summary.tokens_hash_match = summary.tokens_hash === summary.expected_tokens_hash;
  if (!summary.tokens_hash_match) {
    errors.push(
      `retrieval head eval tokens_hash ${summary.tokens_hash || "<missing>"} != corpus tokens hash ${summary.expected_tokens_hash}`,
    );
  }
  return summary;
}

function checkPromptProvenance(trace, config, evalDir, errors) {
  const heldoutRows = Number(trace.heldout_prompt_rows || trace.heldout_prompts?.count || 0);
  const prompts = String(trace.prompts || "");
  const promptsHash = String(trace.prompts_hash || "");
  const summary = {
    required: Boolean(config.promptsPath),
    prompts,
    expected_prompts: config.promptsPath,
    prompts_match: null,
    prompts_hash: promptsHash,
    expected_prompts_hash: "",
    prompts_hash_match: null,
    heldout_prompt_rows: heldoutRows,
    prompt_rows_total: 0,
    prompt_rows_counted: 0,
    unique_targets_counted: 0,
    row_count_match: null,
    unique_targets_match: null,
  };
  if (!config.promptsPath) {
    return summary;
  }
  if (config.promptsPath === "none") {
    summary.prompts_match = prompts ? false : true;
    summary.prompts_hash_match = promptsHash ? false : true;
    summary.row_count_match = heldoutRows === 0;
    if (prompts) {
      errors.push(`retrieval head eval prompts ${prompts} present but --prompts none was expected`);
    }
    if (promptsHash) {
      errors.push(`retrieval head eval prompts_hash ${promptsHash} present but --prompts none was expected`);
    }
    if (heldoutRows !== 0) {
      errors.push(`retrieval head held-out prompt rows ${heldoutRows} != 0 for --prompts none`);
    }
    return summary;
  }
  summary.prompts_match = requireSameReferencedPath(
    "retrieval head eval prompts",
    prompts,
    config.promptsPath,
    evalDir,
    errors,
  );
  summary.expected_prompts_hash = fnv64FileHex(path.resolve(config.promptsPath));
  summary.prompts_hash_match = promptsHash === summary.expected_prompts_hash;
  if (!summary.prompts_hash_match) {
    errors.push(
      `retrieval head eval prompts_hash ${promptsHash || "<missing>"} != prompts hash ${summary.expected_prompts_hash}`,
    );
  }
  const counted = countHeldoutPromptRows(config.promptsPath);
  summary.prompt_rows_total = counted.total_rows;
  summary.prompt_rows_counted = counted.eligible_rows;
  summary.unique_targets_counted = counted.unique_targets;
  summary.row_count_match = heldoutRows === summary.prompt_rows_counted;
  if (!summary.row_count_match) {
    errors.push(`retrieval head held-out prompt rows ${heldoutRows} != eligible prompt file rows ${summary.prompt_rows_counted}`);
  }
  const traceUniqueTargets = Number(trace.heldout_prompt_unique_targets || 0);
  summary.unique_targets_match = traceUniqueTargets === summary.unique_targets_counted;
  if (!summary.unique_targets_match) {
    errors.push(
      `retrieval head held-out prompt unique targets ${traceUniqueTargets} != eligible prompt file unique targets ${summary.unique_targets_counted}`,
    );
  }
  if (summary.unique_targets_counted < config.expectSpirits) {
    errors.push(
      `retrieval head held-out prompt unique targets ${summary.unique_targets_counted} < ${config.expectSpirits}`,
    );
  }
  return summary;
}

function checkRetrievalHeadArtifact(trace, config, evalDir, corpus, errors) {
  const resolved = resolveRetrievalHeadPath(trace, config, evalDir);
  const summary = {
    source: resolved.source,
    path: resolved.path,
    present: false,
    schema: "",
    model_hash: "",
    recomputed_model_hash: "",
    hash_verified: false,
    hash_matches_eval: false,
    feature_count: 0,
    labels: 0,
    labels_cover_expected_spirits: false,
    text_head: false,
    image_head: false,
    text_nonzero_weights: 0,
    image_nonzero_weights: 0,
    corpus_examples_match: null,
    corpus_examples_hash_match: null,
    corpus_tokens_match: null,
    corpus_tokens_hash_match: null,
  };
  if (!resolved.source) {
    errors.push("retrieval head artifact path is missing");
    return summary;
  }
  if (!resolved.exists) {
    errors.push(`retrieval head artifact ${resolved.source} was not found`);
    return summary;
  }
  const model = readJson(resolved.path, "retrieval head artifact");
  summary.present = true;
  summary.schema = String(model.schema || "");
  summary.model_hash = String(model.model_hash || "");
  summary.feature_count = Number(model.feature_count || 0);
  summary.labels = Array.isArray(model.labels) ? model.labels.length : 0;
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    errors.push(`retrieval head artifact schema ${JSON.stringify(model.schema)} != nsrl.solomon_v2_retrieval_head.v1`);
  }
  if (summary.labels !== config.expectSpirits) {
    errors.push(`retrieval head artifact labels ${summary.labels} != ${config.expectSpirits}`);
  }
  summary.labels_cover_expected_spirits = coversExpectedSpiritIds(model.labels, config.expectSpirits);
  if (!summary.labels_cover_expected_spirits) {
    errors.push(`retrieval head artifact labels do not cover spirit ids 1..${config.expectSpirits}`);
  }
  if (summary.feature_count < config.minFeatureCount) {
    errors.push(`retrieval head artifact feature_count ${summary.feature_count} < ${config.minFeatureCount}`);
  }
  if (Number(trace.feature_count || 0) > 0 && summary.feature_count !== Number(trace.feature_count || 0)) {
    errors.push(
      `retrieval head artifact feature_count ${summary.feature_count} != eval feature_count ${Number(trace.feature_count || 0)}`,
    );
  }

  checkModelCorpus(model, config, evalDir, corpus, summary, errors);

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
  if (textHead.nonzero_weights <= 0) {
    errors.push("retrieval head artifact text_head has no nonzero weights");
  }
  if (imageHead.nonzero_weights <= 0) {
    errors.push("retrieval head artifact image_head has no nonzero weights");
  }
  summary.recomputed_model_hash = recomputeRetrievalHeadHash(model);
  summary.hash_verified = Boolean(summary.model_hash) && summary.model_hash === summary.recomputed_model_hash;
  if (!summary.hash_verified) {
    errors.push(`retrieval head artifact model_hash ${summary.model_hash || "<missing>"} != recomputed ${summary.recomputed_model_hash}`);
  }
  summary.hash_matches_eval =
    Boolean(summary.model_hash) &&
    Boolean(trace.model_hash) &&
    summary.model_hash === String(trace.model_hash);
  if (!summary.hash_matches_eval) {
    errors.push(`retrieval head artifact model_hash ${summary.model_hash || "<missing>"} != eval model_hash ${trace.model_hash || "<missing>"}`);
  }
  return summary;
}

function checkModelCorpus(model, config, evalDir, corpus, summary, errors) {
  const modelCorpus = model.corpus || {};
  summary.corpus_examples_match = requireSameReferencedPath(
    "retrieval head model corpus examples",
    String(modelCorpus.examples || ""),
    config.examplesPath,
    evalDir,
    errors,
  );
  summary.corpus_tokens_match = requireSameReferencedPath(
    "retrieval head model corpus tokens",
    String(modelCorpus.tokens || ""),
    config.tokensPath,
    evalDir,
    errors,
  );
  summary.corpus_examples_hash_match = String(modelCorpus.examples_hash || "") === corpus.expected_examples_hash;
  if (!summary.corpus_examples_hash_match) {
    errors.push(
      `retrieval head model corpus examples_hash ${modelCorpus.examples_hash || "<missing>"} != corpus examples hash ${corpus.expected_examples_hash}`,
    );
  }
  summary.corpus_tokens_hash_match = String(modelCorpus.tokens_hash || "") === corpus.expected_tokens_hash;
  if (!summary.corpus_tokens_hash_match) {
    errors.push(
      `retrieval head model corpus tokens_hash ${modelCorpus.tokens_hash || "<missing>"} != corpus tokens hash ${corpus.expected_tokens_hash}`,
    );
  }
}

function checkRetrievalMetrics(trace, config, expectedImageTasks, errors) {
  const summary = {
    known_prompts: metricSummary(trace.known_prompts),
    identity_bindings: {
      total: metricSummary(trace.identity_bindings?.total),
      by_kind: {},
    },
    heldout_prompts: metricSummary(trace.heldout_prompts),
    image_to_text: metricSummary(trace.image_to_text),
    image_tasks: {},
    match: {
      yes: metricSummary(trace.match?.yes),
      no: metricSummary(trace.match?.no),
      no_by_role: {
        image: metricSummary(trace.match?.no_by_role?.image),
        prompt: metricSummary(trace.match?.no_by_role?.prompt),
      },
    },
  };
  requireAllTop1(trace.known_prompts, "known prompts", errors);
  requireMarginFloor(trace.known_prompts, "known prompts", config.minRetrievalMargin, errors);
  requireAllTop1(trace.identity_bindings?.total, "identity bindings", errors);
  requireMarginFloor(trace.identity_bindings?.total, "identity bindings", config.minRetrievalMargin, errors);
  for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
    summary.identity_bindings.by_kind[kind] = metricSummary(trace.identity_bindings?.by_kind?.[kind]);
    requireAllTop1(trace.identity_bindings?.by_kind?.[kind], `identity binding ${kind}`, errors);
    requireMarginFloor(trace.identity_bindings?.by_kind?.[kind], `identity binding ${kind}`, config.minRetrievalMargin, errors);
  }
  if (Number(trace.heldout_prompts?.count || 0) > 0) {
    requireAllTop1(trace.heldout_prompts, "held-out prompts", errors);
    requireMarginFloor(trace.heldout_prompts, "held-out prompts", config.minRetrievalMargin, errors);
  }
  requireAllTop1(trace.image_to_text, "image-to-text/source", errors);
  requireCountEquals(
    trace.image_to_text,
    "image-to-text/source",
    sumExpectedCounts(expectedImageTasks, REVERSE_IMAGE_RETRIEVAL_TASKS),
    errors,
  );
  requireMarginFloor(trace.image_to_text, "image-to-text/source", config.minRetrievalMargin, errors);
  for (const task of IMAGE_RETRIEVAL_TASKS) {
    summary.image_tasks[task] = metricSummary(trace.image_tasks?.[task]);
    requireAllTop1(trace.image_tasks?.[task], task, errors);
    requireCountEquals(trace.image_tasks?.[task], task, Number(expectedImageTasks[task] || 0), errors);
    requireMarginFloor(trace.image_tasks?.[task], task, config.minRetrievalMargin, errors);
  }
  requireAllTop1(trace.match?.yes, "match yes", errors);
  requireAllTop1(trace.match?.no, "match no", errors);
  requireAllTop1(trace.match?.no_by_role?.image, "match no image", errors);
  requireAllTop1(trace.match?.no_by_role?.prompt, "match no prompt", errors);
  requireMarginFloor(trace.match?.yes, "match yes", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no, "match no", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no_by_role?.image, "match no image", config.minRetrievalMargin, errors);
  requireMarginFloor(trace.match?.no_by_role?.prompt, "match no prompt", config.minRetrievalMargin, errors);
  return summary;
}

function metricSummary(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    min_margin: Number(metric?.min_margin ?? 0),
    top1_accuracy_per_mille: Number(metric?.top1_accuracy_per_mille || 0),
    top5_accuracy_per_mille: Number(metric?.top5_accuracy_per_mille || 0),
  };
}

function requireAllTop1(metric, label, errors) {
  const count = Number(metric?.count || 0);
  const top1 = Number(metric?.top1 || 0);
  if (count <= 0) {
    errors.push(`${label} has no retrieval rows`);
    return;
  }
  if (top1 !== count) {
    errors.push(`${label} top1 ${top1} != count ${count}`);
  }
}

function requireCountEquals(metric, label, expected, errors) {
  const target = Number(expected || 0);
  const count = Number(metric?.count || 0);
  if (count !== target) {
    errors.push(`${label} count ${count} != expected corpus rows ${target}`);
  }
}

function requireMarginFloor(metric, label, floor, errors) {
  const minimum = Number(floor || 0);
  if (minimum <= 0 || !metric || Number(metric.count || 0) <= 0) {
    return;
  }
  const margin = Number(metric.min_margin ?? Number.MIN_SAFE_INTEGER);
  if (margin < minimum) {
    errors.push(`${label} min_margin ${margin} < ${minimum}`);
  }
}

function resolveRetrievalHeadPath(trace, config, evalDir) {
  const source = config.retrievalHeadPath || String(trace.model || "");
  if (!source) {
    return { source: "", path: "", exists: false };
  }
  for (const candidate of candidateReferencedPaths(source, evalDir)) {
    if (fs.existsSync(candidate)) {
      return { source, path: candidate, exists: true };
    }
  }
  return { source, path: candidateReferencedPaths(source, evalDir)[0] || source, exists: false };
}

function requireSameReferencedPath(label, actual, expected, baseDir, errors) {
  if (!actual) {
    errors.push(`${label} path is missing`);
    return false;
  }
  const match = sameReferencedPath(actual, expected, baseDir);
  if (!match) {
    errors.push(`${label} ${actual} does not match expected ${expected}`);
  }
  return match;
}

function sameReferencedPath(actual, expected, baseDir) {
  const normalizedExpected = normalizeReferencedPath(expected);
  return candidateReferencedPaths(actual, baseDir).some(
    (candidate) => normalizeReferencedPath(candidate) === normalizedExpected,
  );
}

function candidateReferencedPaths(reference, baseDir) {
  if (!reference) {
    return [];
  }
  if (path.isAbsolute(reference)) {
    return [path.resolve(reference)];
  }
  const candidates = [path.resolve(reference), path.resolve(baseDir, reference)];
  return [...new Set(candidates)];
}

function normalizeReferencedPath(filePath) {
  const resolved = path.resolve(filePath);
  try {
    return fs.realpathSync.native(resolved);
  } catch (_error) {
    return resolved;
  }
}

function coversExpectedSpiritIds(labels, expectSpirits) {
  if (!Array.isArray(labels) || labels.length !== expectSpirits) {
    return false;
  }
  const ids = new Set(labels.map((label) => Number(label.spirit_id)));
  for (let id = 1; id <= expectSpirits; id += 1) {
    if (!ids.has(id)) {
      return false;
    }
  }
  return true;
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

function countJsonlRows(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return 0;
  }
  return text.split(/\r?\n/).filter(Boolean).length;
}

function readJsonl(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function countImageRetrievalTasks(filePath) {
  const counts = Object.fromEntries(IMAGE_RETRIEVAL_TASKS.map((task) => [task, 0]));
  for (const row of readJsonl(filePath)) {
    const task = String(row.task || "");
    if (Object.prototype.hasOwnProperty.call(counts, task)) {
      counts[task] += 1;
    }
  }
  return counts;
}

function sumExpectedCounts(counts, tasks) {
  return tasks.reduce((total, task) => total + Number(counts[task] || 0), 0);
}

function countHeldoutPromptRows(filePath) {
  const rows = readJsonl(filePath)
    .map((row) => ({
      spirit_id: Number(row.spirit_id),
      text: String(row.text || row.prompt || ""),
      source: row.source || "",
      tier: row.tier || "",
    }))
    .filter((row) => Number.isInteger(row.spirit_id) && row.spirit_id > 0 && row.text);
  const eligible = rows.filter(isHeldoutPromptRow);
  return {
    total_rows: rows.length,
    eligible_rows: eligible.length,
    unique_targets: new Set(eligible.map((row) => row.spirit_id)).size,
  };
}

function isHeldoutPromptRow(row) {
  const tier = String(row.tier || "").toLowerCase();
  const source = String(row.source || "").toLowerCase();
  return source !== "canonical" && (tier.includes("holdout") || tier.includes("novel"));
}

function readJson(filePath, label) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`${label} ${filePath} could not be read as JSON: ${error.message}`);
  }
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

function fnv64BytesHex(bytes) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error && error.stack ? error.stack : String(error));
  process.exit(1);
}
