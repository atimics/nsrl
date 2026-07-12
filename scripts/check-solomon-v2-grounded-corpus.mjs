#!/usr/bin/env node

import fs from "node:fs";

const SOURCE_TEXT_TASKS = [
  "explain",
  "image-to-explain",
  "text-image-explain",
  "description-to-image",
];
const ATTRIBUTE_TASKS = ["image-to-attributes"];
const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;
const FNV64_MASK = 0xffffffffffffffffn;

const defaults = {
  examplesPath: "",
  textIndexPath: "",
  outPath: "",
  expectSpirits: 72,
  minSourceOverlapTokens: 2,
  minAttributeSourceOverlapTokens: 8,
  maxSourcePlaceholderRows: 0,
  maxAttributeGenericRankRows: 0,
  requireSourceProvenance: true,
  requireNameSourceExplain: true,
  requireDescriptionSourceImage: true,
  requireImageAttributeGenericPrompt: true,
  sourceTextTasks: SOURCE_TEXT_TASKS,
  attributeTasks: ATTRIBUTE_TASKS,
};

const STOPWORDS = new Set([
  "and", "are", "but", "for", "from", "has", "have", "him", "his", "into",
  "its", "not", "of", "or", "over", "she", "that", "the", "thee", "this",
  "thou", "to", "unto", "upon", "with",
]);

function usage() {
  console.log(
    [
      "Usage: check-solomon-v2-grounded-corpus.mjs --examples PATH --text-index PATH [options]",
      "",
      "Checks that v2 source/explanation and attribute rows carry grounded",
      "source-derived content across the Solomon spirit set.",
      "",
      "Options:",
      "  --out PATH",
      "  --expect-spirits N",
      "  --source-text-tasks LIST",
      "  --attribute-tasks LIST",
      "  --min-source-overlap-tokens N",
      "  --min-attribute-source-overlap-tokens N",
      "  --max-source-placeholder-rows N",
      "  --max-attribute-generic-rank-rows N",
      "  --require-source-provenance",
      "  --no-require-source-provenance",
      "  --require-name-source-explain",
      "  --no-require-name-source-explain",
      "  --require-description-source-image",
      "  --no-require-description-source-image",
      "  --require-image-attribute-generic-prompt",
      "  --no-require-image-attribute-generic-prompt",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = {
    ...defaults,
    sourceTextTasks: [...defaults.sourceTextTasks],
    attributeTasks: [...defaults.attributeTasks],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--examples") {
      config.examplesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--text-index") {
      config.textIndexPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--expect-spirits") {
      config.expectSpirits = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--source-text-tasks") {
      config.sourceTextTasks = parseList(requireValue(argv, ++index, arg));
    } else if (arg === "--attribute-tasks") {
      config.attributeTasks = parseList(requireValue(argv, ++index, arg));
    } else if (arg === "--min-source-overlap-tokens") {
      config.minSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-attribute-source-overlap-tokens") {
      config.minAttributeSourceOverlapTokens = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-source-placeholder-rows") {
      config.maxSourcePlaceholderRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-attribute-generic-rank-rows") {
      config.maxAttributeGenericRankRows = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-source-provenance") {
      config.requireSourceProvenance = true;
    } else if (arg === "--no-require-source-provenance") {
      config.requireSourceProvenance = false;
    } else if (arg === "--require-name-source-explain") {
      config.requireNameSourceExplain = true;
    } else if (arg === "--no-require-name-source-explain") {
      config.requireNameSourceExplain = false;
    } else if (arg === "--require-description-source-image") {
      config.requireDescriptionSourceImage = true;
    } else if (arg === "--no-require-description-source-image") {
      config.requireDescriptionSourceImage = false;
    } else if (arg === "--require-image-attribute-generic-prompt") {
      config.requireImageAttributeGenericPrompt = true;
    } else if (arg === "--no-require-image-attribute-generic-prompt") {
      config.requireImageAttributeGenericPrompt = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.examplesPath) {
    throw new Error("--examples is required");
  }
  if (!config.textIndexPath) {
    throw new Error("--text-index is required");
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

function parseList(value) {
  const items = String(value).split(",").map((item) => item.trim()).filter(Boolean);
  if (items.length === 0) {
    throw new Error("task list must not be empty");
  }
  return items;
}

function readJsonl(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  return text.split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    row.__line = index + 1;
    return row;
  });
}

function fnv64FileHex(filePath) {
  let hash = FNV64_OFFSET;
  for (const byte of fs.readFileSync(filePath)) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function readTextIndex(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    throw new Error(`${filePath} is empty`);
  }
  const lines = text.split(/\r?\n/);
  const header = lines[0].split("\t");
  const rows = new Map();
  for (const [rowIndex, line] of lines.slice(1).entries()) {
    if (!line) continue;
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    const id = Number(row.number);
    if (!Number.isInteger(id)) {
      throw new Error(`${filePath}:${rowIndex + 2} has invalid number`);
    }
    rows.set(id, row);
  }
  return rows;
}

function checkRows(examples, sourceRows, config) {
  const errors = [];
  const requiredTasks = [...config.sourceTextTasks, ...config.attributeTasks];
  const stats = new Map(requiredTasks.map((task) => [task, emptyTaskStats()]));
  for (const row of examples) {
    const task = String(row.task || "");
    if (!stats.has(task)) {
      continue;
    }
    const source = sourceRows.get(Number(row.spirit_id));
    if (!source) {
      errors.push(`examples line ${row.__line}: missing source row for spirit_id ${row.spirit_id}`);
      continue;
    }
    const taskStats = stats.get(task);
    taskStats.records += 1;
    taskStats.spirits.add(Number(row.spirit_id));
    const check = config.attributeTasks.includes(task)
      ? checkAttributeRow(row, source, config)
      : checkSourceTextRow(row, source, config);
    const provenance = checkSourceProvenance(row, source, config);
    const nameSourcePrompt = checkNameSourceExplainPrompt(row, source, config);
    const descriptionSourcePrompt = checkDescriptionSourceImagePrompt(row, source, config);
    const imageAttributePrompt = checkImageAttributePrompt(row, source, config);
    taskStats.min_source_overlap_tokens = Math.min(taskStats.min_source_overlap_tokens, check.source_overlap_tokens);
    taskStats.min_content_tokens = Math.min(taskStats.min_content_tokens, check.content_tokens);
    if (check.source_substring) taskStats.source_substring_rows += 1;
    if (check.placeholder_count > 0) taskStats.placeholder_rows += 1;
    taskStats.placeholder_count += check.placeholder_count;
    if (check.generic_attribute_rank) taskStats.generic_attribute_rank_rows += 1;
    if (provenance.present) taskStats.source_provenance_rows += 1;
    if (provenance.hash_mismatch) taskStats.source_provenance_hash_mismatches += 1;
    if (provenance.excerpt_hash_mismatch) taskStats.source_excerpt_hash_mismatches += 1;
    if (nameSourcePrompt.applicable) {
      taskStats.name_source_prompt_rows += 1;
      if (nameSourcePrompt.ok) taskStats.name_source_prompt_ok_rows += 1;
    }
    if (descriptionSourcePrompt.applicable) {
      taskStats.description_source_prompt_rows += 1;
      if (descriptionSourcePrompt.ok) taskStats.description_source_prompt_ok_rows += 1;
    }
    if (imageAttributePrompt.applicable) {
      taskStats.image_attribute_prompt_rows += 1;
      if (imageAttributePrompt.ok) taskStats.image_attribute_prompt_ok_rows += 1;
    }
    if (!check.ok || !provenance.ok || !nameSourcePrompt.ok || !descriptionSourcePrompt.ok || !imageAttributePrompt.ok) {
      errors.push(
        ...[
          ...check.errors,
          ...provenance.errors,
          ...nameSourcePrompt.errors,
          ...descriptionSourcePrompt.errors,
          ...imageAttributePrompt.errors,
        ].map(
          (error) => `examples line ${row.__line} ${task}: ${error}`,
        ),
      );
    }
  }
  for (const task of requiredTasks) {
    const taskStats = stats.get(task);
    if (taskStats.records <= 0) {
      errors.push(`examples are missing grounded task ${task}`);
      continue;
    }
    if (config.expectSpirits > 0 && taskStats.spirits.size !== config.expectSpirits) {
      errors.push(`grounded task ${task} covers ${taskStats.spirits.size} spirits, expected ${config.expectSpirits}`);
    }
    if (config.sourceTextTasks.includes(task) && taskStats.placeholder_rows > config.maxSourcePlaceholderRows) {
      errors.push(
        `grounded task ${task} source placeholder rows ${taskStats.placeholder_rows} > ${config.maxSourcePlaceholderRows}`,
      );
    }
    if (config.attributeTasks.includes(task) && taskStats.generic_attribute_rank_rows > config.maxAttributeGenericRankRows) {
      errors.push(
        `grounded task ${task} generic rank rows ${taskStats.generic_attribute_rank_rows} > ${config.maxAttributeGenericRankRows}`,
      );
    }
    if (config.requireSourceProvenance && taskStats.source_provenance_rows !== taskStats.records) {
      errors.push(
        `grounded task ${task} source provenance rows ${taskStats.source_provenance_rows} != records ${taskStats.records}`,
      );
    }
    if (
      config.requireNameSourceExplain &&
      task === "explain" &&
      taskStats.name_source_prompt_ok_rows !== taskStats.records
    ) {
      errors.push(
        `grounded task explain name-source prompt rows ${taskStats.name_source_prompt_ok_rows} != records ${taskStats.records}`,
      );
    }
    if (
      config.requireDescriptionSourceImage &&
      task === "description-to-image" &&
      taskStats.description_source_prompt_ok_rows !== taskStats.records
    ) {
      errors.push(
        `grounded task description-to-image description-source prompt rows ${taskStats.description_source_prompt_ok_rows} != records ${taskStats.records}`,
      );
    }
    if (
      config.requireImageAttributeGenericPrompt &&
      task === "image-to-attributes" &&
      taskStats.image_attribute_prompt_ok_rows !== taskStats.records
    ) {
      errors.push(
        `grounded task image-to-attributes generic attribute prompt rows ${taskStats.image_attribute_prompt_ok_rows} != records ${taskStats.records}`,
      );
    }
  }
  return {
    ok: errors.length === 0,
    errors,
    tasks: Object.fromEntries(
      [...stats.entries()].map(([task, taskStats]) => [task, summarizeTaskStats(taskStats)]),
    ),
  };
}

function emptyTaskStats() {
  return {
    records: 0,
    spirits: new Set(),
    min_source_overlap_tokens: Number.POSITIVE_INFINITY,
    min_content_tokens: Number.POSITIVE_INFINITY,
    source_substring_rows: 0,
    placeholder_rows: 0,
    placeholder_count: 0,
    generic_attribute_rank_rows: 0,
    source_provenance_rows: 0,
    source_provenance_hash_mismatches: 0,
    source_excerpt_hash_mismatches: 0,
    name_source_prompt_rows: 0,
    name_source_prompt_ok_rows: 0,
    description_source_prompt_rows: 0,
    description_source_prompt_ok_rows: 0,
    image_attribute_prompt_rows: 0,
    image_attribute_prompt_ok_rows: 0,
  };
}

function summarizeTaskStats(taskStats) {
  return {
    records: taskStats.records,
    spirits: taskStats.spirits.size,
    min_source_overlap_tokens:
      taskStats.min_source_overlap_tokens === Number.POSITIVE_INFINITY ? 0 : taskStats.min_source_overlap_tokens,
    min_content_tokens:
      taskStats.min_content_tokens === Number.POSITIVE_INFINITY ? 0 : taskStats.min_content_tokens,
    source_substring_rows: taskStats.source_substring_rows,
    placeholder_rows: taskStats.placeholder_rows,
    placeholder_count: taskStats.placeholder_count,
    generic_attribute_rank_rows: taskStats.generic_attribute_rank_rows,
    source_provenance_rows: taskStats.source_provenance_rows,
    source_provenance_hash_mismatches: taskStats.source_provenance_hash_mismatches,
    source_excerpt_hash_mismatches: taskStats.source_excerpt_hash_mismatches,
    name_source_prompt_rows: taskStats.name_source_prompt_rows,
    name_source_prompt_ok_rows: taskStats.name_source_prompt_ok_rows,
    description_source_prompt_rows: taskStats.description_source_prompt_rows,
    description_source_prompt_ok_rows: taskStats.description_source_prompt_ok_rows,
    image_attribute_prompt_rows: taskStats.image_attribute_prompt_rows,
    image_attribute_prompt_ok_rows: taskStats.image_attribute_prompt_ok_rows,
  };
}

function checkSourceTextRow(row, source, config) {
  const content = sourceTextContent(row);
  const sourceText = String(source.text || "");
  const contentTokens = contentTokenSet(content);
  const sourceTokens = contentTokenSet(sourceText);
  const overlap = intersectionSize(contentTokens, sourceTokens);
  const normalizedContent = normalizeForSubstring(content);
  const normalizedSource = normalizeForSubstring(sourceText);
  const sourceSubstring = normalizedContent.length > 0 && normalizedSource.includes(normalizedContent);
  const errors = [];
  if (contentTokens.size < config.minSourceOverlapTokens) {
    errors.push(`content has ${contentTokens.size} content tokens < ${config.minSourceOverlapTokens}`);
  }
  if (!sourceSubstring && overlap < config.minSourceOverlapTokens) {
    errors.push(`source overlap ${overlap} < ${config.minSourceOverlapTokens}`);
  }
  return {
    ok: errors.length === 0,
    errors,
    source_overlap_tokens: overlap,
    content_tokens: contentTokens.size,
    source_substring: sourceSubstring,
    placeholder_count: placeholderCount(String(row.text || "")),
  };
}

function checkAttributeRow(row, source, config) {
  const text = String(row.text || "");
  const name = String(row.primary_name || source.primary_name || "").trim();
  const sourceText = String(source.text || "");
  const contentTokens = contentTokenSet(text);
  const sourceTokens = contentTokenSet(sourceText);
  const overlap = intersectionSize(contentTokens, sourceTokens);
  const placeholders = placeholderCount(text);
  const errors = [];
  if (name && !normalizeForSubstring(text).includes(normalizeForSubstring(name))) {
    errors.push(`attribute text does not include primary name ${JSON.stringify(name)}`);
  }
  for (const field of ["rank", "appearance", "office"]) {
    if (!new RegExp(`\\b${field}\\b`, "i").test(text)) {
      errors.push(`attribute text missing ${field} field`);
    }
  }
  if (overlap < config.minAttributeSourceOverlapTokens) {
    errors.push(`attribute source overlap ${overlap} < ${config.minAttributeSourceOverlapTokens}`);
  }
  if (/legions recorded in source/i.test(text)) {
    errors.push("attribute text used generic legions placeholder");
  }
  const genericAttributeRank = /\brank\s+(?:Goetic spirit|not stated in source)\b/i.test(text);
  if (genericAttributeRank && config.maxAttributeGenericRankRows <= 0) {
    errors.push("attribute text used generic rank placeholder");
  }
  return {
    ok: errors.length === 0,
    errors,
    source_overlap_tokens: overlap,
    content_tokens: contentTokens.size,
    source_substring: false,
    placeholder_count: placeholders,
    generic_attribute_rank: genericAttributeRank,
  };
}

function checkImageAttributePrompt(row, source, config) {
  if (!config.requireImageAttributeGenericPrompt || String(row.task || "") !== "image-to-attributes") {
    return { ok: true, applicable: false, errors: [] };
  }
  const prompt = String(row.prompt || "");
  const normalizedPrompt = normalizeForSubstring(prompt);
  const expectedPrompt = "seal attributes";
  const expected = normalizeForSubstring(expectedPrompt);
  const primaryName = String(row.primary_name || source.primary_name || "").trim();
  const errors = [];
  if (normalizedPrompt !== expected) {
    errors.push(`image-to-attributes prompt ${JSON.stringify(prompt)} != ${JSON.stringify(expectedPrompt)}`);
  }
  if (primaryName && normalizedPrompt.includes(normalizeForSubstring(primaryName))) {
    errors.push(`image-to-attributes prompt leaks primary name ${JSON.stringify(primaryName)}`);
  }
  return {
    ok: errors.length === 0,
    applicable: true,
    errors,
  };
}

function checkSourceProvenance(row, source, config) {
  const sourceTextHash = String(row.source_text_hash || "");
  const sourceExcerpt = String(row.source_excerpt || "");
  const sourceExcerptHash = String(row.source_excerpt_hash || "");
  const sourceSpiritId = Number(row.source_spirit_id);
  const expectedSourceHash = fnv64TextHex(normalizeSourceText(source.text || ""));
  const expectedExcerptHash = sourceExcerpt ? fnv64TextHex(normalizeSourceText(sourceExcerpt)) : "";
  const present = Boolean(sourceTextHash || sourceExcerpt || sourceExcerptHash || Number.isFinite(sourceSpiritId));
  const errors = [];
  let hashMismatch = false;
  let excerptHashMismatch = false;
  if (!present && !config.requireSourceProvenance) {
    return {
      ok: true,
      errors,
      present: false,
      hash_mismatch: false,
      excerpt_hash_mismatch: false,
    };
  }
  if (!Number.isInteger(sourceSpiritId) || sourceSpiritId !== Number(row.spirit_id)) {
    errors.push(`source_spirit_id ${JSON.stringify(row.source_spirit_id || "")} != spirit_id ${row.spirit_id}`);
  }
  if (!/^0x[0-9a-f]{16}$/i.test(sourceTextHash)) {
    errors.push(`source_text_hash ${JSON.stringify(sourceTextHash)} is not a fnv64 hex hash`);
  } else if (sourceTextHash.toLowerCase() !== expectedSourceHash) {
    hashMismatch = true;
    errors.push(`source_text_hash ${sourceTextHash} != ${expectedSourceHash}`);
  }
  if (!sourceExcerpt) {
    errors.push("source_excerpt is missing");
  } else if (!normalizeForSubstring(normalizeSourceText(source.text || "")).includes(normalizeForSubstring(sourceExcerpt))) {
    errors.push("source_excerpt is not contained in source text");
  }
  if (!/^0x[0-9a-f]{16}$/i.test(sourceExcerptHash)) {
    errors.push(`source_excerpt_hash ${JSON.stringify(sourceExcerptHash)} is not a fnv64 hex hash`);
  } else if (expectedExcerptHash && sourceExcerptHash.toLowerCase() !== expectedExcerptHash) {
    excerptHashMismatch = true;
    errors.push(`source_excerpt_hash ${sourceExcerptHash} != ${expectedExcerptHash}`);
  }
  return {
    ok: errors.length === 0,
    errors,
    present,
    hash_mismatch: hashMismatch,
    excerpt_hash_mismatch: excerptHashMismatch,
  };
}

function checkNameSourceExplainPrompt(row, source, config) {
  if (!config.requireNameSourceExplain || String(row.task || "") !== "explain") {
    return { ok: true, applicable: false, errors: [] };
  }
  const expectedName = String(row.primary_name || source.primary_name || "").trim();
  const expected = normalizeForSubstring(expectedName);
  const actual = normalizeForSubstring(row.prompt || "");
  const errors = [];
  if (!expectedName) {
    errors.push("explain row is missing primary_name for name-source prompt check");
  } else if (actual !== expected) {
    errors.push(`explain prompt ${JSON.stringify(row.prompt || "")} != primary name ${JSON.stringify(expectedName)}`);
  }
  return {
    ok: errors.length === 0,
    applicable: true,
    errors,
  };
}

function checkDescriptionSourceImagePrompt(row, source, config) {
  if (!config.requireDescriptionSourceImage || String(row.task || "") !== "description-to-image") {
    return { ok: true, applicable: false, errors: [] };
  }
  const prompt = String(row.prompt || "");
  const sourceText = String(source.text || "");
  const promptTokens = contentTokenSet(prompt);
  const sourceTokens = contentTokenSet(sourceText);
  const overlap = intersectionSize(promptTokens, sourceTokens);
  const normalizedPrompt = normalizeForSubstring(prompt);
  const normalizedSource = normalizeForSubstring(sourceText);
  const sourceSubstring = normalizedPrompt.length > 0 && normalizedSource.includes(normalizedPrompt);
  const errors = [];
  if (!prompt.trim()) {
    errors.push("description-to-image prompt is missing");
  }
  if (promptTokens.size < config.minSourceOverlapTokens) {
    errors.push(`description-to-image prompt has ${promptTokens.size} content tokens < ${config.minSourceOverlapTokens}`);
  }
  if (!sourceSubstring && overlap < config.minSourceOverlapTokens) {
    errors.push(`description-to-image prompt source overlap ${overlap} < ${config.minSourceOverlapTokens}`);
  }
  return {
    ok: errors.length === 0,
    applicable: true,
    errors,
  };
}

function sourceTextContent(row) {
  let text = String(row.text || "");
  const name = String(row.primary_name || "").trim();
  if (name) {
    const prefix = new RegExp(`^\\s*Solomon\\s+selects\\s+${escapeRegExp(name)}\\s*:\\s*`, "i");
    text = text.replace(prefix, "");
  }
  return text;
}

function contentTokenSet(text) {
  return new Set(
    normalizeForTokens(text)
      .split(" ")
      .filter((token) => token.length >= 3 && !STOPWORDS.has(token)),
  );
}

function normalizeForTokens(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function normalizeForSubstring(text) {
  return normalizeForTokens(text);
}

function intersectionSize(left, right) {
  let count = 0;
  for (const value of left) {
    if (right.has(value)) count += 1;
  }
  return count;
}

function placeholderCount(text) {
  let count = 0;
  for (const pattern of [
    /recorded in source/i,
    /described in the source/i,
    /Goetic spirit/i,
  ]) {
    if (pattern.test(text)) count += 1;
  }
  return count;
}

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeSourceText(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, " - ")
    .replace(/\[[0-9]+\]/g, " ")
    .replace(/[^ -~]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function fnv64TextHex(value) {
  let hash = FNV64_OFFSET;
  for (const byte of Buffer.from(String(value), "utf8")) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * FNV64_PRIME) & FNV64_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function writeJson(filePath, row) {
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const examples = readJsonl(config.examplesPath);
  const sourceRows = readTextIndex(config.textIndexPath);
  const check = checkRows(examples, sourceRows, config);
  const report = {
    schema: "nsrl.solomon_v2_grounded_corpus_check.v1",
    ok: check.ok,
    examples: config.examplesPath,
    examples_hash: fnv64FileHex(config.examplesPath),
    text_index: config.textIndexPath,
    text_index_hash: fnv64FileHex(config.textIndexPath),
    expect_spirits: config.expectSpirits,
    source_text_tasks: config.sourceTextTasks,
    attribute_tasks: config.attributeTasks,
    min_source_overlap_tokens: config.minSourceOverlapTokens,
    min_attribute_source_overlap_tokens: config.minAttributeSourceOverlapTokens,
    max_source_placeholder_rows: config.maxSourcePlaceholderRows,
    max_attribute_generic_rank_rows: config.maxAttributeGenericRankRows,
    require_source_provenance: config.requireSourceProvenance,
    require_name_source_explain: config.requireNameSourceExplain,
    require_description_source_image: config.requireDescriptionSourceImage,
    require_image_attribute_generic_prompt: config.requireImageAttributeGenericPrompt,
    tasks: check.tasks,
    errors: check.errors,
  };
  if (config.outPath) {
    writeJson(config.outPath, report);
  }
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
