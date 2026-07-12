#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  inputDir: "",
  examplesPath: "",
  tokensPath: "",
  manifestPath: "",
  vocabPath: "",
  outDir: "",
  tasks: "all",
  matchLabels: "all",
  matchRoles: "all",
};

function usage() {
  console.log(
    [
      "Usage: filter-solomon-multimodal-corpus.mjs --input-dir PATH --out-dir PATH [options]",
      "",
      "Writes a stage-specific Solomon corpus by selecting examples from an existing",
      "examples.jsonl/corpus.tokens.u8 pair and rewriting token offsets.",
      "",
      "Options:",
      "  --examples PATH",
      "  --tokens PATH",
      "  --manifest PATH",
      "  --vocab PATH",
      "  --tasks LIST|all",
      "  --match-labels LIST|all",
      "  --match-roles LIST|all",
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
    } else if (arg === "--input-dir") {
      config.inputDir = requireValue(argv, ++index, arg);
    } else if (arg === "--examples") {
      config.examplesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--tokens") {
      config.tokensPath = requireValue(argv, ++index, arg);
    } else if (arg === "--manifest") {
      config.manifestPath = requireValue(argv, ++index, arg);
    } else if (arg === "--vocab") {
      config.vocabPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--tasks") {
      config.tasks = requireValue(argv, ++index, arg);
    } else if (arg === "--match-labels") {
      config.matchLabels = requireValue(argv, ++index, arg);
    } else if (arg === "--match-roles") {
      config.matchRoles = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.inputDir && (!config.examplesPath || !config.tokensPath)) {
    throw new Error("--input-dir is required unless --examples and --tokens are supplied");
  }
  if (!config.outDir) {
    throw new Error("--out-dir is required");
  }
  if (!config.examplesPath) {
    config.examplesPath = path.join(config.inputDir, "examples.jsonl");
  }
  if (!config.tokensPath) {
    config.tokensPath = path.join(config.inputDir, "corpus.tokens.u8");
  }
  if (!config.manifestPath && config.inputDir) {
    config.manifestPath = path.join(config.inputDir, "manifest.json");
  }
  if (!config.vocabPath && config.inputDir) {
    config.vocabPath = path.join(config.inputDir, "vocab.tsv");
  }
  config.taskSet = parseList(config.tasks, "tasks");
  config.matchLabelSet = parseList(config.matchLabels, "match-labels");
  config.matchRoleSet = parseList(config.matchRoles, "match-roles");
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parseList(value, label) {
  if (value === "all") {
    return null;
  }
  const items = String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (items.length === 0) {
    throw new Error(`--${label} requires LIST or all`);
  }
  return new Set(items);
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

function selected(row, config) {
  const task = String(row.task || "");
  if (config.taskSet && !config.taskSet.has(task)) {
    return false;
  }
  if (task === "match") {
    const label = String(row.match_label || row.text || "").toLowerCase();
    const role = matchRole(row);
    if (config.matchLabelSet && !config.matchLabelSet.has(label)) {
      return false;
    }
    if (config.matchRoleSet && !config.matchRoleSet.has(role)) {
      return false;
    }
  }
  return true;
}

function matchRole(row) {
  const role = String(row.negative_role || "positive").toLowerCase();
  if (role === "prompt" || role === "text" || role === "name") {
    return "prompt";
  }
  if (role === "image" || role === "seal") {
    return "image";
  }
  return "positive";
}

function rewriteRows(rows, sourceTokens) {
  const outTokens = [];
  const outRows = [];
  for (const row of rows) {
    const tokenOffset = nonNegativeInteger(row.token_offset, "token_offset", row.__line);
    const tokenCount = positiveInteger(row.token_count, "token_count", row.__line);
    const paddingBefore = nonNegativeInteger(row.padding_before || 0, "padding_before", row.__line);
    if (paddingBefore > tokenOffset) {
      throw new Error(`examples line ${row.__line}: padding_before exceeds token_offset`);
    }
    const sourceStart = tokenOffset - paddingBefore;
    const sourceEnd = tokenOffset + tokenCount;
    if (sourceEnd > sourceTokens.length) {
      throw new Error(`examples line ${row.__line}: token range exceeds source token file`);
    }
    const newTokenOffset = outTokens.length + paddingBefore;
    for (let index = sourceStart; index < sourceEnd; index += 1) {
      outTokens.push(sourceTokens[index]);
    }
    const rewritten = { ...row };
    delete rewritten.__line;
    rewritten.token_offset = newTokenOffset;
    rewritten.token_count = tokenCount;
    rewritten.padding_before = paddingBefore;
    rewritten.token_hash = fnv64Hex(sourceTokens.subarray(tokenOffset, tokenOffset + tokenCount));
    outRows.push(rewritten);
  }
  return { outRows, outTokens };
}

function nonNegativeInteger(value, field, line) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`examples line ${line}: invalid ${field}`);
  }
  return parsed;
}

function positiveInteger(value, field, line) {
  const parsed = nonNegativeInteger(value, field, line);
  if (parsed === 0) {
    throw new Error(`examples line ${line}: ${field} must be positive`);
  }
  return parsed;
}

function writeU16Tokens(tokens, outPath) {
  const bytes = Buffer.alloc(tokens.length * 2);
  for (let index = 0; index < tokens.length; index += 1) {
    bytes.writeUInt16LE(tokens[index], index * 2);
  }
  fs.writeFileSync(outPath, bytes);
}

function writeU8Tokens(tokens, outPath) {
  fs.writeFileSync(outPath, Buffer.from(tokens));
}

function fnv64Hex(tokens) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const token of tokens) {
    if (Number(token) < 0 || Number(token) > 255) {
      throw new Error(`token ${token} is outside byte range`);
    }
    hash ^= BigInt(token & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64TextHex(text) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= BigInt(text.charCodeAt(index) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function identityBindingSummary(rows) {
  const summary = {
    rows: 0,
    binding_hash: fnv64TextHex(""),
    by_task: new Map(),
    by_kind: new Map(),
  };
  const keys = [];
  for (const row of rows) {
    if (row.identity_binding !== true && row.identity_binding !== "true") {
      continue;
    }
    const task = String(row.task || "");
    const kind = String(row.binding_kind || "");
    const spiritId = String(row.spirit_id || "");
    const prompt = normalizeBindingText(row.prompt || "");
    const key = `${task}\t${kind}\t${spiritId}\t${prompt}`;
    keys.push(key);
    summary.rows += 1;
    incrementIdentityGroup(summary.by_task, task, kind, spiritId, key);
    incrementIdentityGroup(summary.by_kind, kind, task, spiritId, key);
  }
  summary.binding_hash = hashIdentityKeys(keys);
  return {
    rows: summary.rows,
    binding_hash: summary.binding_hash,
    by_task: identityGroupMapJson(summary.by_task),
    by_kind: identityGroupMapJson(summary.by_kind),
  };
}

function taskCoverageSummary(rows) {
  const tasks = new Map();
  const allSpirits = new Set();
  for (const row of rows) {
    const task = String(row.task || "");
    const spiritId = normalizedSpiritId(row.spirit_id);
    if (spiritId !== null) {
      allSpirits.add(spiritId);
    }
    if (!tasks.has(task)) {
      tasks.set(task, {
        records: 0,
        spirits: new Set(),
        identity_binding_rows: 0,
        labels: new Map(),
        roles: new Map(),
      });
    }
    const group = tasks.get(task);
    group.records += 1;
    if (spiritId !== null) {
      group.spirits.add(spiritId);
    }
    if (row.identity_binding === true || row.identity_binding === "true") {
      group.identity_binding_rows += 1;
    }
    if (task === "match") {
      const label = String(row.match_label || row.text || "").toLowerCase();
      const role = matchRole(row);
      incrementCoverageGroup(group.labels, label, spiritId);
      incrementCoverageGroup(group.roles, role, spiritId);
    }
  }
  return {
    records: rows.length,
    spirits: allSpirits.size,
    tasks: Object.fromEntries(
      [...tasks.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([task, group]) => [
        task,
        {
          records: group.records,
          spirits: group.spirits.size,
          identity_binding_rows: group.identity_binding_rows,
          labels: coverageGroupJson(group.labels),
          roles: coverageGroupJson(group.roles),
        },
      ]),
    ),
  };
}

function incrementCoverageGroup(map, key, spiritId) {
  if (!map.has(key)) {
    map.set(key, { records: 0, spirits: new Set() });
  }
  const group = map.get(key);
  group.records += 1;
  if (spiritId !== null) {
    group.spirits.add(spiritId);
  }
}

function coverageGroupJson(map) {
  return Object.fromEntries(
    [...map.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([key, group]) => [
      key,
      {
        records: group.records,
        spirits: group.spirits.size,
      },
    ]),
  );
}

function normalizedSpiritId(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function incrementIdentityGroup(map, primaryKey, secondaryKey, spiritId, bindingKey) {
  if (!map.has(primaryKey)) {
    map.set(primaryKey, { rows: 0, spirits: new Set(), keys: [], by_key: new Map() });
  }
  const group = map.get(primaryKey);
  group.rows += 1;
  if (spiritId) {
    group.spirits.add(spiritId);
  }
  group.keys.push(bindingKey);
  group.by_key.set(secondaryKey, (group.by_key.get(secondaryKey) || 0) + 1);
}

function identityGroupMapJson(map) {
  return Object.fromEntries(
    [...map.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([key, group]) => [
      key,
      {
        rows: group.rows,
        spirits: group.spirits.size,
        binding_hash: hashIdentityKeys(group.keys),
        counts: Object.fromEntries([...group.by_key.entries()].sort(([left], [right]) => left.localeCompare(right))),
      },
    ]),
  );
}

function hashIdentityKeys(keys) {
  return fnv64TextHex([...keys].sort().join("\n"));
}

function normalizeBindingText(value) {
  return String(value ?? "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201c\u201d]/g, '"')
    .replace(/[\u2013\u2014]/g, " ")
    .toLowerCase()
    .replace(/[^a-z0-9']+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function copyIfExists(source, dest) {
  if (source && fs.existsSync(source)) {
    fs.copyFileSync(source, dest);
    return path.basename(dest);
  }
  return null;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const sourceRows = readJsonl(config.examplesPath);
  const sourceTokens = fs.readFileSync(config.tokensPath);
  const selectedRows = sourceRows.filter((row) => selected(row, config));
  if (selectedRows.length === 0) {
    throw new Error("filter selected zero examples");
  }
  const { outRows, outTokens } = rewriteRows(selectedRows, sourceTokens);

  fs.mkdirSync(config.outDir, { recursive: true });
  const outExamples = path.join(config.outDir, "examples.jsonl");
  const outU8 = path.join(config.outDir, "corpus.tokens.u8");
  const outU16 = path.join(config.outDir, "corpus.tokens.u16");
  const outManifest = path.join(config.outDir, "manifest.json");
  const outVocab = path.join(config.outDir, "vocab.tsv");

  fs.writeFileSync(outExamples, `${outRows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  writeU8Tokens(outTokens, outU8);
  writeU16Tokens(outTokens, outU16);
  const vocabTsv = copyIfExists(config.vocabPath, outVocab);
  const sourceManifest = config.manifestPath && fs.existsSync(config.manifestPath)
    ? JSON.parse(fs.readFileSync(config.manifestPath, "utf8"))
    : {};
  const selectedIdentityBindings = identityBindingSummary(outRows);
  const sourceIdentityBindings = identityBindingSummary(sourceRows);
  const selectedTaskCoverage = taskCoverageSummary(outRows);
  const sourceTaskCoverage = taskCoverageSummary(sourceRows);
  const manifest = {
    ...sourceManifest,
    schema: "nsrl.solomon_multimodal_corpus_filter.v1",
    source_manifest_schema: sourceManifest.schema || null,
    source_dir: config.inputDir || path.dirname(config.examplesPath),
    source_examples_jsonl: config.examplesPath,
    source_corpus_tokens_u8: config.tokensPath,
    examples: outRows.length,
    training_sequences: outRows.length,
    token_count: outTokens.length,
    token_hash: fnv64Hex(outTokens),
    source_task_coverage: sourceTaskCoverage,
    task_coverage: selectedTaskCoverage,
    source_identity_bindings: sourceIdentityBindings,
    identity_bindings: selectedIdentityBindings,
    filter: {
      tasks: config.tasks,
      match_labels: config.matchLabels,
      match_roles: config.matchRoles,
    },
    corpus_tokens_u16: path.relative(config.outDir, outU16),
    corpus_tokens_u8: path.relative(config.outDir, outU8),
    examples_jsonl: path.relative(config.outDir, outExamples),
    vocab_tsv: vocabTsv,
  };
  fs.writeFileSync(outManifest, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(
    JSON.stringify({
      schema: manifest.schema,
      out_dir: config.outDir,
      examples: outRows.length,
      token_count: outTokens.length,
      token_hash: manifest.token_hash,
      task_coverage: selectedTaskCoverage,
      identity_bindings: selectedIdentityBindings,
      filter: manifest.filter,
    }),
  );
}

main();
