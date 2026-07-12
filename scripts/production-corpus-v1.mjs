#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const CONFIG_SCHEMA = "nsrl.production_corpus_config.v1";
const MANIFEST_SCHEMA = "nsrl.production_corpus_manifest.v1";
const RECORD_SCHEMA = "nsrl.production_corpus_record.v1";

function usage() {
  console.log(`Usage:
  node scripts/production-corpus-v1.mjs build --config PATH --out-dir PATH
  node scripts/production-corpus-v1.mjs bind-tokenizer --manifest PATH --tokenizer PATH --trace PATH
  node scripts/production-corpus-v1.mjs bind-encoding --manifest PATH --split train|dev|test --tokens PATH --trace PATH
  node scripts/production-corpus-v1.mjs check --manifest PATH`);
}

function parseArgs(argv) {
  const options = { command: null, config: null, outDir: null, manifest: null, tokenizer: null, trace: null, split: null, tokens: null };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (["build", "bind-tokenizer", "bind-encoding", "check"].includes(arg) && options.command === null) {
      options.command = arg;
    } else if (arg === "--config") {
      options.config = argv[++index];
    } else if (arg === "--out-dir") {
      options.outDir = argv[++index];
    } else if (arg === "--manifest") {
      options.manifest = argv[++index];
    } else if (arg === "--tokenizer") {
      options.tokenizer = argv[++index];
    } else if (arg === "--trace") {
      options.trace = argv[++index];
    } else if (arg === "--split") {
      options.split = argv[++index];
    } else if (arg === "--tokens") {
      options.tokens = argv[++index];
    } else if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return options;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function sha256File(filePath) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  return hash.digest("hex");
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function portablePath(filePath) {
  const absolute = path.resolve(filePath);
  const relative = path.relative(process.cwd(), absolute);
  return relative.startsWith("..") ? absolute : relative || ".";
}

function normalizeText(input) {
  return input
    .replace(/^\uFEFF/, "")
    .replace(/\r\n?/g, "\n")
    .normalize("NFC")
    .replace(/[\u0000\u000B\u000C\u000E-\u001F\u007F]/g, "")
    .replace(/[\t ]+$/gm, "")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function cleanGutenberg(input) {
  let text = input;
  const start = text.search(/^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (start >= 0) {
    const lineEnd = text.indexOf("\n", start);
    text = lineEnd >= 0 ? text.slice(lineEnd + 1) : "";
  }
  const end = text.search(/^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (end >= 0) text = text.slice(0, end);
  return normalizeText(text);
}

function safeUtf8Prefix(buffer, limit) {
  if (buffer.length <= limit) return buffer;
  let end = limit;
  while (end > 0 && (buffer[end] & 0xc0) === 0x80) end -= 1;
  return buffer.subarray(0, end);
}

function chunkUtf8(text, maxBytes) {
  const chunks = [];
  let remaining = Buffer.from(text, "utf8");
  while (remaining.length > maxBytes) {
    const prefix = safeUtf8Prefix(remaining, maxBytes);
    const minimum = Math.floor(prefix.length / 2);
    let split = prefix.lastIndexOf("\n\n");
    if (split < minimum) split = prefix.lastIndexOf("\n");
    if (split < minimum) split = prefix.lastIndexOf(" ");
    if (split < minimum) split = prefix.length;
    const chunk = normalizeText(prefix.subarray(0, split).toString("utf8"));
    if (chunk) chunks.push(chunk);
    remaining = remaining.subarray(split);
    while (remaining.length > 0 && [9, 10, 13, 32].includes(remaining[0])) {
      remaining = remaining.subarray(1);
    }
  }
  const tail = normalizeText(remaining.toString("utf8"));
  if (tail) chunks.push(tail);
  return chunks;
}

function fnv1a32(value) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

function fnv1a64(bytes) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function wordShingles(text, width = 5) {
  const words = text.toLocaleLowerCase("en-US").match(/[\p{L}\p{N}]+/gu) ?? [];
  const hashes = new Set();
  if (words.length < width) {
    if (words.length > 0) hashes.add(fnv1a32(words.join(" ")));
    return hashes;
  }
  for (let index = 0; index + width <= words.length; index += 1) {
    hashes.add(fnv1a32(words.slice(index, index + width).join(" ")));
  }
  return hashes;
}

function bottomSignature(shingles, size) {
  return [...shingles].sort((left, right) => left - right).slice(0, size);
}

function intersectionSize(left, right) {
  let count = 0;
  const [small, large] = left.size <= right.size ? [left, right] : [right, left];
  for (const item of small) if (large.has(item)) count += 1;
  return count;
}

function isNearDuplicate(left, right, thresholdPermille) {
  const intersection = intersectionSize(left, right);
  const union = left.size + right.size - intersection;
  return union > 0 && intersection * 1000 >= union * thresholdPermille;
}

function validateConfig(config) {
  if (config.schema !== CONFIG_SCHEMA || config.contract_id !== "production-corpus-v1") {
    throw new Error(`config must use ${CONFIG_SCHEMA} and production-corpus-v1`);
  }
  if (!Array.isArray(config.sources) || config.sources.length === 0) {
    throw new Error("config requires at least one source");
  }
  const ids = new Set();
  for (const source of config.sources) {
    if (!source.id || ids.has(source.id)) throw new Error(`duplicate or empty source id: ${source.id}`);
    ids.add(source.id);
    if (!["gutenberg_text", "nsrl_simplewiki_pages", "plain_text"].includes(source.format)) {
      throw new Error(`unsupported source format for ${source.id}: ${source.format}`);
    }
    for (const field of ["input_path", "expected_sha256", "source_url", "license_id", "rights_basis_url", "attribution"]) {
      if (!source[field]) throw new Error(`source ${source.id} requires ${field}`);
    }
    if (source.rights_status !== "approved") {
      throw new Error(`source ${source.id} is not rights-approved`);
    }
    for (const evidence of source.provenance_files ?? []) {
      if (!evidence.path || !evidence.expected_sha256) {
        throw new Error(`source ${source.id} has incomplete provenance evidence`);
      }
    }
  }
  const splitTotal = Object.values(config.split.permyriad).reduce((sum, value) => sum + value, 0);
  if (splitTotal !== 10_000) throw new Error("split permyriad values must sum to 10000");
  if (config.document.min_bytes < 1 || config.document.max_bytes < config.document.min_bytes) {
    throw new Error("invalid document byte limits");
  }
  if (config.near_dedup.signature_size % config.near_dedup.bands !== 0) {
    throw new Error("near-dedup signature size must divide evenly into bands");
  }
  if (config.tokenizer_training.source_split !== "train") {
    throw new Error("tokenizer training source must be the train split");
  }
}

function makeDocument(source, ordinal, label, text, config) {
  const normalized = normalizeText(text);
  const bytes = Buffer.byteLength(normalized);
  if (bytes < config.document.min_bytes) return null;
  return {
    source,
    ordinal,
    label,
    text: normalized,
    bytes,
    sha256: sha256(normalized),
  };
}

async function readTextSource(source, config) {
  const raw = await readFile(source.input_path, "utf8");
  const cleaned = source.format === "gutenberg_text" ? cleanGutenberg(raw) : normalizeText(raw);
  return chunkUtf8(cleaned, config.document.max_bytes)
    .map((text, index) => makeDocument(source, index, `${source.id}:${index}`, text, config))
    .filter(Boolean)
    .slice(0, source.max_documents ?? Number.POSITIVE_INFINITY);
}

async function readSimpleWikiSource(source, config) {
  const stream = createReadStream(source.input_path, { encoding: "utf8" });
  const lines = createInterface({ input: stream, crlfDelay: Number.POSITIVE_INFINITY });
  const documents = [];
  let inWiki = false;
  let title = null;
  let body = [];

  function flush() {
    if (title === null) return;
    const page = normalizeText(body.join("\n"));
    for (const text of chunkUtf8(page, config.document.max_bytes)) {
      const document = makeDocument(source, documents.length, title, text, config);
      if (document) documents.push(document);
      if (documents.length >= (source.max_documents ?? Number.POSITIVE_INFINITY)) return;
    }
  }

  for await (const line of lines) {
    if (line === "<|source:simplewiki|>") {
      inWiki = true;
      continue;
    }
    if (!inWiki) continue;
    const marker = line.match(/^<\|page:(.*)\|>$/);
    if (marker) {
      flush();
      if (documents.length >= (source.max_documents ?? Number.POSITIVE_INFINITY)) break;
      title = marker[1];
      body = [];
    } else if (title !== null) {
      body.push(line);
    }
  }
  if (documents.length < (source.max_documents ?? Number.POSITIVE_INFINITY)) flush();
  lines.close();
  stream.destroy();
  return documents.slice(0, source.max_documents ?? Number.POSITIVE_INFINITY);
}

async function loadPanels(panelConfigs) {
  const prompts = [];
  const panels = [];
  for (const panel of panelConfigs) {
    const bytes = await readFile(panel.path);
    const hash = sha256(bytes);
    if (panel.expected_sha256 && panel.expected_sha256 !== hash) {
      throw new Error(`contamination panel hash mismatch: ${panel.id}`);
    }
    const text = bytes.toString("utf8");
    const values = panel.format === "open_generation_tsv"
      ? text.trimEnd().split("\n").slice(1).map((line) => Buffer.from(line.split("\t").at(-1), "hex").toString("utf8"))
      : text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    for (const value of values) {
      const normalized = normalizeText(value).toLocaleLowerCase("en-US");
      prompts.push({ panel: panel.id, sha256: sha256(normalized), normalized, shingles: wordShingles(normalized) });
    }
    panels.push({ id: panel.id, path: portablePath(panel.path), sha256: hash, prompts: values.length });
  }
  return { prompts, panels };
}

function exactAndNearDeduplicate(documents, config) {
  const exact = new Map();
  const kept = [];
  const duplicateRecords = [];
  const buckets = new Map();
  const shingleSets = [];
  const bandWidth = config.near_dedup.signature_size / config.near_dedup.bands;

  for (const document of documents) {
    const exactIndex = exact.get(document.sha256);
    if (exactIndex !== undefined) {
      duplicateRecords.push({ kind: "exact", dropped_sha256: document.sha256, kept_sha256: kept[exactIndex].sha256 });
      continue;
    }
    const shingles = wordShingles(document.text, config.near_dedup.shingle_words);
    const signature = bottomSignature(shingles, config.near_dedup.signature_size);
    const keys = [];
    const candidates = new Set();
    for (let band = 0; band < config.near_dedup.bands; band += 1) {
      const values = signature.slice(band * bandWidth, (band + 1) * bandWidth);
      if (values.length !== bandWidth) continue;
      const key = `${band}:${values.join(",")}`;
      keys.push(key);
      for (const candidate of buckets.get(key) ?? []) candidates.add(candidate);
    }
    let duplicateOf = null;
    for (const candidate of [...candidates].sort((left, right) => left - right)) {
      if (isNearDuplicate(shingles, shingleSets[candidate], config.near_dedup.threshold_per_mille)) {
        duplicateOf = candidate;
        break;
      }
    }
    if (duplicateOf !== null) {
      duplicateRecords.push({ kind: "near", dropped_sha256: document.sha256, kept_sha256: kept[duplicateOf].sha256 });
      continue;
    }
    const index = kept.length;
    exact.set(document.sha256, index);
    kept.push(document);
    shingleSets.push(shingles);
    for (const key of keys) {
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(index);
    }
  }
  return { kept, duplicateRecords };
}

function quarantineContamination(documents, panelData, config) {
  const kept = [];
  const quarantine = [];
  for (const document of documents) {
    const normalized = document.text.toLocaleLowerCase("en-US");
    const documentShingles = wordShingles(normalized, config.contamination.shingle_words);
    let match = null;
    for (const prompt of panelData.prompts) {
      const direct = Buffer.byteLength(prompt.normalized) >= config.contamination.min_direct_bytes
        && normalized.includes(prompt.normalized);
      const overlap = prompt.shingles.size > 0
        && intersectionSize(documentShingles, prompt.shingles) * 1000
          >= prompt.shingles.size * config.contamination.prompt_overlap_per_mille;
      if (direct || overlap) {
        match = { panel: prompt.panel, prompt_sha256: prompt.sha256, reason: direct ? "direct" : "shingle_overlap" };
        break;
      }
    }
    if (match) quarantine.push({ document_sha256: document.sha256, source_id: document.source.id, ...match });
    else kept.push(document);
  }
  return { kept, quarantine };
}

function chooseSplit(document, config) {
  const value = Number.parseInt(sha256(`${config.split.seed}\0${document.source.id}\0${document.sha256}`).slice(0, 8), 16) % 10_000;
  const trainEnd = config.split.permyriad.train;
  const devEnd = trainEnd + config.split.permyriad.dev;
  return value < trainEnd ? "train" : value < devEnd ? "dev" : "test";
}

function splitDocuments(documents, config) {
  const splits = { train: [], dev: [], test: [] };
  for (const document of documents) splits[chooseSplit(document, config)].push(document);
  if (config.split.require_nonempty && Object.entries(splits).some(([, values]) => values.length === 0)) {
    throw new Error(`deterministic split produced an empty partition: ${Object.entries(splits).map(([key, value]) => `${key}=${value.length}`).join(", ")}`);
  }
  return splits;
}

function tokenizerSample(documents, limit) {
  const ordered = [...documents].sort((left, right) => left.sha256.localeCompare(right.sha256));
  const parts = [];
  let remaining = limit;
  for (const document of ordered) {
    if (remaining <= 0) break;
    const bytes = Buffer.from(`${document.text}\n\n`, "utf8");
    const part = safeUtf8Prefix(bytes, remaining);
    if (part.length > 0) parts.push(part);
    remaining -= part.length;
  }
  return Buffer.concat(parts);
}

async function build(configPath, outDir) {
  if (!configPath || !outDir) throw new Error("build requires --config and --out-dir");
  const configBytes = await readFile(configPath);
  const config = JSON.parse(configBytes);
  validateConfig(config);
  const sourceEvidence = [];
  const documents = [];
  for (const source of config.sources) {
    const inputHash = await sha256File(source.input_path);
    if (inputHash !== source.expected_sha256) throw new Error(`source hash mismatch: ${source.id}`);
    const provenanceFiles = [];
    for (const evidence of source.provenance_files ?? []) {
      const evidenceHash = await sha256File(evidence.path);
      if (evidenceHash !== evidence.expected_sha256) {
        throw new Error(`source provenance hash mismatch: ${source.id}:${evidence.path}`);
      }
      provenanceFiles.push({ path: portablePath(evidence.path), sha256: evidenceHash });
    }
    const sourceDocuments = source.format === "nsrl_simplewiki_pages"
      ? await readSimpleWikiSource(source, config)
      : await readTextSource(source, config);
    sourceEvidence.push({
      id: source.id,
      path: portablePath(source.input_path),
      sha256: inputHash,
      format: source.format,
      source_url: source.source_url,
      license_id: source.license_id,
      rights_basis_url: source.rights_basis_url,
      attribution: source.attribution,
      rights_scope: source.rights_scope ?? null,
      upstream: source.upstream ?? null,
      provenance_files: provenanceFiles,
      documents_loaded: sourceDocuments.length,
    });
    documents.push(...sourceDocuments);
  }

  const dedup = exactAndNearDeduplicate(documents, config);
  const panels = await loadPanels(config.contamination.panels);
  const contamination = quarantineContamination(dedup.kept, panels, config);
  const splits = splitDocuments(contamination.kept, config);
  const resolvedOut = path.resolve(outDir);
  await mkdir(resolvedOut, { recursive: true });
  const artifactEvidence = {};
  const records = [];
  for (const split of ["train", "dev", "test"]) {
    const parts = [];
    const indexLines = ["schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256"];
    let offset = 0;
    for (let index = 0; index < splits[split].length; index += 1) {
      const document = splits[split][index];
      const documentId = `${document.source.id}:${document.ordinal}:${document.sha256.slice(0, 16)}`;
      const textBytes = Buffer.from(document.text, "utf8");
      parts.push(textBytes);
      indexLines.push([RECORD_SCHEMA, split, documentId, offset, textBytes.length, fnv1a64(textBytes), document.sha256].join("\t"));
      records.push({
        schema: RECORD_SCHEMA,
        document_id: documentId,
        source_id: document.source.id,
        source_label: document.label,
        split,
        corpus_offset: offset,
        bytes: document.bytes,
        fnv64: fnv1a64(textBytes),
        sha256: document.sha256,
        license_id: document.source.license_id,
      });
      offset += textBytes.length;
      if (index + 1 < splits[split].length) {
        parts.push(Buffer.from("\n\n"));
        offset += 2;
      }
    }
    parts.push(Buffer.from("\n"));
    const bytes = Buffer.concat(parts);
    const filePath = path.join(resolvedOut, `${split}.txt`);
    await writeFile(filePath, bytes);
    artifactEvidence[split] = { path: portablePath(filePath), bytes: bytes.length, sha256: sha256(bytes), documents: splits[split].length };
    const indexBytes = Buffer.from(`${indexLines.join("\n")}\n`);
    const indexPath = path.join(resolvedOut, `${split}.index.tsv`);
    await writeFile(indexPath, indexBytes);
    artifactEvidence[`${split}_index`] = { path: portablePath(indexPath), bytes: indexBytes.length, sha256: sha256(indexBytes), documents: splits[split].length };
  }
  const recordsBytes = Buffer.from(`${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
  const recordsPath = path.join(resolvedOut, "records.jsonl");
  await writeFile(recordsPath, recordsBytes);
  const tokenizerBytes = tokenizerSample(splits.train, config.tokenizer_training.max_bytes);
  const tokenizerPath = path.join(resolvedOut, "tokenizer-train.txt");
  await writeFile(tokenizerPath, tokenizerBytes);
  const quarantineBytes = Buffer.from(`${JSON.stringify({
    schema: "nsrl.production_corpus_contamination.v1",
    panels: panels.panels,
    quarantined: contamination.quarantine,
  }, null, 2)}\n`);
  const quarantinePath = path.join(resolvedOut, "contamination.json");
  await writeFile(quarantinePath, quarantineBytes);

  const manifest = {
    schema: MANIFEST_SCHEMA,
    contract_id: config.contract_id,
    config: { path: portablePath(configPath), sha256: sha256(configBytes), canonical_sha256: sha256(stableJson(config)) },
    sources: sourceEvidence,
    policy: {
      document: config.document,
      near_dedup: config.near_dedup,
      contamination: { ...config.contamination, panels: panels.panels },
      split: config.split,
      tokenizer_training: config.tokenizer_training,
    },
    counts: {
      loaded: documents.length,
      exact_duplicates_removed: dedup.duplicateRecords.filter((record) => record.kind === "exact").length,
      near_duplicates_removed: dedup.duplicateRecords.filter((record) => record.kind === "near").length,
      contaminated_quarantined: contamination.quarantine.length,
      accepted: contamination.kept.length,
    },
    artifacts: {
      ...artifactEvidence,
      records: { path: portablePath(recordsPath), bytes: recordsBytes.length, sha256: sha256(recordsBytes), records: records.length },
      tokenizer_training: { path: portablePath(tokenizerPath), bytes: tokenizerBytes.length, sha256: sha256(tokenizerBytes), source_split: "train" },
      contamination: { path: portablePath(quarantinePath), bytes: quarantineBytes.length, sha256: sha256(quarantineBytes), quarantined: contamination.quarantine.length },
    },
    gates: {
      rights_approved_sources: sourceEvidence.length,
      no_cross_split_documents: true,
      tokenizer_training_is_train_only: true,
      evaluation_panels_excluded: true,
      tokenizer_bound: false,
      all_splits_encoded: false,
    },
  };
  const manifestPath = path.join(resolvedOut, "manifest.json");
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({ manifest: portablePath(manifestPath), counts: manifest.counts, splits: artifactEvidence }));
  return manifestPath;
}

async function bindEncoding(manifestPath, split, tokensPath, tracePath) {
  if (!manifestPath || !["train", "dev", "test"].includes(split) || !tokensPath || !tracePath) {
    throw new Error("bind-encoding requires --manifest, --split train|dev|test, --tokens, and --trace");
  }
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  if (!manifest.gates.tokenizer_bound || !manifest.tokenizer) throw new Error("bind the tokenizer before split encodings");
  const tokensBytes = await readFile(tokensPath);
  const traceBytes = await readFile(tracePath);
  const trace = JSON.parse(traceBytes);
  if (tokensBytes.length < 24 || tokensBytes.subarray(0, 8).toString("ascii") !== "NSRLTOK1") {
    throw new Error("invalid token stream header");
  }
  const tokenizerHash = `0x${tokensBytes.readBigUInt64LE(8).toString(16).padStart(16, "0")}`;
  const tokenCount = Number(tokensBytes.readBigUInt64LE(16));
  if (!Number.isSafeInteger(tokenCount) || tokensBytes.length !== 24 + tokenCount * 4) {
    throw new Error("invalid token stream length");
  }
  let tokenHash = 0xcbf29ce484222325n;
  for (let offset = 24; offset < tokensBytes.length; offset += 4) {
    const token = tokensBytes.readUInt32LE(offset);
    if (token >= manifest.tokenizer.actual_vocab_size) throw new Error("out-of-vocabulary token in stream");
    for (let shift = 0; shift < 32; shift += 8) {
      tokenHash ^= BigInt((token >>> shift) & 0xff);
      tokenHash = BigInt.asUintN(64, tokenHash * 0x100000001b3n);
    }
  }
  const tokenHashHex = `0x${tokenHash.toString(16).padStart(16, "0")}`;
  const records = (await readFile(manifest.artifacts.records.path, "utf8"))
    .trimEnd().split("\n").filter(Boolean).map((line) => JSON.parse(line)).filter((record) => record.split === split);
  const inputBytes = records.reduce((sum, record) => sum + record.bytes, 0);
  if (trace.schema !== "nsrl.subword_indexed_encode_trace.v1"
    || trace.tokenizer_hash !== tokenizerHash
    || tokenizerHash !== manifest.tokenizer.artifact_hash_fnv64
    || trace.vocab_size !== manifest.tokenizer.actual_vocab_size
    || trace.documents !== records.length
    || trace.input_bytes !== inputBytes
    || trace.tokens !== tokenCount
    || trace.token_hash !== tokenHashHex
    || trace.bos_tokens !== records.length
    || trace.eos_tokens !== records.length) {
    throw new Error(`token trace does not match ${split} corpus or stream`);
  }
  manifest.encodings ??= {};
  manifest.encodings[split] = {
    documents: trace.documents,
    input_bytes: trace.input_bytes,
    tokens: trace.tokens,
    token_hash_fnv64: trace.token_hash,
    tokens_per_input_byte_per_mille: trace.tokens_per_input_byte_per_mille,
    bos_tokens: trace.bos_tokens,
    eos_tokens: trace.eos_tokens,
  };
  manifest.artifacts[`${split}_tokens`] = { path: portablePath(tokensPath), bytes: tokensBytes.length, sha256: sha256(tokensBytes) };
  manifest.artifacts[`${split}_token_trace`] = { path: portablePath(tracePath), bytes: traceBytes.length, sha256: sha256(traceBytes) };
  manifest.gates.all_splits_encoded = ["train", "dev", "test"].every((name) => manifest.encodings[name]);
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({ schema: "nsrl.production_corpus_encoding_binding.v1", ok: true, split, encoding: manifest.encodings[split] }));
}

async function bindTokenizer(manifestPath, tokenizerPath, tracePath) {
  if (!manifestPath || !tokenizerPath || !tracePath) {
    throw new Error("bind-tokenizer requires --manifest, --tokenizer, and --trace");
  }
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  if (manifest.schema !== MANIFEST_SCHEMA) throw new Error("invalid production corpus manifest");
  const tokenizerBytes = await readFile(tokenizerPath);
  const traceBytes = await readFile(tracePath);
  const trace = JSON.parse(traceBytes);
  if (tokenizerBytes.length < 32 || tokenizerBytes.subarray(0, 8).toString("ascii") !== "NSRLBPE1") {
    throw new Error("invalid tokenizer artifact header");
  }
  const target = tokenizerBytes.readUInt32LE(12);
  const minPairFrequency = tokenizerBytes.readUInt32LE(16);
  const sourceHash = `0x${tokenizerBytes.readBigUInt64LE(20).toString(16).padStart(16, "0")}`;
  const merges = tokenizerBytes.readUInt32LE(28);
  const trainingBytes = await readFile(manifest.artifacts.tokenizer_training.path);
  const policy = manifest.policy.tokenizer_training;
  if (target !== policy.target_vocab_size || minPairFrequency !== policy.min_pair_frequency) {
    throw new Error("tokenizer configuration does not match corpus policy");
  }
  if (sourceHash !== fnv1a64(trainingBytes) || trace.input.hash !== sourceHash || trace.input.bytes !== trainingBytes.length) {
    throw new Error("tokenizer is not bound to the train-only tokenizer corpus");
  }
  if (trace.vocabulary.actual !== target || trace.vocabulary.merges !== merges) {
    throw new Error("tokenizer did not reach the frozen vocabulary target");
  }
  if (trace.artifact.hash !== fnv1a64(tokenizerBytes) || trace.artifact.bytes !== tokenizerBytes.length) {
    throw new Error("tokenizer trace does not match its artifact");
  }
  manifest.tokenizer = {
    id: "deterministic_byte_bpe_v1",
    target_vocab_size: target,
    actual_vocab_size: trace.vocabulary.actual,
    merges,
    min_pair_frequency: minPairFrequency,
    source_hash_fnv64: sourceHash,
    artifact_hash_fnv64: trace.artifact.hash,
    training_tokens: trace.training_encoding.tokens,
    tokens_per_input_byte_per_mille: trace.training_encoding.tokens_per_input_byte_per_mille,
  };
  manifest.artifacts.tokenizer = {
    path: portablePath(tokenizerPath),
    bytes: tokenizerBytes.length,
    sha256: sha256(tokenizerBytes),
  };
  manifest.artifacts.tokenizer_trace = {
    path: portablePath(tracePath),
    bytes: traceBytes.length,
    sha256: sha256(traceBytes),
  };
  manifest.gates.tokenizer_bound = true;
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify({ schema: "nsrl.production_corpus_tokenizer_binding.v1", ok: true, tokenizer: manifest.tokenizer }));
}

async function check(manifestPath) {
  if (!manifestPath) throw new Error("check requires --manifest");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  if (manifest.schema !== MANIFEST_SCHEMA || manifest.contract_id !== "production-corpus-v1") {
    throw new Error("invalid production corpus manifest schema or contract");
  }
  const documentHashes = new Set();
  const splitCounts = { train: 0, dev: 0, test: 0 };
  const recordsText = await readFile(manifest.artifacts.records.path, "utf8");
  const records = recordsText.trimEnd().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  for (const record of records) {
    if (record.schema !== RECORD_SCHEMA || !Object.hasOwn(splitCounts, record.split)) throw new Error("invalid corpus record");
    if (documentHashes.has(record.sha256)) throw new Error(`cross-split or duplicate document hash: ${record.sha256}`);
    documentHashes.add(record.sha256);
    splitCounts[record.split] += 1;
  }
  for (const [name, artifact] of Object.entries(manifest.artifacts)) {
    const bytes = await readFile(artifact.path);
    if (bytes.length !== artifact.bytes || sha256(bytes) !== artifact.sha256) throw new Error(`artifact mismatch: ${name}`);
  }
  for (const split of ["train", "dev", "test"]) {
    if (splitCounts[split] !== manifest.artifacts[split].documents) throw new Error(`split record mismatch: ${split}`);
  }
  if (manifest.counts.accepted !== records.length || manifest.artifacts.contamination.quarantined !== manifest.counts.contaminated_quarantined) {
    throw new Error("manifest count mismatch");
  }
  if (!manifest.gates.no_cross_split_documents || !manifest.gates.tokenizer_training_is_train_only || !manifest.gates.evaluation_panels_excluded) {
    throw new Error("required production corpus gate is false");
  }
  console.log(JSON.stringify({ schema: "nsrl.production_corpus_check.v1", ok: true, manifest: portablePath(manifestPath), records: records.length, splits: splitCounts }));
}

export { bindEncoding, bindTokenizer, build, check, cleanGutenberg, chunkUtf8, exactAndNearDeduplicate, normalizeText, quarantineContamination, sha256, wordShingles };

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.command === "build") await build(options.config, options.outDir);
  else if (options.command === "bind-tokenizer") await bindTokenizer(options.manifest, options.tokenizer, options.trace);
  else if (options.command === "bind-encoding") await bindEncoding(options.manifest, options.split, options.tokens, options.trace);
  else if (options.command === "check") await check(options.manifest);
  else throw new Error("expected build or check");
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(`production-corpus-v1: ${error.message}`);
    process.exit(1);
  });
}
