#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const acquisitionPath = process.argv[2]
  ?? "data/raw/production-cross-source-exchange-v1/acquisition.json";
const outputDirectory = process.argv[3]
  ?? "data/processed/production-cross-source-exchange-v1";
const framePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const acquisition = JSON.parse(fs.readFileSync(acquisitionPath, "utf8"));
const SOURCE_SELECTION_SEED = "nsrl-m3-source-selection-2026-07-15-v1";
const ROLE_SEED = "nsrl-m3-source-role-permutation-2026-07-15-v1";
const PANEL_SEED = "nsrl-m3-source-panel-passage-2026-07-15-v1";
const FITTING_SOURCES = 16;
const CALIBRATION_SOURCES = 39;
const EVALUATION_SOURCES = 16;
const PANEL_BYTES = 16_384;
const PREFIX_SENTINELS = 8;
const ATOMIC_DOCUMENTS = 64;

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const shaKey = (...parts) => sha256(parts.join("\0"));
const authorKey = (author) => author.normalize("NFKD").toLowerCase()
  .replace(/[^a-z0-9]+/g, " ").trim();
const normalizeText = (text) => text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n")
  .normalize("NFC").replace(/[\u0000\u000B\u000C\u000E-\u001F\u007F]/g, "")
  .replace(/[\t ]+$/gm, "").replace(/\n{4,}/g, "\n\n\n").trim();
const stripGutenberg = (text) => {
  let body = text;
  const start = body.search(/^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (start >= 0) {
    const lineEnd = body.indexOf("\n", start);
    body = lineEnd >= 0 ? body.slice(lineEnd + 1) : "";
  }
  const end = body.search(/^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (end >= 0) body = body.slice(0, end);
  return normalizeText(body);
};
const fnv64 = (bytes) => {
  let value = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    value ^= BigInt(byte);
    value = BigInt.asUintN(64, value * 0x100000001b3n);
  }
  return `0x${value.toString(16).padStart(16, "0")}`;
};
const stableJson = (value) => {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
};
const safePanel = (body, key, ordinal = 0) => {
  const bytes = Buffer.from(body, "utf8");
  assert(bytes.length >= PANEL_BYTES * 2, `source body too short for frozen panel: ${key}`);
  const digest = crypto.createHash("sha256").update(`${PANEL_SEED}\0${key}\0${ordinal}`).digest();
  const available = bytes.length - PANEL_BYTES;
  let start = Number(digest.readBigUInt64BE(0) % BigInt(available));
  while (start < bytes.length && (bytes[start] & 0xc0) === 0x80) start += 1;
  const whitespace = bytes.subarray(start, Math.min(start + 256, bytes.length)).findIndex(
    (byte) => byte === 10 || byte === 32);
  if (whitespace >= 0) start += whitespace + 1;
  let end = Math.min(start + PANEL_BYTES, bytes.length);
  while (end > start && (bytes[end] & 0xc0) === 0x80) end -= 1;
  const panel = Buffer.from(normalizeText(bytes.subarray(start, end).toString("utf8")), "utf8");
  assert(panel.length >= 12_000, `sampled panel too short: ${key}`);
  return {bytes: panel, byte_offset: start, sha256: sha256(panel)};
};

assert(acquisition.schema === "nsrl.production_cross_source_acquisition.v1"
  && acquisition.acquisition_only_no_model_outcomes === true,
"wrong acquisition manifest");
const eligible = acquisition.records.filter((record) =>
  ["cached", "downloaded"].includes(record.status)
  && record.language === "English"
  && record.author && authorKey(record.author) !== "unknown")
  .map((record) => {
    const raw = fs.readFileSync(record.path);
    assert(sha256(raw) === record.sha256, `acquired source hash changed: ${record.ebook_id}`);
    const body = stripGutenberg(raw.toString("utf8"));
    return {...record, body, cleaned_body_bytes: Buffer.byteLength(body)};
  })
  .filter((record) => record.cleaned_body_bytes >= PANEL_BYTES * 2);
const byAuthor = new Map();
for (const record of eligible.sort((left, right) => left.ebook_id - right.ebook_id)) {
  const key = authorKey(record.author);
  if (!byAuthor.has(key)) byAuthor.set(key, record);
}
const distinct = [...byAuthor.entries()].map(([key, record]) => ({...record, author_key: key}));
const required = FITTING_SOURCES + CALIBRATION_SOURCES + EVALUATION_SOURCES;
assert(distinct.length >= required, `need ${required} distinct eligible authors, found ${distinct.length}`);
const selected = distinct.sort((left, right) => shaKey(
  SOURCE_SELECTION_SEED, left.ebook_id, left.sha256).localeCompare(shaKey(
  SOURCE_SELECTION_SEED, right.ebook_id, right.sha256))).slice(0, required);
const roleOrdered = selected.sort((left, right) => shaKey(
  ROLE_SEED, left.ebook_id, left.sha256).localeCompare(shaKey(
  ROLE_SEED, right.ebook_id, right.sha256)));
const roleForIndex = (index) => index < FITTING_SOURCES ? "fitting"
  : index < FITTING_SOURCES + CALIBRATION_SOURCES ? "calibration" : "evaluation";
const sources = roleOrdered.map((record, index) => {
  const body = record.body;
  const panel = safePanel(body, `${record.ebook_id}\0${record.sha256}`);
  return {
    source_id: `gutenberg-${record.ebook_id}`,
    ebook_id: record.ebook_id,
    title: record.title,
    author: record.author,
    author_key: record.author_key,
    role: roleForIndex(index),
    source_url: record.url,
    raw_sha256: record.sha256,
    raw_bytes: record.bytes,
    cleaned_body_bytes: record.cleaned_body_bytes,
    panel_byte_offset: panel.byte_offset,
    panel_bytes: panel.bytes.length,
    panel_sha256: panel.sha256,
    panel,
    body,
  };
});
assert(new Set(sources.map((source) => source.author_key)).size === required,
  "selected source authors are not distinct");

fs.mkdirSync(outputDirectory, {recursive: true});
const writePhase = (phase, roleSources, paddingCount) => {
  const documents = [];
  for (let index = 0; index < PREFIX_SENTINELS; index += 1) {
    const source = sources[index % sources.length];
    documents.push({
      source_id: `prefix-sentinel-${String(index).padStart(2, "0")}`,
      ordinal: 0,
      bytes: safePanel(source.body, `${source.source_id}\0prefix`, index + 1).bytes,
      analysis_role: "implementation_prefix_sentinel",
    });
  }
  for (const source of roleSources) {
    documents.push({
      source_id: source.source_id,
      ordinal: 0,
      bytes: source.panel.bytes,
      analysis_role: source.role,
    });
  }
  for (let index = 0; index < paddingCount; index += 1) {
    const source = roleSources[index % roleSources.length];
    documents.push({
      source_id: `excluded-padding-${phase}-${String(index).padStart(2, "0")}`,
      ordinal: index + 1,
      bytes: safePanel(source.body, `${source.source_id}\0${phase}\0padding`, index + 1).bytes,
      analysis_role: "implementation_padding_excluded_from_analysis",
    });
  }
  assert(documents.length === PREFIX_SENTINELS + ATOMIC_DOCUMENTS,
    `${phase} phase must contain exactly 72 documents`);
  const corpusParts = [];
  const indexRows = ["schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256"];
  const documentBindings = [];
  let offset = 0;
  for (const document of documents) {
    const separator = corpusParts.length === 0 ? Buffer.alloc(0) : Buffer.from("\n\n");
    offset += separator.length;
    corpusParts.push(separator, document.bytes);
    const digest = sha256(document.bytes);
    const documentId = `${document.source_id}:${document.ordinal}:${digest.slice(0, 16)}`;
    indexRows.push([
      "nsrl.production_corpus_record.v1", "dev", documentId, offset,
      document.bytes.length, fnv64(document.bytes), digest,
    ].join("\t"));
    documentBindings.push({
      document: documentBindings.length,
      document_id: documentId,
      source_id: document.source_id,
      analysis_role: document.analysis_role,
      sha256: digest,
    });
    offset += document.bytes.length;
  }
  const corpus = Buffer.concat(corpusParts);
  const index = Buffer.from(`${indexRows.join("\n")}\n`);
  const corpusPath = path.join(outputDirectory, `${phase}.txt`);
  const indexPath = path.join(outputDirectory, `${phase}.index.tsv`);
  fs.writeFileSync(corpusPath, corpus);
  fs.writeFileSync(indexPath, index);
  return {
    phase,
    corpus_path: corpusPath,
    corpus_sha256: sha256(corpus),
    index_path: indexPath,
    index_sha256: sha256(index),
    document_start: PREFIX_SENTINELS,
    atomic_documents: ATOMIC_DOCUMENTS,
    role_documents: roleSources.length,
    excluded_padding_documents: paddingCount,
    document_bindings: documentBindings,
  };
};

const fitting = sources.filter((source) => source.role === "fitting");
const calibration = sources.filter((source) => source.role === "calibration");
const evaluation = sources.filter((source) => source.role === "evaluation");
assert(fitting.length === FITTING_SOURCES && calibration.length === CALIBRATION_SOURCES
  && evaluation.length === EVALUATION_SOURCES, "source role counts changed");
const fittingPhase = writePhase("fitting", fitting, ATOMIC_DOCUMENTS - fitting.length);
const confirmationPhase = writePhase(
  "calibration-evaluation", [...calibration, ...evaluation],
  ATOMIC_DOCUMENTS - calibration.length - evaluation.length);

const publicSources = sources.map(({panel, body, ...source}) => source);
const frame = {
  schema: "nsrl.production_cross_source_exchange_source_frame.v1",
  analysis_role: "prospective_pre_outcome_source_frame",
  acquisition_manifest_sha256: sha256(fs.readFileSync(acquisitionPath)),
  source_definition: {
    unit: "one Project Gutenberg ebook from one distinct normalized author",
    candidate_ebook_ids_frozen_in_acquisition_script: true,
    eligibility: {
      language: "English",
      author_required: true,
      unknown_author_excluded: true,
      minimum_raw_bytes: PANEL_BYTES * 2,
      one_lowest_ebook_id_per_normalized_author_before_hash_selection: true,
    },
    selection_seed: SOURCE_SELECTION_SEED,
    selected_sources: required,
    distinct_authors_required: true,
    intended_population:
      "the frozen distinct-author English Project Gutenberg acquisition frame; no claim to arbitrary web or SimpleWiki sources",
  },
  role_partition: {
    seed: ROLE_SEED,
    algorithm: "ascending SHA256(seed, ebook_id, raw_sha256)",
    fitting_sources: FITTING_SOURCES,
    calibration_sources: CALIBRATION_SOURCES,
    evaluation_sources: EVALUATION_SOURCES,
    entire_sources_disjoint_across_roles: true,
  },
  panel_sampling: {
    seed: PANEL_SEED,
    panel_documents_per_source: 1,
    sampled_utf8_bytes_before_normalization: PANEL_BYTES,
    model_windows_per_panel_document: 2,
    model_context_tokens: 64,
    byte_offset_algorithm: "SHA256(seed, ebook_id, raw_sha256, ordinal) modulo available cleaned body bytes, advanced to UTF-8 and whitespace boundary",
    missing_source_policy: "abort before outcome evaluation; no source replacement after freeze",
  },
  sources: publicSources,
  phases: {
    fitting: fittingPhase,
    calibration_evaluation: confirmationPhase,
  },
  outcome_firewall: {
    action_cube_outcomes_read: false,
    documents_200_212_read: false,
    source_content_and_tokenization_metadata_are_pre_outcome_features: true,
  },
  frame_fingerprint_sha256: sha256(stableJson({
    source_selection_seed: SOURCE_SELECTION_SEED,
    role_seed: ROLE_SEED,
    panel_seed: PANEL_SEED,
    sources: publicSources,
    phases: {fitting: fittingPhase, calibration_evaluation: confirmationPhase},
  })),
};
fs.mkdirSync(path.dirname(framePath), {recursive: true});
fs.writeFileSync(framePath, `${JSON.stringify(frame, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: frame.schema,
  frame: framePath,
  source_counts: frame.role_partition,
  frame_fingerprint_sha256: frame.frame_fingerprint_sha256,
  action_cube_outcomes_read: false,
  documents_200_212_read: false,
}, null, 2)}\n`);
