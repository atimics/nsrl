#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const acquisitionPath = process.argv[2]
  ?? "data/raw/production-multifamily-exchange-v1/acquisition.json";
const parentFramePath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json";
const outputDirectory = process.argv[4]
  ?? "data/processed/solomonic-judgment-v1";
const framePath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-source-frame.json";

const FAMILIES = ["federal_register", "rfc", "science"];
const SOURCE_SELECTION_SEED = "nsrl-m4-solomonic-source-selection-2026-07-15-v1";
const PASSAGE_SEED = "nsrl-m4-solomonic-quartile-passages-2026-07-15-v1";
const SOURCES_PER_FAMILY = {
  federal_register: 2,
  rfc: 2,
  science: 2,
};
const PASSAGES_PER_SOURCE = 4;
const PASSAGE_BYTES = 12_288;
const MINIMUM_CLEANED_BYTES = 65_536;
const PREFIX_SENTINELS = 8;
const ATOMIC_DOCUMENTS = 64;

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const shaKey = (...parts) => sha256(parts.join("\0"));
const normalizeText = (text) => text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n")
  .normalize("NFC").replace(/[\u0000\u000B\u000C\u000E-\u001F\u007F]/g, "")
  .replace(/[\t ]+$/gm, "").replace(/\n{4,}/g, "\n\n\n").trim();
const decodeXml = (text) => text
  .replace(/&#x([0-9a-f]+);/gi, (_, value) => String.fromCodePoint(Number.parseInt(value, 16)))
  .replace(/&#([0-9]+);/g, (_, value) => String.fromCodePoint(Number.parseInt(value, 10)))
  .replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/&lt;/gi, "<")
  .replace(/&gt;/gi, ">").replace(/&quot;/gi, "\"").replace(/&apos;/gi, "'");
const stripTags = (text) => normalizeText(decodeXml(text
  .replace(/<(?:xref|ext-link)[^>]*>/gi, " ").replace(/<\/[^>]+>/g, "\n")
  .replace(/<[^>]+>/g, " ")));
const cleanedBody = (family, text) => {
  if (family === "science") {
    const body = text.match(/<body(?:\s[^>]*)?>([\s\S]*?)<\/body>/i)?.[1] ?? "";
    return stripTags(body.replace(/<ref-list(?:\s[^>]*)?>[\s\S]*$/i, ""));
  }
  return normalizeText(text);
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
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(
    (key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  return JSON.stringify(value);
};
const passageFromStratum = (body, sourceKey, ordinal) => {
  const bytes = Buffer.from(body, "utf8");
  assert(bytes.length >= MINIMUM_CLEANED_BYTES, `source body too short: ${sourceKey}`);
  const stratumStart = Math.floor(bytes.length * ordinal / PASSAGES_PER_SOURCE);
  const stratumEnd = Math.floor(bytes.length * (ordinal + 1) / PASSAGES_PER_SOURCE);
  const available = stratumEnd - stratumStart - PASSAGE_BYTES;
  assert(available > 512, `quartile cannot hold passage: ${sourceKey}:${ordinal}`);
  const digest = crypto.createHash("sha256").update(
    `${PASSAGE_SEED}\0${sourceKey}\0${ordinal}`).digest();
  let start = stratumStart + Number(digest.readBigUInt64BE(0) % BigInt(available));
  while (start < stratumEnd && (bytes[start] & 0xc0) === 0x80) start += 1;
  const whitespace = bytes.subarray(start, Math.min(start + 256, stratumEnd)).findIndex(
    (byte) => byte === 10 || byte === 32);
  if (whitespace >= 0) start += whitespace + 1;
  let end = Math.min(start + PASSAGE_BYTES, stratumEnd);
  while (end > start && (bytes[end] & 0xc0) === 0x80) end -= 1;
  const content = Buffer.from(normalizeText(bytes.subarray(start, end).toString("utf8")), "utf8");
  assert(content.length >= 9_000, `sampled passage too short: ${sourceKey}:${ordinal}`);
  return {
    ordinal, stratum: ordinal + 1, stratum_byte_start: stratumStart,
    stratum_byte_end: stratumEnd, byte_offset: start, bytes: content.length,
    sha256: sha256(content), content,
  };
};

const acquisitionBytes = fs.readFileSync(acquisitionPath);
const parentFrameBytes = fs.readFileSync(parentFramePath);
const acquisition = JSON.parse(acquisitionBytes);
const parentFrame = JSON.parse(parentFrameBytes);
assert(acquisition.schema === "nsrl.production_multifamily_acquisition.v1"
  && acquisition.acquisition_only_no_model_outcomes === true,
"wrong acquisition manifest");
assert(parentFrame.schema === "nsrl.production_multifamily_exchange_source_frame.v1",
  "wrong parent source frame");

const parentIds = new Set(parentFrame.sources.map((source) => source.source_id));
const parentKeys = new Set(parentFrame.sources.map(
  (source) => `${source.family}\0${source.independence_key}`));
const selectedSources = [];
for (const family of FAMILIES) {
  const eligible = acquisition.records.filter((record) => record.family === family
    && ["cached", "downloaded"].includes(record.status)
    && record.cleaned_body_bytes >= MINIMUM_CLEANED_BYTES
    && !parentIds.has(record.source_id)
    && !parentKeys.has(`${family}\0${record.independence_key}`))
    .map((record) => {
      const raw = fs.readFileSync(record.path);
      assert(sha256(raw) === record.sha256, `raw source hash changed: ${record.source_id}`);
      const body = cleanedBody(family, raw.toString("utf8"));
      assert(Buffer.byteLength(body) === record.cleaned_body_bytes,
        `cleaned source length changed: ${record.source_id}`);
      return {...record, body};
    }).sort((left, right) => shaKey(
      SOURCE_SELECTION_SEED, family, left.source_id, left.sha256).localeCompare(shaKey(
      SOURCE_SELECTION_SEED, family, right.source_id, right.sha256)));
  const target = SOURCES_PER_FAMILY[family];
  assert(eligible.length >= target, `${family} has too few untouched sources`);
  const chosen = eligible.slice(0, target);
  assert(new Set(chosen.map((source) => source.independence_key)).size === chosen.length,
    `${family} selected independence keys repeat`);
  for (const record of chosen) {
    const passages = Array.from({length: PASSAGES_PER_SOURCE}, (_, ordinal) =>
      passageFromStratum(record.body, `${family}\0${record.source_id}\0${record.sha256}`, ordinal));
    selectedSources.push({...record, passages});
  }
}
assert(selectedSources.length === Object.values(SOURCES_PER_FAMILY).reduce((sum, value) => sum + value, 0),
  "wrong untouched source count");

const orderedSources = selectedSources.sort((left, right) => left.family.localeCompare(right.family)
  || left.source_id.localeCompare(right.source_id));
const evaluationDocuments = orderedSources.flatMap((source) => source.passages.map((passage) => ({
  source_id: source.source_id, passage_ordinal: passage.ordinal, family: source.family,
  content: passage.content, analysis_role: "untouched_evaluation",
})));
const implementationDocuments = [];
for (let index = 0; index < PREFIX_SENTINELS; index += 1) {
  const source = orderedSources[index % orderedSources.length];
  const passage = source.passages[index % PASSAGES_PER_SOURCE];
  implementationDocuments.push({
    source_id: source.source_id, passage_ordinal: passage.ordinal, family: source.family,
    content: passage.content, analysis_role: "implementation_prefix_sentinel",
  });
}
implementationDocuments.push(...evaluationDocuments);
for (let index = evaluationDocuments.length; index < ATOMIC_DOCUMENTS; index += 1) {
  const source = orderedSources[index % orderedSources.length];
  const passage = source.passages[(index + 1) % PASSAGES_PER_SOURCE];
  implementationDocuments.push({
    source_id: source.source_id, passage_ordinal: passage.ordinal, family: source.family,
    content: passage.content, analysis_role: "implementation_padding_excluded",
  });
}
assert(implementationDocuments.length === PREFIX_SENTINELS + ATOMIC_DOCUMENTS,
  "atomic corpus must contain 72 documents");

fs.mkdirSync(outputDirectory, {recursive: true});
const corpusParts = [];
const indexRows = ["schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256"];
const documentBindings = [];
let offset = 0;
for (const document of implementationDocuments) {
  const separator = corpusParts.length === 0 ? Buffer.alloc(0) : Buffer.from("\n\n");
  offset += separator.length;
  corpusParts.push(separator, document.content);
  const digest = sha256(document.content);
  const documentId = `${document.source_id}-passage-${document.passage_ordinal}`
    + `:${document.passage_ordinal}:${digest.slice(0, 16)}`;
  indexRows.push(["nsrl.production_corpus_record.v1", "dev", documentId, offset,
    document.content.length, fnv64(document.content), digest].join("\t"));
  documentBindings.push({
    document: documentBindings.length, document_id: documentId,
    source_id: document.source_id, passage_ordinal: document.passage_ordinal,
    family: document.family, analysis_role: document.analysis_role, sha256: digest,
  });
  offset += document.content.length;
}
const corpus = Buffer.concat(corpusParts);
const index = Buffer.from(`${indexRows.join("\n")}\n`);
const corpusPath = path.join(outputDirectory, "evaluation.txt");
const indexPath = path.join(outputDirectory, "evaluation.index.tsv");
fs.writeFileSync(corpusPath, corpus);
fs.writeFileSync(indexPath, index);

const publicSources = orderedSources.map(({body, passages, ...source}) => ({
  ...source,
  passages: passages.map(({content, ...passage}) => passage),
}));
const frame = {
  schema: "nsrl.solomonic_judgment_source_frame.v1",
  analysis_role: "prospective_pre_outcome_source_frame",
  source_selection_seed: SOURCE_SELECTION_SEED,
  passage_sampling_seed: PASSAGE_SEED,
  parent_acquisition: {path: acquisitionPath, sha256: sha256(acquisitionBytes)},
  excluded_parent_frame: {path: parentFramePath, sha256: sha256(parentFrameBytes)},
  population: {
    families: FAMILIES, sources_per_family: SOURCES_PER_FAMILY,
    source_unit: "one complete publication absent by source id and independence key from M18",
    intended_population:
      "the cached eligible Federal Register, RFC, and Europe PMC acquisition surfaces only",
    independence_design:
      "distinct most-specific agency, first-listed RFC author, or first-author/journal key",
  },
  panel_sampling: {
    passages_per_source: PASSAGES_PER_SOURCE,
    passage_bytes_before_normalization: PASSAGE_BYTES,
    strata: "four byte quartiles with one hash-selected nonoverlapping passage per quartile",
    model_windows_per_passage: 2,
    context_tokens: 64,
  },
  sources: publicSources,
  execution: {
    corpus_path: corpusPath, corpus_sha256: sha256(corpus),
    index_path: indexPath, index_sha256: sha256(index),
    document_start: PREFIX_SENTINELS, atomic_documents: ATOMIC_DOCUMENTS,
    evaluation_documents: evaluationDocuments.length,
    excluded_padding_documents: ATOMIC_DOCUMENTS - evaluationDocuments.length,
    hard_stop_before_document: PREFIX_SENTINELS + ATOMIC_DOCUMENTS,
    document_bindings: documentBindings,
  },
  outcome_firewall: {
    action_cube_outcomes_read: false,
    selected_sources_have_no_prior_action_cube_outcome_in_m18: true,
    original_gutenberg_fitting_frame_used: false,
    documents_200_212_read: false,
  },
};
frame.frame_fingerprint_sha256 = sha256(stableJson({
  source_selection_seed: SOURCE_SELECTION_SEED, passage_sampling_seed: PASSAGE_SEED,
  parent_acquisition_sha256: frame.parent_acquisition.sha256,
  excluded_parent_frame_sha256: frame.excluded_parent_frame.sha256,
  sources: publicSources, execution: frame.execution,
}));
fs.mkdirSync(path.dirname(framePath), {recursive: true});
fs.writeFileSync(framePath, `${JSON.stringify(frame, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: frame.schema, frame: framePath, families: FAMILIES,
  selected_source_ids: publicSources.map((source) => source.source_id),
  evaluation_documents: evaluationDocuments.length,
  frame_fingerprint_sha256: frame.frame_fingerprint_sha256,
  action_cube_outcomes_read: false, original_gutenberg_fitting_frame_used: false,
}, null, 2)}\n`);
