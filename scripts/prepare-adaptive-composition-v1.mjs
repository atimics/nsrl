#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const acquisitionPath = process.argv[2]
  ?? "data/raw/p10m-adaptive-composition-v1/acquisition.json";
const outputDirectory = process.argv[3]
  ?? "data/processed/p10m-adaptive-composition-v1";
const framePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-adaptive-composition-v1-source-frame.json";
const exclusionPaths = (process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json,benchmarks/production-model-v1/p10m-solomonic-judgment-v1-source-frame.json")
  .split(",").filter(Boolean);

const FAMILIES = ["federal_register", "rfc", "science"];
const ROLE_COUNTS = {fitting: 12, calibration: 119, adaptive: 2, endpoint: 19};
const SOURCES_PER_FAMILY = Object.values(ROLE_COUNTS).reduce((sum, count) => sum + count, 0);
const SOURCE_SELECTION_SEED = "nsrl-m5-adaptive-source-selection-2026-07-15-v1";
const ROLE_SEED = "nsrl-m5-adaptive-role-permutation-2026-07-15-v1";
const ADAPTIVE_ORDER_SEED = "nsrl-m5-adaptive-panel-order-2026-07-15-v1";
const PASSAGE_SEED = "nsrl-m5-adaptive-quartile-passages-2026-07-15-v1";
const PASSAGES_PER_SOURCE = 4;
const PASSAGE_BYTES = 12_288;
const MINIMUM_CLEANED_BYTES = 65_536;

const assert = (condition, message) => {
  if (!condition) throw new Error(`adaptive composition prepare: ${message}`);
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
  if (family !== "science") return normalizeText(text);
  const body = text.match(/<body(?:\s[^>]*)?>([\s\S]*?)<\/body>/i)?.[1] ?? "";
  return stripTags(body.replace(/<ref-list(?:\s[^>]*)?>[\s\S]*$/i, ""));
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
const acquisition = JSON.parse(acquisitionBytes);
assert(acquisition.schema === "nsrl.production_multifamily_acquisition.v1"
  && acquisition.acquisition_only_no_model_outcomes === true
  && acquisition.independence_key_mode === "whole_publication",
"wrong acquisition manifest");
assert(FAMILIES.every((family) => acquisition.active_families.includes(family)
    && acquisition.family_summary[family].eligible >= SOURCES_PER_FAMILY),
"acquisition does not contain 152 eligible sources per family");

const exclusionBindings = exclusionPaths.map((exclusionPath) => {
  const bytes = fs.readFileSync(exclusionPath);
  const frame = JSON.parse(bytes);
  assert(Array.isArray(frame.sources), `exclusion frame has no sources: ${exclusionPath}`);
  return {path: exclusionPath, sha256: sha256(bytes), frame};
});
const excludedIds = new Set(exclusionBindings.flatMap(({frame}) =>
  frame.sources.map((source) => source.source_id)));
const excludedKeys = new Set(exclusionBindings.flatMap(({frame}) =>
  frame.sources.map((source) => `${source.family}\0${source.independence_key}`)));

const selectedSources = [];
for (const family of FAMILIES) {
  const eligible = acquisition.records.filter((record) => record.family === family
    && ["cached", "downloaded"].includes(record.status)
    && record.cleaned_body_bytes >= MINIMUM_CLEANED_BYTES
    && !excludedIds.has(record.source_id)
    && !excludedKeys.has(`${family}\0${record.independence_key}`))
    .sort((left, right) => shaKey(SOURCE_SELECTION_SEED, family, left.source_id, left.sha256)
      .localeCompare(shaKey(SOURCE_SELECTION_SEED, family, right.source_id, right.sha256)))
    .slice(0, SOURCES_PER_FAMILY);
  assert(eligible.length === SOURCES_PER_FAMILY, `${family} fresh source count changed`);
  assert(new Set(eligible.map((source) => source.source_id)).size === SOURCES_PER_FAMILY
    && new Set(eligible.map((source) => source.independence_key)).size === SOURCES_PER_FAMILY,
  `${family} source identities repeat`);
  const roleOrdered = [...eligible].sort((left, right) =>
    shaKey(ROLE_SEED, family, left.source_id, left.sha256)
      .localeCompare(shaKey(ROLE_SEED, family, right.source_id, right.sha256)));
  for (const [index, record] of roleOrdered.entries()) {
    const role = index < ROLE_COUNTS.fitting ? "fitting"
      : index < ROLE_COUNTS.fitting + ROLE_COUNTS.calibration ? "calibration"
        : index < ROLE_COUNTS.fitting + ROLE_COUNTS.calibration + ROLE_COUNTS.adaptive
          ? "adaptive" : "endpoint";
    const raw = fs.readFileSync(record.path);
    assert(sha256(raw) === record.sha256, `raw source hash changed: ${record.source_id}`);
    const body = cleanedBody(family, raw.toString("utf8"));
    assert(Buffer.byteLength(body) === record.cleaned_body_bytes,
      `cleaned source length changed: ${record.source_id}`);
    const passages = Array.from({length: PASSAGES_PER_SOURCE}, (_, ordinal) =>
      passageFromStratum(body, `${family}\0${record.source_id}\0${record.sha256}`, ordinal));
    for (let ordinal = 1; ordinal < passages.length; ordinal += 1) {
      assert(passages[ordinal - 1].byte_offset + passages[ordinal - 1].bytes
        <= passages[ordinal].byte_offset, `passages overlap: ${record.source_id}`);
    }
    selectedSources.push({...record, role, passages});
  }
}

fs.mkdirSync(outputDirectory, {recursive: true});
const writeRoleCorpus = (role) => {
  const roleSources = selectedSources.filter((source) => source.role === role)
    .sort((left, right) => role === "adaptive"
      ? shaKey(ADAPTIVE_ORDER_SEED, left.family, left.source_id, left.sha256)
        .localeCompare(shaKey(ADAPTIVE_ORDER_SEED, right.family, right.source_id, right.sha256))
      : left.family.localeCompare(right.family) || left.source_id.localeCompare(right.source_id));
  const documents = roleSources
    .flatMap((source) => source.passages.map((passage) => ({source, passage})));
  const parts = [];
  const rows = ["schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256"];
  const bindings = [];
  const panelRows = ["document\tfamily\tsource_id\tindependence_key\tpassage_ordinal\trole\tsha256"];
  let offset = 0;
  for (const {source, passage} of documents) {
    const separator = parts.length === 0 ? Buffer.alloc(0) : Buffer.from("\n\n");
    offset += separator.length;
    parts.push(separator, passage.content);
    const documentId = `${source.source_id}:${passage.ordinal}:${passage.sha256.slice(0, 16)}`;
    rows.push(["nsrl.production_corpus_record.v1", "dev", documentId, offset,
      passage.content.length, fnv64(passage.content), passage.sha256].join("\t"));
    bindings.push({document: bindings.length, document_id: documentId,
      source_id: source.source_id, independence_key: source.independence_key,
      family: source.family, role, passage_ordinal: passage.ordinal, sha256: passage.sha256});
    panelRows.push([bindings.length - 1, source.family, source.source_id,
      source.independence_key, passage.ordinal, role, passage.sha256].join("\t"));
    offset += passage.content.length;
  }
  const corpus = Buffer.concat(parts);
  const index = Buffer.from(`${rows.join("\n")}\n`);
  const corpusPath = path.join(outputDirectory, `${role}.txt`);
  const indexPath = path.join(outputDirectory, `${role}.index.tsv`);
  const panelPath = path.join(outputDirectory, `${role}.panels.tsv`);
  fs.writeFileSync(corpusPath, corpus);
  fs.writeFileSync(indexPath, index);
  fs.writeFileSync(panelPath, `${panelRows.join("\n")}\n`);
  return {role, corpus_path: corpusPath, corpus_sha256: sha256(corpus),
    index_path: indexPath, index_sha256: sha256(index), documents: documents.length,
    panel_path: panelPath, panel_sha256: sha256(Buffer.from(`${panelRows.join("\n")}\n`)),
    document_bindings: bindings};
};
const phases = Object.fromEntries(Object.keys(ROLE_COUNTS).map(
  (role) => [role, writeRoleCorpus(role)]));
for (const [role, count] of Object.entries(ROLE_COUNTS)) {
  assert(phases[role].documents === count * FAMILIES.length * PASSAGES_PER_SOURCE,
    `${role} document count changed`);
}

const publicSources = selectedSources.map(({passages, ...source}) => ({...source,
  passages: passages.map(({content, ...passage}) => passage)}));
const frame = {
  schema: "nsrl.adaptive_composition_source_frame.v1",
  analysis_role: "prospective_pre_outcome_source_frame",
  acquisition: {path: acquisitionPath, sha256: sha256(acquisitionBytes)},
  exclusions: exclusionBindings.map(({path: bindingPath, sha256: digest}) =>
    ({path: bindingPath, sha256: digest})),
  source_design: {
    families: FAMILIES, unit: "one whole publication sampled without replacement",
    independence_key: "family-qualified whole-publication source ID",
    sources_per_family: SOURCES_PER_FAMILY, role_counts_per_family: ROLE_COUNTS,
    source_selection_seed: SOURCE_SELECTION_SEED, role_permutation_seed: ROLE_SEED,
    adaptive_order_seed: ADAPTIVE_ORDER_SEED,
    exchangeability_design:
      "roles are assigned by a frozen hash permutation within each family before model outcomes",
  },
  panel_sampling: {
    seed: PASSAGE_SEED, passages_per_source: PASSAGES_PER_SOURCE,
    passage_bytes_before_normalization: PASSAGE_BYTES,
    strata: "four source-body byte quartiles; one hash-selected nonoverlapping passage per quartile",
    model_windows_per_passage: 2, context_tokens: 64,
  },
  sources: publicSources,
  phases,
  outcome_firewall: {
    fitting_outcomes_read: false, calibration_outcomes_read: false,
    adaptive_outcomes_read: false, endpoint_outcomes_read: false,
    all_m18_m19_source_ids_and_independence_keys_excluded: true,
  },
};
frame.frame_fingerprint_sha256 = sha256(stableJson({
  acquisition: frame.acquisition, exclusions: frame.exclusions,
  source_design: frame.source_design, panel_sampling: frame.panel_sampling,
  sources: frame.sources, phases: frame.phases,
}));
fs.mkdirSync(path.dirname(framePath), {recursive: true});
fs.writeFileSync(framePath, `${JSON.stringify(frame, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: frame.schema, frame: framePath, sources: frame.sources.length,
  roles: Object.fromEntries(Object.entries(phases).map(([role, phase]) =>
    [role, phase.documents / PASSAGES_PER_SOURCE])),
  frame_fingerprint_sha256: frame.frame_fingerprint_sha256,
  all_evaluation_outcomes_read: false,
}, null, 2)}\n`);
