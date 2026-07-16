#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const acquisitionPath = process.argv[2]
  ?? "data/raw/production-multifamily-exchange-v1/acquisition.json";
const outputDirectory = process.argv[3]
  ?? "data/processed/production-multifamily-exchange-v1";
const framePath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json";
const FAMILIES = ["federal_register", "gutenberg", "rfc", "science"];
const SOURCE_SELECTION_SEED = "nsrl-m4-multifamily-source-selection-2026-07-15-v1";
const ROLE_SEED = "nsrl-m4-multifamily-role-permutation-2026-07-15-v1";
const PASSAGE_SEED = "nsrl-m4-quartile-passage-sampling-2026-07-15-v1";
const SOURCES_PER_FAMILY = 26;
const FITTING_PER_FAMILY = 3;
const CALIBRATION_PER_FAMILY = 19;
const EVALUATION_PER_FAMILY = 4;
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
  if (family === "gutenberg") {
    let body = text;
    const start = body.search(/^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
    if (start >= 0) body = body.slice(body.indexOf("\n", start) + 1);
    const end = body.search(/^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
    if (end >= 0) body = body.slice(0, end);
    return normalizeText(body);
  }
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
  const passage = Buffer.from(normalizeText(bytes.subarray(start, end).toString("utf8")), "utf8");
  assert(passage.length >= 9_000, `sampled passage too short: ${sourceKey}:${ordinal}`);
  return {
    ordinal, stratum: ordinal + 1, stratum_byte_start: stratumStart,
    stratum_byte_end: stratumEnd, byte_offset: start, bytes: passage.length,
    sha256: sha256(passage), content: passage,
  };
};

const acquisitionBytes = fs.readFileSync(acquisitionPath);
const acquisition = JSON.parse(acquisitionBytes);
assert(acquisition.schema === "nsrl.production_multifamily_acquisition.v1"
  && acquisition.acquisition_only_no_model_outcomes === true,
"wrong prospective acquisition manifest");
assert(acquisition.documents_200_212_read === false, "sealed documents were read during acquisition");
const sources = [];
for (const family of FAMILIES) {
  const eligible = acquisition.records.filter((record) => record.family === family
    && ["cached", "downloaded"].includes(record.status))
    .map((record) => {
      const raw = fs.readFileSync(record.path);
      assert(sha256(raw) === record.sha256, `acquired source hash changed: ${record.source_id}`);
      const body = cleanedBody(family, raw.toString("utf8"));
      assert(Buffer.byteLength(body) === record.cleaned_body_bytes,
        `cleaned source length changed: ${record.source_id}`);
      return {...record, body};
    });
  assert(new Set(eligible.map((record) => record.independence_key)).size === eligible.length,
    `${family} acquisition independence keys repeat`);
  const selected = eligible.sort((left, right) => shaKey(
    SOURCE_SELECTION_SEED, family, left.source_id, left.sha256).localeCompare(shaKey(
    SOURCE_SELECTION_SEED, family, right.source_id, right.sha256))).slice(0, SOURCES_PER_FAMILY);
  assert(selected.length === SOURCES_PER_FAMILY, `${family} source frame is undersized`);
  const roleOrdered = selected.sort((left, right) => shaKey(
    ROLE_SEED, family, left.source_id, left.sha256).localeCompare(shaKey(
    ROLE_SEED, family, right.source_id, right.sha256)));
  for (const [index, record] of roleOrdered.entries()) {
    const role = index < FITTING_PER_FAMILY ? "fitting"
      : index < FITTING_PER_FAMILY + CALIBRATION_PER_FAMILY ? "calibration" : "evaluation";
    const passages = Array.from({length: PASSAGES_PER_SOURCE}, (_, ordinal) =>
      passageFromStratum(record.body, `${family}\0${record.source_id}\0${record.sha256}`, ordinal));
    for (let ordinal = 1; ordinal < passages.length; ordinal += 1) {
      assert(passages[ordinal - 1].byte_offset + passages[ordinal - 1].bytes
        <= passages[ordinal].byte_offset, `passages overlap: ${record.source_id}`);
    }
    sources.push({...record, role, passages});
  }
}
assert(sources.length === FAMILIES.length * SOURCES_PER_FAMILY, "source total changed");

const fitting = sources.filter((source) => source.role === "fitting")
  .sort((left, right) => left.family.localeCompare(right.family)
    || left.source_id.localeCompare(right.source_id));
const calibration = sources.filter((source) => source.role === "calibration")
  .sort((left, right) => left.family.localeCompare(right.family)
    || left.source_id.localeCompare(right.source_id));
const evaluation = sources.filter((source) => source.role === "evaluation")
  .sort((left, right) => left.family.localeCompare(right.family)
    || left.source_id.localeCompare(right.source_id));
assert(fitting.length === 12 && calibration.length === 76 && evaluation.length === 16,
  "stratified role counts changed");

fs.mkdirSync(outputDirectory, {recursive: true});
const writePhase = (phase, mainDocuments, paddingCount) => {
  const documents = [];
  for (let index = 0; index < PREFIX_SENTINELS; index += 1) {
    const source = fitting[index % fitting.length];
    const passage = source.passages[index % PASSAGES_PER_SOURCE];
    documents.push({
      source_id: `prefix-sentinel-${String(index).padStart(2, "0")}`,
      source_panel_id: source.source_id, passage_ordinal: passage.ordinal,
      bytes: passage.content, analysis_role: "implementation_prefix_sentinel",
    });
  }
  documents.push(...mainDocuments);
  for (let index = 0; index < paddingCount; index += 1) {
    const source = fitting[index % fitting.length];
    const passage = source.passages[(index + 1) % PASSAGES_PER_SOURCE];
    documents.push({
      source_id: `excluded-fitting-padding-${phase}-${String(index).padStart(2, "0")}`,
      source_panel_id: source.source_id, passage_ordinal: passage.ordinal,
      bytes: passage.content, analysis_role: "fitting_padding_excluded_from_analysis",
    });
  }
  assert(documents.length === PREFIX_SENTINELS + ATOMIC_DOCUMENTS,
    `${phase} must contain exactly 72 documents`);
  const corpusParts = [];
  const indexRows = ["schema\tsplit\tdocument_id\toffset\tbytes\tfnv64\tsha256"];
  const documentBindings = [];
  let offset = 0;
  for (const document of documents) {
    const separator = corpusParts.length === 0 ? Buffer.alloc(0) : Buffer.from("\n\n");
    offset += separator.length;
    corpusParts.push(separator, document.bytes);
    const digest = sha256(document.bytes);
    const documentId = `${document.source_id}:${document.passage_ordinal}:${digest.slice(0, 16)}`;
    indexRows.push(["nsrl.production_corpus_record.v1", "dev", documentId, offset,
      document.bytes.length, fnv64(document.bytes), digest].join("\t"));
    documentBindings.push({
      document: documentBindings.length, document_id: documentId,
      source_id: document.source_panel_id, passage_ordinal: document.passage_ordinal,
      family: document.family ?? null, analysis_role: document.analysis_role, sha256: digest,
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
    phase, corpus_path: corpusPath, corpus_sha256: sha256(corpus),
    index_path: indexPath, index_sha256: sha256(index),
    document_start: PREFIX_SENTINELS, atomic_documents: ATOMIC_DOCUMENTS,
    role_documents: mainDocuments.length, excluded_fitting_padding_documents: paddingCount,
    document_bindings: documentBindings,
  };
};
const fittingDocuments = fitting.flatMap((source) => source.passages.map((passage) => ({
  source_id: `${source.source_id}-passage-${passage.ordinal}`,
  source_panel_id: source.source_id, passage_ordinal: passage.ordinal,
  family: source.family, bytes: passage.content, analysis_role: "fitting",
})));
const fittingPhase = writePhase("fitting", fittingDocuments, ATOMIC_DOCUMENTS - fittingDocuments.length);
const confirmationPhases = Array.from({length: PASSAGES_PER_SOURCE}, (_, passageOrdinal) => {
  const roleDocuments = [...calibration, ...evaluation].map((source) => ({
    source_id: `${source.source_id}-passage-${passageOrdinal}`,
    source_panel_id: source.source_id, passage_ordinal: passageOrdinal,
    family: source.family, bytes: source.passages[passageOrdinal].content,
    analysis_role: source.role,
  }));
  const shards = [roleDocuments.slice(0, ATOMIC_DOCUMENTS), roleDocuments.slice(ATOMIC_DOCUMENTS)]
    .map((documents, shard) => writePhase(
      `confirmation-passage-${passageOrdinal}-shard-${shard}`, documents,
      ATOMIC_DOCUMENTS - documents.length));
  return {passage_ordinal: passageOrdinal, role_documents: roleDocuments.length, shards};
});

const publicSources = sources.map(({body, passages, ...source}) => ({
  ...source,
  passages: passages.map(({content, ...passage}) => passage),
}));
const frame = {
  schema: "nsrl.production_multifamily_exchange_source_frame.v1",
  analysis_role: "prospective_pre_outcome_source_frame",
  acquisition_manifest_sha256: sha256(acquisitionBytes),
  source_definition: {
    unit: "one complete publication assigned wholly to one experimental role",
    families: acquisition.family_definitions,
    family_order: FAMILIES,
    sources_per_family: SOURCES_PER_FAMILY,
    eligibility: {
      minimum_cleaned_body_bytes: MINIMUM_CLEANED_BYTES,
      acquisition_status: ["cached", "downloaded"],
      unique_independence_key_within_family: true,
    },
    selection_seed: SOURCE_SELECTION_SEED,
    intended_population:
      "the frozen four-family acquisition frame only; no claim to arbitrary text, authors, agencies, standards, journals, or future publications",
  },
  role_partition: {
    seed: ROLE_SEED,
    algorithm: "within each family, ascending SHA256(seed, family, source_id, raw_sha256)",
    fitting_sources_per_family: FITTING_PER_FAMILY,
    calibration_sources_per_family: CALIBRATION_PER_FAMILY,
    evaluation_sources_per_family: EVALUATION_PER_FAMILY,
    fitting_sources: fitting.length,
    calibration_sources: calibration.length,
    evaluation_sources: evaluation.length,
    entire_sources_disjoint_across_roles: true,
  },
  panel_sampling: {
    seed: PASSAGE_SEED,
    passage_documents_per_source: PASSAGES_PER_SOURCE,
    passage_bytes_before_normalization: PASSAGE_BYTES,
    strata: "four consecutive source-body byte quartiles; one hashed passage wholly inside each quartile",
    passages_nonoverlapping: true,
    model_windows_per_passage_document: 2,
    model_context_tokens: 64,
    source_panel_score_scope: "maximum over all four passages and every frozen exchange",
    missing_source_policy: "abort before outcome evaluation; no source or passage replacement after freeze",
  },
  sources: publicSources,
  phases: {fitting: fittingPhase, confirmation_passages: confirmationPhases},
  outcome_firewall: {
    action_cube_outcomes_read: false,
    documents_200_212_read: false,
    source_content_tokenization_metadata_and_license_metadata_are_pre_outcome_features: true,
  },
  frame_fingerprint_sha256: sha256(stableJson({
    acquisition_manifest_sha256: sha256(acquisitionBytes), source_selection_seed: SOURCE_SELECTION_SEED,
    role_seed: ROLE_SEED, passage_seed: PASSAGE_SEED, sources: publicSources,
    phases: {fitting: fittingPhase, confirmation_passages: confirmationPhases},
  })),
};
fs.mkdirSync(path.dirname(framePath), {recursive: true});
fs.writeFileSync(framePath, `${JSON.stringify(frame, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: frame.schema, frame: framePath,
  families: FAMILIES, role_partition: frame.role_partition,
  passages_per_source: PASSAGES_PER_SOURCE,
  frame_fingerprint_sha256: frame.frame_fingerprint_sha256,
  action_cube_outcomes_read: false, documents_200_212_read: false,
}, null, 2)}\n`);
