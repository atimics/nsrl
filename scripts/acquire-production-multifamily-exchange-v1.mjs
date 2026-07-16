#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const outputDirectory = process.argv[2]
  ?? "data/raw/production-multifamily-exchange-v1";
const requestedEligible = Number(process.argv[3] ?? "28");
const requestedFamilies = (process.argv[4]
  ?? "gutenberg,rfc,federal_register,science").split(",").filter(Boolean);
const exclusionFramePaths = (process.argv[5] ?? "").split(",").filter(Boolean);
const PRIOR_FRAME = "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-source-frame.json";
const SEED = process.argv[6] ?? "nsrl-m4-multifamily-acquisition-2026-07-15-v1";
const INDEPENDENCE_KEY_MODE = process.argv[7] ?? "legacy_creator_cluster";
const MINIMUM_CLEANED_BYTES = 65_536;
const ELIGIBLE_PER_FAMILY = requestedEligible;
const USER_AGENT = "NSRL prospective research corpus acquisition/1.0";
const KNOWN_FAMILIES = ["gutenberg", "rfc", "federal_register", "science"];
const ACTIVE_FAMILIES = new Set(requestedFamilies);

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
assert(Number.isSafeInteger(ELIGIBLE_PER_FAMILY) && ELIGIBLE_PER_FAMILY > 0,
  "eligible records per family must be a positive integer");
assert(ACTIVE_FAMILIES.size > 0
  && [...ACTIVE_FAMILIES].every((family) => KNOWN_FAMILIES.includes(family)),
"requested acquisition families are invalid");
assert(["legacy_creator_cluster", "whole_publication"].includes(INDEPENDENCE_KEY_MODE),
  "independence key mode is invalid");
const identityKey = (legacy) => (candidate) => INDEPENDENCE_KEY_MODE === "whole_publication"
  ? candidate.source_id : legacy(candidate);
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const shaKey = (...parts) => sha256(parts.join("\0"));
const normalizeKey = (value) => value.normalize("NFKD").toLowerCase()
  .replace(/[^a-z0-9]+/g, " ").trim();
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
    const withoutReferences = body.replace(/<ref-list(?:\s[^>]*)?>[\s\S]*$/i, "");
    return stripTags(withoutReferences);
  }
  return normalizeText(text);
};
const xmlValue = (text, tag) => decodeXml(text.match(
  new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tag}>`, "i"))?.[1]
  ?.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim() ?? "");
const parseCsv = (text) => {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') quoted = false;
      else field += character;
    } else if (character === '"') quoted = true;
    else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      field = "";
    } else field += character;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  const header = rows.shift();
  return rows.map((values) => Object.fromEntries(header.map((name, index) => [name, values[index] ?? ""])));
};
const fetchBytes = async (url) => {
  const response = await fetch(url, {headers: {"User-Agent": USER_AGENT}});
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
  return Buffer.from(await response.arrayBuffer());
};
const cachedDownload = async (url, outputPath) => {
  try {
    return {bytes: await fs.readFile(outputPath), status: "cached"};
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const bytes = await fetchBytes(url);
  await fs.mkdir(path.dirname(outputPath), {recursive: true});
  await fs.writeFile(outputPath, bytes);
  return {bytes, status: "downloaded"};
};
const catalogDownload = async (name, url) => {
  const outputPath = path.join(outputDirectory, "catalogs", name);
  const {bytes, status} = await cachedDownload(url, outputPath);
  return {name, url, path: outputPath, status, bytes: bytes.length, sha256: sha256(bytes), content: bytes};
};
const excludedSourceIds = new Map(KNOWN_FAMILIES.map((family) => [family, new Set()]));
const excludedIndependenceKeys = new Map(KNOWN_FAMILIES.map((family) => [family, new Set()]));
const acquireUntil = async (family, candidates, identityKey, recordForCandidate) => {
  const records = [];
  const seen = new Set();
  for (const candidate of candidates) {
    const key = normalizeKey(identityKey(candidate));
    if (!key || seen.has(key)
      || excludedSourceIds.get(family).has(candidate.source_id)
      || excludedIndependenceKeys.get(family).has(key)) continue;
    let record;
    try {
      record = await recordForCandidate(candidate, key);
    } catch (error) {
      records.push({family, source_id: candidate.source_id, status: "acquisition_error", error: error.message});
      continue;
    }
    if (record.cleaned_body_bytes < MINIMUM_CLEANED_BYTES) {
      records.push({...record, status: "too_short"});
      continue;
    }
    seen.add(key);
    records.push({...record, status: record.status, independence_key: key});
    if (records.filter((item) => ["cached", "downloaded"].includes(item.status)).length
      >= ELIGIBLE_PER_FAMILY) break;
  }
  const usable = records.filter((record) => ["cached", "downloaded"].includes(record.status));
  assert(usable.length >= ELIGIBLE_PER_FAMILY,
    `${family}: need ${ELIGIBLE_PER_FAMILY} eligible independent records, found ${usable.length}`);
  return records;
};

await fs.mkdir(outputDirectory, {recursive: true});
const manifestPath = path.join(outputDirectory, "acquisition.json");
try {
  const existing = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  assert(existing.schema === "nsrl.production_multifamily_acquisition.v1"
    && existing.acquisition_only_no_model_outcomes === true, "existing manifest schema changed");
  for (const record of existing.records.filter((item) => item.sha256)) {
    const bytes = await fs.readFile(record.path);
    assert(sha256(bytes) === record.sha256, `cached acquisition hash changed: ${record.source_id}`);
  }
  if (existing.eligible_records_required_per_family === ELIGIBLE_PER_FAMILY
    && Object.values(existing.family_summary).every(
      (summary) => summary.eligible >= ELIGIBLE_PER_FAMILY)) {
    process.stdout.write(`${JSON.stringify({
      schema: existing.schema,
      families: existing.family_summary,
      manifest: manifestPath,
      existing_manifest_verified: true,
    }, null, 2)}\n`);
    process.exit(0);
  }
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const priorFrameBytes = await fs.readFile(PRIOR_FRAME);
const priorFrame = JSON.parse(priorFrameBytes);
for (const framePath of exclusionFramePaths) {
  const frame = JSON.parse(await fs.readFile(framePath, "utf8"));
  assert(Array.isArray(frame.sources), `exclusion frame has no sources: ${framePath}`);
  for (const source of frame.sources) {
    if (!KNOWN_FAMILIES.includes(source.family)) continue;
    excludedSourceIds.get(source.family).add(source.source_id);
    if (source.independence_key) {
      excludedIndependenceKeys.get(source.family).add(normalizeKey(source.independence_key));
    }
  }
}
const excludedGutenbergIds = new Set(priorFrame.sources.map((source) => source.ebook_id));
const catalogs = [];
const records = [];

if (ACTIVE_FAMILIES.has("gutenberg")) {
const gutenbergCatalog = await catalogDownload(
  "pg_catalog.csv", "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv");
catalogs.push({...gutenbergCatalog, content: undefined});
const gutenbergRows = parseCsv(gutenbergCatalog.content.toString("utf8"));
const gutenbergCandidates = gutenbergRows.filter((row) => row.Type === "Text"
  && row.Language.split(";").map((value) => value.trim()).includes("en")
  && row.Authors && !/^anonymous$/i.test(row.Authors.trim()))
  .map((row) => ({
    source_id: `gutenberg-${row["Text#"]}`,
    ebook_id: Number(row["Text#"]),
    title: row.Title,
    author: row.Authors.split(";")[0].trim(),
  }))
  .filter((candidate) => Number.isSafeInteger(candidate.ebook_id)
    && !excludedGutenbergIds.has(candidate.ebook_id))
  .sort((left, right) => shaKey(SEED, "gutenberg", left.source_id)
    .localeCompare(shaKey(SEED, "gutenberg", right.source_id)));
records.push(...await acquireUntil("gutenberg", gutenbergCandidates,
  identityKey((candidate) => candidate.author),
  async (candidate) => {
    const url = `https://www.gutenberg.org/cache/epub/${candidate.ebook_id}/pg${candidate.ebook_id}.txt`;
    const outputPath = path.join(outputDirectory, "gutenberg", `pg${candidate.ebook_id}.txt`);
    const {bytes, status} = await cachedDownload(url, outputPath);
    const body = cleanedBody("gutenberg", bytes.toString("utf8"));
    return {
      family: "gutenberg", source_id: candidate.source_id, ebook_id: candidate.ebook_id,
      title: candidate.title, creator: candidate.author, source_url: url, path: outputPath,
      status, bytes: bytes.length, sha256: sha256(bytes), cleaned_body_bytes: Buffer.byteLength(body),
      license_id: "LicenseRef-Public-Domain-US-Project-Gutenberg",
    };
  }));
}

if (ACTIVE_FAMILIES.has("rfc")) {
const rfcCatalog = await catalogDownload("rfc-index.xml", "https://www.rfc-editor.org/rfc-index.xml");
catalogs.push({...rfcCatalog, content: undefined});
const rfcEntries = [...rfcCatalog.content.toString("utf8").matchAll(/<rfc-entry>([\s\S]*?)<\/rfc-entry>/g)]
  .map((match) => {
    const entry = match[1];
    const documentId = xmlValue(entry, "doc-id");
    const rfcNumber = Number(documentId.replace(/^RFC/i, ""));
    return {
      source_id: `rfc-${rfcNumber}`,
      rfc_number: rfcNumber,
      title: xmlValue(entry, "title"),
      author: xmlValue(entry, "name"),
      page_count: Number(xmlValue(entry, "page-count")),
    };
  })
  .filter((candidate) => Number.isSafeInteger(candidate.rfc_number)
    && candidate.rfc_number >= 2_000 && candidate.author && candidate.page_count >= 20)
  .sort((left, right) => shaKey(SEED, "rfc", left.source_id)
    .localeCompare(shaKey(SEED, "rfc", right.source_id)));
records.push(...await acquireUntil("rfc", rfcEntries, identityKey((candidate) => candidate.author),
  async (candidate) => {
    const url = `https://www.rfc-editor.org/rfc/rfc${candidate.rfc_number}.txt`;
    const outputPath = path.join(outputDirectory, "rfc", `rfc${candidate.rfc_number}.txt`);
    const {bytes, status} = await cachedDownload(url, outputPath);
    const body = cleanedBody("rfc", bytes.toString("utf8"));
    return {
      family: "rfc", source_id: candidate.source_id, rfc_number: candidate.rfc_number,
      title: candidate.title, creator: candidate.author, page_count: candidate.page_count,
      source_url: url, path: outputPath, status, bytes: bytes.length, sha256: sha256(bytes),
      cleaned_body_bytes: Buffer.byteLength(body), license_id: "LicenseRef-IETF-Trust-Legal-Provisions",
    };
  }));
}

if (ACTIVE_FAMILIES.has("federal_register")) {
const federalFields = ["document_number", "title", "publication_date", "page_length",
  "raw_text_url", "agencies", "type"];
const federalParameters = new URLSearchParams({
  per_page: "100", order: "newest",
  "conditions[publication_date][gte]": INDEPENDENCE_KEY_MODE === "whole_publication"
    ? "2010-01-01" : "2024-01-01",
  "conditions[publication_date][lte]": INDEPENDENCE_KEY_MODE === "whole_publication"
    ? "2023-12-31" : "2025-12-31",
});
for (const field of federalFields) federalParameters.append("fields[]", field);
let federalUrl = `https://www.federalregister.gov/api/v1/documents.json?${federalParameters}`;
const federalCandidates = [];
for (let page = 0; page < (INDEPENDENCE_KEY_MODE === "whole_publication" ? 200 : 50)
  && federalUrl; page += 1) {
  const response = JSON.parse((await fetchBytes(federalUrl)).toString("utf8"));
  federalCandidates.push(...response.results.filter((record) => record.raw_text_url
    && record.page_length >= (INDEPENDENCE_KEY_MODE === "whole_publication" ? 12 : 20)
    && record.agencies?.length > 0).map((record) => {
    const agency = record.agencies.at(-1);
    return {
      ...record, source_id: `federal-register-${record.document_number}`,
      agency: agency.name, agency_slug: agency.slug,
    };
  }));
  federalUrl = response.next_page_url ?? null;
}
const federalCandidatesWithAgency = federalCandidates.filter(
  (candidate) => candidate.agency_slug && candidate.agency);
federalCandidatesWithAgency.sort((left, right) => shaKey(SEED, "federal_register", left.source_id)
  .localeCompare(shaKey(SEED, "federal_register", right.source_id)));
records.push(...await acquireUntil("federal_register", federalCandidatesWithAgency,
  identityKey((candidate) => candidate.agency_slug), async (candidate) => {
    const outputPath = path.join(outputDirectory, "federal-register", `${candidate.document_number}.txt`);
    const {bytes, status} = await cachedDownload(candidate.raw_text_url, outputPath);
    const body = cleanedBody("federal_register", bytes.toString("utf8"));
    return {
      family: "federal_register", source_id: candidate.source_id,
      document_number: candidate.document_number, title: candidate.title,
      creator: candidate.agency, agency_slug: candidate.agency_slug,
      publication_date: candidate.publication_date, document_type: candidate.type,
      page_count: candidate.page_length, source_url: candidate.raw_text_url,
      path: outputPath, status, bytes: bytes.length, sha256: sha256(bytes),
      cleaned_body_bytes: Buffer.byteLength(body), license_id: "LicenseRef-US-Government-Work",
    };
  }));
}

if (ACTIVE_FAMILIES.has("science")) {
const scienceQuery = INDEPENDENCE_KEY_MODE === "whole_publication"
  ? "OPEN_ACCESS:Y AND FIRST_PDATE:[2020-01-01 TO 2024-12-31]"
  : "OPEN_ACCESS:Y AND FIRST_PDATE:[2024-01-01 TO 2024-12-31]";
const scienceResults = [];
let scienceCursor = "*";
const sciencePages = INDEPENDENCE_KEY_MODE === "whole_publication" ? 6 : 1;
for (let page = 0; page < sciencePages && scienceCursor; page += 1) {
  const scienceUrl = `https://www.ebi.ac.uk/europepmc/webservices/rest/search?${new URLSearchParams({
    query: scienceQuery, format: "json", resultType: "core", pageSize: "1000",
    cursorMark: scienceCursor,
  })}`;
  const scienceCatalog = await catalogDownload(
    `europe-pmc-search-v3-page-${String(page + 1).padStart(2, "0")}.json`, scienceUrl);
  catalogs.push({...scienceCatalog, content: undefined});
  const pageResults = JSON.parse(scienceCatalog.content.toString("utf8"));
  scienceResults.push(...(pageResults.resultList?.result ?? []));
  scienceCursor = pageResults.nextCursorMark && pageResults.nextCursorMark !== scienceCursor
    ? pageResults.nextCursorMark : null;
}
const scienceCandidates = [...new Map(scienceResults
  .filter((record) => record.pmcid && record.isOpenAccess === "Y"
    && record.language === "eng" && record.firstPublicationDate
    && /^(?:cc\s*(?:by|0)|public domain)/i.test(record.license ?? ""))
  .map((record) => ({
    source_id: `europe-pmc-${record.pmcid.toLowerCase()}`,
    pmcid: record.pmcid, title: record.title,
    author: record.authorList?.author?.[0]?.fullName ?? record.authorString?.split(",")[0] ?? "",
    journal: record.journalInfo?.journal?.title ?? record.journalTitle ?? "",
    publication_date: record.firstPublicationDate, license: record.license,
  }))
  .filter((candidate) => candidate.author && candidate.journal)
  .map((candidate) => [candidate.source_id, candidate])).values()]
  .sort((left, right) => shaKey(SEED, "science", left.source_id)
    .localeCompare(shaKey(SEED, "science", right.source_id)));
const scienceJournalKeys = new Set();
records.push(...await acquireUntil("science", scienceCandidates,
  identityKey((candidate) => `${normalizeKey(candidate.author)}::${normalizeKey(candidate.journal)}`),
  async (candidate) => {
    const journalKey = normalizeKey(candidate.journal);
    if (INDEPENDENCE_KEY_MODE === "legacy_creator_cluster" && scienceJournalKeys.has(journalKey)) {
      throw new Error("duplicate_journal");
    }
    const url = `https://www.ebi.ac.uk/europepmc/webservices/rest/${candidate.pmcid}/fullTextXML`;
    const outputPath = path.join(outputDirectory, "science", `${candidate.pmcid}.xml`);
    const {bytes, status} = await cachedDownload(url, outputPath);
    const text = bytes.toString("utf8");
    const licenseText = xmlValue(text, "license-p");
    const body = cleanedBody("science", text);
    if (Buffer.byteLength(body) >= MINIMUM_CLEANED_BYTES) scienceJournalKeys.add(journalKey);
    return {
      family: "science", source_id: candidate.source_id, pmcid: candidate.pmcid,
      title: candidate.title, creator: candidate.author, journal: candidate.journal,
      publication_date: candidate.publication_date, source_url: url, path: outputPath,
      status, bytes: bytes.length, sha256: sha256(bytes),
      cleaned_body_bytes: Buffer.byteLength(body), license_id: candidate.license,
      license_excerpt: licenseText.slice(0, 500),
    };
  }));
}

const familySummary = Object.fromEntries(requestedFamilies
  .map((family) => [family, {
    eligible: records.filter((record) => record.family === family
      && ["cached", "downloaded"].includes(record.status)).length,
    attempted: records.filter((record) => record.family === family).length,
  }]));
for (const [family, summary] of Object.entries(familySummary)) {
  assert(summary.eligible >= ELIGIBLE_PER_FAMILY, `${family} acquisition fell below target`);
}
const manifest = {
  schema: "nsrl.production_multifamily_acquisition.v1",
  analysis_role: "prospective_acquisition_only",
  acquisition_only_no_model_outcomes: true,
  seed: SEED,
  minimum_cleaned_body_bytes: MINIMUM_CLEANED_BYTES,
  eligible_records_required_per_family: ELIGIBLE_PER_FAMILY,
  active_families: requestedFamilies,
  independence_key_mode: INDEPENDENCE_KEY_MODE,
  exclusion_frames: exclusionFramePaths,
  excluded_source_ids_by_family: Object.fromEntries(KNOWN_FAMILIES.map(
    (family) => [family, excludedSourceIds.get(family).size])),
  excluded_independence_keys_by_family: Object.fromEntries(KNOWN_FAMILIES.map(
    (family) => [family, excludedIndependenceKeys.get(family).size])),
  prior_gutenberg_frame: {path: PRIOR_FRAME, sha256: sha256(priorFrameBytes)},
  family_definitions: {
    gutenberg: "one English Project Gutenberg ebook from an author absent from the M3 frame",
    rfc: INDEPENDENCE_KEY_MODE === "whole_publication"
      ? "one RFC plain-text whole publication"
      : "one RFC plain-text publication from a distinct first-listed author",
    federal_register: INDEPENDENCE_KEY_MODE === "whole_publication"
      ? "one Federal Register whole publication"
      : "one Federal Register document from a distinct most-specific listed agency",
    science: INDEPENDENCE_KEY_MODE === "whole_publication"
      ? "one Europe PMC open-access English whole publication"
      : "one Europe PMC open-access article with a distinct first-author/journal pair and distinct journal",
  },
  catalogs,
  family_summary: familySummary,
  records,
  documents_200_212_read: false,
};
await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: manifest.schema,
  families: manifest.family_summary,
  records: manifest.records.length,
  manifest: manifestPath,
  documents_200_212_read: false,
}, null, 2)}\n`);
