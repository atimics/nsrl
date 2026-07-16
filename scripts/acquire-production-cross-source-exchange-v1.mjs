#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const outputDirectory = process.argv[2]
  ?? "data/raw/production-cross-source-exchange-v1";

// This is an acquisition frame, not the experimental source frame. Metadata,
// length, language, and distinct-author checks are applied before any model
// action-cube outcome is evaluated.
const ebookIds = [
  11, 16, 23, 33, 35, 41, 45, 55, 61, 74, 76, 84, 98, 110, 113, 120,
  131, 132, 140, 145, 146, 155, 160, 161, 174, 205, 209, 215, 219, 236,
  244, 254, 280, 311, 394, 408, 421, 514, 541, 768, 769, 829, 863, 880,
  910, 996, 1023, 1080, 1155, 1184, 1228, 1232, 1245, 1257, 1259, 1260,
  1292, 1322, 1342, 1399, 1400, 1497, 1661, 1727, 1837, 1952, 1998, 2084,
  2097, 2147, 2148, 2500, 2527, 2542, 2680, 2701, 2775, 2892, 3207, 3300,
  3600, 3825, 4300, 4705, 5200, 5827, 6130, 6761, 7178, 7370, 8800,
  10007, 16328, 19810, 20203, 27827, 28054, 32449, 38769, 64317,
];

const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const field = (text, name) => text.match(new RegExp(`^${name}:\\s*(.+)$`, "im"))?.[1].trim()
  ?? null;

await fs.mkdir(outputDirectory, {recursive: true});
const manifestPath = path.join(outputDirectory, "acquisition.json");
try {
  const existing = JSON.parse(await fs.readFile(manifestPath, "utf8"));
  if (existing.schema !== "nsrl.production_cross_source_acquisition.v1"
    || existing.acquisition_only_no_model_outcomes !== true) {
    throw new Error("existing acquisition manifest has the wrong schema");
  }
  for (const record of existing.records.filter((item) => item.sha256)) {
    const bytes = await fs.readFile(record.path);
    if (sha256(bytes) !== record.sha256) {
      throw new Error(`cached acquisition hash changed: ${record.ebook_id}`);
    }
  }
  process.stdout.write(`${JSON.stringify({
    schema: existing.schema,
    requested: existing.requested,
    usable: existing.usable,
    manifest: manifestPath,
    existing_manifest_verified: true,
  }, null, 2)}\n`);
  process.exit(0);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
const records = [];
for (const ebookId of ebookIds) {
  const outputPath = path.join(outputDirectory, `pg${ebookId}.txt`);
  const url = `https://www.gutenberg.org/cache/epub/${ebookId}/pg${ebookId}.txt`;
  let bytes;
  let status = "cached";
  try {
    bytes = await fs.readFile(outputPath);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const response = await fetch(url, {headers: {"User-Agent": "NSRL research corpus acquisition/1.0"}});
    if (!response.ok) {
      records.push({ebook_id: ebookId, url, status: `http_${response.status}`});
      continue;
    }
    bytes = Buffer.from(await response.arrayBuffer());
    await fs.writeFile(outputPath, bytes);
    status = "downloaded";
  }
  const text = bytes.toString("utf8");
  const title = field(text, "Title");
  const author = field(text, "Author");
  const language = field(text, "Language");
  const valid = text.includes("Project Gutenberg") && title && author && language;
  records.push({
    ebook_id: ebookId,
    url,
    path: outputPath,
    status: valid ? status : "invalid_metadata",
    bytes: bytes.length,
    sha256: sha256(bytes),
    title,
    author,
    language,
  });
}

const manifest = {
  schema: "nsrl.production_cross_source_acquisition.v1",
  acquisition_only_no_model_outcomes: true,
  source_url_pattern: "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
  license_id: "LicenseRef-Public-Domain-US-Project-Gutenberg",
  rights_basis_url: "https://www.gutenberg.org/policy/permission",
  requested: ebookIds.length,
  usable: records.filter((record) => ["cached", "downloaded"].includes(record.status)).length,
  records,
};
await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: manifest.schema,
  requested: manifest.requested,
  usable: manifest.usable,
  manifest: manifestPath,
}, null, 2)}\n`);
