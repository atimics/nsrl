#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const defaults = {
  html: "data/raw/key-solomon-goetia-pg72679/pg72679-images.html",
  slicesManifest: "data/processed/key-solomon-goetia-bitmaps-pg72679/slices/manifest.json",
  outDir: "data/processed/key-solomon-goetia-text-index-pg72679",
  imageSize: 128,
  signatureGrid: 8,
};

function usage() {
  console.log(
    "Usage: build-solomon-text-index.mjs [--html PATH] [--slices-manifest PATH] [--out-dir PATH]",
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--html") {
      config.html = requireValue(argv, ++index, arg);
    } else if (arg === "--slices-manifest") {
      config.slicesManifest = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function decodeEntities(text) {
  return String(text)
    .replace(/&mdash;|&#8212;/g, " - ")
    .replace(/&ndash;|&#8211;/g, " - ")
    .replace(/&nbsp;|&#160;/g, " ")
    .replace(/&rsquo;|&#8217;/g, "'")
    .replace(/&lsquo;|&#8216;/g, "'")
    .replace(/&ldquo;|&#8220;/g, '"')
    .replace(/&rdquo;|&#8221;/g, '"')
    .replace(/&aelig;/g, "ae")
    .replace(/&AElig;/g, "Ae")
    .replace(/&eacute;|&#233;/g, "e")
    .replace(/&Eacute;|&#201;/g, "E")
    .replace(/&iuml;|&#239;/g, "i")
    .replace(/&ouml;|&#246;/g, "o")
    .replace(/&ucirc;|&#251;/g, "u")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function stripTags(html) {
  return decodeEntities(html.replace(/<[^>]*>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function extractSpirits(html) {
  const start = html.indexOf("[Transcriber’s Note:");
  const end = html.indexOf("(Seal in Gold.)");
  if (start < 0 || end <= start) {
    throw new Error("could not locate spirit description section in HTML");
  }
  const spiritHtml = html.slice(start, end);
  const spirits = new Map();
  const paragraphRe = /<p class="c00[57]">[\s\S]*?<\/p>/g;
  for (const match of spiritHtml.matchAll(paragraphRe)) {
    const paragraph = match[0];
    const numberMatch = paragraph.match(/\((\d+)\.?\)/);
    if (!numberMatch) {
      continue;
    }
    const number = Number(numberMatch[1]);
    if (number < 1 || number > 72 || spirits.has(number)) {
      continue;
    }
    const beforeDash = paragraph.split(/—|&mdash;|&#8212;/)[0] ?? paragraph;
    const spanNames = [...beforeDash.matchAll(/<span class="sc">([\s\S]*?)<\/span>/g)]
      .map((span) => stripTags(span[1]))
      .filter(Boolean);
    const aliases = unique(
      spanNames
        .flatMap((name) => name.split(/\s*,\s*|\s+or\s+/i))
        .map(cleanName)
        .filter(Boolean),
    );
    const primaryName = aliases[0] ?? `Spirit ${number}`;
    const text = stripTags(paragraph);
    spirits.set(number, {
      number,
      primaryName,
      aliases,
      text,
    });
  }
  return spirits;
}

function cleanName(name) {
  return decodeEntities(name)
    .replace(/[.]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function unique(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    const key = value.toLocaleLowerCase("en-US");
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(value);
  }
  return out;
}

function spiritNumberForSlice(slice) {
  const match = slice.label.match(/^front-([345])-seal-grid-r(\d\d)-c(\d\d)$/);
  if (!match) {
    return null;
  }
  const front = Number(match[1]);
  const row = Number(match[2]);
  const col = Number(match[3]);
  return (front - 3) * 24 + (row - 1) * 4 + col;
}

function signatureForInk(ink, imageSize, grid) {
  const sums = Array.from({ length: grid * grid }, () => 0);
  const counts = Array.from({ length: grid * grid }, () => 0);
  for (let y = 0; y < imageSize; y += 1) {
    const binY = Math.floor((y * grid) / imageSize);
    for (let x = 0; x < imageSize; x += 1) {
      const binX = Math.floor((x * grid) / imageSize);
      const bin = binY * grid + binX;
      sums[bin] += ink[y * imageSize + x];
      counts[bin] += 1;
    }
  }
  return sums.map((sum, index) => Math.floor((sum + Math.floor(counts[index] / 2)) / counts[index]));
}

function escapeTsv(value) {
  return String(value ?? "")
    .replace(/\t/g, " ")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const html = fs.readFileSync(config.html, "utf8");
  const spirits = extractSpirits(html);
  if (spirits.size !== 72) {
    throw new Error(`expected 72 spirit descriptions, found ${spirits.size}`);
  }

  const manifest = JSON.parse(fs.readFileSync(config.slicesManifest, "utf8"));
  const slicesRoot = path.dirname(path.dirname(config.slicesManifest));
  const rows = [];
  for (const slice of manifest.slices) {
    const number = spiritNumberForSlice(slice);
    if (number == null) {
      continue;
    }
    const spirit = spirits.get(number);
    if (!spirit) {
      throw new Error(`missing spirit text for number ${number}`);
    }
    const inkPath = path.join(slicesRoot, slice.ink_128_u8);
    const ink = fs.readFileSync(inkPath);
    const expectedBytes = config.imageSize * config.imageSize;
    if (ink.length !== expectedBytes) {
      throw new Error(`${inkPath} has ${ink.length} bytes, expected ${expectedBytes}`);
    }
    rows.push({
      number,
      primaryName: spirit.primaryName,
      aliases: spirit.aliases,
      text: spirit.text,
      sliceId: slice.id,
      label: slice.label,
      sourceFile: slice.source_file,
      ink128: slice.ink_128_u8,
      signature: signatureForInk(ink, config.imageSize, config.signatureGrid),
    });
  }
  rows.sort((left, right) => left.number - right.number);
  if (rows.length !== 72) {
    throw new Error(`expected 72 mapped seal slices, found ${rows.length}`);
  }

  fs.mkdirSync(config.outDir, { recursive: true });
  const tsvPath = path.join(config.outDir, "solomon-spirit-text-signatures.tsv");
  const header = [
    "number",
    "primary_name",
    "aliases",
    "slice_id",
    "label",
    "source_file",
    "ink_128_u8",
    "signature_8x8",
    "text",
  ];
  const tsv = [
    header.join("\t"),
    ...rows.map((row) =>
      [
        row.number,
        escapeTsv(row.primaryName),
        escapeTsv(row.aliases.join("|")),
        escapeTsv(row.sliceId),
        escapeTsv(row.label),
        escapeTsv(row.sourceFile),
        escapeTsv(row.ink128),
        row.signature.join(","),
        escapeTsv(row.text),
      ].join("\t"),
    ),
  ].join("\n");
  fs.writeFileSync(tsvPath, `${tsv}\n`, "utf8");

  const manifestPath = path.join(config.outDir, "manifest.json");
  fs.writeFileSync(
    manifestPath,
    `${JSON.stringify(
      {
        schema: "nsrl.solomon_text_signature_index.v1",
        source_html: config.html,
        source_slices_manifest: config.slicesManifest,
        rows: rows.length,
        image_size: config.imageSize,
        signature_grid: config.signatureGrid,
        index_tsv: path.relative(config.outDir, tsvPath),
        mapping: "front-3..front-5 seal grids in printed spirit-number order",
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  console.log(JSON.stringify({ tsv: tsvPath, manifest: manifestPath, rows: rows.length }));
}

main();
