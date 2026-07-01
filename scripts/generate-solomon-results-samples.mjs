#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { SolomonAttentionSampler } from "../web/attention-sampler.js";

const defaults = {
  model: "web/assets/solomon-attention.nsrllmm",
  outDir: "docs/assets/results-samples",
  out: "docs/solomon-sample-gallery.tsv",
  seed: 1,
  topK: 1,
  maxTextTokens: 96,
};

const prompts = [
  { id: "bael", kind: "fixed-known", prompt: "seal of Bael" },
  { id: "stolas", kind: "fixed-known", prompt: "seal of Stolas" },
  { id: "marbas", kind: "fixed-known", prompt: "seal of Marbas" },
  { id: "generic", kind: "fixed-generic", prompt: "king solomon seal" },
  { id: "eastern-king", kind: "fixed-held-out-phrase", prompt: "eastern king invisible spirit" },
  { id: "owl-astronomy", kind: "fixed-held-out-phrase", prompt: "owl astronomy seal" },
];

const crcTable = Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

const config = parseArgs(process.argv.slice(2));

if (!existsSync(config.model)) {
  throw new Error(`missing attention model: ${config.model}`);
}
mkdirSync(config.outDir, { recursive: true });

const sampler = new SolomonAttentionSampler(readFileSync(config.model));
const rows = prompts.map((sample) => {
  const result = sampler.sample(sample.prompt, {
    seed: config.seed,
    topK: config.topK,
    maxTextTokens: config.maxTextTokens,
  });
  const imagePath = path.join(config.outDir, `${sample.id}.png`);
  writeFileSync(imagePath, encodePng(result.width, result.height, result.rgba));
  return {
    model: result.metadata.model_kind,
    prompt_id: sample.id,
    prompt_kind: sample.kind,
    prompt: sample.prompt,
    seed: config.seed,
    top_k: config.topK,
    width: result.width,
    height: result.height,
    text_source: result.metadata.text_source,
    image_source: result.metadata.image_source,
    text_lm_fallback: result.metadata.text_lm_fallback,
    text: result.text,
    image_path: imagePath,
    model_hash: result.metadata.model_hash,
    token_hash: result.metadata.token_hash,
    image_sha256: sha256File(imagePath),
  };
});

const header = [
  "model",
  "prompt_id",
  "prompt_kind",
  "prompt",
  "seed",
  "top_k",
  "width",
  "height",
  "text_source",
  "image_source",
  "text_lm_fallback",
  "text",
  "image_path",
  "model_hash",
  "token_hash",
  "image_sha256",
];
const output = `${header.join("\t")}\n${rows
  .map((row) => header.map((column) => tsvCell(row[column])).join("\t"))
  .join("\n")}\n`;
writeFileSync(config.out, output, "utf8");
console.log(JSON.stringify({ schema: "nsrl.solomon_sample_gallery.v1", out: config.out, rows: rows.length }));

function parseArgs(args) {
  const parsed = { ...defaults };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--model") {
      parsed.model = requiredValue(args, ++index, arg);
    } else if (arg === "--out-dir") {
      parsed.outDir = requiredValue(args, ++index, arg);
    } else if (arg === "--out") {
      parsed.out = requiredValue(args, ++index, arg);
    } else if (arg === "--seed") {
      parsed.seed = parsePositive(requiredValue(args, ++index, arg), arg);
    } else if (arg === "--top-k") {
      parsed.topK = parsePositive(requiredValue(args, ++index, arg), arg);
    } else if (arg === "--max-text-tokens") {
      parsed.maxTextTokens = parsePositive(requiredValue(args, ++index, arg), arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return parsed;
}

function usage() {
  console.log(
    [
      "Usage: generate-solomon-results-samples.mjs [--model PATH] [--out-dir PATH]",
      "       [--out PATH] [--seed N] [--top-k N] [--max-text-tokens N]",
      "",
      "Generates the fixed NSRLLMM1 prompt gallery used by the published results page.",
    ].join("\n"),
  );
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function parsePositive(value, flag) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function encodePng(width, height, rgba) {
  if (rgba.length !== width * height * 4) {
    throw new Error(`rgba length mismatch for ${width}x${height}`);
  }
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * (stride + 1);
    raw[rowOffset] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * stride, stride).copy(raw, rowOffset + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr(width, height)),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

function ihdr(width, height) {
  const out = Buffer.alloc(13);
  out.writeUInt32BE(width, 0);
  out.writeUInt32BE(height, 4);
  out[8] = 8;
  out[9] = 6;
  out[10] = 0;
  out[11] = 0;
  out[12] = 0;
  return out;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 0);
  return Buffer.concat([length, typeBytes, data, crc]);
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = (crc >>> 8) ^ crcTable[(crc ^ byte) & 0xff];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function sha256File(filePath) {
  return createHash("sha256").update(readFileSync(filePath)).digest("hex");
}

function tsvCell(value) {
  return String(value ?? "").replace(/[\t\r\n]+/g, " ");
}
