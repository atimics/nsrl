#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const schema = "nsrl.solomon_token_layout_self_test.v1";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const canonicalLayout = {
  pad: 0,
  bos: 1,
  prompt: 2,
  text: 3,
  image: 4,
  eos: 5,
  task_text_to_image: 6,
  task_image_to_text: 7,
  task_match: 8,
  task_explain: 9,
  task_identify: 10,
  image_channel_ink: 11,
  image_channel_edge: 12,
  image_channel_component: 13,
  image_channel_radial: 14,
  image_channel_direction: 15,
  text_base: 16,
  text_count: 128,
  image_base: 144,
  image_bins: 16,
};
const uppercaseToLayout = {
  PAD: "pad",
  BOS: "bos",
  PROMPT: "prompt",
  TEXT: "text",
  IMAGE: "image",
  EOS: "eos",
  TASK_TEXT_TO_IMAGE: "task_text_to_image",
  TASK_IMAGE_TO_TEXT: "task_image_to_text",
  TASK_MATCH: "task_match",
  TASK_EXPLAIN: "task_explain",
  TASK_IDENTIFY: "task_identify",
  IMAGE_CHANNEL_INK: "image_channel_ink",
  IMAGE_CHANNEL_EDGE: "image_channel_edge",
  IMAGE_CHANNEL_COMPONENT: "image_channel_component",
  IMAGE_CHANNEL_RADIAL: "image_channel_radial",
  IMAGE_CHANNEL_DIRECTION: "image_channel_direction",
  TEXT_BASE: "text_base",
  TEXT_COUNT: "text_count",
  IMAGE_BASE: "image_base",
  IMAGE_BINS: "image_bins",
};
const fullLayoutKeys = Object.keys(canonicalLayout);
const imageConsumerKeys = [
  "image_channel_ink",
  "image_channel_edge",
  "image_channel_component",
  "image_channel_radial",
  "image_channel_direction",
  "image_base",
  "image_bins",
];

function usage() {
  console.log([
    "Usage: check-solomon-token-layout-self-test.mjs [--out PATH]",
    "",
    "Checks Solomon's integer token layout parity across the JS v2 corpus builder,",
    "Rust native attention binary, JS quality gates, retrieval consumers, and the",
    "shared symbolic image encoder defaults.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
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

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function runCase(name, fn) {
  try {
    const evidence = fn();
    return { name, ok: true, ...evidence };
  } catch (error) {
    return {
      name,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function repoPath(filePath) {
  return path.join(repoRoot, filePath);
}

function parseConstantLayout(filePath) {
  const source = fs.readFileSync(repoPath(filePath), "utf8");
  const constants = {};
  const regex = /\bconst\s+([A-Z][A-Z0-9_]*)\s*(?::\s*[^=]+)?=\s*([^;]+);/g;
  for (const match of source.matchAll(regex)) {
    const [, name, expression] = match;
    const value = evaluateIntegerExpression(expression, constants);
    if (value !== null) {
      constants[name] = value;
    }
  }
  const layout = {};
  for (const [name, key] of Object.entries(uppercaseToLayout)) {
    if (Number.isInteger(constants[name])) {
      layout[key] = constants[name];
    }
  }
  return layout;
}

function parseObjectLayout(filePath, objectName) {
  const source = fs.readFileSync(repoPath(filePath), "utf8");
  const regex = new RegExp(`\\bconst\\s+${escapeRegExp(objectName)}\\s*=\\s*\\{([\\s\\S]*?)\\};`);
  const match = regex.exec(source);
  if (!match) {
    throw new Error(`${filePath} is missing const ${objectName}`);
  }
  const layout = {};
  const fieldRegex = /\b([a-z][a-z0-9_]*)\s*:\s*([0-9]+)/g;
  for (const field of match[1].matchAll(fieldRegex)) {
    const [, key, value] = field;
    layout[key] = Number(value);
  }
  return layout;
}

function evaluateIntegerExpression(expression, constants) {
  const normalized = String(expression)
    .replace(/\/\/.*$/gm, "")
    .replace(/\b([A-Z][A-Z0-9_]*)\s+as\s+[A-Za-z0-9_:]+/g, "$1")
    .replace(/([0-9])_([0-9])/g, "$1$2")
    .trim();
  if (!/^[A-Z0-9_+\-\s()]+$/.test(normalized)) {
    return null;
  }
  const tokens = normalized.match(/[A-Z][A-Z0-9_]*|[0-9]+|[()+-]/g);
  if (!tokens || tokens.length === 0) {
    return null;
  }
  let index = 0;

  const parseExpression = () => {
    let value = parseTerm();
    while (index < tokens.length && (tokens[index] === "+" || tokens[index] === "-")) {
      const operator = tokens[index++];
      const right = parseTerm();
      value = operator === "+" ? value + right : value - right;
    }
    return value;
  };

  const parseTerm = () => {
    const token = tokens[index++];
    if (token === "(") {
      const value = parseExpression();
      if (tokens[index++] !== ")") {
        throw new Error(`unclosed expression ${expression}`);
      }
      return value;
    }
    if (/^[0-9]+$/.test(token)) {
      return Number(token);
    }
    if (Number.isInteger(constants[token])) {
      return constants[token];
    }
    throw new Error(`unknown constant ${token} in ${expression}`);
  };

  try {
    const value = parseExpression();
    if (index !== tokens.length || !Number.isInteger(value)) {
      return null;
    }
    return value;
  } catch {
    return null;
  }
}

function compareLayout(source, layout, keys) {
  const errors = [];
  for (const key of keys) {
    const actual = layout[key];
    const expected = canonicalLayout[key];
    if (actual !== expected) {
      errors.push(`${source} ${key} ${JSON.stringify(actual)} != ${expected}`);
    }
  }
  return { ok: errors.length === 0, errors };
}

function assertLayout(source, layout, keys) {
  const result = compareLayout(source, layout, keys);
  if (!result.ok) {
    throw new Error(result.errors.join("; "));
  }
  return {
    source,
    keys,
    layout: Object.fromEntries(keys.map((key) => [key, layout[key]])),
  };
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function markerOffsets(tokens) {
  return [
    canonicalLayout.image_channel_ink,
    canonicalLayout.image_channel_edge,
    canonicalLayout.image_channel_component,
    canonicalLayout.image_channel_radial,
    canonicalLayout.image_channel_direction,
  ].map((marker) => tokens.indexOf(marker));
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const cases = [];
  cases.push(runCase("good-js-corpus-builder-layout", () =>
    assertLayout(
      "scripts/build-solomon-multimodal-corpus.mjs",
      parseConstantLayout("scripts/build-solomon-multimodal-corpus.mjs"),
      fullLayoutKeys,
    ),
  ));
  cases.push(runCase("good-rust-native-attention-layout", () =>
    assertLayout(
      "crates/nsrl-train/src/bin/nsrl-solomon-attention.rs",
      parseConstantLayout("crates/nsrl-train/src/bin/nsrl-solomon-attention.rs"),
      fullLayoutKeys,
    ),
  ));
  cases.push(runCase("good-js-fallback-layouts", () => {
    const sources = [
      ["scripts/check-solomon-attention-task-eval.mjs", "TOKEN_LAYOUT_FALLBACK"],
      ["scripts/check-solomon-v2-curriculum-stages.mjs", "TASK_TOKEN_LAYOUT_FALLBACK"],
      ["scripts/check-solomon-v2-quality-report.mjs", "TASK_TOKEN_LAYOUT_FALLBACK"],
      ["scripts/check-solomon-attention-task-eval-self-test.mjs", "TOKEN_LAYOUT"],
      ["scripts/check-solomon-v2-quality-report-self-test.mjs", "TOKEN_LAYOUT"],
    ];
    return {
      layouts: sources.map(([filePath, objectName]) =>
        assertLayout(`${filePath} ${objectName}`, parseObjectLayout(filePath, objectName), fullLayoutKeys.filter((key) => key !== "pad" && key !== "text_base" && key !== "text_count")),
      ),
    };
  }));
  cases.push(runCase("good-js-retrieval-consumer-layouts", () => {
    const sources = [
      "scripts/train-solomon-v2-retrieval-head.mjs",
      "scripts/infer-solomon-v2-identity.mjs",
      "scripts/score-solomon-generative-eval-retrieval.mjs",
      "scripts/run-solomon-generative-eval.mjs",
      "scripts/check-solomon-attention-sample-binding.mjs",
      "scripts/check-solomon-attention-denoise-bridge.mjs",
    ];
    return {
      layouts: sources.map((filePath) =>
        assertLayout(filePath, parseConstantLayout(filePath), imageConsumerKeys),
      ),
    };
  }));
  cases.push(runCase("good-shared-symbolic-image-defaults", () => {
    const signature = new Array(16 * 16).fill(0);
    const ink = solomonImage.imageTokens(signature);
    const symbolic = solomonImage.symbolicImageTokens(signature);
    const offsets = markerOffsets(symbolic);
    assert(ink.every((token) => token === canonicalLayout.image_base), "shared ink tokens do not use image_base 144");
    assert(symbolic.length === 5 * (signature.length + 1), `symbolic length ${symbolic.length} is not 5 channel payloads`);
    assert(offsets.every((offset, index) => offset === index * (signature.length + 1)), `bad symbolic marker offsets ${offsets.join(",")}`);
    assert(
      solomonImage.imageTokenChannels("symbolic16").join(",") === "ink,edge,component,radial,direction",
      "symbolic16 channel order drifted",
    );
    return { ink_token: ink[0], symbolic_tokens: symbolic.length, marker_offsets: offsets };
  }));
  cases.push(runCase("bad-js-layout-mismatch", () => {
    const result = compareLayout("fixture", { image_base: canonicalLayout.image_base - 1 }, ["image_base"]);
    assert(!result.ok, "layout mismatch was not rejected");
    return { rejected_errors: result.errors };
  }));
  cases.push(runCase("bad-rust-task-marker-mismatch", () => {
    const result = compareLayout("fixture", { task_match: canonicalLayout.task_match + 1 }, ["task_match"]);
    assert(!result.ok, "task marker mismatch was not rejected");
    return { rejected_errors: result.errors };
  }));
  cases.push(runCase("bad-shared-marker-order", () => {
    const signature = new Array(16 * 16).fill(0);
    const tokens = solomonImage.symbolicImageTokens(signature);
    const corrupted = [...tokens];
    corrupted[0] = canonicalLayout.image_channel_edge;
    corrupted[signature.length + 1] = canonicalLayout.image_channel_ink;
    const offsets = markerOffsets(corrupted);
    const good = offsets.every((offset, index) => offset === index * (signature.length + 1));
    assert(!good, "corrupt symbolic marker order was not rejected");
    return { rejected_marker_offsets: offsets };
  }));

  const report = {
    schema,
    ok: cases.every((item) => item.ok),
    canonical_layout: canonicalLayout,
    cases,
    errors: cases.filter((item) => !item.ok).map((item) => `${item.name}: ${item.error || "failed"}`),
  };
  if (config.outPath) {
    const resolved = path.resolve(config.outPath);
    fs.mkdirSync(path.dirname(resolved), { recursive: true });
    fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  }
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
