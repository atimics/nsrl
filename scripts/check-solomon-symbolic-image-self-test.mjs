#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import * as solomonImage from "./lib/solomon-symbolic-image.mjs";

const schema = "nsrl.solomon_symbolic_image_self_test.v1";
const GRID = 16;
const BINS = GRID * GRID;
const IMAGE_BASE = 144;
const IMAGE_BINS = 16;
const CHANNEL_TOKENS = {
  ink: 11,
  edge: 12,
  component: 13,
  radial: 14,
  direction: 15,
};

function usage() {
  console.log([
    "Usage: check-solomon-symbolic-image-self-test.mjs [--out PATH]",
    "",
    "Checks the shared Solomon symbolic16 image-token encoder: marker order,",
    "profile lengths, component crossing hints, radial position, and direction",
    "junction bins.",
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

function imageOptions() {
  return {
    grid: GRID,
    imageBase: IMAGE_BASE,
    imageBins: IMAGE_BINS,
    channelTokens: CHANNEL_TOKENS,
  };
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function plusSignature() {
  const signature = new Array(BINS).fill(0);
  const set = (x, y, value = 255) => {
    signature[y * GRID + x] = value;
  };
  set(8, 8);
  set(7, 8);
  set(9, 8);
  set(8, 7);
  set(8, 9);
  set(3, 3, 96);
  set(12, 12, 180);
  return signature;
}

function diagonalSignature() {
  const signature = new Array(BINS).fill(0);
  for (let index = 2; index < 14; index += 1) {
    signature[index * GRID + index] = index % 2 === 0 ? 192 : 255;
  }
  signature[4 * GRID + 5] = 160;
  signature[5 * GRID + 4] = 160;
  return signature;
}

function channelPayload(tokens, marker) {
  const markerIndex = tokens.indexOf(marker);
  assert(markerIndex >= 0, `missing channel marker ${marker}`);
  return tokens.slice(markerIndex + 1, markerIndex + 1 + BINS);
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

function main() {
  const config = parseArgs(process.argv.slice(2));
  const cases = [];
  cases.push(runCase("good-profile-lengths", () => {
    const signature = plusSignature();
    const options = imageOptions();
    const ink = solomonImage.imageTaskTokens(signature, "ink16", options);
    const edge = solomonImage.imageTaskTokens(signature, "ink-edge16", options);
    const symbolic = solomonImage.imageTaskTokens(signature, "symbolic16", options);
    assert(ink.length === BINS, `ink16 length ${ink.length} != ${BINS}`);
    assert(edge.length === 2 * (BINS + 1), `ink-edge16 length ${edge.length} != ${2 * (BINS + 1)}`);
    assert(symbolic.length === 5 * (BINS + 1), `symbolic16 length ${symbolic.length} != ${5 * (BINS + 1)}`);
    return { ink_tokens: ink.length, ink_edge_tokens: edge.length, symbolic_tokens: symbolic.length };
  }));
  cases.push(runCase("good-channel-marker-order", () => {
    const tokens = solomonImage.symbolicImageTokens(plusSignature(), imageOptions());
    const markers = Object.values(CHANNEL_TOKENS);
    const offsets = markers.map((marker) => tokens.indexOf(marker));
    assert(offsets.every((offset) => offset >= 0), `missing marker offset in ${JSON.stringify(offsets)}`);
    assert(offsets.every((offset, index) => offset === index * (BINS + 1)), `bad marker offsets ${offsets.join(",")}`);
    return { marker_offsets: offsets };
  }));
  cases.push(runCase("good-crossing-and-direction-hints", () => {
    const tokens = solomonImage.symbolicImageTokens(plusSignature(), imageOptions());
    const component = channelPayload(tokens, CHANNEL_TOKENS.component);
    const radial = channelPayload(tokens, CHANNEL_TOKENS.radial);
    const direction = channelPayload(tokens, CHANNEL_TOKENS.direction);
    const center = 8 * GRID + 8;
    const componentBin = component[center] - IMAGE_BASE;
    const radialBin = radial[center] - IMAGE_BASE;
    const directionBin = direction[center] - IMAGE_BASE;
    assert(componentBin >= 8, `component crossing bin ${componentBin} < 8`);
    assert(radialBin > 0, `radial center bin ${radialBin} <= 0`);
    assert(directionBin === 15, `direction junction bin ${directionBin} != 15`);
    return { component_crossing_bin: componentBin, radial_bin: radialBin, direction_junction_bin: directionBin };
  }));
  cases.push(runCase("good-channel-stats", () => {
    const stats = solomonImage.imageTokenChannelStats([plusSignature(), diagonalSignature()], "symbolic16", imageOptions());
    for (const channel of solomonImage.imageTokenChannels("symbolic16")) {
      const item = stats[channel];
      assert(item.records === 2, `${channel} records ${item.records} != 2`);
      assert(item.tokens_per_record === BINS, `${channel} tokens_per_record ${item.tokens_per_record} != ${BINS}`);
      assert(item.active_records === 2, `${channel} active_records ${item.active_records} != 2`);
      assert(item.distinct_bins >= 2, `${channel} distinct_bins ${item.distinct_bins} < 2`);
    }
    return { channel_stats: stats };
  }));
  cases.push(runCase("bad-unknown-profile", () => {
    let rejected = false;
    try {
      solomonImage.imageTaskTokens(plusSignature(), "raw-anything", imageOptions());
    } catch (error) {
      rejected = String(error instanceof Error ? error.message : error).includes("unknown image token profile");
    }
    assert(rejected, "unknown image token profile was not rejected");
    return {};
  }));

  const report = {
    schema,
    ok: cases.every((item) => item.ok),
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
