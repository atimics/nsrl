#!/usr/bin/env node
import fs from "node:fs";

const minTextFeatureChannels = 30;

function usage() {
  console.log("Usage: check-solomon-denoiser-model.mjs --model PATH");
}

function parseArgs(argv) {
  let model = "";
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (arg === "--model") {
      model = argv[++index] ?? "";
      continue;
    }
    throw new Error(`unknown option: ${arg}`);
  }
  if (!model) {
    throw new Error("--model is required");
  }
  return { model };
}

function readU32(buffer, state, label) {
  if (state.offset + 4 > buffer.length) {
    throw new Error(`${label}: unexpected end of model`);
  }
  const value = buffer.readUInt32LE(state.offset);
  state.offset += 4;
  return value;
}

function main() {
  const { model } = parseArgs(process.argv.slice(2));
  const buffer = fs.readFileSync(model);
  if (buffer.length < 36) {
    throw new Error(`${model}: model is too small`);
  }
  const magic = buffer.subarray(0, 8).toString("latin1");
  const state = { offset: 8 };
  const imageSize = readU32(buffer, state, "image_size");
  const timesteps = readU32(buffer, state, "timesteps");
  const hiddenShift = readU32(buffer, state, "hidden_shift");
  const outputShift = readU32(buffer, state, "output_shift");
  const featureChannels = readU32(buffer, state, "feature_count");
  const corruptions = readU32(buffer, state, "corruptions");
  const layers = readU32(buffer, state, "layers");
  const failures = [];
  if (magic !== "NSRLTCH\n") {
    failures.push(`${model}: expected NSRLTCH text-conditioned denoiser, got ${JSON.stringify(magic)}`);
  }
  if (featureChannels < minTextFeatureChannels) {
    failures.push(
      `${model}: NSRLTCH feature_count ${featureChannels} < ${minTextFeatureChannels}; text/layout channels are unreachable`,
    );
  }
  if (imageSize <= 0 || timesteps <= 0 || corruptions <= 0 || layers <= 0) {
    failures.push(`${model}: invalid non-positive header field`);
  }
  const report = {
    schema: "nsrl.solomon_denoiser_model_check.v1",
    passed: failures.length === 0,
    model,
    magic,
    image_size: imageSize,
    timesteps,
    hidden_shift: hiddenShift,
    output_shift: outputShift,
    feature_channels: featureChannels,
    corruptions,
    layers,
    min_text_feature_channels: minTextFeatureChannels,
    failures,
  };
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) {
    process.exit(1);
  }
}

main();
