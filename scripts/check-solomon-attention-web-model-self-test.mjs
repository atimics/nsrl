#!/usr/bin/env node

import { MiniTransformerModel, SolomonAttentionSampler } from "../web/attention-sampler.js";

function fixture({ magic, dModel, heads, hiddenDim, layers, contextSeqLen, rms }) {
  const embeddingCount = 256 * dModel;
  const positionCount = contextSeqLen * dModel;
  const attentionCount = layers * dModel * dModel;
  const upCount = layers * dModel * hiddenDim;
  const downCount = layers * hiddenDim * dModel;
  const outputCount = 256 * dModel;
  const rmsCount = rms ? layers * dModel : 0;
  const headerBytes = 8 + 5 * 4 + 10 * 8 + 8 * 8;
  const payloadBytes =
    (embeddingCount + positionCount + 2 * rmsCount) * 2 +
    4 * attentionCount +
    2 * upCount +
    downCount +
    outputCount;
  const bytes = Buffer.alloc(headerBytes + payloadBytes);
  let offset = 0;
  bytes.write(magic, offset, "ascii");
  offset += 8;
  for (const value of [256, dModel, heads, hiddenDim, contextSeqLen]) {
    bytes.writeUInt32LE(value, offset);
    offset += 4;
  }
  for (const value of [
    embeddingCount,
    positionCount,
    attentionCount,
    attentionCount,
    attentionCount,
    attentionCount,
    upCount,
    upCount,
    downCount,
    outputCount,
  ]) {
    bytes.writeBigUInt64LE(BigInt(value), offset);
    offset += 8;
  }
  offset += 8 * 8;
  if (offset !== headerBytes) throw new Error("fixture header size mismatch");
  return bytes;
}

function wrapSolomonV5(transformerBytes, contextSeqLen) {
  const header = Buffer.alloc(8 + 13 * 4 + 4 * 8);
  let offset = 0;
  header.write("NSRLLMM1", offset, "ascii");
  offset += 8;
  for (const value of [5, 0, 256, 16, 128, 144, 16, 16, 0, 0, 0]) {
    header.writeUInt32LE(value, offset);
    offset += 4;
  }
  // No embedded text memory, then context length.
  header.writeUInt32LE(0, offset);
  offset += 4;
  header.writeUInt32LE(contextSeqLen, offset);
  offset += 4;
  for (const value of [0n, 0n, 0n, BigInt(transformerBytes.length)]) {
    header.writeBigUInt64LE(value, offset);
    offset += 8;
  }
  if (offset !== header.length) throw new Error("Solomon fixture header size mismatch");
  const body = Buffer.concat([header, transformerBytes]);
  let hash = 0xcbf29ce484222325n;
  for (const byte of body) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  const trailer = Buffer.alloc(8);
  trailer.writeBigUInt64LE(hash);
  return Buffer.concat([body, trailer]);
}

const cases = [
  {
    name: "historical-v4",
    bytes: fixture({
      magic: "NSRLMT4\n",
      dModel: 32,
      heads: 2,
      hiddenDim: 64,
      layers: 1,
      contextSeqLen: 2,
      rms: false,
    }),
    expected: { dModel: 32, layers: 1, rms: false },
  },
  {
    name: "promoted-v5-stacked-rms",
    bytes: fixture({
      magic: "NSRLMT5\n",
      dModel: 128,
      heads: 2,
      hiddenDim: 256,
      layers: 2,
      contextSeqLen: 2,
      rms: true,
    }),
    expected: { dModel: 128, layers: 2, rms: true },
  },
];

const report = [];
for (const testCase of cases) {
  const model = new MiniTransformerModel(testCase.bytes);
  const row = model.forward(Uint8Array.from([1, 2]), 0);
  const actual = {
    dModel: model.dModel,
    layers: model.layers,
    rms: model.attentionRmsWeights !== null,
  };
  const ok =
    JSON.stringify(actual) === JSON.stringify(testCase.expected) &&
    row.logits.length === 256 &&
    row.probabilities.length === 256;
  report.push({ name: testCase.name, ok, actual });
}

const ok = report.every((entry) => entry.ok);
const promoted = cases.find((entry) => entry.name === "promoted-v5-stacked-rms");
const outer = new SolomonAttentionSampler(wrapSolomonV5(promoted.bytes, 2));
const outerOk =
  outer.transformer.dModel === 128 &&
  outer.transformer.layers === 2 &&
  outer.transformer.attentionRmsWeights !== null;
report.push({ name: "solomon-v5-wrapper", ok: outerOk });
const finalOk = ok && outerOk;
process.stdout.write(`${JSON.stringify({ schema: "nsrl.solomon_attention_web_model_self_test.v1", ok: finalOk, cases: report }, null, 2)}\n`);
if (!finalOk) process.exitCode = 1;
