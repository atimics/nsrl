#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

const analyzer = new URL("./analyze-production-optimizer-residuals-v1.mjs", import.meta.url).pathname;
const groups = [
  "embeddings", "attention_rms", "mlp_rms", "final_rms", "q", "k", "v",
  "o", "up", "gate", "down", "output", "bias",
];
const lengths = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2];

function fnv1a(bytes) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash;
}

function optimizerBytes(corruptChecksum = false) {
  const residuals = lengths.flatMap((length, group) =>
    Array.from({ length }, () => groups[group] === "v" ? 16n : 1n));
  const bytes = Buffer.alloc(76 + residuals.length * 8);
  bytes.write("NSRLPO2\n", 0);
  bytes.writeUInt32LE(2, 8);
  for (const offset of [12, 20, 28, 44]) bytes.writeBigUInt64LE(1n, offset);
  bytes.writeBigUInt64LE(4n, 36);
  bytes.writeBigUInt64LE(BigInt(residuals.length), 68);
  residuals.forEach((value, index) => bytes.writeBigInt64LE(value, 76 + index * 8));
  const checksum = Buffer.alloc(8);
  checksum.writeBigUInt64LE(fnv1a(bytes) ^ (corruptChecksum ? 1n : 0n));
  return Buffer.concat([bytes, checksum]);
}

const root = await mkdtemp(path.join(tmpdir(), "nsrl-residual-analysis-self-test-"));
try {
  const optimizerPath = path.join(root, "optimizer.nsrlpo");
  const tracePath = path.join(root, "trace.json");
  const planPath = path.join(root, "plan.json");
  const outPath = path.join(root, "analysis.json");
  const shifts = Object.fromEntries(groups.map((group) => [group, group === "v" ? 6 : 10]));
  const counts = Object.fromEntries(groups.map((group) => [group, 1]));
  const updates = Object.fromEntries(groups.map((group) => [group, group === "output" ? 1 : 0]));
  await Promise.all([
    writeFile(optimizerPath, optimizerBytes()),
    writeFile(tracePath, JSON.stringify({
      profile: "tiny", parameter_count: 16,
      training: { learning_rate_shifts: shifts },
      diagnostics: { gradient_nonzero_count: counts, update_nonzero_count: updates },
      hashes: { optimizer_state: "0x1", final_model: "0x2" },
    })),
    writeFile(planPath, JSON.stringify({
      tokenizer: { vocab_size: 2 },
      points: [{ id: "tiny", d_model: 1, hidden_dim: 1, layers: 1, parameter_count: 16 }],
    })),
  ]);
  const result = spawnSync(process.execPath, [analyzer, "--optimizer", optimizerPath,
    "--trace", tracePath, "--plan", planPath, "--out", outPath], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  const analysis = JSON.parse(await readFile(outPath, "utf8"));
  assert.equal(analysis.recommendation.group, "v");
  assert.equal(analysis.recommendation.source_shift, 6);
  assert.equal(analysis.recommendation.candidate_shift, 5);
  assert.equal(analysis.recommendation.shift_reduction, 1);
  assert.equal(analysis.recommendation.predicted_parameter_crossings, 1);

  await writeFile(optimizerPath, optimizerBytes(true));
  const corrupt = spawnSync(process.execPath, [analyzer, "--optimizer", optimizerPath,
    "--trace", tracePath, "--plan", planPath], { encoding: "utf8" });
  assert.notEqual(corrupt.status, 0);
  assert.match(corrupt.stderr, /checksum mismatch/);
  console.log(JSON.stringify({ passed: true, checks: 8 }));
} finally {
  await rm(root, { recursive: true, force: true });
}
