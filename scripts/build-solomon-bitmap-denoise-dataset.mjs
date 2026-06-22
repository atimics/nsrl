#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = [
  "run",
  "--release",
  "-q",
  "-p",
  "nsrl-train",
  "--bin",
  "nsrl-build-solomon-bitmap-denoise-dataset",
  "--",
  ...process.argv.slice(2),
];

const result = spawnSync("cargo", args, {
  cwd: repoRoot,
  stdio: "inherit",
});

if (result.error) {
  console.error(`build-solomon-bitmap-denoise-dataset: ${result.error.message}`);
  process.exit(1);
}

process.exit(result.status ?? 1);
