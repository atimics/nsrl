#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const runner = path.join(scriptDir, "run-solomon-eval-scaling-curve.mjs");

const defaults = [
  "--prompts",
  "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  "--sizes",
  "1152,1425",
  "--latent-dims",
  "32,64,128",
  "--text-features",
  "512,2048,8192",
  "--epochs",
  "12",
  "--out-dir",
  "data/processed/key-solomon-goetia-latent-v1/text-feature-sweep",
  "--report-out",
  "docs/solomon-text-feature-sweep.tsv",
];

function usage() {
  console.log(
    [
      "Usage: run-solomon-text-feature-sweep.mjs [runner options]",
      "",
      "Runs the next Solomon eval sweep:",
      "  sizes: 1152,1425",
      "  latent dims: 32,64,128",
      "  text features: 512,2048,8192",
      "  epochs: 12",
      "",
      "Any option after the script name is passed to run-solomon-eval-scaling-curve.mjs.",
      "Repeated options use the later value, so overrides can be appended.",
    ].join("\n"),
  );
}

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  usage();
  process.exit(0);
}

const result = spawnSync(process.execPath, [runner, ...defaults, ...process.argv.slice(2)], {
  stdio: "inherit",
});

if (result.error) {
  throw result.error;
}
process.exit(result.status ?? 1);
