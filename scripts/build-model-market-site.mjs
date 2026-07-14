#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

import { buildDeterministicMarketDemo } from "./lib/model-market-demo-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");
const DEFAULT_OUT = path.join(ROOT, "web/launches/market.json");

function parseArgs(argv) {
  let out = DEFAULT_OUT;
  let check = false;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--out" && argv[index + 1]) {
      out = path.resolve(argv[index + 1]);
      index += 1;
    } else if (argv[index] === "--check") {
      check = true;
    } else {
      throw new Error(`unknown or incomplete argument ${argv[index]}`);
    }
  }
  return { out, check };
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-model-market-site-"));
  const { snapshot } = buildDeterministicMarketDemo(directory);
  const output = `${JSON.stringify(snapshot, null, 2)}\n`;
  if (options.check) {
    if (!fs.existsSync(options.out) || fs.readFileSync(options.out, "utf8") !== output) {
      throw new Error(
        `${path.relative(ROOT, options.out)} is stale; rebuild it with scripts/build-model-market-site.mjs`,
      );
    }
    process.stdout.write(`${path.relative(ROOT, options.out)} is current\n`);
    return;
  }
  fs.mkdirSync(path.dirname(options.out), { recursive: true });
  fs.writeFileSync(options.out, output);
  process.stdout.write(`${path.relative(ROOT, options.out)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
}
