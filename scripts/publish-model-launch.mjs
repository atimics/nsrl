#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  createModelPublishReceipt,
  validateModelLaunchRecipe,
} from "./lib/model-launch-v1.mjs";

function parseArgs(argv) {
  const options = {
    recipe: "protocol/examples/integer-transformer-proof-v1.launch.json",
    event: "model_promoted",
    height: 0,
    previousBlockSha256: "0".repeat(64),
    cumulativeSupplyUnits: "0",
    out: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = argv[index + 1];
    if (arg === "--recipe" && value) {
      options.recipe = value;
      index += 1;
    } else if (arg === "--event" && value) {
      options.event = value;
      index += 1;
    } else if (arg === "--height" && value) {
      options.height = Number.parseInt(value, 10);
      index += 1;
    } else if (arg === "--previous-block" && value) {
      options.previousBlockSha256 = value;
      index += 1;
    } else if (arg === "--cumulative-supply" && value) {
      options.cumulativeSupplyUnits = value;
      index += 1;
    } else if (arg === "--out" && value) {
      options.out = value;
      index += 1;
    } else {
      throw new Error(`unknown or incomplete argument ${arg}`);
    }
  }
  return options;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const recipePath = path.resolve(options.recipe);
  const recipe = JSON.parse(fs.readFileSync(recipePath, "utf8"));
  validateModelLaunchRecipe(recipe);
  const receipt = createModelPublishReceipt(recipe, options);
  const output = `${JSON.stringify(receipt, null, 2)}\n`;
  if (options.out) {
    const outputPath = path.resolve(options.out);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, output);
    process.stdout.write(`${outputPath}\n`);
  } else {
    process.stdout.write(output);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
