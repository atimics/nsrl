#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {evaluateWisdom} from "./lib/solomon-wisdom-eval-v0.mjs";

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath || process.argv.length !== 4) {
  process.stderr.write("Usage: node scripts/evaluate-solomon-wisdom-v0.mjs INPUT.json RESULT.json\n");
  process.exitCode = 2;
} else {
  const input = JSON.parse(fs.readFileSync(inputPath));
  const evaluatorSha256 = crypto.createHash("sha256")
    .update(fs.readFileSync(fileURLToPath(import.meta.url)))
    .update(fs.readFileSync(new URL("./lib/solomon-wisdom-eval-v0.mjs", import.meta.url)))
    .digest("hex");
  const result = evaluateWisdom(input, {evaluatorSha256});
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    schema: result.schema,
    analysis_role: result.analysis_role,
    all_dimensions_outperform: result.verdict.all_dimensions_outperform,
    promotion_gate_passed: result.verdict.promotion_gate_passed,
    output: outputPath,
  }, null, 2)}\n`);
}
