#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

import {sha256Bytes} from "./lib/solomon-council-v0.mjs";
import {compileWisdomCeremony} from "./lib/solomon-wisdom-ceremony-v0.mjs";
import {evaluateWisdom} from "./lib/solomon-wisdom-eval-v0.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const paths = {
  casebook: "benchmarks/solomon-council-v0/production-v0/casebook.json",
  solo: "benchmarks/solomon-council-v0/production-v0/solo-bundle.json",
  council: "benchmarks/solomon-council-v0/production-v0/council-bundle.json",
  opening: "benchmarks/solomon-council-v0/production-v0/gold-opening.json",
  generation: "benchmarks/solomon-council-v0/production-v0/generation-integrity.json",
  provenance: "benchmarks/solomon-council-v0/production-v0/provenance.json",
  input: "benchmarks/solomon-council-v0/production-v0/eval-input.json",
  result: "benchmarks/solomon-council-v0/wisdom-eval-result.json",
};
const absolute = (relative) => path.join(root, relative);
const readJson = (relative) => JSON.parse(fs.readFileSync(absolute(relative), "utf8"));
const binding = (relative) => ({
  path: relative,
  sha256: sha256Bytes(fs.readFileSync(absolute(relative))),
});
const expectedInput = compileWisdomCeremony({
  casebook: readJson(paths.casebook),
  soloBundle: readJson(paths.solo),
  councilBundle: readJson(paths.council),
  opening: readJson(paths.opening),
  integrityBindings: {
    generation_integrity_report: binding(paths.generation),
    provenance_report: binding(paths.provenance),
  },
  ceremonyBindings: {
    casebook: binding(paths.casebook),
    solo_bundle: binding(paths.solo),
    council_bundle: binding(paths.council),
    gold_opening: binding(paths.opening),
  },
}, {baseDir: root});
const expectedInputBytes = Buffer.from(`${JSON.stringify(expectedInput, null, 2)}\n`);
assert(fs.readFileSync(absolute(paths.input)).equals(expectedInputBytes),
  "frozen wisdom evaluation input does not byte-replay from the sealed ceremony");

const evaluatorPath = absolute("scripts/evaluate-solomon-wisdom-v0.mjs");
const evaluatorSha256 = crypto.createHash("sha256")
  .update(fs.readFileSync(evaluatorPath))
  .update(fs.readFileSync(absolute("scripts/lib/solomon-wisdom-eval-v0.mjs")))
  .digest("hex");
const expectedResult = evaluateWisdom(expectedInput, {evaluatorSha256, artifactBase: root});
const expectedResultBytes = Buffer.from(`${JSON.stringify(expectedResult, null, 2)}\n`);
assert(fs.readFileSync(absolute(paths.result)).equals(expectedResultBytes),
  "frozen wisdom result does not byte-replay from its evaluation input");
assert(expectedResult.analysis_role === "frozen_same_model_comparison"
  && expectedResult.verdict.all_dimensions_outperform === true
  && expectedResult.verdict.promotion_gate_passed === true
  && expectedResult.authorization.council_promotion_authorized === true
  && expectedResult.authorization.product_release_authorized === false,
"production wisdom verdict or authorization changed");
assert(Object.values(expectedResult.dimensions).every(
  (dimension) => dimension.cases === 72 && dimension.council_outperforms === true),
"a production wisdom dimension lacks 72 cases or strict council improvement");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_production_check.v0",
  ok: true,
  cases: expectedInput.episodes.length,
  input_sha256: sha256Bytes(expectedInputBytes),
  result_sha256: sha256Bytes(expectedResultBytes),
  evaluator_sha256: evaluatorSha256,
  all_dimensions_outperform: true,
  council_promotion_authorized: true,
  product_release_authorized: false,
}, null, 2)}\n`);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
