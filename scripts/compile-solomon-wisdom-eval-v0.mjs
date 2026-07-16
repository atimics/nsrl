#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {sha256Bytes} from "./lib/solomon-council-v0.mjs";
import {compileWisdomCeremony} from "./lib/solomon-wisdom-ceremony-v0.mjs";

const args = process.argv.slice(2);
if (args.length !== 7 || args.includes("--help") || args.includes("-h")) {
  process.stdout.write([
    "Usage: node scripts/compile-solomon-wisdom-eval-v0.mjs \\",
    "  CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json GOLD-OPENING.json \\",
    "  GENERATION-INTEGRITY.json PROVENANCE.json OUT.json",
    "",
    "Byte-verifies the frozen wisdom ceremony and compiles the only production",
    "input accepted by the same-model wisdom evaluator.",
  ].join("\n") + "\n");
  process.exit(args.length === 0 || args.includes("--help") || args.includes("-h") ? 0 : 2);
}

const [casebookPath, soloPath, councilPath, openingPath,
  generationPath, provenancePath, outPath] = args;
const readJson = (filePath) => JSON.parse(fs.readFileSync(filePath, "utf8"));
const binding = (filePath) => ({
  path: filePath,
  sha256: sha256Bytes(fs.readFileSync(filePath)),
});

const input = compileWisdomCeremony({
  casebook: readJson(casebookPath),
  soloBundle: readJson(soloPath),
  councilBundle: readJson(councilPath),
  opening: readJson(openingPath),
  integrityBindings: {
    generation_integrity_report: binding(generationPath),
    provenance_report: binding(provenancePath),
  },
  ceremonyBindings: {
    casebook: binding(casebookPath),
    solo_bundle: binding(soloPath),
    council_bundle: binding(councilPath),
    gold_opening: binding(openingPath),
  },
}, {baseDir: process.cwd()});

fs.mkdirSync(path.dirname(path.resolve(outPath)), {recursive: true});
fs.writeFileSync(outPath, `${JSON.stringify(input, null, 2)}\n`);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_ceremony_compile.v0",
  out: outPath,
  episodes: input.episodes.length,
  underlying_model_sha256: input.underlying_model.artifact_sha256,
  production: input.analysis_role === "frozen_same_model_comparison",
}, null, 2)}\n`);
