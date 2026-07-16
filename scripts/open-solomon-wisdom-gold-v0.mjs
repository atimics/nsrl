#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {stableJson} from "./lib/solomon-council-v0.mjs";
import {
  openWisdomGold,
  verifyWisdomLanesForOpening,
} from "./lib/solomon-wisdom-ceremony-v0.mjs";

const args = process.argv.slice(2);
const check = args[0] === "--check";
if (check) args.shift();
if (args.length !== 5 || args.includes("--help") || args.includes("-h")) {
  process.stdout.write([
    "Usage: node scripts/open-solomon-wisdom-gold-v0.mjs [--check] \\",
    "  CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json PRIVATE-GOLD-VAULT.json \\",
    "  GOLD-OPENING.json",
    "",
    "The command validates both complete lane bundles before reading the private",
    "gold vault. The emitted opening binds the casebook and both lane hashes.",
  ].join("\n") + "\n");
  process.exit(args.length === 0 || args.includes("--help") || args.includes("-h") ? 0 : 2);
}

const [casebookPath, soloPath, councilPath, vaultPath, openingPath] = args;
const readJson = (filePath) => JSON.parse(fs.readFileSync(filePath, "utf8"));
const casebook = readJson(casebookPath);
const soloBundle = readJson(soloPath);
const councilBundle = readJson(councilPath);

// Do not read the vault until both lane bundles pass byte and receipt replay.
verifyWisdomLanesForOpening({casebook, soloBundle, councilBundle}, {baseDir: process.cwd()});
const vault = readJson(vaultPath);
const opening = openWisdomGold(
  {casebook, soloBundle, councilBundle, vault}, {baseDir: process.cwd()});
if (check) {
  const actual = readJson(openingPath);
  if (stableJson(actual) !== stableJson(opening)) {
    throw new Error("wisdom gold opening does not replay from the sealed artifacts");
  }
} else {
  if (fs.existsSync(openingPath)) throw new Error("refusing to overwrite an existing gold opening");
  fs.mkdirSync(path.dirname(path.resolve(openingPath)), {recursive: true});
  fs.writeFileSync(openingPath, `${JSON.stringify(opening, null, 2)}\n`, {flag: "wx"});
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_gold_open.v0",
  checked: check,
  opening: openingPath,
  cases: opening.gold.length,
  solo_bundle_sha256: opening.solo_bundle_sha256,
  council_bundle_sha256: opening.council_bundle_sha256,
}, null, 2)}\n`);
