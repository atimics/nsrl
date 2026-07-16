#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {stableJson} from "./lib/solomon-council-v0.mjs";
import {freezeWisdomCasebook} from "./lib/solomon-wisdom-ceremony-v0.mjs";

const args = process.argv.slice(2);
const check = args[0] === "--check";
if (check) args.shift();
if (args.length !== 3 || args.includes("--help") || args.includes("-h")) {
  process.stdout.write([
    "Usage: node scripts/freeze-solomon-wisdom-casebook-v0.mjs [--check] \\",
    "  PRIVATE-DRAFT.json PUBLIC-CASEBOOK.json PRIVATE-GOLD-VAULT.json",
    "",
    "The draft and vault contain secret gold. Publish only the casebook before",
    "lane generation. Production nonces must be independent 256-bit hex values.",
  ].join("\n") + "\n");
  process.exit(args.length === 0 || args.includes("--help") || args.includes("-h") ? 0 : 2);
}

const [draftPath, casebookPath, vaultPath] = args;
if (path.resolve(casebookPath) === path.resolve(vaultPath)) {
  throw new Error("public casebook and private gold vault must use different paths");
}
const draft = JSON.parse(fs.readFileSync(draftPath, "utf8"));
const frozen = freezeWisdomCasebook(draft, {baseDir: process.cwd()});
if (check) {
  const actualCasebook = JSON.parse(fs.readFileSync(casebookPath, "utf8"));
  const actualVault = JSON.parse(fs.readFileSync(vaultPath, "utf8"));
  if (stableJson(actualCasebook) !== stableJson(frozen.casebook)) {
    throw new Error("public wisdom casebook does not replay from the private draft");
  }
  if (stableJson(actualVault) !== stableJson(frozen.vault)) {
    throw new Error("private wisdom gold vault does not replay from the private draft");
  }
} else {
  if (fs.existsSync(casebookPath) || fs.existsSync(vaultPath)) {
    throw new Error("refusing to overwrite an existing casebook or gold vault");
  }
  fs.mkdirSync(path.dirname(path.resolve(casebookPath)), {recursive: true});
  fs.mkdirSync(path.dirname(path.resolve(vaultPath)), {recursive: true});
  fs.writeFileSync(vaultPath, `${JSON.stringify(frozen.vault, null, 2)}\n`, {mode: 0o600, flag: "wx"});
  fs.writeFileSync(casebookPath, `${JSON.stringify(frozen.casebook, null, 2)}\n`, {flag: "wx"});
}
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_wisdom_casebook_freeze.v0",
  checked: check,
  public_casebook: casebookPath,
  private_gold_vault: vaultPath,
  cases: frozen.casebook.cases.length,
  analysis_role: frozen.casebook.analysis_role,
}, null, 2)}\n`);
