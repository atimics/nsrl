#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {
  reviseReceipt,
  verifyReceiptRevision,
} from "./lib/solomon-council-v0.mjs";

function usage() {
  process.stdout.write([
    "Usage:",
    "  node scripts/revise-solomon-wisdom-receipt-v0.mjs PRIOR.json OBSERVATION.json REVISED.json",
    "  node scripts/revise-solomon-wisdom-receipt-v0.mjs --check PRIOR.json OBSERVATION.json REVISED.json",
  ].join("\n") + "\n");
}

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
  usage();
  process.exit(0);
}
const check = args[0] === "--check";
const priorPath = check ? args[1] : args[0];
const observationPath = check ? args[2] : args[1];
const revisedPath = check ? args[3] : args[2];
if (!priorPath || !observationPath || !revisedPath || args.length !== (check ? 4 : 3)) {
  usage();
  process.exitCode = 2;
} else {
  const prior = JSON.parse(fs.readFileSync(priorPath));
  const observation = JSON.parse(fs.readFileSync(observationPath));
  if (check) {
    const revised = JSON.parse(fs.readFileSync(revisedPath));
    verifyReceiptRevision(revised, prior, observation);
    const replay = `${JSON.stringify(reviseReceipt(prior, observation), null, 2)}\n`;
    if (!Buffer.from(replay).equals(fs.readFileSync(revisedPath))) {
      throw new Error("revised wisdom receipt JSON byte replay changed");
    }
    process.stdout.write(`${JSON.stringify({
      schema: "nsrl.wisdom_receipt_revision_check.v0",
      receipt_id: revised.receipt_id,
      revision_index: revised.revisions.at(-1).revision_index,
      prior_receipt_sha256: prior.identity.receipt_sha256,
      revised_receipt_sha256: revised.identity.receipt_sha256,
      byte_replay: true,
    }, null, 2)}\n`);
  } else {
    const revised = reviseReceipt(prior, observation);
    fs.mkdirSync(path.dirname(revisedPath), {recursive: true});
    fs.writeFileSync(revisedPath, `${JSON.stringify(revised, null, 2)}\n`);
    process.stdout.write(`${JSON.stringify({
      schema: revised.schema,
      receipt_id: revised.receipt_id,
      revision_index: revised.revisions.at(-1).revision_index,
      prior_receipt_sha256: prior.identity.receipt_sha256,
      revised_receipt_sha256: revised.identity.receipt_sha256,
      output: revisedPath,
    }, null, 2)}\n`);
  }
}
