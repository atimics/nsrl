#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

import {
  deliberate,
  loadCouncilAuthority,
  verifyReceipt,
} from "./lib/solomon-council-v0.mjs";

function usage() {
  process.stdout.write([
    "Usage:",
    "  node scripts/run-solomon-council-v0.mjs REQUEST.json RECEIPT.json",
    "  node scripts/run-solomon-council-v0.mjs --check REQUEST.json RECEIPT.json",
    "",
    "Solomon Council v0 is shadow-only. It records a recommendation, evidence/user",
    "request, or abstention, but never executes the selected action.",
  ].join("\n") + "\n");
}

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
  usage();
  process.exit(0);
}
const check = args[0] === "--check";
const requestPath = check ? args[1] : args[0];
const receiptPath = check ? args[2] : args[1];
if (!requestPath || !receiptPath || args.length !== (check ? 3 : 2)) {
  usage();
  process.exitCode = 2;
} else {
  const request = JSON.parse(fs.readFileSync(requestPath));
  const authority = loadCouncilAuthority();
  if (check) {
    const receipt = JSON.parse(fs.readFileSync(receiptPath));
    verifyReceipt(receipt, request, authority);
    const replay = `${JSON.stringify(deliberate(request, authority), null, 2)}\n`;
    if (!Buffer.from(replay).equals(fs.readFileSync(receiptPath))) {
      throw new Error("wisdom receipt JSON byte replay changed");
    }
    process.stdout.write(`${JSON.stringify({
      schema: "nsrl.solomon_council_check.v0",
      request_id: request.request_id,
      receipt_id: receipt.receipt_id,
      decision: receipt.decision.kind,
      selected_action_id: receipt.decision.selected_action_id,
      seals_verified: receipt.faculty_invocations.length,
      shadow_execution: receipt.shadow_execution.action_executed === false,
      byte_replay: true,
    }, null, 2)}\n`);
  } else {
    const receipt = deliberate(request, authority);
    const bytes = `${JSON.stringify(receipt, null, 2)}\n`;
    fs.mkdirSync(path.dirname(receiptPath), {recursive: true});
    fs.writeFileSync(receiptPath, bytes);
    process.stdout.write(`${JSON.stringify({
      schema: receipt.schema,
      request_id: request.request_id,
      receipt_id: receipt.receipt_id,
      decision: receipt.decision.kind,
      selected_action_id: receipt.decision.selected_action_id,
      controller_allowed: receipt.decision.mathematical_controller_allowed,
      action_executed: false,
      receipt_sha256: receipt.identity.receipt_sha256,
      output: receiptPath,
    }, null, 2)}\n`);
  }
}
