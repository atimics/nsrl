#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

import {
  deliberate,
  loadCouncilAuthority,
  verifyReceipt,
  verifyReceiptRevision,
  verifySeal,
} from "./lib/solomon-council-v0.mjs";

const requestPath = process.argv[2]
  ?? "benchmarks/solomon-council-v0/fixtures/select-request.json";
const receiptPath = process.argv[3]
  ?? "benchmarks/solomon-council-v0/fixtures/select-receipt.json";
const observationPath = process.argv[4]
  ?? "benchmarks/solomon-council-v0/fixtures/select-observation.json";
const revisedReceiptPath = process.argv[5]
  ?? "benchmarks/solomon-council-v0/fixtures/select-revised-receipt.json";

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const sha256File = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const clone = (value) => structuredClone(value);
const expectFailure = (operation, pattern) => {
  try {
    operation();
  } catch (error) {
    assert(pattern.test(String(error.message)), `wrong fail-closed error: ${error.message}`);
    return;
  }
  throw new Error(`expected failure matching ${pattern}`);
};

const requestBytes = fs.readFileSync(requestPath);
const receiptBytes = fs.readFileSync(receiptPath);
const request = JSON.parse(requestBytes);
const receipt = JSON.parse(receiptBytes);
const observation = JSON.parse(fs.readFileSync(observationPath));
const revisedReceipt = JSON.parse(fs.readFileSync(revisedReceiptPath));
const authority = loadCouncilAuthority();

for (const model of request.models) {
  assert(sha256File(model.artifact_uri) === model.artifact_sha256,
    `fixture model binding changed: ${model.artifact_uri}`);
}
for (const evidence of request.evidence) {
  assert(sha256File(evidence.source_uri) === evidence.source_sha256,
    `fixture source binding changed: ${evidence.source_uri}`);
}
verifyReceipt(receipt, request, authority);
verifyReceiptRevision(revisedReceipt, receipt, observation);
const replayBytes = Buffer.from(`${JSON.stringify(deliberate(request, authority), null, 2)}\n`);
assert(replayBytes.equals(receiptBytes), "frozen wisdom receipt byte replay changed");
assert(receipt.decision.kind === "select"
  && receipt.decision.selected_action_id === "record-substrate-promotion"
  && receipt.decision.mathematical_controller_allowed === true,
"base fixture did not select its controller-allowed action");
assert(receipt.deliberation.dissent.some((entry) => entry.faculty_id === "skeptic"
  && entry.disposition === "oppose"), "skeptic dissent was not preserved");
assert(receipt.shadow_execution.action_execution_allowed === false
  && receipt.shadow_execution.action_executed === false,
"shadow fixture claims execution authority");
assert(receipt.outcome.status === "pending" && receipt.revisions.length === 0,
  "initial wisdom receipt fabricated an outcome or revision");

const tamperedSeal = clone(authority.manifests.get("engineer").manifest);
tamperedSeal.capabilities.push("execute_repository_change");
expectFailure(() => verifySeal(tamperedSeal, authority.trust), /signature invalid/);

const overBudget = clone(request);
overBudget.invocations.find((entry) => entry.faculty_id === "mathematician")
  .circle.usage.tokens = 2049;
expectFailure(() => deliberate(overBudget, authority), /usage tokens/);

const forbidden = clone(request);
forbidden.controller.allowed_action_ids = [];
forbidden.controller.forbidden_action_ids = ["record-substrate-promotion"];
const abstentionReceipt = deliberate(forbidden, authority);
assert(abstentionReceipt.decision.kind === "abstain"
  && abstentionReceipt.shadow_execution.action_executed === false,
"controller-forbidden action did not force abstention");

const askUser = clone(forbidden);
askUser.invocations.find((entry) => entry.faculty_id === "consequence_planner")
  .recommendation.missing_information.push({
    kind: "user",
    action_id: "record-substrate-promotion",
    question: "Should this recommendation remain limited to the substrate gate?",
  });
const askReceipt = deliberate(askUser, authority);
assert(askReceipt.decision.kind === "ask_user" && askReceipt.decision.questions.length === 1,
  "material user-only information did not route to ask_user");

const requestEvidence = clone(forbidden);
requestEvidence.invocations.find((entry) => entry.faculty_id === "historian")
  .recommendation.missing_information.push({
    kind: "evidence",
    action_id: "record-substrate-promotion",
    question: "Provide the frozen evaluator replay binding.",
  });
const evidenceReceipt = deliberate(requestEvidence, authority);
assert(evidenceReceipt.decision.kind === "request_evidence"
  && evidenceReceipt.decision.questions.length === 1,
"material evidence deficiency did not route to request_evidence");

const inaccessible = clone(request);
inaccessible.evidence[0].accessible_to = inaccessible.evidence[0].accessible_to.filter(
  (faculty) => faculty !== "skeptic");
expectFailure(() => deliberate(inaccessible, authority), /skeptic circle accessed forbidden evidence/);
const unsealedDisposition = clone(request);
unsealedDisposition.invocations.find((entry) => entry.faculty_id === "mathematician")
  .recommendation.disposition = "ask_user";
expectFailure(() => deliberate(unsealedDisposition, authority), /seal does not permit disposition/);

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.solomon_council_self_check.v0",
  faculties: receipt.faculty_invocations.map((entry) => entry.faculty_id),
  seals_verified: receipt.faculty_invocations.length,
  decision_states_exercised: ["select", "request_evidence", "ask_user", "abstain"],
  controller_forbidden_action_rejected: true,
  signature_tamper_rejected: true,
  circle_overrun_rejected: true,
  evidence_boundary_rejected: true,
  unsealed_disposition_rejected: true,
  dissent_preserved: true,
  outcome_not_fabricated: true,
  shadow_execution: true,
  exact_receipt_replay: true,
  exact_revision_replay: true,
  receipt_sha256: receipt.identity.receipt_sha256,
  revised_receipt_sha256: revisedReceipt.identity.receipt_sha256,
}, null, 2)}\n`);
