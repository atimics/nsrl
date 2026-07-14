#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  LOCALNET_EVENT_TYPES,
  ModelLocalnetLedger,
  buildModelPublicationPayload,
  replayLocalnetEvents,
  signLocalnetIntent,
} from "./lib/model-localnet-v1.mjs";
import { buildDeterministicLocalnetDemo } from "./lib/model-localnet-demo-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "..");

function expectFailure(operation, pattern) {
  assert.throws(operation, pattern);
}

function writeLedgerPrefix(directory, events) {
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(
    path.join(directory, "ledger.jsonl"),
    `${events.map((event) => JSON.stringify(event)).join("\n")}\n`,
  );
}

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-model-localnet-check-"));
const demo = buildDeterministicLocalnetDemo(directory);
const { events, state } = demo.ledger.inspect();
const summary = demo.snapshot.summary;

assert.equal(demo.duplicate_result.duplicate, true);
assert.equal(JSON.stringify(demo.snapshot).includes("private_key_pem"), false);
assert.equal(state.height, 31);
assert.equal(summary.accounts, 10);
assert.equal(summary.launches.length, 1);
assert.equal(summary.launches[0].status, "promoted");
assert.equal(summary.accepted_stages, 3);
assert.equal(summary.attestations, 9);
assert.deepEqual(summary.challenges, { rejected: 1 });
assert.equal(summary.publications, 1);
assert.deepEqual(
  summary.bounty_settlements["integer-transformer-proof-v1-localnet"][
    "beat-float-reference"
  ],
  {
    sponsor: "nsrl:sponsor:prototype",
    recipient: "nsrl:builder:integer-core",
    escrow_units: "120000",
    settled_units: "120000",
    refunded_units: "0",
  },
);
assert.deepEqual(summary.model_balances.ITP1, {
  "nsrl:builder:integer-core": "19200",
  "nsrl:compute:graviton-pool": "22400",
  "nsrl:validator:proof-v1": "12800",
  "nsrl:sponsor:prototype": "6400",
  "nsrl:treasury:public-goods": "3200",
});
assert.equal(
  Object.values(summary.model_balances.ITP1).reduce((sum, value) => sum + BigInt(value), 0n),
  64000n,
);

const tamperedSignature = structuredClone(events);
tamperedSignature[4].signed_intent.payload.account = "nsrl:tampered:account";
expectFailure(() => replayLocalnetEvents(tamperedSignature), /signature is invalid/);

const tamperedHash = structuredClone(events);
tamperedHash[8].event_sha256 = "f".repeat(64);
expectFailure(() => replayLocalnetEvents(tamperedHash), /invalid event_sha256/);

const reordered = structuredClone(events);
[reordered[12], reordered[13]] = [reordered[13], reordered[12]];
expectFailure(
  () => replayLocalnetEvents(reordered),
  /event height .* does not match expected|does not link to the previous event/,
);

const schema = JSON.parse(
  fs.readFileSync(path.join(ROOT, "protocol/model-localnet-v1.schema.json"), "utf8"),
);
assert.deepEqual(schema.$defs.eventType.enum, LOCALNET_EVENT_TYPES);

const firstStage = demo.stage_event_ids[0];
const firstAttestationIndex = events.findIndex(
  (event) =>
    event.signed_intent.event_type === "validation_attested" &&
    event.signed_intent.payload.subject_event_id === firstStage,
);
const quorumDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-model-localnet-quorum-"));
writeLedgerPrefix(quorumDirectory, events.slice(0, firstAttestationIndex));
const quorumLedger = new ModelLocalnetLedger(quorumDirectory);
expectFailure(
  () =>
    quorumLedger.append(
      signLocalnetIntent(demo.identities["nsrl:authority:localnet"], "stage_accepted", {
        stage_event_id: firstStage,
      }),
    ),
  /has not reached a clean validator quorum/,
);

const firstStageIndex = events.findIndex((event) => event.event_id === firstStage);
const upheldDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-model-localnet-upheld-"));
writeLedgerPrefix(upheldDirectory, events.slice(0, firstStageIndex + 1));
const upheldLedger = new ModelLocalnetLedger(upheldDirectory);
const upheldChallenge = upheldLedger.append(
  signLocalnetIntent(demo.identities["nsrl:challenger:audit"], "challenge_opened", {
    subject_type: "stage",
    subject_event_id: firstStage,
    reason: "The stage evidence is invalid in this adversarial branch.",
    evidence_sha256: "c".repeat(64),
  }),
);
upheldLedger.append(
  signLocalnetIntent(demo.identities["nsrl:authority:localnet"], "challenge_resolved", {
    challenge_event_id: upheldChallenge.event.event_id,
    outcome: "upheld",
    evidence_sha256: "d".repeat(64),
  }),
);
expectFailure(
  () =>
    upheldLedger.append(
      signLocalnetIntent(demo.identities["nsrl:authority:localnet"], "stage_accepted", {
        stage_event_id: firstStage,
      }),
    ),
  /challenged or invalid stage evidence cannot be accepted/,
);

const thirdCandidateAttestationIndex = events.findIndex(
  (event) =>
    event.signed_intent.event_type === "validation_attested" &&
    event.signed_intent.payload.subject_event_id === demo.candidate_event_id &&
    event.signed_intent.actor === "nsrl:validator:replay-three",
);
const candidateQuorumDirectory = fs.mkdtempSync(
  path.join(os.tmpdir(), "nsrl-model-localnet-candidate-quorum-"),
);
writeLedgerPrefix(candidateQuorumDirectory, events.slice(0, thirdCandidateAttestationIndex));
const candidateQuorumLedger = new ModelLocalnetLedger(candidateQuorumDirectory);
const candidateQuorumState = candidateQuorumLedger.inspect().state;
expectFailure(
  () =>
    candidateQuorumLedger.append(
      signLocalnetIntent(
        demo.identities["nsrl:authority:localnet"],
        "model_published",
        buildModelPublicationPayload(candidateQuorumState, demo.candidate_event_id),
      ),
    ),
  /has not reached the required clean replay quorum/,
);

expectFailure(
  () =>
    demo.ledger.append(
      signLocalnetIntent(
        demo.identities["nsrl:builder:integer-core"],
        "validation_attested",
        {
          subject_type: "candidate",
          subject_event_id: demo.candidate_event_id,
          verdict: "valid",
          check_mode: "artifact_check",
          evidence_sha256: "a".repeat(64),
        },
      ),
    ),
  /conflicts with launch execution or funding roles/,
);

expectFailure(
  () =>
    demo.ledger.append(
      signLocalnetIntent(
        demo.identities["nsrl:challenger:audit"],
        "challenge_opened",
        {
          subject_type: "stage",
          subject_event_id: firstStage,
          reason: "Attempt to reopen finalized evidence.",
          evidence_sha256: "b".repeat(64),
        },
      ),
    ),
  /finalized evidence cannot be challenged/,
);

process.stdout.write(
  `model localnet v1 passed: ${state.height} signed events, ${summary.accounts} accounts, ` +
    `${summary.accepted_stages} accepted stages, ${summary.publications} publication\n`,
);
