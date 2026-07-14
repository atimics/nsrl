#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  ModelLocalnetLedger,
  buildStageAuctionClosePayload,
  signLocalnetIntent,
} from "./lib/model-localnet-v1.mjs";
import { buildDeterministicMarketDemo } from "./lib/model-market-demo-v1.mjs";

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

function forkLedger(events, endIndex, label) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), `nsrl-market-${label}-`));
  writeLedgerPrefix(directory, events.slice(0, endIndex));
  return new ModelLocalnetLedger(directory);
}

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-model-market-check-"));
const demo = buildDeterministicMarketDemo(directory);
const { events, state } = demo.ledger.inspect();
const summary = demo.snapshot.summary;
const market = summary.market;
const launchId = summary.launches[0].id;
const authority = demo.identities["nsrl:authority:market"];
const graviton = demo.identities["nsrl:compute:graviton-pool"];
const copper = demo.identities["nsrl:compute:copper-grid"];
const validator = demo.identities["nsrl:validator:proof-v1"];

assert.equal(state.height, 76);
assert.equal(summary.accounts, 12);
assert.equal(summary.launches[0].status, "promoted");
assert.equal(summary.launches[0].assigned_stages, 3);
assert.equal(summary.launches[0].paid_stages, 3);
assert.equal(market.enabled, true);
assert.equal(market.credit_symbol, "FORGE-TEST");
assert.equal(market.bid_commits, 9);
assert.equal(market.bid_reveals, 9);
assert.equal(market.auctions.length, 3);
assert.equal(market.meter_receipts, 3);
assert.equal(market.compute_escrows[launchId].status, "settled");
assert.equal(market.compute_escrows[launchId].refunded_units, "4832");
assert.equal(market.issued_supply_units, "146000");
assert.equal(market.accounted_supply_units, "146000");
assert.equal(market.conservation_valid, true);
assert.deepEqual(
  market.auctions.map((auction) => [auction.stage_id, auction.provider, auction.payment_units]),
  [
    ["train-candidate", "nsrl:compute:graviton-pool", "6144"],
    ["evaluate-candidate", "nsrl:compute:copper-grid", "768"],
    ["freeze-publication", "nsrl:compute:glacier-lab", "256"],
  ],
);
assert.deepEqual(market.balances, {
  "nsrl:sponsor:prototype": "12832",
  "nsrl:compute:graviton-pool": "8144",
  "nsrl:compute:copper-grid": "2768",
  "nsrl:compute:glacier-lab": "2256",
  "nsrl:builder:integer-core": "120000",
});
assert.deepEqual(summary.model_balances.ITP1, {
  "nsrl:builder:integer-core": "19200",
  "nsrl:compute:graviton-pool": "16800",
  "nsrl:validator:proof-v1": "12800",
  "nsrl:sponsor:prototype": "6400",
  "nsrl:treasury:public-goods": "3200",
  "nsrl:compute:copper-grid": "4200",
  "nsrl:compute:glacier-lab": "1400",
});
assert.equal(
  Object.values(summary.model_balances.ITP1).reduce((sum, value) => sum + BigInt(value), 0n),
  64000n,
);
assert.equal(JSON.stringify(demo.snapshot).includes("private_key_pem"), false);

const firstRevealIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "provider_bid_revealed",
);
const revealLedger = forkLedger(events, firstRevealIndex, "wrong-reveal");
const firstBid = demo.bids[0];
expectFailure(
  () =>
    revealLedger.append(
      signLocalnetIntent(graviton, "provider_bid_revealed", {
        ...firstBid,
        nonce: "wrong-nonce",
      }),
    ),
  /does not match its commitment/,
);
expectFailure(
  () =>
    revealLedger.append(
      signLocalnetIntent(graviton, "provider_bid_committed", {
        launch_id: firstBid.launch_id,
        stage_id: firstBid.stage_id,
        commitment_sha256: "a".repeat(64),
      }),
    ),
  /after the bid deadline/,
);

const firstCloseIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "stage_auction_closed",
);
const closeLedger = forkLedger(events, firstCloseIndex, "wrong-winner");
const expectedClose = buildStageAuctionClosePayload(
  closeLedger.inspect().state,
  launchId,
  "train-candidate",
);
expectFailure(
  () =>
    closeLedger.append(
      signLocalnetIntent(authority, "stage_auction_closed", {
        ...expectedClose,
        provider: copper.account,
      }),
    ),
  /does not match deterministic auction ranking/,
);

const firstStageIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "stage_submitted",
);
const submitLedger = forkLedger(events, firstStageIndex, "nonwinner");
const firstStage = state.launches[launchId].recipe.run.stages[0];
expectFailure(
  () =>
    submitLedger.append(
      signLocalnetIntent(copper, "stage_submitted", {
        launch_id: launchId,
        stage_id: firstStage.id,
        input_sha256: "1".repeat(64),
        output_sha256: "2".repeat(64),
        evidence_sha256: "3".repeat(64),
        compute_units: firstStage.compute_units,
      }),
    ),
  /assigned compute provider/,
);

const firstMeterIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "compute_metered",
);
const meterLedger = forkLedger(events, firstMeterIndex, "bad-meter");
const submittedEvent = events[firstStageIndex];
expectFailure(
  () =>
    meterLedger.append(
      signLocalnetIntent(graviton, "compute_metered", {
        stage_event_id: submittedEvent.event_id,
        start_slot: 4,
        end_slot: 5,
        compute_units: "3071",
        input_sha256: submittedEvent.signed_intent.payload.input_sha256,
        output_sha256: submittedEvent.signed_intent.payload.output_sha256,
        evidence_sha256: submittedEvent.signed_intent.payload.evidence_sha256,
      }),
    ),
  /must equal the submitted stage claim/,
);

const lateMeterLedger = forkLedger(events, firstMeterIndex, "late-meter");
lateMeterLedger.append(signLocalnetIntent(authority, "slot_advanced", { slot: 11 }));
expectFailure(
  () =>
    lateMeterLedger.append(
      signLocalnetIntent(graviton, "compute_metered", {
        stage_event_id: submittedEvent.event_id,
        start_slot: 4,
        end_slot: 5,
        compute_units: submittedEvent.signed_intent.payload.compute_units,
        input_sha256: submittedEvent.signed_intent.payload.input_sha256,
        output_sha256: submittedEvent.signed_intent.payload.output_sha256,
        evidence_sha256: submittedEvent.signed_intent.payload.evidence_sha256,
      }),
    ),
  /after the execution deadline/,
);

const firstAcceptIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "stage_accepted",
);
const paymentLedger = forkLedger(events, firstAcceptIndex, "early-payment");
const meterEvent = events[firstMeterIndex];
expectFailure(
  () =>
    paymentLedger.append(
      signLocalnetIntent(authority, "stage_payment_settled", {
        stage_event_id: submittedEvent.event_id,
        provider: graviton.account,
        payment_units: "6144",
        meter_event_id: meterEvent.event_id,
      }),
    ),
  /requires accepted stage evidence/,
);

const expiryLedger = forkLedger(events, firstStageIndex, "expiry");
expiryLedger.append(signLocalnetIntent(authority, "slot_advanced", { slot: 11 }));
expiryLedger.append(
  signLocalnetIntent(authority, "launch_expired", {
    launch_id: launchId,
    compute_refund_units: "12000",
    bounty_refund_units: "120000",
    slashed_collateral_units: "1500",
  }),
);
const expirySummary = expiryLedger.inspect().state;
assert.equal(expirySummary.launches[launchId].status, "expired");
assert.equal(expirySummary.test_balances["nsrl:sponsor:prototype"], "141500");
assert.equal(expirySummary.compute_escrows[launchId].status, "expired");
assert.equal(expirySummary.bounty_settlements[launchId]["beat-float-reference"].refunded_units, "120000");
assert.equal(
  Object.values(expirySummary.provider_collateral).every(
    (collateral) => collateral.locked_units === "1000" && collateral.reserved_units === "0",
  ),
  true,
);
expectFailure(
  () =>
    expiryLedger.append(
      signLocalnetIntent(graviton, "stage_submitted", {
        launch_id: launchId,
        stage_id: firstStage.id,
        input_sha256: "4".repeat(64),
        output_sha256: "5".repeat(64),
        evidence_sha256: "6".repeat(64),
        compute_units: firstStage.compute_units,
      }),
    ),
  /expired launches cannot accept stage evidence/,
);

const expiredEvidenceLedger = forkLedger(events, firstStageIndex + 1, "expired-evidence");
expiredEvidenceLedger.append(signLocalnetIntent(authority, "slot_advanced", { slot: 11 }));
expiredEvidenceLedger.append(
  signLocalnetIntent(authority, "launch_expired", {
    launch_id: launchId,
    compute_refund_units: "12000",
    bounty_refund_units: "120000",
    slashed_collateral_units: "1500",
  }),
);
expectFailure(
  () =>
    expiredEvidenceLedger.append(
      signLocalnetIntent(validator, "validation_attested", {
        subject_type: "stage",
        subject_event_id: submittedEvent.event_id,
        verdict: "valid",
        check_mode: "artifact_check",
        evidence_sha256: "7".repeat(64),
      }),
    ),
  /expired launches cannot accept validator attestations/,
);
expectFailure(
  () =>
    expiredEvidenceLedger.append(
      signLocalnetIntent(authority, "stage_accepted", {
        stage_event_id: submittedEvent.event_id,
      }),
    ),
  /expired launches cannot accept stage evidence/,
);

process.stdout.write(
  `Forge market v1 passed: ${state.height} signed events, ${market.bid_reveals} revealed bids, ` +
    `${market.auctions.length} paid stage auctions, ${market.compute_escrows[launchId].refunded_units} refunded units\n`,
);
