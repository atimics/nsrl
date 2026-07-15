#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import {
  automationCycleSpendUnits,
  automationTargetValue,
  bountyAutomationPolicySha256,
  buildAutomationCycleOpenPayload,
} from "./lib/bounty-automation-v1.mjs";
import { buildDeterministicBountyAutomationDemo } from "./lib/bounty-automation-demo-v1.mjs";
import {
  planBountyKeeperCycle,
  runBountyKeeperCycle,
} from "./lib/bounty-keeper-v1.mjs";
import {
  localnetStateSummary,
  ModelLocalnetLedger,
  signLocalnetIntent,
} from "./lib/model-localnet-v1.mjs";
import { sha256Canonical } from "./lib/model-launch-v1.mjs";

function expectFailure(operation, pattern) {
  assert.throws(operation, pattern);
}

function forkLedger(events, endIndex, label) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), `nsrl-auto-${label}-`));
  fs.writeFileSync(
    path.join(directory, "ledger.jsonl"),
    `${events.slice(0, endIndex).map((event) => JSON.stringify(event)).join("\n")}\n`,
  );
  return new ModelLocalnetLedger(directory);
}

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-bounty-automation-check-"));
const demo = buildDeterministicBountyAutomationDemo(directory);
const { events, state } = demo.ledger.inspect();
const summary = demo.snapshot.summary;
const policy = demo.policy;
const policySummary = summary.automation.policies[0];
const cycle = policySummary.cycles[0];
const authority = demo.identities["nsrl:authority:market"];
const sponsor = demo.identities[policy.sponsor];
const proposer = demo.identities[policy.proposer];
const keeper = demo.identities[policy.keeper];
const policySchema = JSON.parse(
  fs.readFileSync(
    path.join(import.meta.dirname, "../protocol/bounty-automation-policy-v1.schema.json"),
    "utf8",
  ),
);

assert.equal(policySchema.properties.schema.const, policy.schema);
assert.deepEqual(policySchema.properties.trigger.properties.event, {
  const: policy.trigger.event,
});
assert.equal(automationTargetValue("101", "minimize", 1000), "90");
assert.equal(automationTargetValue("101", "maximize", 1000), "112");
assert.equal(state.height, 84);
assert.equal(summary.accounts, 13);
assert.equal(summary.launches.length, 2);
assert.equal(summary.market.issued_supply_units, "446000");
assert.equal(summary.market.accounted_supply_units, "446000");
assert.equal(summary.market.conservation_valid, true);
assert.equal(demo.waiting_plan.reason, "cooldown");
assert.equal(demo.result.status, "opened");
assert.equal(demo.result.metric, "probability_error_q15");
assert.equal(demo.result.baseline, "260536589");
assert.equal(demo.result.target, "234482930");
assert.equal(demo.result.cycle_spend_units, "132000");
assert.equal(policySummary.status, "active");
assert.equal(policySummary.spent_units, "132000");
assert.equal(policySummary.remaining_units, "264000");
assert.equal(cycle.status, "funded");
assert.equal(cycle.committed_units, "132000");
assert.equal(state.compute_escrows[cycle.launch_id].bid_deadline_slot, 12);
assert.equal(state.compute_escrows[cycle.launch_id].reveal_deadline_slot, 14);
assert.equal(state.compute_escrows[cycle.launch_id].execution_deadline_slot, 20);
assert.equal(state.test_balances[policy.sponsor], "180832");
assert.equal(
  planBountyKeeperCycle(state, policy.id, "2026-07-14T18:00:00Z").reason,
  "active_limit_reached",
);
assert.equal(JSON.stringify(demo.snapshot).includes("private_key_pem"), false);

const policyIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "bounty_automation_policy_registered",
);
const policyPayload = events[policyIndex].signed_intent.payload;
const wrongSponsorLedger = forkLedger(events, policyIndex, "wrong-sponsor");
expectFailure(
  () =>
    wrongSponsorLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_policy_registered", policyPayload),
    ),
  /only the declared sponsor/,
);
expectFailure(
  () =>
    wrongSponsorLedger.append(
      signLocalnetIntent(sponsor, "bounty_automation_policy_registered", {
        ...policyPayload,
        policy_sha256: "a".repeat(64),
      }),
    ),
  /SHA-256 does not match/,
);

const slotIndex = events.findIndex(
  (event) =>
    event.signed_intent.event_type === "slot_advanced" &&
    event.signed_intent.payload.slot === 10,
);
const cooldownLedger = forkLedger(events, slotIndex, "cooldown");
const cooldownPayload = buildAutomationCycleOpenPayload(
  cooldownLedger.inspect().state,
  policy,
  1,
  policy.source_launch_id,
  "2026-07-14T17:00:00Z",
);
expectFailure(
  () =>
    cooldownLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_cycle_opened", cooldownPayload),
    ),
  /cooldown has not elapsed/,
);

const cycleIndex = events.findIndex(
  (event) => event.signed_intent.event_type === "bounty_automation_cycle_opened",
);
const openLedger = forkLedger(events, cycleIndex, "bad-open");
const correctOpen = events[cycleIndex].signed_intent.payload;
expectFailure(
  () =>
    openLedger.append(
      signLocalnetIntent(proposer, "bounty_automation_cycle_opened", correctOpen),
    ),
  /declared keeper/,
);
expectFailure(
  () =>
    openLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_cycle_opened", {
        ...correctOpen,
        recipe_sha256: "b".repeat(64),
      }),
    ),
  /deterministic policy plan/,
);

const pausedLedger = forkLedger(events, slotIndex, "paused");
pausedLedger.append(
  signLocalnetIntent(sponsor, "bounty_automation_policy_paused", {
    policy_id: policy.id,
  }),
);
pausedLedger.append(signLocalnetIntent(authority, "slot_advanced", { slot: 10 }));
expectFailure(
  () =>
    pausedLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_cycle_opened", correctOpen),
    ),
  /policy is paused/,
);
pausedLedger.append(
  signLocalnetIntent(sponsor, "bounty_automation_policy_resumed", {
    policy_id: policy.id,
  }),
);

const manualPolicy = structuredClone(policy);
manualPolicy.id = "integer-transformer-frontier-manual";
manualPolicy.budgets.manual_approval_above_units = "100000";
const manualLedger = forkLedger(events, policyIndex, "manual");
manualLedger.append(
  signLocalnetIntent(sponsor, "bounty_automation_policy_registered", {
    policy_sha256: bountyAutomationPolicySha256(manualPolicy),
    policy: manualPolicy,
  }),
);
manualLedger.append(signLocalnetIntent(authority, "slot_advanced", { slot: 10 }));
const manualOpen = buildAutomationCycleOpenPayload(
  manualLedger.inspect().state,
  manualPolicy,
  1,
  manualPolicy.source_launch_id,
  "2026-07-14T17:00:00Z",
);
expectFailure(
  () =>
    manualLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_cycle_opened", manualOpen),
    ),
  /requires explicit sponsor approval/,
);
expectFailure(
  () =>
    manualLedger.append(
      signLocalnetIntent(keeper, "bounty_automation_cycle_approved", {
        policy_id: manualPolicy.id,
        cycle_index: 1,
        approved_units: automationCycleSpendUnits(manualPolicy),
      }),
    ),
  /only the automation sponsor/,
);
manualLedger.append(
  signLocalnetIntent(sponsor, "bounty_automation_cycle_approved", {
    policy_id: manualPolicy.id,
    cycle_index: 1,
    approved_units: automationCycleSpendUnits(manualPolicy),
  }),
);
manualLedger.append(signLocalnetIntent(keeper, "bounty_automation_cycle_opened", manualOpen));

const launchIndex = events.findIndex(
  (event) =>
    event.signed_intent.event_type === "launch_published" &&
    event.signed_intent.payload.automation_cycle_event_id,
);
const recipeLedger = forkLedger(events, launchIndex, "tampered-recipe");
const launchPayload = structuredClone(events[launchIndex].signed_intent.payload);
launchPayload.recipe.bounties[0].target = "1";
launchPayload.recipe_sha256 = sha256Canonical(launchPayload.recipe);
expectFailure(
  () =>
    recipeLedger.append(signLocalnetIntent(proposer, "launch_published", launchPayload)),
  /does not match its keeper-signed cycle commitment/,
);

const bountyIndex = events.findIndex(
  (event, index) =>
    index > launchIndex && event.signed_intent.event_type === "bounty_funded",
);
const computeIndex = events.findIndex(
  (event, index) =>
    index > launchIndex && event.signed_intent.event_type === "compute_budget_funded",
);

for (const [cut, label, appendedEvents] of [
  [cycleIndex + 1, "resume-after-open", 3],
  [launchIndex + 1, "resume-after-publish", 2],
  [bountyIndex + 1, "resume-after-bounty", 1],
]) {
  const resumeLedger = forkLedger(events, cut, label);
  const reservedSummary = localnetStateSummary(resumeLedger.inspect().state);
  assert.equal(reservedSummary.market.conservation_valid, true);
  const resumed = runBountyKeeperCycle(
    resumeLedger,
    policy.id,
    { keeper, proposer, sponsor },
    "2099-01-01T00:00:00Z",
  );
  const resumedState = resumeLedger.inspect().state;
  const resumedCycle = resumedState.automation_policies[policy.id].cycles["1"];
  assert.equal(resumed.status, "resumed");
  assert.equal(resumed.events.length, appendedEvents);
  assert.equal(resumedCycle.status, "funded");
  assert.equal(resumedCycle.reserve_balance_units, "0");
  assert.equal(resumedState.test_balances[policy.sponsor], "180832");
  assert.equal(localnetStateSummary(resumedState).market.conservation_valid, true);
}

const auctionLedger = forkLedger(events, computeIndex, "tampered-auction");
const computePayload = structuredClone(events[computeIndex].signed_intent.payload);
computePayload.bid_deadline_slot += 1;
expectFailure(
  () =>
    auctionLedger.append(
      signLocalnetIntent(sponsor, "compute_budget_funded", computePayload),
    ),
  /keeper-bound auction terms/,
);

process.stdout.write(
  `bounty automation v1 passed: ${state.height} signed events, ${policySummary.cycles.length} funded cycle, ` +
    `${policySummary.spent_units} committed units, target ${demo.result.target}\n`,
);
