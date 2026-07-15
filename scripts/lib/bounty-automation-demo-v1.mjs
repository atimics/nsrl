import fs from "node:fs";
import path from "node:path";

import {
  bountyAutomationPolicySha256,
} from "./bounty-automation-v1.mjs";
import {
  planBountyKeeperCycle,
  runBountyKeeperCycle,
} from "./bounty-keeper-v1.mjs";
import {
  createDeterministicLocalnetIdentity,
  registerLocalnetIdentity,
  signLocalnetIntent,
} from "./model-localnet-v1.mjs";
import { publicLocalnetSnapshot } from "./model-localnet-demo-v1.mjs";
import { buildDeterministicMarketDemo } from "./model-market-demo-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const POLICY_PATH = path.join(
  ROOT,
  "protocol/examples/integer-transformer-bounty-automation-v1.json",
);

function append(ledger, identity, eventType, payload) {
  return ledger.append(signLocalnetIntent(identity, eventType, payload));
}

export function buildDeterministicBountyAutomationDemo(directory) {
  const market = buildDeterministicMarketDemo(directory);
  const { ledger, identities } = market;
  const authority = identities["nsrl:authority:market"];
  const proposer = identities["nsrl:lab:genesis"];
  const sponsor = identities["nsrl:sponsor:prototype"];
  const keeper = createDeterministicLocalnetIdentity(
    "nsrl:keeper:frontier",
    "nsrl Forge public bounty keeper identity · frontier",
  );
  identities[keeper.account] = keeper;
  registerLocalnetIdentity(ledger, keeper);
  append(ledger, authority, "test_credit_issued", {
    recipient: sponsor.account,
    units: "300000",
  });

  const policy = JSON.parse(fs.readFileSync(POLICY_PATH, "utf8"));
  append(ledger, sponsor, "bounty_automation_policy_registered", {
    policy_sha256: bountyAutomationPolicySha256(policy),
    policy,
  });
  const waitingPlan = planBountyKeeperCycle(
    ledger.inspect().state,
    policy.id,
    "2026-07-14T17:00:00Z",
  );
  append(ledger, authority, "slot_advanced", { slot: 10 });
  const result = runBountyKeeperCycle(
    ledger,
    policy.id,
    { keeper, proposer, sponsor },
    "2026-07-14T17:00:00Z",
  );
  const { events, state } = ledger.inspect();
  const cycle = state.automation_policies[policy.id].cycles["1"];

  return {
    ledger,
    identities,
    policy,
    waiting_plan: waitingPlan,
    result,
    state,
    snapshot: publicLocalnetSnapshot(events, state, {
      generated_at: "2026-07-14T17:00:00Z",
      notice:
        "Deterministic public bounty keeper fixture · promotion-triggered successor · bounded test credit · no financial value",
      launch_event_id: cycle.launch_event_id,
      candidate_event_id: market.candidate_event_id,
      automation_cycle_event_id: cycle.event_id,
    }),
  };
}
