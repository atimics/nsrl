#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  automationCycleSpendUnits,
  bountyAutomationPolicySha256,
  validateBountyAutomationPolicy,
} from "./lib/bounty-automation-v1.mjs";
import {
  planBountyKeeperCycle,
  runBountyKeeperCycle,
} from "./lib/bounty-keeper-v1.mjs";
import {
  ModelLocalnetLedger,
  localnetStateSummary,
  readLocalnetIdentity,
  signLocalnetIntent,
} from "./lib/model-localnet-v1.mjs";

function usage() {
  return `Usage:
  nsrl-bounty-keeper.mjs register --dir DIR --policy FILE --sponsor-key FILE
  nsrl-bounty-keeper.mjs plan --dir DIR --policy-id ID [--published-at ISO_DATE]
  nsrl-bounty-keeper.mjs tick --dir DIR --policy-id ID --keeper-key FILE --proposer-key FILE --sponsor-key FILE [--published-at ISO_DATE]
  nsrl-bounty-keeper.mjs approve --dir DIR --policy-id ID --cycle N --sponsor-key FILE
  nsrl-bounty-keeper.mjs pause --dir DIR --policy-id ID --sponsor-key FILE
  nsrl-bounty-keeper.mjs resume --dir DIR --policy-id ID --sponsor-key FILE
  nsrl-bounty-keeper.mjs status --dir DIR [--policy-id ID]`;
}

function parseArgs(argv) {
  const command = argv[0];
  if (!command || command === "help" || command === "--help") {
    return { command: "help", options: {} };
  }
  const options = {};
  for (let index = 1; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--") || !argv[index + 1]) {
      throw new Error(`unknown or incomplete argument ${arg}`);
    }
    options[arg.slice(2)] = argv[index + 1];
    index += 1;
  }
  return { command, options };
}

function required(options, name) {
  const value = options[name];
  if (!value) {
    throw new Error(`--${name} is required`);
  }
  return value;
}

function positiveInteger(options, name) {
  const value = required(options, name);
  if (!/^[1-9][0-9]*$/.test(value) || !Number.isSafeInteger(Number(value))) {
    throw new Error(`--${name} must be a positive safe integer`);
  }
  return Number(value);
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(path.resolve(filePath), "utf8"));
}

function append(ledger, identity, eventType, payload) {
  const result = ledger.append(signLocalnetIntent(identity, eventType, payload));
  return {
    event_type: result.event.signed_intent.event_type,
    event_id: result.event.event_id,
    height: result.event.height,
    duplicate: result.duplicate,
    ledger_height: result.state.height,
  };
}

function main() {
  const { command, options } = parseArgs(process.argv.slice(2));
  if (command === "help") {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const ledger = new ModelLocalnetLedger(required(options, "dir"));
  let output;

  if (command === "register") {
    const policy = readJson(required(options, "policy"));
    validateBountyAutomationPolicy(policy);
    const sponsor = readLocalnetIdentity(required(options, "sponsor-key"));
    output = append(ledger, sponsor, "bounty_automation_policy_registered", {
      policy_sha256: bountyAutomationPolicySha256(policy),
      policy,
    });
  } else if (command === "plan") {
    output = planBountyKeeperCycle(
      ledger.inspect().state,
      required(options, "policy-id"),
      options["published-at"],
    );
  } else if (command === "tick") {
    output = runBountyKeeperCycle(
      ledger,
      required(options, "policy-id"),
      {
        keeper: readLocalnetIdentity(required(options, "keeper-key")),
        proposer: readLocalnetIdentity(required(options, "proposer-key")),
        sponsor: readLocalnetIdentity(required(options, "sponsor-key")),
      },
      options["published-at"],
    );
  } else if (command === "approve") {
    const policyId = required(options, "policy-id");
    const state = ledger.inspect().state;
    const record = state.automation_policies[policyId];
    if (!record) {
      throw new Error(`unknown bounty automation policy ${policyId}`);
    }
    output = append(
      ledger,
      readLocalnetIdentity(required(options, "sponsor-key")),
      "bounty_automation_cycle_approved",
      {
        policy_id: policyId,
        cycle_index: positiveInteger(options, "cycle"),
        approved_units: automationCycleSpendUnits(record.policy),
      },
    );
  } else if (command === "pause" || command === "resume") {
    output = append(
      ledger,
      readLocalnetIdentity(required(options, "sponsor-key")),
      `bounty_automation_policy_${command === "pause" ? "paused" : "resumed"}`,
      { policy_id: required(options, "policy-id") },
    );
  } else if (command === "status") {
    const summary = localnetStateSummary(ledger.inspect().state);
    output = options["policy-id"]
      ? summary.automation.policies.find(
          (record) => record.policy.id === options["policy-id"],
        ) ?? null
      : summary.automation;
  } else {
    throw new Error(`unknown command ${command}\n${usage()}`);
  }

  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
