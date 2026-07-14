#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  ModelLocalnetLedger,
  buildModelPublicationPayload,
  createLocalnetIdentity,
  localnetStateSummary,
  readLocalnetIdentity,
  registerLocalnetIdentity,
  signLocalnetIntent,
  writeLocalnetIdentity,
} from "./lib/model-localnet-v1.mjs";
import { sha256Canonical, validateModelLaunchRecipe } from "./lib/model-launch-v1.mjs";

function usage() {
  return `Usage:
  nsrl-model-localnet.mjs init --dir DIR --authority ACCOUNT [--network ID]
  nsrl-model-localnet.mjs account --dir DIR --account ACCOUNT
  nsrl-model-localnet.mjs publish-launch --dir DIR --recipe FILE --key FILE
  nsrl-model-localnet.mjs fund-bounty --dir DIR --launch ID --bounty ID --key FILE
  nsrl-model-localnet.mjs submit-stage --dir DIR --launch ID --stage ID --input-sha HASH --output-sha HASH --evidence-sha HASH --compute-units N --key FILE
  nsrl-model-localnet.mjs attest --dir DIR --subject-type stage|candidate --subject EVENT_ID --verdict valid|invalid --mode artifact_check|full_replay --evidence-sha HASH --key FILE
  nsrl-model-localnet.mjs challenge --dir DIR --subject-type stage|candidate --subject EVENT_ID --reason TEXT --evidence-sha HASH --key FILE
  nsrl-model-localnet.mjs resolve-challenge --dir DIR --challenge EVENT_ID --outcome upheld|rejected --evidence-sha HASH --key FILE
  nsrl-model-localnet.mjs accept-stage --dir DIR --subject EVENT_ID --key AUTHORITY_FILE
  nsrl-model-localnet.mjs submit-candidate --dir DIR --launch ID --candidate FILE --key FILE
  nsrl-model-localnet.mjs publish-model --dir DIR --candidate EVENT_ID --key AUTHORITY_FILE
  nsrl-model-localnet.mjs status --dir DIR`;
}

function parseArgs(argv) {
  const command = argv[0];
  if (!command || command === "--help" || command === "help") {
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

function requireOption(options, name) {
  const value = options[name];
  if (!value) {
    throw new Error(`--${name} is required`);
  }
  return value;
}

function identityFile(directory, account) {
  return path.join(
    path.resolve(directory),
    "identities",
    `${account.replaceAll(":", "-")}.identity.json`,
  );
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(path.resolve(filePath), "utf8"));
}

function sha256File(filePath) {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function eventOutput(result) {
  return {
    schema: "nsrl.model_localnet_cli_event.v1",
    event_type: result.event.signed_intent.event_type,
    event_id: result.event.event_id,
    event_sha256: result.event.event_sha256,
    height: result.event.height,
    duplicate: result.duplicate,
    ledger_height: result.state.height,
    ledger_head_sha256: result.state.head_sha256,
  };
}

function append(ledger, identity, eventType, payload) {
  return eventOutput(ledger.append(signLocalnetIntent(identity, eventType, payload)));
}

function main() {
  const { command, options } = parseArgs(process.argv.slice(2));
  if (command === "help") {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const directory = requireOption(options, "dir");
  const ledger = new ModelLocalnetLedger(directory);
  let output;

  if (command === "init") {
    if (ledger.inspect().events.length !== 0) {
      throw new Error("localnet ledger is already initialized; refusing to replace its authority key");
    }
    const account = requireOption(options, "authority");
    const identity = createLocalnetIdentity(account);
    const keyPath = options["key-out"]
      ? path.resolve(options["key-out"])
      : identityFile(directory, account);
    writeLocalnetIdentity(keyPath, identity);
    const result = ledger.initialize(identity, {
      network_id: options.network ?? "nsrl-model-localnet-v1",
      stage_quorum: options["stage-quorum"]
        ? Number.parseInt(options["stage-quorum"], 10)
        : 2,
      candidate_quorum: options["candidate-quorum"]
        ? Number.parseInt(options["candidate-quorum"], 10)
        : 3,
      full_replay_required: options["full-replay-required"]
        ? Number.parseInt(options["full-replay-required"], 10)
        : 1,
    });
    output = { ...eventOutput(result), authority: account, identity_file: keyPath };
  } else if (command === "account") {
    const account = requireOption(options, "account");
    if (ledger.inspect().state.accounts[account]) {
      throw new Error(`account ${account} is already registered; refusing to replace its key`);
    }
    const identity = createLocalnetIdentity(account);
    const keyPath = options["key-out"]
      ? path.resolve(options["key-out"])
      : identityFile(directory, account);
    writeLocalnetIdentity(keyPath, identity);
    output = {
      ...eventOutput(registerLocalnetIdentity(ledger, identity)),
      account,
      identity_file: keyPath,
    };
  } else if (command === "status") {
    output = localnetStateSummary(ledger.inspect().state);
  } else {
    const identity = readLocalnetIdentity(requireOption(options, "key"));
    if (command === "publish-launch") {
      const recipe = readJson(requireOption(options, "recipe"));
      validateModelLaunchRecipe(recipe);
      output = append(ledger, identity, "launch_published", {
        recipe_sha256: sha256Canonical(recipe),
        recipe,
      });
    } else if (command === "fund-bounty") {
      const state = ledger.inspect().state;
      const launchId = requireOption(options, "launch");
      const bountyId = requireOption(options, "bounty");
      const bounty = state.launches[launchId]?.recipe.bounties.find((row) => row.id === bountyId);
      if (!bounty) {
        throw new Error(`unknown bounty ${bountyId} for launch ${launchId}`);
      }
      output = append(ledger, identity, "bounty_funded", {
        launch_id: launchId,
        bounty_id: bountyId,
        escrow_units: bounty.escrow_units,
      });
    } else if (command === "submit-stage") {
      output = append(ledger, identity, "stage_submitted", {
        launch_id: requireOption(options, "launch"),
        stage_id: requireOption(options, "stage"),
        input_sha256: requireOption(options, "input-sha"),
        output_sha256: requireOption(options, "output-sha"),
        evidence_sha256: requireOption(options, "evidence-sha"),
        compute_units: requireOption(options, "compute-units"),
      });
    } else if (command === "attest") {
      output = append(ledger, identity, "validation_attested", {
        subject_type: requireOption(options, "subject-type"),
        subject_event_id: requireOption(options, "subject"),
        verdict: requireOption(options, "verdict"),
        check_mode: requireOption(options, "mode"),
        evidence_sha256: requireOption(options, "evidence-sha"),
      });
    } else if (command === "challenge") {
      output = append(ledger, identity, "challenge_opened", {
        subject_type: requireOption(options, "subject-type"),
        subject_event_id: requireOption(options, "subject"),
        reason: requireOption(options, "reason"),
        evidence_sha256: requireOption(options, "evidence-sha"),
      });
    } else if (command === "resolve-challenge") {
      output = append(ledger, identity, "challenge_resolved", {
        challenge_event_id: requireOption(options, "challenge"),
        outcome: requireOption(options, "outcome"),
        evidence_sha256: requireOption(options, "evidence-sha"),
      });
    } else if (command === "accept-stage") {
      output = append(ledger, identity, "stage_accepted", {
        stage_event_id: requireOption(options, "subject"),
      });
    } else if (command === "submit-candidate") {
      const candidate = readJson(requireOption(options, "candidate"));
      const artifactPath = path.resolve(candidate.artifact_path);
      const proofPath = path.resolve(candidate.proof_path);
      const artifactSha256 = sha256File(artifactPath);
      const proofSha256 = sha256File(proofPath);
      if (candidate.artifact_sha256 && candidate.artifact_sha256 !== artifactSha256) {
        throw new Error("candidate artifact SHA-256 does not match its file");
      }
      if (candidate.proof_sha256 && candidate.proof_sha256 !== proofSha256) {
        throw new Error("candidate proof SHA-256 does not match its file");
      }
      output = append(ledger, identity, "candidate_submitted", {
        launch_id: requireOption(options, "launch"),
        model_hash: candidate.model_hash,
        artifact_sha256: artifactSha256,
        artifact_path: candidate.artifact_path,
        proof_sha256: proofSha256,
        metrics: candidate.metrics,
      });
    } else if (command === "publish-model") {
      const candidateEventId = requireOption(options, "candidate");
      const payload = buildModelPublicationPayload(ledger.inspect().state, candidateEventId);
      output = append(ledger, identity, "model_published", payload);
    } else {
      throw new Error(`unknown command ${command}\n${usage()}`);
    }
  }

  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
