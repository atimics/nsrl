import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  ModelLocalnetLedger,
  buildModelPublicationPayload,
  createDeterministicLocalnetIdentity,
  localnetStateSummary,
  registerLocalnetIdentity,
  signLocalnetIntent,
} from "./model-localnet-v1.mjs";
import { sha256Canonical } from "./model-launch-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const RECIPE_PATH = path.join(
  ROOT,
  "protocol/examples/integer-transformer-proof-v1.launch.json",
);

function sha256(label) {
  return createHash("sha256").update(label).digest("hex");
}

function readPromotedRecipe() {
  return JSON.parse(fs.readFileSync(RECIPE_PATH, "utf8"));
}

function localnetRecipe() {
  const promoted = readPromotedRecipe();
  const recipe = structuredClone(promoted);
  recipe.launch.id = "integer-transformer-proof-v1-localnet";
  recipe.launch.title = "Integer Transformer Proof v1 · signed localnet";
  recipe.launch.summary =
    "Re-run the frozen integer-transformer model launch through signed funding, compute, validation, challenge, and publication events.";
  recipe.launch.network = "nsrl-model-localnet-v1";
  recipe.launch.mode = "test";
  recipe.launch.status = "open";
  recipe.publication = null;
  return { recipe, publication: promoted.publication };
}

function deterministicIdentities() {
  const accounts = [
    "nsrl:authority:localnet",
    "nsrl:lab:genesis",
    "nsrl:builder:integer-core",
    "nsrl:compute:graviton-pool",
    "nsrl:sponsor:prototype",
    "nsrl:treasury:public-goods",
    "nsrl:validator:proof-v1",
    "nsrl:validator:replay-two",
    "nsrl:validator:replay-three",
    "nsrl:challenger:audit",
  ];
  return Object.fromEntries(
    accounts.map((account) => [
      account,
      createDeterministicLocalnetIdentity(
        account,
        `nsrl model localnet public demo identity · ${account}`,
      ),
    ]),
  );
}

function append(ledger, identity, eventType, payload) {
  return ledger.append(signLocalnetIntent(identity, eventType, payload));
}

export function publicLocalnetSnapshot(events, state, metadata = {}) {
  return {
    schema: "nsrl.model_localnet_snapshot.v1",
    generated_at: "2026-07-13T00:00:00Z",
    notice:
      "Deterministic public test fixture · Ed25519 signatures · simulated credits · no wallet, custody, or financial value",
    ...metadata,
    summary: localnetStateSummary(state),
    public_identities: Object.entries(state.accounts).map(([account, value]) => ({
      account,
      public_key: value.public_key,
      authority: value.authority,
    })),
    events,
  };
}

export function buildDeterministicLocalnetDemo(directory) {
  const ledger = new ModelLocalnetLedger(directory);
  const identities = deterministicIdentities();
  const authority = identities["nsrl:authority:localnet"];
  const proposer = identities["nsrl:lab:genesis"];
  const builder = identities["nsrl:builder:integer-core"];
  const compute = identities["nsrl:compute:graviton-pool"];
  const sponsor = identities["nsrl:sponsor:prototype"];
  const validatorOne = identities["nsrl:validator:proof-v1"];
  const validatorTwo = identities["nsrl:validator:replay-two"];
  const validatorThree = identities["nsrl:validator:replay-three"];
  const challenger = identities["nsrl:challenger:audit"];

  ledger.initialize(authority, {
    network_id: "nsrl-model-localnet-v1",
    stage_quorum: 2,
    candidate_quorum: 3,
    full_replay_required: 1,
  });
  for (const identity of Object.values(identities)) {
    if (identity.account !== authority.account) {
      registerLocalnetIdentity(ledger, identity);
    }
  }

  const { recipe, publication } = localnetRecipe();
  const launchIntent = signLocalnetIntent(proposer, "launch_published", {
    recipe_sha256: sha256Canonical(recipe),
    recipe,
  });
  const launchResult = ledger.append(launchIntent);
  const duplicateResult = ledger.append(launchIntent);
  append(ledger, sponsor, "bounty_funded", {
    launch_id: recipe.launch.id,
    bounty_id: recipe.bounties[0].id,
    escrow_units: recipe.bounties[0].escrow_units,
  });

  const stageEventIds = [];
  for (const [index, stage] of recipe.run.stages.entries()) {
    const stageResult = append(ledger, compute, "stage_submitted", {
      launch_id: recipe.launch.id,
      stage_id: stage.id,
      input_sha256: sha256(`${stage.id}:input`),
      output_sha256: sha256(`${stage.id}:output`),
      evidence_sha256: sha256(`${stage.id}:compute-receipt`),
      compute_units: stage.compute_units,
    });
    const stageEventId = stageResult.event.event_id;
    stageEventIds.push(stageEventId);
    append(ledger, validatorOne, "validation_attested", {
      subject_type: "stage",
      subject_event_id: stageEventId,
      verdict: "valid",
      check_mode: "artifact_check",
      evidence_sha256: sha256(`${stage.id}:validator-one`),
    });
    append(ledger, validatorTwo, "validation_attested", {
      subject_type: "stage",
      subject_event_id: stageEventId,
      verdict: "valid",
      check_mode: index === 0 ? "full_replay" : "artifact_check",
      evidence_sha256: sha256(`${stage.id}:validator-two`),
    });
    if (index === 0) {
      const challenge = append(ledger, challenger, "challenge_opened", {
        subject_type: "stage",
        subject_event_id: stageEventId,
        reason: "Confirm the submitted compute ceiling against the frozen recipe.",
        evidence_sha256: sha256(`${stage.id}:challenge`),
      });
      append(ledger, authority, "challenge_resolved", {
        challenge_event_id: challenge.event.event_id,
        outcome: "rejected",
        evidence_sha256: sha256(`${stage.id}:challenge-resolution`),
      });
    }
    append(ledger, authority, "stage_accepted", { stage_event_id: stageEventId });
  }

  const candidate = append(ledger, builder, "candidate_submitted", {
    launch_id: recipe.launch.id,
    ...publication,
  });
  const candidateEventId = candidate.event.event_id;
  for (const [index, validator] of [validatorOne, validatorTwo, validatorThree].entries()) {
    append(ledger, validator, "validation_attested", {
      subject_type: "candidate",
      subject_event_id: candidateEventId,
      verdict: "valid",
      check_mode: index === 0 ? "full_replay" : "artifact_check",
      evidence_sha256: sha256(`candidate:validator-${index + 1}`),
    });
  }
  append(
    ledger,
    authority,
    "model_published",
    buildModelPublicationPayload(ledger.inspect().state, candidateEventId),
  );

  const { events, state } = ledger.inspect();
  return {
    ledger,
    identities,
    stage_event_ids: stageEventIds,
    candidate_event_id: candidateEventId,
    duplicate_result: duplicateResult,
    snapshot: publicLocalnetSnapshot(events, state, {
      launch_event_id: launchResult.event.event_id,
      candidate_event_id: candidateEventId,
    }),
    state,
  };
}
