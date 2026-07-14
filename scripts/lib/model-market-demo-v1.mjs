import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  ModelLocalnetLedger,
  buildComputeRewardDistributionPayload,
  buildModelPublicationPayload,
  buildStageAuctionClosePayload,
  createDeterministicLocalnetIdentity,
  createProviderBidCommitment,
  registerLocalnetIdentity,
  signLocalnetIntent,
} from "./model-localnet-v1.mjs";
import { publicLocalnetSnapshot } from "./model-localnet-demo-v1.mjs";
import { sha256Canonical } from "./model-launch-v1.mjs";

const ROOT = path.resolve(import.meta.dirname, "../..");
const RECIPE_PATH = path.join(
  ROOT,
  "protocol/examples/integer-transformer-proof-v1.launch.json",
);

function sha256(label) {
  return createHash("sha256").update(label).digest("hex");
}

function marketRecipe() {
  const promoted = JSON.parse(fs.readFileSync(RECIPE_PATH, "utf8"));
  const recipe = structuredClone(promoted);
  recipe.launch.id = "integer-transformer-proof-v1-market";
  recipe.launch.title = "Integer Transformer Proof v1 · Forge market";
  recipe.launch.summary =
    "Auction every bounded compute stage, settle accepted meter receipts, replay the candidate, and publish the verified model.";
  recipe.launch.network = "nsrl-forge-market-v1";
  recipe.launch.mode = "test";
  recipe.launch.status = "open";
  recipe.publication = null;
  return { recipe, publication: promoted.publication };
}

function deterministicIdentities() {
  const accounts = [
    "nsrl:authority:market",
    "nsrl:lab:genesis",
    "nsrl:builder:integer-core",
    "nsrl:compute:graviton-pool",
    "nsrl:compute:copper-grid",
    "nsrl:compute:glacier-lab",
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
        `nsrl Forge market public demo identity · ${account}`,
      ),
    ]),
  );
}

function append(ledger, identity, eventType, payload) {
  return ledger.append(signLocalnetIntent(identity, eventType, payload));
}

function bidPlan(recipe) {
  const providers = [
    "nsrl:compute:graviton-pool",
    "nsrl:compute:copper-grid",
    "nsrl:compute:glacier-lab",
  ];
  const prices = {
    "train-candidate": ["2", "3", "4"],
    "evaluate-candidate": ["3", "1", "2"],
    "freeze-publication": ["4", "2", "1"],
  };
  return recipe.run.stages.flatMap((stage) =>
    providers.map((provider, index) => ({
      launch_id: recipe.launch.id,
      stage_id: stage.id,
      provider,
      unit_price_units: prices[stage.id][index],
      max_compute_units: stage.compute_units,
      nonce: `${stage.id}:${provider}:sealed-v1`,
    })),
  );
}

export function buildDeterministicMarketDemo(directory) {
  const ledger = new ModelLocalnetLedger(directory);
  const identities = deterministicIdentities();
  const authority = identities["nsrl:authority:market"];
  const proposer = identities["nsrl:lab:genesis"];
  const builder = identities["nsrl:builder:integer-core"];
  const sponsor = identities["nsrl:sponsor:prototype"];
  const providers = [
    identities["nsrl:compute:graviton-pool"],
    identities["nsrl:compute:copper-grid"],
    identities["nsrl:compute:glacier-lab"],
  ];
  const validatorOne = identities["nsrl:validator:proof-v1"];
  const validatorTwo = identities["nsrl:validator:replay-two"];
  const validatorThree = identities["nsrl:validator:replay-three"];
  const challenger = identities["nsrl:challenger:audit"];

  ledger.initialize(authority, {
    network_id: "nsrl-forge-market-v1",
    stage_quorum: 2,
    candidate_quorum: 3,
    full_replay_required: 1,
    test_credit_enabled: true,
    credit_symbol: "FORGE-TEST",
  });
  for (const identity of Object.values(identities)) {
    if (identity.account !== authority.account) {
      registerLocalnetIdentity(ledger, identity);
    }
  }

  append(ledger, authority, "test_credit_issued", {
    recipient: sponsor.account,
    units: "140000",
  });
  for (const provider of providers) {
    append(ledger, authority, "test_credit_issued", {
      recipient: provider.account,
      units: "2000",
    });
  }

  const { recipe, publication } = marketRecipe();
  const launchResult = append(ledger, proposer, "launch_published", {
    recipe_sha256: sha256Canonical(recipe),
    recipe,
  });
  append(ledger, sponsor, "bounty_funded", {
    launch_id: recipe.launch.id,
    bounty_id: recipe.bounties[0].id,
    escrow_units: recipe.bounties[0].escrow_units,
  });
  append(ledger, sponsor, "compute_budget_funded", {
    launch_id: recipe.launch.id,
    escrow_units: "12000",
    bid_deadline_slot: 1,
    reveal_deadline_slot: 3,
    execution_deadline_slot: 10,
    minimum_collateral_units: "500",
  });
  for (const provider of providers) {
    append(ledger, provider, "provider_collateral_deposited", { units: "1500" });
  }

  const bids = bidPlan(recipe);
  for (const bid of bids) {
    append(ledger, identities[bid.provider], "provider_bid_committed", {
      launch_id: bid.launch_id,
      stage_id: bid.stage_id,
      commitment_sha256: createProviderBidCommitment(bid),
    });
  }
  append(ledger, authority, "slot_advanced", { slot: 2 });
  for (const bid of bids) {
    append(ledger, identities[bid.provider], "provider_bid_revealed", bid);
  }
  append(ledger, authority, "slot_advanced", { slot: 4 });
  for (const stage of recipe.run.stages) {
    append(
      ledger,
      authority,
      "stage_auction_closed",
      buildStageAuctionClosePayload(ledger.inspect().state, recipe.launch.id, stage.id),
    );
  }
  append(ledger, authority, "slot_advanced", { slot: 5 });

  const stageEventIds = [];
  for (const [index, stage] of recipe.run.stages.entries()) {
    const state = ledger.inspect().state;
    const assignment = state.stage_assignments[`${recipe.launch.id}:${stage.id}`];
    const inputSha256 = sha256(`${stage.id}:market-input`);
    const outputSha256 = sha256(`${stage.id}:market-output`);
    const evidenceSha256 = sha256(`${stage.id}:market-evidence`);
    const submitted = append(ledger, identities[assignment.provider], "stage_submitted", {
      launch_id: recipe.launch.id,
      stage_id: stage.id,
      input_sha256: inputSha256,
      output_sha256: outputSha256,
      evidence_sha256: evidenceSha256,
      compute_units: stage.compute_units,
    });
    const stageEventId = submitted.event.event_id;
    stageEventIds.push(stageEventId);
    const meter = append(ledger, identities[assignment.provider], "compute_metered", {
      stage_event_id: stageEventId,
      start_slot: 4,
      end_slot: 5,
      compute_units: stage.compute_units,
      input_sha256: inputSha256,
      output_sha256: outputSha256,
      evidence_sha256: evidenceSha256,
    });
    append(ledger, validatorOne, "validation_attested", {
      subject_type: "stage",
      subject_event_id: stageEventId,
      verdict: "valid",
      check_mode: "artifact_check",
      evidence_sha256: sha256(`${stage.id}:market-validator-one`),
    });
    append(ledger, validatorTwo, "validation_attested", {
      subject_type: "stage",
      subject_event_id: stageEventId,
      verdict: "valid",
      check_mode: index === 0 ? "full_replay" : "artifact_check",
      evidence_sha256: sha256(`${stage.id}:market-validator-two`),
    });
    if (index === 0) {
      const challenge = append(ledger, challenger, "challenge_opened", {
        subject_type: "stage",
        subject_event_id: stageEventId,
        reason: "Verify the winning provider's meter receipt against stage evidence.",
        evidence_sha256: sha256(`${stage.id}:market-challenge`),
      });
      append(ledger, authority, "challenge_resolved", {
        challenge_event_id: challenge.event.event_id,
        outcome: "rejected",
        evidence_sha256: sha256(`${stage.id}:market-challenge-resolution`),
      });
    }
    append(ledger, authority, "stage_accepted", { stage_event_id: stageEventId });
    append(ledger, authority, "stage_payment_settled", {
      stage_event_id: stageEventId,
      provider: assignment.provider,
      payment_units: assignment.reserved_payment_units,
      meter_event_id: meter.event.event_id,
    });
  }

  const escrowBeforeRefund = ledger.inspect().state.compute_escrows[recipe.launch.id];
  append(ledger, authority, "compute_budget_refunded", {
    launch_id: recipe.launch.id,
    sponsor: sponsor.account,
    refund_units: escrowBeforeRefund.balance_units,
  });
  for (const provider of providers) {
    const collateral = ledger.inspect().state.provider_collateral[provider.account];
    append(ledger, provider, "provider_collateral_withdrawn", {
      units: collateral.locked_units,
    });
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
      evidence_sha256: sha256(`market-candidate:validator-${index + 1}`),
    });
  }
  append(
    ledger,
    authority,
    "model_published",
    buildModelPublicationPayload(ledger.inspect().state, candidateEventId),
  );
  append(
    ledger,
    authority,
    "compute_reward_distributed",
    buildComputeRewardDistributionPayload(ledger.inspect().state, recipe.launch.id),
  );

  const { events, state } = ledger.inspect();
  return {
    ledger,
    identities,
    bids,
    stage_event_ids: stageEventIds,
    candidate_event_id: candidateEventId,
    snapshot: publicLocalnetSnapshot(events, state, {
      launch_event_id: launchResult.event.event_id,
      candidate_event_id: candidateEventId,
    }),
    state,
  };
}
