import {
  createHash,
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign,
  verify,
} from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  automationAuctionDeadlines,
  automationCycleSpendUnits,
  buildAutomationCycleOpenPayload,
  validateAutomatedBountyRecipe,
  validateBountyAutomationPolicy,
} from "./bounty-automation-v1.mjs";
import {
  bountyPayoutUnits,
  canonicalJson,
  createModelPublishReceipt,
  sha256Canonical,
  validateModelLaunchRecipe,
  validateModelPublishReceipt,
} from "./model-launch-v1.mjs";

export const LOCALNET_INTENT_SCHEMA = "nsrl.model_localnet_intent.v1";
export const LOCALNET_EVENT_SCHEMA = "nsrl.model_localnet_event.v1";
export const LOCALNET_IDENTITY_SCHEMA = "nsrl.model_localnet_identity.v1";
export const LOCALNET_STATE_SCHEMA = "nsrl.model_localnet_state.v1";

export const LOCALNET_EVENT_TYPES = [
  "network_initialized",
  "account_registered",
  "test_credit_issued",
  "bounty_automation_policy_registered",
  "bounty_automation_policy_paused",
  "bounty_automation_policy_resumed",
  "bounty_automation_cycle_approved",
  "bounty_automation_cycle_opened",
  "launch_published",
  "bounty_funded",
  "compute_budget_funded",
  "provider_collateral_deposited",
  "provider_bid_committed",
  "slot_advanced",
  "provider_bid_revealed",
  "stage_auction_closed",
  "stage_submitted",
  "compute_metered",
  "validation_attested",
  "challenge_opened",
  "challenge_resolved",
  "stage_accepted",
  "stage_payment_settled",
  "compute_budget_refunded",
  "provider_collateral_withdrawn",
  "candidate_submitted",
  "model_published",
  "compute_reward_distributed",
  "launch_expired",
];

const ZERO_HASH = "0".repeat(64);
const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const MODEL_HASH_PATTERN = /^0x[a-f0-9]{16}$/;
const ACCOUNT_PATTERN = /^nsrl:[a-z0-9][a-z0-9:_-]{2,127}$/;
const NETWORK_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const UINT_PATTERN = /^(0|[1-9][0-9]*)$/;
const PRIVATE_KEY_PREFIX = Buffer.from("302e020100300506032b657004220420", "hex");

function fail(message) {
  throw new Error(`model localnet v1: ${message}`);
}

function asObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(`${label} must be an object`);
  }
  return value;
}

function asString(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    fail(`${label} must be a non-empty string`);
  }
  return value;
}

function asAccount(value, label) {
  const account = asString(value, label);
  if (!ACCOUNT_PATTERN.test(account)) {
    fail(`${label} must use the nsrl: account namespace`);
  }
  return account;
}

function asHash(value, label) {
  const digest = asString(value, label);
  if (!SHA256_PATTERN.test(digest)) {
    fail(`${label} must be a lowercase SHA-256 digest without a prefix`);
  }
  return digest;
}

function asUintString(value, label, { positive = false } = {}) {
  if (typeof value !== "string" || !UINT_PATTERN.test(value)) {
    fail(`${label} must be an unsigned base-10 integer string`);
  }
  if (positive && value === "0") {
    fail(`${label} must be positive`);
  }
  return value;
}

function asPositiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail(`${label} must be a positive integer`);
  }
  return value;
}

function asNonnegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    fail(`${label} must be a nonnegative integer`);
  }
  return value;
}

function clone(value) {
  return structuredClone(value);
}

function publicKeyFromPrivate(privateKeyPem) {
  return createPublicKey(createPrivateKey(privateKeyPem))
    .export({ format: "der", type: "spki" })
    .toString("base64");
}

function privatePem(privateKey) {
  return privateKey.export({ format: "pem", type: "pkcs8" }).toString();
}

export function createLocalnetIdentity(account) {
  asAccount(account, "identity account");
  const { privateKey } = generateKeyPairSync("ed25519");
  const privateKeyPem = privatePem(privateKey);
  return {
    schema: LOCALNET_IDENTITY_SCHEMA,
    account,
    public_key: publicKeyFromPrivate(privateKeyPem),
    private_key_pem: privateKeyPem,
  };
}

export function createDeterministicLocalnetIdentity(account, seedLabel) {
  asAccount(account, "identity account");
  const seed = createHash("sha256").update(asString(seedLabel, "identity seed label")).digest();
  const privateKey = createPrivateKey({
    key: Buffer.concat([PRIVATE_KEY_PREFIX, seed]),
    format: "der",
    type: "pkcs8",
  });
  const privateKeyPem = privatePem(privateKey);
  return {
    schema: LOCALNET_IDENTITY_SCHEMA,
    account,
    public_key: publicKeyFromPrivate(privateKeyPem),
    private_key_pem: privateKeyPem,
  };
}

export function writeLocalnetIdentity(filePath, identity) {
  validateIdentity(identity);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(identity, null, 2)}\n`, { mode: 0o600 });
  fs.chmodSync(filePath, 0o600);
}

export function readLocalnetIdentity(filePath) {
  const identity = JSON.parse(fs.readFileSync(filePath, "utf8"));
  validateIdentity(identity);
  return identity;
}

function validateIdentity(identity) {
  asObject(identity, "identity");
  if (identity.schema !== LOCALNET_IDENTITY_SCHEMA) {
    fail(`identity schema must be ${LOCALNET_IDENTITY_SCHEMA}`);
  }
  asAccount(identity.account, "identity.account");
  asString(identity.private_key_pem, "identity.private_key_pem");
  asString(identity.public_key, "identity.public_key");
  if (publicKeyFromPrivate(identity.private_key_pem) !== identity.public_key) {
    fail("identity public key does not match its private key");
  }
}

function signedBody(intent) {
  return {
    schema: intent.schema,
    event_type: intent.event_type,
    actor: intent.actor,
    public_key: intent.public_key,
    payload: intent.payload,
  };
}

export function signLocalnetIntent(identity, eventType, payload) {
  validateIdentity(identity);
  if (!LOCALNET_EVENT_TYPES.includes(eventType)) {
    fail(`unknown event type ${eventType}`);
  }
  asObject(payload, "event payload");
  const body = {
    schema: LOCALNET_INTENT_SCHEMA,
    event_type: eventType,
    actor: identity.account,
    public_key: identity.public_key,
    payload: clone(payload),
  };
  const signature = sign(
    null,
    Buffer.from(canonicalJson(body)),
    createPrivateKey(identity.private_key_pem),
  ).toString("base64");
  return { ...body, signature };
}

export function verifyLocalnetIntent(intent) {
  asObject(intent, "signed intent");
  if (intent.schema !== LOCALNET_INTENT_SCHEMA) {
    fail(`signed intent schema must be ${LOCALNET_INTENT_SCHEMA}`);
  }
  if (!LOCALNET_EVENT_TYPES.includes(intent.event_type)) {
    fail(`unknown event type ${intent.event_type}`);
  }
  asAccount(intent.actor, "signed intent actor");
  asString(intent.public_key, "signed intent public_key");
  asObject(intent.payload, "signed intent payload");
  asString(intent.signature, "signed intent signature");
  let verified = false;
  try {
    verified = verify(
      null,
      Buffer.from(canonicalJson(signedBody(intent))),
      createPublicKey({
        key: Buffer.from(intent.public_key, "base64"),
        format: "der",
        type: "spki",
      }),
      Buffer.from(intent.signature, "base64"),
    );
  } catch {
    verified = false;
  }
  if (!verified) {
    fail("signed intent signature is invalid");
  }
  return true;
}

function emptyState() {
  return {
    schema: LOCALNET_STATE_SCHEMA,
    network: null,
    accounts: {},
    launches: {},
    stage_submissions: {},
    attestations: {},
    challenges: {},
    invalid_subjects: {},
    accepted_stages: {},
    candidates: {},
    publications: {},
    bounty_settlements: {},
    model_balances: {},
    compute_reward_distributions: {},
    automation_policies: {},
    automation_approvals: {},
    test_credit_supply_units: "0",
    test_balances: {},
    compute_escrows: {},
    provider_collateral: {},
    bid_commits: {},
    bid_reveals: {},
    stage_assignments: {},
    meter_receipts: {},
    stage_payments: {},
    expired_launches: {},
    slot: 0,
    event_ids: {},
    height: 0,
    head_sha256: ZERO_HASH,
  };
}

function stageKey(launchId, stageId) {
  return `${launchId}:${stageId}`;
}

function requireStage(launch, stageId) {
  const stage = launch.recipe.run.stages.find(
    (row) => row.id === asString(stageId, "stage id"),
  );
  if (!stage) {
    fail(`unknown stage ${stageId}`);
  }
  return stage;
}

function balanceUnits(state, account) {
  return BigInt(state.test_balances[account] ?? "0");
}

function addBalance(state, account, units) {
  state.test_balances[account] = (balanceUnits(state, account) + BigInt(units)).toString();
}

function debitBalance(state, account, units, label) {
  const amount = BigInt(units);
  const balance = balanceUnits(state, account);
  if (balance < amount) {
    fail(`${label} exceeds ${account}'s available test-credit balance`);
  }
  state.test_balances[account] = (balance - amount).toString();
}

function accountedTestCreditUnits(state) {
  const balances = Object.values(state.test_balances).reduce(
    (sum, units) => sum + BigInt(units),
    0n,
  );
  const computeEscrows = Object.values(state.compute_escrows).reduce(
    (sum, escrow) => sum + BigInt(escrow.balance_units),
    0n,
  );
  const bountyEscrows = Object.values(state.launches).reduce(
    (sum, launch) =>
      sum +
      Object.values(launch.funded_bounties).reduce(
        (launchSum, bounty) => launchSum + BigInt(bounty.escrow_balance_units ?? "0"),
        0n,
      ),
    0n,
  );
  const collateral = Object.values(state.provider_collateral).reduce(
    (sum, row) => sum + BigInt(row.locked_units),
    0n,
  );
  const automationReserves = Object.values(state.automation_policies).reduce(
    (sum, policy) =>
      sum +
      Object.values(policy.cycles).reduce(
        (cycleSum, cycle) => cycleSum + BigInt(cycle.reserve_balance_units ?? "0"),
        0n,
      ),
    0n,
  );
  return balances + computeEscrows + bountyEscrows + collateral + automationReserves;
}

function assertTestCreditConservation(state) {
  if (
    state.network?.test_credit_enabled === true &&
    accountedTestCreditUnits(state) !== BigInt(state.test_credit_supply_units)
  ) {
    fail("test-credit supply is not conserved across balances, escrows, and collateral");
  }
}

function requireTestCredits(state) {
  const network = requireNetwork(state);
  if (!network.test_credit_enabled) {
    fail("test-credit settlement is not enabled for this network");
  }
  return network;
}

function requireComputeEscrow(state, launchId) {
  const escrow = state.compute_escrows[launchId];
  if (!escrow) {
    fail(`launch ${launchId} has no funded compute escrow`);
  }
  if (escrow.status !== "open") {
    fail(`launch ${launchId} compute escrow is ${escrow.status}`);
  }
  return escrow;
}

function requireAutomationPolicy(state, policyId) {
  const id = asString(policyId, "automation policy id");
  const record = state.automation_policies[id];
  if (!record) {
    fail(`unknown bounty automation policy ${id}`);
  }
  return record;
}

function automationCycleForLaunch(state, launch) {
  if (!launch.automation) {
    return null;
  }
  const policy = requireAutomationPolicy(state, launch.automation.policy_id);
  const cycle = policy.cycles[String(launch.automation.cycle_index)];
  if (!cycle || cycle.launch_id !== launch.id) {
    fail(`automation state is inconsistent for launch ${launch.id}`);
  }
  return { policy, cycle };
}

function consumeAutomationReserve(state, launch, units, kind, eventId) {
  const automation = automationCycleForLaunch(state, launch);
  if (!automation) {
    return false;
  }
  const { policy, cycle } = automation;
  const expected =
    kind === "bounty"
      ? policy.policy.budgets.bounty_units
      : policy.policy.budgets.compute_budget_units;
  if (units !== expected) {
    fail(`automated ${kind} funding must equal its signed policy amount`);
  }
  if (cycle[`${kind}_funding_event_id`]) {
    fail(`automation cycle ${cycle.cycle_index} already recorded ${kind} funding`);
  }
  const reserve = BigInt(cycle.reserve_balance_units);
  if (reserve < BigInt(units)) {
    fail(`automation cycle ${cycle.cycle_index} has insufficient reserved funding`);
  }
  cycle.reserve_balance_units = (reserve - BigInt(units)).toString();
  cycle.committed_units = (BigInt(cycle.committed_units) + BigInt(units)).toString();
  cycle[`${kind}_funding_event_id`] = eventId;
  if (cycle.bounty_funding_event_id && cycle.compute_funding_event_id) {
    if (cycle.reserve_balance_units !== "0") {
      fail(`automation cycle ${cycle.cycle_index} retained an unexpected reserve`);
    }
    cycle.status = "funded";
  }
  return true;
}

export function createProviderBidCommitment({
  launch_id,
  stage_id,
  provider,
  unit_price_units,
  max_compute_units,
  nonce,
}) {
  return sha256Canonical({
    schema: "nsrl.provider_bid_reveal.v1",
    launch_id: asString(launch_id, "bid launch_id"),
    stage_id: asString(stage_id, "bid stage_id"),
    provider: asAccount(provider, "bid provider"),
    unit_price_units: asUintString(unit_price_units, "bid unit_price_units", {
      positive: true,
    }),
    max_compute_units: asUintString(max_compute_units, "bid max_compute_units", {
      positive: true,
    }),
    nonce: asString(nonce, "bid nonce"),
  });
}

export function buildStageAuctionClosePayload(state, launchId, stageId) {
  const launch = requireLaunch(state, launchId);
  const escrow = requireComputeEscrow(state, launch.id);
  const stage = requireStage(launch, stageId);
  const key = stageKey(launch.id, stage.id);
  const availableBudget = BigInt(escrow.balance_units) - BigInt(escrow.reserved_units);
  const eligible = Object.values(state.bid_reveals[key] ?? {})
    .filter((bid) => {
      const collateral = state.provider_collateral[bid.provider];
      const availableCollateral = collateral
        ? BigInt(collateral.locked_units) - BigInt(collateral.reserved_units)
        : 0n;
      const total = BigInt(stage.compute_units) * BigInt(bid.unit_price_units);
      return (
        BigInt(bid.max_compute_units) >= BigInt(stage.compute_units) &&
        total <= availableBudget &&
        availableCollateral >= BigInt(escrow.minimum_collateral_units)
      );
    })
    .sort((left, right) => {
      const priceDelta = BigInt(left.unit_price_units) - BigInt(right.unit_price_units);
      if (priceDelta !== 0n) {
        return priceDelta < 0n ? -1 : 1;
      }
      return left.event_id.localeCompare(right.event_id);
    });
  if (eligible.length === 0) {
    fail(`stage ${stage.id} has no eligible revealed provider bid`);
  }
  const winner = eligible[0];
  return {
    launch_id: launch.id,
    stage_id: stage.id,
    provider: winner.provider,
    winning_bid_event_id: winner.event_id,
    reserved_payment_units: (
      BigInt(stage.compute_units) * BigInt(winner.unit_price_units)
    ).toString(),
  };
}

export function buildComputeRewardDistributionPayload(state, launchId) {
  const launch = requireLaunch(state, launchId);
  const publication = state.publications[launch.id];
  if (!publication) {
    fail(`launch ${launch.id} is not published`);
  }
  const computeAllocation = publication.receipt.reward_block.reward.allocations.find(
    (allocation) => allocation.role === "compute",
  );
  if (!computeAllocation) {
    fail("publication reward has no compute allocation");
  }
  const byProvider = {};
  for (const payment of Object.values(state.stage_payments)) {
    if (payment.launch_id === launch.id) {
      byProvider[payment.provider] = (
        BigInt(byProvider[payment.provider] ?? "0") + BigInt(payment.compute_units)
      ).toString();
    }
  }
  const totalCompute = Object.values(byProvider).reduce(
    (sum, units) => sum + BigInt(units),
    0n,
  );
  if (totalCompute === 0n) {
    fail("compute reward distribution requires paid accepted stages");
  }
  const totalReward = BigInt(computeAllocation.units);
  const rows = Object.entries(byProvider)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([provider, computeUnits]) => {
      const numerator = totalReward * BigInt(computeUnits);
      return {
        provider,
        accepted_compute_units: computeUnits,
        units: numerator / totalCompute,
        remainder: numerator % totalCompute,
      };
    });
  let assigned = rows.reduce((sum, row) => sum + row.units, 0n);
  const remainderOrder = [...rows].sort((left, right) => {
    if (left.remainder === right.remainder) {
      return left.provider.localeCompare(right.provider);
    }
    return left.remainder > right.remainder ? -1 : 1;
  });
  let cursor = 0;
  while (assigned < totalReward) {
    remainderOrder[cursor % remainderOrder.length].units += 1n;
    assigned += 1n;
    cursor += 1;
  }
  return {
    launch_id: launch.id,
    asset_symbol: publication.receipt.reward_block.reward.asset_symbol,
    source_account: computeAllocation.account,
    total_units: computeAllocation.units,
    allocations: rows.map(({ provider, accepted_compute_units, units }) => ({
      provider,
      accepted_compute_units,
      units: units.toString(),
    })),
  };
}

export function buildStagePaymentSettlementPayload(state, stageEventId) {
  const subjectId = asHash(stageEventId, "paid stage event id");
  const accepted = state.accepted_stages[subjectId];
  const submission = state.stage_submissions[subjectId];
  if (!accepted || !submission) {
    fail("stage payment requires accepted stage evidence");
  }
  const assignment = state.stage_assignments[
    stageKey(submission.launch_id, submission.stage_id)
  ];
  if (!assignment) {
    fail("stage payment requires a market assignment");
  }
  const meter = state.meter_receipts[subjectId];
  if (!meter) {
    fail("stage payment requires a signed meter receipt");
  }
  return {
    stage_event_id: subjectId,
    provider: assignment.provider,
    payment_units: (
      BigInt(submission.compute_units) * BigInt(assignment.unit_price_units)
    ).toString(),
    meter_event_id: meter.event_id,
  };
}

export function buildComputeBudgetRefundPayload(state, launchId) {
  const launch = requireLaunch(state, launchId);
  const escrow = requireComputeEscrow(state, launch.id);
  return {
    launch_id: launch.id,
    sponsor: escrow.sponsor,
    refund_units: escrow.balance_units,
  };
}

export function buildLaunchExpiryPayload(state, launchId) {
  const launch = requireLaunch(state, launchId);
  const escrow = requireComputeEscrow(state, launch.id);
  const slashed = Object.values(state.stage_assignments)
    .filter((assignment) => assignment.launch_id === launch.id && assignment.status === "assigned")
    .reduce((sum, assignment) => sum + BigInt(assignment.collateral_units), 0n);
  const bountyRefund = Object.values(launch.funded_bounties).reduce(
    (sum, bounty) => sum + BigInt(bounty.escrow_balance_units ?? "0"),
    0n,
  );
  return {
    launch_id: launch.id,
    compute_refund_units: escrow.balance_units,
    bounty_refund_units: bountyRefund.toString(),
    slashed_collateral_units: slashed.toString(),
  };
}

function requireNetwork(state) {
  if (!state.network) {
    fail("network is not initialized");
  }
  return state.network;
}

function requireRegisteredActor(state, intent) {
  const account = state.accounts[intent.actor];
  if (!account) {
    fail(`actor ${intent.actor} is not registered`);
  }
  if (account.public_key !== intent.public_key) {
    fail(`actor ${intent.actor} signed with an unregistered key`);
  }
  return account;
}

function requireAuthority(state, intent) {
  const network = requireNetwork(state);
  requireRegisteredActor(state, intent);
  if (intent.actor !== network.authority_account) {
    fail(`${intent.event_type} must be signed by the network authority`);
  }
}

function requireLaunch(state, launchId) {
  const launch = state.launches[asString(launchId, "launch id")];
  if (!launch) {
    fail(`unknown launch ${launchId}`);
  }
  return launch;
}

function requireSubject(state, subjectType, subjectEventId) {
  const id = asHash(subjectEventId, "subject event id");
  if (subjectType === "stage") {
    const stage = state.stage_submissions[id];
    if (!stage) {
      fail(`unknown stage subject ${id}`);
    }
    return { type: "stage", record: stage, launch_id: stage.launch_id };
  }
  if (subjectType === "candidate") {
    const candidate = state.candidates[id];
    if (!candidate) {
      fail(`unknown candidate subject ${id}`);
    }
    return { type: "candidate", record: candidate, launch_id: candidate.launch_id };
  }
  fail("subject type must be stage or candidate");
}

function openChallenges(state, subjectEventId) {
  return Object.values(state.challenges).filter(
    (challenge) => challenge.subject_event_id === subjectEventId && challenge.status === "open",
  );
}

function attestationCounts(state, subjectEventId) {
  const rows = Object.values(state.attestations[subjectEventId] ?? {});
  return {
    valid: rows.filter((row) => row.verdict === "valid").length,
    invalid: rows.filter((row) => row.verdict === "invalid").length,
    full_replay: rows.filter(
      (row) => row.verdict === "valid" && row.check_mode === "full_replay",
    ).length,
    artifact_check: rows.filter(
      (row) => row.verdict === "valid" && row.check_mode === "artifact_check",
    ).length,
  };
}

function conflictAccounts(state, launch) {
  const conflicts = new Set([
    launch.recipe.proposer.account,
    launch.recipe.participants.builder,
    launch.recipe.participants.compute,
    launch.recipe.participants.sponsor,
    launch.recipe.participants.treasury,
  ]);
  for (const assignment of Object.values(state.stage_assignments)) {
    if (assignment.launch_id === launch.id) {
      conflicts.add(assignment.provider);
    }
  }
  return conflicts;
}

function applyIntent(state, intent, eventId) {
  const payload = intent.payload;
  switch (intent.event_type) {
    case "network_initialized": {
      if (state.network) {
        fail("network is already initialized");
      }
      const networkId = asString(payload.network_id, "network_id");
      if (!NETWORK_PATTERN.test(networkId)) {
        fail("network_id must be a lowercase kebab-case slug");
      }
      const authority = asAccount(payload.authority_account, "authority_account");
      if (authority !== intent.actor || payload.authority_public_key !== intent.public_key) {
        fail("network authority identity does not match its signed intent");
      }
      state.network = {
        id: networkId,
        authority_account: authority,
        stage_quorum: asPositiveInteger(payload.stage_quorum, "stage_quorum"),
        candidate_quorum: asPositiveInteger(payload.candidate_quorum, "candidate_quorum"),
        full_replay_required: asPositiveInteger(
          payload.full_replay_required,
          "full_replay_required",
        ),
        test_credit_enabled: payload.test_credit_enabled === true,
        credit_symbol:
          payload.test_credit_enabled === true
            ? asString(payload.credit_symbol, "credit_symbol")
            : null,
        initialized_event_id: eventId,
      };
      if (state.network.full_replay_required > state.network.candidate_quorum) {
        fail("full_replay_required cannot exceed candidate_quorum");
      }
      state.accounts[authority] = {
        public_key: intent.public_key,
        registered_event_id: eventId,
        authority: true,
      };
      break;
    }
    case "account_registered": {
      requireNetwork(state);
      const account = asAccount(payload.account, "registered account");
      if (account !== intent.actor || payload.public_key !== intent.public_key) {
        fail("account registration must be self-signed by the registered key");
      }
      if (state.accounts[account]) {
        fail(`account ${account} is already registered`);
      }
      state.accounts[account] = {
        public_key: intent.public_key,
        registered_event_id: eventId,
        authority: false,
      };
      break;
    }
    case "test_credit_issued": {
      requireAuthority(state, intent);
      requireTestCredits(state);
      const recipient = asAccount(payload.recipient, "test-credit recipient");
      if (!state.accounts[recipient]) {
        fail(`test-credit recipient ${recipient} is not registered`);
      }
      const units = asUintString(payload.units, "issued test-credit units", {
        positive: true,
      });
      addBalance(state, recipient, units);
      state.test_credit_supply_units = (
        BigInt(state.test_credit_supply_units) + BigInt(units)
      ).toString();
      break;
    }
    case "bounty_automation_policy_registered": {
      requireRegisteredActor(state, intent);
      requireTestCredits(state);
      const policy = clone(asObject(payload.policy, "bounty automation policy"));
      validateBountyAutomationPolicy(policy);
      if (intent.actor !== policy.sponsor) {
        fail("only the declared sponsor can register a bounty automation policy");
      }
      if (state.automation_policies[policy.id]) {
        fail(`bounty automation policy ${policy.id} is already registered`);
      }
      if (payload.policy_sha256 !== sha256Canonical(policy)) {
        fail("bounty automation policy SHA-256 does not match its canonical policy");
      }
      for (const [role, account] of [
        ["sponsor", policy.sponsor],
        ["proposer", policy.proposer],
        ["keeper", policy.keeper],
      ]) {
        if (!state.accounts[account]) {
          fail(`automation ${role} ${account} is not registered`);
        }
      }
      const source = requireLaunch(state, policy.source_launch_id);
      if (!state.publications[source.id]) {
        fail("automation source launch must already be promoted");
      }
      if (source.recipe.proposer.account !== policy.proposer) {
        fail("automation proposer must match the source launch proposer");
      }
      if (source.recipe.participants.sponsor !== policy.sponsor) {
        fail("automation sponsor must match the source launch sponsor");
      }
      const sourceBounty = source.recipe.bounties.find(
        (row) => row.metric === policy.objective.metric,
      );
      if (!sourceBounty || sourceBounty.direction !== policy.objective.direction) {
        fail("automation objective must match a source bounty metric and direction");
      }
      state.automation_policies[policy.id] = {
        policy,
        policy_sha256: payload.policy_sha256,
        status: "active",
        spent_units: "0",
        last_opened_slot: null,
        registered_event_id: eventId,
        cycles: {},
      };
      state.automation_approvals[policy.id] = {};
      break;
    }
    case "bounty_automation_policy_paused":
    case "bounty_automation_policy_resumed": {
      requireRegisteredActor(state, intent);
      const record = requireAutomationPolicy(state, payload.policy_id);
      if (intent.actor !== record.policy.sponsor) {
        fail("only the automation sponsor can change policy status");
      }
      const next = intent.event_type.endsWith("paused") ? "paused" : "active";
      if (record.status === next) {
        fail(`bounty automation policy is already ${next}`);
      }
      record.status = next;
      record.status_event_id = eventId;
      break;
    }
    case "bounty_automation_cycle_approved": {
      requireRegisteredActor(state, intent);
      const record = requireAutomationPolicy(state, payload.policy_id);
      if (intent.actor !== record.policy.sponsor) {
        fail("only the automation sponsor can approve a high-value cycle");
      }
      const cycleIndex = asPositiveInteger(payload.cycle_index, "automation cycle_index");
      const expectedCycle = Object.keys(record.cycles).length + 1;
      if (cycleIndex !== expectedCycle) {
        fail(`automation approval must target next cycle ${expectedCycle}`);
      }
      const spend = automationCycleSpendUnits(record.policy);
      if (payload.approved_units !== spend) {
        fail("automation approval amount must equal the complete cycle spend");
      }
      if (BigInt(spend) <= BigInt(record.policy.budgets.manual_approval_above_units)) {
        fail("automation cycle does not require manual approval");
      }
      if (state.automation_approvals[record.policy.id][cycleIndex]) {
        fail(`automation cycle ${cycleIndex} is already approved`);
      }
      state.automation_approvals[record.policy.id][cycleIndex] = {
        event_id: eventId,
        cycle_index: cycleIndex,
        approved_units: spend,
      };
      break;
    }
    case "bounty_automation_cycle_opened": {
      requireRegisteredActor(state, intent);
      const record = requireAutomationPolicy(state, payload.policy_id);
      const policy = record.policy;
      if (intent.actor !== policy.keeper) {
        fail("automation cycle must be signed by the declared keeper");
      }
      if (record.status !== "active") {
        fail(`bounty automation policy is ${record.status}`);
      }
      const cycleIndex = asPositiveInteger(payload.cycle_index, "automation cycle_index");
      const expectedCycle = Object.keys(record.cycles).length + 1;
      if (cycleIndex !== expectedCycle || cycleIndex > policy.limits.max_cycles) {
        fail(`automation cycle must be the next allowed cycle ${expectedCycle}`);
      }
      const sourceLaunchId =
        cycleIndex === 1
          ? policy.source_launch_id
          : record.cycles[String(cycleIndex - 1)]?.launch_id;
      if (!sourceLaunchId || payload.source_launch_id !== sourceLaunchId) {
        fail("automation cycle source must be the previous promoted lineage tip");
      }
      const sourcePublication = state.publications[sourceLaunchId];
      if (!sourcePublication) {
        fail("automation cycle requires a promoted source launch");
      }
      const activeCycles = Object.values(record.cycles).filter((cycle) => {
        const launch = state.launches[cycle.launch_id];
        return !launch || !["promoted", "expired"].includes(launch.status);
      }).length;
      if (activeCycles >= policy.limits.max_active_bounties) {
        fail("automation policy reached its active bounty limit");
      }
      const cooldownBase = record.last_opened_slot ?? sourcePublication.slot ?? 0;
      if (state.slot < cooldownBase + policy.limits.cooldown_slots) {
        fail("automation policy cooldown has not elapsed");
      }
      const cycleSpend = automationCycleSpendUnits(policy);
      if (
        BigInt(record.spent_units) + BigInt(cycleSpend) >
        BigInt(policy.budgets.max_total_spend_units)
      ) {
        fail("automation cycle exceeds the policy lifetime spend cap");
      }
      if (balanceUnits(state, policy.sponsor) < BigInt(cycleSpend)) {
        fail("automation sponsor balance cannot fund the complete cycle");
      }
      if (
        BigInt(cycleSpend) > BigInt(policy.budgets.manual_approval_above_units) &&
        !state.automation_approvals[policy.id][cycleIndex]
      ) {
        fail("automation cycle requires explicit sponsor approval");
      }
      const expected = buildAutomationCycleOpenPayload(
        state,
        policy,
        cycleIndex,
        sourceLaunchId,
        payload.published_at,
      );
      if (canonicalJson(payload) !== canonicalJson(expected)) {
        fail("automation cycle does not match its deterministic policy plan");
      }
      record.cycles[String(cycleIndex)] = {
        event_id: eventId,
        cycle_index: cycleIndex,
        source_launch_id: sourceLaunchId,
        launch_id: expected.launch_id,
        recipe_sha256: expected.recipe_sha256,
        published_at: expected.published_at,
        cycle_spend_units: cycleSpend,
        committed_units: "0",
        reserve_balance_units: cycleSpend,
        auction: automationAuctionDeadlines(policy, state.slot),
        opened_slot: state.slot,
        status: "opened",
      };
      debitBalance(state, policy.sponsor, cycleSpend, "automation cycle reservation");
      record.spent_units = (BigInt(record.spent_units) + BigInt(cycleSpend)).toString();
      record.last_opened_slot = state.slot;
      break;
    }
    case "launch_published": {
      requireRegisteredActor(state, intent);
      const recipe = clone(asObject(payload.recipe, "launch recipe"));
      validateModelLaunchRecipe(recipe);
      if (recipe.proposer.account !== intent.actor) {
        fail("only the recipe proposer can publish a launch");
      }
      if (!["draft", "open"].includes(recipe.launch.status) || recipe.publication !== null) {
        fail("localnet launches must begin as draft/open recipes without publication evidence");
      }
      const recipeSha256 = sha256Canonical(recipe);
      if (payload.recipe_sha256 !== recipeSha256) {
        fail("launch recipe SHA-256 does not match its canonical recipe");
      }
      if (state.launches[recipe.launch.id]) {
        fail(`launch ${recipe.launch.id} is already published`);
      }
      let automation = null;
      if (payload.automation_cycle_event_id !== undefined) {
        const cycleEventId = asHash(
          payload.automation_cycle_event_id,
          "automation cycle event id",
        );
        const record = Object.values(state.automation_policies).find((policy) =>
          Object.values(policy.cycles).some((cycle) => cycle.event_id === cycleEventId),
        );
        const cycle = record
          ? Object.values(record.cycles).find((row) => row.event_id === cycleEventId)
          : null;
        if (!record || !cycle || cycle.status !== "opened") {
          fail("launch references an unknown or consumed automation cycle");
        }
        if (intent.actor !== record.policy.proposer) {
          fail("automated launch must be signed by the policy proposer");
        }
        if (recipeSha256 !== cycle.recipe_sha256 || recipe.launch.id !== cycle.launch_id) {
          fail("automated launch does not match its keeper-signed cycle commitment");
        }
        validateAutomatedBountyRecipe(
          state,
          record.policy,
          cycle.cycle_index,
          cycle.source_launch_id,
          recipe,
        );
        cycle.status = "published";
        cycle.launch_event_id = eventId;
        automation = {
          policy_id: record.policy.id,
          cycle_index: cycle.cycle_index,
          cycle_event_id: cycle.event_id,
        };
      }
      state.launches[recipe.launch.id] = {
        id: recipe.launch.id,
        recipe,
        recipe_sha256: recipeSha256,
        published_event_id: eventId,
        funded_bounties: {},
        status: "open",
        automation,
      };
      break;
    }
    case "bounty_funded": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      const bounty = launch.recipe.bounties.find((row) => row.id === payload.bounty_id);
      if (!bounty) {
        fail(`unknown bounty ${payload.bounty_id}`);
      }
      if (bounty.sponsor !== intent.actor) {
        fail("only the declared sponsor can fund a bounty");
      }
      if (payload.escrow_units !== bounty.escrow_units) {
        fail("funded escrow must equal the immutable bounty amount");
      }
      if (launch.funded_bounties[bounty.id]) {
        fail(`bounty ${bounty.id} is already funded`);
      }
      const automated = automationCycleForLaunch(state, launch);
      if (state.network.test_credit_enabled && !automated) {
        debitBalance(state, intent.actor, bounty.escrow_units, "bounty funding");
      }
      if (automated) {
        consumeAutomationReserve(state, launch, bounty.escrow_units, "bounty", eventId);
      }
      launch.funded_bounties[bounty.id] = {
        sponsor: intent.actor,
        escrow_units: bounty.escrow_units,
        escrow_balance_units: bounty.escrow_units,
        funded_event_id: eventId,
      };
      break;
    }
    case "compute_budget_funded": {
      requireRegisteredActor(state, intent);
      requireTestCredits(state);
      const launch = requireLaunch(state, payload.launch_id);
      if (intent.actor !== launch.recipe.participants.sponsor) {
        fail("only the declared sponsor can fund the compute budget");
      }
      if (state.compute_escrows[launch.id]) {
        fail(`launch ${launch.id} already has a compute escrow`);
      }
      const units = asUintString(payload.escrow_units, "compute escrow units", {
        positive: true,
      });
      const bidDeadline = asPositiveInteger(payload.bid_deadline_slot, "bid_deadline_slot");
      const revealDeadline = asPositiveInteger(
        payload.reveal_deadline_slot,
        "reveal_deadline_slot",
      );
      const executionDeadline = asPositiveInteger(
        payload.execution_deadline_slot,
        "execution_deadline_slot",
      );
      if (
        bidDeadline <= state.slot ||
        revealDeadline <= bidDeadline ||
        executionDeadline <= revealDeadline
      ) {
        fail("compute escrow deadlines must increase after the current slot");
      }
      const minimumCollateral = asUintString(
        payload.minimum_collateral_units,
        "minimum_collateral_units",
        { positive: true },
      );
      const automated = automationCycleForLaunch(state, launch);
      if (!automated) {
        debitBalance(state, intent.actor, units, "compute budget funding");
      } else {
        const expectedAuction = automated.cycle.auction;
        if (
          bidDeadline !== expectedAuction.bid_deadline_slot ||
          revealDeadline !== expectedAuction.reveal_deadline_slot ||
          executionDeadline !== expectedAuction.execution_deadline_slot ||
          minimumCollateral !== automated.policy.policy.auction.minimum_collateral_units
        ) {
          fail("automated compute funding must match its keeper-bound auction terms");
        }
        consumeAutomationReserve(state, launch, units, "compute", eventId);
      }
      state.compute_escrows[launch.id] = {
        launch_id: launch.id,
        sponsor: intent.actor,
        funded_units: units,
        balance_units: units,
        reserved_units: "0",
        bid_deadline_slot: bidDeadline,
        reveal_deadline_slot: revealDeadline,
        execution_deadline_slot: executionDeadline,
        minimum_collateral_units: minimumCollateral,
        funded_event_id: eventId,
        status: "open",
      };
      break;
    }
    case "provider_collateral_deposited": {
      requireRegisteredActor(state, intent);
      requireTestCredits(state);
      const units = asUintString(payload.units, "provider collateral units", {
        positive: true,
      });
      debitBalance(state, intent.actor, units, "collateral deposit");
      const collateral = state.provider_collateral[intent.actor] ?? {
        provider: intent.actor,
        locked_units: "0",
        reserved_units: "0",
        deposit_event_ids: [],
      };
      collateral.locked_units = (BigInt(collateral.locked_units) + BigInt(units)).toString();
      collateral.deposit_event_ids.push(eventId);
      state.provider_collateral[intent.actor] = collateral;
      break;
    }
    case "provider_bid_committed": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      const escrow = requireComputeEscrow(state, launch.id);
      const stage = requireStage(launch, payload.stage_id);
      if (state.slot > escrow.bid_deadline_slot) {
        fail("provider bid commitment arrived after the bid deadline");
      }
      const collateral = state.provider_collateral[intent.actor];
      const availableCollateral = collateral
        ? BigInt(collateral.locked_units) - BigInt(collateral.reserved_units)
        : 0n;
      if (availableCollateral < BigInt(escrow.minimum_collateral_units)) {
        fail("provider bid requires the minimum available collateral");
      }
      const key = stageKey(launch.id, stage.id);
      state.bid_commits[key] ??= {};
      if (state.bid_commits[key][intent.actor]) {
        fail(`provider ${intent.actor} already committed a bid for ${stage.id}`);
      }
      state.bid_commits[key][intent.actor] = {
        event_id: eventId,
        launch_id: launch.id,
        stage_id: stage.id,
        provider: intent.actor,
        commitment_sha256: asHash(payload.commitment_sha256, "bid commitment_sha256"),
        slot: state.slot,
      };
      break;
    }
    case "slot_advanced": {
      requireAuthority(state, intent);
      const nextSlot = asPositiveInteger(payload.slot, "slot");
      if (nextSlot <= state.slot) {
        fail("logical slot must advance monotonically");
      }
      state.slot = nextSlot;
      break;
    }
    case "provider_bid_revealed": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      const escrow = requireComputeEscrow(state, launch.id);
      const stage = requireStage(launch, payload.stage_id);
      if (state.slot <= escrow.bid_deadline_slot) {
        fail("provider bid cannot be revealed before the bid window closes");
      }
      if (state.slot > escrow.reveal_deadline_slot) {
        fail("provider bid reveal arrived after the reveal deadline");
      }
      const key = stageKey(launch.id, stage.id);
      const commit = state.bid_commits[key]?.[intent.actor];
      if (!commit) {
        fail(`provider ${intent.actor} has no committed bid for ${stage.id}`);
      }
      state.bid_reveals[key] ??= {};
      if (state.bid_reveals[key][intent.actor]) {
        fail(`provider ${intent.actor} already revealed its bid for ${stage.id}`);
      }
      const reveal = {
        launch_id: launch.id,
        stage_id: stage.id,
        provider: intent.actor,
        unit_price_units: asUintString(payload.unit_price_units, "bid unit_price_units", {
          positive: true,
        }),
        max_compute_units: asUintString(
          payload.max_compute_units,
          "bid max_compute_units",
          { positive: true },
        ),
        nonce: asString(payload.nonce, "bid nonce"),
      };
      if (createProviderBidCommitment(reveal) !== commit.commitment_sha256) {
        fail("revealed provider bid does not match its commitment");
      }
      state.bid_reveals[key][intent.actor] = {
        ...reveal,
        event_id: eventId,
        commitment_event_id: commit.event_id,
        slot: state.slot,
      };
      break;
    }
    case "stage_auction_closed": {
      requireAuthority(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      const escrow = requireComputeEscrow(state, launch.id);
      const stage = requireStage(launch, payload.stage_id);
      if (state.slot <= escrow.reveal_deadline_slot) {
        fail("stage auction cannot close before the reveal deadline");
      }
      if (state.slot > escrow.execution_deadline_slot) {
        fail("stage auction cannot close after the execution deadline");
      }
      const key = stageKey(launch.id, stage.id);
      if (state.stage_assignments[key]) {
        fail(`stage ${stage.id} is already assigned`);
      }
      const expected = buildStageAuctionClosePayload(state, launch.id, stage.id);
      if (canonicalJson(payload) !== canonicalJson(expected)) {
        fail("stage assignment does not match deterministic auction ranking");
      }
      const winner = state.bid_reveals[key][expected.provider];
      const reservedCost = expected.reserved_payment_units;
      const collateral = state.provider_collateral[winner.provider];
      collateral.reserved_units = (
        BigInt(collateral.reserved_units) + BigInt(escrow.minimum_collateral_units)
      ).toString();
      escrow.reserved_units = (BigInt(escrow.reserved_units) + BigInt(reservedCost)).toString();
      state.stage_assignments[key] = {
        event_id: eventId,
        launch_id: launch.id,
        stage_id: stage.id,
        provider: winner.provider,
        winning_bid_event_id: winner.event_id,
        unit_price_units: winner.unit_price_units,
        max_compute_units: winner.max_compute_units,
        reserved_payment_units: reservedCost,
        collateral_units: escrow.minimum_collateral_units,
        status: "assigned",
      };
      break;
    }
    case "stage_submitted": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      if (state.expired_launches[launch.id]) {
        fail("expired launches cannot accept stage evidence");
      }
      for (const bounty of launch.recipe.bounties) {
        if (!launch.funded_bounties[bounty.id]) {
          fail(`bounty ${bounty.id} must be funded before compute begins`);
        }
      }
      const stage = requireStage(launch, payload.stage_id);
      const assignment = state.stage_assignments[stageKey(launch.id, stage.id)];
      const expectedProvider = assignment?.provider ?? launch.recipe.participants.compute;
      if (intent.actor !== expectedProvider) {
        fail("stage submission must be signed by the assigned compute provider");
      }
      if (assignment) {
        const escrow = requireComputeEscrow(state, launch.id);
        if (state.slot > escrow.execution_deadline_slot) {
          fail("stage evidence arrived after the execution deadline");
        }
      }
      if (
        Object.values(state.stage_submissions).some(
          (row) => row.launch_id === launch.id && row.stage_id === stage.id,
        )
      ) {
        fail(`stage ${stage.id} already has a submission`);
      }
      const computeUnits = asUintString(payload.compute_units, "stage compute_units", {
        positive: true,
      });
      if (BigInt(computeUnits) > BigInt(stage.compute_units)) {
        fail("stage compute claim exceeds its recipe ceiling");
      }
      state.stage_submissions[eventId] = {
        event_id: eventId,
        actor: intent.actor,
        launch_id: launch.id,
        stage_id: stage.id,
        input_sha256: asHash(payload.input_sha256, "stage input_sha256"),
        output_sha256: asHash(payload.output_sha256, "stage output_sha256"),
        evidence_sha256: asHash(payload.evidence_sha256, "stage evidence_sha256"),
        compute_units: computeUnits,
      };
      break;
    }
    case "compute_metered": {
      requireRegisteredActor(state, intent);
      const subjectId = asHash(payload.stage_event_id, "metered stage event id");
      const submission = state.stage_submissions[subjectId];
      if (!submission) {
        fail(`unknown stage submission ${subjectId}`);
      }
      const key = stageKey(submission.launch_id, submission.stage_id);
      const assignment = state.stage_assignments[key];
      if (!assignment || assignment.provider !== intent.actor) {
        fail("meter receipt must be signed by the assigned provider");
      }
      if (state.meter_receipts[subjectId]) {
        fail("stage submission already has a meter receipt");
      }
      const startSlot = asNonnegativeInteger(payload.start_slot, "meter start_slot");
      const endSlot = asPositiveInteger(payload.end_slot, "meter end_slot");
      if (endSlot <= startSlot || endSlot > state.slot) {
        fail("meter slots must increase and end no later than the current slot");
      }
      const escrow = requireComputeEscrow(state, submission.launch_id);
      if (endSlot > escrow.execution_deadline_slot || state.slot > escrow.execution_deadline_slot) {
        fail("meter receipt arrived after the execution deadline");
      }
      const computeUnits = asUintString(payload.compute_units, "meter compute_units", {
        positive: true,
      });
      if (computeUnits !== submission.compute_units) {
        fail("metered compute units must equal the submitted stage claim");
      }
      if (
        payload.input_sha256 !== submission.input_sha256 ||
        payload.output_sha256 !== submission.output_sha256 ||
        payload.evidence_sha256 !== submission.evidence_sha256
      ) {
        fail("meter receipt hashes must match the submitted stage evidence");
      }
      state.meter_receipts[subjectId] = {
        event_id: eventId,
        stage_event_id: subjectId,
        launch_id: submission.launch_id,
        stage_id: submission.stage_id,
        provider: intent.actor,
        start_slot: startSlot,
        end_slot: endSlot,
        compute_units: computeUnits,
        input_sha256: submission.input_sha256,
        output_sha256: submission.output_sha256,
        evidence_sha256: submission.evidence_sha256,
      };
      break;
    }
    case "validation_attested": {
      requireRegisteredActor(state, intent);
      const subject = requireSubject(state, payload.subject_type, payload.subject_event_id);
      const launch = requireLaunch(state, subject.launch_id);
      if (state.expired_launches[launch.id]) {
        fail("expired launches cannot accept validator attestations");
      }
      if (conflictAccounts(state, launch).has(intent.actor)) {
        fail(`validator ${intent.actor} conflicts with launch execution or funding roles`);
      }
      if (
        (subject.type === "stage" && state.accepted_stages[payload.subject_event_id]) ||
        (subject.type === "candidate" && state.publications[launch.id])
      ) {
        fail("finalized evidence cannot receive another attestation");
      }
      if (!["valid", "invalid"].includes(payload.verdict)) {
        fail("attestation verdict must be valid or invalid");
      }
      if (!["artifact_check", "full_replay"].includes(payload.check_mode)) {
        fail("attestation check_mode must be artifact_check or full_replay");
      }
      const subjectId = payload.subject_event_id;
      state.attestations[subjectId] ??= {};
      if (state.attestations[subjectId][intent.actor]) {
        fail(`validator ${intent.actor} already attested this subject`);
      }
      state.attestations[subjectId][intent.actor] = {
        event_id: eventId,
        validator: intent.actor,
        verdict: payload.verdict,
        check_mode: payload.check_mode,
        evidence_sha256: asHash(payload.evidence_sha256, "attestation evidence_sha256"),
      };
      break;
    }
    case "challenge_opened": {
      requireRegisteredActor(state, intent);
      const subject = requireSubject(state, payload.subject_type, payload.subject_event_id);
      const launch = requireLaunch(state, subject.launch_id);
      if (state.expired_launches[launch.id]) {
        fail("expired launches cannot accept challenges");
      }
      if (
        (subject.type === "stage" && state.accepted_stages[payload.subject_event_id]) ||
        (subject.type === "candidate" && state.publications[launch.id])
      ) {
        fail("finalized evidence cannot be challenged");
      }
      if (subject.record.actor === intent.actor) {
        fail("a subject submitter cannot challenge its own evidence");
      }
      if (
        Object.values(state.challenges).some(
          (row) =>
            row.subject_event_id === payload.subject_event_id &&
            row.challenger === intent.actor &&
            row.status === "open",
        )
      ) {
        fail("challenger already has an open challenge for this subject");
      }
      state.challenges[eventId] = {
        event_id: eventId,
        challenger: intent.actor,
        subject_type: subject.type,
        subject_event_id: payload.subject_event_id,
        reason: asString(payload.reason, "challenge reason"),
        evidence_sha256: asHash(payload.evidence_sha256, "challenge evidence_sha256"),
        status: "open",
      };
      break;
    }
    case "challenge_resolved": {
      requireAuthority(state, intent);
      const challenge = state.challenges[asHash(payload.challenge_event_id, "challenge event id")];
      if (!challenge || challenge.status !== "open") {
        fail("challenge is unknown or already resolved");
      }
      if (!["upheld", "rejected"].includes(payload.outcome)) {
        fail("challenge outcome must be upheld or rejected");
      }
      challenge.status = payload.outcome;
      challenge.resolution_event_id = eventId;
      challenge.resolution_evidence_sha256 = asHash(
        payload.evidence_sha256,
        "challenge resolution evidence_sha256",
      );
      if (payload.outcome === "upheld") {
        state.invalid_subjects[challenge.subject_event_id] = eventId;
      }
      break;
    }
    case "stage_accepted": {
      requireAuthority(state, intent);
      const subjectId = asHash(payload.stage_event_id, "stage event id");
      const stage = state.stage_submissions[subjectId];
      if (!stage) {
        fail(`unknown stage submission ${subjectId}`);
      }
      if (state.accepted_stages[subjectId]) {
        fail("stage submission is already accepted");
      }
      if (state.expired_launches[stage.launch_id]) {
        fail("expired launches cannot accept stage evidence");
      }
      if (state.invalid_subjects[subjectId] || openChallenges(state, subjectId).length > 0) {
        fail("challenged or invalid stage evidence cannot be accepted");
      }
      const counts = attestationCounts(state, subjectId);
      if (counts.invalid > 0 || counts.valid < state.network.stage_quorum) {
        fail("stage submission has not reached a clean validator quorum");
      }
      const assignment = state.stage_assignments[stageKey(stage.launch_id, stage.stage_id)];
      if (assignment && !state.meter_receipts[subjectId]) {
        fail("market-assigned stage evidence requires a signed meter receipt");
      }
      state.accepted_stages[subjectId] = {
        event_id: eventId,
        stage_event_id: subjectId,
        launch_id: stage.launch_id,
        stage_id: stage.stage_id,
        compute_units: stage.compute_units,
        validator_counts: counts,
      };
      break;
    }
    case "stage_payment_settled": {
      requireAuthority(state, intent);
      requireTestCredits(state);
      const subjectId = asHash(payload.stage_event_id, "paid stage event id");
      const accepted = state.accepted_stages[subjectId];
      const submission = state.stage_submissions[subjectId];
      if (!accepted || !submission) {
        fail("stage payment requires accepted stage evidence");
      }
      if (state.stage_payments[subjectId]) {
        fail("accepted stage evidence is already paid");
      }
      const expectedPayment = buildStagePaymentSettlementPayload(state, subjectId);
      if (canonicalJson(payload) !== canonicalJson(expectedPayment)) {
        fail("stage payment does not match assignment, meter, and accepted compute");
      }
      const key = stageKey(submission.launch_id, submission.stage_id);
      const assignment = state.stage_assignments[key];
      if (!assignment) {
        fail("stage payment requires a market assignment");
      }
      const escrow = requireComputeEscrow(state, submission.launch_id);
      const payment = (
        BigInt(submission.compute_units) * BigInt(assignment.unit_price_units)
      ).toString();
      if (
        BigInt(payment) > BigInt(escrow.balance_units) ||
        BigInt(payment) > BigInt(escrow.reserved_units)
      ) {
        fail("stage payment exceeds the funded or reserved compute escrow");
      }
      escrow.balance_units = (BigInt(escrow.balance_units) - BigInt(payment)).toString();
      escrow.reserved_units = (BigInt(escrow.reserved_units) - BigInt(payment)).toString();
      addBalance(state, assignment.provider, payment);

      const collateral = state.provider_collateral[assignment.provider];
      collateral.locked_units = (
        BigInt(collateral.locked_units) - BigInt(assignment.collateral_units)
      ).toString();
      collateral.reserved_units = (
        BigInt(collateral.reserved_units) - BigInt(assignment.collateral_units)
      ).toString();
      addBalance(state, assignment.provider, assignment.collateral_units);
      assignment.status = "paid";
      assignment.payment_event_id = eventId;
      state.stage_payments[subjectId] = {
        event_id: eventId,
        stage_event_id: subjectId,
        launch_id: submission.launch_id,
        stage_id: submission.stage_id,
        provider: assignment.provider,
        compute_units: submission.compute_units,
        unit_price_units: assignment.unit_price_units,
        payment_units: payment,
        collateral_released_units: assignment.collateral_units,
        meter_event_id: payload.meter_event_id,
      };
      break;
    }
    case "compute_budget_refunded": {
      requireAuthority(state, intent);
      requireTestCredits(state);
      const launch = requireLaunch(state, payload.launch_id);
      const escrow = requireComputeEscrow(state, launch.id);
      for (const stage of launch.recipe.run.stages) {
        const paid = Object.values(state.stage_payments).some(
          (row) => row.launch_id === launch.id && row.stage_id === stage.id,
        );
        if (!paid) {
          fail(`compute budget cannot close before stage ${stage.id} is paid`);
        }
      }
      if (escrow.reserved_units !== "0") {
        fail("compute budget cannot close with reserved stage payments");
      }
      const refund = escrow.balance_units;
      if (canonicalJson(payload) !== canonicalJson(buildComputeBudgetRefundPayload(state, launch.id))) {
        fail("compute budget refund does not match remaining escrow");
      }
      addBalance(state, escrow.sponsor, refund);
      escrow.balance_units = "0";
      escrow.refunded_units = refund;
      escrow.refund_event_id = eventId;
      escrow.status = "settled";
      break;
    }
    case "provider_collateral_withdrawn": {
      requireRegisteredActor(state, intent);
      requireTestCredits(state);
      const collateral = state.provider_collateral[intent.actor];
      if (!collateral) {
        fail(`provider ${intent.actor} has no collateral balance`);
      }
      const units = asUintString(payload.units, "collateral withdrawal units", {
        positive: true,
      });
      const available = BigInt(collateral.locked_units) - BigInt(collateral.reserved_units);
      if (BigInt(units) > available) {
        fail("collateral withdrawal exceeds the unreserved balance");
      }
      collateral.locked_units = (BigInt(collateral.locked_units) - BigInt(units)).toString();
      addBalance(state, intent.actor, units);
      break;
    }
    case "candidate_submitted": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      if (state.expired_launches[launch.id]) {
        fail("expired launches cannot accept a candidate");
      }
      if (intent.actor !== launch.recipe.participants.builder) {
        fail("candidate submission must be signed by the declared builder");
      }
      for (const stage of launch.recipe.run.stages) {
        const accepted = Object.values(state.accepted_stages).some(
          (row) => row.launch_id === launch.id && row.stage_id === stage.id,
        );
        if (!accepted) {
          fail(`candidate cannot be submitted before stage ${stage.id} is accepted`);
        }
      }
      const computeEscrow = state.compute_escrows[launch.id];
      if (computeEscrow && computeEscrow.status !== "settled") {
        fail("candidate cannot be submitted before compute payments and refund settle");
      }
      if (Object.values(state.candidates).some((row) => row.launch_id === launch.id)) {
        fail(`launch ${launch.id} already has a candidate`);
      }
      if (!MODEL_HASH_PATTERN.test(asString(payload.model_hash, "candidate model_hash"))) {
        fail("candidate model_hash must be a 64-bit 0x-prefixed model hash");
      }
      const metrics = clone(asObject(payload.metrics, "candidate metrics"));
      for (const [metric, value] of Object.entries(metrics)) {
        asString(metric, "candidate metric name");
        asUintString(value, `candidate metric ${metric}`);
      }
      state.candidates[eventId] = {
        event_id: eventId,
        actor: intent.actor,
        launch_id: launch.id,
        model_hash: payload.model_hash,
        artifact_sha256: asHash(payload.artifact_sha256, "candidate artifact_sha256"),
        artifact_path: asString(payload.artifact_path, "candidate artifact_path"),
        proof_sha256: asHash(payload.proof_sha256, "candidate proof_sha256"),
        metrics,
      };
      launch.status = "candidate";
      break;
    }
    case "model_published": {
      requireAuthority(state, intent);
      const candidateId = asHash(payload.candidate_event_id, "candidate event id");
      const candidate = state.candidates[candidateId];
      if (!candidate) {
        fail(`unknown candidate ${candidateId}`);
      }
      const launch = requireLaunch(state, candidate.launch_id);
      if (state.expired_launches[launch.id]) {
        fail("expired launches cannot publish a model");
      }
      if (state.publications[launch.id]) {
        fail(`launch ${launch.id} is already published`);
      }
      if (state.invalid_subjects[candidateId] || openChallenges(state, candidateId).length > 0) {
        fail("challenged or invalid candidate evidence cannot be published");
      }
      const counts = attestationCounts(state, candidateId);
      if (
        counts.invalid > 0 ||
        counts.valid < state.network.candidate_quorum ||
        counts.full_replay < state.network.full_replay_required
      ) {
        fail("candidate has not reached the required clean replay quorum");
      }
      const expected = buildModelPublicationPayload(state, candidateId);
      if (canonicalJson(payload.published_recipe) !== canonicalJson(expected.published_recipe)) {
        fail("published recipe does not match the accepted candidate");
      }
      if (canonicalJson(payload.receipt) !== canonicalJson(expected.receipt)) {
        fail("publication receipt does not match deterministic settlement");
      }
      validateModelLaunchRecipe(payload.published_recipe);
      validateModelPublishReceipt(payload.published_recipe, payload.receipt);

      const bountyRows = {};
      for (const bounty of launch.recipe.bounties) {
        const funded = launch.funded_bounties[bounty.id];
        if (!funded) {
          fail(`funded bounty evidence is missing for ${bounty.id}`);
        }
        const payout = bountyPayoutUnits(
          bounty,
          candidate.metrics[bounty.metric],
          true,
          candidate.metrics,
        );
        bountyRows[bounty.id] = {
          sponsor: bounty.sponsor,
          recipient: launch.recipe.participants.builder,
          escrow_units: bounty.escrow_units,
          settled_units: payout,
          refunded_units: (BigInt(bounty.escrow_units) - BigInt(payout)).toString(),
        };
        if (state.network.test_credit_enabled) {
          const refund = BigInt(bounty.escrow_units) - BigInt(payout);
          if (BigInt(funded.escrow_balance_units) !== BigInt(bounty.escrow_units)) {
            fail(`bounty escrow balance is inconsistent for ${bounty.id}`);
          }
          addBalance(state, launch.recipe.participants.builder, payout);
          addBalance(state, bounty.sponsor, refund);
          funded.escrow_balance_units = "0";
        }
      }
      state.bounty_settlements[launch.id] = bountyRows;

      const reward = payload.receipt.reward_block.reward;
      state.model_balances[reward.asset_symbol] ??= {};
      for (const allocation of reward.allocations) {
        const current = BigInt(state.model_balances[reward.asset_symbol][allocation.account] ?? "0");
        state.model_balances[reward.asset_symbol][allocation.account] = (
          current + BigInt(allocation.units)
        ).toString();
      }
      state.publications[launch.id] = {
        event_id: eventId,
        slot: state.slot,
        candidate_event_id: candidateId,
        model_hash: candidate.model_hash,
        artifact_sha256: candidate.artifact_sha256,
        receipt: clone(payload.receipt),
        validator_counts: counts,
      };
      launch.status = "promoted";
      launch.published_recipe_sha256 = payload.receipt.recipe_sha256;
      const automation = automationCycleForLaunch(state, launch);
      if (automation) {
        automation.cycle.status = "promoted";
        automation.cycle.publication_event_id = eventId;
      }
      break;
    }
    case "compute_reward_distributed": {
      requireAuthority(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      if (state.compute_reward_distributions[launch.id]) {
        fail(`launch ${launch.id} compute reward is already distributed`);
      }
      const expected = buildComputeRewardDistributionPayload(state, launch.id);
      if (canonicalJson(payload) !== canonicalJson(expected)) {
        fail("compute reward distribution does not match accepted stage work");
      }
      const balances = state.model_balances[payload.asset_symbol];
      if (!balances || BigInt(balances[payload.source_account] ?? "0") < BigInt(payload.total_units)) {
        fail("compute reward pool does not contain the distributable allocation");
      }
      balances[payload.source_account] = (
        BigInt(balances[payload.source_account]) - BigInt(payload.total_units)
      ).toString();
      for (const allocation of payload.allocations) {
        balances[allocation.provider] = (
          BigInt(balances[allocation.provider] ?? "0") + BigInt(allocation.units)
        ).toString();
      }
      state.compute_reward_distributions[launch.id] = {
        event_id: eventId,
        ...clone(payload),
      };
      break;
    }
    case "launch_expired": {
      requireAuthority(state, intent);
      requireTestCredits(state);
      const launch = requireLaunch(state, payload.launch_id);
      const escrow = requireComputeEscrow(state, launch.id);
      if (state.publications[launch.id]) {
        fail("published launches cannot expire");
      }
      if (state.slot <= escrow.execution_deadline_slot) {
        fail("launch cannot expire before its execution deadline");
      }
      const expectedExpiry = buildLaunchExpiryPayload(state, launch.id);
      if (canonicalJson(payload) !== canonicalJson(expectedExpiry)) {
        fail("launch expiry settlement does not match deterministic refunds and slashing");
      }
      let slashed = 0n;
      for (const assignment of Object.values(state.stage_assignments)) {
        if (assignment.launch_id !== launch.id || assignment.status !== "assigned") {
          continue;
        }
        const collateral = state.provider_collateral[assignment.provider];
        const amount = BigInt(assignment.collateral_units);
        collateral.locked_units = (BigInt(collateral.locked_units) - amount).toString();
        collateral.reserved_units = (BigInt(collateral.reserved_units) - amount).toString();
        assignment.status = "slashed";
        assignment.slash_event_id = eventId;
        slashed += amount;
      }
      addBalance(state, escrow.sponsor, slashed);

      const computeRefund = BigInt(escrow.balance_units);
      addBalance(state, escrow.sponsor, computeRefund);
      escrow.balance_units = "0";
      escrow.reserved_units = "0";
      escrow.refunded_units = computeRefund.toString();
      escrow.status = "expired";
      escrow.refund_event_id = eventId;

      let bountyRefund = 0n;
      const bountyRows = {};
      for (const bounty of launch.recipe.bounties) {
        const funded = launch.funded_bounties[bounty.id];
        if (!funded) {
          continue;
        }
        const refund = BigInt(funded.escrow_balance_units);
        addBalance(state, funded.sponsor, refund);
        funded.escrow_balance_units = "0";
        bountyRefund += refund;
        bountyRows[bounty.id] = {
          sponsor: funded.sponsor,
          recipient: null,
          escrow_units: funded.escrow_units,
          settled_units: "0",
          refunded_units: refund.toString(),
        };
      }
      state.bounty_settlements[launch.id] = bountyRows;
      state.expired_launches[launch.id] = {
        event_id: eventId,
        slot: state.slot,
        compute_refund_units: computeRefund.toString(),
        bounty_refund_units: bountyRefund.toString(),
        slashed_collateral_units: slashed.toString(),
      };
      launch.status = "expired";
      const automation = automationCycleForLaunch(state, launch);
      if (automation) {
        automation.cycle.status = "expired";
        automation.cycle.expiry_event_id = eventId;
      }
      break;
    }
    default:
      fail(`unhandled event type ${intent.event_type}`);
  }
  assertTestCreditConservation(state);
}

function verifyAndApplyEvent(state, event, expectedHeight, previousHash) {
  asObject(event, "localnet event");
  if (event.schema !== LOCALNET_EVENT_SCHEMA) {
    fail(`event schema must be ${LOCALNET_EVENT_SCHEMA}`);
  }
  if (event.height !== expectedHeight) {
    fail(`event height ${event.height} does not match expected ${expectedHeight}`);
  }
  if (event.previous_event_sha256 !== previousHash) {
    fail(`event ${expectedHeight} does not link to the previous event`);
  }
  verifyLocalnetIntent(event.signed_intent);
  const eventId = sha256Canonical(signedBody(event.signed_intent));
  if (event.event_id !== eventId) {
    fail(`event ${expectedHeight} has an invalid event_id`);
  }
  const core = {
    schema: event.schema,
    height: event.height,
    previous_event_sha256: event.previous_event_sha256,
    event_id: event.event_id,
    signed_intent: event.signed_intent,
  };
  const eventSha256 = sha256Canonical(core);
  if (event.event_sha256 !== eventSha256) {
    fail(`event ${expectedHeight} has an invalid event_sha256`);
  }
  if (state.event_ids[eventId] !== undefined) {
    fail(`event ${expectedHeight} replays existing event_id ${eventId}`);
  }
  applyIntent(state, event.signed_intent, eventId);
  state.event_ids[eventId] = event.height;
  state.height = event.height + 1;
  state.head_sha256 = eventSha256;
  return eventSha256;
}

export function replayLocalnetEvents(events) {
  if (!Array.isArray(events)) {
    fail("ledger events must be an array");
  }
  const state = emptyState();
  let previousHash = ZERO_HASH;
  events.forEach((event, height) => {
    previousHash = verifyAndApplyEvent(state, event, height, previousHash);
  });
  return state;
}

export function readLocalnetEvents(ledgerPath) {
  if (!fs.existsSync(ledgerPath)) {
    return [];
  }
  return fs
    .readFileSync(ledgerPath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        fail(`ledger line ${index + 1} is not valid JSON: ${error.message}`);
      }
    });
}

export class ModelLocalnetLedger {
  constructor(directory) {
    this.directory = path.resolve(asString(directory, "localnet directory"));
    this.ledgerPath = path.join(this.directory, "ledger.jsonl");
  }

  inspect() {
    const events = readLocalnetEvents(this.ledgerPath);
    return { events, state: replayLocalnetEvents(events) };
  }

  initialize(authorityIdentity, config = {}) {
    if (this.inspect().events.length !== 0) {
      fail("localnet ledger is already initialized");
    }
    const payload = {
        network_id: config.network_id ?? "nsrl-model-localnet-v1",
        authority_account: authorityIdentity.account,
        authority_public_key: authorityIdentity.public_key,
        stage_quorum: config.stage_quorum ?? 2,
        candidate_quorum: config.candidate_quorum ?? 3,
        full_replay_required: config.full_replay_required ?? 1,
    };
    if (config.test_credit_enabled === true) {
      payload.test_credit_enabled = true;
      payload.credit_symbol = config.credit_symbol ?? "FORGE-TEST";
    }
    return this.append(
      signLocalnetIntent(authorityIdentity, "network_initialized", payload),
    );
  }

  append(signedIntent) {
    verifyLocalnetIntent(signedIntent);
    const { events, state } = this.inspect();
    const eventId = sha256Canonical(signedBody(signedIntent));
    const duplicate = events.find((event) => event.event_id === eventId);
    if (duplicate) {
      if (canonicalJson(duplicate.signed_intent) !== canonicalJson(signedIntent)) {
        fail(`event_id ${eventId} already exists with different signed bytes`);
      }
      return { event: duplicate, duplicate: true, state };
    }
    applyIntent(state, signedIntent, eventId);
    const core = {
      schema: LOCALNET_EVENT_SCHEMA,
      height: events.length,
      previous_event_sha256: state.head_sha256,
      event_id: eventId,
      signed_intent: clone(signedIntent),
    };
    const event = { ...core, event_sha256: sha256Canonical(core) };
    fs.mkdirSync(this.directory, { recursive: true });
    fs.appendFileSync(this.ledgerPath, `${JSON.stringify(event)}\n`);
    const verified = this.inspect();
    return { event, duplicate: false, state: verified.state };
  }
}

export function registerLocalnetIdentity(ledger, identity) {
  return ledger.append(
    signLocalnetIntent(identity, "account_registered", {
      account: identity.account,
      public_key: identity.public_key,
    }),
  );
}

export function buildModelPublicationPayload(state, candidateEventId) {
  const candidate = state.candidates[asHash(candidateEventId, "candidate event id")];
  if (!candidate) {
    fail(`unknown candidate ${candidateEventId}`);
  }
  const launch = requireLaunch(state, candidate.launch_id);
  const publishedRecipe = clone(launch.recipe);
  publishedRecipe.launch.status = "promoted";
  publishedRecipe.publication = {
    model_hash: candidate.model_hash,
    artifact_sha256: candidate.artifact_sha256,
    artifact_path: candidate.artifact_path,
    proof_sha256: candidate.proof_sha256,
    metrics: clone(candidate.metrics),
  };
  validateModelLaunchRecipe(publishedRecipe);
  const receipt = createModelPublishReceipt(publishedRecipe);
  return {
    candidate_event_id: candidateEventId,
    published_recipe: publishedRecipe,
    receipt,
  };
}

export function localnetStateSummary(state) {
  asObject(state, "localnet state");
  const launches = Object.values(state.launches).map((launch) => ({
    id: launch.id,
    status: launch.status,
    recipe_sha256: launch.recipe_sha256,
    funded_bounties: Object.keys(launch.funded_bounties).length,
    required_bounties: launch.recipe.bounties.length,
    accepted_stages: Object.values(state.accepted_stages).filter(
      (stage) => stage.launch_id === launch.id,
    ).length,
    required_stages: launch.recipe.run.stages.length,
    candidate_event_id:
      Object.values(state.candidates).find((candidate) => candidate.launch_id === launch.id)
        ?.event_id ?? null,
    publication_event_id: state.publications[launch.id]?.event_id ?? null,
    compute_escrow_status: state.compute_escrows[launch.id]?.status ?? null,
    assigned_stages: Object.values(state.stage_assignments).filter(
      (assignment) => assignment.launch_id === launch.id,
    ).length,
    paid_stages: Object.values(state.stage_payments).filter(
      (payment) => payment.launch_id === launch.id,
    ).length,
    automation: clone(launch.automation ?? null),
  }));
  const auctionRows = Object.entries(state.stage_assignments).map(([key, assignment]) => ({
    key,
    launch_id: assignment.launch_id,
    stage_id: assignment.stage_id,
    provider: assignment.provider,
    unit_price_units: assignment.unit_price_units,
    reserved_payment_units: assignment.reserved_payment_units,
    status: assignment.status,
    revealed_bids: Object.keys(state.bid_reveals[key] ?? {}).length,
    payment_units:
      Object.values(state.stage_payments).find(
        (payment) =>
          payment.launch_id === assignment.launch_id &&
          payment.stage_id === assignment.stage_id,
      )?.payment_units ?? null,
  }));
  const accountedSupplyUnits = accountedTestCreditUnits(state).toString();
  return {
    schema: "nsrl.model_localnet_summary.v1",
    network: state.network,
    height: state.height,
    head_sha256: state.head_sha256,
    accounts: Object.keys(state.accounts).length,
    launches,
    stage_submissions: Object.keys(state.stage_submissions).length,
    accepted_stages: Object.keys(state.accepted_stages).length,
    attestations: Object.values(state.attestations).reduce(
      (sum, rows) => sum + Object.keys(rows).length,
      0,
    ),
    challenges: Object.values(state.challenges).reduce(
      (counts, challenge) => {
        counts[challenge.status] = (counts[challenge.status] ?? 0) + 1;
        return counts;
      },
      {},
    ),
    publications: Object.keys(state.publications).length,
    bounty_settlements: clone(state.bounty_settlements),
    model_balances: clone(state.model_balances),
    automation: {
      policies: Object.values(state.automation_policies).map((record) => {
        const cycleSpend = automationCycleSpendUnits(record.policy);
        const nextCycle = Object.keys(record.cycles).length + 1;
        return {
          policy: clone(record.policy),
          policy_sha256: record.policy_sha256,
          status: record.status,
          spent_units: record.spent_units,
          remaining_units: (
            BigInt(record.policy.budgets.max_total_spend_units) - BigInt(record.spent_units)
          ).toString(),
          cycle_spend_units: cycleSpend,
          next_cycle_index: nextCycle,
          next_cycle_requires_approval:
            BigInt(cycleSpend) >
              BigInt(record.policy.budgets.manual_approval_above_units) &&
            !state.automation_approvals[record.policy.id]?.[nextCycle],
          last_opened_slot: record.last_opened_slot,
          cycles: Object.values(record.cycles).map((cycle) => clone(cycle)),
        };
      }),
      approvals: clone(state.automation_approvals),
    },
    market: {
      enabled: state.network?.test_credit_enabled === true,
      credit_symbol: state.network?.credit_symbol ?? null,
      slot: state.slot,
      issued_supply_units: state.test_credit_supply_units,
      accounted_supply_units: accountedSupplyUnits,
      balances: clone(state.test_balances),
      compute_escrows: clone(state.compute_escrows),
      provider_collateral: clone(state.provider_collateral),
      bid_commits: Object.values(state.bid_commits).reduce(
        (sum, rows) => sum + Object.keys(rows).length,
        0,
      ),
      bid_reveals: Object.values(state.bid_reveals).reduce(
        (sum, rows) => sum + Object.keys(rows).length,
        0,
      ),
      auctions: auctionRows,
      meter_receipts: Object.keys(state.meter_receipts).length,
      stage_payments: clone(state.stage_payments),
      compute_reward_distributions: clone(state.compute_reward_distributions),
      expired_launches: clone(state.expired_launches),
      conservation_valid:
        state.network?.test_credit_enabled !== true ||
        state.test_credit_supply_units === accountedSupplyUnits,
    },
    valid: true,
  };
}
