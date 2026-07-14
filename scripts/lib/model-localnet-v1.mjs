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
  "launch_published",
  "bounty_funded",
  "stage_submitted",
  "validation_attested",
  "challenge_opened",
  "challenge_resolved",
  "stage_accepted",
  "candidate_submitted",
  "model_published",
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
    event_ids: {},
    height: 0,
    head_sha256: ZERO_HASH,
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

function conflictAccounts(recipe) {
  return new Set([
    recipe.proposer.account,
    recipe.participants.builder,
    recipe.participants.compute,
    recipe.participants.sponsor,
    recipe.participants.treasury,
  ]);
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
      state.launches[recipe.launch.id] = {
        id: recipe.launch.id,
        recipe,
        recipe_sha256: recipeSha256,
        published_event_id: eventId,
        funded_bounties: {},
        status: "open",
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
      launch.funded_bounties[bounty.id] = {
        sponsor: intent.actor,
        escrow_units: bounty.escrow_units,
        funded_event_id: eventId,
      };
      break;
    }
    case "stage_submitted": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
      if (intent.actor !== launch.recipe.participants.compute) {
        fail("stage submission must be signed by the declared compute provider");
      }
      for (const bounty of launch.recipe.bounties) {
        if (!launch.funded_bounties[bounty.id]) {
          fail(`bounty ${bounty.id} must be funded before compute begins`);
        }
      }
      const stage = launch.recipe.run.stages.find((row) => row.id === payload.stage_id);
      if (!stage) {
        fail(`unknown stage ${payload.stage_id}`);
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
    case "validation_attested": {
      requireRegisteredActor(state, intent);
      const subject = requireSubject(state, payload.subject_type, payload.subject_event_id);
      const launch = requireLaunch(state, subject.launch_id);
      if (conflictAccounts(launch.recipe).has(intent.actor)) {
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
      if (state.invalid_subjects[subjectId] || openChallenges(state, subjectId).length > 0) {
        fail("challenged or invalid stage evidence cannot be accepted");
      }
      const counts = attestationCounts(state, subjectId);
      if (counts.invalid > 0 || counts.valid < state.network.stage_quorum) {
        fail("stage submission has not reached a clean validator quorum");
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
    case "candidate_submitted": {
      requireRegisteredActor(state, intent);
      const launch = requireLaunch(state, payload.launch_id);
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
        if (!launch.funded_bounties[bounty.id]) {
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
        candidate_event_id: candidateId,
        model_hash: candidate.model_hash,
        artifact_sha256: candidate.artifact_sha256,
        receipt: clone(payload.receipt),
        validator_counts: counts,
      };
      launch.status = "promoted";
      launch.published_recipe_sha256 = payload.receipt.recipe_sha256;
      break;
    }
    default:
      fail(`unhandled event type ${intent.event_type}`);
  }
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
    return this.append(
      signLocalnetIntent(authorityIdentity, "network_initialized", {
        network_id: config.network_id ?? "nsrl-model-localnet-v1",
        authority_account: authorityIdentity.account,
        authority_public_key: authorityIdentity.public_key,
        stage_quorum: config.stage_quorum ?? 2,
        candidate_quorum: config.candidate_quorum ?? 3,
        full_replay_required: config.full_replay_required ?? 1,
      }),
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
  }));
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
    valid: true,
  };
}
