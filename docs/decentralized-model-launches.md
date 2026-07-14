# Decentralized model launches

`model-launch-v1` is an experimental coordination protocol for funding,
executing, verifying, and publishing deterministic NSRL model runs. It treats a
model run as a finite public project and a promoted model as a durable,
content-addressed result.

The protocol is subordinate to NSRL's evaluation contracts. Capital can choose
which frozen objective to fund, but capital cannot change whether a candidate
passed that objective.

## Protocol objects

### Model launch recipe

A recipe binds the facts that must not change after funding begins:

- proposer and bonded participant accounts;
- model family, artifact format, architecture, tokenizer, and parent model;
- source commit and evaluator hash;
- dataset and held-out evaluation contract;
- ordered run stages and maximum compute budget;
- metric bounties, guardrails, and payout curves;
- promotion checker and accepted proof hash;
- model-local reward asset, supply cap, emission schedule, and allocation; and
- the published artifact, model hash, proof, and metrics.

The machine-readable contract is
[`protocol/model-launch-v1.schema.json`](../protocol/model-launch-v1.schema.json).
The first specimen wraps the already-promoted integer Transformer proof rather
than inventing an unverified model result.

### Signed localnet event

The localnet wraps each coordination action in an Ed25519-signed canonical
intent and then appends it to a replay-protected, hash-linked JSONL ledger. Its
events cover account registration, test-credit issuance, recipe publication,
bounty and compute funding, sealed provider auctions, collateral, signed compute
meters, validator attestations, challenges, stage payment, expiry, candidate
submission, model publication, and compute-reward distribution.

[`protocol/model-localnet-v1.schema.json`](../protocol/model-localnet-v1.schema.json)
defines the public envelopes. [`model-localnet-v1.md`](model-localnet-v1.md)
documents the state machine, CLI, invariants, and threat boundary.

### Metric bounty

A metric bounty is demand for a measurable result. Sponsors escrow a fixed
quantity of credits against a metric, direction, baseline, target, guardrails,
and promotion contract. Several sponsors can attach independent bounties to the
same model lineage.

The v1 progress payout is linear between baseline and target. A portion of the
escrow can be reserved for the full promotion result:

```text
progress = clamp((candidate - baseline) / (target - baseline), 0, 1)
payout = progress_pool * progress + promotion_bonus_if_passed
```

For minimized metrics the signs reverse. All arithmetic is integer arithmetic.
A metric improvement does not receive the promotion bonus unless every frozen
guardrail and the promotion checker pass.

### Run budget

The run budget pays actual accepted compute independently of model success.
Honest negative results are useful evidence and do not slash a compute provider.
Bonds apply to forged, missing, duplicated, or contract-incompatible evidence.

The local market binds those units to sealed provider offers, deterministic
prices, runner identity, signed input/output/evidence hashes, accepted stage
receipts, and test-credit payment. A production contract must additionally bind
hardware attestations or auditable meter semantics and durable artifact roots.

### Sealed provider auction

Each provider deposits conserved test collateral, then signs a commitment to a
stage ID, unit price, maximum compute, and private nonce. After the bid deadline,
providers reveal the bid. After the reveal deadline, the reducer selects the
lowest eligible unit price, using the reveal event ID as the deterministic tie
break, and reserves the stage ceiling plus minimum collateral.

Only the assigned key may submit and meter that stage. Settlement requires a
clean validator quorum and pays actual accepted units rather than the ceiling.
Unused compute budget returns to the sponsor. If the execution deadline passes,
expiry refunds open escrows and slashes collateral still reserved against
unfinished assignments.

### Model publication

Publishing a model creates a receipt bound to:

- the canonical recipe SHA-256;
- model and artifact hashes;
- promotion evidence;
- the observed metric vector; and
- a hash-linked reward block.

`scripts/publish-model-launch.mjs` produces the deterministic receipt.
`scripts/check-model-launch-v1.mjs` reopens the recipe, model artifact, proof,
promotion freeze, reward calculation, and committed publication receipt.

### Proof-of-useful-compute rewards

The reward ledger borrows the useful properties of block rewards without
rewarding arbitrary hash expenditure. Each accepted state transition can append
one reward block:

```text
launch_published
stage_accepted
candidate_valid
independent_replay
model_promoted
```

The block commits to its predecessor, recipe, event, evidence, model, metrics,
and exact allocation. Its deterministic subsidy is:

```text
era = floor(block_height / halving_interval)
subsidy = max(initial_reward >> era, minimum_reward)
event_reward = subsidy * event_multiplier_bps / 10_000
minted = min(event_reward, max_supply - cumulative_supply)
```

Integer largest-remainder allocation divides the mint among builders, compute
providers, validators, sponsors, and the public-goods treasury. The allocation
always sums exactly to the minted units.

V1 creates a separate reward asset for each model lineage. The specimen asset
is `ITP1`, capped at 1,000,000 indivisible units. It is explicitly
non-transferable and can only represent prototype participation, future-run
sponsorship, or inference quota. It is not a global NSRL currency and does not
promise passive profit.

Metric bounties and reward emission remain separate ledgers:

- bounty credits come from sponsor escrow and pay for outcomes;
- reward credits come from a capped model schedule and recognize accepted work.

This prevents emission from silently spending a sponsor's bounty and makes both
sources of value auditable.

## Settlement lifecycle

```text
recipe proposed
  -> sponsor signs simulated escrow funding
  -> providers commit and reveal sealed stage bids
  -> deterministic assignment reserves payment + collateral
  -> assigned provider signs stage evidence + meter
  -> independent validators attest
  -> challenge resolved
  -> authority accepts clean stage quorum
  -> accepted provider payment + unused compute refund
  -> candidate submitted
  -> three-validator candidate quorum + full replay
  -> bounty settlement
  -> model publication + reward block
  -> compute pool allocated to actual providers
```

Every step must be idempotent. Re-submitting the same evidence cannot create a
second stage payment, bounty payout, or reward block.

## Invariants

- An active recipe is immutable. Changes create a new launch ID and hash.
- Token holders cannot vote a model into promotion.
- Governance may create a new contract version but cannot rewrite a funded one.
- A bounty always includes non-regression guardrails.
- A publication is invalid when its artifact, model, recipe, evaluator, dataset,
  or proof hashes disagree.
- A reward event requires new accepted evidence, not merely consumed compute.
- Total model-local emission cannot exceed the recipe's supply cap.
- Failed research can be paid for valid work without receiving a promotion
  reward.
- A validator cannot also be the launch proposer, builder, compute provider,
  sponsor, or treasury.
- Finalized stage evidence and published candidates cannot be reopened.
- Test-credit supply is conserved across balances, escrows, and collateral.
- A revealed bid must open its prior commitment and may not arrive outside its
  signed logical-slot window.
- Stage payment cannot exceed reserved escrow and requires the assigned
  provider's matching signed meter receipt.

## Current boundary

This repository implements a signed, single-process localnet specimen, not a
live financial or blockchain system. It has Ed25519 identities, a hash-linked
append-only event log, replay protection, clean validator quorums, an explicit
challenge flow, and simulated bounty/reward settlement. It has no wallet,
custody, transferable asset, external escrow, multi-writer consensus, Sybil
resistance, or deployed smart contract. Its provider auction and escrow adapter
operate only over deterministic test credit. Production financial
claims would require a separate threat model, jurisdiction-specific legal
review, and audited settlement implementation.
