# Signed model localnet v1

`model-localnet-v1` is a single-writer test network for replaying a complete
model launch with public-key identities and simulated, non-transferable
credits. It turns the launch recipe into an executable coordination state
machine without claiming blockchain consensus, custody, or financial value.

The machine-readable public-object schema is
[`protocol/model-localnet-v1.schema.json`](../protocol/model-localnet-v1.schema.json).
The reducer and ledger are implemented in
[`scripts/lib/model-localnet-v1.mjs`](../scripts/lib/model-localnet-v1.mjs).

## Signed event envelope

Every actor owns an Ed25519 keypair and signs a canonical JSON intent containing
the event schema, event type, actor account, public key, and event payload. The
ledger derives two hashes:

- `event_id`: SHA-256 of the canonical signed-intent body. The reducer accepts
  each event ID once, so resubmission is idempotent rather than rewarding the
  same evidence twice.
- `event_sha256`: SHA-256 of the event height, predecessor hash, event ID, and
  full signed intent. This commits ordering, signature bytes, and the entire
  history prefix.

Reopening the JSONL ledger verifies every signature, account-to-key binding,
event ID, height, predecessor link, event hash, and state transition from
genesis. A mutation or reorder invalidates the replay.

## State machine

The accepted event sequence is:

```text
network_initialized
  -> account_registered
  -> test_credit_issued
  -> launch_published
  -> bounty_funded
  -> compute_budget_funded
  -> provider_collateral_deposited
  -> provider_bid_committed
  -> slot_advanced
  -> provider_bid_revealed
  -> stage_auction_closed
  -> stage_submitted
  -> compute_metered
  -> validation_attested
  -> challenge_opened / challenge_resolved
  -> stage_accepted
  -> stage_payment_settled
  -> compute_budget_refunded
  -> provider_collateral_withdrawn
  -> candidate_submitted
  -> validation_attested
  -> model_published
  -> compute_reward_distributed
  -> bounty_automation_policy_registered
  -> bounty_automation_cycle_opened
  -> launch_published / bounty_funded / compute_budget_funded
```

An incomplete launch can instead transition to `launch_expired` after its
logical execution deadline. Expiry refunds unused compute and bounty escrow,
slashes collateral reserved by unfinished assignments, and permanently blocks
new candidate or stage evidence for that launch.

The reducer enforces these role and settlement rules:

- the recipe proposer alone publishes the immutable open recipe;
- the declared sponsor alone funds the exact bounty amount;
- compute providers commit a SHA-256 of their bid before revealing its price,
  compute ceiling, and nonce in a later logical-slot window;
- the authority closes each auction to the eligible lowest price, breaking ties
  by reveal event ID and reserving both the maximum payment and collateral;
- only the assigned provider can submit and meter a stage; meter hashes and
  units must exactly match the stage evidence;
- validators cannot also be the proposer, builder, compute provider, sponsor,
  or treasury for that launch;
- stage acceptance requires a clean validator quorum and no open or upheld
  challenge;
- candidate publication requires the configured clean quorum and at least the
  configured number of full replays;
- accepted stages and published candidates are final and cannot receive new
  attestations or challenges;
- model publication deterministically binds the artifact, proof, metric vector,
  bounty payout/refund rows, and capped model-local reward allocation;
- accepted metered work pays exactly `compute_units * winning_unit_price`, and
  the remainder of the compute escrow returns to the sponsor;
- the model-local compute allocation is distributed among actual paid providers
  in proportion to their accepted units using exact largest-remainder arithmetic;
  and
- an automation keeper may open only the deterministic successor of a promoted
  source under its sponsor-signed budget, cooldown, approval, and cycle limits;
  the full spend is reserved once and interrupted linked funding resumes from
  the ledger without duplication.

The checked fixture uses a two-validator stage quorum and a three-validator
candidate quorum with one full replay. Challenge outcomes are resolved by the
localnet authority. That is an explicit centralization boundary, not consensus.

## Run it

Create a new localnet and self-register accounts:

```sh
node scripts/nsrl-model-localnet.mjs init \
  --dir /tmp/nsrl-localnet \
  --authority nsrl:authority:localnet \
  --test-credit-symbol FORGE-TEST

node scripts/nsrl-model-localnet.mjs account \
  --dir /tmp/nsrl-localnet \
  --account nsrl:sponsor:example

node scripts/nsrl-model-localnet.mjs status --dir /tmp/nsrl-localnet
```

The command prints the generated identity path. Identity files include private
keys, are written with mode `0600`, and must not be committed. The CLI also
provides commands for issuing test credit, publishing a recipe, funding bounty
and compute escrows, depositing collateral, committing/revealing bids, advancing
logical slots, closing auctions, submitting/metering/attesting/accepting/paying
stages, refunding or expiring launches, publishing candidates, and distributing
the compute reward; run it with `--help` for the complete argument list.

Rebuild and validate the deterministic public transcript used by Forge:

```sh
node scripts/check-model-localnet-v1.mjs
node scripts/build-model-localnet-site.mjs
node scripts/build-model-localnet-site.mjs --check
node scripts/check-model-market-v1.mjs
node scripts/build-model-market-site.mjs --check
node scripts/check-bounty-automation-v1.mjs
node scripts/build-bounty-automation-site.mjs --check
```

`scripts/nsrl-bounty-keeper.mjs` registers and inspects signed policies, plans
or executes eligible cycles, and exposes sponsor `approve`, `pause`, and
`resume` controls. The full policy and restart semantics are documented in
[`bounty-automation-v1.md`](bounty-automation-v1.md).

The public fixture contains only public keys, signatures, events, and reduced
state. Its keypairs are deterministic test identities and must never be reused
for authority over external systems.

## Security boundary

V1 is deliberately narrow:

- one process appends events; there is no concurrent-writer lock, gossip,
  consensus, fork choice, or finality across machines;
- account registration is permissionless and has no Sybil resistance, key
  rotation, delegation, or recovery;
- the authority configures quorums, accepts stages, and resolves challenges;
- escrow, balances, deadlines, slashing, bids, meters, and stage payments are
  deterministic test accounting, not custodied assets or an external rail;
- artifact hashes do not guarantee replicated availability.

The next useful milestone is multi-process networking, a durable event index,
replicated artifact availability, and an explicit validator-selection/finality
design. Transferability should remain out of scope until those mechanisms have
survived economic, legal, and security review.
