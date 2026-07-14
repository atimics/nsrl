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
  -> launch_published
  -> bounty_funded
  -> stage_submitted
  -> validation_attested
  -> challenge_opened / challenge_resolved
  -> stage_accepted
  -> candidate_submitted
  -> validation_attested
  -> model_published
```

The reducer enforces these role and settlement rules:

- the recipe proposer alone publishes the immutable open recipe;
- the declared sponsor alone funds the exact bounty amount;
- the declared compute account alone submits each bounded stage once;
- validators cannot also be the proposer, builder, compute provider, sponsor,
  or treasury for that launch;
- stage acceptance requires a clean validator quorum and no open or upheld
  challenge;
- candidate publication requires the configured clean quorum and at least the
  configured number of full replays;
- accepted stages and published candidates are final and cannot receive new
  attestations or challenges;
- model publication deterministically binds the artifact, proof, metric vector,
  bounty payout/refund rows, and capped model-local reward allocation.

The checked fixture uses a two-validator stage quorum and a three-validator
candidate quorum with one full replay. Challenge outcomes are resolved by the
localnet authority. That is an explicit centralization boundary, not consensus.

## Run it

Create a new localnet and self-register accounts:

```sh
node scripts/nsrl-model-localnet.mjs init \
  --dir /tmp/nsrl-localnet \
  --authority nsrl:authority:localnet

node scripts/nsrl-model-localnet.mjs account \
  --dir /tmp/nsrl-localnet \
  --account nsrl:sponsor:example

node scripts/nsrl-model-localnet.mjs status --dir /tmp/nsrl-localnet
```

The command prints the generated identity path. Identity files include private
keys, are written with mode `0600`, and must not be committed. The CLI also
provides commands for publishing a recipe, funding a bounty, submitting and
attesting stages, opening/resolving challenges, accepting stages, submitting a
candidate, and publishing the model; run it with `--help` for the complete
argument list.

Rebuild and validate the deterministic public transcript used by Forge:

```sh
node scripts/check-model-localnet-v1.mjs
node scripts/build-model-localnet-site.mjs
node scripts/build-model-localnet-site.mjs --check
```

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
- escrow and balances are deterministic accounting rows, not custodied assets;
- there are no deadlines, slashing, provider auctions, stage pricing, or
  external payment adapters; and
- artifact hashes do not guarantee replicated availability.

The next useful milestone is a sealed provider auction plus a test escrow
adapter, followed by multi-process networking and an explicit validator
selection/finality design. Transferability should remain out of scope until
those mechanisms have survived economic and security review.
