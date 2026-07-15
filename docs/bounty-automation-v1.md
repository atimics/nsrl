# Bounty automation v1

`bounty-automation-v1` turns a promoted model into an optional trigger for the
next bounded metric bounty. A sponsor signs one immutable policy; a keeper may
execute only the exact successor recipe and funding envelope derived from that
policy and accepted publication evidence.

The policy schema is
[`protocol/bounty-automation-policy-v1.schema.json`](../protocol/bounty-automation-policy-v1.schema.json).
The deterministic specimen is
[`protocol/examples/integer-transformer-bounty-automation-v1.json`](../protocol/examples/integer-transformer-bounty-automation-v1.json).

## Signed policy

The policy fixes:

- sponsor, proposer, keeper, and initial promoted launch;
- metric, direction, and relative improvement in basis points;
- bounty and compute units per cycle;
- lifetime spend and separate-approval threshold;
- maximum active bounties, maximum cycles, and logical-slot cooldown; and
- sealed-auction windows and minimum provider collateral.

The keeper cannot choose a metric, target, recipe, recipient, or funding
amount. It clones the frozen source recipe, points lineage at the accepted
model hash, derives the next target with integer arithmetic, clears publication
evidence, and commits the resulting canonical recipe hash.

## Decision and event flow

One keeper tick evaluates these gates in order:

```text
policy exists and is active
  -> prior source is promoted
  -> no interrupted cycle needs resuming
  -> cycle and active-bounty caps have room
  -> cooldown has elapsed
  -> lifetime spend cap holds
  -> separate sponsor approval exists when required
  -> sponsor test-credit balance covers the whole cycle
  -> reserve the complete cycle once
  -> publish the committed successor recipe
  -> fund its bounty and compute escrows from the reserve
```

The signed events are:

- `bounty_automation_policy_registered`;
- `bounty_automation_policy_paused` / `bounty_automation_policy_resumed`;
- `bounty_automation_cycle_approved` for a cycle above the automatic threshold;
- `bounty_automation_cycle_opened` by the declared keeper;
- `launch_published` by the declared proposer; and
- `bounty_funded` and `compute_budget_funded` by the sponsor.

Opening a cycle debits and reserves its complete bounded spend. The reserve is
included in test-credit conservation until the linked bounty and compute
escrows consume it. If a process stops after any event boundary, the next tick
reconstructs the same recipe and appends only the missing events. It cannot
open a second cycle or debit the sponsor twice. A pause blocks new and resumed
ticks until the sponsor signs a resume event.

## Keeper CLI

The keeper uses the same JSONL ledger and local Ed25519 identity files as the
model localnet:

```sh
node scripts/nsrl-bounty-keeper.mjs register \
  --dir /tmp/nsrl-localnet \
  --policy protocol/examples/integer-transformer-bounty-automation-v1.json \
  --sponsor-key /path/to/sponsor.identity.json

node scripts/nsrl-bounty-keeper.mjs plan \
  --dir /tmp/nsrl-localnet \
  --policy-id integer-transformer-frontier-auto

node scripts/nsrl-bounty-keeper.mjs tick \
  --dir /tmp/nsrl-localnet \
  --policy-id integer-transformer-frontier-auto \
  --keeper-key /path/to/keeper.identity.json \
  --proposer-key /path/to/proposer.identity.json \
  --sponsor-key /path/to/sponsor.identity.json
```

`approve`, `pause`, `resume`, and `status` provide the sponsor control path.
`plan` is read-only and reports explicit waiting reasons such as `cooldown`,
`active_limit_reached`, `manual_approval_required`, or `insufficient_balance`.

Validate and rebuild the 84-event public transcript with:

```sh
node scripts/check-bounty-automation-v1.mjs
node scripts/build-bounty-automation-site.mjs
node scripts/build-bounty-automation-site.mjs --check
```

The checker covers wrong signers, policy and recipe tampering, early cooldown,
pause, approval enforcement, auction-term tampering, supply conservation, and
restart recovery after each funding boundary.

## Security boundary

This is an auditable local keeper over non-transferable test credit. It is not
a scheduler service, wallet, custodian, smart contract, or permissionless
consensus system. The fixture keeps role keys together only to demonstrate the
complete flow; a networked deployment must isolate key custody, authenticate
the proposer and sponsor signing path, persist indexed events, and define a
recovery action for a reserved cycle whose auction window expires before
funding. Evaluators remain the sole promotion authority.
