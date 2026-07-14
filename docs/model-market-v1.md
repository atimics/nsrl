# Sealed compute market v1

The Forge compute market prices immutable model-recipe stages without giving a
sponsor or provider any control over promotion. It is an additive state machine
inside `model-localnet-v1`, backed only by conserved, non-transferable
`FORGE-TEST` units.

## Contract

The launch recipe freezes stage IDs, input dependencies, commands, and maximum
compute units. A sponsor then funds one compute escrow and signs strictly
increasing logical deadlines:

```text
current slot < bid deadline < reveal deadline < execution deadline
```

Each registered provider deposits the minimum collateral and commits:

```text
sha256(canonical_json({
  schema: "nsrl.provider_bid_reveal.v1",
  launch_id,
  stage_id,
  provider,
  unit_price_units,
  max_compute_units,
  nonce
}))
```

The full values become public only in the reveal window. Auction close filters
out bids that cannot cover the frozen stage ceiling, compute escrow, or minimum
collateral. It selects the lowest integer unit price and breaks a tie by reveal
event ID. The reducer recomputes that result; an authority cannot choose another
winner.

## Execution and settlement

The winning key alone may submit the assigned stage before the execution
deadline. Its signed meter must exactly match the stage event's input, output,
evidence, and compute-unit fields. Validators remain role-separated from every
assigned provider.

After a clean stage quorum, payment is:

```text
accepted compute units * winning unit price
```

The maximum stage cost was reserved at assignment. Settlement transfers only
the actual accepted cost, releases that assignment's collateral, and leaves the
remainder in sponsor escrow. The compute budget closes only when every frozen
stage is paid, returning all unused units to the sponsor.

On model publication, the recipe's capped model-local compute allocation is
redistributed to paid providers in proportion to accepted compute units. Exact
largest-remainder arithmetic assigns every unit deterministically.

If the logical execution deadline passes first, `launch_expired` refunds the
remaining compute escrow and funded bounty escrow, transfers reserved
collateral on unfinished assignments to the sponsor, and permanently closes the
launch.

## Checked specimen

The deterministic Forge fixture contains 76 Ed25519-signed, hash-linked events:

- 9 bid commitments and 9 matching reveals across 3 stages;
- 3 deterministic assignments, meter receipts, acceptances, and payments;
- 12,000 compute units funded, 7,168 paid, and 4,832 refunded;
- 146,000 test-credit units conserved exactly; and
- 22,400 `ITP1` compute-reward units distributed to the actual providers.

Run its positive and adversarial checks with:

```sh
node scripts/check-model-market-v1.mjs
node scripts/build-model-market-site.mjs --check
```

The adversarial suite rejects invalid reveals, late bids, fabricated winners,
non-winner submissions, mismatched meters, premature payment, and evidence after
expiry. It also checks the full refund/slashing branch and confirms that the
public snapshot contains no private keys.

## Boundary

This is a deterministic economic simulation, not a currency, wallet, smart
contract, custody system, or promise of financial value. Logical slots are
authority-signed test events, the ledger has one writer, and challenge
resolution is centralized. A networked pilot still needs durable ordering,
validator selection and rotation, content-addressed artifact availability,
wall-clock finality, abuse controls, and independent security/economic review.
