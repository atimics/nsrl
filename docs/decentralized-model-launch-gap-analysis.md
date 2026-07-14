# Decentralized model launch gap analysis

## Vision

Anyone can sponsor a measurable model outcome. Independent builders can publish
an immutable training recipe, obtain compute, submit a candidate, and receive
objective settlement. Validated model publications append proof blocks and
allocate capped model-local rewards to the people and machines that produced,
checked, and funded the artifact.

NSRL has an unusually strong base for this vision because its arithmetic,
datasets, checkpoints, evaluation rows, and promotion decisions are designed
for deterministic replay. The missing pieces are primarily coordination,
identity, settlement, and adversarial network operation.

## Capability map

| Capability | Existing NSRL evidence | MVP added here | Production gap |
| --- | --- | --- | --- |
| Frozen metric contract | `integer-transformer-proof-v1`, fixed baselines, dataset hash, typed results | Recipe binds evaluator, dataset, target, and guardrails | General metric registry and private/commit-reveal evaluation sets |
| Deterministic model identity | Model hashes, artifact SHA-256s, replay hashes, frozen candidate manifests | Publication receipt binds model, artifact, recipe, proof, and metrics; Ed25519 actors sign every intent | Multi-writer content-addressed registry, key lifecycle, and artifact availability guarantees |
| Model recipe | Training commands and manifests exist across scripts and docs | `nsrl.model_launch_recipe.v1` JSON Schema and checked specimen | Recipe compiler, migrations, compatibility policy, and secret-free container lock |
| Metric bounty | Promotion thresholds exist but have no sponsor object | Signed sponsor funding, integer progress curve, promotion bonus, deterministic payout/refund rows, guardrails | Custodied escrow, deadlines, stacked claims, disputes, and audited settlement |
| Compute contract | AWS stage plans record runner and artifact provenance | Recipe records bounded stages; declared provider signs input/output/evidence hashes and compute units | Provider discovery, sealed bids, pricing, collateral, metering, redundancy, and stage payment |
| Model publication | Promoted artifacts are frozen in repository JSON | Authority publishes after a clean replay quorum into a signed hash-linked event log | Multi-writer registry, distributed ordering/finality, availability, and revocation policy |
| Block-style rewards | No protocol reward accounting | Capped model-local asset, exact role allocation, append-only ledger, replay protection | Multi-node consensus/finality, recovery, long-horizon economic simulation, and issuance governance |
| Independent validation | Checkers can replay deterministic artifacts | Independent keys sign two-validator stage and three-validator candidate quorums; one full replay; challenge flow | Validator selection, timed windows, distributed resolution, slashing, and sampling policy |
| Sponsor and contributor identity | Repository and AWS provenance only | Namespaced accounts bind Ed25519 keys to every event | Key rotation/recovery, delegation, allowlists or stake, Sybil resistance, and organization policy |
| Decentralized storage | S3 artifact manifests and local frozen files | Content hashes are settlement inputs | Replicated content-addressed storage, retention incentives, privacy, and deletion policy |
| Inference revenue | Browser inference exists | Credits reserve inference quota conceptually | Metered service receipts, operator market, pricing, revenue routing, and abuse controls |
| Public product surface | Solomon sampler and results dashboard | Forge launch site, bounty composer, reward simulator, recipe download, signed transcript, and gap map | Wallet/identity UI, durable API/indexer, accessibility audit, and operational support |
| Governance | Promotion boundary is documented | Active recipes cannot be changed; token voting cannot promote | Contract-version governance, emergency policy, treasury controls, and transparent upgrades |

## Readiness assessment

### Ready to reuse

- deterministic evaluation and exact pass/fail semantics;
- model, dataset, corpus, trace, and artifact hashing;
- resumable training evidence and replay checks;
- promotion manifests and health guardrails; and
- CPU/WASM execution suitable for independent verification.

### Implemented as a local protocol prototype

- versioned launch recipe and JSON Schema;
- sponsor bounty with metric direction, baseline, target, and guardrails;
- deterministic bounty payout arithmetic;
- model-local capped reward schedule;
- role-based reward allocation with exact integer conservation;
- hash-linked model publication receipt;
- Ed25519 account registration and signed event intents;
- append-only JSONL ledger with event IDs, predecessor hashes, and full replay;
- role-conflict checks, stage and candidate validator quorums, one full replay,
  challenge resolution, and finality rules;
- deterministic sponsor settlement and model-local balances;
- positive, quorum, conflict, replay, reorder, signature-tamper, and hash-tamper
  validation; and
- interactive static website using real promoted evidence and the 31-event
  public localnet transcript.

### Not yet implemented

- money, custody, deadline-triggered refund execution, or smart-contract escrow;
- transferable tokens or a wallet;
- compute-provider bids, prices, collateral, or stage-payment settlement;
- multi-process networking, shared ordering, consensus, or fork recovery;
- decentralized validator selection, timed challenge windows, or slashing;
- key rotation, delegation, recovery, or Sybil resistance;
- decentralized artifact storage and availability proofs; and
- production service revenue.

## Highest-risk gaps

1. **Verification cost.** Exact replay is credible but can cost almost as much
   as the original run. The protocol needs a policy for redundant execution,
   sampled checkpoint challenges, and full replay of promotion candidates.
2. **Metric capture.** Public evaluation data can become a training target.
   Bounties need hidden sets, commit-reveal rotation, or multiple independent
   evaluation owners without weakening deterministic settlement.
3. **Useful-work definition.** Rewards must require a new accepted artifact or
   reproduction. Paying raw CPU time creates incentives to waste compute.
4. **Identity and Sybil resistance.** The localnet binds keys and rejects role
   conflicts, but permissionless self-registration does not establish that
   validator keys belong to independent people or infrastructure.
5. **Artifact availability.** A hash proves identity but not that other parties
   can retrieve the bytes throughout the challenge and service periods.
6. **Economic and legal design.** Transferability, passive revenue claims, and
   external custody materially change the system. V1 avoids these features.

## Recommended build order

### P0: signed localnet — implemented

1. Add Ed25519 identities and signatures for recipes, stage receipts, proofs,
   and publication blocks.
2. Add an append-only local registry with unique event IDs and replay
   protection.
3. Add accepted stage receipts bound to runner, inputs, outputs, evidence, and
   recipe compute ceiling. Provider offers, duration, and price move to P1.
4. Implement a three-validator quorum with one full replay and two artifact
   checks for the current proof contract.
5. Add an explicit challenge/resolution flow and invalid/failing outcomes.
6. Exercise the entire lifecycle with simulated credits and adversarial tests.

### P1: provider auction + test settlement

1. Add sealed compute-provider bids, deterministic assignment, collateral, and
   signed metering receipts.
2. Add a test-only escrow adapter, sponsor deadlines, and refund paths.
3. Add a content-addressed artifact mirror and availability checks.
4. Move the single-writer reducer behind a durable API/indexer and exercise
   concurrent submissions without silently creating forks.
5. Simulate emissions and adversarial provider/validator behavior across many
   launches before enabling transferability.
6. Commission security and economic reviews of the reward and challenge rules.

### P2: narrow production pilot

1. Launch one capped bounty on an existing frozen NSRL contract.
2. Use a small allowlisted compute and validator set while preserving public
   artifacts and challenges.
3. Settle in a conventional payment rail first.
4. Introduce on-chain components only where they remove a demonstrated trust
   bottleneck.

## Decision

The signed, simulated-credit localnet is now implemented. The next
implementation should be a provider auction plus test escrow adapter—not a
tradable global token. This tests whether compute can be priced and settled
without weakening objective promotion before adding financial and consensus
risk.
