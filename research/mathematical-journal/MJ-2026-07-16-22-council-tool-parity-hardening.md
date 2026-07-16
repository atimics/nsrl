# MJ-2026-07-16-22: Council tool-parity hardening

- Date: 2026-07-16
- Status: historical Council-v0 promotion claim falsified under the stronger
  Council-v1 contract; exact v0 replay and integrity remain valid; Council is
  shadow-only
- Contract:
  [`hardening-contract.json`](../../benchmarks/solomon-council-v1/hardening-contract.json)
- Result:
  [`hardening-result.json`](../../benchmarks/solomon-council-v1/hardening-result.json)
- Auditor:
  [`audit-solomon-council-hardening-v1.mjs`](../../scripts/audit-solomon-council-hardening-v1.mjs)
- Independent checker:
  [`check-solomon-council-hardening-v1.mjs`](../../scripts/check-solomon-council-hardening-v1.mjs)

## Question

Does the 576-case Council-v0 ceremony still demonstrate strict Council
improvement when the same underlying model acting alone receives equivalent
public evidence, tool observations, permissions, and budget? Does that ceremony
also cover stale evidence, tool failures, permission denials, human-authored
ambiguity, production outcomes and revisions, and longer unfamiliar-source and
cross-modal transfer?

## Frozen boundary

This is a retrospective falsification audit. The Council-v0 lanes and gold were
already open when the stronger requirements were frozen. The audit may revoke
effective promotion, but it may not rewrite the historical result or claim a
prospective Council-v1 pass.

The contract requires:

- a same-model, same-evidence solo baseline with equivalent tools,
  permissions, and resource budgets;
- strict Council improvement over that baseline;
- at least 72 misleading, incomplete, stale, and conflicting evidence cases;
- at least 24 actual tool failures and 24 permission denials;
- at least 72 human-authored ambiguous decisions, with at least 36 appropriate
  `ask_user` or abstention decisions;
- at least 72 observed production outcomes and calibration revisions from at
  least three observation sources;
- at least 144 unfamiliar-source and 144 cross-modal cases with broader source
  families;
- exact replay, generation integrity, provenance, no forbidden assistance,
  and shadow-only receipts.

## Tool-parity result

The historical solo lane contains zero tool observations. The Council lane
contains 2,880: five deterministic public-evidence observations for each of 576
cases. Both lanes are bound to the same native successor-v2 model and the same
casebook, but they do not have equivalent access to the information-producing
verifier.

For diagnosis, the auditor applies the already-public deterministic verifier to
each solo case. This reconstructed parity baseline ties the Council exactly on
all eight dimensions:

| Dimension | Tool-parity solo | Council | Strict Council improvement |
|---|---:|---:|---:|
| Source-grounded correctness | 1000‰ | 1000‰ | no |
| Calibration | 990‰ | 990‰ | no |
| Hard-negative rejection | 1000‰ | 1000‰ | no |
| Contradiction detection | 1000‰ | 1000‰ | no |
| Decision regret | 0 mean milli-regret | 0 mean milli-regret | no |
| Appropriate abstention | 1000‰ | 1000‰ | no |
| Cross-modal agreement | 1000‰ | 1000‰ | no |
| Unfamiliar-source transfer | 1000‰ | 1000‰ | no |

Because this reconstruction occurs after the v0 gold opening, it is diagnostic
only. It cannot authorize a new baseline or Council. It does establish that the
published v0 advantage is not identified separately from unequal tool access.

## Missing hardening evidence

The historical ceremony contains 72 misleading, 72 incomplete, and 72
conflicting cases, but zero stale-evidence cases. It contains 2,880 successful
faculty tool observations, but zero tool failures and zero permission denials.
All 576 cases use deterministic public verifiers; none is a human-authored
ambiguous decision.

All 576 production receipts remain outcome-pending and unrevised. The one exact
outcome/revision fixture continues to replay, but it does not count as
production outcome evidence. Unfamiliar-source and cross-modal coverage remain
72 cases each, across three and one source families respectively, below the
frozen longer-transfer requirements.

## Decision

**Decision 22.1.** The Council-v0 strict-outperformance claim is falsified under
the Council-v1 hardening contract. The v0 ceremony remains a valid historical
structured comparison with green exact replay, generation integrity,
provenance, seal/circle enforcement, dissent preservation, and shadow-mode
receipts. Those properties do not demonstrate incremental judgment value over
an equivalent-tool solo baseline.

Effective Council promotion is revoked. Operational action execution and
product release remain unauthorized. A future promotion claim requires
prospectively generated tool-parity lanes and unopened gold over the complete
hardening surface. Until then, Council remains shadow-only while product-facing
multimodal work proceeds independently.
