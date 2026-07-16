# Solomon Council v0

Solomon Council v0 is an auditable shadow-mode judgment protocol. It accepts
bounded recommendations from five expert faculties, routes them through a
sixth deterministic judge, and emits a hash-bound wisdom receipt. It can
recommend an action, request evidence, ask the user for missing information,
or abstain. It cannot execute an action.

The occult vocabulary is governance vocabulary:

- a **seal** is an Ed25519-signed capability manifest;
- a **circle** is the invocation's permission, evidence-access, tool, and
  resource boundary;
- the **court of evidence** is the frozen integer controller and its explicit
  checks;
- the **law of execution** is fail-closed shadow mode.

No symbolic correspondence changes an eligibility check, score, confidence,
or authorization boundary.

## Faculties

The required faculty order is:

1. `mathematician`
2. `engineer`
3. `historian`
4. `skeptic`
5. `consequence_planner`
6. `judge`

The first five provide bounded recommendations. The judge's recommendation is
always `null` in the request because the judge's result is derived by code. A
caller cannot inject a judge result.

The six manifests live under `council/seals/`. Their signatures verify against
`council/trust-root-v0.json`. A signature authorizes only the capabilities in
the manifest. The trust root explicitly grants no tool execution authority.

## Circles

Every invocation declares:

- permissions and tools, each a subset of its seal;
- an input, output, tool-call, token, and wall-clock budget no larger than the
  seal ceiling;
- measured usage no larger than that budget; and
- the exact evidence IDs made available to the faculty.

The request is rejected before deliberation when a seal is invalid, a budget
is exceeded, a permission or tool escapes the seal, or a faculty cites evidence
outside its circle. This is a protocol failure, not a weak recommendation.

## Mathematical controller

For each candidate action the controller records boolean checks for:

- explicit allowlisting and absence from the denylist;
- risk and reversibility limits;
- required permission and tool containment in the judge's circle;
- a minimum number of distinct cited source hashes;
- a minimum number and confidence floor for supporting faculties;
- a completed skeptic review;
- no hard contradiction; and
- no material missing information.

An action is selectable only when every check is true. Eligible actions are
ordered by an integer margin:

```text
sum(support confidence)
- sum(opposition confidence)
- action risk
+ sum(predicted impact * consequence confidence / 1000)
```

The lexicographically smallest action ID breaks an exact margin tie. The judge
cannot override this ordering or select an ineligible action.

When no action is eligible, the deterministic precedence is:

1. ask the user if material user-only information is missing;
2. request evidence if evidence is missing or a hard contradiction remains;
3. abstain.

All recommendations remain in the receipt. Recommendations that do not agree
with the decision are also copied into the explicit dissent ledger.

## Wisdom receipts

`nsrl.wisdom_receipt.v0` records:

- underlying model artifact hashes;
- source and retrieved-content hashes;
- each verified seal and invocation circle;
- every faculty recommendation, confidence bucket, contradiction, predicted
  consequence, and missing-information request;
- per-action controller checks and integer margins;
- preserved dissent;
- the selected shadow recommendation, question, evidence request, or
  abstention;
- tool permissions, budgets, and measured usage;
- a pending or observed outcome; and
- an append-only revision chain containing the prior receipt hash and outcome
  observation hash.

The receipt ID and SHA-256 use recursively key-sorted JSON. The checked fixture
replays byte-for-byte. A revised receipt does not rewrite the original decision;
it records an observed outcome and revised confidence in a hash-linked revision.

Generate and check a receipt:

```bash
node scripts/run-solomon-council-v0.mjs REQUEST.json RECEIPT.json
node scripts/run-solomon-council-v0.mjs --check REQUEST.json RECEIPT.json
```

Append and check an outcome/revision:

```bash
node scripts/revise-solomon-wisdom-receipt-v0.mjs \
  RECEIPT.json OBSERVATION.json REVISED.json
node scripts/revise-solomon-wisdom-receipt-v0.mjs --check \
  RECEIPT.json OBSERVATION.json REVISED.json
```

Run the adversarial protocol self-check:

```bash
node scripts/check-solomon-council-v0.mjs
```

The self-check exercises all four decisions and rejects a forged seal, an
over-budget circle, forbidden evidence access, and a controller-forbidden
selection.

## Wisdom promotion evaluation

The promotion evaluator requires frozen solo and council traces from the same
underlying model and scores eight dimensions:

- source-grounded correctness;
- calibration;
- hard-negative rejection;
- contradiction detection;
- decision regret;
- appropriate abstention;
- cross-modal agreement; and
- unfamiliar-source transfer.

Production input cannot be assembled as one self-attested JSON file. It must be
compiled from a four-stage ceremony:

1. Freeze and publish a public casebook before either lane runs. Each case binds
   its evidence bytes, allowed decision IDs, and a salted canonical-JSON
   commitment to still-hidden gold. The freezer writes the public casebook and
   a separate mode-0600 private gold vault; the draft and vault must never be
   published with the casebook.
2. Seal the solo and council lane bundles while gold remains closed. Every lane
   trace byte-binds its runner, inputs, outputs, model artifact, and leakage
   flags. Solo has exactly one model invocation. Council has exactly one
   invocation for each of the five recommending faculties, all using the same
   model hash; the deterministic judge is not a sixth model call.
3. Publish the gold opening only after both lane bundle hashes exist. Each
   revealed gold row must match its casebook commitment, and the opening binds
   both complete lane bundles.
4. Compile the evaluator input. The compiler replays every council receipt,
   requires faculty model outputs to equal the recommendations in that receipt,
   verifies that receipt evidence exactly equals case evidence, and requires
   exact provenance source and trace sets.

The production evaluator deterministically recompiles those byte-bound ceremony
artifacts before scoring; a hand-written production input is rejected. The
integrity contract requires no oracle target lookup, hidden memory, retrieval
target leakage, or generation-integrity failure. It also byte-verifies green
`nsrl.wisdom_generation_integrity.v0` and `nsrl.wisdom_provenance_gate.v0`
reports bound to the identical underlying model artifact and requires explicit
same-model invocation, trace replay, and faculty-output-binding gates. Council
promotion requires strict improvement on every dimension across at least 72
frozen cases per dimension; ties do not pass. Self-test data can exercise the
ceremony and scorer but can never authorize promotion.

```bash
node scripts/check-solomon-wisdom-eval-v0.mjs
node scripts/check-solomon-wisdom-ceremony-v0.mjs
node scripts/freeze-solomon-wisdom-casebook-v0.mjs \
  PRIVATE-DRAFT.json PUBLIC-CASEBOOK.json PRIVATE-GOLD-VAULT.json
# Publish/commit PUBLIC-CASEBOOK.json, then run and seal both lanes.
node scripts/open-solomon-wisdom-gold-v0.mjs \
  PUBLIC-CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json \
  PRIVATE-GOLD-VAULT.json GOLD-OPENING.json
node scripts/compile-solomon-wisdom-eval-v0.mjs \
  PUBLIC-CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json GOLD-OPENING.json \
  GENERATION-INTEGRITY.json PROVENANCE.json FROZEN-SAME-MODEL-INPUT.json
node scripts/evaluate-solomon-wisdom-v0.mjs \
  FROZEN-SAME-MODEL-INPUT.json \
  benchmarks/solomon-council-v0/wisdom-eval-result.json
```

The compiler proves artifact consistency, not public chronology by itself. A
production run must publish or commit the casebook before lane generation and
both lane bundles before publishing the opening; those publication identities
belong in the wisdom receipt set. Production nonces must be independently
generated 256-bit secrets encoded as lowercase hexadecimal. Reusing a nonce or
checking the private draft/vault into the repository defeats gold hiding even
though the commitment hashes still replay.

No production same-model wisdom result is currently frozen. The canonical
status therefore reports the council core as `shadow_ready` and the wisdom gate
as `not_measured`. The canonical production ceremony directory is
`benchmarks/solomon-council-v0/production-v0/`; the status surface advances its
pipeline stage only as the casebook, both lane bundles, opening, integrity
reports, and compiled input appear. At present it truthfully reports
`casebook_not_frozen`.

## Relationship to the bounded regret experiment

`p10m-solomonic-judgment-v1` is separate evidence. Its prospectively frozen
source controller fired favorably on eight passages, recorded signed regret
`-52381` Q32, and falsified its occult hash-parity hypothesis. Its original
publication marks the sequential claim supported under an explicit conditional
unsafe-intensity null. MJ-20 later gives an exact exchangeable counterexample to
deriving that null from marginal conformal coverage, so the non-crossing
e-process is not affirmative sequential-safety evidence. The checked replacement
uses simultaneous state/action conformalization plus finite-horizon alpha
spending and requires 119 calibration source panels per family; it is
preregistered but not execution-ready. The empirical regret result also compares
against always abstaining rather than the required same-underlying-model solo
lane across all eight wisdom dimensions. The status surface keeps those claims
separate.
