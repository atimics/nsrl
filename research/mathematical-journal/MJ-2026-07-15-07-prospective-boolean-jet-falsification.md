# MJ-2026-07-15-07: Prospective Boolean-jet falsification and stability theory

- Date: 2026-07-15
- Status: frozen transfer-synergy candidate falsified; stability theory proposed
- Supersedes: the transfer-only synergy interpretation in MJ-2026-07-15-06
- Code binding: `crates/nsrl-train/src/production/boolean_jet.rs`
- Artifact binding:
  [`p10m-boolean-jet-confirmation-v1-contract.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1-contract.json)
  and
  [`p10m-boolean-jet-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json)

## Question

Does the one-Q20 conditional trunk-after-head advantage observed post hoc on
four transfer documents survive a frozen test on unused document blocks and a
wider observation objective?

## Frozen experiment

The move family was fixed before selected-document evaluation. It contains the
four `final_rms` unit moves `T` and the two output/bias unit moves `H` from the
v3 mass-corrected lane. The contract binds the base model, tokenizer, token
stream, move fingerprint `0xc11353911a5130fb`, and full manifest hash
`0x263f6984eeccfa84`.

The observation objective is the separately versioned MJ-05 Q47
logit-anchored base-2 NLL, reported in Q20. It changes neither deployed forward
behavior nor optimizer state. Two windows are summed within each document.

- Proposal diagnostic: documents 8 through 71.
- Primary transfer endpoint: documents 72 through 135.
- Reserved replication: documents 136 through 212, not evaluated here.
- Primary contrast per document:

  ```text
  C_d = ell_d(TH) - ell_d(H) = mu_d(T) + mu_d(TH).
  ```

- Test: exact two-sided paired sign test excluding ties.
- Gate: direction `C_d < 0`, `p <= 0.05`, and at least 32 non-tied documents.
- Even a pass would not authorize an optimizer change.

## Artifact observations

### Proposal diagnostic

The aggregate vertex losses were

```text
ell(empty) = 1745648100
ell(T)     = 1745648092
ell(H)     = 1745648028
ell(TH)    = 1745648016
```

Thus

```text
mu_T = -8,  mu_H = -72,  mu_TH = -4,
ell(TH)-ell(H) = -12.
```

At document level, `TH` beat `H` on 10 documents, lost on 2, and tied on 52.
The exact sign-test value was `79/2048 = 0.03857421875`, but only 12 documents
were non-ties. The diagnostic direction is favorable but fails the predeclared
minimum-information gate.

### Primary transfer endpoint

The aggregate vertex losses were

```text
ell(empty) = 1745271996
ell(T)     = 1745271998
ell(H)     = 1745271906
ell(TH)    = 1745271912
```

Therefore

```text
mu_T = +2,  mu_H = -90,  mu_TH = +4,
ell(TH)-ell(H) = +6.
```

The head-only vertex is the unique aggregate minimizer. At document level,
`TH` beat `H` on 7 documents, lost on 11, and tied on 46. The exact two-sided
sign-test value was `15751/32768 = 0.480682373046875`; only 18 documents were
non-ties. Both the direction and minimum-information requirements fail.

The result replayed byte for byte with SHA-256
`662ba99d0c830ac01d45777792bf929205a7fb5936ad6d38e47795522516ac00`.

## Decision

The frozen transfer-only trunk/head synergy candidate is **falsified**. The
post-hoc one-Q20 advantage was not a stable property of the move family. Do not
apply the joint move, change the optimizer around it, or use it to authorize
scaling.

This does not falsify the Boolean-jet algebra, which reconstructed every cube
exactly, nor the broader conjecture that some structured low-rank proposals can
transfer. It falsifies this particular move manifest and the inference from a
four-document post-hoc interaction to a reusable optimizer repair.

## Revised theory: coefficients are random fields

For a document or independently sampled evaluation unit `d`, define

```text
mu_d(S) = sum_(T subseteq S) (-1)^(|S|-|T|) ell_d(T).
```

The global coefficient `mu_E(S)` is a sum over `d`; it is not evidence that the
same interaction holds on most documents. The optimizer-relevant object is the
distribution of `mu_d(S)` and of conditional effects such as

```text
C_d(T | H) = ell_d(TH)-ell_d(H) = mu_d(T)+mu_d(TH).
```

This yields three distinct requirements:

1. **Function visibility:** the action changes deployed logits.
2. **Objective visibility:** the change survives the declared loss lattice.
3. **Distributional stability:** the conditional effect has a favorable,
   sufficiently informative distribution on unused evaluation units.

Define the objective-visibility rate

```text
v_E(T | H) = Pr_d[C_d(T | H) != 0].
```

The observed rates were `12/64` on the proposal diagnostic and `18/64` on the
transfer endpoint. All nonempty vertices were function-visible, yet most
document contrasts were objective ties. Function visibility is therefore not
a sufficient optimization signal.

For a finite frozen sample, a structured move should be eligible only when it
passes a declared stability functional, for example

```text
S_E(T | H) = (sign-test direction, exact p, non-tie count, aggregate effect),
```

with no component silently substituted after observation. Aggregate descent
alone is especially unsafe on a quantized objective because a small number of
one-LSB changes can determine the sum.

## The missing theoretical breakthrough, revised

The missing object is not merely a compressed boundary adjoint. It is a
**stability-aware compressed proposal operator**: a low-cost map from boundary
events and hidden optimizer state to a small move family whose negative
low-order Möbius coefficients persist across unused documents and compatible
observation objectives.

In operational terms, the proposal operator must concentrate not just
`E[mu_d(S)] < 0`, but a favorable distribution with enough objective-visible
mass to distinguish signal from the loss lattice. A reverse-mode compression
that predicts unstable one-LSB coefficients would be computationally elegant
and still optimization-useless.

## Conjectures and falsifiers

### C07-A: the frozen v3 trunk-after-head effect transfers

- State: `falsified`
- Falsifier observed: on the predeclared transfer documents, head-only won more
  non-tied document contrasts than trunk-plus-head and the aggregate conditional
  effect reversed sign.

### C07-B: proposal-visible interactions predict transfer interactions

- State: `open`
- Claim: a move subset selected prospectively by document-level proposal
  coefficients beats a matched control on an unused transfer block.
- Falsifier: proposal-selected interactions reverse or tie at the same rate as
  matched controls.

### C07-C: a higher-resolution observation increases useful visibility

- State: `open`
- Claim: a separately frozen observation objective with more than Q20 loss
  resolution reduces ties without reversing robust conditional rankings.
- Falsifier: extra resolution mainly creates unstable sign changes or fails to
  improve the effective non-tie count.

## Engineering consequences

1. Treat calibration and confirmation as different trace schemas and require a
   nonzero manifest hash for confirmation.
2. Report document-level coefficients, conditional effects, exact tie counts,
   and the effective non-tied sample size.
3. Make the minimum non-tied count part of the decision gate; total document
   count is not a substitute.
4. Preserve the reserved document block for a genuinely new proposal rule.
5. Add a machine checker that validates contract bindings, Möbius
   reconstruction, sign-test arithmetic, and the final authorization decision.
6. Do not promote a move family selected from this failed confirmation. Design
   the next candidate generator without inspecting documents 136 through 212.
7. The matched-control extension added after this v1 run is a new protocol, not
   a retroactive part of the frozen confirmation. Preserve an immutable v1
   replay path and version the matched-control protocol separately before using
   it on reserved documents.

## 2026-07-15 engineering closure

The implementation now enforces those consequences without changing the
frozen v1 evidence:

- atoms carry source lane, move kind, canonical order, and exact
  collision/repetition/boundary reject semantics;
- the full manifest hash binds the model, tokenizer, token stream, and every
  ordered atom;
- calibration authorization is always false;
- source equivalence uses an ordered hash of every complete per-window output
  gradient vector rather than min/max/L1 summaries;
- Q15 canonical and MJ-05 Q47 observation objectives have separate versioned
  specifications including fractional bits, zero floor, and aggregation;
- window/document losses, per-document Möbius coefficients, gamma-one, vertex
  model/function hashes, and runtime inversion checks are retained;
- Rust tests cover rank-two and rank-three transforms, action order,
  collisions, boundaries, deterministic replay, and restoration; the
  repository checker independently verifies the frozen cube and sign test; and
- systematic fixed-mass lanes and a seeded group/cardinality/width-matched
  control-manifest freezer are implemented.

The matched-control path is optional so the checked-in v1 remains
byte-replayable. It has not been used on reserved documents 136--212. Those
documents remain unavailable until a genuinely new stability-aware proposal
rule and its complete decision contract are frozen.
