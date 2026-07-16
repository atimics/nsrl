# MJ-2026-07-15-13: Proposal-only six-atom structure audit

- Date: 2026-07-15
- Status: Q32 cubic aggregate structure observed; exact low width falsified;
  population stability unidentifiable from the single-source proposal block
- Extends: MJ-2026-07-15-10 through MJ-2026-07-15-12
- Code binding:
  [`structure_audit.rs`](../../crates/nsrl-train/src/production/structure_audit.rs),
  [`check-production-atomic-structure-v1.mjs`](../../scripts/check-production-atomic-structure-v1.mjs),
  [`analyze-production-atomic-harmonics-v1.mjs`](../../scripts/analyze-production-atomic-harmonics-v1.mjs)
- Contract binding:
  [`p10m-atomic-structure-proposal-v1-contract.json`](../../benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json)
- Artifact binding:
  [`p10m-atomic-structure-proposal-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json)
- Harmonic artifact binding:
  [`p10m-atomic-harmonics-proposal-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-harmonics-proposal-v1.json)

## Question

Which of the tail, exchange, or interaction-width structures from MJ-10 is
actually present in the complete loss field of the six frozen p10m atoms?

## Protocol

The audit evaluated all `64` action subsets on two windows from each proposal
document `8--71`, for `8,192` exact production forwards. It read zero transfer
documents and zero reserved documents. The contract binds the model, tokenizer,
token stream, corpus index, move family, Rust source, and executable.

The source index adds a decisive limitation: all 64 proposal documents belong
to `simplewiki-pages-2026-06-20`. The artifact therefore has one source cluster
and explicitly sets `source_clustered_fold_estimation_available=false`. Fields
named `population` in the trace are finite proposal-aggregate coefficients;
they are not population estimates over independent source units.

The two objectives use the same Q47 exponent weights and integer logarithm,
with Q20 or Q32 final fractional precision. Q20 components are exact prefixes
of their Q32 counterparts. The audit records all 63 nonconstant aggregate and
`64 x 63` document coefficients, exact reconstruction, sharp MJ-11 tail bounds,
exchange defects, all 720 elimination orders, representation discrepancy, and
the MJ-09 boundary taxonomy.

## Artifact observations

### 1. The exact support is dense

| Objective | Nonzero interaction hyperedges | Best induced width | Orders at that width |
| --- | ---: | ---: | ---: |
| Q20 | 27 | 5 | 720 / 720 |
| Q32 | 37 | 5 | 720 / 720 |

For six variables, width five is maximal. Exact zero-support variable
elimination therefore gives no compression. C10-B is falsified for exact
support on this proposal aggregate. A thresholded retained surrogate remains a
different, uncertified question.

### 2. The refined objective has a tiny cubic tail

| Objective | Nonconstant absolute mass | Tail above order 3 | Tail fraction | Cubic minimizer gap |
| --- | ---: | ---: | ---: | ---: |
| Q20 | 130 | 14 | 10.769% | 3 Q20 |
| Q32 | 409,784 | 16 | 0.003904% | 0 Q32 |

The Q32 cubic truncation selects mask `63`, the exact global minimizer, and its
sharp empirical regret bound is `16` Q32. Its document-absolute tail is `18`,
so only `2` units cancel above order three. By contrast, Q20 reports a much
larger relative high-order tail and its cubic minimizer is mask `57`.

This is meaningful evidence that most high-order Q20 mass is a coarse-boundary
effect. It is not yet evidence that a cubic field is stable across sources.

### 3. Representation agreement is decision-level, not coefficient-level

Both objectives have the unique full-cube minimizer mask `63`. The exact
Q20-to-Q32 discrepancy residual has oscillation `37,011` Q32, while the Q32 cube
range is `356,414`; the observed representation-transfer envelope is therefore
10.384% of the aggregate cube range, and the actual Q32 regret of the Q20
minimizer is zero.

The coefficient picture is less stable:

- 24 of 63 aggregate supports disagree between Q20 and Q32;
- 10 of the 26 coefficients nonzero in both representations disagree in sign;
- at document level, 215 of 4,032 supports disagree and 22 of 202 jointly
  nonzero coefficients disagree in sign;
- 63 of 64 documents have at least one shared Q20/Q32 minimizer, but 33 have a
  nonzero worst-Q32 regret among all tied Q20 minimizers.

Thus the complete-cube decision agrees on this aggregate even though the
coefficient field is representation-sensitive. C11-D is supported only for the
observed full-cube minimizer, not for a selected sparse support.

### 4. A cubic Walsh surrogate is exact on the observed cubes

The MJ-12 analysis was derived from the checked vertex tables without new model
evaluation. For both Q20 and Q32, the aggregate degree-three Walsh truncation
has unique minimizer mask `63` and zero exact gap. The Q32 residual has
unnormalized energy `360`, exact oscillation numerator `76/64`, and spectral
regret bound `3` Q32. Q20 has energy `448`, oscillation `96/64`, and the same
bound of `3` Q20.

The canonical cubic-surrogate minimizer also has zero exact gap on every one of
the 64 document cubes in both representations. Document cubic spectral bounds
are at most `2` Q20 and `1` Q32. This is stronger representation agreement than
the Möbius support comparison, although 16 of 41 aggregate Walsh characters
nonzero in both grids still disagree in sign.

C12-A is therefore descriptively supported on the single observed source: the
field has low-degree orthogonal optimization structure with a nontrivial exact
oscillation certificate. It is not yet source-stable evidence, and it still
selects the already-falsified all-atom move.

### 5. Exchange arithmetic is exact but not a promotion certificate

Every aggregate exchange-local minimum is global on its fixed-cardinality
slice. The nontrivial Q32 defects for cardinalities two through four are
`16,429`, `18,578`, and `18,681`, giving bounds `32,858`, `55,734`, and
`74,724`. The corresponding Q20 defects are `2`, `5`, and `9`.

These are valid empirical slice bounds, but they are neither zero nor
source-replicated. The zero defects on cardinalities one and five are not
evidence for a general exchange-convex move field. C10-C remains unsupported as
a population optimizer premise.

### 6. Most atomic document edges are not visible at Q20

Across all `12,288` directed atom/document/base-subset edges:

| Boundary class | Count | Fraction |
| --- | ---: | ---: |
| Q32 component inactive | 6,688 | 54.427% |
| Q32 active, Q20 phase-masked | 1,350 | 10.986% |
| Q20 components cancel | 376 | 3.060% |
| Q20 objective visible | 3,874 | 31.527% |

The two head atoms account for `3,002` of the `3,874` visible edges. The four
final-RMS trunk atoms are mostly inactive at the refined component grid. Move
visibility is therefore strongly group-dependent.

## Interpretation

The exact cube does not reveal one representation-stable structural class:

- exact low-width structure is absent;
- aggregate Q32 is exceptionally close to cubic in the Möbius basis, while a
  cubic Walsh surrogate is exact on both observed representations;
- exchange-locality happens to identify aggregate slice minima, but its defect
  is nonzero and unreplicated;
- low-order cancellation is material;
- source-cluster stability cannot be measured because the proposal block has
  only one source unit.

Mask `63` is the all-atom trunk-plus-head move. Its proposal attraction is not a
new candidate: MJ-07 already showed that the corresponding joint move reverses
prospectively against head-only. The refined cubic result cannot rehabilitate a
falsified direct contrast.

## Decision

No optimizer change and no paid scaling are authorized.

The next empirical step is to construct a new proposal-only corpus with
multiple independent source clusters, define a genuinely new move generator,
and then test whether the cubic Walsh certificate, Q32 Möbius tail, selected
low-order terms, and direct candidate direction cross-fit across those
clusters. Q20 must remain a robustness surface, with exact discrepancy
oscillation reported. Before any new confirmation, add the leakage-safe
pre-action phase features required by MJ-12. Do not use documents `72--135` or
`136--212` for this selection work.

## Replay bindings

- Contract SHA-256:
  `a55fa73adf8c2650e36cc367dcae6aca1fec87a6bb422fa337e0d0b00a306184`
- Result SHA-256:
  `7d7432057cfa2b86796abdcaf73604541699a9fed1a939a57ad94b42bf9ce7ca`
- Harmonic result SHA-256:
  `9141c8d95b352e3144f02f47f13c4b4e937ab68df294d28cf751152f1c70e26b`
- Harmonic analyzer SHA-256:
  `3f12e41656f8888804a467e69c6da795c3374fd90ebe3d349c9c6d1f819c2b98`
- Source FNV-1a: `0xc19b8402483e9d33`
- Binary FNV-1a: `0xfa0351bca0c5631e`
- Manifest FNV-1a: `0x2ce8888292aa6852`

The independent checker recomputed every Möbius transform and reconstruction,
document aggregation, mass and tail, sharp tail inequality, exchange gate,
width histogram, Q20/Q32 discrepancy certificate, and boundary count.

## Open work

- Replicate the exact cubic Walsh decision and spectral bound across independent
  source clusters before using it as a compressed proposal premise.
- Extend the trace with leakage-safe pre-action phase observations; aggregate
  post-action boundary categories are insufficient for predictive phase work.
- Design a multi-source proposal corpus contract without consuming the frozen
  transfer and reserved document ranges.
- Define a new move generator whose candidate is not the already-falsified
  trunk-plus-head move.
