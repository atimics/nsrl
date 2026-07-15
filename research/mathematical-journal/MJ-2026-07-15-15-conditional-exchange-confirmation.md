# MJ-2026-07-15-15: Untouched Ising confirmation and conditional-exchange revision

- Date: 2026-07-15
- Status: all three C14 document endpoints supported on the untouched
  same-source surface; probe-routed conditional exchange is the strongest
  mechanism; cross-source transfer remains unidentified
- Extends: MJ-2026-07-15-14
- Frozen contract:
  [`p10m-atomic-ising-confirmation-v1-contract.json`](../../benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json)
- Structure artifact:
  [`p10m-atomic-structure-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json)
- Primary result:
  [`p10m-atomic-ising-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json)
- Revised-theory artifact:
  [`p10m-atomic-conditional-exchange-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-conditional-exchange-confirmation-v1.json)

## Protocol integrity

The statistical choices were frozen before the first confirmation forward
pass. The checked repository contract serializes them at file SHA-256
`57084a50c82883cb7d6c6d449b699cf793d495e293cf33ea1318b04aad71c9ce`.
It fixes masks `59`, `61`, and `47`, the two medoid vectors, cluster actions
`[47,59]`, three one-sided exact sign tests, and Holm familywise alpha `0.05`.

The evaluator then computed all 64 masks on two windows for each document
`136--199`: `8,192` exact production forwards. The trace hard-stopped before
document `200`. The structure checker reconstructed every cube, checked all
document coefficient sums and representation envelopes, and verified the
surface. The confirmation checker independently recomputed the exact binomial
fractions and Holm ordering and byte-replayed the analyzer. Documents
`200--212` remain unread.

All 64 confirmation documents belong to the same
`simplewiki-pages-2026-06-20` source cluster as the proposal block. Exact p
values below are therefore within-source document evidence conditional on the
exchangeability/sign-symmetry assumptions in MJ-14. They are not cross-source
population p values.

## Preregistered results

| Endpoint | Favorable | Unfavorable | Ties | Aggregate Q32 | Raw one-sided p | Holm-adjusted p | C14 status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pairwise mask `59` vs baseline | 51 | 13 | 0 | -259,574 | 9.405e-7 | 2.821e-6 | supported |
| Gibbs mask `61` vs baseline | 50 | 14 | 0 | -257,908 | 3.535e-6 | 7.069e-6 | supported |
| Routed `[47,59]` vs global `47` | 17 | 0 | 47 | -88,447 | 7.629e-6 | 7.629e-6 | supported |

Every endpoint rejects in the frozen Holm family and has negative aggregate
Q32 contrast. Thus C14-A, C14-B, and C14-C are supported on this surface.

The Q20 robustness summaries agree directionally:

- mask `59`: `40/9/15`, aggregate `-61`;
- mask `61`: `41/8/15`, aggregate `-64`;
- routed versus `47`: `16/0/48`, aggregate `-21`.

The routed action itself is favorable versus baseline on 62 of 64 Q32
documents, unfavorable on two, with aggregate `-351,270`.

## What did and did not replicate

### The only stable low-order parameter replicated exactly

Applying the frozen stability rule to confirmation again selects only Walsh
character `32`, the one-body field for atom 5:

| Surface | Q20 negative/zero/positive | Q20 numerator | Q32 negative/zero/positive | Q32 numerator |
| --- | ---: | ---: | ---: | ---: |
| Proposal | 61/3/0 | -1,896 | 64/0/0 | -8,328,716 |
| Confirmation | 61/3/0 | -2,036 | 64/0/0 | -8,470,084 |

No pair coupling passes the rule on either surface. The stable field is a real
same-source regularity; “stable low-order couplings” is falsified for this move
family because the stable set contains no order-two character.

### The pairwise action transferred, but the pairwise MAP did not

Proposal Q32 pairwise truncation selected mask `59`; confirmation Q32 pairwise
truncation selects mask `46`. Confirmation Q20 pairwise truncation and the full
Q20 cube select mask `60`; the full Q32 cube selects mask `46`. The exact Q20
and Q32 ground states have no shared minimizer, though the representation-
discrepancy certificate bounds the Q32 regret of mask `60` by the observed
residual oscillation (`9,001 <= 39,408`).

Therefore the prospective success of mask `59` establishes that one frozen
low-order proposal beat baseline. It does not establish a stable pairwise
Hamiltonian or a transferable MAP rule.

### The Gibbs action transferred, but the magnetization mask did not

Proposal quenched magnetization selected mask `61` at fugacities
`1/4,1/2,3/4`. Confirmation selects mask `60` at `1/4` and mask `62` at
`1/2,3/4`. Atom 5 remains strongly negative, but the weaker atom moments change
sign. Thus C14-B's frozen action succeeds while the stronger claim—stable Gibbs
magnetization as a document-population parameter—is falsified.

This distinction matches Proposition 14.4: magnetization thresholding is a
thermal Hamming-consensus action, not an energy-minimization theorem.

## The mechanism revealed by the router

Masks `47` and `59` share base

\[
  B=\{0,1,3,5\}\quad(\text{mask }43).
\]

Mask `47` adds atom 2; mask `59` instead adds atom 4. The routed comparison is
therefore a conditional exchange, not a choice between unrelated Hamiltonian
states.

Define singleton effect `s_d(i)=L_d({i})-L_d(0)` and

\[
  \Delta_d(B;i\to j)=L_d(B\cup\{j\})-L_d(B\cup\{i\}).
\]

**Proposition 15.1 (exact conditional-exchange decomposition).** For
`B` disjoint from `i,j`, with document Möbius coefficients `mu_d`,

\[
\begin{aligned}
  \Delta_d(B;i\to j)
  &=s_d(j)-s_d(i)+\rho_d(B;i,j),\\
  \rho_d(B;i,j)
  &=\sum_{\varnothing\ne T\subseteq B}
    \left[\mu_d(T\cup\{j\})-\mu_d(T\cup\{i\})\right].
\end{aligned}
\]

**Proof.** Expand both losses as sums of Möbius coefficients. Terms supported
inside `B` cancel. The two singleton terms remain, followed by the difference
of every interaction between a nonempty subset of `B` and the exchanged atom.
`square`

**Corollary 15.2 (exchange-margin certificate).** If
`|rho_d(B;i,j)| <= epsilon` and

\[
  s_d(j)-s_d(i)+\epsilon<0,
\]

then the exchange strictly improves the document objective.

The integer theory checker verifies the identity and margin implication. This
is more operational than a global pairwise coefficient: singleton probes give
the leading term, while the residual states exactly what must be bounded.

### Empirical partition

For the actual exchange `i=2`, `j=4`:

| Surface and frozen route | Q32 favorable/unfavorable/ties | Aggregate exchange |
| --- | ---: | ---: |
| Proposal cluster 0 (`47`) | 12/27/1 | +86,904 |
| Proposal cluster 1 (`59`) | 24/0/0 | -142,412 |
| Confirmation cluster 0 (`47`) | 13/30/4 | +91,696 |
| Confirmation cluster 1 (`59`) | 17/0/0 | -88,447 |

The partition, not a global action, replicated. Globally on confirmation,
mask `59` versus `47` is `30/30/4` with aggregate `+3,249`; always choosing
`59` would be worse in aggregate. The frozen router sends 47 documents to
mask `47` and 17 to mask `59`, exactly isolating the favorable subgroup.

The medoids differ principally in atom 4's Q32 singleton feature. On
confirmation, cluster 1 has that feature in `[-8,384,-2,214]`, while cluster 0
has `[-2,127,8,084]`. For all 17 routed documents the exact exchange lies in
`[-8,384,-2,214]`, and the interaction residual relative to the singleton
difference lies in `[-2,2]` Q32. This is the cleanest empirical mechanism in
the program so far: strong head-output singleton visibility predicts when to
replace an inactive trunk atom inside a shared stable base.

The maximum proposal residual was much larger, so Corollary 15.2 did not supply
a frozen uniform population certificate. The result supports a probabilistic
conditional-exchange law, not a proven worst-case one.

## Revised optimization theory

The evidence ranks the candidate theories as follows:

1. **Probe-routed conditional exchange:** prospectively supported with a
   replicated directional partition. This is the leading proposal mechanism.
2. **Stable one-body field:** exactly replicated for atom 5 and useful as a
   common base component.
3. **Frozen pairwise MAP action:** beats baseline, but its re-estimated MAP and
   every pair-coupling stability claim fail to replicate.
4. **Frozen Gibbs action:** beats baseline, but the magnetization mask fails to
   replicate outside the stable atom-5 component.

The missing transfer theorem is now narrower. For probe feature `phi_d`, base
`B(phi_d)`, and proposed exchange `i -> j`, establish a high-probability or
cluster-conditional bound on `rho_d(B;i,j)`. A uniform bound yields Corollary
15.2; a conditional sign bound yields a valid routed sign test. Global spectral
sparsity is neither necessary nor sufficient for this operator.

## Engineering consequences

- Represent proposal moves as **base plus exchange**, not only as an
  unconstrained mask MAP.
- Record singleton Q32 effects before evaluating multi-atom candidates; Q20 is
  too phase-masked to route reliably by itself.
- Track `rho_d(B;i,j)` explicitly in audit cubes and cross-fit its conditional
  envelope. Do not infer it from a global pair coupling.
- Retain atom 5 as a candidate base component only within this move family; its
  stability is same-source, not universal.
- Do not refit Gibbs masks or pairwise MAPs on the confirmation surface and
  report them as prospective actions.
- Build a multi-source proposal/confirmation split. Document-level exact p
  values cannot substitute for independent source clusters.
- Keep documents `200--212` sealed; 13 same-source documents are too few to
  repair the source-identification problem.

## Decision

C14-A, C14-B, and C14-C are marked `supported` on the frozen within-source
document surface. The mechanism-level conclusion is asymmetric: the router's
conditional exchange replicated, while the pairwise and Gibbs parameter maps
did not.

No optimizer change and no paid scaling are authorized. The next promotion
gate is a frozen multi-source conditional-exchange experiment with an explicit
residual-envelope or conditional-sign criterion.

## Replay bindings

- Structure contract SHA-256:
  `37951ceda0e6e10015c8de8cbdb10f137354d03ea893f64612ef9ae94262bb73`
- Structure result SHA-256:
  `353fff30b136bfbfe68130740a97ed60c89c99c5a71a499b68bb514b23475e7e`
- Statistical contract file SHA-256:
  `57084a50c82883cb7d6c6d449b699cf793d495e293cf33ea1318b04aad71c9ce`
- Primary result file SHA-256:
  `106f4724056a483d84e9ee2cd178fa54a7164ed85ad6475770cea172e7dbf0a3`
- Conditional-exchange result SHA-256:
  `2e6a23b09a2ff060b0a2ca5cd5f4adb160fafdf7e5972a6df65ab3d5a02418e2`

## Open work

- Specify a source-cluster sampling contract and the minimum independent-source
  count before further inference.
- Replace the medoid heuristic with a predeclared exchange-margin router once a
  proposal-only bound for `rho_d` is available.
- Test whether the conditional residual is controlled by the joint
  Ramanujan-phase/Walsh features from MJ-12.
- Determine whether other move families have a replicated stable base field or
  whether atom 5 is peculiar to this frozen coordinate family.
