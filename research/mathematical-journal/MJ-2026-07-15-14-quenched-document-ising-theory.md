# MJ-2026-07-15-14: Quenched document Ising theory and prospective mechanism test

- Date: 2026-07-15
- Status: mathematical model and three confirmation mechanisms frozen before
  documents `136--199`; optimizer and scaling remain unauthorized
- Extends: MJ-2026-07-15-11 through MJ-2026-07-15-13
- Code binding:
  [`analyze-production-document-ising-v1.mjs`](../../scripts/analyze-production-document-ising-v1.mjs),
  [`check-production-document-ising-proposal-v1.mjs`](../../scripts/check-production-document-ising-proposal-v1.mjs),
  [`check-document-ising-theory-v1.mjs`](../../scripts/check-document-ising-theory-v1.mjs)
- Proposal artifact binding:
  [`p10m-atomic-ising-proposal-v1.json`](../../benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json)

## Question

Does a low-order aggregate Hamiltonian, a quenched document Gibbs response, or
a probe-only document router predict lower loss on untouched documents? The
question is deliberately comparative: an Ising reparameterization by itself is
an identity, not an optimization theory.

## Exact document Hamiltonian

**Definition.** For document `d`, action mask `x in F_2^k`, and spin
`sigma_i=(-1)^{x_i}`, let `L_d(x)` be its exact integer objective. Define the
normalized Walsh coefficient

\[
  \widehat L_d(A)=2^{-k}\sum_x L_d(x)(-1)^{A\cdot x}.
\]

Then

\[
  H_d(\sigma)=L_d(x)
  =\widehat L_d(\varnothing)+
    \sum_{A\ne\varnothing}\widehat L_d(A)\prod_{i\in A}\sigma_i.
\]

In standard Ising signs,

\[
  H_d=C_d-\sum_i h_{di}\sigma_i-\sum_{i<j}J_{dij}\sigma_i\sigma_j
      +H_d^{(\ge3)},
\]

where `h_di=-widehat L_d({i})` and
`J_dij=-widehat L_d({i,j})`. The artifact stores unnormalized integer
numerators `W_d(A)=64 widehat L_d(A)`; all comparisons remain exact.

**Proposition 14.1 (Ising--Walsh equivalence).** This Hamiltonian reconstructs
every cube value exactly.

**Proof.** Character orthogonality gives
`sum_A (-1)^{A.(x+y)}=2^k 1{x=y}`. Substitution into the inverse transform
leaves `L_d(x)`. The checker verifies the identity in integer numerators on a
rank-three nontrivial field. `square`

This is a change of basis. It does not imply pairwise sufficiency,
ferromagnetism, a thermodynamic phase, or generalization.

## Aggregate disorder and cancellation

Let `w_d >= 0`, `sum_d w_d=1`, and

\[
  \bar H(\sigma)=\sum_d w_d H_d(\sigma).
\]

**Proposition 14.2 (aggregation commutes with the transform).** Every
coefficient of `bar H` is the correspondingly weighted document coefficient.
Thus an aggregate field or coupling can be small because each document term is
small or because large terms cancel. Only document-level signs and magnitudes
distinguish those cases.

**Proof.** The Walsh transform is linear. `square`

For a retained pairwise Hamiltonian `G` and residual `R=bar H-G`, the robust
surrogate result from MJ-11 becomes

\[
  \bar H(\hat x)-\min_x\bar H(x)\le \operatorname{osc}(R),
  \qquad \hat x\in\arg\min_x G(x).
\]

This is a deterministic regret certificate, not a transfer theorem.

## Quenched Gibbs response

For inverse temperature `beta > 0`, define the document Gibbs law

\[
  P_{d,\beta}(x)=\frac{e^{-\beta H_d(x)}}{Z_d(\beta)},\qquad
  m_{di}(\beta)=E_{P_{d,\beta}}[\sigma_i].
\]

The **quenched magnetization** is

\[
  \bar m_i^Q(\beta)=\sum_d w_d m_{di}(\beta).
\]

The frozen implementation evaluates this exactly at Q20 fugacities
`e^{-beta} in {1/4,1/2,3/4}` and rounds only the final moments to Q30.

**Proposition 14.3 (quenched and aggregate Gibbs operations do not commute).**
In general,

\[
  \sum_d w_d P_{d,\beta}\ne P_{\bar H,\beta},
  \qquad
  \sum_d w_d m_d(\beta)\ne m_{\bar H}(\beta).
\]

**Proof by exact counterexample.** For one spin, equal document weights,
fugacity `1/2`, and state energies `H_1=(0,4)`, `H_2=(2,0)`, quenched
magnetization is `12/85`. The mean Hamiltonian is `(1,2)`, whose magnetization
is `1/3`. `square`

This noncommutation is the mathematical reason document heterogeneity can
reverse an aggregate proposal.

**Proposition 14.4 (magnetization threshold is a Hamming Bayes action).** Let
`P^Q=sum_d w_d P_{d,beta}`. The mask

\[
  x_i^Q=1\{\bar m_i^Q<0\}
\]

minimizes `E_{X~P^Q}[d_H(x,X)]`, with either bit admissible when the
corresponding magnetization is zero.

**Proof.** Expected Hamming loss separates by coordinate. Choosing spin `+1`
costs `P(sigma_i=-1)` and choosing `-1` costs `P(sigma_i=+1)`; their
difference is the magnetization. `square`

This theorem is narrower than the desired conclusion. It says the frozen Gibbs
mask is the coordinatewise thermal consensus, not that it minimizes expected
energy. The untouched energy test is therefore essential.

**Proposition 14.5 (Gibbs stability under Hamiltonian error).** For two finite
Hamiltonians `H,G`,

\[
  \|P_{H,\beta}-P_{G,\beta}\|_{TV}
  \le \tanh\!\left(\frac{\beta\operatorname{osc}(H-G)}4\right),
\]

and for every spin,

\[
  |m_i(H)-m_i(G)|
  \le 2\tanh\!\left(\frac{\beta\operatorname{osc}(H-G)}4\right).
\]

**Proof.** The log likelihood ratio has range at most
`beta osc(H-G)`. Among likelihood ratios with fixed range ratio and mean one,
total variation is maximized by a two-point endpoint law, giving
`(sqrt R-1)/(sqrt R+1)=tanh(log R/4)`. A spin has range two. `square`

This supplies a usable clustering criterion: small within-cluster Hamiltonian
oscillation implies stable magnetization. Feature proximity alone does not.

**Proposition 14.6 (zero-temperature recovery).** If `H` has a unique ground
state `x*`, gap `Delta` to every other state, and `N=2^k`, then

\[
  P_\beta(x^*)\ge
  \frac1{1+(N-1)e^{-\beta\Delta}}.
\]

If `(N-1)e^{-beta Delta}<1`, every magnetization sign equals the spin of `x*`,
so thresholding recovers `x*`.

**Proof.** Bound every excited-state weight by
`e^{-beta Delta}` times the ground weight. Ground mass above one half dominates
the worst possible opposing contribution to every spin moment. `square`

Multiple ground states can keep a magnetization at zero; no unique mask follows
without a tie convention.

## Document routing

For each document define the probe feature

\[
  \phi_d=(L_d(e_1)-L_d(0),\ldots,L_d(e_6)-L_d(0)).
\]

The frozen router assigns `phi_d` to the nearest of two proposal medoids in L1
distance, breaking ties toward cluster zero. Its candidate masks have
cardinality at least two, so neither routed candidate is one of the probe
vertices.

**Proposition 14.7 (cluster representative regret).** If cluster `c` has a
representative Hamiltonian `G_c`, `a_c in argmin G_c`, and
`osc(H_d-G_c)<=delta_c`, then

\[
  H_d(a_c)-\min_x H_d(x)\le\delta_c.
\]

**Proof.** Apply the MJ-11 oscillation-regret lemma to `H_d` and `G_c`.
`square`

The current router does not establish the premise: it clusters singleton
features, not full Hamiltonian oscillations. Its prospective comparison is a
test of whether those cheap probes carry enough information in practice.

**Inference assumption.** Because the router and actions are frozen before the
confirmation cube, route selection does not inspect the candidate-versus-
control contrast. An exact routed sign test still requires conditional sign
symmetry under the null, and the document units must be independent or
exchangeable at the level used by the test. One SimpleWiki source cluster makes
that assumption doubtful; p-values will be reported as within-source document
evidence, not source-population evidence.

## Relation to the literature

Edwards and Anderson introduced random signed couplings to model frozen local
directions without global ferro- or antiferromagnetic order
([1975, DOI 10.1088/0305-4608/5/5/017](https://doi.org/10.1088/0305-4608/5/5/017)).
The relevant NSRL analogy is **quenched document disorder**: each document has
a fixed Hamiltonian during an action decision. No spin-glass phase transition
is claimed.

Work on joint quenched measures shows that replacing disordered systems by a
single annealed Gibbs description can be subtle even in conventional lattice
models
([Kuelske 1999](https://arxiv.org/abs/math-ph/9910048)). Proposition 14.3 is the
finite NSRL obstruction in its simplest form.

Inverse-Ising research studies recovery of unknown parameters from spin
samples, with identifiability depending on the regime
([Bhattacharya and Mukherjee 2018](https://arxiv.org/abs/1507.07055)). NSRL's
rank-six proposal cube is complete, so its coefficients are known exactly;
inverse-Ising estimation is not the present bottleneck. The bottleneck is
whether a document-disorder summary transfers to unseen documents.

The prior Ramanujan/Tao program remains complementary. Walsh analysis supplies
the exact Hamiltonian coordinates; structured-versus-uniform decomposition can
diagnose the high-order residual. Neither harmonic sparsity nor a small Gowers
norm alone proves that a selected action improves held-out energy.

The phrase “primes as local minima” does not add a theorem here. Local minima
are defined only after choosing an energy and an adjacency graph; on this cube
the adjacency is Hamming-one. Primality supplies neither the Hamiltonian nor a
descent certificate.

## Proposal-only observations

All observations below use documents `8--71` from the single
`simplewiki-pages-2026-06-20` source cluster. Documents `72--212` were not read
by the proposal analyzer.

1. Of the 21 nonconstant order-one/two characters, only character `32`—the
   one-body field for atom 5—passes the frozen rule: at least 32 Q32-visible
   documents, at least three-quarter directional agreement among visible
   documents, and Q20/Q32 aggregate-sign agreement. Its standard field
   numerator is negative on all 64 Q32 documents and on 61 of 61 Q20-visible
   documents. No pair coupling is stable by this rule.
2. The Q32 pairwise aggregate MAP is mask `59`. Its exact aggregate gap is
   `2,024` Q32 above full mask `63`; the certified pairwise-residual oscillation
   is `231,984/64 = 3,624.75` Q32. Q20 instead selects mask `63`. Thus pairwise
   compression is certified only by a loose regret envelope and is
   representation-sensitive at the decision level.
3. The quenched Q20 Gibbs mask is `61` at all three frozen fugacities
   `1/4,1/2,3/4`. At the central fugacity, five mean spin moments are negative;
   atom 1 is weakly positive.
4. The global directional control is mask `47`, favorable on all 64 proposal
   documents. The two-medoid probe router has medoid documents `61` and `66`,
   sizes `40` and `24`, and candidates `[47,59]`.
5. Contiguous eight-fold cross-fitting gives router-versus-baseline `64/0` and
   router-versus-global-control `26/5` among 31 non-ties, aggregate increment
   `-126,143` Q32. Interleaved folds give `25/4` among 29 non-ties and
   `-131,187`. These are same-source internal estimates, not untouched results.

## Frozen conjectures and falsifiers

The confirmation surface is documents `136--199`, two windows per document,
with a hard stop before document `200`. Documents `200--212` remain sealed.
The three one-sided document-direction endpoints form one Holm family at
familywise alpha `0.05`; ties are omitted from the exact binomial sign test.

### C14-A: pairwise aggregate MAP transfers (`open`)

- Action: mask `59`.
- Control: baseline mask `0`.
- Primary endpoint: Q32 paired document direction.
- Falsifier: the one-sided endpoint does not reject after Holm correction, or
  its aggregate Q32 contrast is nonnegative.

### C14-B: quenched magnetization transfers (`open`)

- Action: central-fugacity Q20 magnetization mask `61`.
- Control: baseline mask `0`.
- Primary endpoint: Q32 paired document direction.
- Descriptive replication: Q20 direction and the confirmation quenched
  magnetization mask, computed without changing the frozen action.
- Falsifier: the one-sided endpoint does not reject after Holm correction, or
  its aggregate Q32 contrast is nonnegative.

### C14-C: probe routing adds value (`open`)

- Router: frozen L1 medoids with feature vectors
  `[0,0,0,0,1977,-4068]` and `[0,0,0,0,-6398,-4020]`.
- Routed actions: `[47,59]`; ties route to mask `47`.
- Control: global directional mask `47`.
- Primary endpoint: Q32 paired document direction on non-ties.
- Falsifier: the one-sided endpoint does not reject after Holm correction, or
  its aggregate Q32 incremental contrast is nonnegative.

The stable-field replication and all coupling, aggregate-loss, Q20,
magnetization, and route-versus-baseline summaries are descriptive. No
threshold may be changed after opening the surface.

## The missing theoretical breakthrough

The missing result is a **document-disorder transfer law**: assumptions stated
in observable proposal quantities under which a compressed Hamiltonian,
thermal consensus, or probe partition has bounded held-out energy regret.
Walsh/Ising coordinates make this question precise, but they do not solve it.
Propositions 14.5 and 14.7 show what a sufficient law would need—control of
Hamiltonian oscillation within the relevant document population. The present
experiment asks which cheap statistic, if any, predicts that control.

## Decision

The untouched confirmation is authorized as a bounded theory-discrimination
experiment. No optimizer change and no paid scaling are authorized. The same-
source design cannot establish cross-source generalization even if all three
document endpoints pass.

## Open work

- Freeze the execution contract, including evaluator and analysis hashes,
  before computing documents `136--199`.
- Preserve documents `200--212` for a later independent audit.
- Add a multi-source corpus surface; without multiple source units, the desired
  document-disorder transfer law remains statistically unidentified.
- Measure within-cluster full-Hamiltonian oscillation after the prospective
  test, descriptively, to determine whether Proposition 14.7 explains any
  router gain.
