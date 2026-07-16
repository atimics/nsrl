# MJ-2026-07-15-10: Discrete structure certificates

- Date: 2026-07-15
- Status: local-to-global certificate theory established; exploitable p10m structure unproved
- Extends: MJ-2026-07-15-06 through MJ-2026-07-15-09
- Code binding:
  [`check-discrete-structure-theory-v1.mjs`](../../scripts/check-discrete-structure-theory-v1.mjs)
- Artifact binding:
  [`p10m-boolean-jet-confirmation-v1.json`](../../benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json)

## Question

Which measurable property would turn Boolean local information into an
efficient optimizer or a quantitative global certificate?

## Result

Another finite-difference identity is not the missing breakthrough. A move-cube
loss is an arbitrary pseudo-Boolean function unless the substrate supplies more
structure. Three structures are sufficient to make materially stronger claims:

1. small absolute high-order Möbius mass gives an explicit global-gap bound;
2. small exchange-convexity defect makes exchange-local minima near-global on
   fixed-cardinality slices;
3. a low-width sparse interaction graph permits exact variable elimination.

The current rank-two p10m evidence establishes none of them. Its aggregate
interaction reverses sign between proposal and transfer documents, and most of
the observed interaction mass cancels within each surface.

## 1. Literature correction: this is pseudo-Boolean optimization

The Boolean loss table

```text
S -> ell(S),    S subseteq {1,...,k},
```

is a pseudo-Boolean function. Its Möbius expansion is its unique multilinear
polynomial in the AND basis. “Boolean jet” remains useful NSRL terminology for
the collection of coefficients around a declared state, but it is an
application of established pseudo-Boolean and incidence-algebra machinery.

The literature adds four constraints that were missing from the first journal
entries:

- [Boros and Hammer's pseudo-Boolean survey](https://doi.org/10.1016/S0166-218X(01)00341-9)
  organizes exact optimization by algebraic degree, sign structure, and factor
  graph rather than treating every Boolean polynomial alike.
- [Murota's discrete convex analysis](https://epubs.siam.org/doi/10.1137/1.9780898718508.ch6)
  gives exchange classes for which a local optimality criterion is global.
- [MADS for granular and discrete variables](https://doi.org/10.1137/18M1175872)
  supplies a mature polling/refinement precedent for expensive black-box
  objectives. Its continuous convergence theory does not automatically apply
  to NSRL's hidden-state integer transition system.
- [Adaptive Sparse Möbius Transforms](https://arxiv.org/abs/2602.06246)
  gives near-optimal adaptive query algorithms for exactly sparse, bounded-degree
  Boolean polynomials. This February 2026 preprint is the closest algorithmic
  match to the proposed compressed Boolean adjoint, but assumes exact sparsity
  and query access to selected vertices.

Two additional experimental constraints follow from adjacent literature:

- [Stochastic rounding theory for LLM training](https://proceedings.mlr.press/v258/ozkara25b.html)
  includes convergence analysis under Adam and large-model experiments, making
  a seeded stochastic lane a theoretical as well as empirical control.
- [Adaptive holdout theory](https://papers.nips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html)
  shows that repeated hypothesis generation against the same validation data
  invalidates ordinary fixed-analysis guarantees. A fresh split for one test is
  not a license to reuse it for later proposal design.

## 2. Population Boolean field

For one frozen move family and objective representation, define the population
loss and population Möbius coefficient

```text
F(S)  = E_d[ell_d(S)],
nu(U) = E_d[mu_d(U)].
```

Linearity gives

```text
F(S) = sum_(U subseteq S) nu(U).
```

The executable checker verifies in 300 integer cases that environment
aggregation commutes exactly with the Möbius transform. This fact does not make
the coefficients stable: `nu(U)` can be small because `mu_d(U)` is consistently
small or because large document effects cancel.

Define the population absolute tail above order `r` and its stronger
environmental counterpart by

```text
tau_r = sum_(|U|>r) |nu(U)|,

A_r   = sum_(|U|>r) E_d[|mu_d(U)|].
```

Jensen's inequality gives `tau_r <= A_r`. The difference `A_r-tau_r` is
distributional cancellation mass. Proposal search needs estimates of `A_r`, not
only the aggregate `tau_r`.

## 3. Absolute interaction-tail certificate

Define the order-`r` truncation

```text
F_r(S) = sum_(U subseteq S, |U|<=r) nu(U).
```

### Theorem 3.1 (uniform tail bound)

For every vertex `S`,

```text
|F(S)-F_r(S)| <= tau_r.
```

**Proof.** The difference is the sum of omitted coefficients whose supports are
subsets of `S`. The triangle inequality bounds this partial absolute sum by the
absolute sum over every omitted coefficient.

### Corollary 3.2 (global gap of a truncated minimizer)

Let `S_hat` minimize `F_r` and let `S_star` minimize `F`. Then

```text
F(S_hat)-F(S_star) <= 2 tau_r.
```

**Proof.** Apply Theorem 3.1 on both sides of
`F_r(S_hat)<=F_r(S_star)`.

The checker verifies the uniform and two-tail bounds on 50,800 vertices from
700 random integer cubes through rank eight.

This is the first explicit local-model-to-global-loss certificate in the
journal. Its limitation is equally explicit: computing or upper-bounding
`tau_r` is generally as difficult as finding the omitted interactions. The
theorem converts the missing breakthrough into a measurable target—an absolute
tail bound—rather than assuming low-order coefficients suffice.

### Proposition 3.3 (low-order nonidentifiability)

For every `r<k`, two rank-`k` cubes can agree on every vertex of cardinality at
most `r` yet have different global minima.

**Construction.** Give one cube zero coefficients and the other one nonzero
coefficient supported only on all `k` actions. Every proper low-order vertex is
identical, while the full vertex differs. The checker preserves a rank-six
example that is indistinguishable through order two but has a hidden loss `-7`.

Therefore no amount of rank-two accuracy alone can certify a small high-order
tail.

## 4. Approximate exchange certificate

Many proposal families have a fixed action budget. On the fixed-cardinality
slice

```text
D_m = {S : |S|=m},
```

define a uniform exchange defect `epsilon_ex` by requiring that for every
`X,Y in D_m` and every `i in X\Y`, some `j in Y\X` satisfies

```text
F(X)+F(Y)
  >= F(X-i+j)+F(Y+i-j)-epsilon_ex.
```

`epsilon_ex=0` is the relevant M-convex exchange inequality on the slice.

### Theorem 4.1 (exchange-local global-gap bound)

If `X` has no improving one-for-one exchange, then

```text
F(X)-min_(Y in D_m) F(Y) <= m epsilon_ex.
```

**Proof.** Let `Y` be a global minimizer. If `X!=Y`, choose `i in X\Y` and the
exchange partner `j` supplied by the approximate axiom. Write
`X'=X-i+j` and `Y'=Y+i-j`. Exchange-locality gives `F(X')>=F(X)`, so the
approximate axiom implies `F(Y')<=F(Y)+epsilon_ex`. The new `Y'` is one exchange
closer to `X`. Repeat at most `m` times until `Y=X`.

The checker computes the minimal uniform defect for 300 arbitrary random
fixed-cardinality losses and verifies the bound for 430 exchange-local minima.
It also verifies 100 zero-defect modular slices, where every exchange-local
minimum is global.

This suggests a stronger meaning for “primes as local minima”:

- an ordinary Boolean prime is only relative to its audited neighborhood;
- an exchange prime plus a measured small `epsilon_ex` has a quantitative
  global-gap certificate on its fixed-cardinality slice.

The p10m rank-two trunk/head cube is not an exchange audit because its vertices
have different cardinalities and each block contains multiple atoms.

## 5. Sparse low-width exact optimization

Let the nonzero population coefficients define hyperedges

```text
E = {U : nu(U) != 0}.
```

Connect two action variables in the primal interaction graph whenever they
appear in a common hyperedge. A variable-elimination order has induced width
`w` when no elimination bucket contains more than `w+1` variables.

### Proposition 5.1 (exact minimization at bounded induced width)

Given the factorized coefficients and an elimination order of induced width
`w`, exact minimization takes

```text
O((k+|E|) 2^(w+1))
```

integer table operations, up to representation constants.

**Proof sketch.** Collect all factors containing the next variable, sum them on
their joint scope, minimize over that bit, and replace the bucket by the
resulting factor. Every intermediate table contains at most `2^(w+1)` entries.
Induction over the elimination order preserves the minimum over eliminated
assignments. The checker compares this algorithm with brute force on 200 sparse
integer cubes of induced width at most two.

Low polynomial degree alone is insufficient. A dense quadratic function has
only order-two interactions but can have interaction width `k-1`. Conversely,
a higher-order factorization arranged along a narrow graph can remain exactly
tractable.

The relevant structural tuple is therefore

```text
(absolute interaction tail, support sparsity, factor rank, induced width,
 exchange defect),
```

not merely maximum observed interaction order.

## 6. Audit of the existing p10m rank-two field

For a two-action minimization cube, document-level submodularity requires
`mu_d(TH)<=0`. The confirmation artifact gives:

| Surface | negative `mu_d(TH)` | zero | positive | sum | absolute sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| proposal | 7 | 52 | 5 | -4 | 14 |
| transfer | 9 | 42 | 13 | +4 | 22 |

Consequences:

1. Five proposal documents and thirteen transfer documents violate the
   rank-two submodular sign.
2. The aggregate interaction reverses from `-4` to `+4` Q20.
3. Aggregate interaction coherence is only `4/14` on proposal and `4/22` on
   transfer. Equivalently, cancellation consumes `10/14` and `18/22` of the
   observed absolute interaction mass.
4. Six of 52 proposal conditional ties and nine of 46 transfer conditional ties
   have nonzero `mu_d(T)` or `mu_d(TH)` that cancel in
   `C_d(T|H)=mu_d(T)+mu_d(TH)`.

These are Boolean-component cancellations, not yet the denominator/target
boundary cancellations of MJ-09. The current artifact cannot distinguish the
output-boundary taxonomy without a refined per-window trace.

**Decision 6.1.** Stable document-level submodularity is not supported by the
rank-two evidence. No local-to-global claim should be based on the aggregate
negative proposal interaction.

## 7. Revised compressed proposal theory

Replace the single opaque proposal operator with a structure-conditional
pipeline:

```text
semantic and boundary sketch
  -> sparse interaction discovery on proposal documents
  -> estimate absolute tails, support graph, exchange defect, and visibility
  -> choose solver justified by the observed structure
  -> freeze at most M candidates
  -> source-clustered prospective confirmation.
```

The solver choice is conditional:

- small `tau_r` or a proved upper bound: optimize the truncated polynomial and
  report the `2 tau_r` certificate;
- small exchange defect on a fixed-budget slice: use exchange descent and
  report the `m epsilon_ex` certificate;
- low induced width: use exact variable elimination;
- none of the above: use a budgeted MADS-style or stochastic direct search and
  label it a heuristic without a global certificate.

The 2026 sparse Möbius algorithms make adaptive group testing a concrete
candidate for the discovery stage. Their exact-sparsity and noiseless-query
assumptions must be replaced by approximate-tail and document-distribution
bounds before they can authorize model changes.

## 8. Statistical firewall

Structure discovery is adaptive analysis. It must not repeatedly interrogate
the completed transfer block. Use:

1. source-clustered folds inside proposal documents for structure discovery and
   cross-fitting;
2. one untouched confirmation block for the frozen candidate family;
3. a separately sealed final corpus for any headline optimization claim;
4. familywise or sequential error control tied to the number of released
   candidate decisions.

The document sign test remains a useful directional endpoint, but a structural
claim additionally needs confidence bounds for `E|mu_d(U)|`, exchange defects,
tail mass, and representation concordance.

## 9. Next bounded experiment

Run a proposal-only exact atomic cube using the six already frozen p10m actions:
four trunk atoms and two head atoms. Evaluate all 64 vertices only on the
existing proposal documents 8--71. Do not reuse transfer documents 72--135 or
reserved documents 136--212 for structure selection.

Record, per document and objective representation:

- all 63 nonempty atomic Möbius coefficients;
- absolute interaction mass by order;
- `tau_r` and empirical `A_r` for `r=1,...,5`;
- the exact support hypergraph and best induced width over all 720 elimination
  orders;
- exchange defects on cardinality slices `m=1,...,5`;
- Q20/Q32 sign and support concordance;
- phase masking and output-component cancellation from MJ-09.

This diagnostic directly tests whether the apparent 45-term coarse interaction
has sparse atomic structure. It consumes no new confirmation data and authorizes
neither an optimizer change nor scaling.

## Conjectures and falsifiers

### C10-A: the atomic Boolean field has a small absolute high-order tail

- State: `open`
- Claim: above some small order `r`, empirical `A_r` is a small declared
  fraction of total absolute interaction mass across proposal folds.
- Falsifier: high-order absolute mass remains material or changes rank across
  folds and objective representations.

### C10-B: useful interactions have sparse low-width support

- State: `open`
- Claim: non-negligible atomic coefficients form a support graph with small
  induced width and stable semantic localization.
- Falsifier: the support graph is dense, unstable, or only appears sparse after
  representation-dependent thresholding.

### C10-C: fixed-budget move losses are approximately exchange-convex

- State: `open`
- Claim: cardinality slices have an `m epsilon_ex` gap materially smaller than
  the observed benefit scale.
- Falsifier: exchange defects are comparable to or larger than all candidate
  advantages.

### C10-D: sparse structure discovered on proposal folds transfers

- State: `open`
- Claim: a structure-conditioned proposal beats phase-, group-, and
  cardinality-matched controls on a new source-clustered block.
- Falsifier: support, direction, or visibility reverses under the frozen
  prospective protocol.

## Decision

Treat structural identification—not another scalar interaction—as the next
theoretical gate. “Primes as local minima” acquires global force only after an
exchange, tail, or factor-width certificate is measured.

No new transfer experiment is authorized yet. The next work is the exact
six-atom proposal cube and same-algorithm refined output trace.

## Open work

- Extend the tail theorem to empirical confidence sequences for adaptively
  queried coefficients.
- Derive approximate sparse-Möbius recovery under integer observation noise and
  clustered document sampling.
- Bound atomic tail mass from branch-local boundary envelopes without evaluating
  every vertex.
- Define a representation-independent threshold for the support graph.
- Add the generating source and replay runner to the immutable research freeze.
