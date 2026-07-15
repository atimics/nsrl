# NSRL Mathematical Journal

This is the append-only mathematical notebook for NSRL. It exists to turn
implementation choices and experiment results into explicit definitions,
derivations, conjectures, and falsifiers.

The journal is not a status report. `PROJECT_STATUS.md` says what currently
passes. The research notes summarize literature and experiment programs. This
journal records the mathematical model those programs are testing.

## Rules

1. Give every entry a stable ID of the form `MJ-YYYY-MM-DD-NN`.
2. Separate derived facts from code observations, artifact observations, and
   conjectures.
3. State the numeric representation and scale of every quantity in an equation.
4. State the admissible controls when using words such as *reachable* or
   *capacity*.
5. Name the objective and evaluation surface when using words such as *better*,
   *useful*, or *descent*.
6. Give every conjecture a falsifier before using it to authorize a larger run.
7. Do not silently rewrite a failed claim. Mark it falsified or superseded and
   link the replacement entry.
8. Treat deterministic replay as reproducibility evidence, not as evidence that
   the chosen objective or update direction is correct.

## Evidence labels

- **Definition** — a convention introduced by the journal.
- **Lemma** — a derived consequence of definitions and stated assumptions.
- **Proposition** — a substantial derived claim with a proof or proof sketch.
- **Code observation** — a statement bound to named implementation lines.
- **Artifact observation** — a statement bound to a frozen result artifact.
- **Literature result** — a claim attributed to a primary source.
- **Conjecture** — a falsifiable claim not yet established.
- **Decision** — an experiment or promotion consequence of the preceding work.

Each conjecture has one of these states:

- `open`
- `supported`
- `falsified`
- `superseded`

`supported` means that a bounded experiment agreed with the conjecture. It does
not turn an empirical conjecture into a theorem.

## Canonical notation

| Symbol | Meaning |
| --- | --- |
| `z` | stored integer value |
| `f_z` | fractional-bit exponent, so the represented real is `z 2^{-f_z}` |
| `R_s(z)` | deterministic integer rounding of `z / 2^s` |
| `Sat_b(z)` | saturation to the signed `b`-bit integer range |
| `theta_t` | persistent model parameters after optimizer step `t` |
| `r_t` | persistent error-feedback residuals after optimizer step `t` |
| `D_t` | deployed state, including parameters and every forward-affecting scale |
| `H_t` | hidden training state, including residuals, controller, batch, seed, and counters |
| `pi(X)` | exact deployed-function observation of full state `X` |
| `K` | fixed integer mass assigned to one output-gradient proposal |
| `C` | a complete discrete training contract |
| `F_C(theta, x)` | exact deployed forward result under contract `C` |
| `L_E(theta)` | declared loss on frozen evaluation surface `E` |
| `A_i` | the `i`th deterministic admissible state action in a declared move family |
| `ell_X(S)` | exact loss after applying the canonically ordered action subset `S` to `X` |
| `mu_X(S)` | Boolean Möbius coefficient measuring the irreducible interaction of `S` |
| `C_d^q(A\|B)` | conditional effect of action block `A` after `B` on evaluation unit `d` under objective `q` |
| `v` | objective-visibility probability `Pr(C != 0)` |
| `m` | signed visibility margin `Pr(C < 0)-Pr(C > 0)` for a negative-is-better contrast |
| `lambda_f(n)` | `f`-fractional-bit output of the declared integer logarithm algorithm |
| `chi_delta(y_0,y_1)` | exact number of coarse quantizer cells crossed between two fine-grid values |
| `A_d` | total absolute output-boundary activity for evaluation unit `d` |
| `kappa_d` | coherence of signed output-boundary components for evaluation unit `d` |
| `U` | explicitly declared set of admissible control sequences |
| `R_T(X_0, U)` | states reachable from `X_0` within `T` steps using controls in `U` |

Subscripts identify tensor roles or parameter groups. Superscripts identify
representations only when needed; for example, `p^(23)` is a Q23 probability.

## Index

| Entry | Title | Principal status |
| --- | --- | --- |
| [MJ-2026-07-14-01](MJ-2026-07-14-01-foundations.md) | Foundations of deterministic integer training | Five open conjectures; paid scaling not authorized |
| [MJ-2026-07-14-02](MJ-2026-07-14-02-discrete-optimization.md) | Optimization on the functional quotient | Discrete residual trust-region theory proposed |
| [MJ-2026-07-14-03](MJ-2026-07-14-03-fibered-recon.md) | Fibered optimization and reciprocal-free descent | Partially superseded by MJ-04 |
| [MJ-2026-07-15-04](MJ-2026-07-15-04-three-geometry-optimization.md) | Three-geometry optimization with fixed-mass proposals | Rescue-stratified v2 fails trunk gate; optimization and scaling unauthorized |
| [MJ-2026-07-15-05](MJ-2026-07-15-05-monotone-systematic-fixed-mass.md) | Monotone wide loss and systematic fixed-mass proposals | Systematic-`K` audit lanes implemented; empirical conjectures remain open |
| [MJ-2026-07-15-06](MJ-2026-07-15-06-boolean-jet-discrete-sensitivity.md) | Boolean-jet optimization and exact discrete sensitivity | Algebra established; post-hoc transfer synergy superseded by MJ-07 |
| [MJ-2026-07-15-07](MJ-2026-07-15-07-prospective-boolean-jet-falsification.md) | Prospective Boolean-jet falsification and stability theory | Frozen synergy falsified; audit engineering closed; new proposal required |
| [MJ-2026-07-15-08](MJ-2026-07-15-08-hierarchical-distributional-boolean-jets.md) | Hierarchical distributional Boolean jets and certifiability | Block pushforward and certification bounds established; hierarchical proposal operator open |
| [MJ-2026-07-15-09](MJ-2026-07-15-09-objective-boundary-phase-calculus.md) | Objective-boundary phase calculus | Exact Q20 boundary decomposition established; reverse boundary proposal map open |
| [MJ-2026-07-15-13](MJ-2026-07-15-13-six-atom-structure-audit.md) | Proposal-only six-atom structure audit | Compact aggregate structure observed; selected move already falsified; no scaling |
| [MJ-2026-07-15-14](MJ-2026-07-15-14-quenched-document-ising-theory.md) | Quenched document Ising theory | Three within-source confirmation mechanisms frozen prospectively |
| [MJ-2026-07-15-15](MJ-2026-07-15-15-conditional-exchange-confirmation.md) | Conditional-exchange confirmation | Routed partition replicated within source; pairwise and Gibbs maps did not |

## Entry template

```markdown
# MJ-YYYY-MM-DD-NN: Title

- Date:
- Status:
- Supersedes:
- Code binding:
- Artifact binding:

## Question

## Definitions and assumptions

## Derivation

## Observations

## Conjectures and falsifiers

## Decision

## Open work
```
