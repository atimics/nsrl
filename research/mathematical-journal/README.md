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
| `nu(U)` | population Möbius coefficient `E_d[mu_d(U)]` |
| `tau_r` | absolute population Möbius mass above order `r` |
| `A_r` | expected absolute document-level Möbius mass above order `r` |
| `epsilon_ex` | uniform approximate exchange-convexity defect on a fixed-cardinality slice |
| `osc_D(h)` | objective-relevant oscillation `max_D h-min_D h` |
| `R` | retained nonempty Möbius support of a sparse surrogate |
| `C(R)` | simultaneous population-regret certificate for retained support `R` |
| `W_d(A)` | unnormalized integer Walsh coefficient of action character `A` |
| `H_d(sigma)` | exact document-indexed Ising Hamiltonian on the atomic move cube |
| `m_di(beta)` | Gibbs magnetization of atom `i` under document `d` at inverse temperature `beta` |
| `bar m_i^Q(beta)` | quenched document-average magnetization |
| `Delta_d(B;i->j)` | document contrast for replacing atom `i` by atom `j` inside base mask `B` |
| `rho_d(B;i,j)` | conditional-exchange interaction residual after subtracting singleton effects |
| `lambda_d(e)` | singleton-effect margin for conditional exchange `e` |
| `q_e(phi)` | proper-training predictor of exchange residual `rho(e)` from pre-outcome probes `phi` |
| `P_u` | frozen document panel sampled from source unit `u` |
| `A_u` | simultaneous one-sided conformal score for calibration unit `u` |
| `Q_alpha` | split-conformal upper correction at marginal error level `alpha` |
| `J_t` | formal evidence-bound judgment record at sequential invocation `t` |
| `E_t` | unsafe-action e-process after source-panel round `t` |
| `S_K` | states reachable after at most `K` accepted persistent physical actions |
| `epsilon_t` | predeclared marginal error spend for fresh source-panel round `t` |
| `G_r^+` | cumulative positive exact regret through source-panel prefix `r` |
| `m_T` | maximum number of residual Walsh characters disagreeing on a vertex pair |
| `c_q(r)` | integer-valued Ramanujan sum of exact period `q` at phase `r` |
| `U` | explicitly declared set of admissible control sequences |
| `R_T(X_0, U)` | states reachable from `X_0` within `T` steps using controls in `U` |
| `B_di` | Q32 singleton influence of lattice action `i` on fitting document `d` |
| `Disc_E(x;u)` | worst-document linearized discrepancy of lattice rounding `x` from fine proposal `u` |
| `H_n` | unnormalized order-`n` Walsh-Hadamard matrix |
| `tau_i^(15)` | Q15 fitting-only sign-trust score for proposal coordinate `i` |
| `d_t` | wide diagonal curvature state for a preconditioned recurrent memory update |

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
| [MJ-2026-07-15-10](MJ-2026-07-15-10-discrete-structure-certificates.md) | Discrete structure certificates | Tail and exchange global-gap bounds established; empirical audit continued in MJ-13 |
| [MJ-2026-07-15-11](MJ-2026-07-15-11-robust-surrogate-certificates.md) | Robust surrogate and finite-sample certificates | Tail bound sharpened; width/error certificate established; six-atom population reconstruction sample-limited |
| [MJ-2026-07-15-12](MJ-2026-07-15-12-finite-group-harmonic-diagnostics.md) | Finite-group harmonic diagnostics | Exact Walsh/Ramanujan analysis and spectral regret bound established; predictive structure untested |
| [MJ-2026-07-15-13](MJ-2026-07-15-13-six-atom-structure-audit.md) | Proposal-only six-atom structure audit | Q32 cubic aggregate tail observed; exact width maximal; source stability unidentifiable |
| [MJ-2026-07-15-14](MJ-2026-07-15-14-quenched-document-ising-theory.md) | Quenched document Ising theory and prospective mechanism test | Exact theory and three untouched confirmation mechanisms frozen; document-disorder transfer law remains open |
| [MJ-2026-07-15-15](MJ-2026-07-15-15-conditional-exchange-confirmation.md) | Untouched Ising confirmation and conditional-exchange revision | Three endpoints pass; routed conditional exchange replicates while pairwise/Gibbs parameter maps do not |
| [MJ-2026-07-15-16](MJ-2026-07-15-16-conformal-conditional-exchange.md) | Conformal certificates for conditional exchange | Finite-sample unsafe-action theorem established; source-level prospective validation remains open |
| [MJ-2026-07-15-17](MJ-2026-07-15-17-prospective-cross-source-exchange.md) | Prospective cross-source conditional exchange | Checked verdict supported on frozen 71-source frame with 16/16 coverage, 5/16 firing, 0/16 unsafe action, and negative regret; vacuity is inconclusive |
| [MJ-2026-07-15-18](MJ-2026-07-15-18-multifamily-multipassage-exchange.md) | Multi-family, multi-passage conditional exchange | Overall coverage inconclusive with 14/16 panels covered; 12/64 passages fire across three families, all favorable; Federal Register and RFC promote locally, Gutenberg is withheld, science abstains |
| [MJ-2026-07-15-19](MJ-2026-07-15-19-solomonic-judgment-calculus.md) | Solomonic judgment calculus | Exact six-source record has 8/24 favorable firings and negative regret in three families; sequential e-process interpretation is superseded by MJ-20; occult hash parity falsified |
| [MJ-2026-07-15-20](MJ-2026-07-15-20-exchangeable-adaptive-composition.md) | Exchangeable adaptive composition and its finite-sample price | MJ-19 conditional bridge falsified; bounded six-panel theorem established; fresh-source optimizer execution falsified with zero firings and exact replay |
| [MJ-2026-07-15-21](MJ-2026-07-15-21-correlated-lattice-neural-updates.md) | Correlated lattice optimization and scale-stable neural updates | Five recent-result-derived NSRL conjectures and a bounded experiment ladder established; optimizer, architecture, scaling, and release promotion remain unauthorized |
| [MJ-2026-07-16-22](MJ-2026-07-16-22-council-tool-parity-hardening.md) | Council tool-parity hardening | Historical v0 promotion falsified under stronger requirements: solo had 0 tool observations versus Council's 2,880, a diagnostic parity baseline ties all eight dimensions, seven hardening surfaces remain missing, and Council returns to shadow-only |

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
