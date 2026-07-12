# Literary model scale-up results and architecture plan

## Experiment contract

All comparable runs use the same deterministic, author-balanced corpus:

- Authors: Aleister Crowley, William Shakespeare, and William Blake.
- Available balanced bytes: 133,092 per author.
- Held out: the final 8,192 bytes per author.
- Training corpus SHA-256:
  `edc84532da9ef3a5b41f16e558e19de2c901030dbc6553d04fcc38d50ba2044f`.
- Holdout SHA-256:
  `b618579f3c5fa427f66b086a7408a026d30e335a890171eb5fdad6c83e00280f`.
- Model: `d_model=128`, two heads, hidden dimension 256, linear attention,
  NOPE positions, integer weights and activations.

The holdout evaluator reloads the saved checkpoint and scores deterministic
next-byte prediction without updating it.

## Results

| Run | Status | Updates | Context | Train accuracy | Holdout accuracy | Holdout mean probability error | Attention delta L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K, fixed shifts | pass | 7,954 | 32 | 171‰ | **170‰** | **57,640** | 2,660 |
| 8K, adaptive shifts | pass | 7,954 | 32 | 100‰ | 103‰ | 62,194 | 441,839,437 |
| 24K, fixed shifts | pass | 22,866 | 32 | 112‰ | 117‰ | 61,802 | 21,823 |
| 8K, fixed shifts | pass | 7,953 | 64 | 85‰ | 87‰ | 62,900 | **0** |
| 24K, adaptive shifts | failed | 9,728 | 32 | — | — | — | 3,057,376,462 |

The failed adaptive run drove the attention QK learning-rate shift from 18 to
1, then NSRL rejected the effective training configuration. Its partial trace
is retained under
`data/local-runs/literary-scale-24k-seq32-adaptive-failed/`.

### Readout

1. More windows do not currently improve generalization. The fixed 24K run is
   worse than the fixed 8K run on the identical holdout.
2. The adaptive shift controller is too aggressive for long runs. Its large
   attention movement does not translate to holdout accuracy and eventually
   becomes invalid.
3. The original fixed-shift trainer could not exploit doubled context because
   it produced zero attention-weight movement at sequence length 64. The later
   Adam + RMSNorm path fixes that mechanical failure; quality is measured
   separately below.
4. Generated samples remain incoherent. Held-out next-byte metrics are useful
   optimization diagnostics, not a claim of language quality.

The fixed 8K checkpoint is therefore the current baseline to beat.

## Phase 1: bounded integer Adam-style optimizer

Implement an optimizer profile without introducing float master weights:

- Keep model weights i8 and embeddings/RMS scales i16.
- Accumulate gradients in i64 as today.
- Store signed first moments and unsigned second moments per trainable value.
- Use base-2 exponential averages, initially `beta1=7/8` and `beta2=127/128`,
  so decay is implemented with deterministic shifts.
- Normalize the first moment with the existing integer reciprocal-square-root
  primitive and an explicit integer epsilon.
- Carry sub-i8 update residuals between steps instead of discarding them.
- Save optimizer state separately from inference weights, with hashes and a
  versioned resume contract.
- Bound all effective learning-rate shifts. A controller must never cross the
  kernel's valid range as the failed 24K adaptive run did.

Acceptance gates:

- Deterministic replay and serial/map-reduce parity.
- Zero-gradient, constant-gradient, saturation, resume, and corrupt-state
  tests.
- No invalid forwards or rejected configurations through 24K updates.
- Nonzero attention movement at sequence length 64.
- Beat 170‰ holdout accuracy or 57,640 mean probability error on the frozen
  holdout without regressing the other metric.

### Implementation and first measurements

The bounded integer Adam-style path is now implemented end to end:

- Signed i64 first moments and unsigned u64 second moments.
- Base-two `beta1=7/8` and `beta2=127/128` defaults.
- Existing integer reciprocal-square-root normalization with explicit epsilon.
- Separate error-feedback residuals for sub-i8 and sub-i16 updates.
- A versioned, checksummed `NSRLAD2` optimizer artifact bound to the exact
  inference-model hash.
- Deterministic serial/two-worker map-reduce parity and byte-exact resumed vs
  uninterrupted training.
- CLI training through `--mode mini-transformer-adam`, with separate
  `--optimizer-state` and `--optimizer-state-out` paths.

The 512-window balanced probe beat its identical SGD comparison: Adam reached
61,932 mean held-out probability error and 49‰ accuracy, versus SGD at 63,090
and 37‰. Both had zero invalid forwards.

The frozen 8K sweep used 7,954 windows and 498 Adam batches:

| Adam step shift | Holdout accuracy | Holdout mean error | Attention delta L1 | Invalid forwards |
|---:|---:|---:|---:|---:|
| 4 | 114‰ | **58,170** | 1,160,792 | 0 |
| 5 | 25‰ | 61,621 | 719,895 | 0 |
| 6 | **183‰** | 58,723 | 352,162 | 0 |

Shift 6 beats the old 170‰ accuracy, but regresses error from 57,640 to 58,723;
shift 4 improves error relative to the untrained and short-SGD probes but does
not beat the frozen SGD baseline. The optimizer therefore passes its safety,
replay, resume, parallelism, and nonzero-attention gates but is not promoted as
the language-quality winner yet. RMSNorm is the next hypothesis rather than
silently weakening the promotion rule.

The best-accuracy 8K artifacts are under
`data/local-runs/literary-adam-8k-shift6/`; the best-error Adam artifacts are
under `data/local-runs/literary-adam-8k-shift4/`.

`NSRLAD1` was the pre-RMSNorm development artifact. Adding learned gamma
vectors changed the optimizer parameter order, so those state files are
intentionally not resumable as `NSRLAD2`. The inference-only `NSRLMT4`
checkpoints remain loadable with RMSNorm disabled.

## Phase 2: real RMSNorm forward and backward

RMSNorm is now implemented on both pre-attention and pre-MLP paths. Every
layer has learned Q15 gamma vectors without beta, forward normalization uses
an explicit integer epsilon, and backward includes the cross-channel RMS term
and gamma gradients. Gamma is updated through integer Adam and serialized in
the `NSRLMT5` model format. Saturated forward outputs correctly mask their
backward derivative.

The implementation passed integer/float reference-gradient, constant-vector,
extreme-value, serialization, legacy-load, deterministic replay, resume, and
serial/map-reduce parity tests. The full `nsrl-core`, `nsrl-train-core`, and
`nsrl-train` suites passed after integration.

### RMSNorm measurements

The 8K sequence-32 sweep used the same 7,954 windows, 498 batches, initial
model, and frozen holdout as the Adam comparison:

| Adam step shift | Holdout accuracy | Holdout mean error | RMS gamma delta L1 | Attention delta L1 | Invalid forwards |
|---:|---:|---:|---:|---:|---:|
| 7 | 124‰ | 59,328 | 461 | 113,488 | 0 |
| 8 | **187‰** | **58,811** | 184 | 46,877 | 0 |

Shift 8 beats the fixed-SGD baseline's 170‰ accuracy but regresses mean
probability error from 57,640 to 58,811. It therefore does not pass the joint
quality promotion rule.

A three-run sequence-64 smoke used 512 windows to test the former zero-update
failure:

| Adam step shift | Holdout accuracy | Holdout mean error | RMS gamma delta L1 | Attention delta L1 | Invalid forwards |
|---:|---:|---:|---:|---:|---:|
| 3 | 65‰ | 61,546 | 1,436 | 353,988 | 0 |
| 4 | 62‰ | 61,595 | 401 | 124,689 | 0 |
| 5 | **68‰** | **61,343** | 113 | 50,128 | 0 |

RMSNorm therefore passes the mechanical sequence-64 gate: every run has
nonzero attention and gamma updates with zero invalid forwards. The bounded
512-window smoke is undertrained and does not pass the held-out quality gate.
Artifacts are under `data/local-runs/literary-rms-adam-8k-shift*/` and
`data/local-runs/literary-rms-adam-seq64-512-shift*/`.

## Phase 3: recursive swarms of small models

The primary scaling architecture is a recursive ternary swarm, not one wider
checkpoint. Keep the current `128 x 2 x 256` model as the leaf/router building
block, train several short diverse runs, and scale through depth-two router
triads with fixed integer consensus and top-two beams. The executable contract
is in `docs/recursive-literary-swarm-experiment.md`.

Adam-style optimization and real RMSNorm apply to every small expert and
router. Promotion is based on routed-vs-best-leaf gain, routing regret, oracle
coverage, utilization, and deterministic replay.

The first depth-two swarm is complete. Its frozen root consensus chose the
best pod on 94.0% of 185 final prompts and reached 98.3% top-two coverage. The
recursive route reduced mean probability error from 62,121 for the
calibration-selected fixed leaf to 61,412, while accuracy rose from 130 to 135
per mille. This validates utility-based recursive routing, but not language
quality: the leaf models remain the dominant limitation.

The next routing ablation keeps corpus, total parameters, and active compute
fixed while comparing prompt-level, 16-token-span, and per-token routing.
Low-level experts share the mixed corpus and specialize through routing; they
are not trained on isolated token shards. Shared attention carries context
between routes, and the router chooses the top one or two feed-forward experts
for each hidden state.

## Optional architecture probe: configurable multi-head profiles

The present code compiles one global `128 x 2 x 256` profile. Scaling a
constant in only one crate previously caused silent shape rejection, so model
dimensions must become a shared, serialized architecture contract.

Wider profiles are optional leaf comparisons rather than the main scaling
strategy. Start with two supported profiles:

- `small`: `d_model=128`, 2 heads, hidden 256.
- `medium`: `d_model=256`, 4 heads, hidden 512; head dimension remains 64,
  preserving the current power-of-four scaling assumptions.

Required work:

- Move dimension validation and workspace sizing behind an architecture
  profile shared by host and no-std paths.
- Record dimensions in the artifact header and reject incompatible profiles
  explicitly.
- Generalize per-head state, QK scaling, initialization, trace fixtures, and
  parameter-count reporting.
- Preserve the small profile for parity and regression testing.
- Benchmark memory and updates/second before attempting 64K windows.

Acceptance gates:

- Byte-stable small-profile replay remains green.
- Medium-profile forward/backward host/no-std parity is green.
- Every head receives a nonzero gradient/update on a constructed fixture.
- The medium profile beats the best small-profile frozen-holdout metric before
  it is used for larger training runs.

## Next experiment ladder

After each phase, use the same corpus and frozen holdout:

1. Adam-style optimizer: implemented and measured at 8K/seq32.
2. RMSNorm: implemented; 8K/seq32 and bounded seq64 gates measured.
3. Recursive small-model swarm: nine leaves and depth-two router triads
   implemented and measured.
4. Token-granularity routing ablation: prompt vs 16-token span vs token-level
   top-one/top-two routing with shared context. The oracle ceiling and first
   target-blind hidden-state router are complete; the learned token consensus
   improves the frozen representative baseline from 62,023 to 61,644 mean
   error and from 136 to 140 per-mille accuracy.
5. Optional medium four-head leaf: 8K/seq64 only after the token router
   baseline.
6. Expand source data again before a 64K aggregate experiment; do not
   manufacture scale through repeated overlapping windows alone.

Two single-trunk expert runtimes are complete. Three 256-parameter author bias
adapters and three 128-parameter diagonal hidden adapters each share one frozen
RMSNorm transformer, reducing trunk forwards by 3x. Their token-oracle ceilings
are only 57 and 119 Q15 respectively; the hidden experts also avoid six
mistakes, but neither ceiling is large enough to promote another learned
router. The next shared-trunk run uses a zero-initialized low-rank residual to
mix contextual hidden channels before the optional wider multi-head profile.

Use `scripts/summarize-literary-runs.mjs` to produce the comparison artifact.
