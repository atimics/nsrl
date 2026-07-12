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

## Configurable multi-head small profiles

The transformer dimensions now remain in the shared host/no-std contract while
the head layout has two build profiles:

- `small-h2-d128-ff256`: 2 heads of dimension 64, the byte-stable default.
- `small-h8-d128-ff256`: 8 heads of dimension 16 via `mini-heads-8`.

Both head dimensions are powers of four, which is required by NSRL's exact
integer QK scaling. A four-head width-128 model would have dimension-32 heads
and is therefore invalid; four 64-dimensional heads still require a future
width-256 profile. The small eight-head profile tests attention specialization
without turning the project into one large model.

The model header already serializes dimensions and head count. Cross-profile
loads are rejected explicitly, and separate Cargo target directories keep the
binaries isolated. Host/no-std single-step parity passes for H8, and a new
head-delta audit proves nonzero Q, K, V, and O updates for every head.

On the same 512-window sequence-64 Adam sweep:

| Profile | Best Adam shift | Holdout accuracy | Holdout mean error | Mistakes |
|---|---:|---:|---:|---:|
| H2 default | 5 | 68‰ | 61,343 | 3,175 |
| H8 small | 5 | **189‰** | **53,753** | **2,763** |

H8 improves mean error by 7,590 Q15 and avoids 412 mistakes, with zero invalid
forwards. A second H8 shift-5 run reproduced the model, Adam state, trace, and
per-head deltas byte-for-byte. The small H8 profile is promoted; the next step
is several H8 leaves, not immediate width growth.

The consolidated evidence is
`data/experiments/literary-multi-head-profile-v1/report.json`.

The follow-up H8 swarm uses three 512-window leaves with Adam shifts 3/4/5.
Author-isolated and disjoint-offset leaves fail their diversity gates, while
optimizer-scale leaves produce a 4,103-Q15 token-oracle ceiling. A
calibration-selected target-blind token router captures 174 Q15 on frozen
final data, raises accuracy from 148‰ to 151‰, and avoids 102 mistakes. A
16-token router captures 27 Q15 and avoids 15 mistakes with only 76 switches.
Both learned routers are promoted. Evidence is in
`data/experiments/literary-h8-swarm-v1/report.json`.

The efficiency follow-up distills training into resumable rank-16 i16
residuals over one frozen H8 trunk. Teacher imitation of the other whole models
does not preserve enough diversity, but target-trained curriculum residuals do
continue learning safely where direct i8 Adam cannot. Seven 512-window stages
lower mixed holdout error from 53,753 to 53,497 and raise accuracy from 189‰ to
191‰. Stage 8 is rejected on exact holdout error. On frozen literary targets,
fixed stage 7 reaches 55,991 error and 153‰ accuracy; its token oracle ceiling
is only 96 Q15, so no new router is trained.

Adapter-aware generation is deterministic, but greedy and top-8 samples fail
the prose-quality gate by collapsing toward spaces and a few letters. The next
phase must improve trainable precision inside transformer blocks and expand
balanced data; next-token metrics are not treated as language quality.
Evidence is in `data/experiments/literary-h8-curriculum-v1/report.json`.

RMSNorm-only Adam is now an explicit training scope. It updates just the 512
i16 gamma values inside H8, freezes all i8 matrices and embeddings, and passes
its isolation test. It adds a tiny holdout gain to the residual curriculum but
does not beat the non-RMS winner on frozen literary targets, so it is a proven
mechanism rather than the selected final checkpoint.

Fine-grained trainable precision is now also available after every H8 block.
Each small expert uses a deterministic sign projection and a learned i16 Q15
expansion, is hash-bound to one frozen trunk, backpropagates through later
frozen blocks, and stores an explicit residual shift so fixed-point saturation
is part of the artifact contract. Rank-8 experts add 2,048 parameters; rank-32
capacity probes add 8,192.

The first stable rank/rate sweep learned its 512-window slice, but no point
improved the exact 3,407-window holdout. The closest rank-8 point missed the
trunk by 278 total Q15 error with the same 2,763 mistakes. Rank 32 did not fix
the generalization gap, and its deterministic top-8 samples still failed all
three prose gates. The mechanism is retained for author/span expert swarms,
but no checkpoint is promoted. Evidence is in
`data/experiments/literary-h8-block-expert-v1/report.json`.

The next experiment made that decomposition explicit. Source markers from the
balanced corpus were parsed into disjoint per-author leaf, router-training,
calibration, and untouched-final chunks. The shard builder reproduces the
repository's ASCII-lower token file byte-for-byte before emitting any split.
Three rank-8 block experts were selected only by their own leaf loss, then
scored with target-blind shared-trunk features.

On 3,313 final targets, every fixed author expert was worse than the zero
expert trunk: Crowley by 3,118 total Q15, Shakespeare by 824, and Blake by
1,782. Conditional utility nevertheless exists: target-aware prompt, span-16,
and token oracles improve the trunk by 237, 4,339, and 8,023 Q15 respectively.
The gap is too small for the learned routers. A three-router child swarm and a
second neural root router were trained for both token and span decisions; the
best recursive result was still 863 Q15 worse than the trunk and routed 96.6%
of tokens to Shakespeare. Generation again failed the prose gate.

This rejects author identity as the expert boundary, not the recursive-router
architecture. The next leaves should specialize on contiguous spans or
token-context clusters, and router training should be skipped unless the
calibration oracle gap is materially larger. Evidence is in
`data/experiments/literary-h8-author-block-swarm-v1/report.json`.

A below-author follow-up clusters 524 non-overlapping 512-token spans from all
three authors using 24 byte-bigram buckets and eight structural ratios. The
deterministic target-blind k-means converges in 13 iterations to clusters of
247, 136, and 141 spans; every cluster contains Crowley, Shakespeare, and
Blake and supports more than 1,500 sequence-64 training windows. Three matched
rank-8 block experts are then trained on those clusters.

This surface-context decomposition is also rejected. Every fixed expert loses
to the zero-expert trunk; the best is 1,244 total Q15 worse. Its final token
oracle improves the trunk by only 4,462 Q15, and its calibration oracle gain
of 9,581 is smaller than the rejected author swarm's 13,117. The router gate
therefore correctly skips another neural hierarchy. A directly deployable
nearest-centroid router briefly gains 216 Q15 on calibration at span
granularity but regresses by 2,244 against the trunk on final data. Prose again
fails.

The next clustering signal should be model-native: frozen-trunk residual or
gradient signatures can define expert training groups, while a separate
target-blind hidden-state router learns to predict those utility-defined
groups. Evidence is in
`data/experiments/literary-h8-context-block-swarm-v1/report.json`.

That model-native experiment is now complete. Each of the same 524 disjoint
512-token spans is represented by 16 signed and 16 magnitude buckets from the
frozen trunk's final-hidden gradient. Deterministic standardized k-means forms
cross-author clusters of 237, 133, and 154 spans. These groups separate the
trunk's error regimes without using author identity as a shortcut.

The experiment also exposed a fixed-point precision bug in block-expert
training: each sample's Q30 outer product was rounded before the learning rate
was applied, erasing small metric-aligned gradients. Accumulating raw products
across the batch and dividing once makes those updates observable. With the
probability-error objective, all three frozen experts now beat the shared trunk
on the same 3,313 final targets by 1,408, 1,110, and 179 total Q15 respectively.
This is the first promoted internal per-block expert decomposition.

Conditional utility is larger than the fixed gain: the prompt, span, and token
oracles beat the trunk by 1,969, 5,448, and 8,462 Q15. The token oracle therefore
has 7,054 Q15 of room beyond the best fixed leaf. Three target-blind child
routers and a second neural router were calibration-selected for both token and
span routing. They improve on the trunk but do not beat the fixed cluster-0
expert; the best recursive result gains 1,247 Q15 over the trunk versus 1,408
for that fixed expert. Routing is therefore not promoted yet. Top-8 generation
still collapses toward spaces and a few letters, so prose is also rejected.
Evidence is in
`data/experiments/literary-h8-gradient-block-swarm-v1/report.json`.

The short-run continuation gate is now measured as well. Each promoted leaf
was resumed on two successive non-overlapping bands of its own gradient
cluster. Two learning rates per leaf were tested at each stage; candidates
were selected by the complete triad's token-oracle loss on router calibration,
not by local training loss. This is the intended scale unit: many bounded runs
with independent acceptance rather than one long optimization trajectory.

The result generalizes. The best frozen expert improves the untouched-final
trunk by 5,506 total Q15, versus 1,408 before the curriculum. The final token
oracle improves the trunk by 18,360 and retains 12,854 beyond the new best
fixed expert. Calibration's token-oracle gap beyond fixed grows from 11,311 to
23,090. Next-byte mistakes are unchanged, so this is a probability-quality
gain rather than a prose claim.

Classification-style child and recursive routers still miss the fixed leaf by
737 and 961 Q15. Their utility-soft targets were found to erase most small
loss differences in a Q8 softmax. A new expected-regret objective differentiates
the router's expected child loss directly and preserves even one-unit integer
regret. It closes the frozen-final router gap to 109 Q15, but still does not
beat the fixed expert. The remaining bottleneck is target-blind observability:
the current router averages each four adjacent contextual channels into one.
The next router view should use signed projections of the full hidden state.
Generation remains space-heavy and fails every prose gate. Consolidated
evidence is in
`data/experiments/literary-h8-gradient-block-curriculum-v1/report.json`.

The first observability follow-up adds an opt-in signed-projection feature mode
to the block-expert scorer. It maps all 128 final contextual channels into 32
deterministic features while leaving the historical contiguous-pooling mode
byte-exact by default. The projection seed and scale are bound into the v2
score report. A unit test verifies that the projection preserves within-bucket
signals that pooled features erase.

Scale matters: projection shift 4 clips at least one feature in 5,881 of 5,887
calibration rows and is rejected before router fitting. Shift 7 has zero
calibration saturation. Two projection seeds are compared on train and
calibration only; seed 1 wins. Its expected-regret span router improves fixed
by 199 Q15 on calibration, then exactly ties fixed on final by making no final
switches. Three projected child routers and direct-regret token/span roots also
tie fixed. This removes the pooled router's 109-Q15 regression but does not
capture any of the 12,854-Q15 final token-oracle ceiling.

The signed-projection runtime is validated, but no routed checkpoint is
promoted. The evidence now rejects further 32-channel compression tuning as
the next scale step. A versioned router should consume all 128 contextual
channels plus nine prior-token probes directly and widen its hidden layer from
16 to 32. Evidence is in
`data/experiments/literary-h8-gradient-block-projected-router-shift7-v1/report.json`.

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
5. Small eight-head profile: implemented, replayed, and promoted on the bounded
   sequence-64 comparison.
6. H8 optimizer swarm: three leaves plus learned token/span router implemented
   and promoted.
7. Shared-trunk H8 residual curriculum: seven stages implemented and promoted;
   prose-generation gate measured and failed.
8. Fine-grained i16 residuals inside every H8 block: implemented and
   mechanically validated; first rank-8/rank-32 checkpoints rejected by exact
   holdout and prose gates.
9. Train provenance-labelled author/span block experts and a target-blind
   hierarchical router: author-level version completed and rejected; it
   exposes an 8,023-Q15 token ceiling but learned routes do not beat the trunk.
10. Subdivide leaves by contiguous span/token-context clusters and require a
   larger calibration oracle gap before fitting another router hierarchy:
   surface-context clustering completed and rejected by this gate.
11. Cluster training spans by frozen-trunk residual/gradient signatures, then
   learn a separate target-blind hidden-state predictor for those groups:
   completed. All fixed gradient experts beat the trunk; the learned recursive
   routes improve the trunk but do not beat the best fixed leaf.
12. Resume metric-aligned gradient experts through short, independently gated
   stages to widen conditional utility before fitting another router swarm:
   two stages completed and promoted; fixed and oracle gains generalize.
13. Train routers with direct expected regret instead of quantized soft utility:
   implemented and measured; it closes most of the router gap but finishes 109
   Q15 behind the best fixed expert.
14. Replace contiguous four-channel hidden averaging with deterministic signed
   projections of all 128 contextual channels: implemented and measured;
   saturation-free projection removes router regression but only ties fixed.
15. Add a versioned 137-input, width-32 integer router over all 128 contextual
   channels plus nine prior-token probes, then integrate a winning checkpoint
   as a block-local top-one/top-two dispatcher.
16. Expand source data again before a 64K aggregate experiment; do not
   manufacture scale through repeated overlapping windows alone.

Four single-trunk expert runtimes are complete. Bias, diagonal hidden,
rank-32 low-rank, and diagonal-plus-rank-16 hybrid adapters each share one
frozen RMSNorm transformer, reducing trunk forwards by 3x. The hybrid fixed
expert is the winner at 57,818 mean error, improving the former diagonal fixed
result by 557 Q15. Its token oracle reaches 57,670, but a three-replica
target-blind router collapses to 99.9% Blake utilization and captures none of
that ceiling. The experts are promoted while that router is not. The small H8
profile and its optimizer-scale token/span routers are promoted, demonstrating
that the new attention layout creates a much larger conditional gap.

Use `scripts/summarize-literary-runs.mjs` to produce the comparison artifact.
