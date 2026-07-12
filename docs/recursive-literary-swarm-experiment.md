# Recursive literary swarm experiment

This experiment tests NSRL scaling through a recursive swarm of small models.
It does not treat the existing manifest/prompt-affinity router as a neural
router; that path remains a flat heuristic baseline.

## Topology

The experiment is a depth-two ternary tree:

```text
root router triad
  -> Crowley router triad -> Crowley experts 0, 1, 2
  -> Shakespeare router triad -> Shakespeare experts 0, 1, 2
  -> Blake router triad -> Blake experts 0, 1, 2
```

Every internal node has three neural-router replicas (`semantic`,
`structural`, and `full`), three children, fixed Q15 rank/confidence consensus,
and a top-two beam. Fixed consensus terminates recursion and prevents a router
from requiring another learned router forever.

## Leakage-safe data contract

Each author is truncated to the same 133,092-byte budget and split contiguously
in this order:

| Split | Bytes per author | Use |
|---|---:|---|
| Leaf train | 100,324 | Train the three small leaf variants |
| Router train | 16,384 | Fit neural routers after utility labels exist |
| Router calibration | 8,192 | Calibrate consensus and beam thresholds |
| Final test | 8,192 | Frozen routed-vs-flat-vs-oracle result only |

Leaf variants rotate the same author-only paragraph set by zero, one-third,
and two-thirds, then train at sequence length 32 with approximately 8K windows.
This changes SGD order and boundaries without leaking router or final-test text.

Initial router records contain 32 Q15 features: 24 hashed byte-bigram buckets
and eight structural ratios. After child scoring, nine inference-safe probe
features are appended: three child quality estimates, three prefix accuracies,
and three relative loss advantages. The resulting neural-router input has 41
features. Root author labels are warm-start metadata only; all trained routers
use measured child utility.

## Prepare

```bash
scripts/prepare-recursive-literary-swarm.sh
```

The generated experiment is under
`data/experiments/literary-recursive-swarm-v1/`:

- `experiment.manifest.json`: topology, splits, hashes, jobs, and contracts.
- `preflight.json`: preparation checks and remaining blockers.
- `leaf-jobs.tsv`: nine deterministic leaf jobs.
- `router-jobs.tsv`: twelve neural-router jobs across four triads.
- `router-data/*.jsonl`: root and local training/calibration/final records.

The preparation passes 177 checks. The preflight report now discovers trained
artifacts as well as validating the immutable preparation contract.

## Train leaves

Train all nine sequentially:

```bash
scripts/run-recursive-literary-leaves.sh
```

Limit a run while testing orchestration:

```bash
NSRL_RECURSIVE_SWARM_MAX_JOBS=1 \
  scripts/run-recursive-literary-leaves.sh
```

Each completed leaf is evaluated against every author's frozen final split.
This produces the cross-author utility matrix required for routing and catches
cases where an author-trained expert unexpectedly performs better elsewhere.

All nine leaves have now completed, producing nine unique model hashes and 27
cross-author evaluations with no invalid forwards. The detailed comparison is
`data/experiments/literary-recursive-swarm-v1/leaf-comparison.json`.

The best generalist is `blake-expert-1`, with mean cross-author accuracy of 162
per mille. It is also best by probability error on the Crowley and Shakespeare
final splits. `crowley-expert-2` is marginally best by probability error on the
Blake split. Most experts have little or negative own-author advantage.

This means the current byte-level leaves are diverse but not meaningfully
author-specialized: they mainly learn common character/spacing statistics.
That result is why source labels are not accepted as final router oracle labels
and why per-sample utility scoring is the next required stage.

## Router stages

Router training proceeds bottom-up:

1. Train and cross-evaluate all nine leaves.
2. Score every local router prompt with its three child experts.
3. Fill `oracle_child_losses_q15` and `oracle_target`; never use final-test rows
   for this step.
4. Train three router replicas per author pod using separate feature views.
5. Calibrate fixed consensus and top-two beam on router calibration rows.
6. Measure each complete author pod on root router-train rows.
7. Replace root author warm-start labels with measured pod utility.
8. Train and calibrate the three root routers.
9. Run the frozen comparison: best leaf, flat average, flat heuristic router,
   recursive neural top-one, recursive neural top-two, and oracle top-two.

All bottom-up stages are now complete. The experiment produced nine leaf
checkpoints, nine local router checkpoints, three root router checkpoints, and
separate hashed oracle, calibration, and final-test artifacts.

## Frozen result

The root consensus weights were selected only on calibration data and frozen
at `semantic=1`, `structural=2`, and `full=2`. On 185 final prompts it achieved:

- 94.0% best-pod selection accuracy.
- 98.3% top-two best-pod coverage.
- Mean root routing regret of 6 Q15 loss units.
- Mean end-to-end regret of 8 Q15 units versus the best of all nine leaves.
- 66.4% replica disagreement, confirming that the router replicas did not
  collapse into identical copies.

The recursive route selected a leaf with mean probability error 61,412 and
next-byte accuracy 135 per mille. The calibration-selected fixed baseline,
`blake-expert-1`, scored 62,121 and 130 per mille on the same final prompts.
Thus routing improves error by 709 Q15 units and accuracy by 5 per mille, but
the generated-language model remains weak. Routing is no longer the principal
bottleneck; leaf learning is.

The auditable report is
`data/experiments/literary-recursive-swarm-v1/root-router-report.json`.

## Next granularity experiment

The next swarm should subdivide routing decisions, not split the corpus into
isolated token shards. Keep shared embeddings and attention so every decision
retains sentence context, then compare equal-budget routing at three levels:

1. One route per prompt (the completed baseline).
2. One route per 16-token span, with a later 8-token probe.
3. One route per token, selecting the top one or two tiny feed-forward experts.

Train low-level experts over the whole mixed corpus with a load-balancing term,
allowing specialization to emerge around reusable functions such as dialogue,
meter, syntax, or vocabulary. Keep author pods as a coarse outer hierarchy.
Measure held-out probability error, routing regret, utilization entropy,
expert starvation, wall-clock cost, and deterministic replay while holding the
data, parameter budget, and approximate active compute constant.

### Oracle ceiling result

The first granularity ablation is complete over all 185 frozen final passages
and 42,120 stride-one prediction windows. It uses one representative model per
author (`crowley-expert-2`, `shakespeare-expert-1`, and `blake-expert-1`) and
asks a target-aware oracle which already-trained whole-model expert has the
lowest measured probability error. This is an upper bound, not a deployable
router result.

| Route granularity | Accuracy | Mean probability error | Expert utilization per mille |
|---|---:|---:|---:|
| Best fixed expert | 136‰ | 62,023 | 0 / 0 / 1000 |
| Prompt oracle | 135‰ | 61,892 | 399 / 0 / 600 |
| 16-token-span oracle | 141‰ | 61,332 | 427 / 98 / 473 |
| Per-token oracle | **168‰** | **58,757** | 428 / 247 / 323 |

Relative to prompt routing, 16-token routing improves accuracy by 6 per mille
and error by 560 Q15. Per-token routing improves accuracy by 33 per mille and
error by 3,135 Q15, while using all three experts substantially. All routes
had zero invalid forwards.

The signal justifies token-aware routing, but not running three complete
transformers for every token. The deployable architecture should keep
embeddings and attention shared, route only small feed-forward expert chunks,
and begin with 16-token decisions before shortening toward per-token top-one
or top-two routing. The token oracle switched experts 27,093 times, so the
learned router also needs a switching penalty or route hysteresis.

The report and target-level training table are under
`data/experiments/literary-token-routing-v1/`. The scorer is
`nsrl-mini-transformer-routing-ablation`; it emits explicit non-claims that
the target-aware oracle and whole-model experts are not the final MoE design.

### Learned target-blind result

The follow-up experiment converts separate router-train, calibration, and
final passages into 85,094 / 42,139 / 42,120 token decisions. Current-target
losses create training labels, but router inputs exclude the current target.
Selection uses calibration only; the final split is never used for epochs,
feature views, consensus weights, or hysteresis.

The first router used byte-bigram histograms and hard one-hot labels. It
collapsed to one expert after a single epoch. Replacing hard labels with a
soft distribution derived from all three measured utilities eliminated
optimizer saturation, but raw byte features still mostly reproduced a fixed
expert.

The successful router consumes 32 pooled channels from the generalist
transformer's final contextual state plus nine rolling utility probes from
already-observed tokens. Three small router replicas use separate feature
views, and integer Q15 probability consensus weights are selected on
calibration.

| Frozen final route | Accuracy | Mean probability error | Expert utilization per mille |
|---|---:|---:|---:|
| Best fixed representative | 136‰ | 62,023 | 0 / 0 / 1000 |
| Learned 16-token-span router | 137‰ | 62,011 | 75 / 0 / 924 |
| Learned per-token router | **140‰** | **61,644** | 503 / 98 / 398 |
| Per-token oracle ceiling | 168‰ | 58,757 | 428 / 247 / 323 |

The learned token router improves both frozen metrics and captures about 12%
of the oracle's available error reduction. Its selection accuracy is only
39.9%, which is expected when many child losses are near-ties; measured routed
utility, not label accuracy, is the promotion metric.

The token consensus still switches 23,542 times. A calibration-only
hysteresis search correctly chose zero margin because every tested token
margin lost more utility than it saved in switches. Span routing selected a
512-Q15 margin and switched only 346 times, but its quality gain is negligible.
The next integrated MoE should therefore charge switching cost directly
during training or keep a top-two expert set warm rather than imposing a
post-hoc threshold.

Auditable artifacts are under
`data/experiments/literary-token-routing-v1/learned-router/`, especially
`dataset-manifest-hidden.json` and `learned-router-hidden-report.json`.
The current hidden state comes from the frozen Blake generalist as a shared
context proxy; routing tiny FFN experts inside one transformer remains the
next implementation step.

### Shared-trunk execution result

The next ablation removes the largest inefficiency in the whole-model oracle.
One frozen RMSNorm transformer now produces the contextual state and logits
once, then three 256-parameter i32/Q8 bias adapters provide the Crowley,
Shakespeare, and Blake expert branches. All artifacts are checksummed and
bound to the same trunk hash, `0x93ab43678ff8e7a0`.

Directly fine-tuning the frozen trunk's final i8 FFN, with and without its
output head, exposed a quantization/catastrophic-forgetting problem: a useful
single-window descent direction exists, but multi-window i8 changes are either
zero or too coarse. The bias adapters therefore use fine-grained i32 values
and train against the exact probability-error objective rather than one-hot
cross entropy. Each author adapter improves its own training error in one
full-corpus update while the trunk remains byte-identical.

On the 42,120 frozen final targets:

| Shared-trunk route | Accuracy | Mean probability error | Trunk forwards |
|---|---:|---:|---:|
| Best fixed bias expert | 156‰ | 58,897 | 42,120 |
| 16-token bias oracle | 156‰ | 58,891 | 42,120 |
| Per-token bias oracle | 156‰ | 58,840 | 42,120 |
| Naive three-model execution | — | — | 126,360 |

This proves single-trunk execution and a threefold reduction in transformer
forwards, but the bias experts are too weak: even the target-aware token oracle
improves error by only 57 Q15 and accuracy by zero. Training another learned
router is not promoted on such a small ceiling.

The next expert representation should preserve the successful fine-grained
residual idea at the hidden/FFN level: an i16 hidden adapter or low-rank FFN
residual initialized at zero, added to the base FFN rather than replacing it.
That keeps the trunk stable, avoids the i8 replacement cliff, and gives the
router a materially larger conditional function to select.

The consolidated evidence is
`data/experiments/literary-shared-trunk-moe-v1/shared-trunk-report.json`.

### Context-dependent hidden experts

The next adapter moves specialization before the shared output head. Each
author gets 128 learned Q15 gains applied as a zero-initialized residual to
the transformer's final contextual state. The trunk and output projection are
still frozen, so one transformer forward supplies all three branches.

All three adapters improve their own full training split, but the frozen
42,120-target comparison remains narrow:

| Shared-trunk route | Accuracy | Mean probability error | Mistakes |
|---|---:|---:|---:|
| Best fixed hidden expert | 158‰ | 58,375 | 35,457 |
| Prompt hidden oracle | 158‰ | 58,371 | 35,454 |
| 16-token hidden oracle | 158‰ | 58,349 | 35,453 |
| Per-token hidden oracle | 158‰ | 58,256 | 35,451 |

Token-level selection now uses every expert and improves the best fixed branch
by 119 Q15 plus six mistakes. That is twice the bias-adapter ceiling, but still
too small to justify training another router. The result supports token-level
subdivision at the expert boundary while rejecting isolated token corpora:
shared attention builds context first, then small residual experts specialize
the contextual token state.

The consolidated evidence is
`data/experiments/literary-shared-trunk-hidden-moe-v1/shared-trunk-hidden-report.json`.
The next capacity gate is a zero-initialized low-rank residual that mixes
hidden channels instead of scaling them independently.

### Fixed-projection low-rank result

The low-rank probe uses a deterministic signed projection from 128 contextual
channels into a small latent, followed by a learned zero-initialized expansion
back into the hidden state. A nine-run rank/rate swarm found that ranks 4 and 8
could quantize to zero updates for some projections. Rank 32 at learning rate
16,384 learned without saturation and was promoted to three author runs. Each
expert has 4,096 Q15 parameters while retaining one shared trunk forward.

| Shared-trunk route | Accuracy | Mean probability error | Mistakes |
|---|---:|---:|---:|
| Best fixed low-rank expert | 155‰ | 58,656 | 35,562 |
| Prompt low-rank oracle | 156‰ | 58,644 | 35,528 |
| 16-token low-rank oracle | 157‰ | 58,572 | 35,481 |
| Per-token low-rank oracle | 157‰ | 58,421 | 35,471 |

The 235-Q15 token ceiling and 91 avoided mistakes prove more conditional
diversity than diagonal scaling. However, even its target-aware token oracle is
46 Q15 worse than the best fixed diagonal expert at 58,375. A learned router
over these branches therefore cannot win the current overall comparison and is
not promoted. The next adapter should preserve the diagonal gains and add the
low-rank residual on top, then repeat the same frozen oracle gate.

The consolidated evidence is
`data/experiments/literary-shared-trunk-low-rank-moe-v1/shared-trunk-low-rank-report.json`.

### Hybrid expert and learned-router result

The promoted composition keeps each author's learned diagonal gains and adds a
rank-16 low-rank residual. It starts byte-for-byte from the stronger diagonal
function, trains only 2,048 new weights, and uses 2,176 parameters per expert
including the frozen gains. Rank 32 produced the same integer update as rank 16
in the probe, so the smaller form was selected.

| Hybrid route | Accuracy | Mean probability error | Mistakes |
|---|---:|---:|---:|
| Best fixed hybrid expert | 158‰ | **57,818** | 35,444 |
| 16-token hybrid oracle | 158‰ | 57,803 | 35,440 |
| Per-token hybrid oracle | 158‰ | **57,670** | 35,437 |
| Learned target-blind token router | 158‰ | 57,818 | 35,444 |

The fixed hybrid improves the prior fixed diagonal result by 557 Q15. That is
the strongest shared-trunk expert result in this experiment. The oracle uses
all three branches, but its remaining ceiling is only 148 Q15 and seven
mistakes.

Three target-blind router replicas were trained from two contextual-hidden
views and one full view. Calibration selected consensus weights `3/2/2` with
zero hysteresis, but final routing still sends 42,096 of 42,120 tokens to
Blake. It changes no mistakes and adds 1,496 total Q15 error versus fixed
Blake. Four hard-label diagnostics also collapsed, this time to Crowley, with
substantially larger regret. The hybrid experts are promoted; this router is
not. More router tuning is not justified until experts expose a larger
conditional utility gap.

Consolidated evidence is
`data/experiments/literary-shared-trunk-hybrid-moe-v1/shared-trunk-hybrid-report.json`,
with the router-only audit in `learned-token-router-report.json` beside it.

### Eight-head optimizer swarm

The promoted `small-h8-d128-ff256` profile was tested as three whole-model
leaves trained on the same balanced mixed corpus with Adam step shifts 3, 4,
and 5. Two diversity controls failed: author-isolated leaves underfit even at
2,048 windows, and later disjoint mixed-corpus offsets were dominated by the
first 512-window leaf. Optimizer-scale diversity is the promoted swarm source;
corpus isolation is not.

Sequence length 64 leaves 36,200 comparable frozen targets:

| H8 route | Accuracy | Mean probability error | Mistakes | Switches |
|---|---:|---:|---:|---:|
| Best fixed shift 5 | 148‰ | 56,364 | 30,820 | 0 |
| Learned 16-token span | 149‰ | 56,337 | 30,805 | 76 |
| Learned per-token | **151‰** | **56,190** | **30,718** | 1,633 |
| Per-token oracle | 209‰ | 52,261 | 28,620 | 23,142 |

Both target-blind routers are calibration-selected and improve frozen final
utility. Token routing gains 174 Q15, three per-mille accuracy points, and 102
mistakes; span routing retains a smaller 27-Q15 gain while cutting route
switches by more than twentyfold. The oracle uses all three leaves, but the
learned routers mostly choose shift 5, so better features or joint routing
losses still have substantial room.

This is the strongest evidence for the requested architecture: many small
runs can expose useful alternatives, and a neural router can select them below
the corpus level. The next efficiency step is to distill H8 optimizer-scale
differences into shared-trunk residual experts so routing no longer requires
three full transformer forwards.

Consolidated evidence is
`data/experiments/literary-h8-swarm-v1/report.json`.

### Shared-trunk H8 residual curriculum

Directly continuing the trained H8 trunk exposed the remaining i8 optimizer
cliff. Adam shift 5 changed whole i8 units and regressed both stage and holdout
loss; shifts 6–8 made zero updates at batch 16. Smaller batches accumulated
updates but still moved in the wrong direction. The loss guard rejected every
parameter-changing continuation, preserving the checkpoint but making no
progress.

The successful replacement freezes the H8 trunk and trains a resumable
rank-16 i16 residual in 512-window stages. Each stage sweeps learning rates
64/256/1024, selects on the same frozen holdout, and becomes the starting point
for the next stage. Stage 8 is rejected because every candidate increases
exact holdout error, so the chain stops at stage 7.

| Curriculum checkpoint | Holdout accuracy | Holdout mean error | Mistakes |
|---|---:|---:|---:|
| Frozen H8 trunk | 189‰ | 53,753 | 2,763 |
| Stage 2 residual | 190‰ | 53,582 | 2,757 |
| Stage 4 residual | 191‰ | 53,539 | 2,755 |
| Stage 7 residual | **191‰** | **53,497** | **2,755** |

On the 36,200 frozen literary targets, stage 7 improves the original fixed H8
leaf from 56,364 to 55,991 mean error, from 148‰ to 153‰ accuracy, and avoids
164 mistakes. A stage-1/2/7 token oracle reaches 55,895, only 96 Q15 beyond
fixed stage 7, so another learned router is not promoted. Shared execution
still reduces H8 trunk forwards from 108,600 to 36,200.

Adapter-aware greedy generation and deterministic top-8 sampling are now
implemented. Greedy output collapses to spaces; sampled output remains mostly
spaces with sparse letters and fails the explicit non-space, alphabetic,
vocabulary, and repetition gate. Next-token improvement is therefore promoted
while prose generation is explicitly not promoted.

An `rms-norm` Adam scope also trains only the 512 i16 gamma values inside the
two H8 blocks. It passes the mechanical isolation test and slightly improves
mixed holdout error when composed with the residual chain (53,497 to 53,496 at
stage 6), but its frozen literary total is slightly worse than the non-RMS
stage-7 winner. The scope is kept as a valid fine-grained internal mechanism;
the composed checkpoint is not promoted over stage 7.

Consolidated evidence is
`data/experiments/literary-h8-curriculum-v1/report.json`.

### Per-block i16 experts

The next precision mechanism inserts one low-rank residual after every frozen
transformer block instead of adapting only the final hidden row. Its fixed
sign down projection and learned i16 Q15 expansion operate on every token row;
backpropagation crosses later frozen attention/MLP blocks, so lower-layer
experts receive exact integer gradients. Artifacts bind the trunk hash, layer
count, rank, projection seed, and residual shift. A zero artifact is byte-exact
with the base trunk.

An unscaled probe exposed millions of hidden clamps, so residual scaling was
made explicit rather than hidden in a training recipe. Stable residual shifts
4 and 8 eliminated weight clipping across the useful rank-8 sweep. On the
3,407-window untouched holdout, however, the best point scored 183,137,562
total error versus the trunk's 183,137,284, with the same 2,763 mistakes. Two
rank-32 probes also failed to beat the trunk. Deterministic top-8 samples had
only 42–72 non-space bytes per thousand, 5–6 distinct bytes, and space runs up
to 95, so prose remains rejected.

This separates two issues: trainable sub-i8 precision now exists inside the
network, but one mixed residual is not a useful expert decomposition. The next
experiment should train many provenance-labelled author/span block experts,
then learn target-blind token/span routers over their conditional utility.
Consolidated evidence is
`data/experiments/literary-h8-block-expert-v1/report.json`.

### Author block experts and recursive neural routing

The first provenance-labelled block swarm uses 24 source chunks per author for
leaf training, four disjoint chunks per author for router training, four for
calibration, and the original three holdout chunks per author for final test.
The extractor validates source hashes and reproduces the canonical ASCII-lower
corpus and holdout tokens exactly. Author labels select training provenance but
are absent from inference features.

Each author leaf is a 2,048-parameter rank-8 per-block Q15 residual over the
same frozen H8 trunk. The final zero-expert comparison is:

| Route | Delta vs trunk Q15 | Mistake delta |
|---|---:|---:|
| Fixed Crowley | +3,118 | 0 |
| Fixed Shakespeare | +824 | 0 |
| Fixed Blake | +1,782 | 0 |
| Prompt oracle | -237 | 0 |
| Span-16 oracle | -4,339 | 0 |
| Token oracle | -8,023 | 0 |

Three child neural routers consume 32 shared-trunk features plus nine rolling
prior-token utility features. Their outputs then feed a second integer neural
router together with the shared hidden features, giving the requested
router-of-routers topology. Calibration selects epochs and consensus settings;
final data is never used for selection.

Neither level is promoted. Child token routing is 1,004 Q15 worse than the
trunk; the recursive token root is 1,258 worse. The recursive span root is the
closest at +863 Q15 but sends 96.6% of final tokens to Shakespeare. This is a
measured collapse caused by an expert gap of only a few Q15 per token, not
evidence that recursive routing is inherently ineffective. The next swarm
should divide experts below author level—contiguous spans or token-context
clusters—and require a larger calibration oracle ceiling before training
routers. Consolidated evidence is
`data/experiments/literary-h8-author-block-swarm-v1/report.json`.

### Cross-author surface-context clusters

The next triad removes author identity entirely. It partitions 524 disjoint
512-token leaf spans by deterministic target-blind k-means over 32 surface
features. All three resulting clusters remain cross-author and contain enough
data for matched 512-window expert runs. Rate selection again uses only each
leaf's own training objective.

The frozen-final comparison is:

| Context route | Delta vs trunk Q15 | Mistake delta |
|---|---:|---:|
| Best fixed cluster | +1,244 | 0 |
| Span-16 oracle | -1,789 | 0 |
| Token oracle | -4,462 | 0 |
| Target-blind centroid token | +2,172 | 0 |
| Target-blind centroid span | +2,244 | 0 |

Calibration token-oracle gain is 9,581 Q15, below the already-rejected author
swarm's 13,117. The explicit router gate therefore prevents another neural
router sweep. This saves compute while preserving the negative evidence:
surface similarity does not align with the frozen model's residual errors.
The next expert labels should come from model-native residual or gradient
signatures, with inference still performed by a separately learned
target-blind hidden-state router. Consolidated evidence is
`data/experiments/literary-h8-context-block-swarm-v1/report.json`.

### Frozen-gradient clusters

The model-native follow-up emits a 32-channel signature for each of the same
524 disjoint 512-token spans: 16 signed final-hidden gradient buckets and 16
gradient-magnitude buckets. Deterministic standardized k-means converges to
cross-author groups of 237, 133, and 154 spans, with mean trunk errors of
53,984, 56,466, and 51,709 Q15. The labels therefore describe how the frozen
model fails, not which author supplied the text.

During this run, exact loss guards revealed that small probability-error
gradients were disappearing before an optimizer step. The block expert had
rounded each sample's Q30 outer product to Q15 too early. It now accumulates
raw products across the whole batch, applies the learning rate, and divides
once with error-feedback residuals. A locked regression test verifies a
nonzero metric-aligned update, per-layer isolation, and non-regression under
the bidirectional loss guard.

The frozen-final comparison is:

| Gradient route | Delta vs trunk Q15 | Mistake delta |
|---|---:|---:|
| Fixed cluster 0 | -1,408 | 0 |
| Fixed cluster 1 | -1,110 | 0 |
| Fixed cluster 2 | -179 | 0 |
| Prompt oracle | -1,969 | 0 |
| Span oracle | -5,448 | 0 |
| Token oracle | -8,462 | 0 |

This promotes gradient-defined per-block leaves: every leaf beats the zero
expert trunk, and the token oracle retains 7,054 Q15 of conditional gain beyond
the best fixed leaf. The target-blind router result is more limited. Three
child routers and a second neural root were selected by calibration loss for
both token and span labels. Child token consensus collapses to fixed cluster 0;
the best recursive route gains 1,247 Q15 over the trunk but remains 161 Q15
worse than the fixed winner. Learned recursive routing is not promoted.

Top-8 samples still contain only about 58--62 non-space bytes per thousand and
six distinct bytes, so prose remains rejected. The next run should resume many
short metric-aligned leaf stages, keep only independently measured gains, and
retrain routers only after that process widens the conditional utility gap.
Consolidated evidence is
`data/experiments/literary-h8-gradient-block-swarm-v1/report.json`.

## Promotion gates

- Recursive top-two must beat the best single leaf on frozen final data.
- Routing regret and oracle top-two coverage must be reported at every level.
- No author may be starved; utilization and path entropy are required.
- Router replicas must disagree on at least some calibration rows, or the
  router swarm has collapsed into redundant copies.
- Every result must include model, dataset, route, and logits hashes.
- Author labels may be reported as a bootstrap baseline, never as proof of
  utility-based routing.

Bounded integer Adam and real RMSNorm are now implemented. The next routing
artifact is a learned span/token dispatcher integrated into the transformer
block rather than a standalone prompt router. Its labels come from the emitted
per-target child-loss table, with author identity excluded from the inference
features.
