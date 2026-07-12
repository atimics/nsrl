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
