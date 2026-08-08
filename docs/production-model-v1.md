# Production model v1

`production-model-v1` introduces `NSRLPM1`, a variable-vocabulary integer
decoder artifact separate from the frozen byte-vocabulary MT5/MT6 formats. It
is bound to one `NSRLBPE1` tokenizer hash and consumes tokenizer-bound
`NSRLTOK1` u32 streams.

## Implemented gates

The runtime now provides:

- exact dynamic parameter accounting for the frozen p10m, p20m, and p30m
  shapes;
- deterministic integer initialization for embeddings, causal linear
  attention, gated MLP, RMS vectors, output weights, and output bias;
- checksummed `NSRLPM1` serialization and strict shape validation;
- tokenizer-hash and vocabulary validation when loading `NSRLTOK1` streams;
- integer forward execution over u32 subword contexts;
- unassisted autoregressive `NSRLPM1` generation with tokenizer-bound prompts,
  bounded rolling context, greedy or deterministic seeded top-k selection,
  EOS handling, special-token masking, and exact per-step replay traces; and
- a bounded output-head perceptron smoke trainer with model-hash and saturation
  evidence;
- full quantized backpropagation through embeddings, attention projections,
  MLP projections, RMS vectors, output weights, and bias;
- a checksummed residual-SGD optimizer with four-window batches, exact
  epoch/window cursor state, and one carried i64 residual per parameter;
- per-parameter-group gradient, carry, update, movement, and saturation
  diagnostics;
- explicit residual-accumulator overflow counts globally and by parameter
  group;
- phase-aware interval liveness state bound to the exact preceding model hash,
  with a chained event-history hash and hard output, trunk-gradient, and
  trunk-update deadlines;
- a same-shape NumPy float reference runner mapped from the integer
  initialization and trained on the same bounded windows;
- a durable 2,048-window matched integer/float readiness runner with
  baseline-relative held-out gates and exact midpoint replay for both lanes.
- an executable forward/training numeric contract that records tensor
  exponents, rounding rules, output-backward damping, real update coefficients,
  and worst-case projection/attention/RMS/softmax accumulator bounds;
- validated p10m, p20m, and p30m profile geometry for causal linear attention;
- a normalization-independent, logit-shift-invariant canonical integer NLL
  evaluator that scores the pre-reciprocal base-2 exponent weights in Q20 and
  reports millibits without Q15/Q31 target-probability collapse;
- a faithful float relaxation using base-2 softmax, bit NLL, and the integer-
  mapped RMS epsilon `2^-30`, while retaining an explicit `legacy-v1` mode for
  frozen artifact replay; and
- a deterministic integer-lattice lane audit that uses one coordinate sample
  for the existing normalized, mass-corrected normalized, raw reciprocal-free,
  late-RHU, and seeded-stochastic proposal oracles. It reports exact
  stored-parameter `-1`/`+1` canonical-NLL changes separately on the proposal
  surface and on a document-separated (or full-context-gap) transfer surface,
  with rescue attribution and a shared random-direction control. Raw exponent
  lanes are explicitly labeled sample-reweighted surrogates. Trace schema v5
  adds opt-in source-specific rescue strata and strict balanced document blocks
  while retaining byte-compatible v4 output for the frozen v1 runner. Schema
  v6 adds an evaluation-only normalized mass-corrected no-rescue lane and an
  explicit causal comparison without changing the v2 sample. All schemas bind
  the effective per-group shifts, output-backward shift, probability precision,
  and normalization method; and
- a rank-two Boolean-jet audit that replays a fingerprint-bound alignment move
  set, evaluates the exact empty/trunk/head/joint cube on proposal and
  document-disjoint transfer surfaces, and reports Möbius coefficients,
  visibility, saturation, minimizing vertices, and escape order without
  mutating optimizer state; and
- a prospective Boolean-jet confirmation audit that requires an explicit move
  manifest plus bound model/tokenizer/stream hashes, evaluates declared unused
  document ranges, reports per-document conditional coefficients and an exact
  paired sign test, and never authorizes an optimizer change by itself; and
- a proposal-only rank-six atomic structure audit that binds source and binary
  hashes plus the corpus index, evaluates all 64 vertices before document 72,
  records Q20/Q32 coefficients, sharp interaction-tail and representation
  bounds, exchange defects, all 720 elimination orders, and boundary masking,
  and explicitly disables source-cluster inference when fewer than two source
  clusters are present.

The alignment trace is calibration evidence. It hard-codes optimizer
authorization to false, rejects a sampled boundary coordinate instead of
silently dropping it, and hashes the complete ordered per-window output
gradient vectors for source-equivalence claims. Summary statistics alone are
not used as an equivalence proof.

Inspect the numeric contracts without allocating a model:

```bash
cargo run -p nsrl-train --bin nsrl-production-model -- \
  numeric-contract --profile p10m
```

Exercise native text generation independently of the training stream:

```bash
cargo run --release -p nsrl-train --bin nsrl-production-model -- generate \
  --tokenizer data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  --model data/experiments/production-model-v1/CANDIDATE.nsrlpm \
  --prompt "The king" \
  --max-new-tokens 64 \
  --top-k 1 \
  --trace data/experiments/production-model-v1/generation-trace.json
```

`top-k 1` is canonical greedy decoding. Larger values use Q15 probability
weights and the declared integer seed. Generation currently replays the bounded
context through the exact production forward path for every token. This closes
the correctness and traceability gap; an incremental cached path and a passing
`open-generation-v1` result remain separate performance and quality gates.

The `numeric-contract` command emits separate forward and training JSONL
contracts. Run canonical
evaluation with `evaluate-canonical`; frozen `evaluate` remains the legacy Q15
surface so v1 checkpoints remain reproducible. The bounded lattice calibration
surface is exposed through `gradient-alignment-audit`; `--max-windows` binds the
proposal batch and `--transfer-windows` binds the separated transfer batch
(`--acceptance-windows` remains a CLI alias). The trace records document,
context-start, and target offsets. Proposal-surface finite differences measure
backward directional fidelity; transfer-surface differences measure held-out
transfer and are not interpreted as gradient correctness. The command rejects
a stream that cannot supply both surfaces with the declared separation. Use
`--documents-per-surface N --rescue-stratified-sampling` to require balanced,
strictly disjoint document blocks and add one shared coordinate sample per
source-specific rescue stratum. Add
`--include-mass-corrected-no-rescue` to evaluate a source-matched plain-RHU
control without allowing that control to alter the frozen coordinate sample.
Add `--include-systematic-fixed-mass` to expose separately labeled
`K=2^15`, `2^16`, and `2^18` systematic lanes. They use MJ-05 Q47 weights, a
contract-seeded phase, token-ID order, and runtime exact-mass/zero-sum checks.
They are measurement lanes, not a training default.
Use `boolean-jet-rank-two-audit` with expected trunk/head move counts and a move
fingerprint to evaluate the corresponding four-vertex cube. Collisions,
repeated actions, and boundary-saturating moves are rejected.
That command preserves the frozen legacy JSON. Use `boolean-jet-audit` with the
same arguments for the hardened `nsrl.production_boolean_jet.v1` schema with
the first-class manifest, objective specification, document losses, vertex
hashes, gamma-one, and reconstruction evidence.
Use `boolean-jet-confirmation-audit` only with a pre-frozen nonzero manifest
hash, explicit move atoms, disjoint document ranges, and a minimum informative
document count. The checked-in v1 confirmation falsifies the post-hoc
trunk-after-head transfer effect; it is a negative result, not a promoted
optimizer rule.

`boolean-jet-freeze-matched-control` constructs a control manifest without
evaluating the objective. It matches parameter group, atom count, and stored
width, then selects distinct boundary-safe coordinates and signs from a frozen
seed. Explicit `--control-move` atoms add a paired `H -> HR` comparison. This
extension must be frozen before its evaluation documents are inspected; it is
not retroactively part of v1. The later document-Ising contract used documents
136--199 for a genuinely new, frozen comparison of pairwise MAP, quenched Gibbs
magnetization, and singleton-probe routing. Documents 200--212 remain sealed.
That follow-up is same-source document evidence and does not authorize an
optimizer or scaling change.

The subsequent prospective cross-source contract retains the same control mask
`47` and candidate mask `59`, but calibrates the interaction residual at source
level. It freezes one ebook per distinct author, one hash-sampled passage per
source, 16 fitting panels, 39 calibration panels, and 16 untouched evaluation
panels. The rank-38 correction is `4,326` Q32. All 16 evaluation panels are
covered; 5 panels fire; all 5 exact contrasts are favorable; and their aggregate
is `-40,769` Q32. This is bounded evidence on the frozen English Project
Gutenberg frame, not arbitrary-source or whole-book coverage, and it still
authorizes neither an optimizer change nor paid scaling.

The M4 extension freezes 26 sources in each of four families: Federal Register,
new Gutenberg books disjoint from M3, RFCs, and Europe PMC open-access science.
Whole-source roles are 3 fitting, 19 conformal calibration, and 4 untouched
evaluation per family. Each source contributes four nonoverlapping
quartile-stratified passages, and the family score is their maximum. The
rank-19 family corrections are `2,326`, `2,141`, `4,307`, and `4,272` Q32.
Fourteen of sixteen evaluation panels are covered; both failures are
Gutenberg, so the frozen overall verdict is
`coverage_inconclusive_no_promotion`. Twelve evaluation passages fire across
Federal Register, Gutenberg, and RFC; all twelve improve, none is unsafe, and
net held-out improvement is `63,541` Q32. Federal Register and RFC pass their
frozen family gates, Gutenberg is withheld by coverage, and science abstains.
This is not an overall multi-family promotion and changes no optimizer or
scaling authorization.

The frozen p10m smoke artifact has 9,317,632 parameters and is bound to
tokenizer hash `0xf4fe71d93c438c1a` and train-stream token hash
`0x97e5254c31c27bda`. Eight windows move from eight mistakes to zero with eight
updates, zero weight saturation, and zero residual saturation. The 13 MB model
artifacts stay in ignored experiment storage; their SHA-256 and internal model
hashes are frozen in `benchmarks/production-model-v1/p10m-smoke.json`.

Reproduce it with:

```bash
scripts/run-production-model-v1-smoke.sh
scripts/run-production-full-train-v1-smoke.sh
scripts/run-production-float-twin-v1-smoke.sh
scripts/run-production-integer-stabilization-v1.sh
scripts/run-production-kv-scaling-readiness-v1.sh
scripts/run-production-gate-boundary-preflight-v1.sh
python3 scripts/benchmark-production-training-v1.py
node scripts/freeze-production-model-v1.mjs --check
node scripts/freeze-production-full-train-v1.mjs --check
node scripts/freeze-production-float-twin-v1.mjs --check
node scripts/freeze-production-integer-stabilization-v1.mjs --check
node scripts/freeze-production-stabilized-pilot-v1.mjs --check
node scripts/freeze-production-liveness-audit-v1.mjs --check
node scripts/check-production-training-liveness-self-test.mjs
node scripts/check-production-optimizer-residual-analysis-self-test.mjs
node scripts/freeze-production-trunk-unlock-preflight-v1.mjs --check
node scripts/freeze-production-k-stabilization-preflight-v1.mjs --check
node scripts/freeze-production-kv-boundary-pilot-v1.mjs --check
node scripts/freeze-production-kv-scaling-readiness-v1.mjs --check
node scripts/freeze-production-gate-boundary-preflight-v1.mjs --check
node scripts/check-production-model-v1.mjs
node scripts/check-production-optimization-v1.mjs
```

The optimized full-backward p10m checkpoint runs four four-window optimizer
steps. All 13 parameter groups move, mistakes improve from 8 to 7, and both
gradient and weight saturation are zero. A run interrupted after one optimizer
step resumes to byte-identical model and optimizer artifacts. The optimizer
artifact is about 71 MiB because it retains exact residuals for all 9,317,632
parameters.

The matched float twin uses recurrent causal linear attention in both forward
and backward passes, reuses gradient buffers, and follows the same four-window
batch schedule. It moves all 13 groups, remains finite, reduces mean loss from
9.011 to 8.904, and moves from 8 mistakes to 0. A locked self-test compares the
recurrent attention forward and backward results with the explicit quadratic
reference.

The local ARM64 preflight measures one complete p10m forward/backward/update at
contexts 4, 16, 64, and 256. The frozen sample ranges from 0.63 to 4.17 seconds
for integer and 5.28 to 5.46 seconds for float as context grows. These
single-sample timings include process startup, serialization, and evaluation,
so they are engineering bounds rather than capacity forecasts.

## Current boundary

The full backward, float-twin, and pre-pilot optimization gates are complete.
The integer backward still uses explicit straight-through rules at internal
quantization dead zones, while parameter updates carry sub-quantum gradients
in residual state instead of forcing one-unit steps. The float twin remains a
NumPy reference rather than an accelerator runner. Neither bounded smoke is a
language-quality result.

The production artifact can now generate subword continuations with an exact
incremental causal-linear-attention cache. A parity test compares every prefix
through the configured training context against the original full forward path.
The frozen p10m development run uses 405,504 bytes of cache state and 10,240
bytes of workspace; all 60 samples continue beyond the 256-token training
context with zero residual saturation. This closes full-window replay as a
serving-correctness gap, but no latency or tokens-per-second promotion budget is
claimed yet.

No candidate has passed `open-generation-v1`. The first complete p10m row is
frozen at `benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json`: the
serving and provenance gates pass, while all six measured generation-quality
gates fail. Candidate modeling is 3,687 millibits per original UTF-8 byte, but
the required baseline matrix, float-twin retention, blinded human comparison,
hidden-panel run, and candidate proof binding remain absent.

The production trainer now has a causal-suffix mean objective. For a
power-of-two suffix, it evaluates every selected causal row, keeps full-scale
integer gradients through the transformer, and applies the exact mean as an
additional parameter-update shift. Target count and the optional embedding
learning-rate boost are bound into resumable optimizer state. The legacy
one-target path remains byte-identical to the pre-change model, optimizer, and
trace artifacts.

The last passing context-64 prerequisite is
`benchmarks/production-model-v1/p10m-causal-sequence-scale-v3-bias-r3.json`.
It supervises 131,072 targets across 2,048 corpus-spread windows, improves
canonical 512-window development NLL by 3,580 millibits and test NLL by 18,939,
moves all eleven trunk groups, records zero saturation, and byte-replays its
second half from the midpoint. Independent output-bias, per-layer saturation,
parameter-delta, and optimizer-residual audits repaired two numeric failure
modes: output-bias domination and layer-3 O-projection overflow.

This is valid generalization, numeric-health, and restart evidence, not a
language-quality promotion. The corresponding generation checkpoint remains
red at 999 per mille worst repetition, 1 per mille minimum unique four-gram
share, zero minimum entropy, and zero context use and distractor resistance.
All 12 raw audited prompts enter one-token feedback loops despite prompt-specific
hidden and logit states. A corpus-spread rollout-divergence audit further shows
zero teacher-forced top-one matches across 128 held-out continuation positions,
mean correct-token rank 2,426, mean correct-token Q15 probability 5, zero
free-running reference matches, and 958 per mille self-loop transitions. The
failure therefore starts in next-token ranking rather than only after exposure
to generated history. Equal-size context swaps also move logits about five
times more for the older half than the recent half, which is a diagnostic signal
for context weighting rather than a promotion claim.

The prospectively frozen fourfold-coverage test is complete and rejected.
`benchmarks/production-model-v1/p10m-causal-sequence-scale-v4.json` supervises
524,288 targets across 8,192 corpus-spread windows. Development NLL improves by
54,691 millibits, but test NLL regresses by 6,033 millibits. K accumulates 3,742
weight saturations in the first half and 171 in the second, while the selected
candidate also exposes 22 inference residual saturations on one adversarial
prompt. Exact midpoint replay and all-group movement still pass. The frozen
gate therefore withholds both preflight promotion and open-generation rerun.

Scale-v4 does not repair conditional ranking. Its held-out rollout audit still
has zero teacher-forced top-one matches across 128 positions; mean target rank
improves only from 2,426 to 2,265 and mean target probability from 5 to 6 Q15.
Free running becomes a one-token loop on all 120 transitions. Equal-size older-
prefix swaps now move logits 5,693 per mille as much as recent-suffix swaps.
Because the causal-suffix objective spends 56 of every 64 losses on contexts
shorter than 57 tokens, while evaluation and generation score the final
context-64 row, the next prospective experiment changes the target distribution
to the last eight causal rows. It holds corpus-window coverage fixed and slows
the K base update by two bits; this tests objective alignment and numeric health
instead of repeating the falsified scale-only intervention.

That recent-tail experiment is complete and rejected, but it identifies the
first large conditional-learning signal. The frozen checkpoint at
`benchmarks/production-model-v1/p10m-causal-tail-context-v1.json` improves
development NLL by 445,246 millibits and test NLL by 403,430. Its rollout audit
records 2/128 teacher-forced top-one matches, mean target rank 1,711, and mean
target probability 27 Q15, crossing all three prospectively frozen ranking
thresholds. Scale-v4 had 0/128, rank 2,265, and probability 6. Restricting loss
to positions with 57--64 visible tokens therefore materially improves the
context-64 objective and next-token ranking.

It is not a numerically valid candidate. The less-cancelled tail gradients make
the unchanged group learning coefficients much too large. By the 4,096-window
midpoint, all public prompts already saturate internal residuals; the second
half then has zero nonzero gradients in Q/K/V/O and every MLP projection, while
final RMS accumulates 3,495 weight saturations. The selected model records
2,285,126 rollout residual saturations and still self-loops on all 120 free-
running transitions. Exact midpoint replay, dev improvement, and test
improvement pass, but the training/inference saturation and repetition gates
correctly reject promotion and generation.

The prospectively frozen contract is a 2,048-window numeric preflight at
`benchmarks/production-model-v1/p10m-causal-tail-stability-v1-contract.json`.
It preserves the successful tail-8 objective and damps each group by the
power-of-two ceiling of its midpoint movement excess relative to the last
healthy scale-v4 reference, with extra boundary margin. It must move all eleven
trunk groups, improve dev and test, retain exact replay, and record zero
training and public-manifest inference saturation before a fresh 8,192-window
tail run can be frozen. It cannot authorize open generation itself.

That stability preflight is complete and numerically successful, but it is not
promotion-eligible. Development and test NLL improve by 9,431 and 4,430
millibits, exact midpoint replay passes, and training, held-out evaluation,
rollout, context, and all 12 public-manifest prompts record zero saturation.
The older-prefix/recent-suffix logit-effect ratio also falls from roughly five
times to 1,644 per mille. The gate still rejects the candidate because MLP RMS,
Q, up, and gate remain below their integer update boundaries.

The prospective residual-boundary repair is frozen at
`benchmarks/production-model-v1/p10m-causal-tail-unlock-v1-contract.json`.
Audits of the bound final optimizer state locate the first nonzero thresholds:
49 MLP-RMS coordinates at effective shift 5, 59 Q coordinates at shift 22,
36 up coordinates at shift 21, and 60 gate coordinates at shift 21. The repair
changes only vector 14 to 12, Q 23 to 19, and up/gate 20 to 18 at the CLI; all
already-live matrix schedules remain unchanged. Because vector is shared by
the two RMS groups, attention RMS also changes from effective shift 13 to 11.
This remains a bounded numeric preflight and cannot authorize open generation
or hidden-panel access.

Unlock-v1 is complete and remains one boundary short of promotion. The intended
MLP-RMS, Q, up, and gate groups move by 68, 41, 54, and 71 L1 units; development
and test NLL improve by 9,891 and 5,192 millibits; exact replay and every zero-
saturation gate pass. O alone remains unchanged, so the all-trunk gate rejects
the candidate. Its final optimizer residual audit finds 574 coordinates at
effective shift 18 and none at shift 19. The prospectively frozen unlock-v2
contract changes only the O CLI shift from 16 to 15 and repeats the same bounded
preflight before any full-tail schedule is authorized.

Unlock-v2 passes that bounded gate. All eleven trunk groups move, development
and test NLL improve by 6,822 and 12,161 millibits, exact midpoint replay is
byte-identical, and training, evaluation, rollout, context, and all public-
manifest inference checks remain at zero saturation. O moves 2,430 L1 units,
well below the 123,029-unit last healthy scale-v4 reference.

The full 8,192-window schedule is prospectively frozen at
`benchmarks/production-model-v1/p10m-causal-tail-full-v1-contract.json`. It
changes no learning or numeric coefficient from unlock-v2 and supervises
65,536 recent-tail targets over 2,048 optimizer steps. Unlike the numeric
preflights, its postflight quality gates restore the original tail-context
ranking, repetition, context-use, and zero-saturation thresholds. Passing all
of them authorizes only the public `open-generation-v1` rerun; hidden-panel
access, paid scaling, and final promotion remain separately gated.

The full stable run is complete and passes its numeric/loss gate but fails its
quality gate. Development and test NLL improve by 50,131 and 39,500 millibits;
all eleven trunk groups move; replay is exact; and every training and inference
saturation count is zero. Mean target rank 2,106, prefix/suffix effect ratio
1,475 per mille, and 7/12 prompt self-loops pass. Teacher-forced top-one remains
0/128, target probability is 5 Q15, free running self-loops on all 120
transitions, and only three prompt-greedy tokens appear, so the public open-
generation rerun remains unauthorized.

The bound movement and residual evidence localizes a head imbalance. Output
matrix movement is only 4,031 L1 units versus 508,316 in the last healthy
scale-v4 run, while bias movement is 63,536. At the probability-adjusted output
shift 41, 83,279 final residual coordinates cross the update threshold. The
prospective head-balance continuation freezes all eleven trunk groups, makes
the output matrix three bits faster, makes bias three bits slower, and trains
only those two groups for 2,048 spread windows. Its contract is
`benchmarks/production-model-v1/p10m-causal-tail-head-balance-v1-contract.json`;
it retains the full strong quality gate and cannot open hidden evaluation.

The isolated head-balance continuation passes loss, replay, health, and
isolation, but fails the quality hypothesis. Output moves 72,172 L1 units while
bias moves 515 and every trunk group remains unchanged. Development and test
NLL improve by 6,661 and 6,166 millibits, but mean target rank regresses from
2,106 to 2,202, target probability reaches only 6 Q15, all 120 free-running
transitions still self-loop, and all 12 public prompts self-loop. A stronger
head alone therefore cannot recover conditional language behavior.

The next prospective continuation isolates representation learning at
`benchmarks/production-model-v1/p10m-causal-tail-representation-v1-contract.json`.
It returns embeddings, K, V, and O to their last zero-saturation scale-v2
effective shifts and freezes the output head plus every other trunk group.
The bound full-v1 residuals predict 1,558 embedding, 188,582 K, 104,370 V, and
31,295 O coordinates at those thresholds, spanning every attention layer. It
uses 2,048 spread windows and retains the same strong public quality gates.

The controlled p10m train/dev pilot completed on a c8g.2xlarge Graviton runner.
Its frozen schedule used 1,024 train windows and 256 held-out dev windows at
context 64, with durable chunking, a midpoint replay, and concurrent
integer/float lanes. The replay finished with byte-identical model and optimizer
artifacts, proving that interruption recovery is exact.

The training result is not promotion-eligible. Float held-out loss improved
slightly from 13.000 to 12.988 bits/token, while integer held-out loss regressed
from 13.000 to 31.731 bits/token. Integer training accumulated 25,810 gradient
saturations and 83,163 weight saturations; K accounted for 76,102 parameter
saturations, and one durable chunk retained gradients only for final RMS,
output, and bias before the full path revived. The frozen checkpoint is
`benchmarks/production-model-v1/p10m-pilot.json`.

The bounded integer shift-stabilization preflight is now complete and eligible
for a controlled replay. The trainer supports independent Q, K, V, O, up,
gate, and down update shifts, records the effective 13-group schedule, and
binds that schedule plus the output backward shift into resumable optimizer
state. A deterministic one-unit output initialization and a finer forward
output scale activate the trunk without immediately disturbing held-out
predictions; the explicit straight-through backward scale remains separately
frozen.

The 256-window validation improves the fixed training probe from 64 to 45
mistakes and held-out loss from 13.065 to 13.062 bits/token. Every parameter
group receives nonzero gradients, the output projection moves, and both
gradient and weight saturation remain zero. The frozen result is
`benchmarks/production-model-v1/p10m-stabilization.json`.

The first scale-preserving replay attempt added two update-shift bits to every
group. Its durable gate stopped after 256 windows because output shift 36 did
not cross an update boundary, leaving only final RMS, output, and bias with
nonzero gradients. Held-out loss stayed flat and saturation stayed zero, so the
early stop reduced this to a five-minute schedule-discovery run. The frozen
attempt is `benchmarks/production-model-v1/p10m-stabilized-pilot-attempt-1.json`.

The corrected v2 contract retains the proven output-unlock shift of 34 and
applies the two-bit scaling adjustment only to the still-locked non-output
groups. Its 1,024-window Graviton replay passed all four durable checks. Every
chunk retained all 13 gradient paths with zero gradient or weight saturation;
integer held-out loss improved from 13.065 to 13.060 bits/token, while the
matched float reference improved from 12.994 to 12.976. Integer finished 6 per
mille behind float, inside the 150-per-mille bound, and the 512-window midpoint
restart reproduced the final model and optimizer byte-for-byte. The frozen
checkpoint is `benchmarks/production-model-v1/p10m-stabilized-pilot.json`.

The follow-up liveness audit ran locally in 16-window probes before allowing
another long runner. With output update shift 34, the first output update lands
in interval 3 (windows 48-64), while all 13 quantized gradient paths first
become active in interval 6 (windows 96-112). Treating those as one event would
therefore reject a healthy warm-up. The phase-aware policy gives output unlock
four intervals and subsequent trunk activation three intervals; after the
trunk is live, any gradient-path loss is immediately fatal. The known-dead
shift-36 control remains locked for all four probes and exits at window 64.

The same audit found a previously silent failure channel: i64 optimizer
residual accumulation used saturating addition without reporting overflow.
Residual saturation is now counted globally and per parameter group and is a
hard liveness failure alongside gradient and weight saturation. Interval state
also rejects skipped intervals or a model hash that does not match the prior
state, and an explicit trunk-update deadline turns persistent sub-quantum
updates into `trunk_update_timeout`. All local probes had zero saturation and
non-increasing held-out loss, but the trunk still had not crossed an integer
update boundary by 256 windows. The frozen evidence is
`benchmarks/production-model-v1/p10m-liveness-audit.json`.

This remains a warm-up/stability result rather than model promotion: only the
output projection crosses an integer update boundary. The next gate is a
bounded trunk-unlock preflight that must make non-output parameter groups move
before its declared deadline without losing the now-proven held-out,
saturation, liveness-state binding, and restart properties. A larger run is not
authorized merely because held-out loss is improving while the trunk remains
unmoved.

The bounded local trunk-unlock preflight is now complete. Instead of sweeping
arbitrary static shifts, it inspected the exact residual-SGD state from the
liveness checkpoint and estimated the smallest one-group action that would
cross an integer update boundary. V was nearest: its maximum accumulated
absolute residual implied shift 30, three bits below the frozen shift 33, with
303 parameters predicted to cross. Output and bias were explicitly excluded
from trunk identity so head-only movement cannot satisfy this gate.

The fresh four-interval run changed only V. V crossed the boundary at the hard
256-window deadline with 269 updates, while output made 1,052 updates across
the run. All 13 gradient paths remained active after warm-up, gradient,
residual, and weight saturation stayed zero, and held-out total improved by 415
millibits (mean 13.065 to 13.063 bits/token). Replaying windows 128-256 from the
midpoint reproduced the final model and optimizer byte-for-byte. The runner
writes interval artifacts atomically and skips complete intervals when
restarted. The contract and frozen evidence are
`benchmarks/production-model-v1/p10m-trunk-unlock-contract.json` and
`benchmarks/production-model-v1/p10m-trunk-unlock-preflight.json`.

This is the first real trunk update, not full trunk training and not a learned
hyperparameter controller. The residual policy is deliberately a bounded
bootstrap that produces state/action/outcome data for a future controller. The
same exact-update gate was then applied to K, the group responsible for 76,102
of the original pilot's 83,163 parameter saturations. Its accumulated residual
predicted a safe boundary at shift 26, six bits more conservative than the
unstable original shift 20.

The isolated K preflight stayed locked for three intervals, then produced
5,184 exact K updates at the declared 256-window deadline. All 13 gradient
paths were active after output warm-up, gradient/residual/weight saturation
remained zero, held-out total improved by 830 millibits, and the midpoint replay
was byte-identical for both model and optimizer. Movement groups, nonzero update
counts, movement L1, and model-hash changes are now required to agree exactly;
an active gradient or large residual alone cannot satisfy reachability. The
contract and checkpoint are
`benchmarks/production-model-v1/p10m-k-stabilization-contract.json` and
`benchmarks/production-model-v1/p10m-k-stabilization-preflight.json`.

That K+V boundary pilot is now complete. Its predeclared local contract runs
1,024 windows in eight durable intervals, requires both groups to move by
window 256 and again after the 512-window midpoint, and permits movement only
in K, V, and output. Both groups met their deadlines, all 13 gradient paths
remained active after unlock, all gradient/residual/weight saturation counters
stayed zero, and held-out total improved by 5,209 millibits (mean 13.065 to
13.044 bits/token). Replaying windows 513–1,024 reproduced the final model and
optimizer byte-for-byte. The contract and checkpoint are
`benchmarks/production-model-v1/p10m-kv-boundary-pilot-contract.json` and
`benchmarks/production-model-v1/p10m-kv-boundary-pilot.json`.

The `p10m_kv_scaling_readiness_review` is now complete. Its predeclared local
contract doubles the horizon to 2,048 windows in eight durable chunks and binds
the stable K/V schedule to a float32 SGD reference with the same integer-mapped
initialization, train/dev streams, window order, context, batch geometry, and
window budget. Integer K, V, and output moved in every chunk, all 13 gradient
paths stayed active, and gradient, residual, and weight saturation remained
zero. Integer held-out total ended 5,209 millibits below initialization; it was
non-monotone after the midpoint but never exceeded the lane baseline. The float
lane moved all 13 arrays and ended 98 mean millibits below initialization. Its
first chunk had a seven-loss-millionth increase hidden below the rounded
millibit gate, so both resolutions are retained in the checkpoint.

Replaying windows 1,025–2,048 from the midpoint reproduced the integer model
and optimizer byte-for-byte and all 13 float tensors exactly. The final integer
mean was 13,044 millibits versus 12,896 for float, a 12-per-mille regression
under the contracted 150-per-mille ceiling. The final residual state identifies
the gated-MLP `gate` projection as the nearest still-unmoving trunk boundary:
shift 25 to 23, with 12 predicted crossings. That is a candidate for a fresh
isolated preflight, not authorization to mutate this checkpoint. The contract
and frozen evidence are
`benchmarks/production-model-v1/p10m-kv-scaling-readiness-contract.json` and
`benchmarks/production-model-v1/p10m-kv-scaling-readiness.json`.

The `p10m_gate_boundary_preflight_contract` is now complete. It preserves the
full K/V readiness schedule and changes only the gated-MLP `gate` shift from 25
to 23 over the same 2,048-window horizon. `gate` first crossed an exact update
boundary at window 768, accumulated 26 updates across the run, and moved only
alongside K, V, and output. All 13 gradient paths remained active, every
saturation counter stayed zero, and held-out total ended 5,209 millibits below
initialization. Replaying windows 1,025–2,048 reproduced the final model and
optimizer byte-for-byte.

The source analysis predicted 12 parameters at a single boundary snapshot; the
fresh run accumulated 26 crossings over multiple later chunks. The checkpoint
retains both numbers so snapshot prediction is not confused with cumulative
movement. Final residual analysis selects the `up` projection at shift 23,
with 6 predicted crossings, as the next isolated action. The contract and
frozen evidence are
`benchmarks/production-model-v1/p10m-gate-boundary-preflight-contract.json` and
`benchmarks/production-model-v1/p10m-gate-boundary-preflight.json`.

The source-relative `up` quality gate is also complete. Shift 23 first moved at
window 768 and accumulated 26 exact updates with zero saturation and exact
midpoint replay, but its final dev score exactly tied the gate source at 13,044
mean millibits. It is safe reachability evidence, not a quality breakthrough.

The stronger shift-22 discovery run exposed the distinction decisively. `up`
first moved at window 512 and accumulated 101,543 exact updates over 2,048
windows, about 3,900 times the shift-23 count, while every arithmetic safety
gate stayed green. The deterministically selected window-1,024 checkpoint only
tied source dev quality. Exact replay reproduced its model and optimizer, and
the one-shot test comparison was worse by 1,245 total and 5 mean millibits.
The outcome is `no_dev_discovery`, not promotion.

A matched-horizon functional comparison then evaluated the shift-23 and
shift-22 window-1,024 models on all 256 dev windows. Although their model
hashes and `up` weights differ, every final feature vector, logit vector,
probability vector, prediction, and per-window loss is identical. The extra
integer weight updates are masked inside the forward trunk before the final
feature boundary. Frozen evidence is in
`benchmarks/production-model-v1/p10m-up-useful-update.json`,
`benchmarks/production-model-v1/p10m-up-shift22-breakthrough.json`, and
`benchmarks/production-model-v1/p10m-up-functional-comparison.json`.

The predeclared forward-scale sweep resolves that boundary. Common `up` forward
shifts 10, 9, and 8 remain completely masked. Shift 7 is the first safe
functional row: 250 of 256 final feature and logit vectors differ, 124
probability vectors differ, one prediction differs, and both models retain zero
forward saturation. But the target probability changes on 0 of 256 windows, so
all per-window losses remain equal.

Fresh training at `up` learning shift 22 and forward shift 7 preserves the
functional scale while testing its gradients. Across 1,024 windows it makes
50,568 exact `up` updates with all 13 gradient paths active and zero saturation.
The selected endpoint exactly ties source dev at 13,044 mean millibits, and a
full 256-step replay reproduces its model and optimizer byte-for-byte. The
outcome is `safe_functional_training_without_dev_gain`. Frozen evidence is in
`benchmarks/production-model-v1/p10m-up-forward-scale-sensitivity.json` and
`benchmarks/production-model-v1/p10m-up-forward-scale-training.json`.

The `p10m_target_probability_resolution_review` is now complete. It evaluates
the same frozen integer logits at Q15, Q19, Q23, Q27, and Q31; the Q31 path
requantizes exactly to the existing Q15 production output. With an 8,192-token
vocabulary the uniform probability is only 4 units at Q15. Source targets take
three distinct Q15 values and the candidate changes no target probability. Q19
raises the uniform quantum to 64 and exposes 1 changed target; Q23 raises it to
1,024 and exposes 13, matching the changed-target coverage at Q27 and Q31.
Frozen evidence is in
`benchmarks/production-model-v1/p10m-target-probability-resolution.json`.

The follow-up preflight feeds Q19 and Q23 probabilities into training while
adding the exact fractional-bit delta to the output backward, output-weight,
and bias update shifts. That preserves effective real learning rates and keeps
the legacy Q15 path byte-identical. Both candidates keep all 13 gradient paths
active with zero saturation, yet after 256 windows both candidate models are
byte-identical to the Q15 control and tie its 3,344,185 total / 13,063 mean
millibit dev result. Their optimizer artifacts differ, proving the wider
information is retained only in residual state at this horizon. The selected
Q19 replay is byte-identical. Frozen evidence is in
`benchmarks/production-model-v1/p10m-wide-probability-gradient-preflight.json`.

The `p10m_probability_normalization_accuracy_review` is now complete. It holds
the models, data, logits, Q23 output scale, and every forward shift fixed while
comparing the legacy Q31 LUT reciprocal, a retained-Q47 LUT reciprocal, one Q47
integer Newton refinement, and rounded exact Q47 division. Worst-case source /
candidate mass error falls from 98,925 / 98,929 ppm for the legacy path to
6,354 / 6,349 ppm for retained Q47, 98 / 83 ppm after one Newton step, and 73 /
74 ppm for exact division. Newton therefore recovers mass accuracy without a
runtime division and lands close to the exact ceiling.

The same audit changes the interpretation of the earlier resolution result.
Legacy Q23 normalization changes 13 target windows, but Newton changes 5 and
exact division changes 4. The predeclared rule required an accuracy candidate
to retain all 13, so the frozen outcome is
`normalization_accuracy_not_recovered` and no training default is selected,
even though Newton independently clears the 1,000 ppm mass threshold. Frozen
evidence is in
`benchmarks/production-model-v1/p10m-probability-normalization-accuracy.json`.
The follow-up `p10m_probability_normalization_signal_attribution_review` is also
complete. Exact division changes windows 6, 79, 173, and 193; Newton preserves
all four and adds only window 174, where the target logit and unnormalized
target weight are unchanged, the denominator moves, and Newton differs from
the exact Q23 result by one unit. Across both full probability surfaces,
Newton's maximum per-value and per-target error against exact division is one
Q23 unit. The nine legacy-only windows all have unchanged target logits,
changed relative target weights and denominators, and zero exact Q23 delta.
This attributes them to reciprocal amplification of sub-Q23 movement rather
than exact normalized target signal. Frozen evidence is in
`benchmarks/production-model-v1/p10m-probability-normalization-signal-attribution.json`.

The `p10m_normalized_wide_gradient_preflight` is now complete. The Q23
`q47_newton1` control changes optimizer bytes relative to the legacy Q23 lane
but remains byte-identical to the Q15-control model and dev score. Its exact
replay reproduces both model and optimizer bytes. The residual-selected `up`
shift-21 boundary materializes 155 `up` updates, changes 84 of 256 feature and
logit windows and 29 probability windows, and retains zero saturation, but
changes no target probability and ties dev exactly. The isolated output shift-33
boundary (effective Q23-compensated shift 41) materializes 3,597 output updates,
changes 64 target logits and three target probabilities, and remains
zero-saturation, but worsens dev by 415 total / 2 mean millibits. Frozen evidence
is in
`benchmarks/production-model-v1/p10m-normalized-wide-gradient-preflight.json`.

The outcome is `integer_precision_recovered_without_dev_gain`: Q23 plus the Q47
Newton reciprocal carries information through optimizer residuals, materialized
weights, features, logits, probabilities, and finally target probabilities.
The remaining failure is update direction or objective alignment, not an
unreached integer boundary. The next gate is
`p10m_target_aligned_integer_objective_review`. Paid p20m/p30m cloud execution
remains explicitly unauthorized. Assisted retrieval, suffix memory, and routing
oracles remain forbidden in headline generation rows.

The follow-up objective infrastructure is now implemented but has not been
promoted as quality evidence. `integer-transformer-successor-v2` replaces the
improper probability-error primary metric with canonical integer NLL, requires a
transformer-only candidate, and names a real `float-transformer` baseline rather
than the v1 smoothed n-gram mixture. The frozen v1 checker is unchanged. The
rescue-stratified v2 p10m audit also passed neither promotion gate. It used
eight windows balanced across four proposal documents and eight across four
different transfer documents. Four primary-lane rescue-exposed trunk
coordinates were sampled: proposal agreement was `1/3` versus random `3/3`,
with zero exact descents versus random one; transfer exposed no exact descent.
Both output-head coordinates aligned and descended on both surfaces. Before an
optimizer or paid scale is selected, v3 applied the normalized no-rescue causal
control on the same coordinates. It removed all 222 rescues and changed all
four exposed aggregate magnitudes by one count, but changed no signs and left
proposal and transfer summaries identical. Rescue distortion is measurable but
does not explain the sampled directional failure. Global rescue removal is not
authorized. The fingerprint-bound rank-two follow-up found proposal coefficients
`mu_T=+1`, `mu_H=-6`, `mu_TH=0` and transfer coefficients `mu_T=+1`,
`mu_H=-4`, `mu_TH=-2`. Thus the trunk block is harmful alone, while its
conditional effect after the head is `-1` on transfer; the head is the proposal
minimizer and the joint block is the transfer minimizer. This is post-hoc
four-document calibration, so the next bounded experiment is a prospectively
frozen head-versus-joint comparison on unseen document blocks. It does not
authorize an optimizer or paid scale.
