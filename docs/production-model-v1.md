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
- integer forward execution over u32 subword contexts; and
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
  initialization and trained on the same bounded windows.

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
python3 scripts/benchmark-production-training-v1.py
node scripts/freeze-production-model-v1.mjs --check
node scripts/freeze-production-full-train-v1.mjs --check
node scripts/freeze-production-float-twin-v1.mjs --check
node scripts/freeze-production-integer-stabilization-v1.mjs --check
node scripts/freeze-production-stabilized-pilot-v1.mjs --check
node scripts/freeze-production-liveness-audit-v1.mjs --check
node scripts/check-production-training-liveness-self-test.mjs
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
Assisted retrieval, suffix memory, and routing oracles remain forbidden in
headline generation rows.
