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
- a checksummed, resumable stateless-SGD optimizer state bound to the exact
  model, tokenizer, token stream, and training schedule; and
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
node scripts/freeze-production-model-v1.mjs --check
node scripts/freeze-production-full-train-v1.mjs --check
node scripts/freeze-production-float-twin-v1.mjs --check
node scripts/check-production-model-v1.mjs
```

The bounded full-backward p10m checkpoint runs 16 optimizer steps. All 13
parameter groups move, mistakes improve from 8 to 6, and the trace records zero
gradient saturation and 1,137 clipped weight updates. The matched float twin
also moves all 13 groups, remains finite, reduces mean loss from 9.011 to 8.500,
and moves from 8 mistakes to 0. These are wiring and numerical-health checks,
not quality comparisons.

## Current boundary

The full backward and float-twin implementation gates are complete. The
integer derivative uses an explicit straight-through rule at quantization
dead zones, and the float twin is a NumPy reference rather than an accelerator
runner. Neither bounded smoke is a language-quality result.

The next checkpoint is a controlled p10m train/dev pilot with a larger frozen
window schedule, held-out evaluation, restart evidence, and integer/float
comparison. The scaling plan keeps `training_started` false until that pilot is
launched deliberately. Assisted retrieval, suffix memory, and routing oracles
remain forbidden in headline generation rows.
