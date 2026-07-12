# Production model v1

`production-model-v1` introduces `NSRLPM1`, a variable-vocabulary integer
decoder artifact separate from the frozen byte-vocabulary MT5/MT6 formats. It
is bound to one `NSRLBPE1` tokenizer hash and consumes tokenizer-bound
`NSRLTOK1` u32 streams.

## Implemented gate

The runtime now provides:

- exact dynamic parameter accounting for the frozen p10m, p20m, and p30m
  shapes;
- deterministic integer initialization for embeddings, causal linear
  attention, gated MLP, RMS vectors, output weights, and output bias;
- checksummed `NSRLPM1` serialization and strict shape validation;
- tokenizer-hash and vocabulary validation when loading `NSRLTOK1` streams;
- integer forward execution over u32 subword contexts; and
- a bounded output-head perceptron smoke trainer with model-hash and saturation
  evidence.

The frozen p10m smoke artifact has 9,317,632 parameters and is bound to
tokenizer hash `0xf4fe71d93c438c1a` and train-stream token hash
`0x97e5254c31c27bda`. Eight windows move from eight mistakes to zero with eight
updates, zero weight saturation, and zero residual saturation. The 13 MB model
artifacts stay in ignored experiment storage; their SHA-256 and internal model
hashes are frozen in `benchmarks/production-model-v1/p10m-smoke.json`.

Reproduce it with:

```bash
scripts/run-production-model-v1-smoke.sh
node scripts/freeze-production-model-v1.mjs --check
node scripts/check-production-model-v1.mjs
```

## Current boundary

The smoke trains only the output head against cached deterministic trunk
features. It proves the variable-vocabulary artifact, u32 data path, forward
kernels, update path, serialization, and health instrumentation work together.
It is not a language-quality run and does not claim full transformer training.

Before the controlled p10m scaling run can be labeled started, two gates remain:

1. backpropagation and optimizer state for every attention, MLP, RMS, embedding,
   and output parameter in `NSRLPM1`; and
2. a same-shape float runner using identical initialization mapping, token order,
   contexts, batches, splits, and evaluation contract.

The scaling plan keeps `training_started` false until both exist. Assisted
retrieval, suffix memory, and routing oracles remain forbidden in headline
generation rows.
