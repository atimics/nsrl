# NSRLMT6 development path

`NSRLMT6` is an explicit successor architecture, not an in-place change to the
frozen `NSRLMT5` proof-v1 candidate. MT5 decoding, hashes, and artifact magic
remain unchanged.

The first MT6 milestone addresses the measured MT5 representation collapse:

- serialized per-projection right shifts for Q/K/V, O, SwiGLU up/gate/down,
  local mixing, and the vocabulary head;
- deterministic non-aliased byte embeddings and independent MLP branches;
- two calibrated integer linear-attention transformer layers;
- an ordered four-tap causal local path, retaining a full 128-dimensional
  block for each of the four most recent bytes;
- final RMSNorm over the transformer state plus ordered local blocks;
- an i16 vocabulary head and i32 output bias;
- a checksummed `NSRLMT6` artifact that is rejected by the MT5 loader and vice
  versa.

## Local overfit gate

Before full-trunk training or frozen evaluation, MT6 must memorize 256
deterministically spread context-64 windows without numerical failure:

```bash
scripts/run-mt6-local-overfit-gate.sh \
  data/experiments/nsrlmt6/local-overfit-default
```

The default gate requires at least 900 per mille training accuracy, zero output
weight saturation, and no more than 4,096 residual saturation events per
window. It writes `candidate.nsrlmt6` and `overfit.trace.jsonl`.

The reference implementation reaches 1,000 per mille (256/256), with zero
output-weight saturation and 184 residual saturation events per window. This
is a representation-capacity gate, not held-out language-model evidence.

Inspect an artifact on held-out bytes without treating it as a proof-v1 row:

```bash
cargo run --release -p nsrl-train --bin nsrl-mt6-eval -- \
  --tokens benchmarks/integer-transformer-proof-v1/eval.txt \
  --model data/experiments/nsrlmt6/local-overfit-default/candidate.nsrlmt6
```

The output-head-only reference artifact scores 1,149/5,896 correct
(194 per mille), probability error `311,063,644`, 30 distinct predicted bytes,
and a 191-per-mille dominant-byte share. This is already less collapsed than
the MT5 diagnostic, but its 4,747 mistakes are not near a production or proof
threshold.

## Next promotion boundary

MT6 must not replace the proof-v1 candidate. The next contract should add
full-trunk integer optimization, evaluate the existing frozen bytes as a
diagnostic, and define a new candidate magic explicitly. Retrieval or n-gram
tables remain outside the candidate model.
