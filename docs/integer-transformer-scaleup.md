# Integer Transformer Scale-Up (Fork B)

## Why

The deployed text model is the **lexeme mean-reduce** model (`.nsrllm`,
`v4096.seq8-mean-reduce`). Mean-pooling the context is permutation-invariant —
"dog bites man" == "man bites dog" — so it structurally cannot model word order.
That is the root cause of the word-salad bot text, and no amount of vocab/embedding
scaling fixes it.

The right foundation is **Fork B**: the integer mini-transformer
(`nsrl-train-core` → `nsrl-core`), a genuine no-float transformer with attention
(`base2_softmax_i32_q15`, Q15/Q8 fixed-point, RMSNorm via 8-bit rsqrt LUT). It
already has a Graviton swarm map-reduce trainer.

**The whole stack is integer-only** (verified: zero `f32`/`f64` in `nsrl-core`,
`nsrl-train-core`, `nsrl-web-wasm`, and the bitmap denoiser). That is the moat:
bit-exact determinism across CPU/WASM/ARM, runs on any integer ALU, same kernels
for train and inference, i8 weights.

## Plan: deprecate mean-reduce, don't delete it yet

The lexeme path is load-bearing — it serves both the X bot text engine and the
entire web chat (including on-device fine-tuning). Sequence:

1. **Freeze** mean-reduce (no new experiments). ← done by committing to Fork B.
2. Keep it serving prod as-is.
3. **Graviton run:** train the scaled integer transformer to parity-then-better.
4. **Swap inference** (Lambda `X_REPLY_ENGINE`, rebuild WASM). TTT/on-device
   adaptation survives — the mini-transformer already supports `..._ttt_shift`.
5. **Then delete** the mean-reduce head + bins.

Keep the **lexeme tokenizer + 4096 vocab** (`nsrl-corpus`) — it is an asset Fork B
should inherit (see "Next").

## What changed in this branch

`crates/nsrl-train-core/src/lib.rs`, widened (uniform `FixedScale` arrays auto-resize,
no per-dim retuning):

| Constant | Before | After |
|---|---|---|
| `MINI_TRANSFORMER_D_MODEL` | 32 | **64** |
| `MINI_TRANSFORMER_HEADS` | 2 | **4** (head dim 16) |
| `MINI_TRANSFORMER_HIDDEN_DIM` | 64 | **256** |

≈3× parameters. Still byte-level (`BYTE_VOCAB=256`) and a **single block** — those
are the two next levers (below). Change is isolated: the deployed lexeme path is
untouched.

### Local validation (10s smoke, 172 KB byte corpus, seq_len 16, 3 epochs)

- Compiles and trains integer-only.
- Next-byte error **8192 → 6144 (−25%)**, accuracy **250‰** (chance ≈ 4‰ for 256 classes).
- Deterministic generation works end-to-end at the new dims.
- Greedy decode on a 10s smoke is degenerate (expected) — real coherence needs the
  full-corpus run.

## The Graviton run

Pipeline: `scripts/aws/run-mini-transformer-training.sh` (runs on a Graviton
instance, publishes checkpoints + dashboard to S3). Swarm map-reduce shards
windows across workers and averages.

```sh
# On a Graviton (arm64) instance with the repo + AWS role:
NSRL_S3_URI=s3://<bucket>/<prefix> \
NSRL_RUN_NAME=fork-b-d64-h4-mlp256-001 \
NSRL_TRAIN_MODE=mini-transformer-swarm \
NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8 \
NSRL_SEQ_LEN=64 \
NSRL_MAX_WINDOWS=2000000 \
NSRL_EPOCHS=2 \
NSRL_BATCH_WINDOWS=8 \
NSRL_SWARM_WORKERS=32 \
NSRL_SWARM_COMPOSITION=average \
NSRL_ATTENTION=linear \
NSRL_POSITION=nope \
NSRL_RUSTFLAGS='-C target-cpu=native' \
NSRL_PUBLISH_CHECKPOINT=fork-b-golden \
scripts/aws/run-mini-transformer-training.sh
```

**Instance / cost (order-of-magnitude):** integer CPU math is fast — the smoke did
~2.5K windows/s on one laptop core. With `target-cpu=native` + a `c7g.8xlarge`
(32 Graviton3 vCPU, ~$1.16/hr), a ~2M-window × 2-epoch run is **~minutes to ~1 hr**,
i.e. **a few dollars**. The serverless alternative is the arm64 Lambda swarm
(`run-crowley-bard-lambda-mapreduce.sh`).

## Next levers (in priority order)

1. **Lexeme vocab (biggest coherence win):** flip `BYTE_VOCAB 256 → 4096`, add a
   u16 token reader, retrain on `*-lexeme-v4096.tokens.u16`. Reuses existing vocab;
   far better semantics per token. ~moderate code change (token I/O + model format).
2. **Multi-block depth:** currently one transformer block (`known_non_claims:
   single_mini_transformer_block_only`). Add a configurable block count — structural
   change in `nsrl-core`/`nsrl-train-core`.
3. **Multimodal:** text and image are separate integer models bridged only by the
   text→signature→sigil conditioning. The unification is one integer transformer
   emitting text **and** image/latent tokens on the same fixed-point substrate
   (the bitmap denoiser is already integer-only). Foundation for image/video.
