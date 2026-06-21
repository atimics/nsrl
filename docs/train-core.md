# NSRL Train Core

`nsrl-train-core` is the first `no_std` extraction from the host trainer. It is
not a replacement for `nsrl-train`; it is the training step engine that a Linux
CLI, AWS runner, appliance, or Neural OS caller can drive with caller-owned
memory.

## Crate Boundary

```text
nsrl-core          no_std inference and numeric kernels
nsrl-train-core    no_std fixed-workspace training steps
nsrl-train         std CLI, files, S3, traces, experiments
nsrl-os-demo       future bare-metal proof
```

The hard rule for `nsrl-train-core` is:

```rust
train_step(model_slices, token_window, workspaces) -> StepStats
```

The `no_std` layer owns no heap allocations and opens no files. It receives all
model state and scratch space as borrowed slices, performs one deterministic
integer training step, and returns compact counters for the host trace layer.

## Current Supported Path

The initial extracted path is intentionally narrow:

- mini-transformer byte model
- linear attention
- NoPE position policy
- batch window size 1
- output head, gated MLP, Q/K/V/O attention, and embeddings updated in-place

This is enough to prove the shape needed for Neural OS without freezing the
larger host experiment system. The host crate keeps batch accumulators,
rollback policy, adaptive shifts, corpus loading, checkpointing, progress JSON,
and S3 orchestration.

## Batch Primitive

The first batch-training primitive is also extracted:

```rust
LinearWeightGradientI64Workspace {
    input_dim,
    output_dim,
    sample_count,
    accumulators,
    residuals,
}
```

The caller owns the i64 accumulator and residual buffers. The no-std functions
accumulate prescaled outer products and apply averaged i8 updates with the same
residual-carry behavior used by the Linux trainer. `nsrl-train` still owns the
`Vec` wrapper for now, but delegates the linear batch-gradient math to
`nsrl-train-core`.

## Parity Contract

`nsrl-train` contains a byte-for-byte fixture:

```text
mini_transformer_train_core_linear_nope_step_matches_std_single_window
```

It initializes the same model twice, runs the existing host trainer for one
linear-NoPE window, runs `nsrl-train-core` on the same window with fixed
borrowed buffers, and compares every trainable slice. This is the safety rail
for continuing the extraction.
