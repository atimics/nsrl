# Mini-Transformer Map-Reduce Training

Map-reduce training is an explicit mini-transformer batch mode for the fast
Graviton and Lambda worker paths. It preserves the serial batched-update weight
result for the supported contract while giving each batch a deterministic
parallel gradient accumulation step.

## Intended Semantics

At each batch boundary:

1. Freeze the current model.
2. Split batch windows across workers.
3. Each worker accumulates gradients privately from the frozen model.
4. The main thread reduces gradients in deterministic worker/chunk order.
5. A candidate model receives one reduced update.
6. The candidate is validated.
7. The batch is committed or rejected atomically.

This is a batched training semantic. It prioritizes throughput and deterministic
batch commits; for the supported contract, serial and map-reduce runs should
land on the same final model hash when their configs match.

## Public Surface

`MiniTransformerMlpTrainConfig` now has:

- `batch_mode: MiniTransformerBatchMode`
- `map_reduce_workers: usize`

CLI:

```sh
--mini-transformer-batch-mode serial|map-reduce
--mini-transformer-map-reduce-workers N
```

AWS runner/API:

```text
NSRL_BATCH_MODE=serial|map-reduce
NSRL_MAP_REDUCE_WORKERS=N
```

`N=0` uses available host parallelism. Each batch is split into deterministic
contiguous worker chunks, worker accumulators are reduced in chunk order, and
one candidate update is applied at the batch boundary.

## Current Gate

The v1 gate accepts only the fast linear/NoPE batch contract:

- linear attention
- NoPE position policy
- identity or ascii-lower byte tokenizer metadata
- `batch_windows > 1`
- rule/adaptive shift controllers may be enabled
- no VO oracle or error feedback
- no loss-regression rejection
- no streaming attention modes

Multi-worker mode is live for this gate. The implementation keeps the model
read-only while workers run forward/backward over their assigned windows into
private i64 gradient buffers. The main thread reduces those buffers in
deterministic worker/chunk order before applying a single checked batch update.

## Next Implementation Step

Move the read-only per-sample forward/backward path into `nsrl-train-core` so
the hot map worker can reuse fixed workspaces instead of allocating host-side
vectors per window. After that, `N=0` should become the default Graviton path
for long linear/NoPE byte runs.
