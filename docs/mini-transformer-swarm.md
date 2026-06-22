# Mini-Transformer Swarm Training

`mini-transformer-swarm` trains many tiny mini-transformer workers in parallel.
Each worker receives an interleaved shard of the same token stream:

```text
worker_window_offset = base_window_offset + worker_index * base_stride
worker_stride        = base_stride * worker_count
```

This keeps workers independent and deterministic while using available CPU
cores. On ARM64 hosts, the mode is intended to pair with the existing integer
kernels and binary trace path: hot work remains local to each worker, and the
swarm summary is written only after the workers finish.

Example:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --swarm-workers 8 \
  --max-windows 8192 \
  --batch-windows 1 \
  --model-out data/processed/wiki-bard-mini-transformer-swarm-best.nsrlmt \
  --swarm-model-out data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --trace data/processed/wiki-bard-mini-transformer-swarm.trace.jsonl
```

The trace uses schema `nsrl.training_mini_transformer_swarm_trace.v1`. It lists
every worker shard, its final metrics, and `best_worker_index`. The model
written to `--model-out` is the promoted best worker, ranked by final total
error, then final probability error, then invalid forward count, then worker
index for deterministic ties.

## Island Fan-Out

`mini-transformer-swarm-worker` runs one deterministic worker shard and writes a
self-validating binary worker artifact. This is the Lambda-friendly path: invoke
one worker per shard, then assemble the artifacts after every invocation
finishes.

Worker invocation:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-worker \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --swarm-worker-index 0 \
  --swarm-worker-count 8 \
  --max-windows 8192 \
  --batch-windows 1 \
  --swarm-worker-out data/processed/worker-000.nsrlwk \
  --trace data/processed/worker-000.trace.jsonl
```

Assembler invocation:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-assemble \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --max-windows 8192 \
  --batch-windows 1 \
  --swarm-worker-artifact data/processed/worker-000.nsrlwk \
  --swarm-worker-artifact data/processed/worker-001.nsrlwk \
  --swarm-model-out data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --model-out data/processed/wiki-bard-mini-transformer-swarm-best.nsrlmt \
  --manifest-out data/processed/wiki-bard-mini-transformer-swarm.manifest.jsonl \
  --trace data/processed/wiki-bard-mini-transformer-swarm.trace.jsonl
```

The worker artifact uses magic `NSRLWK1` and stores the worker shard metadata,
base model hash, worker summary metrics, and the worker `.nsrlmt` payload. The
assembler validates token hash, base config, base model hash, worker indexes,
window hashes, final model hashes, and recomputed shard eval metrics before
packing the normal `.nsrlswarm` artifact. Worker artifacts must cover indexes
`0..worker_count`; duplicate or missing workers are rejected.

### Lambda Cost/Speed Lane

Lambda map-reduce is the default cheap lane for swarm comparisons. It uses the
same deterministic worker artifact path as island fan-out, but each Lambda
worker now defaults to the hash-matched inner batch reducer:
`ascii-lower`, linear attention, NoPE position, `stride=1`, `batch_windows=2`,
`batch_mode=map-reduce`, two in-invocation reducers, adaptive rule shifts, and
sparse progress writes.

For the current Crowley Bard token dataset, the convenience wrapper is the
normal path:

```sh
BUILD=1 DEPLOY=1 scripts/aws/run-crowley-bard-lambda-mapreduce.sh
```

After the Lambda package has been deployed once, repeat runs can omit the build
and deploy step:

```sh
scripts/aws/run-crowley-bard-lambda-mapreduce.sh
```

Override `MAX_WINDOWS`, `WORKERS`, `MEMORY_MB`, or
`NSRL_TOKENS_S3_URI` when comparing scale or a new corpus version. The generic
orchestrator remains available when you need an explicit dataset:

```sh
scripts/aws/build-lambda-swarm-worker.sh

node scripts/aws/run-lambda-swarm-comparison.mjs \
  --deploy \
  --run \
  --s3-uri s3://BUCKET/PREFIX \
  --tokens-s3-uri s3://BUCKET/PREFIX/corpus/datasets/DATASET/VERSION/tokens/tokens.u8 \
  --run-name lambda-swarm-64k \
  --workers 4 \
  --max-windows 65536 \
  --seq-len 8 \
  --stride 1 \
  --batch-windows 2 \
  --batch-mode map-reduce \
  --map-reduce-workers 2 \
  --tokenizer ascii-lower \
  --adaptive-rule-shifts 1 \
  --progress-interval-batches 1024
```

The orchestrator creates or updates the ARM64 Python Lambda function, invokes
one `mini-transformer-swarm-worker` shard per worker, waits for `.nsrlwk`
artifacts in S3, assembles them locally with `mini-transformer-swarm-assemble`,
uploads the packed `.nsrlswarm`, and writes `metrics.json`/`metrics.tsv`. Pass
`--lambda-gb-second-usd` and `--lambda-request-usd` to pin whatever current AWS
price sheet you want used for the estimate.

## Scaling Sweep

Use `mini-transformer-swarm-scaling` to measure host scaling before committing
to a long run:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-scaling \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --max-windows 4096 \
  --batch-windows 1 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --swarm-workers 16 \
  --trace data/processed/wiki-bard-mini-transformer-swarm-scaling.trace.jsonl
```

The scaling trace uses schema
`nsrl.training_mini_transformer_swarm_scaling_trace.v1` and sweeps worker counts
`1, 2, 4, ... N`, including `N` when it is not a power of two. Each row records
elapsed nanoseconds, windows per second, updates per second, speedup, parallel
efficiency, effective worker count, and the best worker's final error metrics.
Timing rows are host observations, not universal benchmark claims.

Pass `--swarm-model-out PATH` to also write a packed swarm artifact. The
artifact stores every worker as length-prefixed `.nsrlmt` payloads under magic
`NSRLSW1`, plus the best-worker index and a swarm hash. This keeps the
individual worker model validator authoritative while allowing cold-path tools
and generation to use the composed swarm.

Pass `--manifest-out PATH` with swarm training to write an expert manifest
sidecar for routers and dashboards. The manifest schema is
`nsrl.mini_transformer_swarm_expert_manifest.v1`; it declares capability tags,
tokenizer and I/O contracts, numeric contract, routing hints, model/component
hashes, artifact bytes, and parameter bytes.

Generate a manifest from an existing packed artifact with:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-manifest \
  --model data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --trace data/processed/wiki-bard-mini-transformer-swarm.manifest.jsonl
```

Route over one or more packed swarm experts with:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-route \
  --expert data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --prompt "To be" \
  --route-capability byte_generation \
  --route-prompt-affinity \
  --route-active-experts 1 \
  --trace data/processed/wiki-bard-mini-transformer-swarm.route.jsonl
```

The route trace uses schema `nsrl.mini_transformer_swarm_route_trace.v1`. It is
a deterministic manifest router with optional prompt replay scoring: each
candidate records capability match, budget checks, manifest score, prompt
affinity score, rejection reason, and selected expert index. Use
`--route-max-artifact-bytes` and `--route-max-parameter-bytes` to enforce local
memory budgets before inference. `--route-prompt-affinity` evaluates each expert
over the prompt's own next-byte transitions before selection; streaming
attention modes are not used for that route-only scoring path.

Route and generate from an active expert set with:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-routed-generate \
  --expert data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --prompt "To be" \
  --max-new-tokens 64 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --route-capability byte_generation \
  --route-prompt-affinity \
  --route-active-experts 1 \
  --text-out data/processed/wiki-bard-mini-transformer-swarm-routed.txt \
  --trace data/processed/wiki-bard-mini-transformer-swarm-routed-generate.trace.jsonl
```

The routed generation trace uses schema
`nsrl.mini_transformer_swarm_routed_generation_trace.v1`. It embeds the route
decision and the normal swarm generation trace. When more than one expert is
selected, the selected experts' workers are concatenated into one active-set
swarm before generation, so existing composition modes still apply.
`--route-prompt-affinity` adds a cheap pre-generation score based on each
expert's fixed-point probability error over the prompt's own next-byte
transitions. Use `--route-prompt-affinity-windows N` to cap how many prompt
windows are evaluated.

Generate from the composed artifact with:

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-swarm-generate \
  --model data/processed/wiki-bard-mini-transformer-swarm.nsrlswarm \
  --prompt "To be" \
  --max-new-tokens 64 \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --text-out data/processed/wiki-bard-mini-transformer-swarm.txt \
  --trace data/processed/wiki-bard-mini-transformer-swarm-generate.trace.jsonl
```

Swarm generation evaluates every worker on the same context window, averages the
worker logits, recomputes probabilities, and then uses the normal deterministic
byte decoder. Streaming attention modes are intentionally left to the
single-model path for now.

Composition modes:

```text
--swarm-composition average
--swarm-composition confidence-weighted
--swarm-composition confidence-router
```

`average` is the default and gives every worker equal weight. `confidence-
weighted` weights each worker's logits by its top-logit margin for the current
context. `confidence-router` chooses the single most confident worker per token
using the same margin, with worker index as the deterministic tie breaker. These
router modes are cheap integer baselines for later trained routing nets.

By default, worker traces use `--mini-transformer-trace-detail none` inside the
swarm to avoid per-step memory growth. Pass `--mini-transformer-trace-detail
summary` for short diagnostics.
