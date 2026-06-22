# NSRL Binary Trace Format

JSONL remains useful for research, dashboards, and quick inspection. Long
training runs should use the binary trace format so the training path does not
pay integer-to-ASCII formatting costs for high-volume telemetry.

## File Contract

All integer fields are little-endian. The first implemented schema is the
mini-transformer MLP training trace.

```text
magic:              4 bytes   "NSRL"
version:            u8        1
schema_id:          u8        1 = mini-transformer MLP training
reserved:           u16       0
initial_model_hash: u64
records:            tagged append-only stream
```

Record tags:

```text
0x01 step sample          fixed 32 bytes including tag
0x02 adaptive shift event fixed 22 bytes including tag
0x7f final summary        fixed-width run summary
```

The step sample intentionally stores only hot-path counters and compact identity
fields. Full tensor/cache hashes live in the final summary and model artifacts.
This keeps sampled step telemetry small enough for large runs while preserving
replay anchors.

## CLI

```sh
cargo run --release -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --trace data/processed/run.nsrlt \
  --trace-format binary
```

Binary traces default to `--mini-transformer-trace-detail summary`, which keeps
the first 16 update samples and then samples at the progress interval, or every
1024 updates when no progress interval is configured. Use
`--mini-transformer-trace-detail full` only for short diagnostic runs.

When `--trace-format binary` is used with `--trace PATH`, `nsrl-train` streams
the header and committed step records directly to a buffered file during
training, then appends bounded adaptive-shift events and the final summary at
the end. The stdout binary path still serializes after a successful run so a
failed command does not emit a partial binary stream into a pipe.

## Reader Direction

Use `nsrl-trace-read` after training to convert `.nsrlt` bytes into text.
Human-readable formatting stays out of the training loop and moves analysis
cost to the cold path.

```sh
cargo run --release -p nsrl-train --bin nsrl-trace-read -- \
  data/processed/run.nsrlt
```

The default output is a compact table with the run summary, adaptive shift
events, and the first 16 sampled step records. Use JSON when feeding dashboards
or batch analysis:

```sh
cargo run --release -p nsrl-train --bin nsrl-trace-read -- \
  data/processed/run.nsrlt \
  --format json \
  --max-step-records all
```
