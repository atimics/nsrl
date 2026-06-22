# Simulator-State Corpora

These builders generate paired `private_state` to `expected_output` rows from
real local simulators, keeping the cheap template corpora as fallback/noise.

## CosyWorld C Kernel

```sh
node scripts/build-cosyworld-kernel-corpus.mjs
```

The builder compiles a tiny C probe against:

```text
/Users/ratimics/develop/cosyworld/v2/core-c
```

It runs a deterministic cottage sequence: seed, visitor entry, speech, movement,
checks, item pickup/use, sparring, fleeing, giving charms, and avatar evolution.
Output defaults to `data/processed/cosyworld-kernel-corpus/`:

- `states.jsonl`: raw C-kernel world snapshots, events, actors, items, exits,
  and action-offer masks.
- `frames.jsonl`: event-level in-world text frames derived from the snapshots.
- `training-pairs.jsonl`: compact `private_state` and `expected_output` rows.
- `voice.txt`: output-only lines for smoke inspection.

## Signal Replay

```sh
node scripts/build-signal-replay-corpus.mjs
```

The builder uses Signal's CMake target:

```text
/Users/ratimics/develop/signal/build/signal_replay
```

It runs deterministic replay branches across seeds, provenance scripts, control
vectors, and horizons. Output defaults to `data/processed/signal-replay-corpus/`:

- `states.jsonl`: raw `signal_replay` JSONL branch rows.
- `frames.jsonl`: branch-level pilot state and clipped radio output frames.
- `training-pairs.jsonl`: compact `private_state` and `expected_output` rows.
- `voice.txt`: output-only radio lines for smoke inspection.

## Ollama Expansion

After building the simulator-state rows, a local Ollama model can draft more
varied `expected_output` lines from the same grounded `private_state` values:

```sh
node scripts/generate-state-outputs-ollama.mjs \
  --model gemma4:latest \
  --input data/processed/signal-replay-corpus/training-pairs.jsonl \
  --out-dir data/processed/ollama-state-outputs/signal-replay-gemma4 \
  --domain signal \
  --limit 24 \
  --variants-per-state 2

node scripts/generate-state-outputs-ollama.mjs \
  --model gemma4:latest \
  --input data/processed/cosyworld-kernel-corpus/training-pairs.jsonl \
  --out-dir data/processed/ollama-state-outputs/cosyworld-kernel-gemma4 \
  --domain cosyworld \
  --limit 24 \
  --variants-per-state 3 \
  --attempts 8
```

See `docs/ollama-state-output-generation.md` for the cache, filters, and
training lanes.

## Training

Use the focused pair trainer on the simulator lanes:

```sh
node scripts/train-focused-pair-models.mjs \
  --out-root data/processed/cheap-trained/sim-state-pair-v1024-d16-seq16 \
  --lanes signal-replay,cosyworld-kernel \
  --cosyworld-repeat 32
```

These lanes flatten each row as:

```text
private_state
expected_output
```

There is no `RANKED` wrapper in the focused training text, so the tiny models
are less likely to learn wrapper tokens as raw output.
