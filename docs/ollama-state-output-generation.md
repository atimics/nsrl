# Ollama State Output Generation

`scripts/generate-state-outputs-ollama.mjs` uses a local Ollama model to draft
raw in-world outputs from simulator `private_state` rows. It is for expanding
the paired corpora after the simulator has already supplied grounded states.

The script does not send data to a hosted API. It calls the local Ollama HTTP
server at `http://127.0.0.1:11434/api/generate`.

## Smoke Runs

Generate a small Signal radio pilot set with Gemma4:

```sh
node scripts/generate-state-outputs-ollama.mjs \
  --model gemma4:latest \
  --input data/processed/signal-replay-corpus/training-pairs.jsonl \
  --out-dir data/processed/ollama-state-outputs/signal-replay-gemma4 \
  --domain signal \
  --limit 24 \
  --variants-per-state 2
```

Generate a small CosyWorld narration set:

```sh
node scripts/generate-state-outputs-ollama.mjs \
  --model gemma4:latest \
  --input data/processed/cosyworld-kernel-corpus/training-pairs.jsonl \
  --out-dir data/processed/ollama-state-outputs/cosyworld-kernel-gemma4 \
  --domain cosyworld \
  --limit 24 \
  --variants-per-state 3 \
  --attempts 8
```

Use `--limit 0` only when you intentionally want the whole source file. Increase
`--variants-per-state` to expand output diversity from the same simulator state.

## Outputs

Each output directory contains:

- `training-pairs.jsonl`: rows with the same `private_state` and a
  Gemma-drafted `expected_output`.
- `expected-output.txt`: raw generated lines only, useful for quick inspection.
- `cache.jsonl`: append-only accepted/rejected generations for resumable runs.
- `rejects.jsonl`: source variants that failed all attempts.
- `manifest.json`: model, source path, counts, and output paths.

The deterministic simulator line is preserved as `source_expected_output`.
The tiny trainers still read `expected_output`.

## Training

The focused trainer has Gemma4 lanes for the default output paths:

```sh
node scripts/train-focused-pair-models.mjs \
  --out-root data/processed/cheap-trained/gemma4-state-pair-v1024-d16-seq16 \
  --lanes signal-replay-gemma4,cosyworld-kernel-gemma4 \
  --cosyworld-repeat 16
```

## Quality Rules

The generator asks for one raw in-world line only and rejects obvious wrappers:

- `ranked`
- `private_state`
- `expected_output`
- `assistant`
- `chatbot`
- `model`
- `training`
- standalone `AI`
- JSON-like braces

This is still draft data. Keep a small smoke set first, read samples, then scale
the simulator rows and variants once the voice sounds right.
