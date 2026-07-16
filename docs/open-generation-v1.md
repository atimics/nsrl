# Open Generation v1

`open-generation-v1` is the frozen development contract for NSRL's native
language-quality lane. It does not claim that an open-generation model exists
yet; it prevents evaluation policy from moving after candidate training starts.

The machine-readable contract is available through:

```bash
cargo run -p nsrl-eval -- open-generation-contract
cargo run -p nsrl-eval -- open-generation-manifest \
  --manifest benchmarks/open-generation-v1/manifest.tsv
```

## Frozen development surface

- 12 retained prompts across continuation, constrained style, explanation,
  dialogue, long-context reference, and adversarial repetition.
- 512 generated subword tokens per eligible prompt.
- Greedy replay plus sampling seeds `7`, `19`, `43`, and `97`.
- Bits per original UTF-8 byte as the cross-tokenizer modeling metric.
- Required byte n-gram, retrieval, best-smaller-NSRL, and same-shape float-twin
  baselines.
- At least 900 per mille retained improvement from the statistical baseline to
  the float twin.
- At most 150 per mille repeated four-gram share and at least 600 per mille
  unique four-gram share.
- Entropy floor `2048` in Q10, 1000 per mille valid UTF-8, 750 per mille context
  use, and 700 per mille distractor resistance.
- Blinded human preference no worse than 100 per mille below the same-shape
  float twin.

Retrieval, corpus priors, memory injection, target lookup, and routing oracles
are forbidden in headline rows. Assisted product diagnostics must remain
separate.

The public development panel is hash-bound in the manifest. The final hidden
panel remains under `data/private/open-generation-v1/` and is represented in
the checked-in contract only by its SHA-256 commitment. The hidden panel must
not be used for training or model selection.

This checkpoint freezes evaluation structure and tokenizer conformance.
`production-corpus-v1` now freezes the first licensed-source registry,
deduplicated split, contamination gate, actual 8,192-token vocabulary, and
document-indexed token streams.

## Reproducible p10m baseline

Run the complete public development row with:

```bash
scripts/run-open-generation-development-v1.sh
```

The command builds hash-bound generation and modeling runners, emits greedy plus
four seeded samples for every prompt, retains all incremental decoder traces,
evaluates the public gates, byte-replays the result, and freezes a compact
checkpoint. Large ledgers stay under ignored experiment storage; their FNV-1a
and SHA-256 bindings are retained in
`benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json`.

The modeling policy resets the incremental decoder for each prompt, consumes a
BOS token, scores every candidate-tokenizer token in the original prompt, and
does not score EOS. Dividing canonical integer NLL by original prompt bytes
makes the candidate row tokenizer-comparable. This candidate-only measurement
does not satisfy the modeling gate without all required baselines.

The first p10m result is diagnostic, not a promotion:

| Surface | Result | Gate |
| --- | ---: | --- |
| Candidate modeling | 3,687 millibits/original UTF-8 byte | baselines missing |
| Incremental cache | 405,504 state bytes; 10,240 workspace bytes | pass |
| Complete generation matrix | 12 prompts; 60 samples; 30,720 tokens | pass |
| Worst repeated four-gram share | 989 per mille | fail (`<= 150`) |
| Minimum unique four-gram share | 11 per mille | fail (`>= 600`) |
| Minimum entropy | 431 Q10 | fail (`>= 2048`) |
| Valid UTF-8 | 166 per mille | fail (`>= 1000`) |
| Context use | 0 per mille | fail (`>= 750`) |
| Distractor resistance | 0 per mille | fail (`>= 700`) |

The hidden panel was not opened or used for selection. The next modeling work is
the required byte-ngram, retrieval, best-smaller-NSRL, and same-shape float-twin
matrix plus a prospectively frozen full-trunk candidate that improves both
development and untouched test evidence before generation is rerun.
