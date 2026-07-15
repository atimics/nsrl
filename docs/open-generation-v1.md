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
document-indexed token streams. The next modeling work is the variable-vocabulary
training artifact followed by three controlled 10M-30M scaling points with
same-shape float twins.
