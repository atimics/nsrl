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

## Context-64 causal scale diagnostic

The latest full-trunk prerequisite is frozen at
`benchmarks/production-model-v1/p10m-causal-sequence-scale-v3-bias-r3.json`.
It supervises 131,072 causal targets over 2,048 corpus-spread windows;
development and test canonical NLL both improve, all eleven trunk groups move,
saturation is zero, and midpoint replay is byte-identical. Independent audits
bind output-bias behavior, per-layer residual health, parameter deltas, and
optimizer residual thresholds. Its complete generation rerun is frozen at
`benchmarks/open-generation-v1/p10m-causal-sequence-scale-v3-bias-r3.json`.

| Surface | Latest result | Gate |
| --- | ---: | --- |
| Candidate modeling | 3,604 millibits/original UTF-8 byte | baselines missing |
| Incremental cache | 405,504 state bytes; 10,240 workspace bytes | pass |
| Complete generation matrix | 12 prompts; 60 samples; 30,720 tokens | pass |
| Worst repeated four-gram share | 999 per mille | fail (`<= 150`) |
| Minimum unique four-gram share | 1 per mille | fail (`>= 600`) |
| Minimum entropy | 0 Q10 | fail (`>= 2048`) |
| Valid UTF-8 | 1,000 per mille | pass |
| Context use | 0 per mille | fail (`>= 750`) |
| Distractor resistance | 0 per mille | fail (`>= 700`) |

The raw context audit records distinct hidden and logit hashes for every prompt,
so the trunk is not globally context-blind. Nevertheless, only two greedy first
tokens appear and all 12 prompts enter one-token feedback loops. Output-bias
damping and an O-projection stability repair eliminated numeric collapse but
did not produce conditional language.

The complementary rollout-divergence checkpoint is frozen at
`benchmarks/open-generation-v1/p10m-causal-sequence-scale-v3-bias-r3-rollout-divergence.json`.
Across eight corpus-spread development windows and 16 continuation positions
per window, teacher forcing produces zero top-one matches, mean correct-token
rank 2,426, and mean correct-token Q15 probability 5. Free running also matches
zero reference tokens and self-loops on 115 of 120 transitions. Prefix and
suffix counterfactuals each change 255 input tokens in aggregate, while the
older-half prefix swaps move logits 5,011 per mille as much as recent-half
suffix swaps. Collapse therefore precedes rollout distribution shift and
includes an anomalous context-weighting signal. This isolates the remaining
quality gap to next-token ranking, learned context conditioning, and training
coverage; the incremental serving path, artifact bindings, matrix completeness,
replay, saturation, and UTF-8 gates are already green. The hidden panel remains
unopened.
