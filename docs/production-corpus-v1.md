# Production corpus v1

`production-corpus-v1` is the first source-bound, document-deduplicated,
contamination-checked subword corpus checkpoint for NSRL. The checked-in
checkpoint records the hashes and policy; the 100+ MB corpus and token streams
remain under ignored `data/processed/production-corpus-v1/`.

## Frozen result

- 20,341 accepted documents from a capped 20,000-page Simple English Wikipedia
  slice plus Shakespeare and Blake source texts.
- Deterministic 98/1/1 document splits: 19,929 train, 213 dev, and 199 test.
- 54,789,038 document bytes in train, 629,854 in dev, and 468,609 in test.
- Exact SHA-256 deduplication followed by deterministic 5-word-shingle LSH and
  an 850-per-mille Jaccard near-duplicate threshold.
- Direct and 5-word-shingle contamination checks against all 12 public and 6
  hidden `open-generation-v1` prompts; zero documents were quarantined in the
  frozen build.
- An 8,192-token byte-complete BPE vocabulary trained only from a deterministic
  1 MiB sample of the train split. It contains 7,933 learned merges and encodes
  the sample at 238 tokens per 1,000 source bytes.
- Document-indexed token streams with one BOS and EOS token per document. The
  train stream contains 14,357,405 tokens.

The complete machine-readable evidence is in
`benchmarks/production-corpus-v1/checkpoint.json`.

## Rebuild

The default config binds exact local input hashes. Recreate the corpus and bind
the tokenizer and split token streams with:

```bash
node scripts/production-corpus-v1.mjs build \
  --config benchmarks/production-corpus-v1/config.json \
  --out-dir data/processed/production-corpus-v1

cargo run --release -p nsrl-corpus --bin nsrl-subword -- train \
  --corpus data/processed/production-corpus-v1/tokenizer-train.txt \
  --tokenizer-out data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  --trace data/processed/production-corpus-v1/tokenizer.trace.json \
  --vocab-size 8192 --min-pair-frequency 2

node scripts/production-corpus-v1.mjs bind-tokenizer \
  --manifest data/processed/production-corpus-v1/manifest.json \
  --tokenizer data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  --trace data/processed/production-corpus-v1/tokenizer.trace.json
```

For each of `train`, `dev`, and `test`, run the indexed encoder with the matching
`.txt` and `.index.tsv`, then bind the resulting `.nsrltok` and trace with
`production-corpus-v1.mjs bind-encoding`. Finally regenerate the small frozen
record:

```bash
node scripts/freeze-production-corpus-v1.mjs
scripts/check-production-corpus-v1.sh
```

## Rights boundary

The source registry preserves source URLs, hashes, license identifiers,
attribution, and rights-basis URLs. Simple English Wikipedia reuse must retain
the applicable attribution and share-alike obligations. Project Gutenberg
states that its public-domain determinations are U.S.-specific, so deployment
outside that scope still requires a jurisdiction and trademark review. Crowley
texts were deliberately excluded from this checkpoint.

This is a corpus and tokenizer promotion, not a model-quality promotion. MT5
and MT6 remain fixed to the 256-byte vocabulary, but the separate `NSRLPM1`
artifact now consumes the `NSRLTOK1` u32 streams and passes bounded p10m
output-head, full-backward, and float-twin smokes. The controlled p10m
train/dev pilot is the remaining gate before the matched scaling run begins.

The exact scaling shapes are frozen in
`benchmarks/production-model-v1/scaling-plan.json`: 9,317,632 parameters,
21,641,600 parameters, and 28,229,056 parameters. Each point requires matched
integer and float runs with identical seeds, token order, context, batches, and
splits. The plan validator reports the controlled pilot as the next unsatisfied
gate rather than treating any bounded smoke as a quality training run.
