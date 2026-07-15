# Deterministic Subword Tokenizer v1

`deterministic_byte_bpe_v1` is NSRL's first versioned subword artifact. It is a
byte-complete BPE tokenizer: every input byte is always representable, including
invalid UTF-8 and arbitrary binary input.

## Token allocation

```text
0..255    literal byte fallback tokens
256       BOS
257       EOS
258       PAD
259..     learned merges in deterministic rank order
```

Special tokens never participate in learned merges and therefore cannot alias
ordinary bytes. The tokenizer artifact uses magic `NSRLBPE1`; encoded u32 token
streams use `NSRLTOK1` and bind the tokenizer artifact hash in their header.

Training repeatedly selects the most frequent adjacent token pair. Ties choose
the lexicographically smallest `(left, right)` token pair, and replacements are
non-overlapping from left to right. Training stops at the requested vocabulary
size or when no pair meets the frozen frequency floor. Encoding replays merge
rules in artifact order, so no host hash-map iteration order affects output.

## CLI

```bash
cargo run -p nsrl-corpus --bin nsrl-subword -- train \
  --corpus corpus.txt \
  --tokenizer-out tokenizer.nsrlbpe \
  --trace tokenizer.trace.json \
  --vocab-size 8192 \
  --min-pair-frequency 2

cargo run -p nsrl-corpus --bin nsrl-subword -- encode \
  --corpus corpus.txt \
  --tokenizer tokenizer.nsrlbpe \
  --tokens-out corpus.nsrltok \
  --trace corpus.tokens.json

cargo run -p nsrl-corpus --bin nsrl-subword -- decode \
  --tokens corpus.nsrltok \
  --tokenizer tokenizer.nsrlbpe \
  --out roundtrip.txt
```

The checked-in `open-generation-v1` development tokenizer is a conformance
artifact trained on the frozen substrate corpus. It requests 8,192 tokens but
stops at 1,081 because the small 9,388-byte corpus exhausts pairs occurring at
least twice. It is not the future quality tokenizer. The 8K-16K production
vocabulary must be retrained only after the larger licensed, deduplicated,
contamination-checked language split is frozen.

That production gate is now represented by `production-corpus-v1`: its
train-only 1 MiB tokenizer sample reaches all 8,192 tokens and its checked-in
checkpoint binds the tokenizer plus document-indexed train/dev/test token
streams. Indexed encoding uses rank-priority merge replay, which is
semantically locked against sequential replay while avoiding one full corpus
scan per merge.

Run the end-to-end conformance check with:

```bash
scripts/check-open-generation-v1.sh
```
