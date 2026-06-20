# Wiki-Bard Corpus

Wiki-Bard starts as a byte/character language model over two auditable sources:

- Project Gutenberg eBook 100, The Complete Works of William Shakespeare:
  `https://www.gutenberg.org/cache/epub/100/pg100.txt`
- Simple English Wikipedia latest pages/articles dump:
  `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

The corpus pipeline intentionally emits a trace before training. A model can
only claim it was trained on Wiki-Bard data after the corpus hash, tokenizer
hash, model seed, and training trace all line up.

## Prepare Sources

```sh
mkdir -p data/raw data/processed

curl -L \
  https://www.gutenberg.org/cache/epub/100/pg100.txt \
  -o data/raw/shakespeare-gutenberg-100.txt

curl -L \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2 \
  -o data/raw/simplewiki-latest-pages-articles.xml.bz2
```

Wikimedia publishes checksum files in the same directory:

```sh
curl -L \
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-sha1sums.txt \
  -o data/raw/simplewiki-latest-sha1sums.txt
```

## Build Corpus

For a fast smoke corpus:

```sh
bzip2 -dc data/raw/simplewiki-latest-pages-articles.xml.bz2 | \
  cargo run -p nsrl-corpus -- prepare \
    --shakespeare data/raw/shakespeare-gutenberg-100.txt \
    --simplewiki-xml - \
    --out data/processed/wiki-bard-corpus-smoke.txt \
    --trace data/processed/wiki-bard-corpus-smoke.trace.jsonl \
    --max-simplewiki-pages 128
```

For the full first corpus:

```sh
bzip2 -dc data/raw/simplewiki-latest-pages-articles.xml.bz2 | \
  cargo run -p nsrl-corpus -- prepare \
    --shakespeare data/raw/shakespeare-gutenberg-100.txt \
    --simplewiki-xml - \
    --out data/processed/wiki-bard-corpus.txt \
    --trace data/processed/wiki-bard-corpus.trace.jsonl
```

The output trace uses `nsrl.corpus_trace.v1` and records source URLs, input
byte counts, accepted/skipped wiki page counts, output bytes, output lines, and
a stable FNV-1a corpus hash.

## Current Limits

This is the deterministic data lane, not the final tokenizer or trainer:

- SimpleWiki XML must be decompressed outside the crate.
- Wiki markup cleaning is intentionally minimal.
- Output is plain text with source/page markers.
- Tokenization and training traces are separate next steps.
