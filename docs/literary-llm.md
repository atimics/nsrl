# Crowley, Shakespeare, and Blake literary model

This workflow trains NSRL's native integer mini-transformer on an
author-balanced corpus of Aleister Crowley, William Shakespeare, and William
Blake. It produces an `NSRLMT5` checkpoint, deterministic corpus/token traces,
and a sample. It does not fine-tune or convert an external floating-point LLM.

## Sources

The default paths use the cleaned texts already present in this workspace:

- Shakespeare: the Project Gutenberg complete works (ebook 100).
- Blake: *Poems of William Blake* (ebook 574), *The Marriage of Heaven and
  Hell* (ebook 45315), and the additional cleaned Blake collections under
  `data/processed/crowley-bard-sources/`.
- Crowley: *The Household Gods* (ebook 14040), *Tannhäuser* (ebook 70261), and
  the cleaned Crowley texts under `data/processed/crowley-bard-sources/`.

Project Gutenberg marks the cited ebook editions as public domain in the USA.
Confirm the rights status in the jurisdiction and deployment context where the
model will be used. Other texts can be supplied through the environment
variables in the runner.

## Train

Run the current-format 4,096-window profile:

```bash
scripts/run-literary-llm.sh
```

The output defaults to
`data/local-runs/crowley-shakespeare-blake/`. The important files are:

- `corpus.manifest.json`: source hashes, byte balance, and corpus hash.
- `tokens.trace.jsonl`: deterministic byte-token evidence.
- `train.trace.jsonl`: training configuration and before/after metrics.
- `holdout.eval.jsonl`: read-only next-byte metrics on reserved author-balanced
  text.
- `model.nsrlmt`: the current NSRL mini-transformer checkpoint.
- `sample.txt`: a deterministic top-k sample seeded with `the soul`.

The builder caps every author at the shortest author's available UTF-8 byte
count, then interleaves author-labelled chunks. It never repeats a short source
to make its share look larger. It reserves 8,192 bytes per author by default
for held-out evaluation. Unless `NSRL_LITERARY_STRIDE` is set, the runner
derives a stride that spreads the requested windows across the full corpus.

Multiple source paths can be supplied as colon-delimited lists through
`NSRL_SHAKESPEARE_TEXTS`, `NSRL_BLAKE_TEXTS`, and `NSRL_CROWLEY_TEXTS`.
`NSRL_LITERARY_ADAPTIVE_ATTENTION=1` enables the experimental adaptive shift
controller; fixed shifts are the stable default after the measured 24K
adaptive failure.

For a longer run, increase windows and epochs explicitly:

```bash
NSRL_LITERARY_MAX_WINDOWS=65536 \
NSRL_LITERARY_EPOCHS=2 \
NSRL_LITERARY_WORKERS=8 \
NSRL_LITERARY_OUT_DIR=data/local-runs/crowley-shakespeare-blake-64k \
  scripts/run-literary-llm.sh
```

The current architecture is small (`d_model=128`, two heads, hidden dimension
256) and experimental. A completed run proves that NSRL trained a compatible
integer model on the requested corpus; it does not by itself establish fluent
generation or author-style quality. Judge that with a held-out panel rather
than the training trace alone.
