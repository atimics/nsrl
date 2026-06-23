# Integer Source-Grounded SimpleWiki Explainer

This is the current hero-result candidate: a deterministic integer lexeme model
generates a short SimpleWiki-style Earth explanation, then a typed Rust
evaluator scores BPT, source grounding, repetition, trace size, examples, and a
bit-exact replay row hash.

## Replay Gate

Full proof replay:

```bash
scripts/check-simplewiki-grounded-replay.sh
```

Fast development replay:

```bash
FAST=1 scripts/check-simplewiki-grounded-replay.sh
```

The full gate runs `nsrl-simplewiki-grounded-eval` with a fixed timestamp,
10,000 BPT windows, and expected row hash `ba2028c5`. The fast gate uses 1,024
BPT windows and expected row hash `ba35b515`.

## Artifact

```text
data/processed/simplewiki-expository-v1/topic-earth-curriculum-holo-sentence-stop-smoke-20260621/paragraph-bestof-earth3-lastprompt-grounded16-20260621
```

Generation recipe:

```bash
RUN_ID=earth3-lastprompt-grounded16-20260621 \
OUT_DIR=data/processed/simplewiki-expository-v1/topic-earth-curriculum-holo-sentence-stop-smoke-20260621/paragraph-bestof-earth3-lastprompt-grounded16-20260621 \
PARAGRAPH_CANDIDATES=16 \
PARAGRAPH_PROMPT_MODE=last \
scripts/run-simplewiki-topic-paragraph.sh
```

## Output

```text
the earth is an ancient planet which has been changing the whole time since its formation. different parts of earth get different amounts of sunlight. the air and water then move these pieces to lower places.
```

## Scoreboard

| Metric | Value |
| --- | ---: |
| BPT windows | 10,000 |
| Bits per token | 9.332 |
| Uniform baseline BPT | 12.000 |
| Reduction vs uniform | 2.668 |
| Selected sentences | 3 |
| Source exact-span rate | 1000 / 1000 |
| Min source bigram coverage | 1000 / 1000 |
| Min source trigram coverage | 1000 / 1000 |
| Repeated bigrams | 0 |
| Repeated trigrams | 0 |
| Max token run | 1 |
| Selected generation trace bytes | 39,063 |
| Model artifact bytes | 794,748 |
| Full replay row hash | `ba2028c5` |

## Sample Rows

| Sentence | Candidate | Seed | Exact span | Trigram coverage | Text |
| ---: | ---: | ---: | --- | ---: | --- |
| 1 | 0 | 1106 | true | 1000 | which has been changing the whole time since its formation. |
| 2 | 2 | 2309 | true | 1000 | different parts of earth get different amounts of sunlight. |
| 3 | 3 | 3415 | true | 1000 | the air and water then move these pieces to lower places. |

## Current Claim

This result proves a local deterministic integer pipeline can produce a
source-grounded SimpleWiki explanation and replay the typed eval row
bit-for-bit. The evaluator is Rust-owned (`nsrl-eval` plus
`nsrl-simplewiki-grounded-eval`), not the older Perl scoring embedded in the
paragraph-generation shell script.

## Remaining Gap

The proof is not clean-checkout complete yet. The replay currently depends on
ignored local artifacts under `data/processed/...`: model, topic token stream,
vocab, paragraph choices, and selected generation traces. The next required
move is to package a small fixture or add a fetch/build script so a stranger can
run the replay from a fresh clone without preexisting local data.
