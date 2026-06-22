# Signal Romance Voice

NSRL is the voice layer for Signal ship radio; `crlplrimes` remains the ranking and authority layer. The ranker chooses grounded facts and candidate lines, then the NSRL mini-transformer learns to continue a compact `<|ranker|>...<|voice|>` prompt into short ship speech. Runtime integration should validate the generated line and fall back to the deterministic candidate when it drifts.

Run the local smoke workflow:

```sh
scripts/run-signal-romance-smoke.sh
```

For the current practical training path, use the lexeme runner:

```sh
scripts/run-signal-romance-lexeme.sh
```

The lexeme runner trains a word-ish NSRL model over the same ranker-to-voice frames. In current local experiments it moved raw generation from printable-but-ungrounded byte noise to Signal-shaped lines with about 70% raw grounding on known-frame probes.

To fetch and include optional public-domain/public style lanes:

```sh
FETCH_SOURCES=1 STYLE_BYTES=6000 OUT_DIR=data/processed/signal-romance-expanded-lexeme \
  scripts/run-signal-romance-lexeme.sh
```

The source fetcher writes `data/processed/signal-romance-sources/` with Project Gutenberg old sci-fi, NASA Apollo air-to-ground transcripts, FAA phraseology, Earhart/Itasca radio-log context, and a small non-verbatim radio-procedure seed. ATCO2 is a good future ATC source, but it is not automatically ingested because the available subset points to an end-user data agreement.

Useful knobs:

```sh
OUT_DIR=data/processed/signal-romance-smoke \
MAX_WINDOWS=32768 \
SEQ_LEN=128 \
EPOCHS=1 \
EVAL_COUNT=20 \
scripts/run-signal-romance-smoke.sh
```

Training frames use a compact plain-text shape so the model spends its early windows on the voice continuation instead of delimiter syntax:

```text
RANKED: CO stack warm at Kepler Yard.
VOICE: CO stack warm at Kepler Yard.
END
```

The workflow writes ignored artifacts under `data/processed/signal-romance-smoke/`:

- `corpus.txt`: flattened next-token corpus with Signal truth frames and small Blake/Crowley/Shakespeare/SimpleWiki style chunks.
- `corpus.tokens.u8`: identity byte token stream.
- `signal-romance.nsrlmt`: trained mini-transformer smoke model.
- `eval/eval-report.json`: grounding, printable, bounded, and fallback-oriented validation summary.

Lexeme artifacts use the same output directory shape with:

- `v*.tokens.u16`: lexeme token stream.
- `v*.vocab.tsv`: learned lexeme vocabulary.
- `v*-seq16.nsrllm`: trained lexeme voice model.
- `lexeme-eval/eval-report.json`: raw/fallback validation report.

The evaluator is intentionally conservative. It keeps raw neural metrics separate from delivered-line metrics. A raw generated line is acceptable only when it is printable, compact, grounded in the prompt terms, and free of assistant/chatbot phrasing. A failed generated line is not a failed ranker decision; it is a voice-layer miss, and the delivered line falls back to the deterministic Signal target.
