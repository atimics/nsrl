# CosyWorld Corpus

`scripts/build-cosyworld-corpus.mjs` builds a cheap, deterministic CosyWorld
voice pack for NSRL. It does not call an LLM, launch AWS, or train a model.

Run a local smoke build:

```sh
node scripts/build-cosyworld-corpus.mjs \
  --out-dir data/processed/cosyworld-smoke \
  --max-csv-items 6
```

By default the builder reads item names from:

```text
/Users/ratimics/develop/cosyworld/items_export.csv
```

If that file is missing, it falls back to a small built-in item list. The output
directory contains:

- `frames.jsonl`: schema-shaped avatar, location, and item frames.
- `training-pairs.jsonl`: compact paired records with `private_state` and
  `expected_output`.
- `corpus.txt`: flattened `RANKED`/`VOICE`/`END` training text.
- `voice.txt`: one generated line per frame for quick inspection.
- `catalog.json`: structured avatars, locations, and softened item records.
- `manifest.json`: paths, counts, and source notes.

The corpus intentionally softens copied item names into cosy descriptions rather
than copying long generated prose from CosyWorld. This keeps the local NSRL lane
small, auditable, and cheap to rebuild.

For a real simulator-state lane, use `scripts/build-cosyworld-kernel-corpus.mjs`.
That builder compiles the CosyWorld C kernel from
`/Users/ratimics/develop/cosyworld/v2/core-c`, exports raw world snapshots, and
then maps each event to `private_state` and `expected_output`. See
`docs/simulator-state-corpora.md`.

For the cheapest tiny models, train on `voice.txt` or another raw-line corpus
instead of `corpus.txt`. The `RANKED`/`VOICE`/`END` wrapper is useful for larger
ranker-to-voice experiments, but the smallest lexeme heads tend to learn the
wrapper words as output. A raw-line prompt such as `Brindle Mosscup` matches the
cheap model better than a wrapped prompt.

Frames and `training-pairs.jsonl` include `private_state` and `expected_output`.
Use the paired file for future private-state-to-output training. For the tiniest
output-only models, train on `voice.txt` so field names do not leak into
generation. The expected output is in-world prose with no awareness of being an
AI, model, or chatbot.

For context-dependent reading, use
`scripts/build-contextual-reading-pairs.mjs` instead. That builder simulates a
CosyWorld private state and pairs it with fixed Shakespeare, Blake, or Crowley
source text; see `docs/contextual-reading-pairs.md`.

For the mixed Shakespeare/Blake/Crowley storybook lane, see
`docs/cosyworld-shared-literary.md`.
