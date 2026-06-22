# Contextual Reading Pairs

`scripts/build-contextual-reading-pairs.mjs` builds cheap paired training data
for context-dependent reading. It does not train a model or call an LLM.

World-state corpora learn from observed state:

```json
{"private_state":"Kepler Yard>Prospect Ref may be unsafe; warn LM before committing.","expected_output":"Caution LM traffic on Kepler Yard>Prospect Ref."}
```

Reading corpora go the other direction. The source text is already fixed, so the
builder simulates a private Signal or CosyWorld state that could plausibly have
produced that literary passage. In other words: when a character reads, it
imagines what that text would be like as an experience; when it experiences a
state, it produces text from that private condition.

```json
{"private_state":"PILOT N2 carries the route concern as formal danger, keeping station, cargo, and consequence under a measured old cadence. Kepler Yard>Prospect Ref may be unsafe; warn LM before committing.","expected_output":"From fairest creatures we desire increase,\nThat thereby beauty's rose might never die,\n..."}
```

Run:

```sh
node scripts/build-contextual-reading-pairs.mjs
```

By default this reads simulator-state pair files:

```text
data/processed/signal-replay-corpus/training-pairs.jsonl
data/processed/cosyworld-kernel-corpus/training-pairs.jsonl
```

Build those first with `scripts/build-signal-replay-corpus.mjs` and
`scripts/build-cosyworld-kernel-corpus.mjs` when starting from a clean checkout.
Use `--signal-pairs` or `--cosyworld-pairs` to point back at older template
corpora if you intentionally want that noise mix.

To build a CosyWorld-only shared literary corpus with Shakespeare, Blake, and
Crowley currents:

```sh
node scripts/build-contextual-reading-pairs.mjs \
  --out-dir data/processed/cosyworld-shared-literary-corpus \
  --domains cosyworld \
  --styles shakespeare,blake,crowley \
  --max-pairs-per-lane 36
```

Default output goes to `data/processed/contextual-reading-pairs/`:

- `training-pairs.jsonl`: combined `private_state` to `expected_output` rows.
- `signal-shakespeare.training-pairs.jsonl`: Signal-conditioned Shakespeare.
- `signal-blake.training-pairs.jsonl`: Signal-conditioned Blake.
- `cosyworld-shakespeare.training-pairs.jsonl`: CosyWorld-conditioned Shakespeare.
- `cosyworld-blake.training-pairs.jsonl`: CosyWorld-conditioned Blake.
- `cosyworld-crowley.training-pairs.jsonl`: CosyWorld-conditioned Crowley when
  `crowley` is included in `--styles`.
- `expected-output.txt`: output-only literary chunks, only for intentional raw
  literary continuation.
- `manifest.json`: counts, input paths, and source notes.

Keep this lane separate from the tiny raw Signal and CosyWorld voice models.
Those smallest models should still train on raw output files such as
`sim-log-voice.txt` or `voice.txt`. They are not smart enough to treat JSON keys
or wrapper words as structure; they learn them as ordinary output.

For paired context-dependent training, use the JSONL rows and feed
`private_state` as the conditioning input. `simulated_private_state` is included
as an explicit alias to make the provenance clear. The `expected_output` field is
the real Shakespeare, Blake, or Crowley chunk, with no awareness of being a
model, assistant, chatbot, or training example.
