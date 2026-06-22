# CosyWorld Shared Literary Corpus

CosyWorld now has a shared literary lane that mixes Shakespeare, Blake, and
Crowley over the same CosyWorld private states.

This is related to, but distinct from, Crowley Bard. Crowley Bard is the
standalone Shakespeare x Blake x Crowley twitterbot-style corpus/model tracked
in `docs/world-llm-corpus-plan.md`; this CosyWorld lane is a world-state reading
adapter that lets CosyWorld characters imagine or produce literary text.

The purpose is the two-way simulation loop:

- Reading: a character imagines what the text would be like as private
  experience.
- Experiencing: the private state becomes generated text.

Build the shared corpus:

```sh
node scripts/build-contextual-reading-pairs.mjs \
  --out-dir data/processed/cosyworld-shared-literary-corpus \
  --domains cosyworld \
  --styles shakespeare,blake,crowley \
  --max-pairs-per-lane 36
```

That writes:

- `training-pairs.jsonl`: all CosyWorld literary pairs.
- `cosyworld-shakespeare.training-pairs.jsonl`: Shakespeare current.
- `cosyworld-blake.training-pairs.jsonl`: Blake current.
- `cosyworld-crowley.training-pairs.jsonl`: Crowley current.
- `expected-output.txt`: mixed output-only literary text.
- `manifest.json`: counts and source paths.

Train the focused shared model:

```sh
node scripts/train-focused-pair-models.mjs \
  --lanes cosyworld-shared-literary \
  --out-root data/processed/cheap-trained/cosyworld-shared-literary-v1024-d16-seq16
```

The current model is:

```text
data/processed/cheap-trained/cosyworld-shared-literary-v1024-d16-seq16/cosyworld-shared-literary/v1024-d16-seq16.nsrllm
```

Use style-specific decode priors when you want a single current from the shared
model. Use the mixed decode prior when you want blended storybook drift.
