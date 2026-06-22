# World LLM Corpus Plan

The current planning target has three sibling corpora/models:

1. Signal LLM
2. CosyWorld LLM
3. Crowley Bard

They should share tooling where it helps, but they should not collapse into one
training stream until each lane is coherent on its own.

## Signal LLM

Purpose: terse in-world radio chatter from simulator state.

Input shape:

```text
Signal replay private_state
short radio expected_output
```

Main sources:

- `data/processed/signal-replay-corpus/training-pairs.jsonl`
- `data/processed/ollama-state-outputs/signal-replay-gemma4/training-pairs.jsonl`

Growth plan:

- Expand Signal replay seeds, scripts, horizons, stations, vectors, hazards, and
  traffic events.
- Use local Gemma4 only as a drafting teacher for grounded radio variants.
- Keep outputs short, operational, and grounded in callsign, station, vector,
  hazard, hull, cargo, or docking facts.

Target lanes:

- `pilot-radio`
- `control-radio`
- `station-traffic`
- `distress`
- `trade-cargo`
- `quiet-flight`

## CosyWorld LLM

Purpose: whimsical but grounded character narration, dialogue, item language,
and location texture from CosyWorld state.

Input shape:

```text
CosyWorld private_state
in-world narration/dialogue expected_output
```

Main sources:

- `data/processed/cosyworld-kernel-corpus/training-pairs.jsonl`
- `data/processed/ollama-state-outputs/cosyworld-kernel-gemma4/training-pairs.jsonl`

Growth plan:

- Expand C-kernel playthroughs with more rooms, residents, items, movement,
  checks, pickup/use/give, rest, cooking, gardening, mending, trading, sparring,
  fleeing, memories, and relationships.
- Use local Gemma4 for multiple grounded variants per state, then filter and
  review before training.
- Keep whimsy attached to actual world entities and actions.

Target lanes:

- `cosyworld-narration`
- `rati-dialogue`
- `whiskerwind-dialogue`
- `skull-dialogue`
- `item-description`
- `location-description`
- `quest-event-line`
- `cosyworld-reading-adapter`

The CosyWorld reading adapter is the two-way simulation lane: a character reads
Shakespeare, Blake, or Crowley and imagines a private world-state that could
produce it. It is useful, but it is not the Crowley Bard itself.

## Crowley Bard

Purpose: the Shakespeare x Blake x Crowley twitterbot voice: compact,
source-balanced, strange public-domain oracle posts.

Current artifact:

```text
data/processed/visionary-twitter-bot-demo/
```

Current source mix:

- Shakespeare: 120,000 bytes
- Blake: 220,000 bytes
- Crowley: 260,000 bytes
- synthetic SimpleWiki scaffold: 100,000 bytes

Current model:

```text
data/processed/visionary-twitter-bot-demo/v4096.nsrllm
```

Growth plan:

- Keep Crowley Bard as its own output-only corpus/model first.
- Use source-balanced interleaving so Shakespeare does not drown out Blake and
  Crowley, and wiki scaffold does not dominate the vocabulary.
- Expand corpus before model size, but freeze the current Twitter bot vocab for
  continuation runs. The first expanded frozen-vocab Lambda sweep
  (`visionary-expanded-frozen-v4096-sweep-20260622T075244Z`) found
  `w16384-lr24` at `8.660` bits/token versus the current expanded-corpus
  baseline at `9.901`; the candidate lives at
  `data/aws-lambda-lexeme/candidates/visionary-expanded-frozen-v4096-w16384-lr24.nsrllm`.
- A fixed-size Stage 2 continuation sweep
  (`crowley-bard-fixed-size-curriculum-stage2-20260622T171622Z`) did not
  improve the rotated held-out panel: the base scored mean `8.443`, worst
  `8.823`, while the best continuation scored mean `8.461`, worst `8.881`.
  Do not keep pushing the same short-continuation recipe.
- The aphorism-only corpus path is cleaner source material. A decode-time policy
  with source-name token bans, a function-word run cap, prose-aware quality
  weights, and word-boundary leakage checks moved the expanded candidate from
  `0/96` to `25/96` accepted strict tweets and the aphorism seq8/mean candidate
  from `0/96` to `9/96`. The expanded accepts are still mostly word-salad; the
  aphorism accepts are more sentence-like. Use aphorism as the next base and
  treat decode policy as triage, not as a replacement for a better model.
- The promoted local Crowley Bard model is now the aphorism seq8/mean base
  reduced with a gentle continuation, without increasing model size:
  `data/processed/crowley-bard-aphorism-v2/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm`.
  It keeps the 196K artifact size, improves the 10-offset held-out panel from
  mean `9.479` / worst `9.517` bits/token to mean `9.093` / worst `9.134`, and
  passed the strict tweet audit at `173/192` accepted candidates with top-24
  mean score `150.3`.
- Do not promote raw continuation candidates on eval alone. The best direct
  continuation (`lr25-o98304-w32768`) scored mean `8.516`, but collapsed into
  glue-word scaffolding and only accepted `88/192` strict samples. Weighted
  lexeme reduction with the base model is the current safe path for fixed-size
  Crowley improvements.
- Add tweet/post-length generation filters: compactness, no metadata, no source
  labels, no malformed markup, low repetition, and strong lexical flavor.
- Use strict candidate selection before any public-facing bot workflow.

Crowley Bard is being built out as a public twitterbot demo. The posting path
lives in `scripts/x-bot/` (an AWS Lambda: `lambda_function.py`, `package.sh`,
`test_lambda_function.py`). The original gate — no external posting until local
generation quality is stable and reviewable — still applies to what gets posted:
the Lambda should only publish strictly selected candidates, and quality review
stays ahead of any automated cadence.

Possible next source additions:

- King James Bible for prophetic cadence.
- Milton for long-period theological syntax.
- Romantic poetry for a bridge between Blake and dramatic verse.
- Classical myth and Arthurian material for named action and symbolic texture.
- Carefully reviewed public-domain occult/esoteric works.

## Shared Infrastructure

All three should use:

- Clean corpus manifests.
- Reproducible train/eval commands.
- Held-out eval states or prompts.
- Meta-leak filters for `AI`, `assistant`, `training`, `prompt`, wrappers, and
  JSON-shaped residue.
- Duplicate and repetition checks.
- Separate train/test splits before scaling.

Do not train the tiny raw-output models on wrapper labels unless the experiment
is specifically about wrapper syntax. For paired world models, flatten as:

```text
private_state
expected_output
```

For Crowley Bard, keep the primary stream output-only:

```text
short strange public-domain-flavored post text
```
