# NSRL Swarm Cycle

The public bot should become the tip of a small-model population, not a single
generator with a few hand-tuned filters.

Current Rust entry point:

```sh
scripts/run-swarm-post-cycle.sh \
  --prompt "the first omen today is" \
  --candidates 32 \
  --parallel 8
```

The runner builds and calls two Rust binaries:

- `nsrl-swarm-cycle`: orchestration, deterministic fan-out, ranking, trace rows.
- `nsrl-train --mode lexeme-generate`: candidate generation.

Default assets are the promoted Crowley/Bard aphorism lexeme bundle:

```text
data/processed/crowley-bard-aphorism-v2/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm
data/processed/crowley-bard-aphorism-v2/v4096.vocab.tsv
data/processed/crowley-bard-aphorism-v2/v4096.tokens.u16
```

## Trace Contract

Each run writes:

- `cycle.jsonl`: run row, candidate rows, final selection row.
- `candidates.tsv`: scan-friendly score table.
- `selected.txt`: selected public text.
- `solomon-media-request.jsonl`: seed and prompt text for the Solomon illustrator.
- `candidates/candidate-*.trace.jsonl`: raw NSRL generation traces.

The ranker label defaults to `crlplrimes-proxy-v1`. That is intentionally
honest: the current ranker is a deterministic Rust proxy that uses
CRL-style evidence discipline, stable IDs, explicit features, rejection
reasons, and replayable trace rows. When `crlplrimes` exposes a tweet/text
candidate scorer, the boundary is already a single replaceable ranker layer.

## Rust Migration

New orchestration should be Rust-first. The `.mjs` scripts can retire
incrementally as each has a Rust equivalent with the same artifact contract:

1. `nsrl-swarm-cycle` replaces post candidate orchestration.
2. Corpus builders move next, starting with focused builders that already have
   stable schemas.
3. Dashboard/watch scripts move last, after their artifact formats stop moving.

This keeps the experimental surface live while pulling the durable pipeline back
into typed, replayable binaries.
