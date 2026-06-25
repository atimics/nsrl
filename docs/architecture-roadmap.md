# NSRL Architecture Roadmap

## Thesis

NSRL's moat is a **no-float, integer/fixed-point, deterministic, replayable**
stack (verified: zero `f32`/`f64` in `nsrl-core`, `nsrl-train-core`,
`nsrl-web-wasm`, and the bitmap denoiser). Every architecture decision must
preserve that.

## Where we are (the ceiling)

- **Text:** the deployed model is the lexeme **mean-reduce** head (order-blind →
  word salad). The **ordered** lexeme head (concatenated per-position
  embeddings) keeps word order and reads far better, but is still shallow
  (no attention, ~one linear/low-depth head) — it will plateau with scale.
- **Image:** a few-layer integer conv denoiser; structurally good seals via the
  **text-index** path, but soft/blurry.
- Both trainers are **single-threaded**; a single Graviton box is not faster.
  Real parallelism exists only in the **Lambda swarm fan-out** (mini-transformer).

## The crlplrimes lesson (inspiration)

`~/develop/crlplrimes` gets coherence/correctness not from a big net but from a
**verifier-gated action-scorer substrate**: state → **legal candidate set**
(symbolic grammar / lexicon trie / adjacency graph / budget) → **tiny neural
scorer ranks survivors** → verifier outcome → **certified rule library** →
closes back to refine the grammar (neural-grammar-discovery). Deterministic,
replayable, certified. The neural trunk is deliberately tiny.

**NSRL already does verifier-*ranking*** (post-hoc best-of-N): `score_public_tweet_text`
for text, `score_sample` (ring/symmetry/ink) for seals. The leg up is verifier-
**gating**: constrain the legal candidate set *during* generation, plus a learned
certified grammar.

Key finding: NSRL's lexeme decoder **already has the gating primitives**
(`strict_adjacency`, `strict_topic`, `strict_memory`, `island_penalty_*`,
`prompt_topic_*`) — currently disabled by default. B1 is largely enabling,
feeding, and tuning them.

## Decision: Path C (hybrid), substrate-first

Tiny integer neural ranker **+** symbolic grammar/verifier **gate** **+**
certified rule library **+** deterministic replay. Verifier-gating now (cheap,
on-rails, preserves no-float); neural capacity scale-up next, gated by the same
verifier; generalizes to multimodal.

Rationale: coherence now without the rewrite; cloud spend only when it buys
capability; the neural scale-up compounds on a structurally-correct generator
instead of learning structure from scratch.

## Roadmap

1. **Ship ordered lexeme** to the bot/web (immediate baseline lift).
2. **B1 — text verifier-gating:** build a corpus **adjacency graph** (legal
   next-lexeme transitions over the 110M-token corpus) + lexicon/topic data;
   **enable + tune** the existing `strict_adjacency` / `prompt_topic` /
   `island_penalty` / memory gates in lexeme generation (bot + WASM). Highest
   coherence-per-effort. No swarm needed.
3. **B2 — image structural gating + reconcile seal path:** make the seal
   `score_sample` ring/symmetry/ink checks *constraints* during denoising (helps
   crispness via structure, not just more training). Resolve that native
   `nsrl-bitmap-sample` **dropped `--text-index`** (bot uses latent-model blobs;
   web uses text-index) — restore text-index natively or route the bot through
   the WASM path.
4. **A1 — attention-on-lexeme transformer:** port the integer mini-transformer
   from byte (256) to the 4096 lexeme vocab (u16 I/O, vocab dim, serialization,
   decode); gated by the B verifier. First justified Lambda-swarm spend.
5. **A2 — depth (multi-block) + long context;** **A3 — multimodal:** one integer
   transformer emitting text + image/latent tokens, structurally gated. Path to
   image/video.
6. **Cross-cutting — neural-grammar-discovery loop:** mine generation failures →
   propose certified grammar deltas → promote.

## Infra fit

- **B1/B2:** local / single instance; no swarm. Grammar construction + gating.
- **A1+:** Lambda swarm (arm64) for true parallel training; that's where
  Graviton compute converts to capability. Lexeme map-reduce reduction is
  currently unproven in the swarm worker and must be validated if we want
  parallel lexeme/transformer training.

## Status (this branch: feat/integer-transformer-scaleup)

- Widened integer mini-transformer (d_model 64 / heads 4 / mlp 256), validated.
- Scaled ordered-lexeme + scaled seal trainings run (local, integer-only).
