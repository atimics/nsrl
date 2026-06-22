# Current Results

This document captures the publishable state of NSRL as of 2026-06-21. The
claim boundary is intentionally narrow: NSRL is an integer-native, deterministic
training and inference system with traceable language experiments. It is not a
drop-in runtime for ordinary HuggingFace checkpoints.

## Implemented System

- `nsrl-core`: `no_std` integer runtime with static Q15 residual streams,
  checked i16/i8 linear layers, integer RMSNorm, base-2 softmax attention,
  full and incremental linear attention, gated MLP, and saturation diagnostics.
- `nsrl-corpus`: deterministic byte and lexeme tokenization, balanced
  vocabulary profiles, SimpleWiki/Gutenberg cleanup, and corpus trace rows.
- `nsrl-train`: integer SGD with i64 batch accumulators, rollback safety,
  lexeme embedding pretraining, lexeme softmax training, mini-transformer
  training, softmax and linear attention backward paths, adaptive-shift
  experiments, and deterministic generation traces.
- `nsrl-demo`: inspectable forward trace, replay test, 1M-weight benchmark, and
  linear-attention microbenchmarks.

## Forward Runtime Evidence

The `bench-1m` preset runs a 4-block, 128-token, d_model=128 transformer-shaped
integer forward pass with 1,048,576 i8 weights. Captured release rows have
reported about 49-58 ms on the current Apple M4 Max development machine,
roughly 1.1 MB of parameter bytes, about 755 KB of workspace bytes, stable
output hashes, and zero saturation events in the checked counters.

Linear attention microbenchmarks show the expected complexity behavior. At small
head width and longer sequence length, full/incremental linear attention beats
softmax attention; at wider heads and short sequence length, the advantage
shrinks because O(n*d^2) becomes comparable to O(n^2*d).

## Integer Training Evidence

Integer updates are live. The trainer accumulates raw Q-format gradient products
in i64 across batch windows before narrowing to i8 weight deltas. This avoids
the fixed-point failure mode where every single-window update quantizes to
zero.

The implemented training lanes include:

- output-head softmax training,
- MLP backward and accumulated weight updates,
- softmax attention backward,
- linear attention backward with denominator treated as constant,
- embedding updates,
- lexeme embedding pretraining,
- lexeme softmax language modeling,
- and mini-transformer training/generation.

The current linear-attention training label is:

```text
linear_numerator_straight_through_denominator_constant
```

An 8192-window d=32 run reached `final_accuracy_per_mille: 184` with zero
rollbacks and zero rejected batches. The open issue is Q gradient movement: K,
V, and O move much more strongly than Q, consistent with the causal
retrieve-after-store learning order and possibly amplified by the simplified
denominator backward.

### Adaptive Shift Controller

The rule-based adaptive shift controller has now been tested on a 32768-window
byte mini-transformer run over `wiki-bard-corpus.tokens.u8`, using
linear attention, NoPE, `seq_len=4`, and `batch_windows=2`.

Release runtime on the current Apple M4 Max development machine was effectively
unchanged versus the static baseline:

| Run | Wall Time | Probability Error Delta | Accuracy | Rollbacks | Attention Delta L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static shifts | 8.52 s | -61705691 | 30 per mille | 0 | 64230652 |
| adaptive rule shifts | 8.45 s | -111329700 | 55 per mille | 0 | 53214974 |

The adaptive run fired 269 shift events and ended at output shift 18, MLP shift
26, embedding shift 14, shared V/O shift 23, Q shift 28, and K shift 30. The
controller is therefore not a runtime bottleneck, and on this run it improved
loss and accuracy while preserving the rollback-free safety contract.

The first corrected holographic controller uses lagged binding and authority
gates, so memory learns from the prior state and can only make cooldown-limited
advisory changes when the explicit rule teacher is silent. A 512-window smoke
improved over rule-only (`191` versus `119` accuracy per mille), but the
32768-window run did not beat rule-only:

| Run | Wall Time | Probability Error Delta | Accuracy | Rollbacks | Holographic Adjustments |
| --- | ---: | ---: | ---: | ---: | ---: |
| rule+holographic advisory | 8.71 s | -104760937 | 41 per mille | 0 | 1013 |

This keeps the holographic controller in the experimental lane. The mechanism
is no longer decorative, but the best long-run policy observed so far is still
the simpler rule-based controller.

The main performance scaling issue is trace volume, not arithmetic. Each 32768
window run emitted 32768 per-window step rows and produced a roughly 40 MB
trace for a roughly 34 KB model artifact. Larger hero runs should add trace
thinning or summary-only traces before scaling to hundreds of thousands of
windows.

## Lexeme Language Evidence

The strongest current text result is a source-grounded SimpleWiki topic
composition over lexeme tokens. The selected run:

```text
data/processed/simplewiki-expository-v1/topic-earth-curriculum-holo-sentence-stop-smoke-20260621/paragraph-bestof-earth3-lastprompt-grounded16-20260621
```

produced:

```text
the earth is an ancient planet which has been changing the whole time since its formation. different parts of earth get different amounts of sunlight. the air and water then move these pieces to lower places.
```

Trace facts:

- schema: `nsrl.simplewiki_topic_paragraph_bestof.v1`
- prompt: `the earth is an ancient planet`
- paragraph sentences: 3
- candidates per sentence: 16
- prompt mode: last accepted sentence
- sentence-terminal stop: enabled
- source exact span required: true
- selected sentence source trigram per mille: 1000 for all selected sentences

This proves the full system can produce coherent, traceable prose under a
declared source-grounded authority. It does not prove that the core model alone
has a durable internal topic representation.

## Current Architectural Reading

The system has two layers:

1. The integer model, which learns local token and lexeme transitions under the
   fixed-point arithmetic contract.
2. The composer, which applies corpus priors, topic priors, memory priors,
   source grounding, repetition controls, and best-of selection.

The next frontier is to move topic control from literal source-span grounding
into an integer latent state: a small topic or Merkle-hologram memory that can
survive across sentence boundaries and bias generation without requiring exact
source overlap.

## World-Model Prototypes

Corpus work is now organized as three sibling tracks, each with a trained tiny
model on disk. Signal and CosyWorld are state-conditioned world models
(`private_state` → `expected_output`); Crowley Bard is an output-only literary
voice. See `docs/world-llm-corpus-plan.md` for the full plan.

| Track | Corpus | Model | Train tokens / windows | Train accuracy |
| --- | --- | --- | --- | --- |
| Signal LLM | 486 replay pairs | `cheap-trained/sim-state-pair-v1024-d16-seq16/signal-replay/v345-d16-seq16.nsrllm` | 37,386 / 37,354 | 915 / 1000 |
| CosyWorld LLM | 23 kernel states / 27 frames | `cheap-trained/sim-state-pair-v1024-d16-seq16/cosyworld-kernel/v464-d16-seq16.nsrllm` | 68,672 / 68,640 | 901 / 1000 |
| Crowley Bard | Shakespeare 120k / Blake 220k / Crowley 260k / synthetic SimpleWiki 100k bytes | `visionary-twitter-bot-demo/v4096.nsrllm` | 194,219 lexeme tokens / 131,072 softmax windows | 9.890 bits/token on local lexeme eval |
| Crowley Bard expanded candidate | Balanced prose expanded corpus retokenized with frozen Twitter bot vocab | `aws-lambda-lexeme/candidates/visionary-expanded-frozen-v4096-w16384-lr24.nsrllm` | 359,879 lexeme tokens / 16,384 continuation windows | 8.660 bits/token on expanded frozen-vocab eval |

- Signal corpus is built from the real `signal_replay` simulator at
  `/Users/ratimics/develop/signal`. CosyWorld corpus is built from the C kernel
  at `/Users/ratimics/develop/cosyworld/v2/core-c`.
- A local Gemma4 teacher pipeline (`scripts/generate-state-outputs-ollama.mjs`)
  drafts grounded output variants from simulator states. Smoke runs accept 6/6
  Signal rows and 5/6 CosyWorld rows (with `--attempts 8`).
- Crowley Bard is the most mature standalone demo and is being prepared as a
  public twitterbot demo (see `scripts/x-bot/` and `docs/world-llm-corpus-plan.md`).
- The 2026-06-22 expanded-corpus lexeme sweep
  (`visionary-expanded-frozen-v4096-sweep-20260622T075244Z`) kept the model
  shape fixed and continued from the current bot on a frozen-vocab expanded
  corpus. The best Lambda candidate was `w16384-lr24` (`worker-006`), improving
  the expanded-corpus eval from `9.901` to `8.660` bits/token. The local live
  dashboard is served from `http://127.0.0.1:8765/` while the run workspace is
  present.

These are deliberately tiny (d_model=16, ~1k vocab for the state lanes). High
train accuracy reflects small, formulaic corpora — the open gap is scale, not
the integer contract. The numbers above are training-set metrics, not held-out
quality.

## Near-Term Plan

1. Run longer comparisons with the implemented rule-based adaptive shift
   controller. This directly tests whether adaptive scheduling improves BPT,
   rollbacks, and dead-component movement before expanding the holographic
   controller.
2. Implement integer-friendly gated linear attention / retention so linear
   states and future holographic memories can forget stale early-training
   bindings.
3. Move source-grounded paragraph scoring out of shell/Perl into Rust so traces,
   grounding checks, and candidate scoring are faster and schema-consistent.
4. Add integer topic state or Merkle-hologram memory as an advisory bias, then
   compare generation with source grounding weakened or disabled.
5. Scale the three sibling tracks past their current prototypes (see
   World-Model Prototypes above): expand Signal replay states and CosyWorld
   kernel playthroughs, generate and filter Gemma4 teacher variants, then train
   each lane separately before blending anything. Crowley Bard remains the
   Shakespeare x Blake x Crowley twitterbot voice, not just a CosyWorld literary
   adapter. See `docs/world-llm-corpus-plan.md`.
