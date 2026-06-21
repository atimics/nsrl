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

## Near-Term Plan

1. Add a rule-based adaptive shift controller before expanding the holographic
   controller. This directly tests whether adaptive scheduling improves BPT,
   rollbacks, and dead-component movement.
2. Implement integer-friendly gated linear attention / retention so linear
   states and future holographic memories can forget stale early-training
   bindings.
3. Move source-grounded paragraph scoring out of shell/Perl into Rust so traces,
   grounding checks, and candidate scoring are faster and schema-consistent.
4. Add integer topic state or Merkle-hologram memory as an advisory bias, then
   compare generation with source grounding weakened or disabled.
5. Continue corpus work with clean, source-balanced SimpleWiki, Shakespeare,
   Blake, Crowley, and other public-domain lanes. The goal is to run out of
   useful clean data before running out of compute.
