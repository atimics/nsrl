# NSRL Schemas

This file defines the machine-readable trace contracts used by NSRL corpus,
demo, training, and generation runs. Schema rows are JSON Lines: one complete
JSON object per line, with deterministic field order for byte-for-byte replay
checks.

Some schema identifiers are not row schemas but embedded I/O contract tags
referenced from a row's `input_schema` / `output_schema` fields — for example
`nsrl.byte_prompt.v1` and `nsrl.byte_generation.v1` are the input/output tags
inside `nsrl.byte_generation_trace.v1`, not standalone rows.

## `nsrl.corpus_trace.v1`

Authority: `deterministic_corpus_preparation`

Purpose: prove that a fixed Shakespeare text file and fixed decompressed
Simple English Wikipedia XML stream produce the same training corpus bytes,
source counts, and output hash.

Non-purpose: this schema does not claim tokenization, training, language-model
quality, semantic deduplication, or exhaustive wiki markup extraction.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.corpus_trace.v1`. |
| `authority` | string | Literal `deterministic_corpus_preparation`. |
| `sources` | array | Source IDs, URLs, and expected input formats. |
| `config` | object | Corpus preparation options, including page limits. |
| `shakespeare` | object | Input and output byte counts for Project Gutenberg text. |
| `simplewiki` | object | Input bytes and wiki page acceptance/skipping counts. |
| `output` | object | Final corpus bytes, line count, and stable hash. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Corpus command:

```sh
bzip2 -dc data/raw/simplewiki-latest-pages-articles.xml.bz2 | \
  cargo run -p nsrl-corpus -- prepare \
    --shakespeare data/raw/shakespeare-gutenberg-100.txt \
    --simplewiki-xml - \
    --out data/processed/wiki-bard-corpus.txt \
    --trace data/processed/wiki-bard-corpus.trace.jsonl
```

## `nsrl.token_trace.v1`

Authority: `deterministic_byte_tokenization`

Purpose: prove that fixed corpus bytes produce the same byte-token stream and
same deterministic next-token training windows.

The default Wiki-Bard tokenizer is deliberately simple:

```text
token_id = corpus_byte
vocab_size = 256
```

This avoids vocabulary training, merge-table drift, Unicode normalization
policy, and tokenizer/runtime mismatch while the integer training stack is
still being built. A second deterministic curriculum profile,
`byte_ascii_lower_text_u8_v1`, lowercases ASCII text, maps bytes outside the
small text alphabet to spaces, and collapses whitespace before emitting byte
tokens. It is for low-entropy text-learning experiments, not a replacement for
the byte-identity audit lane.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.token_trace.v1`. |
| `authority` | string | Literal `deterministic_byte_tokenization`. |
| `tokenizer` | string | `byte_identity_u8_v1` or `byte_ascii_lower_text_u8_v1`. |
| `config` | object | Sequence length, stride, optional max-window cap, and text profile. |
| `input` | object | Corpus input byte count. |
| `tokens` | object | Token count, vocab size, and token stream hash. |
| `windows` | object | Sliding next-token window count, uncovered tail, and window hash. |
| `preview` | object | First input/target token previews for inspection. |
| `output` | object | Raw token output byte count. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Token command:

```sh
cargo run -p nsrl-corpus -- tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus.tokens.u8 \
  --trace data/processed/wiki-bard-corpus.tokens.trace.jsonl \
  --seq-len 128 \
  --stride 1
```

Ascii-lower curriculum token command:

```sh
cargo run -p nsrl-corpus -- tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus-ascii-lower.tokens.u8 \
  --trace data/processed/wiki-bard-corpus-ascii-lower.tokens.trace.jsonl \
  --seq-len 16 \
  --stride 34899 \
  --text-profile ascii-lower
```

## `nsrl.lexeme_token_trace.v1`

Authority: `deterministic_lexeme_tokenization`

Purpose: provide a stable lexical `u16` token stream so the next training lane
can learn embeddings over words and punctuation-like chunks before falling back
to byte spelling.

The tokenizer is deterministic and corpus-local. Token IDs `0..255` are
reserved for byte fallback. By default, ascii-lower lexemes are ranked by
descending count, then lexicographic tie-break, and assigned from `256`
upward. The optional `balanced` vocab profile replaces raw frequency ranking
with a capped-sqrt score so repeated lexemes keep their place without consuming
unbounded vocabulary power. The token stream is written as little-endian `u16`.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.lexeme_token_trace.v1`. |
| `authority` | string | Literal `deterministic_lexeme_tokenization`. |
| `tokenizer` | string | Literal `lexeme_ascii_lower_u16_v1`. |
| `config` | object | Sequence length, stride, optional max-window cap, max vocab, input profile, vocab profile, frequency cap, text profile, and token width. |
| `input` | object | Raw corpus byte count and normalized byte count after marker-line removal and ascii-lower normalization. |
| `lexemes` | object | Raw lexeme count, known-vocab token count, and fallback byte-token count. |
| `vocab` | object | Total vocab size, reserved byte-token count, learned lexeme entry count, and vocab TSV hash. |
| `tokens` | object | Emitted token count, output byte count, and token stream hash. |
| `windows` | object | Sliding next-token window count, uncovered tail, and window hash. |
| `preview` | object | First input/target token IDs plus first lexeme strings for inspection. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Lexeme token command:

```sh
cargo run -p nsrl-corpus -- lexeme-tokenize \
  --corpus data/processed/wiki-bard-corpus.txt \
  --tokens-out data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --vocab-out data/processed/wiki-bard-corpus-lexeme.vocab.tsv \
  --trace data/processed/wiki-bard-corpus-lexeme.tokens.trace.jsonl \
  --seq-len 32 \
  --stride 1 \
  --max-vocab 2048 \
  --lexeme-vocab-profile balanced \
  --lexeme-frequency-cap 4096 \
  --preview-tokens 32
```

## `nsrl.training_lexeme_embedding_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove the first concept-scaffold training lane over stable lexical
`u16` tokens. It trains an i16 embedding table by pulling observed
center/context lexeme pairs together and pushing deterministic negative pairs
apart.

Non-purpose: this schema does not claim language-model training, dynamic
vocabulary growth, semantic quality, grammar, spelling, or final tokenizer
quality.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_lexeme_embedding_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Literal `wiki_bard_lexeme_context_embedding_pretrain`. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and sampled window count. |
| `model` | object | Lexeme embedding model ID, vocab size, embedding width, and trained component. |
| `optimizer` | object | Deterministic integer skip-gram-style hinge update contract, concept frequency cap/min weight, cruft-aware quality profile, and positive/negative dot margins. |
| `training` | object | Epochs, context radius, stride, offset, max windows, examined windows, updates, and pair counts. |
| `metrics` | object | Initial/final positive and negative dot totals, deltas, saturation count, zero-delta count, and embedding L1 movement. |
| `initial_embedding_hash` | string | Stable hash before updates. |
| `final_embedding_hash` | string | Stable hash after updates. |
| `steps` | array | First update steps with pair IDs, concept frequency weights, quality weights, combined update weights, before/after dot products, hashes, saturation, zero-delta, and movement evidence. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Lexeme embedding pretrain command:

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-embedding \
  --tokens data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --model-out data/processed/wiki-bard-lexeme-embedding-spread4096.nsrllex \
  --trace data/processed/wiki-bard-lexeme-embedding-spread4096.trace.jsonl \
  --vocab-size 2048 \
  --embedding-dim 16 \
  --context-radius 2 \
  --stride 33264 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 9 \
  --concept-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware \
  --vocab data/processed/wiki-bard-corpus-lexeme.vocab.tsv
```

## `nsrl.training_lexeme_softmax_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove the first word/concept-level next-token trainer. It freezes a
pretrained i16 lexeme embedding table, mean-pools a power-of-two context window
into Q15 features, and trains a dynamic-vocab i8 output head with the same
integer base-2 softmax gradient used by the byte trainers.

Non-purpose: this schema does not claim coherent language generation, dynamic
vocabulary growth, Transformer backpropagation, or semantic correctness.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_lexeme_softmax_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Literal `wiki_bard_lexeme_next_token_output_head`. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and sampled window count. |
| `model` | object | Model ID, vocab size, embedding width, feature width, context length, trained component, and feature contract. |
| `optimizer` | object | Base-2 CE/SGD contract, Q formats, weight dtype, LR, LR-shift schedule, max weight delta, target frequency cap/min weight, and cruft-aware quality profile. |
| `training` | object | Epochs, sequence length, stride, offset, max windows, examined windows, and updates. |
| `metrics` | object | Initial/final mistakes, probability error, accuracy, saturation, zero deltas, and L1 head movement. |
| `steps` | array | First update steps with context/target IDs, target frequency weight, target quality weight, combined update weight, probabilities, hashes, and update diagnostics. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Lexeme softmax command:

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-softmax \
  --tokens data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --model data/processed/wiki-bard-lexeme-embedding-spread4096.nsrllex \
  --model-out data/processed/wiki-bard-lexeme-softmax-spread4096.nsrllm \
  --trace data/processed/wiki-bard-lexeme-softmax-spread4096.trace.jsonl \
  --seq-len 8 \
  --stride 33264 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 22 \
  --max-weight-delta 1 \
  --target-frequency-cap 4096 \
  --frequency-weight-min-q15 4096 \
  --quality-weight-profile cruft-aware \
  --vocab data/processed/wiki-bard-corpus-lexeme.vocab.tsv
```

Cruft-aware quality weights are deliberately soft. They downweight obvious
document-history lexemes such as `class`, `align`, `bgcolor`, `www`,
`gutenberg`, and `license` without removing them from the model. This lets the
network learn provenance and formatting structure while preventing those tokens
from dominating concept or grammar gradients.

## `nsrl.lexeme_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: prove that a saved lexeme-softmax model artifact, lexeme vocabulary,
prompt text, and integer decode policy produce the same generated `u16` lexeme
sequence and rendered text.

Current model: `lexeme_softmax_embedding_head_v1`

This is a concept-scaffold replay surface. It is phrase-level scaffolding, not
a final language-quality claim.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.lexeme_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `lexeme_softmax_embedding_head_v1`. |
| `tokenizer` | string | Literal `lexeme_ascii_lower_u16_v1`. |
| `decode` | object | Decode strategy, max tokens, sample seed, top-k, repetition controls, corpus-prior toggle, prior logit shift, and strict-adjacency toggle. |
| `decode_priors` | object or null | Source-corpus `u16` prior summary when corpus-prior or strict-adjacency decode is enabled; otherwise `null`. |
| `model_hash` | string | Stable hash of the loaded lexeme embedding/head model. |
| `embedding_hash` | string | Stable hash of the loaded i16 embedding table. |
| `output_weight_hash` | string | Stable hash of the loaded i8 output head. |
| `context_seq_len` | integer | Power-of-two lexeme context window stored in the model artifact. |
| `prompt` | object | Prompt lexeme count and token IDs. |
| `generation` | object | Generated lexeme count and token IDs. |
| `steps` | array | Per-token decode evidence with logits, probabilities, candidate counts, and rejection counts. Lexeme generation rejects reserved byte-fallback IDs during concept-only decode and records them under `rejected_candidates.byte_fallback`. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Lexeme generation command:

```sh
cargo run -p nsrl-train -- \
  --mode lexeme-generate \
  --model data/processed/wiki-bard-lexeme-softmax-spread4096.nsrllm \
  --vocab data/processed/wiki-bard-corpus-lexeme.vocab.tsv \
  --tokens data/processed/wiki-bard-corpus-lexeme.tokens.u16 \
  --prompt "to be or not to be" \
  --max-new-tokens 120 \
  --decode sample \
  --sample-seed 5 \
  --top-k 8 \
  --repeat-window 24 \
  --repeat-penalty-shift 1 \
  --max-repeat-run 3 \
  --corpus-prior \
  --strict-adjacency \
  --text-out data/processed/wiki-bard-lexeme-generation.txt \
  --trace data/processed/wiki-bard-lexeme-generation.trace.jsonl
```

## `nsrl.forward_trace.v1`

Authority: `integer_runtime_determinism`

Purpose: prove that fixed integer inputs and fixed integer weights produce the
same integer outputs, diagnostics, and hash under the declared NSRL arithmetic
contract.

Non-purpose: this schema does not claim language-model quality, PyTorch
checkpoint compatibility, Euler-softmax equivalence, or training convergence.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.forward_trace.v1`. |
| `authority` | string | Literal `integer_runtime_determinism`. |
| `arithmetic` | object | Runtime arithmetic contract. |
| `model_hash` | string | Stable hash of model constants and arithmetic metadata. |
| `input` | object | Input text and fixed-context token IDs. |
| `attention_preview` | object | Final-token attention logits, probabilities, and pre-normalization sum. |
| `attention_residual_saturation_count` | integer | Count of saturating residual additions after attention. |
| `mlp_residual_saturation_count` | integer | Count of saturating residual additions after the gated MLP. |
| `residual_saturation_count` | integer | Total saturating residual additions. |
| `layer_stats` | array | Per-layer integer health diagnostics. |
| `output` | object | Output logits, selected token, and output hash. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

### `arithmetic`

| Field | Type | Meaning |
| --- | --- | --- |
| `residual_scale_q` | integer | Static residual fractional scale. Current demo uses Q15. |
| `rounding` | string | Current demo uses `branchless_round_half_up`. |
| `seq_len` | integer | Fixed context length. |
| `d_model` | integer | Residual trunk width. |
| `hidden_dim` | integer | Gated MLP hidden width. |
| `heads` | integer | Attention head count. |
| `head_dim` | integer | `d_model / heads`; must be a power of four. |
| `sqrt_head_shift` | integer | Exact right shift used for division by `sqrt(head_dim)`. |
| `logit_frac_bits` | integer | Fixed-point fractional bits in attention logits. |
| `probability_shift` | integer | Fixed-point fractional bits in attention probabilities. |
| `attention_temperature` | string | Current demo uses `native_log2`. |
| `mlp_activation` | string | Current demo uses `hard_silu_shift2_q15`. |

### `layer_stats[]`

Each layer stat object carries:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Stable diagnostic name. |
| `hash` | string | Stable hash of the integer tensor. |
| `min` | integer | Minimum i16 value. |
| `max` | integer | Maximum i16 value. |
| `max_abs` | integer | Maximum absolute i16 magnitude, clamped for `i16::MIN`. |
| `saturation_count` | integer | Count of values equal to `i16::MIN` or `i16::MAX`. |

### Demo Command

```sh
cargo run -p nsrl-demo -- --input abba --trace /tmp/nsrl-demo.jsonl
```

Determinism gate:

```sh
cargo run -p nsrl-demo -- --input abba --trace /tmp/nsrl-a.jsonl
cargo run -p nsrl-demo -- --input abba --trace /tmp/nsrl-b.jsonl
diff -u /tmp/nsrl-a.jsonl /tmp/nsrl-b.jsonl
```

An empty diff is the expected result.

The current demo row is a complete pre-norm transformer block:

```text
embedding
  -> RMSNorm -> base-2 causal attention -> residual add
  -> RMSNorm -> gated MLP with power-of-two Hard SiLU -> residual add
  -> integer output head
```

## `nsrl.training_smoke_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove that a deterministic integer training loop can update weights
under a declared arithmetic contract and replay to the same final hashes.

Current task: `tiny_next_char_output_head`

This first training trace intentionally trains only an i8 output head over
fixed Q15 token features. It does not backpropagate through attention, RMSNorm,
or the gated MLP yet. Its job is to establish the training evidence surface:
integer updates, deterministic metrics, before/after hashes, error deltas, and
optimizer saturation counts.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_smoke_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current smoke task name. |
| `model` | object | Vocab, feature width, and trained component. |
| `optimizer` | object | Update rule, Q-format, weight dtype, learning rate, and learning-rate shift. |
| `training` | object | Epoch count, sample count, examined sample count, and applied update count. |
| `metrics` | object | Initial/final total error, mistakes, integer accuracy, and saturation count. |
| `initial_weight_hash` | string | Stable hash before updates. |
| `final_weight_hash` | string | Stable hash after updates. |
| `final_logits_hash` | string | Stable hash of replayed final logits. |
| `steps` | array | One object per applied update. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Each `steps` item includes `update_index`, `epoch`, `sample_index`, `input_id`,
`target_id`, `predicted_id`, `total_error_before`, `total_error_after`,
`error_delta_i32`, `weight_hash_before`, `weight_hash_after`, and
`gradient_saturation_count`.

Smoke command:

```sh
cargo run -p nsrl-train -- --mode perceptron --trace /tmp/nsrl-train-smoke.jsonl
```

## `nsrl.training_softmax_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove the first true classification-gradient training primitive:
`grad_q15 = prob_q15 - target_q15`, where probabilities come from the same
integer base-2 softmax used by the runtime.

Current task: `tiny_next_char_output_head_base2_softmax`

This trace still trains only the i8 output head over fixed Q15 token features.
It is the next boundary after the perceptron smoke: real softmax probabilities,
explicit Q15 gradients, deterministic i8 updates, and saturation/zero-delta
diagnostics.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_softmax_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current softmax-gradient smoke task name. |
| `model` | object | Vocab, feature width, and trained component. |
| `optimizer` | object | Base-2 CE/SGD contract, Q formats, weight dtype, LR, and LR shift. |
| `training` | object | Epoch limit, sample count, examined sample count, updates, and stop reason. |
| `metrics` | object | Classification error, Q15 probability error, saturation, zero deltas, and L1 weight movement. |
| `steps` | array | One object per softmax-gradient update. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Each `steps` item includes the pre-update `logits_q8_before`,
`probabilities_q15_before`, `gradient_q15`, pre/post total error, pre/post Q15
probability error, pre/post weight hashes, saturation count, zero-delta count,
and applied L1 weight movement.

Softmax command:

```sh
cargo run -p nsrl-train -- --mode softmax --trace /tmp/nsrl-train-softmax.jsonl
```

## `nsrl.training_linear_backward_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove the checked Linear backward infrastructure: per-channel
pre-scaling of `dY`, transposed `dX = W^T dY`, and an i64 outer-product weight
update for the i8 matrix.

Current task: `tiny_linear_layer_backward`

This trace is deliberately one layer wide. It does not claim full Transformer
backpropagation. Its job is to prove the numerically dangerous primitive that
all deeper backprop will reuse.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_linear_backward_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current linear backward smoke task name. |
| `model` | object | Input/output dimensions and trained component. |
| `optimizer` | object | Outer-product SGD contract, Q formats, weight dtype, LR, and LR shift. |
| `forward` | object | Input features, forward scales, and pre/post update outputs. |
| `backward` | object | Raw Q15 `dY`, pre-scaled i32 `dY`, input-gradient scales, and Q15 `dX`. |
| `weights` | object | Tiny before/after i8 matrix for inspection. |
| `metrics` | object | Saturation count, zero-delta count, and L1 weight movement. |
| `initial_weight_hash` | string | Stable hash before the update. |
| `final_weight_hash` | string | Stable hash after the update. |
| `output_hash_before` | string | Stable hash of the pre-update forward output. |
| `output_hash_after` | string | Stable hash of the post-update forward output. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Linear backward command:

```sh
cargo run -p nsrl-train -- \
  --mode linear-backward \
  --trace /tmp/nsrl-train-linear-backward.jsonl
```

## `nsrl.training_gated_mlp_backward_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove that the checked chain rule can update all three i8 matrices in
the power-of-two Hard SiLU gated MLP: `up`, `gate`, and `down`.

Current task: `tiny_gated_mlp_weight_backward`

This trace is the first replayable MLP-weight update. It caches the forward
`up`, `gate`, and `gated` activations, computes the backward product rule, and
applies checked outer-product SGD to each matrix.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_gated_mlp_backward_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current gated MLP backward task name. |
| `model` | object | Sequence length, model width, hidden width, and trained component. |
| `optimizer` | object | Checked SGD contract, Q format, activation, weight dtype, LR, and LR shift. |
| `forward` | object | Input, cached MLP activations, and pre/post update output. |
| `backward` | object | Q15 gradient entering the MLP output. |
| `weights` | object | Pre/post `up`, `gate`, and `down` i8 matrices. |
| `metrics` | object | Aggregate and per-matrix saturation, zero-delta, and L1 movement. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Gated MLP backward command:

```sh
cargo run -p nsrl-train -- \
  --mode gated-mlp-backward \
  --lr-shift 20 \
  --trace /tmp/nsrl-train-gated-mlp-backward.jsonl
```

## `nsrl.training_mini_transformer_mlp_trace.v1`

Authority: `deterministic_training_replay`

Purpose: wire the first miniature Transformer-shaped training loop:
trainable byte embeddings -> causal attention -> trainable gated MLP ->
trainable byte output head, then backpropagate through the output head, MLP,
attention `Q`/`K`/`V`/`O` matrices, and byte embedding table.

Current task: `wiki_bard_mini_transformer_mlp_first`

This trace consumes Wiki-Bard byte token windows. It caches embedding,
attention, MLP, and block-output activations, updates the output head, updates
the MLP `up`, `gate`, and `down` matrices from the cached forward pass, updates
all four attention projection matrices, then applies the combined residual and
attention input gradient to the byte embedding rows. The Q/K path uses a fixed
Q15 `ln(2)` gain for the native base-2 softmax derivative and a separate
Q/K-specific learning-rate shift.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_mini_transformer_mlp_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current mini Transformer MLP-first task name. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and window count. |
| `model` | object | Vocab size, sequence length, model width, heads, hidden width, and trained component. |
| `optimizer` | object | Base-2 CE/SGD contract, Q format, activation, weight dtype, LR, LR shifts, and adaptive shift flags. |
| `training` | object | Epochs, sequence length, stride, window offset, max windows, batch windows, derived batch average shift, examined windows, updates, and rollback history limit. |
| `metrics` | object | Classification/probability error, batch counts, output-head, MLP, attention, and embedding accumulator counts, rollback/rejected-window counts, final invalid-forward count, saturation, zero deltas, L1 movement, adaptive shift counts, and final shifts. |
| `adaptive_shift_events` | array | Captured rule-controller shift events, capped by `adaptive_rule_trace_event_limit`. Each event records component, reason, previous/next shift, observation batches, rejected batches, saturation count, zero-delta count, and weight-delta L1. |
| `steps` | array | One object per byte-window update with cache and weight hashes. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Safety semantics:

- Before each accepted mini-Transformer update, the trainer keeps a short
  rolling checkpoint history of prior integer model states.
- If a later window can no longer execute the checked attention forward pass,
  the trainer restores the newest checkpoint that can forward that window,
  records `rollback_count` and `rejected_window_count`, and continues.
- Final evaluation records `final_invalid_forward_count` instead of aborting on
  an invalid sparse window. Invalid final forwards count as mistakes and
  contribute a deterministic sentinel to `final_logits_hash`.
- `batch_windows` applies deterministic integer batch training hygiene. The
  output head, gated MLP, attention projections, and embeddings use true `i64`
  raw-gradient accumulators over accepted windows and apply one averaged update
  per batch without an extra batch-size learning-rate shift. The trace records
  `batch_average_shift = ceil(log2(batch_windows))` as the power-of-two scale
  implied by the batch size. Batch-gradient candidates are checked against the
  current batch windows and sparse guard windows before promotion to the saved
  model state.
- Mini Transformer artifacts with magic `NSRLMT3` serialize learned absolute
  i16 position embeddings alongside token embeddings. The model object reports
  this as `"position":"learned_absolute_i16"`.
- In `mini-transformer-mlp` mode, `--model PATH` or `--resume-from PATH` loads
  an existing `NSRLMT3` artifact and continues training from that exact integer
  state. Without a model path, the trainer initializes a deterministic scratch
  model for the requested `seq_len`.
- `--attention-vo-error-feedback` enables residual/error-feedback buckets for
  attention V/O i8 updates only. The optimizer object records this as
  `attention_vo_error_feedback`.
- `--attention-vo-oracle` disables the hand-written V/O gradient update and
  replaces it with a deterministic coordinate oracle for attention V/O weights:
  each candidate `+1`/`-1` i8 move is kept only if exact checked probability
  error improves on the current batch. This is intentionally slow and is meant
  for validation, not the default training lane.
- `--reject-loss-regression` adds an objective guard around batch-gradient
  promotion. Candidate batches must still pass checked forward validation, and
  must also not increase exact probability error over the full configured
  training window set. The optimizer object records this as
  `reject_loss_regression`, and the metrics object records
  `loss_regression_rejected_batch_count`.
- `--adaptive-rule-shifts` enables a rule-based integer scheduler for
  component learning-rate shifts. It raises shifts immediately on rollback,
  raises shifts on saturation only after sustained pressure over
  `--adaptive-rule-interval-batches`, lowers shifts only after near-total
  zero-delta pressure over the same window, and records fired decisions in
  `adaptive_shift_events`.

Mini Transformer MLP-first command:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --seq-len 4 \
  --stride 1 \
  --window-offset 0 \
  --batch-windows 1 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 18 \
  --mlp-lr-shift 16 \
  --embed-lr-shift 14 \
  --attention-lr-shift 24 \
  --attention-qk-lr-shift 18 \
  --model-out data/processed/wiki-bard-mini-transformer-mlp.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-mlp.trace.jsonl
```

## `nsrl.training_mini_transformer_progress.v1`

Authority: `deterministic_training_replay`

Purpose: provide a compact heartbeat for long-running mini Transformer training
jobs. This is not a replacement for the final replay trace; it is the live,
batch-boundary state used by the AWS dashboard.

The progress row records the same `data`, `model`, and `training` identity
fields as the final trace, then emits running counters in `metrics`:

- accepted/rejected batches, rollbacks, rejected windows,
- output/MLP/embedding/attention movement,
- attention Q/K/V/O movement,
- adaptive rule and holographic shift adjustment counts,
- current component learning-rate shifts,
- model, embedding, attention, MLP, and output-head hashes.

The runner writes this through `--progress-out PATH` every
`--progress-interval-batches N` observed batches and once more after final
evaluation. Dashboards should prefer the final
`nsrl.training_mini_transformer_mlp_trace.v1` row when it exists, and fall back
to this progress schema while the run is still active.

## `nsrl.mini_transformer_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: reload a saved mini Transformer artifact and emit a deterministic
byte-generation trace from the exact serialized integer weights and decode
configuration.

Generation always forwards the model's serialized context length. If a prompt
is shorter than `context_seq_len`, the runtime left-pads the first context with
ASCII spaces before the checked integer forward pass.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.mini_transformer_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `mini_transformer_byte_qkvo_mlp_v1`. |
| `tokenizer` | string | Byte tokenizer/profile ID used for the loaded model. |
| `decode` | object | Decode strategy, max tokens, sample seed, top-k setting, printable/ascii-lower filters, repetition penalty, run cap, corpus-prior toggle, prior logit shift, and strict-adjacency toggle. |
| `decode_priors` | object or null | Source-corpus prior summary when corpus-prior or strict-adjacency decode is enabled; otherwise `null`. |
| `model_hash` | string | Stable hash of the full serialized model state. |
| `embedding_hash` | string | Stable hash of the i16 byte embedding table. |
| `attention_hash` | string | Stable hash of the `Q`/`K`/`V`/`O` i8 matrices. |
| `mlp_hash` | string | Stable hash of the `up`/`gate`/`down` i8 matrices. |
| `output_head_hash` | string | Stable hash of the byte output-head i8 matrix. |
| `context_seq_len` | integer | Serialized model context length used per generation step. |
| `prompt` | object | Prompt byte count and raw byte tokens. |
| `generation` | object | Generated byte count and raw byte tokens. |
| `steps` | array | Per-token decode choices with selected logit, probability, candidate count, and rejection counts. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Mini Transformer generation command:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-generate \
  --model data/processed/wiki-bard-mini-transformer-mlp.nsrlmt \
  --tokens data/processed/wiki-bard-corpus-ascii-lower.tokens.u8 \
  --tokenizer ascii-lower \
  --prompt "to be" \
  --max-new-tokens 128 \
  --decode sample \
  --sample-seed 1 \
  --top-k 16 \
  --printable-only \
  --ascii-lower-only \
  --repeat-window 32 \
  --repeat-penalty-shift 2 \
  --max-repeat-run 3 \
  --corpus-prior \
  --strict-adjacency \
  --text-out data/processed/wiki-bard-mini-transformer-generation.txt \
  --trace data/processed/wiki-bard-mini-transformer-generation.trace.jsonl
```

All byte-generation trace families share the same decode-prior contract.
`decode_priors` records `token_count`, `token_hash`, and `observed_bigrams`
when `--corpus-prior` or `--strict-adjacency` is active. Each generated step
records `candidate_count` plus `rejected_candidates` buckets:
`non_printable`, `outside_ascii_lower`, `byte_fallback`, `repeat_run`,
`adjacency`, and `top_k_truncated`. `byte_fallback` is normally zero for byte
models and records concept-only lexeme decode rejections for lexeme models. The
prior corpus is loaded from `--tokens`; this is a decode constraint and
reranker, not a model-weight mutation.

## `nsrl.training_byte_softmax_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove that `nsrl-train` can consume a Wiki-Bard `.tokens.u8` stream
and perform deterministic byte-level next-token learning using native base-2
softmax gradients.

Current task: `wiki_bard_byte_next_token_output_head`

This trace trains only a 256-class i8 output head. Its feature vector is:

```text
[bias_q15, one_hot(last_context_byte)_q15]
```

This is intentionally not full Transformer backpropagation. It is the first
corpus-backed training bridge between `nsrl.token_trace.v1` and the deeper
integer trainer.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_byte_softmax_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current byte-level training smoke task name. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and window count. |
| `model` | object | Vocab size, feature width, trained component, and feature contract. |
| `optimizer` | object | Base-2 CE/SGD contract, Q formats, weight dtype, LR, and LR shift. |
| `training` | object | Epochs, sequence length, stride, window offset, max windows, examined windows, and updates. |
| `metrics` | object | Classification/probability error, saturation, zero deltas, and L1 movement. |
| `steps` | array | One object per byte-window update. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Each `steps` item includes the window start, last context byte, target byte,
pre/post predicted byte, target Q15 probability before/after, pre/post weight
hashes, saturation count, zero-delta count, and applied L1 weight movement.

Byte softmax command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-softmax \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --model-out data/processed/wiki-bard-byte-softmax.nsrlbm \
  --seq-len 128 \
  --stride 1 \
  --window-offset 0 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 25 \
  --trace data/processed/wiki-bard-byte-softmax.trace.jsonl
```

## `nsrl.byte_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: prove that a saved byte-softmax model artifact, prompt bytes, and
integer decode policy produce the same generated byte sequence.

Current model: `byte_softmax_bigram_output_head_v1`

This generation trace is a baseline replay surface, not a Transformer language
quality claim. It exists so trained Wiki-Bard artifacts can be loaded and
executed deterministically before the deeper model trainer is complete.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.byte_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `byte_softmax_bigram_output_head_v1`. |
| `tokenizer` | string | Byte tokenizer/profile ID used for the loaded model. |
| `decode` | object | Decode strategy, max tokens, sample seed, top-k setting, printable/ascii-lower filters, repetition penalty, run cap, corpus-prior toggle, prior logit shift, and strict-adjacency toggle. |
| `decode_priors` | object or null | Source-corpus prior summary when corpus-prior or strict-adjacency decode is enabled; otherwise `null`. |
| `model_hash` | string | Stable hash of the loaded i8 model weights. |
| `prompt` | object | Prompt byte count and byte-token IDs. |
| `generation` | object | Generated byte count and byte-token IDs. |
| `steps` | array | Per-token decode evidence with candidate count and rejection counts. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Byte generation command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-generate \
  --model data/processed/wiki-bard-byte-softmax.nsrlbm \
  --prompt "To be" \
  --max-new-tokens 64 \
  --decode sample \
  --sample-seed 1 \
  --top-k 16 \
  --printable-only \
  --ascii-lower-only \
  --repeat-window 32 \
  --repeat-penalty-shift 2 \
  --max-repeat-run 3 \
  --text-out data/processed/wiki-bard-byte-generation.txt \
  --trace data/processed/wiki-bard-byte-generation.trace.jsonl
```

## `nsrl.training_byte_embed_softmax_trace.v1`

Authority: `deterministic_training_replay`

Purpose: prove that `nsrl-train` can learn a small byte embedding table and an
i8 output head from Wiki-Bard token windows using only integer base-2 softmax
gradients.

Current task: `wiki_bard_byte_next_token_embedding_output_head`

This trace trains a 256-token i16 embedding table plus a 256-class i8 output
head. Its feature vector is:

```text
[bias_q15, mean(byte_embedding(context_tokens))_q15]
```

`seq_len` must be a power of two so the context mean is an exact right shift.
This is still not Transformer backpropagation, but it is the first corpus-backed
trainer whose hidden context state is learned rather than hard-coded.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_byte_embed_softmax_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current byte embedding training task name. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and window count. |
| `model` | object | Vocab size, embedding width, feature width, context length, and feature contract. |
| `optimizer` | object | Base-2 CE/SGD contract, Q formats, embedding/head dtypes, LR, and LR shifts. |
| `training` | object | Epochs, sequence length, stride, window offset, max windows, examined windows, and updates. |
| `metrics` | object | Classification/probability error, head/embedding saturation, zero deltas, and L1 movement. |
| `steps` | array | One object per byte-window update. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Each `steps` item includes the window start, first/last context byte, target
byte, pre/post predicted byte, target Q15 probability before/after, pre/post
embedding and head hashes, saturation counts, zero-delta counts, and applied L1
movement.

Byte embedding softmax command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-embed-softmax \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --model-out data/processed/wiki-bard-byte-embed-softmax.nsrlem \
  --seq-len 128 \
  --stride 1 \
  --window-offset 0 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 17 \
  --embed-lr-shift 0 \
  --trace data/processed/wiki-bard-byte-embed-softmax.trace.jsonl
```

## `nsrl.byte_embed_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: prove that a saved byte-embedding-softmax model artifact, prompt
bytes, and integer decode policy produce the same generated byte sequence.

Current model: `byte_embed_softmax_context_head_v1`

This is a baseline replay surface with learned byte embeddings. It is not a
Transformer language-quality claim.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.byte_embed_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `byte_embed_softmax_context_head_v1`. |
| `tokenizer` | string | Byte tokenizer/profile ID used for the loaded model. |
| `decode` | object | Decode strategy, max tokens, sample seed, top-k setting, printable/ascii-lower filters, repetition penalty, run cap, corpus-prior toggle, prior logit shift, and strict-adjacency toggle. |
| `decode_priors` | object or null | Source-corpus prior summary when corpus-prior or strict-adjacency decode is enabled; otherwise `null`. |
| `model_hash` | string | Stable hash of the loaded embedding/head model. |
| `embedding_hash` | string | Stable hash of the loaded i16 embedding table. |
| `output_weight_hash` | string | Stable hash of the loaded i8 output head. |
| `context_seq_len` | integer | Power-of-two context window stored in the model artifact. |
| `prompt` | object | Prompt byte count and byte-token IDs. |
| `generation` | object | Generated byte count and byte-token IDs. |
| `steps` | array | Per-token decode evidence with candidate count and rejection counts. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Byte embedding generation command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-embed-generate \
  --model data/processed/wiki-bard-byte-embed-softmax.nsrlem \
  --prompt "To be" \
  --max-new-tokens 64 \
  --decode sample \
  --sample-seed 1 \
  --top-k 16 \
  --printable-only \
  --ascii-lower-only \
  --repeat-window 32 \
  --repeat-penalty-shift 2 \
  --max-repeat-run 3 \
  --text-out data/processed/wiki-bard-byte-embed-generation.txt \
  --trace data/processed/wiki-bard-byte-embed-generation.trace.jsonl
```

## `nsrl.benchmark_trace.v1`

Authority: `integer_runtime_determinism`

Purpose: measure a deterministic forward pass at a larger model shape while
preserving the arithmetic contract, model hash, output hash, memory envelope,
and saturation diagnostics.

Non-purpose: this schema is not a training trace, not a language-quality claim,
and not a cross-machine determinism proof.

The first preset is `bench-1m`:

| Field | Value |
| --- | ---: |
| `blocks` | 4 |
| `seq_len` | 128 |
| `d_model` | 128 |
| `heads` | 2 |
| `head_dim` | 64 |
| `sqrt_head_shift` | 3 |
| `hidden_dim` | 512 |
| `i8_weight_count` | 1,048,576 |

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.benchmark_trace.v1`. |
| `authority` | string | Literal `integer_runtime_determinism`. |
| `preset` | string | Benchmark preset name, currently `bench-1m`. |
| `linear_kernel` | string | Linear implementation used: `generic_i8` or `ternary`. |
| `arithmetic` | object | Rounding, base-2 attention, fixed-point, and MLP activation contract. |
| `model` | object | Dimensions, weight count, and model hash. |
| `memory` | object | Parameter bytes and workspace bytes for the run. |
| `runtime` | object | Elapsed microseconds and integer throughput metric. |
| `saturation` | object | Aggregate saturation counts. |
| `blocks` | array | Per-block output hashes and health diagnostics. |
| `input_hash` | string | Stable hash of generated integer input activations. |
| `output_hash` | string | Stable hash of the final transformer output tensor. |
| `final_max_abs` | integer | Final tensor maximum absolute i16 magnitude. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

`runtime.tokens_per_second_x100` is an integer fixed-point throughput metric:
block-tokens per second multiplied by 100.

Demo command:

```sh
cargo run --release -p nsrl-demo -- \
  --preset bench-1m \
  --linear-kernel generic \
  --trace /tmp/nsrl-bench-1m.jsonl
```

## `nsrl.benchmark_suite.v1`

Authority: `integer_runtime_determinism`

Purpose: aggregate repeated benchmark runs after optional warmup while checking
that model, input, and output hashes stay stable.

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.benchmark_suite.v1`. |
| `linear_kernel` | string | Linear implementation used: `generic_i8` or `ternary`. |
| `runs` | object | Warmup and measured run counts. |
| `runtime.elapsed_micros` | array | Per-run measured elapsed microseconds. |
| `runtime.min_elapsed_micros` | integer | Fastest measured run. |
| `runtime.median_elapsed_micros` | integer | Median measured run. |
| `runtime.mean_elapsed_micros` | integer | Integer mean of measured runs. |
| `runtime.max_elapsed_micros` | integer | Slowest measured run. |
| `runtime.median_tokens_per_second_x100` | integer | Median block-token throughput times 100. |

Suite command:

```sh
cargo run --release -p nsrl-demo -- \
  --preset bench-1m \
  --warmup 2 \
  --repeat 5 \
  --linear-kernel ternary \
  --trace /tmp/nsrl-bench-suite.jsonl
```

## `nsrl.lexeme_softmax_eval.v1`

Purpose: report held-out lexeme bits-per-token for a softmax model with no weight
updates. Emitted by `--mode lexeme-evaluate`.

This row carries no `authority` field. All metrics sit under a nested `eval`
object.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.lexeme_softmax_eval.v1`. |
| `eval.windows` | integer | Number of evaluation windows scored. |
| `eval.vocab_size` | integer | Lexeme vocabulary size. |
| `eval.bits_per_token` | decimal | Mean −log₂ p_target over evaluation windows. |
| `eval.uniform_bits_per_token` | decimal | log₂(vocab_size) reference baseline. |
| `eval.reduction_vs_uniform` | decimal | Uniform baseline minus model bits/token. |

## `nsrl.simplewiki_extract_trace.v1`

Authority: `deterministic_corpus_preparation`

Purpose: account for Simple English Wikipedia page extraction — how many pages
were seen, accepted, and skipped — so corpus builds are auditable.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.simplewiki_extract_trace.v1`. |
| `authority` | string | Literal `deterministic_corpus_preparation`. |
| `sources` | array | Source ID, URL, and expected input format. |
| `config.max_simplewiki_pages` | integer/null | Page cap, or null for no cap. |
| `simplewiki.input_bytes` | integer | Decompressed XML bytes read. |
| `simplewiki.pages_seen` | integer | Pages encountered in the stream. |
| `simplewiki.pages_accepted` | integer | Pages kept after filtering. |
| `simplewiki.pages_skipped_redirect` | integer | Pages skipped as redirects. |
| `simplewiki.pages_skipped_namespace` | integer | Pages skipped by namespace. |

## `nsrl.linear_attention_microbench.v2`

Purpose: emit the softmax-vs-linear attention microbenchmark row from the
`linear_attention_bench` binary. This is a flat row with no `authority` field;
timing values are host observations, not universal benchmark claims.

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.linear_attention_microbench.v2`. |
| `case` | string | Case name (e.g. `seq128 d=32`). |
| `seq_len` / `d_model` / `heads` / `head_dim` | integer | Attention configuration. |
| `repeat` / `generation_repeat` | integer | Measured iteration counts. |
| `softmax_median_micros` | integer | Median softmax attention microseconds. |
| `linear_median_micros` | integer | Median full linear attention microseconds. |
| `incremental_linear_median_micros` | integer | Median incremental linear microseconds. |
| `rescan_generation_median_micros` | integer | Median prefix-rescan generation microseconds. |
| `softmax_to_linear_speedup_x100` | integer | Speedup ×100. |
| `rescan_to_incremental_speedup_x100` | integer | Speedup ×100. |
| `*_workspace_bytes` / `linear_state_bytes` / `linear_key_sum_bytes` | integer | Memory footprints. |
| `*_output_hash` | string | Stable output hashes for each path, used as determinism checks. |

## `nsrl.training_mini_transformer_mlp_binary_trace.v1`

Authority: `deterministic_training_replay`

This is the binary (non-JSONL) trace variant of
`nsrl.training_mini_transformer_mlp_trace.v1`, emitted when
`--mini-transformer-trace-format binary` is selected for `mini-transformer-mlp`
runs. Its byte layout, header, and record framing are documented separately in
[binary-trace-format.md](binary-trace-format.md). Binary traces are limited to
`mini-transformer-mlp`; swarm runs stay on JSON.

## Swarm and routing schemas

All swarm and routing rows carry `authority: deterministic_training_replay`. The
operational walkthrough — modes, flags, artifact magics, and composition modes —
lives in [mini-transformer-swarm.md](mini-transformer-swarm.md). The contracts
below list the top-level structure of each row.

### `nsrl.training_mini_transformer_swarm_trace.v1`

Purpose: summarize a `mini-transformer-swarm` run: every worker shard, its
metrics, and the promoted `best_worker_index`.

Top-level objects: `schema`, `authority`, `task`, `data` (tokenizer,
token_count, token_hash), `swarm` (worker_count, best_worker_index,
base_window_offset, base_stride, final_model_hash), `model` (vocab, seq_len,
d_model, heads, hidden_dim, position), `training` (epochs, seq_len, stride,
window_offset), and a per-worker shard array with final metrics.

### `nsrl.training_mini_transformer_swarm_progress.v1`

Purpose: compact heartbeat row for swarm runs, written at batch intervals through
`--progress-out`. Same `data`/`swarm`/`model`/`training` envelope as the swarm
trace, without the final per-worker metric block.

### `nsrl.training_mini_transformer_swarm_scaling_trace.v1`

Purpose: host scaling sweep from `mini-transformer-swarm-scaling`, sweeping worker
counts `1, 2, 4, ... N`.

Top-level objects: `schema`, `authority`, `data`, `host`, `benchmark`
(run_count, worker_counts), `model`, `training`. Each benchmark row records
elapsed nanoseconds, windows/updates per second, speedup, parallel efficiency,
effective worker count, and the best worker's final error metrics.

### `nsrl.training_mini_transformer_swarm_worker_artifact.v1`

Purpose: metadata header for one self-validating binary worker artifact (magic
`NSRLWK1`) produced by `mini-transformer-swarm-worker`.

Top-level objects: `schema`, `authority`, `data` (token_count, token_hash),
`swarm` (worker_count, base_window_offset, base_stride, base_max_windows,
base_model_hash), `artifact` (format, magic).

### `nsrl.mini_transformer_swarm_expert_manifest.v1`

Purpose: expert manifest sidecar for routers and dashboards
(`mini-transformer-swarm-manifest`, or `--manifest-out` during swarm training).

Top-level objects: `authority`, `model` (format, magic, bytes, model_hash, id),
`contract` (input_schema, output_schema, residual_scale, weight_dtype,
activation_dtype, accumulator_dtype, softmax, context_seq_len, worker_count),
plus capability tags and routing hints.

### `nsrl.mini_transformer_swarm_route_trace.v1`

Purpose: deterministic manifest router decision (`mini-transformer-swarm-route`).

Top-level objects: `schema`, `authority`, `router` (`deterministic_symbolic`),
`config`, `prompt` (bytes, hash), and a per-candidate array recording capability
match, budget checks, manifest score, prompt affinity score, rejection reason,
and selected expert index.

### `nsrl.mini_transformer_swarm_generation_trace.v1`

Purpose: generation from a composed swarm artifact
(`mini-transformer-swarm-generate`).

Top-level fields: `schema`, `authority`, `model`, `tokenizer`, `attention_kind`,
`position_policy`, `composition`, `decode`, `decode_priors`, `swarm_model_hash`,
`worker_count`, `best_worker_index`, component hashes (embedding/attention/mlp/
output_head), `context_seq_len`, `prompt`, `generation` (new_tokens, steps),
`known_non_claims`.

### `nsrl.mini_transformer_swarm_routed_generation_trace.v1`

Purpose: route-then-generate (`mini-transformer-swarm-routed-generate`). Embeds
the route decision and a normal swarm generation trace.

Top-level objects: `schema`, `authority`, `router`, `active_worker_count`,
`route` (the embedded route decision), and `generation` (the embedded swarm
generation trace).
