# NSRL Schemas

This file defines the machine-readable trace contracts used by NSRL demos and
future training runs. Schema rows are JSON Lines: one complete JSON object per
line, with deterministic field order for byte-for-byte replay checks.

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

The first Wiki-Bard tokenizer is deliberately simple:

```text
token_id = corpus_byte
vocab_size = 256
```

This avoids vocabulary training, merge-table drift, Unicode normalization
policy, and tokenizer/runtime mismatch while the integer training stack is
still being built.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.token_trace.v1`. |
| `authority` | string | Literal `deterministic_byte_tokenization`. |
| `tokenizer` | string | Literal `byte_identity_u8_v1`. |
| `config` | object | Sequence length, stride, and optional max-window cap. |
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
byte embeddings -> causal attention forward pass -> trainable gated MLP ->
trainable byte output head, then backpropagate through the output head, MLP, and
the attention `Q`/`K`/`V`/`O` matrices.

Current task: `wiki_bard_mini_transformer_mlp_first`

This trace consumes Wiki-Bard byte token windows. It caches embedding,
attention, MLP, and block-output activations, updates the output head, updates
the MLP `up`, `gate`, and `down` matrices from the cached forward pass, then
updates all four attention projection matrices. The Q/K path uses a fixed Q15
`ln(2)` gain for the native base-2 softmax derivative and a separate
Q/K-specific learning-rate shift.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.training_mini_transformer_mlp_trace.v1`. |
| `authority` | string | Literal `deterministic_training_replay`. |
| `task` | string | Current mini Transformer MLP-first task name. |
| `data` | object | Tokenizer ID, token count, token hash, window hash, and window count. |
| `model` | object | Vocab size, sequence length, model width, heads, hidden width, and trained component. |
| `optimizer` | object | Base-2 CE/SGD contract, Q format, activation, weight dtype, LR, and LR shifts. |
| `training` | object | Epochs, sequence length, stride, max windows, examined windows, and updates. |
| `metrics` | object | Classification/probability error, saturation, zero deltas, and L1 movement. |
| `steps` | array | One object per byte-window update with cache and weight hashes. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Mini Transformer MLP-first command:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-mlp \
  --tokens data/processed/wiki-bard-corpus.tokens.u8 \
  --seq-len 4 \
  --stride 1 \
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 18 \
  --embed-lr-shift 16 \
  --attention-lr-shift 24 \
  --attention-qk-lr-shift 18 \
  --model-out data/processed/wiki-bard-mini-transformer-mlp.nsrlmt \
  --trace data/processed/wiki-bard-mini-transformer-mlp.trace.jsonl
```

## `nsrl.mini_transformer_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: reload a saved mini Transformer artifact and emit a deterministic
greedy byte-generation trace from the exact serialized integer weights.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.mini_transformer_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `mini_transformer_byte_qkvo_mlp_v1`. |
| `tokenizer` | string | Literal `byte_identity_u8_v1`. |
| `model_hash` | string | Stable hash of the full serialized model state. |
| `embedding_hash` | string | Stable hash of the i16 byte embedding table. |
| `attention_hash` | string | Stable hash of the `Q`/`K`/`V`/`O` i8 matrices. |
| `mlp_hash` | string | Stable hash of the `up`/`gate`/`down` i8 matrices. |
| `output_head_hash` | string | Stable hash of the byte output-head i8 matrix. |
| `context_seq_len` | integer | Maximum byte context loaded from the prompt/generated stream per generation step. |
| `prompt` | object | Prompt byte count and raw byte tokens. |
| `generation` | object | Generated byte count and raw byte tokens. |
| `steps` | array | Greedy per-token predictions with selected logit and probability. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Mini Transformer generation command:

```sh
cargo run -p nsrl-train -- \
  --mode mini-transformer-generate \
  --model data/processed/wiki-bard-mini-transformer-mlp.nsrlmt \
  --prompt "To be" \
  --max-new-tokens 128 \
  --trace data/processed/wiki-bard-mini-transformer-generation.trace.jsonl
```

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
| `training` | object | Epochs, sequence length, stride, max windows, examined windows, and updates. |
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
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 25 \
  --trace data/processed/wiki-bard-byte-softmax.trace.jsonl
```

## `nsrl.byte_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: prove that a saved byte-softmax model artifact, prompt bytes, and
greedy integer decoding policy produce the same generated byte sequence.

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
| `tokenizer` | string | Literal `byte_identity_u8_v1`. |
| `model_hash` | string | Stable hash of the loaded i8 model weights. |
| `prompt` | object | Prompt byte count and byte-token IDs. |
| `generation` | object | Generated byte count and byte-token IDs. |
| `steps` | array | Per-token greedy prediction evidence. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Byte generation command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-generate \
  --model data/processed/wiki-bard-byte-softmax.nsrlbm \
  --prompt "To be" \
  --max-new-tokens 64 \
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
| `training` | object | Epochs, sequence length, stride, max windows, examined windows, and updates. |
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
  --max-windows 4096 \
  --epochs 1 \
  --lr-shift 17 \
  --embed-lr-shift 0 \
  --trace data/processed/wiki-bard-byte-embed-softmax.trace.jsonl
```

## `nsrl.byte_embed_generation_trace.v1`

Authority: `deterministic_integer_generation`

Purpose: prove that a saved byte-embedding-softmax model artifact, prompt bytes,
and greedy integer decoding policy produce the same generated byte sequence.

Current model: `byte_embed_softmax_context_head_v1`

This is a baseline replay surface with learned byte embeddings. It is not a
Transformer language-quality claim.

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Literal `nsrl.byte_embed_generation_trace.v1`. |
| `authority` | string | Literal `deterministic_integer_generation`. |
| `model` | string | Literal `byte_embed_softmax_context_head_v1`. |
| `tokenizer` | string | Literal `byte_identity_u8_v1`. |
| `model_hash` | string | Stable hash of the loaded embedding/head model. |
| `embedding_hash` | string | Stable hash of the loaded i16 embedding table. |
| `output_weight_hash` | string | Stable hash of the loaded i8 output head. |
| `context_seq_len` | integer | Power-of-two context window stored in the model artifact. |
| `prompt` | object | Prompt byte count and byte-token IDs. |
| `generation` | object | Generated byte count and byte-token IDs. |
| `steps` | array | Per-token greedy prediction evidence. |
| `known_non_claims` | array | Claims explicitly outside this row's authority. |

Byte embedding generation command:

```sh
cargo run -p nsrl-train -- \
  --mode byte-embed-generate \
  --model data/processed/wiki-bard-byte-embed-softmax.nsrlem \
  --prompt "To be" \
  --max-new-tokens 64 \
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
