# NSRL Schemas

This file defines the machine-readable trace contracts used by NSRL demos and
future training runs. Schema rows are JSON Lines: one complete JSON object per
line, with deterministic field order for byte-for-byte replay checks.

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
