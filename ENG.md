# Engineering Design

## Overview

NSRL is a pure Rust, `no_std`-compatible, integer-only CPU neural network
runtime. The engineering priority is defensive integer design: preserve signal
bits across depth, avoid undefined arithmetic behavior, and keep the runtime
numerically auditable.

The first forward-runtime version implements a small fixed-point inference
engine, native base-2 integer attention, linear attention, static Q15 residual
streams, branchless requantization, integer RMSNorm, a gated MLP block,
deterministic trace output, and a 1,048,576-i8-weight benchmark preset.

The active engineering frontier is now quality and control in `nsrl-train`, not
whether integer training is possible. The trainer updates NSRL-native weights
under the same arithmetic contract used by inference, using i64 batch
accumulators, component-specific integer shifts, rollback safety, and traced
decode controls.

## Numeric Contract

The inference runtime must not use floating-point operations for model math.
All runtime model computation uses:

- Signed integers.
- Fixed-point multiplication.
- Integer shifts.
- CPU integer leading-zero counts.
- Integer lookup tables.
- Native base-2 softmax.
- Saturating clamps.
- Deterministic branchless rounding.

Offline tools may use floating-point math only for reference checks,
calibration experiments, or table generation. Runtime tables must be stored as
integer constants. Tests must clearly label any offline floating-point reference
path.

## Workspace Architecture

The implementation should use a Rust workspace:

```text
crates/
  nsrl-core/      no_std integer-only inference runtime
  nsrl-demo/      deterministic trace and benchmark binary
  nsrl-train/     calibration and training tools that mirror runtime math
```

`nsrl-core` is the strict no-runtime-floats crate. The current LUT generation
lives in `nsrl-core/build.rs` and emits static integer artifacts. `nsrl-demo`
is the executable evidence surface for forward traces and CPU benchmarks.
`nsrl-train` may use offline references, but its integer simulation path must
match runtime requantization, residual scales, and RMSNorm normalization exactly.
Because attention is natively base-2, `nsrl-train` must train NSRL-born models
rather than adapt standard Euler-softmax checkpoints.

## Tensor Representation

Each tensor has integer data and explicit scale metadata:

```text
Tensor {
  dtype: qint8 | qint16 | qint32
  shape: Shape
  layout: Layout
  scale: FixedScale
  data: contiguous integer buffer
}

FixedScale {
  multiplier: int32
  right_shift: uint8
}
```

Initial rules:

- Use symmetric quantization only: zero point is always zero.
- Support per-tensor activation scales outside the residual trunk.
- Support per-channel weight scales for matrix multiplication.
- Do not support dynamic residual stream scales.

Scale metadata is interpreted as an integer fixed-point transform, not as a
runtime floating-point value. For a scale transition:

```text
dst = round_half_up((src * multiplier) / 2^right_shift)
```

The multiplier must be non-negative. Scale transitions that need a left shift
must perform it as a separate checked step before multiplication.

## Data Types

Initial formats:

```text
weights:         qint8
activations:     qint16
residual stream: qint16 Q15
accumulators:    qint32
large sums:      qint64
```

Rationale:

- `qint8` weights keep memory small and matmul efficient.
- `qint16` activations give headroom for residual and normalization paths.
- Q15 residual streams provide one fixed scale for deep residual trunks.
- `qint32` accumulators are enough for many dot products if ranges are checked.
- `qint64` is reserved for RMSNorm sums, checked products, and debug range
  checks.

## Native Base-2 Attention Contract

Attention is part of the numeric core. NSRL does not approximate standard
Euler-softmax attention; it defines attention in native log2-temperature space.
There is no runtime multiplication by `log2(e)`.

Consequences:

- NSRL models must be trained or calibrated under this exact attention rule.
- Standard HuggingFace checkpoints are not a compatibility target.
- Softmax exponentiation is a shift-and-LUT primitive, not a Taylor expansion.
- The first trainable model is a tiny character predictor, not a converted
  PyTorch model.

## Attention Head Size

The attention head dimension must be a power of four:

```text
d_k in {1, 4, 16, 64, 256, ...}
```

Then:

```text
sqrt(d_k) = 2^n
scaled_qk = qk_acc >> n
```

This makes attention variance scaling an exact arithmetic right shift. The
runtime must reject non-power-of-four head dimensions.

## Static Residual Stream

Residual branches must not dynamically choose their own output scale. The
residual trunk uses one fixed arithmetic scale, initially:

```text
RESIDUAL_SCALE = Q15
RESIDUAL_DTYPE = i16
```

Inside the residual trunk, this is the only allowed residual add:

```text
residual_out = saturating_add_i16(residual_in, block_out_q15)
```

The skip path is not requantized. The block path must project its final output
to Q15 before the add:

```text
block_out_q15 = project_to_residual_q15(block_hidden)
residual_out = saturating_add_i16(residual_in_q15, block_out_q15)
```

This avoids cascading precision loss from repeated right shifts. Dynamic scale
alignment remains available for ingress, egress, serialization, and non-trunk
operations, but it is forbidden for residual trunk adds.

Debug builds must count residual saturation events. A model that repeatedly
saturates the residual stream is numerically invalid even if it returns outputs.

## Rounding

Use one deterministic runtime rounding mode:

```text
branchless round-half-up
```

For a signed `i64` intermediate:

```text
round_shift_rhu(x, shift) = (x + (1 << (shift - 1))) >> shift
```

For `shift == 0`, return `x` unchanged. Rust's signed right shift is arithmetic,
which makes this deterministic. This rule is intentionally biased for negative
halfway cases, but it is branchless, fast, and easy for the CPU pipeline.

All tests and calibration code must mirror this exact rule. Do not use
round-to-nearest-even in the runtime.

Example requantization:

```text
wide = int64(accumulator) * int64(multiplier)
scaled = round_shift_rhu(wide, right_shift)
output = saturate_to_dtype(scaled)
```

Implementation must avoid undefined behavior around shifts and overflow.
Multiplication that may exceed `int32` must promote to `int64`.

## Saturation

All narrowing conversions are saturating:

```text
int32 -> qint16: clamp to [-32768, 32767]
int32 -> qint8:  clamp to [-128, 127]
int64 -> qint16: clamp to [-32768, 32767]
```

Debug builds should count saturation events per operation. Release builds may
make counters optional.

## Core Primitives

The first implementation should provide:

- `saturate_i8`
- `saturate_i16`
- `saturating_add_i16`
- `round_shift_rhu_i64`
- `fixed_mul_i32`
- `requantize_i32_to_i16`
- `requantize_i64_to_i16`
- `project_to_residual_q15`
- `dot_i8_i16_i32`
- `matmul_i8_i16_i32`
- `sum_squares_i16_u64_checked`
- `clz_u64`
- `normalize_u64_to_lut_index`
- `attention_dot_q_k_i16_i32_checked`
- `base2_exp_neg_q15`
- `base2_softmax_i32_q15`
- `attention_weight_v_i16_q15_checked`
- `attention_row_i16_q15_checked`
- `linear_i16_i8_i16_per_channel_checked`
- `self_attention_i16_q15_checked`
- `attention_residual_block_i16_q15_checked`
- `integer_rsqrt_q30`
- `rms_norm_i16_q15_checked`
- `prenorm_attention_residual_block_i16_q15_checked`
- `reciprocal_sum_q31`
- `hard_silu_gate_q15`
- `hard_silu_q15`
- `gated_activation_i16_q15`
- `gated_mlp_i16_q15_checked`
- `prenorm_gated_mlp_residual_block_i16_q15_checked`

Every primitive needs edge-case tests.

## Layer Set

Implement layers in this order:

1. Attention dot and base-2 softmax primitives
2. Linear
3. Static residual add
4. RMSNorm
5. Gated MLP
6. Output projection
7. Optional embedding layer

## Linear Layer

Inputs:

```text
activation: qint16, per-tensor scale or residual Q15
weights:    qint8, per-output-channel scale
bias:       qint32
output:     qint16
```

Computation:

```text
acc[o] = bias[o] + sum_i activation[i] * weight[o, i]
out[o] = requantize_i32_to_i16(acc[o], output_scale[o])
```

Bias must be stored in the accumulator domain. Bias scale must match:

```text
activation_scale * weight_scale[o]
```

represented as integer scale metadata.

If the linear layer feeds the residual trunk, its output scale must be Q15. If
it is an internal expansion layer in the gated MLP, it may use a temporary
per-tensor activation scale, but the final projection must return to Q15 before
the residual add.

## Residual Add

Residual addition is a protected operation. The runtime must reject or debug
assert any attempt to add residual tensors with different scales.

Allowed residual operation:

```text
assert_same_residual_scale(left, right)
sum = saturating_add_i16(left, right)
```

Forbidden residual operation:

```text
left_aligned = requantize(left, residual_output_scale)
right_aligned = requantize(right, residual_output_scale)
sum = saturating_add(left_aligned, right_aligned)
```

Scale alignment is still useful at boundaries, but residual trunk math must not
pay a precision tax at every layer.

## RMSNorm

RMSNorm avoids mean subtraction and is simpler than LayerNorm for fixed-point
runtime, but it must not pretend that a naive lookup table can cover all
possible `i64` sums.

Computation:

```text
sum_sq = sum_i x[i] * x[i]
mean_sq = sum_sq / n
normalized = normalize_with_clz(mean_sq + eps)
inv_rms = rsqrt_lut(normalized.mantissa)
inv_rms = adjust_by_exponent(inv_rms, normalized.exponent)
y[i] = requantize(x[i] * inv_rms * weight[i])
```

This is integer block-floating arithmetic:

- `mean_sq` remains an integer magnitude.
- `leading_zeros` extracts an integer exponent.
- Shifts normalize the magnitude into the LUT input range.
- The LUT returns an integer reciprocal square root estimate.
- Integer exponent adjustment maps the estimate back to the original scale.

Engineering requirements:

- Use Rust's integer `leading_zeros` operation or an equivalent CPU `ctlz`.
- Normalize `mean_sq` into a bounded mantissa range before lookup.
- Generate the reciprocal square root table at build time and serialize it as
  integer constants. The first generated table is `RSQRT_LUT_8BIT`, covering
  normalized mantissas in `[128, 255]`.
- Optionally refine with fixed-point Newton-Raphson after the LUT estimate.
- Define `eps` as an integer in the same domain as `mean_sq`.
- Test approximation error across the full expected activation range.
- Record the normalized exponent distribution in debug traces.

## Gated MLP

Initial block:

```text
a = LinearExpand(x_q15)
b = LinearGate(x_q15)
g = hard_silu_q15(b)
h = fixed_mul(a, g)
out_q15 = LinearProjectToResidual(h)
residual_out = saturating_add_i16(x_q15, out_q15)
```

The product `a * g` creates a temporary scale and must be requantized
explicitly. The first implemented activation is a power-of-two Hard SiLU:

```text
hard_silu_gate_q15(x) = clamp((x >> 2) + 16384, 0, 32767)
hard_silu_q15(x)      = round_shift_rhu(x * hard_silu_gate_q15(x), 15)
```

This avoids division by 6 and avoids an extra sigmoid LUT while keeping the
gate monotonic, bounded, and CPU-friendly. The block output must always return
to Q15 before the residual add.

Backward primitive:

```text
hard_silu_derivative_q15(x) = (x >> 1) + 16384
grad_up                    = round_shift_rhu(grad_out * hard_silu_q15(gate), 15)
grad_gate                  = round_shift_rhu(
                               round_shift_rhu(grad_out * up, 15)
                               * hard_silu_derivative_q15(gate),
                               15
                             )
```

Because `gate` is currently an i16 Q15 activation, the clamp endpoints of
`hard_silu_gate_q15` are outside the representable runtime range. The active
derivative branch for today's MLP is therefore unconditionally
`0.5 + x / 2`, implemented as one arithmetic right shift and one integer add.

Linear transpose backward:

```text
scaled_dy[o] = round_shift_rhu(dy[o] * forward_scales[o].multiplier,
                               forward_scales[o].right_shift)

for i in 0..input_dim:
    acc = sum_o scaled_dy[o] * weights[o, i]
    grad_input[i] = requantize(acc, grad_input_scales[i])
```

The forward per-output scale is paid once in an explicit pre-scale pass into an
i32 workspace. The transpose MAC loop therefore has no dynamic shifts, no scale
branches, and no allocation. The gated MLP backward path is:

Linear weight update:

```text
scaled_dy[o] = round_shift_rhu(dy[o] * forward_scales[o].multiplier,
                               forward_scales[o].right_shift)

for o in 0..output_dim:
    for i in 0..input_dim:
        grad_w_i64 = i64(scaled_dy[o]) * i64(input[i])
        delta_i64 = round_shift_rhu(grad_w_i64 * learning_rate, learning_rate_shift)
        weights[o, i] = saturate_i8(weights[o, i] - delta_i64)
```

The `grad_w` multiply is promoted to `i64` before applying the learning rate.
Training traces must report i8 saturation count, zero-delta count, and L1
weight movement for every weight update step.

The gated MLP backward path is:

```text
grad_gated = down^T(grad_output)
grad_up, grad_gate = gated_activation_backward(up, gate, grad_gated)
grad_input = up^T(grad_up) + gate^T(grad_gate)
```

## Base-2 Integer Softmax

Softmax is defined in base-2, not as an approximation to `e^x`.

```text
shifted = logits - max(logits)
magnitude = -shifted
integer = magnitude >> LOGIT_FRAC_BITS
fraction = magnitude & ((1 << LOGIT_FRAC_BITS) - 1)
exp = EXP2_NEG_FRAC_LUT_8BIT[fraction] >> integer
sum = i64_sum(exp)
inv_sum = block_float_reciprocal(sum)
prob = exp * inv_sum
```

Masked tokens are annihilated by setting their logit to `i32::MIN`. The softmax
implementation must ignore masked logits when computing the row maximum and
must produce exactly zero output for masked positions.

Attention output must project back into the static residual scale before any
residual add.

Risks to test:

- All probabilities collapse to zero.
- One probability saturates to one too aggressively.
- Masks interact incorrectly with max subtraction.
- Long sequence sums overflow.
- Different sequence lengths change precision unexpectedly.

## Calibration And Training

Generic framework PTQ is not a viable path. NSRL needs a Rust calibration and
training crate because the model must learn under the same arithmetic rules used
by the runtime, including native base-2 attention.

`nsrl-train` must provide:

- Calibration for per-channel weight scales.
- Calibration for non-residual activation scales.
- Static residual Q15 enforcement.
- A simulation path for branchless round-half-up.
- A simulation path for native base-2 attention.
- A simulation path for block-floating RMSNorm.
- Saturation and underflow reports per layer.
- Optional high-precision reference comparisons for analysis only.

The first real training target is a roughly 1.1M-parameter character predictor:

```text
d_model:   128
layers:    4
heads:     2
d_k:       64
seq_len:   128
vocab:     character or byte-level
```

This was the first scale at which attention could show useful sequence behavior
without making numeric debugging opaque. Strict integer-only training is now
implemented for the small research lanes. The unsolved engineering questions
are:

- whether the same arithmetic scales cleanly to larger lexeme and transformer
  models,
- how to reduce dependence on source-grounded composition without losing
  coherence,
- how to tune component learning-rate shifts over time without static sweep
  magic constants,
- and whether linear attention plus integer test-time state updates can close
  enough of the softmax quality gap to justify the O(d^2) streaming interface.

## Model Serialization

The first format can be simple and explicit:

```text
model.nsrl/
  manifest.txt
  tensors.bin
  scales.bin
  lut_versions.txt
```

The manifest should include:

- Model version.
- Runtime numeric contract version.
- Residual scale contract.
- Tensor names, shapes, dtypes, offsets.
- Scale metadata.
- Layer graph.
- Lookup table versions.
- Rounding mode identifier.

A text format is acceptable for early prototypes if it keeps review and testing
simple.

## Testing Strategy

Use a layered test suite:

- Unit tests for arithmetic primitives.
- Property tests for saturation and scale transitions using `proptest`.
- Golden tests for branchless round-half-up.
- Golden tests for base-2 softmax and mask annihilation.
- Head-size tests that reject non-power-of-four `d_k`.
- Golden tests for small matrix multiplication.
- Residual tests that prove no dynamic trunk alignment occurs.
- RMSNorm approximation tests against a high-precision offline reference.
- RMSNorm exponent and mantissa normalization tests.
- Layer tests with hand-computed small tensors.
- End-to-end deterministic model tests.

Test categories:

```text
edge values: min, max, zero, one, negative one
rounding: positive and negative halfway cases
saturation: deliberate overflow inputs
scale transitions: compatible and incompatible scales
residuals: same-scale adds and rejected mismatches
reductions: short, medium, and large vectors
normalization: clz boundaries and LUT index boundaries
```

## Runtime Diagnostics

Debug builds should expose:

- Saturation count per operation.
- Residual saturation count per block.
- Minimum and maximum tensor values per layer.
- Accumulator range per matmul.
- Scale transitions per layer.
- Attention logit range, masked count, and softmax row-sum estimate.
- RMSNorm input range.
- RMSNorm leading-zero and exponent histogram.
- `rsqrt` LUT error estimate.
- Optional tensor trace dumps for tiny models.

These diagnostics are part of the product, not temporary debugging leftovers.

## Implemented Forward Evidence

`nsrl-demo` emits deterministic JSONL evidence rows:

- `nsrl.forward_trace.v1` for the inspectable `toy` preset.
- `nsrl.benchmark_trace.v1` for the larger `bench-1m` preset.

The `toy` preset runs:

```text
embedding
  -> RMSNorm -> native base-2 causal attention -> static Q15 residual
  -> RMSNorm -> power-of-two Hard SiLU gated MLP -> static Q15 residual
  -> integer output head
```

The `bench-1m` preset runs:

```text
blocks:     4
seq_len:    128
d_model:    128
heads:      2
head_dim:   64
hidden_dim: 512
weights:    1,048,576 i8 values
```

Latest captured release trace on the current development machine:

```text
elapsed_micros: 58,503
parameter_bytes: 1,097,984
workspace_bytes: 754,688
attention_residual_saturation: 0
mlp_residual_saturation: 0
final_tensor_saturation: 0
output_hash: 0xf7dd983b8ce7a156
```

Timing is machine-load dependent; the stable evidence is the deterministic
model/input/output hashes, memory envelope, arithmetic contract, and zero
saturation counts.

## Performance Plan

Correctness comes first, but CPU sympathy is part of correctness for this
project. The scalar reference should still use arithmetic choices that can
survive optimization.

Initial implementation:

- Plain scalar Rust loops.
- Contiguous row-major tensors.
- Simple cache-friendly matmul layout.
- Branchless round-half-up.
- Static residual adds.
- No handwritten SIMD.

Later implementation:

- Blocked matrix multiplication.
- SIMD kernels.
- Packed weights.
- Threaded batch or output-channel parallelism.

Every optimized kernel must match the scalar reference exactly.

## Milestone Plan

### Milestone 1: Numeric Core

Status: complete for the forward runtime.

- Rust workspace.
- `nsrl-core` crate with a `no_std` runtime surface.
- Fixed-point scale type.
- Tensor type.
- `build.rs` generation for `RSQRT_LUT_8BIT`.
- `build.rs` generation for `EXP2_NEG_FRAC_LUT_8BIT`.
- `build.rs` generation for `RECIP_LUT_8BIT_Q31`.
- Power-of-four attention head validation.
- QK dot scaling by exact arithmetic shift.
- Native base-2 softmax.
- Q15 probability-weighted V accumulation.
- End-to-end single-row attention.
- Per-channel `i16 x i8 -> i32 -> i16` linear projection.
- Q/K/V/O projected self-attention over a tiny sequence.
- Static Q15 attention residual block with saturation count.
- Block-floating RMSNorm with q15 gamma weights.
- Pre-norm attention residual block.
- Saturating casts.
- Branchless rounding and requantization.
- Static Q15 residual add.
- Dot product and small matmul.
- Unit tests for edge cases.

### Milestone 2: Block-Floating RMSNorm

Status: complete for the forward runtime.

- `build.rs` reciprocal square root table generation.
- `leading_zeros` normalization.
- Integer mantissa and exponent adjustment.
- RMSNorm scalar reference.
- Error tests against offline reference.

### Milestone 3: Stable Layers

Status: complete for the forward runtime.

- Linear layer.
- RMSNorm.
- Power-of-two Hard SiLU gate.
- Gated MLP.
- Projection back to Q15.
- Layer-level tests.

### Milestone 4: Tiny Model, Trace, And Benchmark

Status: forward execution, deterministic trace output, and `bench-1m` benchmark
are complete. Calibration and training remain open.

- End-to-end integer model execution.
- Deterministic output tests.
- Debug trace output.
- Small demo task.
- `bench-1m` 1,048,576-i8-weight forward benchmark.
- `nsrl-train` calibration path.

### Milestone 5: Larger Attention Research

- Integer logits.
- Longer-context mask handling.
- Larger base-2 attention models.
- Attention output projection to Q15.
- Collapse and saturation tests.

Status: initial 128-token, 4-block benchmark complete. Cross-machine replay,
WASM comparison, and learned weights are still future work.

## Key Risks

- Static Q15 residual scale is too narrow for deeper models.
- Branchless round-half-up bias affects quality more than expected.
- Coarse activation scales destroy accuracy outside the residual trunk.
- Residual adds silently saturate.
- RMSNorm reciprocal square root has unacceptable error.
- Block-floating exponent adjustment has boundary bugs.
- Base-2 attention learns too slowly or needs different initialization.
- Softmax reciprocal approximation collapses low-probability tokens.
- Lookup table approximations look correct locally but fail across layers.
- Calibration drifts away from runtime arithmetic.
- Tests accidentally validate against behavior that uses runtime floats.
- Premature SIMD optimization obscures numeric bugs.

## Locked Decisions

- Implementation language: Rust.
- Quantization style: symmetric.
- Activation type: `qint16`.
- Weight type: `qint8`.
- Accumulator type: `int32`, with `int64` for large reductions and checked
  intermediate products.
- Residual trunk: static Q15 `i16`.
- Runtime rounding: branchless round-half-up.
- Attention: native base-2 logit temperature and shift-exp softmax.
- Head dimensions: powers of four.
- RMSNorm magnitude handling: integer block-floating normalization with
  leading-zero counts.
- Primary model preparation path: bespoke Rust calibration and training support.
- First model scale: about 1.1M parameters, character or byte-level prediction.
