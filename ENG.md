# Engineering Design

## Overview

NSRL is a pure Rust, `no_std`-compatible, integer-only neural network runtime
and training stack for local-first micro-model agents. The engineering priority
is defensive integer design in service of the agentic edge: preserve signal bits
across depth, avoid undefined arithmetic behavior, keep the runtime numerically
auditable, and make many small experts cheap to route, ship, and replay.

The first forward-runtime version implements a small fixed-point inference
engine, native base-2 integer attention, linear attention, static Q15 residual
streams, branchless requantization, integer RMSNorm, a gated MLP block,
deterministic trace output, and a 1,048,576-i8-weight benchmark preset.

The active engineering frontier is no longer whether integer training is
possible. It is whether integer-native training can support a useful
free-running language model. The next critical path is a quality-capable dense
backbone: better tokenization, fully trained normalization and position
handling, scalable integer optimization, a KV-cached decoder, enough clean
training data, and an evaluation contract that separates native generation from
retrieval or memory assistance. Expert routing, generated backward code, and
WASM packaging remain important, but routing starts after this backbone clears
its quality gate.

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
  nsrl-corpus/    deterministic corpus and tokenizer tooling
  nsrl-demo/      deterministic trace and benchmark binary
  nsrl-eval/      frozen proof contracts and comparison policy
  nsrl-train-core/no_std borrowed-workspace training steps
  nsrl-train/     calibration and training tools that mirror runtime math
```

`nsrl-core` is the strict no-runtime-floats crate. The current LUT generation
lives in `nsrl-core/build.rs` and emits static integer artifacts. `nsrl-corpus`
owns deterministic data preparation and tokenizer contracts. `nsrl-demo` is the
executable evidence surface for forward traces and CPU benchmarks.
`nsrl-train-core` is the allocator-free training-step extraction for callers
that own memory. `nsrl-train` may use offline references, but its integer
simulation path must match runtime requantization, residual scales, and RMSNorm
normalization exactly. Because attention is natively base-2, `nsrl-train` must
train NSRL-born models rather than adapt standard Euler-softmax checkpoints.

Planned workspace surfaces:

```text
crates/
  nsrl-router/    deterministic expert routing and context handoff
  nsrl-graph/     integer graph IR and generated no_std forward/backward Rust
  nsrl-wasm/      browser packaging, SIMD probes, and local-first demos
```

Generated code is permitted only when it emits ordinary reviewable Rust, uses
the same checked integer primitives, and has golden tests against hand-written
references. Code generation must reduce manual backward-pass burden without
becoming an opaque framework.

## Agentic Edge Architecture

NSRL scales upward to the minimum competent dense backbone, then outward. The
default eventual deployment target is one shared base generator plus optional
small expert residuals or models selected by a deterministic router:

```text
request
  -> router
  -> competent base generator
  -> zero or more expert manifests/adapters
  -> base or expert-augmented inference
  -> local trace
  -> response or tool decision
```

Each expert artifact must declare:

- capability tags,
- tokenizer contract,
- input and output schema,
- numeric contract version,
- lookup table versions,
- model and tensor hashes,
- routing hints,
- authority and known non-claims,
- native and WASM bundle budgets.

The router starts symbolic and deterministic. A learned router can be added only
after the symbolic route trace is stable enough to act as a reference. Routing is
part of the product surface: route decisions must be replayable, explainable, and
cheap enough to preserve sub-100ms local interactions for small active expert
sets.

Routing is not on the open-generation critical path until the base generator
passes the frozen native-generation contract. Router, retrieval, corpus-prior,
and memory-assisted outputs must use separate evaluation labels and may not be
reported as evidence that the base model learned open-ended generation.

## Quality-Critical Architecture Progression

The current `small-h8-d128-ff256`/`small-h2-d128-ff256` model is a numeric
substrate probe. Its compile-time dimensions, byte vocabulary, short context,
and two-layer trunk are useful for bit-exact debugging but are not an adequate
open-generation target. Engineering should promote profiles through three
explicit tiers:

```text
substrate:
  layers=2, d_model=128, heads=8 or 2, ff=256, byte vocab, context=64

quality-dev:
  layers=8-12, d_model=256 or 384, head_dim=64 or 16,
  gated ff=3-4x d_model, vocab=8K-16K, context=512-1024

open-gen:
  layers=12-20, d_model=384 or 512, head_dim=64,
  gated ff=3-4x d_model, vocab=16K-32K, context>=1024,
  approximately 30M-100M parameters depending on tied embeddings
```

These are experiment envelopes, not commitments to a parameter count. A larger
profile is authorized only when the prior profile shows improving held-out loss,
healthy activation occupancy, bounded saturation and zero-update rates, and a
narrowing gap to a same-shape floating-point twin. Depth, vocabulary, context,
and token budget must not all change in one experiment.

If a healthy 100M-scale candidate fails `open-generation-v1`, stop the automatic
scale ladder. Produce a decision record that identifies whether the limiting
factor is integer optimization, architecture, data, or the local parameter
budget, then either widen the declared deployment envelope or narrow the
product claim. Expert routing is not an accepted workaround for failed base
language quality.

The quality reference block remains pre-norm causal attention plus a gated MLP,
but requires all of the following:

- trained RMSNorm gamma with backward support through normalization,
- a fixed-point relative-position or rotary-position implementation,
- causal base-2 softmax as the quality reference attention,
- a 3-4x gated MLP expansion unless ablation supports a smaller ratio,
- optional tied token embedding/output matrices after a representation ablation,
- biases only where an ablation demonstrates a material quality gain,
- a final trained RMSNorm before the output projection,
- a reusable incremental KV cache, and
- deterministic fixed-point sampling beyond greedy and top-k.

Linear, streaming, and TTT attention remain research alternatives. They do not
become the quality default because they are asymptotically attractive; they must
first match the base-2 reference on the same frozen language evaluation.

## Tokenization And Position Contract

Byte tokens remain the universal fallback, serialization alphabet, and frozen
substrate contract. They are too sequence-inefficient to be the default path for
high-quality language training. `nsrl-corpus` must add a versioned deterministic
subword tokenizer with:

- byte fallback for every input,
- deterministic normalization and UTF-8 handling,
- a frozen vocabulary and merge/model hash,
- special-token allocation that does not alias ordinary bytes,
- encode/decode round-trip tests,
- token-frequency and sequence-compression reports, and
- identical native and WASM tokenization fixtures.

The first target vocabulary is 8K-16K. Increase it only when reduced sequence
length repays the larger embedding/output table. Weight tying requires an
explicit representation design because current token embeddings are i16 while
the output matrix is i8. Evaluate a canonical i8 shared table with deterministic
i16 lookup expansion against untied i16/i8 tables; keep the untied form as the
quality default until the tied form demonstrates parity.

Learned absolute positions and NOPE remain proof controls. The quality path
must implement one extrapolatable integer scheme. The preferred first candidate
is fixed-point rotary position encoding with build-generated integer sin/cos
tables and bit-exact native/WASM fixtures. Fixed-point ALiBi is the fallback if
rotary range or precision is unhealthy. Runtime floating-point trigonometry is
forbidden.

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
- The first proof trainable model is a tiny character predictor, not a converted
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

The active proof target is the frozen two-layer `NSRLMT5` substrate profile:

```text
d_model:   128
layers:    2
heads:     8 promoted / 2 control
d_k:       16 promoted / 64 control
hidden:    256
seq_len:   64 for integer-transformer-proof-v1
vocab:     256 bytes
```

This profile determines whether the numeric substrate can learn; it is not the
quality target. Strict integer-only training is implemented for small research
lanes. The next engineering shape is:

- a 10M-30M dense quality-development profile before expert routing,
- deterministic parallel batch-gradient accumulation,
- adaptive integer optimizer state instead of hand-tuned static shifts,
- generated checked backward code for new architectures,
- trace-thinning so training evidence does not dominate model size,
- resumable sharded corpus iteration with exact sample-order replay, and
- periodic unassisted generation on prompts that are not used for selection.

The remaining research questions are:

- whether the same arithmetic scales cleanly to larger transformer experts,
- whether strict i8 model weights plus integer optimizer residuals preserve
  enough update resolution across 8-20 layers,
- and whether linear attention plus integer test-time state updates can close
  enough of the softmax quality gap to justify the O(d^2) streaming interface.

The high-quality path does not permit silent relaxation to float master weights.
Integer first/second moments and integer update residuals are optimizer state,
not inference weights, and are allowed. If strict i8-from-initialization cannot
retain quality at the development profile, the project must record that result
and propose a versioned integer-only training contract; it must not quietly use
a float shadow model while preserving the old claim.

Offline floating-point teachers may produce reference metrics, curriculum
labels, or distilled probability targets. They may not provide converted model
weights or runtime computation, and teacher-assisted runs must be labeled
separately from pure next-token training. Distillation is an ablation after the
base data and optimizer path is healthy, not a substitute for them.

## Language Data Contract

Open-ended quality is primarily a data-and-optimization problem once the block
is structurally adequate. Every language run beyond the substrate proof must be
bound to a corpus manifest containing:

- immutable source identifiers, hashes, rights metadata, and collection dates,
- normalization and filtering versions,
- exact train, validation, and test partition hashes,
- document- and span-level deduplication reports,
- benchmark and prompt-panel contamination checks,
- tokenizer ID, vocabulary hash, and token counts by source/domain,
- sampling weights and deterministic sample-order seed,
- number of unique tokens seen and effective epochs, and
- rejected-source and quality-filter counts.

The training progression is:

1. Prove optimization on synthetic and small literary controls.
2. Freeze a diverse, clean language corpus and deterministic tokenizer.
3. Run token-budget scaling experiments at one architecture size.
4. Run depth/width experiments at a fixed token budget.
5. Select the smallest profile with a stable validation-loss and generation
   trend.
6. Train the open-generation candidate once the evaluation contract is frozen.
7. Add instruction or dialogue tuning only after base continuation quality is
   established.

The default token budget should be expressed as tokens per non-embedding
parameter and must cover several increasing budgets. A single epoch count is
not comparable across tokenizer or corpus changes. Repeated data is reported as
effective epochs and may not be hidden by a large raw token count.

## Open-Generation Evaluation Contract

`integer-transformer-proof-v1` remains the substrate gate. A separate versioned
`open-generation-v1` contract must be frozen before claiming language quality.
It has three evidence layers:

```text
modeling:
  held-out loss/bits-per-byte, accuracy, calibration, float-twin gap

generation health:
  repetition, unique n-grams, entropy, premature EOS, UTF-8 validity,
  context-use and distractor-resistance probes

human/product quality:
  blinded coherence, relevance, consistency, style control, and preference
  on a diverse unseen prompt panel
```

Required baselines are byte n-gram, retrieval, the best smaller NSRL checkpoint,
and a same-shape floating-point arithmetic twin trained with the same data,
tokenizer, objective, and token budget. The twin must use the same base-2
attention temperature, activation equations, position scheme, and shape; its
floating-point arithmetic and optimizer are the independent variables. A
larger teacher may be diagnostic but is not the parity baseline.

Cross-tokenizer modeling quality uses bits per original UTF-8 byte as the
primary loss metric. Token loss is reported but cannot compare a byte model with
a subword model. Define retained improvement as:

```text
retained = (bpb_statistical_baseline - bpb_integer)
           / (bpb_statistical_baseline - bpb_float_twin)
```

The contract must require the float twin to improve the baseline by a minimum
non-trivial margin before this ratio is meaningful.

The headline generation panel must:

- disable retrieval, corpus priors, memory injection, target lookup, and routing
  oracles,
- include continuation, constrained style, explanation, dialogue, long-context
  reference, and adversarial repetition prompts,
- preserve a hidden final test partition,
- evaluate greedy replay plus multiple predetermined sampling seeds,
- score at least 512 generated subword tokens where the prompt permits,
- retain every sample, not only selected examples, and
- identify decoding parameters and model/tokenizer hashes in every row.

Initial promotion criteria are:

- pass `integer-transformer-proof-v1`,
- close at least 90% of the held-out-loss improvement from the required
  statistical baseline to the same-shape float twin,
- show monotonic aggregate validation improvement across three token or model
  budgets,
- clear frozen degeneration and context-use floors,
- show no numerical-health regression, and
- beat the smaller NSRL baseline and remain within a predeclared non-inferiority
  margin of the float twin in blinded human preference.

The exact thresholds for repetition, context use, and human preference belong
in the frozen eval contract, not in training scripts. Prompted, routed,
retrieval-assisted, memory-assisted, and native scores remain separate fields.

## Incremental Generation Path

The quality profile requires an incremental decoder rather than recomputing the
entire prefix for every token. The KV-cache contract must specify layer, head,
position, dtype, scale, layout, context limit, and eviction policy. For a prefix
within the supported context, cached and full-prefix logits must match bit for
bit.

The first decoder supports:

- greedy decoding,
- deterministic top-k,
- fixed-point temperature and top-p sampling,
- repetition diagnostics without using penalties in the headline quality run,
- explicit EOS/minimum-length handling, and
- bounded context with an observable truncation policy.

Repetition penalties, corpus priors, lexeme memory, strict adjacency, and
retrieval can be product features, but they cannot repair or mask the native
generation score. Performance traces must report prefill time, time to first
token, decode tokens per second, KV-cache bytes, peak workspace bytes, and
sampler overhead.

## Adaptive Integer Optimizer

Static `learning_rate_shift` values are acceptable for proving integer weight
movement, but they are not a depth strategy. The training stack needs an integer
adaptive optimizer that can rebalance gradients without runtime floats.

Initial optimizer target:

```text
gradient_i64
  -> first moment i32 or i64 accumulator
  -> magnitude/variance tracker u16 or i32
  -> dynamic shift from leading-zero / magnitude history
  -> bounded i8 or i16 weight delta
```

Implementation rules:

- Start blockwise, then move per-parameter only where measurements justify the
  memory cost.
- Keep optimizer state training-only; inference artifacts contain only model
  weights, scales, manifests, and LUT versions.
- Derive update shifts from integer magnitude history, using operations such as
  `leading_zeros`, saturating counters, and bounded right shifts.
- Preserve deterministic reduction order so parallel training remains replayable.
- Trace optimizer state summaries without dumping full optimizer tensors by
  default.

The first accepted version can be closer to integer RMSProp/Adam than exact
floating-point Adam. The requirement is not bitwise compatibility with Adam; the
requirement is automatic per-block or per-weight scale control under NSRL's
integer contract.

## Deterministic Parallel Training

The host trainer must move from "mutate the model per window" toward a
deterministic map-reduce batch shape:

```text
read-only model snapshot + token window -> private gradient accumulator
private accumulators in stable chunk order -> batch accumulator
batch accumulator -> single integer update application
updated candidate model -> validation / rollback / trace
```

This structure gives CPU threads real work while preserving replay. The
single-writer apply step remains responsible for saturation counts, rollback
policy, adaptive optimizer state, and trace ordering.

`nsrl-train-core` should therefore grow two families of APIs:

- step-and-update APIs for tiny no_std appliance demos,
- gradient-only APIs for parallel host training.

The gradient-only APIs are the product path for larger experts.

## Graph-Generated Backward Passes

Hand-written backward passes are useful as golden references, but they are not a
scalable architecture research strategy. `nsrl-graph` should provide a small
static integer graph IR that can emit checked `no_std` Rust for forward and
backward passes.

Initial graph operations:

- linear projection,
- residual add,
- RMSNorm,
- Hard SiLU gated activation,
- base-2 softmax,
- causal attention,
- linear attention,
- embedding lookup,
- output head.

Generation requirements:

- Emit readable Rust source, not bytecode.
- Inject shape checks, range checks, scale assertions, saturation counters, and
  deterministic rounding calls.
- Reuse `nsrl-core` primitives rather than generating novel arithmetic.
- Compare generated backward code against current hand-written backward fixtures.
- Version generated code in traces so model artifacts can identify the graph
  contract that produced them.

The goal is to make new integer architectures cheap to try: write the forward
graph, generate the backward code, and keep the emitted Rust auditable.

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
- Expert capability tags.
- Tokenizer contract.
- Input and output schema.
- Routing hints.
- Authority and known non-claims.
- Native and WASM bundle budgets.
- Tensor names, shapes, dtypes, offsets.
- Scale metadata.
- Layer graph.
- Lookup table versions.
- Rounding mode identifier.

A text format is acceptable for early prototypes if it keeps review and testing
simple.

## WASM Deployment Contract

Browser deployment is a first-class engineering target because it is the
distribution wedge for local-first AI. The WASM path must preserve the same
integer model contract rather than becoming a separate runtime.

Requirements:

- Build `nsrl-core` for `wasm32` without runtime floating-point model math.
- Keep model bundles small enough for ordinary web application delivery.
- Detect WASM SIMD support and provide a scalar fallback.
- Avoid hidden heap growth in the hot path; callers should be able to provide
  workspaces or reuse allocated buffers.
- Emit deterministic traces that can be compared with native traces where target
  integer semantics match.
- Measure browser startup time, bundle size, route-to-first-output latency, and
  steady-state tokens per second.

The browser product promise is not "run a cloud LLM locally." It is "load small,
typed, deterministic experts instantly and compose them without server calls or
GPU drivers."

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

System-level budgets matter as much as kernel throughput. NSRL should measure:

- base-generator parameter and embedding bytes,
- prefill latency and time to first token,
- steady-state cached decode tokens per second,
- KV-cache and peak workspace bytes by context length,
- active expert parameter bytes,
- route-to-first-output latency,
- tokens per watt where measurable,
- native and WASM bundle size,
- browser startup time,
- deterministic trace overhead,
- cache and allocation behavior for active expert sets.

Initial implementation:

- Plain scalar Rust loops.
- Contiguous row-major tensors.
- Simple cache-friendly matmul layout.
- Branchless round-half-up.
- Static residual adds.
- No handwritten SIMD.
- Single dense generator execution.
- Deterministic traces before throughput shortcuts.

Later implementation:

- Blocked matrix multiplication.
- SIMD kernels.
- Packed weights.
- Incremental per-layer KV-cache kernels.
- Tied-embedding output projection optimization if its quality ablation passes.
- Threaded eval and deterministic parallel batch-gradient accumulation.
- Output-channel or row-block parallelism where workspaces can be partitioned.
- WASM SIMD with scalar fallback.
- Expert prefetch and active-set memory budgeting.
- Trace thinning for long training runs.

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

### Milestone 4: Tiny Model, Trace, Benchmark, And Native Training

Status: forward execution, deterministic trace output, `bench-1m`, corpus
tooling, and integer-native training are implemented as research
infrastructure. Promotion remains blocked on `integer-transformer-proof-v1`.

- End-to-end integer model execution.
- Deterministic output tests.
- Debug trace output.
- Small demo task.
- `bench-1m` 1,048,576-i8-weight forward benchmark.
- `nsrl-train` calibration path.
- Byte, MLP, attention, embedding, and mini-transformer training traces.
- i64 batch accumulators and rollback safety.
- Typed `nsrl-eval` proof contract and strict multi-baseline checker.

Completion requires one frozen candidate/result matrix in which the integer
transformer strictly beats retrieval, byte n-gram, and floating-point reference
probability error without increasing mistakes. Solomon and literary experiments
do not close this milestone independently.

### Milestone 5: Scalable Integer Training

Status: partially implemented; promotion work remains.

- Per-parameter or measured blockwise integer first and second moments.
- Integer update residuals with bounded state and resumable serialization.
- Dynamic update control from integer history where it improves on fixed shifts.
- Gradient-only `nsrl-train-core` APIs.
- Private per-worker accumulators and stable chunk-order reduction.
- Single-writer update, validation, and rollback.
- Replay tests proving identical serial and parallel results for fixed chunking.
- Distributed/sharded sample ordering with a reproducible global manifest if one
  host is no longer sufficient.

Completion is measured by both replay and the ability to train Milestone 6
profiles without saturation, zero-update, memory, or throughput collapse.

### Milestone 6: Quality-Development Backbone

Status: planned.

- Deterministic 8K-16K subword tokenizer with byte fallback.
- Frozen, deduplicated, contamination-checked language corpus manifest.
- Fully trained pre-norm stack including RMSNorm backward and final RMSNorm.
- Fixed-point rotary or relative positions with native/WASM fixtures.
- Profile-configurable depth, width, heads, MLP ratio, vocabulary, and context;
  quality experiments must not require source edits or silent artifact changes.
- 10M-30M dense profiles with at least three controlled scaling points.
- Same-shape float twins and a report of retained loss improvement.
- Frozen unassisted generation development panel.

Completion requires a healthy scaling trend, at least 90% retained held-out-loss
improvement relative to the float twin gap, and coherent non-repetitive
continuations over the frozen development panel. Passing this milestone does
not yet promote an open-generation product.

### Milestone 7: Incremental Open-Generation Candidate

Status: planned.

- Versioned `open-generation-v1` contract and hidden test partition.
- Bit-exact KV-cache/full-prefix conformance.
- Deterministic fixed-point greedy, top-k, temperature, and top-p decoding.
- 30M-100M candidate selected from measured development scaling.
- Native, unassisted generation artifacts for every prompt and fixed seed.
- Automatic degeneration, prompt-adherence, and context-use gates.
- Blinded human evaluation against the best smaller NSRL model.
- Float-twin non-inferiority evidence under the frozen human-evaluation rule.
- CPU memory, prefill, first-token, cached decode, and energy traces.

Completion requires all quality and numerical-health gates. Retrieval, memory,
corpus-prior, and router-assisted samples are product diagnostics and cannot
close this milestone.

### Milestone 8: Agentic Expert Packaging

Status: planned after the base generator clears Milestone 7.

- Expert artifact manifest.
- Capability tags and authority strings.
- Tokenizer and input/output schemas.
- Routing hints.
- Model, tensor, and LUT hashes.
- Native and WASM bundle budgets.
- Deterministic router trace schema.
- At least three locally routable experts.
- Domain-quality evidence over the promoted base generator.
- General-quality regression bound after routing or adapter activation.

### Milestone 9: `nsrl-graph`

Status: planned.

- Static integer graph IR.
- Generated readable Rust forward and backward code.
- Generated range checks, scale assertions, and saturation counters.
- Golden parity tests against hand-written kernels.
- Trace fields identifying graph and generator versions.

### Milestone 10: WASM Local-First Surface

Status: planned.

- `wasm32` build profile for `nsrl-core`.
- Browser demo that loads an expert bundle and emits a deterministic trace.
- WASM SIMD probe and scalar fallback.
- Bundle-size and startup-latency budgets.
- Route-to-first-output timing for small expert sets.
- Cross-target replay between native and browser builds where integer semantics
  are expected to match.

## Key Risks

- Static Q15 residual scale is too narrow for deeper models.
- Branchless round-half-up bias affects quality more than expected.
- Coarse activation scales destroy accuracy outside the residual trunk.
- Residual adds silently saturate.
- RMSNorm reciprocal square root has unacceptable error.
- Block-floating exponent adjustment has boundary bugs.
- Base-2 attention learns too slowly or needs different initialization.
- Strict i8 weight updates cannot preserve useful small gradients across the
  depth needed for open-ended generation.
- Byte tokenization makes the apparent context and training budget misleading;
  subword tokenization improves compression but makes embeddings dominate the
  local bundle.
- Fixed-point positional encoding loses phase precision or fails to extrapolate.
- Training data is too small, repetitive, contaminated, or domain-narrow to
  distinguish architecture failure from corpus failure.
- The model improves held-out loss but still collapses during long free-running
  generation.
- Decoding heuristics conceal native model weakness and leak into headline
  evaluation.
- KV-cache requantization drifts from full-prefix inference.
- Softmax reciprocal approximation collapses low-probability tokens.
- Lookup table approximations look correct locally but fail across layers.
- Calibration drifts away from runtime arithmetic.
- Tests accidentally validate against behavior that uses runtime floats.
- Premature SIMD optimization obscures numeric bugs.
- The dense quality backbone grows past the local memory/latency envelope before
  it becomes competent.
- Expert routing begins before the base generator is competent and produces
  impressive assisted demos without transferable language quality.
- Expert routing is brittle, opaque, or slower than direct inference.
- Adaptive optimizer state improves quality but makes training memory too large.
- Generated backward code hides math bugs behind an attractive abstraction.
- Parallel gradient accumulation breaks deterministic replay.
- WASM SIMD availability and browser security constraints reduce real-world
  throughput.
- Trace volume outgrows the model artifacts and slows iteration.

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
- Scaling strategy: pass the tiny substrate proof, scale one dense backbone to
  the minimum measured quality envelope, then add local expert routing.
- First proof model: two-layer `d_model=128`, `ff=256`, byte-level `NSRLMT5`
  under `integer-transformer-proof-v1`.
- Quality-development envelope: 10M-30M parameters with deterministic subword
  tokenization; open-generation candidate envelope: 30M-100M parameters.
- Tokenization: byte-level for the substrate and fallback; deterministic learned
  subwords for language-quality profiles.
- Position quality path: fixed-point rotary or relative positions; learned
  absolute and NOPE policies remain controls.
- Attention quality path: causal base-2 softmax remains the reference. Linear,
  streaming, or TTT attention requires controlled quality parity before
  promotion.
- Native generation claim: retrieval, memory, corpus priors, target lookup, and
  routing oracles are disabled unless the assistance mode is named explicitly.
- Generated code policy: allowed only as readable Rust with golden tests and
  trace-visible graph versions.
- Optimizer direction: adaptive integer optimizer state during training, no
  optimizer state in inference artifacts.
- Deployment wedge: local-first native CPU and WASM inference.
- Parallelism rule: parallel training must reduce gradients in deterministic
  stable order before a single update/validation step.
