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
possible. It is turning the proof stack into a viable local AI substrate:
adaptive integer optimizers, deterministic parallel gradient accumulation,
expert manifests and routing, generated checked backward code, and first-class
WASM/browser deployment.

The proof evaluator exposes `combined`, `transformer-only`, and
`suffix-memory-only` modes. Any future headline model gate must run the matched
component matrix and keep assisted memory out of the unassisted candidate row.
The v1 combined artifact remains replayable historical evidence, not a clean
transformer-only learning result. The first suffix-free successor sweep is also
frozen: none of 16 bounded variants passed, so the next attempt must change the
learning architecture or objective rather than only retune the existing
calibrated trainer.

Successor-v2 now has an end-to-end sealed replay rather than a row-only parser.
Its candidate-specific manifest binds the 5,896-target dataset, identity-byte
tokenizer, physically stripped model, evaluator source set, runner, exact
matrix/evidence bytes, and per-system replay hashes. The repository check
re-trains the float32 causal transformer and re-evaluates transformer-only,
uniform, retrieval, byte n-gram, and float-transformer logits through one
canonical integer NLL function. Suffix-memory, retrieval, and routing-oracle
ablations are replay-invariant. The candidate scores 115,010,055 millibits and
loses to all four baselines, so the engineering result is a frozen
falsification and not scaling authorization.

Prior work already establishes integer and quantized training as a field. The
engineering contribution pursued here is the auditable combination: no float
master weights, exact integer replay, explicit saturation and residual-carry
telemetry, and shared training/deployment arithmetic. Broad literature claims
must be checked against `research/paper-catalog.md`; the proposed relationship
between rank and reachable integer updates remains experimental. The first
rank/shift/carry matrix finds both equivalence and distinctness. Its matched
longitudinal extension shows that early reachable movement is a high-precision
predictor of later disjoint held-out gain, but also has delayed-activation false
negatives and saturates in every early-reachable long run. Exact fingerprints
therefore prioritize scale-up candidates; they do not replace saturation gates
or longer activation deadlines.

Production training has crossed the multi-group boundary and the local
scaling-readiness gate. The frozen p10m K+V schedule uses shifts 26 and 30,
respectively, over 2,048 windows. Both groups move by window 256 and in every
second-half chunk; only K, V, and output cross integer update boundaries, all
13 gradient paths remain active, and every saturation counter stays zero. The
integer lane finishes 5,209 total millibits below initialization and replays
windows 1,025–2,048 byte-for-byte. A float32 SGD reference matched on
initialization, data, ordering, context, batch, budget, and held-out evaluation
moves all 13 arrays, improves by 98 mean millibits, and replays its second half
tensor-for-tensor. Optimizer families are not matched. The residual boundary
policy selected the gated-MLP `gate` projection at shift 23 for a fresh
isolated preflight. That preflight is complete: `gate` first crossed at window
768 and accumulated 26 exact updates over 2,048 windows. Only K, V, `gate`, and
output moved; all 13 gradient paths stayed active, saturation remained zero,
held-out ended 5,209 total millibits below initialization, and replay from
window 1,024 reproduced both model and optimizer byte-for-byte. The following
source-relative `up` gate found 26 safe shift-23 updates but no dev gain. A
shift-22 density probe then made 101,543 exact `up` updates with the same zero-
saturation health, yet its selected checkpoint only tied dev and regressed the
one-shot test by 1,245 total millibits. A matched 1,024-window comparison found
the shift-22 and shift-23 models produce identical final features, logits,
probabilities, and losses on all 256 dev windows. The immediate engineering
bottleneck is therefore forward-path quantization masking, not update
reachability. A predeclared sensitivity sweep finds forward shift 7 is the
first safe functional boundary: 250 feature/logit vectors and 124 probability
vectors change, but target probabilities change on 0 of 256 windows. Fresh
1,024-window training at forward shift 7 remains zero-saturation and exactly
replayable, makes 50,568 `up` updates, and still ties source dev. The next
isolated action was target-probability resolution measurement. That audit is
complete across Q15/Q19/Q23/Q27/Q31 using identical integer logits. Q15
requantizes exactly to the production path but compresses source targets to
three values and hides all target deltas; Q19 reveals one changed target and
Q23 reveals the full 13-window support also visible at Q31. A compensated Q19
and Q23 gradient preflight preserves effective output/backward learning scales,
all 13 gradient paths, and zero saturation. Both wide lanes nevertheless end
at the exact Q15 model bytes and dev loss after 256 windows, while their
optimizer states differ. The precision signal is therefore residual-only at
this horizon. The completed normalization audit then isolates the reciprocal:
legacy normalization reaches 98,925/98,929 ppm worst-case Q23 mass error,
retained-Q47 LUT reaches 6,354/6,349 ppm, one integer Newton step reaches 98/83
ppm, and exact integer division reaches 73/74 ppm. Newton is nearly at the exact
accuracy ceiling, but legacy/Newton/exact target-change coverage is 13/5/4
windows. The conservative contract selects no default until those nine excess
legacy changes are attributed. The completed attribution finds all four exact
windows in the Newton set, no exact misses, and one Newton-only window caused by
a denominator change at a Q23 rounding boundary. Across both 2,097,152-value
probability surfaces, every Newton-versus-exact difference is at most one Q23
unit. All nine legacy-only windows have unchanged target logits, changed target
weights and denominators, and zero exact Q23 delta. Engineering can therefore
use `q47_newton1` in training. The bounded normalized wide-gradient preflight
is complete: its Q23/Newton control replays model and optimizer byte-for-byte
and retains the recovered signal in optimizer state. The first safe `up`
materialization boundary, shift 21, produces 155 `up` updates and changes 84
feature/logit windows plus 29 probability windows without saturation, but no
target probability or dev loss. The first output materialization boundary,
effective shift 41, changes three target-probability windows but regresses dev
by 415 total millibits. Integer precision now reaches the target loss boundary;
optimization direction, not missing fractional bits, is the next isolated
problem. Paid scaling remains outside these local contracts.

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

NSRL scales outward before it scales upward. The default deployment target is a
swarm of small expert models connected by a deterministic router:

```text
request
  -> router
  -> one or more expert manifests
  -> expert inference
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

The first proof training target is a roughly 1.1M-parameter character predictor:

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
implemented for the small research lanes. The next engineering shape is:

- expert-scale training rather than monolithic frontier-scale training,
- deterministic parallel batch-gradient accumulation,
- adaptive integer optimizer state instead of hand-tuned static shifts,
- generated checked backward code for new architectures,
- and trace-thinning so training evidence does not dominate model size.

The remaining research questions are:

- whether the same arithmetic scales cleanly to larger transformer experts,
- how to reduce dependence on source-grounded composition without losing
  coherence,
- and whether linear attention plus integer test-time state updates can close
  enough of the softmax quality gap to justify the O(d^2) streaming interface.

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
- Single-expert execution.
- Deterministic traces before throughput shortcuts.

Later implementation:

- Blocked matrix multiplication.
- SIMD kernels.
- Packed weights.
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

### Milestone 5: Agentic Expert Packaging

Status: planned.

- Expert artifact manifest.
- Capability tags and authority strings.
- Tokenizer and input/output schemas.
- Routing hints.
- Model, tensor, and LUT hashes.
- Native and WASM bundle budgets.
- Deterministic router trace schema.
- At least three locally routable experts.

### Milestone 6: Adaptive Integer Optimizer

Status: planned.

- Blockwise integer momentum.
- Blockwise variance or magnitude tracker.
- Dynamic update shifts from integer history.
- Stable deterministic reduction order.
- Training-only optimizer state.
- Trace summaries for optimizer movement, saturation, and zero-delta rates.

### Milestone 7: Deterministic Parallel Training

Status: planned.

- Gradient-only `nsrl-train-core` step APIs.
- Private per-worker accumulators.
- Stable chunk-order accumulator reduction.
- Single-writer update application.
- Batch validation and rollback after parallel accumulation.
- Replay tests proving identical serial and parallel traces for fixed chunking.

### Milestone 8: `nsrl-graph`

Status: planned.

- Static integer graph IR.
- Generated readable Rust forward and backward code.
- Generated range checks, scale assertions, and saturation counters.
- Golden parity tests against hand-written kernels.
- Trace fields identifying graph and generator versions.

### Milestone 9: WASM Local-First Surface

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
- Softmax reciprocal approximation collapses low-probability tokens.
- Lookup table approximations look correct locally but fail across layers.
- Calibration drifts away from runtime arithmetic.
- Tests accidentally validate against behavior that uses runtime floats.
- Premature SIMD optimization obscures numeric bugs.
- A monolithic-model roadmap recreates the memory wall instead of exploiting the
  local expert niche.
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
- Scaling strategy: local expert swarms and deterministic routing before
  monolithic model scale.
- First proof model scale: about 1.1M parameters, character or byte-level
  prediction.
- Product expert envelope: initially 1M-10M parameters, with larger experts only
  when active memory and latency budgets justify them.
- Generated code policy: allowed only as readable Rust with golden tests and
  trace-visible graph versions.
- Optimizer direction: adaptive integer optimizer state during training, no
  optimizer state in inference artifacts.
- Deployment wedge: local-first native CPU and WASM inference.
- Parallelism rule: parallel training must reduce gradients in deterministic
  stable order before a single update/validation step.
