# Product Requirements Document

## Product Name

NSRL: Numeric Stability Research Lab

## Summary

NSRL is a from-scratch, pure Rust neural network runtime and research platform
for building modern neural networks without floating-point operations in the
inference runtime.

The product is built around defensive integer ML: it does not merely replace
floating-point operations with integer operations. It actively protects signal
range, residual precision, rounding determinism, and normalization behavior
across every layer.

The first target is a small, inspectable native base-2 Transformer that proves
the numeric foundation: quantized QK attention, base-2 softmax, static residual
scales, integer RMSNorm, gated nonlinear blocks, and exhaustive tests for
fixed-point behavior.

## Current Status

The forward runtime target is implemented. `nsrl-core` now provides the
integer-native causal Transformer block primitives:

- fixed-scale integer tensor metadata,
- branchless round-half-up requantization,
- static Q15 residual additions with saturation counts,
- build-generated integer LUTs for reciprocal square root, base-2 fractional
  exponentiation, and reciprocal normalization,
- integer RMSNorm with leading-zero block-floating normalization,
- native base-2 causal attention with power-of-four head scaling,
- power-of-two Hard SiLU gated MLP,
- deterministic full-block forward traces.

`nsrl-demo` provides two proof points:

- `toy`: an inspectable `abba` full-block trace using
  `nsrl.forward_trace.v1`.
- `bench-1m`: a release-mode 4-block benchmark using
  `nsrl.benchmark_trace.v1`.

The latest captured `bench-1m` release row uses 1,048,576 i8 weights, reports
1,097,984 parameter bytes and 754,688 workspace bytes, completes the 4-block
128-token forward pass in 58,503 microseconds on the current development
machine, and records zero attention, MLP, residual, and final tensor
saturation events.

The remaining major product frontier is `nsrl-train`: learning NSRL-native
weights under the same base-2 attention, Q15 residual, integer RMSNorm, and
Hard SiLU arithmetic contracts used by inference.

## Problem

Modern neural networks usually depend on floating-point arithmetic for
stability, scale management, normalization, optimization, and activation
functions. Integer-only neural networks exist, but they are often hidden behind
compiler stacks, framework quantization paths, or hardware-specific kernels.

Naive fixed-point designs fail in a different way: they silently lose bits.
Repeated dynamic requantization, residual scale alignment, and precision-poor
normalization can shift learned features out of existence within a few layers.

This project aims to expose the full design from first principles:

- How tensor scales are represented without runtime floats.
- How residual streams preserve bit-width across depth.
- How attention behaves when softmax is defined natively in base-2 rather than
  approximating Euler's number.
- How integer arithmetic avoids overflow, underflow, and saturation collapse.
- How modern neural network blocks can be built on CPU with deterministic
  fixed-point math.
- How calibration, accuracy, reproducibility, and numerical error are measured.

## Goals

- Build a pure Rust, `no_std`-compatible CPU neural network runtime from
  scratch.
- Use no floating-point operations in the inference runtime.
- Use deterministic integer and fixed-point arithmetic.
- Preserve residual stream precision with static residual scales.
- Treat attention as a first-class runtime primitive, not a deferred layer.
- Define attention logits in native log2-temperature space.
- Make every scale, rounding rule, clamp, and approximation explicit.
- Support modern neural network components such as residual connections,
  RMSNorm, gated MLPs, and native base-2 attention.
- Provide a Rust calibration and training crate that mirrors runtime integer
  behavior.
- Provide strong tests for numeric correctness, stability, and performance
  assumptions.
- Keep the codebase small enough to audit and reason about.

## Non-Goals

- Competing with PyTorch, TensorFlow, ONNX Runtime, or vendor inference engines.
- Supporting GPUs or accelerators in the first phase.
- Training large models.
- Implementing every neural network layer type.
- Running standard Llama, GPT, or HuggingFace checkpoints through post-training
  quantization. NSRL models must be born into NSRL's base-2 attention contract.
- Hiding quantization behavior behind opaque compiler passes.
- Optimizing before the numeric contract is proven.

## Users

Primary users:

- Engineers who want to understand integer neural networks from first
  principles.
- Researchers experimenting with fixed-point neural network stability.
- Systems programmers interested in deterministic CPU inference.

Secondary users:

- Students learning quantization and low-level model execution.
- Practitioners who need reproducible inference on constrained hardware.

## Core Product Principles

- **Numerical clarity:** every operation has a documented integer range,
  rounding mode, saturation behavior, and scale transition.
- **Bit-width preservation:** residual streams use fixed arithmetic scales so
  depth does not destroy learned signal through repeated right shifts.
- **Determinism:** the same input, model, and CPU target produce the same output.
- **Auditability:** implementation choices are visible in ordinary Rust source.
- **Attention-first:** the runtime is designed around native base-2 attention
  from the beginning.
- **CPU sympathy:** use arithmetic patterns that map naturally to scalar and
  SIMD CPU execution.

## Runtime Contract

The inference runtime must not execute floating-point instructions for model
math. All model operations use integer arithmetic, fixed-point arithmetic,
integer lookup tables, branchless rounding, saturating clamps, and deterministic
integer approximations.

Attention uses native base-2 softmax. The runtime does not multiply logits by
`log2(e)` to imitate standard Euler softmax. Training must learn directly inside
this log2-temperature space.

Runtime RMSNorm may use integer block-floating normalization: a significand
derived by shifting integer magnitudes plus an integer exponent derived from
leading-zero counts. This is allowed because it uses integer CPU operations, not
IEEE floating-point arithmetic.

Build-time tools may use floats only when explicitly marked as offline tooling,
for example to generate lookup tables or compare against reference models. Any
artifact generated by such tooling must be serialized as integer constants.

## Product Components

The project has two first-class components that must evolve together:

- `nsrl-core`: pure Rust, `no_std`-compatible integer inference runtime.
- `nsrl-train`: Rust calibration and training support that mirrors runtime
  rounding, residual scales, and block-floating normalization.

The training and calibration crate exists because generic post-training
quantization is not expected to produce reliable weights for this runtime.
After the base-2 attention decision, `nsrl-train` is no longer optional
infrastructure; it is the proof system that creates models compatible with
NSRL's mathematics.

## Initial Model Target

The first advanced model target is a tiny native base-2 Transformer character
predictor:

```text
vocab:       character or byte-level
d_model:     128
layers:      4
heads:       2
d_k:         64
sequence:    128 tokens initially
parameters:  about 1.1M
```

This is intentionally smaller than a TinyStories-scale language model. It is
large enough to test whether base-2 attention heads learn useful structure, but
small enough that numeric traces and training failures remain inspectable.

## Later Model Target

After the first character model learns, the project may scale toward a compact
wordpiece or TinyStories-style model:

```text
input
  -> integer RMSNorm
  -> QKV projections
  -> QK dot with power-of-four head-size shift
  -> native base-2 integer softmax
  -> attention output projection to static residual scale
  -> raw residual add
  -> integer RMSNorm
  -> gated MLP projection to static residual scale
  -> raw residual add
```

Attention is no longer deferred. It is part of the numeric core.

## Functional Requirements

- Represent tensors with integer data and explicit integer scale metadata.
- Support signed quantized tensor formats for activations and weights.
- Support symmetric quantization in the first runtime version.
- Support static residual stream scales, initially Q15 on `i16`.
- Support saturating arithmetic.
- Support deterministic branchless round-half-up requantization.
- Support fixed-point matrix multiplication with integer accumulation.
- Support attention head dimensions that are powers of four so `sqrt(d_k)` is
  an exact arithmetic right shift.
- Support native base-2 integer softmax with fractional LUTs and integer shifts.
- Support mask annihilation with `i32::MIN`.
- Support reciprocal normalization through the same block-floating strategy used
  by RMSNorm.
- Support residual addition without dynamic scale alignment inside the residual
  trunk.
- Support RMSNorm using integer reciprocal square root with leading-zero
  normalization.
- Support lookup-table or piecewise integer activations.
- Provide a test suite for arithmetic primitives and layer-level behavior.
- Provide simple model serialization for integer weights, scales, residual
  contract metadata, and lookup table versions.
- Provide a calibration engine that mirrors runtime math exactly.

## Quality Requirements

- Numeric behavior must be reproducible across supported CPU targets.
- Overflow must be prevented by range analysis or caught by debug assertions.
- Approximation functions must have measured error bounds.
- Test cases must include edge values, saturation cases, scale transitions, and
  signed rounding cases.
- Residual stream additions must report saturation rates.
- Runtime lookup tables must be generated, versioned, and tested as integer
  artifacts.
- Arithmetic primitives should use property testing, including `proptest`, to
  fuzz edge cases across broad integer domains.
- Documentation must distinguish runtime guarantees from offline tooling.

## Success Metrics

Milestone 1: Numeric Core is complete when:

- A fixed-point tensor type exists.
- Core arithmetic primitives are tested.
- `RSQRT_LUT_8BIT` is generated by `build.rs` and compiled as static integer
  data.
- Base-2 fractional exponent and reciprocal LUTs are generated by `build.rs`
  and compiled as static integer data.
- Attention dot scaling and base-2 softmax are implemented and tested.
- Branchless round-half-up requantization is implemented and tested.
- Static Q15 residual stream rules are implemented and tested.
- Linear, RMSNorm, residual add, and gated MLP layers run without runtime floats.
- A tiny model can execute end-to-end with deterministic integer outputs.
- Numeric traces can explain the scale and range at each layer.

Status: complete for the implemented forward runtime. Evidence lives in
`nsrl.forward_trace.v1`, `nsrl.benchmark_trace.v1`, and the `nsrl-demo` replay
test.

Milestone 2: Native Training is successful when:

- `nsrl-train` can train a tiny base-2 attention character predictor from
  scratch.
- The tiny model runs on a small character or byte-level sequence task.
- Accuracy degradation from an equivalent reference model is measured.
- Saturation and underflow rates are tracked per layer.
- The implementation remains simple enough to audit.

Status: not started. The forward runtime now gives `nsrl-train` a stable target.

Milestone 3: Transformer Block is complete when:

- A tiny native base-2 Transformer block runs end-to-end.
- Attention collapse, mask annihilation, and reciprocal normalization cases are
  tested.

Status: complete for inference. The implemented block is:

```text
embedding
  -> RMSNorm -> native base-2 causal attention -> static Q15 residual
  -> RMSNorm -> power-of-two Hard SiLU gated MLP -> static Q15 residual
  -> integer output head
```

## Locked Decisions

- Implementation language: Rust.
- Quantization style: symmetric first.
- Activation type: `qint16`.
- Weight type: `qint8`.
- Accumulator type: `int32`, with `int64` for large reductions and checked
  intermediate products.
- Residual trunk: static Q15 `i16`.
- Runtime rounding: branchless round-half-up.
- Attention: native base-2 softmax, not approximated Euler softmax.
- Head dimensions: powers of four.
- RMSNorm magnitude handling: integer block-floating normalization via
  leading-zero counts.
- Training path: bespoke Rust training support; generic framework PTQ is not a
  compatibility target.

## Open Questions

- How strict should the "no floats" rule be for tests and offline calibration?
- How small can the first character corpus be while still proving base-2
  attention learns useful grammar?
- How much of `nsrl-train` is required before the first end-to-end model demo?
- What LUT size gives the best early tradeoff for RMSNorm accuracy and cache
  behavior?
