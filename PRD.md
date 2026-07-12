# Product Requirements Document

## Product Name

NSRL: Numeric Stability Research Lab

## Summary

NSRL is a local-first integer AI substrate for deterministic micro-model agents.
It is built from scratch in Rust for CPU and WASM environments where predictable
latency, small bundles, auditability, and zero driver dependence matter more than
frontier-scale parameter count.

The product is built around defensive integer ML: it does not merely replace
floating-point operations with integer operations. It actively protects signal
range, residual precision, rounding determinism, optimizer scale, and
normalization behavior across every layer.

The strategic target is the agentic edge, but routing is not a substitute for a
competent generator. NSRL must first train one dense NSRL-born language backbone
that produces coherent, diverse, prompt-responsive text without retrieval,
memorized continuations, corpus priors, or expert-oracle assistance. Only then
should small domain experts and a deterministic Rust router extend that base.
NSRL competes on quality per active byte, tokens per watt, cold-start latency,
and reproducible traces rather than frontier-scale parameter count.

The first proof target remains a small, inspectable native base-2 Transformer
that validates the numeric foundation: quantized QK attention, base-2 softmax,
static residual scales, integer RMSNorm, gated nonlinear blocks, integer
training, and exhaustive tests for fixed-point behavior.

## Active Promotion Milestone

The only headline promotion milestone is `integer-transformer-proof-v1`,
defined in `docs/integer-transformer-proof-v1.md` and enforced by `nsrl-eval`.
One NSRL-born integer transformer must beat retrieval, byte n-gram, and an
independently produced floating-point reference on the same frozen evaluation
partition. It must win aggregate Q15 probability error strictly and cannot
regress mistake count against any required baseline.

Literary routing and Solomon multimodal generation are experiment suites. They
may produce candidate architectures and product evidence, but they do not
replace or redefine the substrate promotion milestone.

Passing the substrate milestone is necessary but not sufficient for a language
product. It proves that integer training learns a frozen next-token task; it
does not establish free-running coherence, instruction following, long-context
use, or open-ended generation quality. Those claims require the separate
quality gates defined below.

## Vision

NSRL should not become a smaller imitation of a frontier cloud LLM. It should
become the reference stack for local, deterministic, inspectable generation:

- a useful dense base generator before specialization,
- micro-model experts packaged with explicit capability manifests,
- symbolic or learned routers that compose experts without leaving the local
  process,
- training-time adaptive integer optimizers so deeper models do not depend on
  hand-tuned global shifts,
- generated, reviewable integer backward passes from a small graph IR, and
- first-class WASM/browser deployment with strict bundle-size and startup
  budgets.

The core bet is that integer-only local agents can own niches where large
floating-point stacks are physically and operationally awkward: private browser
apps, offline-first software, embedded tools, local game logic, deterministic
automation, and inexpensive client-side inference.

"High quality" in this product means high quality for the declared local model
envelope, not parity with frontier cloud systems. A promoted generator must be
coherent over multi-paragraph continuations, respond to varied unseen prompts,
avoid obvious repetition collapse, use its supported context, and compare
credibly with a floating-point architectural twin of the same size and data.

## Current Status

The forward runtime target is implemented, and the project has moved into
integer-native training and traceable generation experiments. `nsrl-core` now
provides the integer-native causal Transformer block primitives:

- fixed-scale integer tensor metadata,
- branchless round-half-up requantization,
- static Q15 residual additions with saturation counts,
- build-generated integer LUTs for reciprocal square root, base-2 fractional
  exponentiation, and reciprocal normalization,
- integer RMSNorm with leading-zero block-floating normalization,
- native base-2 causal attention with power-of-four head scaling,
- full, incremental, and training-oriented linear attention paths,
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

`nsrl-train` is now active infrastructure rather than a future phase. It can
train byte, output-head, MLP, attention, and embedding paths with i64 batch
gradient accumulation, checked rollbacks, deterministic trace rows, and integer
learning-rate shifts. The active language/vision evidence is the Solomon
multimodal pipeline (`nsrl-solomon-attention`, `nsrl-solomon-multimodal`,
`nsrl-solomon-latent-train`): deterministic joint text/image-token training and
sampling with checked integer traces at every step.

The current gap is generation quality. The checked-in attention artifacts are
smoke-scale, the frozen substrate proof has no passing candidate, and coherent
Solomon text still depends on prompted or memory-assisted paths. The next
product frontier is therefore a quality-capable dense generator: deterministic
subword tokenization, fully trained normalization and position handling,
incremental decoding, stronger data, scalable integer optimization, and an
unassisted held-out generation suite. Expert packaging and routing follow that
gate rather than preceding it.

## Problem

Modern neural networks usually depend on floating-point arithmetic for
stability, scale management, normalization, optimization, and activation
functions. Integer-only neural networks exist, but they are often hidden behind
compiler stacks, framework quantization paths, or hardware-specific kernels.

Large floating-point systems also assume a deployment shape that is wrong for
many products: cloud calls, GPU drivers, heavyweight browser runtimes, high
memory bandwidth, and monolithic models that must move too many bytes before
they can do useful local work.

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
- Produce one NSRL-born dense language model that supports useful unassisted
  open-ended generation within a measured local memory and latency envelope.
- Make WASM/browser deployment a first-class runtime target.
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
- Provide adaptive integer optimizer support so training does not depend on
  static global learning-rate shifts.
- Provide an inspectable graph/code-generation path for forward definitions and
  integer backward passes.
- Package models as small experts with explicit capabilities, schemas, routing
  hints, tokenizer contracts, and trace authority.
- Support deterministic composition of multiple local experts through a Rust
  router.
- Make byte tokenization the universal fallback while supporting a
  deterministic learned subword tokenizer for language-quality profiles.
- Support incremental causal generation with a KV cache, bounded sampling, and
  fixed-point positional encoding that can extend beyond tiny fixed windows.
- Evaluate free-running generation separately from retrieval, prompted replay,
  and memory-assisted composition.
- Provide strong tests for numeric correctness, stability, and performance
  assumptions.
- Keep the codebase small enough to audit and reason about.

## Non-Goals

- Competing with PyTorch, TensorFlow, ONNX Runtime, or vendor inference engines.
- Supporting GPUs or accelerators in the first phase.
- Training frontier-scale or cloud-replacement models. Dense local backbones up
  to the evidence-backed quality envelope are in scope.
- Implementing every neural network layer type.
- Running standard Llama, GPT, or HuggingFace checkpoints through post-training
  quantization. NSRL models must be born into NSRL's base-2 attention contract.
- Hiding quantization behavior behind opaque compiler passes. Generated code is
  allowed only when the emitted Rust is deterministic, reviewable, testable, and
  bound to the same trace contracts as hand-written kernels.
- Optimizing before the numeric contract is proven.
- Reporting retrieval-assisted, corpus-prior-assisted, copied, or
  memory-assisted output as native open-ended generation quality.

## Users

Primary users:

- Engineers who want to understand integer neural networks from first
  principles.
- Researchers experimenting with fixed-point neural network stability.
- Systems programmers interested in deterministic CPU and WASM inference.
- Product engineers building private local-first or offline-first AI features.

Secondary users:

- Students learning quantization and low-level model execution.
- Practitioners who need reproducible inference on constrained hardware.
- Browser, mobile, and game developers who need small local models without GPU
  drivers or cloud calls.

## Core Product Principles

- **Numerical clarity:** every operation has a documented integer range,
  rounding mode, saturation behavior, and scale transition.
- **Bit-width preservation:** residual streams use fixed arithmetic scales so
  depth does not destroy learned signal through repeated right shifts.
- **Determinism:** the same input, model, and CPU target produce the same output.
- **Auditability:** implementation choices are visible in ordinary Rust source.
- **Inspectable generation:** code generated from graph definitions must be
  readable Rust with stable tests and explicit range checks.
- **Agentic edge:** small expert models and deterministic routing are the default
  specialization strategy after a capable base generator exists.
- **Quality before composition:** routing may improve specialization and active
  memory, but it cannot close a weak base model's grammar, coherence, or
  decoding gaps.
- **Unassisted evidence:** native generation claims are based on held-out prompts
  with retrieval, memory, corpus priors, and target lookup disabled.
- **Local-first portability:** CPU and WASM targets are product surfaces, not
  afterthoughts.
- **Adaptive training:** integer optimizer state should replace manual
  shift-sweep babysitting as models deepen.
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

The project has current first-class components that must evolve together:

- `nsrl-core`: pure Rust, `no_std`-compatible integer inference runtime.
- `nsrl-corpus`: deterministic corpus and byte-token builders.
- `nsrl-train-core`: `no_std` borrowed-workspace training-step extraction.
- `nsrl-train`: Rust calibration and training support that mirrors runtime
  rounding, residual scales, and block-floating normalization.
- `nsrl-demo`: executable evidence surface for forward traces and benchmarks.

The training and calibration crate exists because generic post-training
quantization is not expected to produce reliable weights for this runtime.
After the base-2 attention decision, `nsrl-train` is no longer optional
infrastructure; it is the proof system that creates models compatible with
NSRL's mathematics.

Planned components:

- `nsrl-router`: deterministic expert selection, context passing, and traceable
  agent orchestration.
- `nsrl-graph`: a small integer graph IR plus code generator for checked
  `no_std` forward and backward Rust.
- `nsrl-wasm`: browser packaging, WASM SIMD validation, bundle budgets, and
  local-first demo surfaces.

## Initial Proof Model Target

The active substrate target is the checked-in `NSRLMT5` profile used by
`integer-transformer-proof-v1`:

```text
vocab:       256 bytes
d_model:     128
layers:      2
heads:       8 promoted / 2 control
d_k:         16 promoted / 64 control
hidden:      256
sequence:    64 bytes for the frozen proof
```

This is a substrate probe, not the open-ended model target. It is intentionally
small enough that numeric traces and training failures remain inspectable.

## Strategic Model Target

After the substrate proof passes, the project must scale upward to the smallest
dense backbone that clears free-running quality gates. It may then scale outward
through expert routing. The sequence is deliberate: substrate proof, dense
quality, efficient decoding, then specialization.

Target envelopes:

```text
substrate probe:       <5M parameters; byte tokens; numeric proof only
quality development:  10M-30M parameters; 8K-16K subword vocabulary
open-gen candidate:   30M-100M parameters; measured by quality/byte, not size
domain adapter/expert: 1M-20M incremental parameters over a competent trunk
active route:          base trunk plus 0-2 experts per request
deployment:            native CPU first; WASM where bundle budgets permit
```

The first quality-development profile should use 8-12 layers, `d_model` 256 or
384, a gated MLP ratio near 3-4x, fully trained RMSNorm, fixed-point relative or
rotary position handling, tied input/output embeddings where measurement
supports it, causal base-2 attention, and a reusable KV cache. The first
open-generation candidate should scale only after that profile shows a clean
validation-loss scaling trend and healthy integer numerics.

The 30M-100M envelope is a hypothesis, not a promise. If the largest healthy
candidate still fails the frozen open-generation gate, the project must choose
explicitly between expanding the local memory envelope and narrowing the
product claim. It must not route around the failure or promote assisted demos as
native quality.

Each expert may still use the same native base-2 Transformer-style block:

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

Attention is no longer deferred. It is part of the numeric core. Neither is the
dense trunk optional: a swarm of weak generators does not become a strong
generator through routing. Expert packaging begins after the base trunk clears
the unassisted generation gate.

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
- Provide expert manifests with capability tags, tokenizer IDs, input/output
  schemas, routing hints, authority strings, model hashes, and bundle budgets.
- Provide deterministic router traces that explain why an expert was selected.
- Provide blockwise adaptive integer optimizer state for training-time momentum
  and variance/magnitude tracking.
- Provide a graph-generated backward path whose emitted Rust can be reviewed,
  tested, and compared against scalar golden references.
- Provide deterministic parallel batch-gradient accumulation with stable reduction
  order before applying integer updates.
- Provide WASM conformance tests for no-runtime-float inference, startup latency,
  bundle size, and SIMD/no-SIMD fallback behavior.
- Provide a versioned deterministic subword tokenizer and retain a byte fallback
  for arbitrary input and tokenizer recovery.
- Provide fixed-point relative or rotary position handling with an explicit
  extrapolation contract; learned absolute positions remain a diagnostic
  baseline, not the quality default.
- Provide an incremental causal path with a reusable KV cache that is bit-exact
  with full-prefix inference for the same context.
- Provide bounded greedy, top-k, and top-p/temperature sampling using fixed-point
  probabilities and deterministic seeds.
- Support tied input/output embeddings as an ablation and measure its
  quality-per-byte effect; do not assume tying is free while embeddings and the
  output matrix use different integer representations.
- Provide corpus manifests with source, license, split, deduplication,
  contamination, tokenizer, and token-count provenance.
- Provide an unassisted open-generation evaluation that disables retrieval,
  memory, corpus priors, target lookup, and prompt-specific routing oracles.

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
- Every promoted model must report held-out bits per byte or token loss,
  repetition and degeneration metrics, context-use probes, prompt-following
  scores, and blinded human preference against named baselines.
- Every quality result must identify whether generation is native, routed,
  retrieval-assisted, memory-assisted, or teacher-scored; those categories may
  not be combined into one headline number.
- A same-shape floating-point architectural twin must be trained on the same
  tokenizer, data split, token budget, and objective to quantify the cost of the
  integer contract. It must preserve the same base-2 attention temperature,
  activation equations, and model shape so the comparison isolates arithmetic
  and optimization rather than changing the architecture.
- Quality promotion requires numerically healthy runs: bounded saturation,
  bounded zero-update rates, no rejected batches, and movement in every trained
  attention and MLP projection.

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

Status: implemented as a research lane. `nsrl-train` now supports deterministic
integer updates with i64 batch accumulators, rollback safety, mini-transformer
training, MLP backward, softmax and linear attention backward paths, and traced
generation. The remaining work is quality, scale, and simplification, not proof
that integer updates can move weights.

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

Milestone 4: Scalable Integer Training is successful when:

- Training supports per-parameter or measured blockwise integer momentum and
  variance/magnitude tracking.
- Integer update residuals preserve sub-i8 update signal without becoming float
  or inference master weights.
- Static global shift sweeps are no longer required for the main quality lane.
- Gradient-only worker steps and stable-order accumulation produce replayable
  parallel updates.
- Optimizer state is resumable, training-only, and excluded from inference
  artifacts.

Milestone 5: Quality-Capable Language Backbone is successful when:

- The frozen substrate proof passes before larger quality claims are promoted.
- A deterministic subword corpus and held-out split are frozen with provenance,
  deduplication, and contamination checks.
- A 10M-30M development profile demonstrates improving held-out loss across at
  least three increasing model or token budgets without numerical-health
  regression.
- The integer model closes at least 90% of the held-out loss improvement between
  the byte n-gram baseline and its same-shape floating-point twin. This is an
  interim quality-retention gate, not permission to call the model high quality.
- Full-prefix and cached incremental logits are bit-exact on conformance cases.
- Unassisted samples remain coherent and non-repetitive over at least 512
  generated subword tokens on a frozen, diverse prompt panel.

Milestone 6: Open-Ended Generation is successful when:

- One 30M-100M candidate produces native, unassisted continuations for unseen
  prompts with no retrieval, corpus prior, memory injection, or target lookup.
- On the frozen generation panel, it clears all automatic degeneration,
  prompt-adherence, and context-use floors, beats the best smaller NSRL model,
  and is non-inferior to its same-shape floating-point twin under a predeclared
  blinded-human-evaluation margin.
- It retains at least 90% of the same-shape floating-point twin's held-out-loss
  improvement over the required statistical baseline.
- Results include greedy replay plus multiple fixed sampling seeds; a single
  curated sample cannot promote the model.
- Native CPU artifacts report model bytes, peak working memory, time to first
  token, steady-state tokens per second, and energy where measurable.

The precise prompt panel and numeric floors must be frozen in a versioned
`open-generation-v1` contract before a candidate is trained against them.

Milestone 7: Agentic Edge is successful when:

- Models can be packaged as expert artifacts with manifests, tokenizer contracts,
  schemas, capabilities, hashes, and trace authority.
- A deterministic Rust router can select one or more experts for a request.
- Router traces explain expert selection and context handoff.
- At least three experts can be composed locally without cloud calls or GPU
  drivers.
- Routed generation improves a named domain metric over the already promoted
  base generator without regressing general held-out quality beyond the
  contract's tolerance.

Milestone 8: Graph And WASM Product Surface is successful when:

- `nsrl-graph` can generate checked forward/backward Rust for the core block
  primitives and match hand-written golden references.
- A browser demo loads a small expert bundle, routes locally, and emits a
  deterministic trace.
- WASM bundle size, startup latency, and route-to-first-output latency are
  measured in CI or release traces.

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
- Scaling strategy: expert swarms and deterministic routing, not monolithic
  frontier-scale models. A competent 30M-100M dense backbone is explicitly in
  scope and precedes expert routing.
- Generated code policy: allowed only when emitted Rust is inspectable,
  deterministic, and trace-compatible.
- Deployment wedge: local-first CPU and WASM inference.
- Optimizer direction: adaptive integer optimizer state during training; no
  optimizer state in inference artifacts.
- Tokenization: byte tokens remain the proof and fallback contract; deterministic
  learned subwords are the language-quality default.
- Quality claim: open-ended generation is native and unassisted unless its
  assistance mode is explicitly named.
- Attention quality path: causal base-2 softmax is the reference. Linear or TTT
  attention may replace it only after same-checkpoint or controlled-retrain
  quality parity is measured.

## Open Questions

- What is the smallest expert size that feels useful for grammar, routing, math,
  domain style, and API/tool selection?
- What is the smallest dense backbone that clears the frozen open-generation
  gate, and which failure is limiting below it: data, tokenizer, optimization,
  depth, context, or integer precision?
- Does strict i8-from-initialization training retain enough quality at 30M-100M
  parameters, or does NSRL need a versioned integer-only training contract with
  higher-precision integer update residuals while preserving i8 inference?
- Which fixed-point position scheme gives the best length extrapolation without
  sacrificing native/WASM determinism?
- What data and token budget produces a stable scaling trend before expensive
  open-generation runs begin?
- How should expert manifests describe capability, authority, and failure modes
  without becoming a second opaque framework?
- Should the first router be purely symbolic, learned, or hybrid?
- What block size gives the best adaptive integer optimizer tradeoff between
  quality, memory, and deterministic trace size?
- How much of `nsrl-train` should be replaced by `nsrl-graph` generated backward
  code, and which hand-written kernels remain the golden references?
- What WASM bundle size and startup latency targets are strict enough to make
  NSRL meaningfully better than browser ONNX/PyTorch-style deployments?
