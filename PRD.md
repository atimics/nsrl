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

The strategic target is the agentic edge: many small NSRL-born expert models,
each narrow enough to keep its active working set cache-friendly and cheap to
ship, connected by a deterministic Rust router. NSRL competes on
tokens-per-watt, cold-start latency, reproducible traces, and sub-100ms local
routing rather than monolithic cloud-model scale.

The first proof target remains a small, inspectable native base-2 Transformer
that validates the numeric foundation: quantized QK attention, base-2 softmax,
static residual scales, integer RMSNorm, gated nonlinear blocks, integer
training, and exhaustive tests for fixed-point behavior.

Integer training itself is not the novelty claim. Earlier work already covers
integer or heavily quantized training and integer transformer components. NSRL
is testing the stricter conjunction of native-integer updates, deterministic
replay, checked health, and local Rust/WASM deployment. Its further claim that
rank can exceed the number of optimizer updates reachable at a given scale is
explicitly a falsifiable research hypothesis; see `research/README.md`,
`research/paper-catalog.md`, and `research/quantized-optimization.md`.

## Active Promotion Milestone

`integer-transformer-proof-v1`, defined in
`docs/integer-transformer-proof-v1.md` and enforced by `nsrl-eval`, remains a
frozen system-level result. A matched component ablation found that its fitted
suffix memory supplies every top-1 improvement; transformer logits improve
probability error but change no predicted tokens. The resulting headline gate is
therefore an unassisted transformer-only successor on the same frozen
partition. A 16-cell suffix-free successor sweep varied update scale, duration,
batch geometry, balancing, coverage, attention, and position policy; no row
passed, and the best remained at 5,094 mistakes versus the 2,510 gate. That
falsification required a change to the learning objective rather than promotion
of a hyperparameter variant. The v1 artifact remains a
transformer-plus-suffix-memory system rather than proof that its parametric
transformer learned the task by itself.

The first successor-v2 trial is preserved as a frozen falsification. Its
deterministic repair is now the active artifact. A constrained native
transformer head trains directly against canonical integer base-2 NLL on the
training partition and retains nonzero support for every byte class. On the
unchanged evaluation surface it scores 25,347,655 millibits with zero
zero-probability windows, versus 47,168,000 uniform, 38,271,425 retrieval,
38,025,720 byte n-gram, and 40,847,697 for a genuine trained float32
transformer. It strictly passes the first gate and every frozen successor
promotion comparison without retrieval, suffix memory, routing, or held-out
lookup. This repairs the substrate gate; it does not by itself satisfy the
separate NSRL-MME product release gate.

Solomon Council v0 is now implemented as a separate shadow-mode judgment gate.
Its mathematician, engineer, historian/source scholar, skeptic, consequence
planner, and judge/router each run under an Ed25519-signed capability seal and a
per-invocation permission/resource circle. The deterministic judge preserves
dissent and can select only a mathematical-controller-allowed recommendation,
request evidence, ask the user, or abstain. Hash-bound wisdom receipts include
model/source identity, evidence, confidence/calibration, predicted consequences,
permissions/budgets, outcome, and append-only revision history. The core and its
adversarial replay checks pass. Its production scorer now requires a pre-lane
public casebook with hidden-gold commitments, byte-bound solo and five-faculty
council traces, a post-lane gold opening bound to both bundles, exact receipt
replay, identical question/evidence inputs, a shared decision surface, and exact
provenance sets; self-test or hand-authored aggregates cannot
promote. Promotion does not: the frozen same-model wisdom evaluation across all
eight required dimensions has not yet been produced.

The p10m training roadmap has completed its local scaling-readiness review. A
contracted 2,048-window run bound the K/V schedule to the same initialization,
data order, context, batch geometry, budget, and held-out split as a float32 SGD
reference. Integer K, V, and output moved in every chunk with all 13 gradient
paths active and zero saturation; the integer lane finished 5,209 total
millibits below initialization while the float lane finished 98 mean millibits
below its initialization. Midpoint replay was byte-identical for the integer
model and optimizer and tensor-identical for all 13 float arrays. The result
does not authorize a paid scale run: optimizer families remain intentionally
different, validation quality was non-monotone across integer chunks, and the
remaining trunk groups still need safe boundaries. The isolated
gate-projection preflight at shift 23 is now complete: `gate` first moved at
window 768 and made 26 exact updates, only K/V/gate/output moved, all saturation
remained zero, held-out finished 5,209 total millibits below initialization,
and midpoint replay was byte-identical. Two source-relative `up` gates then
showed why safe movement is insufficient: shift 23 made 26 exact updates with
no dev gain, while shift 22 made 101,543 with zero saturation but still only
tied the source on selected dev and regressed the one-shot test by 1,245 total
millibits. A matched-horizon comparison found every final feature, logit,
probability, and per-window loss identical between shift 22 and shift 23 on all
256 dev windows. A predeclared forward-scale sweep then found shift 7 as the
first safe functional boundary: 250 feature/logit vectors and 124 probability
vectors changed, but no target probability changed. Fresh 1,024-window training
at that scale made 50,568 `up` updates with zero saturation and exact replay,
yet still tied source dev. The product roadmap now targets Q15 target-
probability resolution for the 8,192-token output, not another trunk shift. The
completed audit confirms the resolution loss: Q15 has only three observed
target values and hides every target delta, Q19 exposes one, and Q23 exposes
all 13 target-delta windows visible at Q31. However, scale-compensated Q19 and
Q23 training preflights produce the exact Q15 model and dev result after 256
windows; only optimizer residual bytes differ. The completed normalization
gate finds that one Q47 integer Newton refinement cuts worst-case mass error
from roughly 98,900 ppm to 98/83 ppm, close to the 73/74 ppm exact-division
ceiling. It also reduces observed target-change windows from 13 to 5, versus 4
under exact division. Because the predeclared selection rule required retaining
all 13 legacy target changes, that gate promoted no normalization. The completed
window-level attribution shows Newton contains all four exact target-change
windows, has one additional one-unit denominator-boundary change, and remains
within one Q23 unit of exact division across both complete probability vectors.
All nine legacy-only target changes have unchanged target logits and zero exact
Q23 delta. The bounded normalized wide-gradient preflight then proves the
selected Q23/`q47_newton1` signal reaches optimizer state exactly, reaches
features and logits after 155 `up` boundary crossings, and reaches three target
probabilities after an isolated output boundary crossing, all without
saturation. Neither materialization lane improves dev; the output boundary is
415 total millibits worse. The product roadmap therefore moves to a
target-aligned integer-objective review rather than more fractional bits or
paid scaling; paid scale remains unauthorized.

Reachable-update fingerprints are now an evidence-backed prioritization signal.
In a predeclared 30-cell longitudinal matrix, early functional movement
predicted later disjoint held-out gain with zero false positives, MCC 0.645,
and Spearman ρ 0.828. It missed six candidates that activated only later, and
all 16 early-reachable long runs saturated. The product implication is to use
early reachability to prioritize candidates while retaining delayed-activation
checks and adding saturation-aware control before longer expert runs.

Literary routing and Solomon multimodal generation are experiment suites. They
may produce candidate architectures and product evidence, but they do not
replace or redefine the substrate promotion milestone.

## Vision

NSRL should not become a smaller imitation of a general cloud LLM. It should
become the reference stack for local, deterministic, inspectable AI components:

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

The next product frontier is turning this into an agentic micro-model system:
expert packaging, deterministic routing, integer topic/memory state, adaptive
integer optimization, and browser deployment.

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
- Provide strong tests for numeric correctness, stability, and performance
  assumptions.
- Keep the codebase small enough to audit and reason about.

## Non-Goals

- Competing with PyTorch, TensorFlow, ONNX Runtime, or vendor inference engines.
- Supporting GPUs or accelerators in the first phase.
- Training monolithic frontier-scale or cloud-replacement models.
- Implementing every neural network layer type.
- Running standard Llama, GPT, or HuggingFace checkpoints through post-training
  quantization. NSRL models must be born into NSRL's base-2 attention contract.
- Hiding quantization behavior behind opaque compiler passes. Generated code is
  allowed only when the emitted Rust is deterministic, reviewable, testable, and
  bound to the same trace contracts as hand-written kernels.
- Optimizing before the numeric contract is proven.

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
  scaling strategy.
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

## Strategic Model Target

After the first character-level models learn, the project scales outward
toward expert swarms rather than upward toward a single giant model.

Target envelopes:

```text
micro expert:    1M-10M parameters
large expert:    10M-50M parameters
active route:    1-3 experts per request
deployment:      native CPU and WASM bundles
latency goal:    sub-100ms route + first useful output on ordinary clients
```

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

Attention is no longer deferred. It is part of the numeric core. The strategic
architecture, however, is not "make this block enormous"; it is "package many
small, typed, traceable experts and route between them deterministically."

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

Milestone 4: Agentic Edge is successful when:

- Models can be packaged as expert artifacts with manifests, tokenizer contracts,
  schemas, capabilities, hashes, and trace authority.
- A deterministic Rust router can select one or more experts for a request.
- Router traces explain expert selection and context handoff.
- At least three experts can be composed locally without cloud calls or GPU
  drivers.

Milestone 5: Adaptive Integer Training is successful when:

- Training supports blockwise integer momentum and variance/magnitude tracking.
- Per-block or per-parameter update shifts are derived from integer history,
  including `leading_zeros`/magnitude signals.
- Static global shift sweeps are no longer required for the main expert training
  lane.
- The optimizer state is training-only and excluded from inference artifacts.

Milestone 6: Graph And WASM Product Surface is successful when:

- `nsrl-graph` can generate checked forward/backward Rust for the core block
  primitives and match hand-written golden references.
- A browser demo loads a small expert bundle, routes locally, and emits a
  deterministic trace.
- WASM bundle size, startup latency, and route-to-first-output latency are
  measured in CI or release traces.

Milestone 7: Decentralized Model Launch Coordination is successful when:

- Sponsors can publish immutable metric bounties with escrow amounts, baselines,
  targets, non-regression guardrails, and frozen evaluator hashes.
- Builders can publish a versioned model/run recipe that binds source, dataset,
  architecture, stages, compute ceilings, promotion, and artifact outputs.
- Accepted stage, candidate, replay, and promotion evidence can append one
  idempotent, hash-linked reward block with exact capped integer allocation.
- Compute providers and validators sign receipts that can be independently
  replayed and challenged.
- Model publication binds its artifact and metric proof without allowing token
  balances or market prices to redefine promotion.

Status: signed localnet prototype. `nsrl.model_launch_recipe.v1` packages the
existing promoted integer-Transformer proof as a checked specimen. Ed25519
actors replay 31 hash-linked events through sponsor funding, bounded compute,
independent stage/candidate quorums, challenge resolution, model publication,
deterministic bounty settlement, and a capped model-local reward block. Forge
exposes the public transcript and gap map. Wallets, custody, transferable
assets, provider auctions, Sybil resistance, multi-writer consensus, and
external settlement remain unimplemented.

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
  frontier-scale models.
- Generated code policy: allowed only when emitted Rust is inspectable,
  deterministic, and trace-compatible.
- Deployment wedge: local-first CPU and WASM inference.
- Optimizer direction: adaptive integer optimizer state during training; no
  optimizer state in inference artifacts.

## Open Questions

- What is the smallest expert size that feels useful for grammar, routing, math,
  domain style, and API/tool selection?
- How should expert manifests describe capability, authority, and failure modes
  without becoming a second opaque framework?
- Should the first router be purely symbolic, learned, or hybrid?
- What block size gives the best adaptive integer optimizer tradeoff between
  quality, memory, and deterministic trace size?
- How much of `nsrl-train` should be replaced by `nsrl-graph` generated backward
  code, and which hand-written kernels remain the golden references?
- What WASM bundle size and startup latency targets are strict enough to make
  NSRL meaningfully better than browser ONNX/PyTorch-style deployments?
