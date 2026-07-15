# Research Library

This directory tracks the primary literature that constrains NSRL's research
claims and experiment design. It is an evidence map, not a list of papers that
happen to mention quantization.

Last literature review: **2026-07-15**.

## Start here

- [mathematical-journal/](mathematical-journal/) — append-only definitions,
  derivations, conjectures, falsifiers, and experiment decisions for the NSRL
  mathematical model.
- [paper-catalog.md](paper-catalog.md) — curated primary-source catalog with
  venue/status, numeric path, float-state boundary, and the concrete NSRL
  takeaway for each paper.
- [integer-training.md](integer-training.md) — history of native integer and
  fully quantized training, plus a corrected statement of NSRL's position.
- [quantized-optimization.md](quantized-optimization.md) — rounding, error
  feedback, update dead zones, oscillation, and the proposed reachable-capacity
  research program.
- [linear-attention.md](linear-attention.md) — recurrent linear attention,
  retention, gating, and delta-rule memories.
- [test-time-training.md](test-time-training.md) — parameter adaptation at
  inference and the precise relationship between TTT and linear attention.
- [adaptive-shift-control.md](adaptive-shift-control.md) — repository-specific
  training-control hypothesis and implemented controllers.

## Classification rules

Every paper is classified by its actual training path. These labels must not be
used interchangeably:

- **Native integer training**: forward pass, backward pass, optimizer update,
  and persistent training tensors use integer arithmetic. Integer scale metadata
  and wider integer accumulators are allowed.
- **Fully quantized training (FQT)**: weights, activations, and gradients are
  quantized for substantial parts of training, but floating-point scale
  computation, master weights, optimizer state, or unquantized operations may
  remain.
- **Quantization-aware training (QAT)**: a floating-point training graph
  simulates the deployment quantizer. The deployed model may be integer-only;
  the training path is not.
- **Low-precision floating point**: FP8, FP6, FP4, block floating point, and
  related formats. These papers are relevant controls, not integer-only prior
  art.
- **Integer-only inference**: training remains floating point and only the
  deployed forward path is integer.

When a paper does not make its master-weight, scale, or optimizer-state boundary
clear, the catalog says **unclear** instead of inferring a stronger claim.

## What the prior art already establishes

The literature already demonstrates all of the following:

1. Fixed-point neural-network training can work when rounding and dynamic range
   are handled carefully.
2. Native integer training has been demonstrated for MLPs, CNNs, and vision
   transformers; adjacent quantized-training work covers recurrent models and
   on-device transfer learning.
3. Error feedback can recover information discarded by a biased compressor or
   quantizer.
4. Different tensors and layers need different precision or scale control.
5. Integer-only transformer nonlinearities are practical at inference.
6. Quantized Transformer pretraining and fine-tuning can run most large matrix
   multiplications in INT8 while retaining floating-point support state.
7. Language models can be trained by directly updating low-precision weights
   without full-precision master copies, and recent optimizers can feed weight
   quantization error into momentum or quantize momentum itself.

NSRL therefore must not claim to have invented integer-only training, integer
gradient accumulation, error feedback, or integer transformer arithmetic.

## NSRL's narrower research position

The defensible differentiators are the combination of:

- a deterministic integer-trained causal language model rather than a vision
  classifier or inference-only conversion;
- no floating-point master weights or runtime training path;
- explicit checked scales, saturation counters, and replay hashes;
- `no_std` Rust inference artifacts for CPU, WASM, and embedded targets;
- exact deterministic parallel reduction and restart;
- and experiments on when nominal parameter capacity becomes reachable under a
  discrete optimizer.

The final item remains a bounded research result rather than an established
capacity law. A predeclared longitudinal rank × shift × carry matrix found that
early functional movement predicted later disjoint held-out gain with MCC
0.645 and Spearman ρ 0.828, but it also missed delayed activations and the long
runs saturated. See [quantized-optimization.md](quantized-optimization.md).

## Maintenance policy

When adding a paper:

1. Link the publisher, proceedings, OpenReview record, or arXiv abstract—not a
   blog summary.
2. Record publication status and distinguish a preprint from a reviewed paper.
3. State the numeric boundary: master weights, optimizer state, scales,
   accumulators, and any float-only operations.
4. Separate the authors' result from the inference NSRL draws from it.
5. Prefer a short claim that can be checked against the abstract or paper.
6. Update the review date above when the catalog is materially refreshed.

## Open questions after the review

- Can a hierarchical, phase-aware Boolean proposal concentrate negative
  document-level conditional effects without exponential cube enumeration?
  The first frozen trunk/head candidate failed prospective transfer despite its
  post-hoc one-Q20 advantage; the next proposal must predict fine-log boundary
  exposure, component cancellation, document visibility, and transfer direction.
- Can deterministic error feedback match the directional fidelity of seeded
  stochastic rounding while retaining exact replay for a fixed contract?
- Can the observed high-precision reachable-update signal replicate across
  trunks and corpora without the long-run saturation seen in this matrix?
- Can the one-group residual policy continue safely beyond K, V, and `gate`?
  Final gate-preflight residuals select `up` at shift 23, while the existing
  float control remains optimizer-family rather than optimizer-identical.
- Does exact integer accumulation improve long-context linear-attention state
  stability, or merely move the overflow boundary?
- Can an integer causal language model close the float quality gap without
  suffix memory, retrieval, or a float-trained teacher?
