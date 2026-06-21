# Research Library

Papers relevant to NSRL's architecture and open research questions.

## Index

### Integer-Only Training & Inference
- [integer-training.md](integer-training.md) — Papers on training neural networks with no floating-point arithmetic

### Efficient Attention
- [linear-attention.md](linear-attention.md) — Linear/subquadratic attention mechanisms and recurrent reformulations

### Test-Time Training
- [test-time-training.md](test-time-training.md) — Inference-time adaptation and the connection to linear attention state updates

### Training Control
- [adaptive-shift-control.md](adaptive-shift-control.md) — Why integer learning-rate shifts are a controller problem, why the holographic controller is premature, and the rule-based adaptive schedule to test first

## NSRL's Position

Most "integer LLM" work splits into two non-overlapping camps:

1. **Post-training quantization** (I-BERT, GPTQ, llama.cpp) — float pretraining, integer deployment only
2. **1-bit training with float internals** (BitNet b1.58) — ternary weights trained from scratch but with float Adam master weights

NSRL occupies a third position: integer arithmetic throughout training and inference, no float master weights, transformer-class architecture, language modeling target, `no_std` Rust with no external dependencies.

The closest prior work is PocketNN (2022) which does integer training+inference in pure C++ but only for toy MLP networks on MNIST via Direct Feedback Alignment.

## Open Questions from the Literature

- **Integer training instability**: NSRL solves quantize-to-zero via i64 batch gradient accumulation. BitNet uses float Adam. NITRO-D uses a different normalization strategy. No consensus approach.
- **Linear attention quality gap**: All linear attention papers (GLA, RetNet, RWKV) use float. The interaction between integer arithmetic and linear attention state accumulation is unexplored.
- **TTT / linear attention duality**: A 2025 paper shows test-time training on linear attention is equivalent to updating the KV state — this is directly implementable in NSRL's integer framework without float gradients.
- **Adaptive shift control**: Static integer learning-rate shifts are wrong over
  training time, but learned holographic control needs a non-gameable reward and
  a forgetting mechanism. The next empirical step is a rule-based controller
  before a learned meta-controller.
