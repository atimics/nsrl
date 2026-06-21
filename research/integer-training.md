# Integer-Only Training & Inference

Papers on neural networks that train and/or infer using only integer arithmetic.

---

## Foundational QAT

### Jacob et al. (2017) — Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- **arXiv**: https://arxiv.org/abs/1712.05877
- **Approach**: Quantization-aware training (QAT) — trains in float with simulated quantization noise, recovers INT8 weights for deployment. Inference uses integer-only arithmetic.
- **Relevance**: The canonical QAT paper. NSRL diverges by not having a float training phase at all — this is the baseline everyone else builds on.
- **Key difference from NSRL**: Float master weights during training; integer is the deployment format, not the training format.

---

## Integer Inference of Transformers

### I-BERT (2021) — Integer-only BERT Quantization
- **arXiv**: https://arxiv.org/abs/2101.01321
- **Authors**: Kim, Gholami, Yao, Mahoney, Keutzer (UC Berkeley)
- **Approach**: End-to-end integer-only BERT inference. Approximates GELU, Softmax, LayerNorm with integer polynomial approximations. 2.4–4× speedup on T4 GPU vs FP32.
- **Relevance**: Shows that transformer nonlinearities (softmax, layernorm) can be approximated in integers without quality collapse. NSRL uses similar LUT-based approximations for exp (base-2 softmax) and rsqrt (RMSNorm).
- **Key difference from NSRL**: Inference-only quantization of a pretrained float model. Training is standard float Adam.

---

## 1-Bit / Ternary Training

### BitNet b1.58 (2024) — The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
- **arXiv**: https://arxiv.org/abs/2402.17764
- **Authors**: Ma, Wang, et al. (Microsoft Research)
- **Approach**: Ternary weights {-1, 0, 1} trained from scratch. Uses absmean quantization to map weights to ternary. INT8 activations. Matches FP16 quality at 3B+ parameters. 71.4× more energy-efficient matrix multiply.
- **Relevance**: Closest published work to NSRL's ternary kernel. Demonstrates ternary weights are sufficient for language modeling quality at scale.
- **Key difference from NSRL**: Uses float Adam with float master weights during training — ternary is the storage/inference format. NSRL has no float path.
- **Follow-on**: BitNet a4.8 (arXiv:2411.04965) extends to 4-bit activations.

---

## Integer Training (No Float Path)

### PocketNN (2022) — Integer-only Training and Inference via Direct Feedback Alignment
- **arXiv**: https://arxiv.org/abs/2201.02863
- **Venue**: tinyML Research Symposium 2022
- **Approach**: Pure C++, integer-only training AND inference, no explicit quantization step. Uses Direct Feedback Alignment (DFA) instead of backpropagation to avoid float gradients. "Pocket activations" maintain integer precision. 96.98% on MNIST, 87.7% on Fashion-MNIST.
- **Relevance**: Closest to NSRL's philosophy — no float anywhere, embedded target. Demonstrates integer training is feasible.
- **Key difference from NSRL**: DFA not backprop; toy MLP networks only (no attention, no language modeling); C++ not Rust.
- **Related**: TIFeD (arXiv:2411.16442) extends this to federated learning.

### NITRO-D (2024) — Native Integer-only Training of Deep Convolutional Neural Networks
- **arXiv**: https://arxiv.org/abs/2407.11698
- **Approach**: Framework for arbitrarily deep integer-only CNNs, training and inference. Uses a novel normalization strategy to maintain integer precision across layers.
- **Relevance**: More recent and more capable than PocketNN. Shows integer training scales to deeper networks.
- **Key difference from NSRL**: Vision/CNN focused, not sequence modeling or language.

### Training Integer-Only Deep Recurrent Neural Networks (2022)
- **arXiv**: https://arxiv.org/abs/2212.11791
- **Approach**: Integer-only QAT for RNNs with layer normalization and attention. Adaptive piecewise linear activation approximation.
- **Relevance**: Applies integer training to sequence models. Closer to NSRL's domain than PocketNN/NITRO-D.

---

## NSRL's Specific Contributions vs Literature

| Aspect | Literature approach | NSRL approach |
|--------|-------------------|---------------|
| Training precision | Float (with optional QAT) | Integer throughout |
| Gradient accumulation | Float Adam master weights | i64 batch accumulation |
| Nonlinearities | Polynomial or LUT approximation | LUT (exp, rsqrt), piecewise hard-SILU |
| Architecture | CNN, BERT, RNN | Multi-head attention + gated MLP + RMSNorm |
| Target | GPU serving | CPU, `no_std`, embedded |
| Language | C++, Python/PyTorch | Rust `no_std` |
| Quantize-to-zero fix | Float master weights | i64 accumulation across batch windows |

The i64 gradient accumulation across batch windows is an unpublished technique for solving the quantize-to-zero problem without float master weights. It is the key training innovation not present in the literature.
