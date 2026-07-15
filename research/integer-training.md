# Integer-Only Training and Inference

This note covers work that trains with integer or aggressively quantized
arithmetic. The complete source list and numeric-boundary annotations live in
[paper-catalog.md](paper-catalog.md).

## The important distinction

“Integer model” can describe at least four different systems:

1. a float-trained model converted to integer inference;
2. QAT with float master weights and simulated integer inference;
3. fully quantized training where the large operations are integer but scales,
   optimizer state, or selected gradients remain floating point;
4. native integer training where forward, backward, optimizer updates, and
   persistent training state avoid floating-point arithmetic.

NSRL targets the fourth category. Papers in the other categories are still
important controls, but they do not establish the same execution contract.

## Prior art timeline

### Fixed-point and discrete training foundations

[Deep Learning with Limited Numerical Precision](https://proceedings.mlr.press/v37/gupta15.html)
(2015) established that fixed-point training is viable when stochastic rounding
preserves small updates. [Training Quantized Nets: A Deeper Understanding](https://proceedings.neurips.cc/paper/7163-training-quantized-nets-a-deeper-understanding)
(2017) explained why methods with high-precision latent representations can
search directions that purely quantized methods cannot.

[WAGE](https://openreview.net/forum?id=HJGXzmspb) (2018) is a major direct
predecessor. It quantizes weights, activations, gradients, and errors, removes
BatchNorm in favor of constant scaling, and describes discrete training and
inference dataflow. Its experiments are an algorithmic simulation rather than a
`no_std` integer runtime, but it invalidates any claim that end-to-end discrete
backpropagation began with NSRL.

### Integer-only training systems

[NITI](https://arxiv.org/abs/2009.13108) (2020) stores parameters and
intermediate values as integers, computes with integer arithmetic, and uses
pseudo-stochastic rounding. It reports 8-bit training for smaller image tasks
and wider accumulation for ImageNet.

[Octo](https://www.usenix.org/conference/atc21/presentation/zhou-qihua)
(2021) applies INT8 forward and backward computation to on-device learning,
using loss-aware compensation and parameterized clipping to manage
quantization error.

[PocketNN](https://arxiv.org/abs/2201.02863) (2022) demonstrates dependency-free
integer-only MLP training in C++ using direct feedback alignment rather than
backpropagation.

[Is Integer Arithmetic Enough for Deep Learning Training?](https://arxiv.org/abs/2207.08822)
(NeurIPS 2022) is the strongest broad predecessor found in this review. It
implements integer forward propagation, backpropagation, normalization, and SGD
features using dynamic fixed-point mapping and stochastic rounding. Its
experiments include classification, object detection, segmentation, and vision
transformers.

[NITRO-D](https://arxiv.org/abs/2407.11698) (2024) extends native integer
training to deep CNNs using local losses, explicit scaling layers, and
IntegerSGD. [TIFeD](https://arxiv.org/abs/2411.16442) applies integer direct
feedback alignment to federated tinyML.

### Sequence models and Transformers

[Training Integer-Only Deep Recurrent Neural Networks](https://arxiv.org/abs/2212.11791)
(2022) uses QAT to produce integer recurrent inference with LayerNorm,
attention, and piecewise-linear activations. It is close in model domain but not
native integer training.

[I-BERT](https://proceedings.mlr.press/v139/kim21d.html) (2021) and
[I-ViT](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_I-ViT_Integer-only_Quantization_for_Efficient_Vision_Transformer_Inference_ICCV_2023_paper.pdf)
(2023) show that Softmax, GELU, and normalization can be implemented with
integer approximations for Transformer inference. Both begin from
floating-point training.

[Jetfire](https://openreview.net/forum?id=ltzTHGFF5i) and
[AMPA](https://proceedings.mlr.press/v235/ding24b.html) (ICML 2024) demonstrate
accurate Transformer training with INT8 or adaptively mixed low-bit data paths.
They retain a conventional support path for scaling and optimization, so they
are fully quantized training rather than NSRL-style no-float training.

[BitNet b1.58](https://arxiv.org/abs/2402.17764) and
[BitNet a4.8](https://arxiv.org/abs/2411.04965) establish that ternary weights
and low-bit activations can support language-model quality at scale. Their
training still relies on floating-point latent weights and optimizer state.

## Closest-work comparison

| System | Global backprop | Native integer optimizer | Sequence/attention | From scratch | Main boundary versus NSRL |
|---|---:|---:|---:|---:|---|
| WAGE | Yes | Integer-oriented discrete SGD | No | Yes | Simulated discrete algorithm; CNN-era architecture |
| NITI | Yes | Yes | No | Yes | Vision models and GPU-oriented implementation |
| PocketNN | No; DFA | Yes | No | Yes | Small MLPs and local feedback |
| Integer Arithmetic Enough | Yes | Yes | Vision Transformer | Yes | Dynamic fixed-point/stochastic scales; not causal language modeling or `no_std` |
| Integer RNN | QAT backward | No native claim | RNN attention | Float/QAT | Integer deployment rather than native training |
| NITRO-D | Local losses | IntegerSGD | No | Yes | CNN classification and local learning |
| Jetfire | Yes | No | Transformer | Yes | INT8 large ops with floating-point support path |
| BitNet | STE/QAT-style | No | Causal LLM | Yes | Float latent/master weights and optimizer |
| NSRL | Yes | Yes | Causal linear/base-2 attention | Yes | Current quality evidence is small; promoted proof includes suffix memory |

## What NSRL can defensibly investigate

The literature review narrows the contribution. NSRL should focus on the
combination of:

- deterministic native integer training of a causal language decoder;
- exact replay across threaded reduction and checkpoint resume;
- checked static runtime scales and explicit saturation/liveness evidence;
- no float master weights or optimizer state;
- deployable `no_std` Rust artifacts;
- and the relationship between discrete optimizer resolution and effective
  model capacity.

The last item is developed as a falsifiable hypothesis in
[quantized-optimization.md](quantized-optimization.md).

## Corrections to earlier repository claims

Earlier versions of this file described `i64` batch accumulation as an
unpublished technique and PocketNN as the closest overall prior. Those claims
were too strong:

- wide accumulation is common in low-precision training;
- carrying discarded update mass is closely related to established error
  feedback;
- WAGE, NITI, Octo, and the NeurIPS 2022 integer-training paper are direct
  predecessors;
- and the NeurIPS 2022 work includes integer training of vision transformers.

The specific NSRL implementation may still be useful or novel in combination,
but novelty cannot rest on integer accumulation alone.

## Evidence still missing in NSRL

1. An unassisted transformer-only candidate that passes the frozen proof; the
   completed v1 component ablation found that suffix memory supplies all top-1
   gain, and a 16-cell suffix-free successor sweep still missed the mistake gate
   by 2,584.
2. Functionally visible activation of additional p10m trunk groups. `up` is now
   safely reachable: shift 22 produced 101,543 exact updates with zero
   saturation. But a matched window-1,024 comparison against shift 23 found
   identical final features, logits, probabilities, and per-window losses on
   all 256 dev windows. Forward shift 7 then exposed safe differences in 250
   feature/logit vectors and 124 probability vectors, but no target probability;
   fresh training still tied dev. The completed precision audit shows Q19
   exposes one target delta and Q23 exposes all 13 seen at Q31, but compensated
   Q19/Q23 training still produces the exact Q15 model bytes and dev loss after
   256 windows. Only optimizer residual states differ. The follow-up audit
   recovers normalized mass accuracy with one Q47 integer Newton step: worst-
   case Q23 error falls from about 98,900 ppm to 98/83 ppm, close to the 73/74
   ppm exact-division ceiling. However, changed target windows fall from 13 to
   5, versus 4 under exact division. Window-level attribution now shows Newton
   preserves every exact target-change window, adds one one-unit denominator-
   boundary change, and stays within one Q23 unit of exact division everywhere
   in both probability vectors. All nine legacy-only target changes have
   unchanged target logits and zero exact Q23 delta. The remaining evidence is
   training-side: test whether normalized Q23 Newton gradients cross a model
   boundary in a bounded preflight.
3. A float comparison matched in optimizer family. The new reference is
   matched in architecture, initialization, data and order, context, batch,
   window budget, and evaluation, but compares integer residual SGD with
   float32 SGD rather than identical optimizer arithmetic.
4. Replication of the bounded longitudinal result across trunks and corpora.
   Early movement predicted disjoint held-out gain with MCC 0.645 and Spearman
   ρ 0.828, but six early no-ops activated later and all early-reachable long
   runs saturated.
5. A systematic literature review suitable for a publication-level novelty
   claim. This repository catalog is a curated engineering review, not a
   systematic review.
