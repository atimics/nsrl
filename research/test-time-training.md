# Test-Time Training & Inference-Time Adaptation

Papers on updating model parameters during inference. Directly relevant to NSRL because integer arithmetic makes this cheap and exact.

---

## Key Papers

### TTT with KV Binding is Secretly Linear Attention (2025)
- **arXiv**: https://arxiv.org/pdf/2602.21204
- **Key insight**: Test-time training on a linear attention model (updating weights on the current context using a gradient step) is mathematically equivalent to updating the KV state S. The "learning" that happens at test time is the same operation as the linear attention state update.
- **Implication for NSRL**: Incremental linear attention state update IS test-time training. NSRL can implement TTT without any separate gradient computation step — just maintain S across the generation window. The integer arithmetic makes this exact and reproducible (no float accumulation drift).
- **Direct connection**: The state S = Σ kₜ⊗vₜ can be interpreted as a weight matrix W that has been "trained" on the current context. Each new token updates W via an outer product step.

### Test-Time Training on Nearest Neighbors for Large Language Models (2024)
- **arXiv**: https://arxiv.org/abs/2511.16691
- **Approach**: At inference, retrieve nearest-neighbor sequences from a corpus, fine-tune on them briefly, then generate. Reduces perplexity across diverse domains.
- **Relevance**: Shows TTT is practically useful for domain adaptation. NSRL's fast integer training loop makes retrieval-and-finetune cheap — a 4096-window fine-tune takes ~4 seconds.

### SLOT: Sample-specific Language Model Optimization at Test-time (2025)
- **arXiv**: https://arxiv.org/html/2505.12392v1
- **Approach**: Treat the prompt itself as training data; run a few gradient steps before generating.
- **Relevance**: NSRL's rollback mechanism + i64 batch accumulation enables this without risk of destructive updates — if the gradient step causes overflow or quality regression, roll back.

---

## NSRL's TTT Opportunity

NSRL has properties that make TTT unusually practical:

**Fast training**: ~1ms per window at d=16, seq=32. A 64-window context update takes ~64ms on CPU. Negligible for interactive use.

**Exact reproducibility**: Integer arithmetic means the same context always produces the same weight delta. No float accumulation drift across TTT steps.

**Built-in rollback**: The ring-buffer checkpoint mechanism already handles destructive updates. TTT can be wrapped in the same try-update / rollback logic used during training.

**Linear attention duality**: With incremental linear attention state, TTT is free — the state update IS the adaptation. No backward pass needed.

**No float state**: Standard TTT requires float optimizer state (Adam moments, etc.). NSRL's integer optimizer has no float state, so TTT adds zero memory overhead beyond the model weights.

---

## Connection to Self-Synthesis Pipeline

The `run-simplewiki-self-synthesis.sh` script is a coarse form of TTT at corpus scale: generate synthetic text, train on it, generate again. The TTT literature suggests doing this at inference time (per prompt) rather than as a separate training run.

A viable architecture:
1. Receive prompt
2. Run 8–16 gradient steps on recent context (O(ms) with integer training)
3. Generate with updated model
4. Optionally revert weights after generation (stateless serving) or keep (continual learning)

The rollback mechanism already implements step 4 optionally.
