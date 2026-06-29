# NSRL — Integer-Only Multimodal Training

NSRL is a pure-Rust integer-only training stack. The active pipeline trains a
text-conditioned bitmap generator for Solomon seal targets without floating
point arithmetic in training or sampling.

![72 text-conditioned seals sampled from the integer denoiser](docs/assets/solomon-text-conditioned-seals.png)

The current model family is:

- `NSRLTCH`: text-conditioned bitmap denoiser, i8 weights and u8 ink.
- `NSRLLAT1`: learned prompt-to-layout latent prior for text prompts.
- `NSRLMOD1`: discrete joint prompt/text/image-token model for coarse
  multimodal Solomon samples.
- `NSRLLMM1`: attention-based causal joint prompt/text/image-token model for
  native Solomon multimodal samples.
- `nsrl-core`: no-std integer kernels used by the trainer.

## Contract

Weights are i8 from initialization. Activations are Q15 i16. Large reductions
and gradient buffers accumulate in i64, then quantize back to i8 at batch
boundaries. The same arithmetic contract used for sampling is used during
training.

```text
source images + text signatures
  -> denoise dataset
  -> NSRLTCH text-conditioned denoiser
  -> NSRLLAT1 prompt/layout prior
  -> sampled bitmap seals
  -> held-out prior and generation eval
```

No float master weights, no post-training quantization, and no target-bitmap
lookup during generation.

## Workspace

```text
crates/
  nsrl-core/       no_std integer inference and numeric kernels
  nsrl-corpus/     corpus utilities retained for deterministic preprocessing
  nsrl-train-core/ no_std training kernels shared by the host trainer
  nsrl-train/      Solomon training, eval, and sampling binaries
  nsrl-web-wasm/   wasm Solomon sampler parity surface
docs/
  solomon-seal-denoise.md      current pipeline contract and commands
  schemas.md                   active Solomon trace/artifact contracts
scripts/
  build-solomon-*.mjs          dataset, prompt, text-index builders
  run-solomon-*.sh|mjs         local eval/sample/sweep runners
  check-solomon-*.mjs|sh       pipeline honesty and replay gates
```

## Build And Check

```bash
cargo build --release -p nsrl-train \
  --bin nsrl-build-solomon-bitmap-denoise-dataset \
  --bin nsrl-bitmap-multichannel-denoise \
  --bin nsrl-bitmap-sample \
  --bin nsrl-solomon-latent-train \
  --bin nsrl-solomon-eval \
  --bin nsrl-solomon-multimodal \
  --bin nsrl-solomon-attention

./scripts/check.sh
```

## Pipeline

Build the bitmap denoise dataset:

```bash
node scripts/build-solomon-bitmap-denoise-dataset.mjs
```

Train the text-conditioned denoiser:

```bash
scripts/run-solomon-text-denoiser-train-local-docker.sh
```

Train and evaluate the prompt-to-layout prior, then sample a fixed prompt panel:

```bash
scripts/run-solomon-prior-smoke-local-docker.sh

NSRL_SOLOMON_LATENT_MODEL=data/local-runs-linux/local-solomon-prior-smoke/latent/model.nsrllat \
  scripts/run-solomon-coherence-panel.sh
```

Run held-out generative eval:

```bash
node scripts/run-solomon-generative-eval.mjs \
  --latent-model current=data/local-runs-linux/local-solomon-prior-smoke/latent/model.nsrllat
```

Build the first joint text/image model:

```bash
scripts/run-solomon-multimodal-smoke.sh
```

This trains `NSRLMOD1` from a single serialized stream:
`<BOS> <PROMPT> ... <TEXT> ... <IMAGE> ... <EOS>`. The image side is a coarse
16x16 token plan, not the high-resolution denoiser.

Build the attention-based joint model:

```bash
scripts/run-solomon-attention-smoke.sh
```

This trains `NSRLLMM1`, a native causal mini-transformer wrapper using
base-2 softmax attention and learned absolute positions over the same joint
byte-token stream. The smoke samples with prompt-conditioned corpus decoding
from `examples.jsonl`, and gates the output text plus 256-token image plans so
coherent known-prompt samples stay checked.

Model-only attention quality is measured separately:

```bash
cargo run -p nsrl-train --bin nsrl-solomon-attention -- eval \
  --model data/processed/key-solomon-goetia-attention-v1/model.nsrllmm \
  --tokens data/processed/key-solomon-goetia-attention-v1/corpus.tokens.u8 \
  --conditioning-examples data/processed/key-solomon-goetia-attention-v1/examples.jsonl
```

The evaluator reports constrained next-token accuracy by prompt, generated
text, image, and special markers. Current free-running model-only text remains
weak; prompt-conditioned decoding is the quality-preserving path for known
Solomon prompts while the underlying attention trainer improves.
`nsrl-solomon-attention train` also exposes `--window-offset N` so raw-quality
experiments can cover different next-token residue classes when using a stride
greater than one. The Solomon attention runners default to `--stride 1` now so
capped runs cover all target phases instead of one modulo-stride class.
For raw continuation probes, use `sample --conditioning-examples none
--text-prior-examples none --no-embedded-text-memory`; add `--text-prefix
"Solomon selects "` to test whether the model can continue a generic scaffold
without prompt-conditioned replay or artifact text memory.
For a non-exact artifact-native quality path, add `--embedded-text-lm-order 6`
with `--no-embedded-text-memory`; this uses lower-order transition statistics
from compact model memory and records `text_prior_source:"embedded_lm"`.

Run the current model-only text-quality curriculum gate:

```bash
scripts/run-solomon-attention-curriculum-smoke.sh
```

This pretrains the same `NSRLLMM1` transformer on prompt/text-only Solomon
sequences, wraps it against the joint prompt/text/image corpus, and gates the
bounded generated-text eval. It is a measured improvement path for text logits,
not a claim that pure free-running attention prose is solved. The same smoke
embeds compact prompt/text memory in the `NSRLLMM1` artifact and gates a
model-native memory-assisted sample with `--conditioning-examples none`. The
embedded memory is prompt-aware for spirit prompts, so `seal of Bael` must
produce a Bael opening rather than any coherent Solomon sentence. This produces
readable sentence-level text without exact prompt-conditioned example replay or
external text-prior flags. It also writes `raw-sample-bael/` with
`--conditioning-examples none --text-prior-examples none
--no-embedded-text-memory --text-prefix "Solomon selects "` so attention-only
continuation quality stays visible. For deeper local runs, tune
`NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE`, `NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE`,
`NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS`, and
`NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS`. The runners also expose
`NSRL_SOLOMON_ATTENTION_OUTPUT_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_MLP_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_EMBED_LR_SHIFT`, and the attention LR-shift variables
for integer update-scale sweeps.

## Native Binaries

- `nsrl-build-solomon-bitmap-denoise-dataset`: creates deterministic clean/noisy
  bitmap pairs and dataset manifests.
- `nsrl-bitmap-multichannel-denoise`: trains the `NSRLTCH` denoiser.
- `nsrl-solomon-latent-train`: trains the `NSRLLAT1` prompt/layout prior.
- `nsrl-solomon-eval`: evaluates held-out prompt partition accuracy.
- `nsrl-bitmap-sample`: samples raw generated bitmap panels.
- `nsrl-solomon-multimodal`: trains and samples the `NSRLMOD1` joint
  text/image-token model.
- `nsrl-solomon-attention`: trains and samples the `NSRLLMM1` attention-based
  joint text/image-token model, evaluates constrained next-token accuracy, and
  optionally uses prompt-conditioned corpus decoding for quality-preserving
  known-prompt samples.

## Evidence Gates

```bash
node scripts/check-solomon-denoiser-model.mjs --model PATH
node scripts/check-solomon-generation-honesty.mjs
scripts/check-solomon-eval-replay.sh
node scripts/check-solomon-prior-smoke.mjs --run-dir PATH
```

`check-solomon-generation-honesty.mjs` is load-bearing: it guards against
reintroducing target bitmap guidance or display-time cleanup into the sampler.

## Current Focus

The active work is model quality inside the Solomon pipeline:

- improve the denoiser without procedural cleanup,
- improve prompt-to-layout generalization on held-out prompts,
- keep native and wasm sampling byte-aligned,
- grow the joint `NSRLMOD1` and `NSRLLMM1` paths from coarse image plans toward
  richer image tokens and stronger text,
- make every claim replayable through checked integer traces.
