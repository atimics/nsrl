# Solomon Pipeline

This is the active NSRL training lane: text-conditioned integer bitmap
generation for Solomon seal targets.

## Contract

- Training and sampling stay integer-only.
- Sampling starts from deterministic sparse noise.
- The denoiser may use learned text and latent conditioning.
- Generation does not read target bitmap files as pixel guidance.
- Output samples are raw generated ink at source resolution.
- Eval may compare generated samples to held-out targets, but eval never feeds
  target pixels back into sampling.

## Commands

Build the denoise dataset:

```bash
node scripts/build-solomon-bitmap-denoise-dataset.mjs
```

Train `NSRLTCH`:

```bash
scripts/run-solomon-text-denoiser-train-local-docker.sh
```

Train/evaluate `NSRLLAT1` and sample a prompt panel:

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

Build the coarse joint text/image-token model:

```bash
scripts/run-solomon-multimodal-smoke.sh
```

This emits an `NSRLMOD1` model and samples both generated text and a 16x16 image
token plan from the same discrete context.

Build the attention-based joint text/image-token model:

```bash
scripts/run-solomon-attention-smoke.sh
```

This emits an `NSRLLMM1` native causal mini-transformer artifact. The smoke run
uses a small attention-training window budget with accumulated integer batches,
then samples with prompt-conditioned corpus decoding from `examples.jsonl`. The
smoke gates coherent Bael/Stolas text and 256-token image plans.

Use the native eval command to measure the model-only attention path:

```bash
cargo run -p nsrl-train --bin nsrl-solomon-attention -- eval \
  --model data/processed/key-solomon-goetia-attention-v1/model.nsrllmm \
  --tokens data/processed/key-solomon-goetia-attention-v1/corpus.tokens.u8 \
  --conditioning-examples data/processed/key-solomon-goetia-attention-v1/examples.jsonl
```

Free-running attention text is still weak at the current model scale; the
checked coherent text path is prompt-conditioned corpus decoding.
The native trainer accepts `--window-offset N`, which is useful for raw-quality
experiments that need to train target positions skipped by a larger stride.
The local Solomon attention runners default to stride 1 so capped runs train
across target phases instead of only one modulo-stride residue class.
Raw sample probes should pass `--conditioning-examples none`,
`--text-prior-examples none`, and `--no-embedded-text-memory`. Add
`--text-prefix "Solomon selects "` when testing generic scaffold continuation
separately from prompt-conditioned replay or embedded text memory.
Use `--embedded-text-lm-order 6` together with `--no-embedded-text-memory` to
test the non-exact artifact-native text LM path; sample traces report this as
`text_prior_source:"embedded_lm"`.

The text-quality curriculum smoke is:

```bash
scripts/run-solomon-attention-curriculum-smoke.sh
```

It builds a prompt/text-only pretraining corpus, initializes the joint
`NSRLLMM1` wrapper from that transformer, and gates bounded generated-text
accuracy. It also embeds compact prompt/text memory in the joint artifact and
gates a memory-assisted sample with `--conditioning-examples none` and no
external text-prior file. The embedded memory selects a prompt-specific opening
for named spirit prompts before applying transition constraints, so the Bael
gate rejects coherent wrong-spirit text. This gives the attention path a
replayable text-logit ratchet and an artifact-native readable decoder path while
pure free-running language quality remains under active work. The smoke also
writes `raw-sample-bael/` with prompt conditioning, external priors, and
embedded memory disabled; this is the attention-only continuation probe, and it
is intentionally reported rather than quality-gated. Larger local experiments
can set `NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE`,
`NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE`,
`NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS`, and
`NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS`, plus the
`NSRL_SOLOMON_ATTENTION_*_LR_SHIFT` variables for update-scale sweeps.

## Gates

```bash
node scripts/check-solomon-denoiser-model.mjs --model PATH
node scripts/check-solomon-generation-honesty.mjs
scripts/check-solomon-eval-replay.sh
node scripts/check-solomon-prior-smoke.mjs --run-dir PATH
scripts/run-solomon-multimodal-smoke.sh
scripts/run-solomon-attention-smoke.sh
scripts/run-solomon-attention-curriculum-smoke.sh
```

If samples are weak, the result is weak. Tune the model, dataset, or prior; do
not add target-pixel guidance or display-time cleanup.
