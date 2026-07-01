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
separately from prompt-conditioned replay or embedded text memory. The optional
`--decode-logit-delta` raw-probe diagnostic subtracts deterministic initial
logits from trained logits; it is useful for measuring learned movement, not a
promoted coherent decoder. `--prompt-name-opening-prior` constrains only
`Solomon selects <prompt-name>: He` from a known spirit name in the prompt for
both char and chunked artifacts, which is useful for raw prompt-binding probes
but does not provide body prose. `--text-chunk-boost-q8 N` can be used with
chunked artifacts to test whether phrase/name chunk logits are useful without
embedded memory; large boosts tend to reveal repeated-chunk failure modes, so
this remains diagnostic rather than a promoted decoder.
Use `--embedded-text-lm-order 12 --text-prior-min-order 3 --text-prior-strict`
together with `--no-embedded-text-memory` to test the artifact-native text LM
path; sample traces report this as `text_prior_source:"embedded_lm"`. Named
prompts scope the LM to exact-prompt or primary-name matches, while
`king solomon seal` keeps the full memory. The order-12 default is long enough
to disambiguate repeated source phrases that shorter n-grams collapse. Strict
prior matches are applied before repeat filters and are not overridden by them.
The optional `chunked` text profile reserves byte tokens for common Solomon
phrases plus all 72 normalized primary spirit names, reducing name generation to
single-token choices in prompt-to-opening experiments.
Use `--prompt-profile seal-names` for the narrowest prompt-binding corpus; it
keeps only `seal of <primary_name>` prompts and removes generic and alias prompt
variants.
Use `nsrl-solomon-attention train --solomon-name-copy-init` with
`--text-token-profile chunked` to seed a copy-style attention path for
prompt-bound `Solomon selects <Name>: He` openings before prose fine-tuning.
`--solomon-name-copy-repair` reapplies just those copy/opening slots after a
fine-tune, which helps test body-training changes without silently losing the
prompt-bound opening circuit. This is not a body-prose solution by itself.
For body-start experiments, `--target-segment body-first-after-he` trains only
the token immediately after the `He ` opening, and
`--target-segment body-first-after-opening` trains the source first token after
`Solomon selects <Name>: ` so `He `, `This `, `His `, and similar openings can
be measured directly.
`--solomon-name-copy-repair-preserve-body-output` preserves non-opening
body-token output rows during the final scaffold repair.
`--solomon-body-scaffold` adds a deterministic raw-attention body sentence after
the prompt-bound opening. It is a quality floor for no-memory samples, not a
replacement for the embedded source-specific prose path.
`--solomon-body-opening-repair` uses embedded text memory to add an
attention-native first-body-token lane after `Solomon selects <Name>: `. The
repair keeps scaffold-owned tokens additive so the no-memory fallback remains
readable while non-`He ` source openings move toward argmax.
The raw-quality checker reports word-level repetition, case-noise, and Solomon
source-vocabulary metrics in addition to character-level metrics so repeated
names, glued repeated chunks, uppercase-heavy pseudo-words, out-of-corpus
fragments, and repeated `Solomon selects` restarts are not mistaken for
readable text. Pass `--prompt "seal of Bael"` to penalize missing or
wrong-spirit openings, and pass `--no-vocab` only for non-Solomon corpora.
Use `node scripts/check-solomon-attention-web-quality.mjs` to gate the
artifact-backed browser path directly; it exercises the JS `NSRLLMM1` sampler
against known prompt-scoped text and embedded image-memory output. Add
`--all-names --summary` to run the same gate across all 72 primary spirit
prompts.
Use `node scripts/probe-solomon-attention-raw-rank.mjs --prompt "seal of Bael"`
to inspect the raw transformer's next-token rank and margin at the
`Solomon selects ` prompt-name boundary. Add `--all-names --summary` to compare
the prompt-name boundary across all 72 primary spirit prompts.
Use `node scripts/probe-solomon-attention-body-start-rank.mjs --summary` to
compare the source-specific first body token after
`Solomon selects <Name>: ` across all 72 prompts. Numeric bracket footnote refs
are stripped before ranking so citation markers such as `[25]` do not count as
body prose. The promoted artifact should satisfy
`--min-top1 72 --min-top5 72 --min-top10 72`.

The text-quality curriculum smoke is:

```bash
scripts/run-solomon-attention-curriculum-smoke.sh
```

Attention eval traces include top-5/top-10 accuracy, mean target rank, and
target-vs-best logit margin so runs can distinguish lower probability error
from genuinely improved argmax readiness.

It builds a prompt/text-only pretraining corpus, initializes the joint
`NSRLLMM1` wrapper from that transformer, and gates bounded generated-text
accuracy. The joint stage has separate defaults for update scale
(`NSRL_SOLOMON_ATTENTION_JOINT_LEARNING_RATE=2`, 512 joint windows, and
joint-specific LR-shift variables), so it can accept real prompt/text/image
updates without changing the text pretrain scale. It also embeds compact
prompt/text/image memory in the joint artifact and gates a memory-assisted
sample with `--conditioning-examples none` and no
external text-prior file. The embedded memory selects a prompt-specific opening
and image-token plan for named spirit prompts before applying transition
constraints, so the Bael gate rejects coherent wrong-spirit text or seals. This
gives the attention path a replayable text/image-logit ratchet and an
artifact-native readable decoder path while
pure free-running language quality remains under active work. The smoke also
writes `raw-sample-bael/` with prompt conditioning, external priors, and
embedded memory disabled; this is the attention-only continuation probe, and it
is intentionally reported rather than quality-gated. The smoke also prints
non-gating raw-quality metrics from
`scripts/check-solomon-attention-raw-quality.mjs` so raw changes are comparable
across training experiments. It also writes
`opening-sample-bael/` with those same priors disabled plus
`--prompt-name-opening-prior`, gating that named prompts bind to the correct
`Solomon selects Bael: He` opening before raw continuation resumes. Larger local experiments
can set `NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE`,
`NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE`,
`NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS`, and
`NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS`.
`NSRL_SOLOMON_ATTENTION_NAME_INITIAL_REPEATS` adds first-name-token training
sequences, and `NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS` adds short
`Solomon selects <name>: He ` sequences. They can also set the
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_PHASE` filter (`all`, `special`,
`text-char`, `text-chunk`, or `image`) and
`NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT` /
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_SEGMENT` (`all`, `generated-text`,
`name-opening`, `name-opening-tail`, or `image`), plus
`NSRL_SOLOMON_ATTENTION_JOINT_*_LR_SHIFT` variables for joint-stage update-scale
sweeps. Text/opening pretrain still uses the non-joint
`NSRL_SOLOMON_ATTENTION_*_LR_SHIFT` variables.
`NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_CAP`,
`NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_MIN_WEIGHT_Q15`, and
`NSRL_SOLOMON_ATTENTION_ARGMAX_MARGIN_WEIGHT_Q15` expose experimental
target-frequency and argmax-margin trainer terms for text/opening pretrain;
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_FREQUENCY_CAP`,
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_FREQUENCY_MIN_WEIGHT_Q15`, and
`NSRL_SOLOMON_ATTENTION_JOINT_ARGMAX_MARGIN_WEIGHT_Q15` override them for the
joint stage. The defaults leave both experimental terms off, aside from the
inert min-weight floor.
`NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT=generated-text` restricts training
updates to generated text tokens after the `<TEXT>` marker; the
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_SEGMENT` override applies the same control
to the joint stage. Current probes show this helps isolate the objective but
does not solve raw free-running prose on its own.
`NSRL_SOLOMON_ATTENTION_NAME_OPENING_PRETRAIN` enables a short opening-only
pretrain corpus for prompt-to-name experiments, and
`NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_ORDER`,
`NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_MIN_ORDER`, and
`NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_STRICT` control the embedded-LM smoke
probe; the default is order-12 strict suffix replay so repeated source phrases
stay prompt-specific instead of collapsing into local loops. Attention train
traces record initial/final Q15 probability error; the smoke scripts fail if a
stage increases that error, and the curriculum gate fails if the joint stage
accepts no updates.
`nsrl-solomon-attention train --zero-output-head-init` is an experimental
diagnostic for starting with a neutral output head; it helps separate inherited
head bias from learned sequence quality, but it is not the promoted default.
`NSRL_SOLOMON_ATTENTION_REJECT_LOSS_REGRESSION` enables the trainer's stricter
per-batch loss-regression guard for larger experiments.

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
