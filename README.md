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

The evaluator reports constrained next-token top-1/top-5/top-10 accuracy, mean
target rank, target-vs-best logit margin, and Q15 probability error by prompt,
generated text, image, and special markers. Current free-running model-only
text remains weak; prompt-conditioned decoding is the quality-preserving path
for known Solomon prompts while the underlying attention trainer improves.
`nsrl-solomon-attention train` also exposes `--window-offset N` so raw-quality
experiments can cover different next-token residue classes when using a stride
greater than one. The Solomon attention runners default to `--stride 1` now so
capped runs cover all target phases instead of one modulo-stride class.
For raw continuation probes, use `sample --conditioning-examples none
--text-prior-examples none --no-embedded-text-memory`; add `--text-prefix
"Solomon selects "` to test whether the model can continue a generic scaffold
without prompt-conditioned replay or artifact text memory. Add
`--decode-logit-delta` for an experimental diagnostic that subtracts the
deterministic initial-model logits from candidate scores; this exposes learned
logit movement but is not yet the promoted quality path.
`--prompt-name-opening-prior` constrains only the short
`Solomon selects <prompt-name>: He` opening from a known spirit name in the
prompt for both char and chunked artifacts; it is useful for raw prompt-binding
probes and does not supply body prose.
`--text-chunk-boost-q8 N` can be used with chunked artifacts to test whether
whole-phrase/name chunk logits contain usable signal without embedded memory;
it is diagnostic, not a promoted decoder.
For an artifact-native text LM path, add `--embedded-text-lm-order 12
--text-prior-min-order 3 --text-prior-strict` with
`--no-embedded-text-memory`; this rebuilds prompt-scoped local transition
statistics from compact model memory and records
`text_prior_source:"embedded_lm"`. Named prompts scope the LM to exact-prompt
or primary-name matches; the generic `king solomon seal` prompt keeps the full
memory. The order-12 default is long enough to disambiguate repeated source
phrases such as Decarabia's two `of Birds` contexts. Strict prior matches are
applied before repeat filters and are not overridden by them.
The optional `chunked` text profile uses reserved byte tokens for high-frequency
Solomon phrases and all 72 normalized primary spirit names, which makes
prompt-to-name experiments less dependent on character-by-character spelling.
For narrow prompt-binding probes, build the corpus with
`--prompt-profile seal-names`; this keeps only `seal of <primary_name>` prompts
and removes generic and alias prompt noise.
`nsrl-solomon-attention train --solomon-name-copy-init` is an experimental
chunked-profile initializer that seeds a copy-style attention path for
`seal of <name>` prompts. It makes raw `Solomon selects <Name>: He` openings
prompt-bound before prose fine-tuning, and requires
`--text-token-profile chunked`. `--solomon-name-copy-repair` reapplies only
those copy/opening slots after a fine-tune, which is useful for checking whether
body-training gains survive without regressing prompt-bound openings.
When training body chunks on top of that scaffold, add
`--solomon-name-copy-repair-preserve-body-output` so the final repair does not
erase non-opening body-token output rows. `--target-segment body-first-after-he`
is a narrow curriculum target for only the token immediately after the `He `
opening; it is useful for diagnosing body-start logits separately from later
function-word frequency. `--target-segment body-first-after-opening` targets the
first token after `Solomon selects <Name>: `, so source openings such as `He `,
`This `, `His `, and `and ` can move toward argmax without forcing every spirit
through the same `He ` start.
`--solomon-body-scaffold` overlays a deterministic no-memory body transition
path after the prompt-bound `He ` opening. It provides a clean raw-attention
fallback sentence for every normalized primary name; source-specific body prose
still comes from the embedded text-memory/LM path.
`--solomon-body-opening-repair` is an experimental chunked-profile repair that
uses embedded text memory to add a name-conditioned first-body-token attention
lane after `Solomon selects <Name>: `. It preserves scaffold-owned body tokens
with an additive signal, so `and`/`A` openings stay near argmax without
damaging the deterministic raw fallback.
The raw-quality checker reports character repetition, word repetition,
case-noise, and Solomon source-vocabulary metrics so repeated names, glued
repeated chunks, uppercase-heavy pseudo-words, out-of-corpus fragments, and
repeated `Solomon selects` restarts no longer score as readable prose. Pass
`--prompt "seal of Bael"` to penalize missing or wrong-spirit openings, and
pass `--no-vocab` only when probing a non-Solomon corpus.
The browser artifact path is checked separately with
`node scripts/check-solomon-attention-web-quality.mjs`; it loads the checked-in
`NSRLLMM1` artifact through the same JS sampler used by the app and verifies
known prompt-scoped text plus embedded image-memory output. Add
`--all-names --summary` to verify prompt-bound text and embedded seal output
across all 72 primary spirit prompts.
Use `node scripts/probe-solomon-attention-raw-rank.mjs --prompt "seal of Bael"`
to inspect the raw attention logits at the prompt-name boundary. The probe
reports the expected embedded-memory continuation token, its raw rank/margin,
and the top raw candidates after a prefix such as `Solomon selects `. Add
`--all-names --summary` to report top-1/top-5/top-10 and median rank across all
72 primary spirit prompts.
Use `node scripts/probe-solomon-attention-body-start-rank.mjs --summary` to
measure the source-specific first token after `Solomon selects <Name>: `. This
is the raw body-start diagnostic; the promoted artifact gates it with
`--min-top1 72 --min-top5 72 --min-top10 72` after stripping numeric bracket
footnote refs, so most cleaned source prose openings are raw argmax and every
opening remains in the near-candidate set.
Use `node scripts/check-solomon-attention-raw-scaffold.mjs --summary` to verify
the checked-in no-memory raw fallback sentence across all 72 normalized primary
spirit names.

Run the current model-only text-quality curriculum gate:

```bash
scripts/run-solomon-attention-curriculum-smoke.sh
```

This pretrains the same `NSRLLMM1` transformer on prompt/text-only Solomon
sequences, wraps it against the joint prompt/text/image corpus, and gates the
bounded generated-text eval. The joint stage uses its own default update scale
(`NSRL_SOLOMON_ATTENTION_JOINT_LEARNING_RATE=2`, 512 joint windows, and
joint-specific LR-shift variables) so it accepts real prompt/text/image updates
without changing the text pretrain scale. It is a measured improvement path for
text and image logits, not a claim that pure free-running attention prose is
solved. The same smoke
embeds compact prompt/text/image memory in the `NSRLLMM1` artifact and gates a
model-native memory-assisted sample with `--conditioning-examples none`. The
embedded memory is prompt-aware for spirit prompts, so `seal of Bael` must
produce a Bael opening and Bael 16x16 image-token plan rather than any coherent
Solomon sentence or seal. This produces readable sentence-level text and a
prompt-scoped image plan without external text-prior flags. It also writes
`raw-sample-bael/` with
`--conditioning-examples none --text-prior-examples none
--no-embedded-text-memory --text-prefix "Solomon selects "` so attention-only
continuation quality stays visible. The smoke reports non-gating raw-quality
metrics from `scripts/check-solomon-attention-raw-quality.mjs` for this sample
so raw changes can be compared without relying only on eyeballing text. It also
writes `opening-sample-bael/` with
the same external and embedded priors disabled plus `--prompt-name-opening-prior`,
gating that named prompts bind to the correct `Solomon selects Bael: He`
opening before raw continuation resumes. For deeper local runs, tune
`NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE`, `NSRL_SOLOMON_ATTENTION_PROMPT_PROFILE`,
`NSRL_SOLOMON_ATTENTION_MAX_TEXT_CHARS`, and
`NSRL_SOLOMON_ATTENTION_JOINT_TEXT_ONLY_REPEATS`.
`NSRL_SOLOMON_ATTENTION_NAME_INITIAL_REPEATS` adds short sequences that train
only the first name token after `Solomon selects `, while
`NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS` adds short
`Solomon selects <name>: He ` sequences. The runners also expose
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_PHASE` (`all`, `special`, `text-char`,
`text-chunk`, or `image`) for stage-specific joint fine-tune experiments,
`NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT` and
`NSRL_SOLOMON_ATTENTION_JOINT_TARGET_SEGMENT` (`all`, `generated-text`,
`name-opening`, `name-opening-tail`, or `image`),
`NSRL_SOLOMON_ATTENTION_JOINT_OUTPUT_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_JOINT_MLP_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_JOINT_EMBED_LR_SHIFT`, and the joint attention
LR-shift variables for joint-stage update-scale sweeps. Text/opening pretrain
still uses
`NSRL_SOLOMON_ATTENTION_OUTPUT_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_MLP_LR_SHIFT`,
`NSRL_SOLOMON_ATTENTION_EMBED_LR_SHIFT`, and the attention LR-shift variables
for integer update-scale sweeps. `NSRL_SOLOMON_ATTENTION_NAME_OPENING_PRETRAIN`
and `NSRL_SOLOMON_ATTENTION_NAME_OPENING_REPEATS` enable an opt-in
prompt-to-opening curriculum. `NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_ORDER`,
`NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_MIN_ORDER`, and
`NSRL_SOLOMON_ATTENTION_EMBEDDED_TEXT_LM_STRICT` control the embedded-LM smoke
probe; the default is order-12 strict suffix replay so repeated source phrases
stay prompt-specific instead of collapsing into local loops. Attention train
traces record initial/final Q15 probability error; the smoke scripts fail if a
stage increases that error, and the curriculum gate fails if the joint stage
accepts no updates.
`NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_CAP`,
`NSRL_SOLOMON_ATTENTION_TARGET_FREQUENCY_MIN_WEIGHT_Q15`, and
`NSRL_SOLOMON_ATTENTION_ARGMAX_MARGIN_WEIGHT_Q15` expose experimental
target-frequency and argmax-margin trainer terms for text/opening pretrain;
the `NSRL_SOLOMON_ATTENTION_JOINT_*` variants override those terms during the
joint stage. These default off except for the inert min-weight floor and should
be treated as raw-quality probes, not promoted decoder settings.
`NSRL_SOLOMON_ATTENTION_TARGET_SEGMENT=generated-text` restricts training
updates to tokens after the Solomon `<TEXT>` marker and before image/end
markers; `NSRL_SOLOMON_ATTENTION_JOINT_TARGET_SEGMENT` overrides it for the
joint stage. This is an experimental raw-quality control and does not currently
solve free-running prose by itself.
`nsrl-solomon-attention train --zero-output-head-init` is an experimental
diagnostic that starts the transformer with a neutral output head; it is useful
for separating inherited head bias from learned sequence quality, but it is not
the promoted default.
`NSRL_SOLOMON_ATTENTION_REJECT_LOSS_REGRESSION` turns on the trainer's stricter
per-batch loss-regression guard for larger experiments.

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
