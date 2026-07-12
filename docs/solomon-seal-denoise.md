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

Run the diagnostic-only oracle conditioning split:

```bash
node scripts/run-solomon-oracle-condition-diagnostic.mjs \
  --source-samples path/to/generative-eval/samples.tsv \
  --retrieval-head path/to/retrieval-head.json
```

This gives the denoiser the true held-out 16x16 signature as an attention plan.
It must not be used as headline evidence. Its purpose is to separate a bad
prompt-to-plan latent prior from a denoiser that cannot follow a correct plan.
The checked-in 72-prompt diagnostic row in
`docs/solomon-oracle-condition-diagnostic.tsv` shows near-perfect signature
identity under oracle plans but weak rendered-image retrieval identity, so the
next model work should first improve learned prompt-to-signature planning.

The v2 quality report treats this as product-generation evidence only when the
run directory includes `summary.tsv`, `config.json`, and `samples.tsv`; those
sidecars prove the samples came from the held-out `eval` partition with
`decoded-latent` sampler targets. Add `--retrieval-head PATH` to also score the
rendered held-out bitmaps with the v2 image retrieval head, or let v2 attention
post-score the existing generative eval run once it has trained
`retrieval-head.json`.

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

This emits an `NSRLLMM1` native causal mini-transformer artifact. The compiled
integer core now uses the promoted width (`d_model=128`, two heads, hidden dim
256) and a two-block stacked forward trunk. Lower stacked blocks initialize as
no-op residual blocks, and the serial and map-reduce host trainers
backpropagate through stacked blocks with conservative lower-layer warm-up. The
map-reduce path now accumulates per-layer gradients for stacked Graviton-style
batches. The smoke run uses a small attention-training window budget with
accumulated integer batches, then samples with
prompt-conditioned corpus decoding from `examples.jsonl`. The smoke gates
coherent Bael/Stolas text and 256-token image plans.
For local experiments set `NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce` and
`NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0` to use the same auto-worker
batched path that the AWS end-to-end runner selects by default.

Set `NSRL_SOLOMON_ATTENTION_CORPUS_VERSION=v2` to build the task-marked
bidirectional binding corpus. It adds identify, text-to-image, image-to-text,
image-to-explain, text-image-explain, image-to-attributes, explain,
description-to-image, positive match, wrong-seal no-match, and
wrong-prompt/name no-match records while
preserving canonical joint examples for sampling memory. The attention eval
trace reports a `tasks` object for these records. Supervised v2 task prompts
are identity-bearing; a generic canonical prompt like `king solomon seal` is
normalized to a spirit-specific prompt before it is used for retrieval or match
training. V2 also emits explicit identity binding rows for primary names,
aliases, and seal-ID prompts, and the retrieval spine requires each one as both
`identify` and `text-to-image` evidence. V2 defaults to
`--image-token-profile symbolic16`, which serializes ink, edge,
component/topology, radial-position, and stroke-direction channels using the
same 16 image-bin tokens.
For v2 runs, the attention smoke scripts call
`scripts/check-solomon-attention-task-eval.mjs` and
`scripts/check-solomon-v2-retrieval-spine.mjs`; these gates require all task
groups to be present, reject skipped or invalid contexts by default, verify
72-spirit task coverage, require the `symbolic16` image-token profile/channels
by default, check held-out prompt retrieval when the prompt corpus is
available, and prove both no-match directions bind to the intended spirit
pair.
By default v2 runners require held-out prompt evidence
(`NSRL_SOLOMON_V2_REQUIRE_HELDOUT_PROMPTS=1`,
`NSRL_SOLOMON_V2_MIN_HELDOUT_PROMPT_ROWS=72`) from
`NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS` or the checked-in expanded prompt
corpus, so promotion-style reports cannot pass with held-out retrieval omitted.
The no-match rows include both wrong-seal examples and mirrored wrong-prompt/name
examples so the retrieval spine learns mismatch direction, not only distance to
the nearest seal.
Product runners expose the no-match floors as `NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1`
and `NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1`, both defaulting to `72`.
They also train and gate `retrieval-head.json`, a sparse integer auxiliary
text/image class head that records explicit identity-anchor scores and provides
a model artifact for the retrieval spine before generation.
Generated sample binding uses that retrieval head to classify the generated
16x16 plan back to a spirit, so the gate proves image-to-text identity as well
as nearest target-signature distance.
For staged curriculum runs, set
`NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_STAGES=identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind`;
the runner uses `scripts/filter-solomon-multimodal-corpus.mjs` to derive ordered
task corpora from the same v2 stream, then continues into the final full joint
pass. The final `native-bind` stage defaults to
`NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS=2`; product gates require that
extra binding pressure unless explicitly disabled for experiments. The AWS
end-to-end runner uses that v2 curriculum stage order by default.
Staged runs also write `curriculum-stages.json`,
which gates the filtered manifests, stage train traces, stage order, and filter
recipes before the final quality report. Filtered manifests carry
identity-binding hashes, and the stage checker requires identity-sensitive
stages to preserve the source alias and seal-ID binding rows. The image-to-text
stage includes seal-to-name, seal-to-source-description,
text+seal-to-source-description, and seal-to-attributes records.
Generated known-prompt samples are then checked by
`scripts/check-solomon-attention-sample-binding.mjs`, which ranks the emitted
`image.ink16.u8` plan against the target signatures and verifies retrieval-head
image identity. The saved `sample-binding.json`/`prior-sample-binding.json`
traces include rank margins and text/image agreement flags, giving each sample
a compact retrieval confidence trace. The smoke scripts also persist
`identity-inference.json` from `scripts/infer-solomon-v2-identity.mjs`, which
uses the same sparse integer retrieval head as a reusable text-to-identity,
seal-plan-to-identity, and sample agreement report with required source-text
evidence.
They also persist
`generation-integrity.json` from
`scripts/check-solomon-generation-integrity.mjs`; this fails if generated
sample traces expose target-pixel/oracle guidance, retrieval-hybrid target
sources, or display-time cleanup/postprocess fields. Finally,
`scripts/check-solomon-v2-quality-report.mjs` writes `quality-report.json`,
joining task eval, retrieval-head eval, sample binding, identity inference, and
generation integrity. Set `NSRL_SOLOMON_V2_MIN_TOTAL_TOP5_PER_MILLE`,
`NSRL_SOLOMON_V2_MIN_TEXT_TOP5_PER_MILLE`, or
`NSRL_SOLOMON_V2_MIN_IMAGE_TOP5_PER_MILLE` to ratchet native model-only top-5
quality on larger runs. Set
`NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=72` to require each task bucket to
evaluate at least a full 72-spirit target set. Product AWS runs combine that
with `NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=none` and
`NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=all=72`, requiring the same breadth across
special/control, prompt, text, and image eval phases.
Set
`NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=1` to require every v2 task bucket
to clear a native top-5 floor, with task-specific overrides such as
`image-to-text=500`. Set `NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS` and
`NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS` to ratchet source text
overlap for explanation/source rows and image-to-attributes rows; product AWS
defaults are `2` and `8`. `NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS`
defaults to `0`, rejecting generic source placeholder prose from grounded
source tasks. Use `NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1`,
`NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1`, `NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1`,
and `NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1` to ratchet positive and
negative match rows; product AWS defaults each to `72`. Use
`NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN` to require score separation from the
nearest wrong spirit; product AWS defaults it to `1`. When
`NSRL_SOLOMON_V2_GENERATIVE_EVAL` is supplied, the
report also verifies the generative eval `config.json` and `samples.tsv`
sidecars before setting `product_generation_ready`; generated retrieval columns
can be ratcheted with
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE` and
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE`, plus
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN`; the final report
recomputes those generated retrieval ranks from raw sample bytes before
accepting them. Product runs also
default `NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY=1`, which requires
the matching product-floor model to have every held-out generated 128x128 sample
identify top-1 with a positive retrieval margin after report-side recomputation. Supplying that
artifact also implies a minimal generated-signature floor:
`effective_min_generated_top5_16_per_mille` is at least `1`, so zero-hit
generated 16x16 signature runs remain incomplete evidence. The same report records the architecture profile and
head split: `NSRLLMM1` token heads for text chars, image-plan bins, and text
chunks, plus the auxiliary 72-way retrieval class head. Set
`NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=1`,
`NSRL_SOLOMON_V2_MIN_D_MODEL`, `NSRL_SOLOMON_V2_MIN_HEADS`,
`NSRL_SOLOMON_V2_MIN_HIDDEN_DIM`, `NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS`, and
`NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN` to ratchet promoted runs toward the target
small-model shape. The base-2 softmax path requires a power-of-four per-head
dimension; with two heads, the valid promoted width is `d_model=128`
(`head_dim=64`), while `d_model=64` would produce invalid `head_dim=32`. Set
`NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1` for promotion-grade runs;
that gate requires `d_model=128`, two heads, hidden dim 256-512, 2-4
transformer layers, and context length 384-768.
Use `NSRL_SOLOMON_V2_REQUIRE_IDENTITY_INFERENCE=1`,
`NSRL_SOLOMON_V2_REQUIRE_CURRICULUM_STAGES=1`, and
`NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE=1` when a run should fail unless the
full bidirectional binding, staged curriculum, and denoise product-path
artifacts are present. V2 smokes also default
`NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE=1`, which requires the final
`quality-report.json` to show prompt retrieval, image retrieval, hard-negative
match checks, source evidence, generated sample agreement, and any required
denoise bridge agreeing in one cross-modal trace.
Set `NSRL_SOLOMON_ATTENTION_DENOISER_MODEL` to add the optional product-path
bridge: the smoke passes each generated `image.ink16.u8` plan to
`nsrl-bitmap-sample --attention-plan`, then
`scripts/check-solomon-attention-denoise-bridge.mjs` verifies that the 128x128
denoiser trace used `latent_target_source:"attention-plan"` and exactly the
generated plan bytes. The checker also hashes the denoiser model named by the
sampler `trace.model`, and the final quality report recomputes that hash so a
bridge cannot swap in a stale or unrelated denoiser endpoint. The AWS
end-to-end runner wires this automatically from the pipeline denoiser model
when the `denoiser` stage or an explicit
`NSRL_SOLOMON_DENOISE_MODEL` is present; set
`NSRL_SOLOMON_ATTENTION_DENOISER_MODEL=none` for an attention-only rerun that
skips bridge sampling. It also downsamples the 128x128 raw samples back to a
16x16 signature, records plan distance, rejects flat output with no ink range,
and, for v2 runs with a retrieval head, requires that downsampled output to
identify as the prompted spirit. The bridge checker also recomputes the
retrieval head `model_hash`, so a stale or forged image scorer cannot bless a
denoised output. The final quality report recomputes those bridge output stats
from raw denoiser bytes before accepting the bridge sidecar.
Set
`NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_SIGNATURE_DISTANCE` after
an initial measuring run to make plan/output distance a hard gate; use
`NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_INK_RANGE` to raise the non-flat
output floor above the default of 1. Use
`NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS` to require distinct expected
spirit coverage in the bridge artifact; product Graviton runs default this to
`2` today and should ratchet toward all 72 Solomon targets as the denoise bridge
sample set grows.

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
replacement for the embedded source-specific prose path. It is limited to
<=64d diagnostic builds; promoted-width runs should use
`--solomon-body-opening-repair` plus retrieval/binding gates instead.
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
node scripts/check-solomon-product-diagnostic.mjs
node scripts/check-solomon-denoiser-model.mjs --model PATH
scripts/check-solomon-eval-replay.sh
node scripts/check-solomon-prior-smoke.mjs --run-dir PATH
node scripts/check-solomon-generation-integrity.mjs --sample-dir PATH
node scripts/check-solomon-attention-denoise-bridge-self-test.mjs
node scripts/check-solomon-attention-denoise-bridge.mjs --pair ATTENTION:DENOISE
node scripts/check-solomon-heldout-retrieval-proof.mjs
node scripts/check-solomon-v2-quality-report.mjs --eval PATH --retrieval-head-eval PATH
node scripts/check-solomon-native-directional-eval-smoke.mjs
scripts/run-solomon-multimodal-smoke.sh
scripts/run-solomon-attention-smoke.sh
scripts/run-solomon-attention-curriculum-smoke.sh
```

Use `check-solomon-product-diagnostic.mjs` as the end-to-end local proof before
promoting a run. It wraps the denoise bridge self-test together with corpus,
held-out retrieval, native eval, generative provenance, promotion bundle, and AWS
dry-run plan evidence; `--fast` keeps only the quicker checks for iteration.

The native directional eval smoke builds the real v2 symbolic corpus and trains
the integer attention model at 384-token context. It is intentionally small, but
it still requires the promoted-width `d_model=128`, two-head, two-layer shape,
measured special/text/image output heads, and all four product directional
groups before passing.

If samples are weak, the result is weak. Tune the model, dataset, or prior; do
not add target-pixel guidance or display-time cleanup.
