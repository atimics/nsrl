# Active Schemas

This file lists the public trace and artifact contracts for the active Solomon
multimodal pipeline.

## `nsrl.bitmap_denoise_dataset.v1`

Emitted by `nsrl-build-solomon-bitmap-denoise-dataset`.

Purpose: record deterministic construction of clean/noisy bitmap pairs,
corruption labels, split assignment, target cleaning policy, image size,
timesteps, and stable hashes for replay.

## `nsrl.bitmap_denoise_multichannel_trace.v1`

Emitted by `nsrl-bitmap-multichannel-denoise`.

Purpose: record `NSRLTCH` text-conditioned denoiser training configuration,
dataset hashes, model dimensions, update counters, preview hashes, and final
model hash.

Artifact magic: `NSRLMCH\n`.

## `nsrl.bitmap_sampler_trace.v1`

Emitted by `nsrl-bitmap-sample`.

Purpose: record deterministic sampling configuration, model hashes, prompt,
latent target metadata, candidate selection metrics, output hashes, and the
generation source contract.

The valid generation contract is raw model sampling from noise. Target bitmap
guidance and display-time cleanup are outside this schema.

## `nsrl.solomon_latent_trace.v1`

Emitted by `nsrl-solomon-latent-train`.

Purpose: record `NSRLLAT1` prompt-to-layout prior training, prompt/gold
partitions, feature dimensions, class-head accuracy, latent hashes, and final
model hash.

Artifact magic: `NSRLLAT1`.

## `nsrl.solomon_multimodal_corpus.v1`

Emitted by `scripts/build-solomon-multimodal-corpus.mjs`.

Purpose: record deterministic construction of the joint Solomon stream over
prompt bytes, generated text bytes, and 16x16 image-bin tokens.

Outputs include `corpus.tokens.u16` for `NSRLMOD1` and `corpus.tokens.u8` for
the byte-vocab attention model.
`prompt_profile` records whether prompts were built from `generic`, `names`,
`seal-names`, or `all`; `seal-names` keeps only `seal of <primary_name>`
prompts for prompt-binding probes. `sequence_profile` may be `joint`,
`text-only`, `name-opening`, or `joint-and-text`; `name_initial_repeats` and
`name_opening_repeats` record optional short prompt-to-opening curriculum
sequences.
When `text_token_profile` is `chunked`, `text_chunk_base` and `text_chunks`
describe the reserved byte-token range. The current table uses 96 chunks:
common Solomon phrases plus all normalized primary spirit names.

## `nsrl.solomon_multimodal_train_trace.v1`

Emitted by `nsrl-solomon-multimodal train`.

Purpose: record `NSRLMOD1` integer discrete transition training, token hashes,
context row counts, and final model hash.

Artifact magic: `NSRLMOD1`.

## `nsrl.solomon_multimodal_sample_trace.v1`

Emitted by `nsrl-solomon-multimodal sample`.

Purpose: record joint sampling from one prompt into generated text plus a coarse
16x16 image token plan.

## `nsrl.solomon_multimodal_eval.v1`

Emitted by `scripts/run-solomon-multimodal-eval.mjs`.

Purpose: record `NSRLMOD1` tracked-corpus replay metrics for the deployed
artifact. The evaluator rebuilds the expected prompt/text/image-token stream
from the tracked Solomon text/signature table, verifies the rebuilt token hash
against the artifact, then reports phase-specific next-token top-k ranks.
This is artifact-native replay evidence, not a free-running quality benchmark.

## `nsrl.solomon_attention_train_trace.v1`

Emitted by `nsrl-solomon-attention train`.

Purpose: record `NSRLLMM1` causal mini-transformer training over the joint
Solomon byte-token stream, including token hash, model hash, inner transformer
hash, attention kind, position policy, text token profile, adaptive-training
flags, learning-rate and component LR-shift settings, optional embedded
text-memory order/example count, target-token min/max filters, whether
`zero_output_head_init`, experimental chunked `solomon_name_copy_init`, or
experimental `solomon_name_copy_repair` was used, whether repair preserved
non-opening body output rows (`solomon_name_copy_repair_preserve_body_output`),
whether the deterministic raw body fallback was overlaid
(`solomon_body_scaffold`),
target-frequency weighting fields
(`target_frequency_cap`, `target_frequency_min_weight_q15`), the optional
argmax margin term (`argmax_margin_weight_q15`), the target segment filter
(`target_segment`, for example `all`, `generated-text`, `name-opening`,
`name-opening-tail`, `body-after-he`, `body-first-after-he`,
`body-first-after-opening`, or `image` in Solomon traces),
sequence length, stride, window offset, window count, examined windows,
accepted/rejected batches, rollback count, rejected window count, and final
integer accuracy/error metrics. Train traces also include
`initial_probability_error_q15`,
`final_probability_error_q15`, and `probability_error_delta_i64`; smoke scripts
use those fields to reject runs whose final probability error is higher than
the initial error. The Solomon attention curriculum smoke also rejects joint
train traces whose `updates` or `accepted_batches` are zero, so the joint stage
cannot pass as a no-op.

Artifact magic: `NSRLLMM1`.

The artifact wraps an `NSRLMT4\n` mini-transformer model plus the Solomon token
layout, token hash, attention kind, position policy, text token profile, and
optional compact prompt/text/image memory examples. Version 4 memory entries
include the 256 image tokens; version 3 memory entries remain readable as
text-only memory.

## `nsrl.solomon_attention_sample_trace.v1`

Emitted by `nsrl-solomon-attention sample`.

Purpose: record grammar-constrained attention sampling from one prompt into
generated text plus exactly 256 image-bin tokens for a coarse 16x16 seal plan.
The trace records the artifact `text_token_profile` used to encode prompts and
decode generated text, plus optional `text_prefix` and `text_prefix_tokens`
when sampling starts from an explicit generated-text prefix.
When prompt-conditioned corpus decoding is active, the trace records the
matched `conditioning_primary_name`, `conditioning_prompt`,
`conditioning_score`, and conditioning text/image token counts.
When text-transition prior decoding is active, the trace records
`text_prior_source` (`external`, `embedded`, `embedded_lm`, or `none`),
`text_prior_order`, `text_prior_min_order`, `text_prior_contexts`,
`text_prior_prompt_starts`, `text_prior_selected_start_tokens`,
`text_prior_boost_q8`, and `text_prior_strict`. `embedded` means the prior was
rebuilt from compact memory stored inside the `NSRLLMM1` artifact as the strict
high-order decoder path. `embedded_lm` means `--embedded-text-lm-order N`
rebuilt transition statistics from that same compact artifact memory; named
prompts scope those transitions to exact-prompt or primary-name matches, while
the generic `king solomon seal` prompt keeps all examples. Strict prior matches
are applied before repeat filters and are not overridden by them. Passing
`--no-embedded-text-memory` without `--embedded-text-lm-order` keeps
`text_prior_source` at `none` for raw attention probes even when the artifact
contains compact memory. When artifact image memory is active, the trace records
`image_prior_source`, `image_prior_primary_name`, `image_prior_prompt`, and
`image_prior_tokens`; `--no-embedded-text-memory` also disables this image prior
so raw attention image probes stay visible. Experimental raw probes may also set
`--decode-logit-delta`; sample traces record this as `decode_logit_delta` when
candidate scores use trained logits minus the deterministic initial-model
logits for the same context. Raw probes may set
`--prompt-name-opening-prior`; sample traces record this as
`prompt_name_opening_prior` when the sampler constrains only the short
`Solomon selects <prompt-name>: He` opening from a known spirit name in the
prompt. Chunked-artifact probes may also set `--text-chunk-boost-q8 N`; sample
traces record this as `text_chunk_boost_q8`. This is a diagnostic for testing
whether whole-phrase chunk logits contain usable signal, not a memory or corpus
conditioning path.

## `nsrl.solomon_attention_eval_trace.v1`

Emitted by `nsrl-solomon-attention eval`.

Purpose: record constrained teacher-forced next-token accuracy for an
`NSRLLMM1` artifact over serialized Solomon examples. The trace reports total,
special-marker, prompt-text, generated-text, and image-token target counts,
correct counts, invalid context counts, per-mille top-1/top-5/top-10 accuracy,
mean target rank, target-vs-best logit margin, and Q15 probability error, plus
the artifact text token profile. Rank and margin expose whether probability
error improvements are moving the true next token near argmax. This is the
quality signal for model-only attention behavior, separate from
prompt-conditioned sampling.

## `nsrl.solomon_attention_raw_rank_summary.v1`

Emitted by `scripts/probe-solomon-attention-raw-rank.mjs --summary`.

Purpose: summarize raw transformer next-token rank at a fixed generated-text
prefix such as `Solomon selects `. With `--all-names`, the summary reports
top-1/top-5/top-10 counts, median and mean expected-token rank, worst rank,
median and worst expected-token margin, and a short miss list across all 72
primary spirit prompts. This is a prompt-name boundary diagnostic; it does not
claim free-running prose quality.

## `nsrl.solomon_attention_raw_scaffold_summary.v1`

Emitted by `scripts/check-solomon-attention-raw-scaffold.mjs`.

Purpose: verify the no-memory raw fallback sentence across all 72 primary
spirit prompts. The checker samples with conditioning, external text priors,
and embedded text memory disabled, keeps only the prompt-name opening prior,
and requires the deterministic body scaffold text for each normalized primary
name. This is a quality-floor gate for raw attention samples, not a
source-specific prose benchmark.

## `nsrl.solomon_attention_body_start_rank_summary.v1`

Emitted by `scripts/probe-solomon-attention-body-start-rank.mjs --summary`.

Purpose: summarize raw transformer next-token rank for the cleaned source first
body token after `Solomon selects <Name>: `. The checker uses embedded model
memory only to identify the expected source token, strips numeric bracket
footnote refs, then ranks the raw model logits at that prefix. The promoted
gate requires high top-1/top-5 and complete top-10 recovery so source-specific
body openings are near argmax without replacing the raw no-memory scaffold.
`top1Misses` lists the remaining near misses even when every prompt is inside
top-10, while `misses` stays reserved for prompts outside top-10.

## `nsrl.solomon_attention_web_quality_summary.v1`

Emitted by `scripts/check-solomon-attention-web-quality.mjs --summary`.

Purpose: summarize browser-sampler quality checks for an `NSRLLMM1` artifact.
With `--all-names`, the checker verifies prompt-bound text starts, weak-repeat
rejection, strict embedded-text-LM output at order 12 or higher, and strict
embedded seal-image memory across all 72 primary spirit prompts.

## `nsrl.solomon_eval_trace.v1`

Emitted by `nsrl-solomon-eval`.

Purpose: record held-out prompt partition metrics for a latent prior, including
top-k retrieval/class accuracy, prompt-set version, partition hash, model hash,
and optional ledger rows.

## `nsrl.solomon_prior_smoke_check.v1`

Emitted by `scripts/check-solomon-prior-smoke.mjs`.

Purpose: gate a full prior-smoke run by checking expected files, latent/eval
metrics, sampler traces, target-source honesty, and prompt panel artifacts.
