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

## `nsrl.solomon_multimodal_train_trace.v1`

Emitted by `nsrl-solomon-multimodal train`.

Purpose: record `NSRLMOD1` integer discrete transition training, token hashes,
context row counts, and final model hash.

Artifact magic: `NSRLMOD1`.

## `nsrl.solomon_multimodal_sample_trace.v1`

Emitted by `nsrl-solomon-multimodal sample`.

Purpose: record joint sampling from one prompt into generated text plus a coarse
16x16 image token plan.

## `nsrl.solomon_attention_train_trace.v1`

Emitted by `nsrl-solomon-attention train`.

Purpose: record `NSRLLMM1` causal mini-transformer training over the joint
Solomon byte-token stream, including token hash, model hash, inner transformer
hash, attention kind, position policy, text token profile, adaptive-training
flags, learning-rate and component LR-shift settings, optional embedded
text-memory order/example count, sequence length, stride, window offset, window
count, examined windows, accepted/rejected batches, rollback count, rejected
window count, and final integer accuracy/error metrics.

Artifact magic: `NSRLLMM1`.

The artifact wraps an `NSRLMT4\n` mini-transformer model plus the Solomon token
layout, token hash, attention kind, position policy, text token profile, and
optional compact prompt/text memory examples.

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
`text_prior_order`, `text_prior_contexts`, `text_prior_prompt_starts`,
`text_prior_selected_start_tokens`, `text_prior_boost_q8`, and
`text_prior_strict`. `embedded` means the prior was rebuilt from compact memory
stored inside the `NSRLLMM1` artifact as the strict high-order decoder path.
`embedded_lm` means `--embedded-text-lm-order N` rebuilt a lower-order
statistical text model from that same compact artifact memory. Passing
`--no-embedded-text-memory` without `--embedded-text-lm-order` keeps
`text_prior_source` at `none` for raw attention probes even when the artifact
contains compact memory.

## `nsrl.solomon_attention_eval_trace.v1`

Emitted by `nsrl-solomon-attention eval`.

Purpose: record constrained teacher-forced next-token accuracy for an
`NSRLLMM1` artifact over serialized Solomon examples. The trace reports total,
special-marker, prompt-text, generated-text, and image-token target counts,
correct counts, invalid context counts, per-mille accuracy, and Q15 probability
error, plus the artifact text token profile. This is the quality signal for
model-only attention behavior, separate from prompt-conditioned sampling.

## `nsrl.solomon_eval_trace.v1`

Emitted by `nsrl-solomon-eval`.

Purpose: record held-out prompt partition metrics for a latent prior, including
top-k retrieval/class accuracy, prompt-set version, partition hash, model hash,
and optional ledger rows.

## `nsrl.solomon_prior_smoke_check.v1`

Emitted by `scripts/check-solomon-prior-smoke.mjs`.

Purpose: gate a full prior-smoke run by checking expected files, latent/eval
metrics, sampler traces, target-source honesty, and prompt panel artifacts.
