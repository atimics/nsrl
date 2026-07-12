# AWS Solomon Training

The cloud lane is a plain Linux runner plus S3 artifacts. The scripts do not
create credentials, buckets, public ACLs, or distributions. They assume the
instance has an IAM role or AWS CLI profile that can read/write the chosen S3
prefix.

## Active Runners

Run the default end-to-end Graviton pipeline:

```bash
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-end-to-end.sh
```

Real product runs require a Linux ARM64/Graviton-compatible runner and EC2
IMDSv2 instance metadata by default. The runner records `runner_kernel`,
`runner_arch`, `require_graviton`, `ec2_instance_id`, `ec2_instance_type`, and
placement fields in `run.env`, and exits before training if
`NSRL_SOLOMON_REQUIRE_GRAVITON=1` is set on a non-Graviton host or
`NSRL_SOLOMON_REQUIRE_EC2_METADATA=1` cannot read EC2 metadata. Set either flag
to `0` only for an intentional non-Graviton or non-EC2 diagnostic run.
The EC2 launcher exports both flags, and the launch/prelaunch checks reject
plans whose user-data would omit the metadata requirement.
They also require `NSRL_S3_URI` by default so `run.env`, logs, models, evals,
and promotion artifacts sync to `s3://.../pipelines/<run-name>/`; set
`NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=0` only for an intentional local diagnostic
run.

Before launching a real run, inspect the resolved stage plan:

```bash
node scripts/check-solomon-product-diagnostic.mjs
bash scripts/check-solomon-aws-product-plan.sh
bash scripts/check-solomon-aws-launch-plan.sh
bash scripts/check-solomon-aws-prelaunch-readiness.sh
scripts/aws/launch-solomon-product-run.sh --dry-run
```

The product diagnostic is the preferred local preflight because it ties the
corpus contract, checked-in held-out prompt retrieval proof, promoted native
directional smoke, symbolic-image encoder self-test, token-layout parity
self-test, held-out retrieval contract self-test, grounded-corpus
contract self-test, prior-smoke contract self-test, provenance self-tests,
generation-integrity guardrail fixture, generated sample-binding fixture,
denoise bridge, promotion bundle, AWS dry-run
plan, and EC2 launch-plan check
plus the prelaunch-readiness, release-proof wrapper, live-launch-readiness,
launch execute-guard, and completed-run artifact self-tests into one JSON
report. Use `--fast` for quick iteration; it skips the slower corpus, held-out
retrieval, and native smokes and keeps `full_product_proof` false.
After a real run finishes, prefer the post-run proof wrapper:
`scripts/aws/prove-solomon-product-run.sh --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME --launch-dir data/aws-launches/RUN_NAME --require-launch-dir`.
It fetches the run, checks the transferred artifacts, runs
`node scripts/check-solomon-product-diagnostic.mjs --aws-run-dir PATH --require-aws-run`,
then runs `node scripts/check-solomon-objective-coverage.mjs --require-release`.
It writes `objective-coverage.json` and `release-proof.json`. That diagnostic
path requires both the local no-spend proof and the synced Graviton artifact
proof, and the objective audit maps that proof back to the narrow bidirectional
Solomon product shape before release. The completed-run artifact proof now also
requires the synced `run.env` and `pipeline-complete.json` to agree on v2
`symbolic16` curriculum settings and map-reduce `auto-online-processors` CPU
scaling.

Dry run writes `run.env`, `plan.tsv`, empty `artifacts.tsv`, and per-stage
status/log files plus `pipeline-complete.json` without executing training or
syncing S3. The plan must end in the final `promotion-bundle-check` gate, which runs
`check-solomon-promotion-bundle.mjs` after the promotion manifest is written.
The product-plan self-test also runs the denoise-bridge fixture check, so CI
proves that cleanup fields, wrong latent target sources, forged bridge
signatures, flat output bytes, and weak denoised-output retrieval margins are
rejected before AWS launch.
The product-plan checker fails if the resolved plan stops being the v2 symbolic
attention curriculum with held-out prompt retrieval, identity inference,
grounded source corpus evidence, generative eval, denoise bridge, confidence
trace, symbolic channel, denoised-output retrieval identity, and promoted
small-profile gates enabled. It also requires nonzero generated 16x16 top-5
signature and
rendered-image retrieval top-1 floors, plus
`NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=none`,
`NSRL_SOLOMON_ATTENTION_HELDOUT_PROMPTS=data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl`,
`NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=72`, and
`NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=1`, plus
`NSRL_SOLOMON_V2_MIN_DIRECTION_TOP5_PER_MILLE=all=1` and
`NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=all=72`, so product generation cannot pass
with only sidecar files, zero generated-seal hits, a tiny native eval slice, a
missing native top-5 hit in any v2 task bucket or product direction group, or an
eval trace that never scored prompt/text/image/control targets. The checker resolves
that held-out prompt JSONL, records its FNV64 byte hash, and counts valid prompt
rows before training begins, so `none`, a missing file, or a tiny/stale prompt
corpus fails fast. Task eval also
checks `task_marker_integrity`, so v2 task labels must agree with the serialized
`BOS,TASK_*,...` token stream and recorded per-example token hashes, and
`task_modality_integrity`, so each bidirectional task keeps the expected
`PROMPT`/`IMAGE`/`TEXT` order in the serialized bytes. The task-eval self-test
now mutates hard-negative role coverage, marker, modality, and channel bytes to
prove those corruptions fail locally before AWS launch. The final quality report also checks
`image_channel_marker_integrity`, so required
symbolic image channels must appear as serialized marker bytes with 256-token
image-bin payloads, not only as manifest metadata; the confidence trace exposes
the same proof as `symbolic_image_tokens`. It also requires
`NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS=2` and
`NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS=8` so source/explanation
and image-to-attributes rows carry real source overlap. It also records
`require_name_source_explain=true`, so `explain` rows prove the primary
`Name -> source description` direction instead of a generic seal prompt, and
`require_description_source_image=true`, so `description-to-image` rows prove a
source-description-to-seal-token direction instead of a generic name prompt, and
`require_image_attribute_generic_prompt=true`, so `image-to-attributes` rows do
not leak the primary name through their prompt, plus
`NSRL_SOLOMON_V2_MAX_SOURCE_PLACEHOLDER_ROWS=0` so placeholder source prose
and `NSRL_SOLOMON_V2_MAX_ATTRIBUTE_GENERIC_RANK_ROWS=0` so generic attribute
rank labels cannot satisfy the grounding gate. The product diagnostic and
objective coverage also require raw source provenance on `image-to-text`
identity rows, tying reverse image classification back to the same source index
used for explanation rows, and require expected `source_query_kind` coverage on
each source-bound task direction. It also keeps
`NSRL_SOLOMON_V2_REQUIRE_IMAGE_CHANNEL_TOKEN_STATS=1` and
`NSRL_SOLOMON_V2_MIN_IMAGE_CHANNEL_DISTINCT_BINS=2`, so the Graviton product
plan rejects symbolic image channels that are declared, visually degenerate, or
collapsed to duplicate per-source payload hashes.
Hard-negative floors
default to `NSRL_SOLOMON_V2_MIN_MATCH_YES_TOP1=72`,
`NSRL_SOLOMON_V2_MIN_MATCH_NO_TOP1=72`,
`NSRL_SOLOMON_V2_MIN_MATCH_NO_IMAGE_TOP1=72`, and
`NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1=72`, making wrong-seal and
wrong-prompt/name directions explicit product evidence, plus
`NSRL_SOLOMON_V2_MIN_RETRIEVAL_MARGIN=1` so top-1 bindings need positive score
separation. The final quality report also compares `retrieval-head-eval.json`
against the same `examples.jsonl` and corpus token file declared by the v2
corpus contract, rejecting stale retrieval-head artifacts from an older corpus
even when their own sidecar metrics pass. The wrapper also runs a
negative self-test by default, mutating a copy of the plan to weaker settings
and requiring the checker to reject it.

The default stages are now
`dataset,denoiser,prior,generative-eval,attention-curriculum`: train the
bitmap denoiser, train/evaluate the latent prior, run held-out generation, then
train the v2 task-marked attention curriculum and write the unified
`quality-report.json`. The AWS wrapper sets the product attention defaults to
`NSRL_SOLOMON_ATTENTION_CORPUS_VERSION=v2`,
`NSRL_SOLOMON_ATTENTION_TEXT_TOKEN_PROFILE=chunked`,
`NSRL_SOLOMON_ATTENTION_IMAGE_TOKEN_PROFILE=symbolic16`,
`NSRL_SOLOMON_ATTENTION_SEQ_LEN=512`, and
`NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1`. It also passes
`NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=1`,
`NSRL_SOLOMON_V2_MIN_D_MODEL=128`, `NSRL_SOLOMON_V2_MIN_HEADS=2`,
`NSRL_SOLOMON_V2_MIN_HIDDEN_DIM=256`,
`NSRL_SOLOMON_V2_MIN_TRANSFORMER_LAYERS=2`, and
`NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN=384` into the final quality report. It also sets
`NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce` and
`NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0`; the dry-run product-plan checker
records those in `run.env` with `attention_cpu_scaling_policy`,
`attention_map_reduce_auto_workers`, `processor_count`, and
`attention_effective_map_reduce_workers`. For the default product contract,
`0-auto` workers must resolve to the visible online CPU count, and plans that
do not use Graviton CPU auto-scaling for the product attention stage are
rejected. It also sets
`NSRL_SOLOMON_ATTENTION_EVAL_MAX_EXAMPLES=none`,
`NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=72`, and
`NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=1` for product attention stages.
It also sets `NSRL_SOLOMON_V2_MIN_PHASE_TARGETS=all=72`, requiring the native
eval to score special/control, prompt, text, and image targets at product
breadth, plus source-grounding floors of `2` source tokens and `8` attribute/source
tokens, zero generic attribute-rank rows, and 72/72 floors for positive match, combined no-match, wrong-seal, and
wrong-prompt/name hard-negative rows.
For the denoise bridge, the product plan records
`attention_denoise_min_unique_targets=2` and requires the final quality report
to expose the same distinct-target floor before promotion.
Completed-run artifact validation reopens `quality-report.json` and checks
those native task and aggregate phase metrics directly, so a synced run cannot
pass by advertising `all=72` floors while carrying only a capped local
directional-smoke eval.

For a cheaper bitmap-only bootstrap, narrow the stage list explicitly:

```bash
NSRL_SOLOMON_AWS_STAGES=dataset,denoiser,prior,generative-eval \
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-end-to-end.sh
```

For a targeted v2 bidirectional binding rerun, invoke the attention stages
directly and add any extra ratchets:

```bash
NSRL_SOLOMON_AWS_STAGES=attention,attention-curriculum \
NSRL_SOLOMON_ATTENTION_DENOISER_MODEL=data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch \
NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_INK_RANGE=1 \
NSRL_SOLOMON_V2_MIN_TOTAL_TOP5_PER_MILLE=500 \
NSRL_SOLOMON_V2_MIN_TASK_TARGETS=all=144 \
NSRL_SOLOMON_V2_MIN_TASK_TOP5_PER_MILLE=all=10 \
NSRL_SOLOMON_V2_MIN_SOURCE_OVERLAP_TOKENS=3 \
NSRL_SOLOMON_V2_MIN_ATTRIBUTE_SOURCE_OVERLAP_TOKENS=10 \
NSRL_SOLOMON_V2_MIN_MATCH_NO_PROMPT_TOP1=144 \
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-end-to-end.sh
```

The denoise bridge records `min_output_signature_distance` in
`denoise-bridge.json`; v2 bridge runs with a retrieval head also record
`output_image_to_text_identification` and
`min_output_retrieval_image_margin` for the downsampled 128x128 output. After a
measuring run, set
`NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_SIGNATURE_DISTANCE` to make that
plan/output alignment a promotion gate. Product-plan preflight also requires
`NSRL_SOLOMON_ATTENTION_DENOISE_MAX_OUTPUT_RETRIEVAL_RANK=1` and
`NSRL_SOLOMON_ATTENTION_DENOISE_MIN_OUTPUT_RETRIEVAL_MARGIN=1`, so the denoised
endpoint must still identify as the generated spirit with a positive retrieval
margin. The bridge artifact recomputes the retrieval head `model_hash`, records
`retrieval_head_hash_verified`, and fails stale or forged scorer JSON before
that output identity can count. It also records `trace_integrity_ok`, and the
final quality report fails it if the denoiser trace exposes target-pixel,
oracle, guidance, postprocess, or display-cleanup side channels. It also records
`denoise_model_hash` from the sampler `trace.model`; the final report recomputes
that file hash and requires every bridge pair to use one consistent denoiser
model. The bridge and final report also record distinct expected target
coverage (`expected_unique_targets` and `unique_expected_spirit_ids`). Product
Graviton plans default `NSRL_SOLOMON_ATTENTION_DENOISE_MIN_UNIQUE_TARGETS=2`
and pass it through `NSRL_SOLOMON_V2_MIN_DENOISE_BRIDGE_UNIQUE_TARGETS`; raise
that floor as the bridge expands toward all 72 Solomon targets. The final report
also recomputes the denoised output's plan distance, ink range, retrieval
identity, retrieval margin, and target coverage from raw denoiser bytes before
trusting the bridge sidecar. Promotion bundles require the same
denoiser-model, sample-binding, output-byte, retrieval-head hash, and
confidence-trace generation-bridge provenance, so a bundle cannot pass with a
forged or loosely attached denoise sidecar. When the bridge is required, AWS
stages also default
`NSRL_SOLOMON_V2_REQUIRE_DENOISE_OUTPUT_IDENTITY=1`, which makes missing
`output_image_to_text_identification` or a missing positive retrieval margin a
quality-report failure.
V2 attention smokes also require the richer symbolic image-token contract by
default: `NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_PROFILE=symbolic16` and
`NSRL_SOLOMON_V2_REQUIRE_IMAGE_TOKEN_CHANNELS=ink,edge,component,radial,direction`.
The AWS v2 curriculum default is
`identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind`,
so description-conditioned image planning and source/explanation binding get
their own update passes before the final joint pass. The final `native-bind`
stage defaults to `NSRL_SOLOMON_ATTENTION_V2_NATIVE_BIND_EPOCHS=2`, and the
product-plan and completed-run artifact gates reject weaker product evidence.
The `image` stage is now
checked with `stage_evidence`: it must contain both text/description-to-image
plan rows and image-to-text classification rows over all 72 spirits. Each
filtered stage also carries `task_marker_integrity`, so stage task labels must
match the serialized task markers and row token hashes before the train trace is
trusted. Symbolic stages also carry `image_channel_marker_integrity`, proving
the filtered token file still contains the required image-channel markers and
payloads.
The AWS `attention-curriculum` stage requires `curriculum-stages.json` by
default. In the end-to-end runner, if the `denoiser` stage is present, or
`NSRL_SOLOMON_DENOISE_MODEL` is explicitly set, attention stages default
`NSRL_SOLOMON_ATTENTION_DENOISER_MODEL` to that sampler model and require
`denoise-bridge.json` through `NSRL_SOLOMON_V2_REQUIRE_DENOISE_BRIDGE=1`.
Set `NSRL_SOLOMON_ATTENTION_DENOISER_MODEL=none` for a targeted attention-only
rerun that intentionally skips the 128x128 bridge.
AWS attention stages also default to
`NSRL_SOLOMON_V2_REQUIRE_CONFIDENCE_TRACE=1`, so prompt retrieval, image
retrieval, hard negatives, generated sample agreement, source evidence, and any
required denoise bridge must agree in `quality-report.json`. The same final
report now records `corpus_contract`, proving the v2 `symbolic16` image profile
and ink/edge/component/radial/direction channels from manifest and example rows,
all required task buckets and hard-negative roles across the 72 spirits,
serialized task-marker/token-hash integrity from the corpus token file,
grounded-corpus examples provenance from the same promoted examples file,
row-level source hashes/excerpts for grounded explanation records, and
identity/grounded source-text index hashes from the promoted manifest
`source_text_index`,
curriculum-stage source examples/token provenance from the same promoted
corpus files, curriculum-stage task-marker integrity, plus
`require_curriculum_stage_names` for the exact ordered
curriculum required by `NSRL_SOLOMON_ATTENTION_V2_CURRICULUM_REQUIRED_STAGES`.
With v2 retrieval head evidence, the denoise bridge agreement includes
image-to-text identity of the denoised 128x128 endpoint, not only the generated
16x16 plan, and carries the bridge trace scan as
`generation_bridge.trace_integrity_ok`. It also records
`generation_bridge.sample_binding_provenance` and
`generation_bridge.output_provenance`, proving the bridge
`attention_plan` came from a sample plan already checked by
`sample-binding.json` and that the raw denoised bytes carry the same expected
spirit identity through the denoised output details. Promotion bundle checks now
make those generation-bridge fields mandatory, including denoiser-model hash
provenance, bridge-result/sample-plan matching, recomputed raw-output
provenance, and retrieval-head hash verification. Current sample-binding traces
also require generated
sample text to retrieve the expected spirit and agree with the generated image,
so a correct prompt/image pair cannot hide empty or wrong generated text. The
confidence trace records the
identity requirement as `generation_bridge.output_identity_required`, and also
records forward `text-to-image`/`description-to-image` native task metrics
separately from reverse image-to-text retrieval. When the
v2 corpus is checked, primary-name, alias, and seal-ID bindings must each exist
as both `identify` and `text-to-image` rows before the run can proceed. When the
`generative-eval` stage is part of the same pipeline, its run directory is
passed through `NSRL_SOLOMON_V2_GENERATIVE_EVAL` and required by default with
`NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_EVAL=1`, adding prompt-to-latent-to-rendered
seal evidence to the same report. Ratchet it with
`NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_16_PER_MILLE`,
`NSRL_SOLOMON_V2_MIN_GENERATED_TOP5_PX_PER_MILLE`, and
`NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS`; product AWS defaults the generated
prompt-row floor and `NSRL_SOLOMON_GENERATIVE_EVAL_LIMIT` to `72`, with
`NSRL_SOLOMON_GENERATIVE_EVAL_PERMILLE=200` as the fallback hash split. For
product `partition=eval` runs, the selector prefers non-canonical prompt rows
whose tier contains `holdout` or `novel`, so generated samples use the same
held-out prompt JSONL as retrieval evidence while still covering all 72 spirits.
Use
`NSRL_SOLOMON_V2_MIN_LATENT_TOP5_PER_MILLE` to ratchet the latent path; product
AWS also defaults
`NSRL_SOLOMON_V2_MAX_GENERATED_MEAN_TARGET_DISTANCE_16_Q8=7000000` so generated
held-out seals must stay near their target signatures by actual 16x16 distance,
not only by top-k rank. The quality report checks the
generative eval `config.json` and `samples.tsv` sidecars, so a product-generation
artifact only passes when it was sampled from the held-out `eval` partition with
`decoded-latent` sampler targets. The config must carry `promptsHash`,
`promptRows`, `selectedPromptRows`, `selectedPromptEligibleRows`,
`selectedPromptUniqueTargets`, `selectedPromptEligibleUniqueTargets`, and
`selectedPromptHash`; the report
recomputes them from the referenced prompt JSONL, cross-checks the prompt hash
against retrieval-head held-out prompt evidence when available, and verifies
each model's `samples.tsv` prompt set matches the selected prompts with enough
unique spirit targets for `NSRL_SOLOMON_V2_MIN_GENERATED_PROMPT_ROWS`. Promotion
bundles also require those selected eligible-row and unique-target fields to be
present and matching, so an AWS bundle is self-evidencing without re-running the
sampler. The same
sidecars must carry `latentModelProvenance` in `config.json` and
`latent_model_hash` in `summary.tsv`; the report recomputes those hashes from
the latent-prior model files and requires every sample `trace.json.latent_model`
to resolve to the same hash before accepting product evidence. The report also
recomputes `samplerModelHash` from the bitmap sampler model and requires every
sample `trace.json.model` to resolve to the same `NSRLTCH` renderer. It opens
every held-out sample `trace.json`, requires clean decoded-latent
provenance and raw generated sample bytes at the sampler-written
`samples.ink${image_size}.u8` path inside that sample's `out_dir`, and fails
target-pixel, oracle, guidance, postprocess, or display-cleanup side channels;
the runner now enforces that same contract before writing scored sidecars, and
the confidence trace records the report recheck as
`product_generation.trace_integrity_ok`. If an existing retrieval head is
available, set `NSRL_SOLOMON_GENERATIVE_EVAL_RETRIEVAL_HEAD=PATH` so generative
eval also classifies rendered held-out bitmaps by image-to-text identity and
records `retrievalHeadModelHash`; retrieval-based product-generation gates
require that hash to match the final retrieval-head eval, and the final report
recomputes every generated retrieval rank/identity/margin from raw sample bytes.
In the normal
all-stage order, attention creates `retrieval-head.json` after generative eval;
v2 attention smokes then post-score the existing generative eval sidecars before
the quality report, refusing traces whose `raw_samples` does not resolve to the
sampler-written bytes inside the sample `out_dir` and requiring clean
`decoded-latent` trace provenance. Ratchet that with
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP1_PER_MILLE` or
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_TOP5_PER_MILLE`; product AWS runs
default both rendered-image retrieval floors to `1000` and
`NSRL_SOLOMON_V2_MIN_GENERATED_RETRIEVAL_MARGIN=1`. Product runs also
default `NSRL_SOLOMON_V2_REQUIRE_GENERATIVE_OUTPUT_IDENTITY=1`, so the matching
product-floor model must have every rendered held-out 128x128 sample identify
top-1 with a positive retrieval margin after report-side recomputation.
The product diagnostic also carries the actual selected held-out generation
coverage as `generated_prompt_rows` and `generated_unique_targets`; objective
coverage and release-candidate checks require both to cover all 72 spirits
before the Graviton handoff is green.

Promoted architecture runs can also require the profile fields that
`quality-report.json` records:

```bash
NSRL_SOLOMON_ATTENTION_SEQ_LEN=512 \
NSRL_SOLOMON_V2_REQUIRE_ARCHITECTURE_PROFILE=1 \
NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1 \
NSRL_SOLOMON_V2_MIN_D_MODEL=128 \
NSRL_SOLOMON_V2_MIN_HEADS=2 \
NSRL_SOLOMON_V2_MIN_CONTEXT_SEQ_LEN=384 \
NSRL_SOLOMON_AWS_STAGES=attention \
NSRL_SOLOMON_ATTENTION_CORPUS_VERSION=v2 \
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-end-to-end.sh
```

`NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=1` turns the architecture check
from a minimum-ratchet smoke into the promotion gate: `d_model=128`, two heads,
`head_dim=64`, hidden dim 256-512, 2-4 transformer layers, and context length
384-768. The `d_model=128, heads=2` target is intentional: the current base-2
softmax attention path requires a power-of-four per-head dimension, so
`128 / 2 = 64` is valid while `64 / 2 = 32` is rejected by
`quality-report.json`. The same report loads `retrieval-head.json`, verifies its
schema and model hash against `retrieval-head-eval.json`, and requires both
text and image retrieval heads before promoted class-head readiness can pass.
It also records `retrieval_head_eval.corpus_provenance`, matching the retrieval
eval's corpus paths and optional byte hashes to the final promoted corpus. When
held-out prompts are required or evaluated, the report also records
`retrieval_head_eval.heldout_prompt_provenance` and verifies the prompt JSONL
path, `prompts_hash`, and valid prompt-row count before accepting held-out
generalization evidence. The AWS runner records this as a final
`promotion-bundle-check` stage after the promotion manifest is written, so the
dry-run plan and the real Graviton run both make the product gate explicit. The
promotion-bundle checker then reads the final
`confidence_trace` itself and requires complete known-prompt, held-out-prompt,
identity-binding, image-to-text, per-image-task, forward image-plan, match
yes/no, wrong-image, wrong-prompt, and generated sample agreement summaries. The
raw retrieval-head eval must also prove the forward `text-to-image` image-plan
bucket across 576 rows plus 72-row `description-to-image` and reverse image
task buckets, so the bundle cannot pass on a strong label without the
bidirectional binding spine under it. The same promotion check also requires
`confidence_trace.symbolic_image_tokens` to prove the `symbolic16`
ink/edge/component/radial/direction channel markers in the promoted corpus and
curriculum-stage evidence, and it requires `confidence_trace.source_grounding`
to prove source text for text/image/sample queries, generated text/source
agreement, prompt/generated-text retrieval margins, grounded source-task
coverage, and grounded `image-to-attributes` coverage.
Generated sample binding, identity inference, and denoise bridge artifacts also
carry retrieval-head hash provenance when produced by current scripts, and the
quality report rejects any downstream hash that differs from the retrieval eval.

Train the text-conditioned denoiser:

```bash
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-text-denoiser-train.sh
```

Train/evaluate the latent prior and sample a fixed panel:

```bash
NSRL_S3_URI=s3://BUCKET/PREFIX \
  scripts/aws/run-solomon-prior-smoke.sh
```

Bake a warm Graviton AMI for the Solomon binaries:

```bash
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_ARTIFACT_S3_URI=s3://BUCKET/PREFIX/artifacts/nsrl-working-tree.tar.gz \
  scripts/aws/bake-training-ami.sh
```

Verify the current launch environment, then launch the checked product pipeline
on the baked AMI:

```bash
NSRL_AMI_ID=ami-... \
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_ARTIFACT_S3_URI=s3://BUCKET/PREFIX/artifacts/nsrl-working-tree.tar.gz \
  scripts/check-solomon-aws-live-launch-readiness.sh

NSRL_AMI_ID=ami-... \
NSRL_S3_URI=s3://BUCKET/PREFIX \
NSRL_ARTIFACT_S3_URI=s3://BUCKET/PREFIX/artifacts/nsrl-working-tree.tar.gz \
  scripts/aws/launch-solomon-product-run.sh --execute
```

The launcher defaults to dry-run and writes `launch.json` plus `user-data.sh`;
`check-solomon-aws-launch-plan.sh` verifies that the launch uses a Graviton
instance family, S3-backed repo artifact, the product stage list, and
map-reduce `0-auto` CPU scaling before any EC2 instance is started.
`check-solomon-aws-prelaunch-readiness.sh` is the stricter no-spend launch gate:
it requires a real-looking `NSRL_AMI_ID`, IAM instance profile, Graviton
instance type, S3 artifact paths, IMDSv2, the exact EC2 tag specification, and
exact optional subnet/security group values in the recorded launch command. Execute mode reruns
that same prelaunch readiness checker with `--allow-execute-plan`, but first
requires explicit `NSRL_S3_URI` and `NSRL_ARTIFACT_S3_URI` values instead of
launcher defaults. It writes `prelaunch-readiness-check.json` beside
`launch.json`, and stops before `aws ec2 run-instances` if the check is not
green.
The launch manifest also stores `post_run_proof_command`, bound to the same
S3 pipeline URI and launch directory, and both launch/prelaunch checkers reject
manifests whose post-run proof command points somewhere else. They also require
the manifest `user_data` path and the EC2 `--user-data file://...` command
argument to point at the same hashed `user-data.sh`, and require critical EC2
command flag values such as `--image-id`, `--instance-type`, IAM profile, IMDSv2
metadata options, shutdown behavior, and `--output json` to match the manifest.
In execute mode, the launcher captures the EC2 `run-instances` JSON response as
`launch-result.json`, records its SHA-256 in `launch.json`, and uses that
response to populate the launched instance id.

After the instance finishes, fetch and verify the completed product evidence:

```bash
scripts/aws/prove-solomon-product-run.sh \
  --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME \
  --launch-dir data/aws-launches/RUN_NAME \
  --require-launch-dir
```

The proof wrapper syncs the S3 prefix to
`/tmp/nsrl-solomon-pipelines/RUN_NAME`, writes `fetch-report.json` and
`aws-run-artifacts-check.json`, reruns the product diagnostic with
`--require-aws-run`, runs objective coverage with `--require-release`, and
writes `release-proof.json`. `--require-launch-dir` makes the proof fail unless
the executed launch evidence is supplied. When `--launch-dir` is present, it
cross-checks the launch manifest's run name and S3 pipeline URI against the
fetched run, and requires `dry_run=false`, a nonempty EC2 `instance_id`, and a
matching captured `launch-result.json` response. The wrapper also rechecks the
`user-data.sh` hash and command binding, the `launch-result.json` hash, and the
launch-result instance id, AMI id, instance type, requested subnet, and requested
security groups against `launch.json`. Use
`--skip-sync --out-dir PATH` to verify an
already-synced bundle; the fetch report still rejects the bundle if its recorded
run name or S3 pipeline URI differs from the requested prefix. The underlying
post-run checker rejects dry-run metadata,
non-Graviton runner metadata, missing EC2 IMDS provenance, missing S3
provenance, missing stage statuses, failed promotion checks, and a missing
`pipeline-complete.json`. It also reads the synced `quality-report.json` and
requires generated product evidence to record/recompute selected held-out rows
and unique targets covering the 72-spirit product floor. By default it reruns
`check-solomon-promotion-bundle.mjs` against the synced `promotion.tsv`, so the
deep `quality-report.json` gates are rechecked after artifact transfer instead
of trusting a stale stored `promotion-bundle-check.json`. The release-candidate
handoff also requires the completed-run artifact self-test case list, including
the generated-product coverage rejection, so stale post-run verification cannot
drop out of the operator proof.

## Artifact Shape

```text
s3://BUCKET/PREFIX/
  pipelines/<run-name>/
    run.env
    plan.tsv
    artifacts.tsv
    promotion.tsv
    promotion-bundle-check.json
    pipeline-complete.json
    logs/
    denoise-dataset/
    text-denoiser/
      model.nsrltch
      trace.json
    prior/
      latent/model.nsrllat
      latent/trace.json
      eval-ledger.jsonl
      partition.tsv
      prior-gate/
      samples/
      manifest.tsv
      smoke-check.json
    generative-eval/current/
      config.json
      samples.tsv
      summary.tsv
    multimodal/                # optional
    attention/                 # optional single-pass run; v2 writes retrieval-head and quality gates
      attention-eval.json
      retrieval-head.json
      retrieval-head-eval.json
      grounded-corpus.json
      sample-binding.json
      identity-inference.json
      generation-integrity.json
      denoise-bridge.json        # optional when the attention denoise bridge is enabled
      denoise-generation-integrity.json
      quality-report.json
    attention-curriculum/      # default product run; same v2 evidence shape
      attention-eval.json
      retrieval-head.json
      retrieval-head-eval.json
      grounded-corpus.json
      prior-sample-binding.json
      identity-inference.json
      generation-integrity.json
      quality-report.json
      denoise-bridge.json        # optional when the attention denoise bridge is enabled
      denoise-generation-integrity.json
  text-denoiser/<run-or-output-dir>/
    model.nsrltch
    trace.jsonl
    preview.*
  runs/<run-name>/
    latent/model.nsrllat
    latent/trace.json
    eval-ledger.jsonl
    partition.tsv
    prior-gate/
    samples/
    manifest.tsv
    smoke-check.json
```

S3 is the durable store. Local `data/` paths on an instance are working copies.

## Instance Notes

- Use Graviton for the native Linux lane.
- `nsrl-bitmap-multichannel-denoise` uses threaded deterministic i64 gradient
  accumulation.
- `run-solomon-end-to-end.sh` is the normal cloud entrypoint; use the lower
  level scripts for targeted reruns after a failed or weak stage.
- `run-solomon-prior-smoke.sh` builds only the latent trainer, evaluator, and
  sampler.
- Use a baked AMI when iterating; it removes cold Rust/toolchain build overhead.
