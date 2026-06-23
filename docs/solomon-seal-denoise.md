# Solomon Seal Denoise & Sampling

An integer-native generative image pipeline built on the same `no_std`, zero-float
arithmetic contract as the NSRL language models. It learns the visual grammar of
the seal plates from the *Key of Solomon* / Lesser Key (Goetia) and samples new
seals in pure `i8`/`i16` arithmetic — no floating-point at any stage, enforced by
`scripts/check-no-floats.sh`.

This is the 2D companion to the language experiments: the same i8 weights, Q15
activations, i64 accumulation, and shift-based learning rates, applied to
128×128 single-channel "ink" bitmaps instead of token streams.

## Source

Project Gutenberg **PG72679** (public domain). The scanned seal plates are
sliced into per-cell `seal-grid-cell` bitmaps. The sliced source and its
manifest live under (git-ignored, regenerate locally):

```
data/raw/key-solomon-goetia-pg72679/
data/processed/key-solomon-goetia-bitmaps-pg72679/slices/manifest.json
```

## Pipeline

```
PG72679 seal plates
   │  slice → 128×128 seal-grid cells (manifest.json)
   ▼
nsrl-build-solomon-bitmap-denoise-dataset   →  clean/ + pairs/ + rows/ (train/eval .ink128.u8)
   │  clean targets + 8 corruption kinds × 8 timesteps
   ▼
nsrl-bitmap-three-layer-denoise             →  baseline-three-layer-conv/model.nsrlcv3
   │  3-layer integer conv denoiser (NSRLCV3)
   ▼
nsrl-bitmap-sample                          →  samples/ (preview grid PNG + .ink128.u8)
       seal-prior init → iterative integer denoise passes → diversity-ranked selection
```

### Binaries (all in `nsrl-train`, `#![deny(unsafe_code)]`)

| Binary | Role | Model magic |
|---|---|---|
| `nsrl-build-solomon-bitmap-denoise-dataset` | Slice manifest → corrupted/clean denoise pairs | — |
| `nsrl-bitmap-denoise` | Local-table (mask LUT) denoiser baseline | `NSRLBM1` |
| `nsrl-bitmap-conv-denoise` | Single residual-conv denoiser | `NSRLCV1` |
| `nsrl-bitmap-three-layer-denoise` | 3-layer integer conv denoiser (primary) | `NSRLCV3` |
| `nsrl-bitmap-multichannel-denoise` | 8 fixed hidden kernels (edge/structure) | `NSRLMCH` |
| `nsrl-bitmap-sample` | Generative sampler over a trained denoiser | reads CV3/MCH/TCH |
| `nsrl-solomon-latent-train` | Shared text/bitmap latent bridge over the 72 spirit rows | `NSRLLAT1` |

The eight corruption kinds (`pixel-dropout`, `salt-pepper`, `block-mask`,
`stroke-thin`, `stroke-thicken`, `line-drop`, `mixed-noise`, `coarse-erase`) are
applied across `timesteps` to teach the denoiser a progressive restoration path;
the sampler then runs that path forward from a seal-shaped prior.

The sampler supports several init modes (`--init`): `noise`, `seal-prior`
(default — concentric-ring prior so samples start inside the seal manifold),
`learned-prior`, `patch-prior`, `coordinate-prior`. Generation draws
`samples × candidate-multiplier` candidates, runs `passes` integer denoise
sweeps, then keeps `samples` ranked by a `diversity-weight` term for spread.

## Learned Text/Bitmap Latent

The text-conditioned sampler originally used a catalog bridge:

```
prompt -> best matching spirit row -> stored 8x8 seal signature -> NSRLTCH sampler
```

`nsrl-solomon-latent-train` adds a learned bottleneck bridge:

```
spirit text/name -> text encoder -> shared latent -> signature decoder
seal signature   -> bitmap encoder -> shared latent
```

The first `NSRLLAT1` model keeps the bitmap encoder as a deterministic
integer projection from each 8x8 ink signature, then learns a hashed text
encoder and a latent-to-signature decoder. The sampler can consume the learned
target field directly:

```sh
cargo run --release -p nsrl-train --bin nsrl-solomon-latent-train

cargo run --release -p nsrl-train --bin nsrl-bitmap-sample -- \
  --model data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch \
  --latent-model data/processed/key-solomon-goetia-latent-v1/model.nsrllat \
  --prompt "hidden geometry and rushing waters" \
  --samples 16 --candidate-multiplier 4
```

The latent trace reports text-to-bitmap retrieval over the 72 paired rows,
plus text/image signature reconstruction error in Q8. The sampler trace records
`latent_model`, `latent_prompt`, `latent_dim`, and `latent_text_features` when
this path is active.

### Growing held-out prompt eval

The prompt eval uses stable hash buckets instead of insertion order, so new
prompts can be appended without reshuffling any existing train/eval assignment:

```sh
node scripts/build-solomon-prompt-corpus.mjs

cargo run --release -p nsrl-train --bin nsrl-solomon-latent-train -- \
  --prompts data/processed/key-solomon-goetia-latent-v1/prompts.jsonl

cargo run --release -p nsrl-train --bin nsrl-solomon-eval
```

`prompts.jsonl` uses schema `nsrl.solomon_prompt.v1` with stable `bucket`,
`tier`, `cluster`, and `prompt_hash` fields. `gold.tsv` pins exact prompt
hashes; gold prompts never enter training even when their bucket would otherwise
be train. The default split seed is `solomon-prompt-split-v1`, and the default
eval bucket is `180` permille to match the bitmap dataset builder.

`nsrl-solomon-latent-train --prompts` trains only on prompt rows outside both
the eval bucket and the frozen gold set. `nsrl-solomon-eval` writes a partition
TSV, ranks held-out prompts against unique seal targets, reports metrics per
tier, and appends `eval-ledger.jsonl` rows with the model hash, prompt set
version, train prompt count, top1/top5, and Q8 MAE.

To grow the prompt pool from source-grounded variants and run the first scaling
curve:

```sh
node scripts/build-solomon-grounded-corpus.mjs --variants-per-row 16

node scripts/append-solomon-prompts.mjs \
  --from-grounded data/processed/key-solomon-goetia-grounded-corpus-v1/grounded-text-signatures.tsv \
  --out data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl

node scripts/run-solomon-eval-scaling-curve.mjs \
  --prompts data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl \
  --sizes 288,576,1152,1425 \
  --latent-dims 32,64,128 \
  --epochs 12 \
  --report-out docs/solomon-eval-scaling-curve.tsv
```

The first checked curve is [docs/solomon-eval-scaling-curve.tsv](solomon-eval-scaling-curve.tsv).
It fixes 512 text features and 12 epochs, then sweeps prompt prefixes and
latent widths `32,64,128`. In that short-budget sweep, held-out eval top1 peaks
at `200` per mille with 352 train prompts and 32 latent channels. Novel-vocab
top1 peaks at the same point (`208` per mille), while cluster-holdout top1 peaks
at 352 train prompts and 128 latent channels (`195` per mille). The larger
prompt pools underperform at 12 epochs, so the next curve should sweep epochs
or learning shifts at 822 and 1040 train prompts rather than claiming monotonic
data scaling from this first pass.

Improved checkpoints can be announced through the existing X/Twitter Lambda
without adding another credential path. By default this only prepares a dry-run
payload:

```sh
node scripts/post-solomon-improved-checkpoint.mjs \
  --curve docs/solomon-eval-scaling-curve.tsv
```

To have the scaling runner check after each completed eval row, add
`--post-improvements`. Use `--post-invoke-lambda` for a Lambda dry run, or
`--post-live` to publish and update the git-ignored checkpoint post state:

```sh
node scripts/run-solomon-eval-scaling-curve.mjs \
  --prompts data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl \
  --sizes 288,576,1152,1425 \
  --latent-dims 32,64,128 \
  --epochs 12 \
  --report-out docs/solomon-eval-scaling-curve.tsv \
  --post-live
```

The default posted metric is `eval_top1_per_mille`; pass `--post-metric
novel_top1_per_mille` when the public criterion should be novel-vocab
generalization instead.

### Grounded synthetic text variants

`scripts/build-solomon-grounded-corpus.mjs` expands each of the 72 demon rows
into balanced, source-grounded text variants while keeping the same seal slice
and 8x8 signature target. It uses the existing Goetia TSV and, when present,
additional OCR sources under `data/raw/`:

```sh
curl -L https://archive.org/download/dictionnaireinfe00coll_1/dictionnaireinfe00coll_1_djvu.txt \
  -o data/raw/dictionnaire-infernal-1863-djvu.txt
curl -L https://archive.org/download/discoveryofwitch00scot/discoveryofwitch00scot_djvu.txt \
  -o data/raw/scot-discovery-witchcraft-djvu.txt

node scripts/build-solomon-grounded-corpus.mjs --variants-per-row 32

cargo run --release -p nsrl-train --bin nsrl-solomon-latent-train -- \
  --text-index data/processed/key-solomon-goetia-grounded-corpus-v1/grounded-text-signatures.tsv \
  --out-dir data/processed/key-solomon-goetia-latent-grounded-v1
```

The expanded TSV keeps the original first nine text-index columns for backward
compatibility, then appends variant provenance (`variant_id`, `source_lanes`,
`prompt_kind`, `support_terms`, `source_urls`). The latent trainer treats these
as many text examples mapped to 72 unique image targets, so retrieval is ranked
against unique seals rather than duplicate text variants.

## Reproduce

```sh
scripts/run-solomon-seal-sample.sh
```

This builds the dataset, trains the three-layer denoiser, and emits a 64-seal
preview grid under
`data/processed/key-solomon-goetia-denoise-v1/baseline-three-layer-conv/samples/`.
It is deterministic: same seeds in, same seals out.

### Manual invocation

```sh
# 1. Build the denoise dataset (128×128, 8 corruptions × 8 timesteps)
cargo run --release -p nsrl-train --bin nsrl-build-solomon-bitmap-denoise-dataset

# 2. Train the 3-layer integer conv denoiser
cargo run --release -p nsrl-train --bin nsrl-bitmap-three-layer-denoise -- --epochs 8

# 3. Sample 64 new seals from the seal-prior
cargo run --release -p nsrl-train --bin nsrl-bitmap-sample -- \
  --samples 64 --init seal-prior --candidate-multiplier 4 --diversity-weight 1
```

## Defaults

- **Image:** 128×128, single channel, `u8` ink values (`.ink128.u8`).
- **Dataset:** kinds `seal-grid-cell`, 8 corruptions/image, 8 timesteps,
  eval split 180‰, seed `solomon-denoise-v1`.
- **Three-layer denoiser:** 3 layers, 8 epochs, `output-shift 12`,
  `learning-shift 31`, `bias-learning-shift 30`, `max-weight-delta 4`,
  `max-bias-delta 12`.
- **Sampler:** 64 samples, `seal-prior` init, 8 passes, seed
  `solomon-sampler-v1`.
- **Latent bridge:** 64 latent channels, 512 hashed text features, 120 epochs,
  model `data/processed/key-solomon-goetia-latent-v1/model.nsrllat`.

## Why it matters

A perplexity curve *argues* that integer-native training learns. A grid of
legible seals *shows* it — and shows the same i8/Q15 runtime generalizing from
1D token prediction to 2D structured image generation, with zero floating-point
operations anywhere in train or sample.

Traces use schemas `nsrl.bitmap_denoise_three_layer_conv_trace.v1` and
`nsrl.bitmap_sampler_trace.v1`.
