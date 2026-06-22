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
