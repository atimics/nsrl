# Solomon Bitmap Sampler Status

Updated 2026-06-23.

The previous Solomon bitmap sampling notes mixed real model behavior with procedural starts, bitmap-derived assistance, and post-generation cleanup. Those paths have been removed from the sampler and bot surface. Treat earlier visual wins from this document as historical debugging artifacts, not evidence that the generator learned full seal structure.

## Current Contract

- Sampling starts from sparse deterministic noise.
- Pixel generation is performed by the trained sampler model only.
- Text or latent conditioning may provide target signatures to the model, but generation no longer reads target bitmap files as pixel guidance.
- The bot renders the raw generated ink sample at source resolution.
- Evaluation scripts may compare generated samples to held-out raw targets, but they no longer pass guidance or cleanup options into the sampler.

## Primary Commands

Build/train/sample the baseline denoise path:

```bash
scripts/run-solomon-seal-sample.sh
```

Run the text-to-bitmap generative eval:

```bash
node scripts/run-solomon-generative-eval.mjs
```

Run the small coherence panel:

```bash
scripts/run-solomon-coherence-panel.sh
```

## What Counts Now

The next valid result needs to show raw generated samples plus held-out metrics from the cleaned-up pipeline. If the samples are weak, that is the result: tune the actual model or dataset, not a display-time helper.
