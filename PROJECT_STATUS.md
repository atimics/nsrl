# Project Status

This repo's canonical status surface is:

```bash
node scripts/nsrl-status.mjs
```

Use JSON when another tool needs to consume the same truth surface:

```bash
node scripts/nsrl-status.mjs --json
```

Run fresh local checks when you want evidence instead of artifact inspection:

```bash
node scripts/nsrl-status.mjs --run-hygiene
node scripts/nsrl-status.mjs --refresh-fast-diagnostic
```

## Current Read

The substrate contract `integer-transformer-proof-v1` has a promoted **combined**
checkpoint. [The frozen record](benchmarks/integer-transformer-proof-v1/promoted-candidate.json)
binds the checkpoint, passing proof matrix, health evidence and 5,896 held-out
targets. Verify its exact bytes with
`node scripts/freeze-integer-transformer-proof-candidate.mjs --check`.
The promoted profile includes fitted suffix memory. On the already opened
[component ablation](benchmarks/integer-transformer-proof-v1/component-ablation.json),
combined and suffix-memory-only each made 2,482 mistakes; transformer-only
made 5,094. Transformer logits reduced probability error when added to suffix
memory without reducing mistakes on that fixture. The ablation is diagnostic,
not a separate promotion test. See [the proof contract](docs/integer-transformer-proof-v1.md)
for baselines and exact decision rules, and the [fresh-document comparison](https://github.com/cenetex/ilXyr/blob/main/experiments/research-step-51/REPORT.md)
for a distinct Brier-score confidence result whose metric is not interchangeable
with the substrate proof's Q15 probability error.

The **Solomon multimodal product** remains on separate gates. The July 10, 2026
snapshot measured 371 per mille on `NSRL-MME v0` against its 700 target, with
quality and generated-output checks still red at that time. That historical
measurement is not a current status reading. Run `node scripts/nsrl-status.mjs`
for the current product blockers and the independently verified substrate state.
Local artifacts and product checks do not themselves establish product readiness.

## Headline Eval

The number we are chasing is:

```text
NSRL-MME v0 headline_score_per_mille
```

It is the floor across model-native multimodal task families:

- text prompt -> symbolic image plan,
- seal image -> identity, attributes, and source text,
- text plus seal -> grounded explanation and match behavior,
- prompt/name -> identity and source binding,
- match/no-match hard negatives.

The first target is `>= 700` per mille with the required source-grounding,
held-out generated-output integrity, green quality report, and objective
coverage gates. Replay and sampler metrics remain useful debugging evidence,
but they are not the headline.

## LLM Path

The immediate LLM path is not a converted HuggingFace/Llama model. NSRL models
must be born into the integer/base-2 attention contract.

The practical sequence is:

1. Keep `node scripts/nsrl-status.mjs` green enough that project state is obvious.
2. Produce `data/processed/nsrl-mme-v0.json` with:

   ```bash
   node scripts/check-nsrl-mme-v0.mjs --out data/processed/nsrl-mme-v0.json
   ```

3. Feed the scorer a measured `quality-report.json` with confidence-trace
   evidence plus objective coverage.
4. Repair the failing product-proof/self-test surface.
5. Run the Graviton product path:

   ```bash
   NSRL_S3_URI=s3://BUCKET/PREFIX scripts/aws/run-solomon-end-to-end.sh
   ```

6. Prove the completed run:

   ```bash
   scripts/aws/prove-solomon-product-run.sh \
     --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME \
     --launch-dir data/aws-launches/RUN_NAME \
     --require-launch-dir
   ```

7. Promote the first narrow NSRL-born `NSRLLMM1` expert before scaling outward
   into routed expert swarms.
