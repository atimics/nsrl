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

As of the first status-surface pass, NSRL is not release-ready. The integer
runtime and research artifacts exist, but the Solomon product proof is
incomplete.

The project headline is now `NSRL-MME v0`, a model-native multimodal LLM eval
defined in `docs/multimodal-llm-eval.md`. The current headline score is **not
measured**. Existing sampler, replay, browser-probe, latent-prior, and denoiser
numbers are diagnostics only until they feed a green `quality-report.json` with
confidence-trace evidence for the headline task families.

Known facts from the status command:

- the working tree is dirty,
- the headline multimodal LLM eval is missing,
- the checked-in attention artifacts are smoke-scale, not promoted-profile,
- no `quality-report.json`, `objective-coverage.json`, `release-proof.json`, or
  completed Solomon `pipeline-complete.json` is present under `data/`,
- raw/free-running attention text is still diagnostic-only,
- coherent Solomon text currently comes from prompted or memory-assisted paths.

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
2. Produce a measured `NSRL-MME v0` quality report with confidence-trace
   evidence.
3. Repair the failing product-proof/self-test surface.
4. Run the Graviton product path:

   ```bash
   NSRL_S3_URI=s3://BUCKET/PREFIX scripts/aws/run-solomon-end-to-end.sh
   ```

5. Prove the completed run:

   ```bash
   scripts/aws/prove-solomon-product-run.sh \
     --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME \
     --launch-dir data/aws-launches/RUN_NAME \
     --require-launch-dir
   ```

6. Promote the first narrow NSRL-born `NSRLLMM1` expert before scaling outward
   into routed expert swarms.
