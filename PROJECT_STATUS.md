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

As of 2026-07-10, NSRL is not release-ready. The integer runtime and research
artifacts exist, but the Solomon product proof is incomplete.

The project headline is `NSRL-MME v0`, a model-native multimodal LLM eval
defined in `docs/multimodal-llm-eval.md`. The current local score is **371 per
mille**, below the 700 target. The quality report and generated-output integrity
gates remain red, so this is measured diagnostic evidence rather than a passing
headline result.

Known facts from the status command:

- the working tree is dirty,
- the headline multimodal LLM eval is measured but failing at 371 per mille,
- the checked-in attention artifacts are smoke-scale, not promoted-profile,
- local quality-report and objective-coverage artifacts exist,
- no `release-proof.json` or completed Solomon `pipeline-complete.json` is
  present under `data/`,
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
