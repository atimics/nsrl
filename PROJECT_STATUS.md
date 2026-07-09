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

Known facts from the status command:

- the working tree is dirty,
- the checked-in attention artifacts are smoke-scale, not promoted-profile,
- no `quality-report.json`, `objective-coverage.json`, `release-proof.json`, or
  completed Solomon `pipeline-complete.json` is present under `data/`,
- raw/free-running attention text is still diagnostic-only,
- coherent Solomon text currently comes from prompted or memory-assisted paths.

## LLM Path

The immediate LLM path is not a converted HuggingFace/Llama model. NSRL models
must be born into the integer/base-2 attention contract.

The practical sequence is:

1. Keep `node scripts/nsrl-status.mjs` green enough that project state is obvious.
2. Repair the failing product-proof/self-test surface.
3. Produce a full local Solomon diagnostic.
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
