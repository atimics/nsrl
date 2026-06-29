# AWS Solomon Training

The cloud lane is a plain Linux runner plus S3 artifacts. The scripts do not
create credentials, buckets, public ACLs, or distributions. They assume the
instance has an IAM role or AWS CLI profile that can read/write the chosen S3
prefix.

## Active Runners

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

## Artifact Shape

```text
s3://BUCKET/PREFIX/
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
- `run-solomon-prior-smoke.sh` builds only the latent trainer, evaluator, and
  sampler.
- Use a baked AMI when iterating; it removes cold Rust/toolchain build overhead.
