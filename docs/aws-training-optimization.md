# AWS Solomon Optimization

The active cloud cost problem is Solomon training time, not generic text-model
sweeps.

## Implemented

| Fix | Status |
| --- | --- |
| Threaded `nsrl-bitmap-multichannel-denoise` with deterministic i64 gradient reduction | Done; bit-identical to serial and uses available CPUs |
| Warm AMI bake for Solomon binaries | Scripted in `scripts/aws/bake-training-ami.sh` |
| Local Docker wrappers for Linux parity from macOS | `run-solomon-*-local-docker.sh` |

## Guidance

- Use `c8g.4xlarge` or larger for denoiser training.
- Keep `NSRL_S3_URI` set for long cloud runs so artifacts sync while the run is
  still alive.
- Re-bake the AMI when Rust dependencies or Solomon binaries change materially.
- Treat a sampler that needs target-pixel guidance as a failed model result, not
  an optimization opportunity.
