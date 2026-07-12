# AWS Solomon Optimization

The active cloud cost problem is Solomon training time, not generic text-model
sweeps.

## Implemented

| Fix | Status |
| --- | --- |
| Threaded `nsrl-bitmap-multichannel-denoise` with deterministic i64 gradient reduction | Done; bit-identical to serial and uses available CPUs |
| Warm AMI bake for Solomon binaries | Scripted in `scripts/aws/bake-training-ami.sh` |
| End-to-end Graviton runner with per-stage artifact sync | `scripts/aws/run-solomon-end-to-end.sh` |
| Local Docker wrappers for Linux parity from macOS | `run-solomon-*-local-docker.sh` |

## Guidance

- Use `c8g.4xlarge` or larger for denoiser training.
- Use the end-to-end runner for clean cloud runs, then rerun individual stages
  only when a trace or quality gate points at one stage.
- For v2 attention runs, inspect `quality-report.json` first. It separates
  task coverage, retrieval-head binding, generated sample binding, generation
  identity inference, generation integrity, ratchetable model-only top-5
  quality, and architecture/profile gates such as `d_model` and context length.
  The end-to-end AWS runner requires the 128d/2-head/2-4-layer promoted small
  profile by default; set `NSRL_SOLOMON_V2_REQUIRE_PROMOTED_SMALL_PROFILE=0`
  only for targeted quick smoke loops.
- Keep `NSRL_S3_URI` set for cloud runs. Product runs require it by default so
  artifacts sync while the run is still alive.
- Re-bake the AMI when Rust dependencies or Solomon binaries change materially.
- Treat a sampler that needs target-pixel guidance as a failed model result, not
  an optimization opportunity.
