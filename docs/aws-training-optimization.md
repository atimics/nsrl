# AWS Training Optimization

## Diagnosis (evidence-based)

Two compounding causes made AWS training slow:

1. **The trainers we ship are single-threaded.** `run_lexeme_softmax_training`
   and `nsrl-bitmap-multichannel-denoise` had no threading, so they peg **one
   core** on any instance — a 16-vCPU Graviton ran them at 1/16 utilization.
   (The `mini-transformer-swarm` path was already multi-threaded:
   `lib.rs:7399/9325`, workers = `available_parallelism()`.)
2. **Every launch rebuilt from cold after a full OS update.** `UserData` did
   `dnf update -y --allowerasing` → toolchain install → `rustup` → `cargo build`
   from scratch: ~5–10 min of overhead per launch.
3. On-demand only (no spot) — full price for a mostly-idle box.

## Implemented

| # | Fix | File | Status |
|---|-----|------|--------|
| 2 | **Thread the seal denoiser** — data-parallel gradient accumulation (chunk pairs across `available_parallelism()` threads, sum partial i64 grads, single apply) | `nsrl-bitmap-multichannel-denoise.rs` | **Done + verified: bit-identical model, ~5.6× faster** (14 cores: 1219s user / 218s real) |
| 5 | **Drop the full `dnf update`** (install only the toolchain) | `launch-api.py` | Done |
| 3 | **Spot instances** (opt-in `NSRL_USE_SPOT=1`; runs checkpoint to S3 so interruptible) → ~70% cheaper | `launch-api.py` | Done |
| 1 | **Reusable AMI** (warm `target/` + toolchain) so launches build *incrementally* (seconds) not cold (minutes). Bake once, set `NSRL_AMI_ID`. | `scripts/aws/bake-training-ami.sh` | Script written; needs one bake run to produce the AMI |
| 4 | **Right-size to workload** | (guidance below) | Documented |

### Determinism

The threaded seal trainer is **bit-identical** to the serial path (verified by
retraining the same config and `cmp`). Safe because gradient accumulation sums
raw i64 values that never approach `i64::MAX`, so `saturating_add` never
saturates → addition is associative → chunking order is irrelevant.

## Right-sizing (#4)

- **Single-threaded jobs (lexeme as-is):** smallest fast-core box; don't pay for
  idle cores. `c8g.xlarge`.
- **Parallel jobs (seal now, mini-transformer, lexeme after #2-lexeme):**
  `c8g.4xlarge`–`8xlarge`; the threaded loops use all vCPUs automatically.
- Always pass `NSRL_AMI_ID` (baked) + `NSRL_USE_SPOT=1`.

## Remaining: thread the lexeme trainer (#2-lexeme)

**Highest-value remaining item** — the deployed text model trains single-threaded.
It is feasible and the pattern is proven (seal), but it's a careful refactor of a
**load-bearing** function, so it was not rushed in.

- The batch path (`config.batch_windows > 1`) already accumulates **raw i64**
  gradients via `accumulate_lexeme_softmax_output_head_gradient_i64` (output head,
  hidden, embeddings), with Q15 scaling applied **once** after — i.e. the same
  accumulate-then-apply shape as the seal trainer, so determinism is preserved by
  summing partials.
- Work: extract the ~150-line per-window body (`lib.rs` ~6440–6560) into a pure
  fn with **per-thread scratch** (`grad_features_q15`, `grad_head_features_q15`,
  `scaled_grad_output`, the `forward` struct, …) and per-thread copies of the
  three i64 accumulators; chunk the batch's windows across threads; sum
  accumulators; apply once. Verify bit-identical via 1-thread vs N-thread `cmp`.
- Expected: text training scales ~Ncores, turning the ~tens-of-minutes lexeme run
  into minutes on a `c8g.8xlarge`.
