# NSRL Scheduling Kernel

NSRL swarm work is modeled as deterministic microtasks scheduled onto ARM64
CPU instances. The scheduler is deliberately integer-only: no floats, no FFT
ranker, no wall-clock floating point cost math.

## Microtask Model

Each task declares:

- task kind: train shard, gradient reduce, checkpoint publish, dashboard sync,
  generation probe, checkpoint evaluation, or instance termination
- optional component: output, MLP, embedding, Q, K, V, or O
- resource estimate: minimum cores, memory bytes, cache bytes, expected micros
- artifact inputs and outputs
- status and attempt count

The scheduler receives a compact integer observation:

- active workers
- pending tasks and artifact backlog
- dashboard staleness
- rollback rate and invalid forward count
- zero-delta ratio
- phase through the run
- component movement counters
- cost spent in micro-dollars

## Policy Shape

The first policy follows the ASIX lesson without importing ASIX's float VSA
engine.

1. A conservative static baseline handles safety:
   infeasible tasks are skipped, stale dashboards preempt training, and
   termination outranks work only when the queue and workers are empty.
2. ARM64 resource estimates cap suggested parallelism by cores and shared cache
   footprint.
3. A small residual trace records signed outcomes as integer state-action
   features. It can boost or suppress future tasks, but it does not own the
   whole policy.

This keeps the scheduler useful before the learned trace is good, and prevents
the trace from spending cloud money against hard safety rules.

## Current Crate

`crates/nsrl-sched` is a library-only kernel with unit coverage for:

- stale dashboard preemption
- memory-infeasible task skipping
- ARM64 cache-limited parallelism
- positive residual task boost
- negative residual task suppression

The next integration step is to have the AWS launch API build a task queue from
run specs, then ask `nsrl-sched` which task to dispatch next instead of
branching directly in shell.
