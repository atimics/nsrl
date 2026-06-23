# Cache-Resident Expert Swarm

## Claim

NSRL should scale out before it scales up. Wider or deeper single models push
directly into the static Q15 residual-depth problem, while many shallow i8
experts stay inside the arithmetic regime that already works.

The first defensible scaling claim is therefore:

```text
N cache-resident i8 experts provide increasing routed capability at a fixed
per-expert cache budget, with bit-exact route traces and generation replay.
```

This is not a generic MoE target. The router is per-prompt or per-segment, not
per-token. A selected expert stays hot for the whole generation, which keeps the
experiment aligned with CPU cache behavior instead of fighting it.

## Current Infrastructure

The repository already has most of the necessary pieces:

- expert manifests with capability tags and parameter-byte budgets,
- `mini-transformer-swarm-route` for deterministic top-k expert selection,
- `--route-active-experts 1` for cache-friendly top-1 routing,
- `--route-prompt-affinity` for fixed replay-error prompt scoring,
- `--route-max-parameter-bytes` as the cache-budget proxy,
- distinct corpus lanes that can become specialized experts.

The missing piece is not another deep model. It is a repeatable run shape that
trains one shallow expert per natural corpus lane and reports the routed result
under a fixed per-expert byte budget.

## First Experiment

Run the dry plan:

```bash
./scripts/run-cache-resident-expert-swarm.sh
```

Run the local experiment:

```bash
NSRL_DRY_RUN=0 \
NSRL_CACHE_BUDGET_BYTES=2097152 \
NSRL_MAX_WINDOWS=8192 \
NSRL_SWARM_WORKERS=4 \
NSRL_ROUTE_ACTIVE_EXPERTS=1 \
./scripts/run-cache-resident-expert-swarm.sh
```

Default lanes:

- `simplewiki`: broad expository prose.
- `signal-romance`: radio and logistics voice.
- `crowley-bard`: literary aphorism and Shakespeare-adjacent voice.
- `cosyworld`: cozy simulated world text.
- `signal-sim-log`: synthetic route/state log language.

The script tokenizes missing byte-token lanes, trains one
`mini-transformer-swarm` expert per lane, writes each expert manifest, and emits
a deterministic route trace filtered by `--route-max-parameter-bytes`.

## Measurement Shape

Report the scaling curve as:

```text
expert_count | per_expert_parameter_bytes | selected_expert | route_score |
prompt_affinity_error_q15 | generated_sample | byte_exact_replay_hash
```

The key sweep is not perplexity versus a float baseline. It is capability versus
expert count at a fixed cache footprint. The float comparison is secondary:
NSRL's stronger claim is one-byte weights, cache-resident breadth, deterministic
routing, and reproducible replay.

## Next Tier

Once the lane experts are measured, add a boosting tier:

1. Evaluate each expert on held-out prompts from every lane.
2. Collect the hardest held-out slices by route miss or replay error.
3. Train a second-tier residual expert on those failures.
4. Route tier 1 first, then allow tier 2 only for the failure capability.

That buys capacity-depth without literal residual depth: the residual is in the
expert system, not in one increasingly fragile Q15 trunk.
