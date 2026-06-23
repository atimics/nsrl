#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run the cache-resident scale-out expert experiment.

Default behavior is a dry run. Set NSRL_DRY_RUN=0 to tokenize missing corpora,
train one shallow swarm expert per lane, emit manifests, and route top-1.

Common knobs:
  NSRL_DRY_RUN=1
  NSRL_RUN_NAME=cache-swarm-20260623
  NSRL_RUN_ROOT=data/processed/cache-resident-expert-swarm
  NSRL_CACHE_BUDGET_BYTES=2097152
  NSRL_MAX_WINDOWS=8192
  NSRL_SWARM_WORKERS=4
  NSRL_SEQ_LEN=32
  NSRL_BATCH_WINDOWS=2
  NSRL_ROUTE_PROMPT='the earth is an ancient planet'
  NSRL_ROUTE_ACTIVE_EXPERTS=1
  NSRL_ROUTE_PROMPT_AFFINITY=1
  NSRL_ROUTE_CAPABILITIES=byte_generation,integer_q15,swarm_ensemble

Override lanes with newline-separated specs:
  NSRL_LANE_SPECS=$'name|capability|corpus_path|tokens_path|prompt\n...'

Artifacts:
  data/processed/cache-resident-expert-swarm/<run-name>/
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${NSRL_RUN_NAME:-cache-resident-swarm-${timestamp}}"
run_root="${NSRL_RUN_ROOT:-data/processed/cache-resident-expert-swarm}"
run_dir="${run_root}/${run_name}"
tokens_dir="${run_dir}/tokens"
experts_dir="${run_dir}/experts"
routes_dir="${run_dir}/routes"
mkdir -p "$tokens_dir" "$experts_dir" "$routes_dir"

dry_run="${NSRL_DRY_RUN:-1}"
cache_budget_bytes="${NSRL_CACHE_BUDGET_BYTES:-2097152}"
max_windows="${NSRL_MAX_WINDOWS:-8192}"
seq_len="${NSRL_SEQ_LEN:-32}"
stride="${NSRL_STRIDE:-1}"
batch_windows="${NSRL_BATCH_WINDOWS:-2}"
swarm_workers="${NSRL_SWARM_WORKERS:-4}"
attention="${NSRL_ATTENTION:-linear}"
position="${NSRL_POSITION:-nope}"
tokenizer="${NSRL_TOKENIZER:-identity}"
text_profile="${NSRL_TEXT_PROFILE:-identity}"
trace_detail="${NSRL_TRACE_DETAIL:-summary}"
route_prompt="${NSRL_ROUTE_PROMPT:-the earth is an ancient planet}"
route_active_experts="${NSRL_ROUTE_ACTIVE_EXPERTS:-1}"
route_prompt_affinity="${NSRL_ROUTE_PROMPT_AFFINITY:-1}"
route_prompt_affinity_windows="${NSRL_ROUTE_PROMPT_AFFINITY_WINDOWS:-32}"
route_capabilities="${NSRL_ROUTE_CAPABILITIES:-byte_generation,integer_q15,swarm_ensemble}"

print_cmd() {
  printf '+'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if [[ "$dry_run" != "1" ]]; then
    "$@"
  fi
}

sanitize_lane_name() {
  printf '%s' "$1" | tr -cs '[:alnum:]_.-' '-'
}

default_lane_specs=(
  "simplewiki|simplewiki_expository|data/processed/simplewiki-expository-v1/simplewiki.clean.txt||the earth is an ancient planet"
  "signal-romance|signal_radio|data/processed/signal-romance-focus-s32/corpus.txt|data/processed/signal-romance-focus-s32/corpus.tokens.u8|relay the cargo route"
  "crowley-bard|visionary_literary|data/processed/crowley-bard-focused-v1/corpus.txt|data/processed/crowley-bard-swarm-v1/crowley-bard-focused-v1.tokens.u8|to be and to become"
  "cosyworld|cosyworld|data/processed/cosyworld-kernel-corpus/corpus.txt||brindle keeps the tea route"
  "signal-sim-log|signal_sim_log|data/processed/signal-sim-log-corpus/sim-log-voice.txt||caution ore traffic on argo"
)

lane_specs=()
if [[ -n "${NSRL_LANE_SPECS:-}" ]]; then
  while IFS= read -r lane_spec; do
    [[ -z "$lane_spec" ]] && continue
    lane_specs+=("$lane_spec")
  done <<< "$NSRL_LANE_SPECS"
else
  lane_specs=("${default_lane_specs[@]}")
fi

expert_paths=()
lane_names=()
lane_prompts=()
lane_manifest="${run_dir}/lanes.tsv"
printf 'lane\tcapability\tcorpus\ttokens\texpert\tmanifest\tprompt\n' > "$lane_manifest"

for lane_spec in "${lane_specs[@]}"; do
  IFS='|' read -r lane capability corpus tokens prompt <<< "$lane_spec"
  safe_lane="$(sanitize_lane_name "$lane")"
  tokens="${tokens:-${tokens_dir}/${safe_lane}.tokens.u8}"
  prompt="${prompt:-$route_prompt}"
  lane_dir="${experts_dir}/${safe_lane}"
  mkdir -p "$lane_dir"

  if [[ ! -f "$tokens" ]]; then
    if [[ ! -f "$corpus" ]]; then
      echo "missing corpus for lane ${lane}: ${corpus}" >&2
      exit 2
    fi
    run_cmd cargo run --release -p nsrl-corpus -- tokenize \
      --corpus "$corpus" \
      --tokens-out "$tokens" \
      --trace "${lane_dir}/${safe_lane}.tokens.trace.jsonl" \
      --seq-len "$seq_len" \
      --stride "$stride" \
      --text-profile "$text_profile"
  fi

  expert_out="${lane_dir}/${safe_lane}.nsrlswarm"
  manifest_out="${lane_dir}/${safe_lane}.manifest.jsonl"
  trace_out="${lane_dir}/${safe_lane}.trace.jsonl"
  progress_out="${lane_dir}/${safe_lane}.progress.jsonl"

  train_cmd=(
    cargo run --release -p nsrl-train --
    --mode mini-transformer-swarm
    --tokens "$tokens"
    --swarm-model-out "$expert_out"
    --manifest-out "$manifest_out"
    --trace "$trace_out"
    --progress-out "$progress_out"
    --seq-len "$seq_len"
    --stride "$stride"
    --batch-windows "$batch_windows"
    --max-windows "$max_windows"
    --swarm-workers "$swarm_workers"
    --mini-transformer-attention "$attention"
    --mini-transformer-position "$position"
    --tokenizer "$tokenizer"
    --mini-transformer-trace-detail "$trace_detail"
  )
  run_cmd "${train_cmd[@]}"

  expert_paths+=("$expert_out")
  lane_names+=("$lane")
  lane_prompts+=("$prompt")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$lane" "$capability" "$corpus" "$tokens" "$expert_out" "$manifest_out" "$prompt" \
    >> "$lane_manifest"
done

route_capability_args=()
IFS=',' read -r -a capability_list <<< "$route_capabilities"
for capability in "${capability_list[@]}"; do
  [[ -z "$capability" ]] && continue
  route_capability_args+=(--route-capability "$capability")
done

route_cmd=(
  cargo run --release -p nsrl-train --
  --mode mini-transformer-swarm-route
  --prompt "$route_prompt"
  --route-max-parameter-bytes "$cache_budget_bytes"
  --route-active-experts "$route_active_experts"
  --mini-transformer-attention "$attention"
  --mini-transformer-position "$position"
)
route_cmd+=("${route_capability_args[@]}")
if [[ "$route_prompt_affinity" == "1" ]]; then
  route_cmd+=(--route-prompt-affinity --route-prompt-affinity-windows "$route_prompt_affinity_windows")
fi
for expert_path in "${expert_paths[@]}"; do
  route_cmd+=(--expert "$expert_path")
done

route_trace="${routes_dir}/top${route_active_experts}.route.jsonl"
print_cmd "${route_cmd[@]}" ">" "$route_trace"
if [[ "$dry_run" != "1" ]]; then
  "${route_cmd[@]}" > "$route_trace"
fi

scaling_tsv="${routes_dir}/scaling.tsv"
printf 'expert_count\tprompt_lane\tselected_expert_index\tselected_expert_id\tscore\tmanifest_score\tprompt_affinity_score\tprompt_eval_windows\tprompt_probability_error_q15\tartifact_bytes\tparameter_bytes\tworker_count\tcontext_seq_len\taccepted_candidates\n' > "$scaling_tsv"

for ((expert_count = 1; expert_count <= ${#expert_paths[@]}; expert_count++)); do
  for ((lane_index = 0; lane_index < ${#lane_names[@]}; lane_index++)); do
    lane="${lane_names[$lane_index]}"
    prompt="${lane_prompts[$lane_index]}"
    safe_lane="$(sanitize_lane_name "$lane")"
    prefix_route_trace="${routes_dir}/k${expert_count}-${safe_lane}.route.jsonl"
    prefix_route_cmd=(
      cargo run --release -p nsrl-train --
      --mode mini-transformer-swarm-route
      --prompt "$prompt"
      --route-max-parameter-bytes "$cache_budget_bytes"
      --route-active-experts "$route_active_experts"
      --mini-transformer-attention "$attention"
      --mini-transformer-position "$position"
    )
    prefix_route_cmd+=("${route_capability_args[@]}")
    if [[ "$route_prompt_affinity" == "1" ]]; then
      prefix_route_cmd+=(--route-prompt-affinity --route-prompt-affinity-windows "$route_prompt_affinity_windows")
    fi
    for ((expert_index = 0; expert_index < expert_count; expert_index++)); do
      prefix_route_cmd+=(--expert "${expert_paths[$expert_index]}")
    done

    print_cmd "${prefix_route_cmd[@]}" ">" "$prefix_route_trace"
    if [[ "$dry_run" != "1" ]]; then
      "${prefix_route_cmd[@]}" > "$prefix_route_trace"
      python3 - "$scaling_tsv" "$expert_count" "$lane" "$prefix_route_trace" <<'PY'
import json
import sys

scaling_tsv, expert_count, lane, route_trace = sys.argv[1:]
with open(route_trace) as fh:
    route = json.loads(fh.readline())
selected = route["selected_expert_indices"][0]
candidate = next(c for c in route["candidates"] if c["expert_index"] == selected)
accepted = sum(1 for c in route["candidates"] if c["accepted"])
row = [
    expert_count,
    lane,
    str(selected),
    candidate["expert_id"],
    str(candidate["score"]),
    str(candidate["manifest_score"]),
    str(candidate["prompt_affinity_score"]),
    str(candidate["prompt_eval_windows"]),
    "" if candidate["prompt_probability_error_q15"] is None else str(candidate["prompt_probability_error_q15"]),
    str(candidate["artifact_bytes"]),
    str(candidate["parameter_bytes"]),
    str(candidate["worker_count"]),
    str(candidate["context_seq_len"]),
    str(accepted),
]
with open(scaling_tsv, "a") as out:
    out.write("\t".join(field.replace("\t", " ") for field in row) + "\n")
PY
    fi
  done
done

experiment_manifest="${run_dir}/experiment.json"
if [[ "$dry_run" != "1" ]]; then
  python3 - "$experiment_manifest" "$run_name" "$cache_budget_bytes" "$max_windows" "$swarm_workers" "$lane_manifest" "$route_trace" "$scaling_tsv" <<'PY'
import csv
import json
import pathlib
import sys

out, run_name, cache_budget, max_windows, swarm_workers, lanes_tsv, route_trace, scaling_tsv = sys.argv[1:]
with open(lanes_tsv, newline="") as fh:
    lanes = list(csv.DictReader(fh, delimiter="\t"))
manifest = {
    "schema": "nsrl.cache_resident_expert_swarm_experiment.v1",
    "run_name": run_name,
    "cache_budget_bytes": int(cache_budget),
    "max_windows_per_lane": int(max_windows),
    "swarm_workers_per_lane": int(swarm_workers),
    "lanes": lanes,
    "route_trace": route_trace,
    "scaling_tsv": scaling_tsv,
    "known_non_claims": [
        "one_lane_expert_per_corpus_not_learned_router_training",
        "prompt_affinity_routes_by_fixed_replay_error",
        "parameter_budget_is_cache_proxy_until_benchmarked",
    ],
}
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
PY
fi

echo "run_dir=${run_dir}"
echo "lanes=${lane_manifest}"
echo "route_trace=${route_trace}"
echo "scaling_tsv=${scaling_tsv}"
echo "dry_run=${dry_run}"
