#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run the Solomon prior smoke in a local Linux Docker container.

This avoids macOS dyld issues with newly built local Mach-O binaries while
keeping artifacts in the repo checkout.

Common knobs:
  NSRL_LOCAL_DOCKER_IMAGE=rust:1.90-bookworm
  NSRL_RUN_NAME=local-solomon-prior-smoke
  NSRL_RUN_ROOT=data/local-runs-linux
  NSRL_SOLOMON_LATENT_EPOCHS=16

The wrapper forwards NSRL_* environment variables into the container.
Absolute paths under this repo are rewritten to /workspace/... inside Docker.
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_repo_root="/workspace"
image="${NSRL_LOCAL_DOCKER_IMAGE:-rust:1.90-bookworm}"
run_root="${NSRL_RUN_ROOT:-data/local-runs-linux}"
run_name="${NSRL_RUN_NAME:-local-solomon-prior-smoke}"

container_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    printf ''
    return
  fi
  case "$value" in
    "$repo_root")
      printf '%s' "$container_repo_root"
      ;;
    "$repo_root"/*)
      printf '%s/%s' "$container_repo_root" "${value#"$repo_root"/}"
      ;;
    *)
      printf '%s' "$value"
      ;;
  esac
}

docker run --rm -t \
  --workdir "$container_repo_root" \
  --volume "${repo_root}:${container_repo_root}" \
  --env PATH="/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env CARGO_TARGET_DIR="${container_repo_root}/target-linux-aarch64" \
  --env NSRL_RUN_ROOT="$(container_path "$run_root")" \
  --env NSRL_RUN_NAME="$run_name" \
  --env NSRL_SOLOMON_TEXT_INDEX="$(container_path "${NSRL_SOLOMON_TEXT_INDEX:-}")" \
  --env NSRL_SOLOMON_PROMPTS="$(container_path "${NSRL_SOLOMON_PROMPTS:-}")" \
  --env NSRL_SOLOMON_GOLD="$(container_path "${NSRL_SOLOMON_GOLD:-}")" \
  --env NSRL_SOLOMON_DENOISE_MODEL="$(container_path "${NSRL_SOLOMON_DENOISE_MODEL:-}")" \
  --env NSRL_SOLOMON_LATENT_EPOCHS="${NSRL_SOLOMON_LATENT_EPOCHS:-}" \
  --env NSRL_SOLOMON_LATENT_DIM="${NSRL_SOLOMON_LATENT_DIM:-}" \
  --env NSRL_SOLOMON_TEXT_FEATURES="${NSRL_SOLOMON_TEXT_FEATURES:-}" \
  --env NSRL_SOLOMON_SAMPLES="${NSRL_SOLOMON_SAMPLES:-}" \
  --env NSRL_SOLOMON_CANDIDATE_MULTIPLIER="${NSRL_SOLOMON_CANDIDATE_MULTIPLIER:-}" \
  --env NSRL_SOLOMON_DIVERSITY_WEIGHT="${NSRL_SOLOMON_DIVERSITY_WEIGHT:-}" \
  --env NSRL_SOLOMON_PASSES="${NSRL_SOLOMON_PASSES:-}" \
  --env NSRL_SOLOMON_SEED_PREFIX="${NSRL_SOLOMON_SEED_PREFIX:-}" \
  --env NSRL_SOLOMON_SEED_VARIANTS="${NSRL_SOLOMON_SEED_VARIANTS:-}" \
  --env NSRL_SOLOMON_MAX_INTRA_PROMPT_DISTANCE="${NSRL_SOLOMON_MAX_INTRA_PROMPT_DISTANCE:-}" \
  --env NSRL_SOLOMON_MAX_TARGET_DISTANCE="${NSRL_SOLOMON_MAX_TARGET_DISTANCE:-}" \
  --env NSRL_SOLOMON_MIN_INTER_CLASS_DISTANCE="${NSRL_SOLOMON_MIN_INTER_CLASS_DISTANCE:-}" \
  --env NSRL_SOLOMON_MIN_TARGET_INK_CELLS="${NSRL_SOLOMON_MIN_TARGET_INK_CELLS:-}" \
  --env NSRL_SOLOMON_MAX_TARGET_INK_CELLS="${NSRL_SOLOMON_MAX_TARGET_INK_CELLS:-}" \
  --env NSRL_SOLOMON_MIN_EVAL_CLASS_TOP1="${NSRL_SOLOMON_MIN_EVAL_CLASS_TOP1:-}" \
  "$image" \
  bash -c '
    set -euo pipefail
    export PATH="/usr/local/cargo/bin:$PATH"
    if ! command -v node >/dev/null 2>&1; then
      apt-get update
      apt-get install -y --no-install-recommends nodejs
      rm -rf /var/lib/apt/lists/*
    fi
    bash scripts/aws/run-solomon-prior-smoke.sh
  '
