#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Train the Solomon text-conditioned denoiser in a local Linux Docker container.

This avoids macOS dyld issues with newly built local Mach-O binaries while
keeping artifacts in the repo checkout.

Common knobs:
  NSRL_LOCAL_DOCKER_IMAGE=rust:1.90-bookworm
  NSRL_SOLOMON_DENOISE_EPOCHS=8
  NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR=data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv
  NSRL_SOLOMON_TEXT_DENOISE_MODEL=data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch

The wrapper forwards the NSRL_SOLOMON_DENOISE_* and NSRL_SOLOMON_TEXT_* knobs.
Absolute paths under this repo are rewritten to /workspace/... inside Docker.
USAGE
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_repo_root="/workspace"
image="${NSRL_LOCAL_DOCKER_IMAGE:-rust:1.90-bookworm}"

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
  --env NSRL_SOLOMON_DENOISE_DATASET="$(container_path "${NSRL_SOLOMON_DENOISE_DATASET:-}")" \
  --env NSRL_SOLOMON_TEXT_INDEX="$(container_path "${NSRL_SOLOMON_TEXT_INDEX:-}")" \
  --env NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR="$(container_path "${NSRL_SOLOMON_TEXT_DENOISE_OUT_DIR:-}")" \
  --env NSRL_SOLOMON_TEXT_DENOISE_MODEL="$(container_path "${NSRL_SOLOMON_TEXT_DENOISE_MODEL:-}")" \
  --env NSRL_SOLOMON_DENOISE_EPOCHS="${NSRL_SOLOMON_DENOISE_EPOCHS:-}" \
  --env NSRL_SOLOMON_DENOISE_PREVIEW_PAIRS="${NSRL_SOLOMON_DENOISE_PREVIEW_PAIRS:-}" \
  --env NSRL_SOLOMON_DENOISE_IMAGE_SIZE="${NSRL_SOLOMON_DENOISE_IMAGE_SIZE:-}" \
  --env NSRL_SOLOMON_DENOISE_TIMESTEPS="${NSRL_SOLOMON_DENOISE_TIMESTEPS:-}" \
  --env NSRL_SOLOMON_DENOISE_LAYERS="${NSRL_SOLOMON_DENOISE_LAYERS:-}" \
  --env NSRL_SOLOMON_DENOISE_HIDDEN_SHIFT="${NSRL_SOLOMON_DENOISE_HIDDEN_SHIFT:-}" \
  --env NSRL_SOLOMON_DENOISE_OUTPUT_SHIFT="${NSRL_SOLOMON_DENOISE_OUTPUT_SHIFT:-}" \
  --env NSRL_SOLOMON_DENOISE_LEARNING_SHIFT="${NSRL_SOLOMON_DENOISE_LEARNING_SHIFT:-}" \
  --env NSRL_SOLOMON_DENOISE_BIAS_LEARNING_SHIFT="${NSRL_SOLOMON_DENOISE_BIAS_LEARNING_SHIFT:-}" \
  "$image" \
  bash -c '
    set -euo pipefail
    export PATH="/usr/local/cargo/bin:$PATH"
    if ! command -v node >/dev/null 2>&1; then
      apt-get update
      apt-get install -y --no-install-recommends nodejs
      rm -rf /var/lib/apt/lists/*
    fi
    bash scripts/aws/run-solomon-text-denoiser-train.sh
  '
