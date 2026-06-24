#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/build/bin"
TRAIN_OUT="$OUT_DIR/nsrl-train"
CORPUS_OUT="$OUT_DIR/nsrl-corpus"
SIGIL_OUT="$OUT_DIR/nsrl-bitmap-sample"
PLATFORM="${LAMBDA_BUILD_PLATFORM:-linux/arm64}"
IMAGE="${LAMBDA_BUILD_IMAGE:-public.ecr.aws/amazonlinux/amazonlinux:2023}"

mkdir -p "$OUT_DIR"

docker run --rm \
  --platform "$PLATFORM" \
  -v "$REPO_ROOT:/work" \
  -w /work \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    dnf install -y gcc gcc-c++ make ca-certificates findutils tar gzip >/dev/null
    if ! command -v cargo >/dev/null 2>&1; then
      curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
      . "$HOME/.cargo/env"
    fi
    export CARGO_TARGET_DIR=/tmp/nsrl-lambda-target
    cargo build --release \
      --package nsrl-train --bin nsrl-train \
      --package nsrl-train --bin nsrl-bitmap-sample \
      --package nsrl-corpus --bin nsrl-corpus
    strip "$CARGO_TARGET_DIR/release/nsrl-train" || true
    strip "$CARGO_TARGET_DIR/release/nsrl-bitmap-sample" || true
    strip "$CARGO_TARGET_DIR/release/nsrl-corpus" || true
    cp "$CARGO_TARGET_DIR/release/nsrl-train" scripts/x-bot/build/bin/nsrl-train
    cp "$CARGO_TARGET_DIR/release/nsrl-bitmap-sample" scripts/x-bot/build/bin/nsrl-bitmap-sample
    cp "$CARGO_TARGET_DIR/release/nsrl-corpus" scripts/x-bot/build/bin/nsrl-corpus
  '

chmod +x "$TRAIN_OUT" "$CORPUS_OUT" "$SIGIL_OUT"
file "$TRAIN_OUT" "$CORPUS_OUT" "$SIGIL_OUT"
