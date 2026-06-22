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
    cargo build --release \
      --package nsrl-train --bin nsrl-train \
      --package nsrl-train --bin nsrl-bitmap-sample \
      --package nsrl-corpus --bin nsrl-corpus
    strip target/release/nsrl-train || true
    strip target/release/nsrl-bitmap-sample || true
    strip target/release/nsrl-corpus || true
  '

cp "$REPO_ROOT/target/release/nsrl-train" "$TRAIN_OUT"
cp "$REPO_ROOT/target/release/nsrl-corpus" "$CORPUS_OUT"
cp "$REPO_ROOT/target/release/nsrl-bitmap-sample" "$SIGIL_OUT"
chmod +x "$TRAIN_OUT" "$CORPUS_OUT" "$SIGIL_OUT"
file "$TRAIN_OUT" "$CORPUS_OUT" "$SIGIL_OUT"
