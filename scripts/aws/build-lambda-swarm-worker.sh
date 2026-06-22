#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${NSRL_LAMBDA_BUILD_DIR:-$REPO_ROOT/data/aws-lambda-swarm/build}"
PACKAGE_DIR="$BUILD_DIR/package"
ZIP_OUT="${NSRL_LAMBDA_ZIP:-$BUILD_DIR/nsrl-lambda-swarm-worker.zip}"
PLATFORM="${LAMBDA_BUILD_PLATFORM:-linux/arm64}"
IMAGE="${LAMBDA_BUILD_IMAGE:-public.ecr.aws/amazonlinux/amazonlinux:2023}"

mkdir -p "$BUILD_DIR"

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
    cargo build --release -p nsrl-train
    strip target/release/nsrl-train || true
  '

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/bin"
cp "$SCRIPT_DIR/lambda-swarm-worker/lambda_function.py" "$PACKAGE_DIR/lambda_function.py"
cp "$REPO_ROOT/target/release/nsrl-train" "$PACKAGE_DIR/bin/nsrl-train"
chmod +x "$PACKAGE_DIR/bin/nsrl-train"

rm -f "$ZIP_OUT"
(
  cd "$PACKAGE_DIR"
  zip -qr "$ZIP_OUT" lambda_function.py bin/nsrl-train
)

file "$PACKAGE_DIR/bin/nsrl-train"
ls -lh "$ZIP_OUT"
echo "lambda_zip=$ZIP_OUT"
