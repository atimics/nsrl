#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/check-no-floats.sh
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
git diff --check
git diff --cached --check
