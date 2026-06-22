#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

cargo build --release -p nsrl-train --bin nsrl-train --bin nsrl-swarm-cycle >/dev/null
exec target/release/nsrl-swarm-cycle "$@"
