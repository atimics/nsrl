#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p web/assets web/pkg

cp data/aws-lambda-lexeme/candidates/visionary-expanded-frozen-v4096-w16384-lr24.nsrllm \
  web/assets/model.nsrllm
cp data/processed/visionary-twitter-bot-demo/v4096.vocab.tsv \
  web/assets/v4096.vocab.tsv
cp data/processed/visionary-twitter-bot-demo/v4096.tokens.u16 \
  web/assets/v4096.tokens.u16
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-banner-1500x500.png \
  web/assets/crowley-bard-banner.png
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-pfp-400.png \
  web/assets/crowley-bard-pfp.png

wasm-pack build crates/nsrl-web-wasm --release --target web --out-dir ../../web/pkg
rm -f web/pkg/.gitignore
