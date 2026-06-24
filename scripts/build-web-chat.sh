#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p web/assets web/pkg

MODEL_ROOT="${WEB_CHAT_MODEL_ROOT:-data/processed/crowley-bard-aphorism-v2}"
MODEL_PATH="${WEB_CHAT_MODEL_PATH:-$MODEL_ROOT/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm}"
VOCAB_PATH="${WEB_CHAT_VOCAB_PATH:-$MODEL_ROOT/v4096.vocab.tsv}"
TOKENS_PATH="${WEB_CHAT_TOKENS_PATH:-$MODEL_ROOT/v4096.tokens.u16}"
# 30-channel denoise-v1 text-multichannel model: 8 fixed kernels + 6 position +
# 16 text-conditioning channels. The WASM sampler (crates/nsrl-web-wasm) mirrors
# the native nsrl-bitmap-sample text features exactly, so it renders this model
# the same way the X bot does. The text index must use the 16x16 (256-bin)
# signatures the text features expect.
SOLOMON_MODEL_PATH="${WEB_CHAT_SOLOMON_MODEL_PATH:-data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch}"
SOLOMON_TEXT_INDEX_PATH="${WEB_CHAT_SOLOMON_TEXT_INDEX_PATH:-data/processed/key-solomon-goetia-text-index-pg72679-16x16/solomon-spirit-text-signatures.tsv}"

cp "$MODEL_PATH" web/assets/model.nsrllm
cp "$VOCAB_PATH" web/assets/v4096.vocab.tsv
cp "$TOKENS_PATH" web/assets/v4096.tokens.u16
cp "$SOLOMON_MODEL_PATH" web/assets/solomon-model.nsrltch
cp "$SOLOMON_TEXT_INDEX_PATH" web/assets/solomon-spirit-text-signatures.tsv
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-banner-1500x500.png \
  web/assets/crowley-bard-banner.png
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-pfp-400.png \
  web/assets/crowley-bard-pfp.png

wasm-pack build crates/nsrl-web-wasm --release --target web --out-dir ../../web/pkg
rm -f web/pkg/.gitignore
