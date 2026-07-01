# Solomon Web Model

Static browser build for Solomon, the NSRL AI model for text-conditioned seal
sampling. The page prefers the checked-in `NSRLLMM1` attention artifact, can
decode char or chunked text-profile artifacts, uses
its embedded prompt-scoped language-model prior for prose, guards visibly
clipped or looping text with the paired artifact memory, uses embedded 16x16
image tokens, and renders the result directly into the canvas. If an attention
artifact lacks image memory, the browser falls back to the artifact
mini-transformer for image tokens.

If `web/assets/solomon-attention.nsrllmm` is absent or unsupported, the browser
falls back to `web/assets/solomon-multimodal.nsrlmod` (`NSRLMOD1`). If that is
also absent, it falls back to the current `NSRLTCH` WASM denoiser and Solomon
text-signature index.

```sh
wasm-pack build crates/nsrl-web-wasm --release --target web --out-dir ../../web/pkg
rm -f web/pkg/.gitignore
node scripts/generate-solomon-results-samples.mjs
node scripts/build-pages-results.mjs --out-dir web/results
NSRL_SOLOMON_MULTIMODAL_MODEL=web/assets/solomon-multimodal.nsrlmod \
  scripts/run-solomon-multimodal-smoke.sh
cp data/processed/key-solomon-goetia-attention-curriculum-v1/model.nsrllmm \
  web/assets/solomon-attention.nsrllmm
python3 -m http.server 5173 --directory web
```

## Publish

`.github/workflows/web-pages.yml` rebuilds the WASM package, generates
`web/results/` from checked-in result tables, tracked sample panels, and fast
artifact probes, then deploys the static `web/` directory to GitHub Pages. The
main CI workflow also runs the guarded results build so unsupported dashboard
claims fail before publish.

1. Commit `web/`, `crates/nsrl-web-wasm/`, `Cargo.toml`, and `Cargo.lock`.
2. Merge to `main` or `master`.
3. In GitHub repository settings, set Pages -> Build and deployment -> Source to
   GitHub Actions.
4. Run the `Deploy Solomon Web Sampler` workflow, or push a later change under
   `web/`.
