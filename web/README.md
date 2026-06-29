# Solomon Web Model

Static browser build for Solomon, the NSRL AI model for text-conditioned seal
sampling. The page loads the checked-in `NSRLTCH` model and Solomon
text-signature index, runs inference in WASM, and renders the selected bitmap
directly into the canvas.

If `web/assets/solomon-multimodal.nsrlmod` exists, the browser loads that
`NSRLMOD1` artifact first and samples generated text plus a coarse 16x16 image
plan from the same discrete context. If it is absent, the page falls back to the
current `NSRLTCH` denoiser.

The attention-based `NSRLLMM1` path is native-only for now via
`nsrl-solomon-attention`, including prompt-conditioned corpus decoding and
prompt-aware embedded text-memory decoding, plus native constrained eval.
Browser support needs a generic mini-transformer forward/sampling surface in
WASM before the web app can load `model.nsrllmm`.

```sh
wasm-pack build crates/nsrl-web-wasm --release --target web --out-dir ../../web/pkg
rm -f web/pkg/.gitignore
NSRL_SOLOMON_MULTIMODAL_MODEL=web/assets/solomon-multimodal.nsrlmod \
  scripts/run-solomon-multimodal-smoke.sh
python3 -m http.server 5173 --directory web
```

## Publish

`.github/workflows/web-pages.yml` rebuilds the WASM package and deploys the
checked-in `web/` directory to GitHub Pages.

1. Commit `web/`, `crates/nsrl-web-wasm/`, `Cargo.toml`, and `Cargo.lock`.
2. Merge to `main` or `master`.
3. In GitHub repository settings, set Pages -> Build and deployment -> Source to
   GitHub Actions.
4. Run the `Deploy Solomon Web Sampler` workflow, or push a later change under
   `web/`.
