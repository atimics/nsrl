# Crowley Bard Web Chat

Static browser build for the Crowley Bard NSRL lexeme model. The page loads the
model, vocab, and corpus priors as static assets, runs generation in WASM, and
fine-tunes the in-memory model on the hidden chat transcript one turn at a time.
The adapted model and hidden transcript are stored locally in IndexedDB so the
session survives reloads without sending chat data anywhere.

```sh
scripts/build-web-chat.sh
python3 -m http.server 5173 --directory web
```

## Publish

The repository includes `.github/workflows/web-pages.yml`, which deploys the
checked-in `web/` directory to GitHub Pages.

1. Commit `web/`, `crates/nsrl-web-wasm/`, `scripts/build-web-chat.sh`,
   `Cargo.toml`, and `Cargo.lock`.
2. Merge to `main` or `master`.
3. In GitHub repository settings, set Pages → Build and deployment → Source to
   GitHub Actions.
4. Run the `Deploy Web Chat` workflow, or push a later change under `web/`.
