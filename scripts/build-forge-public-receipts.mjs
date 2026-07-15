#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const releaseCommit = "9d80ee99d4b01ebb02fdcbffa512f28b82d12675";
const defaultOutDir = "web/launches/audit";

const publicFiles = [
  "research/mathematical-journal/MJ-2026-07-15-13-six-atom-structure-audit.md",
  "research/mathematical-journal/MJ-2026-07-15-14-quenched-document-ising-theory.md",
  "research/mathematical-journal/MJ-2026-07-15-15-conditional-exchange-confirmation.md",
  "docs/deterministic-ising-audit-v1.md",
  "crates/nsrl-train/src/production/training.rs",
  "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json",
  "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json",
  "benchmarks/production-model-v1/p10m-atomic-harmonics-proposal-v1.json",
  "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json",
  "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1.json",
  "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1-contract.json",
  "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json",
  "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json",
  "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json",
  "benchmarks/production-model-v1/p10m-atomic-conditional-exchange-confirmation-v1.json",
  "scripts/check-production-atomic-structure-v1.mjs",
  "scripts/analyze-production-atomic-harmonics-v1.mjs",
  "scripts/analyze-production-atomic-ising-v1.mjs",
  "scripts/check-production-atomic-ising-v1.mjs",
  "scripts/analyze-production-atomic-ising-confirmation-v1.mjs",
  "scripts/check-production-atomic-ising-confirmation-v1.mjs",
  "scripts/analyze-production-conditional-exchange-v1.mjs",
  "scripts/lib/production-atomic-ising-v1.mjs",
];

function parseOutDir() {
  const index = process.argv.indexOf("--out-dir");
  if (index === -1) return defaultOutDir;
  if (!process.argv[index + 1]) throw new Error("--out-dir requires a path");
  return process.argv[index + 1];
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function fileLink(sourcePath) {
  return sourcePath.split(path.sep).map(encodeURIComponent).join("/");
}

function receiptHtml(manifest) {
  const rows = manifest.files.map((file) => `
            <tr>
              <td><a href="${file.public_path}">${file.source_path}</a></td>
              <td><code>${file.sha256}</code></td>
              <td>${file.bytes.toLocaleString("en-US")}</td>
            </tr>`).join("");

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="Public, hash-bound receipts for the Forge six-atom proposal audit and frozen confirmation." />
    <title>Forge six-atom public audit receipts</title>
    <style>
      :root { color-scheme: dark; --ink: #f5f6ee; --muted: #bcc1b5; --lime: #d9ff57; --panel: #171a18; --line: #3c413b; --bg: #0b0d0c; }
      * { box-sizing: border-box; }
      body { margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.6 system-ui, sans-serif; }
      a { color: var(--lime); text-underline-offset: .2em; }
      a:focus-visible { outline: 3px solid var(--lime); outline-offset: 3px; }
      .skip { position: absolute; left: -9999px; }
      .skip:focus { left: 1rem; top: 1rem; background: var(--bg); padding: .7rem; z-index: 2; }
      main { width: min(1100px, calc(100% - 2rem)); margin: 0 auto; padding: 3rem 0 5rem; }
      h1 { max-width: 18ch; font-size: clamp(2.3rem, 7vw, 5.5rem); line-height: .95; letter-spacing: -.05em; }
      h2 { margin-top: 3rem; font-size: clamp(1.5rem, 4vw, 2.4rem); }
      .kicker { color: var(--lime); font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
      .lede { max-width: 70ch; font-size: 1.18rem; }
      .panel { margin: 1.5rem 0; padding: clamp(1rem, 3vw, 2rem); border: 1px solid var(--line); border-radius: 1rem; background: var(--panel); }
      .verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; }
      .verdict div { border-left: .25rem solid var(--lime); padding-left: 1rem; }
      .verdict strong { display: block; font-size: 1.4rem; }
      code { color: #e8ffc0; overflow-wrap: anywhere; }
      pre { padding: 1rem; overflow-x: auto; border: 1px solid var(--line); border-radius: .7rem; background: #090a09; }
      .table-wrap { overflow-x: auto; }
      table { width: 100%; border-collapse: collapse; font-size: .88rem; }
      th, td { padding: .7rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
      th { color: var(--lime); }
      td:first-child { min-width: 22rem; }
      footer { margin-top: 3rem; color: var(--muted); }
    </style>
  </head>
  <body>
    <a class="skip" href="#main">Skip to audit receipt</a>
    <main id="main">
      <p class="kicker">Receipts first · release ${releaseCommit.slice(0, 8)}</p>
      <h1>Six atoms. Public audit.</h1>
      <p class="lede">This is the public evidence surface for Forge’s flagship demonstration. It publishes the proposal audit, frozen confirmation, exact artifacts, and checkers with SHA-256 receipts. It does not claim a new deployed-model performance breakthrough.</p>

      <section class="panel" aria-labelledby="verdict-title">
        <h2 id="verdict-title">Checked outcome</h2>
        <div class="verdict">
          <div><strong>64 combinations</strong><span>Every subset of six frozen model changes.</span></div>
          <div><strong>8,192 forwards</strong><span>Proposal surface: 64 documents × 2 windows × 64 masks.</span></div>
          <div><strong>Candidate rejected</strong><span>The compact rule selected a move already falsified on unused data. Do not scale.</span></div>
          <div><strong>17 / 17 routed</strong><span>The conditional exchange improved every rerouted confirmation document.</span></div>
        </div>
      </section>

      <section id="proposal" aria-labelledby="proposal-title">
        <h2 id="proposal-title">Proposal audit: a useful “no”</h2>
        <p>The observed higher-precision aggregate was almost cubic, but exact support was maximally tangled and every proposal document came from one source cluster. Its compact rule selected the all-atom move, which had already failed a prospective unused-data test. The checked decision was no optimizer change and no paid scaling.</p>
        <p><a href="${fileLink(publicFiles[0])}">Read MJ-13, the full proposal-only structure audit</a>.</p>
      </section>

      <section id="confirmation" aria-labelledby="confirmation-title">
        <h2 id="confirmation-title">Frozen confirmation: actions transferred; explanations did not</h2>
        <p>All three preregistered within-source comparisons passed the exact Holm family. But the re-estimated pairwise and Gibbs parameter maps changed. The strongest surviving mechanism is narrower: a probe router isolated 17 documents where exchanging one atom for another improved all 17, while always taking that exchange was worse overall.</p>
        <p><a href="${fileLink(publicFiles[2])}">Read MJ-15, the confirmation and conditional-exchange revision</a>.</p>
      </section>

      <section class="panel" aria-labelledby="boundary-title">
        <h2 id="boundary-title">Reproducibility boundary</h2>
        <p>The published bundle byte-replays the hash-bound structure cubes and their derived results. The original model and corpus bytes are not public artifacts, so this bundle does not rerun the original production forwards. The source and executable fingerprints used for those forwards remain bound in the frozen contracts.</p>
        <p>Documents 200–212 remain unread. Cross-source transfer is not established.</p>
      </section>

      <section aria-labelledby="replay-title">
        <h2 id="replay-title">Replay the public bundle</h2>
        <p>Download the files below while preserving their paths under one directory, then run:</p>
        <pre><code>node scripts/check-production-atomic-structure-v1.mjs
node scripts/check-production-atomic-ising-v1.mjs
node scripts/check-production-atomic-ising-confirmation-v1.mjs</code></pre>
        <p><a href="manifest.json">Download the machine-readable SHA-256 manifest</a>.</p>
      </section>

      <section aria-labelledby="files-title">
        <h2 id="files-title">Published files</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th scope="col">Public file</th><th scope="col">SHA-256</th><th scope="col">Bytes</th></tr></thead>
            <tbody>${rows}
            </tbody>
          </table>
        </div>
      </section>

      <footer>
        <p><a href="../#challenge">Return to the Forge open challenge</a>. Supported, falsified, and inconclusive reproductions all count.</p>
      </footer>
    </main>
  </body>
</html>\n`;
}

async function main() {
  const root = process.cwd();
  const outDir = path.resolve(root, parseOutDir());
  await fs.rm(outDir, { recursive: true, force: true });
  await fs.mkdir(outDir, { recursive: true });

  const files = [];
  for (const sourcePath of publicFiles) {
    const source = path.resolve(root, sourcePath);
    const bytes = await fs.readFile(source);
    const publicPath = fileLink(sourcePath);
    const destination = path.resolve(outDir, publicPath);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.writeFile(destination, bytes);
    files.push({
      source_path: sourcePath,
      public_path: publicPath,
      sha256: sha256(bytes),
      bytes: bytes.byteLength,
    });
  }

  const manifest = {
    schema: "nsrl.forge_six_atom_public_receipts.v1",
    release_commit: releaseCommit,
    claim_boundary: "hash_bound_cube_replay_not_original_model_forward_rerun",
    files,
  };
  await fs.writeFile(path.join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.writeFile(path.join(outDir, "index.html"), receiptHtml(manifest));
  console.log(`built ${files.length} public Forge receipt files in ${outDir}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
