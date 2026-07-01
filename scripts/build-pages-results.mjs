#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const rootDir = process.cwd();
const config = parseArgs(process.argv.slice(2));
const outDir = path.resolve(rootDir, config.outDir);

const tables = [
  {
    id: "prior-scaling",
    title: "Prior Scaling Curve",
    path: "docs/solomon-eval-scaling-curve.tsv",
    columns: [
      "prompt_rows",
      "n_train_prompts",
      "latent_dim",
      "text_features",
      "eval_top1_per_mille",
      "eval_top5_per_mille",
      "gold_top1_per_mille",
      "gold_top5_per_mille",
      "model_hash",
    ],
    highlights: ["eval_top1_per_mille", "eval_top5_per_mille", "gold_top1_per_mille"],
  },
  {
    id: "text-feature-shape",
    title: "Text Feature Shape Probe",
    path: "docs/solomon-text-feature-shape-probe.tsv",
    columns: [
      "prompt_rows",
      "latent_dim",
      "text_features",
      "eval_top1_per_mille",
      "eval_top5_per_mille",
      "novel_top1_per_mille",
      "gold_top1_per_mille",
      "gold_top5_per_mille",
      "model_hash",
    ],
    highlights: ["eval_top1_per_mille", "novel_top1_per_mille", "gold_top5_per_mille"],
  },
  {
    id: "text-feature",
    title: "Text Feature Probe",
    path: "docs/solomon-text-feature-probe.tsv",
    columns: [
      "prompt_rows",
      "latent_dim",
      "text_features",
      "eval_top1_per_mille",
      "eval_top5_per_mille",
      "novel_top1_per_mille",
      "gold_top1_per_mille",
      "gold_top5_per_mille",
      "model_hash",
    ],
    highlights: ["eval_top1_per_mille", "novel_top1_per_mille", "gold_top5_per_mille"],
  },
  {
    id: "generative-eval",
    title: "Generative Eval",
    path: "docs/solomon-generative-eval.tsv",
    columns: [
      "model",
      "prompts",
      "top1_per_mille",
      "top5_per_mille",
      "latent_top1_per_mille",
      "latent_top5_per_mille",
      "mean_generated_target_distance_q8",
      "text_weight",
    ],
    highlights: ["top1_per_mille", "top5_per_mille", "latent_top1_per_mille"],
  },
];

const assets = [
  { id: "attention", label: "NSRLLMM1 attention artifact", path: "web/assets/solomon-attention.nsrllmm" },
  { id: "multimodal", label: "NSRLMOD1 multimodal artifact", path: "web/assets/solomon-multimodal.nsrlmod" },
  { id: "denoiser", label: "NSRLTCH denoiser artifact", path: "web/assets/solomon-model.nsrltch" },
  {
    id: "text-index",
    label: "Solomon text signature index",
    path: "web/assets/solomon-spirit-text-signatures.tsv",
  },
];

const probes = config.skipProbes
  ? []
  : [
      {
        id: "web-quality",
        label: "Browser artifact quality",
        requires: [
          "scripts/check-solomon-attention-web-quality.mjs",
          "web/attention-sampler.js",
          "web/assets/solomon-attention.nsrllmm",
          "web/assets/solomon-spirit-text-signatures.tsv",
        ],
        args: ["scripts/check-solomon-attention-web-quality.mjs", "--all-names", "--summary"],
      },
      {
        id: "raw-name-rank",
        label: "Raw prompt-name rank",
        requires: [
          "scripts/probe-solomon-attention-raw-rank.mjs",
          "web/attention-sampler.js",
          "web/assets/solomon-attention.nsrllmm",
          "web/assets/solomon-spirit-text-signatures.tsv",
        ],
        args: ["scripts/probe-solomon-attention-raw-rank.mjs", "--all-names", "--summary"],
      },
      {
        id: "body-start-rank",
        label: "Body-start rank",
        requires: [
          "scripts/probe-solomon-attention-body-start-rank.mjs",
          "web/attention-sampler.js",
          "web/assets/solomon-attention.nsrllmm",
        ],
        args: ["scripts/probe-solomon-attention-body-start-rank.mjs", "--summary"],
      },
    ];

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

const report = {
  schema: "nsrl.github_pages_results.v1",
  generatedAt: new Date().toISOString(),
  commit: process.env.GITHUB_SHA || gitRevParse(["rev-parse", "HEAD"]) || "unknown",
  run: {
    id: process.env.GITHUB_RUN_ID || null,
    number: process.env.GITHUB_RUN_NUMBER || null,
    url:
      process.env.GITHUB_SERVER_URL && process.env.GITHUB_REPOSITORY && process.env.GITHUB_RUN_ID
        ? `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`
        : null,
  },
  assets: assets.map(assetSummary),
  probes: probes.map(runProbe),
  tables: tables.map(readConfiguredTable),
};

writeFileSync(path.join(outDir, "results.json"), `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(path.join(outDir, "styles.css"), resultStyles());
writeFileSync(path.join(outDir, "index.html"), resultHtml(report));

const failedProbe = report.probes.find((probe) => probe.status === "failed");
if (failedProbe) {
  console.error(`pages results probe failed: ${failedProbe.label}`);
  process.exitCode = 1;
}

function parseArgs(args) {
  const parsed = {
    outDir: "web/results",
    skipProbes: false,
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--out-dir") {
      parsed.outDir = requiredValue(args, ++index, arg);
    } else if (arg === "--skip-probes") {
      parsed.skipProbes = true;
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function requiredValue(args, index, flag) {
  const value = args[index];
  if (!value) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
}

function assetSummary(asset) {
  const absolutePath = path.join(rootDir, asset.path);
  if (!existsSync(absolutePath)) {
    return {
      ...asset,
      status: "missing",
    };
  }
  const bytes = readFileSync(absolutePath);
  return {
    ...asset,
    status: "present",
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function runProbe(probe) {
  const missing = probe.requires.filter((file) => !existsSync(path.join(rootDir, file)));
  if (missing.length > 0) {
    return {
      id: probe.id,
      label: probe.label,
      status: "skipped",
      reason: `missing ${missing.join(", ")}`,
      command: ["node", ...probe.args].join(" "),
    };
  }
  const result = spawnSync(process.execPath, probe.args, {
    cwd: rootDir,
    encoding: "utf8",
  });
  const stdout = result.stdout.trim();
  const stderr = result.stderr.trim();
  const data = parseLastJsonLine(stdout);
  if (result.status !== 0 || !data) {
    return {
      id: probe.id,
      label: probe.label,
      status: "failed",
      command: ["node", ...probe.args].join(" "),
      exitCode: result.status,
      stdout: stdout.slice(-4000),
      stderr: stderr.slice(-4000),
    };
  }
  return {
    id: probe.id,
    label: probe.label,
    status: "passed",
    command: ["node", ...probe.args].join(" "),
    data,
  };
}

function parseLastJsonLine(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      return JSON.parse(lines[index]);
    } catch {
      // Keep scanning; some commands emit diagnostic lines before JSON.
    }
  }
  return null;
}

function readConfiguredTable(config) {
  const absolutePath = path.join(rootDir, config.path);
  if (!existsSync(absolutePath)) {
    return {
      ...config,
      status: "missing",
      rows: [],
      best: [],
    };
  }
  const table = readTsv(absolutePath);
  return {
    ...config,
    status: "present",
    header: table.header,
    rows: table.rows,
    best: config.highlights.map((column) => bestRow(table.rows, column)).filter(Boolean),
  };
}

function readTsv(filePath) {
  const lines = readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  const header = lines.shift()?.split("\t") || [];
  const rows = lines
    .filter(Boolean)
    .map((line) => {
      const cells = line.split("\t");
      const row = {};
      for (let index = 0; index < header.length; index += 1) {
        row[header[index]] = cells[index] ?? "";
      }
      return row;
    });
  return { header, rows };
}

function bestRow(rows, column) {
  let best = null;
  for (const row of rows) {
    const value = Number(row[column]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (!best || value > best.value) {
      best = { column, value, row };
    }
  }
  return best;
}

function gitRevParse(args) {
  const result = spawnSync("git", args, {
    cwd: rootDir,
    encoding: "utf8",
  });
  return result.status === 0 ? result.stdout.trim() : null;
}

function resultHtml(report) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NSRL Results</title>
    <meta name="description" content="Published NSRL Solomon model results and artifact checks." />
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <main class="page">
      <header class="hero">
        <nav class="nav">
          <a href="../">Sampler</a>
          <a href="./results.json">JSON</a>
        </nav>
        <p class="eyebrow">NSRL RESULTS</p>
        <h1>Solomon Results</h1>
        <p class="lede">Published artifact checks and replayable evaluation tables from the CI Pages build.</p>
        <dl class="build-meta">
          <div><dt>Generated</dt><dd>${escapeHtml(report.generatedAt)}</dd></div>
          <div><dt>Commit</dt><dd>${escapeHtml(shortCommit(report.commit))}</dd></div>
          <div><dt>Workflow</dt><dd>${report.run.url ? `<a href="${escapeHtml(report.run.url)}">run ${escapeHtml(report.run.number || report.run.id)}</a>` : "local"}</dd></div>
        </dl>
      </header>

      <section class="section" aria-labelledby="probe-title">
        <h2 id="probe-title">Artifact Probes</h2>
        <div class="probe-grid">
          ${report.probes.map(probeCard).join("\n")}
        </div>
      </section>

      <section class="section" aria-labelledby="asset-title">
        <h2 id="asset-title">Published Artifacts</h2>
        ${assetTable(report.assets)}
      </section>

      ${report.tables.map(resultTable).join("\n")}
    </main>
  </body>
</html>
`;
}

function probeCard(probe) {
  const status = probe.status || "unknown";
  const fields =
    probe.status === "passed"
      ? probeFields(probe.data)
      : [{ label: status === "skipped" ? "Reason" : "Detail", value: probe.reason || probe.stderr || "No JSON result" }];
  return `<article class="probe ${escapeHtml(status)}">
    <div>
      <p class="status">${escapeHtml(status)}</p>
      <h3>${escapeHtml(probe.label)}</h3>
    </div>
    <dl>
      ${fields
        .map(
          (field) => `<div><dt>${escapeHtml(field.label)}</dt><dd>${escapeHtml(String(field.value))}</dd></div>`,
        )
        .join("\n")}
    </dl>
  </article>`;
}

function probeFields(data) {
  if (!data) {
    return [];
  }
  const ordered = [
    ["prompts", "Prompts"],
    ["ranked", "Ranked"],
    ["top1", "Top 1"],
    ["top5", "Top 5"],
    ["top10", "Top 10"],
    ["medianRank", "Median Rank"],
    ["worstRank", "Worst Rank"],
    ["medianMarginQ8", "Median Margin Q8"],
    ["worstMarginQ8", "Worst Margin Q8"],
    ["allNames", "All Names"],
  ];
  return ordered
    .filter(([key]) => data[key] !== undefined)
    .map(([key, label]) => ({ label, value: data[key] }));
}

function assetTable(rows) {
  return `<div class="table-wrap">
    <table>
      <thead>
        <tr><th>Artifact</th><th>Status</th><th>Bytes</th><th>SHA-256</th></tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (asset) => `<tr>
              <th scope="row">${escapeHtml(asset.label)}</th>
              <td>${escapeHtml(asset.status)}</td>
              <td>${asset.bytes ? escapeHtml(humanBytes(asset.bytes)) : ""}</td>
              <td class="mono">${asset.sha256 ? escapeHtml(asset.sha256.slice(0, 16)) : ""}</td>
            </tr>`,
          )
          .join("\n")}
      </tbody>
    </table>
  </div>`;
}

function resultTable(table) {
  const columns = table.columns.filter((column) => table.header?.includes(column));
  const rows = table.rows || [];
  return `<section class="section" aria-labelledby="${escapeHtml(table.id)}-title">
    <div class="section-head">
      <h2 id="${escapeHtml(table.id)}-title">${escapeHtml(table.title)}</h2>
      <span>${rows.length} rows</span>
    </div>
    ${highlights(table.best || [])}
    ${
      rows.length === 0
        ? `<p class="empty">No published rows yet.</p>`
        : `<div class="table-wrap">
      <table>
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `<tr>${columns
                .map((column) => `<td>${escapeHtml(row[column] || "")}</td>`)
                .join("")}</tr>`,
            )
            .join("\n")}
        </tbody>
      </table>
    </div>`
    }
  </section>`;
}

function highlights(rows) {
  if (rows.length === 0) {
    return "";
  }
  return `<div class="highlights">
    ${rows
      .map(
        (item) => `<div>
          <dt>${escapeHtml(item.column)}</dt>
          <dd>${escapeHtml(String(item.value))}</dd>
          <span>${escapeHtml(compactRowLabel(item.row))}</span>
        </div>`,
      )
      .join("\n")}
  </div>`;
}

function compactRowLabel(row) {
  return [
    row.prompt_rows ? `${row.prompt_rows} prompts` : null,
    row.latent_dim ? `latent ${row.latent_dim}` : null,
    row.text_features ? `text ${row.text_features}` : null,
    row.model_hash ? row.model_hash : null,
  ]
    .filter(Boolean)
    .join(" / ");
}

function resultStyles() {
  return `:root {
  color-scheme: dark;
  --bg: #101314;
  --panel: #171b1d;
  --panel-2: #202629;
  --ink: #f4f0e6;
  --muted: #a9b5b1;
  --line: #354145;
  --brass: #d0a24d;
  --teal: #38a59d;
  --red: #d87070;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: #101314;
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

a { color: var(--brass); }

.page {
  width: min(1180px, 100%);
  margin: 0 auto;
  padding: 28px;
}

.hero {
  display: grid;
  gap: 14px;
  padding: 30px 0 24px;
  border-bottom: 1px solid var(--line);
}

.nav {
  display: flex;
  gap: 18px;
  justify-content: flex-end;
}

.eyebrow {
  margin: 0;
  color: var(--brass);
  font-size: 0.78rem;
  text-transform: uppercase;
}

h1, h2, h3, p { margin: 0; }

h1 {
  font-size: 2.4rem;
  line-height: 1.1;
}

h2 {
  font-size: 1.28rem;
}

h3 {
  font-size: 1rem;
}

.lede {
  max-width: 720px;
  color: var(--muted);
  line-height: 1.55;
}

.build-meta,
.probe dl {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 0;
}

.build-meta div,
.probe dl div {
  min-width: 130px;
}

dt {
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
}

dd {
  margin: 3px 0 0;
}

.section {
  display: grid;
  gap: 14px;
  padding: 26px 0;
  border-bottom: 1px solid var(--line);
}

.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
}

.section-head span,
.empty {
  color: var(--muted);
}

.probe-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.probe {
  display: grid;
  gap: 16px;
  min-height: 188px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.status {
  color: var(--teal);
  font-size: 0.72rem;
  text-transform: uppercase;
}

.probe.failed .status { color: var(--red); }
.probe.skipped .status { color: var(--muted); }

.highlights {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.highlights div {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.highlights dd {
  font-size: 1.55rem;
  font-weight: 760;
}

.highlights span {
  display: block;
  margin-top: 4px;
  color: var(--muted);
  font-size: 0.82rem;
}

.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 760px;
}

th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid rgba(53, 65, 69, 0.72);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}

thead th {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  background: var(--panel-2);
}

tbody tr:last-child th,
tbody tr:last-child td {
  border-bottom: 0;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

@media (max-width: 860px) {
  .page {
    padding: 18px;
  }

  .nav {
    justify-content: flex-start;
  }

  h1 {
    font-size: 2rem;
  }

  .probe-grid,
  .highlights {
    grid-template-columns: 1fr;
  }
}
`;
}

function shortCommit(commit) {
  return commit && commit !== "unknown" ? commit.slice(0, 12) : "unknown";
}

function humanBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KiB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
