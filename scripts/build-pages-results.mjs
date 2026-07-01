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
  {
    id: "multimodal-eval",
    title: "Multimodal Eval",
    path: "docs/solomon-multimodal-eval.tsv",
    columns: [
      "model",
      "eval_scope",
      "examples",
      "overall_top1_per_mille",
      "text_top1_per_mille",
      "image_top1_per_mille",
      "prompt_top1_per_mille",
      "exact_examples_per_mille",
      "context_hit_per_mille",
      "model_hash",
    ],
    highlights: ["overall_top1_per_mille", "text_top1_per_mille", "image_top1_per_mille"],
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

const assetResults = assets.map(assetSummary);
const probeResults = probes.map(runProbe);
const tableResults = tables.map(readConfiguredTable);

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
  assets: assetResults,
  probes: probeResults,
  tables: tableResults,
  models: buildModelSummaries({
    assets: assetResults,
    probes: probeResults,
    tables: tableResults,
  }),
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

function lowestRow(rows, column) {
  let best = null;
  for (const row of rows) {
    const value = Number(row[column]);
    if (!Number.isFinite(value)) {
      continue;
    }
    if (!best || value < best.value) {
      best = { column, value, row };
    }
  }
  return best;
}

function buildModelSummaries(report) {
  const tableById = new Map(report.tables.map((table) => [table.id, table]));
  const probeById = new Map(report.probes.map((probe) => [probe.id, probe]));
  const assetById = new Map(report.assets.map((asset) => [asset.id, asset]));
  const priorScaling = tableById.get("prior-scaling");
  const textShape = tableById.get("text-feature-shape");
  const generative = tableById.get("generative-eval");
  const multimodal = tableById.get("multimodal-eval");
  const webQuality = probeById.get("web-quality");
  const rawNameRank = probeById.get("raw-name-rank");
  const bodyStartRank = probeById.get("body-start-rank");

  return [
    {
      id: "nsrllmm1",
      name: "NSRLLMM1",
      role: "attention joint text/image-token model",
      status: probesPassed([webQuality, rawNameRank, bodyStartRank]) ? "passing" : "incomplete",
      summary:
        "Current browser artifact path. Its own checks exercise prompt-scoped text, embedded image memory, raw name binding, and source-specific body-start logits.",
      metrics: [
        ratioMetric(
          "Browser artifact quality",
          webQuality?.data?.prompts,
          webQuality?.data?.prompts,
          webQuality?.status === "passed"
            ? "All configured prompt samples passed the browser sampler gate."
            : probeFallback(webQuality),
        ),
        ratioMetric(
          "Raw prompt-name top-1",
          rawNameRank?.data?.top1,
          rawNameRank?.data?.prompts,
          rankDetail(rawNameRank, "Expected spirit name token after `Solomon selects `"),
        ),
        ratioMetric(
          "Body-start top-1",
          bodyStartRank?.data?.top1,
          bodyStartRank?.data?.prompts,
          rankDetail(bodyStartRank, "First source prose token after the name opening"),
        ),
        scalarMetric(
          "Worst body-start rank",
          bodyStartRank?.data?.worstRank,
          "Lower is better; 1 means the expected token was argmax.",
        ),
      ],
      evidence: [
        "scripts/check-solomon-attention-web-quality.mjs --all-names --summary",
        "scripts/probe-solomon-attention-raw-rank.mjs --all-names --summary",
        "scripts/probe-solomon-attention-body-start-rank.mjs --summary",
      ],
    },
    {
      id: "nsrllat1",
      name: "NSRLLAT1",
      role: "prompt-to-layout latent prior",
      status: hasRows(priorScaling) || hasRows(textShape) ? "published evals" : "no rows",
      summary:
        "Prompt/layout prior measured on checked-in train/eval/gold prompt partitions. Scores are internal top-k routing accuracy, reported per mille.",
      metrics: [
        perMilleMetric(
          "Best eval top-1",
          bestRow(priorScaling?.rows || [], "eval_top1_per_mille"),
          "eval_top1_per_mille",
        ),
        perMilleMetric(
          "Best eval top-5",
          bestRow(priorScaling?.rows || [], "eval_top5_per_mille"),
          "eval_top5_per_mille",
        ),
        perMilleMetric(
          "Best gold top-1",
          bestRow(priorScaling?.rows || [], "gold_top1_per_mille"),
          "gold_top1_per_mille",
        ),
        perMilleMetric(
          "Best shape-probe eval top-1",
          bestRow(textShape?.rows || [], "eval_top1_per_mille"),
          "eval_top1_per_mille",
        ),
      ],
      evidence: [
        "docs/solomon-eval-scaling-curve.tsv",
        "docs/solomon-text-feature-shape-probe.tsv",
      ],
    },
    {
      id: "nsrltch",
      name: "NSRLTCH",
      role: "text-conditioned bitmap denoiser",
      status: hasRows(generative)
        ? "published evals"
        : assetById.get("denoiser")?.status === "present"
          ? "artifact published"
          : "missing",
      summary: hasRows(generative)
        ? "The denoiser artifact is scored on class-head latent-conditioned Solomon generations. Scores report target retrieval from generated seal images plus the latent prior's own decoded target rank."
        : "The denoiser artifact is published for browser fallback sampling. The checked-in generative eval table currently has no result rows, so this page does not claim a current denoiser score.",
      metrics: [
        artifactMetric(assetById.get("denoiser")),
        perMilleMetric(
          "Best generated top-1",
          bestRow(generative?.rows || [], "top1_per_mille"),
          "top1_per_mille",
        ),
        perMilleMetric(
          "Best generated top-5",
          bestRow(generative?.rows || [], "top5_per_mille"),
          "top5_per_mille",
        ),
        perMilleMetric(
          "Best latent top-1",
          bestRow(generative?.rows || [], "latent_top1_per_mille"),
          "latent_top1_per_mille",
        ),
        lowerIsBetterMetric(
          "Best target distance",
          lowestRow(generative?.rows || [], "mean_generated_target_distance_q8"),
          "mean_generated_target_distance_q8",
        ),
      ],
      evidence: ["web/assets/solomon-model.nsrltch", "docs/solomon-generative-eval.tsv"],
    },
    {
      id: "nsrlmod1",
      name: "NSRLMOD1",
      role: "coarse joint text/image-token model",
      status: hasRows(multimodal)
        ? "published evals"
        : assetById.get("multimodal")?.status === "present"
          ? "artifact published"
          : "missing",
      summary: hasRows(multimodal)
        ? "The multimodal fallback artifact is scored on tracked corpus replay: next-token ranks over prompt bytes, generated text bytes, marker tokens, and 16x16 image bins. This is artifact-native replay quality, not broad free-running quality."
        : "The multimodal artifact is published as a browser fallback. No checked-in model-quality eval table currently tracks this path separately.",
      metrics: [
        artifactMetric(assetById.get("multimodal")),
        perMilleMetric(
          "Overall replay top-1",
          bestRow(multimodal?.rows || [], "overall_top1_per_mille"),
          "overall_top1_per_mille",
        ),
        perMilleMetric(
          "Text replay top-1",
          bestRow(multimodal?.rows || [], "text_top1_per_mille"),
          "text_top1_per_mille",
        ),
        perMilleMetric(
          "Image-token top-1",
          bestRow(multimodal?.rows || [], "image_top1_per_mille"),
          "image_top1_per_mille",
        ),
        perMilleMetric(
          "Exact examples",
          bestRow(multimodal?.rows || [], "exact_examples_per_mille"),
          "exact_examples_per_mille",
        ),
      ],
      evidence: [
        "web/assets/solomon-multimodal.nsrlmod",
        "docs/solomon-multimodal-eval.tsv",
        "scripts/run-solomon-multimodal-eval.mjs",
      ],
    },
  ];
}

function probesPassed(probes) {
  return probes.every((probe) => probe?.status === "passed");
}

function hasRows(table) {
  return (table?.rows?.length || 0) > 0;
}

function ratioMetric(label, passed, total, detail) {
  const numericPassed = Number(passed);
  const numericTotal = Number(total);
  if (!Number.isFinite(numericPassed) || !Number.isFinite(numericTotal) || numericTotal === 0) {
    return {
      label,
      value: "n/a",
      detail: detail || "No published result.",
      bar: null,
    };
  }
  return {
    label,
    value: `${numericPassed}/${numericTotal}`,
    detail: `${formatRatio(numericPassed, numericTotal)}. ${detail || ""}`.trim(),
    bar: Math.round((numericPassed * 1000) / numericTotal) / 10,
  };
}

function scalarMetric(label, value, detail) {
  return {
    label,
    value: value === undefined || value === null ? "n/a" : String(value),
    detail,
    bar: null,
  };
}

function perMilleMetric(label, best, column) {
  if (!best) {
    return {
      label,
      value: "n/a",
      detail: "No published rows.",
      bar: null,
    };
  }
  return {
    label,
    value: formatPerMille(best.value),
    detail: `${best.value} per mille from ${compactRowLabel(best.row)} (${column}).`,
    bar: best.value / 10,
  };
}

function lowerIsBetterMetric(label, best, column) {
  if (!best) {
    return {
      label,
      value: "n/a",
      detail: "No published rows.",
      bar: null,
    };
  }
  return {
    label,
    value: formatInteger(best.value),
    detail: `${column} from ${compactRowLabel(best.row)}. Lower is better.`,
    bar: null,
  };
}

function artifactMetric(asset) {
  return {
    label: "Published artifact",
    value: asset?.status || "missing",
    detail: asset?.bytes ? `${humanBytes(asset.bytes)} / sha256 ${asset.sha256.slice(0, 12)}` : "No artifact found.",
    bar: asset?.status === "present" ? 100 : 0,
  };
}

function rowsMetric(label, rows, detail) {
  return {
    label,
    value: String(rows),
    detail,
    bar: rows > 0 ? 100 : 0,
  };
}

function rankDetail(probe, target) {
  if (probe?.status !== "passed") {
    return probeFallback(probe);
  }
  const data = probe.data || {};
  const parts = [
    target,
    data.medianRank !== undefined ? `median rank ${data.medianRank}` : null,
    data.worstRank !== undefined ? `worst rank ${data.worstRank}` : null,
  ].filter(Boolean);
  return parts.join("; ") + ".";
}

function probeFallback(probe) {
  if (!probe) {
    return "Probe was not configured.";
  }
  if (probe.status === "skipped") {
    return probe.reason || "Probe skipped.";
  }
  if (probe.status === "failed") {
    return "Probe failed.";
  }
  return "No published probe result.";
}

function formatRatio(passed, total) {
  return `${((passed / total) * 100).toFixed(total === passed ? 0 : 1)}%`;
}

function formatPerMille(value) {
  return `${(Number(value) / 10).toFixed(1)}%`;
}

function formatInteger(value) {
  return Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 });
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
    <title>NSRL Model Evals</title>
    <meta name="description" content="Published NSRL Solomon model performance on NSRL's own evaluation suite." />
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <main class="page">
      <header class="hero">
        <nav class="nav">
          <a href="../">Sampler</a>
          <a href="./results.json">JSON</a>
        </nav>
        <p class="eyebrow">NSRL MODEL EVALS</p>
        <h1>How Solomon Models Perform</h1>
        <p class="lede">Current published scores on NSRL's own replayable Solomon evals. These are internal gates and probes, not external benchmarks.</p>
        <dl class="build-meta">
          <div><dt>Generated</dt><dd>${escapeHtml(report.generatedAt)}</dd></div>
          <div><dt>Commit</dt><dd>${escapeHtml(shortCommit(report.commit))}</dd></div>
          <div><dt>Workflow</dt><dd>${report.run.url ? `<a href="${escapeHtml(report.run.url)}">run ${escapeHtml(report.run.number || report.run.id)}</a>` : "local"}</dd></div>
        </dl>
      </header>

      <section class="section" aria-labelledby="model-title">
        <div class="section-head">
          <div>
            <p class="eyebrow">SCOREBOARD</p>
            <h2 id="model-title">Models On Our Evals</h2>
          </div>
          <span>${report.models.length} model paths</span>
        </div>
        <div class="model-grid">
          ${report.models.map(modelCard).join("\n")}
        </div>
      </section>

      <section class="section" aria-labelledby="eval-title">
        <div class="section-head">
          <div>
            <p class="eyebrow">EVIDENCE</p>
            <h2 id="eval-title">Published Eval Inputs</h2>
          </div>
          <span>${report.tables.reduce((sum, table) => sum + (table.rows?.length || 0), 0)} table rows</span>
        </div>
        ${evalCoverage(report)}
      </section>

      <section class="section" aria-labelledby="probe-title">
        <div class="section-head">
          <div>
            <p class="eyebrow">ATTENTION ARTIFACT</p>
            <h2 id="probe-title">Probe Details</h2>
          </div>
          <span>${report.probes.filter((probe) => probe.status === "passed").length}/${report.probes.length} passed</span>
        </div>
        <div class="probe-grid">
          ${report.probes.map(probeCard).join("\n")}
        </div>
      </section>

      <section class="section" aria-labelledby="asset-title">
        <div class="section-head">
          <div>
            <p class="eyebrow">ARTIFACTS</p>
            <h2 id="asset-title">Published Files</h2>
          </div>
        </div>
        ${assetTable(report.assets)}
      </section>

      ${report.tables.map(resultTable).join("\n")}
    </main>
  </body>
</html>
`;
}

function modelCard(model) {
  return `<article class="model-card ${escapeHtml(model.status.replace(/\s+/g, "-"))}">
    <header>
      <div>
        <p class="model-name">${escapeHtml(model.name)}</p>
        <h3>${escapeHtml(model.role)}</h3>
      </div>
      <span class="model-status">${escapeHtml(model.status)}</span>
    </header>
    <p class="model-summary">${escapeHtml(model.summary)}</p>
    <dl class="metric-grid">
      ${model.metrics.map(metricCard).join("\n")}
    </dl>
    ${evidenceList(model.evidence)}
  </article>`;
}

function metricCard(metric) {
  const bar = Number.isFinite(metric.bar)
    ? `<span class="bar"><span style="width: ${escapeHtml(String(clamp(metric.bar, 0, 100)))}%"></span></span>`
    : "";
  return `<div class="metric">
    <dt>${escapeHtml(metric.label)}</dt>
    <dd>${escapeHtml(metric.value)}</dd>
    ${bar}
    <span>${escapeHtml(metric.detail || "")}</span>
  </div>`;
}

function evidenceList(items) {
  if (!items?.length) {
    return "";
  }
  return `<ul class="evidence">
    ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("\n")}
  </ul>`;
}

function evalCoverage(report) {
  const rows = [
    ["Attention artifact probes", `${report.probes.filter((probe) => probe.status === "passed").length}/${report.probes.length} passed`, "Prompt/text/image checks run against the browser artifact."],
    ["Latent prior tables", `${rowsForTable(report, "prior-scaling") + rowsForTable(report, "text-feature-shape") + rowsForTable(report, "text-feature")} rows`, "Prompt routing, shape, and text-feature sweeps."],
    ["Generative eval table", `${rowsForTable(report, "generative-eval")} rows`, "Held-out generated bitmap eval rows, when checked in."],
    ["Multimodal eval table", `${rowsForTable(report, "multimodal-eval")} rows`, "NSRLMOD1 prompt/text/image-token replay rows."],
  ];
  return `<div class="coverage-grid">
    ${rows
      .map(
        ([label, value, detail]) => `<article>
          <dt>${escapeHtml(label)}</dt>
          <dd>${escapeHtml(value)}</dd>
          <p>${escapeHtml(detail)}</p>
        </article>`,
      )
      .join("\n")}
  </div>`;
}

function rowsForTable(report, id) {
  return report.tables.find((table) => table.id === id)?.rows?.length || 0;
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
  const parts = [
    row.model ? row.model : null,
    row.prompt_rows ? `${row.prompt_rows} prompt rows` : null,
    row.prompts ? `${row.prompts} prompts` : null,
    row.examples ? `${row.examples} examples` : null,
    row.latent_dim ? `latent ${row.latent_dim}` : null,
    row.text_features ? `text ${row.text_features}` : null,
    row.model_hash ? row.model_hash : null,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" / ") : "row";
}

function resultStyles() {
  return `:root {
  color-scheme: dark;
  --bg: #101314;
  --panel: #171b1d;
  --panel-2: #202629;
  --panel-3: #111617;
  --ink: #f4f0e6;
  --muted: #a9b5b1;
  --line: #354145;
  --brass: #d0a24d;
  --teal: #38a59d;
  --red: #d87070;
  --green: #85c889;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

a {
  color: var(--brass);
  text-underline-offset: 4px;
}

.page {
  width: min(1240px, 100%);
  margin: 0 auto;
  padding: 30px;
}

.hero {
  display: grid;
  gap: 16px;
  min-height: 290px;
  align-content: end;
  padding: 34px 0 28px;
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
  max-width: 820px;
  font-size: 3rem;
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
  font-size: 1.04rem;
  line-height: 1.55;
}

.build-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 0;
}

.build-meta div {
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
  gap: 16px;
  padding: 30px 0;
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

.model-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.model-card {
  display: grid;
  gap: 16px;
  min-height: 360px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.model-card header {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: start;
}

.model-name {
  color: var(--brass);
  font-size: 0.82rem;
  font-weight: 760;
}

.model-status {
  flex: 0 0 auto;
  padding: 6px 9px;
  border: 1px solid rgba(56, 165, 157, 0.32);
  border-radius: 999px;
  color: var(--green);
  font-size: 0.72rem;
  text-transform: uppercase;
}

.model-summary {
  color: var(--muted);
  line-height: 1.5;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.metric {
  display: grid;
  gap: 6px;
  padding: 13px;
  border: 1px solid rgba(53, 65, 69, 0.75);
  border-radius: 8px;
  background: var(--panel-3);
}

.metric dd {
  font-size: 1.45rem;
  font-weight: 780;
}

.metric span:not(.bar) {
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.35;
}

.bar {
  display: block;
  height: 7px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(244, 240, 230, 0.1);
}

.bar span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--teal), var(--brass));
}

.evidence {
  display: grid;
  gap: 5px;
  margin: 0;
  padding-left: 18px;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.78rem;
}

.coverage-grid,
.probe-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.coverage-grid article,
.probe {
  display: grid;
  gap: 16px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.coverage-grid dd {
  font-size: 1.7rem;
  font-weight: 780;
}

.coverage-grid p {
  color: var(--muted);
  line-height: 1.45;
}

.probe {
  min-height: 188px;
}

.probe dl {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 0;
}

.probe dl div {
  min-width: 130px;
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

  .model-grid,
  .metric-grid,
  .coverage-grid,
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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
