#!/usr/bin/env node
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
  {
    id: "attention-raw-eval",
    title: "Attention Raw Eval",
    path: "docs/solomon-attention-eval.tsv",
    columns: [
      "model",
      "eval_scope",
      "prompts",
      "prompt_name_match_per_mille",
      "mean_raw_quality_score",
      "min_raw_quality_score",
      "distinct_texts",
      "scaffold_output_per_mille",
      "model_hash",
    ],
    highlights: [
      "prompt_name_match_per_mille",
      "mean_raw_quality_score",
      "distinct_texts",
    ],
  },
  {
    id: "sample-gallery",
    title: "Prompt Sample Gallery",
    path: "docs/solomon-sample-gallery.tsv",
    columns: [
      "model",
      "prompt_id",
      "prompt_kind",
      "prompt",
      "text_source",
      "image_source",
      "text_lm_fallback",
      "text",
      "image_path",
      "model_hash",
    ],
    highlights: [],
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

const sampleAssets = [
  {
    id: "text-conditioned-seals",
    label: "NSRLTCH text-conditioned seal panel",
    source: "docs/assets/solomon-text-conditioned-seals.png",
    path: "assets/solomon-text-conditioned-seals.png",
  },
  {
    id: "sample-gallery-trace",
    label: "Fixed prompt gallery trace",
    source: "docs/solomon-sample-gallery.tsv",
    path: "assets/solomon-sample-gallery.tsv",
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
const sampleAssetResults = [...sampleAssets, ...gallerySampleAssets(tableResults)].map(copySampleAsset);

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
  sampleAssets: sampleAssetResults,
  probes: probeResults,
  tables: tableResults,
};
report.models = buildModelSummaries(report);
report.samplePanels = buildSamplePanels(report);
report.promptSamples = buildPromptSamples(report);
report.honesty = validateHonesty(report, { requireProbes: !config.skipProbes });

writeFileSync(path.join(outDir, "results.json"), `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(path.join(outDir, "styles.css"), resultStyles());
writeFileSync(path.join(outDir, "index.html"), resultHtml(report));

const failedProbe = report.probes.find((probe) => probe.status === "failed");
if (failedProbe) {
  console.error(`pages results probe failed: ${failedProbe.label}`);
  process.exitCode = 1;
}
if (report.honesty.errors.length > 0) {
  console.error("pages results honesty guard failed:");
  for (const error of report.honesty.errors) {
    console.error(`- ${error}`);
  }
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

function copySampleAsset(asset) {
  const sourcePath = path.join(rootDir, asset.source);
  if (!existsSync(sourcePath)) {
    return {
      ...asset,
      status: "missing",
    };
  }
  const targetPath = path.join(outDir, asset.path);
  mkdirSync(path.dirname(targetPath), { recursive: true });
  copyFileSync(sourcePath, targetPath);
  const bytes = readFileSync(sourcePath);
  return {
    ...asset,
    status: "present",
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function gallerySampleAssets(tableResults) {
  const gallery = tableResults.find((table) => table.id === "sample-gallery");
  return (gallery?.rows || [])
    .filter((row) => row.prompt_id && row.image_path)
    .map((row) => ({
      id: `prompt-sample-${row.prompt_id}`,
      label: `Prompt sample: ${row.prompt}`,
      source: row.image_path,
      path: publicSampleAssetPath(row.image_path),
    }));
}

function publicSampleAssetPath(sourcePath) {
  return sourcePath.replace(/^docs\/assets\//, "assets/");
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
  const attentionRaw = tableById.get("attention-raw-eval");
  const webQuality = probeById.get("web-quality");
  const rawNameRank = probeById.get("raw-name-rank");
  const bodyStartRank = probeById.get("body-start-rank");

  const attentionProbesPassed = probesPassed([webQuality, rawNameRank, bodyStartRank]);

  return [
    {
      id: "nsrllmm1",
      name: "NSRLLMM1",
      role: "attention joint text/image-token model",
      status: attentionProbesPassed && hasRows(attentionRaw)
        ? "probe-gated + raw control"
        : attentionProbesPassed
          ? "probe-gated"
          : hasRows(attentionRaw)
            ? "raw control only"
          : "incomplete",
      summary:
        "Current browser artifact path. The probe suite exercises prompt-scoped text, embedded image memory, raw name logits, and source-specific body-start logits. The raw no-memory row is a free-running negative control and should not be read as browser-path quality.",
      metrics: [
        ratioMetric(
          "Artifact probe suite",
          webQuality?.data?.prompts,
          webQuality?.data?.prompts,
          webQuality?.status === "passed"
            ? "All configured prompt samples passed the browser sampler gate."
            : probeFallback(webQuality),
        ),
        ratioMetric(
          "Constrained name-logit top-1",
          rawNameRank?.data?.top1,
          rawNameRank?.data?.prompts,
          rankDetail(rawNameRank, "Expected spirit name token after `Solomon selects `"),
        ),
        ratioMetric(
          "Constrained body-start top-1",
          bodyStartRank?.data?.top1,
          bodyStartRank?.data?.prompts,
          rankDetail(bodyStartRank, "First source prose token after the name opening"),
        ),
        perMilleMetric(
          "Raw no-memory name match",
          bestRow(attentionRaw?.rows || [], "prompt_name_match_per_mille"),
          "prompt_name_match_per_mille",
        ),
        scoreMetric(
          "Raw text quality score",
          bestRow(attentionRaw?.rows || [], "mean_raw_quality_score"),
          "mean_raw_quality_score",
        ),
        perMilleMetric(
          "Raw scaffold outputs",
          bestRow(attentionRaw?.rows || [], "scaffold_output_per_mille"),
          "scaffold_output_per_mille",
        ),
      ],
      evidence: [
        "scripts/check-solomon-attention-web-quality.mjs --all-names --summary",
        "scripts/probe-solomon-attention-raw-rank.mjs --all-names --summary",
        "scripts/probe-solomon-attention-body-start-rank.mjs --summary",
        "docs/solomon-attention-eval.tsv",
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

function buildSamplePanels(report) {
  const tableById = new Map(report.tables.map((table) => [table.id, table]));
  const probeById = new Map(report.probes.map((probe) => [probe.id, probe]));
  const sampleAssetById = new Map(report.sampleAssets.map((asset) => [asset.id, asset]));

  const priorScaling = tableById.get("prior-scaling");
  const generative = tableById.get("generative-eval");
  const multimodal = tableById.get("multimodal-eval");
  const attentionRaw = tableById.get("attention-raw-eval");

  const priorBest = bestRow(priorScaling?.rows || [], "eval_top1_per_mille");
  const generativeBest = bestRow(generative?.rows || [], "top1_per_mille");
  const multimodalBest = bestRow(multimodal?.rows || [], "overall_top1_per_mille");
  const attentionRawRow = firstRow(attentionRaw?.rows || []);
  const sealPanel = sampleAssetById.get("text-conditioned-seals");
  const webQuality = probeById.get("web-quality");
  const rawNameRank = probeById.get("raw-name-rank");
  const bodyStartRank = probeById.get("body-start-rank");
  const attentionProbesPassed = probesPassed([webQuality, rawNameRank, bodyStartRank]);
  const attentionProbeMetrics = attentionProbesPassed
    ? [
        textMetric("Artifact prompts", ratioValue(webQuality?.data?.prompts, webQuality?.data?.prompts)),
        textMetric("Name-logit top-1", ratioValue(rawNameRank?.data?.top1, rawNameRank?.data?.prompts)),
        textMetric("Body-start top-1", ratioValue(bodyStartRank?.data?.top1, bodyStartRank?.data?.prompts)),
      ]
    : [];

  return [
    priorBest
      ? {
          id: "nsrllat1-routing",
          model: "NSRLLAT1",
          title: "Latent Prior Routing",
          outcome: "Best checked-in routing row",
          summary:
            "Prompt-to-layout routing is scored on held-out Solomon prompt partitions. This panel shows the strongest eval top-1 row now published.",
          metrics: [
            textMetric("Eval top-1", formatPerMille(priorBest.value)),
            textMetric("Eval top-5", formatPerMille(priorBest.row.eval_top5_per_mille)),
            textMetric("Gold top-5", formatPerMille(priorBest.row.gold_top5_per_mille)),
          ],
          evidence: ["docs/solomon-eval-scaling-curve.tsv", compactRowLabel(priorBest.row)],
        }
      : null,
    generativeBest
      ? {
          id: "nsrltch-generative",
          model: "NSRLTCH",
          title: "Generated Seal Retrieval",
          outcome: "Class-head generation eval",
          summary:
            "Generated bitmap seals are checked against target retrieval. The image panel is the tracked visual sample; the numbers come from the generative eval table.",
          image:
            sealPanel?.status === "present"
              ? {
                  src: sealPanel.path,
                  alt: sealPanel.label,
                }
              : null,
          metrics: [
            textMetric("Generated top-1", formatPerMille(generativeBest.value)),
            textMetric("Generated top-5", formatPerMille(generativeBest.row.top5_per_mille)),
            textMetric("Latent top-1", formatPerMille(generativeBest.row.latent_top1_per_mille)),
          ],
          evidence: [
            "docs/solomon-generative-eval.tsv",
            "docs/assets/solomon-text-conditioned-seals.png",
          ],
        }
      : null,
    multimodalBest
      ? {
          id: "nsrlmod1-replay",
          model: "NSRLMOD1",
          title: "Text/Image Token Replay",
          outcome: "Tracked corpus replay",
          summary:
            "The multimodal artifact is scored on next-token replay over prompt bytes, generated text bytes, marker tokens, and 16x16 image bins.",
          metrics: [
            textMetric("Overall top-1", formatPerMille(multimodalBest.value)),
            textMetric("Text top-1", formatPerMille(multimodalBest.row.text_top1_per_mille)),
            textMetric("Image top-1", formatPerMille(multimodalBest.row.image_top1_per_mille)),
            textMetric("Context hit", formatPerMille(multimodalBest.row.context_hit_per_mille)),
          ],
          evidence: ["docs/solomon-multimodal-eval.tsv", compactRowLabel(multimodalBest.row)],
        }
      : null,
    attentionRawRow
      ? {
          id: "nsrllmm1-control",
          model: "NSRLLMM1",
          title: "Probe-Gated Artifact, Raw Control",
          outcome: attentionProbesPassed
            ? "Browser probes pass; raw no-memory control fails name binding"
            : "Raw no-memory control row",
          summary:
            attentionProbesPassed
              ? "The browser artifact path is probe-gated separately from native free-running sampling with memory and conditioning disabled."
              : "Native free-running sampling is shown with memory and conditioning disabled. Probe metrics are omitted because this build skipped probes.",
          sample: attentionRawRow.sample_output
            ? {
                label: attentionRawRow.sample_prompt || "raw sample",
                text: attentionRawRow.sample_output,
              }
            : null,
          metrics: [
            ...attentionProbeMetrics,
            textMetric("Raw name match", formatPerMille(attentionRawRow.prompt_name_match_per_mille)),
            textMetric("Raw text score", `${attentionRawRow.mean_raw_quality_score}/100`),
          ],
          evidence: [
            "docs/solomon-attention-eval.tsv",
            "scripts/check-solomon-attention-web-quality.mjs --all-names --summary",
          ],
        }
      : null,
  ].filter(Boolean);
}

function buildPromptSamples(report) {
  const tableById = new Map(report.tables.map((table) => [table.id, table]));
  const sampleAssetById = new Map(report.sampleAssets.map((asset) => [asset.id, asset]));
  const gallery = tableById.get("sample-gallery");
  const trace = sampleAssetById.get("sample-gallery-trace");
  return (gallery?.rows || []).map((row) => {
    const asset = sampleAssetById.get(`prompt-sample-${row.prompt_id}`);
    return {
      id: row.prompt_id,
      model: row.model,
      promptKind: row.prompt_kind,
      prompt: row.prompt,
      textSource: row.text_source,
      imageSource: row.image_source,
      textFallback: row.text_lm_fallback,
      text: row.text,
      image:
        asset?.status === "present"
          ? {
              src: asset.path,
              alt: `${row.prompt} sample`,
            }
          : null,
      traceHref: trace?.status === "present" ? trace.path : null,
      modelHash: row.model_hash,
    };
  });
}

function firstRow(rows) {
  return rows[0] || null;
}

function textMetric(label, value) {
  return {
    label,
    value: value === undefined || value === null || value === "NaN%" ? "n/a" : String(value),
  };
}

function ratioValue(passed, total) {
  const numericPassed = Number(passed);
  const numericTotal = Number(total);
  if (!Number.isFinite(numericPassed) || !Number.isFinite(numericTotal) || numericTotal === 0) {
    return "n/a";
  }
  return `${numericPassed}/${numericTotal}`;
}

function validateHonesty(report, options = {}) {
  const errors = [];
  const tableById = new Map(report.tables.map((table) => [table.id, table]));
  const assetById = new Map(report.assets.map((asset) => [asset.id, asset]));
  const probeById = new Map(report.probes.map((probe) => [probe.id, probe]));
  const modelById = new Map(report.models.map((model) => [model.id, model]));
  const sampleAssetById = new Map(report.sampleAssets.map((asset) => [asset.id, asset]));

  for (const [id, minRows] of [
    ["prior-scaling", 1],
    ["text-feature-shape", 1],
    ["text-feature", 1],
    ["generative-eval", 1],
    ["multimodal-eval", 1],
    ["attention-raw-eval", 1],
    ["sample-gallery", 6],
  ]) {
    const rowCount = tableById.get(id)?.rows?.length || 0;
    if (rowCount < minRows) {
      errors.push(`${id} has ${rowCount} rows; expected at least ${minRows}`);
    }
  }

  for (const id of ["attention", "multimodal", "denoiser", "text-index"]) {
    if (assetById.get(id)?.status !== "present") {
      errors.push(`${id} asset is not present`);
    }
  }

  if (options.requireProbes) {
    for (const id of ["web-quality", "raw-name-rank", "body-start-rank"]) {
      const probe = probeById.get(id);
      if (probe?.status !== "passed") {
        errors.push(`${id} probe is ${probe?.status || "missing"}; expected passed`);
      }
    }
  }

  const nsrllmm1 = modelById.get("nsrllmm1");
  if (nsrllmm1?.status.includes("probe-gated") && !probesPassed([
    probeById.get("web-quality"),
    probeById.get("raw-name-rank"),
    probeById.get("body-start-rank"),
  ])) {
    errors.push("NSRLLMM1 claims probe-gated status without all attention probes passing");
  }
  if (nsrllmm1?.status.includes("raw control") && !hasRows(tableById.get("attention-raw-eval"))) {
    errors.push("NSRLLMM1 claims a raw control without attention-raw-eval rows");
  }

  const claimedModelIds = ["nsrllmm1", "nsrllat1", "nsrltch", "nsrlmod1"];
  for (const id of claimedModelIds) {
    const model = modelById.get(id);
    if (!model) {
      errors.push(`${id} model card is missing`);
      continue;
    }
    if (/published evals|probe-gated/.test(model.status)) {
      for (const metric of model.metrics || []) {
        if (metric.value === "n/a") {
          errors.push(`${model.name} claims ${model.status} but metric "${metric.label}" is n/a`);
        }
      }
    }
  }

  if ((report.samplePanels?.length || 0) < 4) {
    errors.push(`sample panel count is ${report.samplePanels?.length || 0}; expected at least 4`);
  }
  for (const panel of report.samplePanels || []) {
    if (!panel.evidence?.length) {
      errors.push(`${panel.id} sample panel lacks evidence`);
    }
    for (const metric of panel.metrics || []) {
      if (metric.value === "n/a") {
        errors.push(`${panel.id} sample metric "${metric.label}" is n/a`);
      }
    }
    if (panel.id === "nsrllmm1-control" && !panel.sample?.text) {
      errors.push("NSRLLMM1 raw-control sample panel lacks representative raw output");
    }
  }
  if (sampleAssetById.get("text-conditioned-seals")?.status !== "present") {
    errors.push("tracked text-conditioned seal sample asset is missing");
  }
  if (sampleAssetById.get("sample-gallery-trace")?.status !== "present") {
    errors.push("prompt sample gallery trace asset is missing");
  }
  if ((report.promptSamples?.length || 0) < 6) {
    errors.push(`prompt sample count is ${report.promptSamples?.length || 0}; expected at least 6`);
  }
  for (const sample of report.promptSamples || []) {
    if (!sample.text) {
      errors.push(`${sample.id} prompt sample lacks generated text`);
    }
    if (!sample.image) {
      errors.push(`${sample.id} prompt sample lacks a copied image`);
    }
    if (!sample.traceHref) {
      errors.push(`${sample.id} prompt sample lacks a trace link`);
    }
  }

  return {
    status: errors.length === 0 ? "passed" : "failed",
    errors,
  };
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

function scoreMetric(label, best, column) {
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
    value: `${best.value}/100`,
    detail: `${column} from ${compactRowLabel(best.row)}.`,
    bar: best.value,
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

      <section class="section" aria-labelledby="sample-title">
        <div class="section-head">
          <div>
            <p class="eyebrow">SAMPLES</p>
            <h2 id="sample-title">Representative Eval Panels</h2>
          </div>
          <span>${report.samplePanels.length} panels</span>
        </div>
        <div class="sample-grid">
          ${report.samplePanels.map(samplePanelCard).join("\n")}
        </div>
        ${promptGallery(report.promptSamples)}
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

function samplePanelCard(panel) {
  const image = panel.image
    ? `<img src="${escapeHtml(panel.image.src)}" alt="${escapeHtml(panel.image.alt)}" loading="lazy" />`
    : "";
  const sample = panel.sample
    ? `<figure>
        <figcaption>${escapeHtml(panel.sample.label)}</figcaption>
        <blockquote>${escapeHtml(panel.sample.text)}</blockquote>
      </figure>`
    : "";
  return `<article class="sample-card">
    <div class="sample-main">
      <div>
        <p class="model-name">${escapeHtml(panel.model)}</p>
        <h3>${escapeHtml(panel.title)}</h3>
      </div>
      <span>${escapeHtml(panel.outcome)}</span>
    </div>
    ${image}
    <p>${escapeHtml(panel.summary)}</p>
    ${sample}
    <dl class="sample-metrics">
      ${(panel.metrics || [])
        .map(
          (metric) => `<div>
            <dt>${escapeHtml(metric.label)}</dt>
            <dd>${escapeHtml(metric.value)}</dd>
          </div>`,
        )
        .join("\n")}
    </dl>
    ${evidenceList(panel.evidence)}
  </article>`;
}

function promptGallery(samples) {
  if (!samples?.length) {
    return "";
  }
  return `<div class="prompt-gallery" id="fixed-prompt-gallery" aria-label="Fixed prompt gallery">
    <div class="prompt-gallery-head">
      <div>
        <p class="eyebrow">FIXED PROMPTS</p>
        <h3>Bael, Stolas, Marbas, Generic, Held-Out Phrases</h3>
      </div>
      <span>${samples.length} samples</span>
    </div>
    <div class="prompt-grid">
      ${samples.map(promptSampleCard).join("\n")}
    </div>
  </div>`;
}

function promptSampleCard(sample) {
  const trace = sample.traceHref
    ? `<a href="${escapeHtml(sample.traceHref)}">trace</a>`
    : "";
  const image = sample.image
    ? `<img src="${escapeHtml(sample.image.src)}" alt="${escapeHtml(sample.image.alt)}" loading="lazy" />`
    : "";
  return `<article class="prompt-card">
    ${image}
    <div class="prompt-card-body">
      <div>
        <p class="model-name">${escapeHtml(sample.model)}</p>
        <h4>${escapeHtml(sample.prompt)}</h4>
      </div>
      <p>${escapeHtml(sample.text)}</p>
      <dl>
        <div><dt>Prompt kind</dt><dd>${escapeHtml(compactPromptKind(sample.promptKind))}</dd></div>
        <div><dt>Text source</dt><dd>${escapeHtml(compactSampleSource(sample.textSource))}</dd></div>
        <div><dt>Image source</dt><dd>${escapeHtml(compactSampleSource(sample.imageSource))}</dd></div>
      </dl>
      ${trace}
    </div>
  </article>`;
}

function compactPromptKind(value) {
  return String(value || "")
    .replace(/^fixed-/, "")
    .replace("held-out-phrase", "held-out");
}

function compactSampleSource(value) {
  return String(value || "")
    .replace("embedded_text_lm_strict", "lm strict")
    .replace("embedded_text_memory_guard", "memory guard")
    .replace("embedded_image_memory_strict", "image memory")
    .replace("raw_attention", "raw attention");
}

function evalCoverage(report) {
  const rows = [
    ["Honesty guard", report.honesty.status, "CI fails when claimed model coverage lacks checked-in rows, probes, assets, or samples."],
    ["Prompt sample gallery", `${rowsForTable(report, "sample-gallery")} rows`, "Fixed NSRLLMM1 prompt samples with copied PNGs and a TSV trace."],
    ["Attention artifact probes", `${report.probes.filter((probe) => probe.status === "passed").length}/${report.probes.length} passed`, "Prompt/text/image checks run against the browser artifact."],
    ["Attention raw eval table", `${rowsForTable(report, "attention-raw-eval")} rows`, "NSRLLMM1 no-memory native sampling control row."],
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

h1, h2, h3, h4, p { margin: 0; }

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

h4 {
  font-size: 0.98rem;
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

.sample-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.sample-card {
  display: grid;
  gap: 14px;
  align-content: start;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.sample-main {
  display: grid;
  gap: 8px;
}

.sample-main span {
  color: var(--muted);
  font-size: 0.8rem;
  line-height: 1.35;
}

.sample-card > p {
  color: var(--muted);
  line-height: 1.45;
}

.sample-card img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: contain;
  border: 1px solid rgba(53, 65, 69, 0.75);
  border-radius: 8px;
  background: #f4f0e6;
}

.sample-card figure {
  display: grid;
  gap: 7px;
  margin: 0;
}

.sample-card figcaption {
  color: var(--brass);
  font-size: 0.76rem;
  text-transform: uppercase;
}

.sample-card blockquote {
  margin: 0;
  padding: 12px;
  border: 1px solid rgba(53, 65, 69, 0.75);
  border-radius: 8px;
  background: var(--panel-3);
  color: var(--ink);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.82rem;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.sample-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
}

.sample-metrics div {
  padding: 10px;
  border: 1px solid rgba(53, 65, 69, 0.72);
  border-radius: 8px;
  background: var(--panel-3);
}

.sample-metrics dd {
  font-size: 1.15rem;
  font-weight: 760;
}

.prompt-gallery {
  display: grid;
  gap: 14px;
  padding-top: 10px;
}

.prompt-gallery-head {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: end;
}

.prompt-gallery-head span {
  color: var(--muted);
}

.prompt-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.prompt-card {
  display: grid;
  grid-template-columns: 116px minmax(0, 1fr);
  gap: 14px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.prompt-card img {
  width: 116px;
  aspect-ratio: 1 / 1;
  object-fit: contain;
  border: 1px solid rgba(53, 65, 69, 0.75);
  border-radius: 8px;
  background: #f4f0e6;
}

.prompt-card-body {
  display: grid;
  gap: 9px;
  min-width: 0;
}

.prompt-card-body > p {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.42;
}

.prompt-card dl {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 7px;
  margin: 0;
}

.prompt-card dd {
  font-size: 0.78rem;
  overflow-wrap: anywhere;
}

.prompt-card a {
  width: fit-content;
  font-size: 0.82rem;
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
  .sample-grid,
  .prompt-grid,
  .metric-grid,
  .coverage-grid,
  .probe-grid,
  .highlights {
    grid-template-columns: 1fr;
  }

  .prompt-card {
    grid-template-columns: 92px minmax(0, 1fr);
  }

  .prompt-card img {
    width: 92px;
  }

  .prompt-card dl {
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
