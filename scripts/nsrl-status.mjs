#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.project_status.v1";
const defaultFastDiagnosticPath = "/tmp/nsrl-solomon-product-diagnostic-fast.json";

const evidenceNames = new Set([
  "quality-report.json",
  "objective-coverage.json",
  "release-proof.json",
  "pipeline-complete.json",
]);

const knownArtifacts = [
  {
    id: "denoiser",
    kind: "NSRLTCH text-conditioned denoiser",
    model: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch",
    trace: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/trace.json",
  },
  {
    id: "latent-prior",
    kind: "NSRLLAT1 prompt/layout prior",
    model: "data/processed/key-solomon-goetia-latent-v1/model.nsrllat",
    trace: "data/processed/key-solomon-goetia-latent-v1/trace.json",
  },
  {
    id: "attention-smoke",
    kind: "NSRLLMM1 attention smoke",
    model: "data/processed/key-solomon-goetia-attention-v1/model.nsrllmm",
    manifest: "data/processed/key-solomon-goetia-attention-v1/manifest.json",
    eval: "data/processed/key-solomon-goetia-attention-v1/attention-eval.json",
    sample: "data/processed/key-solomon-goetia-attention-v1/model-only-bael-current/text.txt",
  },
  {
    id: "attention-curriculum-smoke",
    kind: "NSRLLMM1 attention curriculum smoke",
    model: "data/processed/key-solomon-goetia-attention-curriculum-v1/model.nsrllmm",
    manifest: "data/processed/key-solomon-goetia-attention-curriculum-v1/manifest.json",
    eval: "data/processed/key-solomon-goetia-attention-curriculum-v1/attention-eval.json",
    rawSample: "data/processed/key-solomon-goetia-attention-curriculum-v1/raw-sample-bael/text.txt",
    promptedSample: "data/processed/key-solomon-goetia-attention-curriculum-v1/prior-sample-bael/text.txt",
  },
  {
    id: "multimodal-replay",
    kind: "NSRLMOD1 discrete multimodal replay",
    model: "data/processed/key-solomon-goetia-multimodal-v1/model.nsrlmod",
    manifest: "data/processed/key-solomon-goetia-multimodal-v1/manifest.json",
    sample: "data/processed/key-solomon-goetia-multimodal-v1/sample-bael/text.txt",
  },
  {
    id: "web-attention",
    kind: "deployed web NSRLLMM1 asset",
    model: "web/assets/solomon-attention.nsrllmm",
  },
  {
    id: "web-denoiser",
    kind: "deployed web NSRLTCH asset",
    model: "web/assets/solomon-model.nsrltch",
  },
  {
    id: "web-multimodal",
    kind: "deployed web NSRLMOD1 asset",
    model: "web/assets/solomon-multimodal.nsrlmod",
  },
];

function usage() {
  console.log(
    [
      "Usage: node scripts/nsrl-status.mjs [options]",
      "",
      "Prints the project truth surface: code state, known model artifacts,",
      "Solomon proof evidence, blockers, and next commands.",
      "",
      "Options:",
      "  --json                         emit JSON instead of Markdown",
      "  --out PATH                      write the report to PATH",
      "  --diagnostic PATH               include an existing product diagnostic JSON",
      "  --refresh-fast-diagnostic       run the fast product diagnostic first",
      "  --fast-diagnostic-out PATH      output path for --refresh-fast-diagnostic",
      "  --run-hygiene                   run fmt/no-floats/diff whitespace checks",
      "  --strict                        exit nonzero when release readiness is false",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = {
    json: false,
    outPath: "",
    diagnosticPath: "",
    refreshFastDiagnostic: false,
    fastDiagnosticOut: defaultFastDiagnosticPath,
    runHygiene: false,
    strict: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--json") {
      config.json = true;
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--diagnostic") {
      config.diagnosticPath = requireValue(argv, ++index, arg);
    } else if (arg === "--refresh-fast-diagnostic") {
      config.refreshFastDiagnostic = true;
    } else if (arg === "--fast-diagnostic-out") {
      config.fastDiagnosticOut = requireValue(argv, ++index, arg);
    } else if (arg === "--run-hygiene") {
      config.runHygiene = true;
    } else if (arg === "--strict") {
      config.strict = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function runCommand(command, args, options = {}) {
  const started = Date.now();
  const result = childProcess.spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: options.maxBuffer || 1024 * 1024 * 16,
    timeout: options.timeoutMs || 30000,
  });
  return {
    command: [command, ...args].join(" "),
    ok: result.status === 0,
    status: result.status,
    signal: result.signal || "",
    duration_ms: Date.now() - started,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    error: result.error ? String(result.error.message || result.error) : "",
  };
}

function readJson(relativeOrAbsolutePath) {
  const filePath = resolvePath(relativeOrAbsolutePath);
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readText(relativeOrAbsolutePath) {
  const filePath = resolvePath(relativeOrAbsolutePath);
  return fs.readFileSync(filePath, "utf8");
}

function resolvePath(relativeOrAbsolutePath) {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.join(repoRoot, relativeOrAbsolutePath);
}

function maybeReadJson(relativePath) {
  try {
    return readJson(relativePath);
  } catch {
    return null;
  }
}

function maybeReadText(relativePath) {
  try {
    return readText(relativePath);
  } catch {
    return "";
  }
}

function fileInfo(relativePath) {
  if (!relativePath) {
    return { path: "", present: false };
  }
  const filePath = resolvePath(relativePath);
  try {
    const stat = fs.statSync(filePath);
    return {
      path: relativePath,
      present: true,
      bytes: stat.size,
      size: humanBytes(stat.size),
      modified_at: stat.mtime.toISOString(),
    };
  } catch {
    return { path: relativePath, present: false };
  }
}

function humanBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${bytes} B` : `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
}

function shortText(text, max = 120) {
  const oneLine = String(text || "").replace(/\s+/g, " ").trim();
  if (oneLine.length <= max) {
    return oneLine;
  }
  return `${oneLine.slice(0, max - 1)}...`;
}

function collectGit() {
  const status = runCommand("git", ["status", "--short", "--branch"], { timeoutMs: 10000 });
  const head = runCommand("git", ["log", "--oneline", "-1"], { timeoutMs: 10000 });
  const lines = status.stdout.trimEnd().split("\n").filter(Boolean);
  const branch = lines[0] || "";
  const changes = lines.slice(1);
  return {
    branch,
    head: head.stdout.trim(),
    ok: status.ok && head.ok,
    dirty: changes.length > 0,
    change_count: changes.length,
    tracked_change_count: changes.filter((line) => !line.startsWith("??")).length,
    untracked_count: changes.filter((line) => line.startsWith("??")).length,
    sample_changes: changes.slice(0, 12),
    status_error: status.ok ? "" : status.stderr || status.error,
  };
}

function collectHygiene(runHygiene) {
  if (!runHygiene) {
    return {
      run: false,
      checks: [],
      summary: "not run; pass --run-hygiene for fmt/no-floats/diff checks",
    };
  }
  const checks = [
    runCommand("cargo", ["fmt", "--all", "--check"], { timeoutMs: 30000 }),
    runCommand("./scripts/check-no-floats.sh", [], { timeoutMs: 30000 }),
    runCommand("git", ["diff", "--check"], { timeoutMs: 30000 }),
  ];
  return {
    run: true,
    ok: checks.every((check) => check.ok),
    checks: checks.map((check) => ({
      command: check.command,
      ok: check.ok,
      status: check.status,
      duration_ms: check.duration_ms,
      stdout_tail: tail(check.stdout),
      stderr_tail: tail(check.stderr || check.error),
    })),
  };
}

function tail(text, lines = 12) {
  return String(text || "").trimEnd().split("\n").slice(-lines).join("\n");
}

function collectArtifact(artifact) {
  const model = fileInfo(artifact.model);
  const trace = artifact.trace ? maybeReadJson(artifact.trace) : null;
  const manifest = artifact.manifest ? maybeReadJson(artifact.manifest) : null;
  const evalTrace = artifact.eval ? maybeReadJson(artifact.eval) : null;
  const sample = artifact.sample ? shortText(maybeReadText(artifact.sample)) : "";
  const rawSample = artifact.rawSample ? shortText(maybeReadText(artifact.rawSample)) : "";
  const promptedSample = artifact.promptedSample ? shortText(maybeReadText(artifact.promptedSample)) : "";

  const summary = {
    id: artifact.id,
    kind: artifact.kind,
    model,
  };

  if (trace?.schema === "nsrl.bitmap_denoise_multichannel_trace.v1") {
    summary.trace_schema = trace.schema;
    summary.epochs = trace.epochs;
    summary.eval_input_mean_abs = trace.eval?.input_mean_abs || "";
    summary.eval_predicted_mean_abs = trace.eval?.predicted_mean_abs || "";
    summary.eval_improvement = trace.eval?.input_mean_abs && trace.eval?.predicted_mean_abs
      ? `${trace.eval.input_mean_abs} -> ${trace.eval.predicted_mean_abs}`
      : "";
  } else if (trace?.schema === "nsrl.solomon_latent_trace.v1") {
    summary.trace_schema = trace.schema;
    summary.rows = trace.rows;
    summary.epochs = trace.epochs;
    summary.latent_dim = trace.latent_dim;
    summary.retrieval_top1 = trace.retrieval_top1;
    summary.retrieval_top1_per_mille = trace.retrieval_top1_per_mille;
    summary.retrieval_top5 = trace.retrieval_top5;
    summary.retrieval_top5_per_mille = trace.retrieval_top5_per_mille;
  }

  if (manifest || evalTrace) {
    summary.manifest_schema = manifest?.schema || "";
    summary.corpus_version = manifest?.corpus_version || "";
    summary.text_token_profile = evalTrace?.text_token_profile || manifest?.text_token_profile || "";
    summary.image_token_profile = manifest?.image_token_profile || "";
    summary.token_count = evalTrace?.token_count || manifest?.token_count || null;
    summary.context_seq_len = evalTrace?.context_seq_len || null;
    summary.eval_max_examples = evalTrace?.eval_max_examples ?? null;
    summary.eval_example_count = evalTrace?.example_count ?? null;
    summary.model_hash = evalTrace?.model_hash || "";
    summary.total_accuracy_per_mille = evalTrace?.total?.accuracy_per_mille ?? null;
    summary.text_accuracy_per_mille = evalTrace?.text?.accuracy_per_mille ?? null;
    summary.image_accuracy_per_mille = evalTrace?.image?.accuracy_per_mille ?? null;
    summary.promotion_shape = isPromotionAttentionShape(summary);
  }

  if (sample) {
    summary.sample = sample;
  }
  if (rawSample) {
    summary.raw_sample = rawSample;
  }
  if (promptedSample) {
    summary.prompted_sample = promptedSample;
  }
  return summary;
}

function isPromotionAttentionShape(summary) {
  return Boolean(
    summary.context_seq_len >= 384
      && String(summary.eval_max_examples) === "none"
      && summary.text_token_profile === "chunked"
      && summary.image_token_profile === "symbolic16",
  );
}

function collectArtifacts() {
  return knownArtifacts.map(collectArtifact);
}

function walkForEvidence(startRelativePath, maxDepth = 8) {
  const start = resolvePath(startRelativePath);
  const found = [];
  function visit(dir, depth) {
    if (depth > maxDepth) {
      return;
    }
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name === ".git" || entry.name === "target" || entry.name === "node_modules") {
        continue;
      }
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        visit(fullPath, depth + 1);
      } else if (entry.isFile() && evidenceNames.has(entry.name)) {
        found.push(path.relative(repoRoot, fullPath));
      }
    }
  }
  visit(start, 0);
  return found.sort();
}

function collectProductEvidence() {
  const files = fs.existsSync(resolvePath("data")) ? walkForEvidence("data") : [];
  const byName = {};
  for (const name of evidenceNames) {
    byName[name] = files.filter((file) => path.basename(file) === name);
  }
  return {
    files,
    quality_reports: byName["quality-report.json"],
    objective_coverage: byName["objective-coverage.json"],
    release_proofs: byName["release-proof.json"],
    pipeline_completions: byName["pipeline-complete.json"],
  };
}

function collectPromptEvidence() {
  const prompts = fileInfo("data/processed/key-solomon-goetia-latent-v1/prompts.jsonl");
  const expanded = fileInfo("data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl");
  return {
    prompts: { ...prompts, rows: countLines(prompts.path) },
    expanded_prompts: { ...expanded, rows: countLines(expanded.path) },
  };
}

function countLines(relativePath) {
  if (!relativePath) {
    return null;
  }
  try {
    const text = readText(relativePath);
    return text.length === 0 ? 0 : text.trimEnd().split("\n").length;
  } catch {
    return null;
  }
}

function collectDiagnostic(config) {
  let command = null;
  let diagnosticPath = config.diagnosticPath;
  if (config.refreshFastDiagnostic) {
    diagnosticPath = config.fastDiagnosticOut;
    command = runCommand(
      process.execPath,
      ["scripts/check-solomon-product-diagnostic.mjs", "--fast", "--out", diagnosticPath],
      { timeoutMs: 120000, maxBuffer: 1024 * 1024 * 32 },
    );
  }
  if (!diagnosticPath) {
    return {
      run: false,
      path: "",
      summary: "not supplied; pass --diagnostic or --refresh-fast-diagnostic",
    };
  }
  const diagnostic = maybeReadJson(diagnosticPath);
  if (!diagnostic) {
    return {
      run: Boolean(command),
      path: diagnosticPath,
      command,
      ok: false,
      error: "diagnostic JSON could not be read",
    };
  }
  const failedChecks = Array.isArray(diagnostic.checks)
    ? diagnostic.checks.filter((check) => check && check.ok === false).map((check) => ({
        name: check.name,
        status: check.status,
        schema: check.schema,
        errors: Array.isArray(check.errors) ? check.errors.slice(0, 8) : [],
      }))
    : [];
  return {
    run: Boolean(command),
    path: diagnosticPath,
    command: command
      ? {
          ok: command.ok,
          status: command.status,
          duration_ms: command.duration_ms,
        }
      : null,
    ok: Boolean(diagnostic.ok),
    full_product_proof: Boolean(diagnostic.full_product_proof),
    local_product_proof: Boolean(diagnostic.local_product_proof),
    release_product_proof: Boolean(diagnostic.release_product_proof),
    skipped: diagnostic.skipped || [],
    remaining_product_evidence: diagnostic.remaining_product_evidence || [],
    failed_checks: failedChecks,
  };
}

function deriveStatus(report) {
  const blockers = [];
  const warnings = [];
  if (report.git.dirty) {
    warnings.push(`working tree is dirty (${report.git.change_count} changed paths)`);
  }
  if (report.hygiene.run && !report.hygiene.ok) {
    blockers.push("hygiene checks are not green");
  }
  if (report.product_evidence.quality_reports.length === 0) {
    blockers.push("no Solomon quality-report.json found under data/");
  }
  if (report.product_evidence.objective_coverage.length === 0) {
    blockers.push("no objective-coverage.json found under data/");
  }
  if (report.product_evidence.release_proofs.length === 0) {
    blockers.push("no release-proof.json found under data/");
  }
  if (report.product_evidence.pipeline_completions.length === 0) {
    blockers.push("no completed Solomon pipeline-complete.json found under data/");
  }
  const attentionArtifacts = report.artifacts.filter((artifact) => artifact.id.includes("attention"));
  if (!attentionArtifacts.some((artifact) => artifact.promotion_shape)) {
    blockers.push("checked-in NSRLLMM1 attention artifacts are smoke-scale, not promotion profile");
  }
  const rawAttention = attentionArtifacts.find((artifact) => artifact.raw_sample || artifact.sample);
  if (rawAttention && !/Solomon selects Bael: He/.test(rawAttention.raw_sample || rawAttention.sample || "")) {
    warnings.push("raw/free-running attention text is still weak or diagnostic-only");
  }
  if (report.diagnostic.path && !report.diagnostic.ok) {
    blockers.push("product diagnostic is not green");
    for (const check of report.diagnostic.failed_checks || []) {
      blockers.push(`diagnostic failed check: ${check.name}`);
    }
  }
  if (report.diagnostic.skipped?.length) {
    warnings.push(`diagnostic skipped checks: ${report.diagnostic.skipped.join(", ")}`);
  }
  const releaseReady = blockers.length === 0
    && report.product_evidence.release_proofs.length > 0
    && report.product_evidence.pipeline_completions.length > 0;
  const llmPathState = releaseReady ? "release-ready" : "research/proof-gated";
  return {
    release_ready: releaseReady,
    llm_path_state: llmPathState,
    blockers: [...new Set(blockers)],
    warnings: [...new Set(warnings)],
  };
}

function buildReport(config) {
  const report = {
    schema,
    generated_at: new Date().toISOString(),
    repo_root: repoRoot,
    git: collectGit(),
    hygiene: collectHygiene(config.runHygiene),
    prompts: collectPromptEvidence(),
    artifacts: collectArtifacts(),
    product_evidence: collectProductEvidence(),
    diagnostic: collectDiagnostic(config),
  };
  report.status = deriveStatus(report);
  report.next_commands = nextCommands(report);
  return report;
}

function nextCommands(report) {
  const commands = [];
  if (!report.hygiene.run) {
    commands.push("node scripts/nsrl-status.mjs --run-hygiene");
  }
  if (!report.diagnostic.path) {
    commands.push("node scripts/nsrl-status.mjs --refresh-fast-diagnostic");
  }
  if (report.diagnostic.failed_checks?.some((check) => check.name === "release-candidate-self-test")) {
    commands.push("node scripts/check-solomon-release-candidate-self-test.mjs");
  }
  if (report.product_evidence.quality_reports.length === 0) {
    commands.push("node scripts/check-solomon-product-diagnostic.mjs --out /tmp/nsrl-solomon-product-diagnostic.json");
  }
  if (report.status.blockers.some((blocker) => blocker.includes("pipeline-complete"))) {
    commands.push("NSRL_S3_URI=s3://BUCKET/PREFIX scripts/aws/run-solomon-end-to-end.sh");
  }
  commands.push("scripts/aws/prove-solomon-product-run.sh --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME --launch-dir data/aws-launches/RUN_NAME --require-launch-dir");
  return [...new Set(commands)];
}

function renderMarkdown(report) {
  const lines = [];
  lines.push("# NSRL Project Status");
  lines.push("");
  lines.push(`Generated: ${report.generated_at}`);
  lines.push(`Repo: \`${report.repo_root}\``);
  lines.push("");
  lines.push(`Overall: **${report.status.release_ready ? "release-ready" : "not release-ready"}**`);
  lines.push(`LLM path: **${report.status.llm_path_state}**`);
  lines.push("");

  lines.push("## Current Read");
  lines.push("");
  lines.push(`- Branch: \`${report.git.branch || "unknown"}\``);
  lines.push(`- HEAD: \`${report.git.head || "unknown"}\``);
  lines.push(`- Working tree: ${report.git.dirty ? `${report.git.change_count} changed paths` : "clean"}`);
  lines.push(`- Hygiene: ${renderHygiene(report.hygiene)}`);
  lines.push(`- Product diagnostic: ${renderDiagnosticOneLine(report.diagnostic)}`);
  lines.push(`- Product proof files: ${report.product_evidence.files.length} found`);
  lines.push(`- Held-out prompt rows: ${report.prompts.expanded_prompts.rows ?? "missing"} expanded, ${report.prompts.prompts.rows ?? "missing"} base`);
  lines.push("");

  lines.push("## Blockers");
  lines.push("");
  if (report.status.blockers.length === 0) {
    lines.push("- None detected by this status surface.");
  } else {
    for (const blocker of report.status.blockers) {
      lines.push(`- ${blocker}`);
    }
  }
  if (report.status.warnings.length > 0) {
    lines.push("");
    lines.push("Warnings:");
    for (const warning of report.status.warnings) {
      lines.push(`- ${warning}`);
    }
  }
  lines.push("");

  lines.push("## Artifact Inventory");
  lines.push("");
  for (const artifact of report.artifacts) {
    lines.push(`- **${artifact.id}**: ${artifact.model.present ? artifact.model.size : "missing"} ${artifact.kind}`);
    if (artifact.eval_improvement) {
      lines.push(`  Eval MAE: ${artifact.eval_improvement}`);
    }
    if (artifact.retrieval_top1 !== undefined) {
      lines.push(`  Retrieval: top-1 ${artifact.retrieval_top1}/72 (${artifact.retrieval_top1_per_mille} per mille), top-5 ${artifact.retrieval_top5}/72 (${artifact.retrieval_top5_per_mille} per mille)`);
    }
    if (artifact.total_accuracy_per_mille !== undefined && artifact.total_accuracy_per_mille !== null) {
      lines.push(`  Eval: context ${artifact.context_seq_len}, max examples ${artifact.eval_max_examples}, total/text/image accuracy ${artifact.total_accuracy_per_mille}/${artifact.text_accuracy_per_mille}/${artifact.image_accuracy_per_mille} per mille`);
    }
    if (artifact.raw_sample) {
      lines.push(`  Raw sample: "${artifact.raw_sample}"`);
    }
    if (artifact.prompted_sample) {
      lines.push(`  Prompted sample: "${artifact.prompted_sample}"`);
    } else if (artifact.sample) {
      lines.push(`  Sample: "${artifact.sample}"`);
    }
  }
  lines.push("");

  if (report.diagnostic.failed_checks?.length) {
    lines.push("## Failed Diagnostic Checks");
    lines.push("");
    for (const check of report.diagnostic.failed_checks) {
      lines.push(`- ${check.name}: ${check.errors?.[0] || "failed"}`);
    }
    lines.push("");
  }

  lines.push("## Next Commands");
  lines.push("");
  for (const command of report.next_commands) {
    lines.push(`- \`${command}\``);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function renderHygiene(hygiene) {
  if (!hygiene.run) {
    return hygiene.summary;
  }
  if (hygiene.ok) {
    return "green";
  }
  const failed = hygiene.checks.filter((check) => !check.ok).map((check) => check.command).join(", ");
  return `not green (${failed})`;
}

function renderDiagnosticOneLine(diagnostic) {
  if (!diagnostic.path) {
    return diagnostic.summary;
  }
  const state = diagnostic.ok ? "green" : "not green";
  const failed = diagnostic.failed_checks?.length
    ? `; failed: ${diagnostic.failed_checks.map((check) => check.name).join(", ")}`
    : "";
  const skipped = diagnostic.skipped?.length ? `; skipped: ${diagnostic.skipped.join(", ")}` : "";
  return `${state} at \`${diagnostic.path}\`${failed}${skipped}`;
}

function writeOutput(config, text) {
  if (!config.outPath) {
    process.stdout.write(text);
    return;
  }
  const outPath = resolvePath(config.outPath);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, text, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const report = buildReport(config);
  const output = config.json ? `${JSON.stringify(report, null, 2)}\n` : renderMarkdown(report);
  writeOutput(config, output);
  if (config.strict && !report.status.release_ready) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
