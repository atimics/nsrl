#!/usr/bin/env node
import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");

function arg(name, fallback = "") {
  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0 && index + 1 < process.argv.length) return process.argv[index + 1];
  const envName = `NSRL_${name.replaceAll("-", "_").toUpperCase()}`;
  return process.env[envName] || fallback;
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function readEnvFile(file) {
  try {
    const env = {};
    for (const line of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
      const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (!match) continue;
      env[match[1]] = match[2].replace(/^"|"$/g, "");
    }
    return env;
  } catch {
    return {};
  }
}

const requestedRunDir = arg("run-dir", "");
if (!requestedRunDir) {
  console.error("--run-dir is required");
  process.exit(2);
}

const runDir = path.resolve(repoRoot, requestedRunDir);
const launchEnv = readEnvFile(path.join(runDir, "launch.env"));
const optionsPath = path.join(runDir, "run-options.json");
const runOptions = readJson(optionsPath) || {};
const s3Prefix = arg(
  "s3-prefix",
  launchEnv.S3_PREFIX ||
    runOptions.outputS3Prefix ||
    runOptions.s3Prefix ||
    runOptions.payloads?.[0]?.payload?.output_s3_prefix ||
    "",
).replace(/\/+$/, "");
const tokens = path.resolve(repoRoot, arg("tokens", launchEnv.TOKENS || ""));
const vocab = path.resolve(repoRoot, arg("vocab", launchEnv.VOCAB || ""));
const currentModel = path.resolve(repoRoot, arg("current-model", launchEnv.BASE_MODEL || ""));
const port = Number(arg("port", "8765"));
const pollMs = Number(arg("poll-ms", "5000"));
const fullRefreshMs = Number(arg("full-refresh-ms", "60000"));
const sampleMaxNewTokens = Number(arg("sample-max-new-tokens", "96"));
const evalOffsets = parseOffsets(arg("eval-offsets", "0"));
const evalMaxWindows = Number(arg("eval-max-windows", "32768"));
const startupRefresh = arg("startup-refresh", "1") !== "0" && !process.argv.includes("--no-startup-refresh");
const runsRoot = path.resolve(repoRoot, arg("runs-root", "data/aws-lambda-lexeme/runs"));
const dashboardDir = path.join(runDir, "dashboard");
const workersDir = path.join(runDir, "workers");
const samplesDir = path.join(runDir, "samples");
const interactiveSamplesDir = path.join(samplesDir, "interactive");

fs.mkdirSync(dashboardDir, { recursive: true });
fs.mkdirSync(workersDir, { recursive: true });
fs.mkdirSync(interactiveSamplesDir, { recursive: true });

function contextForRunDir(selectedRunDir) {
  const resolvedRunDir = path.resolve(repoRoot, selectedRunDir);
  const selectedLaunchEnv = readEnvFile(path.join(resolvedRunDir, "launch.env"));
  const selectedRunOptions = readJson(path.join(resolvedRunDir, "run-options.json")) || {};
  const selectedDashboardDir = path.join(resolvedRunDir, "dashboard");
  const selectedWorkersDir = path.join(resolvedRunDir, "workers");
  const selectedSamplesDir = path.join(resolvedRunDir, "samples");
  return {
    runDir: resolvedRunDir,
    launchEnv: selectedLaunchEnv,
    runOptions: selectedRunOptions,
    s3Prefix: (
      selectedLaunchEnv.S3_PREFIX ||
      selectedRunOptions.outputS3Prefix ||
      selectedRunOptions.s3Prefix ||
      selectedRunOptions.payloads?.[0]?.payload?.output_s3_prefix ||
      ""
    ).replace(/\/+$/, ""),
    dashboardDir: selectedDashboardDir,
    workersDir: selectedWorkersDir,
    samplesDir: selectedSamplesDir,
    interactiveSamplesDir: path.join(selectedSamplesDir, "interactive"),
    isActive: path.resolve(resolvedRunDir) === path.resolve(runDir),
  };
}

const activeContext = {
  runDir,
  launchEnv,
  runOptions,
  s3Prefix,
  dashboardDir,
  workersDir,
  samplesDir,
  interactiveSamplesDir,
  isActive: true,
};

function resolveSelectableRunDir(value) {
  const resolved = path.resolve(repoRoot, value || runDir);
  const root = path.resolve(runsRoot);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error(`runDir must be under ${path.relative(repoRoot, root)}`);
  }
  if (!fs.existsSync(path.join(resolved, "run-options.json"))) {
    throw new Error(`missing run-options.json for ${path.relative(repoRoot, resolved)}`);
  }
  return resolved;
}

function awsBase() {
  return [
    "--profile",
    process.env.NSRL_AWS_PROFILE || process.env.AWS_PROFILE || "staging",
    "--region",
    process.env.NSRL_AWS_REGION || process.env.AWS_REGION || "us-east-1",
  ];
}

function run(command, args, options = {}) {
  return spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: options.stdio || ["ignore", "pipe", "pipe"],
    maxBuffer: 64 * 1024 * 1024,
  });
}

function cpS3(uri, file) {
  if (!uri) return false;
  const result = run("aws", [...awsBase(), "s3", "cp", uri, file, "--only-show-errors"]);
  return result.status === 0;
}

function parseOffsets(value) {
  const offsets = String(value || "0")
    .split(/[,\s]+/)
    .map((part) => Number(part.trim()))
    .filter((offset) => Number.isInteger(offset) && offset >= 0);
  return offsets.length ? [...new Set(offsets)] : [0];
}

function evalScore(result) {
  if (!result) return null;
  const score = Number(result.bits_per_token ?? result.mean_bits_per_token);
  return Number.isFinite(score) ? score : null;
}

function evalWorst(result) {
  if (!result) return null;
  const score = Number(result.max_bits_per_token);
  return Number.isFinite(score) ? score : null;
}

function payloadConfig(item) {
  return item?.payload?.config || {};
}

function payloadSeqLen(item, summary = null) {
  return Number(summary?.config?.softmax_seq_len || payloadConfig(item).softmax_seq_len || item.seqLen || 8);
}

function titleFromRunName(runName) {
  if (runName.includes("simplewiki")) return "NSRL SimpleWiki Boring-English Optimizer";
  if (runName.includes("crowley")) return "NSRL Crowley Lexeme Sweep";
  if (runName.includes("visionary")) return "NSRL Visionary Lexeme Sweep";
  return "NSRL Lexeme Sweep";
}

function corpusFromRunName(runName) {
  if (runName.includes("simplewiki")) return "SimpleWiki";
  if (runName.includes("crowley")) return "Crowley/Bard";
  if (runName.includes("visionary")) return "Visionary";
  return "Lexeme";
}

function candidateDetails(item, summary = null) {
  const cfg = { ...payloadConfig(item), ...(summary?.config || {}) };
  const parts = [];
  const seqLen = payloadSeqLen(item, summary);
  if (seqLen) parts.push(`seq ${seqLen}`);
  const windows = Number(cfg.softmax_windows || item.maxWindows || 0);
  if (windows) parts.push(`w ${windows}`);
  const batchWindows = Number(cfg.softmax_batch_windows || 0);
  if (batchWindows) parts.push(`b ${batchWindows}`);
  const lrShift = Number(cfg.softmax_lr_shift || item.lrShift || 0);
  if (lrShift) parts.push(`lr ${lrShift}`);
  if (cfg.train_embeddings) parts.push(`embed lr ${cfg.embedding_lr_shift ?? ""}`.trim());
  const hiddenDim = Number(cfg.hidden_dim || 0);
  if (hiddenDim) parts.push(`hidden ${hiddenDim}`);
  const context = cfg.lexeme_context_features || "";
  if (context) parts.push(context);
  return parts.join(" · ");
}

function fileMtimeMs(file) {
  try {
    return fs.statSync(file).mtimeMs;
  } catch {
    return null;
  }
}

function ageMsFrom(file) {
  const mtimeMs = fileMtimeMs(file);
  return mtimeMs ? Math.max(0, Date.now() - mtimeMs) : null;
}

function loadFullDashboardData(ctx = activeContext) {
  const live = readJson(path.join(ctx.dashboardDir, "live-state.json"));
  const materialized = readJson(path.join(ctx.dashboardDir, "runs.json"));
  if (!live) return materialized || null;
  if (!materialized) return live;
  const liveTime = Date.parse(live.updatedAt || "") || 0;
  const materializedTime = Date.parse(materialized.updatedAt || "") || 0;
  if (materialized.evalOffsets || materializedTime >= liveTime) return materialized;
  return live;
}

function pollSummaries(ctx = activeContext) {
  if (!ctx.s3Prefix) return;
  fs.mkdirSync(ctx.workersDir, { recursive: true });
  for (const item of ctx.runOptions.payloads || []) {
    const workerId = item.workerId;
    const summaryPath = path.join(ctx.workersDir, `${workerId}.summary.json`);
    if (fs.existsSync(summaryPath)) continue;
    cpS3(`${ctx.s3Prefix}/workers/${workerId}.summary.json`, summaryPath);
  }
}

function loadLexemeTrace(file) {
  try {
    const text = fs.readFileSync(file, "utf8").trim();
    if (!text) return null;
    return JSON.parse(text.split(/\r?\n/)[0]);
  } catch {
    return null;
  }
}

function ensureTrace(ctx, summary, workerId) {
  const tracePath = path.join(ctx.workersDir, `${workerId}.softmax.trace.jsonl`);
  if (fs.existsSync(tracePath)) return tracePath;
  return cpS3(summary?.softmax_trace_s3_uri, tracePath) ? tracePath : "";
}

function traceDigest(trace) {
  if (!trace?.metrics) return null;
  const metrics = trace.metrics;
  const errorDelta = Number(metrics.probability_error_delta_i64 ?? metrics.probability_error_delta_i32 ?? 0);
  return {
    finalAccuracyPerMille: Number(metrics.final_accuracy_per_mille ?? 0),
    probabilityErrorImprovement: Math.max(0, -errorDelta),
    probabilityErrorDelta: errorDelta,
    weightDeltaL1: Number(metrics.weight_delta_l1 ?? 0),
    embeddingDeltaL1: Number(metrics.embedding_delta_l1 ?? 0),
    gradientSaturationCount: Number(metrics.gradient_saturation_count ?? 0),
    initialProbabilityErrorQ15: Number(metrics.initial_probability_error_q15 ?? 0),
    finalProbabilityErrorQ15: Number(metrics.final_probability_error_q15 ?? 0),
  };
}

function stepProbabilitySeries(trace) {
  return (trace?.steps || []).slice(0, 96).map((step) => ({
    x: Number(step.update_index ?? step.window_index ?? 0),
    before: Number(step.target_probability_before_q15 ?? 0),
    after: Number(step.target_probability_after_q15 ?? 0),
  }));
}

function buildCharts(candidates, currentEval) {
  const scored = candidates.filter((candidate) => candidate.eval);
  const baselineScore = evalScore(currentEval);
  const baseline = baselineScore
    ? [{ label: "baseline", value: baselineScore, kind: "baseline" }]
    : [];
  const best = [...candidates].sort((a, b) => (a.bitsPerToken ?? Infinity) - (b.bitsPerToken ?? Infinity))[0];
  return {
    bitsPerToken: [
      ...baseline,
      ...scored.map((candidate) => ({
        label: candidate.label,
        workerId: candidate.workerId,
        value: candidate.eval.bits_per_token,
        kind: best?.workerId === candidate.workerId ? "best" : "candidate",
      })),
    ],
    worstBitsPerToken: [
      ...(evalWorst(currentEval) ? [{ label: "baseline", value: evalWorst(currentEval), kind: "baseline" }] : []),
      ...scored
        .filter((candidate) => evalWorst(candidate.eval) !== null)
        .map((candidate) => ({
          label: candidate.label,
          workerId: candidate.workerId,
          value: evalWorst(candidate.eval),
          kind: best?.workerId === candidate.workerId ? "best" : "candidate",
        })),
    ],
    runtimeSeconds: candidates
      .filter((candidate) => candidate.elapsedMs)
      .map((candidate) => ({
        label: candidate.label,
        workerId: candidate.workerId,
        value: candidate.elapsedMs / 1000,
      })),
    accuracyPerMille: candidates
      .filter((candidate) => candidate.traceDigest)
      .map((candidate) => ({
        label: candidate.label,
        workerId: candidate.workerId,
        value: candidate.traceDigest.finalAccuracyPerMille,
      })),
    errorImprovement: candidates
      .filter((candidate) => candidate.traceDigest)
      .map((candidate) => ({
        label: candidate.label,
        workerId: candidate.workerId,
        value: candidate.traceDigest.probabilityErrorImprovement,
      })),
    movementL1: candidates
      .filter((candidate) => candidate.traceDigest)
      .map((candidate) => ({
        label: candidate.label,
        workerId: candidate.workerId,
        value: candidate.traceDigest.weightDeltaL1 + candidate.traceDigest.embeddingDeltaL1,
      })),
    bestStepProbability: best?.stepProbability || [],
    bestStepLabel: best?.label || "",
  };
}

function buildRunData(ctx = activeContext, options = {}) {
  const fullData = loadFullDashboardData(ctx);
  const fullByWorker = new Map((fullData?.candidates || []).map((candidate) => [candidate.workerId, candidate]));
  const candidates = [];
  for (const item of ctx.runOptions.payloads || []) {
    const workerId = item.workerId;
    const label = item.label || payloadConfig(item).label || workerId;
    const summaryPath = path.join(ctx.workersDir, `${workerId}.summary.json`);
    const summary = readJson(summaryPath);
    const full = fullByWorker.get(workerId) || {};
    const seqLen = payloadSeqLen(item, summary);
    const modelPath = path.join(ctx.workersDir, `${workerId}.nsrllm`);
    const evalPath = path.join(ctx.workersDir, `${workerId}.seq${seqLen}.eval.json`);
    const evalResult = full.eval || readJson(evalPath) || null;
    const evalPanel = full.evalPanel || (Array.isArray(evalResult?.rows) ? evalResult : null);
    const tracePath = summary?.ok ? ensureTrace(ctx, summary, workerId) : "";
    const trace = tracePath ? loadLexemeTrace(tracePath) : null;
    const digest = traceDigest(trace);
    const samples = Array.isArray(summary?.samples)
      ? summary.samples.map((sample) => ({
          label: sample.label || sample.prompt || "sample",
          prompt: sample.prompt || "",
          text: sample.text || "",
          decodeRecipe: sample.decode_recipe || "",
          textS3Uri: sample.text_s3_uri || "",
          traceS3Uri: sample.trace_s3_uri || "",
        }))
      : full.samples || [];
    const status = summary ? (summary.ok ? "succeeded" : "failed") : "running";
    const invokePath = path.join(ctx.runDir, `invoke-${workerId}.json`);
    const payloadPath = path.join(ctx.runDir, `${workerId}.payload.json`);
    candidates.push({
      workerId,
      label,
      seqLen,
      details: candidateDetails(item, summary),
      status,
      elapsedMs: summary?.elapsed_ms || full.elapsedMs || null,
      runningMs: status === "running" ? ageMsFrom(invokePath) || ageMsFrom(payloadPath) : null,
      eval: evalResult,
      evalPanel,
      bitsPerToken: evalScore(evalResult),
      samples,
      sampleCount: samples.filter((sample) => sample.text).length,
      modelS3Uri: summary?.model_s3_uri || "",
      localModel: fs.existsSync(modelPath) ? path.relative(repoRoot, modelPath) : full.localModel || "",
      traceDigest: digest,
      stepProbability: trace ? stepProbabilitySeries(trace) : [],
      localTrace: tracePath && fs.existsSync(tracePath) ? path.relative(repoRoot, tracePath) : "",
    });
  }
  const completed = candidates.filter((candidate) => candidate.status === "succeeded").length;
  const failed = candidates.filter((candidate) => candidate.status === "failed").length;
  const scored = candidates.filter((candidate) => candidate.eval);
  const best = [...scored].sort((a, b) => (a.bitsPerToken ?? Infinity) - (b.bitsPerToken ?? Infinity))[0] || null;
  const currentSeqLen =
    fullData?.currentSeqLen ||
    Number(ctx.runOptions.payloads?.[0] ? payloadSeqLen(ctx.runOptions.payloads[0]) : 8);
  const currentEval =
    fullData?.currentEval || readJson(path.join(ctx.dashboardDir, `current-eval-seq${currentSeqLen}.json`)) || null;
  const data = {
    schema: "nsrl.lexeme_live_dashboard.v1",
    title: titleFromRunName(ctx.runOptions.runName || path.basename(ctx.runDir)),
    runName: ctx.runOptions.runName || path.basename(ctx.runDir),
    corpus: ctx.runOptions.corpus || corpusFromRunName(ctx.runOptions.runName || path.basename(ctx.runDir)),
    context: ctx.runOptions.context || payloadConfig(ctx.runOptions.payloads?.[0]).lexeme_context_features || "",
    status: failed ? "failed" : completed === candidates.length ? "succeeded" : "running",
    updatedAt: new Date().toISOString(),
    fullDataUpdatedAt: fullData?.updatedAt || null,
    s3Prefix: ctx.s3Prefix,
    runDir: path.relative(repoRoot, ctx.runDir),
    isActive: ctx.isActive,
    total: candidates.length,
    completed,
    failed,
    running: candidates.length - completed - failed,
    currentSeqLen,
    evalOffsets: fullData?.evalOffsets || evalOffsets,
    evalMaxWindows: fullData?.evalMaxWindows || evalMaxWindows,
    currentEval,
    best: best ? { label: best.label, workerId: best.workerId, bitsPerToken: best.bitsPerToken } : null,
    candidates,
    charts: buildCharts(candidates, currentEval),
    refresh: ctx.isActive ? refreshState : null,
  };
  if (ctx.isActive || options.write) {
    fs.mkdirSync(ctx.dashboardDir, { recursive: true });
    fs.writeFileSync(path.join(ctx.dashboardDir, "live-state.json"), `${JSON.stringify(data, null, 2)}\n`);
  }
  return data;
}

function inferHistoricalRun(dir) {
  const options = readJson(path.join(dir, "run-options.json")) || {};
  const dashboard =
    readJson(path.join(dir, "dashboard", "live-state.json")) ||
    readJson(path.join(dir, "dashboard", "runs.json")) ||
    {};
  const payloads = options.payloads || [];
  let completed = dashboard.completed;
  let failed = dashboard.failed;
  let total = dashboard.total || payloads.length;
  if (completed === undefined || failed === undefined) {
    completed = 0;
    failed = 0;
    for (const item of payloads) {
      const summary = readJson(path.join(dir, "workers", `${item.workerId}.summary.json`));
      if (!summary) continue;
      if (summary.ok) completed += 1;
      else failed += 1;
    }
  }
  const status = dashboard.status || (failed ? "failed" : total && completed === total ? "succeeded" : "running");
  const runName = options.runName || path.basename(dir);
  return {
    runName,
    title: titleFromRunName(runName),
    corpus: options.corpus || corpusFromRunName(runName),
    context: options.context || payloadConfig(payloads[0]).lexeme_context_features || "",
    status,
    total,
    completed,
    failed,
    best: dashboard.best || null,
    baselineBits: evalScore(dashboard.currentEval),
    updatedAt: dashboard.updatedAt || new Date(fileMtimeMs(path.join(dir, "run-options.json")) || Date.now()).toISOString(),
    runDir: path.relative(repoRoot, dir),
    s3Prefix: options.outputS3Prefix || payloads[0]?.payload?.output_s3_prefix || "",
  };
}

function listHistory() {
  let entries = [];
  try {
    entries = fs
      .readdirSync(runsRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(runsRoot, entry.name))
      .filter((dir) => fs.existsSync(path.join(dir, "run-options.json")))
      .map(inferHistoricalRun)
      .sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  } catch {
    entries = [];
  }
  fs.writeFileSync(path.join(dashboardDir, "history.json"), `${JSON.stringify(entries, null, 2)}\n`);
  return entries;
}

const refreshState = {
  active: false,
  lastStartedAt: null,
  lastFinishedAt: null,
  lastError: null,
  lastReason: null,
  lastOutput: "",
};

function startFullRefresh(reason = "manual") {
  if (refreshState.active) return false;
  refreshState.active = true;
  refreshState.lastStartedAt = new Date().toISOString();
  refreshState.lastReason = reason;
  refreshState.lastError = null;
  refreshState.lastOutput = "";
  const watchScript = path.join(repoRoot, "scripts/aws/watch-lexeme-sweep-dashboard.mjs");
  const args = [
    watchScript,
    "--run-dir",
    path.relative(repoRoot, runDir),
    "--s3-prefix",
    s3Prefix,
    "--tokens",
    path.relative(repoRoot, tokens),
    "--vocab",
    path.relative(repoRoot, vocab),
    "--current-model",
    path.relative(repoRoot, currentModel),
    "--sample-max-new-tokens",
    String(sampleMaxNewTokens),
    "--eval-offsets",
    evalOffsets.join(","),
    "--eval-max-windows",
    String(evalMaxWindows),
    "--once",
  ];
  const child = spawn("node", args, { cwd: repoRoot, env: process.env });
  let output = "";
  child.stdout.on("data", (chunk) => {
    output += chunk.toString();
    refreshState.lastOutput = output.slice(-4000);
  });
  child.stderr.on("data", (chunk) => {
    output += chunk.toString();
    refreshState.lastOutput = output.slice(-4000);
  });
  child.on("close", (code) => {
    refreshState.active = false;
    refreshState.lastFinishedAt = new Date().toISOString();
    refreshState.lastOutput = output.slice(-4000);
    if (code !== 0) refreshState.lastError = `full refresh exited ${code}`;
    try {
      pollSummaries();
      buildRunData();
      listHistory();
    } catch (error) {
      refreshState.lastError = error.message;
    }
  });
  return true;
}

function maybeScheduledRefresh() {
  if (refreshState.active) return;
  const last = refreshState.lastFinishedAt || refreshState.lastStartedAt;
  if (!last || Date.now() - Date.parse(last) >= fullRefreshMs) startFullRefresh("scheduled");
}

function safeFilePart(value) {
  return String(value || "sample")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "sample";
}

function ensureCandidateModel(ctx, candidate) {
  if (candidate.localModel) return path.resolve(repoRoot, candidate.localModel);
  if (!candidate.modelS3Uri) return "";
  const modelPath = path.join(ctx.workersDir, `${candidate.workerId}.nsrllm`);
  if (fs.existsSync(modelPath)) return modelPath;
  return cpS3(candidate.modelS3Uri, modelPath) ? modelPath : "";
}

function sampleConfig(ctx = activeContext) {
  return payloadConfig(ctx.runOptions.payloads?.[0]);
}

const simpleWikiContentDecodeDefaults = {
  corpus_prior_order: 3,
  corpus_prior_logit_shift: 7,
  repeat_window: 96,
  repeat_penalty_shift: 3,
  max_repeat_run: 2,
  no_repeat_ngram: 3,
  decode_frequency_cap: 2048,
  decode_frequency_min_q15: 2048,
  decode_frequency_logit_shift: 4,
  decode_local_frequency_cap: 2,
  decode_local_frequency_min_q15: 4096,
  decode_local_frequency_logit_shift: 4,
  decode_local_frequency_hard_cap: 2,
  prompt_topic_radius: 2,
  prompt_topic_min_q15: 4096,
  prompt_topic_logit_shift: 4,
};

function sampleDecodeConfig(ctx, cfg) {
  const recipe = String(cfg.sample_decode_recipe || "");
  const isSimpleWiki = /simplewiki/i.test(ctx.runOptions.runName || path.basename(ctx.runDir));
  const contentRecipe = recipe === "simplewiki-content" || recipe === "content";
  if (recipe === "classic" || (!isSimpleWiki && !contentRecipe)) return cfg;
  return { ...simpleWikiContentDecodeDefaults, ...cfg, sample_decode_recipe: recipe || "simplewiki-content" };
}

function appendOptionalIntArg(args, flag, value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric) && numeric > 0) args.push(flag, String(numeric));
}

function generateInteractiveSample(ctx, modelPath, target, prompt, options) {
  const seed = Number(options.seed || 17);
  const topK = Number(options.topK || 12);
  const maxNewTokens = Number(options.maxNewTokens || sampleMaxNewTokens);
  const cfg = sampleDecodeConfig(ctx, sampleConfig(ctx));
  fs.mkdirSync(ctx.interactiveSamplesDir, { recursive: true });
  const outPath = path.join(
    ctx.interactiveSamplesDir,
    `${Date.now()}-${safeFilePart(target)}-${safeFilePart(prompt)}-seed${seed}.txt`,
  );
  const args = [
    "run",
    "--release",
    "-q",
    "-p",
    "nsrl-train",
    "--",
    "--mode",
    "lexeme-generate",
    "--model",
    modelPath,
    "--vocab",
    vocab,
    "--tokens",
    tokens,
    "--prompt",
    prompt,
    "--max-new-tokens",
    String(maxNewTokens),
    "--decode-profile",
    options.decodeProfile || "coherent-prose",
    "--sample-seed",
    String(seed),
    "--top-k",
    String(topK),
    "--corpus-prior",
    "--corpus-prior-logit-shift",
    String(cfg.corpus_prior_logit_shift || 7),
    "--corpus-prior-order",
    String(cfg.corpus_prior_order || 2),
    "--repeat-window",
    String(cfg.repeat_window || 80),
    "--repeat-penalty-shift",
    String(cfg.repeat_penalty_shift || 3),
    "--max-repeat-run",
    String(cfg.max_repeat_run || 2),
    "--no-repeat-ngram",
    String(cfg.no_repeat_ngram || 3),
    "--generated-only",
    "--stop-on-sentence-terminal",
  ];
  appendOptionalIntArg(args, "--decode-frequency-cap", cfg.decode_frequency_cap);
  appendOptionalIntArg(args, "--decode-frequency-min-q15", cfg.decode_frequency_min_q15);
  appendOptionalIntArg(args, "--decode-frequency-logit-shift", cfg.decode_frequency_logit_shift);
  appendOptionalIntArg(args, "--decode-local-frequency-cap", cfg.decode_local_frequency_cap);
  appendOptionalIntArg(args, "--decode-local-frequency-min-q15", cfg.decode_local_frequency_min_q15);
  appendOptionalIntArg(args, "--decode-local-frequency-logit-shift", cfg.decode_local_frequency_logit_shift);
  appendOptionalIntArg(args, "--decode-local-frequency-hard-cap", cfg.decode_local_frequency_hard_cap);
  appendOptionalIntArg(args, "--prompt-topic-radius", cfg.prompt_topic_radius);
  appendOptionalIntArg(args, "--prompt-topic-min-q15", cfg.prompt_topic_min_q15);
  appendOptionalIntArg(args, "--prompt-topic-logit-shift", cfg.prompt_topic_logit_shift);
  args.push("--text-out", outPath);
  const result = run("cargo", args);
  if (result.status !== 0) {
    return { ok: false, target, error: result.stderr || result.stdout || `cargo exited ${result.status}` };
  }
  return {
    ok: true,
    target,
    prompt,
    seed,
    topK,
    maxNewTokens,
    decodeRecipe: cfg.sample_decode_recipe || "classic",
    path: path.relative(repoRoot, outPath),
    text: fs.existsSync(outPath) ? fs.readFileSync(outPath, "utf8").trim() : "",
  };
}

function handleSample(body) {
  const prompt = String(body.prompt || "the world is").trim();
  if (!prompt) return { ok: false, error: "prompt is required" };
  const selectedRunDir = resolveSelectableRunDir(body.runDir || runDir);
  if (path.resolve(selectedRunDir) !== path.resolve(runDir)) {
    return {
      ok: false,
      error: "interactive sampling is currently limited to the active run to avoid mixing vocabularies",
    };
  }
  const ctx = activeContext;
  pollSummaries(ctx);
  const data = buildRunData(ctx);
  const target = body.target || "best";
  const targets = [];
  if (target === "baseline" || body.includeBaseline) {
    if (currentModel && fs.existsSync(currentModel)) {
      targets.push({ id: "baseline", label: "baseline", modelPath: currentModel });
    }
  }
  if (target === "best") {
    const best = data.best ? data.candidates.find((candidate) => candidate.workerId === data.best.workerId) : null;
    if (best) targets.push({ id: best.workerId, label: best.label, candidate: best });
  } else if (target === "all") {
    for (const candidate of data.candidates.filter((item) => item.status === "succeeded")) {
      targets.push({ id: candidate.workerId, label: candidate.label, candidate });
    }
  } else if (target !== "baseline") {
    const candidate = data.candidates.find((item) => item.workerId === target || item.label === target);
    if (candidate) targets.push({ id: candidate.workerId, label: candidate.label, candidate });
  }
  const results = [];
  for (const item of targets) {
    const modelPath = item.modelPath || ensureCandidateModel(ctx, item.candidate);
    if (!modelPath) {
      results.push({ ok: false, target: item.label, error: "model is not downloaded yet" });
      continue;
    }
    results.push(generateInteractiveSample(ctx, modelPath, item.label, prompt, body));
  }
  return { ok: true, prompt, results, run: buildRunData(ctx) };
}

function html() {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NSRL Live Lexeme Dashboard</title>
  <style>
    :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7fa; color: #17202a; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px 20px 44px; }
    h1 { margin: 0 0 4px; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }
    button, input, select { font: inherit; }
    button { border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; color: #17202a; padding: 7px 10px; cursor: pointer; }
    button.primary { background: #174ea6; border-color: #174ea6; color: #ffffff; }
    button:disabled { cursor: wait; opacity: 0.58; }
    input, select { border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px 9px; background: #ffffff; color: #17202a; min-width: 0; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .sub { margin: 0 0 18px; color: #5d6773; font-size: 14px; overflow-wrap: anywhere; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 14px 0; }
    .stat, .panel, .sample-card { background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }
    .stat span { color: #667085; display: block; font-size: 12px; }
    .stat b { display: block; font-size: 22px; margin-top: 5px; }
    .progress { background: #e6ebf2; border-radius: 999px; height: 10px; overflow: hidden; margin: 14px 0 4px; }
    .bar { background: #257a4c; height: 100%; transition: width 220ms ease; width: 0; }
    table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }
    th, td { text-align: left; padding: 10px 11px; border-bottom: 1px solid #e8ecf2; font-size: 13px; vertical-align: top; }
    th { background: #eef2f7; color: #344054; font-weight: 650; }
    tr:last-child td { border-bottom: 0; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #ced6e0; background: #f8fafc; }
    .running { color: #865c00; }
    .done { color: #087443; }
    .pending { color: #475467; }
    .bad { color: #b42318; }
    .workbench { display: grid; grid-template-columns: minmax(260px, 0.34fr) minmax(0, 1fr); gap: 16px; align-items: start; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(340px, 0.8fr); gap: 16px; align-items: start; }
    .run-list { display: grid; gap: 8px; max-height: 720px; overflow: auto; }
    .run-row { width: 100%; text-align: left; display: grid; gap: 5px; border: 1px solid #d9dee7; border-radius: 8px; padding: 10px; background: #ffffff; color: inherit; }
    .run-row.selected { border-color: #174ea6; box-shadow: inset 3px 0 0 #174ea6; }
    .run-row-top { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
    .run-row b { overflow-wrap: anywhere; }
    .mini-progress { height: 7px; border-radius: 999px; background: #e6ebf2; overflow: hidden; }
    .mini-progress i { display: block; height: 100%; background: #257a4c; }
    .charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .chart { border: 1px solid #d9dee7; border-radius: 8px; padding: 12px; background: #ffffff; min-width: 0; }
    .chart h3 { margin: 0 0 10px; font-size: 13px; letter-spacing: 0; }
    .chart svg { display: block; width: 100%; height: 180px; }
    .bar-row { display: grid; grid-template-columns: minmax(92px, 0.54fr) minmax(0, 1fr) 70px; gap: 8px; align-items: center; margin: 6px 0; }
    .bar-track { height: 9px; border-radius: 999px; background: #e6ebf2; overflow: hidden; }
    .bar-track i { display: block; height: 100%; min-width: 2px; background: #2d6cdf; }
    .bar-track i.best { background: #257a4c; }
    .bar-track i.baseline { background: #865c00; }
    .sample-controls { display: grid; grid-template-columns: minmax(220px, 1fr) 76px 74px 96px; gap: 8px; margin-bottom: 10px; }
    .sample-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .sample-results { display: grid; gap: 10px; }
    .sample-card.best { border-color: #77b255; box-shadow: inset 3px 0 0 #77b255; }
    .sample-topline { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 8px; color: #687280; font-size: 13px; }
    .sample-title { color: #17202a; font-weight: 700; }
    .sample-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
    .sample-item { border-top: 1px solid #e8ecf2; padding-top: 10px; }
    .sample-item:first-child { border-top: 0; padding-top: 0; }
    .sample-label { color: #687280; font-size: 12px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }
    .tweet { margin: 0; font-size: 18px; line-height: 1.45; color: #17202a; overflow-wrap: anywhere; max-width: 76ch; }
    .muted { color: #667085; }
    @media (max-width: 940px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .workbench, .layout, .charts { grid-template-columns: 1fr; }
      .sample-grid, .sample-controls { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
      .topbar { align-items: flex-start; flex-direction: column; }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111418; color: #eef2f7; }
      button, input, select, .stat, .panel, .sample-card, table { background: #171b21; border-color: #2b3340; color: #eef2f7; }
      .run-row, .chart { background: #171b21; border-color: #2b3340; color: #eef2f7; }
      button.primary { background: #2d6cdf; border-color: #2d6cdf; color: #ffffff; }
      th { background: #202733; color: #e4e9f1; }
      td { border-color: #29313d; }
      .sub, .muted, .sample-topline, .sample-label { color: #abb3bf; }
      .sample-title, .tweet { color: #eef2f7; }
      .sample-item { border-color: #29313d; }
      .pill { background: #1d2430; border-color: #3a4554; }
      .progress, .mini-progress, .bar-track { background: #26303b; }
    }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1 id="title">NSRL Live Lexeme Dashboard</h1>
      <p class="sub" id="meta">Loading...</p>
    </div>
    <div class="toolbar">
      <button id="refreshBtn" class="primary">Refresh Scores</button>
      <button id="historyBtn">Refresh History</button>
    </div>
  </div>
  <div class="grid">
    <div class="stat"><span>Status</span><b id="status">...</b></div>
    <div class="stat"><span>Completed</span><b id="completed">...</b></div>
    <div class="stat"><span>Running</span><b id="running">...</b></div>
    <div class="stat"><span>Best Mean Bits/Token</span><b id="best">...</b></div>
    <div class="stat"><span>Baseline Mean</span><b id="baseline">...</b></div>
  </div>
  <div class="progress"><div id="bar" class="bar"></div></div>
  <p class="sub" id="refreshState"></p>
  <div class="workbench">
    <aside class="panel">
      <h2>Runs</h2>
      <div class="run-list" id="historyRows"></div>
    </aside>
    <section>
      <section class="panel">
        <h2>Charts</h2>
        <div class="charts" id="charts"></div>
      </section>
      <div class="layout" style="margin-top:16px">
        <section class="panel">
          <h2>Workers</h2>
          <table>
            <thead><tr><th>Candidate</th><th>Status</th><th>Progress</th><th>Recipe</th><th>Bits/Token</th><th>Samples</th><th></th></tr></thead>
            <tbody id="rows"></tbody>
          </table>
        </section>
        <section class="panel">
          <h2>Sample</h2>
          <div class="sample-controls">
            <input id="prompt" value="the world is" />
            <input id="seed" type="number" value="17" />
            <input id="topK" type="number" value="12" />
            <input id="maxNewTokens" type="number" value="96" />
          </div>
          <div class="sample-buttons">
            <button data-target="best" class="primary">Sample Best</button>
            <button data-target="all">Sample All Models</button>
            <button data-target="baseline">Sample Baseline</button>
          </div>
          <div id="sampleStatus" class="sub"></div>
          <div class="sample-results" id="sampleResults"></div>
        </section>
      </div>
      <section class="panel" style="margin-top:16px">
        <h2>Run Samples</h2>
        <div class="sample-results" id="runSamples"></div>
      </section>
    </section>
  </div>
</main>
<script>
const state = { run: null, history: [], selectedRunDir: null, sampling: false };
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function fmtMs(ms) {
  if (!ms) return '';
  const sec = Math.round(ms / 1000);
  if (sec < 60) return sec + 's';
  const min = Math.floor(sec / 60);
  return min + 'm ' + String(sec % 60).padStart(2, '0') + 's';
}
function statusClass(status) {
  return status === 'succeeded' ? 'done' : (status === 'failed' ? 'bad' : (status === 'running' ? 'running' : 'pending'));
}
function runApiUrl(runDir) {
  return '/api/run?ts=' + Date.now() + (runDir ? '&runDir=' + encodeURIComponent(runDir) : '');
}
async function refreshRun(runDir) {
  const data = await fetch(runApiUrl(runDir || state.selectedRunDir)).then(r => r.json());
  state.run = data;
  state.selectedRunDir = data.runDir;
  renderRun(data);
  renderHistory(state.history);
}
async function refreshHistory() {
  state.history = await fetch('/api/history?ts=' + Date.now()).then(r => r.json());
  renderHistory(state.history);
}
async function selectRun(runDir) {
  state.selectedRunDir = runDir;
  await refreshRun(runDir);
}
function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function scoreBits(evalResult) {
  if (!evalResult) return null;
  return num(evalResult.bits_per_token ?? evalResult.mean_bits_per_token);
}
function scoreText(evalResult) {
  const score = scoreBits(evalResult);
  if (score === null) return '';
  const worst = num(evalResult.max_bits_per_token);
  const suffix = worst === null ? '' : ' / worst ' + nice(worst, 3);
  return nice(score, 3) + suffix;
}
function nice(value, digits) {
  const n = num(value);
  if (n === null) return '';
  if (Math.abs(n) >= 1000000) return n.toExponential(2);
  return n.toFixed(digits ?? (Math.abs(n) < 10 ? 3 : 1)).replace(/\\.0+$/, '');
}
function barChart(title, bars, options) {
  const clean = (bars || []).filter(b => num(b.value) !== null);
  if (!clean.length) return '<div class="chart"><h3>' + esc(title) + '</h3><p class="muted">No chart data yet</p></div>';
  const max = Math.max(1, ...clean.map(b => Math.abs(Number(b.value))));
  return '<div class="chart"><h3>' + esc(title) + '</h3>' + clean.map(b => {
    const width = Math.max(2, Math.abs(Number(b.value)) * 100 / max);
    const cls = b.kind === 'best' ? 'best' : (b.kind === 'baseline' ? 'baseline' : '');
    return '<div class="bar-row">' +
      '<span title="' + esc(b.label) + '">' + esc(b.label) + '</span>' +
      '<span class="bar-track"><i class="' + cls + '" style="width:' + width.toFixed(1) + '%"></i></span>' +
      '<b>' + esc(options && options.format ? options.format(b.value) : nice(b.value)) + '</b>' +
      '</div>';
  }).join('') + '</div>';
}
function lineChart(title, points) {
  const clean = (points || []).filter(p => num(p.x) !== null && (num(p.before) !== null || num(p.after) !== null));
  if (!clean.length) return '<div class="chart"><h3>' + esc(title) + '</h3><p class="muted">No step data yet</p></div>';
  const width = 640, height = 180, pad = 28;
  const xs = clean.map(p => Number(p.x));
  const ys = clean.flatMap(p => [num(p.before), num(p.after)]).filter(v => v !== null);
  let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const sx = x => pad + (x - minX) * (width - pad * 2) / (maxX - minX);
  const sy = y => height - pad - (y - minY) * (height - pad * 2) / (maxY - minY);
  const poly = key => clean
    .map(p => [num(p.x), num(p[key])])
    .filter(([x, y]) => x !== null && y !== null)
    .map(([x, y]) => sx(x).toFixed(1) + ',' + sy(y).toFixed(1))
    .join(' ');
  return '<div class="chart"><h3>' + esc(title) + '</h3>' +
    '<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="' + esc(title) + '">' +
    '<line x1="' + pad + '" y1="' + (height - pad) + '" x2="' + (width - pad) + '" y2="' + (height - pad) + '" stroke="#8a94a3" opacity="0.35" />' +
    '<line x1="' + pad + '" y1="' + pad + '" x2="' + pad + '" y2="' + (height - pad) + '" stroke="#8a94a3" opacity="0.35" />' +
    '<polyline points="' + poly('before') + '" fill="none" stroke="#865c00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
    '<polyline points="' + poly('after') + '" fill="none" stroke="#257a4c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
    '<text x="' + pad + '" y="14" fill="currentColor" font-size="11">' + esc(nice(maxY, 0)) + '</text>' +
    '<text x="' + (width - pad) + '" y="' + (height - 5) + '" text-anchor="end" fill="currentColor" font-size="11">' + esc(nice(maxX, 0)) + '</text>' +
    '</svg><div class="muted">before / after target probability q15</div></div>';
}
function renderCharts(data) {
  const charts = data.charts || {};
  document.getElementById('charts').innerHTML = [
    barChart('Mean Bits/Token', charts.bitsPerToken || [], { format: v => nice(v, 3) }),
    barChart('Worst Validation Slice', charts.worstBitsPerToken || [], { format: v => nice(v, 3) }),
    barChart('Runtime Seconds', charts.runtimeSeconds || [], { format: v => nice(v, 1) + 's' }),
    barChart('Training Error Improvement', charts.errorImprovement || [], { format: v => nice(v, 0) }),
    barChart('Weight Movement L1', charts.movementL1 || [], { format: v => nice(v, 0) }),
    barChart('Accuracy Per Mille', charts.accuracyPerMille || [], { format: v => nice(v, 0) }),
    lineChart('Best Early Step Probability' + (charts.bestStepLabel ? ' · ' + charts.bestStepLabel : ''), charts.bestStepProbability || [])
  ].join('');
}
function renderRun(data) {
  document.getElementById('title').textContent = data.title || 'NSRL Live Lexeme Dashboard';
  document.getElementById('meta').textContent = data.runName + ' · ' + data.corpus + (data.context ? ' · ' + data.context : '') + ' · ' + data.runDir;
  document.getElementById('status').textContent = data.status;
  document.getElementById('completed').textContent = data.completed + '/' + data.total;
  document.getElementById('running').textContent = data.running;
  document.getElementById('best').textContent = data.best ? data.best.bitsPerToken.toFixed(3) : '...';
  document.getElementById('baseline').textContent = data.currentEval ? scoreText(data.currentEval) : '...';
  document.getElementById('bar').style.width = data.total ? ((data.completed + data.failed) / data.total * 100).toFixed(1) + '%' : '0';
  const refresh = data.refresh || {};
  const refreshText = !data.isActive
    ? 'Historical run selected · live refresh remains on the active run'
    : refresh.active
    ? 'Scoring refresh running since ' + new Date(refresh.lastStartedAt).toLocaleTimeString()
    : 'Last score refresh ' + (refresh.lastFinishedAt ? new Date(refresh.lastFinishedAt).toLocaleTimeString() : 'not finished yet') + (refresh.lastError ? ' · ' + refresh.lastError : '');
  document.getElementById('refreshState').textContent = refreshText + ' · live poll ' + new Date(data.updatedAt).toLocaleTimeString();
  document.getElementById('refreshBtn').disabled = !data.isActive || !!refresh.active;
  renderCharts(data);
  document.getElementById('rows').innerHTML = data.candidates.map(c => {
    const progress = c.status === 'running' ? ('running ' + fmtMs(c.runningMs)) : fmtMs(c.elapsedMs);
    const bits = c.eval ? scoreText(c.eval) : '';
    const sampleButton = data.isActive && c.status === 'succeeded' ? '<button data-target="' + esc(c.workerId) + '">Sample</button>' : '';
    return '<tr>' +
      '<td><b>' + esc(c.label) + '</b><br><code>' + esc(c.workerId) + '</code></td>' +
      '<td><span class="pill ' + statusClass(c.status) + '">' + esc(c.status) + '</span></td>' +
      '<td>' + esc(progress) + '</td>' +
      '<td>' + esc(c.details || '') + '</td>' +
      '<td>' + esc(bits) + '</td>' +
      '<td>' + esc(c.sampleCount || 0) + '</td>' +
      '<td>' + sampleButton + '</td>' +
      '</tr>';
  }).join('');
  document.querySelectorAll('.sample-buttons button[data-target]').forEach(button => {
    button.disabled = !data.isActive;
  });
  document.querySelectorAll('button[data-target]').forEach(button => {
    button.onclick = () => sample(button.dataset.target);
  });
  renderRunSamples(data);
}
function renderRunSamples(data) {
  const candidates = data.candidates.filter(c => c.samples && c.samples.length);
  document.getElementById('runSamples').innerHTML = candidates.map(c => {
    const isBest = data.best && data.best.workerId === c.workerId;
    return '<article class="sample-card' + (isBest ? ' best' : '') + '">' +
      '<div class="sample-topline"><span class="sample-title">' + esc(c.label) + ' · ' + esc(c.workerId) + '</span><span>' + esc(c.details || '') + (c.eval ? ' · ' + scoreText(c.eval) + ' bits/token' : '') + (isBest ? ' · best' : '') + '</span></div>' +
      '<div class="sample-grid">' + c.samples.map(s => '<div class="sample-item"><div class="sample-label">' + esc(s.label || s.prompt || 'sample') + (s.decodeRecipe ? ' · ' + esc(s.decodeRecipe) : '') + '</div><p class="tweet">' + esc(s.text || 'Waiting for sample...') + '</p></div>').join('') + '</div>' +
      '</article>';
  }).join('') || '<p class="muted">Waiting for worker samples...</p>';
}
function renderHistory(items) {
  document.getElementById('historyRows').innerHTML = items.map(item => {
    const best = item.best ? item.best.bitsPerToken.toFixed(3) + ' · ' + item.best.label : '';
    const baseline = item.baselineBits ? item.baselineBits.toFixed(3) : '';
    const pct = item.total ? Math.round((item.completed + item.failed) * 100 / item.total) : 0;
    const selected = item.runDir === state.selectedRunDir ? ' selected' : '';
    return '<button class="run-row' + selected + '" data-run-dir="' + esc(item.runDir) + '">' +
      '<span class="run-row-top"><b>' + esc(item.runName) + '</b><span class="pill ' + statusClass(item.status) + '">' + esc(item.status) + '</span></span>' +
      '<span class="mini-progress"><i style="width:' + Math.max(2, pct) + '%"></i></span>' +
      '<span class="muted">' + esc(item.completed) + '/' + esc(item.total || 0) + (best ? ' · best ' + esc(best) : '') + (baseline ? ' · base ' + esc(baseline) : '') + '</span>' +
      '<span class="muted">' + esc(item.corpus) + (item.context ? ' · ' + esc(item.context) : '') + '</span>' +
      '</button>';
  }).join('');
  document.querySelectorAll('[data-run-dir]').forEach(button => {
    button.onclick = () => selectRun(button.dataset.runDir);
  });
}
async function requestFullRefresh() {
  document.getElementById('refreshBtn').disabled = true;
  try {
    await fetch('/api/refresh', { method: 'POST' }).then(r => r.json());
    await refreshRun();
  } finally {
    document.getElementById('refreshBtn').disabled = false;
  }
}
async function sample(target) {
  if (state.sampling) return;
  state.sampling = true;
  document.getElementById('sampleStatus').textContent = 'Sampling ' + target + '...';
  document.querySelectorAll('button[data-target]').forEach(button => button.disabled = true);
  const body = {
    target,
    prompt: document.getElementById('prompt').value,
    seed: Number(document.getElementById('seed').value || 17),
    topK: Number(document.getElementById('topK').value || 12),
    maxNewTokens: Number(document.getElementById('maxNewTokens').value || 96),
    runDir: state.selectedRunDir,
  };
  try {
    const response = await fetch('/api/sample', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
    if (!response.ok) {
      document.getElementById('sampleStatus').textContent = response.error || 'sample failed';
      return;
    }
    const results = response.results || [];
    document.getElementById('sampleResults').innerHTML = results.map(result => {
      return '<article class="sample-card">' +
        '<div class="sample-topline"><span class="sample-title">' + esc(result.target) + '</span><span>seed ' + esc(result.seed || '') + (result.decodeRecipe ? ' · ' + esc(result.decodeRecipe) : '') + '</span></div>' +
        (result.ok ? '<p class="tweet">' + esc(result.text) + '</p>' : '<p class="bad">' + esc(result.error) + '</p>') +
        '</article>';
    }).join('');
    document.getElementById('sampleStatus').textContent = 'Sampled ' + results.length + ' target' + (results.length === 1 ? '' : 's') + '.';
    if (response.run) renderRun(response.run);
  } catch (error) {
    document.getElementById('sampleStatus').textContent = error.message;
  } finally {
    state.sampling = false;
    document.querySelectorAll('button[data-target]').forEach(button => button.disabled = false);
  }
}
document.getElementById('refreshBtn').onclick = requestFullRefresh;
document.getElementById('historyBtn').onclick = refreshHistory;
refreshRun();
refreshHistory();
setInterval(() => refreshRun(state.selectedRunDir), 3000);
setInterval(refreshHistory, 15000);
</script>
</body>
</html>`;
}

function sendJson(res, value, status = 200) {
  const body = JSON.stringify(value, null, 2);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(`${body}\n`);
}

function sendText(res, value, status = 200, contentType = "text/plain; charset=utf-8") {
  res.writeHead(status, { "content-type": contentType, "cache-control": "no-store" });
  res.end(value);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk.toString();
      if (body.length > 1024 * 1024) reject(new Error("request body too large"));
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(error);
      }
    });
  });
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "127.0.0.1"}`);
  try {
    if (req.method === "GET" && url.pathname === "/") {
      sendText(res, html(), 200, "text/html; charset=utf-8");
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/run") {
      const selectedRunDir = resolveSelectableRunDir(url.searchParams.get("runDir") || runDir);
      const ctx = contextForRunDir(selectedRunDir);
      pollSummaries(ctx);
      if (ctx.isActive) {
        maybeScheduledRefresh();
      }
      sendJson(res, buildRunData(ctx));
      return;
    }
    if (req.method === "GET" && url.pathname === "/api/history") {
      sendJson(res, listHistory());
      return;
    }
    if (req.method === "POST" && url.pathname === "/api/refresh") {
      sendJson(res, { ok: true, started: startFullRefresh("manual"), refresh: refreshState });
      return;
    }
    if (req.method === "POST" && url.pathname === "/api/sample") {
      const body = await readBody(req);
      sendJson(res, handleSample(body));
      return;
    }
    if (req.method === "GET" && ["/runs.json", "/live-state.json"].includes(url.pathname)) {
      pollSummaries(activeContext);
      sendJson(res, buildRunData(activeContext));
      return;
    }
    if (req.method === "GET" && url.pathname === "/history.json") {
      sendJson(res, listHistory());
      return;
    }
    sendJson(res, { ok: false, error: "not found" }, 404);
  } catch (error) {
    sendJson(res, { ok: false, error: error.message }, 500);
  }
});

pollSummaries(activeContext);
buildRunData(activeContext);
listHistory();
if (startupRefresh) {
  startFullRefresh("startup");
} else {
  refreshState.lastFinishedAt = new Date().toISOString();
  refreshState.lastReason = "startup skipped";
}
setInterval(() => {
  try {
    pollSummaries(activeContext);
    buildRunData(activeContext);
    maybeScheduledRefresh();
  } catch (error) {
    refreshState.lastError = error.message;
  }
}, pollMs);

server.listen(port, "127.0.0.1", () => {
  console.log(`serving ${runOptions.runName || path.basename(runDir)} at http://127.0.0.1:${port}/`);
});
