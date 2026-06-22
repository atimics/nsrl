#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");

function arg(name, fallback = "") {
  const index = process.argv.indexOf(`--${name}`);
  if (index >= 0 && index + 1 < process.argv.length) return process.argv[index + 1];
  const envName = `NSRL_${name.replaceAll("-", "_").toUpperCase()}`;
  return process.env[envName] || fallback;
}

const runDir = path.resolve(repoRoot, arg("run-dir"));
const s3Prefix = arg("s3-prefix").replace(/\/+$/, "");
const intervalMs = Number(arg("interval-ms", "10000"));
const once = process.argv.includes("--once");
const currentModel = path.resolve(repoRoot, arg("current-model", "data/processed/visionary-twitter-bot-demo/v4096.nsrllm"));
const tokens = path.resolve(repoRoot, arg("tokens", "data/processed/visionary-expanded-frozen-v4096/v4096.tokens.u16"));
const vocab = path.resolve(repoRoot, arg("vocab", "data/processed/visionary-expanded-frozen-v4096/v4096.vocab.tsv"));
const sampleMaxNewTokens = Number(arg("sample-max-new-tokens", "96"));
const evalOffsets = parseOffsets(arg("eval-offsets", "0"));
const evalMaxWindows = Number(arg("eval-max-windows", "32768"));
const dashboardDir = path.join(runDir, "dashboard");
const workersDir = path.join(runDir, "workers");
const samplesDir = path.join(runDir, "samples");

fs.mkdirSync(dashboardDir, { recursive: true });
fs.mkdirSync(workersDir, { recursive: true });
fs.mkdirSync(samplesDir, { recursive: true });

function run(cmd, args, options = {}) {
  return spawnSync(cmd, args, {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: options.stdio || ["ignore", "pipe", "pipe"],
  });
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function cpS3(uri, file) {
  const result = run("aws", [
    "--profile",
    process.env.NSRL_AWS_PROFILE || process.env.AWS_PROFILE || "staging",
    "--region",
    process.env.NSRL_AWS_REGION || process.env.AWS_REGION || "us-east-1",
    "s3",
    "cp",
    uri,
    file,
    "--only-show-errors",
  ]);
  return result.status === 0;
}

function parseOffsets(value) {
  const offsets = String(value || "0")
    .split(/[,\s]+/)
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isInteger(value) && value >= 0);
  return offsets.length ? [...new Set(offsets)] : [0];
}

function evalCacheSuffix(seqLen) {
  const offsets = evalOffsets.join("-");
  return evalOffsets.length === 1 && evalOffsets[0] === 0
    ? `seq${seqLen}.eval.json`
    : `seq${seqLen}.offsets-${offsets}.mw${evalMaxWindows}.eval.json`;
}

function evalScore(result) {
  if (!result) return null;
  return Number(result.bits_per_token ?? result.mean_bits_per_token ?? NaN);
}

function evalModelAtOffset(modelPath, seqLen, offset) {
  const result = run("cargo", [
    "run",
    "--release",
    "-q",
    "-p",
    "nsrl-train",
    "--",
    "--mode",
    "lexeme-evaluate",
    "--tokens",
    tokens,
    "--model",
    modelPath,
    "--seq-len",
    String(seqLen),
    "--stride",
    "1",
    "--window-offset",
    String(offset),
    "--max-windows",
    String(evalMaxWindows),
  ]);
  if (result.status !== 0) return null;
  try {
    const evalResult = JSON.parse(result.stdout).eval || null;
    return evalResult ? { offset, ...evalResult } : null;
  } catch {
    return null;
  }
}

function evalModel(modelPath, seqLen) {
  const rows = evalOffsets.map((offset) => evalModelAtOffset(modelPath, seqLen, offset)).filter(Boolean);
  if (!rows.length) return null;
  if (rows.length === 1 && evalOffsets.length === 1 && evalOffsets[0] === 0) {
    const { offset: _offset, ...single } = rows[0];
    return single;
  }
  const bits = rows.map((row) => Number(row.bits_per_token)).filter(Number.isFinite);
  if (!bits.length) return null;
  const mean = bits.reduce((sum, value) => sum + value, 0) / bits.length;
  const best = rows.reduce((winner, row) => (row.bits_per_token < winner.bits_per_token ? row : winner), rows[0]);
  const worst = rows.reduce((winner, row) => (row.bits_per_token > winner.bits_per_token ? row : winner), rows[0]);
  const uniform = rows.find((row) => Number.isFinite(Number(row.uniform_bits_per_token)))?.uniform_bits_per_token;
  return {
    schema: "nsrl.lexeme_eval_offset_panel.v1",
    offsets: evalOffsets,
    max_windows: evalMaxWindows,
    windows: rows.reduce((sum, row) => sum + Number(row.windows || 0), 0),
    vocab_size: rows[0].vocab_size,
    bits_per_token: Number(mean.toFixed(3)),
    mean_bits_per_token: Number(mean.toFixed(3)),
    min_bits_per_token: Number(best.bits_per_token.toFixed(3)),
    max_bits_per_token: Number(worst.bits_per_token.toFixed(3)),
    best_offset: best.offset,
    worst_offset: worst.offset,
    uniform_bits_per_token: uniform,
    reduction_vs_uniform: Number.isFinite(Number(uniform)) ? Number((uniform - mean).toFixed(3)) : undefined,
    rows,
  };
}

function generateSample(modelPath, label, prompt, seed) {
  const outPath = path.join(samplesDir, `${label}.txt`);
  const result = run("cargo", [
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
    String(sampleMaxNewTokens),
    "--decode-profile",
    "coherent-prose",
    "--sample-seed",
    String(seed),
    "--top-k",
    "12",
    "--corpus-prior",
    "--corpus-prior-logit-shift",
    "7",
    "--corpus-prior-order",
    "2",
    "--repeat-window",
    "80",
    "--repeat-penalty-shift",
    "3",
    "--max-repeat-run",
    "2",
    "--no-repeat-ngram",
    "3",
    "--generated-only",
    "--stop-on-sentence-terminal",
    "--text-out",
    outPath,
  ]);
  if (result.status !== 0 || !fs.existsSync(outPath)) return "";
  return fs.readFileSync(outPath, "utf8").trim();
}

function payloadConfig(item) {
  return item?.payload?.config || {};
}

function payloadSeqLen(item, summary = null) {
  return Number(summary?.config?.softmax_seq_len || payloadConfig(item).softmax_seq_len || item.seqLen || 8);
}

function baselineSeqLen(options) {
  const seqLens = new Set((options.payloads || []).map((item) => payloadSeqLen(item)).filter(Boolean));
  if (seqLens.size === 1) return [...seqLens][0];
  return Number(arg("eval-seq-len", "8"));
}

function titleFromRunName(runName) {
  if (runName.includes("simplewiki")) return "NSRL SimpleWiki Boring-English Optimizer";
  if (runName.includes("crowley")) return "NSRL Crowley Lexeme Sweep";
  if (runName.includes("visionary")) return "NSRL Visionary Lexeme Sweep";
  return "NSRL Lexeme Sweep";
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
  if (cfg.train_embeddings) {
    parts.push(`embed lr ${cfg.embedding_lr_shift ?? ""}`.trim());
  }
  const hiddenDim = Number(cfg.hidden_dim || 0);
  if (hiddenDim) parts.push(`hidden ${hiddenDim}`);
  const context = cfg.lexeme_context_features || "";
  if (context) parts.push(context);
  return parts.join(" · ");
}

function renderHtml() {
  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NSRL Lexeme Sweep</title>
  <style>
    :root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #17202a; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px 20px 40px; }
    h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: 0; }
    .sub { margin: 0 0 20px; color: #5d6773; font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }
    .stat, .panel { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }
    .stat b { display: block; font-size: 24px; margin-top: 6px; }
    .stat span { color: #687280; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }
    th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #e8ecf2; font-size: 13px; vertical-align: top; }
    th { background: #eef2f7; color: #344054; font-weight: 650; }
    tr:last-child td { border-bottom: 0; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #ced6e0; background: #f8fafc; }
    .sample-board { display: grid; gap: 12px; margin-top: 16px; }
    .sample-card { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 16px 18px; }
    .sample-card.best { border-color: #77b255; box-shadow: inset 3px 0 0 #77b255; }
    .sample-topline { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px; color: #687280; font-size: 13px; }
    .sample-title { color: #17202a; font-weight: 700; }
    .sample-grid { display: grid; grid-template-columns: 1fr; gap: 10px; }
    .sample-item { border-top: 1px solid #e8ecf2; padding-top: 10px; }
    .sample-item:first-child { border-top: 0; padding-top: 0; }
    .sample-label { color: #687280; font-size: 12px; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }
    .tweet { margin: 0; font-size: 20px; line-height: 1.45; color: #17202a; overflow-wrap: anywhere; max-width: 76ch; }
    .model-path { margin-top: 12px; color: #687280; font-size: 12px; overflow-wrap: anywhere; }
    .running { color: #865c00; }
    .done { color: #087443; }
    .pending { color: #475467; }
    .bad { color: #b42318; }
    @media (max-width: 780px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .sample-grid { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
      .tweet { font-size: 17px; }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #111418; color: #eef2f7; }
      .sub { color: #abb3bf; }
      .stat, .panel, table, .sample-card { background: #171b21; border-color: #2b3340; }
      th { background: #202733; color: #e4e9f1; }
      td { border-color: #29313d; }
      .pill { background: #1d2430; border-color: #3a4554; }
      .sample-title, .tweet { color: #eef2f7; }
      .sample-item { border-color: #29313d; }
      .sample-topline, .sample-label, .model-path { color: #abb3bf; }
    }
  </style>
</head>
<body>
<main>
  <h1 id="title">NSRL Lexeme Sweep</h1>
  <p class="sub" id="meta">Loading...</p>
  <div class="grid">
    <div class="stat"><span>Status</span><b id="status">...</b></div>
    <div class="stat"><span>Completed</span><b id="completed">...</b></div>
    <div class="stat"><span>Best Mean Bits/Token</span><b id="best">...</b></div>
    <div class="stat"><span>Baseline Mean</span><b id="baseline">...</b></div>
  </div>
  <table>
    <thead><tr><th>Candidate</th><th>Status</th><th>Runtime</th><th>Context</th><th>Bits/Token</th><th>Model</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <section class="sample-board" id="sampleCards"></section>
</main>
<script>
function esc(value) {
  return String(value || '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
async function refresh() {
  const data = await fetch('runs.json?ts=' + Date.now()).then(r => r.json());
  document.getElementById('title').textContent = data.title || 'NSRL Lexeme Sweep';
  document.getElementById('meta').textContent = data.runName + ' · refresh ' + new Date(data.updatedAt).toLocaleTimeString();
  document.getElementById('status').textContent = data.status;
  document.getElementById('completed').textContent = data.completed + '/' + data.total;
  document.getElementById('best').textContent = data.best ? data.best.bitsPerToken.toFixed(3) : '...';
  document.getElementById('baseline').textContent = data.currentEval ? data.currentEval.bits_per_token.toFixed(3) : '...';
  document.getElementById('rows').innerHTML = data.candidates.map(c => {
    const cls = c.status === 'succeeded' ? 'done' : (c.status === 'failed' ? 'bad' : (c.status === 'running' ? 'running' : 'pending'));
    const bits = c.eval ? c.eval.bits_per_token.toFixed(3) + (c.eval.max_bits_per_token ? ' / worst ' + c.eval.max_bits_per_token.toFixed(3) : '') : '';
    const runtime = c.elapsedMs ? (c.elapsedMs / 1000).toFixed(1) + 's' : '';
    return '<tr>' +
      '<td><b>' + esc(c.label) + '</b><br><code>' + esc(c.workerId) + '</code></td>' +
      '<td><span class="pill ' + cls + '">' + esc(c.status) + '</span></td>' +
      '<td>' + esc(runtime) + '</td>' +
      '<td>' + esc(c.details || c.seqLen || '') + '</td>' +
      '<td>' + esc(bits) + '</td>' +
      '<td><code>' + esc(c.localModel || '') + '</code></td>' +
      '</tr>';
  }).join('');
  document.getElementById('sampleCards').innerHTML = data.candidates.map(c => {
    const bits = c.eval ? c.eval.bits_per_token.toFixed(3) : '...';
    const isBest = data.best && data.best.workerId === c.workerId;
    const samples = c.samples && c.samples.length ? c.samples : [{ label: 'world', prompt: 'the world is', text: c.worldSample || 'Waiting for sample...' }];
    return '<article class="sample-card' + (isBest ? ' best' : '') + '">' +
      '<div class="sample-topline"><span class="sample-title">' + esc(c.label) + ' · ' + esc(c.workerId) + '</span><span>' + esc(c.details || ('seq ' + (c.seqLen || ''))) + ' · ' + esc(bits) + ' bits/token' + (isBest ? ' · best' : '') + '</span></div>' +
      '<div class="sample-grid">' + samples.map(s => '<div class="sample-item"><div class="sample-label">' + esc(s.label || s.prompt || 'sample') + '</div><p class="tweet">' + esc(s.text || 'Waiting for sample...') + '</p></div>').join('') + '</div>' +
      '<div class="model-path"><code>' + esc(c.localModel || '') + '</code></div>' +
      '</article>';
  }).join('');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
`;
  fs.writeFileSync(path.join(dashboardDir, "index.html"), html);
}

function update() {
  const options = readJson(path.join(runDir, "run-options.json"));
  if (!options) throw new Error(`missing ${path.join(runDir, "run-options.json")}`);
  const currentSeqLen = baselineSeqLen(options);
  const currentEvalPath = path.join(dashboardDir, `current-eval-${evalCacheSuffix(currentSeqLen)}`);
  let currentEval = readJson(currentEvalPath);
  if (!currentEval) {
    currentEval = evalModel(currentModel, currentSeqLen);
    if (currentEval) fs.writeFileSync(currentEvalPath, `${JSON.stringify(currentEval, null, 2)}\n`);
  }

  const candidates = [];
  for (const item of options.payloads || []) {
    const workerId = item.workerId;
    const label = item.label;
    const summaryPath = path.join(workersDir, `${workerId}.summary.json`);
    cpS3(`${s3Prefix}/workers/${workerId}.summary.json`, summaryPath);
    const summary = readJson(summaryPath);
    let status = "running";
    let evalResult = null;
    let worldSample = "";
    let localModel = "";
    let elapsedMs = null;
    let seqLen = payloadSeqLen(item);
    let samples = [];
    let details = candidateDetails(item);
    if (summary) {
      status = summary.ok ? "succeeded" : "failed";
      elapsedMs = summary.elapsed_ms || null;
      seqLen = payloadSeqLen(item, summary);
      details = candidateDetails(item, summary);
      samples = Array.isArray(summary.samples) ? summary.samples.map((sample) => ({
        label: sample.label || sample.prompt || "sample",
        prompt: sample.prompt || "",
        text: sample.text || "",
      })) : [];
      localModel = path.join(workersDir, `${workerId}.nsrllm`);
      if (summary.ok && !fs.existsSync(localModel)) {
        cpS3(summary.model_s3_uri, localModel);
      }
      const evalPath = path.join(workersDir, `${workerId}.${evalCacheSuffix(seqLen)}`);
      evalResult = readJson(evalPath);
      if (!evalResult && fs.existsSync(localModel)) {
        evalResult = evalModel(localModel, seqLen);
        if (evalResult) fs.writeFileSync(evalPath, `${JSON.stringify(evalResult, null, 2)}\n`);
      }
      const samplePath = path.join(samplesDir, `${workerId}.world.local.txt`);
      if (!fs.existsSync(samplePath) && fs.existsSync(localModel)) {
        worldSample = generateSample(localModel, `${workerId}.world.local`, "the world is", 17);
        fs.writeFileSync(samplePath, `${worldSample}\n`);
      } else if (fs.existsSync(samplePath)) {
        worldSample = fs.readFileSync(samplePath, "utf8").trim();
      }
    }
    candidates.push({
      workerId,
      label,
      seqLen,
      maxWindows: item.maxWindows,
      lrShift: item.lrShift,
      details,
      status,
      elapsedMs,
      eval: evalResult,
      worldSample,
      samples,
      localModel: fs.existsSync(localModel) ? path.relative(repoRoot, localModel) : "",
    });
  }
  const complete = candidates.filter((candidate) => candidate.status === "succeeded").length;
  const failed = candidates.filter((candidate) => candidate.status === "failed").length;
  const scored = candidates.filter((candidate) => candidate.eval);
  const best = scored.sort((a, b) => evalScore(a.eval) - evalScore(b.eval))[0] || null;
  const data = {
    schema: "nsrl.lexeme_sweep_dashboard.v1",
    title: titleFromRunName(options.runName || ""),
    runName: options.runName,
    status: failed ? "failed" : complete === candidates.length ? "succeeded" : "running",
    updatedAt: new Date().toISOString(),
    s3Prefix,
    total: candidates.length,
    completed: complete,
    failed,
    currentSeqLen,
    evalOffsets,
    evalMaxWindows,
    currentEval,
    best: best ? { label: best.label, workerId: best.workerId, bitsPerToken: evalScore(best.eval) } : null,
    candidates,
  };
  fs.writeFileSync(path.join(dashboardDir, "runs.json"), `${JSON.stringify(data, null, 2)}\n`);
  renderHtml();
  console.log(`${data.updatedAt} ${data.status} ${complete}/${candidates.length}${data.best ? ` best=${data.best.label}:${data.best.bitsPerToken}` : ""}`);
  return data.status !== "running";
}

do {
  const done = update();
  if (once || done) break;
  await new Promise((resolve) => setTimeout(resolve, intervalMs));
} while (true);
