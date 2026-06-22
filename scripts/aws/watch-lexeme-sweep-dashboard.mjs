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

function evalModel(modelPath, seqLen) {
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
    "0",
    "--max-windows",
    "32768",
  ]);
  if (result.status !== 0) return null;
  try {
    return JSON.parse(result.stdout).eval || null;
  } catch {
    return null;
  }
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
    .tweet { margin: 0; font-size: 20px; line-height: 1.45; color: #17202a; overflow-wrap: anywhere; }
    .model-path { margin-top: 12px; color: #687280; font-size: 12px; overflow-wrap: anywhere; }
    .running { color: #865c00; }
    .done { color: #087443; }
    .pending { color: #475467; }
    .bad { color: #b42318; }
    @media (max-width: 780px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
      .sample-topline, .model-path { color: #abb3bf; }
    }
  </style>
</head>
<body>
<main>
  <h1>NSRL Lexeme Expanded Corpus Sweep</h1>
  <p class="sub" id="meta">Loading...</p>
  <div class="grid">
    <div class="stat"><span>Status</span><b id="status">...</b></div>
    <div class="stat"><span>Completed</span><b id="completed">...</b></div>
    <div class="stat"><span>Best Bits/Token</span><b id="best">...</b></div>
    <div class="stat"><span>Current Baseline</span><b id="baseline">...</b></div>
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
  document.getElementById('meta').textContent = data.runName + ' · refresh ' + new Date(data.updatedAt).toLocaleTimeString();
  document.getElementById('status').textContent = data.status;
  document.getElementById('completed').textContent = data.completed + '/' + data.total;
  document.getElementById('best').textContent = data.best ? data.best.bitsPerToken.toFixed(3) : '...';
  document.getElementById('baseline').textContent = data.currentEval ? data.currentEval.bits_per_token.toFixed(3) : '...';
  document.getElementById('rows').innerHTML = data.candidates.map(c => {
    const cls = c.status === 'succeeded' ? 'done' : (c.status === 'failed' ? 'bad' : (c.status === 'running' ? 'running' : 'pending'));
    const bits = c.eval ? c.eval.bits_per_token.toFixed(3) : '';
    const runtime = c.elapsedMs ? (c.elapsedMs / 1000).toFixed(1) + 's' : '';
    return '<tr>' +
      '<td><b>' + esc(c.label) + '</b><br><code>' + esc(c.workerId) + '</code></td>' +
      '<td><span class="pill ' + cls + '">' + esc(c.status) + '</span></td>' +
      '<td>' + esc(runtime) + '</td>' +
      '<td>' + esc(c.seqLen || '') + '</td>' +
      '<td>' + esc(bits) + '</td>' +
      '<td><code>' + esc(c.localModel || '') + '</code></td>' +
      '</tr>';
  }).join('');
  document.getElementById('sampleCards').innerHTML = data.candidates.map(c => {
    const bits = c.eval ? c.eval.bits_per_token.toFixed(3) : '...';
    const isBest = data.best && data.best.workerId === c.workerId;
    return '<article class="sample-card' + (isBest ? ' best' : '') + '">' +
      '<div class="sample-topline"><span class="sample-title">' + esc(c.label) + ' · ' + esc(c.workerId) + '</span><span>seq ' + esc(c.seqLen || '') + ' · ' + esc(bits) + ' bits/token' + (isBest ? ' · best' : '') + '</span></div>' +
      '<p class="tweet">' + esc(c.worldSample || 'Waiting for sample...') + '</p>' +
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
  const currentEvalPath = path.join(dashboardDir, `current-eval-seq${currentSeqLen}.json`);
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
    if (summary) {
      status = summary.ok ? "succeeded" : "failed";
      elapsedMs = summary.elapsed_ms || null;
      seqLen = payloadSeqLen(item, summary);
      localModel = path.join(workersDir, `${workerId}.nsrllm`);
      if (summary.ok && !fs.existsSync(localModel)) {
        cpS3(summary.model_s3_uri, localModel);
      }
      const evalPath = path.join(workersDir, `${workerId}.seq${seqLen}.eval.json`);
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
      status,
      elapsedMs,
      eval: evalResult,
      worldSample,
      localModel: fs.existsSync(localModel) ? path.relative(repoRoot, localModel) : "",
    });
  }
  const complete = candidates.filter((candidate) => candidate.status === "succeeded").length;
  const failed = candidates.filter((candidate) => candidate.status === "failed").length;
  const scored = candidates.filter((candidate) => candidate.eval);
  const best = scored.sort((a, b) => a.eval.bits_per_token - b.eval.bits_per_token)[0] || null;
  const data = {
    schema: "nsrl.lexeme_sweep_dashboard.v1",
    runName: options.runName,
    status: failed ? "failed" : complete === candidates.length ? "succeeded" : "running",
    updatedAt: new Date().toISOString(),
    s3Prefix,
    total: candidates.length,
    completed: complete,
    failed,
    currentSeqLen,
    currentEval,
    best: best ? { label: best.label, workerId: best.workerId, bitsPerToken: best.eval.bits_per_token } : null,
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
