#!/usr/bin/env python3
"""Render a static S3 dashboard for NSRL cloud training runs."""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import time
from typing import Any


def load_json_line(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline().strip()
    if not line:
        return None
    return json.loads(line)


def read_text(path: pathlib.Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def tail_lines(path: pathlib.Path | None, limit: int) -> list[str]:
    text = read_text(path)
    if not text:
        return []
    return text.splitlines()[-limit:]


def artifact(name: str, path: pathlib.Path | None, run_s3_uri: str) -> dict[str, Any] | None:
    if path is None:
        return None
    exists = path.exists()
    return {
        "name": name,
        "file": path.name,
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "s3_uri": f"{run_s3_uri.rstrip('/')}/{path.name}",
    }


def trace_summary(trace: dict[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    data = trace.get("data", {}) if isinstance(trace.get("data"), dict) else {}
    model = trace.get("model", {}) if isinstance(trace.get("model"), dict) else {}
    optimizer = trace.get("optimizer", {}) if isinstance(trace.get("optimizer"), dict) else {}
    training = trace.get("training", {}) if isinstance(trace.get("training"), dict) else {}
    metrics = trace.get("metrics", {}) if isinstance(trace.get("metrics"), dict) else {}
    examined = training.get("examined_windows")
    windows = data.get("windows")
    progress_per_mille = None
    if isinstance(examined, int) and isinstance(windows, int) and windows > 0:
        progress_per_mille = min(1000, examined * 1000 // windows)
    return {
        "schema": trace.get("schema"),
        "task": trace.get("task"),
        "token_count": data.get("token_count"),
        "windows": data.get("windows"),
        "token_hash": data.get("token_hash"),
        "seq_len": model.get("seq_len") or training.get("seq_len"),
        "d_model": model.get("d_model"),
        "heads": model.get("heads"),
        "attention_kind": model.get("attention_kind"),
        "position": model.get("position"),
        "max_windows": training.get("max_windows"),
        "examined_windows": training.get("examined_windows"),
        "progress_per_mille": progress_per_mille,
        "batch_windows": training.get("batch_windows"),
        "stride": training.get("stride"),
        "window_offset": training.get("window_offset"),
        "adaptive_rule_shifts": optimizer.get("adaptive_rule_shifts"),
        "adaptive_holographic_shifts": optimizer.get("adaptive_holographic_shifts"),
        "probability_error_delta_i32": metrics.get("probability_error_delta_i32"),
        "final_accuracy_per_mille": metrics.get("final_accuracy_per_mille"),
        "rollback_count": metrics.get("rollback_count"),
        "rejected_batch_count": metrics.get("rejected_batch_count"),
        "final_invalid_forward_count": metrics.get("final_invalid_forward_count"),
        "adaptive_rule_shift_adjustment_count": metrics.get(
            "adaptive_rule_shift_adjustment_count"
        ),
        "adaptive_holographic_shift_adjustment_count": metrics.get(
            "adaptive_holographic_shift_adjustment_count"
        ),
        "output_head_delta_l1": metrics.get("output_head_delta_l1"),
        "mlp_delta_l1": metrics.get("mlp_delta_l1"),
        "embedding_delta_l1": metrics.get("embedding_delta_l1"),
        "attention_delta_l1": metrics.get("attention_delta_l1"),
        "attention_q_delta_l1": metrics.get("attention_q_delta_l1"),
        "attention_k_delta_l1": metrics.get("attention_k_delta_l1"),
        "attention_v_delta_l1": metrics.get("attention_v_delta_l1"),
        "attention_o_delta_l1": metrics.get("attention_o_delta_l1"),
        "final_output_learning_rate_shift": metrics.get(
            "final_output_learning_rate_shift",
            metrics.get("current_output_learning_rate_shift"),
        ),
        "final_mlp_learning_rate_shift": metrics.get(
            "final_mlp_learning_rate_shift",
            metrics.get("current_mlp_learning_rate_shift"),
        ),
        "final_embedding_learning_rate_shift": metrics.get(
            "final_embedding_learning_rate_shift",
            metrics.get("current_embedding_learning_rate_shift"),
        ),
        "final_attention_learning_rate_shift": metrics.get(
            "final_attention_learning_rate_shift",
            metrics.get("current_attention_learning_rate_shift"),
        ),
        "final_attention_q_learning_rate_shift": metrics.get(
            "final_attention_q_learning_rate_shift",
            metrics.get("current_attention_q_learning_rate_shift"),
        ),
        "final_attention_qk_learning_rate_shift": metrics.get(
            "final_attention_qk_learning_rate_shift",
            metrics.get("current_attention_qk_learning_rate_shift"),
        ),
    }


def load_runs(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def render_index(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NSRL Cloud Training</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181b1f;
      --line: #2a3037;
      --text: #edf1f5;
      --muted: #9aa7b4;
      --good: #72d58a;
      --warn: #ffd166;
      --bad: #ff6b6b;
      --accent: #7cc7ff;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 24px;
      border-bottom: 1px solid var(--line);
      background: #121519;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 22px;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); }
    main { padding: 24px; }
    .summary {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin-bottom: 18px;
    }
    .metric, .run {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric { padding: 14px; }
    .metric b {
      display: block;
      font-size: 20px;
      margin-top: 4px;
    }
    .runs {
      display: grid;
      gap: 14px;
    }
    .run {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .run-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .run h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 9px;
      color: var(--muted);
      white-space: nowrap;
    }
    .running { color: var(--warn); }
    .succeeded { color: var(--good); }
    .failed { color: var(--bad); }
    .grid {
      display: grid;
      gap: 8px 14px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .kv span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .kv b {
      font-size: 15px;
      overflow-wrap: anywhere;
    }
    details {
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }
    summary { cursor: pointer; color: var(--accent); }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d7dee6;
      background: #0d0f12;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      max-height: 260px;
      overflow: auto;
    }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <h1>NSRL Cloud Training</h1>
    <div class="subtle" id="updated">Loading runs...</div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="runs" id="runs"></section>
  </main>
  <script>
    const fmt = value => value === null || value === undefined ? "—" : String(value);
    const statusClass = status => String(status || "unknown").replace(/[^a-z0-9_-]/gi, "").toLowerCase();
    const artifactLink = artifact => {
      if (!artifact || !artifact.exists) return "—";
      return `<a href="${artifact.file}">${artifact.name}</a>`;
    };
    function kv(label, value) {
      return `<div class="kv"><span>${label}</span><b>${fmt(value)}</b></div>`;
    }
    function render(runs) {
      runs.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
      const running = runs.filter(r => r.status === "running").length;
      const succeeded = runs.filter(r => r.status === "succeeded").length;
      const failed = runs.filter(r => r.status === "failed").length;
      document.getElementById("updated").textContent =
        `Last refreshed ${new Date().toLocaleString()} · ${runs.length} run(s)`;
      document.getElementById("summary").innerHTML = [
        ["Running", running],
        ["Succeeded", succeeded],
        ["Failed", failed],
        ["Latest", runs[0]?.run_name || "—"]
      ].map(([label, value]) => `<div class="metric"><span class="subtle">${label}</span><b>${value}</b></div>`).join("");
      document.getElementById("runs").innerHTML = runs.map(run => {
        const m = run.metrics || {};
        const artifacts = run.artifacts || {};
        return `<article class="run">
          <div class="run-head">
            <h2>${run.run_name}</h2>
            <span class="pill ${statusClass(run.status)}">${run.status || "unknown"}</span>
          </div>
          <div class="grid">
            ${kv("started", run.started_at)}
            ${kv("updated", run.updated_at)}
            ${kv("elapsed_s", run.elapsed_seconds)}
            ${kv("exit", run.exit_code)}
            ${kv("prob_delta", m.probability_error_delta_i32)}
            ${kv("accuracy_per_mille", m.final_accuracy_per_mille)}
            ${kv("rollbacks", m.rollback_count)}
            ${kv("rejected_batches", m.rejected_batch_count)}
            ${kv("invalid_forward", m.final_invalid_forward_count)}
            ${kv("seq_len", m.seq_len)}
            ${kv("max_windows", m.max_windows)}
            ${kv("examined_windows", m.examined_windows)}
            ${kv("progress_per_mille", m.progress_per_mille)}
            ${kv("attention", m.attention_kind)}
            ${kv("position", m.position)}
            ${kv("rule_adjustments", m.adaptive_rule_shift_adjustment_count)}
            ${kv("holo_adjustments", m.adaptive_holographic_shift_adjustment_count)}
            ${kv("final_shifts", [m.final_output_learning_rate_shift, m.final_mlp_learning_rate_shift, m.final_embedding_learning_rate_shift, m.final_attention_learning_rate_shift, m.final_attention_q_learning_rate_shift, m.final_attention_qk_learning_rate_shift].filter(v => v !== null && v !== undefined).join("/"))}
            ${kv("model", artifactLink(artifacts.model))}
            ${kv("progress", artifactLink(artifacts.progress))}
            ${kv("trace", artifactLink(artifacts.trace))}
            ${kv("log", artifactLink(artifacts.log))}
          </div>
          <details>
            <summary>command</summary>
            <pre>${run.command_escaped || ""}</pre>
          </details>
          <details>
            <summary>log tail</summary>
            <pre>${(run.log_tail || []).join("\\n")}</pre>
          </details>
        </article>`;
      }).join("");
    }
    async function refresh() {
      const response = await fetch("runs.json?cache=" + Date.now());
      render(await response.json());
    }
    refresh();
    setInterval(refresh, 30000);
  </script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dashboard-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--updated-at", required=True)
    parser.add_argument("--finished-at")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--repo-rev", default="")
    parser.add_argument("--tokens", default="")
    parser.add_argument("--command-file")
    parser.add_argument("--log-file")
    parser.add_argument("--progress-file")
    parser.add_argument("--trace-file")
    parser.add_argument("--model-file")
    parser.add_argument("--text-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    dashboard_dir = pathlib.Path(args.dashboard_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    started = time.mktime(time.strptime(args.started_at, "%Y-%m-%dT%H:%M:%SZ"))
    updated = time.mktime(time.strptime(args.updated_at, "%Y-%m-%dT%H:%M:%SZ"))
    elapsed = max(0, int(updated - started))
    run_s3_uri = f"{args.s3_uri.rstrip('/')}/runs/{args.run_name}"

    command_path = pathlib.Path(args.command_file) if args.command_file else None
    log_path = pathlib.Path(args.log_file) if args.log_file else None
    progress_path = pathlib.Path(args.progress_file) if args.progress_file else None
    trace_path = pathlib.Path(args.trace_file) if args.trace_file else None
    model_path = pathlib.Path(args.model_file) if args.model_file else None
    text_path = pathlib.Path(args.text_file) if args.text_file else None

    trace = load_json_line(trace_path) if trace_path else None
    progress_trace = load_json_line(progress_path) if progress_path else None
    metrics_trace = trace or progress_trace
    run = {
        "run_name": args.run_name,
        "status": args.status,
        "started_at": args.started_at,
        "updated_at": args.updated_at,
        "finished_at": args.finished_at,
        "elapsed_seconds": elapsed,
        "exit_code": args.exit_code,
        "repo_rev": args.repo_rev,
        "tokens": args.tokens,
        "s3_uri": run_s3_uri,
        "command": read_text(command_path).strip(),
        "command_escaped": html.escape(read_text(command_path).strip()),
        "log_tail": tail_lines(log_path, 80),
        "metrics": trace_summary(metrics_trace),
        "progress": trace_summary(progress_trace),
        "final": trace_summary(trace),
        "artifacts": {
            "progress": artifact("progress", progress_path, run_s3_uri),
            "trace": artifact("trace", trace_path, run_s3_uri),
            "model": artifact("model", model_path, run_s3_uri),
            "text": artifact("text", text_path, run_s3_uri),
            "log": artifact("log", log_path, run_s3_uri),
        },
    }

    write_json(run_dir / "run.json", run)
    runs_path = dashboard_dir / "runs.json"
    runs = [item for item in load_runs(runs_path) if item.get("run_name") != args.run_name]
    runs.append(run)
    runs.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    write_json(runs_path, runs[:200])
    write_json(dashboard_dir / "latest.json", run)
    render_index(dashboard_dir / "index.html")


if __name__ == "__main__":
    main()
