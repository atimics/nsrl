#!/usr/bin/env python3
"""Render a static S3 dashboard for NSRL cloud training runs."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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


def chart_series(trace: dict[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    training = trace.get("training", {}) if isinstance(trace.get("training"), dict) else {}
    metrics = trace.get("metrics", {}) if isinstance(trace.get("metrics"), dict) else {}
    examined = training.get("examined_windows")
    if not isinstance(examined, int):
        examined = training.get("max_windows")
    if not isinstance(examined, int):
        examined = 0

    initial_prob = metrics.get("initial_probability_error_q15")
    final_prob = metrics.get("final_probability_error_q15")
    loss = []
    if isinstance(initial_prob, int):
        loss.append({"x": 0, "y": initial_prob})
    if isinstance(final_prob, int):
        loss.append({"x": examined, "y": final_prob})

    initial_mistakes = metrics.get("initial_mistakes")
    final_mistakes = metrics.get("final_mistakes")
    accuracy = []
    if isinstance(initial_mistakes, int) and examined > 0:
        accuracy.append({"x": 0, "y": (examined - initial_mistakes) * 1000 // examined})
    if isinstance(final_mistakes, int) and examined > 0:
        accuracy.append({"x": examined, "y": (examined - final_mistakes) * 1000 // examined})

    target_probability = []
    steps = trace.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            x = step.get("window_index", step.get("update_index"))
            before = step.get("target_probability_before_q15")
            after = step.get("target_probability_after_q15")
            if isinstance(x, int) and isinstance(before, int) and isinstance(after, int):
                target_probability.append({"x": x, "before": before, "after": after})

    component_delta_l1 = []
    for key, label in [
        ("output_head_delta_l1", "output"),
        ("mlp_delta_l1", "mlp"),
        ("embedding_delta_l1", "embed"),
        ("attention_q_delta_l1", "q"),
        ("attention_k_delta_l1", "k"),
        ("attention_v_delta_l1", "v"),
        ("attention_o_delta_l1", "o"),
    ]:
        value = metrics.get(key)
        if isinstance(value, int):
            component_delta_l1.append({"label": label, "value": value})

    shift_events = []
    events = trace.get("adaptive_shift_events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            x = event.get("batch_index")
            y = event.get("next_shift")
            component = event.get("component")
            reason = event.get("reason")
            if isinstance(x, int) and isinstance(y, int) and isinstance(component, str):
                shift_events.append(
                    {
                        "x": x,
                        "y": y,
                        "component": component,
                        "reason": reason if isinstance(reason, str) else "",
                    }
                )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "target_probability": target_probability,
        "component_delta_l1": component_delta_l1,
        "shift_events": shift_events,
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


def parse_utc_seconds(value: str | None) -> int | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def cost_summary(
    hourly_usd: str,
    currency: str,
    run_elapsed_seconds: int,
    instance_elapsed_seconds: int | None,
) -> dict[str, Any]:
    if not hourly_usd:
        return {
            "available": False,
            "currency": currency,
            "run_elapsed_seconds": run_elapsed_seconds,
            "instance_elapsed_seconds": instance_elapsed_seconds,
        }
    try:
        hourly = Decimal(hourly_usd)
    except InvalidOperation:
        return {
            "available": False,
            "currency": currency,
            "hourly_rate_raw": hourly_usd,
            "run_elapsed_seconds": run_elapsed_seconds,
            "instance_elapsed_seconds": instance_elapsed_seconds,
        }

    billable_seconds = max(60, instance_elapsed_seconds or run_elapsed_seconds)
    run_cost = (hourly * Decimal(run_elapsed_seconds) / Decimal(3600)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    billable_cost = (hourly * Decimal(billable_seconds) / Decimal(3600)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    return {
        "available": True,
        "currency": currency,
        "hourly_usd": str(hourly),
        "run_elapsed_seconds": run_elapsed_seconds,
        "instance_elapsed_seconds": instance_elapsed_seconds,
        "billable_seconds": billable_seconds,
        "run_compute_usd": str(run_cost),
        "estimated_compute_usd": str(billable_cost),
        "scope": "instance_lifetime_to_report"
        if instance_elapsed_seconds is not None
        else "runner_elapsed",
    }


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
    .compare {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-bottom: 18px;
    }
    .workbench {
      display: grid;
      grid-template-columns: minmax(360px, 0.85fr) minmax(420px, 1.4fr);
      gap: 14px;
      align-items: start;
    }
    .run-list-panel, .run-detail {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .run-list-head {
      display: grid;
      grid-template-columns: minmax(150px, 1fr) 74px 74px 86px;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .run-rows {
      max-height: 72vh;
      overflow-y: auto;
    }
    .run-row {
      display: grid;
      grid-template-columns: minmax(150px, 1fr) 74px 74px 86px;
      gap: 8px;
      width: 100%;
      padding: 10px 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      color: var(--text);
      font: inherit;
      text-align: left;
      cursor: pointer;
    }
    .run-row:hover, .run-row.selected {
      background: #20262d;
    }
    .run-row b, .run-row span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .run-detail {
      padding: 16px;
      display: grid;
      gap: 12px;
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
    .charts {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    .chart {
      background: #111418;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .chart h3 {
      margin: 0 0 8px;
      font-size: 13px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .chart svg {
      display: block;
      width: 100%;
      height: 180px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .swatch {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      margin-right: 5px;
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
    @media (max-width: 900px) {
      .workbench { grid-template-columns: 1fr; }
      .run-rows { max-height: 420px; }
      .run-list-head, .run-row {
        grid-template-columns: minmax(130px, 1fr) 62px 62px 72px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>NSRL Cloud Training</h1>
    <div class="subtle" id="updated">Loading runs...</div>
  </header>
  <main>
    <section class="summary" id="summary"></section>
    <section class="compare" id="compare"></section>
    <section class="workbench">
      <aside class="run-list-panel">
        <div class="run-list-head">
          <span>run</span>
          <span>acc</span>
          <span>cost</span>
          <span>q/k</span>
        </div>
        <div class="run-rows" id="runRows"></div>
      </aside>
      <section class="run-detail" id="runDetail"></section>
    </section>
  </main>
  <script>
    let allRuns = [];
    let selectedRunName = null;
    let visibleRows = 50;
    const fmt = value => value === null || value === undefined ? "—" : String(value);
    const statusClass = status => String(status || "unknown").replace(/[^a-z0-9_-]/gi, "").toLowerCase();
    const artifactLink = artifact => {
      if (!artifact || !artifact.exists) return "—";
      return `<a href="${artifact.file}">${artifact.name}</a>`;
    };
    function kv(label, value) {
      return `<div class="kv"><span>${label}</span><b>${fmt(value)}</b></div>`;
    }
    function num(value) {
      if (value === null || value === undefined || value === "") return null;
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }
    function nice(value) {
      const n = num(value);
      if (n === null) return "—";
      return Math.abs(n) >= 1000000 ? n.toExponential(2) : String(Math.round(n));
    }
    function money(value) {
      const n = num(value);
      if (n === null || n === 0) return "—";
      if (Math.abs(n) < 1) {
        const cents = n * 100;
        const digits = Math.abs(cents) < 10 ? 2 : 1;
        return cents.toFixed(digits).replace(/[.]0+$/, "") + "c";
      }
      return "$" + n.toFixed(2);
    }
    function lineChart(title, points, series) {
      if (!Array.isArray(points) || points.length === 0) {
        return `<div class="chart"><h3>${title}</h3><div class="subtle">No chart points yet</div></div>`;
      }
      const colors = ["#7cc7ff", "#72d58a", "#ffd166", "#ff6b6b", "#c792ea", "#82daca"];
      const width = 640, height = 180, pad = 28;
      const xs = points.map(p => num(p.x)).filter(v => v !== null);
      const ys = [];
      for (const spec of series) {
        for (const p of points) {
          const y = num(p[spec.key]);
          if (y !== null) ys.push(y);
        }
      }
      if (xs.length === 0 || ys.length === 0) {
        return `<div class="chart"><h3>${title}</h3><div class="subtle">No chart points yet</div></div>`;
      }
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minX === maxX) { minX -= 1; maxX += 1; }
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const sx = x => pad + (x - minX) * (width - pad * 2) / (maxX - minX);
      const sy = y => height - pad - (y - minY) * (height - pad * 2) / (maxY - minY);
      const lines = series.map((spec, idx) => {
        const pts = points
          .map(p => [num(p.x), num(p[spec.key])])
          .filter(([x, y]) => x !== null && y !== null)
          .map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`)
          .join(" ");
        return `<polyline points="${pts}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />`;
      }).join("");
      const legend = series.map((spec, idx) =>
        `<span><i class="swatch" style="background:${colors[idx % colors.length]}"></i>${spec.label}</span>`
      ).join("");
      return `<div class="chart">
        <h3>${title}</h3>
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${title}">
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#2a3037" />
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#2a3037" />
          <text x="${pad}" y="14" fill="#9aa7b4" font-size="11">${nice(maxY)}</text>
          <text x="${pad}" y="${height - 5}" fill="#9aa7b4" font-size="11">${nice(minY)}</text>
          <text x="${width - pad}" y="${height - 5}" text-anchor="end" fill="#9aa7b4" font-size="11">${nice(maxX)}</text>
          ${lines}
        </svg>
        <div class="legend">${legend}</div>
      </div>`;
    }
    function barChart(title, bars) {
      if (!Array.isArray(bars) || bars.length === 0) {
        return `<div class="chart"><h3>${title}</h3><div class="subtle">No bar data yet</div></div>`;
      }
      const scaled = bars.map(b => ({...b, scaled: Math.log10((num(b.value) || 0) + 1)}));
      const max = Math.max(1, ...scaled.map(b => b.scaled));
      const rows = bars.map(b => {
        const value = num(b.value) || 0;
        const scaledValue = Math.log10(value + 1);
        const pct = Math.max(1, scaledValue * 100 / max);
        return `<div style="display:grid;grid-template-columns:58px 1fr 86px;gap:8px;align-items:center;margin:5px 0;">
          <span class="subtle">${b.label}</span>
          <span style="height:9px;background:#26303a;border-radius:999px;overflow:hidden;"><i style="display:block;width:${pct}%;height:100%;background:#7cc7ff;"></i></span>
          <b style="font-size:12px;text-align:right;">${nice(value)}</b>
        </div>`;
      }).join("");
      return `<div class="chart"><h3>${title} · log scale</h3>${rows}</div>`;
    }
    function comparisonBarChart(title, bars, options = {}) {
      const clean = (bars || []).filter(b => num(b.value) !== null).slice(0, 24);
      if (clean.length === 0) {
        return `<div class="chart"><h3>${title}</h3><div class="subtle">No comparison data yet</div></div>`;
      }
      const values = clean.map(b => Math.abs(num(b.value)) || 0);
      const rawMax = Math.max(0, ...values);
      const max = rawMax > 0 ? rawMax : 1;
      const rows = clean.map(b => {
        const value = num(b.value) || 0;
        const pct = Math.max(1, Math.abs(value) * 100 / max);
        const color = value < 0 && options.signed ? "#ff6b6b" : (options.color || "#7cc7ff");
        return `<button class="run-row" style="grid-template-columns:minmax(130px,1fr) 1fr 86px;border-bottom:0;padding:6px 0;" data-run="${b.name}">
          <b title="${b.name}">${b.name}</b>
          <span style="height:9px;background:#26303a;border-radius:999px;overflow:hidden;"><i style="display:block;width:${pct}%;height:100%;background:${color};"></i></span>
          <b style="font-size:12px;text-align:right;">${options.format ? options.format(value) : nice(value)}</b>
        </button>`;
      }).join("");
      return `<div class="chart"><h3>${title}</h3>${rows}</div>`;
    }
    function shiftChart(events) {
      if (!Array.isArray(events) || events.length === 0) {
        return `<div class="chart"><h3>Adaptive Shift Curriculum</h3><div class="subtle">No shift events yet</div></div>`;
      }
      const colors = ["#7cc7ff", "#72d58a", "#ffd166", "#ff6b6b", "#c792ea", "#82daca", "#f78c6c"];
      const width = 640, height = 180, pad = 28;
      const components = [...new Set(events.map(e => e.component))].slice(0, 12);
      const xs = events.map(e => num(e.x)).filter(v => v !== null);
      const ys = events.map(e => num(e.y)).filter(v => v !== null);
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minX === maxX) { minX -= 1; maxX += 1; }
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const sx = x => pad + (x - minX) * (width - pad * 2) / (maxX - minX);
      const sy = y => height - pad - (y - minY) * (height - pad * 2) / (maxY - minY);
      const lines = components.map((component, idx) => {
        const componentEvents = events
          .filter(e => e.component === component)
        const pts = componentEvents
          .map(e => `${sx(num(e.x)).toFixed(1)},${sy(num(e.y)).toFixed(1)}`)
          .join(" ");
        const last = componentEvents[componentEvents.length - 1];
        const label = last
          ? `<text x="${Math.min(width - 24, sx(num(last.x)) + 5).toFixed(1)}" y="${sy(num(last.y)).toFixed(1)}" fill="${colors[idx % colors.length]}" font-size="11">${num(last.y)}</text>`
          : "";
        return `<polyline points="${pts}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />${label}`;
      }).join("");
      const legend = components.map((component, idx) =>
        `<span><i class="swatch" style="background:${colors[idx % colors.length]}"></i>${component}</span>`
      ).join("");
      return `<div class="chart">
        <h3>Adaptive Shift Curriculum</h3>
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Adaptive shift curriculum">
          <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#2a3037" />
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#2a3037" />
          <text x="${pad}" y="14" fill="#9aa7b4" font-size="11">${nice(maxY)}</text>
          <text x="${pad}" y="${height - 5}" fill="#9aa7b4" font-size="11">${nice(minY)}</text>
          <text x="${width - pad}" y="${height - 5}" text-anchor="end" fill="#9aa7b4" font-size="11">${nice(maxX)}</text>
          ${lines}
        </svg>
        <div class="legend">${legend}</div>
      </div>`;
    }
    function charts(run) {
      const c = run.charts || {};
      return `<div class="charts">
        ${lineChart("Probability Error Q15", c.loss || [], [{key: "y", label: "loss"}])}
        ${lineChart("Accuracy Per Mille", c.accuracy || [], [{key: "y", label: "accuracy"}])}
        ${lineChart("Sampled Target Probability Q15", c.target_probability || [], [{key: "before", label: "before"}, {key: "after", label: "after"}])}
        ${barChart("Weight Movement L1", c.component_delta_l1 || [])}
        ${shiftChart(c.shift_events || [])}
      </div>`;
    }
    function runMetric(run, key) {
      return num((run.metrics || {})[key]);
    }
    function runCost(run) {
      return num(run.cost?.estimated_compute_usd);
    }
    function compareCharts(runs) {
      return [
        comparisonBarChart("Accuracy Per Mille", runs.map(run => ({
          name: run.run_name,
          value: runMetric(run, "final_accuracy_per_mille")
        })), {color: "#72d58a"}),
        comparisonBarChart("Probability Error Improvement", runs.map(run => ({
          name: run.run_name,
          value: -(runMetric(run, "probability_error_delta_i32") || 0)
        })), {color: "#7cc7ff"}),
        comparisonBarChart("Q/K Movement L1", runs.map(run => ({
          name: run.run_name,
          value: (runMetric(run, "attention_q_delta_l1") || 0) + (runMetric(run, "attention_k_delta_l1") || 0)
        })), {color: "#ffd166"}),
        comparisonBarChart("Estimated Compute Cost", runs.map(run => ({
          name: run.run_name,
          value: runCost(run)
        })), {color: "#c792ea", format: money})
      ].join("");
    }
    function finalShifts(run) {
      const m = run.metrics || {};
      return [
        m.final_output_learning_rate_shift,
        m.final_mlp_learning_rate_shift,
        m.final_embedding_learning_rate_shift,
        m.final_attention_learning_rate_shift,
        m.final_attention_q_learning_rate_shift,
        m.final_attention_qk_learning_rate_shift
      ].filter(v => v !== null && v !== undefined).join("/");
    }
    function renderRows() {
      const rows = allRuns.slice(0, visibleRows).map(run => {
        const m = run.metrics || {};
        const selected = run.run_name === selectedRunName ? " selected" : "";
        const q = num(m.attention_q_delta_l1) || 0;
        const k = num(m.attention_k_delta_l1) || 0;
        return `<button class="run-row${selected}" data-run="${run.run_name}">
          <b title="${run.run_name}">${run.run_name}</b>
          <span>${fmt(m.final_accuracy_per_mille)}</span>
          <span>${run.cost?.estimated_compute_usd ? "$" + Number(run.cost.estimated_compute_usd).toFixed(4) : "—"}</span>
          <span>${nice(q + k)}</span>
        </button>`;
      }).join("");
      const suffix = visibleRows < allRuns.length
        ? `<div class="subtle" style="padding:10px 12px;">Scroll for ${allRuns.length - visibleRows} more run(s)</div>`
        : "";
      document.getElementById("runRows").innerHTML = rows + suffix;
    }
    function renderDetail() {
      const run = allRuns.find(run => run.run_name === selectedRunName) || allRuns[0];
      if (!run) {
        document.getElementById("runDetail").innerHTML = `<div class="subtle">No runs yet</div>`;
        return;
      }
      const m = run.metrics || {};
      const artifacts = run.artifacts || {};
      document.getElementById("runDetail").innerHTML = `
        <div class="run-head">
          <h2>${run.run_name}</h2>
          <span class="pill ${statusClass(run.status)}">${run.status || "unknown"}</span>
        </div>
        <div class="grid">
          ${kv("started", run.started_at)}
          ${kv("updated", run.updated_at)}
          ${kv("stage", run.stage)}
          ${kv("elapsed_s", run.elapsed_seconds)}
          ${kv("instance", run.instance?.instance_type)}
          ${kv("cost_usd", run.cost?.estimated_compute_usd)}
          ${kv("run_cost_usd", run.cost?.run_compute_usd)}
          ${kv("billable_s", run.cost?.billable_seconds)}
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
          ${kv("final_shifts", finalShifts(run))}
          ${kv("model", artifactLink(artifacts.model))}
          ${kv("progress", artifactLink(artifacts.progress))}
          ${kv("trace", artifactLink(artifacts.trace))}
          ${kv("log", artifactLink(artifacts.log))}
        </div>
        ${charts(run)}
        <details>
          <summary>command</summary>
          <pre>${run.command_escaped || ""}</pre>
        </details>
        <details>
          <summary>log tail</summary>
          <pre>${(run.log_tail || []).join("\\n")}</pre>
        </details>`;
    }
    function selectRun(name) {
      selectedRunName = name;
      renderRows();
      renderDetail();
    }
    function render(runs) {
      allRuns = runs.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
      if (!selectedRunName || !allRuns.some(run => run.run_name === selectedRunName)) {
        selectedRunName = allRuns[0]?.run_name || null;
      }
      const running = allRuns.filter(r => r.status === "running").length;
      const succeeded = allRuns.filter(r => r.status === "succeeded").length;
      const failed = allRuns.filter(r => r.status === "failed").length;
      document.getElementById("updated").textContent =
        `Last refreshed ${new Date().toLocaleString()} · ${allRuns.length} run(s)`;
      document.getElementById("summary").innerHTML = [
        ["Running", running],
        ["Succeeded", succeeded],
        ["Failed", failed],
        ["Latest", allRuns[0]?.run_name || "—"]
      ].map(([label, value]) => `<div class="metric"><span class="subtle">${label}</span><b>${value}</b></div>`).join("");
      document.getElementById("compare").innerHTML = compareCharts(allRuns);
      renderRows();
      renderDetail();
    }
    document.addEventListener("click", event => {
      const row = event.target.closest("[data-run]");
      if (!row) return;
      selectRun(row.dataset.run);
    });
    document.addEventListener("scroll", () => {
      const list = document.getElementById("runRows");
      if (!list) return;
      if (list.scrollTop + list.clientHeight > list.scrollHeight - 160 && visibleRows < allRuns.length) {
        visibleRows += 50;
        renderRows();
      }
    }, true);
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
    parser.add_argument("--stage", default="")
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--updated-at", required=True)
    parser.add_argument("--finished-at")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--repo-rev", default="")
    parser.add_argument("--tokens", default="")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--instance-type", default="")
    parser.add_argument("--instance-region", default="")
    parser.add_argument("--instance-availability-zone", default="")
    parser.add_argument("--instance-launch-time", default="")
    parser.add_argument("--cost-hourly-usd", default="")
    parser.add_argument("--cost-currency", default="USD")
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

    started = parse_utc_seconds(args.started_at)
    updated = parse_utc_seconds(args.updated_at)
    if started is None or updated is None:
        raise ValueError("started-at and updated-at must be UTC timestamps")
    elapsed = max(0, int(updated - started))
    instance_started = parse_utc_seconds(args.instance_launch_time)
    instance_elapsed = (
        max(0, int(updated - instance_started)) if instance_started is not None else None
    )
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
        "stage": args.stage,
        "started_at": args.started_at,
        "updated_at": args.updated_at,
        "finished_at": args.finished_at,
        "elapsed_seconds": elapsed,
        "exit_code": args.exit_code,
        "repo_rev": args.repo_rev,
        "tokens": args.tokens,
        "s3_uri": run_s3_uri,
        "instance": {
            "instance_id": args.instance_id,
            "instance_type": args.instance_type,
            "region": args.instance_region,
            "availability_zone": args.instance_availability_zone,
            "launch_time": args.instance_launch_time,
        },
        "cost": cost_summary(
            args.cost_hourly_usd,
            args.cost_currency,
            elapsed,
            instance_elapsed,
        ),
        "command": read_text(command_path).strip(),
        "command_escaped": html.escape(read_text(command_path).strip()),
        "log_tail": tail_lines(log_path, 80),
        "metrics": trace_summary(metrics_trace),
        "progress": trace_summary(progress_trace),
        "final": trace_summary(trace),
        "charts": chart_series(trace or progress_trace),
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
