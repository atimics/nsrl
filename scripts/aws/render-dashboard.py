#!/usr/bin/env python3
"""Render a static S3 dashboard for NSRL cloud training runs."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


def load_json_line(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
    except UnicodeDecodeError:
        return None
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


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


def int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def trace_workers(trace: dict[str, Any]) -> list[dict[str, Any]]:
    workers = trace.get("workers", [])
    if not isinstance(workers, list):
        return []
    return [worker for worker in workers if isinstance(worker, dict)]


def swarm_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    data = trace.get("data", {}) if isinstance(trace.get("data"), dict) else {}
    model = trace.get("model", {}) if isinstance(trace.get("model"), dict) else {}
    optimizer = trace.get("optimizer", {}) if isinstance(trace.get("optimizer"), dict) else {}
    training = trace.get("training", {}) if isinstance(trace.get("training"), dict) else {}
    swarm = trace.get("swarm", {}) if isinstance(trace.get("swarm"), dict) else {}
    workers = trace_workers(trace)

    windows = sum(int_or_zero(worker.get("windows")) for worker in workers)
    examined = sum(int_or_zero(worker.get("examined_windows")) for worker in workers)
    updates = sum(int_or_zero(worker.get("updates")) for worker in workers)
    accepted_batches = sum(int_or_zero(worker.get("accepted_batch_count")) for worker in workers)
    rejected_batches = sum(int_or_zero(worker.get("rejected_batch_count")) for worker in workers)
    rollbacks = sum(int_or_zero(worker.get("rollback_count")) for worker in workers)
    rejected_windows = sum(int_or_zero(worker.get("rejected_window_count")) for worker in workers)
    invalid = sum(int_or_zero(worker.get("final_invalid_forward_count")) for worker in workers)
    initial_error = sum(int_or_zero(worker.get("initial_total_error")) for worker in workers)
    final_error = sum(int_or_zero(worker.get("final_total_error")) for worker in workers)
    initial_prob = sum(int_or_zero(worker.get("initial_probability_error_q15")) for worker in workers)
    final_prob = sum(int_or_zero(worker.get("final_probability_error_q15")) for worker in workers)
    output_delta = sum(int_or_zero(worker.get("output_head_delta_l1")) for worker in workers)
    mlp_delta = sum(int_or_zero(worker.get("mlp_delta_l1")) for worker in workers)
    embedding_delta = sum(int_or_zero(worker.get("embedding_delta_l1")) for worker in workers)
    attention_delta = sum(int_or_zero(worker.get("attention_delta_l1")) for worker in workers)
    attention_q_delta = sum(int_or_zero(worker.get("attention_q_delta_l1")) for worker in workers)
    attention_k_delta = sum(int_or_zero(worker.get("attention_k_delta_l1")) for worker in workers)
    attention_v_delta = sum(int_or_zero(worker.get("attention_v_delta_l1")) for worker in workers)
    attention_o_delta = sum(int_or_zero(worker.get("attention_o_delta_l1")) for worker in workers)
    has_final_error = any(isinstance(worker.get("final_total_error"), int) for worker in workers)
    has_probability_error = any(
        isinstance(worker.get("initial_probability_error_q15"), int)
        and isinstance(worker.get("final_probability_error_q15"), int)
        for worker in workers
    )

    progress_base = int_or_zero(training.get("max_windows")) or windows
    progress_per_mille = None
    if progress_base > 0:
        progress_per_mille = min(1000, examined * 1000 // progress_base)

    final_accuracy = None
    if examined > 0 and has_final_error:
        final_accuracy = max(0, (examined - final_error) * 1000 // examined)

    return {
        "schema": trace.get("schema"),
        "task": trace.get("task"),
        "trace_kind": "swarm",
        "token_count": data.get("token_count"),
        "windows": windows or None,
        "token_hash": data.get("token_hash"),
        "seq_len": model.get("seq_len") or training.get("seq_len"),
        "d_model": model.get("d_model"),
        "heads": model.get("heads"),
        "attention_kind": model.get("attention_kind"),
        "position": model.get("position"),
        "max_windows": training.get("max_windows"),
        "examined_windows": examined or None,
        "updates": updates or None,
        "accepted_batch_count": accepted_batches or None,
        "progress_per_mille": progress_per_mille,
        "batch_windows": training.get("batch_windows"),
        "stride": training.get("stride"),
        "window_offset": training.get("window_offset"),
        "adaptive_rule_shifts": optimizer.get("adaptive_rule_shifts"),
        "adaptive_holographic_shifts": optimizer.get("adaptive_holographic_shifts"),
        "initial_mistakes": initial_error if has_final_error else None,
        "final_mistakes": final_error if has_final_error else None,
        "initial_probability_error_q15": initial_prob if has_probability_error else None,
        "final_probability_error_q15": final_prob if has_probability_error else None,
        "probability_error_delta_i32": final_prob - initial_prob if has_probability_error else None,
        "probability_error_delta_i64": final_prob - initial_prob if has_probability_error else None,
        "probability_error_delta_q15": final_prob - initial_prob if has_probability_error else None,
        "final_accuracy_per_mille": final_accuracy,
        "rollback_count": rollbacks,
        "rejected_batch_count": rejected_batches,
        "rejected_window_count": rejected_windows,
        "final_invalid_forward_count": invalid,
        "worker_count": swarm.get("worker_count", len(workers)),
        "best_worker_index": swarm.get("best_worker_index"),
        "final_model_hash": swarm.get("final_model_hash"),
        "output_head_delta_l1": output_delta or None,
        "mlp_delta_l1": mlp_delta or None,
        "embedding_delta_l1": embedding_delta or None,
        "attention_delta_l1": attention_delta or None,
        "attention_q_delta_l1": attention_q_delta or None,
        "attention_k_delta_l1": attention_k_delta or None,
        "attention_v_delta_l1": attention_v_delta or None,
        "attention_o_delta_l1": attention_o_delta or None,
        "adaptive_rule_shift_adjustment_count": None,
        "adaptive_holographic_shift_adjustment_count": None,
    }


def trace_summary(trace: dict[str, Any] | None) -> dict[str, Any]:
    if not trace:
        return {}
    if trace_workers(trace):
        return swarm_trace_summary(trace)
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
    probability_error_delta = first_int(
        metrics.get("probability_error_delta_i32"),
        metrics.get("probability_error_delta_i64"),
    )
    output_head_delta = first_int(
        metrics.get("output_head_delta_l1"),
        metrics.get("weight_delta_l1"),
    )

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
        "probability_error_delta_i32": probability_error_delta,
        "probability_error_delta_i64": probability_error_delta,
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
        "output_head_delta_l1": output_head_delta,
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
    workers = trace_workers(trace)
    if workers:
        examined = sum(int_or_zero(worker.get("examined_windows")) for worker in workers)
        initial_prob = sum(
            int_or_zero(worker.get("initial_probability_error_q15")) for worker in workers
        )
        final_prob = sum(
            int_or_zero(worker.get("final_probability_error_q15")) for worker in workers
        )
        initial_mistakes = sum(int_or_zero(worker.get("initial_total_error")) for worker in workers)
        final_mistakes = sum(int_or_zero(worker.get("final_total_error")) for worker in workers)
        has_final_error = any(isinstance(worker.get("final_total_error"), int) for worker in workers)
        has_probability_error = any(
            isinstance(worker.get("initial_probability_error_q15"), int)
            and isinstance(worker.get("final_probability_error_q15"), int)
            for worker in workers
        )
        loss = []
        if has_probability_error:
            loss.append({"x": 0, "y": initial_prob})
            loss.append({"x": examined, "y": final_prob})
        accuracy = []
        if examined > 0 and has_final_error:
            accuracy.append({"x": 0, "y": (examined - initial_mistakes) * 1000 // examined})
            accuracy.append({"x": examined, "y": (examined - final_mistakes) * 1000 // examined})
        worker_accuracy = []
        worker_probability_delta = []
        for worker in workers:
            worker_index = worker.get("worker_index")
            label = f"w{worker_index}" if isinstance(worker_index, int) else "worker"
            accuracy_value = worker.get("final_accuracy_per_mille")
            if isinstance(accuracy_value, int):
                worker_accuracy.append({"label": label, "value": accuracy_value})
            worker_initial = worker.get("initial_probability_error_q15")
            worker_final = worker.get("final_probability_error_q15")
            if isinstance(worker_initial, int) and isinstance(worker_final, int):
                worker_probability_delta.append(
                    {"label": label, "value": worker_initial - worker_final}
                )
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
            value = sum(int_or_zero(worker.get(key)) for worker in workers)
            if value:
                component_delta_l1.append({"label": label, "value": value})
        return {
            "loss": loss,
            "accuracy": accuracy,
            "target_probability": [],
            "component_delta_l1": component_delta_l1,
            "shift_events": [],
            "worker_accuracy": worker_accuracy,
            "worker_probability_delta": worker_probability_delta,
        }

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
    for keys, label in [
        (("output_head_delta_l1", "weight_delta_l1"), "output"),
        (("mlp_delta_l1",), "mlp"),
        (("embedding_delta_l1",), "embed"),
        (("attention_q_delta_l1",), "q"),
        (("attention_k_delta_l1",), "k"),
        (("attention_v_delta_l1",), "v"),
        (("attention_o_delta_l1",), "o"),
    ]:
        value = first_int(*(metrics.get(key) for key in keys))
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


PROJECT_LABELS = {
    "signal": "Signal",
    "cosyworld": "CosyWorld",
    "lab": "Lab",
}

PHASE_LABELS = {
    "corpus": "Corpus",
    "training": "Training",
}


def slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or fallback


def normalize_choice(value: str | None, choices: dict[str, str], fallback: str) -> str:
    if not value:
        return fallback
    candidate = slugify(value, fallback)
    return candidate if candidate in choices else fallback


def corpus_version_from_path(path: str) -> str:
    for part in pathlib.PurePosixPath(path.replace("\\", "/")).parts:
        if re.fullmatch(r"20\d{6}T\d{6}Z", part):
            return part
    match = re.search(r"20\d{6}T\d{6}Z", path)
    return match.group(0) if match else ""


def corpus_id_from_path(path: str) -> str:
    if not path:
        return ""
    pure = pathlib.PurePosixPath(path.replace("\\", "/"))
    parts = list(pure.parts)
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 1 < len(parts):
            return slugify(parts[idx + 1], "corpus")
    for part in reversed(parts):
        stem = pathlib.PurePosixPath(part).stem
        cleaned = re.sub(r"\.tokens|\.lexeme-v\d+|\.v\d+|\.corpus", "", stem)
        cleaned = re.sub(r"-?20\d{6}T\d{6}Z", "", cleaned)
        if cleaned and cleaned not in {"tokens", "vocab", "corpus", "manifest"}:
            return slugify(cleaned, "corpus")
    return ""


def infer_project(args: argparse.Namespace, command: str) -> str:
    explicit = normalize_choice(args.project, PROJECT_LABELS, "")
    if explicit:
        return explicit
    haystack = " ".join(
        [
            args.run_name,
            args.stage,
            args.tokens,
            args.corpus_id or "",
            args.corpus_name or "",
            args.corpus_file or "",
            args.manifest_file or "",
            command,
        ]
    ).lower()
    if any(key in haystack for key in ("signal", "romance", "ship-radio")):
        return "signal"
    if any(key in haystack for key in ("cosyworld", "cosy-world", "cosy_world")):
        return "cosyworld"
    return "lab"


def infer_phase(args: argparse.Namespace, command: str) -> str:
    explicit = normalize_choice(args.phase, PHASE_LABELS, "")
    if explicit:
        return explicit
    haystack = " ".join([args.run_name, args.stage, args.tokens, command]).lower()
    corpus_markers = (
        "build-corpus",
        "corpus/datasets",
        "clean-gutenberg",
        "extract-simplewiki",
        "lexeme-tokenize",
        "tokenize --corpus",
    )
    training_markers = ("train", "training", "softmax", "embedding", "mini-transformer")
    if any(marker in haystack for marker in corpus_markers) and not any(
        marker in haystack for marker in training_markers
    ):
        return "corpus"
    if "corpus-" in args.run_name.lower() or args.stage.lower().startswith("corpus"):
        return "corpus"
    return "training"


def corpus_metadata(args: argparse.Namespace) -> dict[str, str]:
    candidate_paths = [
        args.corpus_file or "",
        args.manifest_file or "",
        args.tokens_file or "",
        args.tokens,
    ]
    corpus_id = args.corpus_id or next(
        (corpus_id_from_path(path) for path in candidate_paths if corpus_id_from_path(path)),
        "",
    )
    version = args.corpus_version or next(
        (corpus_version_from_path(path) for path in candidate_paths if corpus_version_from_path(path)),
        "",
    )
    corpus_id = slugify(corpus_id, "unversioned")
    version = slugify(version, "unversioned")
    name = args.corpus_name or corpus_id.replace("-", " ")
    return {
        "id": corpus_id,
        "name": name,
        "version": version,
        "key": f"{corpus_id}:{version}",
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
      grid-template-columns: minmax(150px, 1fr) 86px 118px 62px 76px;
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
      grid-template-columns: minmax(150px, 1fr) 86px 118px 62px 76px;
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
    .run-row.trouble-orange {
      box-shadow: inset 4px 0 0 var(--warn);
      background: rgba(255, 209, 102, 0.045);
    }
    .run-row.trouble-red {
      box-shadow: inset 4px 0 0 var(--bad);
      background: rgba(255, 107, 107, 0.06);
    }
    .run-row.trouble-orange.selected,
    .run-row.trouble-red.selected {
      background: #20262d;
    }
    .run-row b, .run-row span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .run-row .progress-wrap,
    .run-row .progress-wrap span {
      overflow: visible;
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
    .pill.running {
      border-color: rgba(255, 209, 102, 0.42);
      background: rgba(255, 209, 102, 0.08);
    }
    .live-dot {
      display: inline-block;
      width: 7px;
      height: 7px;
      margin-right: 6px;
      border-radius: 50%;
      background: var(--warn);
      box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.65);
      animation: pulse 1.4s infinite;
      vertical-align: middle;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.65); }
      70% { box-shadow: 0 0 0 7px rgba(255, 209, 102, 0); }
      100% { box-shadow: 0 0 0 0 rgba(255, 209, 102, 0); }
    }
    .progress-wrap {
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .progress-meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
    }
    .progress-bar {
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #26303a;
    }
    .progress-fill {
      display: block;
      height: 100%;
      min-width: 2px;
      border-radius: inherit;
      background: var(--accent);
    }
    .progress-fill.estimated { background: var(--warn); }
    .progress-fill.done { background: var(--good); }
    .progress-fill.failed { background: var(--bad); }
    .trouble-banner {
      border: 1px solid rgba(255, 209, 102, 0.36);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--warn);
      background: rgba(255, 209, 102, 0.08);
    }
    .trouble-banner.red {
      border-color: rgba(255, 107, 107, 0.42);
      color: var(--bad);
      background: rgba(255, 107, 107, 0.08);
    }
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
    .chat {
      display: grid;
      gap: 10px;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .chat-window {
      display: grid;
      align-content: start;
      gap: 10px;
      min-height: 240px;
      max-height: 380px;
      overflow-y: auto;
      padding: 12px;
      background: #0d0f12;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .chat-msg {
      display: flex;
      min-width: 0;
    }
    .chat-msg.user {
      justify-content: flex-end;
    }
    .chat-msg.model,
    .chat-msg.fallback,
    .chat-msg.system {
      justify-content: flex-start;
    }
    .chat-bubble {
      max-width: min(78%, 620px);
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #15191e;
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .chat-msg.user .chat-bubble {
      background: #1b303c;
      border-color: rgba(124, 199, 255, 0.36);
    }
    .chat-msg.model .chat-bubble {
      background: #162019;
      border-color: rgba(114, 213, 138, 0.28);
    }
    .chat-msg.fallback .chat-bubble {
      background: #251f12;
      border-color: rgba(255, 209, 102, 0.36);
    }
    .chat-label {
      display: block;
      margin-bottom: 3px;
      color: var(--muted);
      font-size: 11px;
    }
    .chat-composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: end;
    }
    .chat-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    textarea {
      box-sizing: border-box;
      width: 100%;
      min-height: 94px;
      resize: vertical;
      color: var(--text);
      background: #0d0f12;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .chat-composer textarea {
      min-height: 52px;
      max-height: 130px;
    }
    button.action {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #20262d;
      color: var(--text);
      padding: 7px 11px;
      font: inherit;
      cursor: pointer;
    }
    button.action:hover {
      border-color: var(--accent);
    }
    button.action:disabled {
      cursor: default;
      opacity: 0.55;
    }
    .chat-status {
      color: var(--muted);
      font-size: 12px;
    }
    a { color: var(--accent); }
    @media (max-width: 900px) {
      .workbench { grid-template-columns: 1fr; }
      .run-rows { max-height: 420px; }
      .chat-composer { grid-template-columns: 1fr; }
      .run-list-head, .run-row {
        grid-template-columns: minmax(130px, 1fr) 72px 96px 54px 68px;
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
          <span>status</span>
          <span>progress</span>
          <span>acc</span>
          <span>cost</span>
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
    let chatState = { key: "", model: null, vocab: null, wasm: null };
    const chatThreads = new Map();
    const fmt = value => value === null || value === undefined ? "—" : String(value);
    const statusClass = status => String(status || "unknown").replace(/[^a-z0-9_-]/gi, "").toLowerCase();
    const escapeHtml = value => String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
    const artifactHref = (run, artifact) => {
      if (!artifact || !artifact.exists) return "";
      return `../runs/${encodeURIComponent(run.run_name)}/${encodeURIComponent(artifact.file)}`;
    };
    const artifactLink = (run, artifact) => {
      if (!artifact || !artifact.exists) return "—";
      return `<a href="${artifactHref(run, artifact)}">${artifact.name}</a>`;
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
    function logScale(value) {
      const n = num(value);
      if (n === null) return null;
      const magnitude = Math.log10(Math.abs(n) + 1);
      return n < 0 ? -magnitude : magnitude;
    }
    function logMagnitude(value) {
      const n = num(value);
      return n === null ? null : Math.log10(Math.abs(n) + 1);
    }
    function runProject(run) {
      const project = String(run?.project || "").toLowerCase();
      if (project === "signal" || project === "cosyworld" || project === "lab") return project;
      const haystack = `${run?.run_name || ""} ${run?.stage || ""} ${run?.tokens || ""}`.toLowerCase();
      if (/signal|romance|ship-radio/.test(haystack)) return "signal";
      if (/cosyworld|cosy-world|cosy_world/.test(haystack)) return "cosyworld";
      return "lab";
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
    function commandNumber(run, flag) {
      const command = String(run.command || "");
      const escaped = flag.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&");
      const match = command.match(new RegExp(`${escaped}(?:=|\\s+)([^\\s]+)`));
      if (!match) return null;
      return num(match[1]);
    }
    function runMaxWindows(run) {
      const m = run.metrics || {};
      return num(m.max_windows) || num(m.windows) || commandNumber(run, "--max-windows");
    }
    function statusMarkup(run) {
      const status = run.status || "unknown";
      const dot = status === "running" ? `<i class="live-dot"></i>` : "";
      return `${dot}<span class="${statusClass(status)}">${status}</span>`;
    }
    function stageFloorPerMille(stage) {
      const value = String(stage || "").toLowerCase();
      if (value.includes("complete") || value.includes("upload")) return 920;
      if (value.includes("train")) return 200;
      if (value.includes("build")) return 140;
      if (value.includes("boot") || value.includes("sync")) return 100;
      if (value.includes("ec2") || value.includes("launch")) return 60;
      if (value.includes("request")) return 20;
      return 40;
    }
    function estimatedProgress(run) {
      const status = run.status || "";
      const m = run.metrics || {};
      const real = num(m.progress_per_mille);
      if (real !== null) {
        return {
          perMille: Math.max(0, Math.min(1000, real)),
          source: status === "running" ? "live" : "actual",
        };
      }
      if (status === "succeeded") {
        return { perMille: 1000, source: "complete" };
      }
      if (status === "failed") {
        return { perMille: 1000, source: "failed" };
      }
      const maxWindows = runMaxWindows(run);
      const elapsed = num(run.elapsed_seconds) || 0;
      const command = String(run.command || "");
      const throughput = command.includes("mini-transformer-swarm")
        ? 170
        : command.includes("map-reduce")
          ? 1400
          : 1200;
      const expectedSeconds = maxWindows ? Math.max(45, maxWindows / throughput) : 120;
      const estimate = Math.floor((elapsed * 1000) / expectedSeconds);
      return {
        perMille: Math.max(
          stageFloorPerMille(run.stage),
          Math.min(980, Math.max(0, estimate)),
        ),
        source: "estimated",
      };
    }
    function progressBar(run, compact = false) {
      const progress = estimatedProgress(run);
      const pct = Math.max(0, Math.min(100, Math.round(progress.perMille / 10)));
      const status = run.status || "";
      const fillClass = status === "succeeded"
        ? "done"
        : status === "failed"
          ? "failed"
          : progress.source === "estimated"
            ? "estimated"
            : "";
      const label = compact
        ? `${pct}%`
        : `${pct}% ${progress.source}`;
      const suffix = compact ? progress.source : (run.stage || "");
      return `<div class="progress-wrap" title="${pct}% ${progress.source}">
        <div class="progress-bar"><i class="progress-fill ${fillClass}" style="width:${Math.max(2, pct)}%;"></i></div>
        <div class="progress-meta"><span>${label}</span><span>${suffix}</span></div>
      </div>`;
    }
    function lineChart(title, points, series) {
      if (!Array.isArray(points) || points.length === 0) {
        return `<div class="chart"><h3>${title} · log scale</h3><div class="subtle">No chart points yet</div></div>`;
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
        return `<div class="chart"><h3>${title} · log scale</h3><div class="subtle">No chart points yet</div></div>`;
      }
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      let minYScaled = Math.min(...ys.map(logScale)), maxYScaled = Math.max(...ys.map(logScale));
      if (minX === maxX) { minX -= 1; maxX += 1; }
      if (minYScaled === maxYScaled) { minYScaled -= 1; maxYScaled += 1; }
      const sx = x => pad + (x - minX) * (width - pad * 2) / (maxX - minX);
      const sy = y => height - pad - (logScale(y) - minYScaled) * (height - pad * 2) / (maxYScaled - minYScaled);
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
        <h3>${title} · log scale</h3>
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
        return `<div class="chart"><h3>${title} · log scale</h3><div class="subtle">No bar data yet</div></div>`;
      }
      const scaled = bars.map(b => ({...b, scaled: logMagnitude(b.value) || 0}));
      const max = Math.max(1, ...scaled.map(b => b.scaled));
      const rows = bars.map(b => {
        const value = num(b.value) || 0;
        const scaledValue = logMagnitude(value) || 0;
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
        return `<div class="chart"><h3>${title} · log scale</h3><div class="subtle">No comparison data yet</div></div>`;
      }
      const logMultiplier = num(options.logMultiplier) || 1;
      const values = clean.map(b => logMagnitude((num(b.value) || 0) * logMultiplier) || 0);
      const max = Math.max(1, ...values);
      const rows = clean.map(b => {
        const value = num(b.value) || 0;
        const pct = Math.max(1, (logMagnitude(value * logMultiplier) || 0) * 100 / max);
        const color = value < 0 && options.signed ? "#ff6b6b" : (options.color || "#7cc7ff");
        return `<button class="run-row${b.trouble || ""}" style="grid-template-columns:minmax(130px,1fr) 1fr 86px;border-bottom:0;padding:6px 0;" data-run="${b.name}">
          <b title="${b.name}">${b.name}</b>
          <span style="height:9px;background:#26303a;border-radius:999px;overflow:hidden;"><i style="display:block;width:${pct}%;height:100%;background:${color};"></i></span>
          <b style="font-size:12px;text-align:right;">${options.format ? options.format(value) : nice(value)}</b>
        </button>`;
      }).join("");
      return `<div class="chart"><h3>${title} · log scale</h3>${rows}</div>`;
    }
    function shiftChart(events) {
      if (!Array.isArray(events) || events.length === 0) {
        return `<div class="chart"><h3>Adaptive Shift Curriculum · log scale</h3><div class="subtle">No shift events yet</div></div>`;
      }
      const colors = ["#7cc7ff", "#72d58a", "#ffd166", "#ff6b6b", "#c792ea", "#82daca", "#f78c6c"];
      const width = 640, height = 180, pad = 28;
      const components = [...new Set(events.map(e => e.component))].slice(0, 12);
      const xs = events.map(e => num(e.x)).filter(v => v !== null);
      const ys = events.map(e => num(e.y)).filter(v => v !== null);
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      let minYScaled = Math.min(...ys.map(logScale)), maxYScaled = Math.max(...ys.map(logScale));
      if (minX === maxX) { minX -= 1; maxX += 1; }
      if (minYScaled === maxYScaled) { minYScaled -= 1; maxYScaled += 1; }
      const sx = x => pad + (x - minX) * (width - pad * 2) / (maxX - minX);
      const sy = y => height - pad - (logScale(y) - minYScaled) * (height - pad * 2) / (maxYScaled - minYScaled);
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
        <h3>Adaptive Shift Curriculum · log scale</h3>
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
        ${(c.worker_accuracy || []).length ? barChart("Worker Accuracy Per Mille", c.worker_accuracy) : ""}
        ${(c.worker_probability_delta || []).length ? barChart("Worker Probability Improvement", c.worker_probability_delta) : ""}
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
    function traceSchema(run) {
      return String((run.metrics || {}).schema || (run.final || {}).schema || (run.progress || {}).schema || "");
    }
    function hasMiniTransformerAttentionMetrics(run) {
      return traceSchema(run).startsWith("nsrl.training_mini_transformer");
    }
    function probabilityErrorImprovement(run) {
      const delta = runMetric(run, "probability_error_delta_i32");
      return delta === null ? null : Math.max(0, -delta);
    }
    function qkMovement(run) {
      return (runMetric(run, "attention_q_delta_l1") || 0)
        + (runMetric(run, "attention_k_delta_l1") || 0);
    }
    function troubleLevel(run) {
      if (!hasMiniTransformerAttentionMetrics(run)) return "";
      const pei = probabilityErrorImprovement(run);
      const peiZero = pei === null || pei === 0;
      const qkmZero = qkMovement(run) === 0;
      if (peiZero && qkmZero) return "red";
      if (peiZero || qkmZero) return "orange";
      return "";
    }
    function troubleClass(run) {
      const level = troubleLevel(run);
      return level ? ` trouble-${level}` : "";
    }
    function troubleBanner(run) {
      const level = troubleLevel(run);
      if (!level) return "";
      const pei = probabilityErrorImprovement(run);
      const qkm = qkMovement(run);
      const message = level === "red"
        ? "Troubled build: PEI and QKM are both zero."
        : "Watch build: either PEI or QKM is zero.";
      return `<div class="trouble-banner ${level === "red" ? "red" : ""}">
        ${message} PEI=${nice(pei)} · QKM=${nice(qkm)}
      </div>`;
    }
    function compareCharts(runs) {
      return [
        comparisonBarChart("Accuracy Per Mille", runs.map(run => ({
          name: run.run_name,
          value: runMetric(run, "final_accuracy_per_mille")
        })), {color: "#72d58a"}),
        comparisonBarChart("Probability Error Improvement", runs.map(run => ({
          name: run.run_name,
          value: probabilityErrorImprovement(run),
          trouble: troubleClass(run)
        })), {color: "#7cc7ff"}),
        comparisonBarChart("Q/K Movement L1", runs.map(run => ({
          name: run.run_name,
          value: hasMiniTransformerAttentionMetrics(run) ? qkMovement(run) : null,
          trouble: troubleClass(run)
        })), {color: "#ffd166"}),
        comparisonBarChart("Estimated Compute Cost", runs.map(run => ({
          name: run.run_name,
          value: runCost(run)
        })), {color: "#c792ea", format: money, logMultiplier: 100})
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
        return `<button class="run-row${selected}${troubleClass(run)}" data-run="${run.run_name}">
          <b title="${run.run_name}">${run.run_name}</b>
          <span>${statusMarkup(run)}</span>
          <span>${progressBar(run, true)}</span>
          <span>${fmt(m.final_accuracy_per_mille)}</span>
          <span>${run.cost?.estimated_compute_usd ? money(run.cost.estimated_compute_usd) : "—"}</span>
        </button>`;
      }).join("");
      const suffix = visibleRows < allRuns.length
        ? `<div class="subtle" style="padding:10px 12px;">Scroll for ${allRuns.length - visibleRows} more run(s)</div>`
        : "";
      document.getElementById("runRows").innerHTML = rows + suffix;
    }
    function chatMessagesForRun(runName) {
      if (!chatThreads.has(runName)) {
        const run = allRuns.find(run => run.run_name === runName);
        const opener = runProject(run) === "cosyworld"
          ? "CosyWorld hearth channel open."
          : runProject(run) === "signal"
            ? "Signal channel open."
            : "Model channel open.";
        chatThreads.set(runName, [
          { role: "system", text: opener },
        ]);
      }
      return chatThreads.get(runName);
    }
    function chatTranscriptHtml(runName) {
      return chatMessagesForRun(runName).map(message => {
        const role = String(message.role || "system").replace(/[^a-z0-9_-]/gi, "").toLowerCase();
        const label = role === "user" ? "ranker" : role === "model" ? "model" : role;
        return `<div class="chat-msg ${role}">
          <div class="chat-bubble">
            <span class="chat-label">${label}</span>
            ${escapeHtml(message.text)}
          </div>
        </div>`;
      }).join("");
    }
    function renderChatLog(runName) {
      const node = document.getElementById("chatLog");
      if (!node) return;
      node.innerHTML = chatTranscriptHtml(runName);
      node.scrollTop = node.scrollHeight;
    }
    function pushChatMessage(runName, role, text) {
      chatMessagesForRun(runName).push({ role, text });
      renderChatLog(runName);
    }
    function defaultChatPrompt(run) {
      if (runProject(run) === "cosyworld") {
        return "Brindle Mosscup";
      }
      if (runProject(run) === "lab") {
        return "the world is";
      }
      return "Caution LM traffic";
    }
    function runUsesRawPrompt(run) {
      const artifacts = run?.artifacts || {};
      const haystack = [
        run?.run_name || "",
        run?.stage || "",
        run?.tokens || "",
        artifacts.model?.file || "",
        artifacts.vocab?.file || "",
      ].join(" ").toLowerCase();
      return /\braw\b|raw-line|voice-only/.test(haystack);
    }
    function chatPromptFromInput(input, run) {
      const trimmed = String(input || "").trim();
      const line = trimmed || defaultChatPrompt(run);
      if (/RANKED:/i.test(line) && /VOICE:/i.test(line)) {
        return { prompt: line, display: rankedFallback(line) || line };
      }
      if (runUsesRawPrompt(run)) {
        return { prompt: line, display: line };
      }
      return {
        prompt: `RANKED: ${line}\nVOICE: `,
        display: line,
      };
    }
    function chatPane(run) {
      const artifacts = run.artifacts || {};
      if (!artifacts.model?.exists || !artifacts.vocab?.exists) {
        return "";
      }
      const prompt = defaultChatPrompt(run);
      return `<section class="chat">
        <div class="run-head">
          <h2>Chat With Model</h2>
          <span class="pill">client-side wasm</span>
        </div>
        <div class="chat-window" id="chatLog">${chatTranscriptHtml(run.run_name)}</div>
        <div class="chat-composer">
          <textarea id="chatPrompt" spellcheck="false" placeholder="${escapeHtml(prompt)}">${escapeHtml(prompt)}</textarea>
          <div class="chat-controls">
            <button class="action" id="chatLoad" type="button">Load Model</button>
            <button class="action" id="chatSend" type="button">Send</button>
          </div>
        </div>
        <span class="chat-status" id="chatStatus">model and vocab ready to load</span>
      </section>`;
    }
    const CHAT_WASM_BASE64 = "AGFzbQEAAAABCAFgA39/fwF/AwIBAAUDAQAQBhkDfwFBgIDAAAt/AEGAgMAAC38AQYCAwAALBzkEBm1lbW9yeQIAEWRvdF9pMTZfaThfc2hpZnQ4AAAKX19kYXRhX2VuZAMBC19faGVhcF9iYXNlAwIKwAIBvQIDAX4EfwJ+AkACQCACDQBCACEDDAELIAJBAXEhBAJAAkAgAkEBRw0AQQAhBUIAIQMMAQsgAkF+cSEGQQAhBSAAIQJCACEDA0AgAyABIAVqIgcwAAAgAjIBAH4iCHwiCUI/h0KAgICAgICAgIB/hSAJIAhCAFMgCSADU3MbIgkgB0EBajAAACACQQJqMgEAfiIIfCIDQj+HQoCAgICAgICAgH+FIAMgCEIAUyADIAlTcxshAyACQQRqIQIgBiAFQQJqIgVHDQALCwJAIARFDQAgAyABIAVqMAAAIAAgBUEBdGoyAQB+Igh8IglCP4dCgICAgICAgICAf4UgCSAIQgBTIAkgA1NzGyEDC0L/AEKAASADQgBTGyADfEIIhyEDCyADQoCAfiADQoCAflUbIgNC//8BIANC//8BUxunCwBHBG5hbWUAFhVuc3JsX2NoYXRfa2VybmVsLndhc20BFAEAEWRvdF9pMTZfaThfc2hpZnQ4BxIBAA9fX3N0YWNrX3BvaW50ZXIARQlwcm9kdWNlcnMBDHByb2Nlc3NlZC1ieQEFcnVzdGMlMS45NS4wLW5pZ2h0bHkgKGYxMzRiYmM3OCAyMDY2LTAxLTI0KQCUAQ90YXJnZXRfZmVhdHVyZXMIKwtidWxrLW1lbW9yeSsPYnVsay1tZW1vcnktb3B0KxZjYWxsLWluZGlyZWN0LW92ZXJsb25nKwptdWx0aXZhbHVlKw9tdXRhYmxlLWdsb2JhbHMrE25vbnRyYXBwaW5nLWZwdG9pbnQrD3JlZmVyZW5jZS10eXBlcysIc2lnbi1leHQ=";
    function setChatStatus(text) {
      const node = document.getElementById("chatStatus");
      if (node) node.textContent = text;
    }
    function setChatOutput(text) {
      const runName = selectedRunName || allRuns[0]?.run_name || "chat";
      pushChatMessage(runName, "system", text);
    }
    async function loadChatWasm() {
      if (chatState.wasm) return chatState.wasm;
      const bytes = Uint8Array.from(atob(CHAT_WASM_BASE64), ch => ch.charCodeAt(0));
      const { instance } = await WebAssembly.instantiate(bytes, {});
      chatState.wasm = {
        memory: instance.exports.memory,
        dot: instance.exports.dot_i16_i8_shift8,
      };
      return chatState.wasm;
    }
    function parseVocab(text) {
      const entries = Array.from({ length: 256 }, (_, id) => String.fromCharCode(id));
      const lookup = new Map();
      for (const line of text.split(/\\r?\\n/).slice(1)) {
        if (!line.trim()) continue;
        const parts = line.split("\\t");
        const id = Number(parts[0]);
        const lexeme = parts[1] || "";
        if (Number.isInteger(id) && id >= 256 && lexeme) {
          entries[id] = lexeme;
          lookup.set(lexeme, id);
        }
      }
      return { entries, lookup };
    }
    function parseLexemeModel(bytes) {
      const decoder = new TextDecoder("ascii");
      const magic = decoder.decode(bytes.slice(0, 8));
      if (magic !== "NSRLLM6\\n") {
        throw new Error(`unsupported lexeme model ${JSON.stringify(magic.trim())}`);
      }
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      let offset = 8;
      const u32 = () => {
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
      };
      const u64 = () => {
        const value = Number(view.getBigUint64(offset, true));
        offset += 8;
        return value;
      };
      const seqLen = u32();
      const vocabSize = u32();
      const embeddingDim = u32();
      const contextFeatures = u32();
      const hiddenDim = u32();
      const headLayout = u32();
      const adapterLogitShift = u32();
      const embeddingCount = u64();
      const hiddenWeightCount = u64();
      const outputWeightCount = u64();
      const embeddingHash = u64();
      const hiddenHash = u64();
      const outputHash = u64();
      const modelHash = u64();
      const embeddings = new Int16Array(embeddingCount);
      for (let i = 0; i < embeddingCount; i += 1) {
        embeddings[i] = view.getInt16(offset, true);
        offset += 2;
      }
      const hiddenWeights = new Int8Array(hiddenWeightCount);
      for (let i = 0; i < hiddenWeightCount; i += 1) {
        hiddenWeights[i] = view.getInt8(offset);
        offset += 1;
      }
      const outputWeights = new Int8Array(outputWeightCount);
      for (let i = 0; i < outputWeightCount; i += 1) {
        outputWeights[i] = view.getInt8(offset);
        offset += 1;
      }
      const dModel = contextFeatures === 0 ? embeddingDim + 1 : seqLen * embeddingDim + 1;
      const headDim = headLayout === 0 ? dModel : headLayout === 1 ? hiddenDim : dModel + hiddenDim;
      return {
        seqLen,
        vocabSize,
        embeddingDim,
        contextFeatures,
        hiddenDim,
        headLayout,
        adapterLogitShift,
        dModel,
        headDim,
        embeddings,
        hiddenWeights,
        outputWeights,
        hashes: { embeddingHash, hiddenHash, outputHash, modelHash },
      };
    }
    function normalizeLexemeText(input) {
      let out = "";
      let pendingSpace = false;
      for (const raw of input.normalize("NFKD")) {
        const ch = raw.toLowerCase();
        const ok = (ch >= "a" && ch <= "z") || ".,;:?!'-".includes(ch);
        if (ok) {
          if (pendingSpace && out) out += " ";
          out += ch;
          pendingSpace = false;
        } else {
          pendingSpace = true;
        }
      }
      return out;
    }
    function lexemesForText(text) {
      const input = normalizeLexemeText(text);
      const lexemes = [];
      let index = 0;
      const isWord = ch => ch >= "a" && ch <= "z";
      while (index < input.length) {
        const ch = input[index];
        if (ch === " ") {
          index += 1;
          continue;
        }
        if (isWord(ch)) {
          const start = index;
          index += 1;
          while (index < input.length) {
            const current = input[index];
            const joins = (current === "'" || current === "-")
              && index + 1 < input.length
              && isWord(input[index - 1])
              && isWord(input[index + 1]);
            if (isWord(current) || joins) {
              index += 1;
            } else {
              break;
            }
          }
          lexemes.push(input.slice(start, index));
          continue;
        }
        if (".,;:?!'-".includes(ch)) lexemes.push(ch);
        index += 1;
      }
      return lexemes;
    }
    function encodePrompt(prompt, vocab) {
      const tokens = [];
      for (const lexeme of lexemesForText(prompt)) {
        if (vocab.lookup.has(lexeme)) {
          tokens.push(vocab.lookup.get(lexeme));
        } else {
          for (const ch of lexeme) tokens.push(ch.charCodeAt(0));
        }
      }
      return tokens.length > 0 ? tokens : [32];
    }
    function roundShift(value, shift) {
      if (shift <= 0) return Math.trunc(value);
      const divisor = 2 ** shift;
      const half = divisor / 2;
      return value >= 0
        ? Math.floor((value + half) / divisor)
        : Math.ceil((value - half) / divisor);
    }
    function sat16(value) {
      return Math.max(-32768, Math.min(32767, Math.trunc(value)));
    }
    function hardSilu(value) {
      const gate = Math.max(0, Math.min(32767, (value >> 2) + 16384));
      return sat16(roundShift(value * gate, 15));
    }
    function wasmDot(features, weights, rowStart, dim) {
      const wasm = chatState.wasm;
      const featurePtr = 0;
      const weightPtr = 8192;
      const mem8 = new Uint8Array(wasm.memory.buffer);
      const mem16 = new Int16Array(wasm.memory.buffer);
      for (let i = 0; i < dim; i += 1) mem16[i] = features[i];
      for (let i = 0; i < dim; i += 1) mem8[weightPtr + i] = weights[rowStart + i] & 255;
      return wasm.dot(featurePtr, weightPtr, dim);
    }
    function jsDotShift(features, weights, rowStart, dim, shift) {
      let acc = 0;
      for (let i = 0; i < dim; i += 1) acc += features[i] * weights[rowStart + i];
      return sat16(roundShift(acc, shift));
    }
    function contextWindow(tokens, seqLen) {
      if (tokens.length >= seqLen) return tokens.slice(tokens.length - seqLen);
      const pad = tokens[0] ?? 32;
      return Array(seqLen - tokens.length).fill(pad).concat(tokens);
    }
    function contextFeatures(model, window) {
      const features = [32767];
      if (model.contextFeatures === 0) {
        const shift = Math.log2(window.length) | 0;
        for (let dim = 0; dim < model.embeddingDim; dim += 1) {
          let acc = 0;
          for (const token of window) {
            const id = Math.max(0, Math.min(model.vocabSize - 1, token));
            acc += model.embeddings[id * model.embeddingDim + dim];
          }
          features.push(sat16(roundShift(acc, shift)));
        }
      } else {
        for (const token of window) {
          const id = Math.max(0, Math.min(model.vocabSize - 1, token));
          const start = id * model.embeddingDim;
          for (let dim = 0; dim < model.embeddingDim; dim += 1) {
            features.push(model.embeddings[start + dim]);
          }
        }
      }
      return features;
    }
    function headFeatures(model, tokens) {
      const features = contextFeatures(model, contextWindow(tokens, model.seqLen));
      if (model.headLayout === 0) return features;
      const hidden = [];
      for (let row = 0; row < model.hiddenDim; row += 1) {
        const pre = wasmDot(features, model.hiddenWeights, row * model.dModel, model.dModel);
        hidden.push(hardSilu(pre));
      }
      if (model.headLayout === 1) return hidden;
      return features.concat(hidden);
    }
    function scoreToken(model, features, tokenId) {
      const rowStart = tokenId * model.headDim;
      if (model.headLayout === 3) {
        const base = wasmDot(features, model.outputWeights, rowStart, model.dModel);
        const adapter = jsDotShift(
          features.slice(model.dModel),
          model.outputWeights,
          rowStart + model.dModel,
          model.hiddenDim,
          8 + model.adapterLogitShift,
        );
        return sat16(base + adapter);
      }
      return wasmDot(features, model.outputWeights, rowStart, model.headDim);
    }
    function tokenText(tokenId, entries) {
      if (tokenId < 256) return String.fromCharCode(tokenId);
      return entries[tokenId] || "";
    }
    function detokenize(tokens, entries) {
      let out = "";
      for (const token of tokens) {
        const piece = tokenText(token, entries);
        if (!piece) continue;
        if (!out) {
          out = piece;
        } else if (/^[.,;:?!]$/.test(piece)) {
          out += piece;
        } else if (piece === "'" || piece === "-") {
          out += piece;
        } else {
          out += " " + piece;
        }
      }
      return out.replace(/\\s+([.,;:?!])/g, "$1").trim();
    }
    function chooseNext(model, vocab, tokens, generated) {
      const features = headFeatures(model, tokens);
      const forbidden = new Set(["ranked", "voice", "end", "nsrlpageboundary"]);
      let bestToken = 32;
      let bestScore = -Infinity;
      for (let token = 0; token < model.vocabSize; token += 1) {
        const piece = tokenText(token, vocab.entries);
        if (!piece || forbidden.has(piece)) continue;
        if (token < 32 || token === 127) continue;
        if (token < 256 && /^[a-z]$/.test(piece)) continue;
        let score = scoreToken(model, features, token);
        if (generated.slice(-3).includes(token)) score -= 512;
        if (score > bestScore) {
          bestScore = score;
          bestToken = token;
        }
      }
      return bestToken;
    }
    function generateChatLine(model, vocab, prompt, maxNewTokens = 20) {
      const tokens = encodePrompt(prompt, vocab);
      const generated = [];
      for (let step = 0; step < maxNewTokens; step += 1) {
        const next = chooseNext(model, vocab, tokens.concat(generated), generated);
        const piece = tokenText(next, vocab.entries);
        if (!piece || piece === "end") break;
        generated.push(next);
        const text = detokenize(generated, vocab.entries);
        if (/[.!?]$/.test(text) && generated.length >= 3) break;
      }
      return detokenize(generated, vocab.entries);
    }
    function rankedFallback(prompt) {
      const match = String(prompt || "").match(/RANKED:\\s*([\\s\\S]*?)\\n\\s*VOICE:/i);
      return match ? match[1].replace(/\\s+/g, " ").trim() : "";
    }
    function modelLineLooksWeak(line, prompt) {
      if (!line || line.length > 96 || /\b(ranked|voice|assistant|chatbot)\b/i.test(line)) {
        return true;
      }
      const promptTerms = new Set(
        lexemesForText(rankedFallback(prompt))
          .filter(term => term.length >= 2 && !["at", "is", "on", "to"].includes(term))
      );
      if (promptTerms.size === 0) return false;
      const lineTerms = new Set(lexemesForText(line));
      for (const term of promptTerms) {
        if (lineTerms.has(term)) return false;
      }
      return true;
    }
    async function loadChatModelForRun(run) {
      const artifacts = run.artifacts || {};
      if (!artifacts.model?.exists || !artifacts.vocab?.exists) {
        throw new Error("selected run has no model/vocab artifacts");
      }
      const key = `${run.run_name}:${artifacts.model.file}:${artifacts.vocab.file}`;
      if (chatState.key === key && chatState.model && chatState.vocab) {
        return chatState;
      }
      setChatStatus("loading model, vocab, and wasm...");
      await loadChatWasm();
      const [modelResponse, vocabResponse] = await Promise.all([
        fetch(artifactHref(run, artifacts.model)),
        fetch(artifactHref(run, artifacts.vocab)),
      ]);
      if (!modelResponse.ok) throw new Error(`model fetch failed: ${modelResponse.status}`);
      if (!vocabResponse.ok) throw new Error(`vocab fetch failed: ${vocabResponse.status}`);
      const model = parseLexemeModel(new Uint8Array(await modelResponse.arrayBuffer()));
      const vocab = parseVocab(await vocabResponse.text());
      chatState = { key, model, vocab, wasm: chatState.wasm };
      setChatStatus(`loaded seq=${model.seqLen} vocab=${model.vocabSize} head=${model.headDim}`);
      return chatState;
    }
    async function runChatPrompt() {
      const run = allRuns.find(run => run.run_name === selectedRunName) || allRuns[0];
      if (!run) return;
      try {
        const state = await loadChatModelForRun(run);
        const input = document.getElementById("chatPrompt");
        const request = chatPromptFromInput(input?.value || "");
        const prompt = request.prompt;
        pushChatMessage(run.run_name, "user", request.display);
        if (input) input.value = "";
        setChatStatus("generating...");
        const line = generateChatLine(state.model, state.vocab, prompt, 20);
        const fallback = rankedFallback(prompt);
        if (modelLineLooksWeak(line, prompt) && fallback) {
          pushChatMessage(run.run_name, "model", line || "[empty]");
          pushChatMessage(run.run_name, "fallback", fallback);
          setChatStatus("generated in browser; fallback suggested");
        } else {
          pushChatMessage(run.run_name, "model", line || "[empty]");
          setChatStatus("generated in browser");
        }
      } catch (error) {
        setChatStatus("chat unavailable");
        pushChatMessage(run.run_name, "system", String(error.message || error));
      }
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
          <span class="pill ${statusClass(run.status)}">${statusMarkup(run)}</span>
        </div>
        ${progressBar(run)}
        ${troubleBanner(run)}
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
          ${kv("progress_per_mille", estimatedProgress(run).perMille)}
          ${kv("attention", m.attention_kind)}
          ${kv("position", m.position)}
          ${kv("worker_count", m.worker_count)}
          ${kv("best_worker", m.best_worker_index)}
          ${kv("rule_adjustments", m.adaptive_rule_shift_adjustment_count)}
          ${kv("holo_adjustments", m.adaptive_holographic_shift_adjustment_count)}
          ${kv("final_shifts", finalShifts(run))}
          ${kv("model", artifactLink(run, artifacts.model))}
          ${kv("vocab", artifactLink(run, artifacts.vocab))}
          ${kv("eval", artifactLink(run, artifacts.eval_report))}
          ${kv("progress", artifactLink(run, artifacts.progress))}
          ${kv("trace", artifactLink(run, artifacts.trace))}
          ${kv("log", artifactLink(run, artifacts.log))}
        </div>
        ${chatPane(run)}
        ${charts(run)}
        <details>
          <summary>command</summary>
          <pre>${run.command_escaped || ""}</pre>
        </details>
        <details>
          <summary>log tail</summary>
          <pre>${(run.log_tail || []).join("\\n")}</pre>
        </details>`;
      renderChatLog(run.run_name);
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
      if (event.target.closest("#chatLoad")) {
        const run = allRuns.find(run => run.run_name === selectedRunName) || allRuns[0];
        if (run) {
          loadChatModelForRun(run).catch(error => {
            setChatStatus("chat unavailable");
            setChatOutput(String(error.message || error));
          });
        }
        return;
      }
      if (event.target.closest("#chatSend")) {
        runChatPrompt();
        return;
      }
      const row = event.target.closest("[data-run]");
      if (!row) return;
      selectRun(row.dataset.run);
    });
    document.addEventListener("keydown", event => {
      if (event.target?.id !== "chatPrompt") return;
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        runChatPrompt();
      }
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
    parser.add_argument("--project", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--corpus-id", default="")
    parser.add_argument("--corpus-name", default="")
    parser.add_argument("--corpus-file", default="")
    parser.add_argument("--manifest-file", default="")
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
    parser.add_argument("--vocab-file")
    parser.add_argument("--eval-report-file")
    parser.add_argument("--eval-summary-file")
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
    vocab_path = pathlib.Path(args.vocab_file) if args.vocab_file else None
    eval_report_path = pathlib.Path(args.eval_report_file) if args.eval_report_file else None
    eval_summary_path = pathlib.Path(args.eval_summary_file) if args.eval_summary_file else None
    text_path = pathlib.Path(args.text_file) if args.text_file else None

    trace = load_json_line(trace_path) if trace_path else None
    progress_trace = load_json_line(progress_path) if progress_path else None
    metrics_trace = trace or progress_trace
    command_text = read_text(command_path).strip()
    project = infer_project(args, command_text)
    phase = infer_phase(args, command_text)
    run = {
        "run_name": args.run_name,
        "status": args.status,
        "stage": args.stage,
        "project": project,
        "project_label": PROJECT_LABELS.get(project, project),
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, phase),
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
        "command": command_text,
        "command_escaped": html.escape(command_text),
        "log_tail": tail_lines(log_path, 80),
        "metrics": trace_summary(metrics_trace),
        "progress": trace_summary(progress_trace),
        "final": trace_summary(trace),
        "charts": chart_series(trace or progress_trace),
        "artifacts": {
            "progress": artifact("progress", progress_path, run_s3_uri),
            "trace": artifact("trace", trace_path, run_s3_uri),
            "model": artifact("model", model_path, run_s3_uri),
            "vocab": artifact("vocab", vocab_path, run_s3_uri),
            "eval_report": artifact("eval", eval_report_path, run_s3_uri),
            "eval_summary": artifact("eval_summary", eval_summary_path, run_s3_uri),
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
