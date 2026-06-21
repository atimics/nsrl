#!/usr/bin/env python3
"""Export NSRL S3 dashboard runs to TSV or Markdown."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


COLUMNS = [
    "run_name",
    "status",
    "elapsed_seconds",
    "max_windows",
    "examined_windows",
    "seq_len",
    "batch_windows",
    "attention_kind",
    "position",
    "probability_error_delta_i32",
    "final_accuracy_per_mille",
    "rollback_count",
    "rejected_batch_count",
    "adaptive_rule_shift_adjustment_count",
    "adaptive_holographic_shift_adjustment_count",
    "final_shifts",
    "model_s3_uri",
    "trace_s3_uri",
]


def value(run: dict[str, Any], column: str) -> Any:
    metrics = run.get("metrics") or {}
    artifacts = run.get("artifacts") or {}
    if column in run:
        return run[column]
    if column == "final_shifts":
        keys = [
            "final_output_learning_rate_shift",
            "final_mlp_learning_rate_shift",
            "final_embedding_learning_rate_shift",
            "final_attention_learning_rate_shift",
            "final_attention_q_learning_rate_shift",
            "final_attention_qk_learning_rate_shift",
        ]
        return "/".join(str(metrics.get(key)) for key in keys if metrics.get(key) is not None)
    if column == "model_s3_uri":
        return (artifacts.get("model") or {}).get("s3_uri")
    if column == "trace_s3_uri":
        return (artifacts.get("trace") or {}).get("s3_uri")
    return metrics.get(column)


def cell(run: dict[str, Any], column: str) -> str:
    current = value(run, column)
    if current is None:
        return ""
    return str(current)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", choices=["tsv", "md"], default="tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = json.loads(pathlib.Path(args.runs_json).read_text(encoding="utf-8"))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "tsv":
        lines = ["\t".join(COLUMNS)]
        for run in runs:
            lines.append("\t".join(cell(run, column) for column in COLUMNS))
    else:
        lines = [
            "| " + " | ".join(COLUMNS) + " |",
            "| " + " | ".join("---" for _ in COLUMNS) + " |",
        ]
        for run in runs:
            lines.append(
                "| "
                + " | ".join(cell(run, column).replace("|", "\\|") for column in COLUMNS)
                + " |"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
