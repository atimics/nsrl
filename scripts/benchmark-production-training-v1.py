#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path


def run_timed(command):
    started = time.perf_counter_ns()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return round((time.perf_counter_ns() - started) / 1_000_000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/nsrl-production-model")
    parser.add_argument("--float-runner", default="scripts/production-float-twin-v1.py")
    parser.add_argument("--model", default="data/experiments/production-model-v1/p10m-smoke/initial.nsrlpm")
    parser.add_argument("--tokenizer", default="data/processed/production-corpus-v1/tokenizer.nsrlbpe")
    parser.add_argument("--tokens", default="data/processed/production-corpus-v1/train.nsrltok")
    parser.add_argument("--contexts", default="4,16,64,256")
    parser.add_argument("--out", default="benchmarks/production-model-v1/prepilot-performance.json")
    args = parser.parse_args()

    contexts = [int(value) for value in args.contexts.split(",")]
    if not contexts or any(value <= 0 for value in contexts):
        raise ValueError("contexts must be positive")
    results = []
    with tempfile.TemporaryDirectory(prefix="nsrl-production-bench-") as temporary:
        temporary = Path(temporary)
        for context in contexts:
            integer_trace = temporary / "integer.json"
            integer_ms = run_timed([
                args.binary,
                "full-train-smoke",
                "--tokenizer", args.tokenizer,
                "--tokens", args.tokens,
                "--model", args.model,
                "--model-out", str(temporary / "integer.nsrlpm"),
                "--optimizer-state-out", str(temporary / "integer.nsrlpo"),
                "--trace", str(integer_trace),
                "--context-tokens", str(context),
                "--max-windows", "1",
                "--epochs", "1",
                "--batch-windows", "1",
            ])
            integer = json.loads(integer_trace.read_text())

            float_trace = temporary / "float.json"
            float_ms = run_timed([
                "python3", args.float_runner,
                "--model", args.model,
                "--tokens", args.tokens,
                "--out", str(temporary / "float.npz"),
                "--trace", str(float_trace),
                "--context-tokens", str(context),
                "--max-windows", "1",
                "--epochs", "1",
                "--batch-windows", "1",
                "--allow-partial-gates",
            ])
            float_trace_value = json.loads(float_trace.read_text())
            results.append({
                "context_tokens": context,
                "integer_milliseconds": integer_ms,
                "float_milliseconds": float_ms,
                "integer_weight_saturation_count": integer["health"]["weight_saturation_count"],
                "integer_gradient_saturation_count": integer["health"]["gradient_saturation_count"],
                "integer_optimizer_state_bytes": (temporary / "integer.nsrlpo").stat().st_size,
                "float_attention_algorithm": float_trace_value["training"]["attention_algorithm"],
            })

    output = {
        "schema": "nsrl.production_preflight_performance.v1",
        "profile": "p10m",
        "measurement": "one_full_forward_backward_optimizer_step",
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python": platform.python_version(),
        },
        "results": results,
        "known_non_claims": [
            "single_sample_wall_clock_not_capacity_forecast",
            "includes_process_startup_serialization_and_evaluation",
            "local_machine_measurement_not_cross_platform_benchmark",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
