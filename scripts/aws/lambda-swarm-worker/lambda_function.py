"""AWS Lambda worker for NSRL training jobs.

The original path trains mini-transformer swarm shards. The lexeme path keeps
the readable Crowley Bard recipe on the same Lambda/S3 runner without claiming
that lexeme models are reducible across workers yet.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import time
from urllib.parse import urlparse

import boto3


S3 = boto3.client("s3")
TASK_ROOT = pathlib.Path(os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
NSRL_TRAIN = TASK_ROOT / "bin" / "nsrl-train"
TMP_ROOT = pathlib.Path("/tmp/nsrl-lambda-swarm")


def s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"expected s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def s3_join(prefix: str, *parts: str) -> str:
    return "/".join([prefix.rstrip("/"), *[part.strip("/") for part in parts if part]])


def download_s3(uri: str, path: pathlib.Path) -> None:
    bucket, key = s3_parts(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    S3.download_file(bucket, key, str(path))


def upload_s3(path: pathlib.Path, uri: str) -> None:
    bucket, key = s3_parts(uri)
    S3.upload_file(str(path), bucket, key)


def int_field(event: dict, key: str, default: int) -> int:
    value = event.get(key, default)
    if value is None:
        return default
    return int(value)


def str_field(event: dict, key: str, default: str) -> str:
    value = event.get(key, default)
    if value is None:
        return default
    return str(value)


def append_flag(args: list[str], flag: str, value: str | int) -> None:
    args.extend([flag, str(value)])


def append_optional_int_flag(args: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return
    if numeric > 0:
        append_flag(args, flag, numeric)


def run_command(cmd: list[str]) -> tuple[int, str, int]:
    started = time.time()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed_ms = round((time.time() - started) * 1000)
    return completed.returncode, completed.stdout, elapsed_ms


def lexeme_sample_specs(config: dict) -> list[dict]:
    samples = config.get("samples")
    if isinstance(samples, list) and samples:
        return samples
    return [
        {"label": "world", "prompt": "the world is", "seed": 17, "top_k": 12},
        {"label": "soul", "prompt": "the soul is", "seed": 11, "top_k": 12},
        {"label": "to-be", "prompt": "to be or not to be", "seed": 7, "top_k": 8},
    ]


SIMPLEWIKI_CONTENT_DECODE_DEFAULTS = {
    "corpus_prior_order": 3,
    "corpus_prior_logit_shift": 7,
    "repeat_window": 96,
    "repeat_penalty_shift": 3,
    "max_repeat_run": 2,
    "no_repeat_ngram": 3,
    "decode_frequency_cap": 2048,
    "decode_frequency_min_q15": 2048,
    "decode_frequency_logit_shift": 4,
    "decode_local_frequency_cap": 2,
    "decode_local_frequency_min_q15": 4096,
    "decode_local_frequency_logit_shift": 4,
    "decode_local_frequency_hard_cap": 2,
    "prompt_topic_radius": 2,
    "prompt_topic_min_q15": 4096,
    "prompt_topic_logit_shift": 4,
}


def lexeme_sample_decode_config(run_name: str, config: dict) -> dict:
    recipe = str_field(config, "sample_decode_recipe", "")
    is_simplewiki = "simplewiki" in run_name.lower()
    if recipe == "classic" or (not is_simplewiki and recipe not in {"simplewiki-content", "content"}):
        return config
    merged = dict(SIMPLEWIKI_CONTENT_DECODE_DEFAULTS)
    merged.update(config)
    if not recipe:
        merged["sample_decode_recipe"] = "simplewiki-content"
    return merged


def handle_lexeme_crowley(event: dict, started: float) -> dict:
    run_name = str_field(event, "run_name", "lambda-lexeme-crowley")
    worker_index = int_field(event, "worker_index", 0)
    worker_count = int_field(event, "worker_count", 1)
    output_s3_prefix = str_field(event, "output_s3_prefix", "")
    tokens_s3_uri = str_field(event, "tokens_s3_uri", "")
    vocab_s3_uri = str_field(event, "vocab_s3_uri", "")
    base_model_s3_uri = str_field(event, "base_model_s3_uri", "")
    if not output_s3_prefix or not tokens_s3_uri or not vocab_s3_uri:
        raise ValueError("output_s3_prefix, tokens_s3_uri, and vocab_s3_uri are required")

    config = event.get("config") or {}
    worker_id = f"worker-{worker_index:03d}"
    work_dir = TMP_ROOT / run_name / worker_id
    work_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = work_dir / "tokens.u16"
    vocab_path = work_dir / "v4096.vocab.tsv"
    base_model_path = work_dir / "base.nsrllm"
    embedding_path = work_dir / f"{worker_id}.nsrllex"
    model_path = work_dir / f"{worker_id}.nsrllm"
    embedding_trace_path = work_dir / f"{worker_id}.embedding.trace.jsonl"
    softmax_trace_path = work_dir / f"{worker_id}.softmax.trace.jsonl"
    stdout_path = work_dir / f"{worker_id}.stdout.txt"
    summary_path = work_dir / f"{worker_id}.summary.json"
    sample_dir = work_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    download_started = time.time()
    if not tokens_path.exists() or tokens_path.stat().st_size == 0:
        download_s3(tokens_s3_uri, tokens_path)
    if not vocab_path.exists() or vocab_path.stat().st_size == 0:
        download_s3(vocab_s3_uri, vocab_path)
    if base_model_s3_uri and (not base_model_path.exists() or base_model_path.stat().st_size == 0):
        download_s3(base_model_s3_uri, base_model_path)
    download_ended = time.time()

    vocab_size = int_field(config, "vocab_size", 4096)
    embedding_dim = int_field(config, "embedding_dim", 16)
    frequency_cap = int_field(config, "frequency_cap", 4096)
    embedding_windows = int_field(config, "embedding_windows", 131072)
    embedding_epochs = int_field(config, "embedding_epochs", 1)
    softmax_windows = int_field(config, "softmax_windows", 131072)
    softmax_epochs = int_field(config, "softmax_epochs", 1)
    softmax_seq_len = int_field(config, "softmax_seq_len", 8)
    softmax_lr_shift = int_field(config, "softmax_lr_shift", 21)
    softmax_max_lr_shift = int_field(config, "softmax_max_lr_shift", 23)
    softmax_lr_decay_windows = int_field(
        config,
        "softmax_lr_decay_windows",
        max(1, (softmax_windows * softmax_epochs) // 2),
    )
    softmax_batch_windows = int_field(config, "softmax_batch_windows", 1)
    hidden_dim = int_field(config, "hidden_dim", 0)
    hidden_lr_shift = int_field(config, "hidden_lr_shift", 8)
    adapter_logit_shift = int_field(config, "adapter_logit_shift", 0)
    stride = int_field(config, "stride", 1)
    window_offset = int_field(config, "window_offset", worker_index)
    context_features = str_field(config, "lexeme_context_features", "mean")
    quality_profile = str_field(config, "quality_weight_profile", "cruft-aware")

    embedding_cmd = None
    softmax_input_path = base_model_path if base_model_s3_uri else embedding_path
    if not base_model_s3_uri:
        embedding_cmd = [
            str(NSRL_TRAIN),
            "--mode",
            "lexeme-embedding",
            "--tokens",
            str(tokens_path),
            "--vocab",
            str(vocab_path),
            "--model-out",
            str(embedding_path),
            "--trace",
            str(embedding_trace_path),
            "--vocab-size",
            str(vocab_size),
            "--embedding-dim",
            str(embedding_dim),
            "--context-radius",
            str(int_field(config, "context_radius", 2)),
            "--stride",
            str(stride),
            "--window-offset",
            str(window_offset),
            "--max-windows",
            str(embedding_windows),
            "--epochs",
            str(embedding_epochs),
            "--lr-shift",
            str(int_field(config, "embedding_lr_shift", 8)),
            "--concept-frequency-cap",
            str(frequency_cap),
            "--frequency-weight-min-q15",
            str(int_field(config, "frequency_weight_min_q15", 4096)),
            "--quality-weight-profile",
            quality_profile,
        ]
    softmax_cmd = [
        str(NSRL_TRAIN),
        "--mode",
        "lexeme-softmax",
        "--tokens",
        str(tokens_path),
        "--vocab",
        str(vocab_path),
        "--model",
        str(softmax_input_path),
        "--model-out",
        str(model_path),
        "--trace",
        str(softmax_trace_path),
        "--seq-len",
        str(softmax_seq_len),
        "--lexeme-context-features",
        context_features,
        "--stride",
        str(stride),
        "--window-offset",
        str(window_offset),
        "--max-windows",
        str(softmax_windows),
        "--epochs",
        str(softmax_epochs),
        "--batch-windows",
        str(softmax_batch_windows),
        "--lr-shift",
        str(softmax_lr_shift),
        "--lr-shift-decay-windows",
        str(softmax_lr_decay_windows),
        "--lr-shift-decay-step",
        str(int_field(config, "softmax_lr_decay_step", 1)),
        "--max-lr-shift",
        str(softmax_max_lr_shift),
        "--max-weight-delta",
        str(int_field(config, "max_weight_delta", 1)),
        "--max-embedding-delta",
        str(int_field(config, "max_embedding_delta", 1)),
        "--max-hidden-weight-delta",
        str(int_field(config, "max_hidden_weight_delta", 1)),
        "--target-frequency-cap",
        str(frequency_cap),
        "--frequency-weight-min-q15",
        str(int_field(config, "frequency_weight_min_q15", 4096)),
        "--quality-weight-profile",
        quality_profile,
        "--embed-lr-shift",
        str(int_field(config, "embedding_lr_shift", 8)),
    ]
    if hidden_dim > 0:
        append_flag(softmax_cmd, "--lexeme-hidden-dim", hidden_dim)
        append_flag(softmax_cmd, "--lexeme-hidden-lr-shift", hidden_lr_shift)
    if adapter_logit_shift > 0:
        append_flag(softmax_cmd, "--lexeme-adapter-logit-shift", adapter_logit_shift)
    if bool(config.get("train_embeddings", False)):
        softmax_cmd.append("--train-lexeme-embeddings")

    stdout_chunks = []
    command_records = []
    train_started = time.time()
    commands = []
    if embedding_cmd is not None:
        commands.append(("embedding", embedding_cmd))
    commands.append(("softmax", softmax_cmd))
    for label, cmd in commands:
        returncode, output, elapsed_ms = run_command(cmd)
        stdout_chunks.append(f"## {label}\n$ {' '.join(shlex.quote(part) for part in cmd)}\n{output}")
        command_records.append(
            {
                "label": label,
                "returncode": returncode,
                "elapsed_ms": elapsed_ms,
                "command": " ".join(shlex.quote(part) for part in cmd),
            }
        )
        if returncode != 0:
            train_ended = time.time()
            stdout_path.write_text("\n".join(stdout_chunks), encoding="utf-8")
            summary = {
                "schema": "nsrl.lambda_lexeme_crowley_summary.v1",
                "run_name": run_name,
                "worker_index": worker_index,
                "worker_count": worker_count,
                "job_kind": "lexeme-crowley",
                "ok": False,
                "failed_step": label,
                "returncode": returncode,
                "elapsed_ms": round((time.time() - started) * 1000),
                "download_ms": round((download_ended - download_started) * 1000),
                "train_ms": round((train_ended - train_started) * 1000),
                "commands": command_records,
                "stdout_tail": output[-4000:],
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            upload_s3(stdout_path, s3_join(output_s3_prefix, "workers", f"{worker_id}.stdout.txt"))
            upload_s3(summary_path, s3_join(output_s3_prefix, "workers", f"{worker_id}.summary.json"))
            raise RuntimeError(f"lexeme worker {worker_index} failed during {label}")

    sample_cfg = lexeme_sample_decode_config(run_name, config)
    sample_records = []
    for sample in lexeme_sample_specs(sample_cfg):
        label = str(sample.get("label", "sample")).replace("/", "-")
        prompt = str(sample.get("prompt", "the world is"))
        sample_text = sample_dir / f"{label}.txt"
        sample_trace = sample_dir / f"{label}.trace.jsonl"
        generate_cmd = [
            str(NSRL_TRAIN),
            "--mode",
            "lexeme-generate",
            "--model",
            str(model_path),
            "--vocab",
            str(vocab_path),
            "--tokens",
            str(tokens_path),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(int(sample.get("max_new_tokens", int_field(sample_cfg, "sample_max_new_tokens", 64)))),
            "--decode-profile",
            str_field(sample_cfg, "sample_decode_profile", "coherent-prose"),
            "--sample-seed",
            str(int(sample.get("seed", 17))),
            "--top-k",
            str(int(sample.get("top_k", 12))),
            "--corpus-prior",
            "--corpus-prior-logit-shift",
            str(int_field(sample_cfg, "corpus_prior_logit_shift", 7)),
            "--corpus-prior-order",
            str(int_field(sample_cfg, "corpus_prior_order", 2)),
            "--repeat-window",
            str(int_field(sample_cfg, "repeat_window", 80)),
            "--repeat-penalty-shift",
            str(int_field(sample_cfg, "repeat_penalty_shift", 3)),
            "--max-repeat-run",
            str(int_field(sample_cfg, "max_repeat_run", 2)),
            "--no-repeat-ngram",
            str(int_field(sample_cfg, "no_repeat_ngram", 3)),
            "--generated-only",
            "--stop-on-sentence-terminal",
        ]
        for flag, key in [
            ("--decode-frequency-cap", "decode_frequency_cap"),
            ("--decode-frequency-min-q15", "decode_frequency_min_q15"),
            ("--decode-frequency-logit-shift", "decode_frequency_logit_shift"),
            ("--decode-local-frequency-cap", "decode_local_frequency_cap"),
            ("--decode-local-frequency-min-q15", "decode_local_frequency_min_q15"),
            ("--decode-local-frequency-logit-shift", "decode_local_frequency_logit_shift"),
            ("--decode-local-frequency-hard-cap", "decode_local_frequency_hard_cap"),
            ("--prompt-topic-radius", "prompt_topic_radius"),
            ("--prompt-topic-min-q15", "prompt_topic_min_q15"),
            ("--prompt-topic-logit-shift", "prompt_topic_logit_shift"),
        ]:
            append_optional_int_flag(generate_cmd, flag, sample_cfg.get(key))
        generate_cmd.extend(
            [
                "--text-out",
                str(sample_text),
                "--trace",
                str(sample_trace),
            ]
        )
        returncode, output, elapsed_ms = run_command(generate_cmd)
        stdout_chunks.append(f"## generate {label}\n$ {' '.join(shlex.quote(part) for part in generate_cmd)}\n{output}")
        if returncode != 0:
            raise RuntimeError(f"lexeme generation sample {label} failed")
        text = sample_text.read_text(encoding="utf-8") if sample_text.exists() else ""
        text_s3 = s3_join(output_s3_prefix, "samples", f"{worker_id}.{label}.txt")
        trace_s3 = s3_join(output_s3_prefix, "samples", f"{worker_id}.{label}.trace.jsonl")
        upload_s3(sample_text, text_s3)
        upload_s3(sample_trace, trace_s3)
        sample_records.append(
            {
                "label": label,
                "prompt": prompt,
                "elapsed_ms": elapsed_ms,
                "chars": len(text),
                "decode_recipe": str_field(sample_cfg, "sample_decode_recipe", "classic"),
                "text": text.strip(),
                "text_s3_uri": text_s3,
                "trace_s3_uri": trace_s3,
            }
        )
    train_ended = time.time()
    stdout_path.write_text("\n".join(stdout_chunks), encoding="utf-8")

    embedding_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.nsrllex")
    model_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.nsrllm")
    embedding_trace_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.embedding.trace.jsonl")
    softmax_trace_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.softmax.trace.jsonl")
    stdout_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.stdout.txt")
    summary_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.summary.json")
    if embedding_cmd is not None:
        upload_s3(embedding_path, embedding_s3)
        upload_s3(embedding_trace_path, embedding_trace_s3)
    upload_s3(model_path, model_s3)
    upload_s3(softmax_trace_path, softmax_trace_s3)
    upload_s3(stdout_path, stdout_s3)

    summary = {
        "schema": "nsrl.lambda_lexeme_crowley_summary.v1",
        "run_name": run_name,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "job_kind": "lexeme-crowley",
        "ok": True,
        "returncode": 0,
        "started_at_epoch_ms": round(started * 1000),
        "finished_at_epoch_ms": round(time.time() * 1000),
        "elapsed_ms": round((time.time() - started) * 1000),
        "download_ms": round((download_ended - download_started) * 1000),
        "train_ms": round((train_ended - train_started) * 1000),
        "embedding_bytes": embedding_path.stat().st_size if embedding_path.exists() else 0,
        "model_bytes": model_path.stat().st_size,
        "embedding_trace_bytes": embedding_trace_path.stat().st_size if embedding_trace_path.exists() else 0,
        "softmax_trace_bytes": softmax_trace_path.stat().st_size,
        "base_model_s3_uri": base_model_s3_uri,
        "embedding_s3_uri": embedding_s3 if embedding_cmd is not None else None,
        "model_s3_uri": model_s3,
        "embedding_trace_s3_uri": embedding_trace_s3 if embedding_cmd is not None else None,
        "softmax_trace_s3_uri": softmax_trace_s3,
        "stdout_s3_uri": stdout_s3,
        "summary_s3_uri": summary_s3,
        "tokens_s3_uri": tokens_s3_uri,
        "vocab_s3_uri": vocab_s3_uri,
        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "frequency_cap": frequency_cap,
            "embedding_windows": embedding_windows,
            "embedding_epochs": embedding_epochs,
            "softmax_windows": softmax_windows,
            "softmax_epochs": softmax_epochs,
            "softmax_seq_len": softmax_seq_len,
            "softmax_batch_windows": softmax_batch_windows,
            "hidden_dim": hidden_dim,
            "hidden_lr_shift": hidden_lr_shift,
            "adapter_logit_shift": adapter_logit_shift,
            "stride": stride,
            "window_offset": window_offset,
            "lexeme_context_features": context_features,
            "quality_weight_profile": quality_profile,
            "train_embeddings": bool(config.get("train_embeddings", False)),
            "max_weight_delta": int_field(config, "max_weight_delta", 1),
            "max_embedding_delta": int_field(config, "max_embedding_delta", 1),
            "max_hidden_weight_delta": int_field(config, "max_hidden_weight_delta", 1),
            "sample_decode_recipe": str_field(sample_cfg, "sample_decode_recipe", "classic"),
        },
        "commands": command_records,
        "samples": sample_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload_s3(summary_path, summary_s3)
    return summary


def handler(event: dict, context) -> dict:  # noqa: ANN001
    started = time.time()
    job_kind = str_field(event, "job_kind", "mini-transformer-swarm-worker")
    if job_kind in {"lexeme-crowley", "lexeme_crowley"}:
        return handle_lexeme_crowley(event, started)

    run_name = str_field(event, "run_name", "lambda-swarm")
    worker_index = int_field(event, "worker_index", 0)
    worker_count = int_field(event, "worker_count", 1)
    output_s3_prefix = str_field(event, "output_s3_prefix", "")
    tokens_s3_uri = str_field(event, "tokens_s3_uri", "")
    if not output_s3_prefix or not tokens_s3_uri:
        raise ValueError("output_s3_prefix and tokens_s3_uri are required")

    config = event.get("config") or {}
    worker_id = f"worker-{worker_index:03d}"
    work_dir = TMP_ROOT / run_name / worker_id
    work_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = work_dir / "tokens.u8"
    artifact_path = work_dir / f"{worker_id}.nsrlwk"
    trace_path = work_dir / f"{worker_id}.trace.jsonl"
    progress_path = work_dir / f"{worker_id}.progress.jsonl"
    stdout_path = work_dir / f"{worker_id}.stdout.txt"
    summary_path = work_dir / f"{worker_id}.summary.json"

    download_started = time.time()
    if not tokens_path.exists() or tokens_path.stat().st_size == 0:
        download_s3(tokens_s3_uri, tokens_path)
    download_ended = time.time()

    cmd = [
        str(NSRL_TRAIN),
        "--mode",
        "mini-transformer-swarm-worker",
        "--tokens",
        str(tokens_path),
        "--swarm-worker-index",
        str(worker_index),
        "--swarm-worker-count",
        str(worker_count),
        "--swarm-worker-out",
        str(artifact_path),
        "--trace",
        str(trace_path),
        "--trace-format",
        "json",
        "--mini-transformer-trace-detail",
        str_field(config, "trace_detail", "none"),
        "--progress-out",
        str(progress_path),
        "--progress-interval-batches",
        str(int_field(config, "progress_interval_batches", 1024)),
    ]

    for flag, key, default in [
        ("--seq-len", "seq_len", 8),
        ("--stride", "stride", 1),
        ("--window-offset", "window_offset", 0),
        ("--batch-windows", "batch_windows", 2),
        ("--max-windows", "max_windows", 65536),
        ("--epochs", "epochs", 1),
        ("--lr-shift", "out_shift", 18),
        ("--mlp-lr-shift", "mlp_shift", 17),
        ("--embed-lr-shift", "embed_shift", 13),
        ("--attention-lr-shift", "attention_shift", 22),
        ("--attention-q-lr-shift", "attention_q_shift", 18),
        ("--attention-qk-lr-shift", "attention_qk_shift", 16),
    ]:
        append_flag(cmd, flag, int_field(config, key, default))

    append_flag(cmd, "--mini-transformer-attention", str_field(config, "attention", "linear"))
    append_flag(cmd, "--mini-transformer-position", str_field(config, "position", "nope"))
    append_flag(cmd, "--tokenizer", str_field(config, "tokenizer", "ascii-lower"))
    append_flag(cmd, "--mini-transformer-batch-mode", str_field(config, "batch_mode", "map-reduce"))
    append_flag(
        cmd,
        "--mini-transformer-map-reduce-workers",
        int_field(config, "map_reduce_workers", 2),
    )

    if int_field(config, "adaptive_rule_shifts", 1) != 0:
        cmd.append("--adaptive-rule-shifts")
        append_flag(
            cmd,
            "--adaptive-rule-interval-batches",
            int_field(config, "adaptive_rule_interval_batches", 128),
        )
    if int_field(config, "adaptive_holographic_shifts", 0) != 0:
        cmd.append("--adaptive-holographic-shifts")
    if int_field(config, "reject_loss_regression", 0) != 0:
        cmd.append("--reject-loss-regression")

    train_started = time.time()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    train_ended = time.time()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        summary = {
            "schema": "nsrl.lambda_swarm_worker_summary.v1",
            "run_name": run_name,
            "worker_index": worker_index,
            "worker_count": worker_count,
            "ok": False,
            "returncode": completed.returncode,
            "elapsed_ms": round((time.time() - started) * 1000),
            "download_ms": round((download_ended - download_started) * 1000),
            "train_ms": round((train_ended - train_started) * 1000),
            "command": " ".join(shlex.quote(part) for part in cmd),
            "stdout_tail": completed.stdout[-4000:],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        upload_s3(stdout_path, s3_join(output_s3_prefix, "workers", f"{worker_id}.stdout.txt"))
        upload_s3(summary_path, s3_join(output_s3_prefix, "workers", f"{worker_id}.summary.json"))
        raise RuntimeError(f"worker {worker_index} failed with exit {completed.returncode}")

    artifact_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.nsrlwk")
    trace_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.trace.jsonl")
    progress_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.progress.jsonl")
    stdout_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.stdout.txt")
    summary_s3 = s3_join(output_s3_prefix, "workers", f"{worker_id}.summary.json")
    upload_s3(artifact_path, artifact_s3)
    upload_s3(trace_path, trace_s3)
    if progress_path.exists():
        upload_s3(progress_path, progress_s3)
    upload_s3(stdout_path, stdout_s3)

    summary = {
        "schema": "nsrl.lambda_swarm_worker_summary.v1",
        "run_name": run_name,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "ok": True,
        "returncode": completed.returncode,
        "started_at_epoch_ms": round(started * 1000),
        "finished_at_epoch_ms": round(time.time() * 1000),
        "elapsed_ms": round((time.time() - started) * 1000),
        "download_ms": round((download_ended - download_started) * 1000),
        "train_ms": round((train_ended - train_started) * 1000),
        "artifact_bytes": artifact_path.stat().st_size,
        "trace_bytes": trace_path.stat().st_size,
        "artifact_s3_uri": artifact_s3,
        "trace_s3_uri": trace_s3,
        "progress_s3_uri": progress_s3 if progress_path.exists() else None,
        "stdout_s3_uri": stdout_s3,
        "summary_s3_uri": summary_s3,
        "command": " ".join(shlex.quote(part) for part in cmd),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload_s3(summary_path, summary_s3)
    return summary
