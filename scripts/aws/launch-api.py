#!/usr/bin/env python3
"""Local HTTP API for launching NSRL EC2 training runs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_S3_URI = "s3://nsrl-training-022118847419-us-east-1/wikibard"
DEFAULT_TOKENS_S3_URI = (
    "s3://nsrl-training-022118847419-us-east-1/"
    "wikibard/corpus/tokens/wiki-bard-corpus.tokens.u8"
)
DEFAULT_ARTIFACT_S3_URI = (
    "s3://nsrl-training-022118847419-us-east-1/"
    "wikibard/artifacts/nsrl-working-trace-summary.tar.gz"
)
DEFAULT_IAM_PROFILE = "NSRLTrainingEc2InstanceProfile"
DEFAULT_RUNNER = "scripts/aws/run-mini-transformer-training.sh"
DEFAULT_DASHBOARD_RENDERER = "scripts/aws/render-dashboard.py"

INSTANCE_HOURLY_USD = {
    "c8g.xlarge": "0.15952",
    "c8g.2xlarge": "0.31904",
    "c8g.4xlarge": "0.63808",
    "c8g.8xlarge": "1.27616",
    "c8g.12xlarge": "1.91424",
    "c8g.16xlarge": "2.55232",
}

RUN_ENV_DEFAULTS = {
    "NSRL_RUN_ROOT": "/mnt/nsrl/aws-runs",
    "NSRL_MODE": "mini-transformer-swarm",
    "NSRL_TOKENS": "/mnt/nsrl/tokens/wiki-bard-corpus.tokens.u8",
    "NSRL_TOKENS_S3_URI": DEFAULT_TOKENS_S3_URI,
    "NSRL_MAX_WINDOWS": "65536",
    "NSRL_SEQ_LEN": "4",
    "NSRL_STRIDE": "1211",
    "NSRL_WINDOW_OFFSET": "0",
    "NSRL_BATCH_WINDOWS": "2",
    "NSRL_BATCH_MODE": "serial",
    "NSRL_MAP_REDUCE_WORKERS": "1",
    "NSRL_EPOCHS": "1",
    "NSRL_OUT_SHIFT": "18",
    "NSRL_MLP_SHIFT": "17",
    "NSRL_EMBED_SHIFT": "13",
    "NSRL_ATTENTION_SHIFT": "22",
    "NSRL_ATTENTION_Q_SHIFT": "18",
    "NSRL_ATTENTION_QK_SHIFT": "16",
    "NSRL_ATTENTION": "linear",
    "NSRL_POSITION": "nope",
    "NSRL_TOKENIZER": "identity",
    "NSRL_TRACE_FORMAT": "json",
    "NSRL_TRACE_DETAIL": "summary",
    "NSRL_SWARM_WORKERS": "0",
    "NSRL_SWARM_COMPOSITION": "average",
    "NSRL_RUSTFLAGS": "-C target-cpu=native",
    "NSRL_ADAPTIVE_RULE_SHIFTS": "1",
    "NSRL_ADAPTIVE_RULE_INTERVAL_BATCHES": "128",
    "NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS": "0",
    "NSRL_SYNC_SECONDS": "30",
    "NSRL_PROGRESS_INTERVAL_BATCHES": "128",
    "NSRL_TERMINATE_ON_EXIT": "1",
    "NSRL_COST_CURRENCY": "USD",
}

FIELD_TO_ENV = {
    "mode": "NSRL_MODE",
    "max_windows": "NSRL_MAX_WINDOWS",
    "seq_len": "NSRL_SEQ_LEN",
    "stride": "NSRL_STRIDE",
    "window_offset": "NSRL_WINDOW_OFFSET",
    "batch_windows": "NSRL_BATCH_WINDOWS",
    "batch_mode": "NSRL_BATCH_MODE",
    "map_reduce_workers": "NSRL_MAP_REDUCE_WORKERS",
    "epochs": "NSRL_EPOCHS",
    "out_shift": "NSRL_OUT_SHIFT",
    "mlp_shift": "NSRL_MLP_SHIFT",
    "embed_shift": "NSRL_EMBED_SHIFT",
    "attention_shift": "NSRL_ATTENTION_SHIFT",
    "attention_q_shift": "NSRL_ATTENTION_Q_SHIFT",
    "attention_qk_shift": "NSRL_ATTENTION_QK_SHIFT",
    "attention": "NSRL_ATTENTION",
    "position": "NSRL_POSITION",
    "tokenizer": "NSRL_TOKENIZER",
    "trace_format": "NSRL_TRACE_FORMAT",
    "trace_detail": "NSRL_TRACE_DETAIL",
    "adaptive_rule_shifts": "NSRL_ADAPTIVE_RULE_SHIFTS",
    "adaptive_rule_interval_batches": "NSRL_ADAPTIVE_RULE_INTERVAL_BATCHES",
    "adaptive_holographic_shifts": "NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS",
    "sync_seconds": "NSRL_SYNC_SECONDS",
    "progress_interval_batches": "NSRL_PROGRESS_INTERVAL_BATCHES",
    "publish_checkpoint": "NSRL_PUBLISH_CHECKPOINT",
    "resume_checkpoint": "NSRL_RESUME_CHECKPOINT",
    "model_s3_uri": "NSRL_MODEL_S3_URI",
    "tokens_s3_uri": "NSRL_TOKENS_S3_URI",
    "train_mode": "NSRL_TRAIN_MODE",
    "swarm_workers": "NSRL_SWARM_WORKERS",
    "swarm_composition": "NSRL_SWARM_COMPOSITION",
    "rustflags": "NSRL_RUSTFLAGS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_command(args: list[str], *, cwd: pathlib.Path, input_text: str | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def aws_base(profile: str | None, region: str | None) -> list[str]:
    args = ["aws"]
    if profile:
        args.extend(["--profile", profile])
    if region:
        args.extend(["--region", region])
    return args


def latest_al2023_arm64_ami(cwd: pathlib.Path, profile: str | None, region: str) -> str:
    query = "sort_by(Images,&CreationDate)[-1].ImageId"
    return run_command(
        aws_base(profile, region)
        + [
            "ec2",
            "describe-images",
            "--owners",
            "amazon",
            "--filters",
            "Name=name,Values=al2023-ami-2023.*-arm64",
            "Name=architecture,Values=arm64",
            "Name=state,Values=available",
            "--query",
            query,
            "--output",
            "text",
        ],
        cwd=cwd,
    )


def git_rev(cwd: pathlib.Path) -> str:
    try:
        return run_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    except subprocess.CalledProcessError:
        return ""


def normalize_run_name(value: str | None) -> str:
    if value:
        allowed = []
        for char in value:
            allowed.append(char if char.isalnum() or char in "-_." else "-")
        name = "".join(allowed).strip("-_.")
        if name:
            return name
    return f"nsrl-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def shell_exports(env: dict[str, str]) -> str:
    lines = []
    for key in sorted(env):
        lines.append(f"export {key}={shlex.quote(str(env[key]))}")
    return "\n".join(lines)


def user_data_script(env: dict[str, str], artifact_s3_uri: str, runner: str) -> str:
    return f"""#!/bin/bash
set -euxo pipefail

exec > >(tee -a /var/log/nsrl-bootstrap.log) 2>&1
shutdown -h +180 || true

export AWS_REGION={shlex.quote(env.get("AWS_REGION", "us-east-1"))}
export AWS_DEFAULT_REGION={shlex.quote(env.get("AWS_DEFAULT_REGION", env.get("AWS_REGION", "us-east-1")))}
export HOME=/root

# Skip the full `dnf update` (re-patching the whole OS adds minutes per launch);
# install only the toolchain we need.
dnf install -y --allowerasing awscli git gzip make gcc gcc-c++ openssl-devel pkgconf-pkg-config python3 tar zstd

mkdir -p /opt/nsrl /mnt/nsrl/tokens /mnt/nsrl/aws-runs
cd /opt

aws s3 cp {shlex.quote(artifact_s3_uri)} /tmp/nsrl.tar.gz
tar -xzf /tmp/nsrl.tar.gz -C /opt/nsrl
cd /opt/nsrl

curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
source /root/.cargo/env

cat >/opt/nsrl/run-nsrl-training.sh <<'RUNNER'
#!/bin/bash
set -euxo pipefail

export HOME=/root
source /root/.cargo/env
cd /opt/nsrl

{shell_exports(env)}

{runner}
RUNNER

chmod +x /opt/nsrl/run-nsrl-training.sh

cat >/etc/systemd/system/nsrl-training.service <<'SERVICE'
[Unit]
Description=NSRL cloud training run
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/nsrl
ExecStart=/opt/nsrl/run-nsrl-training.sh
Restart=no
StandardOutput=append:/var/log/nsrl-training.log
StandardError=append:/var/log/nsrl-training.log

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable nsrl-training.service
systemctl start nsrl-training.service
"""


class Launcher:
    def __init__(self, args: argparse.Namespace):
        self.repo = pathlib.Path(args.repo).resolve()
        self.profile = args.profile
        self.region = args.region
        self.s3_uri = args.s3_uri.rstrip("/")
        self.artifact_s3_uri = args.artifact_s3_uri
        self.iam_instance_profile = args.iam_instance_profile
        self.default_instance_type = args.instance_type
        self.ami_id = args.ami_id
        self.subnet_id = args.subnet_id
        self.security_group_ids = args.security_group_ids
        self.key_name = args.key_name
        self.launch_root = pathlib.Path(args.launch_root)
        self.dashboard_dir = pathlib.Path(args.dashboard_dir)
        self.runner = args.runner

    def run_dir(self, run_name: str) -> pathlib.Path:
        return self.launch_root / run_name

    def normalize_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        run_name = normalize_run_name(spec.get("run_name"))
        instance_type = str(spec.get("instance_type") or self.default_instance_type)
        hourly_usd = str(
            spec.get("hourly_usd")
            or INSTANCE_HOURLY_USD.get(instance_type)
            or INSTANCE_HOURLY_USD.get(self.default_instance_type, "")
        )
        env = dict(RUN_ENV_DEFAULTS)
        env.update(
            {
                "AWS_REGION": self.region,
                "AWS_DEFAULT_REGION": self.region,
                "NSRL_S3_URI": self.s3_uri,
                "NSRL_RUN_NAME": run_name,
                "NSRL_INSTANCE_HOURLY_USD": hourly_usd,
            }
        )
        for field, env_key in FIELD_TO_ENV.items():
            if field in spec and spec[field] is not None:
                value = spec[field]
                if isinstance(value, bool):
                    value = "1" if value else "0"
                env[env_key] = str(value)
        for key, value in (spec.get("env") or {}).items():
            if not key.startswith("NSRL_") and key not in {"AWS_REGION", "AWS_DEFAULT_REGION"}:
                raise ValueError(f"unsupported env key: {key}")
            env[key] = str(value)
        return {
            "run_name": run_name,
            "instance_type": instance_type,
            "hourly_usd": hourly_usd,
            "env": env,
            "dry_run": bool(spec.get("dry_run", False)),
        }

    def render_pending(
        self,
        normalized: dict[str, Any],
        run_dir: pathlib.Path,
        status: str,
        stage: str,
        instance_id: str = "",
    ) -> None:
        now = utc_now()
        command_file = run_dir / "launch.json"
        user_data_file = run_dir / "user-data.sh"
        command_file.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
        env = normalized["env"]
        args = [
            sys.executable,
            DEFAULT_DASHBOARD_RENDERER,
            "--run-dir",
            str(run_dir),
            "--dashboard-dir",
            str(self.dashboard_dir),
            "--run-name",
            normalized["run_name"],
            "--s3-uri",
            self.s3_uri,
            "--status",
            status,
            "--stage",
            stage,
            "--started-at",
            now,
            "--updated-at",
            now,
            "--repo-rev",
            git_rev(self.repo),
            "--tokens",
            env.get("NSRL_TOKENS_S3_URI", env.get("NSRL_TOKENS", "")),
            "--instance-id",
            instance_id,
            "--instance-type",
            normalized["instance_type"],
            "--instance-region",
            self.region,
            "--cost-hourly-usd",
            normalized["hourly_usd"],
            "--cost-currency",
            env.get("NSRL_COST_CURRENCY", "USD"),
            "--command-file",
            str(command_file),
            "--log-file",
            str(run_dir / "train.log"),
            "--progress-file",
            str(run_dir / f"{normalized['run_name']}.progress.jsonl"),
            "--trace-file",
            str(run_dir / f"{normalized['run_name']}.trace.jsonl"),
            "--model-file",
            str(run_dir / f"{normalized['run_name']}.nsrlmt"),
            "--text-file",
            str(user_data_file),
        ]
        run_command(args, cwd=self.repo)

    def sync_run_and_dashboard(self, run_dir: pathlib.Path, run_name: str) -> None:
        run_command(
            aws_base(self.profile, self.region)
            + ["s3", "sync", str(run_dir), f"{self.s3_uri}/runs/{run_name}", "--only-show-errors"],
            cwd=self.repo,
        )
        run_command(
            aws_base(self.profile, self.region)
            + [
                "s3",
                "sync",
                str(self.dashboard_dir),
                f"{self.s3_uri}/dashboard",
                "--only-show-errors",
            ],
            cwd=self.repo,
        )

    def launch(self, spec: dict[str, Any]) -> dict[str, Any]:
        normalized = self.normalize_spec(spec)
        run_name = normalized["run_name"]
        run_dir = self.run_dir(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            aws_base(self.profile, self.region)
            + ["s3", "cp", f"{self.s3_uri}/dashboard/runs.json", str(self.dashboard_dir / "runs.json")],
            cwd=self.repo,
        ) if not (self.dashboard_dir / "runs.json").exists() else None

        user_data = user_data_script(normalized["env"], self.artifact_s3_uri, self.runner)
        user_data_path = run_dir / "user-data.sh"
        user_data_path.write_text(user_data, encoding="utf-8")
        user_data_path.chmod(0o600)
        self.render_pending(normalized, run_dir, "planned" if normalized["dry_run"] else "launching", "dry-run" if normalized["dry_run"] else "ec2-requested")
        self.sync_run_and_dashboard(run_dir, run_name)

        if normalized["dry_run"]:
            return {
                "run_name": run_name,
                "dry_run": True,
                "run_dir": str(run_dir),
                "user_data": str(user_data_path),
                "dashboard": f"{self.s3_uri}/dashboard/index.html",
            }

        ami_id = str(spec.get("ami_id") or self.ami_id or latest_al2023_arm64_ami(self.repo, self.profile, self.region))
        tags = [
            {"Key": "Name", "Value": f"nsrl-{run_name}"},
            {"Key": "Project", "Value": "NSRL"},
            {"Key": "NSRLRun", "Value": run_name},
            {"Key": "AutoTerminate", "Value": "true"},
        ]
        for key, value in (spec.get("tags") or {}).items():
            tags.append({"Key": str(key), "Value": str(value)})

        launch_args = aws_base(self.profile, self.region) + [
            "ec2",
            "run-instances",
            "--image-id",
            ami_id,
            "--instance-type",
            normalized["instance_type"],
            "--iam-instance-profile",
            f"Name={self.iam_instance_profile}",
            "--instance-initiated-shutdown-behavior",
            "terminate",
            "--metadata-options",
            "HttpTokens=required,HttpEndpoint=enabled",
            "--user-data",
            f"file://{user_data_path}",
            "--tag-specifications",
            json.dumps(
                [{"ResourceType": "instance", "Tags": tags}],
                separators=(",", ":"),
            ),
            "--query",
            "Instances[0].InstanceId",
            "--output",
            "text",
        ]
        if self.subnet_id:
            launch_args.extend(["--subnet-id", self.subnet_id])
        if self.security_group_ids:
            launch_args.extend(["--security-group-ids", *self.security_group_ids.split(",")])
        if self.key_name:
            launch_args.extend(["--key-name", self.key_name])
        # Spot is ~70% cheaper; runs checkpoint to S3 (NSRL_SYNC_SECONDS) and can
        # resume, so interruptions are tolerable. Opt-in via NSRL_USE_SPOT=1.
        if os.environ.get("NSRL_USE_SPOT", "").strip().lower() in ("1", "true", "yes"):
            launch_args.extend(
                ["--instance-market-options", "MarketType=spot,SpotOptions={SpotInstanceType=one-time}"]
            )

        instance_id = run_command(launch_args, cwd=self.repo)
        normalized["instance_id"] = instance_id
        self.render_pending(normalized, run_dir, "running", "ec2-launched", instance_id)
        self.sync_run_and_dashboard(run_dir, run_name)
        return {
            "run_name": run_name,
            "instance_id": instance_id,
            "instance_type": normalized["instance_type"],
            "ami_id": ami_id,
            "run_s3_uri": f"{self.s3_uri}/runs/{run_name}",
            "dashboard_s3_uri": f"{self.s3_uri}/dashboard/index.html",
            "local_run_dir": str(run_dir),
            "auto_terminate": True,
        }


class Handler(BaseHTTPRequestHandler):
    launcher: Launcher

    def _send(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True, "time": utc_now()})
            return
        if self.path == "/runs":
            runs_path = self.launcher.dashboard_dir / "runs.json"
            if not runs_path.exists():
                self._send(200, [])
                return
            self._send(200, json.loads(runs_path.read_text(encoding="utf-8")))
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/runs", "/runs/dry-run"}:
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            spec = json.loads(raw)
            if self.path == "/runs/dry-run":
                spec["dry_run"] = True
            payload = self.launcher.launch(spec)
            self._send(200, payload)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "staging"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--s3-uri", default=os.environ.get("NSRL_S3_URI", DEFAULT_S3_URI))
    parser.add_argument("--artifact-s3-uri", default=os.environ.get("NSRL_ARTIFACT_S3_URI", DEFAULT_ARTIFACT_S3_URI))
    parser.add_argument("--iam-instance-profile", default=os.environ.get("NSRL_IAM_INSTANCE_PROFILE", DEFAULT_IAM_PROFILE))
    parser.add_argument("--instance-type", default=os.environ.get("NSRL_INSTANCE_TYPE", "c8g.xlarge"))
    parser.add_argument("--ami-id", default=os.environ.get("NSRL_AMI_ID", ""))
    parser.add_argument("--subnet-id", default=os.environ.get("NSRL_SUBNET_ID", ""))
    parser.add_argument("--security-group-ids", default=os.environ.get("NSRL_SECURITY_GROUP_IDS", ""))
    parser.add_argument("--key-name", default=os.environ.get("NSRL_KEY_NAME", ""))
    parser.add_argument("--launch-root", default=os.environ.get("NSRL_LAUNCH_ROOT", "data/aws-launches"))
    parser.add_argument("--dashboard-dir", default=os.environ.get("NSRL_DASHBOARD_DIR", "data/aws-dashboard-live/dashboard"))
    parser.add_argument("--runner", default=os.environ.get("NSRL_RUNNER", DEFAULT_RUNNER))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Handler.launcher = Launcher(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NSRL launch API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
