#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="${NSRL_AWS_LIVE_LAUNCH_READINESS_ROOT:-/tmp/nsrl-aws-live-launch-readiness}"
name="${NSRL_AWS_LIVE_LAUNCH_READINESS_NAME:-solomon-live-launch-readiness-${timestamp}-$$}"
run_dir="${root%/}/${name}"
launch_log="${run_dir}/launch-dry-run.log"
readiness_log="${run_dir}/prelaunch-readiness.log"
readiness_json="${run_dir}/prelaunch-readiness-check.json"
summary_json="${run_dir}/live-launch-readiness.json"

mkdir -p "$run_dir"

set +e
scripts/aws/launch-solomon-product-run.sh --dry-run --out-dir "$run_dir" > "$launch_log" 2>&1
launch_status=$?
set -e

readiness_status=127
if [[ "$launch_status" -eq 0 ]]; then
  set +e
  node scripts/check-solomon-aws-prelaunch-readiness.mjs \
    --launch-dir "$run_dir" \
    --out "$readiness_json" > "$readiness_log" 2>&1
  readiness_status=$?
  set -e
fi

RUN_DIR="$run_dir" \
LAUNCH_STATUS="$launch_status" \
READINESS_STATUS="$readiness_status" \
LAUNCH_LOG="$launch_log" \
READINESS_LOG="$readiness_log" \
READINESS_JSON="$readiness_json" \
SUMMARY_JSON="$summary_json" \
node --input-type=module - <<'NODE'
import fs from "node:fs";
import path from "node:path";

const env = process.env;
const runDir = env.RUN_DIR;
const launchPath = path.join(runDir, "launch.json");
const userDataPath = path.join(runDir, "user-data.sh");
const readinessPath = env.READINESS_JSON;
const launch = readJsonMaybe(launchPath);
const readiness = readJsonMaybe(readinessPath);
const launchStatus = Number(env.LAUNCH_STATUS || 0);
const readinessStatus = Number(env.READINESS_STATUS || 0);
const importantEnv = [
  "AWS_REGION",
  "AWS_PROFILE",
  "NSRL_AMI_ID",
  "NSRL_S3_URI",
  "NSRL_ARTIFACT_S3_URI",
  "NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE",
  "NSRL_IAM_INSTANCE_PROFILE",
  "NSRL_SUBNET_ID",
  "NSRL_SECURITY_GROUP_IDS",
  "NSRL_KEY_NAME",
  "NSRL_PIPELINE_RUN_ROOT",
  "NSRL_PIPELINE_RUN_NAME",
];
const envSummary = Object.fromEntries(
  importantEnv.map((key) => [key, {
    provided: Object.prototype.hasOwnProperty.call(env, key) && String(env[key] || "") !== "",
    value: safeEnvValue(key, env[key] || ""),
  }]),
);
const requiredEnv = [
  "NSRL_AMI_ID",
  "NSRL_S3_URI",
  "NSRL_ARTIFACT_S3_URI",
];
const requiredMissing = requiredEnv.filter((key) => !envSummary[key].provided);
const recommendedMissing = [
  "NSRL_SUBNET_ID",
  "NSRL_SECURITY_GROUP_IDS",
].filter((key) => !envSummary[key].provided);
const report = {
  schema: "nsrl.solomon_aws_live_launch_readiness_check.v1",
  ok: launchStatus === 0 && readinessStatus === 0 && readiness?.ok === true && requiredMissing.length === 0,
  generated_at: new Date().toISOString(),
  run_dir: runDir,
  launch_dry_run: {
    status: launchStatus,
    log: env.LAUNCH_LOG,
    launch_json: fs.existsSync(launchPath) ? launchPath : "",
    user_data: fs.existsSync(userDataPath) ? userDataPath : "",
  },
  prelaunch_readiness: {
    status: readinessStatus,
    log: fs.existsSync(env.READINESS_LOG) ? env.READINESS_LOG : "",
    report: fs.existsSync(readinessPath) ? readinessPath : "",
    ok: readiness?.ok === true,
    errors: Array.isArray(readiness?.errors) ? readiness.errors : [],
  },
  launch: launch ? {
    run_name: launch.run_name || "",
    region: launch.region || "",
    ami_id: launch.ami_id || "",
    instance_type: launch.instance_type || "",
    iam_instance_profile: launch.iam_instance_profile || "",
    subnet_id: launch.subnet_id || "",
    security_group_ids: launch.security_group_ids || "",
    s3_uri: launch.s3_uri || "",
    s3_pipeline_uri: launch.s3_pipeline_uri || "",
    artifact_s3_uri: launch.artifact_s3_uri || "",
    product_stages: String(launch.env?.NSRL_SOLOMON_AWS_STAGES || "").split(",").filter(Boolean),
    cpu_scaling: {
      batch_mode: launch.env?.NSRL_SOLOMON_ATTENTION_BATCH_MODE || "",
      map_reduce_workers: launch.env?.NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS || "",
      policy: launch.env?.NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY || "",
    },
  } : null,
  env: envSummary,
  required_env_missing: requiredMissing,
  recommended_env_missing: recommendedMissing,
};

fs.writeFileSync(env.SUMMARY_JSON, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
if (!report.ok) {
  process.exit(1);
}

function readJsonMaybe(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function safeEnvValue(key, value) {
  if (!value) {
    return "";
  }
  if (key === "AWS_PROFILE" || key === "AWS_REGION" || key === "NSRL_PIPELINE_RUN_NAME") {
    return value;
  }
  if (key === "NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE" || key === "NSRL_IAM_INSTANCE_PROFILE") {
    return value;
  }
  return "<set>";
}
NODE
