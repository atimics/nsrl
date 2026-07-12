#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${NSRL_AWS_LAUNCH_PLAN_CHECK_ROOT:-/tmp/nsrl-aws-launch-plan-check}"
run_name="${NSRL_AWS_LAUNCH_PLAN_CHECK_NAME:-solomon-product-launch-check-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
run_dir="${run_root}/${run_name}"
self_test="${NSRL_AWS_LAUNCH_PLAN_CHECK_SELF_TEST:-1}"

NSRL_SOLOMON_PRODUCT_LAUNCH_ROOT="$run_root" \
NSRL_PIPELINE_RUN_NAME="$run_name" \
NSRL_S3_URI="s3://nsrl-product-plan-check/solomon" \
NSRL_AMI_ID="ami-1234567890abcdef0" \
  scripts/aws/launch-solomon-product-run.sh --dry-run --out-dir "$run_dir"

node scripts/check-solomon-aws-launch-plan.mjs \
  --launch-dir "$run_dir" \
  --out "$run_dir/launch-plan-check.json"

if [[ "$self_test" != "0" ]]; then
  broken_run_dir="${run_dir}-broken"
  mkdir -p "$broken_run_dir"
  cp "$run_dir/launch.json" "$broken_run_dir/launch.json"
  cp "$run_dir/user-data.sh" "$broken_run_dir/user-data.sh"
  node --input-type=module -e '
import fs from "node:fs";
const dir = process.argv[1];
const launch = JSON.parse(fs.readFileSync(`${dir}/launch.json`, "utf8"));
launch.instance_type = "m7i.4xlarge";
delete launch.env.NSRL_SOLOMON_REQUIRE_EC2_METADATA;
launch.env.NSRL_SOLOMON_ATTENTION_BATCH_MODE = "serial";
launch.env.NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS = "1";
launch.env.NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY = "fixed";
launch.post_run_proof_command = ["scripts/aws/prove-solomon-product-run.sh", "--s3-pipeline-uri", "s3://wrong/pipelines/run"];
const userDataIndex = launch.command.indexOf("--user-data");
if (userDataIndex >= 0) {
  launch.command[userDataIndex + 1] = "file:///tmp/not-solomon-user-data.sh";
}
fs.writeFileSync(`${dir}/launch.json`, `${JSON.stringify(launch, null, 2)}\n`);
let userData = fs.readFileSync(`${dir}/user-data.sh`, "utf8");
userData = userData
  .replace(/\nexport NSRL_SOLOMON_REQUIRE_EC2_METADATA=1/g, "")
  .replace(/NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce/g, "NSRL_SOLOMON_ATTENTION_BATCH_MODE=serial")
  .replace(/NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0/g, "NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=1")
  .replace(/NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=auto-online-processors/g, "NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=fixed");
fs.writeFileSync(`${dir}/user-data.sh`, userData);
' "$broken_run_dir"
  if node scripts/check-solomon-aws-launch-plan.mjs \
    --launch-dir "$broken_run_dir" \
    --out "$broken_run_dir/launch-plan-check.json"; then
    echo "expected broken Solomon AWS launch plan to fail" >&2
    exit 1
  fi
  echo "solomon_aws_launch_plan_negative_check: $broken_run_dir/launch-plan-check.json"
fi

echo "solomon_aws_launch_plan_check: $run_dir/launch-plan-check.json"
