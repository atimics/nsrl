#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="${NSRL_AWS_PRELAUNCH_READINESS_CHECK_ROOT:-/tmp/nsrl-aws-prelaunch-readiness-check}"
name="${NSRL_AWS_PRELAUNCH_READINESS_CHECK_NAME:-solomon-prelaunch-readiness-check-${timestamp}-$$}"
run_dir="${root%/}/${name}"
self_test="${NSRL_AWS_PRELAUNCH_READINESS_SELF_TEST:-1}"

mkdir -p "$run_dir"

NSRL_SOLOMON_PRODUCT_LAUNCH_ROOT="$root" \
NSRL_PIPELINE_RUN_NAME="$name" \
NSRL_S3_URI="${NSRL_S3_URI:-s3://nsrl-product-plan-check/solomon}" \
NSRL_ARTIFACT_S3_URI="${NSRL_ARTIFACT_S3_URI:-s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz}" \
NSRL_AMI_ID="${NSRL_AMI_ID:-ami-0123456789abcdef0}" \
NSRL_IAM_INSTANCE_PROFILE="${NSRL_IAM_INSTANCE_PROFILE:-NSRLTrainingEc2InstanceProfile}" \
NSRL_SUBNET_ID="${NSRL_SUBNET_ID:-subnet-0123456789abcdef0}" \
NSRL_SECURITY_GROUP_IDS="${NSRL_SECURITY_GROUP_IDS:-sg-0123456789abcdef0 sg-0fedcba9876543210}" \
  scripts/aws/launch-solomon-product-run.sh --dry-run --out-dir "$run_dir"

node scripts/check-solomon-aws-prelaunch-readiness.mjs \
  --launch-dir "$run_dir" \
  --out "$run_dir/prelaunch-readiness-check.json"

if [[ "$self_test" != "0" ]]; then
  broken_dir="${run_dir}-broken"
  rm -rf "$broken_dir"
  cp -R "$run_dir" "$broken_dir"
  node -e '
const fs = require("node:fs");
const path = process.argv[1];
const launchPath = `${path}/launch.json`;
const launch = JSON.parse(fs.readFileSync(launchPath, "utf8"));
launch.ami_id = "ami-required-for-execute";
launch.instance_type = "m7i.4xlarge";
launch.iam_instance_profile = "";
delete launch.env.NSRL_SOLOMON_REQUIRE_EC2_METADATA;
launch.s3_uri = "none";
launch.s3_pipeline_uri = "none";
launch.artifact_s3_uri = "s3://wrong-bucket/artifacts/nsrl.tar.gz";
launch.security_group_ids = "not-a-security-group";
launch.post_run_proof_command = ["scripts/aws/prove-solomon-product-run.sh", "--s3-pipeline-uri", "s3://wrong/pipelines/run"];
const userDataIndex = launch.command.indexOf("--user-data");
if (userDataIndex >= 0) {
  launch.command[userDataIndex + 1] = "file:///tmp/not-solomon-user-data.sh";
}
launch.command = launch.command.filter((item) => {
  const text = String(item);
  return item !== "--tag-specifications" && !text.includes("Key=Project,Value=nsrl-solomon");
});
fs.writeFileSync(launchPath, `${JSON.stringify(launch, null, 2)}\n`);
const userDataPath = `${path}/user-data.sh`;
const userData = fs.readFileSync(userDataPath, "utf8")
  .replace(/\nexport NSRL_SOLOMON_REQUIRE_EC2_METADATA=1/g, "");
fs.writeFileSync(userDataPath, userData);
' "$broken_dir"
  set +e
  node scripts/check-solomon-aws-prelaunch-readiness.mjs \
    --launch-dir "$broken_dir" \
    --out "$broken_dir/prelaunch-readiness-check.json"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "expected broken prelaunch readiness check to fail" >&2
    exit 1
  fi
  echo "solomon_aws_prelaunch_readiness_negative_check: $broken_dir/prelaunch-readiness-check.json"

  execute_guard_dir="${run_dir}-execute-guard"
  fake_bin="${execute_guard_dir}/bin"
  fake_aws_marker="${execute_guard_dir}/aws-was-called"
  rm -rf "$execute_guard_dir"
  mkdir -p "$fake_bin"
  cat > "${fake_bin}/aws" <<FAKEAWS
#!/usr/bin/env bash
echo "aws was called" > "$fake_aws_marker"
exit 99
FAKEAWS
  chmod +x "${fake_bin}/aws"
  set +e
  PATH="${fake_bin}:$PATH" \
  NSRL_AMI_ID="ami-0123456789abcdef0" \
  NSRL_S3_URI="s3://nsrl-product-plan-check/solomon" \
  NSRL_ARTIFACT_S3_URI="s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz" \
  NSRL_IAM_INSTANCE_PROFILE="NSRLTrainingEc2InstanceProfile" \
  NSRL_SUBNET_ID="subnet-0123456789abcdef0" \
  NSRL_SECURITY_GROUP_IDS="sg-0123456789abcdef0 sg-0fedcba9876543210" \
  NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE="m7i.4xlarge" \
    scripts/aws/launch-solomon-product-run.sh --execute --out-dir "$execute_guard_dir" \
    > "${execute_guard_dir}/launch-execute.log" 2>&1
  execute_status=$?
  set -e
  if [[ "$execute_status" -eq 0 ]]; then
    echo "expected execute prelaunch guard to fail before AWS launch" >&2
    exit 1
  fi
  if [[ -f "$fake_aws_marker" ]]; then
    echo "execute prelaunch guard called aws before failing" >&2
    exit 1
  fi
  echo "solomon_aws_prelaunch_execute_guard_check: ${execute_guard_dir}/prelaunch-readiness-check.json"
fi

echo "solomon_aws_prelaunch_readiness_check: $run_dir/prelaunch-readiness-check.json"
