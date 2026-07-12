#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Plan or launch a Solomon v1 product run on an EC2 Graviton instance.

Default mode is dry-run: write launch.json and user-data.sh, but do not call AWS.
Use --execute only when ready to spend money and launch the instance.

Environment:
  NSRL_SOLOMON_PRODUCT_LAUNCH_ROOT=data/aws-launches
  NSRL_PIPELINE_RUN_NAME=solomon-product-YYYYMMDDTHHMMSSZ
  NSRL_PIPELINE_RUN_ROOT=/mnt/nsrl/aws-pipelines
  NSRL_S3_URI=s3://bucket/prefix
  NSRL_ARTIFACT_S3_URI=s3://bucket/prefix/artifacts/nsrl-working-trace-summary.tar.gz
  NSRL_AMI_ID=ami-...
  NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE=c8g.4xlarge
  NSRL_IAM_INSTANCE_PROFILE=NSRLTrainingEc2InstanceProfile
  NSRL_SUBNET_ID=subnet-...
  NSRL_SECURITY_GROUP_IDS=sg-...
  NSRL_KEY_NAME=optional-ssh-key

Examples:
  scripts/aws/launch-solomon-product-run.sh --dry-run
  NSRL_AMI_ID=ami-... NSRL_S3_URI=s3://bucket/prefix \
    scripts/aws/launch-solomon-product-run.sh --execute
USAGE
}

dry_run=1
out_dir=""
while (($# > 0)); do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      ;;
    --execute)
      dry_run=0
      ;;
    --out-dir)
      shift
      out_dir="${1:-}"
      [[ -n "$out_dir" ]] || { echo "--out-dir requires a value" >&2; exit 2; }
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "try: scripts/aws/launch-solomon-product-run.sh --help" >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
region="${AWS_REGION:-us-east-1}"
launch_root="${NSRL_SOLOMON_PRODUCT_LAUNCH_ROOT:-data/aws-launches}"
run_name="${NSRL_PIPELINE_RUN_NAME:-solomon-product-${timestamp}}"
pipeline_run_root="${NSRL_PIPELINE_RUN_ROOT:-/mnt/nsrl/aws-pipelines}"
pipeline_run_dir="${pipeline_run_root}/${run_name}"
s3_uri="${NSRL_S3_URI:-s3://nsrl-training-022118847419-us-east-1/solomon}"
s3_pipeline_uri="${s3_uri%/}/pipelines/${run_name}"
artifact_s3_uri="${NSRL_ARTIFACT_S3_URI:-${s3_uri%/}/artifacts/nsrl-working-trace-summary.tar.gz}"
ami_id="${NSRL_AMI_ID:-}"
instance_type="${NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE:-c8g.4xlarge}"
iam_profile="${NSRL_IAM_INSTANCE_PROFILE:-NSRLTrainingEc2InstanceProfile}"
subnet_id="${NSRL_SUBNET_ID:-}"
security_group_ids="${NSRL_SECURITY_GROUP_IDS:-}"
key_name="${NSRL_KEY_NAME:-}"
stages="${NSRL_SOLOMON_AWS_STAGES:-dataset,denoiser,prior,generative-eval,attention-curriculum}"
terminate_on_exit="${NSRL_TERMINATE_ON_EXIT:-0}"

if [[ "$dry_run" == "0" && -z "$ami_id" ]]; then
  echo "NSRL_AMI_ID is required for --execute. Bake one with scripts/aws/bake-training-ami.sh." >&2
  exit 2
fi
if [[ "$dry_run" == "0" && -z "${NSRL_S3_URI:-}" ]]; then
  echo "NSRL_S3_URI is required for --execute so release artifacts land in an explicit S3 prefix." >&2
  exit 2
fi
if [[ "$dry_run" == "0" && -z "${NSRL_ARTIFACT_S3_URI:-}" ]]; then
  echo "NSRL_ARTIFACT_S3_URI is required for --execute so the launched instance boots the intended artifact." >&2
  exit 2
fi
if [[ -z "$ami_id" ]]; then
  ami_id="ami-required-for-execute"
fi
if [[ "$s3_uri" != s3://* ]]; then
  echo "NSRL_S3_URI must start with s3://" >&2
  exit 2
fi
if [[ "$artifact_s3_uri" != s3://* ]]; then
  echo "NSRL_ARTIFACT_S3_URI must start with s3://" >&2
  exit 2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="${launch_root}/${run_name}"
fi
mkdir -p "$out_dir"
user_data_path="${out_dir}/user-data.sh"
launch_json_path="${out_dir}/launch.json"
launch_result_path="${out_dir}/launch-result.json"
prelaunch_check_path="${out_dir}/prelaunch-readiness-check.json"

cat > "$user_data_path" <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > >(tee -a /var/log/nsrl-solomon-product.log) 2>&1
shutdown -h +2880 || true

export HOME=/root
export AWS_REGION=${region}
export AWS_DEFAULT_REGION=${region}
export RUSTFLAGS='-C target-cpu=native'
export NSRL_PIPELINE_RUN_ROOT=${pipeline_run_root}
export NSRL_PIPELINE_RUN_NAME=${run_name}
export NSRL_S3_URI=${s3_uri}
export NSRL_SOLOMON_AWS_STAGES=${stages}
export NSRL_SOLOMON_REQUIRE_GRAVITON=1
export NSRL_SOLOMON_REQUIRE_EC2_METADATA=1
export NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=1
export NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce
export NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0
export NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=auto-online-processors

dnf install -y --allowerasing awscli git gzip make gcc gcc-c++ openssl-devel pkgconf-pkg-config python3 tar zstd
mkdir -p /opt/nsrl /mnt/nsrl
aws s3 cp ${artifact_s3_uri} /tmp/nsrl.tar.gz
tar -xzf /tmp/nsrl.tar.gz -C /opt/nsrl
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
else
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  source /root/.cargo/env
fi
cd /opt/nsrl
scripts/aws/run-solomon-end-to-end.sh
aws s3 cp /var/log/nsrl-solomon-product.log ${s3_pipeline_uri}/logs/launch-user-data.log || true
if [[ "${terminate_on_exit}" == "1" ]]; then
  shutdown -h now
fi
USERDATA
chmod +x "$user_data_path"

write_launch_json() {
  LAUNCH_JSON_PATH="$launch_json_path" \
  LAUNCH_RESULT_PATH="$launch_result_path" \
  LAUNCH_DIR="$out_dir" \
  USER_DATA_PATH="$user_data_path" \
  REPO_ROOT="$repo_root" \
  DRY_RUN="$dry_run" \
  REGION="$region" \
  RUN_NAME="$run_name" \
  PIPELINE_RUN_ROOT="$pipeline_run_root" \
  PIPELINE_RUN_DIR="$pipeline_run_dir" \
  S3_URI="$s3_uri" \
  S3_PIPELINE_URI="$s3_pipeline_uri" \
  ARTIFACT_S3_URI="$artifact_s3_uri" \
  AMI_ID="$ami_id" \
  INSTANCE_TYPE="$instance_type" \
  IAM_PROFILE="$iam_profile" \
  SUBNET_ID="$subnet_id" \
  SECURITY_GROUP_IDS="$security_group_ids" \
  KEY_NAME="$key_name" \
  STAGES="$stages" \
  TERMINATE_ON_EXIT="$terminate_on_exit" \
  INSTANCE_ID="${INSTANCE_ID:-}" \
  node --input-type=module -e '
    import crypto from "node:crypto";
    import fs from "node:fs";
    const env = process.env;
    const userData = fs.readFileSync(env.USER_DATA_PATH, "utf8");
    const launchResult = fs.existsSync(env.LAUNCH_RESULT_PATH)
      ? fs.readFileSync(env.LAUNCH_RESULT_PATH, "utf8")
      : "";
    const launch = {
      schema: "nsrl.solomon_aws_product_launch_plan.v1",
      dry_run: env.DRY_RUN === "1",
      created_at: new Date().toISOString(),
      repo_root: env.REPO_ROOT,
      run_name: env.RUN_NAME,
      pipeline_run_root: env.PIPELINE_RUN_ROOT,
      pipeline_run_dir: env.PIPELINE_RUN_DIR,
      s3_uri: env.S3_URI,
      s3_pipeline_uri: env.S3_PIPELINE_URI,
      artifact_s3_uri: env.ARTIFACT_S3_URI,
      region: env.REGION,
      aws_profile: env.AWS_PROFILE || "",
      ami_id: env.AMI_ID,
      instance_type: env.INSTANCE_TYPE,
      iam_instance_profile: env.IAM_PROFILE,
      subnet_id: env.SUBNET_ID,
      security_group_ids: env.SECURITY_GROUP_IDS,
      key_name: env.KEY_NAME,
      tags: {
        Name: env.RUN_NAME,
        Project: "nsrl-solomon",
        Product: "solomon-v1",
      },
      instance_id: env.INSTANCE_ID || "",
      launch_result: launchResult ? env.LAUNCH_RESULT_PATH : "",
      launch_result_sha256: launchResult
        ? crypto.createHash("sha256").update(launchResult).digest("hex")
        : "",
      user_data: env.USER_DATA_PATH,
      user_data_sha256: crypto.createHash("sha256").update(userData).digest("hex"),
      env: {
        NSRL_PIPELINE_RUN_ROOT: env.PIPELINE_RUN_ROOT,
        NSRL_PIPELINE_RUN_NAME: env.RUN_NAME,
        NSRL_S3_URI: env.S3_URI,
        NSRL_SOLOMON_AWS_STAGES: env.STAGES,
        NSRL_SOLOMON_REQUIRE_GRAVITON: "1",
        NSRL_SOLOMON_REQUIRE_EC2_METADATA: "1",
        NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS: "1",
        NSRL_SOLOMON_ATTENTION_BATCH_MODE: "map-reduce",
        NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS: "0",
        NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY: "auto-online-processors",
      },
      post_run_proof_command: [
        "scripts/aws/prove-solomon-product-run.sh",
        "--s3-pipeline-uri",
        env.S3_PIPELINE_URI,
        "--launch-dir",
        env.LAUNCH_DIR,
        "--require-launch-dir",
      ],
      command: (() => {
        const command = [
        "aws",
        ];
        if (env.AWS_PROFILE) {
          command.push("--profile", env.AWS_PROFILE);
        }
        command.push(
        "--region", env.REGION,
        "ec2", "run-instances",
        "--image-id", env.AMI_ID,
        "--instance-type", env.INSTANCE_TYPE,
        "--iam-instance-profile", `Name=${env.IAM_PROFILE}`,
        "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
        "--instance-initiated-shutdown-behavior", "stop",
        "--user-data", `file://${env.USER_DATA_PATH}`,
        "--tag-specifications", `ResourceType=instance,Tags=[{Key=Name,Value=${env.RUN_NAME}},{Key=Project,Value=nsrl-solomon},{Key=Product,Value=solomon-v1}]`,
        "--output", "json",
        );
        if (env.SUBNET_ID) command.push("--subnet-id", env.SUBNET_ID);
        for (const securityGroup of env.SECURITY_GROUP_IDS.split(/[,\s]+/).filter(Boolean)) {
          if (!command.includes("--security-group-ids")) command.push("--security-group-ids");
          command.push(securityGroup);
        }
        if (env.KEY_NAME) command.push("--key-name", env.KEY_NAME);
        return command;
      })(),
    };
    fs.writeFileSync(env.LAUNCH_JSON_PATH, `${JSON.stringify(launch, null, 2)}\n`);
  '
}

write_launch_json

if [[ "$dry_run" != "0" ]]; then
  echo "solomon_product_launch_plan: $launch_json_path"
  echo "solomon_product_user_data: $user_data_path"
  echo "solomon_product_prelaunch_check: run node scripts/check-solomon-aws-prelaunch-readiness.mjs --launch-dir $out_dir --out $prelaunch_check_path"
  exit 0
fi

node scripts/check-solomon-aws-prelaunch-readiness.mjs \
  --launch-dir "$out_dir" \
  --allow-execute-plan \
  --out "$prelaunch_check_path" >/dev/null

aws_cmd=(aws --region "$region")
if [[ -n "${AWS_PROFILE:-}" ]]; then
  aws_cmd=(aws --profile "$AWS_PROFILE" --region "$region")
fi
run_args=(
  ec2 run-instances
  --image-id "$ami_id"
  --instance-type "$instance_type"
  --iam-instance-profile "Name=${iam_profile}"
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled"
  --instance-initiated-shutdown-behavior stop
  --user-data "file://${user_data_path}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${run_name}},{Key=Project,Value=nsrl-solomon},{Key=Product,Value=solomon-v1}]"
  --output json
)
if [[ -n "$subnet_id" ]]; then
  run_args+=(--subnet-id "$subnet_id")
fi
if [[ -n "$security_group_ids" ]]; then
  # shellcheck disable=SC2206
  security_groups=( ${security_group_ids//,/ } )
  run_args+=(--security-group-ids "${security_groups[@]}")
fi
if [[ -n "$key_name" ]]; then
  run_args+=(--key-name "$key_name")
fi

rm -f "$launch_result_path"
"${aws_cmd[@]}" "${run_args[@]}" > "$launch_result_path"
INSTANCE_ID="$(LAUNCH_RESULT_PATH="$launch_result_path" node -e '
const fs = require("node:fs");
const result = JSON.parse(fs.readFileSync(process.env.LAUNCH_RESULT_PATH, "utf8"));
const instanceId = String(result?.Instances?.[0]?.InstanceId || "");
if (!instanceId) {
  console.error("aws run-instances response did not include Instances[0].InstanceId");
  process.exit(2);
}
console.log(instanceId);
')"
export INSTANCE_ID
write_launch_json

echo "solomon_product_launch_plan: $launch_json_path"
echo "solomon_product_launch_result: $launch_result_path"
echo "solomon_product_user_data: $user_data_path"
echo "solomon_product_prelaunch_check: $prelaunch_check_path"
echo "solomon_product_instance: $INSTANCE_ID"
