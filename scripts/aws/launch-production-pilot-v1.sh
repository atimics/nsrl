#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Prepare or launch the production-model-v1 p10m pilot on EC2 Graviton.

Dry-run is the default and does not call AWS. Execute mode spends money.

Required for --execute:
  NSRL_AMI_ID=ami-...
  NSRL_PRODUCTION_PILOT_ARTIFACT_S3_URI=s3://bucket/prefix/artifact.tar.gz
  NSRL_PRODUCTION_PILOT_S3_URI=s3://bucket/prefix/run

Optional:
  NSRL_PRODUCTION_PILOT_INSTANCE_TYPE=c8g.2xlarge
  NSRL_IAM_INSTANCE_PROFILE=NSRLTrainingEc2InstanceProfile
  NSRL_SUBNET_ID=subnet-...
  NSRL_SECURITY_GROUP_IDS=sg-...,sg-...
  NSRL_TERMINATE_ON_EXIT=1
USAGE
}

execute=0
out_dir=""
while (($# > 0)); do
  case "$1" in
    --dry-run) execute=0 ;;
    --execute) execute=1 ;;
    --out-dir)
      shift
      out_dir="${1:?--out-dir requires a value}"
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
region="${AWS_REGION:-us-east-1}"
run_name="${NSRL_PRODUCTION_PILOT_RUN_NAME:-production-p10m-pilot-${timestamp}}"
ami_id="${NSRL_AMI_ID:-ami-required-for-execute}"
instance_type="${NSRL_PRODUCTION_PILOT_INSTANCE_TYPE:-c8g.2xlarge}"
iam_profile="${NSRL_IAM_INSTANCE_PROFILE:-NSRLTrainingEc2InstanceProfile}"
artifact_uri="${NSRL_PRODUCTION_PILOT_ARTIFACT_S3_URI:-s3://nsrl-training-022118847419-us-east-1/production-model-v1/artifacts/pilot-v1.tar.gz}"
run_s3_uri="${NSRL_PRODUCTION_PILOT_S3_URI:-s3://nsrl-training-022118847419-us-east-1/production-model-v1/runs/${run_name}}"
subnet_id="${NSRL_SUBNET_ID:-}"
security_group_ids="${NSRL_SECURITY_GROUP_IDS:-}"
terminate_on_exit="${NSRL_TERMINATE_ON_EXIT:-1}"
out_dir="${out_dir:-data/aws-launches/${run_name}}"

if [[ "$artifact_uri" != s3://* || "$run_s3_uri" != s3://* ]]; then
  echo "artifact and run URIs must start with s3://" >&2
  exit 2
fi
if [[ "$instance_type" != c6g.* && "$instance_type" != c7g.* && "$instance_type" != c8g.* ]]; then
  echo "instance type must be a c6g/c7g/c8g Graviton instance" >&2
  exit 2
fi
if ((execute == 1)) && [[ "$ami_id" == "ami-required-for-execute" ]]; then
  echo "NSRL_AMI_ID is required for --execute" >&2
  exit 2
fi
if ((execute == 1)) && [[ -z "${NSRL_PRODUCTION_PILOT_ARTIFACT_S3_URI:-}" || -z "${NSRL_PRODUCTION_PILOT_S3_URI:-}" ]]; then
  echo "execute mode requires explicit artifact and run S3 URIs" >&2
  exit 2
fi

mkdir -p "$out_dir"
user_data="$out_dir/user-data.sh"
launch_json="$out_dir/launch.json"
launch_result="$out_dir/launch-result.json"

cat > "$user_data" <<USERDATA
#!/bin/bash
set -uo pipefail
exec > >(tee -a /var/log/nsrl-production-pilot.log) 2>&1
shutdown -h +240 || true
export HOME=/root AWS_REGION=${region} AWS_DEFAULT_REGION=${region}
dnf install -y --allowerasing awscli curl gcc gcc-c++ gzip nodejs openssl-devel pkgconf-pkg-config python3 python3-pip tar
python3 -m pip install --upgrade numpy
mkdir -p /opt/nsrl /mnt/nsrl/production-model-v1/p10m-pilot
aws s3 cp ${artifact_uri} /tmp/nsrl-production-pilot.tar.gz
tar -xzf /tmp/nsrl-production-pilot.tar.gz -C /opt/nsrl
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
else
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
  source /root/.cargo/env
fi
token="\$(curl -fsS --max-time 2 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token || true)"
metadata() { curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: \$token" "http://169.254.169.254/latest/meta-data/\$1" || true; }
export NSRL_EC2_INSTANCE_ID="\$(metadata instance-id)"
export NSRL_EC2_INSTANCE_TYPE="\$(metadata instance-type)"
export NSRL_PRODUCTION_PILOT_S3_URI=${run_s3_uri}
cd /opt/nsrl
set +e
scripts/aws/run-production-pilot-v1-graviton.sh
status=\$?
set -e
aws s3 cp /var/log/nsrl-production-pilot.log ${run_s3_uri}/launch.log --only-show-errors || true
echo "\$status" | aws s3 cp - ${run_s3_uri}/exit-status.txt --only-show-errors || true
if [[ "${terminate_on_exit}" == "1" ]]; then
  shutdown -h now
fi
exit "\$status"
USERDATA
chmod +x "$user_data"

EXECUTE="$execute" REGION="$region" RUN_NAME="$run_name" AMI_ID="$ami_id" \
INSTANCE_TYPE="$instance_type" IAM_PROFILE="$iam_profile" ARTIFACT_URI="$artifact_uri" \
RUN_S3_URI="$run_s3_uri" SUBNET_ID="$subnet_id" SECURITY_GROUP_IDS="$security_group_ids" \
USER_DATA="$user_data" LAUNCH_JSON="$launch_json" node --input-type=module - <<'NODE'
import crypto from "node:crypto";
import fs from "node:fs";
const env = process.env;
const bytes = fs.readFileSync(env.USER_DATA);
const value = {
  schema: "nsrl.production_pilot_aws_launch.v1",
  dry_run: env.EXECUTE !== "1",
  created_at: new Date().toISOString(),
  region: env.REGION,
  run_name: env.RUN_NAME,
  ami_id: env.AMI_ID,
  instance_type: env.INSTANCE_TYPE,
  iam_instance_profile: env.IAM_PROFILE,
  artifact_s3_uri: env.ARTIFACT_URI,
  run_s3_uri: env.RUN_S3_URI,
  subnet_id: env.SUBNET_ID,
  security_group_ids: env.SECURITY_GROUP_IDS.split(/[\s,]+/).filter(Boolean),
  user_data: env.USER_DATA,
  user_data_sha256: crypto.createHash("sha256").update(bytes).digest("hex"),
};
fs.writeFileSync(env.LAUNCH_JSON, `${JSON.stringify(value, null, 2)}\n`);
NODE

if ((execute == 0)); then
  echo "production_pilot_launch_plan: $launch_json"
  echo "production_pilot_user_data: $user_data"
  exit 0
fi

aws_cmd=(aws --region "$region")
run_args=(
  ec2 run-instances --image-id "$ami_id" --instance-type "$instance_type"
  --iam-instance-profile "Name=$iam_profile"
  --metadata-options HttpTokens=required,HttpEndpoint=enabled
  --instance-initiated-shutdown-behavior terminate
  --user-data "file://$user_data"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$run_name},{Key=Project,Value=nsrl},{Key=Product,Value=production-model-v1}]"
  --output json
)
if [[ -n "$subnet_id" ]]; then run_args+=(--subnet-id "$subnet_id"); fi
if [[ -n "$security_group_ids" ]]; then
  read -r -a groups <<< "${security_group_ids//,/ }"
  run_args+=(--security-group-ids "${groups[@]}")
fi
"${aws_cmd[@]}" "${run_args[@]}" > "$launch_result"
INSTANCE_ID="$(LAUNCH_RESULT="$launch_result" node -e 'const fs=require("node:fs"); const x=JSON.parse(fs.readFileSync(process.env.LAUNCH_RESULT)); console.log(x.Instances[0].InstanceId)')"
echo "production_pilot_instance: $INSTANCE_ID"
echo "production_pilot_run_s3_uri: $run_s3_uri"
echo "production_pilot_launch_result: $launch_result"
