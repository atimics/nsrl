#!/usr/bin/env bash
set -euo pipefail

execute=0
while (($# > 0)); do
  case "$1" in
    --dry-run) execute=0 ;;
    --execute) execute=1 ;;
    *) echo "usage: $0 [--dry-run|--execute]" >&2; exit 2 ;;
  esac
  shift
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
region="${AWS_REGION:-us-east-1}"
run_name="${NSRL_PRODUCTION_LIVENESS_RUN_NAME:-production-p10m-liveness-audit-${timestamp}}"
ami_id="${NSRL_AMI_ID:-ami-required-for-execute}"
instance_type="${NSRL_PRODUCTION_LIVENESS_INSTANCE_TYPE:-c8g.2xlarge}"
iam_profile="${NSRL_IAM_INSTANCE_PROFILE:-NSRLTrainingEc2InstanceProfile}"
artifact_uri="${NSRL_PRODUCTION_LIVENESS_ARTIFACT_S3_URI:-s3://nsrl-training-022118847419-us-east-1/production-model-v1/artifacts/liveness-audit-v1.tar.gz}"
run_s3_uri="${NSRL_PRODUCTION_LIVENESS_S3_URI:-s3://nsrl-training-022118847419-us-east-1/production-model-v1/runs/${run_name}}"
terminate_on_exit="${NSRL_TERMINATE_ON_EXIT:-1}"
out_dir="data/aws-launches/$run_name"

if [[ "$artifact_uri" != s3://* || "$run_s3_uri" != s3://* ]]; then
  echo "artifact and run URIs must start with s3://" >&2; exit 2
fi
if [[ "$instance_type" != c6g.* && "$instance_type" != c7g.* && "$instance_type" != c8g.* ]]; then
  echo "instance type must be Graviton" >&2; exit 2
fi
if ((execute == 1)) && [[ "$ami_id" == "ami-required-for-execute" ]]; then
  echo "NSRL_AMI_ID is required" >&2; exit 2
fi
if ((execute == 1)) && [[ -z "${NSRL_PRODUCTION_LIVENESS_ARTIFACT_S3_URI:-}" || -z "${NSRL_PRODUCTION_LIVENESS_S3_URI:-}" ]]; then
  echo "execute mode requires explicit artifact and run S3 URIs" >&2; exit 2
fi

mkdir -p "$out_dir"
user_data="$out_dir/user-data.sh"; launch_json="$out_dir/launch.json"; launch_result="$out_dir/launch-result.json"
cat > "$user_data" <<USERDATA
#!/bin/bash
set -uo pipefail
exec > >(tee -a /var/log/nsrl-production-liveness-audit.log) 2>&1
shutdown -h +60 || true
export HOME=/root AWS_REGION=${region} AWS_DEFAULT_REGION=${region}
dnf install -y --allowerasing awscli curl gcc gcc-c++ gzip nodejs openssl-devel pkgconf-pkg-config tar
mkdir -p /opt/nsrl /mnt/nsrl/production-model-v1/p10m-liveness-audit
aws s3 cp ${artifact_uri} /tmp/nsrl-production-liveness-audit.tar.gz
tar -xzf /tmp/nsrl-production-liveness-audit.tar.gz -C /opt/nsrl
if [[ -f /root/.cargo/env ]]; then source /root/.cargo/env; else curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal; source /root/.cargo/env; fi
token="\$(curl -fsS --max-time 2 -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' http://169.254.169.254/latest/api/token || true)"
metadata() { curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: \$token" "http://169.254.169.254/latest/meta-data/\$1" || true; }
export NSRL_EC2_INSTANCE_ID="\$(metadata instance-id)" NSRL_EC2_INSTANCE_TYPE="\$(metadata instance-type)"
export NSRL_PRODUCTION_LIVENESS_S3_URI=${run_s3_uri}
cd /opt/nsrl
set +e; scripts/aws/run-production-liveness-audit-v1-graviton.sh; status=\$?; set -e
aws s3 cp /var/log/nsrl-production-liveness-audit.log ${run_s3_uri}/launch.log --only-show-errors || true
echo "\$status" | aws s3 cp - ${run_s3_uri}/exit-status.txt --only-show-errors || true
if [[ "${terminate_on_exit}" == "1" ]]; then shutdown -h now; fi
exit "\$status"
USERDATA
chmod +x "$user_data"
EXECUTE="$execute" REGION="$region" RUN_NAME="$run_name" AMI_ID="$ami_id" INSTANCE_TYPE="$instance_type" \
IAM_PROFILE="$iam_profile" ARTIFACT_URI="$artifact_uri" RUN_S3_URI="$run_s3_uri" USER_DATA="$user_data" \
LAUNCH_JSON="$launch_json" node --input-type=module - <<'NODE'
import crypto from "node:crypto"; import fs from "node:fs";
const e = process.env; const bytes = fs.readFileSync(e.USER_DATA);
fs.writeFileSync(e.LAUNCH_JSON, `${JSON.stringify({
  schema: "nsrl.production_training_liveness_aws_launch.v1", dry_run: e.EXECUTE !== "1",
  created_at: new Date().toISOString(), region: e.REGION, run_name: e.RUN_NAME,
  ami_id: e.AMI_ID, instance_type: e.INSTANCE_TYPE, iam_instance_profile: e.IAM_PROFILE,
  artifact_s3_uri: e.ARTIFACT_URI, run_s3_uri: e.RUN_S3_URI, user_data: e.USER_DATA,
  user_data_sha256: crypto.createHash("sha256").update(bytes).digest("hex"),
}, null, 2)}\n`);
NODE
if ((execute == 0)); then echo "production_liveness_launch_plan: $launch_json"; exit 0; fi
aws --region "$region" ec2 run-instances \
  --image-id "$ami_id" --instance-type "$instance_type" \
  --iam-instance-profile "Name=$iam_profile" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "file://$user_data" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$run_name},{Key=Project,Value=nsrl},{Key=Product,Value=production-model-v1}]" \
  --output json > "$launch_result"
instance_id="$(LAUNCH_RESULT="$launch_result" node -e 'const fs=require("node:fs"),x=JSON.parse(fs.readFileSync(process.env.LAUNCH_RESULT)); console.log(x.Instances[0].InstanceId)')"
echo "production_liveness_instance: $instance_id"
echo "production_liveness_run_s3_uri: $run_s3_uri"
