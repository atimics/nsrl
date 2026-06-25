#!/usr/bin/env bash
# Bake a reusable NSRL training AMI so launches skip the cold build.
#
# Launches one c8g instance that installs the toolchain, extracts the repo
# tarball, and runs a full `cargo build --release` (warming target/ + the cargo
# registry). It then snapshots the instance into an AMI. Subsequent training
# launches set NSRL_AMI_ID=<this> so their `cargo build` is *incremental*
# (seconds) instead of cold (minutes), and the OS packages are pre-installed.
#
# Cost: one ~10-15 min c8g.4xlarge build (~$0.15). Re-bake when deps change.
#
#   NSRL_S3_URI=s3://bucket/prefix \
#   NSRL_ARTIFACT_S3_URI=s3://bucket/prefix/artifacts/nsrl-working-trace-summary.tar.gz \
#   scripts/aws/bake-training-ami.sh
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROFILE_ARGS=(); [[ -n "${AWS_PROFILE:-}" ]] && PROFILE_ARGS=(--profile "$AWS_PROFILE")
S3_URI="${NSRL_S3_URI:-s3://nsrl-training-022118847419-us-east-1/wikibard}"
ARTIFACT="${NSRL_ARTIFACT_S3_URI:-${S3_URI}/artifacts/nsrl-working-trace-summary.tar.gz}"
IAM_PROFILE="${NSRL_IAM_INSTANCE_PROFILE:-NSRLTrainingEc2InstanceProfile}"
INSTANCE_TYPE="${NSRL_BAKE_INSTANCE_TYPE:-c8g.4xlarge}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DONE_MARKER="${S3_URI}/cache/ami-bake-done-${STAMP}"

aws() { command aws "${PROFILE_ARGS[@]}" --region "$REGION" "$@"; }

echo "[bake] resolving latest AL2023 arm64 AMI..."
BASE_AMI="$(aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023.*-arm64' 'Name=architecture,Values=arm64' 'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
echo "[bake] base AMI: $BASE_AMI"

USER_DATA="$(cat <<EOF
#!/bin/bash
set -euxo pipefail
exec > >(tee -a /var/log/nsrl-bake.log) 2>&1
shutdown -h +60 || true
export HOME=/root AWS_REGION=${REGION} AWS_DEFAULT_REGION=${REGION}
dnf install -y --allowerasing awscli git gzip make gcc gcc-c++ openssl-devel pkgconf-pkg-config python3 tar zstd
mkdir -p /opt/nsrl
aws s3 cp ${ARTIFACT} /tmp/nsrl.tar.gz
tar -xzf /tmp/nsrl.tar.gz -C /opt/nsrl
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
source /root/.cargo/env
cd /opt/nsrl
RUSTFLAGS='-C target-cpu=native' cargo build --release -p nsrl-train \
  --bin nsrl-train --bin nsrl-bitmap-sample --bin nsrl-bitmap-multichannel-denoise
cargo build --release -p nsrl-corpus --bin nsrl-corpus || true
aws s3 cp /var/log/nsrl-bake.log ${DONE_MARKER}
EOF
)"

echo "[bake] launching builder ($INSTANCE_TYPE)..."
IID="$(aws ec2 run-instances --image-id "$BASE_AMI" --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile "Name=${IAM_PROFILE}" \
  --instance-initiated-shutdown-behavior stop \
  --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nsrl-ami-bake-${STAMP}}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "[bake] instance: $IID — waiting for build to finish (polling S3 marker)..."

for _ in $(seq 1 60); do
  if aws s3 ls "$DONE_MARKER" >/dev/null 2>&1; then echo "[bake] build complete"; break; fi
  sleep 30
done
aws s3 ls "$DONE_MARKER" >/dev/null 2>&1 || { echo "[bake] ERROR: build did not finish in time; check /var/log/nsrl-bake.log on $IID"; exit 1; }

echo "[bake] stopping instance before image..."
aws ec2 stop-instances --instance-ids "$IID" >/dev/null
aws ec2 wait instance-stopped --instance-ids "$IID"

AMI_ID="$(aws ec2 create-image --instance-id "$IID" --name "nsrl-training-${STAMP}" \
  --description "NSRL training AMI: AL2023 arm64 + Rust + warm target/ (${STAMP})" \
  --query 'ImageId' --output text)"
echo "[bake] creating AMI: $AMI_ID — waiting for available..."
aws ec2 wait image-available --image-ids "$AMI_ID"

echo "[bake] terminating builder..."
aws ec2 terminate-instances --instance-ids "$IID" >/dev/null

echo
echo "[bake] DONE. Use it for fast launches:"
echo "    export NSRL_AMI_ID=${AMI_ID}"
