#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Fetch, verify, and release-proof a completed Solomon AWS product run.

Common usage:
  scripts/aws/prove-solomon-product-run.sh --s3-pipeline-uri s3://bucket/prefix/pipelines/RUN

Or derive the pipeline URI from NSRL_S3_URI and a run name:
  NSRL_S3_URI=s3://bucket/prefix scripts/aws/prove-solomon-product-run.sh --run-name RUN

Options:
  --s3-pipeline-uri URI            full s3://.../pipelines/RUN URI
  --run-name NAME                  run name under ${NSRL_S3_URI}/pipelines
  --out-dir PATH                   local destination directory
  --launch-dir PATH                optional executed launch.json/user-data.sh directory to cross-check
  --require-launch-dir             fail unless --launch-dir provides executed launch evidence
  --out PATH                       release proof JSON path
  --skip-sync                      verify an already-synced local directory

Writes by default:
  <out-dir>/fetch-report.json
  <out-dir>/aws-run-artifacts-check.json
  <out-dir>/product-diagnostic.json
  <out-dir>/objective-coverage.json
  <out-dir>/release-proof.json
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

s3_pipeline_uri="${NSRL_S3_PIPELINE_URI:-}"
run_name="${NSRL_PIPELINE_RUN_NAME:-}"
out_dir=""
launch_dir=""
out_path=""
skip_sync="${NSRL_SOLOMON_FETCH_SKIP_SYNC:-0}"
require_launch_dir="${NSRL_SOLOMON_REQUIRE_LAUNCH_DIR:-0}"

while (($# > 0)); do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --s3-pipeline-uri)
      shift
      s3_pipeline_uri="${1:-}"
      [[ -n "$s3_pipeline_uri" ]] || { echo "--s3-pipeline-uri requires a value" >&2; exit 2; }
      ;;
    --run-name)
      shift
      run_name="${1:-}"
      [[ -n "$run_name" ]] || { echo "--run-name requires a value" >&2; exit 2; }
      ;;
    --out-dir)
      shift
      out_dir="${1:-}"
      [[ -n "$out_dir" ]] || { echo "--out-dir requires a value" >&2; exit 2; }
      ;;
    --launch-dir)
      shift
      launch_dir="${1:-}"
      [[ -n "$launch_dir" ]] || { echo "--launch-dir requires a value" >&2; exit 2; }
      ;;
    --require-launch-dir)
      require_launch_dir=1
      ;;
    --out)
      shift
      out_path="${1:-}"
      [[ -n "$out_path" ]] || { echo "--out requires a value" >&2; exit 2; }
      ;;
    --skip-sync)
      skip_sync=1
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "try: scripts/aws/prove-solomon-product-run.sh --help" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$s3_pipeline_uri" ]]; then
  if [[ -z "${NSRL_S3_URI:-}" || -z "$run_name" ]]; then
    echo "Provide --s3-pipeline-uri, or provide --run-name with NSRL_S3_URI." >&2
    exit 2
  fi
  s3_pipeline_uri="${NSRL_S3_URI%/}/pipelines/${run_name}"
fi
if [[ "$s3_pipeline_uri" != s3://* ]]; then
  echo "--s3-pipeline-uri must start with s3://" >&2
  exit 2
fi
if [[ -z "$run_name" ]]; then
  run_name="${s3_pipeline_uri##*/}"
fi
if [[ -z "$run_name" || "$run_name" == "pipelines" ]]; then
  echo "could not infer run name from $s3_pipeline_uri" >&2
  exit 2
fi
if [[ -z "$out_dir" ]]; then
  out_dir="${NSRL_SOLOMON_FETCH_ROOT:-/tmp/nsrl-solomon-pipelines}/${run_name}"
fi
if [[ -z "$out_path" ]]; then
  out_path="${out_dir}/release-proof.json"
fi
mkdir -p "$out_dir"

fetch_args=(
  scripts/aws/fetch-solomon-product-run.sh
  --s3-pipeline-uri "$s3_pipeline_uri"
  --out-dir "$out_dir"
)
if [[ "$skip_sync" != "0" ]]; then
  fetch_args+=(--skip-sync)
fi

set +e
"${fetch_args[@]}"
fetch_status=$?
set -e

diagnostic_path="${out_dir}/product-diagnostic.json"
diagnostic_status=99
if [[ "$fetch_status" -eq 0 ]]; then
  set +e
  node scripts/check-solomon-product-diagnostic.mjs \
    --aws-run-dir "$out_dir" \
    --require-aws-run \
    --out "$diagnostic_path"
  diagnostic_status=$?
  set -e
fi

objective_coverage_path="${out_dir}/objective-coverage.json"
objective_status=99
if [[ "$diagnostic_status" -eq 0 ]]; then
  set +e
  node scripts/check-solomon-objective-coverage.mjs \
    --diagnostic "$diagnostic_path" \
    --require-release \
    --out "$objective_coverage_path"
  objective_status=$?
  set -e
fi

launch_check_path=""
launch_status=0
if [[ -n "$launch_dir" ]]; then
  launch_check_path="${out_dir}/release-launch-readiness-check.json"
  set +e
  node scripts/check-solomon-aws-prelaunch-readiness.mjs \
    --launch-dir "$launch_dir" \
    --allow-execute-plan \
    --out "$launch_check_path"
  launch_status=$?
  set -e
fi

RELEASE_PROOF_PATH="$out_path" \
S3_PIPELINE_URI="$s3_pipeline_uri" \
RUN_NAME="$run_name" \
RUN_DIR="$out_dir" \
LAUNCH_DIR="$launch_dir" \
REQUIRE_LAUNCH_DIR="$require_launch_dir" \
FETCH_STATUS="$fetch_status" \
DIAGNOSTIC_STATUS="$diagnostic_status" \
OBJECTIVE_STATUS="$objective_status" \
LAUNCH_STATUS="$launch_status" \
FETCH_REPORT_PATH="${out_dir}/fetch-report.json" \
ARTIFACT_CHECK_PATH="${out_dir}/aws-run-artifacts-check.json" \
DIAGNOSTIC_PATH="$diagnostic_path" \
OBJECTIVE_COVERAGE_PATH="$objective_coverage_path" \
LAUNCH_CHECK_PATH="$launch_check_path" \
node -e '
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const env = process.env;
function readJson(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}
const fetchReport = readJson(env.FETCH_REPORT_PATH);
const artifactCheck = readJson(env.ARTIFACT_CHECK_PATH);
const diagnostic = readJson(env.DIAGNOSTIC_PATH);
const objectiveCoverage = readJson(env.OBJECTIVE_COVERAGE_PATH);
const launchCheck = readJson(env.LAUNCH_CHECK_PATH);
const launch = readJson(env.LAUNCH_DIR ? path.join(env.LAUNCH_DIR, "launch.json") : "");
const launchResultPath = resolveLaunchResultPath(launch);
const launchResultText = launchResultPath && fs.existsSync(launchResultPath)
  ? fs.readFileSync(launchResultPath, "utf8")
  : "";
let launchResult = null;
let launchResultParseError = "";
if (launchResultText) {
  try {
    launchResult = JSON.parse(launchResultText);
  } catch (error) {
    launchResultParseError = error instanceof Error ? error.message : String(error);
  }
}
const postRunProofCommand = Array.isArray(launch?.post_run_proof_command)
  ? launch.post_run_proof_command.map(String)
  : [];
function resolveLaunchResultPath(row) {
  if (!env.LAUNCH_DIR || !row) return "";
  const recorded = String(row.launch_result || "");
  if (recorded) return path.isAbsolute(recorded) ? recorded : path.resolve(recorded);
  return path.join(env.LAUNCH_DIR, "launch-result.json");
}
function commandValue(flag) {
  const index = postRunProofCommand.indexOf(flag);
  if (index < 0) return "";
  return postRunProofCommand[index + 1] || "";
}
function splitIds(value) {
  return String(value || "").split(/[,\s]+/).filter(Boolean);
}
function sortedUnique(values) {
  return [...new Set(values.map(String).filter(Boolean))].sort();
}
function resultSecurityGroupIds(instance) {
  const ids = [];
  for (const group of instance?.SecurityGroups || []) {
    if (group?.GroupId) ids.push(group.GroupId);
  }
  for (const networkInterface of instance?.NetworkInterfaces || []) {
    for (const group of networkInterface?.Groups || []) {
      if (group?.GroupId) ids.push(group.GroupId);
    }
  }
  return sortedUnique(ids);
}
const errors = [];
if (Number(env.FETCH_STATUS || 0) !== 0 || fetchReport?.ok !== true) {
  errors.push("fetch/artifact verification did not pass");
  for (const error of fetchReport?.errors || []) {
    errors.push(`fetch: ${error}`);
  }
}
if (artifactCheck?.ok === true) {
  const artifactPipelineUri = artifactCheck?.s3?.pipeline_uri || "";
  if (!artifactPipelineUri) {
    errors.push("artifact check did not report s3.pipeline_uri");
  } else if (artifactPipelineUri !== env.S3_PIPELINE_URI) {
    errors.push(`artifact s3.pipeline_uri ${artifactPipelineUri} != requested ${env.S3_PIPELINE_URI}`);
  }
  const artifactRunName = artifactCheck?.run_name || "";
  if (!artifactRunName) {
    errors.push("artifact check did not report run_name");
  } else if (artifactRunName !== env.RUN_NAME) {
    errors.push(`artifact run_name ${artifactRunName} != requested ${env.RUN_NAME}`);
  }
}
const fetchSyncedDigest = fetchReport?.synced_artifacts?.sha256 || "";
const artifactSyncedDigest = artifactCheck?.synced_artifacts?.sha256 || "";
if (fetchReport?.ok === true || artifactCheck?.ok === true) {
  if (!fetchSyncedDigest) {
    errors.push("fetch report missing synced_artifacts.sha256");
  }
  if (!artifactSyncedDigest) {
    errors.push("artifact check missing synced_artifacts.sha256");
  }
  if (fetchSyncedDigest && artifactSyncedDigest && fetchSyncedDigest !== artifactSyncedDigest) {
    errors.push(`fetch synced artifact digest ${fetchSyncedDigest} != artifact check digest ${artifactSyncedDigest}`);
  }
}
if (Number(env.FETCH_STATUS || 0) !== 0 || fetchReport?.ok !== true) {
  errors.push("product diagnostic skipped because fetch/artifact verification failed");
} else if (Number(env.DIAGNOSTIC_STATUS || 0) !== 0 || diagnostic?.release_product_proof !== true) {
  errors.push("product diagnostic release_product_proof is not true");
}
if (Number(env.FETCH_STATUS || 0) !== 0 || fetchReport?.ok !== true) {
  errors.push("objective coverage skipped because fetch/artifact verification failed");
} else if (Number(env.DIAGNOSTIC_STATUS || 0) !== 0 || diagnostic?.release_product_proof !== true) {
  errors.push("objective coverage skipped because product diagnostic did not pass");
} else if (
  Number(env.OBJECTIVE_STATUS || 0) !== 0 ||
  objectiveCoverage?.release_objective_proof !== true ||
  objectiveCoverage?.ok !== true
) {
  errors.push("objective coverage release_objective_proof is not true");
  for (const missing of objectiveCoverage?.missing || []) {
    errors.push(`objective coverage: ${missing}`);
  }
}
if (env.LAUNCH_DIR) {
  if (Number(env.LAUNCH_STATUS || 0) !== 0 || launchCheck?.ok !== true) {
    errors.push("launch/prelaunch readiness check did not pass");
    for (const error of launchCheck?.errors || []) {
      errors.push(`launch readiness: ${error}`);
    }
  }
  if (launch?.dry_run !== false) {
    errors.push("launch dry_run is not false; --launch-dir must point at executed launch evidence");
  }
  if (!String(launch?.instance_id || "")) {
    errors.push("launch instance_id is missing; --launch-dir must point at executed launch evidence");
  }
  if (launch?.dry_run === false) {
    if (!String(launch?.launch_result || "")) {
      errors.push("launch_result path is missing from executed launch manifest");
    }
    if (!launchResultText) {
      errors.push("launch-result.json is missing; --launch-dir must include the EC2 run-instances response");
    } else if (launchResultParseError) {
      errors.push(`launch-result.json is not valid JSON: ${launchResultParseError}`);
    } else {
      const actualHash = crypto.createHash("sha256").update(launchResultText).digest("hex");
      if (String(launch?.launch_result_sha256 || "") !== actualHash) {
        errors.push("launch_result_sha256 does not match launch-result.json");
      }
      const resultInstanceId = String(launchResult?.Instances?.[0]?.InstanceId || "");
      if (!resultInstanceId) {
        errors.push("launch-result.json missing Instances[0].InstanceId");
      } else if (resultInstanceId !== String(launch?.instance_id || "")) {
        errors.push(`launch-result instance id ${resultInstanceId} != launch instance_id ${launch?.instance_id || ""}`);
      }
      const resultImageId = String(launchResult?.Instances?.[0]?.ImageId || "");
      if (!resultImageId) {
        errors.push("launch-result.json missing Instances[0].ImageId");
      } else if (resultImageId !== String(launch?.ami_id || "")) {
        errors.push(`launch-result image id ${resultImageId} != launch ami_id ${launch?.ami_id || ""}`);
      }
      const resultInstanceType = String(launchResult?.Instances?.[0]?.InstanceType || "");
      if (!resultInstanceType) {
        errors.push("launch-result.json missing Instances[0].InstanceType");
      } else if (resultInstanceType !== String(launch?.instance_type || "")) {
        errors.push(`launch-result instance type ${resultInstanceType} != launch instance_type ${launch?.instance_type || ""}`);
      }
      const resultSubnetId = String(
        launchResult?.Instances?.[0]?.SubnetId ||
        launchResult?.Instances?.[0]?.NetworkInterfaces?.[0]?.SubnetId ||
        "",
      );
      if (String(launch?.subnet_id || "")) {
        if (!resultSubnetId) {
          errors.push("launch-result.json missing Instances[0].SubnetId for launch subnet_id check");
        } else if (resultSubnetId !== String(launch.subnet_id)) {
          errors.push(`launch-result subnet id ${resultSubnetId} != launch subnet_id ${launch.subnet_id}`);
        }
      }
      const expectedSecurityGroupIds = sortedUnique(splitIds(launch?.security_group_ids || ""));
      if (expectedSecurityGroupIds.length > 0) {
        const actualSecurityGroupIds = resultSecurityGroupIds(launchResult?.Instances?.[0] || {});
        if (actualSecurityGroupIds.length === 0) {
          errors.push("launch-result.json missing security group ids for launch security_group_ids check");
        } else if (JSON.stringify(actualSecurityGroupIds) !== JSON.stringify(expectedSecurityGroupIds)) {
          errors.push(`launch-result security group ids ${JSON.stringify(actualSecurityGroupIds)} != launch security_group_ids ${JSON.stringify(expectedSecurityGroupIds)}`);
        }
      }
    }
  }
  if (launch?.run_name && launch.run_name !== env.RUN_NAME) {
    errors.push(`launch run_name ${launch.run_name} != fetched run ${env.RUN_NAME}`);
  }
  if (launch?.s3_pipeline_uri && launch.s3_pipeline_uri !== env.S3_PIPELINE_URI) {
    errors.push(`launch s3_pipeline_uri ${launch.s3_pipeline_uri} != ${env.S3_PIPELINE_URI}`);
  }
  if (!postRunProofCommand.includes("scripts/aws/prove-solomon-product-run.sh")) {
    errors.push("launch post_run_proof_command missing scripts/aws/prove-solomon-product-run.sh");
  }
  if (commandValue("--s3-pipeline-uri") !== env.S3_PIPELINE_URI) {
    errors.push("launch post_run_proof_command does not match requested S3 pipeline URI");
  }
  const commandLaunchDir = commandValue("--launch-dir");
  if (!commandLaunchDir) {
    errors.push("launch post_run_proof_command missing --launch-dir");
  } else if (path.resolve(commandLaunchDir) !== path.resolve(env.LAUNCH_DIR)) {
    errors.push(`launch post_run_proof_command launch dir ${commandLaunchDir} != ${env.LAUNCH_DIR}`);
  }
  const artifactRunner = artifactCheck?.runner || {};
  const artifactEc2 = artifactRunner.ec2 || {};
  const artifactRunInstanceId = String(artifactEc2.instance_id || "");
  const artifactRunInstanceType = String(artifactEc2.instance_type || "");
  if (!artifactRunInstanceId) {
    errors.push("run artifact ec2 instance_id is missing; cannot match launch instance");
  } else if (String(launch?.instance_id || "") && artifactRunInstanceId !== String(launch.instance_id)) {
    errors.push(`launch instance_id ${launch.instance_id} != run artifact ec2 instance_id ${artifactRunInstanceId}`);
  }
  if (!artifactRunInstanceType) {
    errors.push("run artifact ec2 instance_type is missing; cannot match launch instance type");
  } else if (String(launch?.instance_type || "") && artifactRunInstanceType !== String(launch.instance_type)) {
    errors.push(`launch instance_type ${launch.instance_type} != run artifact ec2 instance_type ${artifactRunInstanceType}`);
  }
} else if (env.REQUIRE_LAUNCH_DIR === "1") {
  errors.push("launch evidence is required; pass --launch-dir with executed launch.json");
}
const report = {
  schema: "nsrl.solomon_aws_product_release_proof.v1",
  ok: errors.length === 0,
  run_name: env.RUN_NAME || "",
  run_dir: env.RUN_DIR || "",
  s3_pipeline_uri: env.S3_PIPELINE_URI || "",
  generated_at: new Date().toISOString(),
  fetch: {
    status: Number(env.FETCH_STATUS || 0),
    report: env.FETCH_REPORT_PATH || "",
    ok: fetchReport?.ok === true,
    artifact_check_ok: fetchReport?.artifact_check_ok === true,
    synced_artifacts: fetchReport?.synced_artifacts || {},
  },
  artifact_check: {
    path: env.ARTIFACT_CHECK_PATH || "",
    ok: artifactCheck?.ok === true,
    runner: artifactCheck?.runner || {},
    s3: artifactCheck?.s3 || {},
    synced_artifacts: artifactCheck?.synced_artifacts || {},
    promotion: artifactCheck?.promotion || {},
    quality_report: artifactCheck?.quality_report || {},
  },
  product_diagnostic: {
    status: Number(env.DIAGNOSTIC_STATUS || 0),
    path: env.DIAGNOSTIC_PATH || "",
    ok: diagnostic?.ok === true,
    local_product_proof: diagnostic?.local_product_proof === true,
    release_product_proof: diagnostic?.release_product_proof === true,
    remaining_product_evidence: diagnostic?.remaining_product_evidence || [],
  },
  objective_coverage: {
    status: Number(env.OBJECTIVE_STATUS || 0),
    path: env.OBJECTIVE_COVERAGE_PATH || "",
    ok: objectiveCoverage?.ok === true,
    local_objective_proof: objectiveCoverage?.local_objective_proof === true,
    release_objective_proof: objectiveCoverage?.release_objective_proof === true,
    remaining_release_evidence: objectiveCoverage?.remaining_release_evidence || [],
    missing: objectiveCoverage?.missing || [],
  },
  launch: {
    provided: Boolean(env.LAUNCH_DIR),
    required: env.REQUIRE_LAUNCH_DIR === "1",
    dir: env.LAUNCH_DIR || "",
    status: Number(env.LAUNCH_STATUS || 0),
    check: env.LAUNCH_CHECK_PATH || "",
    ok: env.LAUNCH_DIR ? launchCheck?.ok === true : null,
    dry_run: launch ? launch.dry_run === true : null,
    executed: launch?.dry_run === false && Boolean(String(launch?.instance_id || "")),
    run_name: launch?.run_name || "",
    s3_pipeline_uri: launch?.s3_pipeline_uri || "",
    instance_type: launch?.instance_type || "",
    instance_id: launch?.instance_id || "",
    run_artifact_instance_id: artifactCheck?.runner?.ec2?.instance_id || "",
    run_artifact_instance_type: artifactCheck?.runner?.ec2?.instance_type || "",
    launch_result: launchResultPath || "",
    launch_result_sha256: launch?.launch_result_sha256 || "",
    launch_result_instance_id: launchResult?.Instances?.[0]?.InstanceId || "",
    launch_result_image_id: launchResult?.Instances?.[0]?.ImageId || "",
    launch_result_instance_type: launchResult?.Instances?.[0]?.InstanceType || "",
    launch_result_subnet_id: launchResult?.Instances?.[0]?.SubnetId ||
      launchResult?.Instances?.[0]?.NetworkInterfaces?.[0]?.SubnetId || "",
    launch_result_security_group_ids: resultSecurityGroupIds(launchResult?.Instances?.[0] || {}),
    post_run_proof_command: postRunProofCommand,
  },
  errors,
};
fs.mkdirSync(path.dirname(path.resolve(env.RELEASE_PROOF_PATH)), { recursive: true });
fs.writeFileSync(env.RELEASE_PROOF_PATH, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
process.exit(report.ok ? 0 : 1);
'
