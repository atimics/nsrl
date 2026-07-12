#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Fetch a completed Solomon AWS product run from S3 and verify the synced bundle.

Common usage:
  scripts/aws/fetch-solomon-product-run.sh --s3-pipeline-uri s3://bucket/prefix/pipelines/RUN

Or derive the pipeline URI from NSRL_S3_URI and a run name:
  NSRL_S3_URI=s3://bucket/prefix scripts/aws/fetch-solomon-product-run.sh --run-name RUN

Options:
  --s3-pipeline-uri URI            full s3://.../pipelines/RUN URI
  --run-name NAME                  run name under ${NSRL_S3_URI}/pipelines
  --out-dir PATH                   local destination directory
  --skip-sync                      verify an already-synced local directory
  --allow-dry-run                  pass through to check-solomon-aws-run-artifacts.mjs
  --allow-non-graviton-runner      pass through to check-solomon-aws-run-artifacts.mjs
  --allow-missing-s3-artifacts     pass through to check-solomon-aws-run-artifacts.mjs
  --allow-missing-completion-report
  --allow-missing-ec2-metadata
  --skip-promotion-bundle-validation

Writes:
  <out-dir>/aws-run-artifacts-check.json
  <out-dir>/fetch-report.json
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

s3_pipeline_uri="${NSRL_S3_PIPELINE_URI:-}"
run_name="${NSRL_PIPELINE_RUN_NAME:-}"
out_dir=""
skip_sync="${NSRL_SOLOMON_FETCH_SKIP_SYNC:-0}"
checker_args=()

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
    --skip-sync)
      skip_sync=1
      ;;
    --allow-dry-run|--allow-non-graviton-runner|--allow-missing-s3-artifacts|--allow-missing-completion-report|--allow-missing-ec2-metadata|--skip-promotion-bundle-validation)
      checker_args+=("$1")
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "try: scripts/aws/fetch-solomon-product-run.sh --help" >&2
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
mkdir -p "$out_dir"

sync_status=0
if [[ "$skip_sync" == "0" ]]; then
  aws_args=(aws)
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    aws_args+=(--profile "$AWS_PROFILE")
  fi
  if [[ -n "${AWS_REGION:-}" ]]; then
    aws_args+=(--region "$AWS_REGION")
  fi
  "${aws_args[@]}" s3 sync "$s3_pipeline_uri" "$out_dir" --only-show-errors
  sync_status=$?
fi

check_path="${out_dir}/aws-run-artifacts-check.json"
check_cmd=(
  node scripts/check-solomon-aws-run-artifacts.mjs
  --run-dir "$out_dir"
  --out "$check_path"
)
if ((${#checker_args[@]} > 0)); then
  check_cmd+=("${checker_args[@]}")
fi
set +e
"${check_cmd[@]}"
check_status=$?
set -e

FETCH_REPORT_PATH="${out_dir}/fetch-report.json" \
S3_PIPELINE_URI="$s3_pipeline_uri" \
RUN_NAME="$run_name" \
OUT_DIR="$out_dir" \
SKIP_SYNC="$skip_sync" \
SYNC_STATUS="$sync_status" \
CHECK_STATUS="$check_status" \
CHECK_PATH="$check_path" \
node -e '
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const env = process.env;
let artifactCheck = null;
if (fs.existsSync(env.CHECK_PATH)) {
  artifactCheck = JSON.parse(fs.readFileSync(env.CHECK_PATH, "utf8"));
}
function readTsv(filePath, expectedHeader) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  const lines = text ? text.split(/\r?\n/) : [];
  if (lines[0] !== expectedHeader) {
    throw new Error(`${filePath} must start with ${expectedHeader.replace(/\t/g, "\\t")}`);
  }
  const keys = expectedHeader.split("\t");
  return lines.slice(1).filter((line) => line.trim()).map((line, index) => {
    const fields = line.split("\t");
    if (fields.length !== keys.length) {
      throw new Error(`${filePath}:${index + 2}: expected ${keys.length} tab-separated fields`);
    }
    return Object.fromEntries(keys.map((key, fieldIndex) => [key, fields[fieldIndex]]));
  });
}
function resolveRunPath(runDir, ref) {
  if (!ref) {
    return "";
  }
  const candidates = [];
  if (path.isAbsolute(ref)) {
    candidates.push(ref);
    const parts = path.resolve(ref).split(path.sep).filter(Boolean);
    const runName = path.basename(runDir);
    const runIndex = parts.lastIndexOf(runName);
    if (runIndex >= 0) {
      candidates.push(path.join(runDir, ...parts.slice(runIndex + 1)));
    }
  } else {
    candidates.push(path.join(runDir, ref), path.resolve(ref));
  }
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0];
}
function fileSha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}
function directoryFiles(dirPath) {
  const files = [];
  const stack = [dirPath];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const absolute = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(absolute);
      } else if (entry.isFile()) {
        files.push(absolute);
      }
    }
  }
  return files.sort((left, right) => left.localeCompare(right));
}
function syncedArtifactEntry(runDir, row) {
  const resolved = resolveRunPath(runDir, row.path);
  const base = {
    stage: row.stage,
    artifact: row.artifact,
    path: row.path,
    present: fs.existsSync(resolved),
    type: "missing",
    sha256: "",
    file_count: 0,
  };
  if (!base.present) {
    return base;
  }
  const stat = fs.statSync(resolved);
  if (stat.isDirectory()) {
    const files = directoryFiles(resolved).map((filePath) => ({
      path: path.relative(resolved, filePath).split(path.sep).join("/"),
      sha256: fileSha256(filePath),
    }));
    return {
      ...base,
      type: "directory",
      sha256: crypto.createHash("sha256").update(JSON.stringify(files)).digest("hex"),
      file_count: files.length,
    };
  }
  if (stat.isFile()) {
    return {
      ...base,
      type: "file",
      sha256: fileSha256(resolved),
      file_count: 1,
    };
  }
  return {
    ...base,
    type: "other",
  };
}
function summarizeSyncedArtifacts(runDir) {
  const artifactRows = readTsv(path.join(runDir, "artifacts.tsv"), "stage\tartifact\tpath");
  const entries = artifactRows
    .map((row) => syncedArtifactEntry(runDir, row))
    .sort((left, right) =>
      left.stage.localeCompare(right.stage) ||
      left.artifact.localeCompare(right.artifact) ||
      left.path.localeCompare(right.path)
    );
  return {
    schema: "nsrl.solomon_synced_artifacts.v1",
    artifact_count: entries.length,
    present_count: entries.filter((entry) => entry.present).length,
    file_count: entries.reduce((total, entry) => total + Number(entry.file_count || 0), 0),
    sha256: crypto.createHash("sha256").update(JSON.stringify(entries)).digest("hex"),
    entries,
  };
}
const errors = Array.isArray(artifactCheck?.errors) ? artifactCheck.errors.map(String) : [];
let syncedArtifacts = null;
try {
  syncedArtifacts = summarizeSyncedArtifacts(env.OUT_DIR);
} catch (error) {
  errors.push(`could not summarize synced artifacts: ${error instanceof Error ? error.message : String(error)}`);
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
  const checkDigest = artifactCheck?.synced_artifacts?.sha256 || "";
  const fetchDigest = syncedArtifacts?.sha256 || "";
  if (!checkDigest) {
    errors.push("artifact check did not report synced_artifacts.sha256");
  } else if (!fetchDigest) {
    errors.push("fetch report could not compute synced_artifacts.sha256");
  } else if (checkDigest !== fetchDigest) {
    errors.push(`synced artifact digest ${fetchDigest} != artifact check digest ${checkDigest}`);
  }
}
const report = {
  schema: "nsrl.solomon_aws_run_fetch_check.v1",
  ok: Number(env.SYNC_STATUS || 0) === 0 && Number(env.CHECK_STATUS || 0) === 0 && artifactCheck?.ok === true && errors.length === 0,
  s3_pipeline_uri: env.S3_PIPELINE_URI || "",
  run_name: env.RUN_NAME || "",
  run_dir: env.OUT_DIR || "",
  skipped_sync: env.SKIP_SYNC !== "0",
  sync_status: Number(env.SYNC_STATUS || 0),
  artifact_check_status: Number(env.CHECK_STATUS || 0),
  artifact_check_path: env.CHECK_PATH || "",
  artifact_check_ok: artifactCheck?.ok === true,
  artifact_check_schema: artifactCheck?.schema || "",
  artifact_s3_pipeline_uri: artifactCheck?.s3?.pipeline_uri || "",
  artifact_run_name: artifactCheck?.run_name || "",
  synced_artifacts: syncedArtifacts || {},
  errors,
};
fs.mkdirSync(path.dirname(env.FETCH_REPORT_PATH), { recursive: true });
fs.writeFileSync(env.FETCH_REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
process.exit(report.ok ? 0 : 1);
'
