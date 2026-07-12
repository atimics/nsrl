#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_live_launch_readiness_self_test.v1";
const liveReadinessScript = "scripts/check-solomon-aws-live-launch-readiness.sh";
const scrubbedEnvKeys = [
  "NSRL_AWS_LIVE_LAUNCH_READINESS_ROOT",
  "NSRL_AWS_LIVE_LAUNCH_READINESS_NAME",
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

function usage() {
  console.log([
    "Usage: check-solomon-aws-live-launch-readiness-self-test.mjs [--out PATH] [--keep]",
    "",
    "Verifies that the live EC2 launch readiness wrapper requires explicit S3",
    "pipeline and artifact inputs before it can mark an operator shell ready.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "", keep: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  const resolved = path.resolve(outPath);
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, "utf8");
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

function tailLines(text, maxLines) {
  const lines = String(text || "").split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function baseEnv(root, name, overrides = {}) {
  const env = { ...process.env };
  for (const key of scrubbedEnvKeys) {
    delete env[key];
  }
  return {
    ...env,
    AWS_REGION: "us-east-1",
    NSRL_AWS_LIVE_LAUNCH_READINESS_ROOT: root,
    NSRL_AWS_LIVE_LAUNCH_READINESS_NAME: name,
    NSRL_AMI_ID: "ami-0123456789abcdef0",
    NSRL_IAM_INSTANCE_PROFILE: "NSRLTrainingEc2InstanceProfile",
    NSRL_SUBNET_ID: "subnet-0123456789abcdef0",
    NSRL_SECURITY_GROUP_IDS: "sg-0123456789abcdef0 sg-0fedcba9876543210",
    NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE: "c8g.4xlarge",
    ...overrides,
  };
}

function runLiveReadiness(root, name, env) {
  const runDir = path.join(root, name);
  const result = childProcess.spawnSync("bash", [liveReadinessScript], {
    cwd: repoRoot,
    encoding: "utf8",
    env,
  });
  const reportPath = path.join(runDir, "live-launch-readiness.json");
  const report = readJsonMaybe(reportPath);
  return {
    status: result.status,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    run_dir: runDir,
    report_path: fs.existsSync(reportPath) ? reportPath : "",
    report,
  };
}

function goodExplicitS3Case(root) {
  const name = "good-explicit-s3-artifact";
  const result = runLiveReadiness(root, name, baseEnv(root, name, {
    NSRL_S3_URI: "s3://nsrl-product-plan-check/solomon",
    NSRL_ARTIFACT_S3_URI: "s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz",
  }));
  const requiredMissing = Array.isArray(result.report?.required_env_missing)
    ? result.report.required_env_missing.map(String)
    : [];
  const ok = result.status === 0 &&
    result.report?.ok === true &&
    requiredMissing.length === 0 &&
    result.report?.prelaunch_readiness?.ok === true;
  return {
    name,
    ok,
    status: result.status,
    report: result.report_path,
    readiness_ok: result.report?.ok === true,
    prelaunch_ok: result.report?.prelaunch_readiness?.ok === true,
    required_env_missing: requiredMissing,
    errors: [
      result.status === 0 ? "" : "live readiness command failed",
      result.report ? "" : "live-launch-readiness.json was not written",
      result.report?.ok === true ? "" : "live readiness report was not ok",
      result.report?.prelaunch_readiness?.ok === true ? "" : "prelaunch readiness was not ok",
      requiredMissing.length === 0 ? "" : `required env was missing: ${requiredMissing.join(", ")}`,
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
  };
}

function badMissingS3ArtifactCase(root) {
  const name = "bad-missing-explicit-s3-artifact";
  const result = runLiveReadiness(root, name, baseEnv(root, name));
  const requiredMissing = Array.isArray(result.report?.required_env_missing)
    ? result.report.required_env_missing.map(String)
    : [];
  const expectedMissing = ["NSRL_S3_URI", "NSRL_ARTIFACT_S3_URI"];
  const matchedMissing = expectedMissing.filter((item) => requiredMissing.includes(item));
  const ok = result.status !== 0 &&
    result.report?.ok === false &&
    matchedMissing.length === expectedMissing.length &&
    result.report?.prelaunch_readiness?.ok === true;
  return {
    name,
    ok,
    status: result.status,
    report: result.report_path,
    readiness_ok: result.report?.ok === true,
    prelaunch_ok: result.report?.prelaunch_readiness?.ok === true,
    expected_required_env_missing: expectedMissing,
    matched_required_env_missing: matchedMissing,
    required_env_missing: requiredMissing,
    errors: [
      result.status !== 0 ? "" : "live readiness command unexpectedly succeeded",
      result.report ? "" : "live-launch-readiness.json was not written",
      result.report?.ok === false ? "" : "live readiness report did not fail",
      result.report?.prelaunch_readiness?.ok === true
        ? ""
        : "prelaunch readiness should still pass on launcher defaults",
      matchedMissing.length === expectedMissing.length
        ? ""
        : `missing required-env rejection for ${expectedMissing.filter((item) => !matchedMissing.includes(item)).join(", ")}`,
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
  };
}

function badMissingExplicitAmiCase(root) {
  const name = "bad-missing-explicit-ami";
  const env = baseEnv(root, name, {
    NSRL_S3_URI: "s3://nsrl-product-plan-check/solomon",
    NSRL_ARTIFACT_S3_URI: "s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz",
  });
  delete env.NSRL_AMI_ID;
  const result = runLiveReadiness(root, name, env);
  const requiredMissing = Array.isArray(result.report?.required_env_missing)
    ? result.report.required_env_missing.map(String)
    : [];
  const ok = result.status !== 0 &&
    result.report?.ok === false &&
    requiredMissing.includes("NSRL_AMI_ID");
  return {
    name,
    ok,
    status: result.status,
    report: result.report_path,
    readiness_ok: result.report?.ok === true,
    prelaunch_ok: result.report?.prelaunch_readiness?.ok === true,
    expected_required_env_missing: ["NSRL_AMI_ID"],
    matched_required_env_missing: requiredMissing.filter((item) => item === "NSRL_AMI_ID"),
    required_env_missing: requiredMissing,
    errors: [
      result.status !== 0 ? "" : "live readiness command unexpectedly succeeded",
      result.report ? "" : "live-launch-readiness.json was not written",
      result.report?.ok === false ? "" : "live readiness report did not fail",
      requiredMissing.includes("NSRL_AMI_ID") ? "" : "missing required-env rejection for NSRL_AMI_ID",
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-aws-live-launch-readiness-self-test-"));
  const cases = [];
  try {
    cases.push(goodExplicitS3Case(root));
    cases.push(badMissingS3ArtifactCase(root));
    cases.push(badMissingExplicitAmiCase(root));
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }

  const report = {
    schema,
    ok: cases.every((item) => item.ok),
    scratch_root: config.keep ? root : "",
    kept: config.keep,
    cases,
    errors: cases.filter((item) => !item.ok).flatMap((item) =>
      item.errors.length > 0 ? item.errors.map((error) => `${item.name}: ${error}`) : [`${item.name} failed`],
    ),
  };
  writeReport(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
