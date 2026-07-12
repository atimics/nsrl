#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_launch_execute_guard_self_test.v1";

function usage() {
  console.log([
    "Usage: check-solomon-aws-launch-execute-guard-self-test.mjs [--out PATH] [--keep]",
    "",
    "Verifies that scripts/aws/launch-solomon-product-run.sh --execute runs",
    "explicit S3 handoff checks and the prelaunch readiness gate before aws",
    "ec2 run-instances. The test uses a fake aws binary; success means bad",
    "execute inputs fail before the fake aws binary is invoked.",
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

function argValue(args, flag) {
  const index = args.indexOf(flag);
  return index >= 0 ? args[index + 1] || "" : "";
}

function normalizeAwsRunInstancesCommand(args) {
  return ["aws", ...args];
}

function sameArray(left, right) {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

function runExecuteMissingExplicitInputCase(root, name, removedEnvKeys, expectedStderr) {
  const runDir = path.join(root, name);
  const fakeBin = path.join(runDir, "bin");
  const fakeAwsMarker = path.join(runDir, "aws-was-called");
  const launchLog = path.join(runDir, "launch-execute.log");
  const prelaunchPath = path.join(runDir, "prelaunch-readiness-check.json");
  fs.mkdirSync(fakeBin, { recursive: true });
  const fakeAws = path.join(fakeBin, "aws");
  fs.writeFileSync(fakeAws, [
    "#!/usr/bin/env bash",
    `echo "aws was called" > ${JSON.stringify(fakeAwsMarker)}`,
    "exit 99",
    "",
  ].join("\n"), "utf8");
  fs.chmodSync(fakeAws, 0o755);

  const env = {
    ...process.env,
    PATH: `${fakeBin}${path.delimiter}${process.env.PATH || ""}`,
    NSRL_AMI_ID: "ami-0123456789abcdef0",
    NSRL_S3_URI: "s3://nsrl-product-plan-check/solomon",
    NSRL_ARTIFACT_S3_URI: "s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz",
    NSRL_IAM_INSTANCE_PROFILE: "NSRLTrainingEc2InstanceProfile",
    NSRL_SUBNET_ID: "subnet-0123456789abcdef0",
    NSRL_SECURITY_GROUP_IDS: "sg-0123456789abcdef0 sg-0fedcba9876543210",
    NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE: "c8g.4xlarge",
  };
  for (const key of removedEnvKeys) {
    delete env[key];
  }

  const result = childProcess.spawnSync("bash", [
    "scripts/aws/launch-solomon-product-run.sh",
    "--execute",
    "--out-dir",
    runDir,
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    env,
  });
  fs.writeFileSync(launchLog, [
    "$ scripts/aws/launch-solomon-product-run.sh --execute",
    `status=${result.status}`,
    "",
    "stdout:",
    result.stdout || "",
    "",
    "stderr:",
    result.stderr || "",
  ].join("\n"), "utf8");

  const awsCalled = fs.existsSync(fakeAwsMarker);
  const matchedError = String(result.stderr || "").includes(expectedStderr);
  const prelaunchWritten = fs.existsSync(prelaunchPath);
  const ok = result.status !== 0 &&
    !awsCalled &&
    matchedError &&
    !prelaunchWritten;

  return {
    name,
    ok,
    status: result.status,
    aws_called: awsCalled,
    prelaunch_report: prelaunchWritten ? prelaunchPath : "",
    matched_error: matchedError ? expectedStderr : "",
    errors: [
      result.status !== 0 ? "" : "launch command unexpectedly succeeded",
      awsCalled ? "fake aws binary was invoked before explicit handoff failure" : "",
      matchedError ? "" : `stderr did not include ${JSON.stringify(expectedStderr)}`,
      prelaunchWritten ? "prelaunch should not run before explicit execute handoff is complete" : "",
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
  };
}

function runExecuteGuardCase(root) {
  const runDir = path.join(root, "bad-execute-prelaunch-blocks-before-aws");
  const fakeBin = path.join(runDir, "bin");
  const fakeAwsMarker = path.join(runDir, "aws-was-called");
  const launchLog = path.join(runDir, "launch-execute.log");
  const prelaunchPath = path.join(runDir, "prelaunch-readiness-check.json");
  fs.mkdirSync(fakeBin, { recursive: true });
  const fakeAws = path.join(fakeBin, "aws");
  fs.writeFileSync(fakeAws, [
    "#!/usr/bin/env bash",
    `echo "aws was called" > ${JSON.stringify(fakeAwsMarker)}`,
    "exit 99",
    "",
  ].join("\n"), "utf8");
  fs.chmodSync(fakeAws, 0o755);

  const result = childProcess.spawnSync("bash", [
    "scripts/aws/launch-solomon-product-run.sh",
    "--execute",
    "--out-dir",
    runDir,
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${fakeBin}${path.delimiter}${process.env.PATH || ""}`,
      NSRL_AMI_ID: "ami-0123456789abcdef0",
      NSRL_S3_URI: "s3://nsrl-product-plan-check/solomon",
      NSRL_ARTIFACT_S3_URI: "s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz",
      NSRL_IAM_INSTANCE_PROFILE: "NSRLTrainingEc2InstanceProfile",
      NSRL_SUBNET_ID: "subnet-0123456789abcdef0",
      NSRL_SECURITY_GROUP_IDS: "sg-0123456789abcdef0 sg-0fedcba9876543210",
      NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE: "m7i.4xlarge",
    },
  });
  fs.writeFileSync(launchLog, [
    "$ scripts/aws/launch-solomon-product-run.sh --execute",
    `status=${result.status}`,
    "",
    "stdout:",
    result.stdout || "",
    "",
    "stderr:",
    result.stderr || "",
  ].join("\n"), "utf8");

  const prelaunch = readJsonMaybe(prelaunchPath);
  const prelaunchErrors = Array.isArray(prelaunch?.errors) ? prelaunch.errors.map(String) : [];
  const launchPath = path.join(runDir, "launch.json");
  const userDataPath = path.join(runDir, "user-data.sh");
  const awsCalled = fs.existsSync(fakeAwsMarker);
  const expectedPrelaunchError = prelaunchErrors.some((item) => item.includes("not a Graviton family"));
  const ok = result.status !== 0 &&
    !awsCalled &&
    fs.existsSync(launchPath) &&
    fs.existsSync(userDataPath) &&
    prelaunch?.ok === false &&
    expectedPrelaunchError;

  return {
    name: "bad-execute-prelaunch-blocks-before-aws",
    ok,
    status: result.status,
    aws_called: awsCalled,
    launch_json: fs.existsSync(launchPath) ? launchPath : "",
    user_data: fs.existsSync(userDataPath) ? userDataPath : "",
    prelaunch_report: fs.existsSync(prelaunchPath) ? prelaunchPath : "",
    prelaunch_ok: prelaunch?.ok === true,
    matched_error: expectedPrelaunchError ? "not a Graviton family" : "",
    errors: [
      result.status !== 0 ? "" : "launch command unexpectedly succeeded",
      awsCalled ? "fake aws binary was invoked before prelaunch failure" : "",
      fs.existsSync(launchPath) ? "" : "launch.json was not written",
      fs.existsSync(userDataPath) ? "" : "user-data.sh was not written",
      prelaunch ? "" : "prelaunch-readiness-check.json was not written",
      prelaunch?.ok === false ? "" : "prelaunch check did not fail",
      expectedPrelaunchError ? "" : "prelaunch errors did not include non-Graviton instance rejection",
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
    prelaunch_errors: prelaunchErrors.slice(0, 20),
  };
}

function runExecuteRecordsLaunchResultCase(root, caseName = "good-execute-records-launch-result", options = {}) {
  const runDir = path.join(root, caseName);
  const fakeBin = path.join(runDir, "bin");
  const fakeAwsMarker = path.join(runDir, "aws-was-called");
  const fakeAwsArgs = path.join(runDir, "aws-args.txt");
  const launchLog = path.join(runDir, "launch-execute.log");
  const instanceId = "i-0123456789abcdef0";
  const profileName = options.awsProfile || "";
  fs.mkdirSync(fakeBin, { recursive: true });
  const fakeAws = path.join(fakeBin, "aws");
  fs.writeFileSync(fakeAws, [
    "#!/usr/bin/env bash",
    `printf "%s\\n" "$@" > ${JSON.stringify(fakeAwsArgs)}`,
    `echo "aws was called" > ${JSON.stringify(fakeAwsMarker)}`,
    "cat <<'JSON'",
    JSON.stringify({
      Instances: [
        {
          InstanceId: instanceId,
          InstanceType: "c8g.4xlarge",
          ImageId: "ami-0123456789abcdef0",
          State: { Name: "pending" },
        },
      ],
    }, null, 2),
    "JSON",
    "",
  ].join("\n"), "utf8");
  fs.chmodSync(fakeAws, 0o755);

  const result = childProcess.spawnSync("bash", [
    "scripts/aws/launch-solomon-product-run.sh",
    "--execute",
    "--out-dir",
    runDir,
  ], {
    cwd: repoRoot,
    encoding: "utf8",
    env: {
      ...process.env,
      PATH: `${fakeBin}${path.delimiter}${process.env.PATH || ""}`,
      AWS_PROFILE: profileName,
      AWS_REGION: "us-east-1",
      NSRL_AMI_ID: "ami-0123456789abcdef0",
      NSRL_S3_URI: "s3://nsrl-product-plan-check/solomon",
      NSRL_ARTIFACT_S3_URI: "s3://nsrl-product-plan-check/solomon/artifacts/nsrl-working-trace-summary.tar.gz",
      NSRL_IAM_INSTANCE_PROFILE: "NSRLTrainingEc2InstanceProfile",
      NSRL_SUBNET_ID: "subnet-0123456789abcdef0",
      NSRL_SECURITY_GROUP_IDS: "sg-0123456789abcdef0 sg-0fedcba9876543210",
      NSRL_SOLOMON_PRODUCT_INSTANCE_TYPE: "c8g.4xlarge",
    },
  });
  fs.writeFileSync(launchLog, [
    "$ scripts/aws/launch-solomon-product-run.sh --execute",
    `status=${result.status}`,
    "",
    "stdout:",
    result.stdout || "",
    "",
    "stderr:",
    result.stderr || "",
  ].join("\n"), "utf8");

  const launchPath = path.join(runDir, "launch.json");
  const launchResultPath = path.join(runDir, "launch-result.json");
  const prelaunchPath = path.join(runDir, "prelaunch-readiness-check.json");
  const launch = readJsonMaybe(launchPath);
  const launchResult = readJsonMaybe(launchResultPath);
  const prelaunch = readJsonMaybe(prelaunchPath);
  const awsArgs = fs.existsSync(fakeAwsArgs)
    ? fs.readFileSync(fakeAwsArgs, "utf8").split(/\r?\n/).filter(Boolean)
    : [];
  const awsCalled = fs.existsSync(fakeAwsMarker);
  const launchCommand = Array.isArray(launch?.command) ? launch.command.map(String) : [];
  const normalizedAwsCommand = normalizeAwsRunInstancesCommand(awsArgs);
  const awsCommandMatchesLaunch = sameArray(normalizedAwsCommand, launchCommand);
  const awsRegion = argValue(awsArgs, "--region");
  const awsRegionMatchesLaunch = awsRegion === String(launch?.region || "");
  const awsProfile = argValue(awsArgs, "--profile");
  const awsProfileMatchesLaunch = awsProfile === String(launch?.aws_profile || "");
  const ok = result.status === 0 &&
    awsCalled &&
    launch?.dry_run === false &&
    launch?.instance_id === instanceId &&
    launch?.launch_result === launchResultPath &&
    typeof launch?.launch_result_sha256 === "string" &&
    launch.launch_result_sha256.length === 64 &&
    launchResult?.Instances?.[0]?.InstanceId === instanceId &&
    prelaunch?.ok === true &&
    awsRegionMatchesLaunch &&
    awsProfileMatchesLaunch &&
    awsCommandMatchesLaunch &&
    awsArgs.includes("--output") &&
    awsArgs.includes("json");

  return {
    name: caseName,
    ok,
    status: result.status,
    aws_called: awsCalled,
    launch_json: fs.existsSync(launchPath) ? launchPath : "",
    launch_result: fs.existsSync(launchResultPath) ? launchResultPath : "",
    prelaunch_report: fs.existsSync(prelaunchPath) ? prelaunchPath : "",
    launch_dry_run: launch?.dry_run ?? null,
    launch_instance_id: launch?.instance_id || "",
    launch_result_instance_id: launchResult?.Instances?.[0]?.InstanceId || "",
    launch_result_sha256: launch?.launch_result_sha256 || "",
    prelaunch_ok: prelaunch?.ok === true,
    aws_region: awsRegion,
    aws_region_matches_launch: awsRegionMatchesLaunch,
    aws_profile: awsProfile,
    aws_profile_matches_launch: awsProfileMatchesLaunch,
    aws_command_matches_launch_manifest: awsCommandMatchesLaunch,
    normalized_aws_command: normalizedAwsCommand,
    launch_command: launchCommand,
    aws_args: awsArgs,
    errors: [
      result.status === 0 ? "" : "launch command failed",
      awsCalled ? "" : "fake aws binary was not invoked",
      launch?.dry_run === false ? "" : "launch.json did not record dry_run=false",
      launch?.instance_id === instanceId ? "" : "launch.json did not record the fake instance id",
      launch?.launch_result === launchResultPath ? "" : "launch.json did not record launch-result.json",
      String(launch?.launch_result_sha256 || "").length === 64 ? "" : "launch_result_sha256 was not recorded",
      launchResult?.Instances?.[0]?.InstanceId === instanceId
        ? ""
        : "launch-result.json did not preserve the fake instance id",
      prelaunch?.ok === true ? "" : "prelaunch readiness was not green before fake AWS execute",
      awsRegionMatchesLaunch ? "" : `aws command region ${awsRegion || ""} != launch region ${launch?.region || ""}`,
      awsProfileMatchesLaunch ? "" : `aws command profile ${awsProfile || ""} != launch profile ${launch?.aws_profile || ""}`,
      awsCommandMatchesLaunch ? "" : "aws run-instances command did not match launch manifest command",
      awsArgs.includes("--output") && awsArgs.includes("json") ? "" : "aws command did not request JSON output",
    ].filter(Boolean),
    stdout_tail: tailLines(result.stdout, 20),
    stderr_tail: tailLines(result.stderr, 20),
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-aws-launch-execute-guard-"));
  const cases = [];
  try {
    cases.push(runExecuteMissingExplicitInputCase(
      root,
      "bad-execute-missing-explicit-s3-blocks-before-aws",
      ["NSRL_S3_URI"],
      "NSRL_S3_URI is required for --execute",
    ));
    cases.push(runExecuteMissingExplicitInputCase(
      root,
      "bad-execute-missing-explicit-artifact-blocks-before-aws",
      ["NSRL_ARTIFACT_S3_URI"],
      "NSRL_ARTIFACT_S3_URI is required for --execute",
    ));
    cases.push(runExecuteGuardCase(root));
    cases.push(runExecuteRecordsLaunchResultCase(root));
    cases.push(runExecuteRecordsLaunchResultCase(root, "good-execute-command-matches-launch-manifest"));
    cases.push(runExecuteRecordsLaunchResultCase(
      root,
      "good-execute-command-matches-launch-manifest-with-profile",
      { awsProfile: "solomon-product-profile" },
    ));
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
