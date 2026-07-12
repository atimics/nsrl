#!/usr/bin/env node

import childProcess from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { makeFixture } from "./check-solomon-aws-run-artifacts-self-test.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_product_release_proof_self_test.v1";

function usage() {
  console.log([
    "Usage: check-solomon-aws-release-proof-self-test.mjs [--out PATH] [--keep] [--include-slow-positive]",
    "",
    "Builds synthetic completed-run directories and checks that the release proof",
    "wrapper rejects mismatched S3/run metadata and broken artifact bundles before",
    "running the expensive product diagnostic. --include-slow-positive also runs",
    "a full positive synthetic release proof.",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { outPath: "", keep: false, includeSlowPositive: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--keep") {
      config.keep = true;
    } else if (arg === "--include-slow-positive") {
      config.includeSlowPositive = true;
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

function runProof(runDir, requestedRunName, outPath, launchDir = "", options = {}) {
  const args = [
    "scripts/aws/prove-solomon-product-run.sh",
    "--skip-sync",
    "--s3-pipeline-uri",
    `s3://nsrl-product-run-check/solomon/pipelines/${requestedRunName}`,
    "--out-dir",
    runDir,
    "--out",
    outPath,
  ];
  if (launchDir) {
    args.push("--launch-dir", launchDir);
  }
  if (options.requireLaunchDir) {
    args.push("--require-launch-dir");
  }
  return childProcess.spawnSync("bash", args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function writeLaunchFixture(root, name, runName, options = {}) {
  const launchDir = path.join(root, `${name}-launch`);
  const dryRun = options.dryRun ?? false;
  const instanceId = options.instanceId ?? "i-0123456789abcdef0";
  const s3Uri = "s3://nsrl-product-run-check/solomon";
  const s3PipelineUri = `${s3Uri}/pipelines/${runName}`;
  const artifactS3Uri = `${s3Uri}/artifacts/nsrl-working-tree.tar.gz`;
  const userDataPath = path.join(launchDir, "user-data.sh");
  const launchPath = path.join(launchDir, "launch.json");
  const launchResultPath = path.join(launchDir, "launch-result.json");
  const postRunProofCommand = options.badPostRunProofCommand
    ? [
      "scripts/aws/prove-solomon-product-run.sh",
      "--s3-pipeline-uri",
      `${s3Uri}/pipelines/not-${runName}`,
      "--launch-dir",
      path.join(root, "wrong-launch"),
    ]
    : [
      "scripts/aws/prove-solomon-product-run.sh",
      "--s3-pipeline-uri",
      s3PipelineUri,
      "--launch-dir",
      launchDir,
      "--require-launch-dir",
    ];
  const userData = [
    "#!/bin/bash",
    "set -euxo pipefail",
    `export NSRL_S3_URI=${s3Uri}`,
    "export NSRL_SOLOMON_AWS_STAGES=dataset,denoiser,prior,generative-eval,attention-curriculum",
    "export NSRL_SOLOMON_REQUIRE_GRAVITON=1",
    "export NSRL_SOLOMON_REQUIRE_EC2_METADATA=1",
    "export NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=1",
    "export NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce",
    "export NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0",
    "export NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=auto-online-processors",
    `aws s3 cp ${artifactS3Uri} /tmp/nsrl.tar.gz`,
    "scripts/aws/run-solomon-end-to-end.sh",
    "",
  ].join("\n");
  fs.mkdirSync(launchDir, { recursive: true });
  fs.writeFileSync(userDataPath, userData, "utf8");
  const securityGroupIds = options.securityGroupIds || "";
  const subnetId = options.subnetId || "";
  let launchResultHash = "";
  let launchResultRecord = "";
  if (!dryRun && instanceId) {
    launchResultRecord = launchResultPath;
    if (!options.omitLaunchResult) {
      const launchResultInstanceType = options.launchResultInstanceType || "c8g.4xlarge";
      const launchResultImageId = options.launchResultImageId || "ami-0123456789abcdef0";
      const launchResultSubnetId = options.launchResultSubnetId ?? subnetId;
      const launchResultSecurityGroupIds = String(
        options.launchResultSecurityGroupIds ?? securityGroupIds,
      ).split(/[,\s]+/).filter(Boolean);
      const launchResultText = `${JSON.stringify({
        Instances: [
          {
            InstanceId: instanceId,
            InstanceType: launchResultInstanceType,
            ImageId: launchResultImageId,
            ...(launchResultSubnetId ? { SubnetId: launchResultSubnetId } : {}),
            ...(launchResultSecurityGroupIds.length > 0
              ? { SecurityGroups: launchResultSecurityGroupIds.map((GroupId) => ({ GroupId })) }
              : {}),
            State: { Name: "pending" },
          },
        ],
      }, null, 2)}\n`;
      fs.writeFileSync(launchResultPath, launchResultText, "utf8");
      launchResultHash = crypto.createHash("sha256").update(launchResultText).digest("hex");
      if (options.corruptLaunchResultHash) {
        launchResultHash = "0".repeat(64);
      }
    }
  }
  const commandUserDataPath = options.badCommandUserDataPath
    ? path.join(launchDir, "wrong-user-data.sh")
    : userDataPath;
  const commandImageId = options.badCommandImageId
    ? "ami-0fedcba9876543210"
    : "ami-0123456789abcdef0";
  const commandInstanceType = options.badCommandInstanceType
    ? "m7i.4xlarge"
    : "c8g.4xlarge";
  const commandTagSpecification = options.badCommandTagSpecification
    ? `ResourceType=instance,Tags=[{Key=Name,Value=${runName}},{Key=Project,Value=wrong-project},{Key=Product,Value=solomon-v1}]`
    : `ResourceType=instance,Tags=[{Key=Name,Value=${runName}},{Key=Project,Value=nsrl-solomon},{Key=Product,Value=solomon-v1}]`;
  const commandSecurityGroups = options.badCommandSecurityGroupIds
    ? ["sg-00000000000000000"]
    : String(securityGroupIds).split(/[,\s]+/).filter(Boolean);
  const commandSubnetId = options.badCommandSubnetId
    ? "subnet-00000000000000000"
    : subnetId;
  const keyName = options.keyName || "";
  const commandKeyName = options.badCommandKeyName
    ? "wrong-solomon-key"
    : keyName;
  const region = options.region || "us-east-1";
  const commandRegion = options.badCommandRegion
    ? "us-west-2"
    : region;
  const awsProfile = options.awsProfile || "";
  const commandProfile = options.badCommandProfile
    ? "wrong-solomon-profile"
    : awsProfile;
  const command = ["aws"];
  if (commandProfile) {
    command.push("--profile", commandProfile);
  }
  command.push(
    "--region", commandRegion,
    "ec2", "run-instances",
    "--image-id", commandImageId,
    "--instance-type", commandInstanceType,
    "--iam-instance-profile", "Name=NSRLTrainingEc2InstanceProfile",
    "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
    "--instance-initiated-shutdown-behavior", "stop",
    "--user-data", `file://${commandUserDataPath}`,
    "--tag-specifications", commandTagSpecification,
    "--output", "json",
  );
  if (commandSecurityGroups.length > 0) {
    command.push("--security-group-ids", ...commandSecurityGroups);
  }
  if (commandSubnetId) {
    command.push("--subnet-id", commandSubnetId);
  }
  if (commandKeyName) {
    command.push("--key-name", commandKeyName);
  }
  fs.writeFileSync(launchPath, `${JSON.stringify({
    schema: "nsrl.solomon_aws_product_launch_plan.v1",
    dry_run: dryRun,
    created_at: "2026-07-03T00:00:00.000Z",
    repo_root: repoRoot,
    run_name: runName,
    pipeline_run_root: "/mnt/nsrl/aws-pipelines",
    pipeline_run_dir: `/mnt/nsrl/aws-pipelines/${runName}`,
    s3_uri: s3Uri,
    s3_pipeline_uri: s3PipelineUri,
    artifact_s3_uri: artifactS3Uri,
    region,
    aws_profile: awsProfile,
    ami_id: "ami-0123456789abcdef0",
    instance_type: "c8g.4xlarge",
    iam_instance_profile: "NSRLTrainingEc2InstanceProfile",
    subnet_id: subnetId,
    security_group_ids: securityGroupIds,
    key_name: keyName,
    tags: {
      Name: runName,
      Project: "nsrl-solomon",
      Product: "solomon-v1",
    },
    instance_id: instanceId,
    launch_result: launchResultRecord,
    launch_result_sha256: launchResultHash,
    user_data: userDataPath,
    user_data_sha256: crypto.createHash("sha256").update(userData).digest("hex"),
    env: {
      NSRL_PIPELINE_RUN_ROOT: "/mnt/nsrl/aws-pipelines",
      NSRL_PIPELINE_RUN_NAME: runName,
      NSRL_S3_URI: s3Uri,
      NSRL_SOLOMON_AWS_STAGES: "dataset,denoiser,prior,generative-eval,attention-curriculum",
      NSRL_SOLOMON_REQUIRE_GRAVITON: "1",
      NSRL_SOLOMON_REQUIRE_EC2_METADATA: "1",
      NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS: "1",
      NSRL_SOLOMON_ATTENTION_BATCH_MODE: "map-reduce",
      NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS: "0",
      NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY: "auto-online-processors",
    },
    post_run_proof_command: postRunProofCommand,
    command,
  }, null, 2)}\n`, "utf8");
  if (options.tamperUserDataAfterHash) {
    fs.appendFileSync(userDataPath, "\n# tampered after launch manifest\n", "utf8");
  }
  return launchDir;
}

function readProof(outPath) {
  if (!fs.existsSync(outPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(outPath, "utf8"));
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function caseResult(definition, result, proof) {
  const actualOk = result.status === 0 && proof?.ok === true;
  const syncedArtifactsDigestMatched = Boolean(proof?.fetch?.synced_artifacts?.sha256) &&
    proof.fetch.synced_artifacts.sha256 === proof?.artifact_check?.synced_artifacts?.sha256;
  const expectations = [];
  expectations.push(syncedArtifactsDigestMatched);
  if (definition.expectFetchOk !== undefined) {
    expectations.push(proof?.fetch?.ok === definition.expectFetchOk);
  }
  if (definition.expectArtifactCheckOk !== undefined) {
    expectations.push(proof?.artifact_check?.ok === definition.expectArtifactCheckOk);
  }
  if (definition.expectDiagnosticSkipped) {
    expectations.push(Number(proof?.product_diagnostic?.status || 0) === 99);
    expectations.push((proof?.errors || []).some((error) => String(error).includes("product diagnostic skipped")));
  }
  if (definition.expectObjectiveSkipped) {
    expectations.push(Number(proof?.objective_coverage?.status || 0) === 99);
    expectations.push((proof?.errors || []).some((error) => String(error).includes("objective coverage skipped")));
  }
  if (definition.expectObjectiveReleaseProof !== undefined) {
    expectations.push(proof?.objective_coverage?.release_objective_proof === definition.expectObjectiveReleaseProof);
  }
  if (definition.requiredError) {
    expectations.push((proof?.errors || []).some((error) => String(error).includes(definition.requiredError)));
  }
  return {
    name: definition.name,
    expect_ok: definition.expectOk,
    ok: actualOk === definition.expectOk && expectations.every(Boolean),
    status: result.status,
    proof_ok: proof?.ok === true,
    fetch_ok: proof?.fetch?.ok === true,
    artifact_check_ok: proof?.artifact_check?.ok === true,
    diagnostic_status: proof?.product_diagnostic?.status ?? null,
    release_product_proof: proof?.product_diagnostic?.release_product_proof === true,
    objective_status: proof?.objective_coverage?.status ?? null,
    release_objective_proof: proof?.objective_coverage?.release_objective_proof === true,
    launch_provided: proof?.launch?.provided === true,
    launch_required: proof?.launch?.required === true,
    launch_ok: proof?.launch?.ok ?? null,
    launch_dry_run: proof?.launch?.dry_run ?? null,
    launch_executed: proof?.launch?.executed ?? null,
    launch_instance_id: proof?.launch?.instance_id || "",
    launch_post_run_proof_command: proof?.launch?.post_run_proof_command || [],
    synced_artifacts_sha256: proof?.fetch?.synced_artifacts?.sha256 || "",
    artifact_check_synced_artifacts_sha256: proof?.artifact_check?.synced_artifacts?.sha256 || "",
    synced_artifacts_digest_matched: syncedArtifactsDigestMatched,
    errors: proof?.errors || [],
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-release-proof-self-test-"));
  const cases = [];
  try {
    const definitions = [
      {
        name: "bad-mismatched-s3-pipeline",
        fixtureName: "bad-mismatched-s3-pipeline",
        requestedRunName: "not-bad-mismatched-s3-pipeline",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: true,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "artifact s3.pipeline_uri",
        mutate: () => {},
      },
      {
        name: "bad-missing-stage-status",
        fixtureName: "bad-missing-stage-status",
        requestedRunName: "bad-missing-stage-status",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "attention-curriculum status",
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-native-product-eval-scope",
        fixtureName: "bad-native-product-eval-scope",
        requestedRunName: "bad-native-product-eval-scope",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "quality-report native phase eval image targets 2 < 72",
        mutate: (state) => {
          state.corruptNativeEvalAfterCheck = true;
        },
      },
      {
        name: "bad-missing-required-launch-dir",
        fixtureName: "bad-missing-required-launch-dir",
        requestedRunName: "bad-missing-required-launch-dir",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch evidence is required",
        requireLaunchDir: true,
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-dry-run-launch-dir",
        fixtureName: "bad-dry-run-launch-dir",
        requestedRunName: "bad-dry-run-launch-dir",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch dry_run is not false",
        launch: { dryRun: true, instanceId: "" },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-missing-launch-instance-id",
        fixtureName: "bad-missing-launch-instance-id",
        requestedRunName: "bad-missing-launch-instance-id",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch instance_id is missing",
        launch: { dryRun: false, instanceId: "" },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-user-data-sha256",
        fixtureName: "bad-launch-user-data-sha256",
        requestedRunName: "bad-launch-user-data-sha256",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: user_data_sha256",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          tamperUserDataAfterHash: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-image-id",
        fixtureName: "bad-launch-command-image-id",
        requestedRunName: "bad-launch-command-image-id",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --image-id",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          badCommandImageId: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-instance-type",
        fixtureName: "bad-launch-command-instance-type",
        requestedRunName: "bad-launch-command-instance-type",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --instance-type",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          badCommandInstanceType: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-user-data",
        fixtureName: "bad-launch-command-user-data",
        requestedRunName: "bad-launch-command-user-data",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --user-data",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          badCommandUserDataPath: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-tag-specification",
        fixtureName: "bad-launch-command-tag-specification",
        requestedRunName: "bad-launch-command-tag-specification",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --tag-specifications",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          badCommandTagSpecification: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-security-groups",
        fixtureName: "bad-launch-command-security-groups",
        requestedRunName: "bad-launch-command-security-groups",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --security-group-ids",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          securityGroupIds: "sg-0123456789abcdef0 sg-0fedcba9876543210",
          badCommandSecurityGroupIds: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-subnet",
        fixtureName: "bad-launch-command-subnet",
        requestedRunName: "bad-launch-command-subnet",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --subnet-id",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          subnetId: "subnet-0123456789abcdef0",
          badCommandSubnetId: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-key-name",
        fixtureName: "bad-launch-command-key-name",
        requestedRunName: "bad-launch-command-key-name",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --key-name",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          keyName: "solomon-product-key",
          badCommandKeyName: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-region",
        fixtureName: "bad-launch-command-region",
        requestedRunName: "bad-launch-command-region",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --region",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          badCommandRegion: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-command-profile",
        fixtureName: "bad-launch-command-profile",
        requestedRunName: "bad-launch-command-profile",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch readiness: launch command --profile",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          awsProfile: "solomon-product-profile",
          badCommandProfile: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-missing-launch-result",
        fixtureName: "bad-missing-launch-result",
        requestedRunName: "bad-missing-launch-result",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch-result.json is missing",
        launch: { dryRun: false, instanceId: "i-0123456789abcdef0", omitLaunchResult: true },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-result-sha256",
        fixtureName: "bad-launch-result-sha256",
        requestedRunName: "bad-launch-result-sha256",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch_result_sha256 does not match launch-result.json",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          corruptLaunchResultHash: true,
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-result-image-id",
        fixtureName: "bad-launch-result-image-id",
        requestedRunName: "bad-launch-result-image-id",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch-result image id ami-0fedcba9876543210 != launch ami_id ami-0123456789abcdef0",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          launchResultImageId: "ami-0fedcba9876543210",
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-result-instance-type",
        fixtureName: "bad-launch-result-instance-type",
        requestedRunName: "bad-launch-result-instance-type",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch-result instance type m7i.4xlarge != launch instance_type c8g.4xlarge",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          launchResultInstanceType: "m7i.4xlarge",
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-result-subnet",
        fixtureName: "bad-launch-result-subnet",
        requestedRunName: "bad-launch-result-subnet",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch-result subnet id subnet-00000000000000000 != launch subnet_id subnet-0123456789abcdef0",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          subnetId: "subnet-0123456789abcdef0",
          launchResultSubnetId: "subnet-00000000000000000",
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-result-security-groups",
        fixtureName: "bad-launch-result-security-groups",
        requestedRunName: "bad-launch-result-security-groups",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch-result security group ids",
        launch: {
          dryRun: false,
          instanceId: "i-0123456789abcdef0",
          securityGroupIds: "sg-0123456789abcdef0 sg-0fedcba9876543210",
          launchResultSecurityGroupIds: "sg-00000000000000000",
        },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-launch-run-instance-mismatch",
        fixtureName: "bad-launch-run-instance-mismatch",
        requestedRunName: "bad-launch-run-instance-mismatch",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch instance_id i-0123456789abcdef0 != run artifact ec2 instance_id i-0fedcba9876543210",
        launch: { dryRun: false, instanceId: "i-0123456789abcdef0" },
        mutate: (state) => {
          state.ec2InstanceId = "i-0fedcba9876543210";
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-post-run-proof-command",
        fixtureName: "bad-post-run-proof-command",
        requestedRunName: "bad-post-run-proof-command",
        expectOk: false,
        expectFetchOk: false,
        expectArtifactCheckOk: false,
        expectDiagnosticSkipped: true,
        expectObjectiveSkipped: true,
        requiredError: "launch post_run_proof_command does not match requested S3 pipeline URI",
        launch: { dryRun: false, instanceId: "i-0123456789abcdef0", badPostRunProofCommand: true },
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
    ];
    if (config.includeSlowPositive) {
      definitions.push({
        name: "good-slow-positive",
        fixtureName: "good-slow-positive",
        requestedRunName: "good-slow-positive",
        expectOk: true,
        expectFetchOk: true,
        expectArtifactCheckOk: true,
        expectObjectiveReleaseProof: true,
        launch: { dryRun: false, instanceId: "i-0123456789abcdef0" },
        mutate: () => {},
      });
    }
    for (const definition of definitions) {
      const runDir = makeFixture(root, definition.fixtureName, definition.mutate);
      const launchDir = definition.launch
        ? writeLaunchFixture(root, definition.name, definition.requestedRunName, definition.launch)
        : "";
      const proofPath = path.join(runDir, "release-proof-self-test.json");
      const result = runProof(runDir, definition.requestedRunName, proofPath, launchDir, {
        requireLaunchDir: definition.requireLaunchDir === true,
      });
      const proof = readProof(proofPath);
      cases.push(caseResult(definition, result, proof));
    }
    const report = {
      schema,
      ok: cases.every((item) => item.ok),
      root,
      kept: config.keep,
      include_slow_positive: config.includeSlowPositive,
      cases,
    };
    writeReport(config.outPath, report);
    console.log(JSON.stringify(report, null, 2));
    if (!report.ok) {
      process.exit(1);
    }
  } finally {
    if (!config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
