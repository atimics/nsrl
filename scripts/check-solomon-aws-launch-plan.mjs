#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const schema = "nsrl.solomon_aws_product_launch_plan_check.v1";
const launchSchema = "nsrl.solomon_aws_product_launch_plan.v1";
const requiredStages = "dataset,denoiser,prior,generative-eval,attention-curriculum";
const gravitonInstance = /^(?:c|m|r|t)(?:6|7|8)g[dn]?\./;

const defaults = {
  launchDir: "",
  launchPath: "",
  userDataPath: "",
  outPath: "",
  requireDryRun: true,
  requireGravitonInstance: true,
};

function usage() {
  console.log([
    "Usage: check-solomon-aws-launch-plan.mjs --launch-dir PATH [options]",
    "   or: check-solomon-aws-launch-plan.mjs --launch PATH --user-data PATH",
    "",
    "Checks a dry-run Solomon EC2 launch manifest and user-data script before",
    "launching a real Graviton product run.",
    "",
    "Options:",
    "  --out PATH",
    "  --allow-execute-plan",
    "  --allow-non-graviton-instance",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--launch-dir") {
      config.launchDir = requireValue(argv, ++index, arg);
    } else if (arg === "--launch") {
      config.launchPath = requireValue(argv, ++index, arg);
    } else if (arg === "--user-data") {
      config.userDataPath = requireValue(argv, ++index, arg);
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--allow-execute-plan") {
      config.requireDryRun = false;
    } else if (arg === "--allow-non-graviton-instance") {
      config.requireGravitonInstance = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.launchDir) {
    config.launchPath ||= path.join(config.launchDir, "launch.json");
    config.userDataPath ||= path.join(config.launchDir, "user-data.sh");
  }
  if (!config.launchPath) {
    throw new Error("--launch-dir or --launch is required");
  }
  if (!config.userDataPath) {
    throw new Error("--launch-dir or --user-data is required");
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function sha256(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function requireField(row, key, expected, errors) {
  const actual = row[key];
  if (actual !== expected) {
    errors.push(`${key} ${JSON.stringify(actual ?? "")} != ${JSON.stringify(expected)}`);
  }
}

function requireStartsWith(row, key, prefix, errors) {
  const actual = String(row[key] || "");
  if (!actual.startsWith(prefix)) {
    errors.push(`${key} ${JSON.stringify(actual)} must start with ${JSON.stringify(prefix)}`);
  }
}

function requireUserDataContains(userData, text, errors) {
  if (!userData.includes(text)) {
    errors.push(`user-data missing ${text}`);
  }
}

function commandValue(command, flag) {
  const index = command.indexOf(flag);
  return index >= 0 ? command[index + 1] || "" : "";
}

function commandValues(command, flag) {
  const index = command.indexOf(flag);
  if (index < 0) {
    return [];
  }
  const values = [];
  for (let cursor = index + 1; cursor < command.length; cursor += 1) {
    const value = String(command[cursor] || "");
    if (value.startsWith("--")) {
      break;
    }
    values.push(value);
  }
  return values;
}

function requireCommandValue(command, flag, expected, errors) {
  const actual = commandValue(command, flag);
  if (actual !== expected) {
    errors.push(`launch command ${flag} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function requireOptionalCommandValue(command, flag, expected, errors) {
  if (expected) {
    requireCommandValue(command, flag, expected, errors);
  } else if (command.includes(flag)) {
    errors.push(`launch command ${flag} is present but launch manifest has no value`);
  }
}

function requireCommandValues(command, flag, expected, errors) {
  const actual = commandValues(command, flag);
  if (expected.length === 0 && command.includes(flag)) {
    errors.push(`launch command ${flag} is present but launch manifest has no values`);
  } else if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    errors.push(`launch command ${flag} ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function launchTagSpecification(launch) {
  return `ResourceType=instance,Tags=[{Key=Name,Value=${launch.run_name || ""}},{Key=Project,Value=nsrl-solomon},{Key=Product,Value=solomon-v1}]`;
}

function splitIds(text) {
  return String(text || "").split(/[,\s]+/).filter(Boolean);
}

function requirePostRunProofCommand(launch, config, errors) {
  const command = Array.isArray(launch.post_run_proof_command)
    ? launch.post_run_proof_command.map(String)
    : [];
  const expectedLaunchDir = path.resolve(config.launchDir || path.dirname(config.launchPath));
  const s3Index = command.indexOf("--s3-pipeline-uri");
  const launchDirIndex = command.indexOf("--launch-dir");
  if (!command.includes("scripts/aws/prove-solomon-product-run.sh")) {
    errors.push("post_run_proof_command missing scripts/aws/prove-solomon-product-run.sh");
  }
  if (s3Index < 0 || command[s3Index + 1] !== String(launch.s3_pipeline_uri || "")) {
    errors.push("post_run_proof_command does not use launch s3_pipeline_uri");
  }
  if (launchDirIndex < 0 || !command[launchDirIndex + 1]) {
    errors.push("post_run_proof_command missing --launch-dir");
  } else if (path.resolve(command[launchDirIndex + 1]) !== expectedLaunchDir) {
    errors.push(`post_run_proof_command launch dir ${command[launchDirIndex + 1]} != ${expectedLaunchDir}`);
  }
  if (!command.includes("--require-launch-dir")) {
    errors.push("post_run_proof_command missing --require-launch-dir");
  }
}

function check(config) {
  const errors = [];
  const launch = readJson(config.launchPath);
  const userData = fs.readFileSync(config.userDataPath, "utf8");
  const env = launch.env || {};
  const securityGroups = splitIds(launch.security_group_ids);

  requireField(launch, "schema", launchSchema, errors);
  if (config.requireDryRun && launch.dry_run !== true) {
    errors.push("launch dry_run is not true");
  }
  if (config.requireGravitonInstance && !gravitonInstance.test(String(launch.instance_type || ""))) {
    errors.push(`instance_type ${JSON.stringify(launch.instance_type || "")} is not a Graviton family`);
  }
  requireStartsWith(launch, "s3_uri", "s3://", errors);
  requireStartsWith(launch, "s3_pipeline_uri", "s3://", errors);
  requireStartsWith(launch, "artifact_s3_uri", "s3://", errors);
  if (
    launch.s3_uri &&
    launch.s3_pipeline_uri &&
    !String(launch.s3_pipeline_uri).startsWith(`${String(launch.s3_uri).replace(/\/+$/, "")}/pipelines/`)
  ) {
    errors.push(`s3_pipeline_uri ${launch.s3_pipeline_uri} is not under ${launch.s3_uri}/pipelines/`);
  }
  if (!String(launch.pipeline_run_dir || "").endsWith(String(launch.run_name || ""))) {
    errors.push("pipeline_run_dir does not end with run_name");
  }
  if (!String(launch.user_data || "")) {
    errors.push("launch user_data is missing");
  } else if (path.resolve(String(launch.user_data || "")) !== path.resolve(config.userDataPath)) {
    errors.push(`launch user_data ${JSON.stringify(launch.user_data || "")} != ${config.userDataPath}`);
  }
  const recordedHash = String(launch.user_data_sha256 || "");
  const actualHash = sha256(userData);
  if (recordedHash !== actualHash) {
    errors.push(`user_data_sha256 ${JSON.stringify(recordedHash)} != ${actualHash}`);
  }

  requireField(env, "NSRL_PIPELINE_RUN_NAME", launch.run_name || "", errors);
  requireField(env, "NSRL_PIPELINE_RUN_ROOT", launch.pipeline_run_root || "", errors);
  requireField(env, "NSRL_S3_URI", launch.s3_uri || "", errors);
  requireField(env, "NSRL_SOLOMON_AWS_STAGES", requiredStages, errors);
  requireField(env, "NSRL_SOLOMON_REQUIRE_GRAVITON", "1", errors);
  requireField(env, "NSRL_SOLOMON_REQUIRE_EC2_METADATA", "1", errors);
  requireField(env, "NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS", "1", errors);
  requireField(env, "NSRL_SOLOMON_ATTENTION_BATCH_MODE", "map-reduce", errors);
  requireField(env, "NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS", "0", errors);
  requireField(env, "NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY", "auto-online-processors", errors);

  for (const text of [
    "aws s3 cp",
    String(launch.artifact_s3_uri || ""),
    "scripts/aws/run-solomon-end-to-end.sh",
    "NSRL_SOLOMON_REQUIRE_GRAVITON=1",
    "NSRL_SOLOMON_REQUIRE_EC2_METADATA=1",
    "NSRL_SOLOMON_REQUIRE_S3_ARTIFACTS=1",
    "NSRL_SOLOMON_ATTENTION_BATCH_MODE=map-reduce",
    "NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS=0",
    "NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY=auto-online-processors",
  ]) {
    requireUserDataContains(userData, text, errors);
  }

  const command = Array.isArray(launch.command) ? launch.command.map(String) : [];
  for (const text of [
    "aws",
    "--region",
    "ec2",
    "run-instances",
    "--image-id",
    "--instance-type",
    "--iam-instance-profile",
    "--metadata-options",
    "HttpTokens=required,HttpEndpoint=enabled",
    "--instance-initiated-shutdown-behavior",
    "stop",
    "--user-data",
    "--tag-specifications",
    "--output",
    "json",
  ]) {
    if (!command.includes(text)) {
      errors.push(`launch command missing ${text}`);
    }
  }
  requireCommandValue(command, "--region", String(launch.region || ""), errors);
  requireOptionalCommandValue(command, "--profile", String(launch.aws_profile || ""), errors);
  requireCommandValue(command, "--image-id", String(launch.ami_id || ""), errors);
  requireCommandValue(command, "--instance-type", String(launch.instance_type || ""), errors);
  requireCommandValue(command, "--iam-instance-profile", `Name=${launch.iam_instance_profile || ""}`, errors);
  requireCommandValue(command, "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled", errors);
  requireCommandValue(command, "--instance-initiated-shutdown-behavior", "stop", errors);
  requireCommandValue(command, "--output", "json", errors);
  requireCommandValue(command, "--tag-specifications", launchTagSpecification(launch), errors);
  const userDataCommandValue = commandValue(command, "--user-data");
  if (!userDataCommandValue.startsWith("file://")) {
    errors.push(`launch command --user-data ${JSON.stringify(userDataCommandValue)} is not a file:// URI`);
  } else if (path.resolve(userDataCommandValue.slice("file://".length)) !== path.resolve(config.userDataPath)) {
    errors.push(`launch command --user-data ${userDataCommandValue} != file://${config.userDataPath}`);
  }
  for (const tag of [
    `Key=Name,Value=${launch.run_name || ""}`,
    "Key=Project,Value=nsrl-solomon",
    "Key=Product,Value=solomon-v1",
  ]) {
    if (!command.some((item) => String(item).includes(tag))) {
      errors.push(`launch command missing tag ${tag}`);
    }
  }
  requireOptionalCommandValue(command, "--subnet-id", String(launch.subnet_id || ""), errors);
  requireCommandValues(command, "--security-group-ids", securityGroups, errors);
  requireOptionalCommandValue(command, "--key-name", String(launch.key_name || ""), errors);
  requirePostRunProofCommand(launch, config, errors);

  return {
    schema,
    ok: errors.length === 0,
    launch: config.launchPath,
    user_data: config.userDataPath,
    dry_run: launch.dry_run === true,
    run_name: launch.run_name || "",
    region: launch.region || "",
    aws_profile: launch.aws_profile || "",
    instance_type: launch.instance_type || "",
    graviton_instance: gravitonInstance.test(String(launch.instance_type || "")),
    ec2_metadata_required: env.NSRL_SOLOMON_REQUIRE_EC2_METADATA === "1",
    s3_uri: launch.s3_uri || "",
    s3_pipeline_uri: launch.s3_pipeline_uri || "",
    artifact_s3_uri: launch.artifact_s3_uri || "",
    post_run_proof_command: Array.isArray(launch.post_run_proof_command)
      ? launch.post_run_proof_command.map(String)
      : [],
    product_stages: String(env.NSRL_SOLOMON_AWS_STAGES || "").split(",").filter(Boolean),
    cpu_scaling: {
      batch_mode: env.NSRL_SOLOMON_ATTENTION_BATCH_MODE || "",
      map_reduce_workers: env.NSRL_SOLOMON_ATTENTION_MAP_REDUCE_WORKERS || "",
      policy: env.NSRL_SOLOMON_ATTENTION_CPU_SCALING_POLICY || "",
    },
    user_data_sha256: actualHash,
    errors,
  };
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

try {
  const config = parseArgs(process.argv.slice(2));
  const report = check(config);
  writeReport(config.outPath, report);
  console.log(JSON.stringify(report, null, 2));
  if (!report.ok) {
    process.exit(1);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(2);
}
