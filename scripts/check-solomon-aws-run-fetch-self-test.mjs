#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { makeFixture } from "./check-solomon-aws-run-artifacts-self-test.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_aws_run_fetch_self_test.v1";

function usage() {
  console.log([
    "Usage: check-solomon-aws-run-fetch-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic completed-run directories and checks that the S3 fetch",
    "wrapper succeeds on an already-synced good run while rejecting an already",
    "synced bad run, without touching AWS.",
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

function runFetch(runDir, item) {
  const args = [
    "scripts/aws/fetch-solomon-product-run.sh",
    "--skip-sync",
  ];
  const env = { ...process.env };
  if (item.invoke === "run-name") {
    env.NSRL_S3_URI = "s3://nsrl-product-run-check/solomon";
    args.push("--run-name", item.name);
  } else {
    const runName = item.requestedRunName || item.name;
    args.push("--s3-pipeline-uri", `s3://nsrl-product-run-check/solomon/pipelines/${runName}`);
  }
  args.push("--out-dir", runDir);
  return childProcess.spawnSync("bash", args, {
    cwd: repoRoot,
    encoding: "utf8",
    env,
  });
}

function extractReport(stdout) {
  const start = stdout.lastIndexOf("{");
  if (start < 0) {
    return null;
  }
  return JSON.parse(stdout.slice(start));
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-aws-run-fetch-self-test-"));
  const cases = [];
  try {
    const definitions = [
      {
        name: "good",
        expectOk: true,
        invoke: "s3-pipeline-uri",
        mutate: () => {},
      },
      {
        name: "good-run-name",
        expectOk: true,
        invoke: "run-name",
        mutate: () => {},
      },
      {
        name: "bad-mismatched-s3-pipeline",
        expectOk: false,
        invoke: "s3-pipeline-uri",
        requestedRunName: "not-bad-mismatched-s3-pipeline",
        mutate: () => {},
      },
      {
        name: "bad-missing-status",
        expectOk: false,
        invoke: "s3-pipeline-uri",
        mutate: (state) => {
          state.omitStatus = "attention-curriculum";
        },
      },
      {
        name: "bad-stale-promotion",
        expectOk: false,
        invoke: "s3-pipeline-uri",
        mutate: (state) => {
          state.corruptQualityAfterCheck = true;
        },
      },
      {
        name: "bad-native-product-eval-scope",
        expectOk: false,
        invoke: "s3-pipeline-uri",
        requiredError: "quality-report native phase eval image targets 2 < 72",
        mutate: (state) => {
          state.corruptNativeEvalAfterCheck = true;
        },
      },
    ];
    for (const item of definitions) {
      const runDir = makeFixture(root, item.name, item.mutate);
      const result = runFetch(runDir, item);
      const reportPath = path.join(runDir, "fetch-report.json");
      const report = fs.existsSync(reportPath)
        ? JSON.parse(fs.readFileSync(reportPath, "utf8"))
        : extractReport(result.stdout || "");
      const artifactCheck = report?.artifact_check_path && fs.existsSync(report.artifact_check_path)
        ? JSON.parse(fs.readFileSync(report.artifact_check_path, "utf8"))
        : null;
      const actualOk = result.status === 0 && report?.ok === true;
      const errors = report?.errors || [];
      const requiredErrorMatched = !item.requiredError ||
        errors.some((error) => String(error).includes(item.requiredError));
      const digestMatched = Boolean(report?.synced_artifacts?.sha256) &&
        report.synced_artifacts.sha256 === artifactCheck?.synced_artifacts?.sha256;
      cases.push({
        name: item.name,
        expect_ok: item.expectOk,
        invocation: item.invoke,
        ok: actualOk === item.expectOk && requiredErrorMatched && digestMatched,
        status: result.status,
        fetch_ok: report?.ok === true,
        artifact_check_ok: report?.artifact_check_ok === true,
        synced_artifacts_sha256: report?.synced_artifacts?.sha256 || "",
        artifact_check_synced_artifacts_sha256: artifactCheck?.synced_artifacts?.sha256 || "",
        synced_artifacts_digest_matched: digestMatched,
        required_error: item.requiredError || "",
        required_error_matched: requiredErrorMatched,
        errors,
      });
    }
    const report = {
      schema,
      ok: cases.every((item) => item.ok),
      root,
      kept: config.keep,
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
