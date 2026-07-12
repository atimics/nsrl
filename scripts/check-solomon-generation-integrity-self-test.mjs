#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_generation_integrity_self_test.v1";
const checkScript = "scripts/check-solomon-generation-integrity.mjs";
const integritySchema = "nsrl.solomon_generation_integrity_check.v1";

function usage() {
  console.log([
    "Usage: check-solomon-generation-integrity-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds tiny generation traces and proves the integrity checker accepts",
    "clean latent/attention generation while rejecting target-pixel, oracle,",
    "cleanup, and raw-sample side channels.",
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

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writeReport(outPath, report) {
  if (!outPath) {
    return;
  }
  writeJson(path.resolve(outPath), report);
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

function bitmapTrace(overrides = {}) {
  return {
    schema: "nsrl.bitmap_sampler_trace.v1",
    image_size: 128,
    latent_target_source: "decoded-latent",
    latent_target_number: 11,
    latent_target_name: "fixture-bael",
    raw_samples: "samples.ink128.u8",
    ...overrides,
  };
}

function attentionTrace(overrides = {}) {
  return {
    schema: "nsrl.solomon_attention_sample_trace.v1",
    text_prior_source: "embedded_lm",
    image_prior_source: "embedded",
    init_mode: "native-bind",
    sample: {
      spirit_id: "fixture-bael",
      text: "bael",
      image_ink16_u8: "image.ink16.u8",
    },
    ...overrides,
  };
}

function writeBitmapSample(root, name, trace, { writeRaw = true } = {}) {
  const dir = path.join(root, name, "sample");
  fs.mkdirSync(dir, { recursive: true });
  writeJson(path.join(dir, "trace.json"), trace);
  if (writeRaw) {
    fs.writeFileSync(path.join(dir, "samples.ink128.u8"), Buffer.alloc(128 * 128, 17));
  }
  return dir;
}

function writeTrace(root, name, trace) {
  const tracePath = path.join(root, name, "trace.json");
  writeJson(tracePath, trace);
  return tracePath;
}

function runIntegrity(root, name, args) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  const outPath = path.join(dir, "generation-integrity.json");
  const result = childProcess.spawnSync(process.execPath, [checkScript, ...args, "--out", outPath], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  return {
    status: result.status,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    report_path: fs.existsSync(outPath) ? outPath : "",
    report: readJsonMaybe(outPath),
  };
}

function evaluateCase(name, run, { expectOk, expectedFailureIncludes = [] }) {
  const combined = `${run.stdout}\n${run.stderr}\n${JSON.stringify(run.report || {})}`;
  const matchedFailureIncludes = expectedFailureIncludes.filter((needle) => combined.includes(needle));
  const statusOk = expectOk ? run.status === 0 : run.status !== 0;
  const reportOk = expectOk ? run.report?.ok === true : run.report?.ok === false;
  const schemaOk = run.report?.schema === integritySchema;
  const failuresOk = matchedFailureIncludes.length === expectedFailureIncludes.length;
  return {
    name,
    ok: statusOk && reportOk && schemaOk && failuresOk,
    expected_ok: expectOk,
    status: run.status,
    schema: run.report?.schema || "",
    report_ok: run.report?.ok === true,
    trace_count: run.report?.trace_count || 0,
    report: run.report_path,
    violation_fields: Array.isArray(run.report?.violations)
      ? run.report.violations.map((item) => item.field || "")
      : [],
    expected_failure_includes: expectedFailureIncludes,
    matched_failure_includes: matchedFailureIncludes,
    errors: [
      statusOk ? "" : `status ${run.status} did not match expected ${expectOk ? "success" : "failure"}`,
      reportOk ? "" : `report ok did not match expected ${expectOk}`,
      schemaOk ? "" : `schema ${JSON.stringify(run.report?.schema || "")} did not match ${integritySchema}`,
      failuresOk
        ? ""
        : `missing expected failure text: ${expectedFailureIncludes.filter((item) => !matchedFailureIncludes.includes(item)).join(", ")}`,
    ].filter(Boolean),
    stdout_tail: tailLines(run.stdout, 20),
    stderr_tail: tailLines(run.stderr, 20),
  };
}

function goodBitmapDecodedLatentCase(root) {
  const name = "good-bitmap-decoded-latent";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace());
  return evaluateCase(name, runIntegrity(root, name, [
    "--sample-dir",
    sampleDir,
    "--expected-latent-target-source",
    "decoded-latent",
  ]), {
    expectOk: true,
  });
}

function goodAttentionEmbeddedCase(root) {
  const name = "good-attention-embedded";
  const tracePath = writeTrace(root, name, attentionTrace());
  return evaluateCase(name, runIntegrity(root, name, ["--trace", tracePath]), {
    expectOk: true,
  });
}

function badTargetSourceCase(root) {
  const name = "bad-target-source";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace({
    target_source: "target-pixel",
  }));
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["generation traces must use latent_target_source"],
  });
}

function badTargetPixelKeyCase(root) {
  const name = "bad-target-pixel-key";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace({
    sampler: {
      target_pixel_guidance: "fixture-plan",
    },
  }));
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["forbidden target-pixel, oracle, guidance, or cleanup field"],
  });
}

function badOracleValueCase(root) {
  const name = "bad-oracle-value";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace({
    init_mode: "oracle",
  }));
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["forbidden target-pixel, oracle, retrieval-hybrid, or cleanup value"],
  });
}

function badDisplayCleanupCase(root) {
  const name = "bad-display-cleanup";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace({
    display_cleanup: "contrast polish",
  }));
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["forbidden target-pixel, oracle, guidance, or cleanup field"],
  });
}

function badRawPathCase(root) {
  const name = "bad-raw-path";
  const outsidePath = path.join(root, name, "outside.ink128.u8");
  fs.mkdirSync(path.dirname(outsidePath), { recursive: true });
  fs.writeFileSync(outsidePath, Buffer.alloc(128 * 128, 3));
  const sampleDir = writeBitmapSample(root, name, bitmapTrace({
    raw_samples: "../outside.ink128.u8",
  }));
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["raw_samples must resolve to samples.ink128.u8"],
  });
}

function badMissingRawCase(root) {
  const name = "bad-missing-raw";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace(), { writeRaw: false });
  return evaluateCase(name, runIntegrity(root, name, ["--sample-dir", sampleDir]), {
    expectOk: false,
    expectedFailureIncludes: ["missing generated raw sample bytes"],
  });
}

function badExpectedLatentSourceCase(root) {
  const name = "bad-expected-latent-source";
  const sampleDir = writeBitmapSample(root, name, bitmapTrace());
  return evaluateCase(name, runIntegrity(root, name, [
    "--sample-dir",
    sampleDir,
    "--expected-latent-target-source",
    "attention-plan",
  ]), {
    expectOk: false,
    expectedFailureIncludes: ["attention-plan"],
  });
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-generation-integrity-self-test-"));
  const cases = [];
  try {
    cases.push(goodBitmapDecodedLatentCase(root));
    cases.push(goodAttentionEmbeddedCase(root));
    cases.push(badTargetSourceCase(root));
    cases.push(badTargetPixelKeyCase(root));
    cases.push(badOracleValueCase(root));
    cases.push(badDisplayCleanupCase(root));
    cases.push(badRawPathCase(root));
    cases.push(badMissingRawCase(root));
    cases.push(badExpectedLatentSourceCase(root));
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
