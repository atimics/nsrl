#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_prior_smoke_self_test.v1";
const grid = 16;
const imageSize = 128;
const prompts = [
  { slug: "crocell", prompt: "Crocell", number: 49, name: "Crocell" },
  { slug: "stolas", prompt: "Stolas", number: 36, name: "Stolas" },
  { slug: "bael", prompt: "Bael", number: 1, name: "Bael" },
  { slug: "hidden-geometry-waters", prompt: "hidden geometry and rushing waters", number: 49, name: "Crocell" },
  { slug: "astronomy-herbs-teacher", prompt: "astronomy and herbs teacher", number: 36, name: "Stolas" },
];
const seedVariants = ["a", "b", "c"];

function usage() {
  console.log([
    "Usage: check-solomon-prior-smoke-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic prior-smoke run directories and proves the smoke checker",
    "rejects wrong latent routing, missing seed variants, collapsed layouts, and",
    "weak held-out class eval evidence without training a real prior.",
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

function signatureForNumber(number, collapsed = false) {
  const signature = new Array(grid * grid).fill(0);
  const slot = collapsed ? 0 : ({ 49: 0, 36: 1, 1: 2 }[number] ?? 3);
  const yStart = slot * 3;
  for (let y = yStart; y < yStart + 2; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      signature[y * grid + x] = 255;
    }
  }
  return signature;
}

function rawImageFromSignature(signature) {
  const bytes = Buffer.alloc(imageSize * imageSize, 0);
  const cell = imageSize / grid;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const value = signature[gy * grid + gx] >= 128 ? 255 : 0;
      for (let y = gy * cell; y < (gy + 1) * cell; y += 1) {
        for (let x = gx * cell; x < (gx + 1) * cell; x += 1) {
          bytes[y * imageSize + x] = value;
        }
      }
    }
  }
  return bytes;
}

function writeRunFixture(root, name, options = {}) {
  const runDir = path.join(root, name);
  const samplesDir = path.join(runDir, "samples");
  fs.mkdirSync(samplesDir, { recursive: true });
  const manifestRows = [];
  const evalTop1 = options.evalTop1 ?? 1000;

  for (const prompt of prompts) {
    for (const seed of seedVariants) {
      if (options.missingSeedVariant && prompt.slug === "bael" && seed === "c") {
        continue;
      }
      const targetNumber = options.wrongTargetNumber && prompt.slug === "crocell" ? 36 : prompt.number;
      const targetName = prompts.find((item) => item.number === targetNumber)?.name || prompt.name;
      const signature = signatureForNumber(targetNumber, options.collapsedInterclass);
      const outDir = path.join(samplesDir, `${prompt.slug}-${seed}`);
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(outDir, `samples.ink${imageSize}.u8`), rawImageFromSignature(signature));
      fs.writeFileSync(path.join(outDir, "samples.pgm"), "P5\n128 128\n255\n", "utf8");
      writeJson(path.join(outDir, "trace.json"), {
        schema: "nsrl.bitmap_sampler_trace.v1",
        model_format: "NSRLTCH",
        feature_channels: 30,
        image_size: imageSize,
        samples: 1,
        latent_target_source: options.targetSource || "decoded-latent",
        latent_target_number: targetNumber,
        latent_target_name: targetName,
        latent_target_score: 1000,
        latent_target_signature: signature,
        raw_samples: path.join(outDir, `samples.ink${imageSize}.u8`),
      });
      manifestRows.push([
        prompt.slug,
        seed,
        prompt.prompt,
        outDir,
        path.join(outDir, "samples.pgm"),
        "",
        options.targetSource || "decoded-latent",
        "96",
        String(targetNumber),
        targetName,
        "1000",
      ]);
    }
  }

  fs.writeFileSync(
    path.join(runDir, "manifest.tsv"),
    [
      "prompt_slug\tseed_variant\tprompt\tout_dir\tpgm\tpng\tlatent_target_source\ttext_weight\tlatent_target_number\tlatent_target_name\tlatent_target_score",
      ...manifestRows.map((row) => row.join("\t")),
      "",
    ].join("\n"),
    "utf8",
  );
  fs.writeFileSync(
    path.join(runDir, "eval-ledger.jsonl"),
    `${JSON.stringify({ prior_eval: { all: { class_top1_per_mille: evalTop1 } } })}\n`,
    "utf8",
  );
  return runDir;
}

function runChecker(runDir) {
  return childProcess.spawnSync(process.execPath, [
    "scripts/check-solomon-prior-smoke.mjs",
    "--run-dir",
    runDir,
    "--min-seed-variants",
    "3",
    "--max-intra-prompt-distance",
    "8192",
    "--max-target-distance",
    "24576",
    "--min-inter-class-distance",
    "1024",
    "--min-target-ink-cells",
    "8",
    "--max-target-ink-cells",
    "224",
    "--min-eval-class-top1",
    "1",
    "--expected-target-source",
    "decoded-latent",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function readReport(stdout) {
  const text = String(stdout || "").trim();
  if (!text) return null;
  return JSON.parse(text);
}

function caseResult(definition, result, report) {
  const actualOk = result.status === 0 && report?.passed === true;
  const requiredErrorOk = definition.requiredError
    ? (report?.failures || []).some((failure) => String(failure).includes(definition.requiredError))
    : true;
  return {
    name: definition.name,
    expect_ok: definition.expectOk,
    ok: actualOk === definition.expectOk && requiredErrorOk,
    status: result.status,
    passed: report?.passed === true,
    eval_class_top1_per_mille: report?.eval_class_top1_per_mille ?? null,
    prompt_groups: report?.prompt_groups || [],
    failures: report?.failures || [],
    stdout_tail: result.stdout ? tailLines(result.stdout, 20) : "",
    stderr_tail: result.stderr ? tailLines(result.stderr, 20) : "",
  };
}

function tailLines(text, maxLines) {
  const lines = String(text).split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - maxLines)).join("\n");
}

function writeReport(outPath, report) {
  if (!outPath) return;
  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-prior-smoke-self-test-"));
  const cases = [];
  try {
    const definitions = [
      { name: "good", expectOk: true, options: {} },
      {
        name: "bad-target-source",
        expectOk: false,
        options: { targetSource: "target-bitmap-lookup" },
        requiredError: "used target-bitmap-lookup, expected decoded-latent",
      },
      {
        name: "bad-missing-seed-variant",
        expectOk: false,
        options: { missingSeedVariant: true },
        requiredError: "bael has 2 seed variants",
      },
      {
        name: "bad-collapsed-interclass",
        expectOk: false,
        options: { collapsedInterclass: true },
        requiredError: "generated signature distance 0 < 1024",
      },
      {
        name: "bad-eval-class-top1",
        expectOk: false,
        options: { evalTop1: 0 },
        requiredError: "eval class_top1_per_mille 0 < 1",
      },
    ];
    for (const definition of definitions) {
      const runDir = writeRunFixture(root, definition.name, definition.options);
      const result = runChecker(runDir);
      const report = readReport(result.stdout);
      cases.push(caseResult(definition, result, report));
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
