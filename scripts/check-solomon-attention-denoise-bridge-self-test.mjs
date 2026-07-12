#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const imageSize = 128;
const signatureGrid = 16;
const signatureBins = signatureGrid * signatureGrid;

function usage() {
  console.log([
    "Usage: check-solomon-attention-denoise-bridge-self-test.mjs [--keep]",
    "",
    "Builds tiny attention-plan/denoiser fixtures and proves the denoise bridge",
    "checker accepts clean attention-plan provenance while rejecting cleanup,",
    "wrong source, forged signatures, flat output bytes, and weak output",
    "image-to-text retrieval margins.",
    "",
    "Options:",
    "  --keep   keep the temporary fixture directory for debugging",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { keep: false };
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-denoise-bridge-self-test-"));
  let completed = false;
  try {
    const shared = writeSharedFixture(root);
    const cases = [];

    const good = writeCase(root, shared, "good");
    const goodResult = runBridge(shared, good);
    assertStatus(goodResult, 0, "clean denoise bridge should pass");
    const goodReport = JSON.parse(goodResult.stdout);
    assertEqual(goodReport.ok, true, "clean report ok");
    assertEqual(goodReport.trace_integrity_ok, true, "clean trace integrity");
    assertEqual(goodReport.output_image_to_text_identification, true, "clean output identity");
    assertEqual(goodReport.expected_unique_targets, 1, "clean unique target count");
    assertEqual(goodReport.target_coverage_ok, true, "clean target coverage");
    assertEqual(goodReport.min_output_signature_distance, 0, "clean output signature distance");
    if (!(Number(goodReport.min_output_ink_range || 0) > 0)) {
      throw new Error("clean output ink range should be positive");
    }
    if (!(Number(goodReport.min_output_retrieval_image_margin || 0) > 0)) {
      throw new Error("clean output retrieval margin should be positive");
    }
    if (goodReport.denoise_model_provenance?.ok !== true) {
      throw new Error("clean denoise model provenance should be ok");
    }
    if (goodReport.retrieval_head_provenance?.ok !== true) {
      throw new Error("clean retrieval head provenance should be ok");
    }
    cases.push({ name: "good", ok: true });

    for (const badCase of [
      {
        name: "bad-cleanup",
        mutate: ({ trace }) => {
          trace.display_cleanup = "postprocess target bitmap";
        },
        expected: "display_cleanup",
      },
      {
        name: "bad-source",
        mutate: ({ trace }) => {
          trace.latent_target_source = "decoded-latent";
        },
        expected: "latent_target_source",
      },
      {
        name: "bad-signature",
        mutate: ({ trace }) => {
          trace.latent_target_signature[0] = trace.latent_target_signature[0] ^ 1;
        },
        expected: "latent_target_signature differs from attention plan",
      },
      {
        name: "bad-flat-output",
        mutate: ({ rawPath }) => {
          fs.writeFileSync(rawPath, Buffer.alloc(imageSize * imageSize));
        },
        expected: "min output ink range",
      },
    ]) {
      const fixture = writeCase(root, shared, badCase.name, badCase.mutate);
      const result = runBridge(shared, fixture);
      assertFailure(result, badCase.expected, `${badCase.name} should fail`);
      cases.push({ name: badCase.name, ok: true });
    }

    const badRetrievalHeadHash = writeCase(root, shared, "bad-retrieval-head-hash");
    const badRetrievalHeadPath = path.join(root, "bad-retrieval-head-hash.json");
    const badRetrievalHead = JSON.parse(fs.readFileSync(shared.retrievalHeadPath, "utf8"));
    badRetrievalHead.model_hash = "0x0000000000000000";
    fs.writeFileSync(badRetrievalHeadPath, `${JSON.stringify(badRetrievalHead, null, 2)}\n`, "utf8");
    const badRetrievalHeadResult = runBridge(
      { ...shared, retrievalHeadPath: badRetrievalHeadPath },
      badRetrievalHeadHash,
    );
    assertFailure(
      badRetrievalHeadResult,
      "retrieval head model_hash",
      "bad retrieval head hash should fail",
    );
    cases.push({ name: "bad-retrieval-head-hash", ok: true });

    const badOutputRetrievalMargin = writeCase(root, shared, "bad-output-retrieval-margin");
    const badOutputRetrievalMarginPath = path.join(root, "bad-output-retrieval-margin.json");
    const weakMarginHead = JSON.parse(fs.readFileSync(shared.retrievalHeadPath, "utf8"));
    weakMarginHead.image_head.biases[1] = weakMarginHead.image_head.biases[0];
    delete weakMarginHead.model_hash;
    weakMarginHead.model_hash = fnv64TextHex(JSON.stringify(weakMarginHead));
    fs.writeFileSync(badOutputRetrievalMarginPath, `${JSON.stringify(weakMarginHead, null, 2)}\n`, "utf8");
    const badOutputRetrievalMarginResult = runBridge(
      { ...shared, retrievalHeadPath: badOutputRetrievalMarginPath },
      badOutputRetrievalMargin,
    );
    assertFailure(
      badOutputRetrievalMarginResult,
      "output retrieval margin 0 < 1",
      "bad output retrieval margin should fail",
    );
    cases.push({ name: "bad-output-retrieval-margin", ok: true });

    const badUniqueTargets = writeCase(root, shared, "bad-unique-targets");
    const badUniqueTargetsResult = runBridge(shared, badUniqueTargets, ["--min-unique-targets", "2"]);
    assertFailure(
      badUniqueTargetsResult,
      "denoise bridge unique targets 1 < 2",
      "bad unique target coverage should fail",
    );
    cases.push({ name: "bad-unique-targets", ok: true });

    completed = true;
    console.log(JSON.stringify({
      schema: "nsrl.solomon_attention_denoise_bridge_self_test.v1",
      ok: true,
      cases,
    }, null, 2));
  } finally {
    if (completed && !config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    } else {
      console.error(`fixture_dir: ${root}`);
    }
  }
}

function writeSharedFixture(root) {
  const dataDir = path.join(root, "data");
  fs.mkdirSync(dataDir, { recursive: true });
  const plan = fixturePlan();
  const denoiseModelPath = path.join(dataDir, "denoiser.nsrltch");
  fs.writeFileSync(denoiseModelPath, "fixture denoiser\n", "utf8");
  const textIndexPath = path.join(dataDir, "text-index.tsv");
  fs.writeFileSync(
    textIndexPath,
    [
      "number\tprimary_name\taliases\tsignature_16x16",
      `1\tBael\t\t${Array.from(plan).join(",")}`,
      "",
    ].join("\n"),
    "utf8",
  );
  const retrievalHeadPath = path.join(dataDir, "retrieval-head.json");
  writeRetrievalHead(retrievalHeadPath);
  return { denoiseModelPath, plan, retrievalHeadPath, textIndexPath };
}

function writeCase(root, shared, name, mutate = null) {
  const caseDir = path.join(root, name);
  const sampleDir = path.join(caseDir, "attention");
  const denoiseDir = path.join(caseDir, "denoise");
  fs.mkdirSync(sampleDir, { recursive: true });
  fs.mkdirSync(denoiseDir, { recursive: true });

  const planPath = path.join(sampleDir, "image.ink16.u8");
  fs.writeFileSync(planPath, shared.plan);
  fs.writeFileSync(
    path.join(sampleDir, "sample.json"),
    `${JSON.stringify({
      schema: "nsrl.solomon_attention_sample_trace.v1",
      prompt: "Bael seal",
      generated_text: "Bael",
      image_ink16_u8: planPath,
    }, null, 2)}\n`,
    "utf8",
  );

  const rawPath = path.join(denoiseDir, "samples.ink128.u8");
  fs.writeFileSync(rawPath, expandPlanToRaw(shared.plan));
  const previewPath = path.join(denoiseDir, "preview.pgm");
  fs.writeFileSync(previewPath, "P5\n1 1\n255\n\0", "binary");
  const trace = {
    schema: "nsrl.bitmap_sampler_trace.v1",
    model_format: "NSRLTCH",
    model: shared.denoiseModelPath,
    image_size: imageSize,
    feature_channels: 30,
    selected_count: 1,
    samples: 1,
    latent_prompt: "Bael seal",
    latent_target_source: "attention-plan",
    latent_target_plan: planPath,
    latent_target_signature: Array.from(shared.plan),
    raw_samples: rawPath,
    preview_pgm: previewPath,
    selected_min_text_distance: 0,
    selected_min_score: 0,
  };

  if (mutate) {
    mutate({ denoiseDir, planPath, previewPath, rawPath, sampleDir, trace });
  }
  fs.writeFileSync(path.join(denoiseDir, "trace.json"), `${JSON.stringify(trace, null, 2)}\n`, "utf8");
  return { denoiseDir, sampleDir };
}

function fixturePlan() {
  const plan = Buffer.alloc(signatureBins);
  for (let y = 0; y < signatureGrid; y += 1) {
    for (let x = 0; x < signatureGrid; x += 1) {
      const value = x === y || x + y === signatureGrid - 1
        ? 224
        : x === 8 || y === 8
          ? 96
          : 0;
      plan[y * signatureGrid + x] = value;
    }
  }
  return plan;
}

function expandPlanToRaw(plan) {
  const raw = Buffer.alloc(imageSize * imageSize);
  const cell = imageSize / signatureGrid;
  for (let y = 0; y < imageSize; y += 1) {
    const sy = Math.floor(y / cell);
    for (let x = 0; x < imageSize; x += 1) {
      const sx = Math.floor(x / cell);
      raw[y * imageSize + x] = plan[sy * signatureGrid + sx];
    }
  }
  return raw;
}

function writeRetrievalHead(filePath) {
  const labels = Array.from({ length: 72 }, (_, index) => ({
    label: index,
    spirit_id: index + 1,
    primary_name: index === 0 ? "Bael" : `Spirit ${index + 1}`,
  }));
  const model = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    feature_count: 1,
    labels,
    image_head: {
      biases: labels.map((label) => (label.spirit_id === 1 ? 1000 : 0)),
      weights: labels.map(() => []),
    },
  };
  model.model_hash = fnv64TextHex(JSON.stringify(model));
  fs.writeFileSync(filePath, `${JSON.stringify(model, null, 2)}\n`, "utf8");
}

function fnv64TextHex(value) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function runBridge(shared, fixture, extraArgs = []) {
  return childProcess.spawnSync(
    process.execPath,
    [
      "scripts/check-solomon-attention-denoise-bridge.mjs",
      "--pair",
      `${fixture.sampleDir}:${fixture.denoiseDir}`,
      "--text-index",
      shared.textIndexPath,
      "--retrieval-head",
      shared.retrievalHeadPath,
      "--require-retrieval-head",
      "--max-output-signature-distance",
      "0",
      "--min-output-ink-range",
      "1",
      "--max-output-retrieval-rank",
      "1",
      "--min-output-retrieval-margin",
      "1",
      ...extraArgs,
    ],
    { cwd: repoRoot, encoding: "utf8" },
  );
}

function assertFailure(result, expectedText, message) {
  if (result.status === 0) {
    throw new Error(`${message}: command unexpectedly passed`);
  }
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  if (!output.includes(expectedText)) {
    throw new Error(`${message}: expected ${JSON.stringify(expectedText)}, got:\n${output}`);
  }
}

function assertStatus(result, expectedStatus, message) {
  if (result.status !== expectedStatus) {
    throw new Error(`${message}: status ${result.status}\nstdout:\n${result.stdout || ""}\nstderr:\n${result.stderr || ""}`);
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error && error.stack ? error.stack : String(error));
  process.exit(1);
}
