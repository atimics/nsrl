#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const grid = 16;
const bins = grid * grid;

function usage() {
  console.log([
    "Usage: check-solomon-attention-sample-binding-self-test.mjs [--keep]",
    "",
    "Builds tiny Solomon attention sample fixtures and proves the sample-binding",
    "checker accepts aligned generated image/text identity while rejecting wrong",
    "generated text, wrong generated image identity, and cleanup trace fields.",
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
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-sample-binding-self-test-"));
  let completed = false;
  try {
    const shared = writeSharedFixture(root);
    const cases = [];

    const good = writeSample(root, shared, "good", {
      prompt: "Bael seal",
      generatedText: "Bael",
      signature: shared.baelSignature,
    });
    const goodBinding = runSampleBinding(shared, [good.sampleDir]);
    assertStatus(goodBinding, 0, "aligned generated sample should pass binding");
    const goodBindingReport = JSON.parse(goodBinding.stdout);
    assertEqual(goodBindingReport.ok, true, "aligned binding report ok");
    assertEqual(goodBindingReport.image_to_text_identification, true, "aligned image identity");
    assertEqual(goodBindingReport.generated_text_identification, true, "aligned generated text identity");
    assertEqual(goodBindingReport.generated_text_image_agreement, true, "aligned generated text/image agreement");
    if (!(Number(goodBindingReport.min_signature_margin || 0) > 0)) {
      throw new Error("aligned binding signature margin should be positive");
    }
    if (!(Number(goodBindingReport.min_retrieval_image_margin || 0) > 0)) {
      throw new Error("aligned binding retrieval image margin should be positive");
    }
    if (!(Number(goodBindingReport.min_generated_text_margin || 0) > 0)) {
      throw new Error("aligned generated text margin should be positive");
    }
    const goodIntegrity = runGenerationIntegrity([good.sampleDir]);
    assertStatus(goodIntegrity, 0, "aligned generated sample should pass integrity");
    const goodIntegrityReport = JSON.parse(goodIntegrity.stdout);
    assertEqual(goodIntegrityReport.ok, true, "aligned integrity report ok");
    cases.push({ name: "good", ok: true });

    const badText = writeSample(root, shared, "bad-generated-text", {
      prompt: "Bael seal",
      generatedText: "Stolas",
      signature: shared.baelSignature,
    });
    assertFailure(
      runSampleBinding(shared, [badText.sampleDir]),
      "generated text identity",
      "wrong generated text should fail binding",
    );
    cases.push({ name: "bad-generated-text", ok: true });

    const badImage = writeSample(root, shared, "bad-generated-image", {
      prompt: "Stolas seal",
      generatedText: "Stolas",
      signature: shared.baelSignature,
    });
    assertFailure(
      runSampleBinding(shared, [badImage.sampleDir]),
      "signature rank",
      "wrong generated image should fail binding",
    );
    cases.push({ name: "bad-generated-image", ok: true });

    const badCleanup = writeSample(root, shared, "bad-cleanup", {
      prompt: "Bael seal",
      generatedText: "Bael",
      signature: shared.baelSignature,
      mutateTrace: (trace) => {
        trace.display_cleanup = "postprocess generated seal";
      },
    });
    assertFailure(
      runGenerationIntegrity([badCleanup.sampleDir]),
      "display_cleanup",
      "cleanup trace should fail integrity",
    );
    cases.push({ name: "bad-cleanup", ok: true });

    completed = true;
    console.log(JSON.stringify({
      schema: "nsrl.solomon_attention_sample_binding_self_test.v1",
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
  const baelSignature = diagonalSignature();
  const stolasSignature = borderSignature();
  const textIndexPath = path.join(dataDir, "text-index.tsv");
  fs.writeFileSync(
    textIndexPath,
    [
      "number\tprimary_name\taliases\tsignature_16x16",
      `1\tBael\t\t${Array.from(baelSignature).join(",")}`,
      `36\tStolas\t\t${Array.from(stolasSignature).join(",")}`,
      "",
    ].join("\n"),
    "utf8",
  );
  const retrievalHeadPath = path.join(dataDir, "retrieval-head.json");
  writeRetrievalHead(retrievalHeadPath);
  return { baelSignature, retrievalHeadPath, stolasSignature, textIndexPath };
}

function writeSample(root, shared, name, options) {
  const sampleDir = path.join(root, name);
  fs.mkdirSync(sampleDir, { recursive: true });
  const imagePath = path.join(sampleDir, "image.ink16.u8");
  fs.writeFileSync(imagePath, options.signature);
  const trace = {
    schema: "nsrl.solomon_attention_sample_trace.v1",
    prompt: options.prompt,
    generated_text: options.generatedText,
    text_prior_source: "embedded_lm",
    image_prior_source: "embedded",
    conditioning_primary_name: options.prompt.split(/\s+/)[0] || "",
    image_ink16_u8: "image.ink16.u8",
  };
  if (options.mutateTrace) {
    options.mutateTrace(trace);
  }
  fs.writeFileSync(path.join(sampleDir, "sample.json"), `${JSON.stringify(trace, null, 2)}\n`, "utf8");
  return { sampleDir, shared };
}

function diagonalSignature() {
  const signature = Buffer.alloc(bins);
  for (let y = 0; y < grid; y += 1) {
    for (let x = 0; x < grid; x += 1) {
      signature[y * grid + x] = x === y || x + y === grid - 1 ? 224 : 0;
    }
  }
  return signature;
}

function borderSignature() {
  const signature = Buffer.alloc(bins);
  for (let y = 0; y < grid; y += 1) {
    for (let x = 0; x < grid; x += 1) {
      signature[y * grid + x] = x === 0 || y === 0 || x === grid - 1 || y === grid - 1 ? 192 : 0;
    }
  }
  return signature;
}

function writeRetrievalHead(filePath) {
  const labels = Array.from({ length: 72 }, (_, index) => {
    const spiritId = index + 1;
    return {
      label: index,
      spirit_id: spiritId,
      primary_name: spiritId === 1 ? "Bael" : spiritId === 36 ? "Stolas" : `Spirit ${spiritId}`,
      aliases: [],
    };
  });
  const model = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    model_hash: "fixture-sample-binding-retrieval-head",
    feature_count: 1,
    identity_anchor: {
      leading_boost: 100000,
      mention_boost: 80000,
    },
    labels,
    text_head: {
      biases: labels.map(() => 0),
      weights: labels.map(() => []),
    },
    image_head: {
      biases: labels.map((label) => (label.spirit_id === 1 ? 1000 : 0)),
      weights: labels.map(() => []),
    },
  };
  fs.writeFileSync(filePath, `${JSON.stringify(model, null, 2)}\n`, "utf8");
}

function runSampleBinding(shared, sampleDirs) {
  const args = [
    "scripts/check-solomon-attention-sample-binding.mjs",
    "--text-index",
    shared.textIndexPath,
    "--retrieval-head",
    shared.retrievalHeadPath,
    "--require-retrieval-head",
    "--max-signature-rank",
    "1",
    "--max-retrieval-rank",
    "1",
    "--max-text-rank",
    "1",
    "--min-signature-margin",
    "1",
    "--min-retrieval-margin",
    "1",
    "--min-text-margin",
    "1",
  ];
  for (const sampleDir of sampleDirs) {
    args.push("--sample-dir", sampleDir);
  }
  return childProcess.spawnSync(process.execPath, args, { cwd: repoRoot, encoding: "utf8" });
}

function runGenerationIntegrity(sampleDirs) {
  const args = ["scripts/check-solomon-generation-integrity.mjs"];
  for (const sampleDir of sampleDirs) {
    args.push("--sample-dir", sampleDir);
  }
  return childProcess.spawnSync(process.execPath, args, { cwd: repoRoot, encoding: "utf8" });
}

function assertStatus(result, expectedStatus, message) {
  if (result.status !== expectedStatus) {
    throw new Error([
      `${message}: expected status ${expectedStatus}, got ${result.status}`,
      `stdout:\n${result.stdout || ""}`,
      `stderr:\n${result.stderr || ""}`,
    ].join("\n"));
  }
}

function assertFailure(result, expectedText, message) {
  if (result.status === 0) {
    throw new Error([
      `${message}: command unexpectedly passed`,
      `stdout:\n${result.stdout || ""}`,
      `stderr:\n${result.stderr || ""}`,
    ].join("\n"));
  }
  const combined = `${result.stdout || ""}\n${result.stderr || ""}`;
  if (!combined.includes(expectedText)) {
    throw new Error([
      `${message}: expected failure to mention ${JSON.stringify(expectedText)}`,
      `stdout:\n${result.stdout || ""}`,
      `stderr:\n${result.stderr || ""}`,
    ].join("\n"));
  }
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(`${message}: ${JSON.stringify(actual)} !== ${JSON.stringify(expected)}`);
  }
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
