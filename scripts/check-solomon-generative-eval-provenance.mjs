#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const imageSize = 128;
const imageBytes = imageSize * imageSize;
const signatureBins = 16 * 16;

function usage() {
  console.log(
    [
      "Usage: check-solomon-generative-eval-provenance.mjs [--keep]",
      "",
      "Runs a tiny fixture through run-solomon-generative-eval.mjs and proves",
      "that clean decoded-latent sampler traces can write scored sidecars while",
      "bad raw sample paths, cleanup side channels, missing traces, and empty raw",
      "bytes fail before generated retrieval identity scores are written.",
      "",
      "Options:",
      "  --keep   keep the temporary fixture directory for debugging",
    ].join("\n"),
  );
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
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-generative-eval-provenance-"));
  let completed = false;
  try {
    const fixture = writeFixture(root);
    const cases = [];
    const good = runEval(fixture, "good", "");
    assertStatus(good, 0, "clean fixture should pass");
    const goodRunDir = path.join(fixture.outDir, "good");
    assertExists(path.join(goodRunDir, "samples.tsv"), "clean run should write samples.tsv");
    assertExists(path.join(goodRunDir, "summary.tsv"), "clean run should write summary.tsv");
    assertExists(path.join(goodRunDir, "config.json"), "clean run should write config.json");
    const sample = firstTsvRow(path.join(goodRunDir, "samples.tsv"));
    assertEqual(sample.sampler_target_source, "decoded-latent", "clean run should record decoded latent source");
    assertEqual(sample.generated_retrieval_rank, "1", "clean run should record generated retrieval rank");
    assertEqual(sample.generated_retrieval_identity, "1", "clean run should record generated retrieval identity");
    const goodConfig = JSON.parse(fs.readFileSync(path.join(goodRunDir, "config.json"), "utf8"));
    assertEqual(
      goodConfig.retrievalHeadModelHash,
      "fixture-solomon-retrieval-head",
      "clean run should record retrieval head model hash",
    );
    assertEqual(goodConfig.selectedPromptEligibleRows, 1, "clean run should record held-out prompt row count");
    assertEqual(goodConfig.selectedPromptUniqueTargets, 1, "clean run should record unique selected target count");
    assertEqual(
      goodConfig.selectedPromptEligibleUniqueTargets,
      1,
      "clean run should record held-out unique target count",
    );
    assertEqual(
      goodConfig.selectedPromptTiers["tier-novel-vocab"],
      1,
      "clean run should record selected prompt tier counts",
    );
    assertEqual(
      goodConfig.selectedPromptSources.generated,
      1,
      "clean run should record selected prompt source counts",
    );
    const goodSummary = firstTsvRow(path.join(goodRunDir, "summary.tsv"));
    assertLatentModelProvenance(
      goodSummary,
      goodConfig,
      fixture.latentModelPath,
      "tiny",
      "clean run",
    );
    cases.push({ name: "good", ok: true, run_dir: goodRunDir });

    const posthoc = runEval(fixture, "posthoc", "", { retrievalHead: false });
    assertStatus(posthoc, 0, "post-hoc fixture should pass before retrieval scoring");
    const posthocRunDir = path.join(fixture.outDir, "posthoc");
    const posthocSampleBefore = firstTsvRow(path.join(posthocRunDir, "samples.tsv"));
    assertEqual(posthocSampleBefore.generated_retrieval_rank, "", "post-hoc fixture should start without retrieval rank");
    assertEqual(posthocSampleBefore.generated_retrieval_identity, "", "post-hoc fixture should start without retrieval identity");
    const posthocConfigBefore = JSON.parse(fs.readFileSync(path.join(posthocRunDir, "config.json"), "utf8"));
    assertEqual(
      posthocConfigBefore.retrievalHeadModelHash,
      "",
      "post-hoc fixture should start without retrieval head model hash",
    );
    const posthocScore = runPosthocScore(fixture, posthocRunDir);
    assertStatus(posthocScore, 0, "post-hoc retrieval scoring should pass");
    const posthocSampleAfter = firstTsvRow(path.join(posthocRunDir, "samples.tsv"));
    assertEqual(posthocSampleAfter.generated_retrieval_rank, "1", "post-hoc scoring should write generated retrieval rank");
    assertEqual(posthocSampleAfter.generated_retrieval_identity, "1", "post-hoc scoring should write generated retrieval identity");
    const posthocConfigAfter = JSON.parse(fs.readFileSync(path.join(posthocRunDir, "config.json"), "utf8"));
    assertEqual(
      posthocConfigAfter.retrievalHeadModelHash,
      "fixture-solomon-retrieval-head",
      "post-hoc scoring should record retrieval head model hash",
    );
    const posthocSummaryAfter = firstTsvRow(path.join(posthocRunDir, "summary.tsv"));
    assertLatentModelProvenance(
      posthocSummaryAfter,
      posthocConfigAfter,
      fixture.latentModelPath,
      "tiny",
      "post-hoc run",
    );
    cases.push({ name: "posthoc-score", ok: true, run_dir: posthocRunDir });

    const freeText = runEval(fixture, "free-text-target-word", "free_text_target_word");
    assertStatus(freeText, 0, "free-text prompt fixture should pass");
    const freeTextRunDir = path.join(fixture.outDir, "free-text-target-word");
    const freeTextSample = firstTsvRow(path.join(freeTextRunDir, "samples.tsv"));
    assertEqual(
      freeTextSample.sampler_target_source,
      "decoded-latent",
      "free-text prompt fixture should preserve decoded latent source",
    );
    cases.push({ name: "free-text-target-word", ok: true, mode: "free_text_target_word" });

    const posthocBad = runEval(fixture, "posthoc-bad-raw-path", "", { retrievalHead: false });
    assertStatus(posthocBad, 0, "post-hoc bad raw-path fixture should build before scoring");
    const posthocBadRunDir = path.join(fixture.outDir, "posthoc-bad-raw-path");
    corruptSampleRawPath(posthocBadRunDir);
    const posthocBadScore = runPosthocScore(fixture, posthocBadRunDir);
    assertFailure(
      posthocBadScore,
      "raw_samples must resolve to samples.ink128.u8 in out_dir",
      "post-hoc scorer should fail bad raw sample provenance",
    );
    const posthocBadConfig = JSON.parse(fs.readFileSync(path.join(posthocBadRunDir, "config.json"), "utf8"));
    assertEqual(
      posthocBadConfig.retrievalHeadModelHash,
      "",
      "failed post-hoc scoring should not record retrieval head model hash",
    );
    cases.push({ name: "posthoc-bad-raw-path", ok: true });

    for (const badCase of [
      {
        name: "bad-raw-path",
        mode: "raw_path",
        expected: "raw_samples must resolve to samples.ink128.u8 in out_dir",
      },
      {
        name: "bad-cleanup",
        mode: "cleanup",
        expected: "display_cleanup: forbidden target-pixel, oracle, guidance, or cleanup field",
      },
      {
        name: "bad-missing-trace",
        mode: "missing_trace",
        expected: "missing trace.json",
      },
      {
        name: "bad-empty-raw",
        mode: "empty",
        expected: "byte count 0 is not a positive multiple of 16384",
      },
    ]) {
      const result = runEval(fixture, badCase.name, badCase.mode);
      assertFailure(result, badCase.expected, `${badCase.name} should fail before scoring`);
      assertNoScoredSidecars(path.join(fixture.outDir, badCase.name), badCase.name);
      cases.push({ name: badCase.name, ok: true, mode: badCase.mode });
    }

    completed = true;
    console.log(JSON.stringify({
      schema: "nsrl.solomon_generative_eval_provenance_check.v1",
      ok: true,
      clean_sample: sampleEvidence(sample, goodSummary, goodConfig),
      posthoc_sample: sampleEvidence(posthocSampleAfter, posthocSummaryAfter, posthocConfigAfter),
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

function writeFixture(root) {
  const binDir = path.join(root, "bin");
  const targetReleaseDir = path.join(root, "target", "release");
  const dataDir = path.join(root, "data");
  const outDir = path.join(root, "out");
  fs.mkdirSync(binDir, { recursive: true });
  fs.mkdirSync(targetReleaseDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });
  fs.mkdirSync(outDir, { recursive: true });

  writeExecutable(path.join(binDir, "cargo"), [
    "#!/usr/bin/env sh",
    "exit 0",
    "",
  ].join("\n"));
  writeExecutable(path.join(targetReleaseDir, "nsrl-bitmap-sample"), fakeSamplerSource());

  const promptsPath = path.join(dataDir, "prompts.jsonl");
  fs.writeFileSync(
    promptsPath,
    `${JSON.stringify({
      prompt_hash: "p1",
      spirit_id: 1,
      tier: "tier-novel-vocab",
      source: "generated",
      text: "Bael seal",
      bucket: 0,
    })}\n`,
    "utf8",
  );

  const targetImagePath = path.join(dataDir, "target.ink128.u8");
  fs.writeFileSync(targetImagePath, Buffer.alloc(imageBytes));
  const textIndexPath = path.join(dataDir, "text-index.tsv");
  fs.writeFileSync(
    textIndexPath,
    [
      "number\tprimary_name\tink_128_u8\tsignature_16x16",
      `1\tBael\t${targetImagePath}\t${new Array(signatureBins).fill(0).join(",")}`,
      "",
    ].join("\n"),
    "utf8",
  );

  const latentModelPath = path.join(dataDir, "latent.nsrllat");
  writeLatentModel(latentModelPath);
  const samplerModelPath = path.join(dataDir, "sampler.nsrltch");
  fs.writeFileSync(samplerModelPath, "fixture sampler\n", "utf8");
  const retrievalHeadPath = path.join(dataDir, "retrieval-head.json");
  writeRetrievalHead(retrievalHeadPath);

  return {
    binDir,
    targetDir: path.join(root, "target"),
    outDir,
    promptsPath,
    textIndexPath,
    samplerModelPath,
    latentModelPath,
    retrievalHeadPath,
  };
}

function fakeSamplerSource() {
  return [
    "#!/usr/bin/env node",
    'const fs = require("fs");',
    'const path = require("path");',
    'const imageSize = 128;',
    'let outDir = "";',
    'let samples = 1;',
    "for (let index = 2; index < process.argv.length; index += 1) {",
    '  if (process.argv[index] === "--out-dir") {',
    "    outDir = process.argv[index + 1] || \"\";",
    "    index += 1;",
    '  } else if (process.argv[index] === "--samples") {',
    "    samples = Number(process.argv[index + 1] || 1);",
    "    index += 1;",
    "  }",
    "}",
    'if (!outDir) throw new Error("--out-dir is required");',
    "fs.mkdirSync(outDir, { recursive: true });",
    "const mode = process.env.NSRL_FAKE_SAMPLER_BAD || \"\";",
    'const rawPath = path.join(outDir, "samples.ink128.u8");',
    'const rawBytes = mode === "empty" ? Buffer.alloc(0) : Buffer.alloc(imageSize * imageSize * samples);',
    "fs.writeFileSync(rawPath, rawBytes);",
    'if (mode === "missing_trace") process.exit(0);',
    "const trace = {",
    '  schema: "nsrl.bitmap_sampler_trace.v1",',
    '  latent_target_source: "decoded-latent",',
    "  latent_target_number: 1,",
    '  latent_target_name: "Bael",',
    "  selected_min_text_distance: 0,",
    "  selected_mean_wash_penalty_q8: 0,",
    "  raw_samples: rawPath,",
    '  preview_pgm: path.join(outDir, "preview.pgm"),',
    "};",
    'if (mode === "raw_path") trace.raw_samples = path.join(outDir, "elsewhere.ink128.u8");',
    'if (mode === "cleanup") trace.display_cleanup = "postprocess target bitmap";',
    'if (mode === "free_text_target_word") {',
    '  trace.latent_prompt = "benign prompt text says target in prose";',
    '  trace.latent_target_name = "benign target name in free text";',
    "}",
    'fs.writeFileSync(path.join(outDir, "trace.json"), `${JSON.stringify(trace, null, 2)}\\n`);',
    "",
  ].join("\n");
}

function writeExecutable(filePath, contents) {
  fs.writeFileSync(filePath, contents, { encoding: "utf8", mode: 0o755 });
  fs.chmodSync(filePath, 0o755);
}

function writeLatentModel(filePath) {
  const chunks = [];
  const pushU32 = (value) => {
    const bytes = Buffer.alloc(4);
    bytes.writeUInt32LE(value);
    chunks.push(bytes);
  };
  const pushI16 = (value) => {
    const bytes = Buffer.alloc(2);
    bytes.writeInt16LE(value);
    chunks.push(bytes);
  };
  chunks.push(Buffer.from("NSRLLAT1", "ascii"));
  pushU32(1);
  pushU32(1);
  pushU32(signatureBins);
  pushU32(0);
  pushU32(0);
  pushU32(0);
  pushU32(16);
  chunks.push(Buffer.alloc(1));
  pushI16(0);
  chunks.push(Buffer.alloc(signatureBins));
  chunks.push(Buffer.alloc(2));
  chunks.push(Buffer.alloc(signatureBins));
  for (let index = 0; index < signatureBins; index += 1) {
    pushI16(0);
  }
  fs.writeFileSync(filePath, Buffer.concat(chunks));
}

function writeRetrievalHead(filePath) {
  const labels = Array.from({ length: 72 }, (_, index) => ({
    label: index,
    spirit_id: index + 1,
    primary_name: index === 0 ? "Bael" : `Spirit ${index + 1}`,
  }));
  const model = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    model_hash: "fixture-solomon-retrieval-head",
    feature_count: 1,
    labels,
    image_head: {
      biases: labels.map((label) => (label.spirit_id === 1 ? 1000 : 0)),
      weights: labels.map(() => []),
    },
  };
  fs.writeFileSync(filePath, `${JSON.stringify(model, null, 2)}\n`, "utf8");
}

function runEval(fixture, runName, mode, options = {}) {
  const env = {
    ...process.env,
    PATH: `${fixture.binDir}${path.delimiter}${process.env.PATH || ""}`,
    CARGO_TARGET_DIR: fixture.targetDir,
  };
  if (mode) {
    env.NSRL_FAKE_SAMPLER_BAD = mode;
  } else {
    delete env.NSRL_FAKE_SAMPLER_BAD;
  }
  const args = [
    "scripts/run-solomon-generative-eval.mjs",
    "--prompts",
    fixture.promptsPath,
    "--text-index",
    fixture.textIndexPath,
    "--sampler-model",
    fixture.samplerModelPath,
    "--out-dir",
    fixture.outDir,
    "--run-name",
    runName,
    "--partition",
    "eval",
    "--limit",
    "1",
    "--latent-model",
    `tiny=${fixture.latentModelPath}`,
    "--samples",
    "1",
    "--candidate-multiplier",
    "1",
    "--passes",
    "1",
  ];
  if (options.retrievalHead !== false) {
    args.splice(7, 0, "--retrieval-head", fixture.retrievalHeadPath);
  }
  return spawnSync(
    process.execPath,
    args,
    {
      cwd: repoRoot,
      env,
      encoding: "utf8",
    },
  );
}

function runPosthocScore(fixture, runDir) {
  return spawnSync(
    process.execPath,
    [
      "scripts/score-solomon-generative-eval-retrieval.mjs",
      "--generative-eval",
      runDir,
      "--retrieval-head",
      fixture.retrievalHeadPath,
    ],
    {
      cwd: repoRoot,
      encoding: "utf8",
    },
  );
}

function corruptSampleRawPath(runDir) {
  const sample = firstTsvRow(path.join(runDir, "samples.tsv"));
  const tracePath = path.join(sample.out_dir, "trace.json");
  const trace = JSON.parse(fs.readFileSync(tracePath, "utf8"));
  trace.raw_samples = path.join(sample.out_dir, "elsewhere.ink128.u8");
  fs.writeFileSync(tracePath, `${JSON.stringify(trace, null, 2)}\n`, "utf8");
}

function firstTsvRow(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").trimEnd().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error(`${filePath} has no data rows`);
  }
  const header = lines[0].split("\t");
  const values = lines[1].split("\t");
  return Object.fromEntries(header.map((key, index) => [key, values[index] || ""]));
}

function sampleEvidence(sample, summary, config) {
  const model = summary.model || "";
  const provenance = latentModelProvenance(config, model);
  return {
    sampler_target_source: sample.sampler_target_source || "",
    latent_model: summary.latent_model || "",
    latent_model_hash: summary.latent_model_hash || "",
    latent_model_config_hash: config.latentModelHashes?.[model] || "",
    latent_model_provenance_hash: provenance?.modelHash || "",
    latent_model_provenance_path: provenance?.path || "",
    generated_retrieval_rank: Number(sample.generated_retrieval_rank || 0),
    generated_retrieval_identity: Number(sample.generated_retrieval_identity || 0),
    mean_generated_retrieval_rank_q8: Number(summary.mean_generated_retrieval_rank_q8 || 0),
    generated_retrieval_top1_per_mille: Number(summary.generated_retrieval_top1_per_mille || 0),
    generated_retrieval_top5_per_mille: Number(summary.generated_retrieval_top5_per_mille || 0),
    retrieval_head_model_hash: config.retrievalHeadModelHash || "",
    selected_prompt_rows: Number(config.selectedPromptRows || 0),
    selected_prompt_eligible_rows: Number(config.selectedPromptEligibleRows || 0),
    selected_prompt_unique_targets: Number(config.selectedPromptUniqueTargets || 0),
    selected_prompt_eligible_unique_targets: Number(config.selectedPromptEligibleUniqueTargets || 0),
    selected_prompt_sources: config.selectedPromptSources || {},
    selected_prompt_tiers: config.selectedPromptTiers || {},
  };
}

function assertLatentModelProvenance(summary, config, expectedPath, expectedLabel, label) {
  const hash = summary.latent_model_hash || "";
  if (!hash) {
    throw new Error(`${label} summary should record latent_model_hash`);
  }
  assertEqual(summary.model, expectedLabel, `${label} summary should record latent label`);
  assertEqual(summary.latent_model, expectedPath, `${label} summary should record latent model path`);
  assertEqual(
    config.latentModelHashes?.[expectedLabel] || "",
    hash,
    `${label} config latentModelHashes should match summary latent_model_hash`,
  );
  const provenance = latentModelProvenance(config, expectedLabel);
  if (!provenance) {
    throw new Error(`${label} config should record latentModelProvenance for ${expectedLabel}`);
  }
  assertEqual(provenance.path, expectedPath, `${label} latentModelProvenance path should match summary`);
  assertEqual(provenance.modelHash, hash, `${label} latentModelProvenance hash should match summary`);
}

function latentModelProvenance(config, label) {
  if (!Array.isArray(config.latentModelProvenance)) {
    return null;
  }
  return config.latentModelProvenance.find((item) => item?.label === label) || null;
}

function assertFailure(result, expectedText, message) {
  if (result.status === 0) {
    throw new Error(`${message}: command unexpectedly passed`);
  }
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  if (!output.includes(expectedText)) {
    throw new Error(`${message}: expected output to include ${JSON.stringify(expectedText)}, got:\n${output}`);
  }
}

function assertStatus(result, expectedStatus, message) {
  if (result.status !== expectedStatus) {
    throw new Error(`${message}: status ${result.status}, stderr:\n${result.stderr || ""}`);
  }
}

function assertNoScoredSidecars(runDir, label) {
  for (const fileName of ["samples.tsv", "summary.tsv", "config.json"]) {
    const filePath = path.join(runDir, fileName);
    if (fs.existsSync(filePath)) {
      throw new Error(`${label} should not write ${fileName}`);
    }
  }
}

function assertExists(filePath, message) {
  if (!fs.existsSync(filePath)) {
    throw new Error(message);
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
