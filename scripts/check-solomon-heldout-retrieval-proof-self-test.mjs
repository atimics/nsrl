#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_heldout_retrieval_proof_self_test.v1";
const labelCount = 72;
const featureCount = 8;
const imageTaskCounts = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};

function usage() {
  console.log([
    "Usage: check-solomon-heldout-retrieval-proof-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds synthetic held-out retrieval/provenance artifacts and proves the",
    "retrieval-head provenance checker rejects stale prompts, row-count drift,",
    "weak held-out top-1/margin evidence, missing image heads, and stale hashes.",
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

function writeJsonl(filePath, rows) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
}

function metric(count, overrides = {}) {
  return {
    count,
    top1: count,
    top5: count,
    top1_per_mille: 1000,
    top5_per_mille: 1000,
    min_margin: 10,
    mean_margin: 12,
    ...overrides,
  };
}

function writeFixture(root, name, mutate = () => {}) {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  const examplesPath = path.join(dir, "examples.jsonl");
  const tokensPath = path.join(dir, "corpus.tokens.u8");
  const promptsPath = path.join(dir, "prompts-expanded.jsonl");
  const modelPath = path.join(dir, "retrieval-head.json");
  const evalPath = path.join(dir, "retrieval-head-eval.json");

  const exampleRows = [
    { task: "identify", spirit_id: 1, prompt: "Bael", token_offset: 0, token_count: 0 },
  ];
  for (const [task, count] of Object.entries(imageTaskCounts)) {
    for (let index = 0; index < count; index += 1) {
      exampleRows.push({
        task,
        spirit_id: (index % labelCount) + 1,
        token_offset: 0,
        token_count: 0,
      });
    }
  }
  writeJsonl(examplesPath, exampleRows);
  fs.writeFileSync(tokensPath, Buffer.from([1, 2, 3, 4, 5]));
  writeJsonl(promptsPath, Array.from({ length: labelCount }, (_, index) => ({
    spirit_id: index + 1,
    text: `held-out paraphrase ${index + 1}`,
    source: "expanded",
    tier: "tier-novel-vocab",
  })));

  const hashes = {
    examples: fnv64FileHex(examplesPath),
    tokens: fnv64FileHex(tokensPath),
    prompts: fnv64FileHex(promptsPath),
  };
  const model = modelJson({ examplesPath, tokensPath, hashes });
  const evalTrace = evalJson({ examplesPath, tokensPath, promptsPath, modelPath, hashes, model });
  mutate({ model, evalTrace, hashes, paths: { examplesPath, tokensPath, promptsPath, modelPath, evalPath } });
  if (!model.model_hash || model.__recompute_hash === true) {
    delete model.__recompute_hash;
    model.model_hash = retrievalHeadHash(model);
  }
  if (evalTrace.__sync_model_hash !== false) {
    evalTrace.model_hash = model.model_hash;
  }
  delete evalTrace.__sync_model_hash;
  writeJson(modelPath, model);
  writeJson(evalPath, evalTrace);
  return { dir, examplesPath, tokensPath, promptsPath, modelPath, evalPath };
}

function modelJson({ examplesPath, tokensPath, hashes }) {
  const labels = Array.from({ length: labelCount }, (_, index) => ({
    label: index,
    spirit_id: index + 1,
    primary_name: `Spirit ${index + 1}`,
    aliases: [`Alias ${index + 1}`],
  }));
  const model = {
    schema: "nsrl.solomon_v2_retrieval_head.v1",
    corpus: {
      examples: examplesPath,
      examples_hash: hashes.examples,
      tokens: tokensPath,
      tokens_hash: hashes.tokens,
    },
    feature_count: featureCount,
    labels,
    identity_anchor: {
      leading_boost: 1,
      mention_boost: 1,
      seal_id_templates: ["seal id {n}", "spirit {n}", "goetic spirit {n}"],
    },
    text_head: componentHead(1),
    image_head: componentHead(2),
    training: {
      text_rows: labelCount,
      image_rows: labelCount,
      epochs: 1,
      seed: "heldout-self-test",
      text_mistakes: 0,
      image_mistakes: 0,
      text_nonzero_weights: labelCount,
      image_nonzero_weights: labelCount,
    },
    eval: {},
  };
  model.model_hash = retrievalHeadHash(model);
  return model;
}

function componentHead(multiplier) {
  return {
    biases: Array.from({ length: labelCount }, () => 0),
    weights: Array.from({ length: labelCount }, (_, index) => [[index % featureCount, multiplier]]),
  };
}

function evalJson({ examplesPath, tokensPath, promptsPath, modelPath, hashes, model }) {
  const imageTasks = Object.fromEntries(
    Object.entries(imageTaskCounts).map(([task, count]) => [task, metric(count)]),
  );
  return {
    schema: "nsrl.solomon_v2_retrieval_head_eval.v1",
    ok: true,
    errors: [],
    model: modelPath,
    model_hash: model.model_hash,
    feature_count: featureCount,
    examples: examplesPath,
    examples_hash: hashes.examples,
    tokens: tokensPath,
    tokens_hash: hashes.tokens,
    prompts: promptsPath,
    prompts_hash: hashes.prompts,
    heldout_prompt_rows: 72,
    heldout_prompt_unique_targets: 72,
    heldout_prompts: metric(72),
    known_prompts: metric(72),
    identity_bindings: {
      total: metric(504),
      by_kind: {
        "primary-name": metric(72),
        "primary-seal": metric(72),
        alias: metric(72),
        "alias-seal": metric(72),
        "seal-id": metric(216),
      },
    },
    image_to_text: metric(288),
    image_tasks: imageTasks,
    match: {
      yes: metric(72),
      no: metric(144),
      no_by_role: {
        image: metric(72),
        prompt: metric(72),
      },
    },
  };
}

function runProvenance(fixture) {
  return childProcess.spawnSync(process.execPath, [
    "scripts/check-solomon-v2-retrieval-head-provenance.mjs",
    "--eval",
    fixture.evalPath,
    "--retrieval-head",
    fixture.modelPath,
    "--examples",
    fixture.examplesPath,
    "--tokens",
    fixture.tokensPath,
    "--prompts",
    fixture.promptsPath,
    "--expect-spirits",
    String(labelCount),
    "--min-feature-count",
    "1",
    "--min-retrieval-margin",
    "1",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function readReport(stdout) {
  const start = String(stdout || "").indexOf("{");
  if (start < 0) return null;
  return JSON.parse(String(stdout).slice(start));
}

function caseResult(definition, result, report) {
  const actualOk = result.status === 0 && report?.ok === true;
  const haystack = [
    ...(report?.errors || []),
    result.stdout || "",
    result.stderr || "",
  ].join("\n");
  const requiredErrorOk = definition.requiredError
    ? haystack.includes(definition.requiredError)
    : true;
  return {
    name: definition.name,
    expect_ok: definition.expectOk,
    ok: actualOk === definition.expectOk && requiredErrorOk,
    status: result.status,
    passed: report?.ok === true,
    required_error: definition.requiredError || "",
    errors: report?.errors || [],
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

function retrievalHeadHash(model) {
  const copy = { ...model };
  delete copy.model_hash;
  delete copy.__recompute_hash;
  return fnv64Hex(JSON.stringify(copy));
}

function fnv64Hex(value) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64BytesHex(bytes) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64FileHex(filePath) {
  return fnv64BytesHex(fs.readFileSync(filePath));
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-heldout-retrieval-self-test-"));
  const cases = [];
  try {
    const definitions = [
      { name: "good", expectOk: true, mutate: () => {} },
      {
        name: "bad-prompts-hash",
        expectOk: false,
        requiredError: "retrieval head eval prompts_hash",
        mutate: ({ evalTrace }) => {
          evalTrace.prompts_hash = "0x0000000000000000";
        },
      },
      {
        name: "bad-heldout-row-count",
        expectOk: false,
        requiredError: "retrieval head held-out prompt rows 71 != eligible prompt file rows 72",
        mutate: ({ evalTrace }) => {
          evalTrace.heldout_prompt_rows = 71;
        },
      },
      {
        name: "bad-heldout-top1",
        expectOk: false,
        requiredError: "held-out prompts top1 71 != count 72",
        mutate: ({ evalTrace }) => {
          evalTrace.heldout_prompts.top1 = 71;
        },
      },
      {
        name: "bad-heldout-margin",
        expectOk: false,
        requiredError: "held-out prompts min_margin 0 < 1",
        mutate: ({ evalTrace }) => {
          evalTrace.heldout_prompts.min_margin = 0;
        },
      },
      {
        name: "bad-missing-image-head",
        expectOk: false,
        requiredError: "retrieval head artifact image_head has no nonzero weights",
        mutate: ({ model }) => {
          model.image_head = {
            biases: Array.from({ length: labelCount }, () => 0),
            weights: Array.from({ length: labelCount }, () => []),
          };
          model.__recompute_hash = true;
        },
      },
      {
        name: "bad-stale-model-hash",
        expectOk: false,
        requiredError: "retrieval head artifact model_hash",
        mutate: ({ model, evalTrace }) => {
          model.model_hash = "0x0000000000000000";
          evalTrace.__sync_model_hash = false;
          evalTrace.model_hash = model.model_hash;
        },
      },
    ];
    for (const definition of definitions) {
      const fixture = writeFixture(root, definition.name, definition.mutate);
      const result = runProvenance(fixture);
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
