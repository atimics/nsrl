#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_heldout_retrieval_proof.v1";

const defaults = {
  outDir: "",
  keepTemp: false,
  textIndex: "web/assets/solomon-spirit-text-signatures.tsv",
  prompts: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  featureCount: "4096",
  epochs: "4",
  minHeldoutPromptRows: "72",
  minHeldoutTop1PerMille: "1000",
  minHeldoutTop5PerMille: "1000",
  minRetrievalMargin: "1",
};

function usage() {
  console.log([
    "Usage: check-solomon-heldout-retrieval-proof.mjs [options]",
    "",
    "Builds the real Solomon v2 symbolic corpus, trains the tiny integer 72-way",
    "retrieval/class head, and proves the checked-in held-out prompt panel binds",
    "top-1/top-5 with provenance tied to the exact corpus and prompt bytes.",
    "",
    "Options:",
    "  --out-dir PATH",
    "  --keep-temp",
    "  --text-index PATH",
    "  --prompts PATH",
    "  --feature-count N",
    "  --epochs N",
    "  --min-heldout-prompt-rows N",
    "  --min-heldout-top1-per-mille N",
    "  --min-heldout-top5-per-mille N",
    "  --min-retrieval-margin N",
  ].join("\n"));
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--keep-temp") {
      config.keepTemp = true;
    } else if (arg === "--text-index") {
      config.textIndex = requireValue(argv, ++index, arg);
    } else if (arg === "--prompts") {
      config.prompts = requireValue(argv, ++index, arg);
    } else if (arg === "--feature-count") {
      config.featureCount = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--epochs") {
      config.epochs = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heldout-prompt-rows") {
      config.minHeldoutPromptRows = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heldout-top1-per-mille") {
      config.minHeldoutTop1PerMille = parsePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-heldout-top5-per-mille") {
      config.minHeldoutTop5PerMille = parsePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-retrieval-margin") {
      config.minRetrievalMargin = parseNonNegativeInteger(requireValue(argv, ++index, arg), arg);
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

function parsePositiveInteger(value, flag) {
  if (!/^[1-9][0-9]*$/.test(String(value))) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return String(value);
}

function parseNonNegativeInteger(value, flag) {
  if (!/^[0-9]+$/.test(String(value))) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return String(value);
}

function parsePerMille(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 1000) {
    throw new Error(`${flag} requires an integer from 0 to 1000`);
  }
  return String(parsed);
}

function run(label, command, args) {
  const result = childProcess.spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error([
      `${label} failed with status ${result.status}`,
      `command: ${[command, ...args].join(" ")}`,
      `stdout:\n${result.stdout || ""}`,
      `stderr:\n${result.stderr || ""}`,
    ].join("\n"));
  }
  return result;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function assertHeldoutProof(evalTrace, model, provenance, config) {
  const errors = [];
  if (evalTrace.schema !== "nsrl.solomon_v2_retrieval_head_eval.v1") {
    errors.push(`retrieval eval schema ${JSON.stringify(evalTrace.schema || "")} != nsrl.solomon_v2_retrieval_head_eval.v1`);
  }
  if (evalTrace.ok !== true) {
    errors.push("retrieval eval ok is not true");
  }
  if (model.schema !== "nsrl.solomon_v2_retrieval_head.v1") {
    errors.push(`retrieval model schema ${JSON.stringify(model.schema || "")} != nsrl.solomon_v2_retrieval_head.v1`);
  }
  if (Array.isArray(model.labels) ? model.labels.length !== 72 : true) {
    errors.push(`retrieval model labels ${Array.isArray(model.labels) ? model.labels.length : 0} != 72`);
  }
  if (model.model_hash !== evalTrace.model_hash) {
    errors.push(`retrieval model hash ${model.model_hash || ""} != eval hash ${evalTrace.model_hash || ""}`);
  }
  const heldout = evalTrace.heldout_prompts || {};
  const heldoutRows = Number(evalTrace.heldout_prompt_rows || heldout.count || 0);
  const heldoutTargets = Number(evalTrace.heldout_prompt_unique_targets || 0);
  if (heldoutRows < Number(config.minHeldoutPromptRows)) {
    errors.push(`held-out rows ${heldoutRows} < ${config.minHeldoutPromptRows}`);
  }
  if (heldoutTargets !== 72) {
    errors.push(`held-out unique targets ${heldoutTargets} != 72`);
  }
  if (Number(heldout.top1_per_mille || 0) < Number(config.minHeldoutTop1PerMille)) {
    errors.push(`held-out top1 ${heldout.top1_per_mille || 0} < ${config.minHeldoutTop1PerMille}`);
  }
  if (Number(heldout.top5_per_mille || 0) < Number(config.minHeldoutTop5PerMille)) {
    errors.push(`held-out top5 ${heldout.top5_per_mille || 0} < ${config.minHeldoutTop5PerMille}`);
  }
  if (Number(heldout.min_margin ?? Number.MIN_SAFE_INTEGER) < Number(config.minRetrievalMargin)) {
    errors.push(`held-out min_margin ${heldout.min_margin ?? ""} < ${config.minRetrievalMargin}`);
  }
  if (provenance.schema !== "nsrl.solomon_v2_retrieval_head_provenance_check.v1") {
    errors.push(`provenance schema ${JSON.stringify(provenance.schema || "")} != nsrl.solomon_v2_retrieval_head_provenance_check.v1`);
  }
  if (provenance.ok !== true) {
    errors.push("retrieval head provenance ok is not true");
  }
  if (provenance.heldout_prompt_provenance?.prompts_hash_match !== true) {
    errors.push("held-out prompt provenance hash did not match");
  }
  if (provenance.heldout_prompt_provenance?.row_count_match !== true) {
    errors.push("held-out prompt provenance row count did not match");
  }
  if (provenance.heldout_prompt_provenance?.unique_targets_match !== true) {
    errors.push("held-out prompt provenance unique target count did not match");
  }
  if (provenance.retrieval_head?.text_head !== true) {
    errors.push("retrieval head text_head is not valid");
  }
  if (provenance.retrieval_head?.image_head !== true) {
    errors.push("retrieval head image_head is not valid");
  }
  if (errors.length > 0) {
    throw new Error(`held-out retrieval proof failed:\n- ${errors.join("\n- ")}`);
  }
}

function metricSummary(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    top1_per_mille: Number(metric?.top1_per_mille || 0),
    top5_per_mille: Number(metric?.top5_per_mille || 0),
    min_margin: Number(metric?.min_margin ?? 0),
    mean_margin: Number(metric?.mean_margin ?? 0),
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = config.outDir
    ? path.resolve(config.outDir)
    : fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-heldout-retrieval-proof-"));
  const corpusDir = path.join(root, "corpus");
  const retrievalHeadPath = path.join(root, "retrieval-head.json");
  const retrievalEvalPath = path.join(root, "retrieval-head-eval.json");
  const provenancePath = path.join(root, "retrieval-head-provenance.json");
  fs.mkdirSync(root, { recursive: true });
  let completed = false;
  try {
    run("build v2 corpus", process.execPath, [
      "scripts/build-solomon-multimodal-corpus.mjs",
      "--text-index",
      config.textIndex,
      "--out-dir",
      corpusDir,
      "--prompt-profile",
      "seal-names",
      "--corpus-version",
      "v2",
      "--text-token-profile",
      "chunked",
      "--image-token-profile",
      "symbolic16",
      "--max-text-chars",
      "180",
    ]);
    const manifest = readJson(path.join(corpusDir, "manifest.json"));
    run("train held-out retrieval head", process.execPath, [
      "scripts/train-solomon-v2-retrieval-head.mjs",
      "--examples",
      path.join(corpusDir, "examples.jsonl"),
      "--tokens",
      path.join(corpusDir, "corpus.tokens.u8"),
      "--text-index",
      config.textIndex,
      "--prompts",
      config.prompts,
      "--model-out",
      retrievalHeadPath,
      "--eval-out",
      retrievalEvalPath,
      "--feature-count",
      config.featureCount,
      "--epochs",
      config.epochs,
      "--require-heldout-prompts",
      "--min-heldout-prompt-rows",
      config.minHeldoutPromptRows,
      "--min-heldout-top1-per-mille",
      config.minHeldoutTop1PerMille,
      "--min-heldout-top5-per-mille",
      config.minHeldoutTop5PerMille,
      "--min-retrieval-margin",
      config.minRetrievalMargin,
    ]);
    const provenanceRun = run("check retrieval head provenance", process.execPath, [
      "scripts/check-solomon-v2-retrieval-head-provenance.mjs",
      "--eval",
      retrievalEvalPath,
      "--retrieval-head",
      retrievalHeadPath,
      "--examples",
      path.join(corpusDir, "examples.jsonl"),
      "--tokens",
      path.join(corpusDir, "corpus.tokens.u8"),
      "--prompts",
      config.prompts,
      "--expect-spirits",
      "72",
      "--min-feature-count",
      "1",
      "--min-retrieval-margin",
      config.minRetrievalMargin,
    ]);
    fs.writeFileSync(provenancePath, provenanceRun.stdout, "utf8");
    const evalTrace = readJson(retrievalEvalPath);
    const model = readJson(retrievalHeadPath);
    const provenance = JSON.parse(provenanceRun.stdout);
    assertHeldoutProof(evalTrace, model, provenance, config);
    completed = true;
    console.log(JSON.stringify({
      schema,
      ok: true,
      artifacts_kept: config.keepTemp || Boolean(config.outDir),
      out_dir: root,
      corpus: {
        examples: path.join(corpusDir, "examples.jsonl"),
        tokens: path.join(corpusDir, "corpus.tokens.u8"),
        manifest: path.join(corpusDir, "manifest.json"),
        rows: Number(manifest.rows || 0),
        corpus_version: manifest.corpus_version || "",
        image_token_profile: manifest.image_token_profile || "",
        image_token_channels: manifest.image_token_channels || [],
      },
      retrieval_head: {
        model: retrievalHeadPath,
        eval: retrievalEvalPath,
        provenance: provenancePath,
        model_hash: model.model_hash || "",
        feature_count: Number(model.feature_count || 0),
        labels: Array.isArray(model.labels) ? model.labels.length : 0,
        text_head: provenance.retrieval_head?.text_head === true,
        image_head: provenance.retrieval_head?.image_head === true,
        text_nonzero_weights: Number(provenance.retrieval_head?.text_nonzero_weights || 0),
        image_nonzero_weights: Number(provenance.retrieval_head?.image_nonzero_weights || 0),
      },
      heldout_prompts: {
        prompts: config.prompts,
        prompts_hash: evalTrace.prompts_hash || "",
        rows: Number(evalTrace.heldout_prompt_rows || evalTrace.heldout_prompts?.count || 0),
        unique_targets: Number(evalTrace.heldout_prompt_unique_targets || 0),
        tiers: evalTrace.heldout_prompt_tiers || {},
        sources: evalTrace.heldout_prompt_sources || {},
        metric: metricSummary(evalTrace.heldout_prompts),
        provenance: provenance.heldout_prompt_provenance || {},
      },
      known_prompts: metricSummary(evalTrace.known_prompts),
      identity_bindings: {
        total: metricSummary(evalTrace.identity_bindings?.total),
        by_kind: Object.fromEntries(
          Object.entries(evalTrace.identity_bindings?.by_kind || {}).map(([key, value]) => [
            key,
            metricSummary(value),
          ]),
        ),
      },
      image_to_text: metricSummary(evalTrace.image_to_text),
      image_tasks: Object.fromEntries(
        Object.entries(evalTrace.image_tasks || {}).map(([key, value]) => [key, metricSummary(value)]),
      ),
      match: {
        yes: metricSummary(evalTrace.match?.yes),
        no: metricSummary(evalTrace.match?.no),
        no_by_role: {
          image: metricSummary(evalTrace.match?.no_by_role?.image),
          prompt: metricSummary(evalTrace.match?.no_by_role?.prompt),
        },
      },
    }, null, 2));
  } finally {
    if (completed && !config.keepTemp && !config.outDir) {
      fs.rmSync(root, { recursive: true, force: true });
    } else if (!completed) {
      console.error(`out_dir: ${root}`);
    }
  }
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
