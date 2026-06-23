#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const defaults = {
  prompts: "data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl",
  fallbackPrompts: "data/processed/key-solomon-goetia-latent-v1/prompts.jsonl",
  gold: "data/processed/key-solomon-goetia-latent-v1/gold.tsv",
  outDir: "data/processed/key-solomon-goetia-latent-v1/scaling-curve",
  reportOut: "",
  sizes: "",
  epochs: 24,
  latentDims: "64",
  textFeatures: "512",
  release: false,
  postImprovements: false,
  postLive: false,
  postInvokeLambda: false,
  postAdvanceStateOnDryRun: false,
  postMetric: "eval_top1_per_mille",
  postState: "data/processed/key-solomon-goetia-latent-v1/scaling-curve/x-post-state.json",
  postLambdaName: process.env.X_BOT_FUNCTION_NAME || "crowley-bard-mention-bot",
  postRegion: process.env.AWS_REGION || "us-east-1",
  postProfile: "",
};

function usage() {
  console.log(
    [
      "Usage: run-solomon-eval-scaling-curve.mjs [--prompts PATH] [--gold PATH]",
      "       [--out-dir PATH] [--report-out PATH] [--sizes LIST|all]",
      "       [--epochs N] [--latent-dims LIST] [--text-features LIST] [--release]",
      "       [--post-improvements] [--post-live] [--post-metric COLUMN]",
      "",
      "Trains/evals deterministic prompt-prefix subsets and writes curve.tsv with",
      "top1/top5 by n_train_prompts, prompt count, and model shape.",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--prompts") {
      config.prompts = requireValue(argv, ++index, arg);
    } else if (arg === "--gold") {
      config.gold = requireValue(argv, ++index, arg);
    } else if (arg === "--out-dir") {
      config.outDir = requireValue(argv, ++index, arg);
    } else if (arg === "--report-out") {
      config.reportOut = requireValue(argv, ++index, arg);
    } else if (arg === "--sizes") {
      config.sizes = requireValue(argv, ++index, arg);
    } else if (arg === "--epochs") {
      config.epochs = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--latent-dims") {
      config.latentDims = requireValue(argv, ++index, arg);
    } else if (arg === "--text-features") {
      config.textFeatures = requireValue(argv, ++index, arg);
    } else if (arg === "--release") {
      config.release = true;
    } else if (arg === "--post-improvements") {
      config.postImprovements = true;
    } else if (arg === "--post-live") {
      config.postImprovements = true;
      config.postLive = true;
      config.postInvokeLambda = true;
    } else if (arg === "--post-invoke-lambda") {
      config.postImprovements = true;
      config.postInvokeLambda = true;
    } else if (arg === "--post-advance-state-on-dry-run") {
      config.postAdvanceStateOnDryRun = true;
    } else if (arg === "--post-metric") {
      config.postMetric = requireValue(argv, ++index, arg);
    } else if (arg === "--post-state") {
      config.postState = requireValue(argv, ++index, arg);
    } else if (arg === "--post-lambda-name") {
      config.postLambdaName = requireValue(argv, ++index, arg);
    } else if (arg === "--post-region") {
      config.postRegion = requireValue(argv, ++index, arg);
    } else if (arg === "--post-profile") {
      config.postProfile = requireValue(argv, ++index, arg);
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!fs.existsSync(config.prompts)) {
    config.prompts = config.fallbackPrompts;
  }
  return config;
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parsePositive(value, flag) {
  if (!/^[0-9]+$/.test(value) || Number(value) === 0) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parseList(value, label) {
  return String(value)
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => parsePositive(item, label));
}

function readPromptLines(filePath) {
  return fs.readFileSync(filePath, "utf8").split(/\r?\n/).filter(Boolean);
}

function promptSizes(config, promptCount) {
  if (config.sizes) {
    const sizes = config.sizes === "all" ? [promptCount] : parseList(config.sizes, "--sizes");
    return uniqueSorted(sizes.map((size) => {
      if (size > promptCount) {
        throw new Error(`--sizes entry ${size} exceeds prompt count ${promptCount}`);
      }
      return size;
    }));
  }
  const base = [288, 576, 1152, promptCount].filter((size) => size <= promptCount);
  return uniqueSorted(base);
}

function uniqueSorted(values) {
  return [...new Set(values)].sort((left, right) => left - right);
}

function cargoArgs(config, binName, args) {
  const cargo = ["run", "-q"];
  if (config.release) {
    cargo.push("--release");
  }
  cargo.push("-p", "nsrl-train", "--bin", binName, "--", ...args);
  return cargo;
}

function runCommand(label, command, args, logPath) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  const log = [
    `$ ${command} ${args.join(" ")}`,
    result.stdout || "",
    result.stderr || "",
  ].join("\n");
  fs.writeFileSync(logPath, log, "utf8");
  if (result.status !== 0) {
    throw new Error(`${label} failed; see ${logPath}`);
  }
  return result.stdout.trim();
}

function parseLastJsonLine(output, label) {
  const lines = output.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    throw new Error(`${label} produced no JSON output`);
  }
  return JSON.parse(lines[lines.length - 1]);
}

function metric(metrics, key, field) {
  return metrics?.[key]?.[field] ?? 0;
}

function writeCurve(rows, outPath) {
  const header = [
    "prompt_rows",
    "n_train_prompts",
    "latent_dim",
    "text_features",
    "epochs",
    "eval_count",
    "eval_top1_per_mille",
    "eval_top5_per_mille",
    "novel_count",
    "novel_top1_per_mille",
    "cluster_count",
    "cluster_top1_per_mille",
    "gold_count",
    "gold_top1_per_mille",
    "gold_top5_per_mille",
    "model_hash",
    "model_dir",
  ];
  const lines = [
    header.join("\t"),
    ...rows.map((row) => header.map((key) => row[key]).join("\t")),
  ];
  fs.writeFileSync(outPath, `${lines.join("\n")}\n`, "utf8");
}

function runCheckpointPost(config, curvePath) {
  const args = [
    "scripts/post-solomon-improved-checkpoint.mjs",
    "--curve", curvePath,
    "--state", config.postState,
    "--metric", config.postMetric,
    "--lambda-name", config.postLambdaName,
    "--region", config.postRegion,
  ];
  if (config.postLive) {
    args.push("--live");
  } else if (config.postInvokeLambda) {
    args.push("--invoke-lambda");
  }
  if (config.postAdvanceStateOnDryRun) {
    args.push("--advance-state-on-dry-run");
  }
  if (config.postProfile) {
    args.push("--profile", config.postProfile);
  }
  const result = spawnSync("node", args, { encoding: "utf8" });
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr);
  }
  if (result.status !== 0) {
    throw new Error(`checkpoint post failed for ${curvePath}`);
  }
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const promptLines = readPromptLines(config.prompts);
  const sizes = promptSizes(config, promptLines.length);
  const latentDims = parseList(config.latentDims, "--latent-dims");
  const textFeatures = parseList(config.textFeatures, "--text-features");
  fs.mkdirSync(config.outDir, { recursive: true });
  const ledgerPath = path.join(config.outDir, "eval-ledger.jsonl");
  const curvePath = path.join(config.outDir, "curve.tsv");
  fs.writeFileSync(ledgerPath, "", "utf8");
  const rows = [];
  for (const size of sizes) {
    const promptSubsetPath = path.join(config.outDir, `prompts-${String(size).padStart(4, "0")}.jsonl`);
    fs.writeFileSync(promptSubsetPath, `${promptLines.slice(0, size).join("\n")}\n`, "utf8");
    for (const latentDim of latentDims) {
      for (const textFeatureCount of textFeatures) {
        const runId = `n${size}-ld${latentDim}-tf${textFeatureCount}-e${config.epochs}`;
        const runDir = path.join(config.outDir, runId);
        fs.mkdirSync(runDir, { recursive: true });
        runCommand(
          `train ${runId}`,
          "cargo",
          cargoArgs(config, "nsrl-solomon-latent-train", [
            "--prompts", promptSubsetPath,
            "--gold", config.gold,
            "--out-dir", runDir,
            "--epochs", String(config.epochs),
            "--latent-dim", String(latentDim),
            "--text-features", String(textFeatureCount),
          ]),
          path.join(runDir, "train.log"),
        );
        const evalOutput = runCommand(
          `eval ${runId}`,
          "cargo",
          cargoArgs(config, "nsrl-solomon-eval", [
            "--prompts", promptSubsetPath,
            "--gold", config.gold,
            "--model", path.join(runDir, "model.nsrllat"),
            "--ledger", ledgerPath,
            "--partition-out", path.join(runDir, "partition.tsv"),
            "--prompt-set-version", `solomon-curve-n${size}`,
          ]),
          path.join(runDir, "eval.log"),
        );
        const evalJson = parseLastJsonLine(evalOutput, `eval ${runId}`);
        rows.push({
          prompt_rows: size,
          n_train_prompts: evalJson.n_train_prompts,
          latent_dim: latentDim,
          text_features: textFeatureCount,
          epochs: config.epochs,
          eval_count: metric(evalJson.retrieval_eval, "all", "count"),
          eval_top1_per_mille: metric(evalJson.retrieval_eval, "all", "top1_per_mille"),
          eval_top5_per_mille: metric(evalJson.retrieval_eval, "all", "top5_per_mille"),
          novel_count: metric(evalJson.retrieval_eval, "tier-novel-vocab", "count"),
          novel_top1_per_mille: metric(evalJson.retrieval_eval, "tier-novel-vocab", "top1_per_mille"),
          cluster_count: metric(evalJson.retrieval_eval, "tier-cluster-holdout", "count"),
          cluster_top1_per_mille: metric(evalJson.retrieval_eval, "tier-cluster-holdout", "top1_per_mille"),
          gold_count: metric(evalJson.retrieval_gold, "all", "count"),
          gold_top1_per_mille: metric(evalJson.retrieval_gold, "all", "top1_per_mille"),
          gold_top5_per_mille: metric(evalJson.retrieval_gold, "all", "top5_per_mille"),
          model_hash: evalJson.model_hash,
          model_dir: runDir,
        });
        writeCurve(rows, curvePath);
        if (config.reportOut) {
          fs.mkdirSync(path.dirname(config.reportOut), { recursive: true });
          writeCurve(rows, config.reportOut);
        }
        if (config.postImprovements) {
          runCheckpointPost(config, curvePath);
        }
        console.log(JSON.stringify(rows[rows.length - 1]));
      }
    }
  }
  console.error(`wrote ${curvePath}`);
}

main();
