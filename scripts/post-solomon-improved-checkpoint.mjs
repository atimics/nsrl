#!/usr/bin/env node
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const defaults = {
  curve: "docs/solomon-eval-scaling-curve.tsv",
  state: "data/processed/key-solomon-goetia-latent-v1/scaling-curve/x-post-state.json",
  metric: "eval_top1_per_mille",
  minImprovement: 1,
  lambdaName: process.env.X_BOT_FUNCTION_NAME || "crowley-bard-mention-bot",
  region: process.env.AWS_REGION || "us-east-1",
  awsCli: process.env.AWS_CLI || "aws",
  profile: "",
  invokeLambda: false,
  dryRun: true,
  advanceStateOnDryRun: false,
  tweetPrompt:
    "Solomon checkpoint improved. Speak one compact omen about integer seals learning from held-out prompts.",
};

const schema = "nsrl.solomon_x_checkpoint_state.v1";

const metricLabels = {
  eval_top1_per_mille: "eval top1",
  novel_top1_per_mille: "novel-vocab top1",
  cluster_top1_per_mille: "cluster-holdout top1",
  gold_top1_per_mille: "gold top1",
};

function usage() {
  console.log(
    [
      "Usage: post-solomon-improved-checkpoint.mjs [--curve PATH] [--state PATH]",
      "       [--metric COLUMN] [--min-improvement N] [--invoke-lambda]",
      "       [--live] [--lambda-name NAME] [--region REGION] [--tweet-prompt TEXT]",
      "",
      "Reads a Solomon eval curve TSV, detects whether the selected metric has",
      "beaten the saved best checkpoint, and prepares or sends an idempotent X",
      "thread through the existing Crowley Bard Lambda: model-generated top-level",
      "tweet with sigil image, deterministic metrics in the reply.",
      "",
      "Default mode is safe: dry-run payload only, no Lambda invoke, no state write.",
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
    } else if (arg === "--curve") {
      config.curve = requireValue(argv, ++index, arg);
    } else if (arg === "--state") {
      config.state = requireValue(argv, ++index, arg);
    } else if (arg === "--metric") {
      config.metric = requireValue(argv, ++index, arg);
    } else if (arg === "--min-improvement") {
      config.minImprovement = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--lambda-name" || arg === "--function-name") {
      config.lambdaName = requireValue(argv, ++index, arg);
    } else if (arg === "--region") {
      config.region = requireValue(argv, ++index, arg);
    } else if (arg === "--aws-cli") {
      config.awsCli = requireValue(argv, ++index, arg);
    } else if (arg === "--profile") {
      config.profile = requireValue(argv, ++index, arg);
    } else if (arg === "--invoke-lambda") {
      config.invokeLambda = true;
    } else if (arg === "--live") {
      config.dryRun = false;
      config.invokeLambda = true;
    } else if (arg === "--dry-run") {
      config.dryRun = true;
    } else if (arg === "--advance-state-on-dry-run") {
      config.advanceStateOnDryRun = true;
    } else if (arg === "--tweet-prompt") {
      config.tweetPrompt = requireValue(argv, ++index, arg);
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

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function readTsv(filePath) {
  const text = fs.readFileSync(filePath, "utf8").trimEnd();
  if (!text) {
    throw new Error(`${filePath} is empty`);
  }
  const lines = text.split(/\r?\n/);
  const header = lines[0].split("\t");
  const rows = lines.slice(1).filter(Boolean).map((line, rowIndex) => {
    const fields = line.split("\t");
    const row = {};
    for (let index = 0; index < header.length; index += 1) {
      row[header[index]] = fields[index] ?? "";
    }
    row.__line = rowIndex + 2;
    return row;
  });
  return { header, rows };
}

function numberField(row, key) {
  const value = Number(row[key]);
  if (!Number.isFinite(value)) {
    throw new Error(`curve row ${row.__line} has non-numeric ${key}`);
  }
  return value;
}

function bestRow(rows, metric) {
  if (rows.length === 0) {
    throw new Error("curve has no data rows");
  }
  let best = rows[0];
  for (const row of rows.slice(1)) {
    if (compareRows(row, best, metric) > 0) {
      best = row;
    }
  }
  return best;
}

function compareRows(left, right, metric) {
  const tieBreakers = [
    metric,
    "novel_top1_per_mille",
    "cluster_top1_per_mille",
    "gold_top1_per_mille",
    "eval_top5_per_mille",
    "n_train_prompts",
  ];
  for (const key of tieBreakers) {
    const diff = numberField(left, key) - numberField(right, key);
    if (diff !== 0) {
      return diff;
    }
  }
  return String(right.model_hash).localeCompare(String(left.model_hash));
}

function readState(filePath) {
  if (!fs.existsSync(filePath)) {
    return { schema, metrics: {}, dry_run_metrics: {} };
  }
  const state = JSON.parse(fs.readFileSync(filePath, "utf8"));
  state.schema = state.schema || schema;
  state.metrics = state.metrics || {};
  state.dry_run_metrics = state.dry_run_metrics || {};
  return state;
}

function stateBucket(config) {
  return config.dryRun ? "dry_run_metrics" : "metrics";
}

function previousBest(state, config) {
  const bucket = stateBucket(config);
  const value = Number(state[bucket]?.[config.metric]?.best_value);
  return Number.isFinite(value) ? value : -1;
}

function checkpointPostId(row, metric) {
  const safeMetric = metric.replace(/[^A-Za-z0-9_.-]+/g, "-");
  const hash = String(row.model_hash || "unknown").replace(/[^A-Za-z0-9_.-]+/g, "");
  return `solomon-checkpoint-${safeMetric}-${row[metric]}-${hash}`;
}

function replyText(row, metric) {
  const label = metricLabels[metric] || metric.replace(/_/g, " ");
  const lines = [
    `Solomon checkpoint improved: ${label} ${row[metric]}/1000.`,
    `Train prompts ${row.n_train_prompts}; prompt rows ${row.prompt_rows}; ld${row.latent_dim} tf${row.text_features} e${row.epochs}.`,
    `Novel ${row.novel_top1_per_mille}/1000, cluster ${row.cluster_top1_per_mille}/1000, gold ${row.gold_top1_per_mille}/1000.`,
    `Model ${row.model_hash}. #NSRL`,
  ];
  const text = lines.join("\n");
  if (text.length > 260) {
    throw new Error(`checkpoint reply is ${text.length} chars; max is 260`);
  }
  return text;
}

function buildPayload(row, config) {
  return {
    post_tweet: true,
    dry_run: config.dryRun,
    id: checkpointPostId(row, config.metric),
    prompt: config.tweetPrompt,
    reply_text: replyText(row, config.metric),
    source: "solomon-eval-checkpoint",
    metric: config.metric,
  };
}

function invokeLambda(payload, config) {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-x-checkpoint-"));
  const outPath = path.join(tempDir, "response.json");
  const args = [
    "lambda",
    "invoke",
    "--region",
    config.region,
    "--function-name",
    config.lambdaName,
    "--cli-binary-format",
    "raw-in-base64-out",
    "--payload",
    JSON.stringify(payload),
  ];
  if (config.profile) {
    args.push("--profile", config.profile);
  }
  args.push(outPath);
  const result = spawnSync(config.awsCli, args, { encoding: "utf8" });
  if (result.status !== 0) {
    throw new Error(
      `Lambda invoke failed: ${(result.stderr || result.stdout || "").trim()}`,
    );
  }
  const responseText = fs.readFileSync(outPath, "utf8").trim() || "{}";
  return JSON.parse(responseText);
}

function lambdaAccepted(response, config) {
  if (!response || response.ok !== true) {
    return false;
  }
  if (config.dryRun) {
    return response.would_post === true;
  }
  return response.posted === true || response.duplicate === true;
}

function writeState(filePath, state, config, row, payload, lambdaResponse) {
  const bucket = stateBucket(config);
  state.schema = schema;
  state[bucket] = state[bucket] || {};
  state[bucket][config.metric] = {
    best_value: numberField(row, config.metric),
    post_id: payload.id,
    model_hash: String(row.model_hash || ""),
    prompt_rows: numberField(row, "prompt_rows"),
    n_train_prompts: numberField(row, "n_train_prompts"),
    latent_dim: numberField(row, "latent_dim"),
    text_features: numberField(row, "text_features"),
    epochs: numberField(row, "epochs"),
    updated_at: new Date().toISOString(),
    dry_run: config.dryRun,
    lambda_status: lambdaResponse ? summarizeLambdaResponse(lambdaResponse) : "local",
  };
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function summarizeLambdaResponse(response) {
  if (response.posted) {
    return "posted";
  }
  if (response.duplicate) {
    return `duplicate:${response.status || "unknown"}`;
  }
  if (response.would_post) {
    return "would_post";
  }
  return response.error || "unknown";
}

function emit(value) {
  console.log(JSON.stringify(value, null, 2));
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const { header, rows } = readTsv(config.curve);
  if (!header.includes(config.metric)) {
    throw new Error(`${config.curve} has no metric column ${config.metric}`);
  }
  const row = bestRow(rows, config.metric);
  const value = numberField(row, config.metric);
  const state = readState(config.state);
  const prior = previousBest(state, config);
  const delta = value - prior;
  if (delta < config.minImprovement) {
    emit({
      ok: true,
      skipped: "no_improvement",
      dry_run: config.dryRun,
      metric: config.metric,
      best_value: value,
      previous_best_value: prior,
      min_improvement: config.minImprovement,
      model_hash: row.model_hash,
    });
    return;
  }

  const payload = buildPayload(row, config);
  let lambdaResponse = null;
  if (config.invokeLambda) {
    lambdaResponse = invokeLambda(payload, config);
    if (!lambdaAccepted(lambdaResponse, config)) {
      emit({
        ok: false,
        error: "lambda_rejected_checkpoint_post",
        payload,
        lambda_response: lambdaResponse,
      });
      process.exit(1);
    }
  }

  const shouldWriteState = !config.dryRun || config.advanceStateOnDryRun;
  if (shouldWriteState) {
    writeState(config.state, state, config, row, payload, lambdaResponse);
  }

  emit({
    ok: true,
    dry_run: config.dryRun,
    would_post: config.dryRun,
    posted: !config.dryRun && lambdaResponse?.posted === true,
    duplicate: !config.dryRun && lambdaResponse?.duplicate === true,
    invoke_lambda: config.invokeLambda,
    state_written: shouldWriteState,
    metric: config.metric,
    best_value: value,
    previous_best_value: prior,
    improvement: delta,
    row,
    payload,
    lambda_response: lambdaResponse,
  });
}

main();
