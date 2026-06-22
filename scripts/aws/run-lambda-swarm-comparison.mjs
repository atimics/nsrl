#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { execFile, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "../..");

function usage() {
  console.log(`Usage:
  node scripts/aws/run-lambda-swarm-comparison.mjs --deploy --run \\
    --tokens-s3-uri s3://bucket/prefix/tokens.u8 \\
    --s3-uri s3://bucket/prefix \\
    --run-name crowley-bard-lambda-smoke \\
    --workers 4

Options:
  --profile staging
  --region us-east-1
  --function-name nsrl-mini-transformer-swarm-worker
  --role-name NSRLLambdaSwarmWorkerRole
  --role-arn arn:aws:iam::ACCOUNT:role/ROLE
  --zip data/aws-lambda-swarm/build/nsrl-lambda-swarm-worker.zip
  --runtime python3.12
  --memory-mb 2048
  --timeout-seconds 900
  --workers 4
  --max-windows 65536
  --seq-len 8
  --stride 1
  --batch-windows 2
  --batch-mode map-reduce
  --map-reduce-workers 2
  --tokenizer ascii-lower
  --adaptive-rule-shifts 1
  --progress-interval-batches 1024
  --lambda-gb-second-usd 0.0000133334
  --lambda-request-usd 0.0000002`);
}

function parseArgs(argv) {
  const options = {
    profile: "staging",
    region: "us-east-1",
    functionName: "nsrl-mini-transformer-swarm-worker",
    roleName: "NSRLLambdaSwarmWorkerRole",
    zip: "data/aws-lambda-swarm/build/nsrl-lambda-swarm-worker.zip",
    runtime: "python3.12",
    memoryMb: 2048,
    timeoutSeconds: 900,
    ephemeralStorageMb: 512,
    workers: 4,
    maxWindows: 65536,
    seqLen: 8,
    stride: 1,
    windowOffset: 0,
    batchWindows: 2,
    batchMode: "map-reduce",
    mapReduceWorkers: 2,
    epochs: 1,
    outShift: 18,
    mlpShift: 17,
    embedShift: 13,
    attentionShift: 22,
    attentionQShift: 18,
    attentionQkShift: 16,
    adaptiveRuleShifts: 1,
    adaptiveRuleIntervalBatches: 128,
    adaptiveHolographicShifts: 0,
    progressIntervalBatches: 1024,
    traceDetail: "none",
    attention: "linear",
    position: "nope",
    tokenizer: "ascii-lower",
    invocationType: "RequestResponse",
    pollSeconds: 5,
    pollTimeoutSeconds: 900,
    runRoot: "data/aws-lambda-swarm/runs",
    lambdaGbSecondUsd: 0.0000133334,
    lambdaRequestUsd: 0.0000002,
    deploy: false,
    run: false,
    assemble: true,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    }
    if (arg === "--deploy") {
      options.deploy = true;
      continue;
    }
    if (arg === "--run") {
      options.run = true;
      continue;
    }
    if (arg === "--no-assemble") {
      options.assemble = false;
      continue;
    }
    if (!arg.startsWith("--")) {
      throw new Error(`unexpected positional argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    const value = argv[++index];
    if (value === undefined) {
      throw new Error(`${arg} requires a value`);
    }
    if (
      [
        "memoryMb",
        "timeoutSeconds",
        "ephemeralStorageMb",
        "workers",
        "maxWindows",
        "seqLen",
        "stride",
        "windowOffset",
        "batchWindows",
        "mapReduceWorkers",
        "epochs",
        "outShift",
        "mlpShift",
        "embedShift",
        "attentionShift",
        "attentionQShift",
        "attentionQkShift",
        "adaptiveRuleShifts",
        "adaptiveRuleIntervalBatches",
        "adaptiveHolographicShifts",
        "progressIntervalBatches",
        "pollSeconds",
        "pollTimeoutSeconds",
      ].includes(key)
    ) {
      options[key] = Number.parseInt(value, 10);
    } else if (["lambdaGbSecondUsd", "lambdaRequestUsd"].includes(key)) {
      options[key] = Number.parseFloat(value);
    } else {
      options[key] = value;
    }
  }
  if (!options.s3Uri) {
    throw new Error("--s3-uri is required");
  }
  if (!options.tokensS3Uri) {
    throw new Error("--tokens-s3-uri is required");
  }
  if (!["serial", "map-reduce"].includes(options.batchMode)) {
    throw new Error(`--batch-mode must be serial or map-reduce, got ${options.batchMode}`);
  }
  if (options.batchMode === "map-reduce") {
    if (options.batchWindows < 2) {
      throw new Error("--batch-windows must be at least 2 for Lambda map-reduce runs");
    }
    if (options.mapReduceWorkers < 1) {
      throw new Error("--map-reduce-workers must be at least 1");
    }
  }
  if (!options.runName) {
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
    options.runName = `lambda-swarm-mapreduce-${stamp}`;
  }
  return options;
}

function awsBase(options) {
  const args = [];
  if (options.profile) args.push("--profile", options.profile);
  if (options.region) args.push("--region", options.region);
  return args;
}

function run(command, args, { cwd = repoRoot, input, allowFailure = false } = {}) {
  const result = spawnSync(command, args, {
    cwd,
    input,
    text: true,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (!allowFailure && result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed with ${result.status}\n${result.stdout}\n${result.stderr}`,
    );
  }
  return result;
}

function runText(command, args, options) {
  return run(command, args, options).stdout.trim();
}

function runJson(command, args, options) {
  const text = runText(command, args, options);
  return text ? JSON.parse(text) : null;
}

function execFilePromise(command, args, { cwd = repoRoot } = {}) {
  return new Promise((resolve) => {
    execFile(
      command,
      args,
      { cwd, encoding: "utf8", maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => resolve({ error, stdout, stderr, args }),
    );
  });
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function s3Parts(uri) {
  const parsed = new URL(uri);
  if (parsed.protocol !== "s3:" || !parsed.hostname || !parsed.pathname.slice(1)) {
    throw new Error(`expected s3://bucket/key, got ${uri}`);
  }
  return { bucket: parsed.hostname, key: parsed.pathname.slice(1) };
}

function s3PrefixParts(uri) {
  const parsed = new URL(uri);
  if (parsed.protocol !== "s3:" || !parsed.hostname) {
    throw new Error(`expected s3://bucket[/prefix], got ${uri}`);
  }
  return { bucket: parsed.hostname, prefix: parsed.pathname.slice(1).replace(/\/$/, "") };
}

function s3Join(prefix, ...parts) {
  return [prefix.replace(/\/$/, ""), ...parts.map((part) => String(part).replace(/^\/|\/$/g, ""))]
    .filter(Boolean)
    .join("/");
}

function ensureRole(options, runDir) {
  if (options.roleArn) {
    return options.roleArn;
  }
  const roleName = options.roleName;
  const trustPath = path.join(runDir, "iam-trust.json");
  const policyPath = path.join(runDir, "iam-policy.json");
  const trust = {
    Version: "2012-10-17",
    Statement: [
      {
        Effect: "Allow",
        Principal: { Service: "lambda.amazonaws.com" },
        Action: "sts:AssumeRole",
      },
    ],
  };
  const s3Buckets = new Set([s3PrefixParts(options.s3Uri).bucket, s3Parts(options.tokensS3Uri).bucket]);
  const bucketResources = [...s3Buckets].flatMap((bucket) => [
    `arn:aws:s3:::${bucket}`,
    `arn:aws:s3:::${bucket}/*`,
  ]);
  const policy = {
    Version: "2012-10-17",
    Statement: [
      {
        Effect: "Allow",
        Action: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource: "arn:aws:logs:*:*:*",
      },
      {
        Effect: "Allow",
        Action: ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
        Resource: bucketResources,
      },
    ],
  };
  writeJson(trustPath, trust);
  writeJson(policyPath, policy);

  const getRoleArgs = [...awsBase(options), "iam", "get-role", "--role-name", roleName, "--output", "json"];
  let role = runJson("aws", getRoleArgs, { allowFailure: true });
  if (!role) {
    run("aws", [
      ...awsBase(options),
      "iam",
      "create-role",
      "--role-name",
      roleName,
      "--assume-role-policy-document",
      `file://${trustPath}`,
    ]);
    run("aws", [
      ...awsBase(options),
      "iam",
      "attach-role-policy",
      "--role-name",
      roleName,
      "--policy-arn",
      "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    ]);
    role = runJson("aws", getRoleArgs);
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10000);
  }
  run("aws", [
    ...awsBase(options),
    "iam",
    "put-role-policy",
    "--role-name",
    roleName,
    "--policy-name",
    "NSRLLambdaSwarmWorkerS3Access",
    "--policy-document",
    `file://${policyPath}`,
  ]);
  return role.Role.Arn;
}

function deployFunction(options, roleArn) {
  const zipPath = path.resolve(repoRoot, options.zip);
  if (!fs.existsSync(zipPath)) {
    throw new Error(`Lambda zip not found: ${zipPath}`);
  }
  const exists = run("aws", [...awsBase(options), "lambda", "get-function", "--function-name", options.functionName], {
    allowFailure: true,
  }).status === 0;
  if (exists) {
    run("aws", [
      ...awsBase(options),
      "lambda",
      "update-function-code",
      "--function-name",
      options.functionName,
      "--zip-file",
      `fileb://${zipPath}`,
    ]);
    run("aws", [...awsBase(options), "lambda", "wait", "function-updated", "--function-name", options.functionName]);
    run("aws", [
      ...awsBase(options),
      "lambda",
      "update-function-configuration",
      "--function-name",
      options.functionName,
      "--role",
      roleArn,
      "--runtime",
      options.runtime,
      "--handler",
      "lambda_function.handler",
      "--timeout",
      String(options.timeoutSeconds),
      "--memory-size",
      String(options.memoryMb),
      "--ephemeral-storage",
      `Size=${options.ephemeralStorageMb}`,
    ]);
    run("aws", [...awsBase(options), "lambda", "wait", "function-updated", "--function-name", options.functionName]);
  } else {
    run("aws", [
      ...awsBase(options),
      "lambda",
      "create-function",
      "--function-name",
      options.functionName,
      "--runtime",
      options.runtime,
      "--handler",
      "lambda_function.handler",
      "--role",
      roleArn,
      "--architectures",
      "arm64",
      "--timeout",
      String(options.timeoutSeconds),
      "--memory-size",
      String(options.memoryMb),
      "--ephemeral-storage",
      `Size=${options.ephemeralStorageMb}`,
      "--zip-file",
      `fileb://${zipPath}`,
    ]);
    run("aws", [...awsBase(options), "lambda", "wait", "function-active", "--function-name", options.functionName]);
  }
}

function workerPayload(options, workerIndex, outputS3Prefix) {
  return {
    run_name: options.runName,
    worker_index: workerIndex,
    worker_count: options.workers,
    tokens_s3_uri: options.tokensS3Uri,
    output_s3_prefix: outputS3Prefix,
    config: {
      max_windows: options.maxWindows,
      seq_len: options.seqLen,
      stride: options.stride,
      window_offset: options.windowOffset,
      batch_windows: options.batchWindows,
      batch_mode: options.batchMode,
      map_reduce_workers: options.mapReduceWorkers,
      epochs: options.epochs,
      out_shift: options.outShift,
      mlp_shift: options.mlpShift,
      embed_shift: options.embedShift,
      attention_shift: options.attentionShift,
      attention_q_shift: options.attentionQShift,
      attention_qk_shift: options.attentionQkShift,
      adaptive_rule_shifts: options.adaptiveRuleShifts,
      adaptive_rule_interval_batches: options.adaptiveRuleIntervalBatches,
      adaptive_holographic_shifts: options.adaptiveHolographicShifts,
      progress_interval_batches: options.progressIntervalBatches,
      trace_detail: options.traceDetail,
      attention: options.attention,
      position: options.position,
      tokenizer: options.tokenizer,
    },
  };
}

async function invokeWorkers(options, runDir, outputS3Prefix) {
  const started = Date.now();
  const invocations = [];
  for (let worker = 0; worker < options.workers; worker += 1) {
    const payloadPath = path.join(runDir, `payload-worker-${worker.toString().padStart(3, "0")}.json`);
    const responsePath = path.join(runDir, `invoke-worker-${worker.toString().padStart(3, "0")}.json`);
    writeJson(payloadPath, workerPayload(options, worker, outputS3Prefix));
    invocations.push(
      execFilePromise("aws", [
        ...awsBase(options),
        "lambda",
        "invoke",
        "--function-name",
        options.functionName,
        "--invocation-type",
        options.invocationType,
        "--cli-binary-format",
        "raw-in-base64-out",
        "--payload",
        `file://${payloadPath}`,
        responsePath,
      ]),
    );
  }
  const results = await Promise.all(invocations);
  const failed = results.filter((result) => result.error);
  if (failed.length > 0) {
    for (const result of failed) {
      console.error(result.stderr || result.stdout || result.error.message);
    }
    throw new Error(`${failed.length} Lambda invocation command(s) failed`);
  }
  return Date.now() - started;
}

function s3ObjectExists(options, uri) {
  const { bucket, key } = s3Parts(uri);
  return (
    run("aws", [...awsBase(options), "s3api", "head-object", "--bucket", bucket, "--key", key], {
      allowFailure: true,
    }).status === 0
  );
}

function readS3JsonIfExists(options, uri) {
  const { bucket, key } = s3Parts(uri);
  const result = run(
    "aws",
    [...awsBase(options), "s3api", "get-object", "--bucket", bucket, "--key", key, "-"],
    { allowFailure: true },
  );
  if (result.status !== 0 || !result.stdout.trim()) {
    return null;
  }
  return JSON.parse(result.stdout);
}

function downloadS3(options, uri, localPath) {
  fs.mkdirSync(path.dirname(localPath), { recursive: true });
  run("aws", [...awsBase(options), "s3", "cp", uri, localPath, "--only-show-errors"]);
}

function uploadS3(options, localPath, uri) {
  run("aws", [...awsBase(options), "s3", "cp", localPath, uri, "--only-show-errors"]);
}

async function pollWorkers(options, outputS3Prefix) {
  const started = Date.now();
  const needed = new Set();
  const summaryUris = [];
  for (let worker = 0; worker < options.workers; worker += 1) {
    const workerId = `worker-${worker.toString().padStart(3, "0")}`;
    needed.add(s3Join(outputS3Prefix, "workers", `${workerId}.nsrlwk`));
    const summaryUri = s3Join(outputS3Prefix, "workers", `${workerId}.summary.json`);
    needed.add(summaryUri);
    summaryUris.push(summaryUri);
  }
  while (needed.size > 0) {
    for (const uri of summaryUris) {
      const summary = readS3JsonIfExists(options, uri);
      if (summary && summary.ok === false) {
        throw new Error(
          `Lambda worker ${summary.worker_index} failed: ${summary.stdout_tail || "see summary"}`,
        );
      }
    }
    for (const uri of [...needed]) {
      if (s3ObjectExists(options, uri)) {
        needed.delete(uri);
      }
    }
    if (needed.size === 0) {
      break;
    }
    if ((Date.now() - started) / 1000 > options.pollTimeoutSeconds) {
      throw new Error(`timed out waiting for ${needed.size} Lambda artifacts`);
    }
    await new Promise((resolve) => setTimeout(resolve, options.pollSeconds * 1000));
  }
}

function downloadWorkerArtifacts(options, runDir, outputS3Prefix) {
  const artifacts = [];
  const summaries = [];
  for (let worker = 0; worker < options.workers; worker += 1) {
    const workerId = `worker-${worker.toString().padStart(3, "0")}`;
    const artifactPath = path.join(runDir, "workers", `${workerId}.nsrlwk`);
    const summaryPath = path.join(runDir, "workers", `${workerId}.summary.json`);
    downloadS3(options, s3Join(outputS3Prefix, "workers", `${workerId}.nsrlwk`), artifactPath);
    downloadS3(options, s3Join(outputS3Prefix, "workers", `${workerId}.summary.json`), summaryPath);
    artifacts.push(artifactPath);
    summaries.push(JSON.parse(fs.readFileSync(summaryPath, "utf8")));
  }
  return { artifacts, summaries };
}

function assemble(options, runDir, artifacts, outputS3Prefix) {
  const tokensPath = path.join(runDir, "tokens.u8");
  downloadS3(options, options.tokensS3Uri, tokensPath);
  const modelOut = path.join(runDir, `${options.runName}.nsrlmt`);
  const swarmOut = path.join(runDir, `${options.runName}.nsrlswarm`);
  const manifestOut = path.join(runDir, `${options.runName}.manifest.jsonl`);
  const traceOut = path.join(runDir, `${options.runName}.assemble.trace.jsonl`);
  const args = [
    "run",
    "--release",
    "-q",
    "-p",
    "nsrl-train",
    "--",
    "--mode",
    "mini-transformer-swarm-assemble",
    "--tokens",
    tokensPath,
    "--seq-len",
    String(options.seqLen),
    "--stride",
    String(options.stride),
    "--window-offset",
    String(options.windowOffset),
    "--batch-windows",
    String(options.batchWindows),
    "--mini-transformer-batch-mode",
    options.batchMode,
    "--mini-transformer-map-reduce-workers",
    String(options.mapReduceWorkers),
    "--max-windows",
    String(options.maxWindows),
    "--epochs",
    String(options.epochs),
    "--lr-shift",
    String(options.outShift),
    "--mlp-lr-shift",
    String(options.mlpShift),
    "--embed-lr-shift",
    String(options.embedShift),
    "--attention-lr-shift",
    String(options.attentionShift),
    "--attention-q-lr-shift",
    String(options.attentionQShift),
    "--attention-qk-lr-shift",
    String(options.attentionQkShift),
    "--mini-transformer-attention",
    options.attention,
    "--mini-transformer-position",
    options.position,
    "--tokenizer",
    options.tokenizer,
    "--trace",
    traceOut,
    "--model-out",
    modelOut,
    "--swarm-model-out",
    swarmOut,
    "--manifest-out",
    manifestOut,
  ];
  for (const artifact of artifacts) {
    args.push("--swarm-worker-artifact", artifact);
  }
  if (options.adaptiveRuleShifts !== 0) {
    args.push("--adaptive-rule-shifts", "--adaptive-rule-interval-batches", String(options.adaptiveRuleIntervalBatches));
  }
  if (options.adaptiveHolographicShifts !== 0) {
    args.push("--adaptive-holographic-shifts");
  }
  run("cargo", args);
  uploadS3(options, modelOut, s3Join(outputS3Prefix, "assembled", path.basename(modelOut)));
  uploadS3(options, swarmOut, s3Join(outputS3Prefix, "assembled", path.basename(swarmOut)));
  uploadS3(options, manifestOut, s3Join(outputS3Prefix, "assembled", path.basename(manifestOut)));
  uploadS3(options, traceOut, s3Join(outputS3Prefix, "assembled", path.basename(traceOut)));
  return { modelOut, swarmOut, manifestOut, traceOut };
}

function writeMetrics(options, runDir, outputS3Prefix, timings, summaries, assembled) {
  const elapsedMs = summaries.map((summary) => Number(summary.elapsed_ms || 0));
  const trainMs = summaries.map((summary) => Number(summary.train_ms || 0));
  const memoryGb = options.memoryMb / 1024;
  const lambdaGbSeconds = elapsedMs.reduce((total, value) => total + (value / 1000) * memoryGb, 0);
  const estimatedLambdaUsd =
    lambdaGbSeconds * options.lambdaGbSecondUsd + options.workers * options.lambdaRequestUsd;
  const metrics = {
    schema: "nsrl.lambda_swarm_comparison_metrics.v1",
    run_name: options.runName,
    function_name: options.functionName,
    output_s3_prefix: outputS3Prefix,
    tokens_s3_uri: options.tokensS3Uri,
    worker_count: options.workers,
    memory_mb: options.memoryMb,
    invocation_type: options.invocationType,
    config: workerPayload(options, 0, outputS3Prefix).config,
    wall_ms: timings.wallMs,
    invoke_ms: timings.invokeMs,
    lambda_elapsed_ms_sum: elapsedMs.reduce((a, b) => a + b, 0),
    lambda_elapsed_ms_max: Math.max(...elapsedMs),
    lambda_train_ms_sum: trainMs.reduce((a, b) => a + b, 0),
    lambda_train_ms_max: Math.max(...trainMs),
    lambda_gb_seconds: Number(lambdaGbSeconds.toFixed(6)),
    lambda_gb_second_usd: options.lambdaGbSecondUsd,
    lambda_request_usd: options.lambdaRequestUsd,
    estimated_lambda_usd: Number(estimatedLambdaUsd.toFixed(6)),
    assembled,
    summaries,
  };
  const jsonPath = path.join(runDir, "metrics.json");
  const tsvPath = path.join(runDir, "metrics.tsv");
  writeJson(jsonPath, metrics);
  fs.writeFileSync(
    tsvPath,
    [
      "run_name\tworkers\tmemory_mb\twall_ms\tinvoke_ms\tlambda_elapsed_ms_sum\tlambda_elapsed_ms_max\tlambda_train_ms_sum\tlambda_train_ms_max\tlambda_gb_seconds\testimated_lambda_usd",
      [
        options.runName,
        options.workers,
        options.memoryMb,
        metrics.wall_ms,
        metrics.invoke_ms,
        metrics.lambda_elapsed_ms_sum,
        metrics.lambda_elapsed_ms_max,
        metrics.lambda_train_ms_sum,
        metrics.lambda_train_ms_max,
        metrics.lambda_gb_seconds,
        metrics.estimated_lambda_usd,
      ].join("\t"),
      "",
    ].join("\n"),
    "utf8",
  );
  uploadS3(options, jsonPath, s3Join(outputS3Prefix, "metrics.json"));
  uploadS3(options, tsvPath, s3Join(outputS3Prefix, "metrics.tsv"));
  return metrics;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const runDir = path.resolve(repoRoot, options.runRoot, options.runName);
  fs.mkdirSync(runDir, { recursive: true });
  const outputS3Prefix = s3Join(options.s3Uri, "lambda-runs", options.runName);
  writeJson(path.join(runDir, "run-options.json"), { ...options, outputS3Prefix });
  uploadS3(options, path.join(runDir, "run-options.json"), s3Join(outputS3Prefix, "run-options.json"));

  if (options.deploy) {
    const roleArn = ensureRole(options, runDir);
    deployFunction(options, roleArn);
  }
  if (!options.run) {
    console.log(JSON.stringify({ run_name: options.runName, deployed: options.deploy, output_s3_prefix: outputS3Prefix }, null, 2));
    return;
  }

  const started = Date.now();
  const invokeMs = await invokeWorkers(options, runDir, outputS3Prefix);
  await pollWorkers(options, outputS3Prefix);
  const { artifacts, summaries } = downloadWorkerArtifacts(options, runDir, outputS3Prefix);
  const assembled = options.assemble ? assemble(options, runDir, artifacts, outputS3Prefix) : null;
  const metrics = writeMetrics(
    options,
    runDir,
    outputS3Prefix,
    { wallMs: Date.now() - started, invokeMs },
    summaries,
    assembled,
  );
  console.log(JSON.stringify({
    run_name: options.runName,
    output_s3_prefix: outputS3Prefix,
    local_run_dir: runDir,
    metrics: {
      wall_ms: metrics.wall_ms,
      lambda_train_ms_max: metrics.lambda_train_ms_max,
      lambda_gb_seconds: metrics.lambda_gb_seconds,
      estimated_lambda_usd: metrics.estimated_lambda_usd,
    },
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
