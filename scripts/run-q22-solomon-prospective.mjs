#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const root = path.resolve(import.meta.dirname, "..");
const defaultContract = path.join(root, "benchmarks/q22-solomon-prospective-v1/contract.json");

function fail(message) {
  throw new Error(message);
}

function parseArgs(argv) {
  const options = {
    contract: defaultContract,
    manifest: path.join(root, "benchmarks/q22-shared-task-v1/manifest.tsv"),
    cargo: "cargo",
    checkContract: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--check-contract") options.checkContract = true;
    else if (arg === "--contract") options.contract = path.resolve(required(argv, ++index, arg));
    else if (arg === "--manifest") options.manifest = path.resolve(required(argv, ++index, arg));
    else if (arg === "--train-dataset") options.trainDataset = path.resolve(required(argv, ++index, arg));
    else if (arg === "--eval") options.eval = path.resolve(required(argv, ++index, arg));
    else if (arg === "--out-dir") options.outDir = path.resolve(required(argv, ++index, arg));
    else if (arg === "--cargo") options.cargo = required(argv, ++index, arg);
    else fail(`unknown argument: ${arg}`);
  }
  return options;
}

function required(argv, index, option) {
  if (!argv[index]) fail(`${option} requires a value`);
  return argv[index];
}

function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sha256File(file) {
  return sha256Bytes(fs.readFileSync(file));
}

function stableJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function atomicWrite(file, bytes) {
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, bytes);
  fs.renameSync(temporary, file);
}

function exactKeys(value, expected, label) {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${label} fields do not match the frozen schema`);
  }
}

function validateContract(contract, contractPath) {
  exactKeys(contract, [
    "schema", "id", "status", "shared_task_id", "family", "hypothesis",
    "dataset", "model", "replicas", "compute_budget", "evaluation_firewall",
    "outcome_rule", "implementation", "development_evidence",
  ], "contract");
  if (contract.schema !== "nsrl.q22_solomon_prospective_contract.v1"
      || contract.id !== "solomon.q22-operation.prospective-v1"
      || contract.status !== "preregistered_not_run"
      || contract.shared_task_id !== "zero-solomon.q22-operation.v1"
      || contract.family !== "solomon") {
    fail("unexpected Q22 prospective contract identity");
  }
  if (JSON.stringify(contract.replicas.seeds) !== JSON.stringify([1, 2, 3])
      || contract.replicas.run_all !== true
      || contract.model.kind !== "sparse_integer_perceptron_class_head"
      || contract.model.schema !== "nsrl.q22_integer_class_head.v1"
      || contract.model.feature_count !== 8192
      || contract.model.epochs !== 4
      || contract.model.integer_only !== true) {
    fail("unexpected frozen Q22 model or replica settings");
  }
  if (contract.dataset.train_sha256 !== "815fac312664f49eaaa33942828ffa1511fd81091ccd88d47b4480b6c27a5fa4"
      || contract.dataset.eval_sha256 !== "9270ea2b72af90235407bd7924a0864b8eba35b2969e1657ed1c15bf04449519"
      || contract.dataset.encoded_train_sha256 !== "2f03c80eb286e64eeb22521a929184d6018991667e0ddd410b6e22acbb9bef61"
      || contract.dataset.train_records !== 9500
      || contract.dataset.eval_records !== 500) {
    fail("unexpected frozen Q22 data binding");
  }
  if (contract.compute_budget.network !== "off"
      || contract.compute_budget.paid_compute_allowed !== false
      || contract.compute_budget.max_wall_seconds_per_seed !== 60
      || contract.compute_budget.max_total_wall_seconds !== 180) {
    fail("unexpected Q22 compute budget");
  }
  if (contract.outcome_rule.minimum_seed_exact_rate_ppm !== 950000
      || contract.outcome_rule.minimum_all_seed_agreement_rate_ppm !== 1000000
      || contract.outcome_rule.all_seeds_must_pass !== true) {
    fail("unexpected Q22 outcome rule");
  }
  for (const source of contract.implementation.source_files) {
    const file = path.join(root, source.path);
    if (!fs.existsSync(file) || sha256File(file) !== source.sha256) {
      fail(`source hash mismatch: ${source.path}`);
    }
  }
  return {
    contract,
    contractPath,
    contractSha256: sha256File(contractPath),
  };
}

function run(program, args, timeout, label) {
  const result = spawnSync(program, args, {
    cwd: root,
    encoding: "utf8",
    timeout: timeout * 1000,
    env: { ...process.env, CARGO_NET_OFFLINE: "true" },
  });
  if (result.error) fail(`${label}: ${result.error.message}`);
  if (result.status !== 0) {
    fail(`${label} exited ${result.status}: ${(result.stderr || result.stdout).trim()}`);
  }
  return result.stdout;
}

function requireEmptyDirectory(directory) {
  if (fs.existsSync(directory) && fs.readdirSync(directory).length > 0) {
    fail(`output directory must be absent or empty: ${directory}`);
  }
  fs.mkdirSync(directory, { recursive: true });
}

function readPredictions(file) {
  const lines = fs.readFileSync(file, "utf8").trimEnd().split("\n");
  if (lines.shift() !== "id\tmodel_request") fail(`bad prediction header: ${file}`);
  const rows = new Map();
  for (const line of lines) {
    const fields = line.split("\t");
    if (fields.length !== 2 || rows.has(fields[0])) fail(`bad prediction row: ${file}`);
    rows.set(fields[0], fields[1]);
  }
  return rows;
}

function execute(options, registration) {
  for (const field of ["trainDataset", "eval", "outDir"]) {
    if (!options[field]) fail(`execution requires --${field.replace(/[A-Z]/g, (ch) => `-${ch.toLowerCase()}`)}`);
  }
  requireEmptyDirectory(options.outDir);
  const contract = registration.contract;
  if (sha256File(options.trainDataset) !== contract.dataset.train_sha256) {
    fail("training dataset SHA-256 mismatch");
  }
  if (sha256File(options.manifest) !== contract.dataset.manifest_sha256) {
    fail("Q22 manifest SHA-256 mismatch");
  }

  run(options.cargo, ["build", "--release", "--offline", "-p", "nsrl-eval", "-p", "nsrl-train", "--bin", "nsrl-eval", "--bin", "nsrl-q22-proposer"], 600, "build");
  const proposer = path.join(root, "target/release/nsrl-q22-proposer");
  const evaluator = path.join(root, "target/release/nsrl-eval");
  const frozenModels = [];
  const encodedTraining = path.join(options.outDir, "q22-solomon-train.txt");
  run(evaluator, [
    "q22-encode", "--manifest", options.manifest, "--dataset", options.trainDataset,
    "--out", encodedTraining,
  ], 60, "encode training data");
  if (sha256File(encodedTraining) !== contract.dataset.encoded_train_sha256) {
    fail("Solomon training encoding SHA-256 mismatch");
  }

  // This phase has no evaluation path in any child process. All three model
  // artifacts are written and hashed before the blinded evaluation is created.
  const trainingStarted = Date.now();
  for (const seed of contract.replicas.seeds) {
    const model = path.join(options.outDir, `seed${seed}.nsrlq22`);
    const trace = path.join(options.outDir, `seed${seed}.train.json`);
    run(proposer, [
      "train", "--manifest", options.manifest, "--encoded", encodedTraining,
      "--seed", String(seed), "--epochs", String(contract.model.epochs),
      "--model-out", model, "--trace-out", trace,
    ], contract.compute_budget.max_wall_seconds_per_seed, `train seed ${seed}`);
    const trainingTrace = JSON.parse(fs.readFileSync(trace, "utf8"));
    if (trainingTrace.seed !== seed
        || trainingTrace.epochs !== contract.model.epochs
        || trainingTrace.train_records !== contract.dataset.train_records
        || trainingTrace.encoded_dataset_sha256 !== contract.dataset.encoded_train_sha256
        || fs.statSync(model).size > contract.compute_budget.max_model_bytes_per_seed) {
      fail(`seed ${seed} exceeded the frozen training or model budget`);
    }
    frozenModels.push({
      seed,
      model: path.basename(model),
      model_sha256: sha256File(model),
      train_trace: path.basename(trace),
      train_trace_sha256: sha256File(trace),
    });
  }
  if (Date.now() - trainingStarted > contract.compute_budget.max_total_wall_seconds * 1000) {
    fail("three-seed training exceeded the frozen total wall-time budget");
  }
  const freeze = {
    schema: "nsrl.q22_model_freeze.v1",
    contract_sha256: registration.contractSha256,
    evaluation_opened: false,
    models: frozenModels,
  };
  const freezePath = path.join(options.outDir, "models-frozen.json");
  atomicWrite(freezePath, stableJson(freeze));

  // This is the first operation allowed to read evaluation bytes. Its output is
  // only the two-column ID/input surface accepted by the proposer.
  const blindInputs = path.join(options.outDir, "promotion-inputs.blind.tsv");
  run(evaluator, [
    "q22-blind", "--manifest", options.manifest, "--eval", options.eval,
    "--out", blindInputs,
  ], 60, "blind evaluation");
  const opened = {
    schema: "nsrl.q22_evaluation_open.v1",
    contract_sha256: registration.contractSha256,
    model_freeze_sha256: sha256File(freezePath),
    eval_sha256: sha256File(options.eval),
    blind_inputs_sha256: sha256File(blindInputs),
    retraining_allowed: false,
  };
  if (opened.eval_sha256 !== contract.dataset.eval_sha256) fail("evaluation SHA-256 mismatch");
  atomicWrite(path.join(options.outDir, "evaluation-opened.json"), stableJson(opened));

  const checks = [];
  const predictionSets = [];
  for (const frozen of frozenModels) {
    const model = path.join(options.outDir, frozen.model);
    if (sha256File(model) !== frozen.model_sha256) fail(`seed ${frozen.seed} model changed after freeze`);
    const predictions = path.join(options.outDir, `seed${frozen.seed}.predictions.tsv`);
    const predictTrace = path.join(options.outDir, `seed${frozen.seed}.predict.json`);
    run(proposer, [
      "predict", "--manifest", options.manifest, "--model", model,
      "--inputs", blindInputs, "--predictions-out", predictions,
      "--trace-out", predictTrace,
    ], 60, `predict seed ${frozen.seed}`);
    const checkText = run(evaluator, [
      "q22-check", "--manifest", options.manifest, "--eval", options.eval,
      "--predictions", predictions,
    ], 60, `check seed ${frozen.seed}`);
    const check = JSON.parse(checkText);
    atomicWrite(path.join(options.outDir, `seed${frozen.seed}.check.json`), stableJson(check));
    checks.push({ seed: frozen.seed, ...check });
    predictionSets.push(readPredictions(predictions));
  }

  const ids = [...predictionSets[0].keys()];
  if (predictionSets.some((rows) => rows.size !== ids.length || ids.some((id) => !rows.has(id)))) {
    fail("seed prediction identities disagree");
  }
  const agreements = ids.filter((id) => {
    const first = predictionSets[0].get(id);
    return predictionSets.every((rows) => rows.get(id) === first);
  }).length;
  const rates = checks.map((check) => check.operation_exact_rate_ppm);
  const minimumRate = Math.min(...rates);
  const meanRate = Math.floor(rates.reduce((sum, value) => sum + value, 0) / rates.length);
  const agreementRate = Math.floor((agreements * 1000000) / ids.length);
  const passed = checks.every((check) => check.operation_exact_rate_ppm >= contract.outcome_rule.minimum_seed_exact_rate_ppm)
    && agreementRate >= contract.outcome_rule.minimum_all_seed_agreement_rate_ppm;
  const result = {
    schema: "nsrl.q22_solomon_prospective_result.v1",
    contract_sha256: registration.contractSha256,
    model_freeze_sha256: sha256File(freezePath),
    eval_sha256: opened.eval_sha256,
    seeds: checks,
    minimum_operation_exact_rate_ppm: minimumRate,
    mean_operation_exact_rate_ppm: meanRate,
    all_seed_agreement_cases: agreements,
    all_seed_agreement_rate_ppm: agreementRate,
    family_passed: passed,
    outcome: passed ? "go" : "no_go",
  };
  atomicWrite(path.join(options.outDir, "result.json"), stableJson(result));
  console.log(JSON.stringify({
    metrics: {
      minimum_operation_exact_rate_ppm: minimumRate,
      mean_operation_exact_rate_ppm: meanRate,
      all_seed_agreement_rate_ppm: agreementRate,
      family_passed: passed ? 1 : 0,
    },
  }));
}

const options = parseArgs(process.argv.slice(2));
const registration = validateContract(JSON.parse(fs.readFileSync(options.contract, "utf8")), options.contract);
if (options.checkContract) {
  console.log(`Q22 Solomon prospective contract passed: ${registration.contractSha256}`);
} else {
  execute(options, registration);
}
