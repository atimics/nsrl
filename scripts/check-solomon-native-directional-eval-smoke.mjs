#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");

const defaults = {
  outDir: "",
  keepTemp: false,
  textIndex: "web/assets/solomon-spirit-text-signatures.tsv",
  seqLen: "384",
  maxWindows: "4",
  batchWindows: "1",
  evalMaxTargetsPerTaskPhase: "2",
  minContextSeqLen: "384",
  maxContextSeqLen: "768",
};
const expectedSmallModelShape = {
  d_model: 128,
  heads: 2,
  head_dim: 64,
  hidden_dim: 256,
  transformer_layers: 2,
};
const productNativeTaskTargetFloor = 72;

function usage() {
  console.log([
    "Usage: check-solomon-native-directional-eval-smoke.mjs [options]",
    "",
    "Builds the real Solomon v2 symbolic corpus, trains a tiny native attention",
    "model for a few windows, runs nsrl-solomon-attention eval, and requires",
    "task_phases plus directional multimodal task groups.",
    "",
    "Options:",
    "  --out-dir PATH",
    "  --keep-temp",
    "  --text-index PATH",
    "  --seq-len N",
    "  --max-windows N",
    "  --batch-windows N",
    "  --eval-max-targets-per-task-phase N",
    "  --min-context-seq-len N",
    "  --max-context-seq-len N",
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
    } else if (arg === "--seq-len") {
      config.seqLen = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-windows") {
      config.maxWindows = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--batch-windows") {
      config.batchWindows = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--eval-max-targets-per-task-phase") {
      config.evalMaxTargetsPerTaskPhase = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-context-seq-len") {
      config.minContextSeqLen = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-context-seq-len") {
      config.maxContextSeqLen = parsePositiveInteger(requireValue(argv, ++index, arg), arg);
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

function run(label, command, args, options = {}) {
  const result = childProcess.spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    ...options,
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

function assertSmallModelShape(trace, label, config) {
  const errors = [];
  for (const [field, expected] of Object.entries(expectedSmallModelShape)) {
    if (field === "head_dim") {
      continue;
    }
    const actual = Number(trace[field] || 0);
    if (actual !== expected) {
      errors.push(`${label} ${field} ${actual} != ${expected}`);
    }
  }
  const dModel = Number(trace.d_model || 0);
  const heads = Number(trace.heads || 0);
  const headDim = heads > 0 && dModel > 0 && dModel % heads === 0 ? dModel / heads : 0;
  if (headDim !== expectedSmallModelShape.head_dim) {
    errors.push(`${label} head_dim ${headDim} != ${expectedSmallModelShape.head_dim}`);
  }
  const contextSeqLen = Number(trace.context_seq_len || trace.seq_len || 0);
  const expectedSeqLen = Number(config.seqLen);
  const minContextSeqLen = Number(config.minContextSeqLen);
  const maxContextSeqLen = Number(config.maxContextSeqLen);
  if (contextSeqLen !== expectedSeqLen) {
    errors.push(`${label} context_seq_len ${contextSeqLen} != requested seq_len ${expectedSeqLen}`);
  }
  if (contextSeqLen < minContextSeqLen || contextSeqLen > maxContextSeqLen) {
    errors.push(`${label} context_seq_len ${contextSeqLen} outside ${minContextSeqLen}-${maxContextSeqLen}`);
  }
  if (errors.length > 0) {
    throw new Error(`native small-model shape failed:\n- ${errors.join("\n- ")}`);
  }
  return {
    ...expectedSmallModelShape,
    context_seq_len: contextSeqLen,
  };
}

function assertTrainScalingTrace(trace) {
  const errors = [];
  const batchMode = String(trace.batch_mode || "");
  const requestedWorkers = Number(trace.map_reduce_workers);
  const effectiveWorkers = Number(trace.effective_map_reduce_workers);
  const availableParallelism = Number(trace.available_parallelism);
  if (!["serial", "map-reduce"].includes(batchMode)) {
    errors.push(`batch_mode ${JSON.stringify(batchMode)} is not serial or map-reduce`);
  }
  if (!Number.isInteger(requestedWorkers) || requestedWorkers < 0) {
    errors.push(`map_reduce_workers ${JSON.stringify(trace.map_reduce_workers)} is not a non-negative integer`);
  }
  if (!Number.isInteger(effectiveWorkers) || effectiveWorkers < 1) {
    errors.push(`effective_map_reduce_workers ${JSON.stringify(trace.effective_map_reduce_workers)} is not positive`);
  }
  if (!Number.isInteger(availableParallelism) || availableParallelism < 1) {
    errors.push(`available_parallelism ${JSON.stringify(trace.available_parallelism)} is not positive`);
  }
  if (batchMode === "serial" && effectiveWorkers !== 1) {
    errors.push(`serial effective_map_reduce_workers ${effectiveWorkers} != 1`);
  }
  if (batchMode === "map-reduce" && requestedWorkers === 0 && effectiveWorkers !== availableParallelism) {
    errors.push(
      `map-reduce 0-auto effective_map_reduce_workers ${effectiveWorkers} != available_parallelism ${availableParallelism}`,
    );
  }
  if (batchMode === "map-reduce" && requestedWorkers > 0 && effectiveWorkers !== requestedWorkers) {
    errors.push(`map-reduce effective_map_reduce_workers ${effectiveWorkers} != requested ${requestedWorkers}`);
  }
  if (errors.length > 0) {
    throw new Error(`native train CPU-scaling trace failed:\n- ${errors.join("\n- ")}`);
  }
}

function assertIntegerTraceContract(trainTrace, evalTrace) {
  const errors = [];
  const numericLeaves = {
    train: countNumericLeaves(trainTrace, "train"),
    eval: countNumericLeaves(evalTrace, "eval"),
  };
  const trainFields = [
    "target_frequency_min_weight_q15",
    "argmax_margin_weight_q15",
    "initial_probability_error_q15",
    "final_probability_error_q15",
    "probability_error_delta_i64",
  ];
  const evalMetricFields = [
    "mean_target_margin_q8",
    "min_target_margin_q8",
    "probability_error_q15",
    "mean_probability_error_q15",
  ];
  if (trainTrace.schema !== "nsrl.solomon_attention_train_trace.v1") {
    errors.push(`train schema ${JSON.stringify(trainTrace.schema || "")} != nsrl.solomon_attention_train_trace.v1`);
  }
  if (evalTrace.schema !== "nsrl.solomon_attention_eval_trace.v1") {
    errors.push(`eval schema ${JSON.stringify(evalTrace.schema || "")} != nsrl.solomon_attention_eval_trace.v1`);
  }
  for (const [label, summary] of Object.entries(numericLeaves)) {
    if (summary.non_integer_numeric_paths.length > 0) {
      errors.push(`${label} trace has non-integer numeric leaves: ${summary.non_integer_numeric_paths.join(", ")}`);
    }
  }
  for (const field of trainFields) {
    if (!Number.isInteger(Number(trainTrace[field]))) {
      errors.push(`train trace ${field} ${JSON.stringify(trainTrace[field])} is not an integer`);
    }
  }
  const evalMetrics = collectEvalMetricObjects(evalTrace);
  if (evalMetrics.length === 0) {
    errors.push("eval trace has no metric objects");
  }
  for (const metric of evalMetrics) {
    for (const field of evalMetricFields) {
      if (!Number.isInteger(Number(metric.value?.[field]))) {
        errors.push(`eval trace ${metric.path}.${field} ${JSON.stringify(metric.value?.[field])} is not an integer`);
      }
    }
  }
  if (errors.length > 0) {
    throw new Error(`native integer train/eval trace failed:\n- ${errors.join("\n- ")}`);
  }
  return {
    ok: true,
    train_schema: trainTrace.schema || "",
    eval_schema: evalTrace.schema || "",
    q_formats: {
      logits: "i32_q8",
      probabilities: "i16_q15",
      probability_error: "q15",
      target_margin: "q8",
      train_delta: "i64",
    },
    train_required_fields: Object.fromEntries(trainFields.map((field) => [field, Number(trainTrace[field])])),
    eval_required_metric_fields: evalMetricFields,
    eval_metric_objects: evalMetrics.length,
    numeric_leaves: {
      train: numericLeaves.train.count,
      eval: numericLeaves.eval.count,
    },
    non_integer_numeric_paths: [
      ...numericLeaves.train.non_integer_numeric_paths,
      ...numericLeaves.eval.non_integer_numeric_paths,
    ],
  };
}

function countNumericLeaves(value, pathName, summary = { count: 0, non_integer_numeric_paths: [] }) {
  if (typeof value === "number") {
    summary.count += 1;
    if (!Number.isInteger(value) && summary.non_integer_numeric_paths.length < 20) {
      summary.non_integer_numeric_paths.push(pathName);
    }
    return summary;
  }
  if (!value || typeof value !== "object") {
    return summary;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => countNumericLeaves(item, `${pathName}[${index}]`, summary));
    return summary;
  }
  for (const [key, child] of Object.entries(value)) {
    countNumericLeaves(child, `${pathName}.${key}`, summary);
  }
  return summary;
}

function collectEvalMetricObjects(evalTrace) {
  const metrics = [];
  for (const key of ["total", "special", "prompt", "text", "image"]) {
    if (evalTrace[key] && typeof evalTrace[key] === "object" && !Array.isArray(evalTrace[key])) {
      metrics.push({ path: key, value: evalTrace[key] });
    }
  }
  for (const [task, metric] of Object.entries(evalTrace.tasks || {})) {
    metrics.push({ path: `tasks.${task}`, value: metric });
  }
  for (const [task, phases] of Object.entries(evalTrace.task_phases || {})) {
    for (const [phase, metric] of Object.entries(phases || {})) {
      metrics.push({ path: `task_phases.${task}.${phase}`, value: metric });
    }
  }
  return metrics;
}

function assertDirectionalEval(evalTrace, gateSummary) {
  const errors = [];
  if (evalTrace.schema !== "nsrl.solomon_attention_eval_trace.v1") {
    errors.push(`eval schema ${JSON.stringify(evalTrace.schema || "")} != nsrl.solomon_attention_eval_trace.v1`);
  }
  if (!evalTrace.task_phases || typeof evalTrace.task_phases !== "object" || Array.isArray(evalTrace.task_phases)) {
    errors.push("native eval is missing task_phases");
  }
  for (const headName of ["special_head", "text_head", "image_head"]) {
    const head = gateSummary.output_heads?.[headName] || {};
    if (head.source !== "nsrllmm-output-token-head") {
      errors.push(`output head ${headName} is not present`);
    }
    if (Number(head.stats?.targets || 0) <= 0) {
      errors.push(`output head ${headName} has no targets`);
    }
  }
  const groups = gateSummary.directional_groups?.groups || {};
  for (const group of [
    "text_prompt_to_image_plan",
    "seal_image_to_text",
    "text_and_seal_to_explanation",
    "identity_source_binding",
  ]) {
    const summary = groups[group] || {};
    if (summary.ok !== true) {
      errors.push(`directional group ${group} is not ok`);
    }
    const stats = summary.stats || {};
    for (const field of ["targets", "accuracy_per_mille", "top5_accuracy_per_mille", "top10_accuracy_per_mille"]) {
      if (!Number.isFinite(Number(stats[field]))) {
        errors.push(`directional group ${group} stats.${field} is missing`);
      }
    }
    if (summary.min_top5_accuracy_per_mille !== 1) {
      errors.push(`directional group ${group} min_top5_accuracy_per_mille ${summary.min_top5_accuracy_per_mille} != 1`);
    }
  }
  if (errors.length > 0) {
    throw new Error(`native directional eval smoke failed:\n- ${errors.join("\n- ")}`);
  }
}

function nativeEvalScope(config) {
  const evalMaxTargetsPerTaskPhase = Number(config.evalMaxTargetsPerTaskPhase);
  const productScale = evalMaxTargetsPerTaskPhase >= productNativeTaskTargetFloor;
  return {
    proof_scope: productScale ? "product-scale-native-eval" : "local-directional-smoke",
    eval_max_examples: "none",
    eval_max_targets_per_task_phase: evalMaxTargetsPerTaskPhase,
    smoke_min_task_targets: "all=1",
    smoke_min_phase_targets: "special=1,prompt=1,text=1,image=1",
    smoke_min_direction_top5_per_mille: "all=1",
    product_min_task_targets: `all=${productNativeTaskTargetFloor}`,
    product_min_phase_targets: `all=${productNativeTaskTargetFloor}`,
    product_scale: productScale,
  };
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = config.outDir
    ? path.resolve(config.outDir)
    : fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-native-directional-eval-"));
  const corpusDir = path.join(root, "corpus");
  const modelPath = path.join(root, "model.nsrllmm");
  const trainPath = path.join(root, "train.json");
  const evalPath = path.join(root, "attention-eval.json");
  const gatePath = path.join(root, "task-eval-check.json");

  fs.mkdirSync(root, { recursive: true });
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

    const train = run("train native attention", "cargo", [
      "run",
      "--quiet",
      "-p",
      "nsrl-train",
      "--bin",
      "nsrl-solomon-attention",
      "--",
      "train",
      "--tokens",
      path.join(corpusDir, "corpus.tokens.u8"),
      "--model-out",
      modelPath,
      "--epochs",
      "1",
      "--seq-len",
      config.seqLen,
      "--stride",
      "1",
      "--window-offset",
      "0",
      "--batch-windows",
      config.batchWindows,
      "--batch-mode",
      "serial",
      "--map-reduce-workers",
      "1",
      "--text-token-profile",
      "chunked",
      "--target-segment",
      "all",
      "--max-windows",
      config.maxWindows,
    ]);
    fs.writeFileSync(trainPath, train.stdout);
    const trainTrace = JSON.parse(train.stdout);
    assertSmallModelShape(trainTrace, "train trace", config);
    assertTrainScalingTrace(trainTrace);

    const evalRun = run("native attention eval", "cargo", [
      "run",
      "--quiet",
      "-p",
      "nsrl-train",
      "--bin",
      "nsrl-solomon-attention",
      "--",
      "eval",
      "--model",
      modelPath,
      "--tokens",
      path.join(corpusDir, "corpus.tokens.u8"),
      "--conditioning-examples",
      path.join(corpusDir, "examples.jsonl"),
      "--eval-max-examples",
      "none",
      "--eval-max-targets-per-task-phase",
      config.evalMaxTargetsPerTaskPhase,
    ]);
    fs.writeFileSync(evalPath, evalRun.stdout);

    const gate = run("check native directional eval", process.execPath, [
      "scripts/check-solomon-attention-task-eval.mjs",
      "--eval",
      evalPath,
      "--require-tasks",
      "canonical-joint,identify,text-to-image,image-to-text,image-to-explain,text-image-explain,image-to-attributes,explain,description-to-image,match",
      "--min-task-targets",
      "all=1",
      "--min-phase-targets",
      "special=1,prompt=1,text=1,image=1",
      "--min-direction-top5",
      "all=1",
      "--require-directional-groups",
    ]);
    fs.writeFileSync(gatePath, gate.stdout);

    const evalTrace = readJson(evalPath);
    const gateSummary = JSON.parse(gate.stdout);
    const architecture = assertSmallModelShape(evalTrace, "eval trace", config);
    const integerTrace = assertIntegerTraceContract(trainTrace, evalTrace);
    assertDirectionalEval(evalTrace, gateSummary);

    console.log(JSON.stringify({
      schema: "nsrl.solomon_native_directional_eval_smoke.v1",
      ok: true,
      artifacts_kept: config.keepTemp || Boolean(config.outDir),
      out_dir: root,
      eval: evalPath,
      task_eval_check: gatePath,
      model_hash: evalTrace.model_hash || "",
      eval_scope: nativeEvalScope(config),
      architecture,
      integer_trace: integerTrace,
      output_heads: {
        special: gateSummary.output_heads?.special_head || null,
        text: gateSummary.output_heads?.text_head || null,
        image: gateSummary.output_heads?.image_head || null,
      },
      tasks: gateSummary.tasks || {},
      task_phases: gateSummary.task_phases || {},
      task_phase_tasks: Object.keys(evalTrace.task_phases || {}).length,
      directional_groups: Object.fromEntries(
        Object.entries(gateSummary.directional_groups?.groups || {}).map(([key, value]) => [
          key,
          {
            ok: value.ok === true,
            targets: Number(value.targets || 0),
            stats: value.stats || {},
            min_targets: value.min_targets ?? null,
            min_accuracy_per_mille: value.min_accuracy_per_mille ?? null,
            min_top5_accuracy_per_mille: value.min_top5_accuracy_per_mille ?? null,
            min_top10_accuracy_per_mille: value.min_top10_accuracy_per_mille ?? null,
            phase_targets: value.phase_targets || {},
          },
        ]),
      ),
    }, null, 2));
  } finally {
    if (!config.keepTemp && !config.outDir) {
      fs.rmSync(root, { recursive: true, force: true });
    }
  }
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
