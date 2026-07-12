#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const AUTHORS = ["crowley", "shakespeare", "blake"];
const AUTHOR_INDEX = Object.fromEntries(AUTHORS.map((author, index) => [author, index]));
const ROUTER_REPLICAS = ["semantic", "structural", "full"];

function parseArgs(argv) {
  const options = {
    sourceManifest: "data/local-runs/literary-scale-8k-seq32-fixed/corpus.manifest.json",
    outDir: "data/experiments/literary-recursive-swarm-v1",
    routerTrainBytes: 16_384,
    routerCalibrationBytes: 8_192,
    finalTestBytes: 8_192,
    promptChars: 256,
    promptStrideChars: 128,
    leafWindows: 8_192,
    seqLen: 32,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      const value = argv[++index];
      if (value === undefined) throw new Error(`${arg} requires a value`);
      return value;
    };
    if (arg === "--source-manifest") options.sourceManifest = next();
    else if (arg === "--out-dir") options.outDir = next();
    else if (arg === "--router-train-bytes") options.routerTrainBytes = Number.parseInt(next(), 10);
    else if (arg === "--router-calibration-bytes") options.routerCalibrationBytes = Number.parseInt(next(), 10);
    else if (arg === "--final-test-bytes") options.finalTestBytes = Number.parseInt(next(), 10);
    else if (arg === "--prompt-chars") options.promptChars = Number.parseInt(next(), 10);
    else if (arg === "--prompt-stride-chars") options.promptStrideChars = Number.parseInt(next(), 10);
    else if (arg === "--leaf-windows") options.leafWindows = Number.parseInt(next(), 10);
    else if (arg === "--seq-len") options.seqLen = Number.parseInt(next(), 10);
    else if (arg === "--help" || arg === "-h") {
      console.log("Usage: node scripts/build-recursive-literary-swarm-experiment.mjs [--source-manifest PATH] [--out-dir PATH] [--router-train-bytes N] [--router-calibration-bytes N] [--final-test-bytes N] [--leaf-windows N] [--seq-len N]");
      process.exit(0);
    } else throw new Error(`unknown argument: ${arg}`);
  }
  for (const [name, value] of Object.entries(options)) {
    if (name.endsWith("Bytes") || ["promptChars", "promptStrideChars", "leafWindows", "seqLen"].includes(name)) {
      if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
    }
  }
  return options;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function cleanText(input) {
  let text = input.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const start = text.search(/^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (start >= 0) text = text.slice(text.indexOf("\n", start) + 1);
  const end = text.search(/^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*$/im);
  if (end >= 0) text = text.slice(0, end);
  return text.normalize("NFC").replace(/[\t ]+$/gm, "").replace(/\n{4,}/g, "\n\n\n").trim();
}

function safeUtf8End(buffer, requestedEnd) {
  let end = Math.min(requestedEnd, buffer.length);
  while (end > 0 && end < buffer.length && (buffer[end] & 0xc0) === 0x80) end -= 1;
  return end;
}

function splitBalancedText(text, sizes) {
  const buffer = Buffer.from(text, "utf8");
  const result = {};
  let start = 0;
  for (const [name, requestedBytes] of sizes) {
    const end = safeUtf8End(buffer, start + requestedBytes);
    const raw = buffer.subarray(start, end);
    result[name] = {
      text: raw.toString("utf8").trim(),
      byteRange: [start, end],
      rawBytes: raw.length,
    };
    start = end;
  }
  if (start !== buffer.length) throw new Error("split sizes do not consume the balanced text");
  return result;
}

function rotateParagraphs(text, variant) {
  const paragraphs = text.split(/\n\s*\n/).map((value) => value.trim()).filter(Boolean);
  if (paragraphs.length < 3 || variant === 0) return text;
  const pivot = Math.floor((paragraphs.length * variant) / 3);
  return [...paragraphs.slice(pivot), ...paragraphs.slice(0, pivot)].join("\n\n");
}

function routerFeaturesQ15(prompt) {
  const bytes = Buffer.from(prompt.toLowerCase(), "utf8");
  const buckets = Array(24).fill(0);
  for (let index = 1; index < bytes.length; index += 1) {
    const bucket = ((bytes[index - 1] * 257) + bytes[index]) % buckets.length;
    buckets[bucket] += 1;
  }
  const pairCount = Math.max(1, bytes.length - 1);
  const ratios = [
    (byte) => byte >= 97 && byte <= 122,
    (byte) => byte >= 48 && byte <= 57,
    (byte) => byte === 32 || byte === 9,
    (byte) => byte === 10,
    (byte) => ",.;:!?".includes(String.fromCharCode(byte)),
    (byte) => "aeiouy".includes(String.fromCharCode(byte)),
    (byte) => byte === 39 || byte === 34,
    (byte) => byte < 32 || byte > 126,
  ];
  const normalize = (count, total) => Math.min(32_767, Math.round((count * 32_767) / Math.max(1, total)));
  return [
    ...buckets.map((count) => normalize(count, pairCount)),
    ...ratios.map((predicate) => normalize([...bytes].filter(predicate).length, bytes.length)),
  ];
}

function promptRows(text, split, author, options, candidates, bootstrapTarget) {
  const rows = [];
  for (let start = 0; start + options.promptChars <= text.length; start += options.promptStrideChars) {
    const prompt = text.slice(start, start + options.promptChars);
    const sampleHash = sha256(`${split}\0${author}\0${start}\0${prompt}`).slice(0, 16);
    rows.push({
      schema: "nsrl.recursive_router_sample.v1",
      sample_id: `${split}-${author}-${sampleHash}`,
      split,
      source_author: author,
      prompt,
      features_q15: routerFeaturesQ15(prompt),
      candidate_ids: candidates,
      bootstrap_target: bootstrapTarget,
      oracle_target: null,
      oracle_child_losses_q15: null,
    });
  }
  return rows;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const sourceManifest = JSON.parse(await readFile(options.sourceManifest, "utf8"));
  const texts = {};
  const sourceFiles = {};
  for (const author of AUTHORS) {
    const entries = sourceManifest.sources?.[author];
    if (!Array.isArray(entries) || entries.length === 0) throw new Error(`source manifest has no ${author} sources`);
    const parts = [];
    sourceFiles[author] = [];
    for (const entry of entries) {
      const raw = await readFile(entry.path, "utf8");
      const cleaned = cleanText(raw);
      parts.push(cleaned);
      sourceFiles[author].push({ path: entry.path, sha256: sha256(raw), cleaned_bytes: Buffer.byteLength(cleaned) });
    }
    texts[author] = parts.join("\n\n");
  }

  const availableBytes = Object.fromEntries(AUTHORS.map((author) => [author, Buffer.byteLength(texts[author])]));
  const balancedBytes = Math.min(...Object.values(availableBytes));
  const reservedBytes = options.routerTrainBytes + options.routerCalibrationBytes + options.finalTestBytes;
  if (reservedBytes >= balancedBytes) throw new Error("router/final splits leave no leaf training data");
  const leafTrainBytes = balancedBytes - reservedBytes;
  const outDir = path.resolve(options.outDir);
  await mkdir(outDir, { recursive: true });

  const splitManifest = {};
  const leafJobs = [];
  const rootRows = { router_train: [], router_calibration: [], final_test: [] };
  const localRows = Object.fromEntries(AUTHORS.map((author) => [author, { router_train: [], router_calibration: [], final_test: [] }]));

  for (const author of AUTHORS) {
    const balanced = Buffer.from(texts[author], "utf8").subarray(0, safeUtf8End(Buffer.from(texts[author], "utf8"), balancedBytes)).toString("utf8");
    const consumedBytes = Buffer.byteLength(balanced);
    const split = splitBalancedText(balanced, [
      ["leaf_train", consumedBytes - reservedBytes],
      ["router_train", options.routerTrainBytes],
      ["router_calibration", options.routerCalibrationBytes],
      ["final_test", options.finalTestBytes],
    ]);
    const authorDir = path.join(outDir, "splits", author);
    await mkdir(authorDir, { recursive: true });
    splitManifest[author] = {};
    for (const [name, value] of Object.entries(split)) {
      const file = path.join(authorDir, `${name.replaceAll("_", "-")}.txt`);
      await writeFile(file, `${value.text}\n`);
      splitManifest[author][name] = {
        path: file,
        tokens_path: path.join(authorDir, `${name.replaceAll("_", "-")}.tokens.u8`),
        tokens_trace_path: path.join(authorDir, `${name.replaceAll("_", "-")}.tokens.trace.jsonl`),
        raw_byte_range: value.byteRange,
        raw_bytes: value.rawBytes,
        output_bytes: Buffer.byteLength(`${value.text}\n`),
        sha256: sha256(`${value.text}\n`),
      };
    }

    const expertIds = [0, 1, 2].map((variant) => `${author}-expert-${variant}`);
    for (let variant = 0; variant < 3; variant += 1) {
      const rotated = `${rotateParagraphs(split.leaf_train.text, variant)}\n`;
      const textPath = path.join(authorDir, `leaf-train-v${variant}.txt`);
      const tokensPath = path.join(authorDir, `leaf-train-v${variant}.tokens.u8`);
      await writeFile(textPath, rotated);
      leafJobs.push({
        expert_id: expertIds[variant],
        author,
        variant,
        text_path: textPath,
        tokens_path: tokensPath,
        model_path: path.join(outDir, "leaves", expertIds[variant], "model.nsrlmt"),
        train_trace_path: path.join(outDir, "leaves", expertIds[variant], "train.trace.jsonl"),
        seq_len: options.seqLen,
        max_windows: options.leafWindows,
        stride: Math.max(1, Math.ceil(Buffer.byteLength(rotated) / options.leafWindows)),
        window_offset: variant * 4,
      });
    }

    for (const splitName of Object.keys(rootRows)) {
      rootRows[splitName].push(...promptRows(
        split[splitName].text,
        splitName,
        author,
        options,
        AUTHORS.map((candidate) => `author-${candidate}-pod`),
        AUTHOR_INDEX[author],
      ));
      localRows[author][splitName].push(...promptRows(
        split[splitName].text,
        splitName,
        author,
        options,
        expertIds,
        null,
      ));
    }
  }

  const routerDataDir = path.join(outDir, "router-data");
  await mkdir(routerDataDir, { recursive: true });
  const routerDatasets = { root: {}, local: {} };
  for (const [split, rows] of Object.entries(rootRows)) {
    const file = path.join(routerDataDir, `root-${split.replaceAll("_", "-")}.jsonl`);
    await writeFile(file, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
    routerDatasets.root[split] = { path: file, rows: rows.length, sha256: sha256(`${rows.map((row) => JSON.stringify(row)).join("\n")}\n`) };
  }
  for (const author of AUTHORS) {
    routerDatasets.local[author] = {};
    for (const [split, rows] of Object.entries(localRows[author])) {
      const file = path.join(routerDataDir, `${author}-${split.replaceAll("_", "-")}.jsonl`);
      const content = `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`;
      await writeFile(file, content);
      routerDatasets.local[author][split] = { path: file, rows: rows.length, sha256: sha256(content) };
    }
  }

  const nodes = AUTHORS.map((author) => ({
    node_id: `author-${author}-pod`,
    kind: "router_triad",
    children: [0, 1, 2].map((variant) => `${author}-expert-${variant}`),
    router_replicas: ROUTER_REPLICAS.map((view) => `${author}-router-${view}`),
    consensus: "q15_rank_sum_then_confidence",
    beam_width: 2,
    bootstrap_target: null,
    oracle_target_source: `router-data/${author}-*.jsonl after leaf scoring`,
  }));
  nodes.push({
    node_id: "root-literary-pod",
    kind: "router_triad",
    children: AUTHORS.map((author) => `author-${author}-pod`),
    router_replicas: ROUTER_REPLICAS.map((view) => `root-router-${view}`),
    consensus: "q15_rank_sum_then_confidence",
    beam_width: 2,
    bootstrap_target: "source_author",
    oracle_target_source: "root router rows; replace bootstrap with measured child-pod utility before final claims",
  });

  const routerJobs = [];
  for (const node of nodes) {
    const isRoot = node.node_id === "root-literary-pod";
    const author = isRoot ? null : node.node_id.replace(/^author-/, "").replace(/-pod$/, "");
    const datasets = isRoot ? routerDatasets.root : routerDatasets.local[author];
    for (const featureView of ROUTER_REPLICAS) {
      const routerId = isRoot ? `root-router-${featureView}` : `${author}-router-${featureView}`;
      routerJobs.push({
        router_id: routerId,
        node_id: node.node_id,
        feature_view: featureView,
        feature_indices: featureView === "semantic"
          ? "0-23"
          : featureView === "structural"
            ? "24-40"
            : "0-40",
        train_dataset: datasets.router_train.path,
        calibration_dataset: datasets.router_calibration.path,
        final_test_dataset: datasets.final_test.path,
        target_field: isRoot ? "bootstrap_target_then_oracle_target" : "oracle_target",
        warm_start_ready: isRoot,
        final_training_ready: false,
        model_path: path.join(outDir, "routers", routerId, "router.nsrlrt"),
        trace_path: path.join(outDir, "routers", routerId, "train.trace.jsonl"),
      });
    }
  }

  const leafJobsPath = path.join(outDir, "leaf-jobs.tsv");
  const header = ["expert_id", "author", "variant", "text_path", "tokens_path", "model_path", "train_trace_path", "seq_len", "max_windows", "stride", "window_offset"];
  await writeFile(leafJobsPath, `${header.join("\t")}\n${leafJobs.map((job) => header.map((key) => job[key]).join("\t")).join("\n")}\n`);
  const routerJobsPath = path.join(outDir, "router-jobs.tsv");
  const routerHeader = ["router_id", "node_id", "feature_view", "feature_indices", "train_dataset", "calibration_dataset", "final_test_dataset", "target_field", "warm_start_ready", "final_training_ready", "model_path", "trace_path"];
  await writeFile(routerJobsPath, `${routerHeader.join("\t")}\n${routerJobs.map((job) => routerHeader.map((key) => job[key]).join("\t")).join("\n")}\n`);

  const manifest = {
    schema: "nsrl.recursive_literary_swarm_experiment.v1",
    experiment_id: path.basename(outDir),
    topology: "depth2_ternary_router_triads",
    root_node_id: "root-literary-pod",
    authors: AUTHORS,
    router_feature_schema: { base_dimensions: 32, child_probe_dimensions: 9, routed_dimensions: 41, dtype: "q15_i16", hash_buckets: 24, structural_ratios: 8, child_probes: ["quality_by_child", "prefix_accuracy_by_child", "relative_loss_advantage_by_child"] },
    routing_contract: {
      routers_per_node: 3,
      children_per_node: 3,
      beam_width: 2,
      consensus: "fixed_integer_consensus_terminates_router_recursion",
      path_score: "sum_base2_route_scores_plus_leaf_confidence",
      flat_heuristic_router_role: "baseline_only",
    },
    split_contract: {
      order: ["leaf_train", "router_train", "router_calibration", "final_test"],
      balanced_bytes: balancedBytes,
      leaf_train_bytes: leafTrainBytes,
      router_train_bytes: options.routerTrainBytes,
      router_calibration_bytes: options.routerCalibrationBytes,
      final_test_bytes: options.finalTestBytes,
      final_test_is_frozen: true,
    },
    available_bytes: availableBytes,
    sources: sourceFiles,
    splits: splitManifest,
    leaf_jobs_path: leafJobsPath,
    leaf_jobs: leafJobs,
    router_jobs_path: routerJobsPath,
    router_jobs: routerJobs,
    router_datasets: routerDatasets,
    nodes,
    required_next_artifacts: [
      "nine leaf checkpoints and cross-evaluation matrix",
      "local router oracle labels from measured per-sample leaf utility",
      "three trained router replicas per node",
      "root oracle labels from measured child-pod utility",
      "frozen final-test routed-vs-flat-vs-oracle report",
    ],
  };
  const manifestPath = path.join(outDir, "experiment.manifest.json");
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(manifestPath);
}

main().catch((error) => {
  console.error(`build-recursive-literary-swarm-experiment: ${error.message}`);
  process.exit(1);
});
