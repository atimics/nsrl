#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const defaults = {
  runDir: "",
  stages: "",
  stageDirs: [],
  requiredStageNames: [],
  outPath: "",
  minStages: 1,
  requireLossNonIncreasing: true,
  minNativeBindEpochs: 2,
};

const STAGE_FILTERS = {
  identity: { tasks: "identify,image-to-text,explain" },
  image: { tasks: "text-to-image,description-to-image,image-to-text" },
  "text-to-image": { tasks: "text-to-image,description-to-image" },
  "description-to-image": { tasks: "description-to-image" },
  "image-to-text": { tasks: "image-to-text,image-to-explain,text-image-explain,image-to-attributes" },
  explain: { tasks: "explain,image-to-explain,text-image-explain,image-to-attributes" },
  match: { tasks: "match" },
  "hard-negative": { tasks: "match", match_labels: "no", match_roles: "image,prompt" },
  "native-bind": {
    tasks:
      "canonical-joint,identify,text-to-image,description-to-image,image-to-text,image-to-explain,text-image-explain,image-to-attributes,explain",
  },
  all: { tasks: "canonical-joint,identify,text-to-image,image-to-text,image-to-explain,text-image-explain,image-to-attributes,explain,description-to-image,match" },
};

const STAGE_IDENTITY_BINDING_TASKS = {
  identity: ["identify"],
  image: ["text-to-image"],
  "text-to-image": ["text-to-image"],
  "native-bind": ["identify", "text-to-image"],
  all: ["identify", "text-to-image"],
};

const REQUIRED_IDENTITY_BINDING_KINDS = [
  "primary-name",
  "primary-seal",
  "alias",
  "alias-seal",
  "seal-id",
];
const TASK_TOKEN_LAYOUT_FALLBACK = {
  bos: 1,
  prompt: 2,
  text: 3,
  image: 4,
  eos: 5,
  task_text_to_image: 6,
  task_image_to_text: 7,
  task_match: 8,
  task_explain: 9,
  task_identify: 10,
  image_channel_ink: 11,
  image_channel_edge: 12,
  image_channel_component: 13,
  image_channel_radial: 14,
  image_channel_direction: 15,
  image_base: 144,
  image_bins: 16,
};
const IMAGE_CHANNEL_PAYLOAD_TOKENS_FALLBACK = 256;
const IMAGE_BEARING_TASKS = new Set([
  "text-to-image",
  "description-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "match",
]);

const STAGE_EVIDENCE_TASKS = {
  identity: {
    required: ["identify", "image-to-text", "explain"],
  },
  image: {
    required: ["text-to-image", "description-to-image", "image-to-text"],
    image_plan: ["text-to-image", "description-to-image"],
    image_classification: ["image-to-text"],
  },
  "text-to-image": {
    required: ["text-to-image", "description-to-image"],
    image_plan: ["text-to-image", "description-to-image"],
  },
  "description-to-image": {
    required: ["description-to-image"],
    image_plan: ["description-to-image"],
  },
  "image-to-text": {
    required: ["image-to-text", "image-to-explain", "text-image-explain", "image-to-attributes"],
    image_classification: ["image-to-text"],
    image_grounding: ["image-to-explain", "text-image-explain", "image-to-attributes"],
  },
  explain: {
    required: ["explain", "image-to-explain", "text-image-explain", "image-to-attributes"],
    image_grounding: ["image-to-explain", "text-image-explain", "image-to-attributes"],
  },
  match: {
    required: ["match"],
    match_labels: ["yes", "no"],
  },
  "hard-negative": {
    required: ["match"],
    match_labels: ["no"],
    match_roles: ["image", "prompt"],
  },
  "native-bind": {
    required: [
      "canonical-joint",
      "identify",
      "text-to-image",
      "description-to-image",
      "image-to-text",
      "image-to-explain",
      "text-image-explain",
      "image-to-attributes",
      "explain",
    ],
    image_plan: ["text-to-image", "description-to-image"],
    image_classification: ["image-to-text"],
    image_grounding: ["image-to-explain", "text-image-explain", "image-to-attributes"],
  },
  all: {
    required: [
      "canonical-joint",
      "identify",
      "text-to-image",
      "image-to-text",
      "image-to-explain",
      "text-image-explain",
      "image-to-attributes",
      "explain",
      "description-to-image",
      "match",
    ],
  },
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-v2-curriculum-stages.mjs --stage-dir PATH [--stage-dir PATH...]",
      "   or: check-solomon-v2-curriculum-stages.mjs --run-dir PATH --stages identity,image,text-to-image,description-to-image,image-to-text,explain,hard-negative,native-bind",
      "",
      "Checks filtered v2 Solomon curriculum stage corpora and their train traces.",
      "",
      "Options:",
      "  --out PATH",
      "  --min-stages N",
      "  --min-native-bind-epochs N",
      "  --require-stage-names LIST",
      "  --allow-loss-increase",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { ...defaults, stageDirs: [], requiredStageNames: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--run-dir") {
      config.runDir = requireValue(argv, ++index, arg);
    } else if (arg === "--stages") {
      config.stages = requireValue(argv, ++index, arg);
    } else if (arg === "--stage-dir") {
      config.stageDirs.push(requireValue(argv, ++index, arg));
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--min-stages") {
      config.minStages = parsePositive(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-native-bind-epochs") {
      config.minNativeBindEpochs = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-stage-names") {
      config.requiredStageNames = stageNames(requireValue(argv, ++index, arg)).map(canonicalStageName);
    } else if (arg === "--allow-loss-increase") {
      config.requireLossNonIncreasing = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (config.stageDirs.length === 0) {
    if (!config.runDir || !config.stages) {
      throw new Error("--stage-dir or --run-dir with --stages is required");
    }
    config.stageDirs = stageNames(config.stages).map((stage, index) =>
      path.join(config.runDir, `v2-stage-${index}-${stage}`),
    );
  }
  if (config.requiredStageNames.length === 0 && config.stages) {
    config.requiredStageNames = stageNames(config.stages).map(canonicalStageName);
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
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${flag} requires a positive integer`);
  }
  return Number(value);
}

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function stageNames(value) {
  const stages = String(value)
    .split(",")
    .map((stage) => stage.trim())
    .filter(Boolean);
  if (stages.length === 0) {
    throw new Error("--stages selected no stages");
  }
  return stages;
}

function canonicalStageName(stageName) {
  if (stageName === "hard-negatives") {
    return "hard-negative";
  }
  return stageName;
}

function inferStageName(stageDir) {
  const basename = path.basename(stageDir);
  const match = /^v2-stage-[0-9]+-(.+)$/.exec(basename);
  return match ? canonicalStageName(match[1]) : "";
}

function checkStageRecipe(stageDir, index, manifest, config, errors) {
  const expectedStageName = config.requiredStageNames[index] || "";
  const inferredStageName = inferStageName(stageDir);
  const stageName = inferredStageName || expectedStageName;
  if (!expectedStageName) {
    return { stageName, expectedStageName };
  }
  if (inferredStageName && inferredStageName !== expectedStageName) {
    errors.push(`stage name ${JSON.stringify(inferredStageName)} != expected ${JSON.stringify(expectedStageName)}`);
  }
  const expectedFilter = STAGE_FILTERS[expectedStageName];
  if (!expectedFilter) {
    errors.push(`unknown required stage name ${JSON.stringify(expectedStageName)}`);
    return { stageName, expectedStageName };
  }
  for (const [key, expectedValue] of Object.entries(expectedFilter)) {
    const actualValue = String(manifest.filter?.[key] || "");
    if (actualValue !== expectedValue) {
      errors.push(`filter ${key} ${JSON.stringify(actualValue)} != ${JSON.stringify(expectedValue)} for stage ${expectedStageName}`);
    }
  }
  return { stageName, expectedStageName };
}

function readJson(filePath, errors) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    errors.push(`${filePath}: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}

function readJsonl(filePath, errors) {
  try {
    const text = fs.readFileSync(filePath, "utf8").trimEnd();
    if (!text) {
      return [];
    }
    return text.split(/\r?\n/).filter(Boolean).map((line, rowIndex) => {
      const row = JSON.parse(line);
      row.__line = rowIndex + 1;
      return row;
    });
  } catch (error) {
    errors.push(`${filePath}: ${error instanceof Error ? error.message : String(error)}`);
    return [];
  }
}

function fnv64ByteHex(bytes) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function resolveStagePath(stageDir, ref, fallback) {
  const selected = ref || fallback;
  return path.isAbsolute(selected) ? selected : path.resolve(stageDir, selected);
}

function firstExistingReferencedPath(ref, stageDir, sourceDir) {
  if (!ref) {
    return "";
  }
  const candidates = path.isAbsolute(ref)
    ? [ref]
    : [
        path.resolve(ref),
        path.resolve(stageDir, ref),
        sourceDir ? path.resolve(sourceDir, ref) : "",
        sourceDir ? path.resolve(sourceDir, path.basename(ref)) : "",
      ];
  for (const candidate of candidates.filter(Boolean)) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return path.isAbsolute(ref) ? ref : path.resolve(ref);
}

function hashReferencedFile(ref, stageDir, sourceDir, label, errors) {
  const resolved = firstExistingReferencedPath(ref, stageDir, sourceDir);
  if (!resolved || !fs.existsSync(resolved)) {
    errors.push(`${label} ${ref || ""} not found`);
    return "";
  }
  try {
    return fnv64ByteHex(fs.readFileSync(resolved));
  } catch (error) {
    errors.push(`${label} ${ref}: ${error instanceof Error ? error.message : String(error)}`);
    return "";
  }
}

function checkStageSourceCorpus(stageDir, manifest, errors) {
  const sourceDir = manifest.source_dir || "";
  const sourceExamples = manifest.source_examples_jsonl || "";
  const sourceTokens = manifest.source_corpus_tokens_u8 || "";
  const source = {
    source_dir: sourceDir,
    source_manifest_schema: manifest.source_manifest_schema || "",
    source_examples: sourceExamples,
    source_examples_hash: "",
    source_tokens: sourceTokens,
    source_tokens_hash: "",
  };
  if (!sourceExamples) {
    errors.push("manifest missing source_examples_jsonl");
  } else {
    source.source_examples_hash = hashReferencedFile(sourceExamples, stageDir, sourceDir, "source examples", errors);
  }
  if (!sourceTokens) {
    errors.push("manifest missing source_corpus_tokens_u8");
  } else {
    source.source_tokens_hash = hashReferencedFile(sourceTokens, stageDir, sourceDir, "source tokens", errors);
  }
  return source;
}

function expectedTaskMarker(task, layout) {
  const bos = Number(layout.bos ?? TASK_TOKEN_LAYOUT_FALLBACK.bos);
  const prompt = Number(layout.prompt ?? TASK_TOKEN_LAYOUT_FALLBACK.prompt);
  const image = Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image);
  if (task === "identify") return [bos, Number(layout.task_identify), prompt];
  if (task === "text-to-image" || task === "description-to-image") {
    return [bos, Number(layout.task_text_to_image), prompt];
  }
  if (task === "image-to-text") return [bos, Number(layout.task_image_to_text), image];
  if (task === "image-to-explain" || task === "image-to-attributes") {
    return [bos, Number(layout.task_explain), image];
  }
  if (task === "text-image-explain" || task === "explain") {
    return [bos, Number(layout.task_explain), prompt];
  }
  if (task === "match") return [bos, Number(layout.task_match), prompt];
  return null;
}

function markersMatch(actual, expected) {
  if (actual.length !== expected.length) {
    return false;
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (Number(actual[index]) !== Number(expected[index])) {
      return false;
    }
  }
  return true;
}

function ensureTaskIntegritySummary(map, task) {
  if (!map.has(task)) {
    map.set(task, {
      checked_records: 0,
      hash_mismatches: 0,
      marker_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
    });
  }
  return map.get(task);
}

function ensureTaskModalitySummary(map, task) {
  if (!map.has(task)) {
    map.set(task, {
      checked_records: 0,
      missing_offsets: 0,
      out_of_bounds: 0,
      modality_mismatches: 0,
    });
  }
  return map.get(task);
}

function checkStageTaskMarkerIntegrity(stageDir, manifest, tokenPath, errors) {
  const examplesPath = resolveStagePath(stageDir, manifest.examples_jsonl, "examples.jsonl");
  const rows = readJsonl(examplesPath, errors);
  let tokens = Buffer.alloc(0);
  try {
    tokens = fs.readFileSync(tokenPath);
  } catch (error) {
    errors.push(`${tokenPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const byTask = new Map();
  let checkedRecords = 0;
  let hashMismatches = 0;
  let markerMismatches = 0;
  let outOfBounds = 0;
  let missingOffsets = 0;
  for (const row of rows) {
    const task = row.task || "";
    const expected = expectedTaskMarker(task, layout);
    if (!expected) {
      continue;
    }
    const taskSummary = ensureTaskIntegritySummary(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`stage examples line ${row.__line}: ${task} missing valid token_offset/token_count`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(`stage examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length}`);
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.subarray(offset, offset + count);
    const actualMarker = Array.from(slice.subarray(0, expected.length));
    if (!markersMatch(actualMarker, expected)) {
      markerMismatches += 1;
      taskSummary.marker_mismatches += 1;
      errors.push(`stage examples line ${row.__line}: ${task} token marker ${JSON.stringify(actualMarker)} != ${JSON.stringify(expected)}`);
    }
    if (row.token_hash) {
      const actualHash = fnv64ByteHex(slice);
      if (actualHash !== row.token_hash) {
        hashMismatches += 1;
        taskSummary.hash_mismatches += 1;
        errors.push(`stage examples line ${row.__line}: ${task} token_hash ${actualHash} != ${row.token_hash}`);
      }
    }
  }
  return {
    ok: hashMismatches === 0 && markerMismatches === 0 && outOfBounds === 0 && missingOffsets === 0,
    examples: examplesPath,
    tokens: tokenPath,
    checked_records: checkedRecords,
    hash_mismatches: hashMismatches,
    marker_mismatches: markerMismatches,
    out_of_bounds: outOfBounds,
    missing_offsets: missingOffsets,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
  };
}

function checkStageTaskModalityIntegrity(stageDir, manifest, tokenPath, errors) {
  const examplesPath = resolveStagePath(stageDir, manifest.examples_jsonl, "examples.jsonl");
  const rows = readJsonl(examplesPath, errors).filter((row) => expectedTaskModalities(row.task || ""));
  let tokens = Buffer.alloc(0);
  try {
    tokens = fs.readFileSync(tokenPath);
  } catch (error) {
    errors.push(`${tokenPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const byTask = new Map();
  let checkedRecords = 0;
  let missingOffsets = 0;
  let outOfBounds = 0;
  let modalityMismatches = 0;
  for (const row of rows) {
    const task = row.task || "";
    const taskSummary = ensureTaskModalitySummary(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`stage examples line ${row.__line}: ${task} missing valid token_offset/token_count for modality order`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `stage examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for modality order`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.subarray(offset, offset + count);
    const rowErrors = checkStageTaskModalityOrder(task, slice, layout);
    if (rowErrors.length > 0) {
      modalityMismatches += 1;
      taskSummary.modality_mismatches += 1;
      for (const error of rowErrors) {
        errors.push(`stage examples line ${row.__line}: ${error}`);
      }
    }
  }
  return {
    ok: modalityMismatches === 0 && outOfBounds === 0 && missingOffsets === 0,
    examples: examplesPath,
    tokens: tokenPath,
    checked_records: checkedRecords,
    missing_offsets: missingOffsets,
    out_of_bounds: outOfBounds,
    modality_mismatches: modalityMismatches,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
  };
}

function expectedTaskModalities(task) {
  if (task === "canonical-joint") return ["prompt", "text", "image"];
  if (task === "identify" || task === "explain") return ["prompt", "text"];
  if (task === "text-to-image" || task === "description-to-image") return ["prompt", "image"];
  if (task === "image-to-text" || task === "image-to-explain") return ["image", "text"];
  if (task === "text-image-explain" || task === "match") return ["prompt", "image", "text"];
  if (task === "image-to-attributes") return ["image", "prompt", "text"];
  return null;
}

function checkStageTaskModalityOrder(task, slice, layout) {
  const expected = expectedTaskModalities(task);
  if (!expected) {
    return [];
  }
  const markerTokens = {
    prompt: Number(layout.prompt ?? TASK_TOKEN_LAYOUT_FALLBACK.prompt),
    text: Number(layout.text ?? TASK_TOKEN_LAYOUT_FALLBACK.text),
    image: Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image),
  };
  const eosToken = Number(layout.eos ?? TASK_TOKEN_LAYOUT_FALLBACK.eos);
  const eosIndex = slice.indexOf(eosToken, 1);
  const searchEnd = eosIndex >= 0 ? eosIndex : slice.length;
  const positions = Object.fromEntries(
    Object.entries(markerTokens).map(([name, token]) => [
      name,
      markerPositions(slice, token).filter((position) => position > 0 && position < searchEnd),
    ]),
  );
  const out = [];
  if (eosIndex < 0) {
    out.push(`${task} modality order is missing EOS marker ${eosToken}`);
  }
  for (const name of expected) {
    const found = positions[name] || [];
    if (found.length !== 1) {
      out.push(`${task} modality order expected exactly one ${name.toUpperCase()} marker, found ${found.length}`);
    }
  }
  for (const name of Object.keys(markerTokens)) {
    if (!expected.includes(name) && (positions[name] || []).length > 0) {
      out.push(`${task} modality order has unexpected ${name.toUpperCase()} marker before EOS`);
    }
  }
  let previousName = "";
  let previousPosition = -1;
  for (const name of expected) {
    const found = positions[name] || [];
    if (found.length !== 1) {
      continue;
    }
    const position = found[0];
    if (position <= previousPosition) {
      out.push(
        `${task} modality order ${expected.join("->")} has ${name.toUpperCase()} at ${position} after ${previousName.toUpperCase()} at ${previousPosition}`,
      );
    }
    previousName = name;
    previousPosition = position;
  }
  return out;
}

function markerPositions(tokens, marker) {
  const out = [];
  for (let index = 0; index < tokens.length; index += 1) {
    if (Number(tokens[index]) === marker) {
      out.push(index);
    }
  }
  return out;
}

function absentStageImageChannelMarkerIntegrity() {
  return {
    ok: true,
    examples: "",
    tokens: "",
    required_channels: [],
    checked_records: 0,
    missing_offsets: 0,
    out_of_bounds: 0,
    missing_image_markers: 0,
    missing_channel_markers: 0,
    short_channel_payloads: 0,
    bad_channel_payloads: 0,
    channel_order_mismatches: 0,
    by_task: {},
    by_channel: {},
  };
}

function checkStageImageChannelMarkerIntegrity(stageDir, manifest, tokenPath, errors) {
  const requiredChannels = Array.isArray(manifest.image_token_channels)
    ? manifest.image_token_channels.map((channel) => String(channel)).filter(Boolean)
    : [];
  if (requiredChannels.length === 0) {
    return absentStageImageChannelMarkerIntegrity();
  }
  const examplesPath = resolveStagePath(stageDir, manifest.examples_jsonl, "examples.jsonl");
  const rows = readJsonl(examplesPath, errors).filter(
    (row) => row?.schema === "nsrl.solomon_multimodal_example.v2" && IMAGE_BEARING_TASKS.has(String(row.task || "")),
  );
  let tokens = Buffer.alloc(0);
  try {
    tokens = fs.readFileSync(tokenPath);
  } catch (error) {
    errors.push(`${tokenPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const layout = {
    ...TASK_TOKEN_LAYOUT_FALLBACK,
    ...(manifest.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const imageToken = Number(layout.image ?? TASK_TOKEN_LAYOUT_FALLBACK.image);
  const imageBase = Number(layout.image_base ?? TASK_TOKEN_LAYOUT_FALLBACK.image_base);
  const imageBins = Number(layout.image_bins ?? TASK_TOKEN_LAYOUT_FALLBACK.image_bins);
  const payloadTokens = Number(manifest.signature_bins || IMAGE_CHANNEL_PAYLOAD_TOKENS_FALLBACK);
  const byTask = new Map();
  const byChannel = new Map();
  let checkedRecords = 0;
  let missingOffsets = 0;
  let outOfBounds = 0;
  let missingImageMarkers = 0;
  let missingChannelMarkers = 0;
  let shortChannelPayloads = 0;
  let badChannelPayloads = 0;
  let channelOrderMismatches = 0;
  if (rows.length === 0) {
    errors.push(`stage ${stageDir} image channel marker integrity found no image-bearing v2 records`);
  }
  for (const row of rows) {
    const task = row.task || "";
    const taskSummary = ensureImageChannelMarkerSummary(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`stage examples line ${row.__line}: ${task} missing valid token_offset/token_count for image channel markers`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(`stage examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for image channel markers`);
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.subarray(offset, offset + count);
    const imageIndex = slice.indexOf(imageToken);
    if (imageIndex < 0) {
      missingImageMarkers += 1;
      taskSummary.missing_image_markers += 1;
      errors.push(`stage examples line ${row.__line}: ${task} token slice is missing IMAGE marker ${imageToken}`);
      continue;
    }
    let previousChannelPosition = imageIndex;
    for (const channel of requiredChannels) {
      const channelSummary = ensureImageChannelMarkerSummary(byChannel, channel);
      channelSummary.checked_records += 1;
      const marker = expectedImageChannelMarker(channel, layout);
      if (!Number.isInteger(marker)) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`stage image channel ${channel} has no token_layout image_channel_${channel} marker`);
        continue;
      }
      const markerCheck = findImageChannelPayload(slice, imageIndex + 1, marker, imageBase, imageBins, payloadTokens);
      if (!markerCheck.found) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`stage examples line ${row.__line}: ${task} missing image channel marker ${channel}:${marker}`);
        continue;
      }
      if (markerCheck.shortPayload) {
        shortChannelPayloads += 1;
        taskSummary.short_channel_payloads += 1;
        channelSummary.short_channel_payloads += 1;
        errors.push(`stage examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has fewer than ${payloadTokens} tokens`);
        continue;
      }
      if (markerCheck.badPayload) {
        badChannelPayloads += 1;
        taskSummary.bad_channel_payloads += 1;
        channelSummary.bad_channel_payloads += 1;
        errors.push(
          `stage examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has token outside ${imageBase}..${
            imageBase + imageBins - 1
          }`,
        );
        continue;
      }
      if (markerCheck.position <= previousChannelPosition) {
        channelOrderMismatches += 1;
        taskSummary.channel_order_mismatches += 1;
        channelSummary.channel_order_mismatches += 1;
        errors.push(`stage examples line ${row.__line}: ${task} image channel ${channel}:${marker} is out of order`);
      }
      previousChannelPosition = markerCheck.position;
      taskSummary.found_markers += 1;
      channelSummary.found_markers += 1;
    }
  }
  return {
    ok:
      missingOffsets === 0 &&
      outOfBounds === 0 &&
      missingImageMarkers === 0 &&
      missingChannelMarkers === 0 &&
      shortChannelPayloads === 0 &&
      badChannelPayloads === 0 &&
      channelOrderMismatches === 0,
    examples: examplesPath,
    tokens: tokenPath,
    required_channels: requiredChannels,
    checked_records: checkedRecords,
    missing_offsets: missingOffsets,
    out_of_bounds: outOfBounds,
    missing_image_markers: missingImageMarkers,
    missing_channel_markers: missingChannelMarkers,
    short_channel_payloads: shortChannelPayloads,
    bad_channel_payloads: badChannelPayloads,
    channel_order_mismatches: channelOrderMismatches,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    by_channel: Object.fromEntries([...byChannel.entries()].sort(([left], [right]) => left.localeCompare(right))),
  };
}

function ensureImageChannelMarkerSummary(map, key) {
  if (!map.has(key)) {
    map.set(key, {
      checked_records: 0,
      found_markers: 0,
      missing_offsets: 0,
      out_of_bounds: 0,
      missing_image_markers: 0,
      missing_channel_markers: 0,
      short_channel_payloads: 0,
      bad_channel_payloads: 0,
      channel_order_mismatches: 0,
    });
  }
  return map.get(key);
}

function expectedImageChannelMarker(channel, layout) {
  const key = `image_channel_${String(channel).replace(/-/g, "_")}`;
  return Number(layout[key]);
}

function findImageChannelPayload(slice, startIndex, marker, imageBase, imageBins, payloadTokens) {
  let position = slice.indexOf(marker, startIndex);
  let sawBadPayload = false;
  while (position >= 0) {
    const payloadStart = position + 1;
    const payloadEnd = payloadStart + payloadTokens;
    if (payloadEnd > slice.length) {
      return { found: true, position, shortPayload: true, badPayload: false };
    }
    let badPayload = false;
    for (let index = payloadStart; index < payloadEnd; index += 1) {
      const token = Number(slice[index]);
      if (token < imageBase || token >= imageBase + imageBins) {
        badPayload = true;
        break;
      }
    }
    if (!badPayload) {
      return { found: true, position, shortPayload: false, badPayload: false };
    }
    sawBadPayload = true;
    position = slice.indexOf(marker, position + 1);
  }
  if (sawBadPayload) {
    return { found: true, position: -1, shortPayload: false, badPayload: true };
  }
  return { found: false, position: -1, shortPayload: false, badPayload: false };
}

function checkStageIdentityBindings(stageName, manifest, errors) {
  const requiredTasks = STAGE_IDENTITY_BINDING_TASKS[stageName] || [];
  if (requiredTasks.length === 0) {
    return;
  }
  const source = manifest.source_identity_bindings;
  const selected = manifest.identity_bindings;
  if (!source || !selected) {
    errors.push(`stage ${stageName} is missing identity binding summaries`);
    return;
  }
  for (const task of requiredTasks) {
    const sourceTask = source.by_task?.[task];
    const selectedTask = selected.by_task?.[task];
    if (!sourceTask || Number(sourceTask.rows || 0) <= 0) {
      errors.push(`source identity bindings have no ${task} rows for stage ${stageName}`);
      continue;
    }
    if (!selectedTask || Number(selectedTask.rows || 0) <= 0) {
      errors.push(`stage ${stageName} selected no ${task} identity bindings`);
      continue;
    }
    if (Number(selectedTask.rows || 0) !== Number(sourceTask.rows || 0)) {
      errors.push(
        `stage ${stageName} ${task} identity bindings ${selectedTask.rows || 0} != source ${sourceTask.rows || 0}`,
      );
    }
    if (selectedTask.binding_hash !== sourceTask.binding_hash) {
      errors.push(`stage ${stageName} ${task} identity binding hash ${selectedTask.binding_hash || ""} != source ${sourceTask.binding_hash || ""}`);
    }
    if (Number(selectedTask.spirits || 0) !== Number(sourceTask.spirits || 0)) {
      errors.push(
        `stage ${stageName} ${task} identity spirits ${selectedTask.spirits || 0} != source ${sourceTask.spirits || 0}`,
      );
    }
    for (const kind of REQUIRED_IDENTITY_BINDING_KINDS) {
      const sourceCount = Number(sourceTask.counts?.[kind] || 0);
      const selectedCount = Number(selectedTask.counts?.[kind] || 0);
      if (sourceCount <= 0) {
        errors.push(`source ${task} identity bindings are missing kind ${kind}`);
      } else if (selectedCount !== sourceCount) {
        errors.push(
          `stage ${stageName} ${task} identity kind ${kind} count ${selectedCount} != source ${sourceCount}`,
        );
      }
    }
  }
}

function checkStageTaskCoverage(stageName, manifest, errors) {
  const coverage = manifest.task_coverage;
  const sourceCoverage = manifest.source_task_coverage;
  const expectedSpirits = Number(sourceCoverage?.spirits || manifest.rows || 0);
  const requirement = STAGE_EVIDENCE_TASKS[stageName] || null;
  if (!requirement) {
    return absentStageEvidence(stageName, expectedSpirits);
  }
  if (!coverage || typeof coverage !== "object" || Array.isArray(coverage)) {
    errors.push(`stage ${stageName} is missing task_coverage`);
    return absentStageEvidence(stageName, expectedSpirits);
  }
  const tasks = coverage.tasks && typeof coverage.tasks === "object" ? coverage.tasks : {};
  const required = {};
  for (const task of requirement.required || []) {
    required[task] = taskEvidence(tasks[task]);
    if (required[task].records <= 0) {
      errors.push(`stage ${stageName} selected no ${task} rows`);
    }
    if (expectedSpirits > 0 && required[task].spirits !== expectedSpirits) {
      errors.push(`stage ${stageName} ${task} spirits ${required[task].spirits} != ${expectedSpirits}`);
    }
  }
  const match = {};
  for (const label of requirement.match_labels || []) {
    const evidence = groupEvidence(tasks.match?.labels?.[label]);
    match[`label:${label}`] = evidence;
    if (evidence.records <= 0) {
      errors.push(`stage ${stageName} selected no match ${label} rows`);
    }
    if (expectedSpirits > 0 && evidence.spirits !== expectedSpirits) {
      errors.push(`stage ${stageName} match ${label} spirits ${evidence.spirits} != ${expectedSpirits}`);
    }
  }
  for (const role of requirement.match_roles || []) {
    const evidence = groupEvidence(tasks.match?.roles?.[role]);
    match[`role:${role}`] = evidence;
    if (evidence.records <= 0) {
      errors.push(`stage ${stageName} selected no match ${role} negative-role rows`);
    }
    if (expectedSpirits > 0 && evidence.spirits !== expectedSpirits) {
      errors.push(`stage ${stageName} match ${role} role spirits ${evidence.spirits} != ${expectedSpirits}`);
    }
  }
  return {
    stage_name: stageName,
    expected_spirits: expectedSpirits,
    records: Number(coverage.records || 0),
    spirits: Number(coverage.spirits || 0),
    required,
    image_plan: categoryEvidence(requirement.image_plan || [], tasks),
    image_classification: categoryEvidence(requirement.image_classification || [], tasks),
    image_grounding: categoryEvidence(requirement.image_grounding || [], tasks),
    match,
  };
}

function absentStageEvidence(stageName, expectedSpirits) {
  return {
    stage_name: stageName,
    expected_spirits: expectedSpirits,
    records: 0,
    spirits: 0,
    required: {},
    image_plan: categoryEvidence([], {}),
    image_classification: categoryEvidence([], {}),
    image_grounding: categoryEvidence([], {}),
    match: {},
  };
}

function categoryEvidence(taskNames, tasks) {
  const byTask = {};
  const spiritCoverage = [];
  let records = 0;
  for (const task of taskNames) {
    const evidence = taskEvidence(tasks[task]);
    byTask[task] = evidence;
    records += evidence.records;
    spiritCoverage.push(evidence.spirits);
  }
  return {
    tasks: taskNames,
    records,
    min_spirits: spiritCoverage.length > 0 ? Math.min(...spiritCoverage) : 0,
    by_task: byTask,
  };
}

function taskEvidence(row) {
  return {
    records: Number(row?.records || 0),
    spirits: Number(row?.spirits || 0),
    identity_binding_rows: Number(row?.identity_binding_rows || 0),
  };
}

function trainTaskEvidence(row) {
  return {
    examples: Number(row?.examples || 0),
    targets: Number(row?.targets || 0),
    special_targets: Number(row?.special_targets || 0),
    prompt_targets: Number(row?.prompt_targets || 0),
    text_targets: Number(row?.text_targets || 0),
    image_targets: Number(row?.image_targets || 0),
  };
}

function checkTrainTaskCoverage(stageDir, manifest, train, trainPath, errors) {
  const coverage = manifest.task_coverage && typeof manifest.task_coverage === "object" ? manifest.task_coverage : {};
  const manifestTasks = coverage.tasks && typeof coverage.tasks === "object" ? coverage.tasks : {};
  const trainTasks = train.tasks && typeof train.tasks === "object" && !Array.isArray(train.tasks) ? train.tasks : null;
  const trainTaskPhases =
    train.task_phases && typeof train.task_phases === "object" && !Array.isArray(train.task_phases)
      ? train.task_phases
      : null;
  const stageExamplesPath = resolveStagePath(stageDir, manifest.examples_jsonl, "examples.jsonl");
  const trainExamplesPath = String(train.corpus_examples_path || "");
  const resolvedTrainExamplesPath = trainExamplesPath
    ? firstExistingReferencedPath(trainExamplesPath, stageDir, "")
    : "";
  let expectedExamplesHash = "";
  try {
    expectedExamplesHash = fnv64ByteHex(fs.readFileSync(stageExamplesPath));
  } catch (error) {
    errors.push(`${stageExamplesPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const summary = {
    source: train.corpus_coverage_source || "",
    examples_path: trainExamplesPath,
    resolved_examples_path: resolvedTrainExamplesPath,
    expected_examples_path: stageExamplesPath,
    examples_hash: train.corpus_examples_hash || "",
    expected_examples_hash: expectedExamplesHash,
    examples_hash_match: Boolean(
      train.corpus_examples_hash && expectedExamplesHash && train.corpus_examples_hash === expectedExamplesHash,
    ),
    examples: Number(train.corpus_examples || 0),
    skipped_examples: Number(train.corpus_skipped_examples || 0),
    prefix_pad_tokens: Number(train.corpus_prefix_pad_tokens || 0),
    orphan_tokens: Number(train.corpus_orphan_tokens || 0),
    task_count: trainTasks ? Object.keys(trainTasks).length : 0,
    tasks: {},
  };
  if (summary.source !== "examples") {
    errors.push(`${trainPath} corpus_coverage_source ${JSON.stringify(summary.source)} != "examples"`);
  }
  if (!trainExamplesPath) {
    errors.push(`${trainPath} missing corpus_examples_path`);
  } else if (!resolvedTrainExamplesPath || !fs.existsSync(resolvedTrainExamplesPath)) {
    errors.push(`${trainPath} corpus_examples_path ${JSON.stringify(trainExamplesPath)} does not resolve`);
  } else if (path.resolve(resolvedTrainExamplesPath) !== path.resolve(stageExamplesPath)) {
    errors.push(
      `${trainPath} corpus_examples_path ${path.resolve(resolvedTrainExamplesPath)} != stage examples ${path.resolve(stageExamplesPath)}`,
    );
  }
  if (!train.corpus_examples_hash) {
    errors.push(`${trainPath} missing corpus_examples_hash`);
  } else if (expectedExamplesHash && train.corpus_examples_hash !== expectedExamplesHash) {
    errors.push(
      `${trainPath} corpus_examples_hash ${train.corpus_examples_hash} != stage examples hash ${expectedExamplesHash}`,
    );
  }
  if (summary.examples !== Number(manifest.examples || 0)) {
    errors.push(`${trainPath} corpus_examples ${summary.examples} != manifest examples ${Number(manifest.examples || 0)}`);
  }
  if (summary.skipped_examples !== 0) {
    errors.push(`${trainPath} corpus_skipped_examples ${summary.skipped_examples} != 0`);
  }
  if (summary.orphan_tokens !== 0) {
    errors.push(`${trainPath} corpus_orphan_tokens ${summary.orphan_tokens} != 0`);
  }
  if (!trainTasks) {
    errors.push(`${trainPath} is missing native train task coverage`);
    return summary;
  }
  if (!trainTaskPhases) {
    errors.push(`${trainPath} is missing native train task phase coverage`);
  }
  for (const [task, taskCoverage] of Object.entries(manifestTasks)) {
    const trainTask = trainTaskEvidence(trainTasks[task]);
    const requiredRecords = Number(taskCoverage?.records || 0);
    summary.tasks[task] = trainTask;
    if (trainTask.examples !== requiredRecords) {
      errors.push(`${trainPath} task ${task} examples ${trainTask.examples} != manifest records ${requiredRecords}`);
    }
    if (requiredRecords > 0 && trainTask.targets <= 0) {
      errors.push(`${trainPath} task ${task} has no native train targets`);
    }
    const phases = trainTaskPhases?.[task] && typeof trainTaskPhases[task] === "object" ? trainTaskPhases[task] : null;
    if (!phases || Object.keys(phases).length === 0) {
      errors.push(`${trainPath} task ${task} has no native train phase targets`);
    }
  }
  for (const task of Object.keys(trainTasks)) {
    if (!Object.hasOwn(manifestTasks, task)) {
      errors.push(`${trainPath} task ${task} is absent from manifest task_coverage`);
    }
  }
  return summary;
}

function groupEvidence(row) {
  return {
    records: Number(row?.records || 0),
    spirits: Number(row?.spirits || 0),
  };
}

function checkStage(stageDir, index, config) {
  const errors = [];
  const manifestPath = path.join(stageDir, "manifest.json");
  const trainPath = path.join(stageDir, "train.json");
  const manifest = readJson(manifestPath, errors);
  const train = readJson(trainPath, errors);
  if (!manifest || !train) {
    return {
      ok: false,
      index,
      stage_dir: stageDir,
      stage_name: inferStageName(stageDir),
      expected_stage_name: config.requiredStageNames[index] || "",
      errors,
    };
  }
  const recipe = checkStageRecipe(stageDir, index, manifest, config, errors);
  const tokenPath = path.join(stageDir, manifest.corpus_tokens_u8 || "corpus.tokens.u8");
  let byteTokenHash = "";
  try {
    byteTokenHash = fnv64ByteHex(fs.readFileSync(tokenPath));
  } catch (error) {
    errors.push(`${tokenPath}: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (manifest.schema !== "nsrl.solomon_multimodal_corpus_filter.v1") {
    errors.push(`${manifestPath} schema ${JSON.stringify(manifest.schema)} != nsrl.solomon_multimodal_corpus_filter.v1`);
  }
  if (train.schema !== "nsrl.solomon_attention_train_trace.v1") {
    errors.push(`${trainPath} schema ${JSON.stringify(train.schema)} != nsrl.solomon_attention_train_trace.v1`);
  }
  if (Number(manifest.examples || 0) <= 0) {
    errors.push(`${manifestPath} examples ${manifest.examples || 0} <= 0`);
  }
  if (Number(manifest.token_count || 0) <= 0) {
    errors.push(`${manifestPath} token_count ${manifest.token_count || 0} <= 0`);
  }
  const sourceCorpus = checkStageSourceCorpus(stageDir, manifest, errors);
  const taskMarkerIntegrity = checkStageTaskMarkerIntegrity(stageDir, manifest, tokenPath, errors);
  const taskModalityIntegrity = checkStageTaskModalityIntegrity(stageDir, manifest, tokenPath, errors);
  const imageChannelMarkerIntegrity = checkStageImageChannelMarkerIntegrity(stageDir, manifest, tokenPath, errors);
  const evidenceStageName = recipe.expectedStageName || recipe.stageName;
  checkStageIdentityBindings(evidenceStageName, manifest, errors);
  const stageEvidence = checkStageTaskCoverage(evidenceStageName, manifest, errors);
  const trainTaskCoverage = checkTrainTaskCoverage(stageDir, manifest, train, trainPath, errors);
  if (train.token_hash && byteTokenHash && train.token_hash !== byteTokenHash) {
    errors.push(`${trainPath} token_hash ${train.token_hash} != byte token hash ${byteTokenHash}`);
  }
  if (Number(train.accepted_batches || 0) <= 0) {
    errors.push(`${trainPath} accepted_batches ${train.accepted_batches || 0} <= 0`);
  }
  if (Number(train.updates || 0) <= 0) {
    errors.push(`${trainPath} updates ${train.updates || 0} <= 0`);
  }
  if (Number(train.examined_windows || 0) <= 0) {
    errors.push(`${trainPath} examined_windows ${train.examined_windows || 0} <= 0`);
  }
  if (Number(train.rejected_batches || 0) > Number(train.accepted_batches || 0)) {
    errors.push(`${trainPath} rejected_batches ${train.rejected_batches} > accepted_batches ${train.accepted_batches}`);
  }
  const delta = Number(train.probability_error_delta_i64 || 0);
  if (config.requireLossNonIncreasing && delta > 0) {
    errors.push(`${trainPath} probability_error_delta_i64 ${delta} > 0`);
  }
  const trainEpochs = Number(train.epochs || 0);
  if (
    evidenceStageName === "native-bind" &&
    config.minNativeBindEpochs > 0 &&
    trainEpochs < config.minNativeBindEpochs
  ) {
    errors.push(`${trainPath} epochs ${trainEpochs} < native-bind minimum ${config.minNativeBindEpochs}`);
  }
  return {
    ok: errors.length === 0,
    index,
    stage_dir: stageDir,
    stage_name: recipe.stageName,
    expected_stage_name: recipe.expectedStageName,
    filter: manifest.filter || {},
    examples: Number(manifest.examples || 0),
    token_count: Number(manifest.token_count || 0),
    ...sourceCorpus,
    identity_bindings: manifest.identity_bindings || null,
    source_identity_bindings: manifest.source_identity_bindings || null,
    task_coverage: manifest.task_coverage || null,
    stage_evidence: stageEvidence,
    task_marker_integrity: taskMarkerIntegrity,
    task_modality_integrity: taskModalityIntegrity,
    image_channel_marker_integrity: imageChannelMarkerIntegrity,
    train_task_coverage: trainTaskCoverage,
    manifest_token_hash: manifest.token_hash || "",
    byte_token_hash: byteTokenHash,
    train: {
      model: train.model || "",
      model_hash: train.model_hash || "",
      attention_kind: train.attention_kind || "",
      text_token_profile: train.text_token_profile || "",
      d_model: Number(train.d_model || 0),
      heads: Number(train.heads || 0),
      hidden_dim: Number(train.hidden_dim || 0),
      transformer_layers: Number(train.transformer_layers || 0),
      epochs: trainEpochs,
      seq_len: Number(train.seq_len || 0),
      context_seq_len: Number(train.context_seq_len || train.seq_len || 0),
      max_windows: train.max_windows === null ? null : Number(train.max_windows || 0),
      batch_mode: train.batch_mode || "",
      map_reduce_workers: Number(train.map_reduce_workers || 0),
      windows: Number(train.windows || 0),
      examined_windows: Number(train.examined_windows || 0),
      updates: Number(train.updates || 0),
      accepted_batches: Number(train.accepted_batches || 0),
      rejected_batches: Number(train.rejected_batches || 0),
      probability_error_delta_i64: delta,
      initial_probability_error_q15: Number(train.initial_probability_error_q15 || 0),
      final_probability_error_q15: Number(train.final_probability_error_q15 || 0),
    },
    errors,
  };
}

function uniqueStrings(values) {
  return [...new Set(values.map((value) => String(value || "")).filter(Boolean))];
}

function summarizeSourceCorpusProvenance(stages, errors) {
  const sourceExamples = uniqueStrings(stages.map((stage) => stage.source_examples));
  const sourceExamplesHashes = uniqueStrings(stages.map((stage) => stage.source_examples_hash));
  const sourceTokens = uniqueStrings(stages.map((stage) => stage.source_tokens));
  const sourceTokensHashes = uniqueStrings(stages.map((stage) => stage.source_tokens_hash));
  const summary = {
    source_examples: sourceExamples[0] || "",
    source_examples_hash: sourceExamplesHashes[0] || "",
    source_examples_consistent: sourceExamples.length <= 1,
    source_tokens: sourceTokens[0] || "",
    source_tokens_hash: sourceTokensHashes[0] || "",
    source_tokens_consistent: sourceTokens.length <= 1,
  };
  if (sourceExamples.length > 1) {
    errors.push(`source examples differ across stages: ${sourceExamples.join(", ")}`);
  }
  if (sourceExamplesHashes.length > 1) {
    summary.source_examples_consistent = false;
    errors.push(`source examples hashes differ across stages: ${sourceExamplesHashes.join(", ")}`);
  }
  if (sourceTokens.length > 1) {
    errors.push(`source tokens differ across stages: ${sourceTokens.join(", ")}`);
  }
  if (sourceTokensHashes.length > 1) {
    summary.source_tokens_consistent = false;
    errors.push(`source tokens hashes differ across stages: ${sourceTokensHashes.join(", ")}`);
  }
  return summary;
}

function writeJson(filePath, row) {
  const dir = path.dirname(filePath);
  if (dir && dir !== ".") {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(filePath, `${JSON.stringify(row, null, 2)}\n`, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const stages = config.stageDirs.map((stageDir, index) => checkStage(stageDir, index, config));
  const errors = stages.flatMap((stage) => stage.errors.map((error) => `stage ${stage.index}: ${error}`));
  const sourceCorpusProvenance = summarizeSourceCorpusProvenance(stages, errors);
  if (stages.length < config.minStages) {
    errors.push(`stage count ${stages.length} < ${config.minStages}`);
  }
  if (config.requiredStageNames.length > 0 && stages.length !== config.requiredStageNames.length) {
    errors.push(`stage count ${stages.length} != required stage count ${config.requiredStageNames.length}`);
  }
  const result = {
    schema: "nsrl.solomon_v2_curriculum_stage_check.v1",
    ok: errors.length === 0,
    stage_count: stages.length,
    min_stages: config.minStages,
    required_stage_names: config.requiredStageNames,
    require_loss_non_increasing: config.requireLossNonIncreasing,
    min_native_bind_epochs: config.minNativeBindEpochs,
    source_corpus_provenance: sourceCorpusProvenance,
    stages,
    errors,
  };
  if (config.outPath) {
    writeJson(config.outPath, result);
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    process.exit(1);
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
