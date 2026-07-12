#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const DEFAULT_REQUIRED_TASKS = [
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
];
const PRODUCT_DIRECTIONAL_GROUPS = [
  {
    key: "text_prompt_to_image_plan",
    label: "text prompt -> 16x16 image plan",
    tasks: ["text-to-image", "description-to-image"],
    required_phases: {
      "text-to-image": ["prompt", "image"],
      "description-to-image": ["prompt", "image"],
    },
  },
  {
    key: "seal_image_to_text",
    label: "seal image -> identity / attributes / source text",
    tasks: ["image-to-text", "image-to-explain", "image-to-attributes"],
    required_phases: {
      "image-to-text": ["image", "text"],
      "image-to-explain": ["image", "text"],
      "image-to-attributes": ["image", "prompt", "text"],
    },
  },
  {
    key: "text_and_seal_to_explanation",
    label: "text + seal -> explanation / retrieval",
    tasks: ["text-image-explain", "match"],
    required_phases: {
      "text-image-explain": ["prompt", "image", "text"],
      match: ["prompt", "image", "text"],
    },
  },
  {
    key: "identity_source_binding",
    label: "prompt/name -> identity / source text",
    tasks: ["canonical-joint", "identify", "explain"],
    required_phases: {
      "canonical-joint": ["prompt", "text", "image"],
      identify: ["prompt", "text"],
      explain: ["prompt", "text"],
    },
  },
];
const TOKEN_LAYOUT_FALLBACK = {
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
const EVAL_PHASES = ["special", "prompt", "text", "image"];
const REQUIRED_OUTPUT_HEADS = ["special_head", "text_head", "image_head"];
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

const defaults = {
  evalPath: "",
  examplesPath: "",
  manifestPath: "",
  tokensPath: "",
  requiredTasks: DEFAULT_REQUIRED_TASKS,
  expectSpirits: null,
  requireCorpusVersion: "",
  requireImageTokenProfile: "",
  requireImageTokenChannels: [],
  requireImageChannelTokenStats: false,
  minImageChannelDistinctBins: 2,
  maxSkippedExamples: 0,
  maxInvalidContexts: 0,
  minTaskTargets: new Map(),
  minTaskAccuracy: new Map(),
  minTaskTop5: new Map(),
  minTaskTop10: new Map(),
  minPhaseTargets: new Map(),
  minDirectionalTargets: new Map(),
  minDirectionalAccuracy: new Map(),
  minDirectionalTop5: new Map(),
  minDirectionalTop10: new Map(),
  minTotalAccuracy: null,
  minTotalTop5: null,
  requireOutputHeads: true,
  requireTaskPhaseStats: false,
  requireDirectionalGroups: false,
};

function usage() {
  console.log(
    [
      "Usage: check-solomon-attention-task-eval.mjs --eval PATH [options]",
      "",
      "Verifies that a v2 Solomon attention eval trace exposes the expected",
      "multimodal task gates and, when examples are supplied, full spirit coverage.",
      "",
      "Options:",
      "  --examples PATH",
      "  --manifest PATH",
      "  --tokens PATH",
      "  --require-tasks LIST",
      "  --expect-spirits N",
      "  --require-corpus-version VALUE",
      "  --require-image-token-profile PROFILE",
      "  --require-image-token-channels LIST",
      "  --require-image-channel-token-stats",
      "  --min-image-channel-distinct-bins N",
      "  --max-skipped-examples N",
      "  --max-invalid-contexts N",
      "  --min-task-targets TASK=N[,TASK=N...]",
      "  --min-task-accuracy TASK=N[,TASK=N...]",
      "  --min-task-top5 TASK=N[,TASK=N...]",
      "  --min-task-top10 TASK=N[,TASK=N...]",
      "  --min-phase-targets PHASE=N[,PHASE=N...]",
      "  --min-direction-targets GROUP=N[,GROUP=N...]",
      "  --min-direction-accuracy GROUP=N[,GROUP=N...]",
      "  --min-direction-top5 GROUP=N[,GROUP=N...]",
      "  --min-direction-top10 GROUP=N[,GROUP=N...]",
      "  --min-total-accuracy N",
      "  --min-total-top5 N",
      "  --require-output-heads",
      "  --no-require-output-heads",
      "  --require-task-phase-stats",
      "  --no-require-task-phase-stats",
      "  --require-directional-groups",
      "  --no-require-directional-groups",
      "",
      "Accuracy thresholds are per-mille integers; 0.0-1.0 decimals are accepted.",
      "Use all=N in task threshold maps to apply the threshold to every required task.",
      `Known directional groups: ${PRODUCT_DIRECTIONAL_GROUPS.map((group) => group.key).join(", ")}`,
      `Known phases: ${EVAL_PHASES.join(", ")}`,
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = {
    ...defaults,
    requiredTasks: [...defaults.requiredTasks],
    minTaskTargets: new Map(defaults.minTaskTargets),
    minTaskAccuracy: new Map(defaults.minTaskAccuracy),
    minTaskTop5: new Map(defaults.minTaskTop5),
    minTaskTop10: new Map(defaults.minTaskTop10),
    minPhaseTargets: new Map(defaults.minPhaseTargets),
    minDirectionalTargets: new Map(defaults.minDirectionalTargets),
    minDirectionalAccuracy: new Map(defaults.minDirectionalAccuracy),
    minDirectionalTop5: new Map(defaults.minDirectionalTop5),
    minDirectionalTop10: new Map(defaults.minDirectionalTop10),
    requireImageTokenChannels: [...defaults.requireImageTokenChannels],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--eval") {
      config.evalPath = requireValue(argv, ++index, arg);
    } else if (arg === "--examples") {
      config.examplesPath = requireValue(argv, ++index, arg);
    } else if (arg === "--manifest") {
      config.manifestPath = requireValue(argv, ++index, arg);
    } else if (arg === "--tokens") {
      config.tokensPath = requireValue(argv, ++index, arg);
    } else if (arg === "--require-tasks") {
      config.requiredTasks = parseList(requireValue(argv, ++index, arg));
    } else if (arg === "--expect-spirits") {
      config.expectSpirits = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-corpus-version") {
      config.requireCorpusVersion = requireValue(argv, ++index, arg);
    } else if (arg === "--require-image-token-profile") {
      config.requireImageTokenProfile = requireValue(argv, ++index, arg);
    } else if (arg === "--require-image-token-channels") {
      config.requireImageTokenChannels = parseList(requireValue(argv, ++index, arg));
    } else if (arg === "--require-image-channel-token-stats") {
      config.requireImageChannelTokenStats = true;
    } else if (arg === "--min-image-channel-distinct-bins") {
      config.minImageChannelDistinctBins = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-skipped-examples") {
      config.maxSkippedExamples = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--max-invalid-contexts") {
      config.maxInvalidContexts = parseNonNegative(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-task-targets") {
      mergeAssignments(config.minTaskTargets, requireValue(argv, ++index, arg), arg, parseNonNegative);
    } else if (arg === "--min-task-accuracy") {
      mergeAssignments(config.minTaskAccuracy, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-task-top5") {
      mergeAssignments(config.minTaskTop5, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-task-top10") {
      mergeAssignments(config.minTaskTop10, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-phase-targets") {
      mergeAssignments(config.minPhaseTargets, requireValue(argv, ++index, arg), arg, parseNonNegative);
    } else if (arg === "--min-direction-targets") {
      mergeAssignments(config.minDirectionalTargets, requireValue(argv, ++index, arg), arg, parseNonNegative);
    } else if (arg === "--min-direction-accuracy") {
      mergeAssignments(config.minDirectionalAccuracy, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-direction-top5") {
      mergeAssignments(config.minDirectionalTop5, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-direction-top10") {
      mergeAssignments(config.minDirectionalTop10, requireValue(argv, ++index, arg), arg, parseRatePerMille);
    } else if (arg === "--min-total-accuracy") {
      config.minTotalAccuracy = parseRatePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--min-total-top5") {
      config.minTotalTop5 = parseRatePerMille(requireValue(argv, ++index, arg), arg);
    } else if (arg === "--require-output-heads") {
      config.requireOutputHeads = true;
    } else if (arg === "--no-require-output-heads") {
      config.requireOutputHeads = false;
    } else if (arg === "--require-task-phase-stats") {
      config.requireTaskPhaseStats = true;
    } else if (arg === "--no-require-task-phase-stats") {
      config.requireTaskPhaseStats = false;
    } else if (arg === "--require-directional-groups") {
      config.requireDirectionalGroups = true;
      config.requireTaskPhaseStats = true;
    } else if (arg === "--no-require-directional-groups") {
      config.requireDirectionalGroups = false;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  if (!config.evalPath) {
    throw new Error("--eval is required");
  }
  if (config.requiredTasks.length === 0) {
    throw new Error("--require-tasks must include at least one task");
  }
  for (const phase of config.minPhaseTargets.keys()) {
    if (phase !== "all" && !EVAL_PHASES.includes(phase)) {
      throw new Error(`--min-phase-targets unknown phase ${phase}; expected ${EVAL_PHASES.join(", ")}, or all`);
    }
  }
  const knownDirectionalGroups = new Set(["all", ...PRODUCT_DIRECTIONAL_GROUPS.map((group) => group.key)]);
  checkDirectionalThresholdKeys("--min-direction-targets", config.minDirectionalTargets, knownDirectionalGroups);
  checkDirectionalThresholdKeys("--min-direction-accuracy", config.minDirectionalAccuracy, knownDirectionalGroups);
  checkDirectionalThresholdKeys("--min-direction-top5", config.minDirectionalTop5, knownDirectionalGroups);
  checkDirectionalThresholdKeys("--min-direction-top10", config.minDirectionalTop10, knownDirectionalGroups);
  if (config.expectSpirits === null) {
    config.expectSpirits = config.examplesPath ? 72 : 0;
  }
  return config;
}

function checkDirectionalThresholdKeys(flag, map, knownDirectionalGroups) {
  for (const group of map.keys()) {
    if (!knownDirectionalGroups.has(group)) {
      throw new Error(
        `${flag} unknown group ${group}; expected ${[...knownDirectionalGroups].join(", ")}`,
      );
    }
  }
}

function requireValue(argv, index, flag) {
  if (index >= argv.length) {
    throw new Error(`${flag} requires a value`);
  }
  return argv[index];
}

function parseList(value) {
  return value
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function parseNonNegative(value, flag) {
  if (!/^[0-9]+$/.test(value)) {
    throw new Error(`${flag} requires a non-negative integer`);
  }
  return Number(value);
}

function parseRatePerMille(value, flag) {
  if (/^[0-9]+$/.test(value)) {
    const parsed = Number(value);
    if (parsed > 1000) {
      throw new Error(`${flag} threshold must be <= 1000 per mille`);
    }
    return parsed;
  }
  if (/^(?:0(?:\.[0-9]+)?|1(?:\.0+)?)$/.test(value)) {
    return Math.round(Number(value) * 1000);
  }
  throw new Error(`${flag} requires a per-mille integer or 0.0-1.0 decimal`);
}

function mergeAssignments(target, value, flag, parser) {
  for (const assignment of parseList(value)) {
    const equals = assignment.indexOf("=");
    if (equals <= 0 || equals === assignment.length - 1) {
      throw new Error(`${flag} entries must look like KEY=N`);
    }
    const key = assignment.slice(0, equals);
    const raw = assignment.slice(equals + 1);
    target.set(key, parser(raw, `${flag} ${key}`));
  }
}

function readJson(path) {
  const text = fs.readFileSync(path, "utf8");
  try {
    return JSON.parse(text);
  } catch (error) {
    const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (lines.length === 0) {
      throw error;
    }
    return JSON.parse(lines[lines.length - 1]);
  }
}

function readJsonl(path) {
  const text = fs.readFileSync(path, "utf8").trimEnd();
  if (!text) {
    return [];
  }
  return text.split(/\r?\n/).filter(Boolean).map((line, rowIndex) => {
    const row = JSON.parse(line);
    row.__line = rowIndex + 1;
    return row;
  });
}

function readTokens(path) {
  const bytes = fs.readFileSync(path);
  if (path.endsWith(".u16")) {
    if (bytes.length % 2 !== 0) {
      throw new Error(`${path} byte length ${bytes.length} is not divisible by 2`);
    }
    const tokens = [];
    for (let index = 0; index < bytes.length; index += 2) {
      tokens.push(bytes.readUInt16LE(index));
    }
    return tokens;
  }
  return Array.from(bytes);
}

function normalizedPath(value) {
  if (!value) {
    return "";
  }
  return path.resolve(String(value));
}

function samePath(left, right) {
  return normalizedPath(left) === normalizedPath(right);
}

function taskThreshold(map, task) {
  if (map.has(task)) {
    return map.get(task);
  }
  if (map.has("all")) {
    return map.get("all");
  }
  return null;
}

function directionalThreshold(map, group) {
  if (map.has(group)) {
    return map.get(group);
  }
  if (map.has("all")) {
    return map.get("all");
  }
  return null;
}

function statsSummary(stats) {
  return {
    targets: numberField(stats, "targets"),
    correct: numberField(stats, "correct"),
    invalid_contexts: numberField(stats, "invalid_contexts"),
    accuracy_per_mille: numberField(stats, "accuracy_per_mille"),
    top5_accuracy_per_mille: numberField(stats, "top5_accuracy_per_mille"),
    top10_accuracy_per_mille: numberField(stats, "top10_accuracy_per_mille"),
    mean_target_rank_per_mille: numberField(stats, "mean_target_rank_per_mille"),
    mean_target_margin_q8: numberField(stats, "mean_target_margin_q8"),
  };
}

function mergeStatsSummaries(items) {
  let targets = 0;
  let correct = 0;
  let invalidContexts = 0;
  let top5Numerator = 0;
  let top10Numerator = 0;
  let rankNumerator = 0;
  let marginNumerator = 0;
  let top5Targets = 0;
  let top10Targets = 0;
  let rankTargets = 0;
  let marginTargets = 0;
  let seen = false;
  for (const stats of items) {
    const itemTargets = numberField(stats, "targets");
    if (itemTargets === null) {
      continue;
    }
    seen = true;
    targets += itemTargets;
    correct += Number(stats.correct || 0);
    invalidContexts += Number(stats.invalid_contexts || 0);
    const top5 = numberField(stats, "top5_accuracy_per_mille");
    if (top5 !== null) {
      top5Numerator += top5 * itemTargets;
      top5Targets += itemTargets;
    }
    const top10 = numberField(stats, "top10_accuracy_per_mille");
    if (top10 !== null) {
      top10Numerator += top10 * itemTargets;
      top10Targets += itemTargets;
    }
    const rank = numberField(stats, "mean_target_rank_per_mille");
    if (rank !== null) {
      rankNumerator += rank * itemTargets;
      rankTargets += itemTargets;
    }
    const margin = numberField(stats, "mean_target_margin_q8");
    if (margin !== null) {
      marginNumerator += margin * itemTargets;
      marginTargets += itemTargets;
    }
  }
  if (!seen) {
    return statsSummary({});
  }
  const valid = Math.max(0, targets - invalidContexts);
  return {
    targets,
    correct,
    invalid_contexts: invalidContexts,
    accuracy_per_mille: targets > 0 ? Math.floor((correct * 1000) / targets) : 0,
    top5_accuracy_per_mille: top5Targets > 0 ? Math.floor(top5Numerator / top5Targets) : null,
    top10_accuracy_per_mille: top10Targets > 0 ? Math.floor(top10Numerator / top10Targets) : null,
    mean_target_rank_per_mille: rankTargets > 0 ? Math.floor(rankNumerator / rankTargets) : null,
    mean_target_margin_q8: marginTargets > 0 ? Math.floor(marginNumerator / marginTargets) : null,
    valid_targets: valid,
  };
}

function numberField(object, field) {
  const value = object?.[field];
  return Number.isFinite(value) ? value : null;
}

function inspectExamples(rows) {
  const groups = new Map();
  const errors = [];
  const allSpirits = new Set();
  const profileCounts = new Map();
  const channelSetCounts = new Map();
  const channelPresenceCounts = new Map();
  let v2Records = 0;
  let missingImageTokenProfile = 0;
  let missingImageTokenChannels = 0;
  for (const row of rows) {
    const task = row.task || "canonical-joint";
    const spiritId = normalizedId(row.spirit_id);
    if (row.schema === "nsrl.solomon_multimodal_example.v2") {
      v2Records += 1;
      const profile = String(row.image_token_profile || "");
      if (profile) {
        increment(profileCounts, profile);
      } else {
        missingImageTokenProfile += 1;
      }
      const channels = Array.isArray(row.image_token_channels)
        ? row.image_token_channels.map((channel) => String(channel))
        : [];
      if (channels.length > 0) {
        increment(channelSetCounts, channels.join(","));
        for (const channel of new Set(channels)) {
          increment(channelPresenceCounts, channel);
        }
      } else {
        missingImageTokenChannels += 1;
      }
    }
    if (spiritId !== null) {
      allSpirits.add(spiritId);
    }
    const group = ensureGroup(groups, task);
    group.records += 1;
    if (spiritId !== null) {
      group.spirits.add(spiritId);
    }
    if (task === "match") {
      const label = String(row.match_label || row.text || "").toLowerCase();
      if (label !== "yes" && label !== "no") {
        errors.push(`examples line ${row.__line}: match row has invalid label ${JSON.stringify(label)}`);
      } else {
        const labelGroup = ensureGroup(group.labels, label);
        labelGroup.records += 1;
        if (spiritId !== null) {
          labelGroup.spirits.add(spiritId);
        }
        if (label === "no") {
          const negativeSpiritId = normalizedId(row.negative_spirit_id);
          if (negativeSpiritId === null) {
            errors.push(`examples line ${row.__line}: negative match row is missing negative_spirit_id`);
          } else if (negativeSpiritId === spiritId) {
            errors.push(`examples line ${row.__line}: negative match row points at its own spirit_id`);
          }
          const negativeRole = matchNegativeRole(row);
          if (negativeRole !== "image" && negativeRole !== "prompt") {
            errors.push(`examples line ${row.__line}: negative match row has invalid negative_role ${JSON.stringify(row.negative_role)}`);
          } else {
            const roleGroup = ensureGroup(labelGroup.roles, negativeRole);
            roleGroup.records += 1;
            if (spiritId !== null) {
              roleGroup.spirits.add(spiritId);
            }
          }
          if (String(row.negative_selection || "") !== "nearest-image-token") {
            errors.push(`examples line ${row.__line}: negative match row negative_selection ${JSON.stringify(row.negative_selection || "")} != nearest-image-token`);
          }
          if (Number(row.negative_image_token_rank) !== 1) {
            errors.push(`examples line ${row.__line}: negative match row negative_image_token_rank ${JSON.stringify(row.negative_image_token_rank || "")} != 1`);
          }
          const distance = Number(row.negative_image_token_distance);
          if (!Number.isInteger(distance) || distance <= 0) {
            errors.push(`examples line ${row.__line}: negative match row has invalid negative_image_token_distance ${JSON.stringify(row.negative_image_token_distance || "")}`);
          }
        }
      }
    }
  }
  return {
    errors,
    distinct_spirits: allSpirits.size,
    v2_records: v2Records,
    missing_image_token_profile: missingImageTokenProfile,
    missing_image_token_channels: missingImageTokenChannels,
    image_token_profiles: Object.fromEntries([...profileCounts.entries()].sort(([left], [right]) => left.localeCompare(right))),
    image_token_channel_sets: Object.fromEntries([...channelSetCounts.entries()].sort(([left], [right]) => left.localeCompare(right))),
    image_token_channel_presence: Object.fromEntries(
      [...channelPresenceCounts.entries()].sort(([left], [right]) => left.localeCompare(right)),
    ),
    tasks: Object.fromEntries(
      [...groups.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([task, group]) => [
        task,
        groupSummary(group),
      ]),
    ),
  };
}

function increment(map, key) {
  map.set(key, (map.get(key) || 0) + 1);
}

function ensureGroup(map, key) {
  if (!map.has(key)) {
    map.set(key, { records: 0, spirits: new Set(), labels: new Map(), roles: new Map() });
  }
  return map.get(key);
}

function groupSummary(group) {
  const summary = {
    records: group.records,
    spirits: group.spirits.size,
  };
  if (group.labels.size > 0) {
    summary.labels = Object.fromEntries(
      [...group.labels.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([label, labelGroup]) => [
        label,
        groupSummary(labelGroup),
      ]),
    );
  }
  if (group.roles.size > 0) {
    summary.roles = Object.fromEntries(
      [...group.roles.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([role, roleGroup]) => [
        role,
        groupSummary(roleGroup),
      ]),
    );
  }
  return summary;
}

function matchNegativeRole(row) {
  const role = String(row.negative_role || "image").toLowerCase();
  if (role === "prompt" || role === "text" || role === "name") {
    return "prompt";
  }
  if (role === "image" || role === "seal") {
    return "image";
  }
  return role;
}

function normalizedId(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function checkEval(config, evalTrace) {
  const errors = [];
  if (evalTrace.schema !== "nsrl.solomon_attention_eval_trace.v1") {
    errors.push(`unexpected eval schema: ${JSON.stringify(evalTrace.schema)}`);
  }
  if (!evalTrace.tasks || typeof evalTrace.tasks !== "object" || Array.isArray(evalTrace.tasks)) {
    errors.push("eval trace is missing the v2 tasks object");
  }
  const skippedExamples = numberField(evalTrace, "skipped_examples");
  if (skippedExamples === null) {
    errors.push("eval trace is missing skipped_examples");
  } else if (skippedExamples > config.maxSkippedExamples) {
    errors.push(
      `skipped_examples ${skippedExamples} > ${config.maxSkippedExamples}`,
    );
  }
  const totalInvalid = numberField(evalTrace.total, "invalid_contexts");
  if (totalInvalid === null) {
    errors.push("eval total is missing invalid_contexts");
  } else if (totalInvalid > config.maxInvalidContexts) {
    errors.push(
      `total invalid_contexts ${totalInvalid} > ${config.maxInvalidContexts}`,
    );
  }
  if (
    config.minTotalAccuracy !== null &&
    numberField(evalTrace.total, "accuracy_per_mille") < config.minTotalAccuracy
  ) {
    errors.push(
      `total accuracy ${evalTrace.total?.accuracy_per_mille} < ${config.minTotalAccuracy}`,
    );
  }
  if (
    config.minTotalTop5 !== null &&
    numberField(evalTrace.total, "top5_accuracy_per_mille") < config.minTotalTop5
  ) {
    errors.push(
      `total top5 accuracy ${evalTrace.total?.top5_accuracy_per_mille} < ${config.minTotalTop5}`,
    );
  }
  for (const phase of EVAL_PHASES) {
    const minTargets = taskThreshold(config.minPhaseTargets, phase);
    if (minTargets === null) {
      continue;
    }
    const targets = numberField(evalTrace[phase], "targets");
    if (targets < minTargets) {
      errors.push(`${phase} targets ${targets} < ${minTargets}`);
    }
  }
  const outputHeadErrors = checkOutputHeads(config, evalTrace);
  errors.push(...outputHeadErrors);
  errors.push(...checkTaskPhaseStats(config, evalTrace));

  const taskStats = evalTrace.tasks && typeof evalTrace.tasks === "object" ? evalTrace.tasks : {};
  for (const task of config.requiredTasks) {
    const stats = taskStats[task];
    if (!stats) {
      errors.push(`missing required task eval stats: ${task}`);
      continue;
    }
    const targets = numberField(stats, "targets");
    if (!(targets > 0)) {
      errors.push(`${task} targets ${targets} must be > 0`);
    }
    const minTargets = taskThreshold(config.minTaskTargets, task);
    if (minTargets !== null && targets < minTargets) {
      errors.push(`${task} targets ${targets} < ${minTargets}`);
    }
    const invalid = numberField(stats, "invalid_contexts");
    if (invalid === null) {
      errors.push(`${task} is missing invalid_contexts`);
    } else if (invalid > config.maxInvalidContexts) {
      errors.push(`${task} invalid_contexts ${invalid} > ${config.maxInvalidContexts}`);
    }
    checkTaskRate(errors, task, stats, "accuracy_per_mille", config.minTaskAccuracy);
    checkTaskRate(errors, task, stats, "top5_accuracy_per_mille", config.minTaskTop5);
    checkTaskRate(errors, task, stats, "top10_accuracy_per_mille", config.minTaskTop10);
  }
  return errors;
}

function checkEvalProvenance(config, evalTrace, tokensPath) {
  const errors = [];
  const summary = {
    ok: true,
    examples_expected: config.examplesPath || "",
    examples_recorded: evalTrace.examples || "",
    examples_path_match: true,
    tokens_expected: tokensPath || "",
    tokens_recorded: evalTrace.tokens || "",
    tokens_path_match: true,
    token_count_expected: null,
    token_count_recorded: numberField(evalTrace, "token_count"),
    token_count_match: true,
    token_hash_expected: "",
    token_hash_recorded: evalTrace.token_hash || "",
    token_hash_match: true,
    errors,
  };

  if (config.examplesPath) {
    if (!evalTrace.examples) {
      summary.examples_path_match = false;
      errors.push("eval trace is missing examples path");
    } else if (!samePath(evalTrace.examples, config.examplesPath)) {
      summary.examples_path_match = false;
      errors.push(
        `eval trace examples ${JSON.stringify(evalTrace.examples)} does not match ${JSON.stringify(config.examplesPath)}`,
      );
    }
  }

  if (tokensPath) {
    if (!evalTrace.tokens) {
      summary.tokens_path_match = false;
      errors.push("eval trace is missing tokens path");
    } else if (!samePath(evalTrace.tokens, tokensPath)) {
      summary.tokens_path_match = false;
      errors.push(
        `eval trace tokens ${JSON.stringify(evalTrace.tokens)} does not match ${JSON.stringify(tokensPath)}`,
      );
    }

    let tokens = [];
    try {
      tokens = readTokens(tokensPath);
      summary.token_count_expected = tokens.length;
      summary.token_hash_expected = fnv64Hex(tokens);
      if (summary.token_count_recorded === null) {
        summary.token_count_match = false;
        errors.push("eval trace is missing token_count");
      } else if (summary.token_count_recorded !== tokens.length) {
        summary.token_count_match = false;
        errors.push(`eval trace token_count ${summary.token_count_recorded} != ${tokens.length}`);
      }
      if (!evalTrace.token_hash) {
        summary.token_hash_match = false;
        errors.push("eval trace is missing token_hash");
      } else if (String(evalTrace.token_hash) !== summary.token_hash_expected) {
        summary.token_hash_match = false;
        errors.push(
          `eval trace token_hash ${JSON.stringify(evalTrace.token_hash)} != ${summary.token_hash_expected}`,
        );
      }
    } catch (error) {
      summary.token_count_match = false;
      summary.token_hash_match = false;
      errors.push(`eval trace token provenance could not read ${tokensPath}: ${error.message}`);
    }
  }

  summary.ok = errors.length === 0;
  return summary;
}

function checkTaskPhaseStats(config, evalTrace) {
  if (!config.requireTaskPhaseStats && !config.requireDirectionalGroups) {
    return [];
  }
  const errors = [];
  const taskPhases = evalTrace.task_phases;
  if (!taskPhases || typeof taskPhases !== "object" || Array.isArray(taskPhases)) {
    return ["eval trace is missing task_phases"];
  }
  for (const task of config.requiredTasks) {
    const phases = taskPhases[task];
    if (!phases || typeof phases !== "object" || Array.isArray(phases)) {
      errors.push(`eval task_phases missing ${task}`);
      continue;
    }
    const targets = EVAL_PHASES.reduce((sum, phase) => sum + Number(phases[phase]?.targets || 0), 0);
    if (!(targets > 0)) {
      errors.push(`eval task_phases ${task} has no phase targets`);
    }
  }
  return errors;
}

function checkOutputHeads(config, evalTrace) {
  if (!config.requireOutputHeads) {
    return [];
  }
  const errors = [];
  const heads = evalTrace.output_heads;
  if (!heads || typeof heads !== "object" || Array.isArray(heads)) {
    return ["eval trace is missing output_heads"];
  }
  for (const headName of REQUIRED_OUTPUT_HEADS) {
    const head = heads[headName];
    if (!head || typeof head !== "object" || Array.isArray(head)) {
      errors.push(`eval output_heads missing ${headName}`);
      continue;
    }
    if (String(head.source || "") !== "nsrllmm-output-token-head") {
      errors.push(`eval output_heads.${headName}.source ${JSON.stringify(head.source)} != nsrllmm-output-token-head`);
    }
    if (!Array.isArray(head.token_ranges) || head.token_ranges.length === 0) {
      errors.push(`eval output_heads.${headName} has no token_ranges`);
    }
    if (!Array.isArray(head.token_classes) || head.token_classes.length === 0) {
      errors.push(`eval output_heads.${headName} has no token_classes`);
    }
    const allowedTokenCount = Number(head.allowed_token_count || 0);
    if (!Number.isInteger(allowedTokenCount) || allowedTokenCount <= 0) {
      errors.push(`eval output_heads.${headName}.allowed_token_count ${head.allowed_token_count} must be > 0`);
    }
    const stats = head.stats;
    if (!stats || typeof stats !== "object" || Array.isArray(stats)) {
      errors.push(`eval output_heads.${headName} is missing stats`);
      continue;
    }
    const targets = numberField(stats, "targets");
    if (!(targets > 0)) {
      errors.push(`eval output_heads.${headName} stats targets ${targets} must be > 0`);
    }
    const invalid = numberField(stats, "invalid_contexts");
    if (invalid === null) {
      errors.push(`eval output_heads.${headName} stats missing invalid_contexts`);
    } else if (invalid > config.maxInvalidContexts) {
      errors.push(`eval output_heads.${headName} invalid_contexts ${invalid} > ${config.maxInvalidContexts}`);
    }
  }
  const textHeadTargets = numberField(heads.text_head?.stats, "targets");
  const expectedTextTargets =
    Number(evalTrace.prompt?.targets || 0) + Number(evalTrace.text?.targets || 0);
  if (textHeadTargets !== null && textHeadTargets !== expectedTextTargets) {
    errors.push(`eval output_heads.text_head targets ${textHeadTargets} != prompt+text targets ${expectedTextTargets}`);
  }
  const imageHeadTargets = numberField(heads.image_head?.stats, "targets");
  const expectedImageTargets = Number(evalTrace.image?.targets || 0);
  if (imageHeadTargets !== null && imageHeadTargets !== expectedImageTargets) {
    errors.push(`eval output_heads.image_head targets ${imageHeadTargets} != image targets ${expectedImageTargets}`);
  }
  const specialHeadTargets = numberField(heads.special_head?.stats, "targets");
  const expectedSpecialTargets = Number(evalTrace.special?.targets || 0);
  if (specialHeadTargets !== null && specialHeadTargets !== expectedSpecialTargets) {
    errors.push(`eval output_heads.special_head targets ${specialHeadTargets} != special targets ${expectedSpecialTargets}`);
  }
  return errors;
}

function directionalGroupSummary(config, evalTrace, coverage) {
  const taskStats = evalTrace.tasks && typeof evalTrace.tasks === "object" ? evalTrace.tasks : {};
  const taskPhases =
    evalTrace.task_phases && typeof evalTrace.task_phases === "object" && !Array.isArray(evalTrace.task_phases)
      ? evalTrace.task_phases
      : {};
  const errors = [];
  const groups = {};
  for (const group of PRODUCT_DIRECTIONAL_GROUPS) {
    const taskTargets = {};
    const phaseTargets = {};
    const taskCoverage = {};
    let aggregateTargets = 0;
    let aggregateCoverageSpirits = null;
    const aggregateStats = [];
    for (const task of group.tasks) {
      const stats = taskStats[task] || {};
      const targets = Number(stats.targets || 0);
      taskTargets[task] = targets;
      aggregateTargets += targets;
      aggregateStats.push(stats);
      if (config.requireDirectionalGroups && targets <= 0) {
        errors.push(`directional group ${group.key} task ${task} has no eval targets`);
      }

      const coverageSpirits = coverage?.tasks?.[task]?.spirits;
      if (Number.isFinite(coverageSpirits)) {
        taskCoverage[task] = coverageSpirits;
        aggregateCoverageSpirits =
          aggregateCoverageSpirits === null ? coverageSpirits : Math.min(aggregateCoverageSpirits, coverageSpirits);
        if (config.requireDirectionalGroups && config.expectSpirits > 0 && coverageSpirits !== config.expectSpirits) {
          errors.push(
            `directional group ${group.key} task ${task} covers ${coverageSpirits} spirits, expected ${config.expectSpirits}`,
          );
        }
      }

      for (const phase of group.required_phases[task] || []) {
        const phaseStats = taskPhases?.[task]?.[phase];
        const phaseTargetCount = Number(phaseStats?.targets || 0);
        phaseTargets[`${task}:${phase}`] = phaseTargetCount;
        if (config.requireDirectionalGroups && phaseTargetCount <= 0) {
          errors.push(`directional group ${group.key} task ${task} phase ${phase} has no eval targets`);
        }
      }
    }
    const minTargets = directionalThreshold(config.minDirectionalTargets, group.key);
    const stats = mergeStatsSummaries(aggregateStats);
    const minAccuracy = directionalThreshold(config.minDirectionalAccuracy, group.key);
    const minTop5 = directionalThreshold(config.minDirectionalTop5, group.key);
    const minTop10 = directionalThreshold(config.minDirectionalTop10, group.key);
    if (minTargets !== null && aggregateTargets < minTargets) {
      errors.push(`directional group ${group.key} targets ${aggregateTargets} < ${minTargets}`);
    }
    checkDirectionalRate(errors, group.key, stats, "accuracy_per_mille", minAccuracy);
    checkDirectionalRate(errors, group.key, stats, "top5_accuracy_per_mille", minTop5);
    checkDirectionalRate(errors, group.key, stats, "top10_accuracy_per_mille", minTop10);
    groups[group.key] = {
      label: group.label,
      required: config.requireDirectionalGroups,
      tasks: group.tasks,
      required_phases: group.required_phases,
      targets: aggregateTargets,
      min_targets: minTargets,
      min_accuracy_per_mille: minAccuracy,
      min_top5_accuracy_per_mille: minTop5,
      min_top10_accuracy_per_mille: minTop10,
      stats,
      task_targets: taskTargets,
      phase_targets: phaseTargets,
      coverage_spirits_min: aggregateCoverageSpirits,
      task_coverage_spirits: taskCoverage,
      ok:
        group.tasks.every((task) => Number(taskTargets[task] || 0) > 0) &&
        Object.values(phaseTargets).every((targets) => Number(targets || 0) > 0) &&
        (minTargets === null || aggregateTargets >= minTargets) &&
        (minAccuracy === null || Number(stats.accuracy_per_mille) >= minAccuracy) &&
        (minTop5 === null || Number(stats.top5_accuracy_per_mille) >= minTop5) &&
        (minTop10 === null || Number(stats.top10_accuracy_per_mille) >= minTop10) &&
        (config.expectSpirits <= 0 ||
          aggregateCoverageSpirits === null ||
          aggregateCoverageSpirits === config.expectSpirits),
    };
  }
  return {
    required: config.requireDirectionalGroups,
    require_task_phase_stats: config.requireTaskPhaseStats,
    groups,
    errors,
  };
}

function checkDirectionalRate(errors, group, stats, field, threshold) {
  if (threshold === null) {
    return;
  }
  const value = numberField(stats, field);
  if (value < threshold) {
    errors.push(`directional group ${group} ${field} ${value} < ${threshold}`);
  }
}

function checkTaskRate(errors, task, stats, field, thresholds) {
  const threshold = taskThreshold(thresholds, task);
  if (threshold === null) {
    return;
  }
  const value = numberField(stats, field);
  if (value < threshold) {
    errors.push(`${task} ${field} ${value} < ${threshold}`);
  }
}

function checkExampleCoverage(config, coverage) {
  const errors = [...coverage.errors];
  if (config.requireImageTokenProfile || config.requireImageTokenChannels.length > 0) {
    if (coverage.v2_records <= 0) {
      errors.push("examples contain no v2 records for image-token contract checks");
    }
    if (coverage.missing_image_token_profile > 0) {
      errors.push(`examples have ${coverage.missing_image_token_profile} v2 records without image_token_profile`);
    }
    if (coverage.missing_image_token_channels > 0) {
      errors.push(`examples have ${coverage.missing_image_token_channels} v2 records without image_token_channels`);
    }
  }
  if (config.requireImageTokenProfile) {
    const count = Number(coverage.image_token_profiles?.[config.requireImageTokenProfile] || 0);
    if (count !== coverage.v2_records) {
      errors.push(
        `examples image_token_profile ${config.requireImageTokenProfile} covers ${count}/${coverage.v2_records} v2 records`,
      );
    }
  }
  for (const channel of config.requireImageTokenChannels) {
    const count = Number(coverage.image_token_channel_presence?.[channel] || 0);
    if (count !== coverage.v2_records) {
      errors.push(`examples image_token_channel ${channel} covers ${count}/${coverage.v2_records} v2 records`);
    }
  }
  if (config.expectSpirits <= 0) {
    return errors;
  }
  if (coverage.distinct_spirits !== config.expectSpirits) {
    errors.push(
      `examples distinct spirits ${coverage.distinct_spirits} != ${config.expectSpirits}`,
    );
  }
  for (const task of config.requiredTasks) {
    const taskCoverage = coverage.tasks[task];
    if (!taskCoverage) {
      errors.push(`examples are missing required task: ${task}`);
      continue;
    }
    if (taskCoverage.spirits !== config.expectSpirits) {
      errors.push(
        `examples task ${task} covers ${taskCoverage.spirits} spirits, expected ${config.expectSpirits}`,
      );
    }
    if (task === "match") {
      const labels = taskCoverage.labels || {};
      for (const label of ["yes", "no"]) {
        if (!labels[label]) {
          errors.push(`examples match task is missing ${label} rows`);
        } else if (labels[label].spirits !== config.expectSpirits) {
          errors.push(
            `examples match ${label} rows cover ${labels[label].spirits} spirits, expected ${config.expectSpirits}`,
          );
        }
      }
      const negativeRoles = labels.no?.roles || {};
      for (const role of ["image", "prompt"]) {
        if (!negativeRoles[role]) {
          errors.push(`examples match no rows are missing ${role} negative_role rows`);
        } else if (negativeRoles[role].spirits !== config.expectSpirits) {
          errors.push(
            `examples match no ${role} rows cover ${negativeRoles[role].spirits} spirits, expected ${config.expectSpirits}`,
          );
        }
      }
    }
  }
  return errors;
}

function resolveTokensPath(config, manifest) {
  if (config.tokensPath) {
    return config.tokensPath;
  }
  if (!config.manifestPath || !manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return "";
  }
  const tokenRef = manifest.corpus_tokens_u8 || manifest.corpus_tokens_u16 || "";
  if (!tokenRef) {
    return "";
  }
  if (tokenRef.startsWith("/") || /^[A-Za-z]:[\\/]/.test(tokenRef)) {
    return tokenRef;
  }
  return `${dirname(config.manifestPath)}/${tokenRef}`;
}

function dirname(path) {
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return slash < 0 ? "." : path.slice(0, slash);
}

function taskMarkerIntegrity(config, examples, manifest, tokensPath) {
  if (!examples || examples.length === 0 || !tokensPath) {
    return {
      ok: true,
      present: false,
      tokens: tokensPath || "",
      checked_records: 0,
      hash_mismatches: 0,
      marker_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
      by_task: {},
      errors: [],
    };
  }
  const errors = [];
  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ok: false,
      present: true,
      tokens: tokensPath,
      checked_records: 0,
      hash_mismatches: 0,
      marker_mismatches: 0,
      out_of_bounds: 0,
      missing_offsets: 0,
      by_task: {},
      errors: [`token file ${tokensPath}: ${error.message}`],
    };
  }
  const layout = {
    ...TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const byTask = new Map();
  let checkedRecords = 0;
  let hashMismatches = 0;
  let markerMismatches = 0;
  let outOfBounds = 0;
  let missingOffsets = 0;
  for (const row of examples) {
    const task = row.task || "canonical-joint";
    const expected = expectedTaskMarker(task, layout);
    if (!expected) {
      continue;
    }
    const taskSummary = ensureTaskMarkerGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`examples line ${row.__line}: ${task} missing valid token_offset/token_count`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(`examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length}`);
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const actualMarker = slice.slice(0, expected.length);
    if (!sameTokenPrefix(actualMarker, expected)) {
      markerMismatches += 1;
      taskSummary.marker_mismatches += 1;
      errors.push(
        `examples line ${row.__line}: ${task} token marker ${JSON.stringify(actualMarker)} != ${JSON.stringify(expected)}`,
      );
    }
    if (row.token_hash) {
      const actualHash = fnv64Hex(slice);
      if (actualHash !== row.token_hash) {
        hashMismatches += 1;
        taskSummary.hash_mismatches += 1;
        errors.push(`examples line ${row.__line}: ${task} token_hash ${actualHash} != ${row.token_hash}`);
      }
    }
  }
  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
    checked_records: checkedRecords,
    hash_mismatches: hashMismatches,
    marker_mismatches: markerMismatches,
    out_of_bounds: outOfBounds,
    missing_offsets: missingOffsets,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    errors,
  };
}

function absentTaskModalityIntegrity() {
  return {
    ok: true,
    present: false,
    tokens: "",
    checked_records: 0,
    missing_offsets: 0,
    out_of_bounds: 0,
    modality_mismatches: 0,
    by_task: {},
    errors: [],
  };
}

function taskModalityIntegrity(config, examples, manifest, tokensPath) {
  if (!examples || examples.length === 0 || !tokensPath) {
    return absentTaskModalityIntegrity();
  }
  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ...absentTaskModalityIntegrity(),
      ok: false,
      present: true,
      tokens: tokensPath,
      errors: [`token file ${tokensPath}: ${error.message}`],
    };
  }

  const layout = {
    ...TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const errors = [];
  const byTask = new Map();
  let checkedRecords = 0;
  let missingOffsets = 0;
  let outOfBounds = 0;
  let modalityMismatches = 0;
  for (const row of examples) {
    const task = row.task || "canonical-joint";
    const expected = expectedTaskModalities(task);
    if (!expected) {
      continue;
    }
    const taskSummary = ensureTaskModalityGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`examples line ${row.__line}: ${task} missing valid token_offset/token_count for modality order`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for modality order`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const rowErrors = checkTaskModalityOrder(task, slice, layout);
    if (rowErrors.length > 0) {
      modalityMismatches += 1;
      taskSummary.modality_mismatches += 1;
      for (const error of rowErrors) {
        errors.push(`examples line ${row.__line}: ${error}`);
      }
    }
  }
  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
    checked_records: checkedRecords,
    missing_offsets: missingOffsets,
    out_of_bounds: outOfBounds,
    modality_mismatches: modalityMismatches,
    by_task: Object.fromEntries([...byTask.entries()].sort(([left], [right]) => left.localeCompare(right))),
    errors,
  };
}

function absentImageChannelMarkerIntegrity(requiredChannels = []) {
  return {
    ok: true,
    present: false,
    tokens: "",
    required_channels: requiredChannels,
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
    errors: [],
  };
}

function imageChannelMarkerIntegrity(config, examples, manifest, tokensPath) {
  const requiredChannels = config.requireImageTokenChannels.map((channel) => String(channel));
  if (requiredChannels.length === 0) {
    return absentImageChannelMarkerIntegrity();
  }
  if (!examples || examples.length === 0) {
    return absentImageChannelMarkerIntegrity(requiredChannels);
  }
  if (!tokensPath) {
    const report = absentImageChannelMarkerIntegrity(requiredChannels);
    report.ok = false;
    report.errors = ["image channel marker integrity requires manifest corpus_tokens_u8/corpus_tokens_u16 or --tokens"];
    return report;
  }

  let tokens = [];
  try {
    tokens = readTokens(tokensPath);
  } catch (error) {
    return {
      ...absentImageChannelMarkerIntegrity(requiredChannels),
      ok: false,
      present: true,
      tokens: tokensPath,
      errors: [`token file ${tokensPath}: ${error.message}`],
    };
  }

  const layout = {
    ...TOKEN_LAYOUT_FALLBACK,
    ...(manifest?.token_layout && typeof manifest.token_layout === "object" ? manifest.token_layout : {}),
  };
  const imageToken = Number(layout.image ?? TOKEN_LAYOUT_FALLBACK.image);
  const imageBase = Number(layout.image_base ?? TOKEN_LAYOUT_FALLBACK.image_base);
  const imageBins = Number(layout.image_bins ?? TOKEN_LAYOUT_FALLBACK.image_bins);
  const payloadTokens = Number(manifest?.signature_bins || IMAGE_CHANNEL_PAYLOAD_TOKENS_FALLBACK);
  const v2ImageExamples = examples.filter(
    (row) => row?.schema === "nsrl.solomon_multimodal_example.v2" && IMAGE_BEARING_TASKS.has(String(row.task || "")),
  );
  const errors = [];
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

  if (v2ImageExamples.length === 0) {
    errors.push("image channel marker integrity found no image-bearing v2 records");
  }
  for (const row of v2ImageExamples) {
    const task = row.task || "";
    const taskSummary = ensureImageChannelMarkerGroup(byTask, task);
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    if (!Number.isInteger(offset) || !Number.isInteger(count) || offset < 0 || count <= 0) {
      missingOffsets += 1;
      taskSummary.missing_offsets += 1;
      errors.push(`examples line ${row.__line}: ${task} missing valid token_offset/token_count for image channel markers`);
      continue;
    }
    if (offset + count > tokens.length) {
      outOfBounds += 1;
      taskSummary.out_of_bounds += 1;
      errors.push(
        `examples line ${row.__line}: ${task} token slice ${offset}+${count} exceeds token file length ${tokens.length} for image channel markers`,
      );
      continue;
    }
    checkedRecords += 1;
    taskSummary.checked_records += 1;
    const slice = tokens.slice(offset, offset + count);
    const imageIndex = slice.indexOf(imageToken);
    if (imageIndex < 0) {
      missingImageMarkers += 1;
      taskSummary.missing_image_markers += 1;
      errors.push(`examples line ${row.__line}: ${task} token slice is missing IMAGE marker ${imageToken}`);
      continue;
    }

    let previousChannelPosition = imageIndex;
    for (const channel of requiredChannels) {
      const channelSummary = ensureImageChannelMarkerGroup(byChannel, channel);
      channelSummary.checked_records += 1;
      const marker = expectedImageChannelMarker(channel, layout);
      if (!Number.isInteger(marker)) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`image channel ${channel} has no token_layout image_channel_${channel} marker`);
        continue;
      }
      const markerCheck = findImageChannelPayload(slice, imageIndex + 1, marker, imageBase, imageBins, payloadTokens);
      if (!markerCheck.found) {
        missingChannelMarkers += 1;
        taskSummary.missing_channel_markers += 1;
        channelSummary.missing_channel_markers += 1;
        errors.push(`examples line ${row.__line}: ${task} missing image channel marker ${channel}:${marker}`);
        continue;
      }
      if (markerCheck.shortPayload) {
        shortChannelPayloads += 1;
        taskSummary.short_channel_payloads += 1;
        channelSummary.short_channel_payloads += 1;
        errors.push(
          `examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has fewer than ${payloadTokens} tokens`,
        );
        continue;
      }
      if (markerCheck.badPayload) {
        badChannelPayloads += 1;
        taskSummary.bad_channel_payloads += 1;
        channelSummary.bad_channel_payloads += 1;
        errors.push(
          `examples line ${row.__line}: ${task} image channel ${channel}:${marker} payload has token outside ${imageBase}..${
            imageBase + imageBins - 1
          }`,
        );
        continue;
      }
      if (markerCheck.position <= previousChannelPosition) {
        channelOrderMismatches += 1;
        taskSummary.channel_order_mismatches += 1;
        channelSummary.channel_order_mismatches += 1;
        errors.push(`examples line ${row.__line}: ${task} image channel ${channel}:${marker} is out of order`);
      }
      previousChannelPosition = markerCheck.position;
      taskSummary.found_markers += 1;
      channelSummary.found_markers += 1;
    }
  }

  return {
    ok: errors.length === 0,
    present: true,
    tokens: tokensPath,
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
    errors,
  };
}

function ensureTaskModalityGroup(map, key) {
  if (!map.has(key)) {
    map.set(key, {
      checked_records: 0,
      missing_offsets: 0,
      out_of_bounds: 0,
      modality_mismatches: 0,
    });
  }
  return map.get(key);
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

function checkTaskModalityOrder(task, slice, layout) {
  const expected = expectedTaskModalities(task);
  if (!expected) {
    return [];
  }
  const markerTokens = {
    prompt: Number(layout.prompt ?? TOKEN_LAYOUT_FALLBACK.prompt),
    text: Number(layout.text ?? 3),
    image: Number(layout.image ?? TOKEN_LAYOUT_FALLBACK.image),
  };
  const eosToken = Number(layout.eos ?? 5);
  const eosIndex = slice.indexOf(eosToken, 1);
  const searchEnd = eosIndex >= 0 ? eosIndex : slice.length;
  const positions = Object.fromEntries(
    Object.entries(markerTokens).map(([name, token]) => [
      name,
      markerPositions(slice, token).filter((position) => position > 0 && position < searchEnd),
    ]),
  );
  const errors = [];
  if (eosIndex < 0) {
    errors.push(`${task} modality order is missing EOS marker ${eosToken}`);
  }
  for (const name of expected) {
    const found = positions[name] || [];
    if (found.length !== 1) {
      errors.push(`${task} modality order expected exactly one ${name.toUpperCase()} marker, found ${found.length}`);
    }
  }
  for (const name of Object.keys(markerTokens)) {
    if (!expected.includes(name) && (positions[name] || []).length > 0) {
      errors.push(`${task} modality order has unexpected ${name.toUpperCase()} marker before EOS`);
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
      errors.push(
        `${task} modality order ${expected.join("->")} has ${name.toUpperCase()} at ${position} after ${previousName.toUpperCase()} at ${previousPosition}`,
      );
    }
    previousName = name;
    previousPosition = position;
  }
  return errors;
}

function markerPositions(tokens, marker) {
  const positions = [];
  for (let index = 0; index < tokens.length; index += 1) {
    if (Number(tokens[index]) === marker) {
      positions.push(index);
    }
  }
  return positions;
}

function ensureImageChannelMarkerGroup(map, key) {
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

function ensureTaskMarkerGroup(map, task) {
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

function expectedTaskMarker(task, layout) {
  const bos = Number(layout.bos ?? TOKEN_LAYOUT_FALLBACK.bos);
  const prompt = Number(layout.prompt ?? TOKEN_LAYOUT_FALLBACK.prompt);
  const image = Number(layout.image ?? TOKEN_LAYOUT_FALLBACK.image);
  if (task === "canonical-joint") return [bos, prompt];
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

function sameTokenPrefix(actual, expected) {
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

function fnv64Hex(tokens) {
  let hash = FNV_OFFSET;
  for (const token of tokens) {
    hash ^= BigInt(Number(token) & 0xff);
    hash = (hash * FNV_PRIME) & FNV_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function checkManifest(config, manifest) {
  const errors = [];
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return ["manifest is not a JSON object"];
  }
  if (config.requireCorpusVersion && manifest.corpus_version !== config.requireCorpusVersion) {
    errors.push(`manifest corpus_version ${JSON.stringify(manifest.corpus_version)} != ${JSON.stringify(config.requireCorpusVersion)}`);
  }
  if (config.requireImageTokenProfile && manifest.image_token_profile !== config.requireImageTokenProfile) {
    errors.push(
      `manifest image_token_profile ${JSON.stringify(manifest.image_token_profile)} != ${JSON.stringify(config.requireImageTokenProfile)}`,
    );
  }
  const channels = Array.isArray(manifest.image_token_channels)
    ? manifest.image_token_channels.map((channel) => String(channel))
    : [];
  if (config.requireImageTokenChannels.length > 0 && channels.length === 0) {
    errors.push("manifest is missing image_token_channels");
  }
  for (const channel of config.requireImageTokenChannels) {
    if (!channels.includes(channel)) {
      errors.push(`manifest image_token_channels missing ${channel}`);
    }
  }
  if (config.requireImageChannelTokenStats) {
    errors.push(...checkImageChannelTokenStats(config, manifest, channels));
  }
  return errors;
}

function checkImageChannelTokenStats(config, manifest, channels) {
  const errors = [];
  const stats = manifest.image_token_channel_stats;
  if (!stats || typeof stats !== "object" || Array.isArray(stats)) {
    return ["manifest is missing image_token_channel_stats"];
  }
  const requiredChannels = config.requireImageTokenChannels.length > 0 ? config.requireImageTokenChannels : channels;
  const expectedRecords = Number(manifest.rows || config.expectSpirits || 0);
  const expectedTokensPerRecord = Number(manifest.signature_bins || 0);
  for (const channel of requiredChannels) {
    const row = stats[channel];
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      errors.push(`manifest image_token_channel_stats missing ${channel}`);
      continue;
    }
    const records = Number(row.records || 0);
    const tokensPerRecord = Number(row.tokens_per_record || 0);
    const activeRecords = Number(row.active_records || 0);
    const multiBinRecords = Number(row.multi_bin_records || 0);
    const nonzeroTokens = Number(row.nonzero_tokens || 0);
    const distinctBins = Number(row.distinct_bins || 0);
    const maxBin = Number(row.max_bin || 0);
    const uniqueRecordHashes = Number(row.unique_record_hashes || 0);
    const duplicateRecordHashes = Number(row.duplicate_record_hashes || 0);
    if (expectedRecords > 0 && records !== expectedRecords) {
      errors.push(`manifest image_token_channel_stats ${channel} records ${records} != ${expectedRecords}`);
    }
    if (expectedTokensPerRecord > 0 && tokensPerRecord !== expectedTokensPerRecord) {
      errors.push(
        `manifest image_token_channel_stats ${channel} tokens_per_record ${tokensPerRecord} != ${expectedTokensPerRecord}`,
      );
    }
    if (records > 0 && activeRecords !== records) {
      errors.push(`manifest image_token_channel_stats ${channel} active_records ${activeRecords} != ${records}`);
    }
    if (records > 0 && multiBinRecords !== records) {
      errors.push(`manifest image_token_channel_stats ${channel} multi_bin_records ${multiBinRecords} != ${records}`);
    }
    if (nonzeroTokens <= 0) {
      errors.push(`manifest image_token_channel_stats ${channel} nonzero_tokens ${nonzeroTokens} <= 0`);
    }
    if (distinctBins < config.minImageChannelDistinctBins) {
      errors.push(
        `manifest image_token_channel_stats ${channel} distinct_bins ${distinctBins} < ${config.minImageChannelDistinctBins}`,
      );
    }
    if (maxBin <= 0) {
      errors.push(`manifest image_token_channel_stats ${channel} max_bin ${maxBin} <= 0`);
    }
    if (records > 0 && uniqueRecordHashes !== records) {
      errors.push(`manifest image_token_channel_stats ${channel} unique_record_hashes ${uniqueRecordHashes} != records ${records}`);
    }
    if (duplicateRecordHashes !== 0) {
      errors.push(`manifest image_token_channel_stats ${channel} duplicate_record_hashes ${duplicateRecordHashes} != 0`);
    }
  }
  return errors;
}

function manifestSummary(manifest) {
  if (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) {
    return null;
  }
  return {
    schema: manifest.schema || "",
    corpus_version: manifest.corpus_version || "",
    image_token_profile: manifest.image_token_profile || "",
    image_token_channels: Array.isArray(manifest.image_token_channels) ? manifest.image_token_channels : [],
    image_token_channel_stats:
      manifest.image_token_channel_stats && typeof manifest.image_token_channel_stats === "object"
        ? manifest.image_token_channel_stats
        : {},
    examples: numberField(manifest, "examples"),
    training_sequences: numberField(manifest, "training_sequences"),
    token_hash: manifest.token_hash || "",
  };
}

function outputHeadSummary(evalTrace) {
  const heads = evalTrace.output_heads;
  if (!heads || typeof heads !== "object" || Array.isArray(heads)) {
    return {};
  }
  return Object.fromEntries(
    REQUIRED_OUTPUT_HEADS.map((headName) => {
      const head = heads[headName] && typeof heads[headName] === "object" ? heads[headName] : {};
      return [
        headName,
        {
          source: String(head.source || ""),
          token_classes: Array.isArray(head.token_classes) ? head.token_classes.map(String) : [],
          token_ranges: Array.isArray(head.token_ranges) ? head.token_ranges : [],
          allowed_token_count: numberField(head, "allowed_token_count"),
          stats: statsSummary(head.stats || {}),
        },
      ];
    }),
  );
}

function taskPhaseSummary(evalTrace, requiredTasks) {
  const taskPhases =
    evalTrace.task_phases && typeof evalTrace.task_phases === "object" && !Array.isArray(evalTrace.task_phases)
      ? evalTrace.task_phases
      : {};
  return Object.fromEntries(
    requiredTasks.map((task) => {
      const phases = taskPhases[task] && typeof taskPhases[task] === "object" ? taskPhases[task] : {};
      return [
        task,
        Object.fromEntries(
          EVAL_PHASES.filter((phase) => phases[phase]).map((phase) => [
            phase,
            statsSummary(phases[phase] || {}),
          ]),
        ),
      ];
    }),
  );
}

function buildSummary(
  config,
  evalTrace,
  manifest,
  coverage,
  markerIntegrity,
  modalityIntegrity,
  imageChannelIntegrity,
  evalProvenance,
  directionalGroups,
  errors,
) {
  const taskStats = evalTrace.tasks && typeof evalTrace.tasks === "object" ? evalTrace.tasks : {};
  return {
    schema: "nsrl.solomon_attention_task_eval_check.v1",
    ok: errors.length === 0,
    eval: config.evalPath,
    manifest: config.manifestPath || null,
    examples: config.examplesPath || null,
    required_tasks: config.requiredTasks,
    expect_spirits: config.expectSpirits,
    required_corpus_version: config.requireCorpusVersion || null,
    required_image_token_profile: config.requireImageTokenProfile || null,
    required_image_token_channels: config.requireImageTokenChannels,
    require_image_channel_token_stats: config.requireImageChannelTokenStats,
    min_image_channel_distinct_bins: config.minImageChannelDistinctBins,
    require_output_heads: config.requireOutputHeads,
    manifest_summary: manifestSummary(manifest),
    skipped_examples: numberField(evalTrace, "skipped_examples"),
    total: statsSummary(evalTrace.total || {}),
    phases: Object.fromEntries(
      EVAL_PHASES.map((phase) => [phase, statsSummary(evalTrace[phase] || {})]),
    ),
    output_heads: outputHeadSummary(evalTrace),
    tasks: Object.fromEntries(
      config.requiredTasks.map((task) => [task, statsSummary(taskStats[task] || {})]),
    ),
    task_phases: taskPhaseSummary(evalTrace, config.requiredTasks),
    directional_groups: directionalGroups,
    coverage,
    task_marker_integrity: markerIntegrity,
    task_modality_integrity: modalityIntegrity,
    image_channel_marker_integrity: imageChannelIntegrity,
    eval_provenance: evalProvenance,
    errors,
  };
}

function main() {
  try {
    const config = parseArgs(process.argv.slice(2));
    const evalTrace = readJson(config.evalPath);
    const evalErrors = checkEval(config, evalTrace);
    const manifest = config.manifestPath ? readJson(config.manifestPath) : null;
    const manifestErrors = config.manifestPath ? checkManifest(config, manifest) : [];
    const tokensPath = resolveTokensPath(config, manifest);
    const evalProvenance = checkEvalProvenance(config, evalTrace, tokensPath);
    let coverage = null;
    let coverageErrors = [];
    let markerIntegrity = taskMarkerIntegrity(config, [], manifest, "");
    let modalityIntegrity = taskModalityIntegrity(config, [], manifest, "");
    let imageChannelIntegrity = imageChannelMarkerIntegrity(config, [], manifest, "");
    if (config.examplesPath) {
      const examples = readJsonl(config.examplesPath);
      coverage = inspectExamples(examples);
      coverageErrors = checkExampleCoverage(config, coverage);
      markerIntegrity = taskMarkerIntegrity(config, examples, manifest, tokensPath);
      modalityIntegrity = taskModalityIntegrity(config, examples, manifest, tokensPath);
      imageChannelIntegrity = imageChannelMarkerIntegrity(config, examples, manifest, tokensPath);
    }
    const directionalGroups = directionalGroupSummary(config, evalTrace, coverage || {});
    const errors = [
      ...evalErrors,
      ...manifestErrors,
      ...coverageErrors,
      ...evalProvenance.errors,
      ...directionalGroups.errors,
      ...markerIntegrity.errors,
      ...modalityIntegrity.errors,
      ...imageChannelIntegrity.errors,
    ];
    const summary = buildSummary(
      config,
      evalTrace,
      manifest,
      coverage,
      markerIntegrity,
      modalityIntegrity,
      imageChannelIntegrity,
      evalProvenance,
      directionalGroups,
      errors,
    );
    console.log(JSON.stringify(summary));
    if (errors.length > 0) {
      console.error(`Solomon attention task eval check failed with ${errors.length} error(s):`);
      for (const error of errors) {
        console.error(`- ${error}`);
      }
      process.exit(1);
    }
  } catch (error) {
    console.error(error.message);
    process.exit(2);
  }
}

main();
