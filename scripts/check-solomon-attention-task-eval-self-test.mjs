#!/usr/bin/env node

import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.solomon_attention_task_eval_self_test.v1";
const REQUIRED_TASKS = [
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
const CHANNELS = ["ink", "edge", "component", "radial", "direction"];
const EVAL_PHASES = ["special", "prompt", "text", "image"];
const TOKEN_LAYOUT = {
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
const IMAGE_CHANNEL_PAYLOAD_TOKENS = 256;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

function usage() {
  console.log([
    "Usage: check-solomon-attention-task-eval-self-test.mjs [--out PATH] [--keep]",
    "",
    "Builds tiny synthetic v2 attention task-eval fixtures and checks that the",
    "task-eval gate accepts the complete bidirectional contract while rejecting",
    "missing conditioning/output directional phase evidence, weak task coverage,",
    "missing hard-negative role coverage, broken channel stats, and inconsistent",
    "output-head accounting.",
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

function writeText(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, text, "utf8");
}

function writeJson(filePath, value) {
  writeText(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function buildFixture(root, name, mutate = () => {}) {
  const dir = path.join(root, name);
  const state = buildState();
  mutate(state);
  state.manifest.examples = state.examples.length;
  fs.mkdirSync(dir, { recursive: true });

  const evalPath = path.join(dir, "attention-eval.json");
  const examplesPath = path.join(dir, "examples.jsonl");
  const manifestPath = path.join(dir, "manifest.json");
  const tokensPath = path.join(dir, "corpus.tokens.u8");
  if (!state.evalTrace.examples) state.evalTrace.examples = examplesPath;
  if (!state.evalTrace.tokens) state.evalTrace.tokens = tokensPath;
  if (!state.evalTrace.token_count) state.evalTrace.token_count = state.tokens.length;
  if (!state.evalTrace.token_hash) state.evalTrace.token_hash = fnv64Hex(state.tokens);

  writeJson(evalPath, state.evalTrace);
  writeJson(manifestPath, state.manifest);
  writeText(examplesPath, `${state.examples.map((row) => JSON.stringify(row)).join("\n")}\n`);
  fs.writeFileSync(tokensPath, Buffer.from(state.tokens));

  return { dir, evalPath, examplesPath, manifestPath };
}

function mutateFirstExampleSlice(state, task, mutateSlice) {
  const row = state.examples.find((item) => item.task === task);
  if (!row) {
    throw new Error(`fixture is missing task ${task}`);
  }
  const offset = Number(row.token_offset);
  const count = Number(row.token_count);
  const slice = state.tokens.slice(offset, offset + count);
  if (slice.length !== count) {
    throw new Error(`fixture task ${task} has truncated token slice`);
  }
  mutateSlice(slice);
  state.tokens.splice(offset, count, ...slice);
  row.token_count = slice.length;
  row.token_hash = fnv64Hex(slice);
}

function buildState() {
  const examples = [];
  const tokens = [];
  for (let spiritId = 1; spiritId <= 2; spiritId += 1) {
    for (const task of REQUIRED_TASKS.filter((item) => item !== "match")) {
      addExample({ examples, tokens, spiritId, task });
    }
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "yes" });
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "no", negativeRole: "image" });
    addExample({ examples, tokens, spiritId, task: "match", matchLabel: "no", negativeRole: "prompt" });
  }
  return {
    evalTrace: syntheticEvalTrace(),
    manifest: syntheticManifest(),
    examples,
    tokens,
  };
}

function addExample({ examples, tokens, spiritId, task, matchLabel = "", negativeRole = "" }) {
  const slice = tokenSliceForTask(task, spiritId, matchLabel);
  const offset = tokens.length;
  tokens.push(...slice);
  const row = {
    schema: "nsrl.solomon_multimodal_example.v2",
    spirit_id: spiritId,
    task,
    text: task === "match" ? matchLabel : `synthetic prompt ${spiritId}`,
    image_token_profile: "symbolic16",
    image_token_channels: CHANNELS,
    token_offset: offset,
    token_count: slice.length,
    token_hash: fnv64Hex(slice),
  };
  if (task === "match") {
    row.match_label = matchLabel;
    if (matchLabel === "no") {
      row.negative_spirit_id = spiritId === 1 ? 2 : 1;
      row.negative_role = negativeRole;
      row.negative_selection = "nearest-image-token";
      row.negative_image_token_rank = 1;
      row.negative_image_token_distance = 100 + spiritId;
    }
  }
  examples.push(row);
}

function tokenSliceForTask(task, spiritId, matchLabel) {
  const promptPayload = 32 + spiritId;
  const textPayload = matchLabel === "no" ? 82 + spiritId : 64 + spiritId;
  const image = imagePayload(spiritId);
  if (task === "canonical-joint") return [1, 2, promptPayload, 3, textPayload, 4, ...image, 5];
  if (task === "identify") return [1, 10, 2, promptPayload, 3, textPayload, 5];
  if (task === "text-to-image") return [1, 6, 2, promptPayload, 4, ...image, 5];
  if (task === "description-to-image") return [1, 6, 2, promptPayload, 4, ...image, 5];
  if (task === "image-to-text") return [1, 7, 4, ...image, 3, textPayload, 5];
  if (task === "image-to-explain") return [1, 9, 4, ...image, 3, textPayload, 5];
  if (task === "image-to-attributes") return [1, 9, 4, ...image, 2, promptPayload, 3, textPayload, 5];
  if (task === "text-image-explain") return [1, 9, 2, promptPayload, 4, ...image, 3, textPayload, 5];
  if (task === "explain") return [1, 9, 2, promptPayload, 3, textPayload, 5];
  if (task === "match") return [1, 8, 2, promptPayload, 4, ...image, 3, textPayload, 5];
  throw new Error(`unknown task ${task}`);
}

function imagePayload(spiritId) {
  const payload = [];
  for (const [channelIndex, channel] of CHANNELS.entries()) {
    payload.push(TOKEN_LAYOUT[`image_channel_${channel}`]);
    for (let index = 0; index < IMAGE_CHANNEL_PAYLOAD_TOKENS; index += 1) {
      payload.push(TOKEN_LAYOUT.image_base + ((index + spiritId + channelIndex) % TOKEN_LAYOUT.image_bins));
    }
  }
  return payload;
}

function syntheticEvalTrace() {
  const tasks = Object.fromEntries(
    REQUIRED_TASKS.map((task) => [task, stat(task === "match" ? 6 : 2)]),
  );
  const taskPhases = {
    "canonical-joint": phaseStats(["prompt", "text", "image"]),
    identify: phaseStats(["prompt", "text"]),
    "text-to-image": phaseStats(["prompt", "image"]),
    "image-to-text": phaseStats(["image", "text"]),
    "image-to-explain": phaseStats(["image", "text"]),
    "text-image-explain": phaseStats(["prompt", "image", "text"]),
    "image-to-attributes": phaseStats(["image", "prompt", "text"]),
    explain: phaseStats(["prompt", "text"]),
    "description-to-image": phaseStats(["prompt", "image"]),
    match: phaseStats(["prompt", "image", "text"]),
  };
  return {
    schema: "nsrl.solomon_attention_eval_trace.v1",
    model_hash: "0xsynthetic",
    skipped_examples: 0,
    total: stat(32),
    special: stat(10),
    prompt: stat(20),
    text: stat(18),
    image: stat(14),
    tasks,
    task_phases: taskPhases,
    output_heads: {
      special_head: outputHead(["special"], [[1, 16]], 10),
      text_head: outputHead(["prompt", "text"], [[16, 144]], 38),
      image_head: outputHead(["image-channel", "image-bin"], [[11, 16], [144, 160]], 14),
    },
  };
}

function phaseStats(phases) {
  return Object.fromEntries(phases.map((phase) => [phase, stat(2)]));
}

function outputHead(tokenClasses, tokenRanges, targets) {
  return {
    source: "nsrllmm-output-token-head",
    token_classes: tokenClasses,
    token_ranges: tokenRanges,
    allowed_token_count: tokenRanges.reduce((sum, range) => sum + Number(range[1] - range[0]), 0),
    stats: stat(targets),
  };
}

function stat(targets) {
  return {
    targets,
    correct: targets,
    invalid_contexts: 0,
    accuracy_per_mille: targets > 0 ? 1000 : 0,
    top5_accuracy_per_mille: targets > 0 ? 1000 : 0,
    top10_accuracy_per_mille: targets > 0 ? 1000 : 0,
    mean_target_rank_per_mille: 1000,
    mean_target_margin_q8: 256,
  };
}

function syntheticManifest() {
  return {
    schema: "nsrl.solomon_multimodal_corpus_manifest.v2",
    corpus_version: "v2",
    image_token_profile: "symbolic16",
    image_token_channels: CHANNELS,
    image_token_channel_stats: Object.fromEntries(CHANNELS.map((channel) => [channel, channelStats()])),
    rows: 2,
    examples: 0,
    signature_bins: IMAGE_CHANNEL_PAYLOAD_TOKENS,
    corpus_tokens_u8: "corpus.tokens.u8",
    token_layout: TOKEN_LAYOUT,
  };
}

function channelStats() {
  return {
    records: 2,
    tokens_per_record: IMAGE_CHANNEL_PAYLOAD_TOKENS,
    active_records: 2,
    multi_bin_records: 2,
    nonzero_tokens: 512,
    distinct_bins: 8,
    max_bin: 15,
    unique_record_hashes: 2,
    duplicate_record_hashes: 0,
  };
}

function fnv64Hex(tokens) {
  let hash = FNV_OFFSET;
  for (const token of tokens) {
    hash ^= BigInt(Number(token) & 0xff);
    hash = (hash * FNV_PRIME) & FNV_MASK;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function runChecker(fixture) {
  return childProcess.spawnSync(process.execPath, [
    "scripts/check-solomon-attention-task-eval.mjs",
    "--eval",
    fixture.evalPath,
    "--examples",
    fixture.examplesPath,
    "--manifest",
    fixture.manifestPath,
    "--require-corpus-version",
    "v2",
    "--require-image-token-profile",
    "symbolic16",
    "--require-image-token-channels",
    CHANNELS.join(","),
    "--require-image-channel-token-stats",
    "--min-image-channel-distinct-bins",
    "2",
    "--expect-spirits",
    "2",
    "--min-task-targets",
    "all=1",
    "--min-phase-targets",
    "all=1",
    "--min-direction-targets",
    "all=1",
    "--min-direction-accuracy",
    "all=1000",
    "--min-direction-top5",
    "all=1000",
    "--min-direction-top10",
    "all=1000",
    "--require-directional-groups",
  ], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function extractReport(stdout) {
  const start = stdout.indexOf("{");
  if (start < 0) {
    return null;
  }
  return JSON.parse(stdout.slice(start));
}

function caseOk(item, result, report) {
  const actualOk = result.status === 0 && report?.ok === true;
  if (actualOk !== item.expectOk) {
    return false;
  }
  if (!item.errorIncludes) {
    return true;
  }
  const errors = (report?.errors || []).join("\n");
  return errors.includes(item.errorIncludes);
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-solomon-attention-task-eval-self-test-"));
  const definitions = [
    {
      name: "good",
      expectOk: true,
      mutate: () => {},
    },
    {
      name: "bad-directional-phase",
      expectOk: false,
      errorIncludes: "directional group text_prompt_to_image_plan task text-to-image phase image has no eval targets",
      mutate: (state) => {
        state.evalTrace.task_phases["text-to-image"].image.targets = 0;
      },
    },
    {
      name: "bad-forward-direction-prompt-phase",
      expectOk: false,
      errorIncludes: "directional group text_prompt_to_image_plan task text-to-image phase prompt has no eval targets",
      mutate: (state) => {
        state.evalTrace.task_phases["text-to-image"].prompt.targets = 0;
      },
    },
    {
      name: "bad-seal-direction-image-phase",
      expectOk: false,
      errorIncludes: "directional group seal_image_to_text task image-to-text phase image has no eval targets",
      mutate: (state) => {
        state.evalTrace.task_phases["image-to-text"].image.targets = 0;
      },
    },
    {
      name: "bad-joint-direction-image-phase",
      expectOk: false,
      errorIncludes: "directional group text_and_seal_to_explanation task text-image-explain phase image has no eval targets",
      mutate: (state) => {
        state.evalTrace.task_phases["text-image-explain"].image.targets = 0;
      },
    },
    {
      name: "bad-identity-source-prompt-phase",
      expectOk: false,
      errorIncludes: "directional group identity_source_binding task canonical-joint phase prompt has no eval targets",
      mutate: (state) => {
        state.evalTrace.task_phases["canonical-joint"].prompt.targets = 0;
      },
    },
    {
      name: "bad-directional-quality",
      expectOk: false,
      errorIncludes: "directional group seal_image_to_text top5_accuracy_per_mille",
      mutate: (state) => {
        state.evalTrace.tasks["image-to-text"].top5_accuracy_per_mille = 0;
      },
    },
    {
      name: "bad-task-coverage",
      expectOk: false,
      errorIncludes: "examples task description-to-image covers 1 spirits, expected 2",
      mutate: (state) => {
        state.examples = state.examples.filter(
          (row) => !(row.task === "description-to-image" && row.spirit_id === 2),
        );
      },
    },
    {
      name: "bad-match-negative-role-coverage",
      expectOk: false,
      errorIncludes: "examples match no rows are missing prompt negative_role rows",
      mutate: (state) => {
        state.examples = state.examples.filter(
          (row) => !(row.task === "match" && row.match_label === "no" && row.negative_role === "prompt"),
        );
      },
    },
    {
      name: "bad-task-marker",
      expectOk: false,
      errorIncludes: "text-to-image token marker",
      mutate: (state) => {
        mutateFirstExampleSlice(state, "text-to-image", (slice) => {
          slice[1] = TOKEN_LAYOUT.task_image_to_text;
        });
      },
    },
    {
      name: "bad-modality-order",
      expectOk: false,
      errorIncludes: "image-to-text modality order has unexpected PROMPT marker before EOS",
      mutate: (state) => {
        mutateFirstExampleSlice(state, "image-to-text", (slice) => {
          const textMarkerIndex = slice.indexOf(TOKEN_LAYOUT.text);
          if (textMarkerIndex < 0) {
            throw new Error("fixture image-to-text slice is missing text marker");
          }
          slice[textMarkerIndex] = TOKEN_LAYOUT.prompt;
        });
      },
    },
    {
      name: "bad-image-channel-marker",
      expectOk: false,
      errorIncludes: "text-to-image missing image channel marker edge:12",
      mutate: (state) => {
        mutateFirstExampleSlice(state, "text-to-image", (slice) => {
          const edgeMarkerIndex = slice.indexOf(TOKEN_LAYOUT.image_channel_edge);
          if (edgeMarkerIndex < 0) {
            throw new Error("fixture text-to-image slice is missing edge marker");
          }
          slice[edgeMarkerIndex] = TOKEN_LAYOUT.image_base;
        });
      },
    },
    {
      name: "bad-channel-stats",
      expectOk: false,
      errorIncludes: "manifest image_token_channel_stats edge distinct_bins 1 < 2",
      mutate: (state) => {
        state.manifest.image_token_channel_stats.edge.distinct_bins = 1;
      },
    },
    {
      name: "bad-channel-duplicate-records",
      expectOk: false,
      errorIncludes: "manifest image_token_channel_stats edge unique_record_hashes 1 != records 2",
      mutate: (state) => {
        state.manifest.image_token_channel_stats.edge.unique_record_hashes = 1;
        state.manifest.image_token_channel_stats.edge.duplicate_record_hashes = 1;
      },
    },
    {
      name: "bad-output-head",
      expectOk: false,
      errorIncludes: "eval output_heads.image_head targets 15 != image targets 14",
      mutate: (state) => {
        state.evalTrace.output_heads.image_head.stats.targets = 15;
      },
    },
    {
      name: "bad-eval-provenance",
      expectOk: false,
      errorIncludes: "eval trace token_hash",
      mutate: (state) => {
        state.evalTrace.token_hash = "0x0000000000000000";
      },
    },
  ];
  const cases = [];
  try {
    for (const definition of definitions) {
      const fixture = buildFixture(root, definition.name, definition.mutate);
      const result = runChecker(fixture);
      const report = extractReport(result.stdout || "");
      cases.push({
        name: definition.name,
        expect_ok: definition.expectOk,
        ok: caseOk(definition, result, report),
        status: result.status,
        checker_ok: report?.ok === true,
        errors: report?.errors || [],
      });
    }
    const report = {
      schema,
      ok: cases.every((item) => item.ok),
      root,
      kept: config.keep,
      cases,
    };
    if (config.outPath) {
      writeJson(config.outPath, report);
    }
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
