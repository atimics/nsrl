#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const signatureGrid = 16;
const signatureBins = signatureGrid * signatureGrid;
const requiredTasks = [
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
const requiredChannels = ["ink", "edge", "component", "radial", "direction"];
const requiredBindingKinds = ["primary-name", "primary-seal", "alias", "alias-seal", "seal-id"];
const retrievalImageTaskMinimums = {
  "text-to-image": 576,
  "description-to-image": 72,
  "image-to-text": 72,
  "image-to-explain": 72,
  "text-image-explain": 72,
  "image-to-attributes": 72,
};
const sourceProvenanceTasks = [
  "text-to-image",
  "image-to-text",
  "image-to-explain",
  "text-image-explain",
  "image-to-attributes",
  "explain",
  "description-to-image",
];
const sourceQueryKindByTask = {
  "text-to-image": "identity-to-image",
  "image-to-text": "image-identity",
  "image-to-explain": "image-source",
  "text-image-explain": "text-image-source",
  "image-to-attributes": "image-attributes",
  explain: "primary-name",
  "description-to-image": "source-description",
};
const solomonNames = [
  "Bael",
  "Agares",
  "Vassago",
  "Samigina",
  "Marbas",
  "Valefor",
  "Amon",
  "Barbatos",
  "Paimon",
  "Buer",
  "Gusion",
  "Sitri",
  "Beleth",
  "Leraje",
  "Eligos",
  "Zepar",
  "Botis",
  "Bathin",
  "Sallos",
  "Purson",
  "Marax",
  "Ipos",
  "Aim",
  "Naberius",
  "Glasya-Labolas",
  "Bune",
  "Ronove",
  "Berith",
  "Astaroth",
  "Forneus",
  "Foras",
  "Asmoday",
  "Gaap",
  "Furfur",
  "Marchosias",
  "Stolas",
  "Phenex",
  "Halphas",
  "Malphas",
  "Raum",
  "Focalor",
  "Vepar",
  "Sabnock",
  "Shax",
  "Vine",
  "Bifrons",
  "Uvall",
  "Haagenti",
  "Crocell",
  "Furcas",
  "Balam",
  "Alloces",
  "Camio",
  "Murmur",
  "Orobas",
  "Gremory",
  "Ose",
  "Amy",
  "Oriax",
  "Vapula",
  "Zagan",
  "Volac",
  "Andras",
  "Haures",
  "Andrealphus",
  "Cimejes",
  "Amdusias",
  "Belial",
  "Decarabia",
  "Seere",
  "Dantalion",
  "Andromalius",
];

function usage() {
  console.log(
    [
      "Usage: check-solomon-v2-corpus-contract.mjs [--keep]",
      "",
      "Builds a synthetic 72-spirit v2 Solomon multimodal corpus with symbolic16",
      "image channels, then proves task markers, hard-negative roles, grounding,",
      "retrieval-spine binding, and retrieval-head labels through the real repo",
      "scripts.",
      "",
      "Options:",
      "  --keep   keep the temporary fixture directory for debugging",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = { keep: false };
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--keep") {
      config.keep = true;
    } else {
      throw new Error(`unknown option: ${arg}`);
    }
  }
  return config;
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-v2-corpus-contract-"));
  let completed = false;
  try {
    const fixture = writeFixture(root);
    runNode("build v2 corpus", [
      "scripts/build-solomon-multimodal-corpus.mjs",
      "--text-index",
      fixture.textIndexPath,
      "--out-dir",
      fixture.corpusDir,
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

    const manifestPath = path.join(fixture.corpusDir, "manifest.json");
    const examplesPath = path.join(fixture.corpusDir, "examples.jsonl");
    const tokensPath = path.join(fixture.corpusDir, "corpus.tokens.u8");
    const manifest = readJson(manifestPath);
    const examples = readJsonl(examplesPath);
    const tokens = fs.readFileSync(tokensPath);
    const taskCounts = assertCorpusContract({ manifest, examples, tokens });
    const syntheticEvalPath = path.join(root, "task-eval.json");
    fs.writeFileSync(
      syntheticEvalPath,
      `${JSON.stringify(syntheticEval(taskCounts, { examplesPath, tokensPath, tokens }), null, 2)}\n`,
      "utf8",
    );

    const taskEvalContractReport = JSON.parse(runNode("check task eval corpus contract", [
      "scripts/check-solomon-attention-task-eval.mjs",
      "--eval",
      syntheticEvalPath,
      "--examples",
      examplesPath,
      "--manifest",
      manifestPath,
      "--tokens",
      tokensPath,
      "--expect-spirits",
      "72",
      "--require-corpus-version",
      "v2",
      "--require-image-token-profile",
      "symbolic16",
      "--require-image-token-channels",
      requiredChannels.join(","),
      "--require-image-channel-token-stats",
      "--min-image-channel-distinct-bins",
      "2",
      "--min-task-targets",
      "all=1",
      "--require-directional-groups",
    ]));

    const groundedCorpusPath = path.join(root, "grounded-corpus.json");
    runNode("check grounded corpus", [
      "scripts/check-solomon-v2-grounded-corpus.mjs",
      "--examples",
      examplesPath,
      "--text-index",
      fixture.textIndexPath,
      "--expect-spirits",
      "72",
      "--out",
      groundedCorpusPath,
    ]);

    runNode("check retrieval spine", [
      "scripts/check-solomon-v2-retrieval-spine.mjs",
      "--examples",
      examplesPath,
      "--tokens",
      tokensPath,
      "--text-index",
      fixture.textIndexPath,
      "--prompts",
      "none",
      "--max-misses",
      "0",
    ]);

    const retrievalHeadPath = path.join(root, "retrieval-head.json");
    const retrievalEvalPath = path.join(root, "retrieval-head-eval.json");
    runNode("train retrieval head", [
      "scripts/train-solomon-v2-retrieval-head.mjs",
      "--examples",
      examplesPath,
      "--tokens",
      tokensPath,
      "--text-index",
      fixture.textIndexPath,
      "--prompts",
      "none",
      "--model-out",
      retrievalHeadPath,
      "--eval-out",
      retrievalEvalPath,
      "--feature-count",
      "4096",
      "--epochs",
      "4",
      "--min-retrieval-margin",
      "0",
    ]);
    const retrievalHead = readJson(retrievalHeadPath);
    const retrievalEval = readJson(retrievalEvalPath);
    assertRetrievalHead(retrievalHead, retrievalEval);
    runNode("check retrieval head provenance", retrievalHeadProvenanceArgs({
      evalPath: retrievalEvalPath,
      retrievalHeadPath,
      examplesPath,
      tokensPath,
    }));
    const sampleBindingPath = path.join(root, "sample-binding.json");
    const generationIntegrityPath = path.join(root, "generation-integrity.json");
    writeJson(sampleBindingPath, sampleBindingTrace(retrievalHead));
    writeJson(generationIntegrityPath, generationIntegrityTrace());
    const curriculum = buildSyntheticCurriculumStage(root, fixture.corpusDir);
    const identityFixture = buildSyntheticIdentityInference(root, retrievalHeadPath, fixture.textIndexPath);
    runNode("check quality report", qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath: identityFixture.identityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath: curriculum.curriculumStagesPath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }));
    const negative_cases = runNegativeSelfTests({
      root,
      manifest,
      examples,
      tokens,
      syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      curriculumStagesPath: curriculum.curriculumStagesPath,
      curriculumStageDir: curriculum.imageStageDir,
      identityInferencePath: identityFixture.identityInferencePath,
      groundedCorpusPath,
      textIndexPath: fixture.textIndexPath,
    });

    completed = true;
    console.log(JSON.stringify({
      schema: "nsrl.solomon_v2_corpus_contract_check.v1",
      ok: true,
      examples: examples.length,
      token_count: tokens.length,
      task_counts: taskCounts,
      image_token_profile: manifest.image_token_profile,
      image_token_channels: manifest.image_token_channels,
      image_token_channel_stats: manifest.image_token_channel_stats || {},
      hard_negative_roles: hardNegativeRoleSummary(examples),
      identity_binding_coverage: identityBindingCoverageSummary(examples),
      source_provenance: sourceProvenanceSummary(examples),
      task_marker_integrity: compactIntegrity(taskEvalContractReport.task_marker_integrity),
      task_modality_integrity: compactIntegrity(taskEvalContractReport.task_modality_integrity),
      image_channel_marker_integrity: compactImageChannelIntegrity(taskEvalContractReport.image_channel_marker_integrity),
      retrieval_model_hash: retrievalHead.model_hash,
      retrieval_eval_hash: retrievalEval.model_hash,
      retrieval_head: retrievalHeadSummary(retrievalHead, retrievalEval),
      negative_cases,
    }, null, 2));
  } finally {
    if (completed && !config.keep) {
      fs.rmSync(root, { recursive: true, force: true });
    } else {
      console.error(`fixture_dir: ${root}`);
    }
  }
}

function writeFixture(root) {
  const dataDir = path.join(root, "data");
  const corpusDir = path.join(root, "corpus");
  fs.mkdirSync(dataDir, { recursive: true });
  fs.mkdirSync(corpusDir, { recursive: true });
  const textIndexPath = path.join(dataDir, "text-index.tsv");
  const lines = ["number\tprimary_name\taliases\ttext\tsignature_16x16"];
  for (let index = 0; index < solomonNames.length; index += 1) {
    const number = index + 1;
    const name = solomonNames[index];
    const alias = `${name} Fixture Alias`;
    const text = sourceText(name, alias, number);
    lines.push([
      number,
      name,
      alias,
      text,
      signatureForSpirit(number).join(","),
    ].map(escapeTsv).join("\t"));
  }
  fs.writeFileSync(textIndexPath, `${lines.join("\n")}\n`, "utf8");
  return { textIndexPath, corpusDir };
}

function sourceText(name, alias, number) {
  const ranks = ["king", "duke", "prince", "marquis", "president", "earl", "knight"];
  const offices = [
    "geometry herbs stones astronomy numbers and hidden language",
    "medicine metals waters stars and faithful answers",
    "rhetoric birds trees measures and secret causes",
    "logic maps plants minerals and swift memory",
  ];
  const rank = ranks[(number - 1) % ranks.length];
  const office = offices[(number - 1) % offices.length];
  const legions = 20 + (number % 50);
  return [
    `${name} is a ${rank} spirit in this source record.`,
    `He appears as a crowned figure bearing ${alias} and the seal of ${name}.`,
    `His office teaches ${office} for Solomon's bounded seal work.`,
    `He commands ${legions} legions and gives true answers from the source description.`,
  ].join(" ");
}

function signatureForSpirit(number) {
  const out = new Array(signatureBins).fill(0);
  const vertical = (number * 3) % signatureGrid;
  const horizontal = (number * 5 + 1) % signatureGrid;
  const diagonal = (number * 7 + 2) % signatureGrid;
  const dotX = (number * 11 + 3) % signatureGrid;
  const dotY = (number * 13 + 5) % signatureGrid;
  for (let y = 0; y < signatureGrid; y += 1) {
    for (let x = 0; x < signatureGrid; x += 1) {
      const index = y * signatureGrid + x;
      const onLine =
        x === vertical ||
        y === horizontal ||
        ((x + y) % signatureGrid) === diagonal ||
        (Math.abs(x - dotX) <= 1 && Math.abs(y - dotY) <= 1);
      if (onLine) {
        out[index] = 96 + ((number + x * 7 + y * 11) % 160);
      }
    }
  }
  for (let y = 12; y < signatureGrid; y += 1) {
    for (let x = 0; x < signatureGrid; x += 1) {
      out[y * signatureGrid + x] = 0;
    }
  }
  for (let bit = 0; bit < 8; bit += 1) {
    if ((number & (1 << bit)) !== 0) {
      const x = 1 + bit * 2;
      const y = 13;
      out[y * signatureGrid + x] = 160 + ((number + bit * 17) % 80);
    }
  }
  return out;
}

function assertCorpusContract({ manifest, examples, tokens }) {
  assertEqual(manifest.schema, "nsrl.solomon_multimodal_corpus.v1", "manifest schema");
  assertEqual(manifest.rows, 72, "manifest row count");
  assertEqual(manifest.corpus_version, "v2", "manifest corpus version");
  assertEqual(manifest.image_token_profile, "symbolic16", "manifest image token profile");
  assertArrayEqual(manifest.image_token_channels, requiredChannels, "manifest image token channels");
  for (const channel of requiredChannels) {
    const stats = manifest.image_token_channel_stats?.[channel];
    if (!stats) {
      throw new Error(`manifest missing image token channel stats for ${channel}`);
    }
    assertEqual(stats.records, 72, `${channel} channel records`);
    assertEqual(stats.tokens_per_record, signatureBins, `${channel} tokens per record`);
    assertEqual(stats.active_records, 72, `${channel} active records`);
    assertEqual(stats.multi_bin_records, 72, `${channel} multi-bin records`);
    assertAtLeast(stats.distinct_bins, 2, `${channel} distinct bins`);
    assertEqual(stats.unique_record_hashes, 72, `${channel} unique record hashes`);
    assertEqual(stats.duplicate_record_hashes, 0, `${channel} duplicate record hashes`);
  }

  const taskCounts = countBy(examples, (row) => row.task || "canonical-joint");
  for (const task of requiredTasks) {
    assertAtLeast(taskCounts[task] || 0, 72, `${task} record count`);
    const spirits = new Set(examples.filter((row) => (row.task || "canonical-joint") === task).map((row) => row.spirit_id));
    assertEqual(spirits.size, 72, `${task} spirit coverage`);
  }
  assertSourceProvenanceSummary(sourceProvenanceSummary(examples));
  const matchNoByRole = { image: new Set(), prompt: new Set() };
  for (const row of examples.filter((item) => item.task === "match" && item.match_label === "no")) {
    if (Number(row.negative_spirit_id) === Number(row.spirit_id)) {
      throw new Error(`match no row for spirit ${row.spirit_id} points at itself`);
    }
    assertEqual(row.negative_selection, "nearest-image-token", "match no negative selection");
    assertEqual(Number(row.negative_image_token_rank), 1, "match no negative image-token rank");
    assertAtLeast(Number(row.negative_image_token_distance), 1, "match no negative image-token distance");
    const role = row.negative_role === "prompt" ? "prompt" : "image";
    matchNoByRole[role].add(Number(row.spirit_id));
  }
  assertEqual(matchNoByRole.image.size, 72, "match no image-role spirit coverage");
  assertEqual(matchNoByRole.prompt.size, 72, "match no prompt-role spirit coverage");

  const bindingCoverage = {};
  for (const kind of requiredBindingKinds) {
    bindingCoverage[kind] = { identify: new Set(), "text-to-image": new Set() };
  }
  for (const row of examples.filter((item) => item.identity_binding === true)) {
    if (bindingCoverage[row.binding_kind]?.[row.task]) {
      bindingCoverage[row.binding_kind][row.task].add(Number(row.spirit_id));
    }
  }
  for (const kind of requiredBindingKinds) {
    assertEqual(bindingCoverage[kind].identify.size, 72, `${kind} identify binding coverage`);
    assertEqual(bindingCoverage[kind]["text-to-image"].size, 72, `${kind} text-to-image binding coverage`);
  }

  for (const row of examples.filter((item) => sourceProvenanceTasks.includes(item.task))) {
    assertEqual(Number(row.source_spirit_id), Number(row.spirit_id), `${row.task} source spirit id`);
    assertMatches(row.source_text_hash, /^0x[0-9a-f]{16}$/i, `${row.task} source text hash`);
    assertMatches(row.source_excerpt_hash, /^0x[0-9a-f]{16}$/i, `${row.task} source excerpt hash`);
    assertAtLeast(String(row.source_excerpt || "").length, 16, `${row.task} source excerpt length`);
  }

  const markerLayout = manifest.token_layout;
  for (const row of examples) {
    const offset = Number(row.token_offset);
    const count = Number(row.token_count);
    const slice = tokens.subarray(offset, offset + count);
    if (slice.length !== count) {
      throw new Error(`example token slice is truncated for line ${row.__line || "?"}`);
    }
    const expected = expectedTaskMarker(row.task || "canonical-joint", markerLayout);
    for (let index = 0; index < expected.length; index += 1) {
      if (slice[index] !== expected[index]) {
        throw new Error(
          `example ${row.task || "canonical-joint"} marker mismatch at ${index}: ${slice[index]} != ${expected[index]}`,
        );
      }
    }
    assertEqual(fnv64Bytes(slice), row.token_hash, `${row.task || "canonical-joint"} token hash`);
  }
  return taskCounts;
}

function expectedTaskMarker(task, layout) {
  if (task === "canonical-joint") return [layout.bos, layout.prompt];
  if (task === "identify") return [layout.bos, layout.task_identify, layout.prompt];
  if (task === "text-to-image" || task === "description-to-image") {
    return [layout.bos, layout.task_text_to_image, layout.prompt];
  }
  if (task === "image-to-text") return [layout.bos, layout.task_image_to_text, layout.image];
  if (task === "image-to-explain" || task === "image-to-attributes") {
    return [layout.bos, layout.task_explain, layout.image];
  }
  if (task === "text-image-explain" || task === "explain") {
    return [layout.bos, layout.task_explain, layout.prompt];
  }
  if (task === "match") return [layout.bos, layout.task_match, layout.prompt];
  throw new Error(`unknown task ${task}`);
}

function syntheticEval(taskCounts, provenance = {}) {
  const taskStats = {};
  const taskPhases = {};
  for (const task of requiredTasks) {
    const targets = Math.max(1, Number(taskCounts[task] || 0));
    taskStats[task] = metric(targets);
    taskPhases[task] = syntheticTaskPhases(task, targets);
  }
  const totalTargets = Object.values(taskStats).reduce((sum, row) => sum + row.targets, 0);
  const special = metric(1);
  const prompt = metric(1);
  const text = metric(1);
  const image = metric(1);
  return {
    schema: "nsrl.solomon_attention_eval_trace.v1",
    examples: provenance.examplesPath || "",
    tokens: provenance.tokensPath || "",
    token_count: provenance.tokens?.length || 0,
    token_hash: provenance.tokens ? fnv64Bytes(provenance.tokens) : "",
    skipped_examples: 0,
    d_model: 128,
    heads: 2,
    hidden_dim: 256,
    transformer_layers: 2,
    context_seq_len: 512,
    total: metric(totalTargets),
    special,
    prompt,
    text,
    image,
    output_heads: syntheticOutputHeads({ special, prompt, text, image }),
    tasks: taskStats,
    task_phases: taskPhases,
  };
}

function syntheticTaskPhases(task, targets) {
  const phases = {
    special: metric(1),
  };
  if ([
    "canonical-joint",
    "identify",
    "text-to-image",
    "description-to-image",
    "text-image-explain",
    "image-to-attributes",
    "explain",
    "match",
  ].includes(task)) {
    phases.prompt = metric(1);
  }
  if ([
    "canonical-joint",
    "identify",
    "image-to-text",
    "image-to-explain",
    "text-image-explain",
    "image-to-attributes",
    "explain",
    "match",
  ].includes(task)) {
    phases.text = metric(Math.max(1, targets));
  }
  if ([
    "canonical-joint",
    "text-to-image",
    "description-to-image",
    "image-to-text",
    "image-to-explain",
    "text-image-explain",
    "image-to-attributes",
    "match",
  ].includes(task)) {
    phases.image = metric(Math.max(1, targets));
  }
  return phases;
}

function syntheticOutputHeads({ special, prompt, text, image }) {
  return {
    special_head: {
      source: "nsrllmm-output-token-head",
      token_classes: ["control", "task"],
      token_ranges: [
        { name: "control", token_min: 2, token_max: 5 },
        { name: "task", token_min: 6, token_max: 10 },
      ],
      allowed_token_count: 9,
      stats: special,
    },
    text_head: {
      source: "nsrllmm-output-token-head",
      token_classes: ["prompt", "text"],
      token_ranges: [
        { name: "text_char_printable", token_min: 48, token_max: 142 },
        { name: "text_chunk", token_min: 160, token_max: 255 },
      ],
      allowed_token_count: 191,
      stats: mergeMetrics(prompt, text),
    },
    image_head: {
      source: "nsrllmm-output-token-head",
      token_classes: ["image_channel", "image_bin"],
      token_ranges: [
        { name: "image_channel", token_min: 11, token_max: 15 },
        { name: "image_bin", token_min: 144, token_max: 159 },
      ],
      allowed_token_count: 21,
      stats: image,
    },
  };
}

function mergeMetrics(...rows) {
  const targets = rows.reduce((sum, row) => sum + Number(row.targets || 0), 0);
  return metric(targets);
}

function metric(targets) {
  return {
    targets,
    correct: 0,
    invalid_contexts: 0,
    accuracy_per_mille: 0,
    top5_accuracy_per_mille: 0,
    top10_accuracy_per_mille: 0,
    mean_target_rank_per_mille: 0,
    mean_target_margin_q8: 0,
  };
}

function sampleBindingTrace(retrievalHead) {
  return {
    schema: "nsrl.solomon_attention_sample_binding_check.v1",
    ok: true,
    errors: [],
    retrieval_head: "retrieval-head.json",
    retrieval_head_model_hash: retrievalHead.model_hash,
    samples: 1,
    text_image_agreement: true,
    signature_retrieval_agreement: true,
    image_to_text_identification: true,
    min_signature_margin: 1,
    min_retrieval_image_margin: 1,
    min_image_to_text_margin: 1,
    min_retrieval_text_margin: 1,
    generated_text_identification: true,
    generated_text_image_agreement: true,
    min_generated_text_margin: 1,
    results: [],
  };
}

function generationIntegrityTrace() {
  return {
    schema: "nsrl.solomon_generation_integrity_check.v1",
    ok: true,
    trace_count: 1,
    violations: [],
  };
}

function buildSyntheticCurriculumStage(root, corpusDir) {
  const identityStageDir = path.join(root, "v2-stage-0-identity");
  const imageStageDir = path.join(root, "v2-stage-1-image-to-text");
  const curriculumStagesPath = path.join(root, "curriculum-stages.json");
  runNode("filter identity curriculum stage", [
    "scripts/filter-solomon-multimodal-corpus.mjs",
    "--input-dir",
    corpusDir,
    "--out-dir",
    identityStageDir,
    "--tasks",
    "identify,image-to-text,explain",
  ]);
  runNode("filter image-to-text curriculum stage", [
    "scripts/filter-solomon-multimodal-corpus.mjs",
    "--input-dir",
    corpusDir,
    "--out-dir",
    imageStageDir,
    "--tasks",
    "image-to-text,image-to-explain,text-image-explain,image-to-attributes",
  ]);
  for (const stageDir of [identityStageDir, imageStageDir]) {
    const manifest = readJson(path.join(stageDir, "manifest.json"));
    writeJson(path.join(stageDir, "train.json"), syntheticTrainTrace(manifest, stageDir));
  }
  runNode("check curriculum stages", [
    "scripts/check-solomon-v2-curriculum-stages.mjs",
    "--stage-dir",
    identityStageDir,
    "--stage-dir",
    imageStageDir,
    "--min-stages",
    "2",
    "--require-stage-names",
    "identity,image-to-text",
    "--out",
    curriculumStagesPath,
  ]);
  return { identityStageDir, imageStageDir, curriculumStagesPath };
}

function syntheticTrainTrace(manifest, stageDir) {
  const { tasks, taskPhases } = syntheticTrainTaskCoverage(manifest);
  const examplesJsonl = manifest.examples_jsonl || "examples.jsonl";
  const examplesPath = path.resolve(stageDir, examplesJsonl);
  return {
    schema: "nsrl.solomon_attention_train_trace.v1",
    model: "synthetic-curriculum-stage.nsrllmm",
    model_hash: "0x0000000000000001",
    token_hash: manifest.token_hash || "",
    corpus_coverage_source: "examples",
    corpus_examples_path: examplesJsonl,
    corpus_examples_hash: fnv64Bytes(fs.readFileSync(examplesPath)),
    corpus_examples: Number(manifest.examples || 0),
    corpus_skipped_examples: 0,
    corpus_prefix_pad_tokens: Number(manifest.examples || 0),
    corpus_orphan_tokens: 0,
    tasks,
    task_phases: taskPhases,
    attention_kind: "base2-softmax",
    text_token_profile: "chunked",
    d_model: 128,
    heads: 2,
    hidden_dim: 256,
    transformer_layers: 2,
    seq_len: 512,
    context_seq_len: 512,
    batch_mode: "serial",
    map_reduce_workers: 0,
    windows: 1,
    examined_windows: 1,
    updates: 1,
    accepted_batches: 1,
    rejected_batches: 0,
    probability_error_delta_i64: 0,
    initial_probability_error_q15: 1,
    final_probability_error_q15: 1,
  };
}

function syntheticTrainTaskCoverage(manifest) {
  const tasks = {};
  const taskPhases = {};
  for (const [task, coverage] of Object.entries(manifest.task_coverage?.tasks || {})) {
    const records = Number(coverage.records || 0);
    const phases = syntheticTrainTaskPhases(task);
    tasks[task] = {
      examples: records,
      targets: records * phases.length,
      special_targets: phases.includes("special") ? records : 0,
      prompt_targets: phases.includes("prompt") ? records : 0,
      text_targets: phases.includes("text") ? records : 0,
      image_targets: phases.includes("image") ? records : 0,
    };
    taskPhases[task] = Object.fromEntries(
      phases.map((phase) => [
        phase,
        {
          examples: records,
          targets: records,
          special_targets: phase === "special" ? records : 0,
          prompt_targets: phase === "prompt" ? records : 0,
          text_targets: phase === "text" ? records : 0,
          image_targets: phase === "image" ? records : 0,
        },
      ]),
    );
  }
  return { tasks, taskPhases };
}

function syntheticTrainTaskPhases(task) {
  switch (task) {
    case "text-to-image":
    case "description-to-image":
      return ["special", "prompt", "image"];
    case "image-to-text":
    case "image-to-explain":
      return ["special", "image", "text"];
    case "text-image-explain":
    case "match":
      return ["special", "prompt", "image", "text"];
    case "image-to-attributes":
      return ["special", "image", "prompt", "text"];
    case "identify":
    case "explain":
    default:
      return ["special", "prompt", "text"];
  }
}

function buildSyntheticIdentityInference(root, retrievalHeadPath, textIndexPath) {
  const sampleDir = path.join(root, "identity-sample-bael");
  const imagePath = path.join(sampleDir, "image.ink16.u8");
  const identityInferencePath = path.join(root, "identity-inference.json");
  fs.mkdirSync(sampleDir, { recursive: true });
  fs.writeFileSync(imagePath, Buffer.from(signatureForSpirit(1)));
  const generatedText = sourceText("Bael", "Bael Fixture Alias", 1);
  fs.writeFileSync(path.join(sampleDir, "text.txt"), `${generatedText}\n`, "utf8");
  writeJson(path.join(sampleDir, "sample.json"), {
    schema: "nsrl.solomon_attention_sample_trace.v1",
    prompt: "seal of Bael",
    generated_text: generatedText,
    image_ink16_u8: "image.ink16.u8",
  });
  runNode("infer synthetic identity", [
    "scripts/infer-solomon-v2-identity.mjs",
    "--retrieval-head",
    retrievalHeadPath,
    "--text-index",
    textIndexPath,
    "--text",
    "seal of Bael",
    "--image-ink16",
    imagePath,
    "--sample-dir",
    sampleDir,
    "--require-sample-agreement",
    "--require-source-evidence",
    "--max-misses",
    "0",
    "--out",
    identityInferencePath,
  ]);
  return { sampleDir, identityInferencePath };
}

function assertRetrievalHead(model, evalTrace) {
  assertEqual(model.schema, "nsrl.solomon_v2_retrieval_head.v1", "retrieval head schema");
  if (!Array.isArray(model.labels) || model.labels.length !== 72) {
    throw new Error(`retrieval head labels ${Array.isArray(model.labels) ? model.labels.length : 0} != 72`);
  }
  const ids = new Set(model.labels.map((label) => Number(label.spirit_id)));
  for (let spiritId = 1; spiritId <= 72; spiritId += 1) {
    if (!ids.has(spiritId)) {
      throw new Error(`retrieval head labels missing spirit ${spiritId}`);
    }
  }
  assertEqual(model.model_hash, evalTrace.model_hash, "retrieval head/eval model hash");
  for (const [task, minimum] of Object.entries(retrievalImageTaskMinimums)) {
    assertAtLeast(evalTrace.image_tasks?.[task]?.count || 0, minimum, `${task} retrieval rows`);
    assertAtLeast(evalTrace.image_tasks?.[task]?.top1 || 0, minimum, `${task} retrieval top1`);
  }
  assertAtLeast(evalTrace.match?.no_by_role?.image?.top1 || 0, 72, "match no image top1");
  assertAtLeast(evalTrace.match?.no_by_role?.prompt?.top1 || 0, 72, "match no prompt top1");
}

function retrievalHeadSummary(model, evalTrace) {
  const textHead = retrievalHeadComponentSummary(model.text_head, model.labels?.length || 0);
  const imageHead = retrievalHeadComponentSummary(model.image_head, model.labels?.length || 0);
  return {
    schema: model.schema || "",
    model_hash: model.model_hash || "",
    eval_model_hash: evalTrace.model_hash || "",
    feature_count: Number(model.feature_count || 0),
    labels: Array.isArray(model.labels) ? model.labels.length : 0,
    text_head: {
      present: textHead.ok,
      nonzero_weights: textHead.nonzero_weights,
    },
    image_head: {
      present: imageHead.ok,
      nonzero_weights: imageHead.nonzero_weights,
    },
    known_prompts: metricSummary(evalTrace.known_prompts),
    identity_bindings: {
      total: metricSummary(evalTrace.identity_bindings?.total),
      by_kind: Object.fromEntries(
        requiredBindingKinds.map((kind) => [
          kind,
          metricSummary(evalTrace.identity_bindings?.by_kind?.[kind]),
        ]),
      ),
    },
    image_to_text: metricSummary(evalTrace.image_to_text),
    image_tasks: Object.fromEntries(
      Object.keys(retrievalImageTaskMinimums).map((task) => [
        task,
        metricSummary(evalTrace.image_tasks?.[task]),
      ]),
    ),
    match: {
      yes: metricSummary(evalTrace.match?.yes),
      no: metricSummary(evalTrace.match?.no),
      no_by_role: {
        image: metricSummary(evalTrace.match?.no_by_role?.image),
        prompt: metricSummary(evalTrace.match?.no_by_role?.prompt),
      },
    },
  };
}

function hardNegativeRoleSummary(examples) {
  const roles = {
    image: new Set(),
    prompt: new Set(),
  };
  let totalNo = 0;
  let nearestImageToken = 0;
  let rankOne = 0;
  let positiveDistance = 0;
  for (const row of examples) {
    if (row.task !== "match" || row.match_label !== "no") {
      continue;
    }
    totalNo += 1;
    const role = row.negative_role === "prompt" ? "prompt" : "image";
    roles[role].add(Number(row.spirit_id));
    if (row.negative_selection === "nearest-image-token") {
      nearestImageToken += 1;
    }
    if (Number(row.negative_image_token_rank) === 1) {
      rankOne += 1;
    }
    if (Number(row.negative_image_token_distance) > 0) {
      positiveDistance += 1;
    }
  }
  return {
    no_rows: totalNo,
    image_role_spirits: roles.image.size,
    prompt_role_spirits: roles.prompt.size,
    nearest_image_token_rows: nearestImageToken,
    rank1_rows: rankOne,
    positive_distance_rows: positiveDistance,
  };
}

function identityBindingCoverageSummary(examples) {
  return Object.fromEntries(requiredBindingKinds.map((kind) => {
    const identify = new Set();
    const textToImage = new Set();
    for (const row of examples) {
      if (row.identity_binding !== true || row.binding_kind !== kind) {
        continue;
      }
      if (row.task === "identify") {
        identify.add(Number(row.spirit_id));
      } else if (row.task === "text-to-image") {
        textToImage.add(Number(row.spirit_id));
      }
    }
    return [kind, {
      identify_spirits: identify.size,
      text_to_image_spirits: textToImage.size,
    }];
  }));
}

function sourceProvenanceSummary(examples) {
  return Object.fromEntries(sourceProvenanceTasks.map((task) => {
    const spirits = new Set();
    let rows = 0;
    let sourceSpiritIds = 0;
    let sourceTextHashes = 0;
    let sourceExcerptHashes = 0;
    let sourceExcerpts = 0;
    let sourceQueryKinds = 0;
    let sourceQueryKindOk = 0;
    const queryKinds = {};
    const expectedSourceQueryKind = sourceQueryKindByTask[task] || "";
    for (const row of examples) {
      if (row.task !== task) {
        continue;
      }
      rows += 1;
      spirits.add(Number(row.spirit_id));
      const sourceQueryKind = String(row.source_query_kind || "");
      if (sourceQueryKind) {
        sourceQueryKinds += 1;
        queryKinds[sourceQueryKind] = (queryKinds[sourceQueryKind] || 0) + 1;
      }
      if (sourceQueryKind === expectedSourceQueryKind) {
        sourceQueryKindOk += 1;
      }
      if (Number(row.source_spirit_id) === Number(row.spirit_id)) {
        sourceSpiritIds += 1;
      }
      if (/^0x[0-9a-f]{16}$/i.test(String(row.source_text_hash || ""))) {
        sourceTextHashes += 1;
      }
      if (/^0x[0-9a-f]{16}$/i.test(String(row.source_excerpt_hash || ""))) {
        sourceExcerptHashes += 1;
      }
      if (String(row.source_excerpt || "").length >= 16) {
        sourceExcerpts += 1;
      }
    }
    return [task, {
      rows,
      spirits: spirits.size,
      source_spirit_id_rows: sourceSpiritIds,
      source_text_hash_rows: sourceTextHashes,
      source_excerpt_hash_rows: sourceExcerptHashes,
      source_excerpt_rows: sourceExcerpts,
      expected_source_query_kind: expectedSourceQueryKind,
      source_query_kind_rows: sourceQueryKinds,
      source_query_kind_ok_rows: sourceQueryKindOk,
      source_query_kinds: queryKinds,
    }];
  }));
}

function assertSourceProvenanceSummary(summary) {
  for (const task of sourceProvenanceTasks) {
    const row = summary[task] || {};
    const rows = Number(row.rows || 0);
    assertAtLeast(rows, 72, `${task} source provenance rows`);
    assertAtLeast(Number(row.spirits || 0), 72, `${task} source provenance spirit coverage`);
    assertEqual(Number(row.source_spirit_id_rows || 0), rows, `${task} source spirit provenance rows`);
    assertEqual(Number(row.source_text_hash_rows || 0), rows, `${task} source text hash rows`);
    assertEqual(Number(row.source_excerpt_hash_rows || 0), rows, `${task} source excerpt hash rows`);
    assertEqual(Number(row.source_excerpt_rows || 0), rows, `${task} source excerpt rows`);
    assertEqual(Number(row.source_query_kind_rows || 0), rows, `${task} source query kind rows`);
    assertEqual(Number(row.source_query_kind_ok_rows || 0), rows, `${task} source query kind ok rows`);
  }
}

function compactIntegrity(integrity) {
  return {
    ok: integrity?.ok === true,
    checked_records: Number(integrity?.checked_records || 0),
    hash_mismatches: Number(integrity?.hash_mismatches || 0),
    marker_mismatches: Number(integrity?.marker_mismatches || 0),
    modality_mismatches: Number(integrity?.modality_mismatches || 0),
    out_of_bounds: Number(integrity?.out_of_bounds || 0),
    missing_offsets: Number(integrity?.missing_offsets || 0),
    by_task: integrity?.by_task || {},
  };
}

function compactImageChannelIntegrity(integrity) {
  return {
    ok: integrity?.ok === true,
    checked_records: Number(integrity?.checked_records || 0),
    required_channels: Array.isArray(integrity?.required_channels)
      ? integrity.required_channels.map(String)
      : [],
    missing_offsets: Number(integrity?.missing_offsets || 0),
    out_of_bounds: Number(integrity?.out_of_bounds || 0),
    missing_image_markers: Number(integrity?.missing_image_markers || 0),
    missing_channel_markers: Number(integrity?.missing_channel_markers || 0),
    short_channel_payloads: Number(integrity?.short_channel_payloads || 0),
    bad_channel_payloads: Number(integrity?.bad_channel_payloads || 0),
    channel_order_mismatches: Number(integrity?.channel_order_mismatches || 0),
    by_task: integrity?.by_task || {},
    by_channel: integrity?.by_channel || {},
  };
}

function metricSummary(metric) {
  return {
    count: Number(metric?.count || 0),
    top1: Number(metric?.top1 || 0),
    top5: Number(metric?.top5 || 0),
    min_margin: Number(metric?.min_margin ?? 0),
    top1_per_mille: Number(metric?.top1_per_mille || metric?.top1_accuracy_per_mille || 0),
    top5_per_mille: Number(metric?.top5_per_mille || metric?.top5_accuracy_per_mille || 0),
  };
}

function retrievalHeadComponentSummary(head, labelCount) {
  const biases = Array.isArray(head?.biases) ? head.biases.length : 0;
  const weights = Array.isArray(head?.weights) ? head.weights.length : 0;
  let malformedRows = 0;
  let nonzeroWeights = 0;
  if (Array.isArray(head?.weights)) {
    for (const row of head.weights) {
      if (!Array.isArray(row)) {
        malformedRows += 1;
        continue;
      } else {
        nonzeroWeights += row.length;
      }
      for (const entry of row) {
        if (!Array.isArray(entry) || entry.length !== 2 || !Number.isInteger(Number(entry[0]))) {
          malformedRows += 1;
          break;
        }
      }
    }
  }
  return {
    ok: labelCount > 0 && biases === labelCount && weights === labelCount && malformedRows === 0,
    biases,
    weights,
    malformed_rows: malformedRows,
    nonzero_weights: nonzeroWeights,
  };
}

function runNegativeSelfTests({
  root,
  manifest,
  examples,
  tokens,
  syntheticEvalPath,
  retrievalHeadPath,
  retrievalEvalPath,
  sampleBindingPath,
  generationIntegrityPath,
  curriculumStagesPath,
  curriculumStageDir,
  identityInferencePath,
  groundedCorpusPath,
  textIndexPath,
}) {
  const out = [];
  const examplesPath = path.join(root, "corpus", "examples.jsonl");
  const manifestPath = path.join(root, "corpus", "manifest.json");
  const tokensPath = path.join(root, "corpus", "corpus.tokens.u8");

  const weakManifestPath = path.join(root, "negative-weak-image-profile-manifest.json");
  const weakManifest = JSON.parse(JSON.stringify(manifest));
  weakManifest.image_token_profile = "ink16";
  weakManifest.image_token_channels = ["ink"];
  weakManifest.image_token_channel_stats = {
    ink: weakManifest.image_token_channel_stats?.ink,
  };
  writeJson(weakManifestPath, weakManifest);
  assertCommandFailure(
    "weak image profile should fail task eval contract",
    taskEvalArgs({
      evalPath: syntheticEvalPath,
      examplesPath,
      manifestPath: weakManifestPath,
      tokensPath,
    }),
    "manifest image_token_profile",
  );
  out.push({ name: "weak-image-profile", ok: true });

  const missingPromptNegativePath = path.join(root, "negative-missing-prompt-hard-negatives.jsonl");
  const missingPromptNegativeExamples = examples.filter(
    (row) => !(row.task === "match" && row.match_label === "no" && row.negative_role === "prompt"),
  );
  writeJsonl(missingPromptNegativePath, missingPromptNegativeExamples);
  assertThrows(
    () => assertCorpusContract({ manifest, examples: missingPromptNegativeExamples, tokens }),
    "match no prompt-role spirit coverage",
    "direct contract should fail without prompt hard negatives",
  );
  assertCommandFailure(
    "missing prompt hard negatives should fail task eval contract",
    taskEvalArgs({
      evalPath: syntheticEvalPath,
      examplesPath: missingPromptNegativePath,
      manifestPath,
      tokensPath,
    }),
    "examples match no rows are missing prompt negative_role rows",
  );
  out.push({ name: "missing-prompt-hard-negatives", ok: true });

  const badHardNegativeMetadataPath = path.join(root, "negative-bad-hard-negative-metadata.jsonl");
  const badHardNegativeMetadataExamples = examples.map((row) => {
    const copy = { ...row };
    if (copy.task === "match" && copy.match_label === "no") {
      copy.negative_selection = "hash-random";
      copy.negative_image_token_rank = 2;
      delete copy.negative_image_token_distance;
    }
    return copy;
  });
  writeJsonl(badHardNegativeMetadataPath, badHardNegativeMetadataExamples);
  assertThrows(
    () => assertCorpusContract({ manifest, examples: badHardNegativeMetadataExamples, tokens }),
    "negative selection",
    "direct contract should fail with bad hard-negative metadata",
  );
  assertCommandFailure(
    "bad hard-negative metadata should fail task eval contract",
    taskEvalArgs({
      evalPath: syntheticEvalPath,
      examplesPath: badHardNegativeMetadataPath,
      manifestPath,
      tokensPath,
    }),
    "negative_selection",
  );
  assertCommandFailure(
    "bad hard-negative metadata should fail retrieval spine",
    [
      "scripts/check-solomon-v2-retrieval-spine.mjs",
      "--examples",
      badHardNegativeMetadataPath,
      "--tokens",
      tokensPath,
      "--text-index",
      textIndexPath,
      "--prompts",
      "none",
      "--max-misses",
      "0",
    ],
    "negative_selection",
  );
  out.push({ name: "bad-hard-negative-metadata", ok: true });

  const missingSourceProvenancePath = path.join(root, "negative-missing-source-provenance.jsonl");
  const missingSourceProvenanceExamples = examples.map((row) => {
    const copy = { ...row };
    if (sourceProvenanceTasks.includes(copy.task)) {
      delete copy.source_spirit_id;
      delete copy.source_text_hash;
      delete copy.source_excerpt;
      delete copy.source_excerpt_hash;
    }
    return copy;
  });
  writeJsonl(missingSourceProvenancePath, missingSourceProvenanceExamples);
  assertCommandFailure(
    "missing source provenance should fail grounded corpus contract",
    [
      "scripts/check-solomon-v2-grounded-corpus.mjs",
      "--examples",
      missingSourceProvenancePath,
      "--text-index",
      textIndexPath,
      "--expect-spirits",
      "72",
    ],
    "source_text_hash",
  );
  out.push({ name: "missing-source-provenance", ok: true });

  const badSourceQueryKindExamples = examples.map((row) => {
    const copy = { ...row };
    if (copy.task === "image-to-explain") {
      copy.source_query_kind = "image-identity";
    }
    return copy;
  });
  assertThrows(
    () => assertSourceProvenanceSummary(sourceProvenanceSummary(badSourceQueryKindExamples)),
    "image-to-explain source query kind ok rows",
    "bad source query kind should fail corpus provenance summary",
  );
  out.push({ name: "bad-source-query-kind", ok: true });

  const corruptTokensPath = path.join(root, "negative-corrupt-task-marker.u8");
  const corruptTokens = Buffer.from(tokens);
  const firstIdentify = examples.find((row) => row.task === "identify");
  if (!firstIdentify) {
    throw new Error("could not find identify example for corrupt marker negative test");
  }
  corruptTokens[Number(firstIdentify.token_offset) + 1] = 8;
  fs.writeFileSync(corruptTokensPath, corruptTokens);
  assertCommandFailure(
    "corrupt task marker should fail task eval contract",
    taskEvalArgs({
      evalPath: syntheticEvalPath,
      examplesPath,
      manifestPath,
      tokensPath: corruptTokensPath,
    }),
    "token marker",
  );
  out.push({ name: "corrupt-task-marker", ok: true });

  const badModalityTokensPath = path.join(root, "negative-bad-modality-order.u8");
  const badModalityExamplesPath = path.join(root, "negative-bad-modality-order.jsonl");
  const badModalityTokens = Buffer.from(tokens);
  const badModalityExamples = examples.map((row) => ({ ...row }));
  const firstIdentifyIndex = badModalityExamples.findIndex((row) => row.task === "identify");
  if (firstIdentifyIndex < 0) {
    throw new Error("could not find identify example for modality-order negative test");
  }
  const identifyRow = badModalityExamples[firstIdentifyIndex];
  const identifyOffset = Number(identifyRow.token_offset);
  const identifyCount = Number(identifyRow.token_count);
  const identifySlice = badModalityTokens.subarray(identifyOffset, identifyOffset + identifyCount);
  const identifyPrompt = identifySlice.indexOf(Number(manifest.token_layout.prompt ?? 2));
  const identifyText = identifySlice.indexOf(Number(manifest.token_layout.text ?? 3));
  if (identifyPrompt < 0 || identifyText <= identifyPrompt + 1) {
    throw new Error("identify example has no prompt payload for modality-order negative test");
  }
  badModalityTokens[identifyOffset + identifyPrompt + 1] = Number(manifest.token_layout.image ?? 4);
  identifyRow.token_hash = fnv64Bytes(badModalityTokens.subarray(identifyOffset, identifyOffset + identifyCount));
  fs.writeFileSync(badModalityTokensPath, badModalityTokens);
  writeJsonl(badModalityExamplesPath, badModalityExamples);
  const badModalityRetrievalEvalPath = path.join(root, "negative-bad-modality-order-retrieval-eval.json");
  const badModalityRetrievalEval = readJson(retrievalEvalPath);
  badModalityRetrievalEval.examples = badModalityExamplesPath;
  badModalityRetrievalEval.examples_hash = fnv64Bytes(fs.readFileSync(badModalityExamplesPath));
  badModalityRetrievalEval.tokens = badModalityTokensPath;
  badModalityRetrievalEval.tokens_hash = fnv64Bytes(badModalityTokens);
  writeJson(badModalityRetrievalEvalPath, badModalityRetrievalEval);
  assertCommandFailure(
    "bad modality order should fail task eval contract",
    taskEvalArgs({
      evalPath: syntheticEvalPath,
      examplesPath: badModalityExamplesPath,
      manifestPath,
      tokensPath: badModalityTokensPath,
    }),
    "modality order",
  );
  assertCommandFailure(
    "bad modality order should fail quality report",
    qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath: badModalityRetrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      examplesPath: badModalityExamplesPath,
      manifestPath,
      tokensPath: badModalityTokensPath,
    }),
    "modality order",
  );
  out.push({ name: "bad-modality-order", ok: true });

  const missingOutputHeadsEvalPath = path.join(root, "negative-missing-output-heads-eval.json");
  const missingOutputHeadsEval = readJson(syntheticEvalPath);
  delete missingOutputHeadsEval.output_heads;
  writeJson(missingOutputHeadsEvalPath, missingOutputHeadsEval);
  assertCommandFailure(
    "missing output heads should fail task eval contract",
    taskEvalArgs({
      evalPath: missingOutputHeadsEvalPath,
      examplesPath,
      manifestPath,
      tokensPath,
    }),
    "output_heads",
  );
  out.push({ name: "missing-output-heads", ok: true });

  const missingTaskPhasesEvalPath = path.join(root, "negative-missing-task-phases-eval.json");
  const missingTaskPhasesEval = readJson(syntheticEvalPath);
  delete missingTaskPhasesEval.task_phases;
  writeJson(missingTaskPhasesEvalPath, missingTaskPhasesEval);
  assertCommandFailure(
    "missing task phases should fail task eval contract",
    taskEvalArgs({
      evalPath: missingTaskPhasesEvalPath,
      examplesPath,
      manifestPath,
      tokensPath,
    }),
    "task_phases",
  );
  assertCommandFailure(
    "missing task phases should fail quality report",
    qualityReportArgs({
      evalPath: missingTaskPhasesEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      examplesPath,
      manifestPath,
      tokensPath,
    }),
    "task_phases",
  );
  out.push({ name: "missing-task-phases", ok: true });

  const badPromotedSmallEvalPath = path.join(root, "negative-bad-promoted-small-eval.json");
  const badPromotedSmallEval = readJson(syntheticEvalPath);
  badPromotedSmallEval.d_model = 64;
  writeJson(badPromotedSmallEvalPath, badPromotedSmallEval);
  assertCommandFailure(
    "bad promoted small profile should fail quality report",
    qualityReportArgs({
      evalPath: badPromotedSmallEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }),
    '"promoted_small_profile_ready": false',
  );
  out.push({ name: "bad-promoted-small-profile", ok: true });

  const badCurriculumTrainProfilePath = path.join(root, "negative-bad-curriculum-train-profile.json");
  const badCurriculumTrainProfile = readJson(curriculumStagesPath);
  badCurriculumTrainProfile.stages[0].train.d_model = 64;
  writeJson(badCurriculumTrainProfilePath, badCurriculumTrainProfile);
  assertCommandFailure(
    "bad curriculum train profile should fail quality report",
    qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath: badCurriculumTrainProfilePath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }),
    '"curriculum_ready": false',
  );
  out.push({ name: "bad-curriculum-train-profile", ok: true });

  const missingCurriculumTrainCoverageStageDir = corruptCurriculumStageTrainCoverage(root, curriculumStageDir);
  assertCommandFailure(
    "missing curriculum train task coverage should fail curriculum stage contract",
    [
      "scripts/check-solomon-v2-curriculum-stages.mjs",
      "--stage-dir",
      missingCurriculumTrainCoverageStageDir,
      "--min-stages",
      "1",
      "--require-stage-names",
      "image-to-text",
    ],
    "native train task coverage",
  );
  out.push({ name: "missing-curriculum-train-task-coverage", ok: true });

  const badCurriculumTrainExamplesStageDir = corruptCurriculumStageTrainExamplesProvenance(root, curriculumStageDir);
  assertCommandFailure(
    "bad curriculum train examples provenance should fail curriculum stage contract",
    [
      "scripts/check-solomon-v2-curriculum-stages.mjs",
      "--stage-dir",
      badCurriculumTrainExamplesStageDir,
      "--min-stages",
      "1",
      "--require-stage-names",
      "image-to-text",
    ],
    "corpus_examples_hash",
  );
  out.push({ name: "bad-curriculum-train-examples-provenance", ok: true });

  const badCurriculumStageDir = corruptCurriculumStageModality(root, curriculumStageDir);
  assertCommandFailure(
    "bad curriculum modality order should fail curriculum stage contract",
    [
      "scripts/check-solomon-v2-curriculum-stages.mjs",
      "--stage-dir",
      badCurriculumStageDir,
      "--min-stages",
      "1",
      "--require-stage-names",
      "image-to-text",
    ],
    "modality order",
  );
  out.push({ name: "bad-curriculum-modality-order", ok: true });

  const missingCurriculumModalityPath = path.join(root, "negative-missing-curriculum-modality.json");
  const missingCurriculumModality = readJson(curriculumStagesPath);
  delete missingCurriculumModality.stages[0].task_modality_integrity;
  writeJson(missingCurriculumModalityPath, missingCurriculumModality);
  assertCommandFailure(
    "missing curriculum modality should fail quality report",
    qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath: missingCurriculumModalityPath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }),
    "task_modality_integrity",
  );
  out.push({ name: "missing-curriculum-modality-integrity", ok: true });

  const staleGroundedCorpusPath = path.join(root, "negative-stale-grounded-corpus.json");
  const staleGroundedCorpus = readJson(groundedCorpusPath);
  staleGroundedCorpus.examples_hash = "0x0000000000000000";
  writeJson(staleGroundedCorpusPath, staleGroundedCorpus);
  assertCommandFailure(
    "stale grounded corpus should fail quality report",
    qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath: staleGroundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }),
    "examples_hash_match",
  );
  out.push({ name: "stale-grounded-corpus", ok: true });

  const staleIdentityInferencePath = path.join(root, "negative-stale-identity-inference.json");
  const staleIdentityInference = readJson(identityInferencePath);
  staleIdentityInference.model_hash = "0x0000000000000000";
  writeJson(staleIdentityInferencePath, staleIdentityInference);
  assertCommandFailure(
    "stale identity inference should fail quality report",
    qualityReportArgs({
      evalPath: syntheticEvalPath,
      retrievalHeadPath,
      retrievalEvalPath,
      sampleBindingPath,
      generationIntegrityPath,
      identityInferencePath: staleIdentityInferencePath,
      examplesPath,
      manifestPath,
      tokensPath,
      curriculumStagesPath,
      requireCurriculumStageNames: "identity,image-to-text",
      groundedCorpusPath,
      requireConfidenceTrace: true,
      requireArchitectureProfile: true,
      requirePromotedSmallProfile: true,
    }),
    '"model_hash": "0x0000000000000000"',
  );
  out.push({ name: "stale-identity-inference", ok: true });

  const staleEvalHashPath = path.join(root, "negative-stale-retrieval-eval-hash.json");
  const staleEvalHash = readJson(retrievalEvalPath);
  staleEvalHash.examples_hash = "0x0000000000000000";
  writeJson(staleEvalHashPath, staleEvalHash);
  assertCommandFailure(
    "stale retrieval eval hash should fail retrieval head provenance",
    retrievalHeadProvenanceArgs({
      evalPath: staleEvalHashPath,
      retrievalHeadPath,
      examplesPath,
      tokensPath,
    }),
    "examples_hash",
  );
  out.push({ name: "stale-retrieval-eval-hash", ok: true });

  const tamperedHeadPath = path.join(root, "negative-tampered-retrieval-head.json");
  const tamperedHead = readJson(retrievalHeadPath);
  tamperedHead.labels[0].primary_name = `${tamperedHead.labels[0].primary_name} stale`;
  writeJson(tamperedHeadPath, tamperedHead);
  assertCommandFailure(
    "tampered retrieval head should fail retrieval head provenance",
    retrievalHeadProvenanceArgs({
      evalPath: retrievalEvalPath,
      retrievalHeadPath: tamperedHeadPath,
      examplesPath,
      tokensPath,
    }),
    "model_hash",
  );
  out.push({ name: "tampered-retrieval-head", ok: true });

  return out;
}

function corruptCurriculumStageModality(root, stageDir) {
  const badStageDir = path.join(root, "negative-v2-stage-0-image-to-text");
  fs.cpSync(stageDir, badStageDir, { recursive: true });
  const manifestPath = path.join(badStageDir, "manifest.json");
  const manifest = readJson(manifestPath);
  const examplesPath = path.join(badStageDir, manifest.examples_jsonl || "examples.jsonl");
  const tokensPath = path.join(badStageDir, manifest.corpus_tokens_u8 || "corpus.tokens.u8");
  const trainPath = path.join(badStageDir, "train.json");
  const examples = readJsonl(examplesPath);
  const tokens = Buffer.from(fs.readFileSync(tokensPath));
  const rowIndex = examples.findIndex((row) => row.task === "image-to-text");
  if (rowIndex < 0) {
    throw new Error("could not find image-to-text row for curriculum modality negative test");
  }
  const row = examples[rowIndex];
  const offset = Number(row.token_offset);
  const count = Number(row.token_count);
  const slice = tokens.subarray(offset, offset + count);
  const layout = { text: 3, prompt: 2, eos: 5, ...(manifest.token_layout || {}) };
  const textIndex = slice.indexOf(Number(layout.text));
  const eosIndex = slice.indexOf(Number(layout.eos), 1);
  if (textIndex < 0 || eosIndex <= textIndex + 1) {
    throw new Error("image-to-text row has no text payload for curriculum modality negative test");
  }
  tokens[offset + textIndex + 1] = Number(layout.prompt);
  row.token_hash = fnv64Bytes(tokens.subarray(offset, offset + count));
  fs.writeFileSync(tokensPath, tokens);
  writeJsonl(examplesPath, examples.map(stripLineNumber));
  manifest.token_hash = fnv64Bytes(tokens);
  writeJson(manifestPath, manifest);
  const train = readJson(trainPath);
  train.token_hash = manifest.token_hash;
  writeJson(trainPath, train);
  return badStageDir;
}

function corruptCurriculumStageTrainCoverage(root, stageDir) {
  const badStageDir = path.join(root, "negative-v2-stage-missing-train-task-coverage");
  fs.cpSync(stageDir, badStageDir, { recursive: true });
  const trainPath = path.join(badStageDir, "train.json");
  const train = readJson(trainPath);
  delete train.tasks;
  writeJson(trainPath, train);
  return badStageDir;
}

function corruptCurriculumStageTrainExamplesProvenance(root, stageDir) {
  const badStageDir = path.join(root, "negative-v2-stage-bad-train-examples-provenance");
  fs.cpSync(stageDir, badStageDir, { recursive: true });
  const trainPath = path.join(badStageDir, "train.json");
  const train = readJson(trainPath);
  train.corpus_examples_hash = "0x0000000000000000";
  writeJson(trainPath, train);
  return badStageDir;
}

function stripLineNumber(row) {
  const out = { ...row };
  delete out.__line;
  return out;
}

function retrievalHeadProvenanceArgs({ evalPath, retrievalHeadPath, examplesPath, tokensPath }) {
  return [
    "scripts/check-solomon-v2-retrieval-head-provenance.mjs",
    "--eval",
    evalPath,
    "--retrieval-head",
    retrievalHeadPath,
    "--examples",
    examplesPath,
    "--tokens",
    tokensPath,
    "--prompts",
    "none",
    "--expect-spirits",
    "72",
    "--min-feature-count",
    "4096",
    "--min-retrieval-margin",
    "0",
  ];
}

function taskEvalArgs({ evalPath, examplesPath, manifestPath, tokensPath }) {
  return [
    "scripts/check-solomon-attention-task-eval.mjs",
    "--eval",
    evalPath,
    "--examples",
    examplesPath,
    "--manifest",
    manifestPath,
    "--tokens",
    tokensPath,
    "--expect-spirits",
    "72",
    "--require-corpus-version",
    "v2",
    "--require-image-token-profile",
    "symbolic16",
    "--require-image-token-channels",
    requiredChannels.join(","),
    "--require-image-channel-token-stats",
    "--min-image-channel-distinct-bins",
    "2",
    "--min-task-targets",
    "all=1",
    "--require-directional-groups",
  ];
}

function qualityReportArgs({
  evalPath,
  retrievalHeadPath,
  retrievalEvalPath,
  sampleBindingPath,
  generationIntegrityPath,
  identityInferencePath = "",
  examplesPath,
  manifestPath,
  tokensPath,
  curriculumStagesPath = "",
  requireCurriculumStageNames = "",
  groundedCorpusPath = "",
  requireConfidenceTrace = false,
  requireArchitectureProfile = false,
  requirePromotedSmallProfile = false,
}) {
  const args = [
    "scripts/check-solomon-v2-quality-report.mjs",
    "--eval",
    evalPath,
    "--retrieval-head",
    retrievalHeadPath,
    "--retrieval-head-eval",
    retrievalEvalPath,
    "--sample-binding",
    sampleBindingPath,
    "--generation-integrity",
    generationIntegrityPath,
    "--examples",
    examplesPath,
    "--manifest",
    manifestPath,
    "--tokens",
    tokensPath,
    "--require-corpus-version",
    "v2",
    "--require-image-token-profile",
    "symbolic16",
    "--require-image-token-channels",
    requiredChannels.join(","),
    "--require-image-channel-token-stats",
    "--min-image-channel-distinct-bins",
    "2",
    "--min-task-targets",
    "all=1",
    "--min-retrieval-margin",
    "0",
  ];
  if (identityInferencePath) {
    args.push("--identity-inference", identityInferencePath);
    args.push("--require-identity-inference");
  }
  if (curriculumStagesPath) {
    args.push("--curriculum-stages", curriculumStagesPath);
    args.push("--require-curriculum-stages");
    if (requireCurriculumStageNames) {
      args.push("--require-curriculum-stage-names", requireCurriculumStageNames);
    }
  }
  if (groundedCorpusPath) {
    args.push("--grounded-corpus", groundedCorpusPath);
    args.push("--require-grounded-corpus");
    args.push("--min-grounded-source-overlap-tokens", "2");
    args.push("--min-grounded-attribute-source-overlap-tokens", "8");
    args.push("--max-grounded-source-placeholder-rows", "0");
    args.push("--max-grounded-attribute-generic-rank-rows", "0");
  }
  if (requireConfidenceTrace) {
    args.push("--require-confidence-trace");
  }
  if (requireArchitectureProfile) {
    args.push("--require-architecture-profile");
  }
  if (requirePromotedSmallProfile) {
    args.push("--require-promoted-small-profile");
  }
  return args;
}

function runNode(label, args) {
  const result = spawnSync(process.execPath, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(
      `${label} failed with status ${result.status}\nstdout:\n${result.stdout || ""}\nstderr:\n${result.stderr || ""}`,
    );
  }
  return result.stdout;
}

function assertCommandFailure(label, args, expectedText) {
  const result = spawnSync(process.execPath, args, {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status === 0) {
    throw new Error(`${label}: command unexpectedly passed`);
  }
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  if (!output.includes(expectedText)) {
    throw new Error(`${label}: expected output to include ${JSON.stringify(expectedText)}, got:\n${output}`);
  }
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, "utf8")
    .trimEnd()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => ({ ...JSON.parse(line), __line: index + 1 }));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
}

function countBy(values, keyFn) {
  const out = {};
  for (const value of values) {
    const key = keyFn(value);
    out[key] = (out[key] || 0) + 1;
  }
  return out;
}

function fnv64Bytes(bytes) {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (const byte of bytes) {
    hash ^= BigInt(Number(byte) & 0xff);
    hash = (hash * prime) & mask;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function escapeTsv(value) {
  return String(value).replace(/\t/g, " ").replace(/\r?\n/g, " ");
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

function assertArrayEqual(actual, expected, label) {
  if (!Array.isArray(actual) || actual.length !== expected.length) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
    }
  }
}

function assertAtLeast(actual, expected, label) {
  if (Number(actual) < expected) {
    throw new Error(`${label}: expected at least ${expected}, got ${actual}`);
  }
}

function assertMatches(actual, pattern, label) {
  if (!pattern.test(String(actual || ""))) {
    throw new Error(`${label}: expected ${pattern}, got ${JSON.stringify(actual || "")}`);
  }
}

function assertThrows(fn, expectedText, label) {
  try {
    fn();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes(expectedText)) {
      throw new Error(`${label}: expected ${JSON.stringify(expectedText)}, got ${JSON.stringify(message)}`);
    }
    return;
  }
  throw new Error(`${label}: expected failure`);
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error && error.stack ? error.stack : String(error));
  process.exit(1);
}
