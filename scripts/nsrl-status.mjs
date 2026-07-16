#!/usr/bin/env node

import childProcess from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const schema = "nsrl.project_status.v1";
const defaultFastDiagnosticPath = "/tmp/nsrl-solomon-product-diagnostic-fast.json";

const evidenceNames = new Set([
  "nsrl-mme-v0.json",
  "quality-report.json",
  "objective-coverage.json",
  "release-proof.json",
  "pipeline-complete.json",
]);

const headlineEvalContract = {
  schema: "nsrl.multimodal_llm_eval.v0",
  id: "nsrl-mme-v0",
  label: "NSRL-MME v0 multimodal LLM eval",
  headline_metric:
    "minimum per-mille score across model-native multimodal task families",
  target_score_per_mille: 700,
  minimum_rows_per_family: 72,
  policy:
    "Sampler, replay, browser-probe, and memory-assisted sample metrics are diagnostics only; they do not define the headline score.",
};

const headlineDirectionalFamilies = [
  {
    key: "text_prompt_to_image_plan",
    label: "text prompt -> symbolic image plan",
  },
  {
    key: "seal_image_to_text",
    label: "seal image -> identity / attributes / source text",
  },
  {
    key: "text_and_seal_to_explanation",
    label: "text + seal -> grounded explanation / match",
  },
  {
    key: "identity_source_binding",
    label: "prompt/name -> identity / source binding",
  },
];

const knownArtifacts = [
  {
    id: "denoiser",
    kind: "NSRLTCH text-conditioned denoiser",
    model: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch",
    trace: "data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/trace.json",
  },
  {
    id: "latent-prior",
    kind: "NSRLLAT1 prompt/layout prior",
    model: "data/processed/key-solomon-goetia-latent-v1/model.nsrllat",
    trace: "data/processed/key-solomon-goetia-latent-v1/trace.json",
  },
  {
    id: "attention-smoke",
    kind: "NSRLLMM1 attention smoke",
    model: "data/processed/key-solomon-goetia-attention-v1/model.nsrllmm",
    manifest: "data/processed/key-solomon-goetia-attention-v1/manifest.json",
    eval: "data/processed/key-solomon-goetia-attention-v1/attention-eval.json",
    sample: "data/processed/key-solomon-goetia-attention-v1/model-only-bael-current/text.txt",
  },
  {
    id: "attention-curriculum-smoke",
    kind: "NSRLLMM1 attention curriculum smoke",
    model: "data/processed/key-solomon-goetia-attention-curriculum-v1/model.nsrllmm",
    manifest: "data/processed/key-solomon-goetia-attention-curriculum-v1/manifest.json",
    eval: "data/processed/key-solomon-goetia-attention-curriculum-v1/attention-eval.json",
    rawSample: "data/processed/key-solomon-goetia-attention-curriculum-v1/raw-sample-bael/text.txt",
    promptedSample: "data/processed/key-solomon-goetia-attention-curriculum-v1/prior-sample-bael/text.txt",
  },
  {
    id: "multimodal-replay",
    kind: "NSRLMOD1 discrete multimodal replay",
    model: "data/processed/key-solomon-goetia-multimodal-v1/model.nsrlmod",
    manifest: "data/processed/key-solomon-goetia-multimodal-v1/manifest.json",
    sample: "data/processed/key-solomon-goetia-multimodal-v1/sample-bael/text.txt",
  },
  {
    id: "web-attention",
    kind: "deployed web NSRLLMM1 asset",
    model: "web/assets/solomon-attention.nsrllmm",
  },
  {
    id: "web-denoiser",
    kind: "deployed web NSRLTCH asset",
    model: "web/assets/solomon-model.nsrltch",
  },
  {
    id: "web-multimodal",
    kind: "deployed web NSRLMOD1 asset",
    model: "web/assets/solomon-multimodal.nsrlmod",
  },
];

function usage() {
  console.log(
    [
      "Usage: node scripts/nsrl-status.mjs [options]",
      "",
      "Prints the project truth surface: code state, known model artifacts,",
      "Solomon proof evidence, blockers, and next commands.",
      "",
      "Options:",
      "  --json                         emit JSON instead of Markdown",
      "  --out PATH                      write the report to PATH",
      "  --diagnostic PATH               include an existing product diagnostic JSON",
      "  --refresh-fast-diagnostic       run the fast product diagnostic first",
      "  --fast-diagnostic-out PATH      output path for --refresh-fast-diagnostic",
      "  --run-hygiene                   run fmt/no-floats/diff whitespace checks",
      "  --strict                        exit nonzero when release readiness is false",
    ].join("\n"),
  );
}

function parseArgs(argv) {
  const config = {
    json: false,
    outPath: "",
    diagnosticPath: "",
    refreshFastDiagnostic: false,
    fastDiagnosticOut: defaultFastDiagnosticPath,
    runHygiene: false,
    strict: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--json") {
      config.json = true;
    } else if (arg === "--out") {
      config.outPath = requireValue(argv, ++index, arg);
    } else if (arg === "--diagnostic") {
      config.diagnosticPath = requireValue(argv, ++index, arg);
    } else if (arg === "--refresh-fast-diagnostic") {
      config.refreshFastDiagnostic = true;
    } else if (arg === "--fast-diagnostic-out") {
      config.fastDiagnosticOut = requireValue(argv, ++index, arg);
    } else if (arg === "--run-hygiene") {
      config.runHygiene = true;
    } else if (arg === "--strict") {
      config.strict = true;
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

function runCommand(command, args, options = {}) {
  const started = Date.now();
  const result = childProcess.spawnSync(command, args, {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: options.maxBuffer || 1024 * 1024 * 16,
    timeout: options.timeoutMs || 30000,
  });
  return {
    command: [command, ...args].join(" "),
    ok: result.status === 0,
    status: result.status,
    signal: result.signal || "",
    duration_ms: Date.now() - started,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
    error: result.error ? String(result.error.message || result.error) : "",
  };
}

function readJson(relativeOrAbsolutePath) {
  const filePath = resolvePath(relativeOrAbsolutePath);
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readText(relativeOrAbsolutePath) {
  const filePath = resolvePath(relativeOrAbsolutePath);
  return fs.readFileSync(filePath, "utf8");
}

function resolvePath(relativeOrAbsolutePath) {
  return path.isAbsolute(relativeOrAbsolutePath)
    ? relativeOrAbsolutePath
    : path.join(repoRoot, relativeOrAbsolutePath);
}

function maybeReadJson(relativePath) {
  try {
    return readJson(relativePath);
  } catch {
    return null;
  }
}

function maybeReadText(relativePath) {
  try {
    return readText(relativePath);
  } catch {
    return "";
  }
}

function fileInfo(relativePath) {
  if (!relativePath) {
    return { path: "", present: false };
  }
  const filePath = resolvePath(relativePath);
  try {
    const stat = fs.statSync(filePath);
    return {
      path: relativePath,
      present: true,
      bytes: stat.size,
      size: humanBytes(stat.size),
      modified_at: stat.mtime.toISOString(),
    };
  } catch {
    return { path: relativePath, present: false };
  }
}

function humanBytes(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${bytes} B` : `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
}

function shortText(text, max = 120) {
  const oneLine = String(text || "").replace(/\s+/g, " ").trim();
  if (oneLine.length <= max) {
    return oneLine;
  }
  return `${oneLine.slice(0, max - 1)}...`;
}

function collectGit() {
  const status = runCommand("git", ["status", "--short", "--branch"], { timeoutMs: 10000 });
  const currentBranch = runCommand("git", ["branch", "--show-current"], { timeoutMs: 10000 });
  const head = runCommand("git", ["log", "--oneline", "-1"], { timeoutMs: 10000 });
  const lines = status.stdout.trimEnd().split("\n").filter(Boolean);
  const branch = currentBranch.stdout.trim()
    || (lines[0] || "").replace(/^##\s+/, "").split("...")[0];
  const changes = lines.slice(1);
  return {
    branch,
    head: head.stdout.trim(),
    ok: status.ok && currentBranch.ok && head.ok,
    dirty: changes.length > 0,
    change_count: changes.length,
    tracked_change_count: changes.filter((line) => !line.startsWith("??")).length,
    untracked_count: changes.filter((line) => line.startsWith("??")).length,
    sample_changes: changes.slice(0, 12),
    status_error: status.ok ? "" : status.stderr || status.error,
  };
}

function sha256File(relativePath) {
  try {
    return crypto.createHash("sha256").update(fs.readFileSync(resolvePath(relativePath))).digest("hex");
  } catch {
    return "";
  }
}

function fnv64Bytes(bytes) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash = ((hash ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return `0x${hash.toString(16).padStart(16, "0")}`;
}

function fnv64File(relativePath) {
  try {
    return fnv64Bytes(fs.readFileSync(resolvePath(relativePath)));
  } catch {
    return "";
  }
}

function parseTsv(text) {
  const lines = String(text || "").trimEnd().split("\n").filter(Boolean);
  if (lines.length < 2) return [];
  const header = lines[0].split("\t");
  return lines.slice(1).map((line) => Object.fromEntries(
    header.map((field, index) => [field, line.split("\t")[index] ?? ""]),
  ));
}

function collectNativeModelEvidence() {
  const paths = {
    candidate: "benchmarks/integer-transformer-proof-v1/successor-v2-candidate.nsrlmt",
    evidence: "benchmarks/integer-transformer-proof-v1/successor-v2-evidence.json",
    manifest: "benchmarks/integer-transformer-proof-v1/successor-v2-manifest.tsv",
    matrix: "benchmarks/integer-transformer-proof-v1/successor-v2-matrix.tsv",
    training: "benchmarks/integer-transformer-proof-v1/successor-v2-training.json",
  };
  const files = Object.fromEntries(Object.entries(paths).map(([key, value]) => [key, fileInfo(value)]));
  const evidence = maybeReadJson(paths.evidence);
  const training = maybeReadJson(paths.training);
  const manifestRows = parseTsv(maybeReadText(paths.manifest));
  const matrixRows = parseTsv(maybeReadText(paths.matrix));
  const manifest = manifestRows.length === 1 ? manifestRows[0] : null;
  const candidate = matrixRows.find((row) => row.system === "transformer-only") || null;
  const baselines = matrixRows.filter((row) => row.system !== "transformer-only");
  const evidenceSystems = new Map(
    (evidence?.systems || []).map((system) => [system.system, system]),
  );
  const allFilesPresent = Object.values(files).every((file) => file.present);
  const trainingReceiptOk = Boolean(manifest && training
    && training.schema === "nsrl.integer_transformer_successor_train.v1"
    && training.objective?.id === "integer_base2_softmax_nll_millibits"
    && training.objective?.partition === "train"
    && training.objective?.context === Number(manifest.context)
    && training.objective?.zero_probability_floor_millibits === 32000
    && training.method?.name === "deterministic_constrained_coordinate_descent"
    && training.metrics?.zero_probability_classes === 0
    && training.metrics?.final_nll_millibits < training.metrics?.uniform_nll_millibits
    && training.bindings?.model_hash === manifest.candidate_model_hash
    && training.bindings?.artifact_fnv64 === manifest.candidate_artifact_hash
    && evidence?.training?.trace_hash === fnv64File(paths.training)
    && evidence?.training?.objective === training.objective.id
    && evidence?.training?.partition === training.objective.partition
    && evidence?.training?.heldout_targets_read === false);
  const artifactIdentityOk = Boolean(manifest && allFilesPresent
    && manifest.candidate_artifact_hash === fnv64File(paths.candidate)
    && manifest.matrix_hash === fnv64File(paths.matrix)
    && manifest.evidence_hash === fnv64File(paths.evidence)
    && evidence?.bindings?.candidate_artifact_hash === manifest.candidate_artifact_hash
    && trainingReceiptOk);
  const replayFields = [
    ["transformer-only", "transformer_replay_hash"],
    ["uniform", "uniform_replay_hash"],
    ["retrieval", "retrieval_replay_hash"],
    ["byte-ngram", "byte_ngram_replay_hash"],
    ["float-transformer", "float_transformer_replay_hash"],
  ];
  const bindingsOk = Boolean(manifest && candidate && matrixRows.length === 5
    && evidence?.schema === "nsrl.integer_transformer_successor_evidence.v2"
    && evidence.contract === manifest.contract
    && evidence.bindings?.dataset_hash === manifest.dataset_hash
    && String(evidence.bindings?.targets) === manifest.targets
    && evidence.bindings?.candidate_model_hash === manifest.candidate_model_hash
    && matrixRows.every((row) => row.contract === manifest.contract
      && row.dataset_hash === manifest.dataset_hash
      && row.tokenizer_hash === manifest.tokenizer_hash
      && row.candidate_model_hash === manifest.candidate_model_hash
      && row.evaluator_hash === manifest.evaluator_hash
      && row.runner_hash === manifest.runner_hash
      && row.targets === manifest.targets)
    && matrixRows.every((row) => {
      const system = evidenceSystems.get(row.system);
      return system
        && String(system.total_nll_millibits) === row.total_nll_millibits
        && String(system.zero_probability_windows) === row.zero_probability_windows
        && system.replay_hash === row.replay_hash;
    })
    && replayFields.every(([system, field]) => matrixRows.find((row) => row.system === system)?.replay_hash
      === manifest[field]));
  const assistanceAbsent = Boolean(candidate
    && candidate.suffix_memory === "false"
    && candidate.retrieval_assistance === "false"
    && candidate.routing_oracle === "false"
    && evidence?.candidate_assistance?.suffix_memory_present === false
    && evidence?.candidate_assistance?.retrieval_assistance_present === false
    && evidence?.candidate_assistance?.routing_oracle_present === false
    && evidence?.candidate_assistance?.position_storage_all_zero === true
    && training?.assistance?.suffix_memory === false
    && training?.assistance?.retrieval === false
    && training?.assistance?.routing_oracle === false
    && training?.assistance?.heldout_targets_read === false);
  const candidateNll = candidate ? Number(candidate.total_nll_millibits) : null;
  const zeroProbabilityWindows = candidate ? Number(candidate.zero_probability_windows) : null;
  const baselineNll = Object.fromEntries(baselines.map((row) => [row.system,
    Number(row.total_nll_millibits)]));
  const gaps = candidateNll === null ? {} : Object.fromEntries(
    Object.entries(baselineNll).map(([system, nll]) => [system, candidateNll - nll]),
  );
  const zeroProbabilityGate = zeroProbabilityWindows === 0;
  const uniformGate = candidateNll !== null && Number.isFinite(baselineNll.uniform)
    && candidateNll < baselineNll.uniform;
  const promotionGate = candidateNll !== null && baselines.length === 4
    && baselines.every((row) => candidateNll < Number(row.total_nll_millibits));
  const integrityOk = artifactIdentityOk && bindingsOk && assistanceAbsent;
  const state = !allFilesPresent ? "missing"
    : !integrityOk ? "invalid"
      : !zeroProbabilityGate ? "falsified_zero_probability"
        : !uniformGate ? "falsified_uniform"
          : !promotionGate ? "falsified_frozen_baselines"
            : "promotion_gate_passed";
  return {
    present: allFilesPresent,
    ok: integrityOk,
    contract: manifest?.contract || "integer-transformer-successor-v2",
    state,
    files,
    dataset_hash: manifest?.dataset_hash || "",
    candidate_model_hash: manifest?.candidate_model_hash || "",
    candidate_artifact_hash: manifest?.candidate_artifact_hash || "",
    candidate_nll_millibits: candidateNll,
    zero_probability_windows: zeroProbabilityWindows,
    target_count: candidate ? Number(candidate.targets) : null,
    baseline_nll_millibits: baselineNll,
    gap_to_baseline_millibits: gaps,
    assistance_absent: assistanceAbsent,
    artifact_identity_ok: artifactIdentityOk,
    training_receipt_ok: trainingReceiptOk,
    replay_bindings_ok: bindingsOk,
    zero_probability_gate: zeroProbabilityGate,
    beats_uniform_gate: uniformGate,
    beats_all_frozen_baselines_gate: promotionGate,
  };
}

function collectOpenGenerationEvidence() {
  const checkpointPath = "benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json";
  const file = fileInfo(checkpointPath);
  const checkpoint = maybeReadJson(checkpointPath);
  const servingGates = [
    "complete_generation_matrix",
    "incremental_cached_decoding",
    "no_residual_saturation",
    "forbidden_assistance_absent",
  ];
  const qualityGates = [
    "repeat_4gram_health",
    "unique_4gram_health",
    "entropy_health",
    "utf8_validity",
    "context_use",
    "distractor_resistance",
  ];
  const bindingsOk = Boolean(checkpoint
    && checkpoint.schema === "nsrl.open_generation_development_checkpoint.v1"
    && checkpoint.contract === "open-generation-v1"
    && /^0x[0-9a-f]{16}$/.test(checkpoint.candidate?.model_fnv64 || "")
    && /^0x[0-9a-f]{16}$/.test(checkpoint.candidate?.tokenizer_fnv64 || ""));
  const servingOk = Boolean(bindingsOk
    && checkpoint.execution?.decoder === "incremental_linear_attention_cache_v1"
    && checkpoint.execution.counts?.samples === 60
    && checkpoint.execution.counts?.generated_tokens === 30_720
    && checkpoint.execution.residual_saturation_count === 0
    && servingGates.every((gate) => checkpoint.generation?.gates?.[gate] === true));
  const qualityOk = Boolean(servingOk
    && qualityGates.every((gate) => checkpoint.generation?.gates?.[gate] === true)
    && checkpoint.generation?.gates?.development_generation_passed === true);
  const promotionGatePassed = Boolean(qualityOk && checkpoint.promotion_passed === true);
  const ok = bindingsOk && servingOk;
  const state = !file.present ? "missing"
    : !ok ? "invalid"
      : promotionGatePassed ? "promotion_gate_passed"
        : "development_failed";
  return {
    present: file.present,
    ok,
    state,
    file,
    candidate: checkpoint?.candidate || {},
    modeling: checkpoint?.modeling || null,
    generation_metrics: checkpoint?.generation?.metrics || {},
    generation_gates: checkpoint?.generation?.gates || {},
    cache: checkpoint?.execution?.cache || {},
    serving_ok: servingOk,
    quality_ok: qualityOk,
    promotion_gate_passed: promotionGatePassed,
    missing_evidence: checkpoint?.missing_evidence || [],
  };
}

function collectCouncilEvidence() {
  const paths = {
    trust_root: "council/trust-root-v0.json",
    request_schema: "protocol/solomon-council-request-v0.schema.json",
    receipt_schema: "protocol/wisdom-receipt-v0.schema.json",
    observation_schema: "protocol/wisdom-outcome-observation-v0.schema.json",
    wisdom_eval_schema: "protocol/solomon-wisdom-eval-v0.schema.json",
    wisdom_casebook_draft_schema: "protocol/solomon-wisdom-casebook-draft-v0.schema.json",
    wisdom_casebook_schema: "protocol/solomon-wisdom-casebook-v0.schema.json",
    wisdom_gold_vault_schema: "protocol/solomon-wisdom-gold-vault-v0.schema.json",
    wisdom_lane_bundle_schema: "protocol/solomon-wisdom-lane-bundle-v0.schema.json",
    wisdom_lane_trace_schema: "protocol/solomon-wisdom-lane-trace-v0.schema.json",
    wisdom_model_input_schema: "protocol/solomon-wisdom-model-input-v0.schema.json",
    wisdom_gold_opening_schema: "protocol/solomon-wisdom-gold-opening-v0.schema.json",
    generation_integrity_schema: "protocol/wisdom-generation-integrity-v0.schema.json",
    provenance_gate_schema: "protocol/wisdom-provenance-gate-v0.schema.json",
    request: "benchmarks/solomon-council-v0/fixtures/select-request.json",
    receipt: "benchmarks/solomon-council-v0/fixtures/select-receipt.json",
    observation: "benchmarks/solomon-council-v0/fixtures/select-observation.json",
    revised_receipt: "benchmarks/solomon-council-v0/fixtures/select-revised-receipt.json",
    wisdom_casebook: "benchmarks/solomon-council-v0/production-v0/casebook.json",
    wisdom_solo_bundle: "benchmarks/solomon-council-v0/production-v0/solo-bundle.json",
    wisdom_council_bundle: "benchmarks/solomon-council-v0/production-v0/council-bundle.json",
    wisdom_gold_opening: "benchmarks/solomon-council-v0/production-v0/gold-opening.json",
    wisdom_generation_integrity:
      "benchmarks/solomon-council-v0/production-v0/generation-integrity.json",
    wisdom_provenance: "benchmarks/solomon-council-v0/production-v0/provenance.json",
    wisdom_eval_input: "benchmarks/solomon-council-v0/production-v0/eval-input.json",
    adaptive_preregistration:
      "protocol/examples/p10m-adaptive-composition-v1-preregistration.json",
    adaptive_theory:
      "research/mathematical-journal/MJ-2026-07-15-20-exchangeable-adaptive-composition.md",
    adaptive_contract:
      "benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json",
    adaptive_result:
      "benchmarks/production-model-v1/p10m-adaptive-composition-v1-result.json",
    adaptive_replay_receipt:
      "benchmarks/production-model-v1/p10m-adaptive-composition-v1-replay-receipt.json",
    adaptive_publication:
      "benchmarks/production-model-v1/p10m-adaptive-composition-v1-publication.json",
    council_hardening_contract:
      "benchmarks/solomon-council-v1/hardening-contract.json",
    council_hardening_result:
      "benchmarks/solomon-council-v1/hardening-result.json",
    wisdom_eval_result: "benchmarks/solomon-council-v0/wisdom-eval-result.json",
    decision_regret_result:
      "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-result.json",
    decision_regret_publication:
      "benchmarks/production-model-v1/p10m-solomonic-judgment-v1-publication.json",
  };
  const sealPaths = [
    "mathematician", "engineer", "historian", "skeptic", "consequence_planner", "judge",
  ].map((faculty) => `council/seals/${faculty}.json`);
  const requiredPaths = Object.entries(paths).filter(
    ([key]) => ![
      "wisdom_casebook", "wisdom_solo_bundle", "wisdom_council_bundle",
      "wisdom_gold_opening", "wisdom_generation_integrity", "wisdom_provenance",
      "wisdom_eval_input", "wisdom_eval_result", "decision_regret_result",
      "decision_regret_publication", "adaptive_contract", "adaptive_result",
      "adaptive_replay_receipt", "adaptive_publication", "council_hardening_contract",
      "council_hardening_result",
    ].includes(key));
  const files = Object.fromEntries(requiredPaths.map(([key, value]) => [key, fileInfo(value)]));
  const seals = sealPaths.map(fileInfo);
  const receipt = maybeReadJson(paths.receipt);
  const revisedReceipt = maybeReadJson(paths.revised_receipt);
  const wisdomEval = maybeReadJson(paths.wisdom_eval_result);
  const regretResult = maybeReadJson(paths.decision_regret_result);
  const regretPublication = maybeReadJson(paths.decision_regret_publication);
  const adaptiveContract = maybeReadJson(paths.adaptive_contract);
  const adaptiveResult = maybeReadJson(paths.adaptive_result);
  const adaptiveReplayReceipt = maybeReadJson(paths.adaptive_replay_receipt);
  const adaptivePublication = maybeReadJson(paths.adaptive_publication);
  const councilHardeningContract = maybeReadJson(paths.council_hardening_contract);
  const councilHardeningResult = maybeReadJson(paths.council_hardening_result);
  const selfCheck = runCommand(process.execPath, ["scripts/check-solomon-council-v0.mjs"], {
    timeoutMs: 10000,
  });
  const ceremonyCheck = runCommand(
    process.execPath, ["scripts/check-solomon-wisdom-ceremony-v0.mjs"], {timeoutMs: 10000});
  const productionWisdomCheck = runCommand(
    process.execPath, ["scripts/check-solomon-wisdom-production-v0.mjs"], {timeoutMs: 20000});
  const adaptiveCheck = runCommand(
    process.execPath, ["scripts/check-adaptive-composition-theory-v1.mjs"], {timeoutMs: 10000});
  const adaptivePublicationCheck = runCommand(
    process.execPath, ["scripts/check-adaptive-composition-publication-v1.mjs"], {
      timeoutMs: 10000,
    });
  const councilHardeningCheck = runCommand(
    process.execPath, ["scripts/check-solomon-council-hardening-v1.mjs"], {
      timeoutMs: 30000,
    });
  const selfCheckOk = selfCheck.ok;
  const ceremonyCheckOk = ceremonyCheck.ok;
  const productionWisdomCheckOk = productionWisdomCheck.ok;
  const receiptOk = receipt?.schema === "nsrl.wisdom_receipt.v0"
    && receipt.mode === "shadow"
    && receipt.faculty_invocations?.length === 6
    && receipt.faculty_invocations.every((entry) => entry.seal?.signature_verified === true)
    && ["select", "request_evidence", "ask_user", "abstain"].includes(receipt.decision?.kind)
    && receipt.shadow_execution?.action_execution_allowed === false
    && receipt.shadow_execution?.action_executed === false
    && receipt.outcome?.status === "pending"
    && receipt.revisions?.length === 0
    && /^[0-9a-f]{64}$/.test(receipt.identity?.receipt_sha256 ?? "");
  const revisionOk = revisedReceipt?.schema === "nsrl.wisdom_receipt.v0"
    && revisedReceipt.mode === "shadow"
    && revisedReceipt.outcome?.status === "observed"
    && revisedReceipt.revisions?.length === 1
    && revisedReceipt.revisions[0]?.prior_receipt_sha256 === receipt?.identity?.receipt_sha256
    && revisedReceipt.shadow_execution?.action_executed === false;
  const filesPresent = Object.values(files).every((file) => file.present)
    && seals.every((file) => file.present);
  const councilCoreOk = filesPresent && receiptOk && revisionOk && selfCheckOk && ceremonyCheckOk;
  const wisdomGatePassed = wisdomEval?.schema === "nsrl.solomon_wisdom_eval_result.v0"
    && wisdomEval.analysis_role === "frozen_same_model_comparison"
    && productionWisdomCheckOk
    && wisdomEval.verdict?.all_dimensions_outperform === true
    && wisdomEval.verdict?.promotion_gate_passed === true
    && wisdomEval.authorization?.council_promotion_authorized === true
    && wisdomEval.authorization?.product_release_authorized === false;
  const hardeningMeasured = councilHardeningContract?.schema
      === "nsrl.solomon_council_hardening_contract.v1"
    && councilHardeningResult?.schema === "nsrl.solomon_council_hardening_result.v1";
  const hardeningGatePassed = hardeningMeasured
    && councilHardeningCheck.ok
    && councilHardeningResult.gates?.all_passed === true
    && councilHardeningResult.authorization?.effective_council_promotion_authorized === true;
  const hardeningState = !hardeningMeasured ? "not_measured"
    : !councilHardeningCheck.ok ? "invalid"
      : councilHardeningResult.verdict?.status === "falsified" ? "falsified"
        : hardeningGatePassed ? "passed" : "failed_or_inconclusive";
  const effectiveWisdomPromotion = wisdomGatePassed && hardeningGatePassed;
  const wisdomState = !wisdomEval ? "not_measured"
    : !wisdomGatePassed ? "failed_or_invalid"
      : hardeningState === "not_measured" ? "v0_passed_hardening_not_measured"
        : hardeningState === "falsified" ? "v0_passed_hardening_falsified"
          : hardeningState === "passed" ? "hardening_passed" : "hardening_invalid";
  const wisdomArtifacts = Object.fromEntries([
    "wisdom_casebook", "wisdom_solo_bundle", "wisdom_council_bundle",
    "wisdom_gold_opening", "wisdom_generation_integrity", "wisdom_provenance",
    "wisdom_eval_input", "wisdom_eval_result",
  ].map((key) => [key, fileInfo(paths[key])]));
  const wisdomPipelineStage = !wisdomArtifacts.wisdom_casebook.present
    ? "casebook_not_frozen"
    : !wisdomArtifacts.wisdom_solo_bundle.present
      ? "awaiting_solo_bundle"
      : !wisdomArtifacts.wisdom_council_bundle.present
        ? "awaiting_council_bundle"
        : !wisdomArtifacts.wisdom_gold_opening.present
          ? "awaiting_gold_opening"
          : !(wisdomArtifacts.wisdom_generation_integrity.present
              && wisdomArtifacts.wisdom_provenance.present)
            ? "awaiting_integrity_reports"
            : !wisdomArtifacts.wisdom_eval_input.present
              ? "awaiting_compilation"
              : !wisdomArtifacts.wisdom_eval_result.present
                ? "awaiting_evaluation"
                : "measured";
  const regretEvidenceOk = regretResult?.schema === "nsrl.solomonic_judgment_result.v1"
    && regretPublication?.schema === "nsrl.solomonic_judgment_publication.v1"
    && regretPublication.verdict?.status === "supported"
    && regretResult.heldout_regret?.fired_passages > 0
    && BigInt(regretResult.heldout_regret?.signed_regret_q32 ?? "0") < 0n;
  const adaptiveArtifactsPresent = Boolean(
    adaptiveContract && adaptiveResult && adaptiveReplayReceipt && adaptivePublication);
  const adaptiveEvidence = adaptivePublication?.evidence;
  const adaptiveTrajectory = adaptiveEvidence?.adaptive_trajectory;
  const adaptiveEndpoints = adaptiveEvidence?.endpoints;
  const adaptiveExecutionValid = adaptiveArtifactsPresent
    && adaptiveContract.schema === "nsrl.adaptive_composition_execution_contract.v1"
    && adaptiveResult.schema === "nsrl.adaptive_composition_result.v1"
    && adaptiveReplayReceipt.schema === "nsrl.adaptive_composition_replay_receipt.v1"
    && adaptivePublication.schema === "nsrl.adaptive_composition_publication.v1"
    && adaptivePublicationCheck.ok
    && adaptiveResult.verdict === "falsified"
    && adaptivePublication.verdict?.status === "falsified"
    && adaptivePublication.verdict?.falsified === true
    && adaptiveTrajectory?.accepted_actions === 0
    && adaptiveTrajectory?.head_fires === 0
    && adaptiveTrajectory?.trunk_fires === 0
    && adaptiveEndpoints?.adaptive?.total_nll_millibits === "5930001"
    && adaptiveEndpoints?.always_abstain?.total_nll_millibits === "5930001"
    && adaptiveEvidence?.exact_byte_replay === true
    && adaptiveReplayReceipt.guarantees?.post_outcome_threshold_change === false;
  return {
    present: filesPresent,
    ok: councilCoreOk,
    state: !filesPresent ? "missing"
      : !councilCoreOk ? "invalid"
        : effectiveWisdomPromotion ? "wisdom_hardening_gate_passed"
          : hardeningState === "falsified" ? "hardening_falsified_shadow_only"
            : wisdomGatePassed ? "v0_passed_hardening_pending" : "shadow_ready",
    files,
    seals,
    faculties: receipt?.faculty_invocations?.map((entry) => entry.faculty_id) ?? [],
    seal_signatures_verified: receipt?.faculty_invocations?.filter(
      (entry) => entry.seal?.signature_verified === true).length ?? 0,
    decision_states_checked: ["select", "request_evidence", "ask_user", "abstain"],
    receipt_sha256: receipt?.identity?.receipt_sha256 ?? "",
    revised_receipt_sha256: revisedReceipt?.identity?.receipt_sha256 ?? "",
    self_check_ok: selfCheckOk,
    wisdom_ceremony_check_ok: ceremonyCheckOk,
    wisdom_ceremony_byte_bound: ceremonyCheckOk,
    wisdom_production_check_ok: productionWisdomCheckOk,
    receipt_ok: receiptOk,
    revision_ok: revisionOk,
    shadow_execution_only: receipt?.shadow_execution?.action_execution_allowed === false
      && receipt?.shadow_execution?.action_executed === false,
    wisdom_evaluation: {
      state: wisdomState,
      pipeline_stage: wisdomPipelineStage,
      path: paths.wisdom_eval_result,
      artifacts: wisdomArtifacts,
      dimensions: wisdomEval?.dimensions ?? {},
      all_dimensions_outperform: wisdomEval?.verdict?.all_dimensions_outperform === true,
      v0_promotion_gate_passed: wisdomGatePassed,
      promotion_gate_passed: effectiveWisdomPromotion,
      hardening: {
        state: hardeningState,
        check_ok: councilHardeningCheck.ok,
        contract: fileInfo(paths.council_hardening_contract),
        result: fileInfo(paths.council_hardening_result),
        verdict: councilHardeningResult?.verdict?.status ?? "not_measured",
        actual_solo_tool_observations:
          councilHardeningResult?.baseline_fairness?.actual_solo_tool_observations ?? 0,
        actual_council_tool_observations:
          councilHardeningResult?.baseline_fairness?.actual_council_tool_observations ?? 0,
        tool_parity_dimensions_tied: Object.values(
          councilHardeningResult?.dimensions ?? {}).filter((dimension) => dimension.exact_tie).length,
        gates_passed: Object.entries(councilHardeningResult?.gates ?? {})
          .filter(([key, value]) => key !== "all_passed" && value === true).length,
        gates_total: Object.keys(councilHardeningResult?.gates ?? {})
          .filter((key) => key !== "all_passed").length,
        missing_coverage: councilHardeningResult?.next_required_evidence ?? [],
        remain_shadow_only:
          councilHardeningResult?.authorization?.remain_shadow_only !== false,
      },
    },
    bounded_decision_regret_evidence: {
      present: Boolean(regretResult && regretPublication),
      empirical_result_supported: regretEvidenceOk,
      publication_status: regretPublication?.verdict?.status ?? "missing",
      fired_passages: regretResult?.heldout_regret?.fired_passages ?? 0,
      signed_regret_q32: regretResult?.heldout_regret?.signed_regret_q32 ?? "",
      same_model_solo_comparison: false,
      conditional_null_assumed: true,
      conditional_bridge_falsified_by_mj20: true,
      sequential_safety_supported: false,
    },
    adaptive_composition: {
      theory_check_ok: adaptiveCheck.ok,
      replacement: "finite_horizon_simultaneous_state_action_conformal_plus_alpha_spending",
      artifacts: {
        contract: fileInfo(paths.adaptive_contract),
        result: fileInfo(paths.adaptive_result),
        replay_receipt: fileInfo(paths.adaptive_replay_receipt),
        publication: fileInfo(paths.adaptive_publication),
      },
      execution_state: !adaptiveArtifactsPresent ? "not_measured"
        : adaptiveExecutionValid ? "falsified" : "invalid",
      execution_completed: adaptiveExecutionValid,
      publication_check_ok: adaptivePublicationCheck.ok,
      calibration_sources_per_family: adaptiveContract?.calibration?.sources_per_family ?? 119,
      calibration_source_panels: adaptiveEvidence?.calibration_source_panels ?? 0,
      calibration_cube_rows: adaptiveEvidence?.calibration_cube_rows ?? 0,
      corrections_q32: adaptiveEvidence?.corrections_q32 ?? {},
      adaptive_fires: adaptiveTrajectory?.accepted_actions ?? 0,
      head_fires: adaptiveTrajectory?.head_fires ?? 0,
      trunk_fires: adaptiveTrajectory?.trunk_fires ?? 0,
      endpoint_nll_millibits: adaptiveEndpoints?.adaptive?.total_nll_millibits ?? "",
      exact_replay: adaptiveEvidence?.exact_byte_replay === true,
      threshold_retuning_authorized:
        adaptivePublication?.interpretation?.threshold_retuning_after_outcome_authorized ?? false,
      optimizer_promotion_authorized: false,
      next_optimizer_experiment:
        adaptivePublication?.interpretation?.next_admissible_optimizer_experiment ?? "",
      next_product_experiment:
        adaptivePublication?.interpretation?.next_product_facing_experiment ?? "",
    },
  };
}

function collectIsingEvidence() {
  const paths = {
    audit_contract: "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json",
    audit_result: "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1.json",
    document_ising: "benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json",
    confirmation_contract:
      "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json",
    confirmation_source:
      "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json",
    confirmation_result:
      "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json",
    cross_source_contract:
      "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json",
    cross_source_result:
      "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json",
    cross_source_publication:
      "benchmarks/production-model-v1/p10m-cross-source-exchange-v1-publication.json",
    multifamily_contract:
      "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json",
    multifamily_result:
      "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json",
  };
  const auditContract = maybeReadJson(paths.audit_contract);
  const auditResult = maybeReadJson(paths.audit_result);
  const documentIsing = maybeReadJson(paths.document_ising);
  const confirmationContract = maybeReadJson(paths.confirmation_contract);
  const confirmationSource = maybeReadJson(paths.confirmation_source);
  const confirmationResult = maybeReadJson(paths.confirmation_result);
  const crossSourceContract = maybeReadJson(paths.cross_source_contract);
  const crossSourceResult = maybeReadJson(paths.cross_source_result);
  const crossSourcePublication = maybeReadJson(paths.cross_source_publication);
  const multifamilyContract = maybeReadJson(paths.multifamily_contract);
  const multifamilyResult = maybeReadJson(paths.multifamily_result);
  const auditContractSha256 = sha256File(paths.audit_contract);
  const auditResultSha256 = sha256File(paths.audit_result);
  const confirmationContractSha256 = sha256File(paths.confirmation_contract);
  const confirmationSourceSha256 = sha256File(paths.confirmation_source);
  const confirmationResultSha256 = sha256File(paths.confirmation_result);
  const crossSourceContractSha256 = sha256File(paths.cross_source_contract);
  const crossSourceResultSha256 = sha256File(paths.cross_source_result);
  const crossSourcePublicationSha256 = sha256File(paths.cross_source_publication);
  const multifamilyContractSha256 = sha256File(paths.multifamily_contract);
  const multifamilyResultSha256 = sha256File(paths.multifamily_result);
  const endpointSupport = confirmationResult?.mechanism_support || {};
  const endpointValues = Object.values(endpointSupport);
  const confirmationVerdict = endpointValues.length === 0
    ? "inconclusive"
    : endpointValues.every((value) => value === true)
      ? "supported_within_source"
      : "falsified_or_partially_supported";
  const authorizationSafe = auditResult?.decision?.optimizer_change_authorized === false
    && auditResult?.decision?.paid_scaling_authorized === false
    && documentIsing?.decision?.optimizer_change_authorized === false
    && documentIsing?.decision?.paid_scaling_authorized === false
    && confirmationResult?.decision?.optimizer_change_authorized === false
    && confirmationResult?.decision?.paid_scaling_authorized === false
    && crossSourceContract?.authorization?.optimizer_change === false
    && crossSourceContract?.authorization?.paid_scaling === false
    && crossSourceResult?.decision?.optimizer_change_authorized === false
    && crossSourceResult?.decision?.paid_scaling_authorized === false
    && crossSourcePublication?.verdict?.optimizer_change_authorized === false
    && crossSourcePublication?.verdict?.paid_scaling_authorized === false
    && multifamilyContract?.authorization?.optimizer_change === false
    && multifamilyContract?.authorization?.paid_scaling === false
    && multifamilyResult?.decision?.optimizer_change_authorized === false
    && multifamilyResult?.decision?.paid_scaling_authorized === false;
  const auditOk = auditContract?.schema === "nsrl.production_atomic_ising_audit_contract.v1"
    && auditResult?.schema === "nsrl.production_atomic_ising_audit.v1"
    && auditResult.audit_contract_sha256 === auditContractSha256;
  const documentIsingOk = documentIsing?.schema === "nsrl.production_atomic_ising_proposal.v1"
    && documentIsing?.analysis_role === "proposal_only_calibration";
  const confirmationOk = confirmationContract?.schema
      === "nsrl.production_atomic_ising_confirmation_contract.v1"
    && confirmationSource?.schema === "nsrl.production_atomic_structure.v1"
    && confirmationSource?.analysis_role === "untouched_confirmation"
    && confirmationResult?.schema === "nsrl.production_atomic_ising_confirmation.v1"
    && confirmationResult.confirmation_contract_sha256 === confirmationContractSha256
    && confirmationResult.source_result_sha256 === confirmationSourceSha256;
  const crossSourceOk = crossSourceContract?.schema
      === "nsrl.production_cross_source_exchange_contract.v1"
    && crossSourceResult?.schema === "nsrl.production_cross_source_exchange_result.v1"
    && crossSourceResult.source_sha256?.contract === crossSourceContractSha256
    && crossSourceResult.decision?.documents_200_212_read === false;
  const crossSourcePublicationOk = crossSourcePublication?.schema
      === "nsrl.production_cross_source_exchange_publication.v1"
    && crossSourcePublication.source_sha256?.frozen_prospective_contract
      === crossSourceContractSha256
    && crossSourcePublication.source_sha256?.checked_prospective_result
      === crossSourceResultSha256
    && ["supported", "falsified", "inconclusive"].includes(
      crossSourcePublication.verdict?.status)
    && crossSourcePublication.verdict?.vacuous_envelope_status === "inconclusive"
    && crossSourcePublication.sealed_material?.read === false;
  const multifamilyOk = multifamilyContract?.schema
      === "nsrl.production_multifamily_exchange_contract.v1"
    && multifamilyResult?.schema === "nsrl.production_multifamily_exchange_result.v1"
    && multifamilyResult.source_sha256?.contract === multifamilyContractSha256
    && multifamilyResult.decision?.documents_200_212_read === false
    && ["supported_on_frozen_four_family_multipassage_frame",
      "coverage_inconclusive_no_promotion",
      "prospective_multifamily_certificate_falsified_or_vacuous"].includes(
      multifamilyResult.decision?.status);
  return {
    present: Boolean(auditContract && auditResult && documentIsing
      && confirmationContract && confirmationSource && confirmationResult
      && crossSourceContract && crossSourceResult && crossSourcePublication
      && multifamilyContract && multifamilyResult),
    ok: auditOk && documentIsingOk && confirmationOk && crossSourceOk
      && crossSourcePublicationOk && multifamilyOk && authorizationSafe,
    audit: {
      state: auditOk ? "replayable" : "missing_or_invalid",
      contract_path: paths.audit_contract,
      contract_sha256: auditContractSha256,
      result_path: paths.audit_result,
      result_sha256: auditResultSha256,
      sigma_delta_scope: "Gray-ordered high-order Walsh loss residual; not optimizer time dynamics",
    },
    document_ising: {
      state: documentIsingOk ? "proposal_frozen" : "missing_or_invalid",
      result_path: paths.document_ising,
      source_clusters: documentIsing?.source_population?.proposal_source_clusters ?? 1,
    },
    confirmation: {
      state: confirmationOk ? "replayable" : "missing_or_invalid",
      verdict: confirmationVerdict,
      contract_path: paths.confirmation_contract,
      contract_sha256: confirmationContractSha256,
      result_path: paths.confirmation_result,
      result_sha256: confirmationResultSha256,
      endpoints_supported: endpointValues.filter((value) => value === true).length,
      endpoints_total: endpointValues.length,
    },
    cross_source: {
      state: crossSourceOk && crossSourcePublicationOk ? "replayable" : "missing_or_invalid",
      verdict: crossSourcePublication?.verdict?.status ?? "missing",
      contract_path: paths.cross_source_contract,
      contract_sha256: crossSourceContractSha256,
      result_path: paths.cross_source_result,
      result_sha256: crossSourceResultSha256,
      publication_path: paths.cross_source_publication,
      publication_sha256: crossSourcePublicationSha256,
      fitting_source_panels: crossSourceResult?.population?.fitting_source_panels ?? 0,
      calibration_source_panels: crossSourceResult?.population?.calibration_source_panels ?? 0,
      evaluation_source_panels: crossSourceResult?.population?.evaluation_source_panels ?? 0,
      envelope_covered: crossSourceResult?.untouched_evaluation?.envelope_covered ?? 0,
      fired_source_panels: crossSourceResult?.untouched_evaluation?.fired_source_panels ?? 0,
      unsafe_firings: crossSourceResult?.untouched_evaluation?.unsafe_firings ?? 0,
      unsafe_action_rate:
        crossSourcePublication?.untouched_evaluation_metrics?.unsafe_action_rate?.exact ?? "",
      firing_rate:
        crossSourcePublication?.untouched_evaluation_metrics?.firing_rate?.exact ?? "",
      coverage_rate:
        crossSourcePublication?.untouched_evaluation_metrics?.source_panel_coverage?.exact ?? "",
      aggregate_signed_regret_relative_to_abstention_q32:
        crossSourcePublication?.untouched_evaluation_metrics?.regret_relative_to_abstention
          ?.aggregate_signed_q32 ?? "",
      aggregate_positive_regret_relative_to_abstention_q32:
        crossSourcePublication?.untouched_evaluation_metrics?.regret_relative_to_abstention
          ?.aggregate_positive_part_q32 ?? "",
    },
    multifamily_cross_source: {
      state: multifamilyOk ? "replayable" : "missing_or_invalid",
      verdict: multifamilyResult?.decision?.status ?? "missing",
      contract_path: paths.multifamily_contract,
      contract_sha256: multifamilyContractSha256,
      result_path: paths.multifamily_result,
      result_sha256: multifamilyResultSha256,
      families: multifamilyResult?.population?.families ?? [],
      fitting_source_panels: multifamilyResult?.population?.fitting_source_panels ?? 0,
      calibration_source_panels: multifamilyResult?.population?.calibration_source_panels ?? 0,
      evaluation_source_panels: multifamilyResult?.population?.evaluation_source_panels ?? 0,
      passages_per_source_panel:
        multifamilyContract?.panel_sampling?.passage_documents_per_source ?? 0,
      envelope_covered: multifamilyResult?.untouched_evaluation?.envelope_covered ?? 0,
      fired_source_panels:
        multifamilyResult?.untouched_evaluation?.fired_source_panels ?? 0,
      fired_passages: multifamilyResult?.untouched_evaluation?.fired_passages ?? 0,
      firing_families: multifamilyResult?.untouched_evaluation?.firing_families ?? [],
      unsafe_firings: multifamilyResult?.untouched_evaluation?.unsafe_firings ?? 0,
      net_heldout_improvement_q32:
        multifamilyResult?.untouched_evaluation?.net_heldout_improvement_q32 ?? "",
      family_promotions: multifamilyResult?.untouched_evaluation?.family_promotions ?? {},
    },
    boundaries: {
      default_optimizer: "integer_residual_sgd",
      optimizer_change_authorized: authorizationSafe ? false : null,
      paid_scaling_authorized: authorizationSafe ? false : null,
      same_source_document_evidence_only: false,
      cross_source_certificate_scope: crossSourceOk
        ? "frozen_distinct_author_english_gutenberg_frame"
        : "unidentified",
      multifamily_certificate_scope: multifamilyOk
        ? "overall_inconclusive_with_federal_register_and_rfc_family_promotion"
        : "unidentified",
      arbitrary_source_generalization_identified: false,
      sealed_documents: "200--212",
      frozen_structure_cube_reexecuted_in_clean_check: false,
    },
  };
}

function collectResearchHarness() {
  const eventsPath = resolvePath("data/research-harness/events.jsonl");
  if (!fs.existsSync(eventsPath)) {
    return {
      present: false,
      ok: true,
      state_root: "data/research-harness",
      experiment_count: 0,
      event_count: 0,
      experiments: [],
    };
  }
  const command = runCommand(
    process.execPath,
    ["scripts/research-harness.mjs", "status", "--json"],
    { timeoutMs: 30_000, maxBuffer: 1024 * 1024 * 8 },
  );
  if (!command.ok) {
    return {
      present: true,
      ok: false,
      state_root: "data/research-harness",
      error: command.stderr || command.error || "research harness status failed",
      experiments: [],
    };
  }
  try {
    return {
      present: true,
      ...JSON.parse(command.stdout),
    };
  } catch (error) {
    return {
      present: true,
      ok: false,
      state_root: "data/research-harness",
      error: `research harness emitted invalid JSON: ${error.message}`,
      experiments: [],
    };
  }
}

function collectHygiene(runHygiene) {
  if (!runHygiene) {
    return {
      run: false,
      checks: [],
      summary: "not run; pass --run-hygiene for fmt/no-floats/diff checks",
    };
  }
  const checks = [
    runCommand("cargo", ["fmt", "--all", "--check"], { timeoutMs: 30000 }),
    runCommand("./scripts/check-no-floats.sh", [], { timeoutMs: 30000 }),
    runCommand("git", ["diff", "--check"], { timeoutMs: 30000 }),
  ];
  return {
    run: true,
    ok: checks.every((check) => check.ok),
    checks: checks.map((check) => ({
      command: check.command,
      ok: check.ok,
      status: check.status,
      duration_ms: check.duration_ms,
      stdout_tail: tail(check.stdout),
      stderr_tail: tail(check.stderr || check.error),
    })),
  };
}

function tail(text, lines = 12) {
  return String(text || "").trimEnd().split("\n").slice(-lines).join("\n");
}

function collectArtifact(artifact) {
  const model = fileInfo(artifact.model);
  const trace = artifact.trace ? maybeReadJson(artifact.trace) : null;
  const manifest = artifact.manifest ? maybeReadJson(artifact.manifest) : null;
  const evalTrace = artifact.eval ? maybeReadJson(artifact.eval) : null;
  const sample = artifact.sample ? shortText(maybeReadText(artifact.sample)) : "";
  const rawSample = artifact.rawSample ? shortText(maybeReadText(artifact.rawSample)) : "";
  const promptedSample = artifact.promptedSample ? shortText(maybeReadText(artifact.promptedSample)) : "";

  const summary = {
    id: artifact.id,
    kind: artifact.kind,
    model,
  };

  if (trace?.schema === "nsrl.bitmap_denoise_multichannel_trace.v1") {
    summary.trace_schema = trace.schema;
    summary.epochs = trace.epochs;
    summary.eval_input_mean_abs = trace.eval?.input_mean_abs || "";
    summary.eval_predicted_mean_abs = trace.eval?.predicted_mean_abs || "";
    summary.eval_improvement = trace.eval?.input_mean_abs && trace.eval?.predicted_mean_abs
      ? `${trace.eval.input_mean_abs} -> ${trace.eval.predicted_mean_abs}`
      : "";
  } else if (trace?.schema === "nsrl.solomon_latent_trace.v1") {
    summary.trace_schema = trace.schema;
    summary.rows = trace.rows;
    summary.epochs = trace.epochs;
    summary.latent_dim = trace.latent_dim;
    summary.retrieval_top1 = trace.retrieval_top1;
    summary.retrieval_top1_per_mille = trace.retrieval_top1_per_mille;
    summary.retrieval_top5 = trace.retrieval_top5;
    summary.retrieval_top5_per_mille = trace.retrieval_top5_per_mille;
  }

  if (manifest || evalTrace) {
    summary.manifest_schema = manifest?.schema || "";
    summary.corpus_version = manifest?.corpus_version || "";
    summary.text_token_profile = evalTrace?.text_token_profile || manifest?.text_token_profile || "";
    summary.image_token_profile = manifest?.image_token_profile || "";
    summary.token_count = evalTrace?.token_count || manifest?.token_count || null;
    summary.context_seq_len = evalTrace?.context_seq_len || null;
    summary.eval_max_examples = evalTrace?.eval_max_examples ?? null;
    summary.eval_example_count = evalTrace?.example_count ?? null;
    summary.model_hash = evalTrace?.model_hash || "";
    summary.total_accuracy_per_mille = evalTrace?.total?.accuracy_per_mille ?? null;
    summary.text_accuracy_per_mille = evalTrace?.text?.accuracy_per_mille ?? null;
    summary.image_accuracy_per_mille = evalTrace?.image?.accuracy_per_mille ?? null;
    summary.promotion_shape = isPromotionAttentionShape(summary);
  }

  if (sample) {
    summary.sample = sample;
  }
  if (rawSample) {
    summary.raw_sample = rawSample;
  }
  if (promptedSample) {
    summary.prompted_sample = promptedSample;
  }
  return summary;
}

function isPromotionAttentionShape(summary) {
  return Boolean(
    summary.context_seq_len >= 384
      && String(summary.eval_max_examples) === "none"
      && summary.text_token_profile === "chunked"
      && summary.image_token_profile === "symbolic16",
  );
}

function collectArtifacts() {
  return knownArtifacts.map(collectArtifact);
}

function walkForEvidence(startRelativePath, maxDepth = 8) {
  const start = resolvePath(startRelativePath);
  const found = [];
  function visit(dir, depth) {
    if (depth > maxDepth) {
      return;
    }
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name === ".git" || entry.name === "target" || entry.name === "node_modules") {
        continue;
      }
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        visit(fullPath, depth + 1);
      } else if (entry.isFile() && evidenceNames.has(entry.name)) {
        found.push(path.relative(repoRoot, fullPath));
      }
    }
  }
  visit(start, 0);
  return found.sort();
}

function collectProductEvidence() {
  const files = fs.existsSync(resolvePath("data")) ? walkForEvidence("data") : [];
  const byName = {};
  for (const name of evidenceNames) {
    byName[name] = files.filter((file) => path.basename(file) === name);
  }
  return {
    files,
    headline_reports: byName["nsrl-mme-v0.json"],
    quality_reports: byName["quality-report.json"],
    objective_coverage: byName["objective-coverage.json"],
    release_proofs: byName["release-proof.json"],
    pipeline_completions: byName["pipeline-complete.json"],
  };
}

function collectHeadlineEval(productEvidence) {
  const headlineReport = latestFileInfo(productEvidence.headline_reports);
  if (headlineReport.present) {
    const stored = maybeReadJson(headlineReport.path);
    if (stored?.schema === headlineEvalContract.schema) {
      return normalizeStoredHeadlineEval(stored, headlineReport);
    }
  }

  const qualityReport = latestFileInfo(productEvidence.quality_reports);
  const objectiveCoverage = latestFileInfo(productEvidence.objective_coverage);
  const quality = qualityReport.present ? maybeReadJson(qualityReport.path) : null;
  const confidence = quality?.confidence_trace || null;
  const missingEvidence = [];

  if (!qualityReport.present) {
    missingEvidence.push("quality-report.json with confidence_trace");
  }
  if (!objectiveCoverage.present) {
    missingEvidence.push("objective-coverage.json");
  }
  if (qualityReport.present && quality?.schema !== "nsrl.solomon_v2_quality_report.v1") {
    missingEvidence.push("quality report schema nsrl.solomon_v2_quality_report.v1");
  }
  if (qualityReport.present && (!confidence || typeof confidence !== "object" || Array.isArray(confidence))) {
    missingEvidence.push("quality report confidence_trace");
  }

  const metricComponents = confidence ? [
    ...headlineDirectionalFamilies.map((family) => directionalHeadlineComponent(confidence, family)),
    hardNegativeHeadlineComponent(confidence),
  ] : [];
  const gates = confidence ? [
    sourceGroundingGate(confidence),
    generatedOutputGate(confidence, quality),
  ] : [];

  const measuredMetrics = metricComponents.filter((component) => component.score_per_mille !== null);
  const allMetricsMeasured = metricComponents.length > 0
    && measuredMetrics.length === metricComponents.length
    && metricComponents.every((component) => component.rows >= headlineEvalContract.minimum_rows_per_family);
  const gatesGreen = gates.length > 0 && gates.every((gate) => gate.ok === true);
  const score = allMetricsMeasured
    ? Math.min(...metricComponents.map((component) => component.score_per_mille))
    : null;
  const weakest = score === null
    ? null
    : metricComponents
        .map((component) => ({ key: component.key, label: component.label, score_per_mille: component.score_per_mille }))
        .sort((left, right) => left.score_per_mille - right.score_per_mille || left.key.localeCompare(right.key))[0];

  let status = "missing";
  if (qualityReport.present && confidence) {
    status = allMetricsMeasured ? "failed" : "incomplete";
    if (
      allMetricsMeasured
      && gatesGreen
      && score >= headlineEvalContract.target_score_per_mille
      && quality.ok === true
    ) {
      status = "passed";
    }
  }

  return {
    ...headlineEvalContract,
    status,
    score_per_mille: score,
    target_met: status === "passed",
    artifact: headlineReport,
    weakest_component: weakest,
    evidence: {
      quality_report: qualityReport,
      objective_coverage: objectiveCoverage,
      quality_report_ok: quality?.ok === true,
      confidence_label: confidence?.label || "",
    },
    metric_components: metricComponents,
    gates,
    missing_evidence: ["nsrl-mme-v0.json", ...missingEvidence],
  };
}

function normalizeStoredHeadlineEval(report, artifact) {
  const score = report.score_per_mille ?? report.headline_score_per_mille ?? null;
  const evidence = report.evidence || {};
  return {
    ...headlineEvalContract,
    ...report,
    status: report.status || "missing",
    score_per_mille: score,
    headline_score_per_mille: score,
    target_score_per_mille: report.target_score_per_mille ?? headlineEvalContract.target_score_per_mille,
    minimum_rows_per_family: report.minimum_rows_per_family ?? headlineEvalContract.minimum_rows_per_family,
    target_met: report.target_met === true || report.ok === true,
    artifact,
    weakest_component: report.weakest_component || null,
    evidence: {
      quality_report: evidence.quality_report || {
        path: report.inputs?.quality_report || "",
        present: Boolean(report.inputs?.quality_report),
      },
      objective_coverage: evidence.objective_coverage || {
        path: report.inputs?.objective_coverage || "",
        present: Boolean(report.inputs?.objective_coverage),
      },
      quality_report_ok: evidence.quality_report?.ok === true,
      confidence_label: evidence.confidence_label || "",
    },
    metric_components: report.metric_components || [],
    gates: report.gates || [],
    missing_evidence: report.missing_evidence || [],
    errors: report.errors || [],
  };
}

function latestFileInfo(paths) {
  const infos = (paths || []).map(fileInfo).filter((info) => info.present);
  if (infos.length === 0) {
    return { path: "", present: false };
  }
  return infos.sort((left, right) => right.modified_at.localeCompare(left.modified_at))[0];
}

function directionalHeadlineComponent(confidence, family) {
  const group = confidence.directional_native_eval?.groups?.[family.key] || {};
  const stats = group.stats || {};
  const rows = Number(group.targets || stats.targets || 0);
  const score = numberOrNull(stats.top5_accuracy_per_mille);
  const errors = Array.isArray(group.errors) ? group.errors.slice(0, 4) : [];
  if (rows < headlineEvalContract.minimum_rows_per_family) {
    errors.push(`rows ${rows} < ${headlineEvalContract.minimum_rows_per_family}`);
  }
  if (score === null) {
    errors.push("missing top5_accuracy_per_mille");
  }
  if (Number(group.invalid_contexts || stats.invalid_contexts || 0) !== 0) {
    errors.push(`invalid_contexts ${Number(group.invalid_contexts || stats.invalid_contexts || 0)} != 0`);
  }
  return {
    key: family.key,
    label: family.label,
    kind: "model_native_directional_task",
    score_metric: "top5_accuracy_per_mille",
    score_per_mille: score,
    rows,
    ok: errors.length === 0 && group.ok === true,
    source: `confidence_trace.directional_native_eval.groups.${family.key}.stats`,
    errors: [...new Set(errors)],
  };
}

function hardNegativeHeadlineComponent(confidence) {
  const cross = confidence.cross_modal_agreement || {};
  const metrics = [
    ["match_yes", "match yes"],
    ["match_no", "match no"],
    ["wrong_image_negatives", "wrong-image hard negatives"],
    ["wrong_prompt_negatives", "wrong-prompt hard negatives"],
  ].map(([key, label]) => {
    const metric = cross[key] || {};
    return {
      key,
      label,
      rows: Number(metric.count || 0),
      top1: Number(metric.top1 || 0),
      score_per_mille: confidenceTop1PerMille(metric),
      min_margin: numberOrNull(metric.min_margin),
    };
  });
  const scores = metrics.map((metric) => metric.score_per_mille).filter((value) => value !== null);
  const rows = metrics.length > 0 ? Math.min(...metrics.map((metric) => metric.rows)) : 0;
  const errors = [];
  for (const metric of metrics) {
    if (metric.rows < headlineEvalContract.minimum_rows_per_family) {
      errors.push(`${metric.label} rows ${metric.rows} < ${headlineEvalContract.minimum_rows_per_family}`);
    }
    if (metric.score_per_mille === null) {
      errors.push(`${metric.label} missing top1 score`);
    }
    if (metric.rows > 0 && metric.top1 !== metric.rows) {
      errors.push(`${metric.label} top1 ${metric.top1} != rows ${metric.rows}`);
    }
    if (metric.min_margin !== null && metric.min_margin <= 0) {
      errors.push(`${metric.label} min_margin ${metric.min_margin} <= 0`);
    }
  }
  return {
    key: "hard_negative_match",
    label: "match / no-match hard-negative agreement",
    kind: "model_native_cross_modal_agreement",
    score_metric: "minimum top1_per_mille across match yes/no and hard negatives",
    score_per_mille: scores.length === metrics.length ? Math.min(...scores) : null,
    rows,
    ok: errors.length === 0,
    source: "confidence_trace.cross_modal_agreement",
    submetrics: metrics,
    errors: [...new Set(errors)],
  };
}

function sourceGroundingGate(confidence) {
  const source = confidence.source_grounding || {};
  const checks = {
    grounded_corpus_present: source.grounded_corpus_present === true,
    grounded_corpus_ok: source.grounded_corpus_ok === true,
    grounded_source_provenance: source.grounded_source_provenance === true,
    text_queries_have_source_text: source.text_queries_have_source_text === true,
    image_queries_have_source_text: source.image_queries_have_source_text === true,
    sample_queries_have_source_text: source.sample_queries_have_source_text === true,
    sample_source_text_evidence: source.sample_source_text_evidence === true,
    generated_text_source_evidence: source.generated_text_source_evidence === true,
    generated_text_image_agreement: source.generated_text_image_agreement === true,
    expected_generated_text_agreement: source.expected_generated_text_agreement === true,
  };
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([key]) => key);
  return {
    key: "source_grounding",
    label: "source-grounded text/image evidence",
    kind: "gate",
    ok: failed.length === 0,
    source: "confidence_trace.source_grounding",
    failed,
  };
}

function generatedOutputGate(confidence, quality) {
  const generation = confidence.product_generation || {};
  const integrity = quality?.generation_integrity || {};
  const checks = {
    product_generation_present: generation.present === true,
    heldout_partition_ready: generation.heldout_partition_ready === true,
    trace_integrity_ok: generation.trace_integrity_ok === true && integrity.ok !== false,
    product_floor_ok: generation.product_floor_ok === true,
  };
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([key]) => key);
  return {
    key: "generated_output_integrity",
    label: "held-out generated output integrity",
    kind: "gate",
    ok: failed.length === 0,
    source: "confidence_trace.product_generation + generation_integrity",
    sample_count: Number(generation.sample_count || 0),
    best_retrieval_top1_per_mille: numberOrNull(generation.best_retrieval_top1_per_mille),
    failed,
  };
}

function confidenceTop1PerMille(metric) {
  const count = Number(metric?.count || 0);
  const top1 = Number(metric?.top1 || 0);
  if (count <= 0) {
    return null;
  }
  return Math.floor((top1 * 1000) / count);
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function collectPromptEvidence() {
  const prompts = fileInfo("data/processed/key-solomon-goetia-latent-v1/prompts.jsonl");
  const expanded = fileInfo("data/processed/key-solomon-goetia-latent-v1/prompts-expanded.jsonl");
  return {
    prompts: { ...prompts, rows: countLines(prompts.path) },
    expanded_prompts: { ...expanded, rows: countLines(expanded.path) },
  };
}

function countLines(relativePath) {
  if (!relativePath) {
    return null;
  }
  try {
    const text = readText(relativePath);
    return text.length === 0 ? 0 : text.trimEnd().split("\n").length;
  } catch {
    return null;
  }
}

function collectDiagnostic(config) {
  let command = null;
  let diagnosticPath = config.diagnosticPath;
  if (config.refreshFastDiagnostic) {
    diagnosticPath = config.fastDiagnosticOut;
    command = runCommand(
      process.execPath,
      ["scripts/check-solomon-product-diagnostic.mjs", "--fast", "--out", diagnosticPath],
      { timeoutMs: 120000, maxBuffer: 1024 * 1024 * 32 },
    );
  }
  if (!diagnosticPath) {
    return {
      run: false,
      path: "",
      summary: "not supplied; pass --diagnostic or --refresh-fast-diagnostic",
    };
  }
  const diagnostic = maybeReadJson(diagnosticPath);
  if (!diagnostic) {
    return {
      run: Boolean(command),
      path: diagnosticPath,
      command,
      ok: false,
      error: "diagnostic JSON could not be read",
    };
  }
  const failedChecks = Array.isArray(diagnostic.checks)
    ? diagnostic.checks.filter((check) => check && check.ok === false).map((check) => ({
        name: check.name,
        status: check.status,
        schema: check.schema,
        errors: Array.isArray(check.errors) ? check.errors.slice(0, 8) : [],
      }))
    : [];
  return {
    run: Boolean(command),
    path: diagnosticPath,
    command: command
      ? {
          ok: command.ok,
          status: command.status,
          duration_ms: command.duration_ms,
        }
      : null,
    ok: Boolean(diagnostic.ok),
    full_product_proof: Boolean(diagnostic.full_product_proof),
    local_product_proof: Boolean(diagnostic.local_product_proof),
    release_product_proof: Boolean(diagnostic.release_product_proof),
    skipped: diagnostic.skipped || [],
    remaining_product_evidence: diagnostic.remaining_product_evidence || [],
    failed_checks: failedChecks,
  };
}

function deriveStatus(report) {
  const blockers = [];
  const warnings = [];
  const councilBlockers = [];
  if (report.git.dirty) {
    warnings.push(`working tree is dirty (${report.git.change_count} changed paths)`);
  }
  if (report.research_harness.present && !report.research_harness.ok) {
    warnings.push("agentic research harness ledger is not valid");
  }
  if (!report.ising_evidence.ok) {
    warnings.push("Ising audit/confirmation evidence is missing or invalid");
  }
  if (!report.native_model.present) {
    blockers.push("native successor-v2 evidence is missing");
  } else if (!report.native_model.ok) {
    blockers.push("native successor-v2 artifact identity or replay bindings are invalid");
  } else {
    if (!report.native_model.zero_probability_gate) {
      blockers.push(`native successor-v2 has ${report.native_model.zero_probability_windows} zero-probability windows`);
    }
    if (!report.native_model.beats_uniform_gate) {
      blockers.push("native successor-v2 does not beat canonical uniform NLL");
    }
    if (!report.native_model.beats_all_frozen_baselines_gate) {
      blockers.push("native successor-v2 has not beaten every frozen promotion baseline");
    }
  }
  if (!report.open_generation.present) {
    blockers.push("open-generation-v1 development evidence is missing");
  } else if (!report.open_generation.ok) {
    blockers.push("open-generation-v1 checkpoint or serving evidence is invalid");
  } else if (!report.open_generation.promotion_gate_passed) {
    blockers.push("open-generation-v1 candidate has not passed development and promotion gates");
    if (report.open_generation.missing_evidence.length > 0) {
      warnings.push(`open-generation-v1 missing evidence: ${report.open_generation.missing_evidence.join(", ")}`);
    }
  }
  if (!report.council.present || !report.council.ok) {
    councilBlockers.push("Solomon Council v0 core or its exact receipt replay is missing/invalid");
  }
  if (report.council.wisdom_evaluation.state === "not_measured") {
    councilBlockers.push(
      `same-model frozen wisdom evaluation is not measured (${report.council.wisdom_evaluation.pipeline_stage})`);
  } else if (report.council.wisdom_evaluation.hardening.state === "falsified") {
    councilBlockers.push(
      "Council-v1 hardening falsified effective v0 promotion: the historical solo lane lacked tool parity and the diagnostic parity baseline ties all eight dimensions");
  } else if (!report.council.wisdom_evaluation.promotion_gate_passed) {
    councilBlockers.push(
      "Council does not strictly outperform the same underlying model on every wisdom dimension");
  }
  if (report.hygiene.run && !report.hygiene.ok) {
    blockers.push("hygiene checks are not green");
  }
  if (report.headline_eval.status === "missing") {
    blockers.push("headline multimodal LLM eval (NSRL-MME v0) is not measured");
  } else if (report.headline_eval.status !== "passed") {
    const score = report.headline_eval.score_per_mille === null
      ? "unscored"
      : `${report.headline_eval.score_per_mille} per mille`;
    blockers.push(`headline multimodal LLM eval (NSRL-MME v0) is ${report.headline_eval.status}: ${score}`);
  }
  if (report.product_evidence.quality_reports.length === 0) {
    blockers.push("no Solomon quality-report.json found under data/");
  }
  if (report.product_evidence.objective_coverage.length === 0) {
    blockers.push("no objective-coverage.json found under data/");
  }
  if (report.product_evidence.release_proofs.length === 0) {
    blockers.push("no release-proof.json found under data/");
  }
  if (report.product_evidence.pipeline_completions.length === 0) {
    blockers.push("no completed Solomon pipeline-complete.json found under data/");
  }
  const attentionArtifacts = report.artifacts.filter((artifact) => artifact.id.includes("attention"));
  if (!attentionArtifacts.some((artifact) => artifact.promotion_shape)) {
    blockers.push("checked-in NSRLLMM1 attention artifacts are smoke-scale, not promotion profile");
  }
  const rawAttention = attentionArtifacts.find((artifact) => artifact.raw_sample || artifact.sample);
  if (rawAttention && !/Solomon selects Bael: He/.test(rawAttention.raw_sample || rawAttention.sample || "")) {
    warnings.push("raw/free-running attention text is still weak or diagnostic-only");
  }
  if (report.diagnostic.path && !report.diagnostic.ok) {
    blockers.push("product diagnostic is not green");
    for (const check of report.diagnostic.failed_checks || []) {
      blockers.push(`diagnostic failed check: ${check.name}`);
    }
  }
  if (report.diagnostic.skipped?.length) {
    warnings.push(`diagnostic skipped checks: ${report.diagnostic.skipped.join(", ")}`);
  }
  const releaseReady = blockers.length === 0
    && report.product_evidence.release_proofs.length > 0
    && report.product_evidence.pipeline_completions.length > 0;
  const llmPathState = releaseReady ? "release-ready" : "research/proof-gated";
  return {
    release_ready: releaseReady,
    llm_path_state: llmPathState,
    blockers: [...new Set(blockers)],
    warnings: [...new Set(warnings)],
    council_ready: councilBlockers.length === 0,
    council_blockers: [...new Set(councilBlockers)],
  };
}

function buildReport(config) {
  const productEvidence = collectProductEvidence();
  const report = {
    schema,
    generated_at: new Date().toISOString(),
    repo_root: repoRoot,
    git: collectGit(),
    native_model: collectNativeModelEvidence(),
    open_generation: collectOpenGenerationEvidence(),
    council: collectCouncilEvidence(),
    ising_evidence: collectIsingEvidence(),
    research_harness: collectResearchHarness(),
    hygiene: collectHygiene(config.runHygiene),
    prompts: collectPromptEvidence(),
    artifacts: collectArtifacts(),
    product_evidence: productEvidence,
    headline_eval: collectHeadlineEval(productEvidence),
    diagnostic: collectDiagnostic(config),
  };
  report.status = deriveStatus(report);
  report.next_commands = nextCommands(report);
  return report;
}

function nextCommands(report) {
  const commands = [];
  if (report.headline_eval.status === "missing") {
    commands.push("node scripts/check-nsrl-mme-v0.mjs --out data/processed/nsrl-mme-v0.json");
  }
  if (!report.hygiene.run) {
    commands.push("node scripts/nsrl-status.mjs --run-hygiene");
  }
  if (!report.diagnostic.path) {
    commands.push("node scripts/nsrl-status.mjs --refresh-fast-diagnostic");
  }
  if (report.research_harness.present && !report.research_harness.ok) {
    commands.push("node scripts/research-harness.mjs verify");
  }
  if (!report.ising_evidence.ok) {
    commands.push("node scripts/check-production-atomic-ising-v1.mjs");
    commands.push("node scripts/check-production-atomic-ising-confirmation-v1.mjs");
    commands.push("node scripts/check-production-cross-source-exchange-v1.mjs");
    commands.push("node scripts/check-production-cross-source-exchange-publication-v1.mjs");
    commands.push("node scripts/check-production-multifamily-exchange-v1.mjs");
  }
  if (!report.native_model.ok) {
    commands.push("node scripts/run-integer-transformer-successor-v2.mjs --check");
  }
  if (!report.open_generation.present) {
    commands.push("scripts/run-open-generation-development-v1.sh");
  } else {
    commands.push("scripts/check-open-generation-v1.sh");
  }
  if (!report.council.ok) {
    commands.push("node scripts/check-solomon-council-v0.mjs");
  }
  if (report.council.wisdom_evaluation.state === "not_measured") {
    commands.push("node scripts/freeze-solomon-wisdom-casebook-v0.mjs PRIVATE-DRAFT.json PUBLIC-CASEBOOK.json PRIVATE-GOLD-VAULT.json");
    commands.push("node scripts/open-solomon-wisdom-gold-v0.mjs PUBLIC-CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json PRIVATE-GOLD-VAULT.json GOLD-OPENING.json");
    commands.push("node scripts/compile-solomon-wisdom-eval-v0.mjs PUBLIC-CASEBOOK.json SOLO-BUNDLE.json COUNCIL-BUNDLE.json GOLD-OPENING.json GENERATION-INTEGRITY.json PROVENANCE.json FROZEN-SAME-MODEL-INPUT.json");
    commands.push("node scripts/evaluate-solomon-wisdom-v0.mjs FROZEN-SAME-MODEL-INPUT.json benchmarks/solomon-council-v0/wisdom-eval-result.json");
  }
  if (report.diagnostic.failed_checks?.some((check) => check.name === "release-candidate-self-test")) {
    commands.push("node scripts/check-solomon-release-candidate-self-test.mjs");
  }
  if (report.product_evidence.quality_reports.length === 0) {
    commands.push("node scripts/check-solomon-product-diagnostic.mjs --out /tmp/nsrl-solomon-product-diagnostic.json");
  }
  if (report.status.blockers.some((blocker) => blocker.includes("pipeline-complete"))) {
    commands.push("NSRL_S3_URI=s3://BUCKET/PREFIX scripts/aws/run-solomon-end-to-end.sh");
  }
  commands.push("scripts/aws/prove-solomon-product-run.sh --s3-pipeline-uri s3://BUCKET/PREFIX/pipelines/RUN_NAME --launch-dir data/aws-launches/RUN_NAME --require-launch-dir");
  return [...new Set(commands)];
}

function renderMarkdown(report) {
  const lines = [];
  lines.push("# NSRL Project Status");
  lines.push("");
  lines.push(`Generated: ${report.generated_at}`);
  lines.push(`Repo: \`${report.repo_root}\``);
  lines.push("");
  lines.push(`Overall: **${report.status.release_ready ? "release-ready" : "not release-ready"}**`);
  lines.push(`LLM path: **${report.status.llm_path_state}**`);
  lines.push("");

  lines.push("## Current Read");
  lines.push("");
  lines.push(`- Branch: \`${report.git.branch || "unknown"}\``);
  lines.push(`- HEAD: \`${report.git.head || "unknown"}\``);
  lines.push(`- Working tree: ${report.git.dirty ? `${report.git.change_count} changed paths` : "clean"}`);
  lines.push(`- Native model: ${renderNativeModelOneLine(report.native_model)}`);
  lines.push(`- Open generation: ${report.open_generation.state}; cached serving ${report.open_generation.serving_ok ? "verified" : "invalid"}; promotion ${report.open_generation.promotion_gate_passed ? "passed" : "not passed"}`);
  lines.push(`- Solomon Council: ${report.council.state}; wisdom evaluation ${report.council.wisdom_evaluation.state}`);
  lines.push(`- Ising evidence: ${renderIsingEvidenceOneLine(report.ising_evidence)}`);
  lines.push(`- Research harness: ${renderResearchHarnessOneLine(report.research_harness)}`);
  lines.push(`- Hygiene: ${renderHygiene(report.hygiene)}`);
  lines.push(`- Product diagnostic: ${renderDiagnosticOneLine(report.diagnostic)}`);
  lines.push(`- Headline eval: ${renderHeadlineOneLine(report.headline_eval)}`);
  lines.push(`- Product proof files: ${report.product_evidence.files.length} found`);
  lines.push(`- Held-out prompt rows: ${report.prompts.expanded_prompts.rows ?? "missing"} expanded, ${report.prompts.prompts.rows ?? "missing"} base`);
  lines.push("");

  lines.push("## Native Model Promotion");
  lines.push("");
  lines.push(`- Contract: \`${report.native_model.contract}\`; state **${report.native_model.state}**`);
  lines.push(`- Identity: artifact ${report.native_model.artifact_identity_ok ? "verified" : "invalid"}; training receipt ${report.native_model.training_receipt_ok ? "verified" : "invalid"}; replay bindings ${report.native_model.replay_bindings_ok ? "verified" : "invalid"}; forbidden assistance ${report.native_model.assistance_absent ? "absent" : "present or unverified"}`);
  lines.push(`- Candidate: ${report.native_model.candidate_nll_millibits ?? "missing"} canonical NLL millibits over ${report.native_model.target_count ?? "missing"} targets; ${report.native_model.zero_probability_windows ?? "missing"} zero-probability windows`);
  for (const [system, nll] of Object.entries(report.native_model.baseline_nll_millibits)) {
    const gap = report.native_model.gap_to_baseline_millibits[system];
    lines.push(`- Gap to ${system}: ${gap >= 0 ? "+" : ""}${gap} millibits (baseline ${nll})`);
  }
  lines.push(`- Gates: zero-probability ${report.native_model.zero_probability_gate ? "green" : "red"}; beats uniform ${report.native_model.beats_uniform_gate ? "green" : "red"}; beats every frozen baseline ${report.native_model.beats_all_frozen_baselines_gate ? "green" : "red"}`);
  lines.push("");

  lines.push("## Open Generation v1");
  lines.push("");
  lines.push(`- State: **${report.open_generation.state}**; candidate model \`${report.open_generation.candidate.model_fnv64 || "missing"}\`; tokenizer \`${report.open_generation.candidate.tokenizer_fnv64 || "missing"}\``);
  lines.push(`- Serving: ${report.open_generation.serving_ok ? "green" : "red"}; cache ${report.open_generation.cache.maximum_state_bytes ?? "missing"} state bytes + ${report.open_generation.cache.maximum_workspace_bytes ?? "missing"} workspace bytes`);
  lines.push(`- Modeling: ${report.open_generation.modeling?.millibits_per_original_utf8_byte ?? "missing"} millibits/original UTF-8 byte; required baselines ${report.open_generation.modeling?.required_baselines_measured ? "measured" : "missing"}`);
  lines.push(`- Generation: repeat ${report.open_generation.generation_metrics.max_repeat_4gram_share_per_mille ?? "missing"}‰; unique floor ${report.open_generation.generation_metrics.min_unique_4gram_share_per_mille ?? "missing"}‰; entropy floor ${report.open_generation.generation_metrics.min_entropy_q10 ?? "missing"} Q10; UTF-8 ${report.open_generation.generation_metrics.utf8_valid_per_mille ?? "missing"}‰; context ${report.open_generation.generation_metrics.context_use_per_mille ?? "missing"}‰; distractor ${report.open_generation.generation_metrics.distractor_resistance_per_mille ?? "missing"}‰`);
  lines.push(`- Missing evidence: ${report.open_generation.missing_evidence.join(", ") || "none"}`);
  lines.push("");

  lines.push("## Solomon Council v0/v1");
  lines.push("");
  lines.push(`- Core: **${report.council.state}**; ${report.council.seal_signatures_verified}/6 Ed25519 faculty seals verified; shadow-only execution ${report.council.shadow_execution_only ? "enforced" : "invalid"}`);
  lines.push(`- Judge states checked: ${report.council.decision_states_checked.join(", ")}; exact initial receipt ${report.council.receipt_ok ? "verified" : "invalid"}; outcome/revision chain ${report.council.revision_ok ? "verified" : "invalid"}`);
  lines.push(`- Wisdom receipt: \`${report.council.receipt_sha256 || "missing"}\`; revised receipt: \`${report.council.revised_receipt_sha256 || "missing"}\``);
  lines.push(`- Bounded decision-regret experiment: publication **${report.council.bounded_decision_regret_evidence.publication_status}** with ${report.council.bounded_decision_regret_evidence.fired_passages} favorable fired passages and signed regret ${report.council.bounded_decision_regret_evidence.signed_regret_q32 || "missing"} Q32; MJ-20 falsifies the marginal-to-conditional bridge, so the non-crossing e-process does not establish sequential safety`);
  const adaptive = report.council.adaptive_composition;
  lines.push(`- Adaptive replacement: theory ${adaptive.theory_check_ok ? "checked" : "invalid"}; execution **${adaptive.execution_state}** with ${adaptive.adaptive_fires} adaptive fires, ${adaptive.calibration_cube_rows} calibration-cube rows across ${adaptive.calibration_source_panels} source panels, and ${adaptive.endpoint_nll_millibits || "missing"} endpoint NLL millibits; exact replay ${adaptive.exact_replay ? "verified" : "absent or invalid"}; threshold retuning and optimizer promotion remain unauthorized`);
  const hardening = report.council.wisdom_evaluation.hardening;
  lines.push(`- Historical v0 wisdom gate: ${report.council.wisdom_evaluation.v0_promotion_gate_passed ? "passed" : "not passed"}; effective promotion ${report.council.wisdom_evaluation.promotion_gate_passed ? "authorized" : "not authorized"}`);
  lines.push(`- Council-v1 hardening: **${hardening.state}**; actual solo/Council tool observations ${hardening.actual_solo_tool_observations}/${hardening.actual_council_tool_observations}; ${hardening.tool_parity_dimensions_tied}/8 parity dimensions tie; ${hardening.gates_passed}/${hardening.gates_total} hardening gates pass; shadow-only ${hardening.remain_shadow_only ? "enforced" : "invalid"}`);
  lines.push(`- Wisdom ceremony: ${report.council.wisdom_ceremony_check_ok ? "byte-bound compiler/replay self-check green" : "invalid"}; no production casebook or paired lanes are inferred from this self-test`);
  for (const blocker of report.status.council_blockers) {
    lines.push(`- Council blocker: ${blocker}`);
  }
  lines.push("");

  lines.push("## Ising Evidence");
  lines.push("");
  lines.push(`- Audit: **${report.ising_evidence.audit.state}**; contract \`${report.ising_evidence.audit.contract_sha256 || "missing"}\`; result \`${report.ising_evidence.audit.result_sha256 || "missing"}\``);
  lines.push(`- Document Ising: **${report.ising_evidence.document_ising.state}**; ${report.ising_evidence.document_ising.source_clusters} source cluster`);
  lines.push(`- Confirmation: **${report.ising_evidence.confirmation.verdict}** (${report.ising_evidence.confirmation.endpoints_supported}/${report.ising_evidence.confirmation.endpoints_total} endpoints); contract \`${report.ising_evidence.confirmation.contract_sha256 || "missing"}\`; result \`${report.ising_evidence.confirmation.result_sha256 || "missing"}\``);
  lines.push(`- Cross-source exchange: **${report.ising_evidence.cross_source.verdict}**; ${report.ising_evidence.cross_source.envelope_covered}/${report.ising_evidence.cross_source.evaluation_source_panels} panels covered, ${report.ising_evidence.cross_source.fired_source_panels} fired, ${report.ising_evidence.cross_source.unsafe_firings} unsafe; contract \`${report.ising_evidence.cross_source.contract_sha256 || "missing"}\`; result \`${report.ising_evidence.cross_source.result_sha256 || "missing"}\``);
  lines.push(`- Multi-family exchange: **${report.ising_evidence.multifamily_cross_source.verdict}**; ${report.ising_evidence.multifamily_cross_source.envelope_covered}/${report.ising_evidence.multifamily_cross_source.evaluation_source_panels} panels covered, ${report.ising_evidence.multifamily_cross_source.fired_passages} passages fired across ${report.ising_evidence.multifamily_cross_source.firing_families.length} families, ${report.ising_evidence.multifamily_cross_source.unsafe_firings} unsafe; net improvement ${report.ising_evidence.multifamily_cross_source.net_heldout_improvement_q32 || "missing"} Q32; contract \`${report.ising_evidence.multifamily_cross_source.contract_sha256 || "missing"}\`; result \`${report.ising_evidence.multifamily_cross_source.result_sha256 || "missing"}\``);
  lines.push(`- Boundary: default \`${report.ising_evidence.boundaries.default_optimizer}\`; optimizer change ${report.ising_evidence.boundaries.optimizer_change_authorized === false ? "not authorized" : "unknown"}; paid scaling ${report.ising_evidence.boundaries.paid_scaling_authorized === false ? "not authorized" : "unknown"}; cross-source certificate bounded to ${report.ising_evidence.boundaries.cross_source_certificate_scope}; arbitrary-source generalization unidentified; documents ${report.ising_evidence.boundaries.sealed_documents} sealed`);
  lines.push(`- Sigma--delta scope: ${report.ising_evidence.audit.sigma_delta_scope}`);
  lines.push("");

  lines.push("## Headline Eval");
  lines.push("");
  lines.push(`- Contract: **${report.headline_eval.label}** (\`${report.headline_eval.schema}\`)`);
  lines.push(`- Score: ${renderHeadlineScore(report.headline_eval)}; target ${report.headline_eval.target_score_per_mille} per mille`);
  lines.push(`- Metric: ${report.headline_eval.headline_metric}`);
  lines.push(`- Policy: ${report.headline_eval.policy}`);
  lines.push(`- Evidence: ${renderHeadlineEvidence(report.headline_eval)}`);
  if (report.headline_eval.missing_evidence.length > 0) {
    lines.push(`- Missing evidence: ${report.headline_eval.missing_evidence.join(", ")}`);
  }
  if (report.headline_eval.metric_components.length > 0) {
    lines.push("");
    lines.push("Metric components:");
    for (const component of report.headline_eval.metric_components) {
      lines.push(`- ${component.label}: ${renderComponentScore(component)} (${component.rows} rows)`);
    }
  }
  if (report.headline_eval.gates.length > 0) {
    lines.push("");
    lines.push("Required gates:");
    for (const gate of report.headline_eval.gates) {
      lines.push(`- ${gate.label}: ${gate.ok ? "green" : `not green (${gate.failed.join(", ")})`}`);
    }
  }
  lines.push("");

  lines.push("## Blockers");
  lines.push("");
  if (report.status.blockers.length === 0) {
    lines.push("- None detected by this status surface.");
  } else {
    for (const blocker of report.status.blockers) {
      lines.push(`- ${blocker}`);
    }
  }
  if (report.status.warnings.length > 0) {
    lines.push("");
    lines.push("Warnings:");
    for (const warning of report.status.warnings) {
      lines.push(`- ${warning}`);
    }
  }
  lines.push("");

  lines.push("## Artifact Inventory");
  lines.push("");
  for (const artifact of report.artifacts) {
    lines.push(`- **${artifact.id}**: ${artifact.model.present ? artifact.model.size : "missing"} ${artifact.kind}`);
    if (artifact.eval_improvement) {
      lines.push(`  Eval MAE: ${artifact.eval_improvement}`);
    }
    if (artifact.retrieval_top1 !== undefined) {
      lines.push(`  Retrieval: top-1 ${artifact.retrieval_top1}/72 (${artifact.retrieval_top1_per_mille} per mille), top-5 ${artifact.retrieval_top5}/72 (${artifact.retrieval_top5_per_mille} per mille)`);
    }
    if (artifact.total_accuracy_per_mille !== undefined && artifact.total_accuracy_per_mille !== null) {
      lines.push(`  Eval: context ${artifact.context_seq_len}, max examples ${artifact.eval_max_examples}, total/text/image accuracy ${artifact.total_accuracy_per_mille}/${artifact.text_accuracy_per_mille}/${artifact.image_accuracy_per_mille} per mille`);
    }
    if (artifact.raw_sample) {
      lines.push(`  Raw sample: "${artifact.raw_sample}"`);
    }
    if (artifact.prompted_sample) {
      lines.push(`  Prompted sample: "${artifact.prompted_sample}"`);
    } else if (artifact.sample) {
      lines.push(`  Sample: "${artifact.sample}"`);
    }
  }
  lines.push("");

  if (report.diagnostic.failed_checks?.length) {
    lines.push("## Failed Diagnostic Checks");
    lines.push("");
    for (const check of report.diagnostic.failed_checks) {
      lines.push(`- ${check.name}: ${check.errors?.[0] || "failed"}`);
    }
    lines.push("");
  }

  lines.push("## Next Commands");
  lines.push("");
  for (const command of report.next_commands) {
    lines.push(`- \`${command}\``);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function renderHeadlineOneLine(headline) {
  const score = renderHeadlineScore(headline);
  const weakest = headline.weakest_component
    ? `; weakest: ${headline.weakest_component.label}`
    : "";
  return `${headline.status} (${score}; target ${headline.target_score_per_mille} per mille${weakest})`;
}

function renderHeadlineScore(headline) {
  if (headline.score_per_mille === null || headline.score_per_mille === undefined) {
    return "not measured";
  }
  return `${headline.score_per_mille} per mille`;
}

function renderHeadlineEvidence(headline) {
  const artifact = headline.artifact?.present
    ? `artifact \`${headline.artifact.path}\`; `
    : "";
  const quality = headline.evidence.quality_report.present
    ? `quality \`${headline.evidence.quality_report.path}\``
    : "quality missing";
  const objective = headline.evidence.objective_coverage.present
    ? `objective \`${headline.evidence.objective_coverage.path}\``
    : "objective missing";
  const ok = headline.evidence.quality_report_ok ? "quality ok" : "quality not ok";
  return `${artifact}${quality}; ${objective}; ${ok}`;
}

function renderComponentScore(component) {
  const score = component.score_per_mille === null ? "not measured" : `${component.score_per_mille} per mille`;
  if (component.ok) {
    return score;
  }
  const errors = component.errors?.length ? `; ${component.errors.join("; ")}` : "";
  return `${score}, not green${errors}`;
}

function renderHygiene(hygiene) {
  if (!hygiene.run) {
    return hygiene.summary;
  }
  if (hygiene.ok) {
    return "green";
  }
  const failed = hygiene.checks.filter((check) => !check.ok).map((check) => check.command).join(", ");
  return `not green (${failed})`;
}

function renderDiagnosticOneLine(diagnostic) {
  if (!diagnostic.path) {
    return diagnostic.summary;
  }
  const state = diagnostic.ok ? "green" : "not green";
  const failed = diagnostic.failed_checks?.length
    ? `; failed: ${diagnostic.failed_checks.map((check) => check.name).join(", ")}`
    : "";
  const skipped = diagnostic.skipped?.length ? `; skipped: ${diagnostic.skipped.join(", ")}` : "";
  return `${state} at \`${diagnostic.path}\`${failed}${skipped}`;
}

function renderResearchHarnessOneLine(harness) {
  if (!harness.present) return "not initialized";
  if (!harness.ok) return `invalid (${harness.error || "verification failed"})`;
  const outcomes = (harness.experiments || [])
    .filter((experiment) => experiment.outcome)
    .map((experiment) => `${experiment.id}=${experiment.outcome}`)
    .join(", ");
  return `${harness.experiment_count} experiments, ${harness.event_count} events${outcomes ? `; ${outcomes}` : ""}`;
}

function renderIsingEvidenceOneLine(evidence) {
  if (!evidence.present) return "missing";
  const state = evidence.ok ? "replayable" : "invalid";
  return `${state}; confirmation ${evidence.confirmation.verdict}; cross-source ${evidence.cross_source.verdict}; multi-family ${evidence.multifamily_cross_source.verdict}; optimizer change not authorized`;
}

function renderNativeModelOneLine(model) {
  if (!model.present) return "missing";
  const nll = model.candidate_nll_millibits ?? "unscored";
  const zeros = model.zero_probability_windows ?? "unknown";
  return `${model.state}; NLL ${nll} millibits; ${zeros} zero-probability windows; replay ${model.ok ? "bound" : "invalid"}`;
}

function writeOutput(config, text) {
  if (!config.outPath) {
    process.stdout.write(text);
    return;
  }
  const outPath = resolvePath(config.outPath);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, text, "utf8");
}

function main() {
  const config = parseArgs(process.argv.slice(2));
  const report = buildReport(config);
  const output = config.json ? `${JSON.stringify(report, null, 2)}\n` : renderMarkdown(report);
  writeOutput(config, output);
  if (config.strict && !report.status.release_ready) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
