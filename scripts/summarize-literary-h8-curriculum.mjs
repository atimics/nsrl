#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-curriculum-v1";
const out = process.argv[3] ?? path.join(root, "report.json");
const sha256 = (value) => createHash("sha256").update(value).digest("hex");

async function json(relative) {
  const bytes = await readFile(path.join(root, relative));
  return { value: JSON.parse(bytes), sha256: sha256(bytes) };
}

const stageSpecs = [
  { stage: 1, offset: 0, learningRate: null, holdout: "stage1-offset0/holdout.json", kind: "base" },
  {
    stage: 2,
    offset: 23_552,
    learningRate: 1024,
    holdout: "residual-sweep-stage2/r16-b16-lr1024-holdout.json",
    train: "residual-sweep-stage2/r16-b16-lr1024.json",
  },
  { stage: 3, offset: 47_104, learningRate: 1024, holdout: "resume-stage3/lr1024-holdout.json", train: "resume-stage3/lr1024.json" },
  { stage: 4, offset: 70_656, learningRate: 1024, holdout: "resume-stage4/lr1024-holdout.json", train: "resume-stage4/lr1024.json" },
  { stage: 5, offset: 94_208, learningRate: 256, holdout: "resume-stage5/lr256-holdout.json", train: "resume-stage5/lr256.json" },
  { stage: 6, offset: 117_760, learningRate: 1024, holdout: "resume-stage6/lr1024-holdout.json", train: "resume-stage6/lr1024.json" },
  { stage: 7, offset: 141_312, learningRate: 256, holdout: "resume-stage7/lr256-holdout.json", train: "resume-stage7/lr256.json" },
];

const stages = [];
for (const spec of stageSpecs) {
  const holdout = await json(spec.holdout);
  const metrics = spec.kind === "base" ? holdout.value.evaluation : holdout.value.metrics;
  const train = spec.train ? await json(spec.train) : null;
  stages.push({
    stage: spec.stage,
    token_offset: spec.offset,
    windows: spec.stage === 1 ? 512 : train.value.initial.windows,
    learning_rate: spec.learningRate,
    holdout: metrics,
    train_initial: train?.value.initial ?? null,
    train_final: train?.value.final ?? null,
    expert_hash: train?.value.artifact_hash ?? null,
    resumed_from_hash: train?.value.resumed_from_hash ?? null,
    source_sha256: { holdout: holdout.sha256, ...(train ? { train: train.sha256 } : {}) },
  });
}

function generationMetrics(bytes) {
  const spaces = [...bytes].filter((byte) => byte === 32).length;
  const alphabetic = [...bytes].filter(
    (byte) => (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122),
  ).length;
  let longestSpaceRun = 0;
  let currentSpaceRun = 0;
  for (const byte of bytes) {
    currentSpaceRun = byte === 32 ? currentSpaceRun + 1 : 0;
    longestSpaceRun = Math.max(longestSpaceRun, currentSpaceRun);
  }
  return {
    bytes: bytes.length,
    distinct_bytes: new Set(bytes).size,
    non_space_per_mille: Math.floor(((bytes.length - spaces) * 1000) / bytes.length),
    alphabetic_per_mille: Math.floor((alphabetic * 1000) / bytes.length),
    longest_space_run: longestSpaceRun,
  };
}

const generation = {};
for (const stage of ["stage1", "stage2", "stage3", "stage7"]) {
  generation[stage] = {};
  for (const prompt of ["soul", "love", "law"]) {
    const relative = `generation/${stage}-${prompt}-top8.txt`;
    const bytes = await readFile(path.join(root, relative));
    generation[stage][prompt] = {
      ...generationMetrics(bytes),
      sha256: sha256(bytes),
    };
  }
}
const generationGatePassed = Object.values(generation.stage7).every(
  (sample) =>
    sample.non_space_per_mille >= 500 &&
    sample.alphabetic_per_mille >= 300 &&
    sample.distinct_bytes >= 15 &&
    sample.longest_space_run <= 16,
);

const oracle = await json("oracles/final-stage7-report.json");
const rmsOracle = await json("rms-oracles/final-report.json");
const fixed = oracle.value.fixed_experts[oracle.value.best_fixed_expert];
const tokenOracle = oracle.value.oracle_routes.token;
const rmsFixed = rmsOracle.value.fixed_experts[rmsOracle.value.best_fixed_expert];
const rmsStage6Holdout = (await json("rms-resume-stage6/holdout.json")).value.metrics;
const stage8 = await Promise.all(
  [64, 256, 1024].map((rate) => json(`resume-stage8/lr${rate}-holdout.json`)),
);
const report = {
  schema: "nsrl.literary_h8_residual_curriculum.v1",
  architecture: {
    trunk_profile: "small-h8-d128-ff256",
    trunk_model_hash: oracle.value.trunk.model_hash,
    frozen_trunk: true,
    residual_rank: oracle.value.experts.rank,
    residual_parameters: oracle.value.experts.parameter_count_each,
    context_seq_len: 64,
  },
  curriculum: {
    stage_windows: 512,
    batch_windows: 16,
    stages,
    stopped_after_stage: 7,
    stop_reason: "all stage-8 learning rates increased exact frozen holdout error",
    stage8_candidates: Object.fromEntries(
      [64, 256, 1024].map((rate, index) => [rate, stage8[index].value.metrics]),
    ),
  },
  frozen_final: {
    samples: oracle.value.dataset.samples,
    windows: oracle.value.dataset.windows,
    fixed,
    prompt_oracle: oracle.value.oracle_routes.prompt,
    span_oracle: oracle.value.oracle_routes.span,
    token_oracle: tokenOracle,
    token_oracle_delta_vs_fixed: {
      accuracy_per_mille: tokenOracle.accuracy_per_mille - fixed.accuracy_per_mille,
      mistakes: tokenOracle.mistakes - fixed.mistakes,
      probability_error_q15: tokenOracle.probability_error_q15 - fixed.probability_error_q15,
      mean_probability_error_q15:
        tokenOracle.mean_probability_error_q15 - fixed.mean_probability_error_q15,
    },
    trunk_forward_reduction_factor: 3,
  },
  diagnostics: {
    direct_i8_adam_continuation_promoted: false,
    direct_i8_adam_reason:
      "shift 5 regressed stage and holdout loss; shifts 6-8 produced zero updates; guarded runs changed no weights",
    final_hidden_teacher_distillation_promoted: false,
    final_hidden_teacher_distillation_reason:
      "ranks 16-128 retained only a small fraction of whole-model optimizer diversity",
    stage2_residual_byte_exact_replay: true,
    rmsnorm_only_internal_training: {
      implemented: true,
      updated_parameter_count: 512,
      non_rms_stage6_holdout: stages.find((stage) => stage.stage === 6).holdout,
      rms_plus_residual_stage6_holdout: rmsStage6Holdout,
      non_rms_frozen_final: fixed,
      rms_plus_residual_frozen_final: rmsFixed,
      promoted_over_non_rms_final: rmsFixed.probability_error_q15 < fixed.probability_error_q15,
    },
  },
  generation: {
    decode: "deterministic_top_k_sample",
    top_k: 8,
    sample_seed: 7,
    samples: generation,
    quality_gate_passed: generationGatePassed,
    gate:
      "each stage-7 sample requires >=500 non-space per mille, >=300 alphabetic per mille, >=15 distinct bytes, and <=16 longest space run",
  },
  promotion: {
    residual_curriculum_promoted: true,
    shared_trunk_execution_promoted: true,
    rmsnorm_only_scope_promoted_mechanically: true,
    rms_plus_residual_promoted_over_non_rms_final:
      rmsFixed.probability_error_q15 < fixed.probability_error_q15,
    learned_router_training_promoted: false,
    router_reason: "stage-7 token oracle ceiling is only 96 mean Q15",
    generation_quality_promoted: generationGatePassed,
    next_step:
      "increase effective trainable precision inside transformer blocks and expand balanced data before claiming language generation",
  },
  source_sha256: { oracle: oracle.sha256, rms_oracle: rmsOracle.sha256 },
  known_non_claims: [
    "next_token_improvement_is_not_prose_quality",
    "top_k_samples_fail_current_generation_gate",
    "does_not_claim_llm_quality",
  ],
};

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    out,
    final_stage: stages.at(-1),
    frozen_final: report.frozen_final,
    generation_gate_passed: generationGatePassed,
    promotion: report.promotion,
  }),
);
