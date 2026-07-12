#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-shared-trunk-low-rank-moe-v1";
const out = process.argv[3] ?? path.join(root, "shared-trunk-low-rank-report.json");
const authors = ["crowley", "shakespeare", "blake"];
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const experts = {};
let trunkModelHash = null;

for (const author of authors) {
  const artifact = await readFile(path.join(root, "experts", author, "low-rank.nsrlle"));
  const trainBytes = await readFile(path.join(root, "experts", author, "train.json"));
  const train = JSON.parse(trainBytes);
  if (trunkModelHash === null) trunkModelHash = train.trunk_model_hash;
  if (train.trunk_model_hash !== trunkModelHash) throw new Error("expert trunk hash mismatch");
  const evaluations = {};
  for (const testAuthor of authors) {
    evaluations[testAuthor] = JSON.parse(
      await readFile(path.join(root, "experts", author, `eval-${testAuthor}.json`), "utf8"),
    ).metrics;
  }
  experts[author] = {
    artifact_bytes: artifact.length,
    artifact_sha256: sha256(artifact),
    train_sha256: sha256(trainBytes),
    train_initial: train.initial,
    train_final: train.final,
    train_updates: train.updates,
    evaluations,
  };
}
if (new Set(authors.map((author) => experts[author].artifact_sha256)).size !== 3) {
  throw new Error("low-rank expert artifacts are not unique");
}

const oracleBytes = await readFile(path.join(root, "oracles", "final-report.json"));
const oracle = JSON.parse(oracleBytes);
if (oracle.trunk.model_hash !== trunkModelHash) throw new Error("oracle trunk hash mismatch");
if (oracle.trunk.forward_count * 3 !== oracle.trunk.naive_three_model_forward_count) {
  throw new Error("shared forward accounting mismatch");
}
const fixed = oracle.fixed_experts[oracle.best_fixed_expert];
const token = oracle.oracle_routes.token;
const strongerDiagonalFixedMean = 58_375;
const report = {
  schema: "nsrl.literary_shared_trunk_low_rank_moe.v1",
  trunk: {
    model_hash: trunkModelHash,
    frozen_during_expert_training: true,
    final_test_forward_count: oracle.trunk.forward_count,
    naive_three_model_forward_count: oracle.trunk.naive_three_model_forward_count,
    forward_reduction_factor: 3,
  },
  expert_type: "fixed_projection_low_rank_hidden_residual_q15",
  expert_rank: oracle.experts.rank,
  expert_parameter_count: oracle.experts.parameter_count_each,
  experts,
  frozen_final: {
    samples: oracle.dataset.samples,
    windows: oracle.dataset.windows,
    best_fixed_expert: authors[oracle.best_fixed_expert],
    fixed,
    prompt_oracle: oracle.oracle_routes.prompt,
    span_oracle: oracle.oracle_routes.span,
    token_oracle: token,
    token_oracle_delta_vs_fixed: {
      accuracy_per_mille: token.accuracy_per_mille - fixed.accuracy_per_mille,
      mistake_count: token.mistakes - fixed.mistakes,
      mean_probability_error_q15:
        token.mean_probability_error_q15 - fixed.mean_probability_error_q15,
    },
    token_oracle_delta_vs_stronger_diagonal_fixed: {
      mean_probability_error_q15: token.mean_probability_error_q15 - strongerDiagonalFixedMean,
    },
  },
  promotion: {
    shared_execution_proven: true,
    expert_diversity_proven: true,
    token_router_training_promoted: false,
    reason:
      "low-rank token oracle is 235 Q15 better than its fixed branch but remains 46 Q15 worse than the existing diagonal fixed expert",
    next_expert_type: "diagonal_plus_low_rank_hybrid_hidden_residual",
  },
  source_sha256: { oracle: sha256(oracleBytes) },
  known_non_claims: [
    "fixed_projection_low_rank_is_not_a_full_ffn_expert",
    "oracle_is_target_aware",
    "does_not_claim_language_model_quality",
  ],
};

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    out,
    promotion: report.promotion,
    delta: report.frozen_final.token_oracle_delta_vs_fixed,
  }),
);
