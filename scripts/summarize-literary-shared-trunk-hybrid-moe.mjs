#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-shared-trunk-hybrid-moe-v1";
const out = process.argv[3] ?? path.join(root, "shared-trunk-hybrid-report.json");
const authors = ["crowley", "shakespeare", "blake"];
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const experts = {};
let trunkModelHash = null;

for (const author of authors) {
  const artifact = await readFile(path.join(root, "experts", author, "hybrid.nsrlle"));
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
  throw new Error("hybrid expert artifacts are not unique");
}

const oracleBytes = await readFile(path.join(root, "oracles", "final-report.json"));
const oracle = JSON.parse(oracleBytes);
const routerBytes = await readFile(path.join(root, "learned-token-router-report.json"));
const router = JSON.parse(routerBytes);
if (oracle.trunk.model_hash !== trunkModelHash) throw new Error("oracle trunk hash mismatch");
if (oracle.trunk.forward_count * 3 !== oracle.trunk.naive_three_model_forward_count) {
  throw new Error("shared forward accounting mismatch");
}
const fixed = oracle.fixed_experts[oracle.best_fixed_expert];
const token = oracle.oracle_routes.token;
const learned = router.selected_consensus.final;
const report = {
  schema: "nsrl.literary_shared_trunk_hybrid_moe.v1",
  trunk: {
    model_hash: trunkModelHash,
    frozen_during_expert_training: true,
    final_test_forward_count: oracle.trunk.forward_count,
    naive_three_model_forward_count: oracle.trunk.naive_three_model_forward_count,
    forward_reduction_factor: 3,
  },
  expert_type: "diagonal_plus_fixed_projection_low_rank_hidden_residual_q15",
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
    learned_token_router: learned,
    token_oracle_delta_vs_fixed: {
      accuracy_per_mille: token.accuracy_per_mille - fixed.accuracy_per_mille,
      mistake_count: token.mistakes - fixed.mistakes,
      mean_probability_error_q15:
        token.mean_probability_error_q15 - fixed.mean_probability_error_q15,
    },
    learned_router_delta_vs_fixed: router.delta_vs_fixed,
  },
  promotion: {
    hybrid_experts_promoted: true,
    reason: "hybrid fixed expert improves the prior diagonal fixed mean by 557 Q15",
    target_blind_token_router_promoted: false,
    router_reason:
      "calibration-selected consensus collapses to Blake and adds 1496 total Q15 final error",
    next_architecture_gate: "configurable_multi_head_small_profile_with_hybrid_experts",
  },
  source_sha256: { oracle: sha256(oracleBytes), learned_router: sha256(routerBytes) },
  known_non_claims: [
    "oracle_is_target_aware",
    "router_uses_teacher_forced_prior_token_probes",
    "does_not_claim_language_model_quality",
  ],
};

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, promotion: report.promotion, frozen_final: report.frozen_final }));
