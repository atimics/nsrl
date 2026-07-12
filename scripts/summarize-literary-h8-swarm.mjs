#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-h8-swarm-v1";
const out = process.argv[3] ?? path.join(root, "report.json");
const sha256 = (value) => createHash("sha256").update(value).digest("hex");
const authors = ["crowley", "shakespeare", "blake"];

async function loadJson(relative) {
  const bytes = await readFile(path.join(root, relative));
  return { value: JSON.parse(bytes), sha256: sha256(bytes) };
}

const oracle = await loadJson("optimizer-swarm-oracle/final-report.json");
const token = await loadJson("learned-router/learned-token-router-report.json");
const span = await loadJson("learned-router/learned-span-router-report.json");
const fixed = oracle.value.fixed_experts[oracle.value.best_fixed_expert];
const matrices = { author_512: {}, author_2048: {}, mixed_offsets: {} };

for (const [key, directory] of [
  ["author_512", "leaves"],
  ["author_2048", "leaves-2k"],
]) {
  for (const model of authors) {
    matrices[key][model] = {};
    for (const testAuthor of authors) {
      matrices[key][model][testAuthor] = (
        await loadJson(`${directory}/${model}/eval-${testAuthor}.json`)
      ).value.evaluation;
    }
  }
}
for (const offset of [0, 512, 1024]) {
  matrices.mixed_offsets[offset] = {};
  for (const test of ["mixed", ...authors]) {
    matrices.mixed_offsets[offset][test] = (
      await loadJson(`mixed-leaves/offset${offset}/eval-${test}.json`)
    ).value.evaluation;
  }
}

const report = {
  schema: "nsrl.literary_h8_optimizer_swarm.v1",
  architecture: {
    profile: "small-h8-d128-ff256",
    seq_len: oracle.value.models.seq_len,
    model_ids: oracle.value.models.ids,
    model_hashes: oracle.value.models.hashes,
    active_transformer_forwards_per_target: 3,
  },
  frozen_final: {
    samples: oracle.value.dataset.samples,
    windows: oracle.value.dataset.windows,
    fixed,
    prompt_oracle: oracle.value.oracle_routes.prompt,
    span_oracle: oracle.value.oracle_routes.span,
    token_oracle: oracle.value.oracle_routes.token,
    learned_span: span.value.selected_consensus.final,
    learned_token: token.value.selected_consensus.final,
    learned_span_delta_vs_fixed: span.value.delta_vs_fixed,
    learned_token_delta_vs_fixed: token.value.delta_vs_fixed,
  },
  failed_diversity_controls: {
    author_isolated_512_windows: matrices.author_512,
    author_isolated_2048_windows: matrices.author_2048,
    disjoint_mixed_offsets_512_windows: matrices.mixed_offsets,
    conclusion:
      "isolated authors and later contiguous offsets underfit; optimizer-scale diversity over the balanced mixed corpus is promoted",
  },
  promotion: {
    h8_optimizer_swarm_promoted: true,
    learned_token_router_promoted: true,
    learned_span_router_promoted: true,
    preferred_quality_policy: "token",
    preferred_low_switch_policy: "span",
    next_step: "distill H8 optimizer diversity into shared-trunk residual experts",
  },
  source_sha256: {
    oracle: oracle.sha256,
    learned_token: token.sha256,
    learned_span: span.sha256,
  },
  known_non_claims: [
    "current_H8_swarm_runs_three_whole_transformers",
    "routers_use_teacher_forced_prior_token_probes",
    "does_not_claim_language_model_quality",
  ],
};

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, promotion: report.promotion, frozen_final: report.frozen_final }));
