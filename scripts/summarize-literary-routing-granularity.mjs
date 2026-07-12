#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.argv[2] ?? "data/experiments/literary-token-routing-v1";
const out = process.argv[3] ?? path.join(root, "overall-report.json");
const authors = ["crowley", "shakespeare", "blake"];
const reports = [];
const sourceHashes = {};

for (const author of authors) {
  const reportPath = path.join(root, `${author}-report.json`);
  const bytes = await readFile(reportPath);
  sourceHashes[author] = createHash("sha256").update(bytes).digest("hex");
  reports.push(JSON.parse(bytes));
}

const sum = (pick) => reports.reduce((total, report) => total + pick(report), 0);
const windows = sum((report) => report.dataset.windows);
const modelIds = reports[0].models.ids;
const modelHashes = reports[0].models.hashes;

function combineRoute(key) {
  const mistakes = sum((report) => report.oracle_routes[key].mistakes);
  const probabilityError = sum(
    (report) => report.oracle_routes[key].probability_error_q15,
  );
  const utilizationTokens = modelIds.map((_, index) =>
    sum((report) => report.oracle_routes[key].utilization_tokens[index]),
  );
  return {
    windows,
    mistakes,
    accuracy_per_mille: Math.floor(((windows - mistakes) * 1000) / windows),
    probability_error_q15: probabilityError,
    mean_probability_error_q15: Math.floor(probabilityError / windows),
    invalid_forward_count: sum(
      (report) => report.oracle_routes[key].invalid_forward_count,
    ),
    decisions: sum((report) => report.oracle_routes[key].decisions),
    route_switches: sum((report) => report.oracle_routes[key].route_switches),
    utilization_tokens: utilizationTokens,
    utilization_per_mille: utilizationTokens.map((count) =>
      Math.floor((count * 1000) / windows),
    ),
  };
}

const fixedExperts = modelIds.map((_, index) => {
  const mistakes = sum((report) => report.fixed_experts[index].mistakes);
  const probabilityError = sum(
    (report) => report.fixed_experts[index].probability_error_q15,
  );
  return {
    expert_index: index,
    expert_id: modelIds[index],
    windows,
    mistakes,
    accuracy_per_mille: Math.floor(((windows - mistakes) * 1000) / windows),
    probability_error_q15: probabilityError,
    mean_probability_error_q15: Math.floor(probabilityError / windows),
    invalid_forward_count: sum(
      (report) => report.fixed_experts[index].invalid_forward_count,
    ),
  };
});
const bestFixedExpert = fixedExperts.reduce((best, candidate) => {
  if (candidate.probability_error_q15 !== best.probability_error_q15) {
    return candidate.probability_error_q15 < best.probability_error_q15
      ? candidate
      : best;
  }
  if (candidate.mistakes !== best.mistakes) {
    return candidate.mistakes < best.mistakes ? candidate : best;
  }
  return candidate.expert_index < best.expert_index ? candidate : best;
});
const prompt = combineRoute("prompt");
const span = combineRoute("span");
const token = combineRoute("token");

const report = {
  schema: "nsrl.literary_routing_granularity_summary.v1",
  source_report_sha256: sourceHashes,
  authors,
  samples: sum((item) => item.dataset.samples),
  windows,
  span_len: 16,
  stride: 1,
  models: { ids: modelIds, hashes: modelHashes, seq_len: reports[0].models.seq_len },
  fixed_experts: fixedExperts,
  best_fixed_expert: bestFixedExpert.expert_index,
  oracle_routes: { prompt, span, token },
  deltas_vs_prompt: {
    span_accuracy_per_mille: span.accuracy_per_mille - prompt.accuracy_per_mille,
    span_mean_probability_error_q15:
      span.mean_probability_error_q15 - prompt.mean_probability_error_q15,
    token_accuracy_per_mille: token.accuracy_per_mille - prompt.accuracy_per_mille,
    token_mean_probability_error_q15:
      token.mean_probability_error_q15 - prompt.mean_probability_error_q15,
  },
  known_non_claims: [
    "target-aware_oracle_ceiling_not_deployable_router",
    "whole_model_experts_not_shared_attention_moe",
    "does_not_claim_language_model_quality",
  ],
};

await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ out, samples: report.samples, windows, deltas: report.deltas_vs_prompt }));
