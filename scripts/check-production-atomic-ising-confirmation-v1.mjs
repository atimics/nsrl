#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json";
const resultPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json";
const analyzerPath = new URL("./analyze-production-atomic-ising-confirmation-v1.mjs", import.meta.url);
const structureContractPath =
  "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1-contract.json";
const sourceBytes = fs.readFileSync(sourcePath);
const contractBytes = fs.readFileSync(contractPath);
const resultBytes = fs.readFileSync(resultPath);
const structureContractBytes = fs.readFileSync(structureContractPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const contract = JSON.parse(contractBytes.toString("utf8"));
const result = JSON.parse(resultBytes.toString("utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (value) => value < 0n ? -value : value;
const gcd = (left, right) => {
  let a = absolute(left);
  let b = absolute(right);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};
const rational = (numerator, denominator) => {
  const divisor = gcd(numerator, denominator);
  return {numerator: (numerator / divisor).toString(), denominator: (denominator / divisor).toString()};
};
const compare = (left, right) => {
  const l = BigInt(left.numerator) * BigInt(right.denominator);
  const r = BigInt(right.numerator) * BigInt(left.denominator);
  return l < r ? -1 : l > r ? 1 : 0;
};
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const spin = (character, vertex) => popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const reconstruct = (coefficients) => Array.from({length: 64}, (_, mask) => {
  let total = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    total += coefficients[subset];
    if (subset === 0) return total;
  }
});
const binomial = (n, k) => {
  let total = 1n;
  const choose = Math.min(k, n - k);
  for (let index = 1; index <= choose; index += 1) {
    total = total * BigInt(n - choose + index) / BigInt(index);
  }
  return total;
};
const summarize = (contrasts) => {
  const favorable = contrasts.filter((value) => value < 0n).length;
  const unfavorable = contrasts.filter((value) => value > 0n).length;
  const ties = contrasts.length - favorable - unfavorable;
  const n = favorable + unfavorable;
  const numerator = n === 0 ? 1n : Array.from(
    {length: n - favorable + 1}, (_, index) => favorable + index,
  ).reduce((sum, successes) => sum + binomial(n, successes), 0n);
  return {
    favorable,
    unfavorable,
    ties,
    non_ties: n,
    aggregate: contrasts.reduce((sum, value) => sum + value, 0n).toString(),
    one_sided_exact_p: rational(numerator, n === 0 ? 1n : 1n << BigInt(n)),
  };
};
const same = (left, right, message) => assert(
  JSON.stringify(left) === JSON.stringify(right), message);

assert(contract.schema === "nsrl.production_atomic_ising_confirmation_contract.v1",
  "wrong confirmation contract schema");
assert(result.schema === "nsrl.production_atomic_ising_confirmation.v1",
  "wrong confirmation result schema");
assert(sha256(contractBytes) === result.confirmation_contract_sha256,
  "confirmation contract hash mismatch");
assert(sha256(sourceBytes) === result.source_result_sha256, "source result hash mismatch");
assert(sha256(structureContractBytes) === contract.execution.structure_contract_sha256,
  "structure contract hash mismatch");
assert(sha256(fs.readFileSync(analyzerPath)) === contract.implementation.analyzer_sha256,
  "analyzer hash mismatch");
assert(sha256(fs.readFileSync(new URL(import.meta.url)))
  === contract.implementation.checker_sha256, "checker hash mismatch");
assert(source.analysis_role === "untouched_confirmation"
  && result.analysis_role === "untouched_confirmation", "confirmation role changed");
assert(source.surface.document_start === 136 && source.surface.documents === 64
  && source.surface.hard_stop_before_document === 200, "surface changed");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 64,
  "document accounting changed");
assert(source.bindings.manifest_hash === contract.execution.structure_manifest_hash
  && result.source_structure_manifest_hash === contract.execution.structure_manifest_hash,
"manifest binding changed");
assert(result.limitations.documents_200_212_read === false,
  "sealed documents were marked read");

const readDocuments = (objective) => objective.documents.map((document) => ({
  document: document.document,
  losses: reconstruct(document.coefficients.map(BigInt)),
}));
const q20 = readDocuments(source.q20);
const q32 = readDocuments(source.q32);
assert(q32.every((document, index) => document.document === 136 + index
  && q20[index].document === document.document), "document order changed");
const candidates = contract.candidates;
same(candidates, result.candidates, "candidate contract changed in result");
assert(candidates.pairwise_ising_map_mask === 59
  && candidates.gibbs_magnetization_mask === 61
  && candidates.global_directional_control_mask === 47
  && JSON.stringify(candidates.cluster_candidate_masks) === JSON.stringify([47, 59]),
"frozen candidate masks changed");
const medoids = candidates.cluster_medoid_feature_vectors.map((row) => row.map(BigInt));
const features = (document) => [1, 2, 4, 8, 16, 32].map(
  (mask) => document.losses[mask] - document.losses[0]);
const distance = (left, right) => left.reduce(
  (sum, value, index) => sum + absolute(value - right[index]), 0n);
const routes = q32.map((document) => {
  const vector = features(document);
  return distance(vector, medoids[0]) <= distance(vector, medoids[1]) ? 0 : 1;
});
const contrasts = [
  q32.map((document) => document.losses[59] - document.losses[0]),
  q32.map((document) => document.losses[61] - document.losses[0]),
  q32.map((document, index) =>
    document.losses[candidates.cluster_candidate_masks[routes[index]]]
      - document.losses[47]),
];
const summaries = contrasts.map(summarize);
for (const [index, recorded] of result.primary_endpoints.entries()) {
  const expected = summaries[index];
  for (const key of ["favorable", "unfavorable", "ties", "non_ties", "aggregate"] ) {
    assert(recorded[key] === expected[key], `primary endpoint ${index} ${key} mismatch`);
  }
  same(recorded.one_sided_exact_p, expected.one_sided_exact_p,
    `primary endpoint ${index} exact p mismatch`);
}

const order = summaries.map((summary, index) => ({summary, index})).sort(
  (left, right) => compare(left.summary.one_sided_exact_p, right.summary.one_sided_exact_p)
    || left.index - right.index);
let continuing = true;
for (const [rank, row] of order.entries()) {
  const threshold = rational(1n, 20n * BigInt(3 - rank));
  const passes = continuing && compare(row.summary.one_sided_exact_p, threshold) <= 0;
  if (!passes) continuing = false;
  const recorded = result.primary_endpoints[row.index];
  assert(recorded.holm_rank === rank + 1 && recorded.holm_rejected === passes,
    `Holm decision ${rank} mismatch`);
  same(recorded.holm_threshold, threshold, `Holm threshold ${rank} mismatch`);
}

const character = contract.stable_low_order_rule.character;
const replicate = (documents) => {
  const parameters = documents.map((document) => -document.losses.reduce(
    (sum, loss, vertex) => sum + loss * spin(character, vertex), 0n));
  return {
    negative_documents: parameters.filter((value) => value < 0n).length,
    zero_documents: parameters.filter((value) => value === 0n).length,
    positive_documents: parameters.filter((value) => value > 0n).length,
    aggregate_numerator: parameters.reduce((sum, value) => sum + value, 0n).toString(),
  };
};
same(replicate(q20), result.descriptive.stable_field_q20,
  "Q20 stable-field replication mismatch");
same(replicate(q32), result.descriptive.stable_field_q32,
  "Q32 stable-field replication mismatch");
same(routes.reduce((counts, route) => {
  counts[route] += 1;
  return counts;
}, [0, 0]), result.descriptive.cluster_route_counts, "route counts changed");
assert(result.decision.optimizer_change_authorized === false
  && result.decision.paid_scaling_authorized === false,
"confirmation result authorized promotion");

const replayDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-document-ising-replay-"));
const replayPath = path.join(replayDirectory, "replay.json");
try {
  const replay = spawnSync(
    process.execPath,
    [fileURLToPath(analyzerPath), sourcePath, contractPath, replayPath],
    {encoding: "utf8"},
  );
  assert(replay.status === 0, `confirmation replay failed: ${replay.stderr || replay.stdout}`);
  assert(fs.readFileSync(replayPath).equals(resultBytes),
    "confirmation result is not byte-replayable");
} finally {
  fs.rmSync(replayDirectory, {recursive: true, force: true});
}

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_confirmation_check.v1",
  contract_sha256: sha256(contractBytes),
  result_sha256: sha256(resultBytes),
  primary_endpoint_summaries: summaries,
  route_counts: result.descriptive.cluster_route_counts,
  exact_holm_verified: true,
  byte_replay_verified: true,
  documents_200_212_read: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
