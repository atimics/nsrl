#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const resultPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json";
const analyzerPath = new URL("./analyze-production-document-ising-v1.mjs", import.meta.url);
const sourceBytes = fs.readFileSync(sourcePath);
const resultBytes = fs.readFileSync(resultPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const result = JSON.parse(resultBytes.toString("utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (value) => value < 0n ? -value : value;
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const sign = (value) => value < 0n ? -1 : value > 0n ? 1 : 0;
const spin = (character, vertex) => popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const minimum = (values) => values.reduce((left, right) => left < right ? left : right);
const roundNearest = (numerator, denominator) => numerator < 0n
  ? -roundNearest(-numerator, denominator)
  : (numerator + denominator / 2n) / denominator;
const reconstruct = (coefficients) => Array.from({length: 64}, (_, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
});
const walsh = (losses) => Array.from({length: 64}, (_, character) => losses.reduce(
  (sum, loss, vertex) => sum + loss * spin(character, vertex), 0n));
const documents = (objective) => objective.documents.map((document) => ({
  document: document.document,
  losses: reconstruct(document.coefficients.map(BigInt)),
}));
const contrastSummary = (values) => ({
  favorable: values.filter((value) => value < 0n).length,
  unfavorable: values.filter((value) => value > 0n).length,
  ties: values.filter((value) => value === 0n).length,
  aggregate: values.reduce((sum, value) => sum + value, 0n).toString(),
});
const equal = (left, right, message) => assert(
  JSON.stringify(left) === JSON.stringify(right), message);

assert(source.schema === "nsrl.production_atomic_structure.v1"
  && source.analysis_role === "proposal_only_calibration", "wrong proposal source");
assert(result.schema === "nsrl.production_atomic_ising_proposal.v1"
  && result.analysis_role === "proposal_only_calibration", "wrong document Ising result");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 0,
  "proposal firewall crossed");
assert(result.source_result_sha256 === sha256(sourceBytes), "source hash mismatch");
assert(result.analyzer_sha256 === sha256(fs.readFileSync(analyzerPath)),
  "analyzer hash mismatch");
assert(result.rank === 6 && result.vertices === 64 && result.documents === 64,
  "wrong Ising population shape");

const q20 = documents(source.q20);
const q32 = documents(source.q32);
assert(q20.every((document, index) => document.document === 8 + index
  && q32[index].document === document.document), "document ordering changed");
const transforms = (rows) => rows.map((document) => walsh(document.losses));
const q20Walsh = transforms(q20);
const q32Walsh = transforms(q32);
const aggregate = (rows) => Array.from({length: 64}, (_, character) => rows.reduce(
  (sum, transform) => sum + transform[character], 0n));

const stableCharacters = [];
for (let character = 1; character < 64; character += 1) {
  if (popcount(character) > 2) continue;
  const summarizeParameter = (rows) => {
    const values = rows.map((transform) => -transform[character]);
    return {
      negative: values.filter((value) => value < 0n).length,
      zero: values.filter((value) => value === 0n).length,
      positive: values.filter((value) => value > 0n).length,
      sum: values.reduce((sum, value) => sum + value, 0n),
    };
  };
  const coarse = summarizeParameter(q20Walsh);
  const fine = summarizeParameter(q32Walsh);
  const visible = fine.negative + fine.positive;
  const majority = Math.max(fine.negative, fine.positive);
  if (visible >= 32 && majority * 4 >= visible * 3
    && sign(coarse.sum) !== 0 && sign(coarse.sum) === sign(fine.sum)) {
    stableCharacters.push(character);
  }
}
equal(stableCharacters, result.stable_low_order_characters,
  "stable low-order Ising characters changed");

const pairwiseMap = (rows) => {
  const coefficients = aggregate(rows);
  const values = Array.from({length: 64}, (_, vertex) => coefficients.reduce(
    (sum, coefficient, character) => popcount(character) <= 2
      ? sum + coefficient * spin(character, vertex) : sum, 0n));
  const best = minimum(values);
  const minimizers = values.flatMap((value, mask) => value === best ? [mask] : []);
  return {minimizers, selected: minimizers[0]};
};
const q20Pairwise = pairwiseMap(q20Walsh);
const q32Pairwise = pairwiseMap(q32Walsh);
equal(q20Pairwise.minimizers, result.pairwise_ising_map.q20.minimizers,
  "Q20 pairwise minimizers changed");
equal(q32Pairwise.minimizers, result.pairwise_ising_map.q32.minimizers,
  "Q32 pairwise minimizers changed");
assert(q20Pairwise.selected === result.pairwise_ising_map.q20.selected,
  "Q20 pairwise MAP changed");
assert(q32Pairwise.selected === result.pairwise_ising_map.q32.selected,
  "Q32 pairwise MAP changed");

const gibbsMask = (numerator, denominator) => {
  const documentMoments = q20.map((document) => {
    const best = minimum(document.losses);
    const gaps = document.losses.map((loss) => Number(loss - best));
    const largest = Math.max(...gaps);
    const weights = gaps.map((gap) => numerator ** BigInt(gap)
      * denominator ** BigInt(largest - gap));
    const partition = weights.reduce((sum, value) => sum + value, 0n);
    return Array.from({length: 6}, (_, atom) => roundNearest(
      weights.reduce((sum, weight, vertex) =>
        sum + weight * spin(1 << atom, vertex), 0n) * (1n << 30n),
      partition,
    ));
  });
  const means = Array.from({length: 6}, (_, atom) => roundNearest(
    documentMoments.reduce((sum, values) => sum + values[atom], 0n), 64n));
  return means.reduce(
    (mask, value, atom) => value < 0n ? mask | (1 << atom) : mask, 0);
};
for (const [numerator, denominator, label] of [[1n, 4n, "1/4"], [1n, 2n, "1/2"], [3n, 4n, "3/4"]]) {
  const recorded = result.gibbs.temperature_grid.find((row) => row.fugacity === label);
  assert(recorded?.selected_mask === gibbsMask(numerator, denominator),
    `Gibbs mask changed at ${label}`);
}
assert(result.gibbs.selected_mask === gibbsMask(1n, 2n), "central Gibbs mask changed");

const singletonFeatures = (document) => [1, 2, 4, 8, 16, 32].map(
  (mask) => document.losses[mask] - document.losses[0]);
const l1 = (left, right) => left.reduce(
  (sum, value, index) => sum + absolute(value - right[index]), 0n);
const chooseCandidate = (rows) => {
  let best;
  for (let mask = 1; mask < 64; mask += 1) {
    if (popcount(mask) < 2) continue;
    const contrasts = rows.map((document) => document.losses[mask] - document.losses[0]);
    const summary = contrastSummary(contrasts);
    const candidate = {...summary, aggregateValue: BigInt(summary.aggregate), mask};
    if (best === undefined
      || candidate.aggregateValue < best.aggregateValue
      || (candidate.aggregateValue === best.aggregateValue
        && (candidate.favorable > best.favorable
          || (candidate.favorable === best.favorable
            && (candidate.unfavorable < best.unfavorable
              || (candidate.unfavorable === best.unfavorable
                && (popcount(candidate.mask) < popcount(best.mask)
                  || (popcount(candidate.mask) === popcount(best.mask)
                    && candidate.mask < best.mask)))))))) best = candidate;
  }
  return best.mask;
};
const fit = (rows) => {
  const records = rows.map((document) => ({document, features: singletonFeatures(document)}));
  let medoidIndices = [0, 1];
  let farthest = -1n;
  for (let left = 0; left < records.length; left += 1) {
    for (let right = left + 1; right < records.length; right += 1) {
      const distance = l1(records[left].features, records[right].features);
      if (distance > farthest) {
        farthest = distance;
        medoidIndices = [left, right];
      }
    }
  }
  let medoids = medoidIndices.map((index) => records[index]);
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const clusters = [[], []];
    for (const record of records) {
      clusters[l1(record.features, medoids[0].features)
        <= l1(record.features, medoids[1].features) ? 0 : 1].push(record);
    }
    const next = clusters.map((cluster) => cluster.reduce((best, candidate) => {
      const total = (record) => cluster.reduce(
        (sum, other) => sum + l1(record.features, other.features), 0n);
      const candidateTotal = total(candidate);
      const bestTotal = total(best);
      return candidateTotal < bestTotal || (candidateTotal === bestTotal
        && candidate.document.document < best.document.document) ? candidate : best;
    }, cluster[0]));
    if (next.every((medoid, index) =>
      medoid.document.document === medoids[index].document.document)) {
      return {
        medoids,
        clusters,
        candidates: clusters.map((cluster) =>
          chooseCandidate(cluster.map((record) => record.document))),
      };
    }
    medoids = next;
  }
  throw new Error("independent two-medoid replay did not converge");
};
const cluster = fit(q32);
equal(cluster.medoids.map((record) => record.document.document),
  result.clustering.medoids.map((record) => record.document), "cluster medoids changed");
equal(cluster.medoids.map((record) => record.features.map(String)),
  result.frozen_confirmation_candidates.cluster_medoid_feature_vectors,
  "cluster medoid features changed");
equal(cluster.candidates, result.frozen_confirmation_candidates.cluster_candidate_masks,
  "cluster candidates changed");
assert(result.frozen_confirmation_candidates.pairwise_ising_map_mask === 59
  && result.frozen_confirmation_candidates.gibbs_magnetization_mask === 61
  && result.frozen_confirmation_candidates.global_directional_control_mask === 47,
"frozen confirmation candidates changed");
assert(result.confirmation_design.document_start === 136
  && result.confirmation_design.documents === 64
  && result.confirmation_design.hard_stop_before_document === 200,
"confirmation surface changed");
assert(result.decision.optimizer_change_authorized === false
  && result.decision.paid_scaling_authorized === false,
"proposal analysis authorized promotion");

const replayDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-document-ising-proposal-"));
const replayPath = path.join(replayDirectory, "replay.json");
try {
  const replay = spawnSync(
    process.execPath,
    [fileURLToPath(analyzerPath), sourcePath, replayPath],
    {encoding: "utf8"},
  );
  assert(replay.status === 0, `proposal replay failed: ${replay.stderr || replay.stdout}`);
  assert(fs.readFileSync(replayPath).equals(resultBytes),
    "document-Ising proposal result is not byte-replayable");
} finally {
  fs.rmSync(replayDirectory, {recursive: true, force: true});
}

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_proposal_independent_check.v1",
  stable_low_order_characters: stableCharacters,
  pairwise_ising_map_mask: result.pairwise_ising_map.q32.selected,
  gibbs_magnetization_mask: result.gibbs.selected_mask,
  cluster_medoid_documents: cluster.medoids.map((record) => record.document.document),
  cluster_candidate_masks: cluster.candidates,
  proposal_only_firewall_verified: true,
  untouched_confirmation_surface_frozen: true,
  byte_replay_verified: true,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
