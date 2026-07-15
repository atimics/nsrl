#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const contractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1-contract.json";
const outputPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json";
const sourceBytes = fs.readFileSync(sourcePath);
const contractBytes = fs.readFileSync(contractPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const contract = JSON.parse(contractBytes.toString("utf8"));
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
  assert(denominator > 0n, "positive rational denominator required");
  const divisor = gcd(numerator, denominator);
  return {numerator: (numerator / divisor).toString(), denominator: (denominator / divisor).toString()};
};
const compareRational = (left, right) => BigInt(left.numerator) * BigInt(right.denominator)
  < BigInt(right.numerator) * BigInt(left.denominator) ? -1
  : BigInt(left.numerator) * BigInt(right.denominator)
    > BigInt(right.numerator) * BigInt(left.denominator) ? 1 : 0;
const multiplyRational = (value, factor) => rational(
  BigInt(value.numerator) * BigInt(factor), BigInt(value.denominator));
const capOne = (value) => compareRational(value, {numerator: "1", denominator: "1"}) > 0
  ? {numerator: "1", denominator: "1"} : value;
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
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
const binomial = (n, k) => {
  let value = 1n;
  for (let index = 1; index <= k; index += 1) {
    value = value * BigInt(n - k + index) / BigInt(index);
  }
  return value;
};
const signTest = (contrasts) => {
  const favorable = contrasts.filter((value) => value < 0n).length;
  const unfavorable = contrasts.filter((value) => value > 0n).length;
  const ties = contrasts.length - favorable - unfavorable;
  const nonTies = favorable + unfavorable;
  const numerator = nonTies === 0 ? 1n : Array.from(
    {length: nonTies - favorable + 1}, (_, index) => favorable + index,
  ).reduce((sum, successes) => sum + binomial(nonTies, successes), 0n);
  const denominator = nonTies === 0 ? 1n : 1n << BigInt(nonTies);
  return {
    favorable,
    unfavorable,
    ties,
    non_ties: nonTies,
    aggregate: contrasts.reduce((sum, value) => sum + value, 0n).toString(),
    one_sided_exact_p: rational(numerator, denominator),
  };
};
const holm = (endpoints) => {
  const ordered = endpoints.map((endpoint, index) => ({endpoint, index})).sort(
    (left, right) => compareRational(
      left.endpoint.one_sided_exact_p, right.endpoint.one_sided_exact_p,
    ) || left.index - right.index);
  let continuing = true;
  let runningAdjusted = {numerator: "0", denominator: "1"};
  for (const [rank, row] of ordered.entries()) {
    const remaining = ordered.length - rank;
    const threshold = rational(1n, 20n * BigInt(remaining));
    const passes = continuing
      && compareRational(row.endpoint.one_sided_exact_p, threshold) <= 0;
    if (!passes) continuing = false;
    const scaled = capOne(multiplyRational(row.endpoint.one_sided_exact_p, remaining));
    if (compareRational(scaled, runningAdjusted) > 0) runningAdjusted = scaled;
    row.endpoint.holm_rank = rank + 1;
    row.endpoint.holm_threshold = threshold;
    row.endpoint.holm_adjusted_p = runningAdjusted;
    row.endpoint.holm_rejected = passes;
  }
  return endpoints;
};

assert(contract.schema === "nsrl.production_atomic_ising_confirmation_contract.v1",
  "wrong confirmation contract schema");
assert(source.schema === "nsrl.production_atomic_structure.v1"
  && source.analysis_role === "untouched_confirmation", "wrong confirmation source");
assert(source.surface.document_start === 136 && source.surface.documents === 64
  && source.surface.windows_per_document === 2
  && source.surface.hard_stop_before_document === 200, "confirmation surface changed");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 64,
  "confirmation document accounting changed");
assert(source.bindings.manifest_hash === contract.execution.structure_manifest_hash,
  "confirmation manifest mismatch");
assert(contract.surface.still_sealed_documents === "200--212",
  "sealed document range changed");

const objectiveDocuments = (objective) => objective.documents.map((document) => ({
  document: document.document,
  losses: reconstruct(document.coefficients.map(BigInt)),
}));
const q20 = objectiveDocuments(source.q20);
const q32 = objectiveDocuments(source.q32);
assert(q20.length === 64 && q32.length === 64 && q20.every(
  (document, index) => document.document === 136 + index
    && q32[index].document === document.document), "confirmation document order changed");

const medoids = contract.candidates.cluster_medoid_feature_vectors.map(
  (values) => values.map(BigInt));
const singletonFeatures = (document) => [1, 2, 4, 8, 16, 32].map(
  (mask) => document.losses[mask] - document.losses[0]);
const l1 = (left, right) => left.reduce(
  (sum, value, index) => sum + absolute(value - right[index]), 0n);
const route = (document) => {
  const features = singletonFeatures(document);
  return l1(features, medoids[0]) <= l1(features, medoids[1]) ? 0 : 1;
};
const routes = q32.map(route);
const masks = contract.candidates;
const contrastsFor = (documents, maskForDocument, controlForDocument) => documents.map(
  (document, index) => document.losses[maskForDocument(index)]
    - document.losses[controlForDocument(index)]);
const endpointSpecifications = [
  {
    id: "pairwise_ising_map_vs_baseline_q32_document_direction",
    contrasts: contrastsFor(q32, () => masks.pairwise_ising_map_mask, () => 0),
  },
  {
    id: "gibbs_magnetization_vs_baseline_q32_document_direction",
    contrasts: contrastsFor(q32, () => masks.gibbs_magnetization_mask, () => 0),
  },
  {
    id: "cluster_routed_vs_global_directional_q32_document_direction",
    contrasts: contrastsFor(
      q32,
      (index) => masks.cluster_candidate_masks[routes[index]],
      () => masks.global_directional_control_mask,
    ),
  },
];
const primaryEndpoints = holm(endpointSpecifications.map((specification) => ({
  id: specification.id,
  ...signTest(specification.contrasts),
})));

const fugacityMoment = (numerator, denominator) => {
  const q30 = 1n << 30n;
  const documentMoments = q20.map((document) => {
    const best = minimum(document.losses);
    const gaps = document.losses.map((loss) => Number(loss - best));
    assert(gaps.every((gap) => Number.isSafeInteger(gap) && gap >= 0),
      "confirmation Gibbs gap is not a safe nonnegative integer");
    const largest = Math.max(...gaps);
    const weights = gaps.map((gap) => numerator ** BigInt(gap)
      * denominator ** BigInt(largest - gap));
    const partition = weights.reduce((sum, value) => sum + value, 0n);
    return Array.from({length: 6}, (_, atom) => roundNearest(
      weights.reduce((sum, weight, vertex) =>
        sum + weight * spin(1 << atom, vertex), 0n) * q30,
      partition,
    ));
  });
  const mean = Array.from({length: 6}, (_, atom) => roundNearest(
    documentMoments.reduce((sum, values) => sum + values[atom], 0n), 64n));
  return {
    fugacity: `${numerator}/${denominator}`,
    mean_spin_magnetization_q30: mean.map(String),
    selected_mask: mean.reduce(
      (mask, value, atom) => value < 0n ? mask | (1 << atom) : mask, 0),
  };
};
const stableCharacter = contract.stable_low_order_rule.character;
const parameterReplication = (documents) => {
  const values = documents.map((document) => -walsh(document.losses)[stableCharacter]);
  return {
    negative_documents: values.filter((value) => value < 0n).length,
    zero_documents: values.filter((value) => value === 0n).length,
    positive_documents: values.filter((value) => value > 0n).length,
    aggregate_numerator: values.reduce((sum, value) => sum + value, 0n).toString(),
  };
};
const q20Replication = parameterReplication(q20);
const q32Replication = parameterReplication(q32);
const stableFieldReplicated = q32Replication.negative_documents
  + q32Replication.positive_documents >= 32
  && Math.max(q32Replication.negative_documents, q32Replication.positive_documents) * 4
    >= (q32Replication.negative_documents + q32Replication.positive_documents) * 3
  && BigInt(q20Replication.aggregate_numerator) < 0n
  && BigInt(q32Replication.aggregate_numerator) < 0n;
const descriptive = {
  q20: {
    pairwise_vs_baseline: signTest(contrastsFor(
      q20, () => masks.pairwise_ising_map_mask, () => 0)),
    gibbs_vs_baseline: signTest(contrastsFor(
      q20, () => masks.gibbs_magnetization_mask, () => 0)),
    cluster_routed_vs_global_directional: signTest(contrastsFor(
      q20,
      (index) => masks.cluster_candidate_masks[routes[index]],
      () => masks.global_directional_control_mask,
    )),
  },
  cluster_route_counts: routes.map((cluster) => cluster).reduce(
    (counts, cluster) => {
      counts[cluster] += 1;
      return counts;
    }, [0, 0]),
  cluster_routed_vs_baseline_q32: signTest(contrastsFor(
    q32, (index) => masks.cluster_candidate_masks[routes[index]], () => 0)),
  stable_field_character: stableCharacter,
  stable_field_q20: q20Replication,
  stable_field_q32: q32Replication,
  stable_field_replicated: stableFieldReplicated,
  confirmation_quenched_gibbs: [[1n, 4n], [1n, 2n], [3n, 4n]].map(
    ([numerator, denominator]) => fugacityMoment(numerator, denominator)),
};
const mechanismSupport = Object.fromEntries(primaryEndpoints.map((endpoint) => [
  endpoint.id,
  endpoint.holm_rejected && BigInt(endpoint.aggregate) < 0n,
]));
const result = {
  schema: "nsrl.production_atomic_ising_confirmation.v1",
  analysis_role: "untouched_confirmation",
  confirmation_contract_sha256: sha256(contractBytes),
  source_result_sha256: sha256(sourceBytes),
  source_structure_manifest_hash: source.bindings.manifest_hash,
  implementation: contract.implementation,
  surface: contract.surface,
  candidates: contract.candidates,
  primary_endpoints: primaryEndpoints,
  descriptive,
  mechanism_support: mechanismSupport,
  limitations: {
    source_clusters: source.source_population.proposal_source_clusters,
    same_source_document_evidence_only: true,
    cross_source_generalization_identified: false,
    documents_200_212_read: false,
  },
  decision: {
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    all_three_mechanisms_supported: Object.values(mechanismSupport).every(Boolean),
  },
};
const bytes = `${JSON.stringify(result, null, 2)}\n`;
fs.mkdirSync(path.dirname(outputPath), {recursive: true});
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, bytes);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_confirmation_run.v1",
  result_sha256: sha256(Buffer.from(bytes)),
  primary_endpoints: primaryEndpoints,
  mechanism_support: mechanismSupport,
  documents_200_212_read: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
