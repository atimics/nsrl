#!/usr/bin/env node

import fs from "node:fs";

const contractPath = process.argv[2];
const resultPath = process.argv[3];
if (!contractPath || !resultPath) {
  throw new Error("usage: check-production-multisource-atomic-structure-v1.mjs CONTRACT RESULT");
}
const contract = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const result = JSON.parse(fs.readFileSync(resultPath, "utf8"));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (value) => value < 0n ? -value : value;
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const mobius = (values) => {
  const coefficients = [...values];
  for (let bit = 0; bit < 6; bit += 1) {
    for (let mask = 0; mask < 64; mask += 1) {
      if ((mask & (1 << bit)) !== 0) coefficients[mask] -= coefficients[mask ^ (1 << bit)];
    }
  }
  return coefficients;
};
const reconstruct = (coefficients, mask) => {
  let sum = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    sum += coefficients[subset];
    if (subset === 0) return sum;
  }
};

assert(contract.schema === "nsrl.production_atomic_structure_contract.v1"
  && result.schema === "nsrl.production_atomic_structure.v1", "wrong atomic structure schema");
assert(contract.analysis_role === "proposal_only_calibration"
  && result.analysis_role === contract.analysis_role, "wrong multisource analysis role");
assert(contract.surface.document_start === 8 && contract.surface.documents === 64
  && contract.surface.windows_per_document === 2
  && contract.surface.hard_stop_before_document === 72
  && result.surface.hard_stop_before_document === 72,
"multisource atomic surface changed");
assert(result.vertices_evaluated === 64 && result.transfer_documents_read === 0
  && result.reserved_documents_read === 0, "multisource document accounting changed");
for (const key of ["source_fnv64", "binary_fnv64", "model_hash", "tokenizer_hash",
  "token_stream_hash", "source_index_hash"]) {
  assert(result.bindings[key] === contract.bindings[key], `${key} binding mismatch`);
}
assert(result.bindings.move_fingerprint === contract.move_fingerprint
  && result.bindings.manifest_hash === contract.manifest_hash,
"move or manifest binding mismatch");
assert(result.source_population.proposal_source_cluster_hash
  === contract.source_population.proposal_source_cluster_hash
  && result.source_population.proposal_source_clusters
    === contract.source_population.proposal_source_clusters,
"multisource population binding mismatch");
assert(contract.source_population.proposal_source_clusters === 64
  && contract.source_population.source_clustered_fold_estimation_available === true
  && result.source_population.source_clustered_fold_estimation_available === true,
"expected 64 bound source clusters with clustered folds available");

const checkObjective = (trace, fractionalBits) => {
  assert(trace.fractional_bits === fractionalBits
    && trace.vertex_losses.length === 64 && trace.coefficients.length === 64
    && trace.documents.length === 64, `Q${fractionalBits} cube shape changed`);
  const losses = trace.vertex_losses.map(BigInt);
  const coefficients = trace.coefficients.map(BigInt);
  const recomputed = mobius(losses);
  assert(recomputed.every((value, mask) => value === coefficients[mask]),
    `Q${fractionalBits} aggregate Möbius mismatch`);
  assert(losses.every((value, mask) => reconstruct(coefficients, mask) === value),
    `Q${fractionalBits} aggregate reconstruction failed`);
  const sums = Array(64).fill(0n);
  const environmentalMass = Array(7).fill(0n);
  for (const [offset, document] of trace.documents.entries()) {
    assert(document.document === 8 + offset && document.coefficients.length === 64,
      `Q${fractionalBits} document order or shape changed`);
    document.coefficients.map(BigInt).forEach((value, mask) => {
      sums[mask] += value;
      environmentalMass[popcount(mask)] += absolute(value);
    });
  }
  assert(sums.every((value, mask) => value === coefficients[mask]),
    `Q${fractionalBits} environment aggregation mismatch`);
  const populationMass = Array(7).fill(0n);
  coefficients.forEach((value, mask) => populationMass[popcount(mask)] += absolute(value));
  assert(trace.absolute_mass_by_order.population.map(BigInt).every(
    (value, order) => value === populationMass[order])
    && trace.absolute_mass_by_order.environmental.map(BigInt).every(
      (value, order) => value === environmentalMass[order]),
  `Q${fractionalBits} mass summary changed`);
  for (const tail of trace.tails) {
    const population = populationMass.slice(tail.retained_order + 1).reduce((a, b) => a + b, 0n);
    const environmental = environmentalMass.slice(tail.retained_order + 1).reduce((a, b) => a + b, 0n);
    assert(BigInt(tail.population_absolute_tail) === population
      && BigInt(tail.environmental_absolute_tail) === environmental
      && BigInt(tail.cancellation_mass) === environmental - population
      && BigInt(tail.tail_regret_bound) === population
      && BigInt(tail.exact_gap) <= population && tail.certificate_verified === true,
    `Q${fractionalBits} tail certificate changed`);
  }
  assert(trace.exchange_slices.length === 5
    && trace.exchange_slices.every((slice) => slice.certificate_verified === true)
    && trace.interaction_width.elimination_orders_evaluated === 720
    && trace.interaction_width.width_histogram.reduce((sum, value) => sum + value, 0) === 720,
  `Q${fractionalBits} exchange or width certificate changed`);
};
checkObjective(result.q20, 20);
checkObjective(result.q32, 32);
assert(result.boundary_taxonomy.by_atom.length === 6
  && Object.values(result.boundary_taxonomy.all_edges).reduce((sum, value) => sum + value, 0)
    === 6 * 32 * 64,
"boundary taxonomy changed");
assert(result.decision.structure_certificate_selected === false
  && result.decision.optimizer_change_authorized === false
  && result.decision.paid_scaling_authorized === false,
"multisource calibration audit escaped its authorization boundary");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_multisource_atomic_structure_check.v1",
  vertices: 64, documents: 64, source_clusters: 64,
  source_clustered_fold_estimation_available: true,
  q20_q32_mobius_reconstruction_verified: true,
  tail_exchange_and_width_certificates_verified: true,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
