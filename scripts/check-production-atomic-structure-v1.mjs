#!/usr/bin/env node

import fs from "node:fs";

const contractPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json";
const resultPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
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

assert(contract.schema === "nsrl.production_atomic_structure_contract.v1", "wrong contract schema");
assert(result.schema === "nsrl.production_atomic_structure.v1", "wrong result schema");
assert(contract.analysis_role === result.analysis_role, "analysis-role mismatch");
assert(["proposal_only_calibration", "untouched_confirmation"].includes(result.analysis_role),
  "unsupported atomic-structure role");
const confirmation = result.analysis_role === "untouched_confirmation";
const expectedDocumentStart = confirmation ? 136 : 8;
const expectedHardStop = confirmation ? 200 : 72;
assert(result.vertices_evaluated === 64, "rank-six cube did not evaluate 64 vertices");
assert(result.transfer_documents_read === 0
  && result.reserved_documents_read === (confirmation ? 64 : 0),
"document-read accounting changed");
assert(contract.surface.document_start === expectedDocumentStart
  && contract.surface.documents === 64, "contract surface changed");
assert(contract.surface.hard_stop_before_document === expectedHardStop
  && result.surface.hard_stop_before_document === expectedHardStop,
"contract hard stop changed");
for (const key of ["source_fnv64", "binary_fnv64", "model_hash", "tokenizer_hash", "token_stream_hash", "source_index_hash"]) {
  assert(result.bindings[key] === contract.bindings[key], `${key} binding mismatch`);
}
assert(result.bindings.move_fingerprint === contract.move_fingerprint, "move fingerprint mismatch");
assert(result.bindings.manifest_hash === contract.manifest_hash, "manifest mismatch");
assert(contract.moves.length === 6 && result.moves.length === 6, "expected six atoms");
assert(result.source_population.proposal_source_cluster_hash
  === contract.source_population.proposal_source_cluster_hash,
"proposal source-cluster binding mismatch");
assert(result.source_population.proposal_source_clusters
  === contract.source_population.proposal_source_clusters,
"proposal source-cluster count mismatch");
assert(contract.source_population.proposal_source_clusters === 1,
  "this frozen proposal block is expected to contain exactly one source cluster");
assert(contract.source_population.source_clustered_fold_estimation_available === false
  && result.source_population.source_clustered_fold_estimation_available === false,
"single-source proposal block incorrectly claims source-cluster fold availability");

const checkObjective = (trace, fractionalBits) => {
  assert(trace.fractional_bits === fractionalBits, `wrong Q${fractionalBits} precision`);
  assert(trace.vertex_losses.length === 64 && trace.coefficients.length === 64,
    `Q${fractionalBits} cube shape mismatch`);
  const losses = trace.vertex_losses.map(BigInt);
  const coefficients = trace.coefficients.map(BigInt);
  const recomputed = mobius(losses);
  assert(recomputed.every((value, mask) => value === coefficients[mask]),
    `Q${fractionalBits} aggregate Möbius mismatch`);
  assert(losses.every((value, mask) => reconstruct(coefficients, mask) === value),
    `Q${fractionalBits} reconstruction failed`);
  assert(trace.documents.length === 64, `Q${fractionalBits} document count mismatch`);
  const sums = Array(64).fill(0n);
  const environmentalMass = Array(7).fill(0n);
  for (const [offset, document] of trace.documents.entries()) {
    assert(document.document === expectedDocumentStart + offset,
      `Q${fractionalBits} document order changed`);
    assert(document.coefficients.length === 64, `Q${fractionalBits} document coefficient shape`);
    document.coefficients.map(BigInt).forEach((value, mask) => {
      sums[mask] += value;
      environmentalMass[popcount(mask)] += absolute(value);
    });
  }
  assert(sums.every((value, mask) => value === coefficients[mask]),
    `Q${fractionalBits} environment aggregation mismatch`);
  const populationMass = Array(7).fill(0n);
  coefficients.forEach((value, mask) => populationMass[popcount(mask)] += absolute(value));
  assert(trace.absolute_mass_by_order.population.map(BigInt)
    .every((value, order) => value === populationMass[order]),
  `Q${fractionalBits} population mass mismatch`);
  assert(trace.absolute_mass_by_order.environmental.map(BigInt)
    .every((value, order) => value === environmentalMass[order]),
  `Q${fractionalBits} environmental mass mismatch`);
  for (const tail of trace.tails) {
    const population = populationMass.slice(tail.retained_order + 1).reduce((a, b) => a + b, 0n);
    const environmental = environmentalMass.slice(tail.retained_order + 1).reduce((a, b) => a + b, 0n);
    assert(BigInt(tail.population_absolute_tail) === population, "population tail mismatch");
    assert(BigInt(tail.environmental_absolute_tail) === environmental, "environmental tail mismatch");
    assert(BigInt(tail.cancellation_mass) === environmental - population, "tail cancellation mismatch");
    assert(BigInt(tail.tail_regret_bound) === population, "sharp tail bound mismatch");
    assert(BigInt(tail.exact_gap) <= BigInt(tail.tail_regret_bound), "sharp tail certificate failed");
    assert(tail.certificate_verified === true, "tail certificate not verified");
  }
  assert(trace.exchange_slices.length === 5, "exchange slices missing");
  assert(trace.exchange_slices.every((slice) => slice.certificate_verified === true),
    "exchange certificate arithmetic failed");
  assert(trace.interaction_width.elimination_orders_evaluated === 720,
    "not all elimination orders were evaluated");
  assert(trace.interaction_width.width_histogram.reduce((a, b) => a + b, 0) === 720,
    "width histogram does not cover all orders");
  return coefficients;
};

const q20 = checkObjective(result.q20, 20);
const q32 = checkObjective(result.q32, 32);
const concordance = {both_zero: 0, q20_zero_q32_nonzero: 0, q20_nonzero_q32_zero: 0,
  both_nonzero_sign_agree: 0, both_nonzero_sign_disagree: 0};
for (let mask = 1; mask < 64; mask += 1) {
  const coarse = q20[mask];
  const fine = q32[mask];
  if (coarse === 0n && fine === 0n) concordance.both_zero += 1;
  else if (coarse === 0n) concordance.q20_zero_q32_nonzero += 1;
  else if (fine === 0n) concordance.q20_nonzero_q32_zero += 1;
  else if ((coarse < 0n) === (fine < 0n)) concordance.both_nonzero_sign_agree += 1;
  else concordance.both_nonzero_sign_disagree += 1;
}
assert(Object.keys(concordance).every(
  (key) => result.representation_concordance.aggregate[key] === concordance[key]),
"aggregate representation concordance mismatch");

const discrepancy = result.representation_discrepancy.aggregate;
assert(discrepancy.q20_to_q32_multiplier === 4096, "wrong Q20-to-Q32 multiplier");
const residuals = result.q20.vertex_losses.map(
  (value, mask) => BigInt(result.q32.vertex_losses[mask]) - BigInt(value) * 4096n);
const residualMinimum = residuals.reduce((left, right) => left < right ? left : right);
const residualMaximum = residuals.reduce((left, right) => left > right ? left : right);
const q20Minimum = result.q20.vertex_losses.reduce((left, right) => left < right ? left : right);
const q32Minimum = result.q32.vertex_losses.reduce((left, right) => left < right ? left : right);
const q20Minimizers = result.q20.vertex_losses.flatMap(
  (value, mask) => value === q20Minimum ? [mask] : []);
const q32Minimizers = result.q32.vertex_losses.flatMap(
  (value, mask) => value === q32Minimum ? [mask] : []);
const sharedMinimizers = q20Minimizers.filter((mask) => q32Minimizers.includes(mask));
const maximumRegret = q20Minimizers.map(
  (mask) => BigInt(result.q32.vertex_losses[mask]) - BigInt(q32Minimum))
  .reduce((left, right) => left > right ? left : right);
assert(BigInt(discrepancy.residual_minimum) === residualMinimum,
  "representation residual minimum mismatch");
assert(BigInt(discrepancy.residual_maximum) === residualMaximum,
  "representation residual maximum mismatch");
assert(BigInt(discrepancy.residual_oscillation) === residualMaximum - residualMinimum,
  "representation discrepancy oscillation mismatch");
assert(JSON.stringify(discrepancy.q20_minimizers) === JSON.stringify(q20Minimizers)
  && JSON.stringify(discrepancy.q32_minimizers) === JSON.stringify(q32Minimizers)
  && JSON.stringify(discrepancy.shared_minimizers) === JSON.stringify(sharedMinimizers),
"representation minimizer sets mismatch");
assert(BigInt(discrepancy.maximum_q32_regret_of_q20_minimizer) === maximumRegret
  && maximumRegret <= BigInt(discrepancy.residual_oscillation)
  && discrepancy.certificate_verified === true,
"representation discrepancy certificate failed");
assert(result.representation_discrepancy.documents.length === 64
  && result.representation_discrepancy.documents.every(
    (document, offset) => document.document === expectedDocumentStart + offset
      && document.certificate_verified),
"document representation discrepancies are incomplete");
for (let offset = 0; offset < 64; offset += 1) {
  const q20Document = result.q20.documents[offset].coefficients.map(BigInt);
  const q32Document = result.q32.documents[offset].coefficients.map(BigInt);
  const q20Losses = Array.from({length: 64}, (_, mask) => reconstruct(q20Document, mask));
  const q32Losses = Array.from({length: 64}, (_, mask) => reconstruct(q32Document, mask));
  const documentResiduals = q20Losses.map(
    (value, mask) => q32Losses[mask] - value * 4096n);
  const minimumResidual = documentResiduals.reduce(
    (left, right) => left < right ? left : right);
  const maximumResidual = documentResiduals.reduce(
    (left, right) => left > right ? left : right);
  const coarseMinimum = q20Losses.reduce((left, right) => left < right ? left : right);
  const fineMinimum = q32Losses.reduce((left, right) => left < right ? left : right);
  const coarseMinimizers = q20Losses.flatMap(
    (value, mask) => value === coarseMinimum ? [mask] : []);
  const fineMinimizers = q32Losses.flatMap(
    (value, mask) => value === fineMinimum ? [mask] : []);
  const documentMaximumRegret = coarseMinimizers.map(
    (mask) => q32Losses[mask] - fineMinimum)
    .reduce((left, right) => left > right ? left : right);
  const recorded = result.representation_discrepancy.documents[offset];
  assert(BigInt(recorded.residual_oscillation) === maximumResidual - minimumResidual,
    `document ${expectedDocumentStart + offset} representation oscillation mismatch`);
  assert(recorded.shared_minimizer
    === coarseMinimizers.some((mask) => fineMinimizers.includes(mask)),
  `document ${expectedDocumentStart + offset} shared-minimizer mismatch`);
  assert(BigInt(recorded.maximum_q32_regret_of_q20_minimizer) === documentMaximumRegret
    && documentMaximumRegret <= BigInt(recorded.residual_oscillation),
  `document ${expectedDocumentStart + offset} representation certificate mismatch`);
}

const taxonomy = result.boundary_taxonomy.all_edges;
const taxonomyTotal = Object.values(taxonomy).reduce((sum, value) => sum + value, 0);
assert(taxonomyTotal === 6 * 32 * 64, "boundary taxonomy does not cover every document edge");
assert(result.boundary_taxonomy.by_atom.length === 6, "per-atom taxonomy missing");
assert(result.boundary_taxonomy.by_atom.every(
  (atom) => Object.values(atom.counts).reduce((sum, value) => sum + value, 0) === 32 * 64),
"per-atom taxonomy does not cover every edge");
assert(result.decision.structure_certificate_selected === false,
  "calibration audit selected a structure certificate automatically");
assert(result.decision.optimizer_change_authorized === false,
  "calibration audit authorized optimizer change");
assert(result.decision.paid_scaling_authorized === false,
  "calibration audit authorized scaling");

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_structure_check.v1",
  analysis_role: result.analysis_role,
  vertices: 64,
  documents: 64,
  coefficients_per_objective: 63,
  elimination_orders_per_objective: 720,
  hard_stop_verified: true,
  source_clusters: result.source_population.proposal_source_clusters,
  source_clustered_fold_estimation_available: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
