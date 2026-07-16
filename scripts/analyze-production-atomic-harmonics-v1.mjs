#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const inputPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const outputPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-harmonics-proposal-v1.json";
const inputBytes = fs.readFileSync(inputPath);
const analysisSourceBytes = fs.readFileSync(new URL(import.meta.url));
const source = JSON.parse(inputBytes.toString("utf8"));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const rank = 6;
const vertices = 1 << rank;
const denominator = BigInt(vertices);
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const sign = (character, vertex) => popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const minimum = (values) => values.reduce((left, right) => left < right ? left : right);
const maximum = (values) => values.reduce((left, right) => left > right ? left : right);
const ceilDivide = (numerator, divisor) => (numerator + divisor - 1n) / divisor;
const ceilSqrt = (value) => {
  if (value <= 1n) return value;
  let low = 0n;
  let high = 1n;
  while (high * high < value) high *= 2n;
  while (low + 1n < high) {
    const middle = (low + high) / 2n;
    if (middle * middle >= value) high = middle;
    else low = middle;
  }
  return high;
};
const walsh = (losses) => Array.from({length: vertices}, (_, character) =>
  losses.reduce((sum, loss, vertex) => sum + loss * sign(character, vertex), 0n));
const reconstructLosses = (coefficients) => Array.from({length: vertices}, (_, mask) => {
  let sum = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    sum += coefficients[subset];
    if (subset === 0) return sum;
  }
});
const maximumPairDisagreements = (characters) => {
  let result = 0;
  for (let difference = 1; difference < vertices; difference += 1) {
    const disagreements = characters.filter(
      (character) => popcount(character & difference) % 2 === 1).length;
    result = Math.max(result, disagreements);
  }
  return result;
};
const analyzeLosses = (losses) => {
  assert(losses.length === vertices, "rank-six loss cube required");
  const coefficients = walsh(losses);
  const energyByDegree = Array(7).fill(0n);
  coefficients.forEach((coefficient, character) => {
    energyByDegree[popcount(character)] += coefficient * coefficient;
  });
  const parsevalLeft = energyByDegree.reduce((sum, value) => sum + value, 0n);
  const parsevalRight = denominator * losses.reduce((sum, value) => sum + value * value, 0n);
  assert(parsevalLeft === parsevalRight, "Walsh Parseval identity failed");
  const exactMinimum = minimum(losses);
  const tails = [];
  for (let retainedDegree = 0; retainedDegree < rank; retainedDegree += 1) {
    const residualCharacters = Array.from({length: vertices}, (_, value) => value)
      .filter((character) => popcount(character) > retainedDegree);
    const retainedCharacters = Array.from({length: vertices}, (_, value) => value)
      .filter((character) => popcount(character) <= retainedDegree);
    const tailEnergy = residualCharacters.reduce(
      (sum, character) => sum + coefficients[character] * coefficients[character], 0n);
    const disagreements = maximumPairDisagreements(residualCharacters);
    const spectralRadicand = 4n * BigInt(disagreements) * tailEnergy;
    const spectralBound = ceilDivide(ceilSqrt(spectralRadicand), denominator);
    const surrogateNumerators = Array.from({length: vertices}, (_, vertex) =>
      retainedCharacters.reduce(
        (sum, character) => sum + coefficients[character] * sign(character, vertex), 0n));
    const surrogateMinimum = minimum(surrogateNumerators);
    const surrogateMinimizers = surrogateNumerators.flatMap(
      (value, vertex) => value === surrogateMinimum ? [vertex] : []);
    const selectedMinimizer = surrogateMinimizers[0];
    const exactGap = losses[selectedMinimizer] - exactMinimum;
    const residualNumerators = Array.from({length: vertices}, (_, vertex) =>
      residualCharacters.reduce(
        (sum, character) => sum + coefficients[character] * sign(character, vertex), 0n));
    const residualOscillation = maximum(residualNumerators) - minimum(residualNumerators);
    assert(residualOscillation * residualOscillation <= spectralRadicand,
      "Walsh spectral oscillation bound failed");
    assert(exactGap * denominator <= residualOscillation,
      "Walsh direct surrogate-regret bound failed");
    assert(exactGap <= spectralBound, "Walsh spectral regret certificate failed");
    tails.push({
      retained_degree: retainedDegree,
      residual_characters: residualCharacters.length,
      maximum_pair_disagreements: disagreements,
      tail_energy_numerator: tailEnergy.toString(),
      residual_oscillation_numerator: residualOscillation.toString(),
      residual_denominator: vertices,
      residual_oscillation_ceil: ceilDivide(residualOscillation, denominator).toString(),
      spectral_regret_bound_ceil: spectralBound.toString(),
      surrogate_minimizers: surrogateMinimizers,
      selected_minimizer: selectedMinimizer,
      exact_gap: exactGap.toString(),
      direct_oscillation_certificate_verified: true,
      spectral_certificate_verified: true,
    });
  }
  return {
    walsh_numerators: coefficients.map(String),
    energy_by_degree: energyByDegree.map(String),
    parseval_verified: true,
    tails,
  };
};
const analyzeObjective = (objective) => {
  const aggregate = analyzeLosses(objective.vertex_losses.map(BigInt));
  const documents = objective.documents.map((document) => {
    const analysis = analyzeLosses(reconstructLosses(document.coefficients.map(BigInt)));
    const {surrogate_minimizers, ...cubic} = analysis.tails[3];
    return {
      document: document.document,
      walsh_numerators: analysis.walsh_numerators,
      energy_by_degree: analysis.energy_by_degree,
      parseval_verified: analysis.parseval_verified,
      cubic: {
        ...cubic,
        surrogate_minimizer_count: surrogate_minimizers.length,
      },
    };
  });
  return {
    fractional_bits: objective.fractional_bits,
    aggregate,
    documents,
  };
};
const concordance = (coarse, fine) => {
  const result = {both_zero: 0, q20_zero_q32_nonzero: 0, q20_nonzero_q32_zero: 0,
    both_nonzero_sign_agree: 0, both_nonzero_sign_disagree: 0};
  for (let character = 1; character < vertices; character += 1) {
    if (coarse[character] === 0n && fine[character] === 0n) result.both_zero += 1;
    else if (coarse[character] === 0n) result.q20_zero_q32_nonzero += 1;
    else if (fine[character] === 0n) result.q20_nonzero_q32_zero += 1;
    else if ((coarse[character] < 0n) === (fine[character] < 0n)) {
      result.both_nonzero_sign_agree += 1;
    } else result.both_nonzero_sign_disagree += 1;
  }
  return result;
};

assert(source.schema === "nsrl.production_atomic_structure.v1", "wrong source schema");
assert(source.rank === rank && source.vertices_evaluated === vertices, "wrong source rank");
assert(source.analysis_role === "proposal_only_calibration", "source is not proposal-only");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 0,
  "harmonic analysis source crossed the proposal firewall");
assert(source.source_population.proposal_source_clusters === 1
  && source.source_population.source_clustered_fold_estimation_available === false,
"source-cluster limitation changed");
const q20 = analyzeObjective(source.q20);
const q32 = analyzeObjective(source.q32);
const result = {
  schema: "nsrl.production_atomic_harmonics.v1",
  analysis_role: "proposal_only_calibration",
  rank,
  vertices,
  walsh_normalization_denominator: vertices,
  analysis_source_sha256: crypto.createHash("sha256").update(analysisSourceBytes).digest("hex"),
  node_runtime: process.version,
  source_result_sha256: crypto.createHash("sha256").update(inputBytes).digest("hex"),
  bindings: source.bindings,
  source_population: source.source_population,
  q20,
  q32,
  representation_concordance: concordance(
    q20.aggregate.walsh_numerators.map(BigInt), q32.aggregate.walsh_numerators.map(BigInt)),
  limitations: {
    source_clustered_stability_estimated: false,
    pre_action_phase_features_available: false,
    ramanujan_prediction_authorized: false,
  },
  decision: {
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
  },
};
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(result, null, 2)}\n`);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_harmonics_check.v1",
  analysis_source_sha256: result.analysis_source_sha256,
  source_result_sha256: result.source_result_sha256,
  q20_cubic: q20.aggregate.tails[3],
  q32_cubic: q32.aggregate.tails[3],
  source_clustered_stability_estimated: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
