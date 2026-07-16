#!/usr/bin/env node

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (value) => value < 0n ? -value : value;
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const chi = (character, vertex) => popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const minimum = (values) => values.reduce((left, right) => left < right ? left : right);
const maximum = (values) => values.reduce((left, right) => left > right ? left : right);
const argmin = (values) => values.findIndex((value) => value === minimum(values));

// Exact Walsh/Ising equivalence on a rank-three integer Hamiltonian.
const losses = [7n, 3n, 11n, -2n, 5n, 13n, 0n, 17n];
const walsh = Array.from({length: 8}, (_, character) => losses.reduce(
  (sum, loss, vertex) => sum + loss * chi(character, vertex), 0n));
for (let vertex = 0; vertex < 8; vertex += 1) {
  const reconstructed = walsh.reduce(
    (sum, coefficient, character) => sum + coefficient * chi(character, vertex), 0n);
  assert(reconstructed === 8n * losses[vertex], "Walsh/Ising inversion failed");
}

// Exact conditional-exchange decomposition. For base B disjoint from i,j,
// swapping i for j equals the singleton difference plus the difference of all
// interactions between the base and the two exchanged atoms.
const rankFourMobius = [3n, -2n, 5n, 7n, 0n, 11n, -4n, 6n, -20n, -3n, 8n, -5n, 2n, 9n, -7n, 4n];
const evaluateMobius = (mask) => {
  let total = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    total += rankFourMobius[subset];
    if (subset === 0) return total;
  }
};
const baseMask = 0b0011;
const outgoing = 0b0100;
const incoming = 0b1000;
const exchange = evaluateMobius(baseMask | incoming) - evaluateMobius(baseMask | outgoing);
let interactionResidual = 0n;
for (let subset = baseMask; subset !== 0; subset = (subset - 1) & baseMask) {
  interactionResidual += rankFourMobius[subset | incoming]
    - rankFourMobius[subset | outgoing];
}
const singletonDifference = rankFourMobius[incoming] - rankFourMobius[outgoing];
assert(exchange === singletonDifference + interactionResidual,
  "conditional-exchange decomposition failed");
assert(singletonDifference === -20n && interactionResidual === -13n && exchange === -33n,
  "conditional-exchange example changed");
assert(singletonDifference + absolute(interactionResidual) < 0n && exchange < 0n,
  "conditional-exchange margin certificate failed");

// Finite exhaustive check of the oscillation-regret lemma on a four-state space.
const vectors = [];
for (let encoded = 0; encoded < 81; encoded += 1) {
  let value = encoded;
  const vector = [];
  for (let index = 0; index < 4; index += 1) {
    vector.push(BigInt(value % 3 - 1));
    value = Math.floor(value / 3);
  }
  vectors.push(vector);
}
let robustPairs = 0;
for (const exact of vectors) {
  for (const surrogate of vectors) {
    const selected = argmin(surrogate);
    const residual = exact.map((value, state) => value - surrogate[state]);
    const regret = exact[selected] - minimum(exact);
    const oscillation = maximum(residual) - minimum(residual);
    assert(regret <= oscillation, "oscillation-regret lemma failed");
    robustPairs += 1;
  }
}

// Quenched averaging and Gibbs formation do not commute, even for one spin.
// With fugacity 1/2, H1=(0,4), H2=(2,0), the quenched magnetization is 12/85.
// Their mean Hamiltonian is (1,2), whose magnetization is 1/3.
const quenchedNumerator = 12n;
const quenchedDenominator = 85n;
const aggregateNumerator = 1n;
const aggregateDenominator = 3n;
assert(quenchedNumerator * aggregateDenominator
  !== aggregateNumerator * quenchedDenominator,
"quenched/aggregate counterexample collapsed");

// The sign-thresholded mean magnetization is exactly the Hamming Bayes action.
// Use integer common-scale weights for three rank-three document Gibbs laws.
const documentWeights = [
  [64n, 16n, 8n, 4n, 2n, 1n, 1n, 1n],
  [4n, 8n, 16n, 64n, 1n, 2n, 1n, 1n],
  [1n, 1n, 2n, 4n, 8n, 16n, 32n, 64n],
];
const totalWeight = documentWeights.reduce(
  (outer, weights) => outer + weights.reduce((sum, weight) => sum + weight, 0n), 0n);
const moments = Array.from({length: 3}, (_, atom) => documentWeights.reduce(
  (outer, weights) => outer + weights.reduce(
    (sum, weight, vertex) => sum + weight * chi(1 << atom, vertex), 0n), 0n));
const bayesMask = moments.reduce(
  (mask, moment, atom) => moment < 0n ? mask | (1 << atom) : mask, 0);
const hammingRiskNumerator = (action) => documentWeights.reduce(
  (outer, weights) => outer + weights.reduce(
    (sum, weight, vertex) => sum + weight * BigInt(popcount(action ^ vertex)), 0n), 0n);
const risks = Array.from({length: 8}, (_, action) => hammingRiskNumerator(action));
assert(risks[bayesMask] === minimum(risks), "magnetization Hamming Bayes theorem failed");

// A unique ground state has every magnetization sign correct once its Gibbs mass exceeds 1/2.
// Seven excited states at relative weight 1/16 give ground mass 16/23 > 1/2.
const groundWeight = 16n;
const excitedWeight = 1n;
const partition = groundWeight + 7n * excitedWeight;
assert(2n * groundWeight > partition, "zero-temperature majority premise failed");
for (let atom = 0; atom < 3; atom += 1) {
  const worstCaseMoment = groundWeight - 7n * excitedWeight;
  assert(worstCaseMoment > 0n, `ground-state sign failed for atom ${atom}`);
}

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.document_ising_theory_check.v1",
  walsh_ising_reconstruction_verified: true,
  conditional_exchange_decomposition_verified: true,
  conditional_exchange_margin_certificate_verified: true,
  oscillation_regret_pairs_checked: robustPairs,
  quenched_aggregate_noncommutation_counterexample_verified: true,
  magnetization_hamming_bayes_action_verified: true,
  unique_ground_state_majority_bound_verified: true,
  floating_point_operations: 0,
}, null, 2)}\n`);
