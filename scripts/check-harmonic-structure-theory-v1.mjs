#!/usr/bin/env node

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};

const parity = (mask) => popcount(mask) & 1;
const character = (frequency, vertex) => (parity(frequency & vertex) === 0 ? 1n : -1n);

const walshTransform = (values) => Array.from({length: values.length}, (_, frequency) =>
  values.reduce(
    (sum, value, vertex) => sum + value * character(frequency, vertex),
    0n,
  ));

const inverseWalshNumerators = (coefficients) => Array.from(
  {length: coefficients.length},
  (_, vertex) => coefficients.reduce(
    (sum, coefficient, frequency) => sum + coefficient * character(frequency, vertex),
    0n,
  ),
);

const minimumValueAndMask = (values) => {
  let minimum = values[0];
  let mask = 0;
  for (let candidate = 1; candidate < values.length; candidate += 1) {
    if (values[candidate] < minimum) {
      minimum = values[candidate];
      mask = candidate;
    }
  }
  return {minimum, mask};
};

let randomState = 0x9e3779b9;
const randomU32 = () => {
  randomState ^= randomState << 13;
  randomState ^= randomState >>> 17;
  randomState ^= randomState << 5;
  return randomState >>> 0;
};
const randomInteger = (minimum, maximum) =>
  minimum + (randomU32() % (maximum - minimum + 1));

let walshIdentityCases = 0;
let walshIdentityVertices = 0;
for (let rank = 1; rank <= 8; rank += 1) {
  const size = 1 << rank;
  for (let trial = 0; trial < 100; trial += 1) {
    const values = Array.from(
      {length: size},
      () => BigInt(randomInteger(-1000, 1000)),
    );
    const coefficients = walshTransform(values);
    const inverseNumerators = inverseWalshNumerators(coefficients);
    for (let vertex = 0; vertex < size; vertex += 1) {
      assert(
        inverseNumerators[vertex] === BigInt(size) * values[vertex],
        "Walsh inversion failed",
      );
      walshIdentityVertices += 1;
    }
    const signalEnergy = values.reduce((sum, value) => sum + value * value, 0n);
    const spectralEnergy = coefficients.reduce(
      (sum, coefficient) => sum + coefficient * coefficient,
      0n,
    );
    assert(
      spectralEnergy === BigInt(size) * signalEnergy,
      "unnormalized Walsh Parseval identity failed",
    );
    const offset = BigInt(randomInteger(-1000, 1000));
    const shiftedCoefficients = walshTransform(values.map((value) => value + offset));
    assert(
      coefficients.slice(1).every(
        (coefficient, frequency) => coefficient === shiftedCoefficients[frequency + 1],
      ),
      "nonconstant Walsh coefficients changed under an objective offset",
    );
    walshIdentityCases += 1;
  }
}

const maximumDisagreementCount = (rank, frequencies) => {
  let maximum = 0;
  for (let difference = 1; difference < 1 << rank; difference += 1) {
    const count = frequencies.filter(
      (frequency) => parity(frequency & difference) === 1,
    ).length;
    if (count > maximum) maximum = count;
  }
  return maximum;
};

let spectralTailCases = 0;
let spectralTailVertices = 0;
for (let rank = 2; rank <= 8; rank += 1) {
  const size = 1 << rank;
  for (let trial = 0; trial < 100; trial += 1) {
    const values = Array.from(
      {length: size},
      () => BigInt(randomInteger(-1000, 1000)),
    );
    const coefficients = walshTransform(values);
    const retainedDegree = randomInteger(0, rank);
    const residualFrequencies = [];
    const retainedCoefficients = coefficients.map((coefficient, frequency) => {
      if (popcount(frequency) <= retainedDegree) return coefficient;
      residualFrequencies.push(frequency);
      return 0n;
    });
    const surrogateScaledValues = inverseWalshNumerators(retainedCoefficients);
    const surrogateMinimizer = minimumValueAndMask(surrogateScaledValues).mask;
    const trueMinimum = minimumValueAndMask(values).minimum;
    const scaledRegret = BigInt(size) * (values[surrogateMinimizer] - trueMinimum);
    const residualEnergy = residualFrequencies.reduce(
      (sum, frequency) => sum + coefficients[frequency] * coefficients[frequency],
      0n,
    );
    const disagreementCount = maximumDisagreementCount(rank, residualFrequencies);
    assert(
      scaledRegret * scaledRegret
        <= 4n * BigInt(disagreementCount) * residualEnergy,
      "Walsh tail minimizer exceeded its spectral oscillation certificate",
    );
    spectralTailCases += 1;
    spectralTailVertices += size;
  }
}

const rankSixTailGeometry = Array.from({length: 6}, (_, retainedDegree) => {
  const residualFrequencies = Array.from(
    {length: 1 << 6},
    (_, frequency) => frequency,
  ).filter((frequency) => popcount(frequency) > retainedDegree);
  return {
    retained_degree: retainedDegree,
    residual_characters: residualFrequencies.length,
    maximum_pair_disagreements: maximumDisagreementCount(6, residualFrequencies),
  };
});

const centeredSpikeCounterexamples = [];
for (let rank = 2; rank <= 10; rank += 1) {
  const size = 1 << rank;
  centeredSpikeCounterexamples.push({
    rank,
    vertices: size,
    residual_oscillation: 1,
    normalized_u2_fourth_power_numerator: size - 1,
    normalized_u2_fourth_power_denominator: size ** 4,
    normalized_u2: ((size - 1) / (size ** 4)) ** 0.25,
  });
}
assert(
  centeredSpikeCounterexamples.every(
    (example, index, examples) => index === 0
      || example.normalized_u2 < examples[index - 1].normalized_u2,
  ),
  "centered-spike U2 counterexample did not become more uniform with rank",
);

const ramanujanPowerTwo = (period, value) => {
  if (period === 1) return 1;
  const residue = ((value % period) + period) % period;
  if (residue === 0) return period / 2;
  if (residue === period / 2) return -(period / 2);
  return 0;
};

const phaseCellWidth = 1 << 12;
const phasePeriods = Array.from({length: 13}, (_, exponent) => 1 << exponent);
let ramanujanPeriodCases = 0;
for (const period of phasePeriods) {
  let sum = 0;
  let squaredSum = 0;
  for (let phase = 0; phase < phaseCellWidth; phase += 1) {
    const value = ramanujanPowerTwo(period, phase);
    sum += value;
    squaredSum += value * value;
  }
  if (period === 1) assert(sum === phaseCellWidth, "constant phase subspace changed");
  else assert(sum === 0, "nonconstant Ramanujan vector has nonzero mean");
  const totient = period === 1 ? 1 : period / 2;
  assert(
    squaredSum === phaseCellWidth * totient,
    "Ramanujan power-two energy identity failed",
  );
  ramanujanPeriodCases += 1;
}

let ramanujanCrossSubspaceCases = 0;
for (let leftIndex = 0; leftIndex < phasePeriods.length; leftIndex += 1) {
  for (let rightIndex = leftIndex + 1; rightIndex < phasePeriods.length; rightIndex += 1) {
    const leftPeriod = phasePeriods[leftIndex];
    const rightPeriod = phasePeriods[rightIndex];
    const leftShifts = [...new Set([0, 1, Math.floor(leftPeriod / 2)])];
    const rightShifts = [...new Set([0, 1, Math.floor(rightPeriod / 2)])];
    for (const leftShift of leftShifts) {
      for (const rightShift of rightShifts) {
        let innerProduct = 0;
        for (let phase = 0; phase < phaseCellWidth; phase += 1) {
          innerProduct += ramanujanPowerTwo(leftPeriod, phase - leftShift)
            * ramanujanPowerTwo(rightPeriod, phase - rightShift);
        }
        assert(innerProduct === 0, "distinct Ramanujan phase subspaces lost orthogonality");
        ramanujanCrossSubspaceCases += 1;
      }
    }
  }
}

const output = {
  schema: "nsrl.harmonic_structure_theory_check.v1",
  walsh_analysis: {
    identity_cases: walshIdentityCases,
    vertices: walshIdentityVertices,
    exact_integer_transform_and_inversion: true,
    parseval_verified: true,
    nonconstant_spectrum_is_offset_invariant: true,
  },
  spectral_optimization_certificate: {
    cases: spectralTailCases,
    vertices: spectralTailVertices,
    regret_bound: "2_times_sqrt(max_pair_disagreements_times_normalized_tail_energy)",
    rank_six_tail_geometry: rankSixTailGeometry,
  },
  uniformity_warning: {
    centered_spike_counterexamples: centeredSpikeCounterexamples,
    small_u2_does_not_bound_objective_oscillation: true,
  },
  ramanujan_phase_analysis: {
    phase_cell_width: phaseCellWidth,
    periods: phasePeriods,
    period_cases: ramanujanPeriodCases,
    cross_subspace_cases: ramanujanCrossSubspaceCases,
    power_two_formula: "zero_except_at_two_highest_divisibility_classes",
    integer_orthogonal_period_subspaces_verified: true,
  },
};

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
