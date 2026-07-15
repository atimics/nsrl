#!/usr/bin/env node

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};

const mobiusTransform = (values, rank) => {
  const coefficients = [...values];
  for (let bit = 0; bit < rank; bit += 1) {
    for (let mask = 0; mask < 1 << rank; mask += 1) {
      if ((mask & (1 << bit)) !== 0) {
        coefficients[mask] -= coefficients[mask ^ (1 << bit)];
      }
    }
  }
  return coefficients;
};

const zetaTransform = (coefficients, rank) => {
  const values = [...coefficients];
  for (let bit = 0; bit < rank; bit += 1) {
    for (let mask = 0; mask < 1 << rank; mask += 1) {
      if ((mask & (1 << bit)) !== 0) {
        values[mask] += values[mask ^ (1 << bit)];
      }
    }
  }
  return values;
};

const pow = (base, exponent) => base ** BigInt(exponent);

const expectationNumerator = (values, numerators, denominator) => {
  let result = 0n;
  for (let mask = 0; mask < values.length; mask += 1) {
    let weight = 1n;
    for (let bit = 0; bit < numerators.length; bit += 1) {
      weight *= (mask & (1 << bit)) !== 0
        ? numerators[bit]
        : denominator - numerators[bit];
    }
    result += values[mask] * weight;
  }
  return result;
};

const polynomialNumerator = (coefficients, numerators, denominator) => {
  const rank = numerators.length;
  let result = 0n;
  for (let mask = 0; mask < coefficients.length; mask += 1) {
    let term = coefficients[mask] * pow(denominator, rank - popcount(mask));
    for (let bit = 0; bit < rank; bit += 1) {
      if ((mask & (1 << bit)) !== 0) term *= numerators[bit];
    }
    result += term;
  }
  return result;
};

let generatorState = 0x9e3779b9;
const nextInteger = () => {
  generatorState ^= generatorState << 13;
  generatorState ^= generatorState >>> 17;
  generatorState ^= generatorState << 5;
  return BigInt((generatorState >>> 0) % 4001) - 2000n;
};

const rationalDenominator = 11n;
let transformCases = 0;
let rationalExtensionCases = 0;
for (let rank = 1; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 40; trial += 1) {
    const values = Array.from({ length: 1 << rank }, nextInteger);
    const coefficients = mobiusTransform(values, rank);
    const reconstructed = zetaTransform(coefficients, rank);
    assert(
      reconstructed.every((value, index) => value === values[index]),
      "Möbius/zeta inversion failed",
    );
    transformCases += 1;

    const numerators = Array.from({ length: rank }, () => {
      const value = nextInteger();
      return ((value % 12n) + 12n) % 12n;
    });
    assert(
      expectationNumerator(values, numerators, rationalDenominator)
        === polynomialNumerator(coefficients, numerators, rationalDenominator),
      "multilinear expectation and Möbius polynomial differ",
    );
    rationalExtensionCases += 1;
  }
}

let gridMinimumCases = 0;
for (let rank = 1; rank <= 4; rank += 1) {
  const values = Array.from({ length: 1 << rank }, nextInteger);
  const coefficients = mobiusTransform(values, rank);
  const minimum = values.reduce((left, right) => left < right ? left : right);
  const maximum = values.reduce((left, right) => left > right ? left : right);
  const commonDenominator = pow(rationalDenominator, rank);
  const numerators = Array(rank).fill(0n);
  const visitGrid = (bit) => {
    if (bit === rank) {
      const expectation = expectationNumerator(
        values,
        numerators,
        rationalDenominator,
      );
      assert(expectation >= minimum * commonDenominator, "extension below vertex minimum");
      assert(expectation <= maximum * commonDenominator, "extension above vertex maximum");
      assert(
        expectation === polynomialNumerator(
          coefficients,
          numerators,
          rationalDenominator,
        ),
        "grid extension identity failed",
      );
      gridMinimumCases += 1;
      return;
    }
    for (let numerator = 0n; numerator <= rationalDenominator; numerator += 1n) {
      numerators[bit] = numerator;
      visitGrid(bit + 1);
    }
  };
  visitGrid(0);

  for (let mask = 0; mask < 1 << rank; mask += 1) {
    const vertex = Array.from(
      { length: rank },
      (_, bit) => (mask & (1 << bit)) !== 0 ? rationalDenominator : 0n,
    );
    assert(
      expectationNumerator(values, vertex, rationalDenominator)
        === values[mask] * commonDenominator,
      "multilinear extension is not vertex-exact",
    );
  }
}

const finiteDifference = (fn, base, directions) => {
  let result = 0n;
  for (let mask = 0; mask < 1 << directions.length; mask += 1) {
    let argument = base;
    for (let bit = 0; bit < directions.length; bit += 1) {
      if ((mask & (1 << bit)) !== 0) argument += directions[bit];
    }
    const parity = (directions.length - popcount(mask)) & 1;
    result += parity === 0 ? fn(argument) : -fn(argument);
  }
  return result;
};

const clamp = (value, low, high) => value < low ? low : value > high ? high : value;
const outerFunctions = [
  (value) => value * value * value - 3n * value * value + 2n * value,
  (value) => clamp(value / 3n, -9n, 11n),
  (value) => (value & 1n) === 0n ? value / 2n : -value,
];

let chainRuleCases = 0;
for (let trial = 0; trial < 2000; trial += 1) {
  const g0 = nextInteger() % 31n;
  const g1 = nextInteger() % 31n;
  const g2 = nextInteger() % 31n;
  const g12 = nextInteger() % 31n;
  const delta1 = g1 - g0;
  const delta2 = g2 - g0;
  const delta12 = g12 - g1 - g2 + g0;
  for (const outer of outerFunctions) {
    const left = outer(g12) - outer(g1) - outer(g2) + outer(g0);
    const right = finiteDifference(outer, g0, [delta12])
      + finiteDifference(outer, g0, [delta1, delta2])
      + finiteDifference(outer, g0, [delta1, delta12])
      + finiteDifference(outer, g0, [delta2, delta12])
      + finiteDifference(outer, g0, [delta1, delta2, delta12]);
    assert(left === right, "rank-two discrete Faà di Bruno identity failed");
    chainRuleCases += 1;
  }
}

let pruningEstimatorCases = 0;
let antitheticEstimatorCases = 0;
for (let rank = 1; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 40; trial += 1) {
    const values = Array.from({ length: 1 << rank }, nextInteger);
    const coefficients = mobiusTransform(values, rank);
    const rates = Array.from(
      { length: rank },
      (_, index) => BigInt(1 + ((trial + 3 * index) % 7)),
    );
    const totalRate = rates.reduce((sum, value) => sum + value, 0n);
    const exactDirectionalDerivative = rates.reduce(
      (sum, rate, index) => sum + rate * coefficients[1 << index],
      0n,
    );
    const exhaustiveEstimatorSum = rates.reduce(
      (sum, rate, index) => sum + rate * totalRate * coefficients[1 << index],
      0n,
    );
    assert(
      exhaustiveEstimatorSum === totalRate * exactDirectionalDerivative,
      "weighted atomic pruning estimator is biased",
    );
    pruningEstimatorCases += 1;

    const fullMask = (1 << rank) - 1;
    for (let bit = 0; bit < rank; bit += 1) {
      let antitheticSum = 0n;
      let conditionalDifferenceSum = 0n;
      for (let mask = 0; mask <= fullMask; mask += 1) {
        const sign = (mask & (1 << bit)) !== 0 ? 1n : -1n;
        antitheticSum += (values[mask] - values[fullMask ^ mask]) * sign;
        if ((mask & (1 << bit)) === 0) {
          conditionalDifferenceSum += values[mask | (1 << bit)] - values[mask];
        }
      }
      assert(
        antitheticSum === 2n * conditionalDifferenceSum,
        "half-cube antithetic estimator is biased",
      );
      antitheticEstimatorCases += 1;
    }
  }
}

const escapeOrder = (values, rank) => {
  const base = values[0];
  let order = Number.POSITIVE_INFINITY;
  for (let mask = 1; mask < 1 << rank; mask += 1) {
    if (values[mask] < base) order = Math.min(order, popcount(mask));
  }
  return Number.isFinite(order) ? order : null;
};

const isPrimeThrough = (values, rank, order) => {
  const base = values[0];
  for (let mask = 1; mask < 1 << rank; mask += 1) {
    if (popcount(mask) <= order && values[mask] < base) return false;
  }
  return true;
};

const pairPrimeValues = [0n, 1n, 1n, -1n];
const pairPrimeCoefficients = mobiusTransform(pairPrimeValues, 2);
assert(isPrimeThrough(pairPrimeValues, 2, 1), "pair example is not one-prime");
assert(escapeOrder(pairPrimeValues, 2) === 2, "pair escape order is not two");
assert(pairPrimeCoefficients[3] === -3n, "pair interaction is not -3");

const triplePrimeValues = Array.from({ length: 8 }, (_, mask) =>
  mask === 7 ? -1n : BigInt(popcount(mask))
);
const triplePrimeCoefficients = mobiusTransform(triplePrimeValues, 3);
assert(isPrimeThrough(triplePrimeValues, 3, 2), "triple example is not two-prime");
assert(escapeOrder(triplePrimeValues, 3) === 3, "triple escape order is not three");
assert(triplePrimeCoefficients[7] === -4n, "triple interaction is not -4");

const result = {
  schema: "nsrl.boolean_jet_theory_check.v1",
  exact_boolean_jet: {
    transform_cases: transformCases,
    mobius_zeta_inversion_verified: true,
    rational_extension_cases: rationalExtensionCases,
    expectation_equals_mobius_polynomial_verified: true,
    rational_grid_minimum_cases: gridMinimumCases,
    vertex_exactness_verified: true,
    global_vertex_minimum_preservation_verified: true,
  },
  discrete_composition: {
    rank_two_cases: chainRuleCases,
    outer_functions: ["cubic", "integer-truncate-and-clamp", "parity-discontinuous"],
    five_covering_term_identity_verified: true,
  },
  stochastic_estimators: {
    weighted_atomic_pruning_cases: pruningEstimatorCases,
    weighted_atomic_unbiasedness_verified: true,
    half_cube_antithetic_coordinate_cases: antitheticEstimatorCases,
    half_cube_antithetic_unbiasedness_verified: true,
  },
  prime_examples: {
    pair: {
      losses: pairPrimeValues.map(String),
      mobius_coefficients: pairPrimeCoefficients.map(String),
      prime_through_order: 1,
      escape_order: 2,
    },
    triple: {
      losses: triplePrimeValues.map(String),
      mobius_coefficients: triplePrimeCoefficients.map(String),
      prime_through_order: 2,
      escape_order: 3,
    },
  },
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
