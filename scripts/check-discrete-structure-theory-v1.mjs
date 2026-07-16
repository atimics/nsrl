#!/usr/bin/env node

import fs from "node:fs";

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const absolute = (value) => (value < 0n ? -value : value);

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

let randomState = 0xa341316c;
const randomU32 = () => {
  randomState ^= randomState << 13;
  randomState ^= randomState >>> 17;
  randomState ^= randomState << 5;
  return randomState >>> 0;
};
const randomInteger = (minimum, maximum) =>
  minimum + (randomU32() % (maximum - minimum + 1));

let interactionTailCases = 0;
let interactionTailVertices = 0;
for (let rank = 2; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 100; trial += 1) {
    const values = Array.from(
      {length: 1 << rank},
      () => BigInt(randomInteger(-10000, 10000)),
    );
    const coefficients = mobiusTransform(values, rank);
    const retainedRank = randomInteger(0, rank);
    let tail = 0n;
    const truncatedCoefficients = coefficients.map((coefficient, mask) => {
      if (popcount(mask) <= retainedRank) return coefficient;
      tail += absolute(coefficient);
      return 0n;
    });
    const truncatedValues = zetaTransform(truncatedCoefficients, rank);
    let exactMinimum = values[0];
    let truncatedMinimum = truncatedValues[0];
    let truncatedMinimizer = 0;
    for (let mask = 0; mask < 1 << rank; mask += 1) {
      const error = absolute(values[mask] - truncatedValues[mask]);
      assert(error <= tail, "low-rank truncation error exceeded the absolute interaction tail");
      if (values[mask] < exactMinimum) exactMinimum = values[mask];
      if (truncatedValues[mask] < truncatedMinimum) {
        truncatedMinimum = truncatedValues[mask];
        truncatedMinimizer = mask;
      }
      interactionTailVertices += 1;
    }
    assert(
      values[truncatedMinimizer] - exactMinimum <= tail,
      "truncated minimizer exceeded the sharp absolute-tail global gap bound",
    );
    interactionTailCases += 1;
  }
}

let expectationCommutationCases = 0;
for (let rank = 2; rank <= 7; rank += 1) {
  for (let trial = 0; trial < 50; trial += 1) {
    const environmentValues = Array.from({length: 5}, () => Array.from(
      {length: 1 << rank},
      () => BigInt(randomInteger(-1000, 1000)),
    ));
    const summedValues = Array(1 << rank).fill(0n);
    const summedCoefficients = Array(1 << rank).fill(0n);
    for (const values of environmentValues) {
      const coefficients = mobiusTransform(values, rank);
      for (let mask = 0; mask < 1 << rank; mask += 1) {
        summedValues[mask] += values[mask];
        summedCoefficients[mask] += coefficients[mask];
      }
    }
    const transformedSum = mobiusTransform(summedValues, rank);
    assert(
      transformedSum.every((value, mask) => value === summedCoefficients[mask]),
      "expectation/summation did not commute with the Möbius transform",
    );
    expectationCommutationCases += 1;
  }
}

const invisibleRank = 6;
const auditedOrder = 2;
const zeroCoefficients = Array(1 << invisibleRank).fill(0n);
const hiddenCoefficients = [...zeroCoefficients];
hiddenCoefficients[(1 << invisibleRank) - 1] = -7n;
const zeroValues = zetaTransform(zeroCoefficients, invisibleRank);
const hiddenValues = zetaTransform(hiddenCoefficients, invisibleRank);
for (let mask = 0; mask < 1 << invisibleRank; mask += 1) {
  if (popcount(mask) <= auditedOrder) {
    assert(
      zeroValues[mask] === hiddenValues[mask],
      "high-order invisibility example changed an audited low-order vertex",
    );
  }
}
assert(Math.min(...zeroValues.map(Number)) === 0, "zero comparison cube changed");
assert(Math.min(...hiddenValues.map(Number)) === -7, "hidden high-order optimum disappeared");

const fixedCardinalityMasks = (rank, cardinality) => Array.from(
  {length: 1 << rank},
  (_, mask) => mask,
).filter((mask) => popcount(mask) === cardinality);

const exchangeDefect = (values, masks, rank) => {
  let epsilon = 0n;
  for (const left of masks) {
    for (const right of masks) {
      const leftOnly = left & ~right;
      const rightOnly = right & ~left;
      for (let source = 0; source < rank; source += 1) {
        if ((leftOnly & (1 << source)) === 0) continue;
        let best;
        for (let target = 0; target < rank; target += 1) {
          if ((rightOnly & (1 << target)) === 0) continue;
          const exchangedLeft = (left ^ (1 << source)) | (1 << target);
          const exchangedRight = (right | (1 << source)) ^ (1 << target);
          const defect = values[exchangedLeft] + values[exchangedRight]
            - values[left] - values[right];
          if (best === undefined || defect < best) best = defect;
        }
        assert(best !== undefined, "exchange pair had no target coordinate");
        if (best > epsilon) epsilon = best;
      }
    }
  }
  return epsilon;
};

const isExchangeLocalMinimum = (values, mask, rank) => {
  for (let source = 0; source < rank; source += 1) {
    if ((mask & (1 << source)) === 0) continue;
    for (let target = 0; target < rank; target += 1) {
      if ((mask & (1 << target)) !== 0) continue;
      const neighbor = (mask ^ (1 << source)) | (1 << target);
      if (values[neighbor] < values[mask]) return false;
    }
  }
  return true;
};

let approximateExchangeCases = 0;
let approximateExchangeLocalMinima = 0;
for (let trial = 0; trial < 300; trial += 1) {
  const rank = randomInteger(3, 8);
  const cardinality = randomInteger(1, rank - 1);
  const masks = fixedCardinalityMasks(rank, cardinality);
  const values = Array(1 << rank).fill(0n);
  for (const mask of masks) values[mask] = BigInt(randomInteger(-100, 100));
  const epsilon = exchangeDefect(values, masks, rank);
  const globalMinimum = masks.reduce(
    (minimum, mask) => (values[mask] < minimum ? values[mask] : minimum),
    values[masks[0]],
  );
  for (const mask of masks) {
    if (!isExchangeLocalMinimum(values, mask, rank)) continue;
    assert(
      values[mask] - globalMinimum <= BigInt(cardinality) * epsilon,
      "exchange-local minimum exceeded the cardinality-times-defect global gap",
    );
    approximateExchangeLocalMinima += 1;
  }
  approximateExchangeCases += 1;
}

let exactExchangeCases = 0;
for (let trial = 0; trial < 100; trial += 1) {
  const rank = randomInteger(3, 8);
  const cardinality = randomInteger(1, rank - 1);
  const costs = Array.from({length: rank}, () => BigInt(randomInteger(-50, 50)));
  const masks = fixedCardinalityMasks(rank, cardinality);
  const values = Array(1 << rank).fill(0n);
  for (const mask of masks) {
    values[mask] = costs.reduce(
      (sum, cost, coordinate) => sum + ((mask & (1 << coordinate)) !== 0 ? cost : 0n),
      0n,
    );
  }
  assert(exchangeDefect(values, masks, rank) === 0n, "modular slice lost exact exchange convexity");
  const globalMinimum = masks.reduce(
    (minimum, mask) => (values[mask] < minimum ? values[mask] : minimum),
    values[masks[0]],
  );
  assert(
    masks.every((mask) => !isExchangeLocalMinimum(values, mask, rank)
      || values[mask] === globalMinimum),
    "exact exchange-convex slice has a nonglobal exchange-local minimum",
  );
  exactExchangeCases += 1;
}

let surrogateExchangeCases = 0;
let surrogateExchangeLocalMinima = 0;
let exchangeDefectStabilityCases = 0;
for (let trial = 0; trial < 300; trial += 1) {
  const rank = randomInteger(3, 8);
  const cardinality = randomInteger(1, rank - 1);
  const masks = fixedCardinalityMasks(rank, cardinality);
  const surrogateValues = Array(1 << rank).fill(0n);
  const populationValues = Array(1 << rank).fill(0n);
  const discrepancies = [];
  for (const mask of masks) {
    surrogateValues[mask] = BigInt(randomInteger(-100, 100));
    const discrepancy = BigInt(randomInteger(-20, 20));
    populationValues[mask] = surrogateValues[mask] + discrepancy;
    discrepancies.push(discrepancy);
  }
  const surrogateDefect = exchangeDefect(surrogateValues, masks, rank);
  const populationDefect = exchangeDefect(populationValues, masks, rank);
  const minimumDiscrepancy = discrepancies.reduce(
    (minimum, value) => (value < minimum ? value : minimum),
    discrepancies[0],
  );
  const maximumDiscrepancy = discrepancies.reduce(
    (maximum, value) => (value > maximum ? value : maximum),
    discrepancies[0],
  );
  const discrepancyOscillation = maximumDiscrepancy - minimumDiscrepancy;
  assert(
    absolute(populationDefect - surrogateDefect) <= 2n * discrepancyOscillation,
    "exchange defect changed by more than twice the objective-discrepancy oscillation",
  );
  const populationMinimum = masks.reduce(
    (minimum, mask) => (populationValues[mask] < minimum ? populationValues[mask] : minimum),
    populationValues[masks[0]],
  );
  for (const mask of masks) {
    if (!isExchangeLocalMinimum(surrogateValues, mask, rank)) continue;
    assert(
      populationValues[mask] - populationMinimum
        <= BigInt(cardinality) * surrogateDefect + discrepancyOscillation,
      "surrogate exchange-local minimum exceeded its transferred population certificate",
    );
    surrogateExchangeLocalMinima += 1;
  }
  surrogateExchangeCases += 1;
  exchangeDefectStabilityCases += 1;
}

const factorValue = (factor, globalMask) => {
  let localMask = 0;
  for (let position = 0; position < factor.variables.length; position += 1) {
    if ((globalMask & (1 << factor.variables[position])) !== 0) {
      localMask |= 1 << position;
    }
  }
  return factor.values[localMask];
};

const eliminate = (inputFactors, order) => {
  let factors = inputFactors.map((factor) => ({
    variables: [...factor.variables],
    values: [...factor.values],
  }));
  let inducedWidth = 0;
  for (const variable of order) {
    const bucket = factors.filter((factor) => factor.variables.includes(variable));
    factors = factors.filter((factor) => !factor.variables.includes(variable));
    if (bucket.length === 0) continue;
    const bucketVariables = [...new Set(bucket.flatMap((factor) => factor.variables))]
      .sort((left, right) => left - right);
    inducedWidth = Math.max(inducedWidth, bucketVariables.length - 1);
    const remainingVariables = bucketVariables.filter((candidate) => candidate !== variable);
    const table = [];
    for (let localMask = 0; localMask < 1 << remainingVariables.length; localMask += 1) {
      let globalMask = 0;
      for (let position = 0; position < remainingVariables.length; position += 1) {
        if ((localMask & (1 << position)) !== 0) {
          globalMask |= 1 << remainingVariables[position];
        }
      }
      const without = bucket.reduce(
        (sum, factor) => sum + factorValue(factor, globalMask),
        0n,
      );
      const withVariable = bucket.reduce(
        (sum, factor) => sum + factorValue(factor, globalMask | (1 << variable)),
        0n,
      );
      table.push(without < withVariable ? without : withVariable);
    }
    factors.push({variables: remainingVariables, values: table});
  }
  assert(
    factors.every((factor) => factor.variables.length === 0),
    "variable elimination left live variables",
  );
  return {
    minimum: factors.reduce((sum, factor) => sum + factor.values[0], 0n),
    inducedWidth,
  };
};

const monomialFactor = (variables, coefficient) => {
  const values = Array(1 << variables.length).fill(0n);
  values[values.length - 1] = coefficient;
  return {variables, values};
};

let variableEliminationCases = 0;
let maximumVerifiedInducedWidth = 0;
for (let trial = 0; trial < 200; trial += 1) {
  const rank = randomInteger(3, 10);
  const factors = [monomialFactor([], BigInt(randomInteger(-20, 20)))];
  for (let variable = 0; variable < rank; variable += 1) {
    factors.push(monomialFactor([variable], BigInt(randomInteger(-20, 20))));
    if (variable + 1 < rank) {
      factors.push(monomialFactor(
        [variable, variable + 1],
        BigInt(randomInteger(-20, 20)),
      ));
    }
    if (variable + 2 < rank && trial % 2 === 0) {
      factors.push(monomialFactor(
        [variable, variable + 1, variable + 2],
        BigInt(randomInteger(-10, 10)),
      ));
    }
  }
  let bruteMinimum;
  for (let mask = 0; mask < 1 << rank; mask += 1) {
    const value = factors.reduce((sum, factor) => sum + factorValue(factor, mask), 0n);
    if (bruteMinimum === undefined || value < bruteMinimum) bruteMinimum = value;
  }
  const result = eliminate(factors, Array.from({length: rank}, (_, index) => index));
  assert(result.minimum === bruteMinimum, "bounded-width variable elimination missed the minimum");
  assert(result.inducedWidth <= 2, "chain factorization exceeded induced width two");
  maximumVerifiedInducedWidth = Math.max(maximumVerifiedInducedWidth, result.inducedWidth);
  variableEliminationCases += 1;
}

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

let retainedSurrogateCases = 0;
let adaptiveRetainedSupportCases = 0;
for (let trial = 0; trial < 400; trial += 1) {
  const rank = randomInteger(2, 8);
  const trueCoefficients = Array.from(
    {length: 1 << rank},
    () => BigInt(randomInteger(-40, 40)),
  );
  const surrogateCoefficients = Array(1 << rank).fill(0n);
  surrogateCoefficients[0] = BigInt(randomInteger(-100, 100));
  let exactCoefficientDiscrepancy = 0n;
  let simultaneousIntervalCertificate = 0n;
  const threshold = BigInt(randomInteger(5, 30));
  for (let mask = 1; mask < 1 << rank; mask += 1) {
    const radius = BigInt(randomInteger(0, 10));
    const error = BigInt(randomInteger(-Number(radius), Number(radius)));
    const estimated = trueCoefficients[mask] + error;
    const retained = absolute(estimated) >= threshold;
    simultaneousIntervalCertificate += radius;
    if (retained) {
      surrogateCoefficients[mask] = estimated;
      exactCoefficientDiscrepancy += absolute(trueCoefficients[mask] - estimated);
    } else {
      simultaneousIntervalCertificate += absolute(estimated);
      exactCoefficientDiscrepancy += absolute(trueCoefficients[mask]);
    }
  }
  assert(
    exactCoefficientDiscrepancy <= simultaneousIntervalCertificate,
    "simultaneous coefficient intervals did not cover selected-surrogate discrepancy",
  );
  const trueValues = zetaTransform(trueCoefficients, rank);
  const surrogateValues = zetaTransform(surrogateCoefficients, rank);
  const trueMinimum = minimumValueAndMask(trueValues).minimum;
  const surrogateMinimizer = minimumValueAndMask(surrogateValues).mask;
  const populationRegret = trueValues[surrogateMinimizer] - trueMinimum;
  assert(
    populationRegret <= exactCoefficientDiscrepancy,
    "surrogate minimizer exceeded exact nonconstant coefficient discrepancy",
  );
  assert(
    populationRegret <= simultaneousIntervalCertificate,
    "adaptively selected support exceeded its simultaneous-interval certificate",
  );
  retainedSurrogateCases += 1;
  adaptiveRetainedSupportCases += 1;
}

let widthTailCertificateCases = 0;
let maximumCertifiedSurrogateWidth = 0;
for (let trial = 0; trial < 200; trial += 1) {
  const rank = randomInteger(3, 8);
  const trueCoefficients = Array.from(
    {length: 1 << rank},
    () => BigInt(randomInteger(-20, 20)),
  );
  const surrogateCoefficients = Array(1 << rank).fill(0n);
  surrogateCoefficients[0] = BigInt(randomInteger(-20, 20));
  let certificate = 0n;
  const threshold = BigInt(randomInteger(2, 12));
  for (let mask = 1; mask < 1 << rank; mask += 1) {
    const radius = BigInt(randomInteger(0, 5));
    const error = BigInt(randomInteger(-Number(radius), Number(radius)));
    const estimated = trueCoefficients[mask] + error;
    const variables = Array.from(
      {length: rank},
      (_, variable) => variable,
    ).filter((variable) => (mask & (1 << variable)) !== 0);
    const span = variables[variables.length - 1] - variables[0];
    const widthCompatible = variables.length <= 3 && span <= 2;
    const retained = widthCompatible && absolute(estimated) >= threshold;
    certificate += radius;
    if (retained) surrogateCoefficients[mask] = estimated;
    else certificate += absolute(estimated);
  }
  const factors = [monomialFactor([], surrogateCoefficients[0])];
  for (let mask = 1; mask < 1 << rank; mask += 1) {
    if (surrogateCoefficients[mask] === 0n) continue;
    const variables = Array.from(
      {length: rank},
      (_, variable) => variable,
    ).filter((variable) => (mask & (1 << variable)) !== 0);
    factors.push(monomialFactor(variables, surrogateCoefficients[mask]));
  }
  const surrogateValues = zetaTransform(surrogateCoefficients, rank);
  const surrogateMinimum = minimumValueAndMask(surrogateValues);
  const eliminationResult = eliminate(
    factors,
    Array.from({length: rank}, (_, variable) => variable),
  );
  assert(
    eliminationResult.minimum === surrogateMinimum.minimum,
    "width-constrained retained surrogate was not minimized exactly",
  );
  assert(
    eliminationResult.inducedWidth <= 2,
    "width-constrained retained support exceeded induced width two",
  );
  const trueValues = zetaTransform(trueCoefficients, rank);
  const trueMinimum = minimumValueAndMask(trueValues).minimum;
  assert(
    trueValues[surrogateMinimum.mask] - trueMinimum <= certificate,
    "low-width surrogate minimizer exceeded its omitted-mass certificate",
  );
  maximumCertifiedSurrogateWidth = Math.max(
    maximumCertifiedSurrogateWidth,
    eliminationResult.inducedWidth,
  );
  widthTailCertificateCases += 1;
}

let coefficientRangeCases = 0;
for (let rank = 1; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 100; trial += 1) {
    const baseline = randomInteger(-10000, 10000);
    const declaredOscillation = randomInteger(1, 100);
    const values = Array.from(
      {length: 1 << rank},
      () => BigInt(baseline + randomInteger(0, declaredOscillation)),
    );
    const observedMinimum = values.reduce(
      (minimum, value) => (value < minimum ? value : minimum),
      values[0],
    );
    const observedMaximum = values.reduce(
      (maximum, value) => (value > maximum ? value : maximum),
      values[0],
    );
    const observedOscillation = observedMaximum - observedMinimum;
    const coefficients = mobiusTransform(values, rank);
    for (let mask = 1; mask < 1 << rank; mask += 1) {
      const order = popcount(mask);
      assert(
        absolute(coefficients[mask]) <= (1n << BigInt(order - 1)) * observedOscillation,
        "document coefficient exceeded its within-cube oscillation envelope",
      );
    }
    coefficientRangeCases += 1;
  }
}

const p10mHoeffdingPlan = (() => {
  const rank = 6;
  const documents = 64;
  const alpha = 0.05;
  const coefficients = (1 << rank) - 1;
  const commonFactor = Math.sqrt(Math.log((2 * coefficients) / alpha) / (2 * documents));
  return {
    rank,
    documents,
    alpha,
    simultaneous_nonconstant_coefficients: coefficients,
    common_factor: commonFactor,
    coefficient_radius_over_document_oscillation_by_order: Array.from(
      {length: rank},
      (_, order) => 2 ** (order + 1) * commonFactor,
    ),
    absolute_coefficient_radius_over_document_oscillation_by_order: Array.from(
      {length: rank},
      (_, order) => 2 ** order * commonFactor,
    ),
    total_l1_radius_over_document_oscillation:
      ((3 ** rank) - 1) * commonFactor,
  };
})();

const artifact = JSON.parse(fs.readFileSync(
  "benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json",
  "utf8",
));

const signSummary = (values) => ({
  negative: values.filter((value) => value < 0).length,
  zero: values.filter((value) => value === 0).length,
  positive: values.filter((value) => value > 0).length,
  sum: values.reduce((sum, value) => sum + value, 0),
  absolute_sum: values.reduce((sum, value) => sum + Math.abs(value), 0),
});

const surfaceStructure = (surfaceName) => {
  const surface = artifact[surfaceName].cube;
  const interactions = surface.documents.map((document) => document.mobius_q20.trunk_head);
  const conditionals = surface.documents.map(
    (document) => document.conditional_trunk_after_head_q20,
  );
  const interaction = signSummary(interactions);
  const conditional = signSummary(conditionals);
  const activeConditionalTies = surface.documents.filter((document) => {
    assert(
      document.mobius_q20.trunk + document.mobius_q20.trunk_head
        === document.conditional_trunk_after_head_q20,
      `${surfaceName} conditional decomposition changed`,
    );
    return document.conditional_trunk_after_head_q20 === 0
      && (document.mobius_q20.trunk !== 0 || document.mobius_q20.trunk_head !== 0);
  }).length;
  assert(
    interaction.sum === surface.mobius_q20.trunk_head,
    `${surfaceName} document interactions do not sum to the aggregate coefficient`,
  );
  return {
    documents: surface.documents.length,
    interaction,
    document_submodularity_violations: interaction.positive,
    interaction_coherence_numerator: Math.abs(interaction.sum),
    interaction_coherence_denominator: interaction.absolute_sum,
    interaction_cancellation_numerator: interaction.absolute_sum - Math.abs(interaction.sum),
    interaction_cancellation_denominator: interaction.absolute_sum,
    conditional,
    active_conditional_ties: activeConditionalTies,
  };
};

const proposal = surfaceStructure("proposal");
const transfer = surfaceStructure("transfer");
assert(
  proposal.interaction.negative === 7
    && proposal.interaction.zero === 52
    && proposal.interaction.positive === 5
    && proposal.interaction.sum === -4
    && proposal.interaction.absolute_sum === 14
    && proposal.active_conditional_ties === 6,
  "proposal structural checkpoint changed",
);
assert(
  transfer.interaction.negative === 9
    && transfer.interaction.zero === 42
    && transfer.interaction.positive === 13
    && transfer.interaction.sum === 4
    && transfer.interaction.absolute_sum === 22
    && transfer.active_conditional_ties === 9,
  "transfer structural checkpoint changed",
);

const output = {
  schema: "nsrl.discrete_structure_theory_check.v1",
  interaction_tail_certificate: {
    cases: interactionTailCases,
    vertices: interactionTailVertices,
    uniform_truncation_error_bounded_by_absolute_tail: true,
    truncated_minimizer_global_gap_bounded_by_absolute_tail: true,
    expectation_commutation_cases: expectationCommutationCases,
    mobius_transform_commutes_with_environment_aggregation: true,
  },
  low_order_nonidentifiability: {
    rank: invisibleRank,
    audited_order: auditedOrder,
    indistinguishable_audited_vertices: true,
    hidden_global_improvement: "-7",
  },
  exchange_certificate: {
    approximate_exchange_cases: approximateExchangeCases,
    exchange_local_minima: approximateExchangeLocalMinima,
    global_gap_bound: "cardinality_times_uniform_exchange_defect",
    exact_exchange_cases: exactExchangeCases,
    zero_defect_local_implies_global_verified: true,
    surrogate_exchange_cases: surrogateExchangeCases,
    surrogate_exchange_local_minima: surrogateExchangeLocalMinima,
    transferred_gap_bound: "cardinality_times_surrogate_defect_plus_discrepancy_oscillation",
    exchange_defect_stability_cases: exchangeDefectStabilityCases,
    defect_change_bounded_by_twice_discrepancy_oscillation: true,
  },
  sparse_factor_optimization: {
    variable_elimination_cases: variableEliminationCases,
    maximum_verified_induced_width: maximumVerifiedInducedWidth,
    exact_minimum_matches_brute_force: true,
    width_tail_certificate_cases: widthTailCertificateCases,
    maximum_certified_surrogate_width: maximumCertifiedSurrogateWidth,
    population_regret_bounded_by_omitted_mass_and_interval_radii: true,
  },
  robust_surrogate_certificate: {
    retained_surrogate_cases: retainedSurrogateCases,
    adaptive_retained_support_cases: adaptiveRetainedSupportCases,
    regret_bounded_by_nonconstant_coefficient_l1_discrepancy: true,
    simultaneous_intervals_allow_data_dependent_support: true,
  },
  finite_sample_envelope: {
    coefficient_range_cases: coefficientRangeCases,
    order_u_absolute_coefficient_bound: "2^(u-1)_times_document_cube_oscillation",
    p10m_hoeffding_plan: p10mHoeffdingPlan,
  },
  observed_rank_two_structure: {
    proposal,
    transfer,
    aggregate_interaction_sign_reversed: proposal.interaction.sum < 0
      && transfer.interaction.sum > 0,
    stable_document_submodularity_supported: false,
  },
};

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
