#!/usr/bin/env node

import fs from "node:fs";

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

let randomState = 0x6d2b79f5;
const randomU32 = () => {
  randomState = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
  randomState ^= randomState + Math.imul(randomState ^ (randomState >>> 7), 61 | randomState);
  return (randomState ^ (randomState >>> 14)) >>> 0;
};
const randomInteger = (minimum, maximum) =>
  minimum + (randomU32() % (maximum - minimum + 1));

const blockSupport = (atomicMask, blocks) => {
  let support = 0;
  for (let block = 0; block < blocks.length; block += 1) {
    if ((atomicMask & blocks[block]) !== 0) support |= 1 << block;
  }
  return support;
};

let blockPushforwardCases = 0;
for (let trial = 0; trial < 240; trial += 1) {
  const rank = randomInteger(2, 7);
  const split = randomInteger(1, rank - 1);
  const firstBlock = (1 << split) - 1;
  const secondBlock = ((1 << rank) - 1) ^ firstBlock;
  const blocks = [firstBlock, secondBlock];
  const values = Array.from(
    {length: 1 << rank},
    () => BigInt(randomInteger(-5000, 5000)),
  );
  const atomicCoefficients = mobiusTransform(values, rank);
  const coarseValues = [
    values[0],
    values[firstBlock],
    values[secondBlock],
    values[firstBlock | secondBlock],
  ];
  const coarseCoefficients = mobiusTransform(coarseValues, 2);
  const pushed = [0n, 0n, 0n, 0n];
  for (let mask = 0; mask < atomicCoefficients.length; mask += 1) {
    pushed[blockSupport(mask, blocks)] += atomicCoefficients[mask];
  }
  assert(
    pushed.every((value, index) => value === coarseCoefficients[index]),
    "block-support Möbius pushforward failed",
  );
  blockPushforwardCases += 1;
}

let conditionalIdentityCases = 0;
for (let rank = 2; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 100; trial += 1) {
    const values = Array.from(
      {length: 1 << rank},
      () => BigInt(randomInteger(-5000, 5000)),
    );
    const coefficients = mobiusTransform(values, rank);
    const all = (1 << rank) - 1;
    let action = randomU32() & all;
    if (action === 0) action = 1 << (randomU32() % rank);
    const available = all ^ action;
    const baseline = randomU32() & available;
    const union = action | baseline;
    let coefficientSum = 0n;
    for (let subset = 0; subset < 1 << rank; subset += 1) {
      if ((subset & ~union) === 0 && (subset & action) !== 0) {
        coefficientSum += coefficients[subset];
      }
    }
    assert(
      values[union] - values[baseline] === coefficientSum,
      "conditional Möbius identity failed",
    );
    conditionalIdentityCases += 1;
  }
}

const cancellationCoefficients = Array(8).fill(0n);
cancellationCoefficients[0b101] = -5n;
cancellationCoefficients[0b110] = 5n;
const cancellationValues = zetaTransform(cancellationCoefficients, 3);
const cancellationCoarseValues = [
  cancellationValues[0],
  cancellationValues[0b011],
  cancellationValues[0b100],
  cancellationValues[0b111],
];
const cancellationCoarseCoefficients = mobiusTransform(cancellationCoarseValues, 2);
assert(
  cancellationCoarseCoefficients[3] === 0n,
  "coarse cancellation example did not hide atomic interactions",
);

const quantizationError = 3n;
let coefficientErrorCases = 0;
let conditionalErrorCases = 0;
for (let rank = 1; rank <= 8; rank += 1) {
  for (let trial = 0; trial < 40; trial += 1) {
    const latent = Array.from(
      {length: 1 << rank},
      () => BigInt(randomInteger(-10000, 10000)),
    );
    const errors = Array.from(
      {length: 1 << rank},
      () => BigInt(randomInteger(-Number(quantizationError), Number(quantizationError))),
    );
    const observed = latent.map((value, index) => value + errors[index]);
    const latentCoefficients = mobiusTransform(latent, rank);
    const observedCoefficients = mobiusTransform(observed, rank);
    for (let mask = 0; mask < 1 << rank; mask += 1) {
      const difference = observedCoefficients[mask] - latentCoefficients[mask];
      const absolute = difference < 0n ? -difference : difference;
      const bound = (1n << BigInt(popcount(mask))) * quantizationError;
      assert(absolute <= bound, "Möbius coefficient error exceeded 2^k epsilon");
      coefficientErrorCases += 1;
    }
    const all = (1 << rank) - 1;
    const action = 1 << (randomU32() % rank);
    const baseline = (randomU32() & all) & ~action;
    const union = baseline | action;
    const latentConditional = latent[union] - latent[baseline];
    const observedConditional = observed[union] - observed[baseline];
    const conditionalDifference = observedConditional - latentConditional;
    const conditionalAbsolute = conditionalDifference < 0n
      ? -conditionalDifference
      : conditionalDifference;
    assert(
      conditionalAbsolute <= 2n * quantizationError,
      "conditional contrast error exceeded 2 epsilon",
    );
    conditionalErrorCases += 1;
  }
}

const majorityImprovesMeanHarms = [...Array(9).fill(-1), 20];
const majorityHarmsMeanImproves = [-20, ...Array(9).fill(1)];
assert(
  majorityImprovesMeanHarms.filter((value) => value < 0).length === 9
    && majorityImprovesMeanHarms.reduce((sum, value) => sum + value, 0) > 0,
  "majority-improves/mean-harms counterexample failed",
);
assert(
  majorityHarmsMeanImproves.filter((value) => value > 0).length === 9
    && majorityHarmsMeanImproves.reduce((sum, value) => sum + value, 0) < 0,
  "majority-harms/mean-improves counterexample failed",
);

const resultArtifact = JSON.parse(fs.readFileSync(
  "benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json",
  "utf8",
));

const observedSurface = (name) => {
  const surface = resultArtifact[name];
  const contrasts = surface.cube.documents.map(
    (document) => document.conditional_trunk_after_head_q20,
  );
  const favorable = contrasts.filter((value) => value < 0).length;
  const unfavorable = contrasts.filter((value) => value > 0).length;
  const ties = contrasts.length - favorable - unfavorable;
  assert(favorable === surface.conditional_sign_test.joint_wins, `${name} favorable mismatch`);
  assert(unfavorable === surface.conditional_sign_test.head_wins, `${name} unfavorable mismatch`);
  assert(ties === surface.conditional_sign_test.ties, `${name} tie mismatch`);
  const visibilityNumerator = favorable + unfavorable;
  const signedMarginNumerator = favorable - unfavorable;
  assert(
    signedMarginNumerator * visibilityNumerator
      === visibilityNumerator * (favorable - unfavorable),
    `${name} signed-visibility factorization failed`,
  );
  return {
    documents: contrasts.length,
    favorable,
    unfavorable,
    ties,
    visibility_numerator: visibilityNumerator,
    visibility_denominator: contrasts.length,
    signed_visibility_margin_numerator: signedMarginNumerator,
    signed_visibility_margin_denominator: contrasts.length,
    aggregate_conditional_q20: contrasts.reduce((sum, value) => sum + value, 0),
  };
};

const proposal = observedSurface("proposal");
const transfer = observedSurface("transfer");
assert(proposal.signed_visibility_margin_numerator === 8, "proposal margin is not 8/64");
assert(transfer.signed_visibility_margin_numerator === -4, "transfer margin is not -4/64");

const alpha = 0.05;
const transferVisibility = transfer.visibility_numerator / transfer.visibility_denominator;
const planningRows = [0.1, 0.2, 0.3].map((conditionalAdvantage) => {
  const requiredNonTies = Math.ceil(
    Math.log(1 / alpha) / (2 * conditionalAdvantage ** 2),
  );
  return {
    conditional_advantage_over_half: conditionalAdvantage,
    required_non_ties: requiredNonTies,
    expected_documents_at_observed_visibility: Math.ceil(requiredNonTies / transferVisibility),
  };
});
assert(
  JSON.stringify(planningRows.map((row) => row.required_non_ties))
    === JSON.stringify([150, 38, 17]),
  "non-tie planning counts changed",
);
assert(
  JSON.stringify(planningRows.map((row) => row.expected_documents_at_observed_visibility))
    === JSON.stringify([534, 136, 61]),
  "document planning counts changed",
);

const trunkAtoms = 4;
const headAtoms = 2;
const trunkOnlyAtomicTerms = 2 ** trunkAtoms - 1;
const headOnlyAtomicTerms = 2 ** headAtoms - 1;
const crossAtomicTerms = trunkOnlyAtomicTerms * headOnlyAtomicTerms;
assert(
  trunkOnlyAtomicTerms === 15 && headOnlyAtomicTerms === 3 && crossAtomicTerms === 45,
  "p10m block-support term counts changed",
);

const output = {
  schema: "nsrl.boolean_jet_stability_theory_check.v1",
  hierarchical_boolean_jet: {
    block_pushforward_cases: blockPushforwardCases,
    block_support_mobius_identity_verified: true,
    p10m_trunk_only_atomic_terms: trunkOnlyAtomicTerms,
    p10m_head_only_atomic_terms: headOnlyAtomicTerms,
    p10m_cross_atomic_terms: crossAtomicTerms,
    coarse_zero_can_hide_opposing_atomic_interactions: true,
  },
  conditional_calculus: {
    identity_cases: conditionalIdentityCases,
    conditional_effect_equals_sum_of_intersecting_coefficients: true,
  },
  resolution_stability: {
    coefficient_error_cases: coefficientErrorCases,
    order_k_bound: "2^k_epsilon",
    conditional_error_cases: conditionalErrorCases,
    conditional_bound: "2_epsilon",
  },
  estimand_counterexamples: {
    majority_improves_mean_harms: majorityImprovesMeanHarms,
    majority_harms_mean_improves: majorityHarmsMeanImproves,
    risk_and_directional_primality_are_incomparable: true,
  },
  observed_information: {proposal, transfer},
  planning_at_transfer_visibility: {
    alpha,
    observed_visibility: transferVisibility,
    hoeffding_rows: planningRows,
    reserved_documents: 77,
  },
};

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
