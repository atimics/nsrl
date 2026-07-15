#!/usr/bin/env node

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const floorDiv = (numerator, denominator) => {
  assert(denominator > 0n, "floorDiv requires a positive denominator");
  let quotient = numerator / denominator;
  const remainder = numerator % denominator;
  if (remainder < 0n) quotient -= 1n;
  return quotient;
};

const log2IntegerQ = (input, fractionalBits) => {
  const value = BigInt(input);
  assert(value > 0n, "integer log2 requires a positive value");
  const integerLog2 = BigInt(value.toString(2).length - 1);
  let normalizedQ63 = value << (63n - integerLog2);
  let fractional = 0n;
  for (let bit = fractionalBits - 1; bit >= 0; bit -= 1) {
    normalizedQ63 = (normalizedQ63 * normalizedQ63) >> 63n;
    if (normalizedQ63 >= (1n << 64n)) {
      normalizedQ63 >>= 1n;
      fractional |= 1n << BigInt(bit);
    }
  }
  return (integerLog2 << BigInt(fractionalBits)) | fractional;
};

const crossing = (before, after, cellWidth) =>
  floorDiv(after, cellWidth) - floorDiv(before, cellWidth);

let randomState = 0x9e3779b9;
const randomU32 = () => {
  randomState ^= randomState << 13;
  randomState ^= randomState >>> 17;
  randomState ^= randomState << 5;
  return randomState >>> 0;
};
const randomPositive = (maximum) => 1n + BigInt(randomU32() % maximum);

const coarseBits = 20;
const fineBits = 32;
const cellWidth = 1n << BigInt(fineBits - coarseBits);

let refinementCases = 0;
for (let value = 1n; value <= 4096n; value += 1n) {
  const coarse = log2IntegerQ(value, coarseBits);
  const refined = log2IntegerQ(value, fineBits);
  assert(
    coarse === refined >> BigInt(fineBits - coarseBits),
    "fine integer log2 did not refine the Q20 result exactly",
  );
  refinementCases += 1;
}

let phaseIdentityCases = 0;
for (let trial = 0; trial < 10000; trial += 1) {
  const before = BigInt(randomU32() % 1000000);
  const displacement = BigInt((randomU32() % 20001) - 10000);
  const after = before + displacement;
  const phase = before % cellWidth;
  const direct = crossing(before, after, cellWidth);
  const phaseForm = floorDiv(phase + displacement, cellWidth);
  assert(direct === phaseForm, "quantizer phase identity failed");
  phaseIdentityCases += 1;
}

let nllDecompositionCases = 0;
for (let trial = 0; trial < 5000; trial += 1) {
  const targetBefore = randomPositive(1 << 20);
  const targetAfter = randomPositive(1 << 20);
  const sumBefore = targetBefore + randomPositive(1 << 24);
  const sumAfter = targetAfter + randomPositive(1 << 24);

  const sumBeforeFine = log2IntegerQ(sumBefore, fineBits);
  const sumAfterFine = log2IntegerQ(sumAfter, fineBits);
  const targetBeforeFine = log2IntegerQ(targetBefore, fineBits);
  const targetAfterFine = log2IntegerQ(targetAfter, fineBits);

  const nllBefore = log2IntegerQ(sumBefore, coarseBits)
    - log2IntegerQ(targetBefore, coarseBits);
  const nllAfter = log2IntegerQ(sumAfter, coarseBits)
    - log2IntegerQ(targetAfter, coarseBits);
  const decomposed = crossing(sumBeforeFine, sumAfterFine, cellWidth)
    - crossing(targetBeforeFine, targetAfterFine, cellWidth);
  assert(
    nllAfter - nllBefore === decomposed,
    "Q20 NLL contrast did not decompose into denominator and numerator crossings",
  );
  nllDecompositionCases += 1;
}

const phaseMixingCellWidth = 64n;
let phaseMixingCases = 0;
for (
  let displacement = -phaseMixingCellWidth;
  displacement <= phaseMixingCellWidth;
  displacement += 1n
) {
  let signedCrossingSum = 0n;
  let visiblePhases = 0n;
  for (let phase = 0n; phase < phaseMixingCellWidth; phase += 1n) {
    const value = floorDiv(phase + displacement, phaseMixingCellWidth);
    signedCrossingSum += value;
    if (value !== 0n) visiblePhases += 1n;
  }
  assert(
    signedCrossingSum === displacement,
    "uniform-phase crossing expectation is not displacement/cell width",
  );
  const absoluteDisplacement = displacement < 0n ? -displacement : displacement;
  assert(
    visiblePhases === absoluteDisplacement,
    "uniform-phase visibility is not |displacement|/cell width inside one cell",
  );
  phaseMixingCases += 1;
}

const componentCrossings = [1n, -1n, 0n, 0n];
const componentActivity = componentCrossings.reduce(
  (sum, value) => sum + (value < 0n ? -value : value),
  0n,
);
const documentContrast = componentCrossings.reduce((sum, value) => sum + value, 0n);
assert(componentActivity === 2n, "component activity example changed");
assert(documentContrast === 0n, "component cancellation example is not a document tie");

const output = {
  schema: "nsrl.objective_boundary_phase_theory_check.v1",
  exact_log_refinement: {
    coarse_fractional_bits: coarseBits,
    fine_fractional_bits: fineBits,
    cases: refinementCases,
    fine_log_downshift_equals_q20_log: true,
  },
  quantizer_phase_calculus: {
    identity_cases: phaseIdentityCases,
    crossing_identity: "Q(y+h)-Q(y)=floor((phase(y)+h)/cell_width)",
  },
  nll_boundary_decomposition: {
    cases: nllDecompositionCases,
    denominator_minus_numerator_crossings: true,
  },
  uniform_phase_model: {
    cases: phaseMixingCases,
    expected_crossing: "displacement/cell_width",
    one_cell_visibility: "abs(displacement)/cell_width",
  },
  document_cancellation: {
    component_crossings: componentCrossings.map(String),
    component_activity: String(componentActivity),
    document_contrast: String(documentContrast),
    active_components_can_sum_to_a_tie: true,
  },
};

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
