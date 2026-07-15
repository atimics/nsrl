#!/usr/bin/env node

import { readFile } from "node:fs/promises";

const lutPath = process.argv[2] ?? "crates/nsrl-core/src/rsqrt_lut_8bit.rs";
const tokenStreamPath = process.argv[3]
  ?? "data/processed/production-corpus-v1/dev.nsrltok";
const source = await readFile(lutPath, "utf8");
const match = source.match(
  /EXP2_NEG_FRAC_LUT_8BIT: \[i16; 256\] = \[([\s\S]*?)\];/,
);
if (!match) throw new Error("EXP2_NEG_FRAC_LUT_8BIT not found");
const lut = [...match[1].matchAll(/\d+/g)].map((item) => BigInt(item[0]));
if (lut.length !== 256 || lut[0] !== 32767n) throw new Error("unexpected exponent LUT");

const wideBits = 47;
const wideLift = 1n << BigInt(wideBits - 15);
const wideWeight = (gapQ8) => {
  const exponent = Math.floor(gapQ8 / 256);
  const fraction = gapQ8 % 256;
  return (lut[fraction] * wideLift) >> BigInt(exponent);
};

let maximumIncrement = 0n;
let maximumIncrementGapQ8 = 0;
for (let gap = 1; gap <= (wideBits + 1) * 256; gap += 1) {
  const increment = wideWeight(gap - 1) - wideWeight(gap);
  if (increment < 0n) throw new Error("wide weights are not monotone");
  if (increment > maximumIncrement) {
    maximumIncrement = increment;
    maximumIncrementGapQ8 = gap;
  }
}

const maximumWeight = wideWeight(0);
const declaredIncrementBound = 88n * wideLift;
if (maximumIncrement > declaredIncrementBound) {
  throw new Error("one-cell wide-weight increment exceeds declared bound");
}

// Exact rational sandwich:
// maximum_increment / maximum_weight < 27/10000 < 2^(1/256)-1.
if (!(maximumIncrement * 10_000n < 27n * maximumWeight)) {
  throw new Error("LUT increment does not fit below 27/10000");
}
if (!(10_027n ** 256n < 2n * 10_000n ** 256n)) {
  throw new Error("27/10000 is not below the one-Q8 exponential step");
}

let systematicCases = 0;
let systematicPhases = 0;
let largestErrorNumerator = 0n;
let largestErrorDenominator = 1n;
for (let w0 = 0; w0 <= 4; w0 += 1) {
  for (let w1 = 0; w1 <= 4; w1 += 1) {
    for (let w2 = 0; w2 <= 4; w2 += 1) {
      const weights = [w0, w1, w2].map(BigInt);
      const total = weights.reduce((sum, value) => sum + value, 0n);
      if (total === 0n) continue;
      for (let mass = 1n; mass <= 16n; mass += 1n) {
        const countSums = [0n, 0n, 0n];
        const squaredErrorNumeratorSums = [0n, 0n, 0n];
        for (let phase = 0n; phase < total; phase += 1n) {
          let cumulative = 0n;
          let previous = phase / total;
          let allocated = 0n;
          for (let index = 0; index < weights.length; index += 1) {
            cumulative += weights[index];
            const next = (mass * cumulative + phase) / total;
            const count = next - previous;
            previous = next;
            allocated += count;
            countSums[index] += count;
            const errorNumerator = count * total - mass * weights[index];
            const absoluteErrorNumerator = errorNumerator < 0n
              ? -errorNumerator
              : errorNumerator;
            if (!(absoluteErrorNumerator < total)) {
              throw new Error("systematic coordinate error reached one count");
            }
            if (absoluteErrorNumerator * largestErrorDenominator
                > largestErrorNumerator * total) {
              largestErrorNumerator = absoluteErrorNumerator;
              largestErrorDenominator = total;
            }
            squaredErrorNumeratorSums[index] += errorNumerator * errorNumerator;
          }
          if (allocated !== mass) throw new Error("systematic mass is not exact");
          systematicPhases += 1;
        }
        for (let index = 0; index < weights.length; index += 1) {
          if (countSums[index] !== mass * weights[index]) {
            throw new Error("systematic allocation is biased over exact phases");
          }
          const quotaNumerator = mass * weights[index];
          const remainder = quotaNumerator % total;
          const expectedSquaredErrorNumeratorSum = total * remainder * (total - remainder);
          if (squaredErrorNumeratorSums[index] !== expectedSquaredErrorNumeratorSum) {
            throw new Error("systematic variance identity failed");
          }
        }
        systematicCases += 1;
      }
    }
  }
}

const vocabulary = 8192;
const proposalMassRows = [15, 16, 18, 20, 23].map((bits) => {
  const mass = 2 ** bits;
  return {
    bits,
    mass,
    uniform_expected_count: mass / vocabulary,
    systematic_normalized_l2_rms_bound: Math.sqrt(vocabulary / 4) / mass,
    categorical_normalized_l2_rms_bound: 1 / Math.sqrt(mass),
    grad_feature_absolute_bound_bits: bits + 16,
  };
});

const confidenceAlpha = 0.05;
const hoeffdingHalfWidth = (documents) =>
  Math.sqrt(Math.log(2 / confidenceAlpha) / (2 * documents));
const confidenceRows = [64, 128, 213, 738].map((documents) => ({
  documents,
  two_sided_alpha: confidenceAlpha,
  hoeffding_half_width: hoeffdingHalfWidth(documents),
}));
const documentsForFivePointWidth = Math.ceil(
  Math.log(2 / confidenceAlpha) / (2 * 0.05 ** 2),
);
if (documentsForFivePointWidth !== 738) {
  throw new Error("unexpected document count for five-point Hoeffding width");
}

const auditDocumentBlocks = async (path, context) => {
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (error) {
    if (error?.code === "ENOENT") return { path, available: false };
    throw error;
  }
  if (bytes.length < 24 || bytes.subarray(0, 8).toString("ascii") !== "NSRLTOK1") {
    throw new Error("unexpected token-stream header");
  }
  const tokenCount = Number(bytes.readBigUInt64LE(16));
  if (bytes.length !== 24 + tokenCount * 4) {
    throw new Error("unexpected token-stream length");
  }
  const bos = 256;
  const eos = 257;
  let active = false;
  let documentLength = 0;
  let documents = 0;
  let eligibleDocuments = 0;
  let slidingWindows = 0;
  for (let offset = 24; offset < bytes.length; offset += 4) {
    const token = bytes.readUInt32LE(offset);
    if (token === bos) {
      active = true;
      documentLength = 0;
    } else if (token === eos) {
      if (active) {
        documents += 1;
        if (documentLength > context) {
          eligibleDocuments += 1;
          slidingWindows += documentLength - context;
        }
      }
      active = false;
      documentLength = 0;
    } else if (active) {
      documentLength += 1;
    }
  }
  if (active) throw new Error("unterminated document in token stream");
  return {
    path,
    available: true,
    token_count_including_boundaries: tokenCount,
    documents,
    context,
    eligible_documents: eligibleDocuments,
    sliding_windows: slidingWindows,
  };
};

const corpusDocumentAudit = await auditDocumentBlocks(tokenStreamPath, 64);

const result = {
  schema: "nsrl.fixed_mass_theory_check.v1",
  lut_path: lutPath,
  wide_objective: {
    exponent_bits: wideBits,
    maximum_weight: maximumWeight.toString(),
    maximum_one_q8_increment: maximumIncrement.toString(),
    maximum_increment_gap_q8: maximumIncrementGapQ8,
    exact_rational_sandwich: [
      "maximum_increment/maximum_weight < 27/10000",
      "(10027/10000)^256 < 2",
    ],
    target_monotonicity_verified: true,
  },
  systematic_apportionment: {
    exhaustive_weight_domain: "three coordinates, each weight in [0,4], excluding all-zero",
    exhaustive_mass_domain: "K in [1,16]",
    cases: systematicCases,
    phases: systematicPhases,
    exact_mass_verified: true,
    exact_unbiasedness_over_uniform_integer_phase_verified: true,
    coordinate_error_strictly_below_one_verified: true,
    variance_identity_verified: true,
    maximum_observed_coordinate_error: Number(largestErrorNumerator)
      / Number(largestErrorDenominator),
  },
  p10m_proposal_mass_rows: proposalMassRows,
  document_confidence: {
    inference_unit: "independent document block",
    rows: confidenceRows,
    documents_for_half_width_at_most_0_05: documentsForFivePointWidth,
  },
  corpus_document_audit: corpusDocumentAudit,
};

process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
