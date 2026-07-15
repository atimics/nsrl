import crypto from "node:crypto";

export const RANK = 6;
export const VERTICES = 1 << RANK;

export const invariant = (condition, message) => {
  if (!condition) throw new Error(message);
};

export const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");

export const optimizerControlBinding = (sourceBytes) => {
  const source = sourceBytes.toString("utf8");
  const magic = source.match(/const OPTIMIZER_MAGIC: &\[u8; 8\] = b"([^";]+)";/);
  const version = source.match(/const OPTIMIZER_VERSION: u32 = ([0-9]+);/);
  invariant(magic !== null && version !== null, "optimizer state control constants missing");
  invariant(source.includes("integer_residual_sgd"), "default optimizer identity missing");
  const binding = {
    default_optimizer: "integer_residual_sgd",
    optimizer_state_magic_literal: magic[1],
    optimizer_state_version: Number(version[1]),
  };
  return {
    ...binding,
    semantic_sha256: sha256(Buffer.from(`${JSON.stringify(binding)}\n`)),
  };
};

export const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};

export const walshSign = (character, vertex) =>
  popcount(character & vertex) % 2 === 0 ? 1n : -1n;

const absolute = (value) => value < 0n ? -value : value;

const gcd = (left, right) => {
  let a = absolute(left);
  let b = absolute(right);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};

export const rational = (numerator, denominator) => {
  invariant(denominator !== 0n, "zero rational denominator");
  let n = numerator;
  let d = denominator;
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const divisor = gcd(n, d);
  return {
    numerator: (n / divisor).toString(),
    denominator: (d / divisor).toString(),
  };
};

export const isCanonicalRational = (value) => {
  if (typeof value !== "object" || value === null) return false;
  let numerator;
  let denominator;
  try {
    numerator = BigInt(value.numerator);
    denominator = BigInt(value.denominator);
  } catch {
    return false;
  }
  return denominator > 0n && gcd(numerator, denominator) === 1n;
};

export const walshTransform = (losses) => {
  invariant(losses.length === VERTICES, "rank-six loss cube required");
  return Array.from({length: VERTICES}, (_, character) =>
    losses.reduce(
      (sum, loss, vertex) => sum + loss * walshSign(character, vertex),
      0n,
    ));
};

export const walshReconstructionNumerators = (couplings) => {
  invariant(couplings.length === VERTICES, "rank-six Walsh field required");
  return Array.from({length: VERTICES}, (_, vertex) =>
    couplings.reduce(
      (sum, coupling, character) => sum + coupling * walshSign(character, vertex),
      0n,
    ));
};

const minimum = (values) =>
  values.reduce((left, right) => left < right ? left : right);

const maximum = (values) =>
  values.reduce((left, right) => left > right ? left : right);

const minimizingMasks = (values) => {
  const value = minimum(values);
  return values.flatMap((candidate, mask) => candidate === value ? [mask] : []);
};

export const buildIsingWalshReconstruction = (losses) => {
  const couplings = walshTransform(losses);
  const reconstructed = walshReconstructionNumerators(couplings);
  invariant(
    reconstructed.every((value, vertex) => value === BigInt(VERTICES) * losses[vertex]),
    "integer Walsh inversion failed",
  );
  const energyByOrder = Array(RANK + 1).fill(0n);
  for (let character = 0; character < VERTICES; character += 1) {
    energyByOrder[popcount(character)] += couplings[character] * couplings[character];
  }
  const parsevalLeft = energyByOrder.reduce((sum, value) => sum + value, 0n);
  const parsevalRight = BigInt(VERTICES)
    * losses.reduce((sum, value) => sum + value * value, 0n);
  invariant(parsevalLeft === parsevalRight, "integer Walsh Parseval identity failed");
  return {
    spin_convention: "s_i=(-1)^bit_i",
    hamiltonian: "loss(mask)=sum_A coupling_numerator[A]*product_(i in A)s_i/64",
    normalization_denominator: VERTICES,
    coupling_numerators: couplings.map(String),
    energy_numerator_squared_by_order: energyByOrder.map(String),
    nonzero_couplings: couplings.filter((value) => value !== 0n).length,
    ground_state_masks: minimizingMasks(losses),
    walsh_inversion_verified: true,
    parseval_verified: true,
  };
};

// This is an exact, integer, piecewise-linear interpolation between adjacent
// powers of two. With delta=q*T+r, the common-scale weight is proportional to
// 2^-q * (1-r/(2*T)). The frozen contract rejects exponent underflow.
export const dyadicBoltzmannWeight = (delta, temperature, weightShift) => {
  invariant(delta >= 0n, "negative energy above ground");
  invariant(temperature > 0n, "nonpositive temperature");
  const bucket = delta / temperature;
  const remainder = delta % temperature;
  invariant(bucket < BigInt(weightShift), "dyadic Boltzmann exponent underflow");
  const shift = BigInt(weightShift - Number(bucket) - 1);
  return (2n * temperature - remainder) << shift;
};

const spin = (mask, atom) => mask & (1 << atom) ? -1n : 1n;

const overlapSum = (left, right) => BigInt(RANK - 2 * popcount(left ^ right));

export const buildTemperatureSweep = (losses, temperatures, weightShift) => {
  invariant(losses.length === VERTICES, "rank-six thermal cube required");
  const ground = minimum(losses);
  const groundMasks = minimizingMasks(losses);
  const selectedGround = groundMasks[0];
  return temperatures.map((temperatureText) => {
    const temperature = BigInt(temperatureText);
    invariant(temperature > 0n, "temperature schedule must be positive");
    const deltas = losses.map((loss) => loss - ground);
    const weights = deltas.map(
      (delta) => dyadicBoltzmannWeight(delta, temperature, weightShift),
    );
    invariant(weights.every((weight) => weight > 0n), "thermal state lost a vertex");
    const partition = weights.reduce((sum, weight) => sum + weight, 0n);
    const groundWeight = groundMasks.reduce((sum, mask) => sum + weights[mask], 0n);
    const weightedDelta = weights.reduce(
      (sum, weight, mask) => sum + weight * deltas[mask],
      0n,
    );
    const weightedGroundOverlap = weights.reduce(
      (sum, weight, mask) => sum + weight * overlapSum(mask, selectedGround),
      0n,
    );
    const magnetizationNumerators = Array.from({length: RANK}, (_, atom) =>
      weights.reduce((sum, weight, mask) => sum + weight * spin(mask, atom), 0n));
    const weightedTotalMagnetization = weights.reduce((sum, weight, mask) => {
      const total = Array.from({length: RANK}, (_, atom) => spin(mask, atom))
        .reduce((left, right) => left + right, 0n);
      return sum + weight * total;
    }, 0n);
    const weightedTotalMagnetizationSquared = weights.reduce((sum, weight, mask) => {
      const total = Array.from({length: RANK}, (_, atom) => spin(mask, atom))
        .reduce((left, right) => left + right, 0n);
      return sum + weight * total * total;
    }, 0n);
    const pairCorrelationNumerators = [];
    for (let left = 0; left < RANK; left += 1) {
      for (let right = 0; right < RANK; right += 1) {
        pairCorrelationNumerators.push(weights.reduce(
          (sum, weight, mask) => sum + weight * spin(mask, left) * spin(mask, right),
          0n,
        ));
      }
    }
    const edwardsAndersonNumerator = magnetizationNumerators.reduce(
      (sum, value) => sum + value * value,
      0n,
    );
    const spinGlassNumerator = pairCorrelationNumerators.reduce(
      (sum, value) => sum + value * value,
      0n,
    );
    const magneticVarianceNumerator =
      weightedTotalMagnetizationSquared * partition
      - weightedTotalMagnetization * weightedTotalMagnetization;
    invariant(magneticVarianceNumerator >= 0n, "negative magnetic variance");
    return {
      temperature_units: temperature.toString(),
      maximum_energy_bucket: Number(maximum(deltas) / temperature),
      partition_weight: partition.toString(),
      minimum_vertex_weight: minimum(weights).toString(),
      maximum_vertex_weight: maximum(weights).toString(),
      ground_state_probability: rational(groundWeight, partition),
      expected_energy_above_ground: rational(weightedDelta, partition),
      selected_ground_state_overlap: rational(
        weightedGroundOverlap,
        BigInt(RANK) * partition,
      ),
      edwards_anderson_replica_overlap: rational(
        edwardsAndersonNumerator,
        BigInt(RANK) * partition * partition,
      ),
      magnetic_susceptibility_per_spin: rational(
        magneticVarianceNumerator,
        BigInt(RANK) * temperature * partition * partition,
      ),
      spin_glass_susceptibility: rational(
        spinGlassNumerator,
        BigInt(RANK) * partition * partition,
      ),
      magnetization_by_atom: magnetizationNumerators.map(
        (value) => rational(value, partition),
      ),
      all_vertex_weights_positive: true,
    };
  });
};

const roundNearestTiesAwayFromZero = (numerator, denominator) => {
  invariant(denominator > 0n, "nonpositive sigma-delta denominator");
  if (numerator >= 0n) return (numerator + denominator / 2n) / denominator;
  return -((-numerator + denominator / 2n) / denominator);
};

export const buildSigmaDeltaResidual = (losses, couplings, retainedDegree) => {
  invariant(retainedDegree >= 0 && retainedDegree < RANK, "invalid retained Walsh degree");
  const denominator = BigInt(VERTICES);
  const retained = Array.from({length: VERTICES}, (_, character) => character)
    .filter((character) => popcount(character) <= retainedDegree);
  const residualCharacters = Array.from({length: VERTICES}, (_, character) => character)
    .filter((character) => popcount(character) > retainedDegree);
  const surrogateNumerators = Array.from({length: VERTICES}, (_, vertex) =>
    retained.reduce(
      (sum, character) => sum + couplings[character] * walshSign(character, vertex),
      0n,
    ));
  const residualNumerators = Array.from({length: VERTICES}, (_, vertex) =>
    residualCharacters.reduce(
      (sum, character) => sum + couplings[character] * walshSign(character, vertex),
      0n,
    ));
  invariant(residualNumerators.every(
    (residual, vertex) => surrogateNumerators[vertex] + residual === denominator * losses[vertex]),
  "Walsh residual decomposition failed");

  let state = 0n;
  let emittedSum = 0n;
  let inputSum = 0n;
  let maximumAbsoluteState = 0n;
  const trace = [];
  for (let step = 0; step < VERTICES; step += 1) {
    const vertex = step ^ (step >>> 1);
    const input = residualNumerators[vertex];
    const before = state;
    const integrator = before + input;
    const emitted = roundNearestTiesAwayFromZero(integrator, denominator);
    state = integrator - emitted * denominator;
    inputSum += input;
    emittedSum += emitted;
    maximumAbsoluteState = maximumAbsoluteState > absolute(state)
      ? maximumAbsoluteState : absolute(state);
    trace.push([
      step,
      vertex,
      input.toString(),
      before.toString(),
      emitted.toString(),
      state.toString(),
    ]);
  }
  invariant(
    inputSum === denominator * emittedSum + state,
    "sigma-delta conservation identity failed",
  );
  invariant(
    maximumAbsoluteState <= denominator / 2n,
    "sigma-delta carry escaped the nearest-integer cell",
  );
  const surrogateMinimizers = minimizingMasks(surrogateNumerators);
  const exactMinimum = minimum(losses);
  const selectedMinimizer = surrogateMinimizers[0];
  const residualEnergy = residualCharacters.reduce(
    (sum, character) => sum + couplings[character] * couplings[character],
    0n,
  );
  return {
    retained_walsh_degree: retainedDegree,
    residual_characters: residualCharacters.length,
    vertex_order: "binary_reflected_gray_code",
    quantizer_denominator: VERTICES,
    rounding: "nearest_ties_away_from_zero",
    trace_columns: [
      "step",
      "vertex_mask",
      "input_residual_numerator",
      "accumulator_before",
      "emitted_integer_residual",
      "accumulator_after",
    ],
    residual_energy_numerator_squared: residualEnergy.toString(),
    residual_oscillation_numerator: (
      maximum(residualNumerators) - minimum(residualNumerators)
    ).toString(),
    surrogate_minimizers: surrogateMinimizers,
    selected_minimizer: selectedMinimizer,
    exact_gap: (losses[selectedMinimizer] - exactMinimum).toString(),
    input_sum_numerator: inputSum.toString(),
    emitted_sum: emittedSum.toString(),
    final_accumulator: state.toString(),
    maximum_absolute_accumulator: maximumAbsoluteState.toString(),
    conservation_verified: true,
    bounded_carry_verified: true,
    trace,
  };
};

export const buildObjectiveAudit = (objective, schedule, contract) => {
  const losses = objective.vertex_losses.map(BigInt);
  const reconstruction = buildIsingWalshReconstruction(losses);
  const couplings = reconstruction.coupling_numerators.map(BigInt);
  return {
    fractional_bits: objective.fractional_bits,
    ising_walsh: reconstruction,
    temperature_sweep: buildTemperatureSweep(
      losses,
      schedule,
      contract.arithmetic.dyadic_weight_shift,
    ),
    sigma_delta_residual: buildSigmaDeltaResidual(
      losses,
      couplings,
      contract.sigma_delta.retained_walsh_degree,
    ),
  };
};

export const encodeCanonicalJson = (value) => `${JSON.stringify(value, null, 2)}\n`;
