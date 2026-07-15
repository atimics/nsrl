#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";

import {optimizerControlBinding} from "./lib/production-atomic-ising-v1.mjs";

const sourcePath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const sourceContractPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json";
const contractPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json";
const resultPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-audit-v1.json";
const kernelPath = new URL("./lib/production-atomic-ising-v1.mjs", import.meta.url);
const analyzerPath = new URL("./analyze-production-atomic-ising-v1.mjs", import.meta.url);
const optimizerPath = new URL(
  "../crates/nsrl-train/src/production/training.rs",
  import.meta.url,
);

const sourceBytes = fs.readFileSync(sourcePath);
const sourceContractBytes = fs.readFileSync(sourceContractPath);
const contractBytes = fs.readFileSync(contractPath);
const resultBytes = fs.readFileSync(resultPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
const contract = JSON.parse(contractBytes.toString("utf8"));
const result = JSON.parse(resultBytes.toString("utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const sign = (character, vertex) =>
  popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const absolute = (value) => value < 0n ? -value : value;
const gcd = (left, right) => {
  let a = absolute(left);
  let b = absolute(right);
  while (b !== 0n) [a, b] = [b, a % b];
  return a;
};
const canonicalRational = (value) => {
  const numerator = BigInt(value.numerator);
  const denominator = BigInt(value.denominator);
  return denominator > 0n && gcd(numerator, denominator) === 1n;
};
const equalRational = (value, numerator, denominator) => {
  assert(canonicalRational(value), "noncanonical rational metric");
  return BigInt(value.numerator) * denominator === numerator * BigInt(value.denominator);
};
const roundNearest = (numerator, denominator) => numerator >= 0n
  ? (numerator + denominator / 2n) / denominator
  : -((-numerator + denominator / 2n) / denominator);
const dyadicWeight = (delta, temperature, shift) => {
  const bucket = delta / temperature;
  const remainder = delta % temperature;
  assert(bucket < BigInt(shift), "temperature weight underflow");
  return (2n * temperature - remainder) << BigInt(shift - Number(bucket) - 1);
};
const spin = (mask, atom) => mask & (1 << atom) ? -1n : 1n;

assert(contract.schema === "nsrl.production_atomic_ising_audit_contract.v1",
  "wrong audit contract schema");
assert(result.schema === "nsrl.production_atomic_ising_audit.v1", "wrong result schema");
assert(contract.analysis_role === "proposal_only_calibration"
  && result.analysis_role === "proposal_only_calibration", "audit is not proposal-only");
assert(sha256(sourceBytes) === contract.source.result_sha256, "source result hash mismatch");
assert(sha256(sourceContractBytes) === contract.source.contract_sha256,
  "source contract hash mismatch");
assert(sha256(contractBytes) === result.audit_contract_sha256, "audit contract hash mismatch");
assert(result.source_result_sha256 === contract.source.result_sha256
  && result.source_contract_sha256 === contract.source.contract_sha256,
"result source bindings changed");
assert(sha256(fs.readFileSync(kernelPath)) === contract.implementation.kernel_sha256,
  "kernel hash mismatch");
assert(sha256(fs.readFileSync(analyzerPath)) === contract.implementation.analyzer_sha256,
  "analyzer hash mismatch");
assert(sha256(fs.readFileSync(new URL(import.meta.url)))
  === contract.implementation.checker_sha256, "checker hash mismatch");
assert(result.implementation.kernel_sha256 === contract.implementation.kernel_sha256
  && result.implementation.analyzer_sha256 === contract.implementation.analyzer_sha256
  && result.implementation.checker_sha256 === contract.implementation.checker_sha256,
"result implementation bindings changed");
const optimizerControl = optimizerControlBinding(fs.readFileSync(optimizerPath));
assert(optimizerControl.semantic_sha256
  === contract.control.optimizer_control_semantic_sha256,
"optimizer control tuple changed before audit replay");
assert(optimizerControl.default_optimizer === contract.control.default_optimizer
  && optimizerControl.optimizer_state_magic_literal
    === contract.control.optimizer_state_magic_literal
  && optimizerControl.optimizer_state_version === contract.control.optimizer_state_version,
"optimizer control fields changed before audit replay");
assert(contract.control.default_optimizer === "integer_residual_sgd",
  "frozen default optimizer changed");
assert(contract.control.optimizer_change_authorized === false
  && result.decision.optimizer_change_authorized === false,
"audit authorized an optimizer change");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 0,
  "source crossed proposal firewall");
assert(result.bindings.manifest_hash === contract.source.manifest_hash,
  "result manifest binding changed");
assert(result.rank === 6 && result.vertices === 64, "wrong audit cube shape");

const checkObjective = (sourceObjective, audit, schedule, label) => {
  const losses = sourceObjective.vertex_losses.map(BigInt);
  const couplings = audit.ising_walsh.coupling_numerators.map(BigInt);
  assert(couplings.length === 64, `${label} Ising field shape mismatch`);
  for (let vertex = 0; vertex < 64; vertex += 1) {
    const reconstructed = couplings.reduce(
      (sum, value, character) => sum + value * sign(character, vertex),
      0n,
    );
    assert(reconstructed === 64n * losses[vertex], `${label} Walsh inversion failed`);
  }
  const parsevalLeft = couplings.reduce((sum, value) => sum + value * value, 0n);
  const parsevalRight = 64n * losses.reduce((sum, value) => sum + value * value, 0n);
  assert(parsevalLeft === parsevalRight && audit.ising_walsh.parseval_verified,
    `${label} Parseval check failed`);

  const minimumLoss = losses.reduce((left, right) => left < right ? left : right);
  const groundMasks = losses.flatMap((loss, mask) => loss === minimumLoss ? [mask] : []);
  const selectedGround = groundMasks[0];
  assert(JSON.stringify(groundMasks) === JSON.stringify(audit.ising_walsh.ground_state_masks),
    `${label} ground states changed`);
  assert(audit.temperature_sweep.length === schedule.length,
    `${label} temperature schedule length mismatch`);
  for (let index = 0; index < schedule.length; index += 1) {
    const temperature = BigInt(schedule[index]);
    const recorded = audit.temperature_sweep[index];
    assert(recorded.temperature_units === schedule[index], `${label} temperature changed`);
    const deltas = losses.map((loss) => loss - minimumLoss);
    const weights = deltas.map(
      (delta) => dyadicWeight(delta, temperature, contract.arithmetic.dyadic_weight_shift),
    );
    const partition = weights.reduce((sum, weight) => sum + weight, 0n);
    assert(BigInt(recorded.partition_weight) === partition, `${label} partition mismatch`);
    assert(BigInt(recorded.minimum_vertex_weight)
      === weights.reduce((left, right) => left < right ? left : right),
    `${label} minimum thermal weight mismatch`);
    assert(recorded.all_vertex_weights_positive && weights.every((weight) => weight > 0n),
      `${label} thermal support is incomplete`);
    const groundWeight = groundMasks.reduce((sum, mask) => sum + weights[mask], 0n);
    assert(equalRational(recorded.ground_state_probability, groundWeight, partition),
      `${label} ground-state probability mismatch`);
    const weightedDelta = weights.reduce(
      (sum, weight, mask) => sum + weight * deltas[mask], 0n);
    assert(equalRational(recorded.expected_energy_above_ground, weightedDelta, partition),
      `${label} expected energy mismatch`);
    const overlap = weights.reduce(
      (sum, weight, mask) => sum + weight * BigInt(6 - 2 * popcount(mask ^ selectedGround)),
      0n);
    assert(equalRational(recorded.selected_ground_state_overlap, overlap, 6n * partition),
      `${label} ground-state overlap mismatch`);
    const moments = Array.from({length: 6}, (_, atom) => weights.reduce(
      (sum, weight, mask) => sum + weight * spin(mask, atom), 0n));
    assert(recorded.magnetization_by_atom.every(
      (value, atom) => equalRational(value, moments[atom], partition)),
    `${label} atom magnetization mismatch`);
    const edwardsAnderson = moments.reduce((sum, value) => sum + value * value, 0n);
    assert(equalRational(
      recorded.edwards_anderson_replica_overlap,
      edwardsAnderson,
      6n * partition * partition,
    ), `${label} replica overlap mismatch`);
    const weightedM = weights.reduce((sum, weight, mask) => {
      const magnetization = Array.from({length: 6}, (_, atom) => spin(mask, atom))
        .reduce((left, right) => left + right, 0n);
      return sum + weight * magnetization;
    }, 0n);
    const weightedM2 = weights.reduce((sum, weight, mask) => {
      const magnetization = Array.from({length: 6}, (_, atom) => spin(mask, atom))
        .reduce((left, right) => left + right, 0n);
      return sum + weight * magnetization * magnetization;
    }, 0n);
    assert(equalRational(
      recorded.magnetic_susceptibility_per_spin,
      weightedM2 * partition - weightedM * weightedM,
      6n * temperature * partition * partition,
    ), `${label} magnetic susceptibility mismatch`);
    let pairSquares = 0n;
    for (let left = 0; left < 6; left += 1) {
      for (let right = 0; right < 6; right += 1) {
        const correlation = weights.reduce(
          (sum, weight, mask) => sum + weight * spin(mask, left) * spin(mask, right),
          0n);
        pairSquares += correlation * correlation;
      }
    }
    assert(equalRational(
      recorded.spin_glass_susceptibility,
      pairSquares,
      6n * partition * partition,
    ), `${label} spin-glass susceptibility mismatch`);
  }

  const sigma = audit.sigma_delta_residual;
  assert(sigma.trace.length === 64 && sigma.quantizer_denominator === 64,
    `${label} sigma-delta trace shape mismatch`);
  let state = 0n;
  let inputSum = 0n;
  let emittedSum = 0n;
  for (let step = 0; step < 64; step += 1) {
    const vertex = step ^ (step >>> 1);
    const residual = couplings.reduce((sum, value, character) =>
      popcount(character) > sigma.retained_walsh_degree
        ? sum + value * sign(character, vertex) : sum, 0n);
    const row = sigma.trace[step];
    assert(row[0] === step && row[1] === vertex, `${label} sigma-delta Gray order changed`);
    assert(BigInt(row[2]) === residual && BigInt(row[3]) === state,
      `${label} sigma-delta input mismatch`);
    const emitted = roundNearest(state + residual, 64n);
    state = state + residual - emitted * 64n;
    assert(BigInt(row[4]) === emitted && BigInt(row[5]) === state,
      `${label} sigma-delta state transition mismatch`);
    assert(absolute(state) <= 32n, `${label} sigma-delta carry bound failed`);
    inputSum += residual;
    emittedSum += emitted;
  }
  assert(inputSum === 64n * emittedSum + state,
    `${label} sigma-delta conservation failed`);
  assert(BigInt(sigma.input_sum_numerator) === inputSum
    && BigInt(sigma.emitted_sum) === emittedSum
    && BigInt(sigma.final_accumulator) === state
    && sigma.conservation_verified === true
    && sigma.bounded_carry_verified === true,
  `${label} sigma-delta summary mismatch`);
};

checkObjective(source.q20, result.q20, contract.temperature_sweep.q20_temperature_units, "Q20");
checkObjective(source.q32, result.q32, contract.temperature_sweep.q32_temperature_units, "Q32");
assert(Object.values(result.gates).every((value) => value === true), "an audit gate failed");
assert(result.decision.audit_contract_passed === true
  && result.decision.structure_certificate_selected === false
  && result.decision.paid_scaling_authorized === false,
"audit decision escaped the frozen control");

const replayDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "nsrl-ising-replay-"));
const replayPath = path.join(replayDirectory, "replay.json");
try {
  const replay = spawnSync(
    process.execPath,
    [fileURLToPath(analyzerPath), sourcePath, contractPath, replayPath],
    {encoding: "utf8"},
  );
  assert(replay.status === 0, `byte replay failed: ${replay.stderr || replay.stdout}`);
  const replayBytes = fs.readFileSync(replayPath);
  assert(resultBytes.equals(replayBytes), "audit artifact is not byte-replayable");
} finally {
  fs.rmSync(replayDirectory, {recursive: true, force: true});
}

process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_audit_check.v1",
  contract_sha256: sha256(contractBytes),
  result_sha256: sha256(resultBytes),
  objectives: 2,
  temperatures_per_objective: result.q20.temperature_sweep.length,
  sigma_delta_rows_per_objective: result.q20.sigma_delta_residual.trace.length,
  integer_walsh_reconstruction_verified: true,
  overlap_and_susceptibility_metrics_verified: true,
  byte_replay_verified: true,
  default_optimizer: contract.control.default_optimizer,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
