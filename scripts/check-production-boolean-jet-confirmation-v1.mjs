#!/usr/bin/env node

import fs from "node:fs";

const contractPath =
  "benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1-contract.json";
const resultPath =
  "benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json";

const contract = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const result = JSON.parse(fs.readFileSync(resultPath, "utf8"));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function gcd(left, right) {
  while (right !== 0n) [left, right] = [right, left % right];
  return left;
}

function binomial(n, k) {
  k = Math.min(k, n - k);
  let value = 1n;
  for (let index = 1; index <= k; index += 1) {
    value = (value * BigInt(n - index + 1)) / BigInt(index);
  }
  return value;
}

function exactTwoSidedSignP(leftWins, rightWins) {
  const n = leftWins + rightWins;
  if (n === 0) return [1n, 1n];
  const tail = Math.min(leftWins, rightWins);
  let numerator = 0n;
  for (let index = 0; index <= tail; index += 1) {
    numerator += binomial(n, index);
  }
  numerator = (2n * numerator) < (1n << BigInt(n))
    ? 2n * numerator
    : 1n << BigInt(n);
  const denominator = 1n << BigInt(n);
  const divisor = gcd(numerator, denominator);
  return [numerator / divisor, denominator / divisor];
}

function checkCube(surface, expectedBlock) {
  assert(surface.document_block.start === expectedBlock.document_start,
    `${surface.cube.surface}: wrong document start`);
  assert(surface.document_block.count === expectedBlock.document_count,
    `${surface.cube.surface}: wrong document count`);
  assert(surface.document_block.windows_per_document === expectedBlock.windows_per_document,
    `${surface.cube.surface}: wrong windows per document`);

  const cube = surface.cube;
  assert(cube.reconstruction_verified === true,
    `${cube.surface}: aggregate reconstruction not verified`);
  assert(cube.vertices.length === 4, `${cube.surface}: expected four vertices`);
  const losses = Object.fromEntries(cube.vertices.map((vertex) => [vertex.vertex, vertex.nll_q20]));
  const muTrunk = losses.trunk - losses.empty;
  const muHead = losses.head - losses.empty;
  const muPair = losses.trunk_head - losses.trunk - losses.head + losses.empty;
  assert(cube.mobius_q20.trunk === muTrunk, `${cube.surface}: trunk coefficient mismatch`);
  assert(cube.mobius_q20.head === muHead, `${cube.surface}: head coefficient mismatch`);
  assert(cube.mobius_q20.trunk_head === muPair, `${cube.surface}: pair coefficient mismatch`);
  assert(cube.joint_delta_q20 === muTrunk + muHead + muPair,
    `${cube.surface}: aggregate Möbius inversion failed`);
  assert(cube.gamma_one_q20 === Math.min(muTrunk, muHead),
    `${cube.surface}: gamma-one mismatch`);

  assert(cube.documents.length === expectedBlock.document_count,
    `${cube.surface}: wrong document trace count`);
  const sums = {empty: 0, trunk: 0, head: 0, trunk_head: 0};
  let jointWins = 0;
  let headWins = 0;
  let ties = 0;
  for (const [offset, document] of cube.documents.entries()) {
    assert(document.document === expectedBlock.document_start + offset,
      `${cube.surface}: noncanonical document order`);
    assert(document.windows === expectedBlock.windows_per_document,
      `${cube.surface}: document window count mismatch`);
    assert(document.reconstruction_verified === true,
      `${cube.surface}: document reconstruction not verified`);
    const values = document.vertex_nll_q20;
    const documentMuTrunk = values.trunk - values.empty;
    const documentMuHead = values.head - values.empty;
    const documentMuPair = values.trunk_head - values.trunk - values.head + values.empty;
    assert(document.mobius_q20.trunk === documentMuTrunk,
      `${cube.surface}: document trunk coefficient mismatch`);
    assert(document.mobius_q20.head === documentMuHead,
      `${cube.surface}: document head coefficient mismatch`);
    assert(document.mobius_q20.trunk_head === documentMuPair,
      `${cube.surface}: document pair coefficient mismatch`);
    const conditional = values.trunk_head - values.head;
    assert(document.conditional_trunk_after_head_q20 === conditional,
      `${cube.surface}: conditional contrast mismatch`);
    if (conditional < 0) jointWins += 1;
    else if (conditional > 0) headWins += 1;
    else ties += 1;
    for (const vertex of Object.keys(sums)) sums[vertex] += values[vertex];
  }
  for (const vertex of cube.vertices) {
    assert(sums[vertex.vertex] === vertex.nll_q20,
      `${cube.surface}: document losses do not sum to aggregate vertex`);
  }

  const sign = surface.conditional_sign_test;
  assert(sign.joint_wins === jointWins, `${cube.surface}: joint-win count mismatch`);
  assert(sign.head_wins === headWins, `${cube.surface}: head-win count mismatch`);
  assert(sign.ties === ties, `${cube.surface}: tie count mismatch`);
  assert(sign.non_ties === jointWins + headWins, `${cube.surface}: non-tie count mismatch`);
  const [pNumerator, pDenominator] = exactTwoSidedSignP(jointWins, headWins);
  assert(BigInt(sign.exact_p_numerator) === pNumerator,
    `${cube.surface}: sign-test numerator mismatch`);
  assert(BigInt(sign.exact_p_denominator) === pDenominator,
    `${cube.surface}: sign-test denominator mismatch`);
  const pPerMillion = Number((pNumerator * 1_000_000n + pDenominator / 2n) / pDenominator);
  assert(sign.p_per_million === pPerMillion, `${cube.surface}: rounded p mismatch`);
  return sign;
}

assert(contract.schema === "nsrl.production_boolean_jet_confirmation_contract.v1",
  "wrong confirmation contract schema");
assert(result.schema === contract.trace_schema, "trace schema does not match contract");
assert(result.analysis_role === "confirmation", "result is not a confirmation trace");
assert(result.profile === contract.profile, "profile mismatch");
for (const key of ["model_hash", "tokenizer_hash", "token_stream_hash"]) {
  assert(result.bindings[key] === contract.bindings[key], `${key} mismatch`);
}
assert(result.move_contract.move_fingerprint === contract.move_manifest.move_fingerprint,
  "move fingerprint mismatch");
assert(result.move_contract.manifest_hash === contract.move_manifest.manifest_hash,
  "manifest hash mismatch");
for (const key of ["algorithm", "version", "fractional_bits", "zero_probability_floor_q20"]) {
  assert(result.objective[key] === contract.objective[key], `objective ${key} mismatch`);
}

const proposalSign = checkCube(result.proposal, contract.surfaces.proposal_diagnostic);
const transferSign = checkCube(result.transfer, contract.surfaces.transfer_primary);
const proposalEnd = contract.surfaces.proposal_diagnostic.document_start
  + contract.surfaces.proposal_diagnostic.document_count;
assert(proposalEnd <= contract.surfaces.transfer_primary.document_start,
  "proposal and transfer document ranges overlap");
const transferEnd = contract.surfaces.transfer_primary.document_start
  + contract.surfaces.transfer_primary.document_count;
assert(transferEnd <= contract.surfaces.reserved_replication.document_start,
  "transfer and reserved replication ranges overlap");

const threshold = contract.decision_rule;
const transferDirection = transferSign.joint_wins > transferSign.head_wins;
const transferSignificant = BigInt(transferSign.exact_p_numerator)
  * BigInt(threshold.significance_denominator)
  <= BigInt(threshold.significance_numerator)
  * BigInt(transferSign.exact_p_denominator);
const expectedDecision = transferDirection
  && transferSignificant
  && transferSign.non_ties >= threshold.minimum_non_tied_documents;
assert(result.decision.prospective_transfer_synergy_supported === expectedDecision,
  "prospective decision does not implement the frozen rule");
assert(result.decision.optimizer_change_authorized === false,
  "confirmation must not authorize an optimizer change");
assert(result.decision.paid_scaling_authorized === false,
  "confirmation must not authorize paid scaling");

console.log(JSON.stringify({
  schema: "nsrl.production_boolean_jet_confirmation_check.v1",
  proposal: {
    joint_wins: proposalSign.joint_wins,
    head_wins: proposalSign.head_wins,
    ties: proposalSign.ties,
    non_ties: proposalSign.non_ties,
  },
  transfer: {
    joint_wins: transferSign.joint_wins,
    head_wins: transferSign.head_wins,
    ties: transferSign.ties,
    non_ties: transferSign.non_ties,
  },
  prospective_transfer_synergy_supported: expectedDecision,
  optimizer_change_authorized: false,
}, null, 2));
