#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const proposalPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const confirmationPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json";
const isingConfirmationPath = process.argv[4]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json";
const outputPath = process.argv[5]
  ?? "benchmarks/production-model-v1/p10m-atomic-conditional-exchange-confirmation-v1.json";
const proposalBytes = fs.readFileSync(proposalPath);
const confirmationBytes = fs.readFileSync(confirmationPath);
const isingConfirmationBytes = fs.readFileSync(isingConfirmationPath);
const analyzerBytes = fs.readFileSync(new URL(import.meta.url));
const proposal = JSON.parse(proposalBytes.toString("utf8"));
const confirmation = JSON.parse(confirmationBytes.toString("utf8"));
const isingConfirmation = JSON.parse(isingConfirmationBytes.toString("utf8"));
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const absolute = (value) => value < 0n ? -value : value;
const reconstruct = (coefficients) => Array.from({length: 64}, (_, mask) => {
  let value = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    value += coefficients[subset];
    if (subset === 0) return value;
  }
});
const summarize = (values) => ({
  favorable: values.filter((value) => value < 0n).length,
  unfavorable: values.filter((value) => value > 0n).length,
  ties: values.filter((value) => value === 0n).length,
  aggregate: values.reduce((sum, value) => sum + value, 0n).toString(),
  minimum: values.reduce((left, right) => left < right ? left : right).toString(),
  maximum: values.reduce((left, right) => left > right ? left : right).toString(),
});
const baseMask = 43;
const outgoingAtom = 2;
const incomingAtom = 4;
const outgoingMask = 1 << outgoingAtom;
const incomingMask = 1 << incomingAtom;
const controlMask = baseMask | outgoingMask;
const candidateMask = baseMask | incomingMask;
assert(controlMask === 47 && candidateMask === 59, "exchange masks changed");
const medoids = [
  [0n, 0n, 0n, 0n, 1977n, -4068n],
  [0n, 0n, 0n, 0n, -6398n, -4020n],
];
const l1 = (left, right) => left.reduce(
  (sum, value, index) => sum + absolute(value - right[index]), 0n);
const features = (losses) => [1, 2, 4, 8, 16, 32].map(
  (mask) => losses[mask] - losses[0]);
const route = (losses) => {
  const vector = features(losses);
  return l1(vector, medoids[0]) <= l1(vector, medoids[1]) ? 0 : 1;
};
const objectiveRows = (objective, q32Objective) => objective.documents.map((document, index) => {
  const coefficients = document.coefficients.map(BigInt);
  const losses = reconstruct(coefficients);
  const q32Losses = reconstruct(q32Objective.documents[index].coefficients.map(BigInt));
  const singletonDifference = (losses[incomingMask] - losses[0])
    - (losses[outgoingMask] - losses[0]);
  const exchangeContrast = losses[candidateMask] - losses[controlMask];
  let interactionResidual = 0n;
  for (let subset = baseMask; subset !== 0; subset = (subset - 1) & baseMask) {
    interactionResidual += coefficients[subset | incomingMask]
      - coefficients[subset | outgoingMask];
  }
  assert(exchangeContrast === singletonDifference + interactionResidual,
    `conditional exchange decomposition failed on document ${document.document}`);
  return {
    document: document.document,
    route: route(q32Losses),
    incoming_q32_singleton: q32Losses[incomingMask] - q32Losses[0],
    singleton_difference: singletonDifference,
    interaction_residual: interactionResidual,
    exchange_contrast: exchangeContrast,
  };
});
const populationAnalysis = (surface, label) => {
  const q20 = objectiveRows(surface.q20, surface.q32);
  const q32 = objectiveRows(surface.q32, surface.q32);
  const summarizeObjective = (rows) => [0, 1].map((cluster) => {
    const selected = rows.filter((row) => row.route === cluster);
    return {
      cluster,
      documents: selected.length,
      incoming_q32_singleton: summarize(
        selected.map((row) => row.incoming_q32_singleton)),
      singleton_difference: summarize(selected.map((row) => row.singleton_difference)),
      conditional_exchange: summarize(selected.map((row) => row.exchange_contrast)),
      interaction_residual: summarize(selected.map((row) => row.interaction_residual)),
      singleton_exchange_sign_agreement: selected.filter((row) =>
        row.singleton_difference !== 0n && row.exchange_contrast !== 0n
          && (row.singleton_difference < 0n) === (row.exchange_contrast < 0n)).length,
    };
  });
  return {
    label,
    analysis_role: surface.analysis_role,
    document_start: surface.surface.document_start,
    documents: surface.surface.documents,
    q20: summarizeObjective(q20),
    q32: summarizeObjective(q32),
    decomposition_verified_documents: q20.length + q32.length,
  };
};

assert(proposal.analysis_role === "proposal_only_calibration"
  && proposal.transfer_documents_read === 0 && proposal.reserved_documents_read === 0,
"proposal firewall changed");
assert(confirmation.analysis_role === "untouched_confirmation"
  && confirmation.surface.hard_stop_before_document === 200,
"confirmation surface changed");
assert(isingConfirmation.confirmation_contract_sha256
  === "57084a50c82883cb7d6c6d449b699cf793d495e293cf33ea1318b04aad71c9ce",
"primary confirmation contract changed");
assert(isingConfirmation.limitations.documents_200_212_read === false,
  "sealed documents were read");
const proposalAnalysis = populationAnalysis(proposal, "proposal_documents_8_71");
const confirmationAnalysis = populationAnalysis(
  confirmation, "untouched_confirmation_documents_136_199");
const proposalRouted = proposalAnalysis.q32[1].conditional_exchange;
const confirmationRouted = confirmationAnalysis.q32[1].conditional_exchange;
const result = {
  schema: "nsrl.production_atomic_conditional_exchange_confirmation.v1",
  analysis_role: "post_confirmation_theory_revision",
  source_sha256: {
    proposal_structure: sha256(proposalBytes),
    confirmation_structure: sha256(confirmationBytes),
    primary_ising_confirmation: sha256(isingConfirmationBytes),
    analyzer: sha256(analyzerBytes),
  },
  exchange: {
    base_mask: baseMask,
    outgoing_atom: outgoingAtom,
    incoming_atom: incomingAtom,
    control_mask: controlMask,
    candidate_mask: candidateMask,
    identity:
      "Delta_swap=(singleton_in-singleton_out)+sum_nonempty_T_subset_base(mu(T+in)-mu(T+out))",
  },
  router: {
    features: "Q32 singleton contrasts",
    medoids: medoids.map((row) => row.map(String)),
    distance: "L1",
    tie_break: "cluster_zero",
  },
  proposal: proposalAnalysis,
  confirmation: confirmationAnalysis,
  replicated_directional_partition: {
    routed_cluster_proposal_q32: proposalRouted,
    routed_cluster_confirmation_q32: confirmationRouted,
    routed_cluster_all_nonties_favorable_on_both_surfaces:
      proposalRouted.unfavorable === 0 && confirmationRouted.unfavorable === 0,
    nonrouted_cluster_exchange_aggregate_positive_on_both_surfaces:
      BigInt(proposalAnalysis.q32[0].conditional_exchange.aggregate) > 0n
      && BigInt(confirmationAnalysis.q32[0].conditional_exchange.aggregate) > 0n,
  },
  interpretation: {
    revised_mechanism:
      "probe-routed conditional exchange, not a globally stable pairwise coupling",
    proposal_max_residual_is_not_a_uniform_transfer_certificate: true,
    same_source_cluster_only: true,
    cross_source_generalization_identified: false,
  },
  decision: {
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    documents_200_212_read: false,
  },
};
const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(result, null, 2)}\n`);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_conditional_exchange_confirmation_check.v1",
  routed_proposal_q32: proposalRouted,
  routed_confirmation_q32: confirmationRouted,
  replicated_directional_partition: result.replicated_directional_partition,
  documents_200_212_read: false,
  optimizer_change_authorized: false,
}, null, 2)}\n`);
