#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";

const inputPath = process.argv[2]
  ?? "benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json";
const outputPath = process.argv[3]
  ?? "benchmarks/production-model-v1/p10m-atomic-ising-proposal-v1.json";
const inputBytes = fs.readFileSync(inputPath);
const analyzerBytes = fs.readFileSync(new URL(import.meta.url));
const source = JSON.parse(inputBytes.toString("utf8"));
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const rank = 6;
const vertices = 1 << rank;
const denominator = BigInt(vertices);
const q30 = 1n << 30n;
const absolute = (value) => value < 0n ? -value : value;
const popcount = (mask) => {
  let count = 0;
  for (let value = mask; value !== 0; value >>>= 1) count += value & 1;
  return count;
};
const spin = (character, vertex) => popcount(character & vertex) % 2 === 0 ? 1n : -1n;
const sign = (value) => value < 0n ? -1 : value > 0n ? 1 : 0;
const minimum = (values) => values.reduce((left, right) => left < right ? left : right);
const maximum = (values) => values.reduce((left, right) => left > right ? left : right);
const reconstructLosses = (coefficients) => Array.from({length: vertices}, (_, mask) => {
  let sum = 0n;
  for (let subset = mask; ; subset = (subset - 1) & mask) {
    sum += coefficients[subset];
    if (subset === 0) return sum;
  }
});
const walsh = (losses) => Array.from({length: vertices}, (_, character) =>
  losses.reduce((sum, loss, vertex) => sum + loss * spin(character, vertex), 0n));
const exactDivideRounded = (numerator, divisor) => {
  assert(divisor > 0n, "positive divisor required");
  if (numerator < 0n) return -exactDivideRounded(-numerator, divisor);
  return (numerator + divisor / 2n) / divisor;
};
const support = (mask) => Array.from(
  {length: rank},
  (_, atom) => atom,
).filter((atom) => (mask & (1 << atom)) !== 0);
const contrastSummary = (contrasts) => ({
  favorable: contrasts.filter((value) => value < 0n).length,
  unfavorable: contrasts.filter((value) => value > 0n).length,
  ties: contrasts.filter((value) => value === 0n).length,
  aggregate: contrasts.reduce((sum, value) => sum + value, 0n).toString(),
});

assert(source.schema === "nsrl.production_atomic_structure.v1", "wrong source schema");
assert(source.analysis_role === "proposal_only_calibration", "source is not proposal-only");
assert(source.rank === rank && source.vertices_evaluated === vertices, "wrong source rank");
assert(source.transfer_documents_read === 0 && source.reserved_documents_read === 0,
  "Ising analysis source crossed the proposal firewall");
assert(source.surface.document_start === 8 && source.surface.documents === 64
  && source.surface.hard_stop_before_document === 72,
"proposal surface changed");

const objectiveDocuments = (objective) => objective.documents.map((document) => {
  const losses = reconstructLosses(document.coefficients.map(BigInt));
  return {
    document: document.document,
    losses,
    walsh: walsh(losses),
  };
});
const q20Documents = objectiveDocuments(source.q20);
const q32Documents = objectiveDocuments(source.q32);
assert(q20Documents.length === 64 && q32Documents.length === 64,
  "proposal document cube count changed");
assert(q20Documents.every((document, index) => document.document === q32Documents[index].document),
  "Q20/Q32 document order changed");

const aggregateWalsh = (documents) => Array.from({length: vertices}, (_, character) =>
  documents.reduce((sum, document) => sum + document.walsh[character], 0n));
const q20AggregateWalsh = aggregateWalsh(q20Documents);
const q32AggregateWalsh = aggregateWalsh(q32Documents);

const parameterStability = (character) => {
  const standardValues = (documents) => documents.map(
    (document) => -document.walsh[character]);
  const summarize = (values) => {
    const negative = values.filter((value) => value < 0n).length;
    const zero = values.filter((value) => value === 0n).length;
    const positive = values.filter((value) => value > 0n).length;
    const sum = values.reduce((total, value) => total + value, 0n);
    const absoluteSum = values.reduce((total, value) => total + absolute(value), 0n);
    return {negative, zero, positive, sum, absoluteSum};
  };
  const coarse = summarize(standardValues(q20Documents));
  const fine = summarize(standardValues(q32Documents));
  const visible = fine.negative + fine.positive;
  const majority = Math.max(fine.negative, fine.positive);
  const representationSignAgree = sign(coarse.sum) !== 0 && sign(coarse.sum) === sign(fine.sum);
  const stable = visible >= 32 && majority * 4 >= visible * 3 && representationSignAgree;
  return {
    character,
    order: popcount(character),
    atoms: support(character),
    standard_parameter: popcount(character) === 1 ? "field_h" : "coupling_J",
    normalization_denominator: vertices,
    q20: {
      negative_documents: coarse.negative,
      zero_documents: coarse.zero,
      positive_documents: coarse.positive,
      aggregate_numerator: coarse.sum.toString(),
      absolute_document_numerator: coarse.absoluteSum.toString(),
    },
    q32: {
      negative_documents: fine.negative,
      zero_documents: fine.zero,
      positive_documents: fine.positive,
      aggregate_numerator: fine.sum.toString(),
      absolute_document_numerator: fine.absoluteSum.toString(),
    },
    stability_rule: {
      minimum_visible_documents: 32,
      minimum_directional_fraction: "3/4",
      aggregate_q20_q32_sign_agreement_required: true,
    },
    stable,
  };
};
const lowOrderParameters = Array.from({length: vertices - 1}, (_, index) => index + 1)
  .filter((character) => popcount(character) <= 2)
  .map(parameterStability);

const minimizeWalshOrder = (documents, retainedOrder) => {
  const coefficients = aggregateWalsh(documents);
  const exactLosses = Array.from({length: vertices}, (_, vertex) => documents.reduce(
    (sum, document) => sum + document.losses[vertex], 0n));
  const values = Array.from({length: vertices}, (_, vertex) => coefficients.reduce(
    (sum, coefficient, character) => popcount(character) <= retainedOrder
      ? sum + coefficient * spin(character, vertex) : sum,
    0n,
  ));
  const best = minimum(values);
  const minimizers = values.flatMap((value, mask) => value === best ? [mask] : []);
  const selected = minimizers[0];
  const exactMinimum = minimum(exactLosses);
  const residualNumerators = exactLosses.map(
    (loss, vertex) => denominator * loss - values[vertex]);
  const residualOscillationNumerator = maximum(residualNumerators)
    - minimum(residualNumerators);
  const selectedExactGap = exactLosses[selected] - exactMinimum;
  assert(denominator * selectedExactGap <= residualOscillationNumerator,
    "pairwise Ising tail regret certificate failed");
  return {
    retained_order: retainedOrder,
    minimizers,
    selected,
    selected_exact_gap: selectedExactGap.toString(),
    residual_oscillation_numerator: residualOscillationNumerator.toString(),
    residual_normalization_denominator: vertices,
    regret_certificate_verified: true,
  };
};

const gibbsMagnetization = (documents, numerator, divisor) => {
  const documentMagnetizations = documents.map((document) => {
    const best = minimum(document.losses);
    const gaps = document.losses.map((loss) => Number(loss - best));
    assert(gaps.every((gap) => Number.isSafeInteger(gap) && gap >= 0),
      "Gibbs energy gap is not a safe nonnegative integer");
    const largestGap = Math.max(...gaps);
    const weights = gaps.map((gap) => numerator ** BigInt(gap)
      * divisor ** BigInt(largestGap - gap));
    const partition = weights.reduce((sum, weight) => sum + weight, 0n);
    return Array.from({length: rank}, (_, atom) => {
      const moment = weights.reduce(
        (sum, weight, vertex) => sum + weight * spin(1 << atom, vertex), 0n);
      return exactDivideRounded(moment * q30, partition);
    });
  });
  const mean = Array.from({length: rank}, (_, atom) => exactDivideRounded(
    documentMagnetizations.reduce((sum, values) => sum + values[atom], 0n),
    BigInt(documentMagnetizations.length),
  ));
  const selectedMask = mean.reduce(
    (mask, magnetization, atom) => magnetization < 0n ? mask | (1 << atom) : mask,
    0,
  );
  return {
    fugacity: `${numerator}/${divisor}`,
    inverse_temperature: `-ln(${numerator}/${divisor})_per_Q20_unit`,
    mean_spin_magnetization_q30: mean.map(String),
    selected_mask: selectedMask,
    selected_atoms: support(selectedMask),
  };
};
const gibbsGrid = [[1n, 4n], [1n, 2n], [3n, 4n]].map(
  ([numerator, divisor]) => gibbsMagnetization(q20Documents, numerator, divisor));
const selectedGibbs = gibbsGrid.find((row) => row.fugacity === "1/2");
assert(selectedGibbs !== undefined, "central Gibbs temperature missing");

const singletonMasks = Array.from({length: rank}, (_, atom) => 1 << atom);
const probeFeatures = (document) => singletonMasks.map(
  (mask) => document.losses[mask] - document.losses[0]);
const l1Distance = (left, right) => left.reduce(
  (sum, value, index) => sum + absolute(value - right[index]), 0n);

const selectDirectionalCandidate = (documents) => {
  const rows = Array.from({length: vertices - 1}, (_, index) => index + 1).map((mask) => {
    const contrasts = documents.map((document) => document.losses[mask] - document.losses[0]);
    const summary = contrastSummary(contrasts);
    return {...summary, mask, aggregateValue: BigInt(summary.aggregate)};
  });
  rows.sort((left, right) => right.favorable - left.favorable
    || left.unfavorable - right.unfavorable
    || (left.aggregateValue < right.aggregateValue ? -1 : left.aggregateValue > right.aggregateValue ? 1 : 0)
    || popcount(left.mask) - popcount(right.mask)
    || left.mask - right.mask);
  return rows[0].mask;
};

const selectClusterCandidate = (documents) => {
  let selected;
  for (let mask = 1; mask < vertices; mask += 1) {
    if (popcount(mask) < 2) continue;
    const contrasts = documents.map((document) => document.losses[mask] - document.losses[0]);
    const summary = contrastSummary(contrasts);
    const row = {...summary, mask, aggregateValue: BigInt(summary.aggregate)};
    if (selected === undefined
      || row.aggregateValue < selected.aggregateValue
      || (row.aggregateValue === selected.aggregateValue && row.favorable > selected.favorable)
      || (row.aggregateValue === selected.aggregateValue && row.favorable === selected.favorable
        && row.unfavorable < selected.unfavorable)
      || (row.aggregateValue === selected.aggregateValue && row.favorable === selected.favorable
        && row.unfavorable === selected.unfavorable
        && popcount(row.mask) < popcount(selected.mask))
      || (row.aggregateValue === selected.aggregateValue && row.favorable === selected.favorable
        && row.unfavorable === selected.unfavorable
        && popcount(row.mask) === popcount(selected.mask) && row.mask < selected.mask)) {
      selected = row;
    }
  }
  return selected.mask;
};

const fitTwoMedoids = (documents) => {
  assert(documents.length >= 4, "two-medoid fit needs at least four documents");
  const rows = documents.map((document) => ({document, features: probeFeatures(document)}));
  let first = 0;
  let second = 1;
  let farthest = -1n;
  for (let left = 0; left < rows.length; left += 1) {
    for (let right = left + 1; right < rows.length; right += 1) {
      const distance = l1Distance(rows[left].features, rows[right].features);
      if (distance > farthest) {
        farthest = distance;
        first = left;
        second = right;
      }
    }
  }
  let medoids = [rows[first], rows[second]];
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const clusters = [[], []];
    for (const row of rows) {
      const distances = medoids.map((medoid) => l1Distance(row.features, medoid.features));
      clusters[distances[0] <= distances[1] ? 0 : 1].push(row);
    }
    assert(clusters.every((cluster) => cluster.length > 0), "two-medoid fit made an empty cluster");
    const next = clusters.map((cluster) => cluster.reduce((best, candidate) => {
      const candidateDistance = cluster.reduce(
        (sum, row) => sum + l1Distance(candidate.features, row.features), 0n);
      const bestDistance = cluster.reduce(
        (sum, row) => sum + l1Distance(best.features, row.features), 0n);
      return candidateDistance < bestDistance
        || (candidateDistance === bestDistance
          && candidate.document.document < best.document.document) ? candidate : best;
    }, cluster[0]));
    if (next[0].document.document === medoids[0].document.document
      && next[1].document.document === medoids[1].document.document) {
      return {
        medoids,
        clusters,
        candidateMasks: clusters.map(
          (cluster) => selectClusterCandidate(cluster.map((row) => row.document))),
      };
    }
    medoids = next;
  }
  throw new Error("two-medoid fit did not converge");
};
const routeCluster = (model, document) => {
  const features = probeFeatures(document);
  const distances = model.medoids.map((medoid) => l1Distance(features, medoid.features));
  return distances[0] <= distances[1] ? 0 : 1;
};

const summarizeRule = (documents, maskForDocument, comparatorForDocument = () => 0) => {
  const contrasts = documents.map((document) => {
    const mask = maskForDocument(document);
    const comparator = comparatorForDocument(document);
    return document.losses[mask] - document.losses[comparator];
  });
  return contrastSummary(contrasts);
};

const fittedClusterModel = fitTwoMedoids(q32Documents);
const globalDirectionalMask = selectDirectionalCandidate(q32Documents);
const pairwiseQ20 = minimizeWalshOrder(q20Documents, 2);
const pairwiseQ32 = minimizeWalshOrder(q32Documents, 2);
const proposalRules = {
  pairwise_q32_vs_baseline: summarizeRule(q32Documents, () => pairwiseQ32.selected),
  gibbs_q20_mask_vs_baseline_q32: summarizeRule(
    q32Documents, () => selectedGibbs.selected_mask),
  global_directional_vs_baseline: summarizeRule(q32Documents, () => globalDirectionalMask),
  cluster_routed_vs_baseline: summarizeRule(
    q32Documents,
    (document) => fittedClusterModel.candidateMasks[routeCluster(fittedClusterModel, document)],
  ),
  cluster_routed_vs_global_directional: summarizeRule(
    q32Documents,
    (document) => fittedClusterModel.candidateMasks[routeCluster(fittedClusterModel, document)],
    () => globalDirectionalMask,
  ),
};

const crossValidate = (foldKind) => {
  const foldFor = (index) => foldKind === "contiguous_eight_by_eight"
    ? Math.floor(index / 8) : index % 8;
  const records = [];
  const allContrasts = {
    pairwise: [],
    gibbs: [],
    global_directional: [],
    cluster_routed: [],
    cluster_incremental: [],
  };
  for (let fold = 0; fold < 8; fold += 1) {
    const trainingIndices = q32Documents.flatMap(
      (_, index) => foldFor(index) === fold ? [] : [index]);
    const validationIndices = q32Documents.flatMap(
      (_, index) => foldFor(index) === fold ? [index] : []);
    const trainingQ32 = trainingIndices.map((index) => q32Documents[index]);
    const trainingQ20 = trainingIndices.map((index) => q20Documents[index]);
    const validation = validationIndices.map((index) => q32Documents[index]);
    const pairwise = minimizeWalshOrder(trainingQ32, 2).selected;
    const gibbs = gibbsMagnetization(trainingQ20, 1n, 2n).selected_mask;
    const globalDirectional = selectDirectionalCandidate(trainingQ32);
    const cluster = fitTwoMedoids(trainingQ32);
    for (const document of validation) {
      const routed = cluster.candidateMasks[routeCluster(cluster, document)];
      allContrasts.pairwise.push(document.losses[pairwise] - document.losses[0]);
      allContrasts.gibbs.push(document.losses[gibbs] - document.losses[0]);
      allContrasts.global_directional.push(
        document.losses[globalDirectional] - document.losses[0]);
      allContrasts.cluster_routed.push(document.losses[routed] - document.losses[0]);
      allContrasts.cluster_incremental.push(
        document.losses[routed] - document.losses[globalDirectional]);
    }
    records.push({
      fold,
      validation_documents: validation.map((document) => document.document),
      pairwise_mask: pairwise,
      gibbs_mask: gibbs,
      global_directional_mask: globalDirectional,
      cluster_medoid_documents: cluster.medoids.map((medoid) => medoid.document.document),
      cluster_candidate_masks: cluster.candidateMasks,
    });
  }
  return {
    fold_kind: foldKind,
    folds: records,
    summaries: Object.fromEntries(Object.entries(allContrasts).map(
      ([key, contrasts]) => [key, contrastSummary(contrasts)])),
  };
};

const result = {
  schema: "nsrl.production_atomic_ising_proposal.v1",
  analysis_role: "proposal_only_calibration",
  source_result_sha256: crypto.createHash("sha256").update(inputBytes).digest("hex"),
  analyzer_sha256: crypto.createHash("sha256").update(analyzerBytes).digest("hex"),
  rank,
  vertices,
  documents: q32Documents.length,
  source_population: source.source_population,
  spin_convention: {
    sigma_i: "(-1)^x_i",
    action_absent: 1,
    action_present: -1,
    hamiltonian: "H=C-sum_i(h_i*sigma_i)-sum_ij(J_ij*sigma_i*sigma_j)-higher_orders",
    standard_parameter_numerator: "negative_Walsh_numerator",
    normalization_denominator: vertices,
  },
  low_order_parameters: lowOrderParameters,
  stable_low_order_characters: lowOrderParameters.filter((row) => row.stable)
    .map((row) => row.character),
  pairwise_ising_map: {
    q20: pairwiseQ20,
    q32: pairwiseQ32,
  },
  gibbs: {
    objective: "q47_logit_anchored_q20",
    ensemble: "quenched_document_average",
    exact_weight: "(p/q)^(H_d(x)-min_x_H_d)",
    magnetization_rounding: "nearest_Q30_after_exact_rational_partition_sum",
    temperature_grid: gibbsGrid,
    selected_fugacity: "1/2",
    selected_mask: selectedGibbs.selected_mask,
  },
  clustering: {
    operational_role: "leakage_safe_sequential_router",
    features: "six_Q32_singleton_contrasts_against_baseline_only",
    distance: "L1",
    clusters: 2,
    initialization: "farthest_document_pair_then_deterministic_PAM_updates",
    candidate_eligibility: "cardinality_at_least_two_so_no_probe_vertex_is_an_outcome",
    medoids: fittedClusterModel.medoids.map((medoid, cluster) => ({
      cluster,
      document: medoid.document.document,
      feature_vector: medoid.features.map(String),
      members: fittedClusterModel.clusters[cluster].length,
      candidate_mask: fittedClusterModel.candidateMasks[cluster],
      candidate_atoms: support(fittedClusterModel.candidateMasks[cluster]),
    })),
  },
  global_directional_control: {
    selection: "max_favorable_then_min_unfavorable_then_min_aggregate_then_cardinality_then_mask",
    mask: globalDirectionalMask,
    atoms: support(globalDirectionalMask),
  },
  proposal_rule_results_q32: proposalRules,
  cross_validation: [
    crossValidate("contiguous_eight_by_eight"),
    crossValidate("interleaved_modulo_eight"),
  ],
  frozen_confirmation_candidates: {
    pairwise_ising_map_mask: pairwiseQ32.selected,
    gibbs_magnetization_mask: selectedGibbs.selected_mask,
    global_directional_control_mask: globalDirectionalMask,
    cluster_medoid_feature_vectors: fittedClusterModel.medoids.map(
      (medoid) => medoid.features.map(String)),
    cluster_candidate_masks: fittedClusterModel.candidateMasks,
  },
  confirmation_design: {
    document_start: 136,
    documents: 64,
    hard_stop_before_document: 200,
    still_sealed_documents: "200--212",
    same_source_cluster_only: true,
    primary_endpoints: [
      "pairwise_ising_map_vs_baseline_q32_document_direction",
      "gibbs_magnetization_vs_baseline_q32_document_direction",
      "cluster_routed_vs_global_directional_q32_document_direction",
    ],
    multiplicity: "Holm_familywise_alpha_0.05_over_three_directional_endpoints",
    descriptive_endpoints: [
      "aggregate_Q32_contrast",
      "Q20_robustness_direction",
      "stable_low_order_coupling_sign_replication",
      "quenched_magnetization_Q30_replication",
      "cluster_route_vs_baseline",
    ],
  },
  limitations: {
    proposal_source_clusters: 1,
    document_fold_cross_validation_is_not_source_cluster_validation: true,
    coupling_clusters_using_full_cubes_are_descriptive_only: true,
    operational_cluster_router_uses_singleton_probes_only: true,
    transfer_documents_72_135_read: false,
    reserved_documents_136_212_read: false,
  },
  decision: {
    optimizer_change_authorized: false,
    paid_scaling_authorized: false,
    untouched_confirmation_authorized_by_this_frozen_design: true,
  },
};

const temporaryPath = `${outputPath}.tmp-${process.pid}`;
fs.writeFileSync(temporaryPath, `${JSON.stringify(result, null, 2)}\n`);
fs.renameSync(temporaryPath, outputPath);
process.stdout.write(`${JSON.stringify({
  schema: "nsrl.production_atomic_ising_proposal_check.v1",
  stable_low_order_characters: result.stable_low_order_characters,
  pairwise_ising_map_mask: result.frozen_confirmation_candidates.pairwise_ising_map_mask,
  gibbs_magnetization_mask: result.frozen_confirmation_candidates.gibbs_magnetization_mask,
  global_directional_control_mask: result.frozen_confirmation_candidates.global_directional_control_mask,
  cluster_candidate_masks: result.frozen_confirmation_candidates.cluster_candidate_masks,
  contiguous_cross_validation: result.cross_validation[0].summaries,
  proposal_only_firewall_verified: true,
}, null, 2)}\n`);
