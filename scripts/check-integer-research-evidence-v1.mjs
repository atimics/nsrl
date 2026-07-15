#!/usr/bin/env node

import fs from "node:fs";

const ablation = JSON.parse(
  fs.readFileSync("benchmarks/integer-transformer-proof-v1/component-ablation.json", "utf8"),
);
const capacity = JSON.parse(
  fs.readFileSync("benchmarks/integer-reachable-capacity-v1/matrix.json", "utf8"),
);
const successor = JSON.parse(
  fs.readFileSync("benchmarks/integer-transformer-proof-v1/transformer-successor-sweep.json", "utf8"),
);
const longitudinal = JSON.parse(
  fs.readFileSync("benchmarks/integer-reachable-capacity-v1/longitudinal.json", "utf8"),
);

if (ablation.schema !== "nsrl.integer_transformer_component_ablation.v1"
  || ablation.source_model_hash !== "0x6ffd37de48a3121b"
  || ablation.data.windows !== 5896
  || ablation.metrics.combined.mistakes !== 2482
  || ablation.metrics["transformer-only"].mistakes !== 5094
  || ablation.metrics["suffix-memory-only"].mistakes !== 2482
  || ablation.contrasts.suffix_memory_added_to_transformer.mistake_reduction !== 2612
  || ablation.contrasts.transformer_logits_added_to_suffix_memory.mistake_reduction !== 0
  || ablation.contrasts.transformer_logits_added_to_suffix_memory
    .probability_error_reduction_q15 !== 124348395) {
  throw new Error("integer-transformer component ablation checkpoint is invalid");
}

const zeroFunctional = capacity.matrix.filter(
  (row) => row.functional_update.nonzero_count === 0,
).length;
const uniqueFunctional = new Set(
  capacity.matrix.map((row) => row.functional_update.hash),
).size;
const rank16Shift3Carry = capacity.matrix.find(
  (row) => row.rank === 16 && row.learning_rate_shift === 3 && row.error_feedback,
);
const rank32Shift3Carry = capacity.matrix.find(
  (row) => row.rank === 32 && row.learning_rate_shift === 3 && row.error_feedback,
);
if (capacity.schema !== "nsrl.integer_reachable_capacity_matrix.v1"
  || capacity.claim_status !== "bounded_observation_not_capacity_proof"
  || capacity.matrix.length !== 30
  || capacity.observed_capacity.runs !== 30
  || capacity.observed_capacity.unique_functional_update_hashes !== 15
  || capacity.observed_capacity.zero_functional_updates !== 14
  || uniqueFunctional !== 15
  || zeroFunctional !== 14
  || rank16Shift3Carry?.functional_update.hash !== rank32Shift3Carry?.functional_update.hash
  || rank16Shift3Carry?.functional_update.nonzero_count !== 3071) {
  throw new Error("integer reachable-capacity checkpoint is invalid");
}

if (successor.schema !== "nsrl.integer_transformer_successor_sweep.v1"
  || successor.candidates.length !== 16
  || successor.candidates.some((row) => row.passed)
  || successor.conclusion.passed_candidates !== 0
  || successor.conclusion.best_variant !== "s5-e1"
  || successor.conclusion.best_mistakes !== 5094
  || successor.conclusion.best_probability_error_q15 !== 337139495
  || successor.conclusion.mistake_gap_to_gate !== 2584
  || successor.conclusion.status !== "blocked_on_top1_generalization") {
  throw new Error("integer-transformer successor sweep checkpoint is invalid");
}

const binary = longitudinal.analysis.binary_prediction;
const saturation = longitudinal.analysis.saturation;
if (longitudinal.schema !== "nsrl.integer_reachable_capacity_longitudinal.v1"
  || longitudinal.matrix.length !== 30
  || longitudinal.baseline.windows !== 4096
  || binary.true_positive !== 16
  || binary.false_positive !== 0
  || binary.true_negative !== 8
  || binary.false_negative !== 6
  || Math.abs(binary.matthews_correlation - 0.6446583712203042) > 1e-12
  || Math.abs(longitudinal.analysis.correlations.functional_delta_l1_spearman
    - 0.8279811480076021) > 1e-12
  || longitudinal.analysis.permutation_test.permutations !== 10000
  || longitudinal.analysis.permutation_test.p_value_one_sided_greater
    !== 0.00009999000099990002
  || saturation.cells_with_any_saturation !== 20
  || saturation.early_reachable_cells_with_any_saturation !== 16
  || longitudinal.conclusion.prediction_supported !== true
  || longitudinal.conclusion.claim_status !== "supported_in_bounded_longitudinal_matrix"
  || longitudinal.conclusion.safety_status
    !== "association_supported_but_long_run_saturation_requires_followup") {
  throw new Error("integer reachable-capacity longitudinal checkpoint is invalid");
}

console.log(JSON.stringify({
  schema: "nsrl.integer_research_evidence_check.v1",
  ok: true,
  component_ablation_windows: ablation.data.windows,
  reachable_capacity_runs: capacity.matrix.length,
  unique_functional_updates: uniqueFunctional,
  suffix_free_successor_passes: successor.conclusion.passed_candidates,
  longitudinal_matthews_correlation: binary.matthews_correlation,
  longitudinal_spearman: longitudinal.analysis.correlations.functional_delta_l1_spearman,
}));
