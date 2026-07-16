use std::fmt::Write;
use std::ops::Range;

use nsrl_core::base2_softmax_nll_q20;
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use super::training::{
    GradientProposalSpec, GradientSource, coarse_gradients_for_window_with_spec,
    effective_learning_rate_shifts,
};
use super::{
    ProductionFullTrainConfig, ProductionGradientProposalLane, ProductionModelV1, TrainError,
    forward_production_model,
};

const ZERO_PROBABILITY_FLOOR_Q20: u64 = 32_u64 << 20;
const LANE_COUNT: usize =
    ProductionGradientProposalLane::WITH_SYSTEMATIC_AND_CAUSAL_NO_RESCUE.len();
const RESCUE_SOURCE_COUNT: usize = 3;
const RESCUE_STRATUM_NAMES: [&str; RESCUE_SOURCE_COUNT] = [
    "normalized_rescue",
    "mass_corrected_rescue",
    "reciprocal_free_rescue",
];
const GROUP_NAMES: [&str; 13] = [
    "embeddings",
    "attention_rms",
    "mlp_rms",
    "final_rms",
    "q",
    "k",
    "v",
    "o",
    "up",
    "gate",
    "down",
    "output",
    "bias",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGradientAlignmentConfig {
    pub proposal_windows: usize,
    pub transfer_windows: usize,
    pub documents_per_surface: usize,
    pub rescue_stratified_sampling: bool,
    pub include_mass_corrected_no_rescue: bool,
    pub include_systematic_fixed_mass: bool,
    pub coordinates_per_group: usize,
    pub sample_seed: u64,
}

impl Default for ProductionGradientAlignmentConfig {
    fn default() -> Self {
        Self {
            proposal_windows: 1,
            transfer_windows: 1,
            documents_per_surface: 0,
            rescue_stratified_sampling: false,
            include_mass_corrected_no_rescue: false,
            include_systematic_fixed_mass: false,
            coordinates_per_group: 1,
            sample_seed: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProductionGradientAlignmentSummary {
    pub sampled_coordinates: usize,
    pub proposed_coordinates: usize,
    pub direction_comparable: usize,
    pub direction_agreements: usize,
    pub random_direction_agreements: usize,
    pub exact_descent_available: usize,
    pub predicted_exact_descents: usize,
    pub random_exact_descents: usize,
    pub function_visible: usize,
    pub objective_ties: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGradientLaneHealth {
    pub lane: ProductionGradientProposalLane,
    pub sample_reweighted_surrogate: bool,
    pub output_gradient_sum_min: i64,
    pub output_gradient_sum_max: i64,
    pub output_gradient_l1_min: u64,
    pub output_gradient_l1_max: u64,
    pub output_gradient_max_abs: u64,
    pub output_gradient_trace_hash: u64,
    pub backward_ste_rescue_count: u64,
    pub backward_quantization_count: u64,
    pub stochastic_round_up_count: u64,
    pub gradient_saturation_count: usize,
    pub residual_saturation_count: usize,
}

impl ProductionGradientLaneHealth {
    fn new(lane: ProductionGradientProposalLane) -> Self {
        Self {
            lane,
            sample_reweighted_surrogate: matches!(
                lane,
                ProductionGradientProposalLane::ReciprocalFreeRescued
                    | ProductionGradientProposalLane::ReciprocalFreeLateRhu
                    | ProductionGradientProposalLane::ReciprocalFreeLateStochastic
            ),
            output_gradient_sum_min: i64::MAX,
            output_gradient_sum_max: i64::MIN,
            output_gradient_l1_min: u64::MAX,
            output_gradient_l1_max: 0,
            output_gradient_max_abs: 0,
            output_gradient_trace_hash: super::FNV_OFFSET,
            backward_ste_rescue_count: 0,
            backward_quantization_count: 0,
            stochastic_round_up_count: 0,
            gradient_saturation_count: 0,
            residual_saturation_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGradientSurfaceDelta {
    pub current_nll_q20: u64,
    pub plus_one_nll_q20: u64,
    pub minus_one_nll_q20: u64,
    pub better_neighbor_delta: i8,
    pub exact_descent_available: bool,
    pub plus_function_visible: bool,
    pub minus_function_visible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGradientLaneSample {
    pub lane: ProductionGradientProposalLane,
    pub rescue_exposed: bool,
    pub coarse_gradient: i64,
    pub predicted_parameter_delta: i8,
    pub proposal_direction_agrees: bool,
    pub proposal_predicted_exact_descent: bool,
    pub transfer_direction_agrees: bool,
    pub transfer_predicted_exact_descent: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGradientAlignmentSample {
    pub group: &'static str,
    pub group_index: usize,
    pub coordinate: usize,
    pub selected_for_union_activity: bool,
    pub selected_for_rescue_strata: [bool; 3],
    pub any_rescue_exposed: bool,
    pub random_control_delta: i8,
    pub proposal: ProductionGradientSurfaceDelta,
    pub transfer: ProductionGradientSurfaceDelta,
    pub lanes: Vec<ProductionGradientLaneSample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGradientLaneTrace {
    pub lane: ProductionGradientProposalLane,
    pub health: ProductionGradientLaneHealth,
    pub proposal_summary: ProductionGradientAlignmentSummary,
    pub transfer_summary: ProductionGradientAlignmentSummary,
    pub rescue_exposed_trunk_proposal: ProductionGradientAlignmentSummary,
    pub rescue_exposed_trunk_transfer: ProductionGradientAlignmentSummary,
    pub natural_trunk_proposal: ProductionGradientAlignmentSummary,
    pub natural_trunk_transfer: ProductionGradientAlignmentSummary,
    pub output_head_proposal: ProductionGradientAlignmentSummary,
    pub output_head_transfer: ProductionGradientAlignmentSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGradientWindowBinding {
    pub document: usize,
    pub context_start: usize,
    pub target_offset: usize,
    pub separation: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGradientAlignmentGate {
    pub primary_lane: ProductionGradientProposalLane,
    pub proposal_fidelity_passed: bool,
    pub held_out_transfer_passed: bool,
    pub optimizer_refinement_authorized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionGradientCausalComparison {
    pub rescued_lane: ProductionGradientProposalLane,
    pub control_lane: ProductionGradientProposalLane,
    pub source_equivalence_verified: bool,
    pub rescue_exposed_trunk_coordinates: usize,
    pub aggregate_gradient_equal: usize,
    pub aggregate_gradient_magnitude_changed: usize,
    pub aggregate_gradient_sign_changed: usize,
    pub rescued_proposal_direction_agreements: usize,
    pub control_proposal_direction_agreements: usize,
    pub rescued_proposal_exact_descents: usize,
    pub control_proposal_exact_descents: usize,
    pub causal_separation_observed: bool,
    pub control_improves_proposal_fidelity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionGradientAlignmentTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub model_hash: u64,
    pub context_tokens: usize,
    pub documents_per_surface: usize,
    pub rescue_stratified_sampling: bool,
    pub include_mass_corrected_no_rescue: bool,
    pub include_systematic_fixed_mass: bool,
    pub coordinates_per_group: usize,
    pub sample_seed: u64,
    pub effective_learning_rate_shifts: [u8; 13],
    pub output_backward_shift: Option<u8>,
    pub probability_gradient_fractional_bits: u8,
    pub probability_normalization: &'static str,
    pub proposal_bindings: Vec<ProductionGradientWindowBinding>,
    pub transfer_bindings: Vec<ProductionGradientWindowBinding>,
    pub lanes: Vec<ProductionGradientLaneTrace>,
    pub samples: Vec<ProductionGradientAlignmentSample>,
    pub causal_comparison: Option<ProductionGradientCausalComparison>,
    pub gate: ProductionGradientAlignmentGate,
}

impl ProductionGradientAlignmentTrace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        let enhanced_audit = self.rescue_stratified_sampling
            || self.documents_per_surface != 0
            || self.include_mass_corrected_no_rescue
            || self.include_systematic_fixed_mass;
        if self.include_mass_corrected_no_rescue {
            write!(
                json,
                concat!(
                    "{{\"schema\":\"nsrl.production_gradient_alignment.v6\",",
                    "\"analysis_role\":\"calibration\",",
                    "\"objective\":\"integer_base2_softmax_nll_q20\",",
                    "\"claims\":{{\"proposal_surface\":\"backward_directional_fidelity\",",
                    "\"transfer_surface\":\"held_out_transfer_not_gradient_correctness\",",
                    "\"gradient_lanes\":\"surrogate_directions_not_derivatives_of_piecewise_integer_objective\",",
                    "\"causal_control\":\"plain_rhu_replaces_rescued_rhu_only_not_optimizer_promotion\"}},",
                    "\"profile\":\"{}\",\"parameter_count\":{},",
                    "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                    "\"token_stream_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\"}},",
                    "\"audit\":{{\"context_tokens\":{},\"documents_per_surface\":{},",
                    "\"rescue_stratified_sampling\":{},",
                    "\"include_mass_corrected_no_rescue\":true,",
                    "\"coordinates_per_group_per_stratum\":{},",
                    "\"sample_seed\":{},\"perturbations\":[-1,1],",
                    "\"boundary_coordinate_policy\":\"reject_audit\",",
                    "\"coordinate_sampling\":\"v2_public_lane_union_plus_source_specific_rescue_strata_control_evaluation_only\",",
                    "\"transfer_policy\":\"{}\"}},"
                ),
                self.profile,
                self.parameter_count,
                self.tokenizer_hash,
                self.token_stream_hash,
                self.model_hash,
                self.context_tokens,
                self.documents_per_surface,
                self.rescue_stratified_sampling,
                self.coordinates_per_group,
                self.sample_seed,
                if self.documents_per_surface == 0 {
                    "different_document_else_nonoverlap_context_plus_target"
                } else {
                    "strict_disjoint_document_blocks"
                },
            )
            .expect("writing causal gradient alignment JSON cannot fail");
        } else if enhanced_audit {
            write!(
                json,
                concat!(
                    "{{\"schema\":\"nsrl.production_gradient_alignment.v5\",",
                    "\"analysis_role\":\"calibration\",",
                    "\"objective\":\"integer_base2_softmax_nll_q20\",",
                    "\"claims\":{{\"proposal_surface\":\"backward_directional_fidelity\",",
                    "\"transfer_surface\":\"held_out_transfer_not_gradient_correctness\",",
                    "\"gradient_lanes\":\"surrogate_directions_not_derivatives_of_piecewise_integer_objective\"}},",
                    "\"profile\":\"{}\",\"parameter_count\":{},",
                    "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                    "\"token_stream_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\"}},",
                    "\"audit\":{{\"context_tokens\":{},\"documents_per_surface\":{},",
                    "\"rescue_stratified_sampling\":{},",
                    "\"coordinates_per_group_per_stratum\":{},",
                    "\"sample_seed\":{},\"perturbations\":[-1,1],",
                    "\"boundary_coordinate_policy\":\"reject_audit\",",
                    "\"coordinate_sampling\":\"{}\",\"transfer_policy\":\"{}\"}},"
                ),
                self.profile,
                self.parameter_count,
                self.tokenizer_hash,
                self.token_stream_hash,
                self.model_hash,
                self.context_tokens,
                self.documents_per_surface,
                self.rescue_stratified_sampling,
                self.coordinates_per_group,
                self.sample_seed,
                if self.rescue_stratified_sampling {
                    "shared_hash_sample_from_union_activity_and_source_specific_rescue_strata"
                } else {
                    "same_hash_sample_from_union_of_lane_activity"
                },
                if self.documents_per_surface == 0 {
                    "different_document_else_nonoverlap_context_plus_target"
                } else {
                    "strict_disjoint_document_blocks"
                },
            )
            .expect("writing enhanced gradient alignment JSON cannot fail");
        } else {
            write!(
                json,
                concat!(
                    "{{\"schema\":\"nsrl.production_gradient_alignment.v4\",",
                    "\"analysis_role\":\"calibration\",",
                    "\"objective\":\"integer_base2_softmax_nll_q20\",",
                    "\"claims\":{{\"proposal_surface\":\"backward_directional_fidelity\",",
                    "\"transfer_surface\":\"held_out_transfer_not_gradient_correctness\",",
                    "\"gradient_lanes\":\"surrogate_directions_not_derivatives_of_piecewise_integer_objective\"}},",
                    "\"profile\":\"{}\",\"parameter_count\":{},",
                    "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                    "\"token_stream_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\"}},",
                    "\"audit\":{{\"context_tokens\":{},\"coordinates_per_group\":{},",
                    "\"sample_seed\":{},\"perturbations\":[-1,1],",
                    "\"boundary_coordinate_policy\":\"reject_audit\",",
                    "\"coordinate_sampling\":\"same_hash_sample_from_union_of_lane_activity\",",
                    "\"transfer_policy\":\"different_document_else_nonoverlap_context_plus_target\"}},"
                ),
                self.profile,
                self.parameter_count,
                self.tokenizer_hash,
                self.token_stream_hash,
                self.model_hash,
                self.context_tokens,
                self.coordinates_per_group,
                self.sample_seed,
            )
            .expect("writing legacy gradient alignment JSON cannot fail");
        }
        write!(
            json,
            concat!(
                "\"systematic_fixed_mass\":{{\"enabled\":{},",
                "\"masses\":[32768,65536,262144],",
                "\"phase\":\"seeded_uniform_modulo_exact_weight_mass\",",
                "\"vocabulary_order\":\"token_id_ascending\",",
                "\"exact_mass_and_zero_sum_checked\":true}},"
            ),
            self.include_systematic_fixed_mass,
        )
        .expect("writing systematic fixed-mass contract cannot fail");
        json.push_str("\"training_numeric_contract\":{");
        json.push_str("\"effective_learning_rate_shifts\":{");
        for (index, (&name, &shift)) in GROUP_NAMES
            .iter()
            .zip(self.effective_learning_rate_shifts.iter())
            .enumerate()
        {
            if index != 0 {
                json.push(',');
            }
            write!(json, "\"{name}\":{shift}")
                .expect("writing gradient numeric contract cannot fail");
        }
        json.push_str("},\"output_backward_shift\":");
        match self.output_backward_shift {
            Some(shift) => {
                write!(json, "{shift}").expect("writing output backward shift cannot fail")
            }
            None => json.push_str("null"),
        }
        write!(
            json,
            ",\"probability_gradient_fractional_bits\":{},\"probability_normalization\":\"{}\"}},\"proposal_windows\":",
            self.probability_gradient_fractional_bits, self.probability_normalization,
        )
        .expect("writing probability gradient contract cannot fail");
        push_window_bindings(&mut json, &self.proposal_bindings);
        json.push_str(",\"transfer_windows\":");
        push_window_bindings(&mut json, &self.transfer_bindings);
        json.push_str(",\"lanes\":[");
        for (index, lane) in self.lanes.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            push_lane_trace_json(&mut json, lane);
        }
        json.push(']');
        if let Some(comparison) = self.causal_comparison {
            json.push_str(",\"causal_no_rescue_comparison\":");
            push_causal_comparison_json(&mut json, comparison);
        }
        json.push_str(",\"samples\":[");
        for (index, sample) in self.samples.iter().enumerate() {
            if index != 0 {
                json.push(',');
            }
            push_sample_json(&mut json, sample, enhanced_audit);
        }
        write!(
            json,
            concat!(
                "],\"gate\":{{\"primary_lane\":\"{}\",",
                "\"proposal_fidelity_passed\":{},\"held_out_transfer_passed\":{},",
                "\"optimizer_refinement_authorized\":{}}}}}\n"
            ),
            self.gate.primary_lane.as_str(),
            self.gate.proposal_fidelity_passed,
            self.gate.held_out_transfer_passed,
            self.gate.optimizer_refinement_authorized,
        )
        .expect("writing gradient alignment JSON cannot fail");
        json
    }
}

#[derive(Clone)]
pub(super) struct DocumentWindow {
    pub(super) document: usize,
    pub(super) context_start: usize,
    pub(super) context: Vec<u32>,
    pub(super) target: u32,
}

#[derive(Clone)]
pub(super) struct SurfaceEval {
    pub(super) nll_q20: u64,
    pub(super) losses_q20: Vec<u64>,
    pub(super) logits: Vec<Vec<i32>>,
}

struct SelectedCoordinate {
    group: usize,
    local: usize,
    selected_for_union_activity: bool,
    selected_for_rescue_strata: [bool; RESCUE_SOURCE_COUNT],
    gradients: [i64; LANE_COUNT],
    rescue_exposed: [bool; LANE_COUNT],
}

pub fn audit_production_gradient_alignment(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    training: ProductionFullTrainConfig,
    audit: ProductionGradientAlignmentConfig,
) -> Result<ProductionGradientAlignmentTrace, TrainError> {
    validate_audit_inputs(model, tokens, training, audit)?;
    let active_lanes = active_lanes(
        audit.include_mass_corrected_no_rescue,
        audit.include_systematic_fixed_mass,
    );
    let all_windows = document_windows_with_coordinates(tokens, training.context_tokens);
    let (proposal_windows, transfer_windows) = select_surfaces(&all_windows, audit)?;
    let ranges = parameter_group_ranges(model)?;
    let mut health = ProductionGradientProposalLane::WITH_SYSTEMATIC_AND_CAUSAL_NO_RESCUE
        .map(ProductionGradientLaneHealth::new);
    let mut selected_by_group = (0..GROUP_NAMES.len())
        .map(|_| Vec::<(u64, usize)>::new())
        .collect::<Vec<_>>();
    let mut rescue_selected_by_source = (0..RESCUE_SOURCE_COUNT)
        .map(|_| {
            (0..GROUP_NAMES.len())
                .map(|_| Vec::<(u64, usize)>::new())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut working = model.clone();

    // Pass one binds one coordinate sample to the union of activity from every
    // public lane plus a source-specific sample from each rescue-exposed
    // stratum. Every public lane is later evaluated on this same union.
    for window in &proposal_windows {
        for &lane in active_lanes {
            let lane_index = lane_index(lane);
            let snapshot = coarse_gradients_for_window_with_spec(
                &mut working,
                &window.context,
                window.target as usize,
                training,
                GradientProposalSpec::lane(
                    lane,
                    stochastic_seed(model, token_stream_hash, audit.sample_seed, lane, window),
                ),
            )?;
            observe_health(&mut health[lane_index], &snapshot)?;
            // The causal control is evaluation-only: allowing it into this
            // union would change the v2 coordinates it is meant to replay.
            if lane != ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue {
                observe_coordinate_candidates(
                    &snapshot.residuals,
                    &ranges,
                    &mut selected_by_group,
                    audit.coordinates_per_group,
                    audit.sample_seed,
                );
            }
            if audit.rescue_stratified_sampling
                && let Some((source_index, source)) = rescued_source_for_lane(lane)
            {
                let natural = coarse_gradients_for_window_with_spec(
                    &mut working,
                    &window.context,
                    window.target as usize,
                    training,
                    GradientProposalSpec::natural_reference(source),
                )?;
                observe_rescue_coordinate_candidates(
                    &snapshot.residuals,
                    &natural.residuals,
                    &ranges,
                    &mut rescue_selected_by_source[source_index],
                    audit.coordinates_per_group,
                    audit.sample_seed
                        ^ 0x5c4f_52ce_8b1d_a7e3
                        ^ (source_index as u64).rotate_left(17),
                );
            }
        }
    }

    let mut selected = Vec::new();
    for (group, values) in selected_by_group.iter().enumerate() {
        for &(_, local) in values {
            insert_selected_coordinate(&mut selected, group, local, None);
        }
    }
    for (source_index, groups) in rescue_selected_by_source.iter().enumerate() {
        for (group, values) in groups.iter().enumerate() {
            for &(_, local) in values {
                insert_selected_coordinate(&mut selected, group, local, Some(source_index));
            }
        }
    }
    if selected.is_empty() {
        return Err(TrainError::CoreRejected(
            "production_alignment_no_active_coordinates",
        ));
    }

    // Pass two accumulates only the fixed sample and attributes rescue by
    // comparing each rescued source against its no-rescue RHU reference.
    for window in &proposal_windows {
        for &lane in active_lanes {
            let snapshot = coarse_gradients_for_window_with_spec(
                &mut working,
                &window.context,
                window.target as usize,
                training,
                GradientProposalSpec::lane(
                    lane,
                    stochastic_seed(model, token_stream_hash, audit.sample_seed, lane, window),
                ),
            )?;
            accumulate_selected(
                &mut selected,
                &ranges,
                lane_index(lane),
                &snapshot.residuals,
            )?;
        }
        for source in [
            GradientSource::NormalizedProbability,
            GradientSource::MassCorrectedNormalizedProbability,
            GradientSource::ReciprocalFreeWeights,
        ] {
            let rescued_lane = rescued_lane(source);
            let rescued = coarse_gradients_for_window_with_spec(
                &mut working,
                &window.context,
                window.target as usize,
                training,
                GradientProposalSpec::lane(
                    rescued_lane,
                    stochastic_seed(
                        model,
                        token_stream_hash,
                        audit.sample_seed,
                        rescued_lane,
                        window,
                    ),
                ),
            )?;
            let natural = coarse_gradients_for_window_with_spec(
                &mut working,
                &window.context,
                window.target as usize,
                training,
                GradientProposalSpec::natural_reference(source),
            )?;
            let affected_lanes: &[usize] = match source {
                GradientSource::NormalizedProbability => &[0],
                // The control is labeled by exposure under its matched
                // rescued counterfactual so both lanes share the same stratum.
                GradientSource::MassCorrectedNormalizedProbability => &[
                    ProductionGradientProposalLane::MassCorrectedNormalized.stable_index(),
                    ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue.stable_index(),
                ],
                // The late-quantization lanes are controls evaluated on the
                // same coordinates where the rescued reciprocal-free lane
                // differs from its natural no-rescue reference.
                GradientSource::ReciprocalFreeWeights => &[2, 3, 4],
                GradientSource::SystematicFixedMassK15
                | GradientSource::SystematicFixedMassK16
                | GradientSource::SystematicFixedMassK18 => &[],
            };
            for coordinate in &mut selected {
                let global = ranges[coordinate.group].start + coordinate.local;
                if rescued.residuals[global] != natural.residuals[global] {
                    for &lane_index in affected_lanes {
                        coordinate.rescue_exposed[lane_index] = true;
                    }
                }
            }
        }
    }

    let current_proposal = evaluate_surface(model, &proposal_windows)?;
    let current_transfer = evaluate_surface(model, &transfer_windows)?;
    let mut samples = Vec::with_capacity(selected.len());
    for coordinate in selected {
        if !can_perturb_both(&working, coordinate.group, coordinate.local) {
            return Err(TrainError::CoreRejected(
                "production_alignment_boundary_coordinate_rejected",
            ));
        }
        set_parameter_delta(&mut working, coordinate.group, coordinate.local, 1)?;
        let plus_proposal = evaluate_surface(&working, &proposal_windows)?;
        let plus_transfer = evaluate_surface(&working, &transfer_windows)?;
        set_parameter_delta(&mut working, coordinate.group, coordinate.local, -2)?;
        let minus_proposal = evaluate_surface(&working, &proposal_windows)?;
        let minus_transfer = evaluate_surface(&working, &transfer_windows)?;
        set_parameter_delta(&mut working, coordinate.group, coordinate.local, 1)?;

        let proposal = surface_delta(&current_proposal, &plus_proposal, &minus_proposal);
        let transfer = surface_delta(&current_transfer, &plus_transfer, &minus_transfer);
        let random_control_delta =
            random_control_delta(audit.sample_seed, coordinate.group, coordinate.local);
        let lanes = active_lanes
            .iter()
            .enumerate()
            .map(|(index, &lane)| {
                lane_sample(
                    lane,
                    coordinate.rescue_exposed[index],
                    coordinate.gradients[index],
                    proposal,
                    transfer,
                )
            })
            .collect();
        samples.push(ProductionGradientAlignmentSample {
            group: GROUP_NAMES[coordinate.group],
            group_index: coordinate.group,
            coordinate: coordinate.local,
            selected_for_union_activity: coordinate.selected_for_union_activity,
            selected_for_rescue_strata: coordinate.selected_for_rescue_strata,
            any_rescue_exposed: coordinate.rescue_exposed.iter().any(|&value| value),
            random_control_delta,
            proposal,
            transfer,
            lanes,
        });
    }

    debug_assert_eq!(working, *model);
    let lanes = build_lane_traces(health, &samples, active_lanes);
    let primary_lane = ProductionGradientProposalLane::MassCorrectedNormalized;
    let primary = lanes
        .iter()
        .find(|lane| lane.lane == primary_lane)
        .expect("primary gradient lane is active");
    let proposal_fidelity_passed =
        alignment_gate_pass(primary.rescue_exposed_trunk_proposal, primary.health);
    let held_out_transfer_passed =
        alignment_gate_pass(primary.rescue_exposed_trunk_transfer, primary.health);
    let causal_comparison = if audit.include_mass_corrected_no_rescue {
        Some(build_causal_comparison(&lanes, &samples))
    } else {
        None
    };
    Ok(ProductionGradientAlignmentTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        model_hash: model.model_hash(),
        context_tokens: training.context_tokens,
        documents_per_surface: audit.documents_per_surface,
        rescue_stratified_sampling: audit.rescue_stratified_sampling,
        include_mass_corrected_no_rescue: audit.include_mass_corrected_no_rescue,
        include_systematic_fixed_mass: audit.include_systematic_fixed_mass,
        coordinates_per_group: audit.coordinates_per_group,
        sample_seed: audit.sample_seed,
        effective_learning_rate_shifts: effective_learning_rate_shifts(training),
        output_backward_shift: training.output_backward_shift,
        probability_gradient_fractional_bits: training.probability_gradient_fractional_bits,
        probability_normalization: training.probability_normalization.as_str(),
        proposal_bindings: proposal_windows
            .iter()
            .map(|window| window_binding(window, "proposal"))
            .collect(),
        transfer_bindings: transfer_windows
            .iter()
            .map(|window| {
                let separation = if proposal_windows
                    .iter()
                    .all(|proposal| proposal.document != window.document)
                {
                    "different_document"
                } else {
                    "nonoverlap_context_plus_target"
                };
                window_binding(window, separation)
            })
            .collect(),
        lanes,
        samples,
        causal_comparison,
        gate: ProductionGradientAlignmentGate {
            primary_lane,
            proposal_fidelity_passed,
            held_out_transfer_passed,
            // A causal calibration cannot by itself authorize optimizer
            // refinement, even if its bounded direction summaries pass.
            optimizer_refinement_authorized: false,
        },
    })
}

fn validate_audit_inputs(
    model: &ProductionModelV1,
    tokens: &[u32],
    training: ProductionFullTrainConfig,
    audit: ProductionGradientAlignmentConfig,
) -> Result<(), TrainError> {
    model.validate()?;
    if audit.proposal_windows == 0
        || audit.transfer_windows == 0
        || audit.coordinates_per_group == 0
        || (audit.include_mass_corrected_no_rescue && !audit.rescue_stratified_sampling)
        || training.context_tokens == 0
        || training.context_tokens > model.config.context_tokens
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    Ok(())
}

pub(super) fn document_windows_with_coordinates(
    tokens: &[u32],
    context: usize,
) -> Vec<DocumentWindow> {
    let mut windows = Vec::new();
    let mut document_tokens = Vec::new();
    let mut active = false;
    let mut document = 0_usize;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document_tokens.clear();
            active = true;
        } else if token == EOS_TOKEN_ID {
            if active && document_tokens.len() > context {
                for start in 0..document_tokens.len() - context {
                    windows.push(DocumentWindow {
                        document,
                        context_start: start,
                        context: document_tokens[start..start + context].to_vec(),
                        target: document_tokens[start + context],
                    });
                }
            }
            document_tokens.clear();
            if active {
                document = document.saturating_add(1);
            }
            active = false;
        } else if active {
            document_tokens.push(token);
        }
    }
    windows
}

pub(super) fn select_surfaces(
    windows: &[DocumentWindow],
    audit: ProductionGradientAlignmentConfig,
) -> Result<(Vec<DocumentWindow>, Vec<DocumentWindow>), TrainError> {
    if audit.documents_per_surface != 0 {
        return select_document_block_surfaces(windows, audit);
    }
    let proposal = windows
        .iter()
        .take(audit.proposal_windows)
        .cloned()
        .collect::<Vec<_>>();
    if proposal.len() != audit.proposal_windows {
        return Err(TrainError::InvalidConfig);
    }
    let mut transfer = windows
        .iter()
        .filter(|candidate| {
            proposal
                .iter()
                .all(|source| source.document != candidate.document)
        })
        .take(audit.transfer_windows)
        .cloned()
        .collect::<Vec<_>>();
    if transfer.len() < audit.transfer_windows {
        for candidate in windows {
            if transfer.len() >= audit.transfer_windows {
                break;
            }
            if transfer
                .iter()
                .any(|existing| same_window(existing, candidate))
                || !proposal
                    .iter()
                    .all(|source| windows_do_not_overlap(source, candidate))
            {
                continue;
            }
            transfer.push(candidate.clone());
        }
    }
    if transfer.len() != audit.transfer_windows {
        return Err(TrainError::CoreRejected(
            "production_alignment_no_separated_transfer_surface",
        ));
    }
    Ok((proposal, transfer))
}

fn select_document_block_surfaces(
    windows: &[DocumentWindow],
    audit: ProductionGradientAlignmentConfig,
) -> Result<(Vec<DocumentWindow>, Vec<DocumentWindow>), TrainError> {
    if audit.proposal_windows < audit.documents_per_surface
        || audit.transfer_windows < audit.documents_per_surface
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut documents = Vec::new();
    for window in windows {
        if documents.last().copied() != Some(window.document) {
            documents.push(window.document);
        }
    }
    let required_documents = audit
        .documents_per_surface
        .checked_mul(2)
        .ok_or(TrainError::InvalidConfig)?;
    if documents.len() < required_documents {
        return Err(TrainError::CoreRejected(
            "production_alignment_insufficient_document_blocks",
        ));
    }
    let proposal_documents = &documents[..audit.documents_per_surface];
    let transfer_documents = &documents[audit.documents_per_surface..required_documents];
    let proposal =
        select_balanced_document_windows(windows, proposal_documents, audit.proposal_windows)?;
    let transfer =
        select_balanced_document_windows(windows, transfer_documents, audit.transfer_windows)?;
    Ok((proposal, transfer))
}

fn select_balanced_document_windows(
    windows: &[DocumentWindow],
    documents: &[usize],
    count: usize,
) -> Result<Vec<DocumentWindow>, TrainError> {
    let grouped = documents
        .iter()
        .map(|document| {
            windows
                .iter()
                .filter(|window| window.document == *document)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut selected = Vec::with_capacity(count);
    let mut row = 0_usize;
    while selected.len() < count {
        let mut progressed = false;
        for group in &grouped {
            if let Some(window) = group.get(row) {
                selected.push((*window).clone());
                progressed = true;
                if selected.len() == count {
                    break;
                }
            }
        }
        if !progressed {
            return Err(TrainError::CoreRejected(
                "production_alignment_insufficient_windows_in_document_blocks",
            ));
        }
        row = row.saturating_add(1);
    }
    Ok(selected)
}

fn same_window(left: &DocumentWindow, right: &DocumentWindow) -> bool {
    left.document == right.document && left.context_start == right.context_start
}

fn windows_do_not_overlap(left: &DocumentWindow, right: &DocumentWindow) -> bool {
    if left.document != right.document {
        return true;
    }
    let left_end = left.context_start + left.context.len();
    let right_end = right.context_start + right.context.len();
    left_end < right.context_start || right_end < left.context_start
}

fn window_binding(
    window: &DocumentWindow,
    separation: &'static str,
) -> ProductionGradientWindowBinding {
    ProductionGradientWindowBinding {
        document: window.document,
        context_start: window.context_start,
        target_offset: window.context_start + window.context.len(),
        separation,
    }
}

fn stochastic_seed(
    model: &ProductionModelV1,
    token_stream_hash: u64,
    sample_seed: u64,
    lane: ProductionGradientProposalLane,
    window: &DocumentWindow,
) -> u64 {
    splitmix64(
        sample_seed
            ^ model.model_hash()
            ^ token_stream_hash.rotate_left(7)
            ^ (lane_index(lane) as u64).rotate_left(19)
            ^ (window.document as u64).rotate_left(31)
            ^ window.context_start as u64,
    )
}

fn active_lanes(
    include_mass_corrected_no_rescue: bool,
    include_systematic_fixed_mass: bool,
) -> &'static [ProductionGradientProposalLane] {
    match (
        include_mass_corrected_no_rescue,
        include_systematic_fixed_mass,
    ) {
        (false, false) => &ProductionGradientProposalLane::ALL,
        (true, false) => &ProductionGradientProposalLane::WITH_CAUSAL_NO_RESCUE,
        (false, true) => &ProductionGradientProposalLane::WITH_SYSTEMATIC,
        (true, true) => &ProductionGradientProposalLane::WITH_SYSTEMATIC_AND_CAUSAL_NO_RESCUE,
    }
}

fn rescued_lane(source: GradientSource) -> ProductionGradientProposalLane {
    match source {
        GradientSource::NormalizedProbability => ProductionGradientProposalLane::NormalizedRescued,
        GradientSource::MassCorrectedNormalizedProbability => {
            ProductionGradientProposalLane::MassCorrectedNormalized
        }
        GradientSource::ReciprocalFreeWeights => {
            ProductionGradientProposalLane::ReciprocalFreeRescued
        }
        GradientSource::SystematicFixedMassK15
        | GradientSource::SystematicFixedMassK16
        | GradientSource::SystematicFixedMassK18 => {
            unreachable!("systematic fixed-mass lanes have no rescued counterpart")
        }
    }
}

fn rescued_source_for_lane(
    lane: ProductionGradientProposalLane,
) -> Option<(usize, GradientSource)> {
    match lane {
        ProductionGradientProposalLane::NormalizedRescued => {
            Some((0, GradientSource::NormalizedProbability))
        }
        ProductionGradientProposalLane::MassCorrectedNormalized => {
            Some((1, GradientSource::MassCorrectedNormalizedProbability))
        }
        ProductionGradientProposalLane::ReciprocalFreeRescued => {
            Some((2, GradientSource::ReciprocalFreeWeights))
        }
        ProductionGradientProposalLane::ReciprocalFreeLateRhu
        | ProductionGradientProposalLane::ReciprocalFreeLateStochastic
        | ProductionGradientProposalLane::SystematicFixedMassK15
        | ProductionGradientProposalLane::SystematicFixedMassK16
        | ProductionGradientProposalLane::SystematicFixedMassK18
        | ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue => None,
    }
}

fn lane_index(lane: ProductionGradientProposalLane) -> usize {
    lane.stable_index()
}

fn observe_health(
    health: &mut ProductionGradientLaneHealth,
    snapshot: &super::training::CoarseGradientSnapshot,
) -> Result<(), TrainError> {
    health.output_gradient_sum_min = health
        .output_gradient_sum_min
        .min(snapshot.output_gradient_sum);
    health.output_gradient_sum_max = health
        .output_gradient_sum_max
        .max(snapshot.output_gradient_sum);
    health.output_gradient_l1_min = health
        .output_gradient_l1_min
        .min(snapshot.output_gradient_l1);
    health.output_gradient_l1_max = health
        .output_gradient_l1_max
        .max(snapshot.output_gradient_l1);
    health.output_gradient_max_abs = health
        .output_gradient_max_abs
        .max(snapshot.output_gradient_max_abs);
    health.output_gradient_trace_hash = snapshot
        .output_gradient_vector_hash
        .to_le_bytes()
        .into_iter()
        .fold(health.output_gradient_trace_hash, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(super::FNV_PRIME)
        });
    health.backward_ste_rescue_count = checked_add_u64(
        health.backward_ste_rescue_count,
        snapshot.backward_ste_rescue_count,
    )?;
    health.backward_quantization_count = checked_add_u64(
        health.backward_quantization_count,
        snapshot.backward_quantization_count,
    )?;
    health.stochastic_round_up_count = checked_add_u64(
        health.stochastic_round_up_count,
        snapshot.stochastic_round_up_count,
    )?;
    health.gradient_saturation_count = health
        .gradient_saturation_count
        .checked_add(snapshot.gradient_saturation_count)
        .ok_or(TrainError::CoreRejected(
            "production_alignment_health_counter_overflow",
        ))?;
    health.residual_saturation_count = health
        .residual_saturation_count
        .checked_add(snapshot.residual_saturation_count)
        .ok_or(TrainError::CoreRejected(
            "production_alignment_health_counter_overflow",
        ))?;
    Ok(())
}

fn checked_add_u64(left: u64, right: u64) -> Result<u64, TrainError> {
    left.checked_add(right).ok_or(TrainError::CoreRejected(
        "production_alignment_health_counter_overflow",
    ))
}

fn observe_coordinate_candidates(
    gradients: &[i64],
    ranges: &[Range<usize>; 13],
    selected: &mut [Vec<(u64, usize)>],
    limit: usize,
    seed: u64,
) {
    for (group, range) in ranges.iter().enumerate() {
        for (local, &gradient) in gradients[range.clone()].iter().enumerate() {
            if gradient == 0 {
                continue;
            }
            let key = splitmix64(seed ^ (group as u64).rotate_left(33) ^ local as u64);
            let values = &mut selected[group];
            if values.iter().any(|&(_, coordinate)| coordinate == local) {
                continue;
            }
            if values.len() < limit {
                values.push((key, local));
                values.sort_unstable();
            } else if (key, local) < values[limit - 1] {
                values[limit - 1] = (key, local);
                values.sort_unstable();
            }
        }
    }
}

fn observe_rescue_coordinate_candidates(
    rescued: &[i64],
    natural: &[i64],
    ranges: &[Range<usize>; 13],
    selected: &mut [Vec<(u64, usize)>],
    limit: usize,
    seed: u64,
) {
    debug_assert_eq!(rescued.len(), natural.len());
    for (group, range) in ranges.iter().enumerate() {
        for (local, (&rescued_gradient, &natural_gradient)) in rescued[range.clone()]
            .iter()
            .zip(&natural[range.clone()])
            .enumerate()
        {
            if rescued_gradient == natural_gradient {
                continue;
            }
            let key = splitmix64(seed ^ (group as u64).rotate_left(33) ^ local as u64);
            let values = &mut selected[group];
            if values.iter().any(|&(_, coordinate)| coordinate == local) {
                continue;
            }
            if values.len() < limit {
                values.push((key, local));
                values.sort_unstable();
            } else if (key, local) < values[limit - 1] {
                values[limit - 1] = (key, local);
                values.sort_unstable();
            }
        }
    }
}

fn insert_selected_coordinate(
    selected: &mut Vec<SelectedCoordinate>,
    group: usize,
    local: usize,
    rescue_source: Option<usize>,
) {
    let index = selected
        .iter()
        .position(|coordinate| coordinate.group == group && coordinate.local == local);
    let coordinate = if let Some(index) = index {
        &mut selected[index]
    } else {
        selected.push(SelectedCoordinate {
            group,
            local,
            selected_for_union_activity: false,
            selected_for_rescue_strata: [false; RESCUE_SOURCE_COUNT],
            gradients: [0; LANE_COUNT],
            rescue_exposed: [false; LANE_COUNT],
        });
        selected.last_mut().expect("selected coordinate was pushed")
    };
    if let Some(source) = rescue_source {
        coordinate.selected_for_rescue_strata[source] = true;
    } else {
        coordinate.selected_for_union_activity = true;
    }
}

fn accumulate_selected(
    selected: &mut [SelectedCoordinate],
    ranges: &[Range<usize>; 13],
    lane: usize,
    gradients: &[i64],
) -> Result<(), TrainError> {
    for coordinate in selected {
        let global = ranges[coordinate.group].start + coordinate.local;
        coordinate.gradients[lane] = coordinate.gradients[lane]
            .checked_add(gradients[global])
            .ok_or(TrainError::CoreRejected(
                "production_alignment_gradient_accumulator_overflow",
            ))?;
    }
    Ok(())
}

pub(super) fn evaluate_surface(
    model: &ProductionModelV1,
    windows: &[DocumentWindow],
) -> Result<SurfaceEval, TrainError> {
    let mut nll_q20 = 0_u64;
    let mut losses_q20 = Vec::with_capacity(windows.len());
    let mut logits = Vec::with_capacity(windows.len());
    for window in windows {
        let forward = forward_production_model(model, &window.context)?;
        let loss = base2_softmax_nll_q20(
            &forward.logits_q8,
            window.target as usize,
            ZERO_PROBABILITY_FLOOR_Q20,
        )
        .ok_or(TrainError::CoreRejected("production_alignment_nll"))?;
        nll_q20 = nll_q20.checked_add(loss).ok_or(TrainError::CoreRejected(
            "production_alignment_nll_accumulator_overflow",
        ))?;
        losses_q20.push(loss);
        logits.push(forward.logits_q8);
    }
    Ok(SurfaceEval {
        nll_q20,
        losses_q20,
        logits,
    })
}

fn surface_delta(
    current: &SurfaceEval,
    plus: &SurfaceEval,
    minus: &SurfaceEval,
) -> ProductionGradientSurfaceDelta {
    ProductionGradientSurfaceDelta {
        current_nll_q20: current.nll_q20,
        plus_one_nll_q20: plus.nll_q20,
        minus_one_nll_q20: minus.nll_q20,
        better_neighbor_delta: match plus.nll_q20.cmp(&minus.nll_q20) {
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Greater => -1,
            std::cmp::Ordering::Equal => 0,
        },
        exact_descent_available: plus.nll_q20.min(minus.nll_q20) < current.nll_q20,
        plus_function_visible: plus.logits != current.logits,
        minus_function_visible: minus.logits != current.logits,
    }
}

fn lane_sample(
    lane: ProductionGradientProposalLane,
    rescue_exposed: bool,
    gradient: i64,
    proposal: ProductionGradientSurfaceDelta,
    transfer: ProductionGradientSurfaceDelta,
) -> ProductionGradientLaneSample {
    let predicted_parameter_delta = match gradient.cmp(&0) {
        std::cmp::Ordering::Greater => -1,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
    };
    ProductionGradientLaneSample {
        lane,
        rescue_exposed,
        coarse_gradient: gradient,
        predicted_parameter_delta,
        proposal_direction_agrees: predicted_parameter_delta != 0
            && predicted_parameter_delta == proposal.better_neighbor_delta,
        proposal_predicted_exact_descent: predicted_loss(proposal, predicted_parameter_delta)
            .is_some_and(|loss| loss < proposal.current_nll_q20),
        transfer_direction_agrees: predicted_parameter_delta != 0
            && predicted_parameter_delta == transfer.better_neighbor_delta,
        transfer_predicted_exact_descent: predicted_loss(transfer, predicted_parameter_delta)
            .is_some_and(|loss| loss < transfer.current_nll_q20),
    }
}

fn predicted_loss(surface: ProductionGradientSurfaceDelta, delta: i8) -> Option<u64> {
    match delta {
        -1 => Some(surface.minus_one_nll_q20),
        1 => Some(surface.plus_one_nll_q20),
        _ => None,
    }
}

fn random_control_delta(seed: u64, group: usize, coordinate: usize) -> i8 {
    if splitmix64(seed ^ 0xd1b5_4a32_d192_ed03 ^ (group as u64).rotate_left(29) ^ coordinate as u64)
        & 1
        == 0
    {
        -1
    } else {
        1
    }
}

fn build_lane_traces(
    health: [ProductionGradientLaneHealth; LANE_COUNT],
    samples: &[ProductionGradientAlignmentSample],
    active_lanes: &[ProductionGradientProposalLane],
) -> Vec<ProductionGradientLaneTrace> {
    active_lanes
        .iter()
        .enumerate()
        .map(|(index, &lane)| {
            let mut trace = ProductionGradientLaneTrace {
                lane,
                health: health[lane_index(lane)],
                proposal_summary: ProductionGradientAlignmentSummary::default(),
                transfer_summary: ProductionGradientAlignmentSummary::default(),
                rescue_exposed_trunk_proposal: ProductionGradientAlignmentSummary::default(),
                rescue_exposed_trunk_transfer: ProductionGradientAlignmentSummary::default(),
                natural_trunk_proposal: ProductionGradientAlignmentSummary::default(),
                natural_trunk_transfer: ProductionGradientAlignmentSummary::default(),
                output_head_proposal: ProductionGradientAlignmentSummary::default(),
                output_head_transfer: ProductionGradientAlignmentSummary::default(),
            };
            for sample in samples {
                let lane_sample = &sample.lanes[index];
                observe_summary(
                    &mut trace.proposal_summary,
                    sample,
                    lane_sample,
                    sample.proposal,
                    true,
                );
                observe_summary(
                    &mut trace.transfer_summary,
                    sample,
                    lane_sample,
                    sample.transfer,
                    false,
                );
                let (proposal_subset, transfer_subset) = if sample.group_index >= 11 {
                    (
                        &mut trace.output_head_proposal,
                        &mut trace.output_head_transfer,
                    )
                } else if lane_sample.rescue_exposed {
                    (
                        &mut trace.rescue_exposed_trunk_proposal,
                        &mut trace.rescue_exposed_trunk_transfer,
                    )
                } else {
                    (
                        &mut trace.natural_trunk_proposal,
                        &mut trace.natural_trunk_transfer,
                    )
                };
                observe_summary(proposal_subset, sample, lane_sample, sample.proposal, true);
                observe_summary(transfer_subset, sample, lane_sample, sample.transfer, false);
            }
            trace
        })
        .collect()
}

fn build_causal_comparison(
    lanes: &[ProductionGradientLaneTrace],
    samples: &[ProductionGradientAlignmentSample],
) -> ProductionGradientCausalComparison {
    let rescued_lane = ProductionGradientProposalLane::MassCorrectedNormalized;
    let control_lane = ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue;
    let rescued = lanes
        .iter()
        .find(|lane| lane.lane == rescued_lane)
        .expect("rescued mass-corrected lane is active");
    let control = lanes
        .iter()
        .find(|lane| lane.lane == control_lane)
        .expect("no-rescue mass-corrected lane is active");
    let source_equivalence_verified =
        rescued.health.output_gradient_trace_hash == control.health.output_gradient_trace_hash;
    let mut rescue_exposed_trunk_coordinates = 0_usize;
    let mut aggregate_gradient_equal = 0_usize;
    let mut aggregate_gradient_magnitude_changed = 0_usize;
    let mut aggregate_gradient_sign_changed = 0_usize;
    for sample in samples.iter().filter(|sample| sample.group_index < 11) {
        let rescued_sample = sample
            .lanes
            .iter()
            .find(|lane| lane.lane == rescued_lane)
            .expect("rescued sample lane is active");
        if !rescued_sample.rescue_exposed {
            continue;
        }
        let control_sample = sample
            .lanes
            .iter()
            .find(|lane| lane.lane == control_lane)
            .expect("control sample lane is active");
        rescue_exposed_trunk_coordinates = rescue_exposed_trunk_coordinates.saturating_add(1);
        aggregate_gradient_equal = aggregate_gradient_equal.saturating_add(usize::from(
            rescued_sample.coarse_gradient == control_sample.coarse_gradient,
        ));
        aggregate_gradient_magnitude_changed =
            aggregate_gradient_magnitude_changed.saturating_add(usize::from(
                rescued_sample.coarse_gradient.unsigned_abs()
                    != control_sample.coarse_gradient.unsigned_abs(),
            ));
        aggregate_gradient_sign_changed =
            aggregate_gradient_sign_changed.saturating_add(usize::from(
                rescued_sample.coarse_gradient.signum() != control_sample.coarse_gradient.signum(),
            ));
    }
    let rescued_summary = rescued.rescue_exposed_trunk_proposal;
    let control_summary = control.rescue_exposed_trunk_proposal;
    let control_improves_proposal_fidelity = (control_summary.direction_agreements
        >= rescued_summary.direction_agreements
        && control_summary.predicted_exact_descents >= rescued_summary.predicted_exact_descents)
        && (control_summary.direction_agreements > rescued_summary.direction_agreements
            || control_summary.predicted_exact_descents > rescued_summary.predicted_exact_descents);
    ProductionGradientCausalComparison {
        rescued_lane,
        control_lane,
        source_equivalence_verified,
        rescue_exposed_trunk_coordinates,
        aggregate_gradient_equal,
        aggregate_gradient_magnitude_changed,
        aggregate_gradient_sign_changed,
        rescued_proposal_direction_agreements: rescued_summary.direction_agreements,
        control_proposal_direction_agreements: control_summary.direction_agreements,
        rescued_proposal_exact_descents: rescued_summary.predicted_exact_descents,
        control_proposal_exact_descents: control_summary.predicted_exact_descents,
        causal_separation_observed: source_equivalence_verified
            && (aggregate_gradient_magnitude_changed > 0 || aggregate_gradient_sign_changed > 0),
        control_improves_proposal_fidelity,
    }
}

fn observe_summary(
    summary: &mut ProductionGradientAlignmentSummary,
    sample: &ProductionGradientAlignmentSample,
    lane: &ProductionGradientLaneSample,
    surface: ProductionGradientSurfaceDelta,
    proposal_surface: bool,
) {
    summary.sampled_coordinates = summary.sampled_coordinates.saturating_add(1);
    let proposed = lane.predicted_parameter_delta != 0;
    summary.proposed_coordinates = summary
        .proposed_coordinates
        .saturating_add(usize::from(proposed));
    let comparable = proposed && surface.better_neighbor_delta != 0;
    summary.direction_comparable = summary
        .direction_comparable
        .saturating_add(usize::from(comparable));
    let agrees = if proposal_surface {
        lane.proposal_direction_agrees
    } else {
        lane.transfer_direction_agrees
    };
    summary.direction_agreements = summary
        .direction_agreements
        .saturating_add(usize::from(comparable && agrees));
    summary.random_direction_agreements =
        summary
            .random_direction_agreements
            .saturating_add(usize::from(
                comparable && sample.random_control_delta == surface.better_neighbor_delta,
            ));
    summary.exact_descent_available = summary
        .exact_descent_available
        .saturating_add(usize::from(proposed && surface.exact_descent_available));
    let predicted_descent = if proposal_surface {
        lane.proposal_predicted_exact_descent
    } else {
        lane.transfer_predicted_exact_descent
    };
    summary.predicted_exact_descents = summary
        .predicted_exact_descents
        .saturating_add(usize::from(proposed && predicted_descent));
    let random_loss = predicted_loss(surface, sample.random_control_delta)
        .expect("random control is a unit direction");
    summary.random_exact_descents = summary.random_exact_descents.saturating_add(usize::from(
        proposed && random_loss < surface.current_nll_q20,
    ));
    let visible = match lane.predicted_parameter_delta {
        -1 => surface.minus_function_visible,
        1 => surface.plus_function_visible,
        _ => false,
    };
    summary.function_visible = summary
        .function_visible
        .saturating_add(usize::from(visible));
    summary.objective_ties = summary.objective_ties.saturating_add(usize::from(
        proposed && surface.plus_one_nll_q20 == surface.minus_one_nll_q20,
    ));
}

fn alignment_gate_pass(
    summary: ProductionGradientAlignmentSummary,
    health: ProductionGradientLaneHealth,
) -> bool {
    summary.proposed_coordinates > 0
        && summary.direction_comparable > 0
        && summary.direction_agreements > summary.random_direction_agreements
        && summary.predicted_exact_descents > summary.random_exact_descents
        && health.gradient_saturation_count == 0
        && health.residual_saturation_count == 0
}

fn parameter_group_ranges(model: &ProductionModelV1) -> Result<[Range<usize>; 13], TrainError> {
    let lengths = [
        model.embeddings.len(),
        model.attention_rms_weights.len(),
        model.mlp_rms_weights.len(),
        model.final_rms_weights.len(),
        model.q_weights.len(),
        model.k_weights.len(),
        model.v_weights.len(),
        model.o_weights.len(),
        model.up_weights.len(),
        model.gate_weights.len(),
        model.down_weights.len(),
        model.output_weights.len(),
        model.output_bias_q8.len(),
    ];
    let mut cursor = 0_usize;
    let mut ranges = std::array::from_fn(|_| 0..0);
    for (index, length) in lengths.into_iter().enumerate() {
        let end = cursor
            .checked_add(length)
            .ok_or(TrainError::InvalidConfig)?;
        ranges[index] = cursor..end;
        cursor = end;
    }
    if cursor != model.parameter_count() {
        return Err(TrainError::InvalidModel(
            "production alignment parameter ranges mismatch",
        ));
    }
    Ok(ranges)
}

pub(super) fn can_perturb_both(model: &ProductionModelV1, group: usize, index: usize) -> bool {
    match group {
        0 => model.embeddings[index] > i16::MIN && model.embeddings[index] < i16::MAX,
        1 => {
            model.attention_rms_weights[index] > i16::MIN
                && model.attention_rms_weights[index] < i16::MAX
        }
        2 => model.mlp_rms_weights[index] > i16::MIN && model.mlp_rms_weights[index] < i16::MAX,
        3 => model.final_rms_weights[index] > i16::MIN && model.final_rms_weights[index] < i16::MAX,
        4 => model.q_weights[index] > i8::MIN && model.q_weights[index] < i8::MAX,
        5 => model.k_weights[index] > i8::MIN && model.k_weights[index] < i8::MAX,
        6 => model.v_weights[index] > i8::MIN && model.v_weights[index] < i8::MAX,
        7 => model.o_weights[index] > i8::MIN && model.o_weights[index] < i8::MAX,
        8 => model.up_weights[index] > i8::MIN && model.up_weights[index] < i8::MAX,
        9 => model.gate_weights[index] > i8::MIN && model.gate_weights[index] < i8::MAX,
        10 => model.down_weights[index] > i8::MIN && model.down_weights[index] < i8::MAX,
        11 => model.output_weights[index] > i16::MIN && model.output_weights[index] < i16::MAX,
        12 => model.output_bias_q8[index] > i32::MIN && model.output_bias_q8[index] < i32::MAX,
        _ => false,
    }
}

pub(super) fn set_parameter_delta(
    model: &mut ProductionModelV1,
    group: usize,
    index: usize,
    delta: i8,
) -> Result<(), TrainError> {
    macro_rules! add_delta {
        ($values:expr, $type:ty) => {{
            let value = $values.get_mut(index).ok_or(TrainError::InvalidConfig)?;
            *value = <$type>::try_from(i64::from(*value) + i64::from(delta))
                .map_err(|_| TrainError::InvalidConfig)?;
        }};
    }
    match group {
        0 => add_delta!(model.embeddings, i16),
        1 => add_delta!(model.attention_rms_weights, i16),
        2 => add_delta!(model.mlp_rms_weights, i16),
        3 => add_delta!(model.final_rms_weights, i16),
        4 => add_delta!(model.q_weights, i8),
        5 => add_delta!(model.k_weights, i8),
        6 => add_delta!(model.v_weights, i8),
        7 => add_delta!(model.o_weights, i8),
        8 => add_delta!(model.up_weights, i8),
        9 => add_delta!(model.gate_weights, i8),
        10 => add_delta!(model.down_weights, i8),
        11 => add_delta!(model.output_weights, i16),
        12 => add_delta!(model.output_bias_q8, i32),
        _ => return Err(TrainError::InvalidConfig),
    }
    Ok(())
}

fn push_window_bindings(json: &mut String, windows: &[ProductionGradientWindowBinding]) {
    json.push('[');
    for (index, window) in windows.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"document\":{},\"context_start\":{},\"target_offset\":{},\"separation\":\"{}\"}}",
            window.document, window.context_start, window.target_offset, window.separation,
        )
        .expect("writing window binding JSON cannot fail");
    }
    json.push(']');
}

fn push_lane_trace_json(json: &mut String, lane: &ProductionGradientLaneTrace) {
    write!(
        json,
        concat!(
            "{{\"lane\":\"{}\",\"gradient_interpretation\":\"{}\",",
            "\"late_quantization_scope\":\"{}\",",
            "\"stochastic_rounding_scope\":\"{}\",",
            "\"health\":{{\"output_gradient_sum_min\":{},\"output_gradient_sum_max\":{},",
            "\"output_gradient_l1_min\":{},\"output_gradient_l1_max\":{},",
            "\"output_gradient_max_abs\":{},\"output_gradient_trace_hash\":\"0x{:016x}\",",
            "\"ste_rescue_count\":{},",
            "\"backward_quantization_count\":{},\"stochastic_round_up_count\":{},",
            "\"gradient_saturation_count\":{},\"residual_saturation_count\":{}}},",
            "\"proposal_summary\":"
        ),
        lane.lane.as_str(),
        if lane.health.sample_reweighted_surrogate {
            "sample_reweighted_surrogate_direction"
        } else {
            "normalized_cross_entropy_surrogate_direction"
        },
        match lane.lane {
            ProductionGradientProposalLane::ReciprocalFreeLateRhu
            | ProductionGradientProposalLane::ReciprocalFreeLateStochastic => {
                "gated_chain_and_attention_v_k_microterm_sums_other_linear_sites_already_wide"
            }
            _ => "none",
        },
        if lane.lane == ProductionGradientProposalLane::ReciprocalFreeLateStochastic {
            "proposal_quantizer_sites_only_rms_core_remains_rhu"
        } else {
            "none"
        },
        lane.health.output_gradient_sum_min,
        lane.health.output_gradient_sum_max,
        lane.health.output_gradient_l1_min,
        lane.health.output_gradient_l1_max,
        lane.health.output_gradient_max_abs,
        lane.health.output_gradient_trace_hash,
        lane.health.backward_ste_rescue_count,
        lane.health.backward_quantization_count,
        lane.health.stochastic_round_up_count,
        lane.health.gradient_saturation_count,
        lane.health.residual_saturation_count,
    )
    .expect("writing lane trace JSON cannot fail");
    push_summary_json(json, lane.proposal_summary);
    json.push_str(",\"transfer_summary\":");
    push_summary_json(json, lane.transfer_summary);
    json.push_str(",\"rescue_exposed_trunk_proposal\":");
    push_summary_json(json, lane.rescue_exposed_trunk_proposal);
    json.push_str(",\"rescue_exposed_trunk_transfer\":");
    push_summary_json(json, lane.rescue_exposed_trunk_transfer);
    json.push_str(",\"natural_trunk_proposal\":");
    push_summary_json(json, lane.natural_trunk_proposal);
    json.push_str(",\"natural_trunk_transfer\":");
    push_summary_json(json, lane.natural_trunk_transfer);
    json.push_str(",\"output_head_proposal\":");
    push_summary_json(json, lane.output_head_proposal);
    json.push_str(",\"output_head_transfer\":");
    push_summary_json(json, lane.output_head_transfer);
    json.push('}');
}

fn push_causal_comparison_json(json: &mut String, comparison: ProductionGradientCausalComparison) {
    write!(
        json,
        concat!(
            "{{\"rescued_lane\":\"{}\",\"control_lane\":\"{}\",",
            "\"source_equivalence_verified\":{},",
            "\"rescue_exposed_trunk_coordinates\":{},",
            "\"aggregate_gradient_equal\":{},",
            "\"aggregate_gradient_magnitude_changed\":{},",
            "\"aggregate_gradient_sign_changed\":{},",
            "\"rescued_proposal_direction_agreements\":{},",
            "\"control_proposal_direction_agreements\":{},",
            "\"rescued_proposal_exact_descents\":{},",
            "\"control_proposal_exact_descents\":{},",
            "\"causal_separation_observed\":{},",
            "\"control_improves_proposal_fidelity\":{}}}"
        ),
        comparison.rescued_lane.as_str(),
        comparison.control_lane.as_str(),
        comparison.source_equivalence_verified,
        comparison.rescue_exposed_trunk_coordinates,
        comparison.aggregate_gradient_equal,
        comparison.aggregate_gradient_magnitude_changed,
        comparison.aggregate_gradient_sign_changed,
        comparison.rescued_proposal_direction_agreements,
        comparison.control_proposal_direction_agreements,
        comparison.rescued_proposal_exact_descents,
        comparison.control_proposal_exact_descents,
        comparison.causal_separation_observed,
        comparison.control_improves_proposal_fidelity,
    )
    .expect("writing causal comparison JSON cannot fail");
}

fn push_summary_json(json: &mut String, summary: ProductionGradientAlignmentSummary) {
    let agreement_per_mille =
        ratio_per_mille(summary.direction_agreements, summary.direction_comparable);
    let random_agreement_per_mille = ratio_per_mille(
        summary.random_direction_agreements,
        summary.direction_comparable,
    );
    let descent_per_mille = ratio_per_mille(
        summary.predicted_exact_descents,
        summary.proposed_coordinates,
    );
    let random_descent_per_mille =
        ratio_per_mille(summary.random_exact_descents, summary.proposed_coordinates);
    write!(
        json,
        concat!(
            "{{\"sampled_coordinates\":{},\"proposed_coordinates\":{},",
            "\"direction_comparable\":{},\"direction_agreements\":{},",
            "\"direction_agreement_per_mille\":{},\"random_direction_agreements\":{},",
            "\"random_direction_agreement_per_mille\":{},\"exact_descent_available\":{},",
            "\"predicted_exact_descents\":{},\"predicted_descent_per_mille\":{},",
            "\"random_exact_descents\":{},\"random_descent_per_mille\":{},",
            "\"function_visible\":{},\"objective_ties\":{}}}"
        ),
        summary.sampled_coordinates,
        summary.proposed_coordinates,
        summary.direction_comparable,
        summary.direction_agreements,
        agreement_per_mille,
        summary.random_direction_agreements,
        random_agreement_per_mille,
        summary.exact_descent_available,
        summary.predicted_exact_descents,
        descent_per_mille,
        summary.random_exact_descents,
        random_descent_per_mille,
        summary.function_visible,
        summary.objective_ties,
    )
    .expect("writing summary JSON cannot fail");
}

fn ratio_per_mille(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        0
    } else {
        numerator.saturating_mul(1_000) / denominator
    }
}

fn push_surface_json(json: &mut String, surface: ProductionGradientSurfaceDelta) {
    write!(
        json,
        concat!(
            "{{\"current_nll_q20\":{},\"plus_one_nll_q20\":{},",
            "\"minus_one_nll_q20\":{},\"better_neighbor_delta\":{},",
            "\"exact_descent_available\":{},\"plus_function_visible\":{},",
            "\"minus_function_visible\":{}}}"
        ),
        surface.current_nll_q20,
        surface.plus_one_nll_q20,
        surface.minus_one_nll_q20,
        surface.better_neighbor_delta,
        surface.exact_descent_available,
        surface.plus_function_visible,
        surface.minus_function_visible,
    )
    .expect("writing surface JSON cannot fail");
}

fn push_sample_json(
    json: &mut String,
    sample: &ProductionGradientAlignmentSample,
    enhanced_audit: bool,
) {
    write!(
        json,
        "{{\"group\":\"{}\",\"coordinate\":{}",
        sample.group, sample.coordinate,
    )
    .expect("writing sample JSON cannot fail");
    if enhanced_audit {
        json.push_str(",\"selection_strata\":[");
        let mut wrote_stratum = false;
        if sample.selected_for_union_activity {
            json.push_str("\"union_activity\"");
            wrote_stratum = true;
        }
        for (selected, name) in sample
            .selected_for_rescue_strata
            .iter()
            .zip(RESCUE_STRATUM_NAMES)
        {
            if !selected {
                continue;
            }
            if wrote_stratum {
                json.push(',');
            }
            write!(json, "\"{name}\"").expect("writing selection stratum cannot fail");
            wrote_stratum = true;
        }
        json.push(']');
    }
    write!(
        json,
        ",\"any_rescue_exposed\":{},\"random_control_delta\":{},\"proposal\":",
        sample.any_rescue_exposed, sample.random_control_delta,
    )
    .expect("writing sample metadata cannot fail");
    push_surface_json(json, sample.proposal);
    json.push_str(",\"transfer\":");
    push_surface_json(json, sample.transfer);
    json.push_str(",\"lanes\":[");
    for (index, lane) in sample.lanes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"lane\":\"{}\",\"coarse_gradient\":{},",
                "\"rescue_exposed\":{},\"predicted_parameter_delta\":{},",
                "\"proposal_direction_agrees\":{},",
                "\"proposal_predicted_exact_descent\":{},\"transfer_direction_agrees\":{},",
                "\"transfer_predicted_exact_descent\":{}}}"
            ),
            lane.lane.as_str(),
            lane.coarse_gradient,
            lane.rescue_exposed,
            lane.predicted_parameter_delta,
            lane.proposal_direction_agrees,
            lane.proposal_predicted_exact_descent,
            lane.transfer_direction_agrees,
            lane.transfer_predicted_exact_descent,
        )
        .expect("writing lane sample JSON cannot fail");
    }
    json.push_str("]}");
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use nsrl_core::SoftmaxNormalization;

    use super::*;
    use crate::production::ProductionModelConfig;

    fn tiny_config() -> ProductionModelConfig {
        ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 2,
            hidden_dim: 48,
            context_tokens: 16,
        }
    }

    fn two_document_tokens() -> Vec<u32> {
        vec![
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            305,
            306,
            307,
            308,
            309,
            EOS_TOKEN_ID,
        ]
    }

    fn four_document_tokens() -> Vec<u32> {
        vec![
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            305,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            306,
            307,
            308,
            309,
            310,
            311,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            312,
            313,
            314,
            315,
            316,
            317,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            300,
            302,
            304,
            306,
            308,
            310,
            EOS_TOKEN_ID,
        ]
    }

    #[test]
    fn surfaces_prefer_different_documents_and_record_coordinates() {
        let windows = document_windows_with_coordinates(&two_document_tokens(), 4);
        let (proposal, transfer) = select_surfaces(
            &windows,
            ProductionGradientAlignmentConfig {
                proposal_windows: 1,
                transfer_windows: 1,
                documents_per_surface: 0,
                rescue_stratified_sampling: false,
                include_mass_corrected_no_rescue: false,
                include_systematic_fixed_mass: false,
                coordinates_per_group: 1,
                sample_seed: 1,
            },
        )
        .expect("separated surfaces");
        assert_eq!(proposal[0].document, 0);
        assert_eq!(transfer[0].document, 1);
        assert_eq!(proposal[0].context_start, 0);
        assert_eq!(transfer[0].context_start, 0);
    }

    #[test]
    fn adjacent_sliding_windows_are_not_valid_transfer_windows() {
        let tokens = [
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            EOS_TOKEN_ID,
        ];
        let windows = document_windows_with_coordinates(&tokens, 4);
        assert!(!windows_do_not_overlap(&windows[0], &windows[1]));
        let (_, transfer) = select_surfaces(
            &windows,
            ProductionGradientAlignmentConfig {
                proposal_windows: 1,
                transfer_windows: 1,
                documents_per_surface: 0,
                rescue_stratified_sampling: false,
                include_mass_corrected_no_rescue: false,
                include_systematic_fixed_mass: false,
                coordinates_per_group: 1,
                sample_seed: 1,
            },
        )
        .expect("gapped transfer surface");
        assert!(transfer[0].context_start >= 5);
    }

    #[test]
    fn strict_document_blocks_balance_windows_and_do_not_share_documents() {
        let windows = document_windows_with_coordinates(&four_document_tokens(), 4);
        let (proposal, transfer) = select_surfaces(
            &windows,
            ProductionGradientAlignmentConfig {
                proposal_windows: 4,
                transfer_windows: 4,
                documents_per_surface: 2,
                rescue_stratified_sampling: false,
                include_mass_corrected_no_rescue: false,
                include_systematic_fixed_mass: false,
                coordinates_per_group: 1,
                sample_seed: 1,
            },
        )
        .expect("strict document blocks");
        assert_eq!(
            proposal
                .iter()
                .map(|window| window.document)
                .collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert_eq!(
            transfer
                .iter()
                .map(|window| window.document)
                .collect::<Vec<_>>(),
            [2, 3, 2, 3]
        );
        assert!(
            proposal
                .iter()
                .all(|left| { transfer.iter().all(|right| left.document != right.document) })
        );
    }

    #[test]
    fn lane_alignment_is_deterministic_zero_mass_and_restores_model() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        model.initialize_output_weights(2).expect("output weights");
        let original = model.clone();
        let training = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 1,
            epochs: 1,
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            ..ProductionFullTrainConfig::default()
        };
        let audit = ProductionGradientAlignmentConfig {
            proposal_windows: 1,
            transfer_windows: 1,
            documents_per_surface: 0,
            rescue_stratified_sampling: true,
            include_mass_corrected_no_rescue: true,
            include_systematic_fixed_mass: false,
            coordinates_per_group: 1,
            sample_seed: 19,
        };
        let tokens = two_document_tokens();
        let left = audit_production_gradient_alignment(&model, &tokens, 0x5678, training, audit)
            .expect("alignment audit");
        let right = audit_production_gradient_alignment(&model, &tokens, 0x5678, training, audit)
            .expect("alignment replay");
        let without_control = audit_production_gradient_alignment(
            &model,
            &tokens,
            0x5678,
            training,
            ProductionGradientAlignmentConfig {
                include_mass_corrected_no_rescue: false,
                ..audit
            },
        )
        .expect("alignment without causal control");
        let with_systematic = audit_production_gradient_alignment(
            &model,
            &tokens,
            0x5678,
            training,
            ProductionGradientAlignmentConfig {
                include_systematic_fixed_mass: true,
                ..audit
            },
        )
        .expect("alignment with systematic fixed-mass lanes");
        assert_eq!(model, original);
        assert_eq!(left, right);
        assert_eq!(left.proposal_bindings[0].document, 0);
        assert_eq!(left.transfer_bindings[0].document, 1);
        assert!(!left.samples.is_empty());
        assert!(left.samples.iter().any(|sample| {
            sample
                .selected_for_rescue_strata
                .iter()
                .any(|&selected| selected)
        }));
        for lane in &left.lanes {
            if lane.lane != ProductionGradientProposalLane::NormalizedRescued {
                assert_eq!(lane.health.output_gradient_sum_min, 0);
                assert_eq!(lane.health.output_gradient_sum_max, 0);
            }
        }
        let json = left.to_json_line();
        assert!(json.contains("backward_directional_fidelity"));
        assert!(json.contains("held_out_transfer_not_gradient_correctness"));
        assert!(json.contains("sample_reweighted_surrogate_direction"));
        assert!(json.contains("mass_corrected_rescue"));
        assert!(json.contains("mass-corrected-normalized-no-rescue-rhu"));
        assert!(left.causal_comparison.unwrap().source_equivalence_verified);
        assert_eq!(without_control.lanes.len(), 5);
        assert_eq!(left.lanes.len(), 6);
        assert_eq!(with_systematic.lanes.len(), 9);
        for lane in with_systematic.lanes.iter().filter(|lane| {
            matches!(
                lane.lane,
                ProductionGradientProposalLane::SystematicFixedMassK15
                    | ProductionGradientProposalLane::SystematicFixedMassK16
                    | ProductionGradientProposalLane::SystematicFixedMassK18
            )
        }) {
            assert_eq!(lane.health.output_gradient_sum_min, 0);
            assert_eq!(lane.health.output_gradient_sum_max, 0);
            assert_eq!(lane.health.gradient_saturation_count, 0);
        }
        assert!(
            with_systematic
                .to_json_line()
                .contains("systematic-fixed-mass-k16-token-id-rhu")
        );
        assert_eq!(
            left.samples
                .iter()
                .map(|sample| {
                    (
                        sample.group_index,
                        sample.coordinate,
                        sample.selected_for_union_activity,
                        sample.selected_for_rescue_strata,
                    )
                })
                .collect::<Vec<_>>(),
            without_control
                .samples
                .iter()
                .map(|sample| {
                    (
                        sample.group_index,
                        sample.coordinate,
                        sample.selected_for_union_activity,
                        sample.selected_for_rescue_strata,
                    )
                })
                .collect::<Vec<_>>()
        );
        let control = left
            .lanes
            .iter()
            .find(|lane| {
                lane.lane == ProductionGradientProposalLane::MassCorrectedNormalizedNoRescue
            })
            .expect("causal no-rescue lane");
        assert_eq!(control.health.backward_ste_rescue_count, 0);
    }
}
