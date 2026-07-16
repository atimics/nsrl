use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use nsrl_core::{base2_softmax_nll_q20, base2_softmax_nll_q47_q20, base2_softmax_nll_q47_q32};

use super::alignment::{
    SurfaceEval, can_perturb_both, document_windows_with_coordinates, evaluate_surface,
    select_surfaces, set_parameter_delta,
};
use super::{
    ProductionFullTrainConfig, ProductionGradientAlignmentConfig, ProductionGradientProposalLane,
    ProductionGradientWindowBinding, ProductionModelV1, TrainError,
    audit_production_gradient_alignment, forward_production_model,
    forward_production_model_branch_hashes,
};

const TRUNK_GROUP_INDEX: usize = 3;
const HEAD_GROUP_START: usize = 11;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_ZERO_PROBABILITY_FLOOR_Q20: u64 = 32_u64 << 20;
const DEFAULT_ZERO_PROBABILITY_FLOOR_Q32: u64 = 32_u64 << 32;
pub const PRODUCTION_BOOLEAN_JET_RESERVED_DOCUMENT_START: usize = 136;
const RESCUE_STRATUM_NAMES: [&str; 3] = [
    "normalized_rescue",
    "mass_corrected_rescue",
    "reciprocal_free_rescue",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetRankTwoConfig {
    pub alignment: ProductionGradientAlignmentConfig,
    pub expected_trunk_moves: usize,
    pub expected_head_moves: usize,
    pub expected_move_fingerprint: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionBooleanJetAnalysisRole {
    Calibration,
    Confirmation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionBooleanJetProtocolVersion {
    ConfirmationV1,
    StabilityV2,
}

impl ProductionBooleanJetProtocolVersion {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ConfirmationV1 => "confirmation_v1_immutable",
            Self::StabilityV2 => "stability_v2_matched_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetProtocolBindings {
    pub source_fnv64: u64,
    pub binary_fnv64: u64,
}

pub fn production_boolean_jet_source_fnv64() -> u64 {
    env!("NSRL_BOOLEAN_JET_SOURCE_FNV64")
        .parse()
        .expect("build script emits a decimal source FNV-1a binding")
}

pub fn production_boolean_jet_binary_fnv64(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(FNV_OFFSET, |hash, &byte| fnv_byte(hash, byte))
}

impl ProductionBooleanJetAnalysisRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Confirmation => "confirmation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionBooleanJetObjectiveAlgorithm {
    CanonicalQ15Lut,
    WideQ47LogitAnchored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductionBooleanJetAggregationRule {
    SumWindowsWithinDocumentThenSumDocuments,
}

impl ProductionBooleanJetAggregationRule {
    const fn as_str(self) -> &'static str {
        match self {
            Self::SumWindowsWithinDocumentThenSumDocuments => {
                "sum_windows_within_document_then_sum_documents_no_surface_averaging"
            }
        }
    }
}

impl ProductionBooleanJetObjectiveAlgorithm {
    const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalQ15Lut => "canonical_q15_lut_base2_nll",
            Self::WideQ47LogitAnchored => "mj05_q47_logit_anchored_base2_nll",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetObjectiveSpec {
    pub algorithm: ProductionBooleanJetObjectiveAlgorithm,
    pub fractional_bits: u8,
    pub zero_probability_floor_q20: u64,
    pub aggregation: ProductionBooleanJetAggregationRule,
    pub version: u8,
}

impl ProductionBooleanJetObjectiveSpec {
    pub const fn canonical_q15_v1() -> Self {
        Self {
            algorithm: ProductionBooleanJetObjectiveAlgorithm::CanonicalQ15Lut,
            fractional_bits: 20,
            zero_probability_floor_q20: DEFAULT_ZERO_PROBABILITY_FLOOR_Q20,
            aggregation:
                ProductionBooleanJetAggregationRule::SumWindowsWithinDocumentThenSumDocuments,
            version: 1,
        }
    }

    pub const fn wide_q47_v1() -> Self {
        Self {
            algorithm: ProductionBooleanJetObjectiveAlgorithm::WideQ47LogitAnchored,
            fractional_bits: 20,
            zero_probability_floor_q20: DEFAULT_ZERO_PROBABILITY_FLOOR_Q20,
            aggregation:
                ProductionBooleanJetAggregationRule::SumWindowsWithinDocumentThenSumDocuments,
            version: 1,
        }
    }

    pub const fn wide_q47_q32_v2() -> Self {
        Self {
            algorithm: ProductionBooleanJetObjectiveAlgorithm::WideQ47LogitAnchored,
            fractional_bits: 32,
            zero_probability_floor_q20: DEFAULT_ZERO_PROBABILITY_FLOOR_Q32,
            aggregation:
                ProductionBooleanJetAggregationRule::SumWindowsWithinDocumentThenSumDocuments,
            version: 2,
        }
    }

    fn validate(self) -> Result<(), TrainError> {
        let supported = matches!(
            (self.algorithm, self.fractional_bits, self.version),
            (
                ProductionBooleanJetObjectiveAlgorithm::CanonicalQ15Lut,
                20,
                1
            ) | (
                ProductionBooleanJetObjectiveAlgorithm::WideQ47LogitAnchored,
                20,
                1
            ) | (
                ProductionBooleanJetObjectiveAlgorithm::WideQ47LogitAnchored,
                32,
                2
            )
        );
        if !supported || self.zero_probability_floor_q20 == 0 {
            return Err(TrainError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetMove {
    pub block: &'static str,
    pub group: &'static str,
    pub group_index: usize,
    pub coordinate: usize,
    pub parameter_delta: i8,
    pub coarse_gradient: i64,
    pub selection_strata: Vec<&'static str>,
    pub source_lane: &'static str,
    pub move_kind: &'static str,
    pub canonical_order: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetMoveContract {
    pub protocol_version: ProductionBooleanJetProtocolVersion,
    pub analysis_role: ProductionBooleanJetAnalysisRole,
    pub expected_source_fnv64: u64,
    pub expected_binary_fnv64: u64,
    pub expected_base_model_hash: u64,
    pub expected_tokenizer_hash: u64,
    pub expected_token_stream_hash: u64,
    pub expected_move_fingerprint: u64,
    pub expected_manifest_hash: u64,
    pub trunk_moves: Vec<ProductionBooleanJetMove>,
    pub head_moves: Vec<ProductionBooleanJetMove>,
    pub matched_control_moves: Vec<ProductionBooleanJetMove>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetMatchedControlManifest {
    pub seed: u64,
    pub model_hash: u64,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub source_fnv64: u64,
    pub binary_fnv64: u64,
    pub visibility_window_hash: u64,
    pub structured_visibility_hash: u64,
    pub control_visibility_hash: u64,
    pub structured_saturation_margin_hash: u64,
    pub control_saturation_margin_hash: u64,
    pub move_fingerprint: u64,
    pub manifest_hash: u64,
    pub moves: Vec<ProductionBooleanJetMove>,
}

impl ProductionBooleanJetMatchedControlManifest {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_boolean_jet_matched_control_manifest.v2\",",
                "\"seed\":{},\"bindings\":{{\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\",",
                "\"source_fnv64\":\"0x{:016x}\",\"binary_fnv64\":\"0x{:016x}\"}},",
                "\"matching\":{{\"group\":true,\"block_cardinality\":true,",
                "\"stored_value_width\":true,\"function_visibility\":true,",
                "\"saturation_margin\":true,",
                "\"coordinate_selection\":\"seeded_hash_with_forward_visibility_no_loss_evaluation\",",
                "\"visibility_window_hash\":\"0x{:016x}\",",
                "\"structured_visibility_hash\":\"0x{:016x}\",",
                "\"control_visibility_hash\":\"0x{:016x}\",",
                "\"structured_saturation_margin_hash\":\"0x{:016x}\",",
                "\"control_saturation_margin_hash\":\"0x{:016x}\"}},",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\",\"moves\":"
            ),
            self.seed,
            self.model_hash,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.source_fnv64,
            self.binary_fnv64,
            self.visibility_window_hash,
            self.structured_visibility_hash,
            self.control_visibility_hash,
            self.structured_saturation_margin_hash,
            self.control_saturation_margin_hash,
            self.move_fingerprint,
            self.manifest_hash,
        )
        .expect("writing matched-control manifest cannot fail");
        push_moves(&mut json, &self.moves);
        json.push_str("}\n");
        json
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetDocumentTrace {
    pub document: usize,
    pub windows: usize,
    pub vertex_nll_q20: [u64; 4],
    pub mu_trunk_q20: i128,
    pub mu_head_q20: i128,
    pub mu_trunk_head_q20: i128,
    pub conditional_trunk_after_head_q20: i128,
    pub reconstruction_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetVertex {
    pub vertex: &'static str,
    pub nll_q20: u64,
    pub function_visible: bool,
    pub applied_moves: usize,
    pub parameter_saturation_count: usize,
    pub model_hash: u64,
    pub function_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetSurfaceTrace {
    pub surface: &'static str,
    pub vertices: [ProductionBooleanJetVertex; 4],
    pub mu_trunk_q20: i128,
    pub mu_head_q20: i128,
    pub mu_trunk_head_q20: i128,
    pub joint_delta_q20: i128,
    pub gamma_one_q20: i128,
    pub boolean_one_prime: bool,
    pub escape_order: Option<u8>,
    pub minimizing_mask: u8,
    pub minimizing_vertices: Vec<&'static str>,
    pub reconstruction_verified: bool,
    pub documents: Vec<ProductionBooleanJetDocumentTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetRankTwoTrace {
    pub analysis_role: ProductionBooleanJetAnalysisRole,
    pub objective: ProductionBooleanJetObjectiveSpec,
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub model_hash: u64,
    pub context_tokens: usize,
    pub sample_seed: u64,
    pub move_fingerprint: u64,
    pub manifest_hash: u64,
    pub proposal_bindings: Vec<ProductionGradientWindowBinding>,
    pub transfer_bindings: Vec<ProductionGradientWindowBinding>,
    pub trunk_moves: Vec<ProductionBooleanJetMove>,
    pub head_moves: Vec<ProductionBooleanJetMove>,
    pub proposal: ProductionBooleanJetSurfaceTrace,
    pub transfer: ProductionBooleanJetSurfaceTrace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetConfirmationConfig {
    pub context_tokens: usize,
    pub objective: ProductionBooleanJetObjectiveSpec,
    pub move_contract: ProductionBooleanJetMoveContract,
    pub proposal_document_start: usize,
    pub proposal_documents: usize,
    pub transfer_document_start: usize,
    pub transfer_documents: usize,
    pub windows_per_document: usize,
    pub minimum_independent_documents: usize,
    pub significance_numerator: u128,
    pub significance_denominator: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetSignTest {
    pub joint_wins: usize,
    pub head_wins: usize,
    pub ties: usize,
    pub non_ties: usize,
    pub exact_p_numerator: u128,
    pub exact_p_denominator: u128,
    pub p_per_million: u64,
    pub direction_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetConfirmationSurfaceTrace {
    pub cube: ProductionBooleanJetSurfaceTrace,
    pub document_start: usize,
    pub document_count: usize,
    pub windows_per_document: usize,
    pub conditional_sign_test: ProductionBooleanJetSignTest,
    pub matched_control: Option<ProductionBooleanJetMatchedControlSurfaceTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetMatchedControlDocumentTrace {
    pub document: usize,
    pub control_nll_q20: u64,
    pub head_control_nll_q20: u64,
    pub conditional_control_after_head_q20: i128,
    pub structured_minus_control_conditional_q20: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetMatchedControlSurfaceTrace {
    pub control_vertex: ProductionBooleanJetVertex,
    pub head_control_vertex: ProductionBooleanJetVertex,
    pub documents: Vec<ProductionBooleanJetMatchedControlDocumentTrace>,
    pub conditional_control_sign_test: ProductionBooleanJetSignTest,
    pub structured_beats_control_sign_test: ProductionBooleanJetSignTest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetConfirmationTrace {
    pub analysis_role: ProductionBooleanJetAnalysisRole,
    pub objective: ProductionBooleanJetObjectiveSpec,
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub model_hash: u64,
    pub move_fingerprint: u64,
    pub manifest_hash: u64,
    pub trunk_moves: Vec<ProductionBooleanJetMove>,
    pub head_moves: Vec<ProductionBooleanJetMove>,
    pub matched_control_moves: Vec<ProductionBooleanJetMove>,
    pub proposal: ProductionBooleanJetConfirmationSurfaceTrace,
    pub transfer: ProductionBooleanJetConfirmationSurfaceTrace,
    pub prospective_transfer_synergy_supported: bool,
    pub optimizer_change_authorized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetDecisionGates {
    pub transfer_direction: bool,
    pub transfer_significance: bool,
    pub minimum_non_ties: bool,
    pub objective_robustness: bool,
    pub matched_control: bool,
    pub source_binary_binding: bool,
    pub optimizer_transition_tested: bool,
    pub optimizer_change_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetObjectiveRobustnessTrace {
    pub objective: ProductionBooleanJetObjectiveSpec,
    pub proposal: ProductionBooleanJetConfirmationSurfaceTrace,
    pub transfer: ProductionBooleanJetConfirmationSurfaceTrace,
    pub aggregate_direction_agrees: bool,
    pub document_direction_agrees: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetConfirmationV2Trace {
    pub protocol_version: ProductionBooleanJetProtocolVersion,
    pub protocol_bindings: ProductionBooleanJetProtocolBindings,
    pub primary: ProductionBooleanJetConfirmationTrace,
    pub robustness: ProductionBooleanJetObjectiveRobustnessTrace,
    pub proposal_branch_localization: ProductionBooleanJetBranchSurfaceTrace,
    pub transfer_branch_localization: ProductionBooleanJetBranchSurfaceTrace,
    pub gates: ProductionBooleanJetDecisionGates,
    pub stability_supported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetBranchVertexTrace {
    pub vertex: &'static str,
    pub embedding_hash: u64,
    pub attention_residual_hashes: Vec<u64>,
    pub layer_output_hashes: Vec<u64>,
    pub final_features_hash: u64,
    pub logits_hash: u64,
    pub first_divergent_boundary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetBranchSurfaceTrace {
    pub surface: &'static str,
    pub vertices: [ProductionBooleanJetBranchVertexTrace; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionBooleanJetConfirmationV2Config {
    pub primary: ProductionBooleanJetConfirmationConfig,
    pub robustness_objective: ProductionBooleanJetObjectiveSpec,
    pub protocol_bindings: ProductionBooleanJetProtocolBindings,
    pub reserved_document_start: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionBooleanJetMatchedControlV2Config {
    pub context_tokens: usize,
    pub visibility_document_start: usize,
    pub visibility_documents: usize,
    pub windows_per_document: usize,
    pub reserved_document_start: usize,
    pub seed: u64,
    pub protocol_bindings: ProductionBooleanJetProtocolBindings,
}

impl ProductionBooleanJetRankTwoTrace {
    pub fn to_legacy_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_boolean_jet_rank_two.v1\",",
                "\"objective\":\"integer_base2_softmax_nll_q20\",",
                "\"claims\":{{\"evidence\":\"post_hoc_v3_calibration\",",
                "\"proposal_surface\":\"exact_move_cube_calibration\",",
                "\"transfer_surface\":\"document_disjoint_transfer_not_selection\"}},",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\",",
                "\"model_hash\":\"0x{:016x}\"}},",
                "\"move_contract\":{{",
                "\"state_scope\":\"deployed_parameters_only\",",
                "\"hidden_optimizer_state\":\"excluded\",",
                "\"action_order\":\"coordinate_ascending_within_block_trunk_then_head\",",
                "\"collision_semantics\":\"reject\",",
                "\"repeated_action_semantics\":\"reject\",",
                "\"boundary_saturation_semantics\":\"reject\"}},",
                "\"selection\":{{\"source\":\"exact_v3_alignment_replay\",",
                "\"primary_lane\":\"mass-corrected-normalized-rhu\",",
                "\"context_tokens\":{},\"sample_seed\":{},",
                "\"move_fingerprint\":\"0x{:016x}\"}},",
                "\"proposal_windows\":"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.model_hash,
            self.context_tokens,
            self.sample_seed,
            self.move_fingerprint,
        )
        .expect("writing legacy Boolean-jet trace header cannot fail");
        push_window_bindings(&mut json, &self.proposal_bindings);
        json.push_str(",\"transfer_windows\":");
        push_window_bindings(&mut json, &self.transfer_bindings);
        json.push_str(",\"moves\":{\"trunk\":");
        push_legacy_moves(&mut json, &self.trunk_moves);
        json.push_str(",\"head\":");
        push_legacy_moves(&mut json, &self.head_moves);
        json.push_str("},\"proposal\":");
        push_legacy_surface(&mut json, &self.proposal);
        json.push_str(",\"transfer\":");
        push_legacy_surface(&mut json, &self.transfer);
        json.push_str(",\"authorization\":{\"optimizer_change\":false,\"paid_scaling\":false}}\n");
        json
    }

    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_boolean_jet.v1\",",
                "\"rank\":2,\"analysis_role\":\"{}\",",
                "\"objective\":{{\"algorithm\":\"{}\",\"version\":{},",
                "\"fractional_bits\":{},\"zero_probability_floor_q20\":{},",
                "\"aggregation\":\"{}\"}},",
                "\"claims\":{{\"evidence\":\"post_hoc_v3_calibration\",",
                "\"proposal_surface\":\"exact_move_cube_calibration\",",
                "\"transfer_surface\":\"document_disjoint_transfer_not_selection\"}},",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\",",
                "\"model_hash\":\"0x{:016x}\"}},",
                "\"move_contract\":{{",
                "\"state_scope\":\"model_only_unit_sign_probe\",",
                "\"hidden_optimizer_state\":\"excluded\",",
                "\"action_order\":\"coordinate_ascending_within_block_trunk_then_head\",",
                "\"collision_semantics\":\"reject\",",
                "\"repeated_action_semantics\":\"reject\",",
                "\"boundary_saturation_semantics\":\"reject\"}},",
                "\"selection\":{{\"source\":\"exact_v3_alignment_replay\",",
                "\"primary_lane\":\"mass-corrected-normalized-rhu\",",
                "\"context_tokens\":{},\"sample_seed\":{},",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\"}},",
                "\"proposal_windows\":"
            ),
            self.analysis_role.as_str(),
            self.objective.algorithm.as_str(),
            self.objective.version,
            self.objective.fractional_bits,
            self.objective.zero_probability_floor_q20,
            self.objective.aggregation.as_str(),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.model_hash,
            self.context_tokens,
            self.sample_seed,
            self.move_fingerprint,
            self.manifest_hash,
        )
        .expect("writing Boolean-jet trace header cannot fail");
        push_window_bindings(&mut json, &self.proposal_bindings);
        json.push_str(",\"transfer_windows\":");
        push_window_bindings(&mut json, &self.transfer_bindings);
        json.push_str(",\"moves\":{\"trunk\":");
        push_moves(&mut json, &self.trunk_moves);
        json.push_str(",\"head\":");
        push_moves(&mut json, &self.head_moves);
        json.push_str("},\"proposal\":");
        push_surface(&mut json, &self.proposal);
        json.push_str(",\"transfer\":");
        push_surface(&mut json, &self.transfer);
        json.push_str(",\"authorization\":{\"optimizer_change\":false,\"paid_scaling\":false}}\n");
        json
    }
}

impl ProductionBooleanJetConfirmationTrace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_boolean_jet_confirmation.v1\",",
                "\"rank\":2,\"analysis_role\":\"{}\",",
                "\"primary_endpoint\":\"transfer_document_conditional_trunk_after_head\",",
                "\"proposal_role\":\"separate_diagnostic_not_pooled\",",
                "\"objective\":{{\"algorithm\":\"{}\",\"version\":{},",
                "\"fractional_bits\":{},\"zero_probability_floor_q20\":{},",
                "\"aggregation\":\"two_windows_summed_per_document_then_paired_sign_test\"}},",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\"}},",
                "\"move_contract\":{{\"state_scope\":\"model_only_unit_sign_probe\",",
                "\"hidden_optimizer_state\":\"excluded\",",
                "\"action_order\":\"{}\",",
                "\"collision_semantics\":\"reject_family\",",
                "\"repeated_action_semantics\":\"reject_family\",",
                "\"boundary_saturation_semantics\":\"reject_family\",",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\"}},",
                "\"moves\":{{\"trunk\":"
            ),
            self.analysis_role.as_str(),
            self.objective.algorithm.as_str(),
            self.objective.version,
            self.objective.fractional_bits,
            self.objective.zero_probability_floor_q20,
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.model_hash,
            if self.matched_control_moves.is_empty() {
                "canonical_order_ascending_trunk_then_head"
            } else {
                "canonical_order_ascending_trunk_then_head_then_matched_control"
            },
            self.move_fingerprint,
            self.manifest_hash,
        )
        .expect("writing Boolean-jet confirmation header cannot fail");
        push_moves(&mut json, &self.trunk_moves);
        json.push_str(",\"head\":");
        push_moves(&mut json, &self.head_moves);
        if !self.matched_control_moves.is_empty() {
            json.push_str(",\"matched_control\":");
            push_moves(&mut json, &self.matched_control_moves);
        }
        json.push_str("},\"proposal\":");
        push_confirmation_surface(&mut json, &self.proposal);
        json.push_str(",\"transfer\":");
        push_confirmation_surface(&mut json, &self.transfer);
        write!(
            json,
            concat!(
                ",\"decision\":{{\"prospective_transfer_synergy_supported\":{},",
                "\"optimizer_change_authorized\":{},",
                "\"paid_scaling_authorized\":false}}}}\n"
            ),
            self.prospective_transfer_synergy_supported, self.optimizer_change_authorized,
        )
        .expect("writing Boolean-jet confirmation decision cannot fail");
        json
    }
}

impl ProductionBooleanJetConfirmationV2Trace {
    pub fn to_json_line(&self) -> String {
        let mut json = String::new();
        write!(
            json,
            concat!(
                "{{\"schema\":\"nsrl.production_boolean_jet_stability_confirmation.v2\",",
                "\"protocol\":\"{}\",\"rank\":2,\"analysis_role\":\"confirmation\",",
                "\"bindings\":{{\"source_fnv64\":\"0x{:016x}\",",
                "\"binary_fnv64\":\"0x{:016x}\",\"model_hash\":\"0x{:016x}\",",
                "\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"objectives\":{{\"primary_high_resolution\":"
            ),
            self.protocol_version.as_str(),
            self.protocol_bindings.source_fnv64,
            self.protocol_bindings.binary_fnv64,
            self.primary.model_hash,
            self.primary.tokenizer_hash,
            self.primary.token_stream_hash,
        )
        .expect("writing v2 confirmation header cannot fail");
        push_objective(&mut json, self.primary.objective);
        json.push_str(",\"compatibility_robustness\":");
        push_objective(&mut json, self.robustness.objective);
        write!(
            json,
            concat!(
                "}},\"move_contract\":{{\"state_scope\":\"model_only_unit_sign_probe\",",
                "\"hidden_optimizer_state\":\"excluded_gate_reported_false\",",
                "\"move_fingerprint\":\"0x{:016x}\",",
                "\"manifest_hash\":\"0x{:016x}\",",
                "\"matched_control_frozen_pre_evaluation\":true}},",
                "\"moves\":{{\"trunk\":"
            ),
            self.primary.move_fingerprint, self.primary.manifest_hash,
        )
        .expect("writing v2 move contract cannot fail");
        push_moves(&mut json, &self.primary.trunk_moves);
        json.push_str(",\"head\":");
        push_moves(&mut json, &self.primary.head_moves);
        json.push_str(",\"matched_control\":");
        push_moves(&mut json, &self.primary.matched_control_moves);
        json.push_str("},\"primary_high_resolution\":{\"proposal\":");
        push_confirmation_surface_units(
            &mut json,
            &self.primary.proposal,
            self.primary.objective.fractional_bits,
        );
        json.push_str(",\"transfer\":");
        push_confirmation_surface_units(
            &mut json,
            &self.primary.transfer,
            self.primary.objective.fractional_bits,
        );
        json.push_str("},\"objective_robustness\":{\"proposal\":");
        push_confirmation_surface_units(
            &mut json,
            &self.robustness.proposal,
            self.robustness.objective.fractional_bits,
        );
        json.push_str(",\"transfer\":");
        push_confirmation_surface_units(
            &mut json,
            &self.robustness.transfer,
            self.robustness.objective.fractional_bits,
        );
        write!(
            json,
            concat!(
                ",\"aggregate_direction_agrees\":{},",
                "\"document_direction_agrees\":{}}},",
                "\"branch_localization\":{{\"proposal\":"
            ),
            self.robustness.aggregate_direction_agrees, self.robustness.document_direction_agrees,
        )
        .expect("writing v2 robustness gates cannot fail");
        push_branch_surface(&mut json, &self.proposal_branch_localization);
        json.push_str(",\"transfer\":");
        push_branch_surface(&mut json, &self.transfer_branch_localization);
        write!(
            json,
            concat!(
                "}},\"decision_gates\":{{\"transfer_direction\":{},",
                "\"transfer_significance\":{},\"minimum_non_ties\":{},",
                "\"objective_robustness\":{},\"matched_control\":{},",
                "\"source_binary_binding\":{},\"optimizer_transition_tested\":{},",
                "\"optimizer_change_authorized\":{}}},",
                "\"decision\":{{\"stability_supported\":{},",
                "\"optimizer_change_authorized\":false,\"paid_scaling_authorized\":false}}}}\n"
            ),
            self.gates.transfer_direction,
            self.gates.transfer_significance,
            self.gates.minimum_non_ties,
            self.gates.objective_robustness,
            self.gates.matched_control,
            self.gates.source_binary_binding,
            self.gates.optimizer_transition_tested,
            self.gates.optimizer_change_authorized,
            self.stability_supported,
        )
        .expect("writing v2 decision gates cannot fail");
        json
    }
}

pub fn audit_production_boolean_jet_rank_two(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    training: ProductionFullTrainConfig,
    config: ProductionBooleanJetRankTwoConfig,
) -> Result<ProductionBooleanJetRankTwoTrace, TrainError> {
    if config.expected_trunk_moves == 0
        || config.expected_head_moves == 0
        || !config.alignment.rescue_stratified_sampling
        || !config.alignment.include_mass_corrected_no_rescue
    {
        return Err(TrainError::InvalidConfig);
    }
    let alignment = audit_production_gradient_alignment(
        model,
        tokens,
        token_stream_hash,
        training,
        config.alignment,
    )?;
    let primary_lane = ProductionGradientProposalLane::MassCorrectedNormalized;
    let mut trunk_moves = Vec::new();
    let mut head_moves = Vec::new();
    for sample in &alignment.samples {
        let block = if sample.group_index == TRUNK_GROUP_INDEX {
            "trunk"
        } else if sample.group_index >= HEAD_GROUP_START {
            "head"
        } else {
            continue;
        };
        let lane = sample
            .lanes
            .iter()
            .find(|lane| lane.lane == primary_lane)
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_primary_lane_missing",
            ))?;
        if !matches!(lane.predicted_parameter_delta, -1 | 1) {
            return Err(TrainError::CoreRejected(
                "production_boolean_jet_zero_block_move",
            ));
        }
        let movement = ProductionBooleanJetMove {
            block,
            group: sample.group,
            group_index: sample.group_index,
            coordinate: sample.coordinate,
            parameter_delta: lane.predicted_parameter_delta,
            coarse_gradient: lane.coarse_gradient,
            selection_strata: selection_strata(sample),
            source_lane: primary_lane.as_str(),
            move_kind: "model_only_unit_sign_probe",
            canonical_order: 0,
        };
        if block == "trunk" {
            trunk_moves.push(movement);
        } else {
            head_moves.push(movement);
        }
    }
    trunk_moves.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    head_moves.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    for (order, movement) in trunk_moves
        .iter_mut()
        .chain(head_moves.iter_mut())
        .enumerate()
    {
        movement.canonical_order = order;
    }
    if trunk_moves.len() != config.expected_trunk_moves
        || head_moves.len() != config.expected_head_moves
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_move_count_mismatch",
        ));
    }
    validate_moves(model, &trunk_moves, &head_moves)?;
    let move_fingerprint = move_fingerprint(&trunk_moves, &head_moves);
    if config.expected_move_fingerprint != 0 && move_fingerprint != config.expected_move_fingerprint
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_move_fingerprint_mismatch",
        ));
    }
    let manifest_hash = manifest_hash(
        model.model_hash(),
        model.tokenizer_hash,
        token_stream_hash,
        &trunk_moves,
        &head_moves,
        &[],
    );

    let all_windows = document_windows_with_coordinates(tokens, training.context_tokens);
    let (proposal_windows, transfer_windows) = select_surfaces(&all_windows, config.alignment)?;
    let base_proposal = evaluate_surface(model, &proposal_windows)?;
    let base_transfer = evaluate_surface(model, &transfer_windows)?;

    let mut trunk_model = model.clone();
    apply_moves(&mut trunk_model, &trunk_moves)?;
    let trunk_proposal = evaluate_surface(&trunk_model, &proposal_windows)?;
    let trunk_transfer = evaluate_surface(&trunk_model, &transfer_windows)?;

    let mut head_model = model.clone();
    apply_moves(&mut head_model, &head_moves)?;
    let head_proposal = evaluate_surface(&head_model, &proposal_windows)?;
    let head_transfer = evaluate_surface(&head_model, &transfer_windows)?;

    let mut joint_model = model.clone();
    apply_moves(&mut joint_model, &trunk_moves)?;
    apply_moves(&mut joint_model, &head_moves)?;
    let joint_proposal = evaluate_surface(&joint_model, &proposal_windows)?;
    let joint_transfer = evaluate_surface(&joint_model, &transfer_windows)?;

    Ok(ProductionBooleanJetRankTwoTrace {
        analysis_role: ProductionBooleanJetAnalysisRole::Calibration,
        objective: ProductionBooleanJetObjectiveSpec::canonical_q15_v1(),
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        model_hash: model.model_hash(),
        context_tokens: training.context_tokens,
        sample_seed: config.alignment.sample_seed,
        move_fingerprint,
        manifest_hash,
        proposal_bindings: alignment.proposal_bindings,
        transfer_bindings: alignment.transfer_bindings,
        proposal: build_surface_trace(
            "proposal",
            &proposal_windows,
            &base_proposal,
            &trunk_proposal,
            &head_proposal,
            &joint_proposal,
            trunk_moves.len(),
            head_moves.len(),
            [
                model.model_hash(),
                trunk_model.model_hash(),
                head_model.model_hash(),
                joint_model.model_hash(),
            ],
        )?,
        transfer: build_surface_trace(
            "transfer",
            &transfer_windows,
            &base_transfer,
            &trunk_transfer,
            &head_transfer,
            &joint_transfer,
            trunk_moves.len(),
            head_moves.len(),
            [
                model.model_hash(),
                trunk_model.model_hash(),
                head_model.model_hash(),
                joint_model.model_hash(),
            ],
        )?,
        trunk_moves,
        head_moves,
    })
}

pub fn audit_production_boolean_jet_confirmation(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionBooleanJetConfirmationConfig,
) -> Result<ProductionBooleanJetConfirmationTrace, TrainError> {
    if config.move_contract.protocol_version != ProductionBooleanJetProtocolVersion::ConfirmationV1
        || config.move_contract.expected_source_fnv64 != 0
        || config.move_contract.expected_binary_fnv64 != 0
        || !config.move_contract.matched_control_moves.is_empty()
        || config.objective != ProductionBooleanJetObjectiveSpec::wide_q47_v1()
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_confirmation_v1_protocol_drift",
        ));
    }
    audit_production_boolean_jet_confirmation_impl(model, tokens, token_stream_hash, config)
}

fn audit_production_boolean_jet_confirmation_impl(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionBooleanJetConfirmationConfig,
) -> Result<ProductionBooleanJetConfirmationTrace, TrainError> {
    model.validate()?;
    config.objective.validate()?;
    let contract = &config.move_contract;
    if contract.analysis_role != ProductionBooleanJetAnalysisRole::Confirmation
        || contract.expected_base_model_hash == 0
        || contract.expected_tokenizer_hash == 0
        || contract.expected_token_stream_hash == 0
        || contract.expected_move_fingerprint == 0
        || contract.expected_manifest_hash == 0
        || contract.expected_base_model_hash != model.model_hash()
        || contract.expected_tokenizer_hash != model.tokenizer_hash
        || contract.expected_token_stream_hash != token_stream_hash
        || config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.windows_per_document == 0
        || config.minimum_independent_documents == 0
        || config.proposal_documents < config.minimum_independent_documents
        || config.transfer_documents < config.minimum_independent_documents
        || config.significance_numerator == 0
        || config.significance_denominator == 0
        || config.significance_numerator >= config.significance_denominator
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let proposal_end = config
        .proposal_document_start
        .checked_add(config.proposal_documents)
        .ok_or(TrainError::InvalidConfig)?;
    let transfer_end = config
        .transfer_document_start
        .checked_add(config.transfer_documents)
        .ok_or(TrainError::InvalidConfig)?;
    if config.proposal_document_start < transfer_end
        && config.transfer_document_start < proposal_end
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_confirmation_document_overlap",
        ));
    }
    if contract.matched_control_moves.is_empty() {
        validate_moves(model, &contract.trunk_moves, &contract.head_moves)?;
    } else {
        validate_confirmation_move_family(model, contract)?;
    }
    let move_fingerprint = move_fingerprint(&contract.trunk_moves, &contract.head_moves);
    let manifest_hash = match contract.protocol_version {
        ProductionBooleanJetProtocolVersion::ConfirmationV1 => manifest_hash(
            model.model_hash(),
            model.tokenizer_hash,
            token_stream_hash,
            &contract.trunk_moves,
            &contract.head_moves,
            &contract.matched_control_moves,
        ),
        ProductionBooleanJetProtocolVersion::StabilityV2 => manifest_hash_v2(
            contract.expected_source_fnv64,
            contract.expected_binary_fnv64,
            model.model_hash(),
            model.tokenizer_hash,
            token_stream_hash,
            &contract.trunk_moves,
            &contract.head_moves,
            &contract.matched_control_moves,
        ),
    };
    if move_fingerprint != contract.expected_move_fingerprint
        || manifest_hash != contract.expected_manifest_hash
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_confirmation_manifest_mismatch",
        ));
    }

    let all_windows = document_windows_with_coordinates(tokens, config.context_tokens);
    let proposal_windows = select_document_range(
        &all_windows,
        config.proposal_document_start,
        config.proposal_documents,
        config.windows_per_document,
    )?;
    let transfer_windows = select_document_range(
        &all_windows,
        config.transfer_document_start,
        config.transfer_documents,
        config.windows_per_document,
    )?;

    let mut trunk_model = model.clone();
    apply_moves(&mut trunk_model, &contract.trunk_moves)?;
    let mut head_model = model.clone();
    apply_moves(&mut head_model, &contract.head_moves)?;
    let mut joint_model = model.clone();
    apply_moves(&mut joint_model, &contract.trunk_moves)?;
    apply_moves(&mut joint_model, &contract.head_moves)?;
    let mut control_model = model.clone();
    apply_moves(&mut control_model, &contract.matched_control_moves)?;
    let mut head_control_model = model.clone();
    apply_moves(&mut head_control_model, &contract.head_moves)?;
    apply_moves(&mut head_control_model, &contract.matched_control_moves)?;

    let models = [model, &trunk_model, &head_model, &joint_model];
    let model_hashes = models.map(ProductionModelV1::model_hash);
    let proposal_evals = models
        .map(|candidate| {
            evaluate_surface_with_objective(candidate, &proposal_windows, config.objective)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let transfer_evals = models
        .map(|candidate| {
            evaluate_surface_with_objective(candidate, &transfer_windows, config.objective)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let proposal_control =
        evaluate_surface_with_objective(&control_model, &proposal_windows, config.objective)?;
    let proposal_head_control =
        evaluate_surface_with_objective(&head_control_model, &proposal_windows, config.objective)?;
    let transfer_control =
        evaluate_surface_with_objective(&control_model, &transfer_windows, config.objective)?;
    let transfer_head_control =
        evaluate_surface_with_objective(&head_control_model, &transfer_windows, config.objective)?;
    let proposal_cube = build_surface_trace(
        "proposal_confirmation",
        &proposal_windows,
        &proposal_evals[0],
        &proposal_evals[1],
        &proposal_evals[2],
        &proposal_evals[3],
        contract.trunk_moves.len(),
        contract.head_moves.len(),
        model_hashes,
    )?;
    let transfer_cube = build_surface_trace(
        "transfer_confirmation",
        &transfer_windows,
        &transfer_evals[0],
        &transfer_evals[1],
        &transfer_evals[2],
        &transfer_evals[3],
        contract.trunk_moves.len(),
        contract.head_moves.len(),
        model_hashes,
    )?;
    let proposal_sign = conditional_sign_test(
        &proposal_cube.documents,
        config.significance_numerator,
        config.significance_denominator,
    )?;
    let transfer_sign = conditional_sign_test(
        &transfer_cube.documents,
        config.significance_numerator,
        config.significance_denominator,
    )?;
    let proposal_matched_control = if contract.matched_control_moves.is_empty() {
        None
    } else {
        Some(build_matched_control_trace(
            &proposal_windows,
            &proposal_evals[0],
            &proposal_control,
            &proposal_head_control,
            &proposal_cube,
            contract.matched_control_moves.len(),
            control_model.model_hash(),
            head_control_model.model_hash(),
            config.significance_numerator,
            config.significance_denominator,
        )?)
    };
    let transfer_matched_control = if contract.matched_control_moves.is_empty() {
        None
    } else {
        Some(build_matched_control_trace(
            &transfer_windows,
            &transfer_evals[0],
            &transfer_control,
            &transfer_head_control,
            &transfer_cube,
            contract.matched_control_moves.len(),
            control_model.model_hash(),
            head_control_model.model_hash(),
            config.significance_numerator,
            config.significance_denominator,
        )?)
    };
    let prospective_transfer_synergy_supported = transfer_sign.direction_supported
        && transfer_sign.non_ties >= config.minimum_independent_documents
        && transfer_matched_control.as_ref().is_none_or(|control| {
            control
                .structured_beats_control_sign_test
                .direction_supported
        });

    Ok(ProductionBooleanJetConfirmationTrace {
        analysis_role: ProductionBooleanJetAnalysisRole::Confirmation,
        objective: config.objective,
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        model_hash: model.model_hash(),
        move_fingerprint,
        manifest_hash,
        trunk_moves: contract.trunk_moves.clone(),
        head_moves: contract.head_moves.clone(),
        matched_control_moves: contract.matched_control_moves.clone(),
        proposal: ProductionBooleanJetConfirmationSurfaceTrace {
            cube: proposal_cube,
            document_start: config.proposal_document_start,
            document_count: config.proposal_documents,
            windows_per_document: config.windows_per_document,
            conditional_sign_test: proposal_sign,
            matched_control: proposal_matched_control,
        },
        transfer: ProductionBooleanJetConfirmationSurfaceTrace {
            cube: transfer_cube,
            document_start: config.transfer_document_start,
            document_count: config.transfer_documents,
            windows_per_document: config.windows_per_document,
            conditional_sign_test: transfer_sign,
            matched_control: transfer_matched_control,
        },
        prospective_transfer_synergy_supported,
        optimizer_change_authorized: false,
    })
}

pub fn audit_production_boolean_jet_confirmation_v2(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionBooleanJetConfirmationV2Config,
) -> Result<ProductionBooleanJetConfirmationV2Trace, TrainError> {
    let contract = &config.primary.move_contract;
    let proposal_end = config
        .primary
        .proposal_document_start
        .checked_add(config.primary.proposal_documents)
        .ok_or(TrainError::InvalidConfig)?;
    if contract.protocol_version != ProductionBooleanJetProtocolVersion::StabilityV2
        || contract.expected_source_fnv64 == 0
        || contract.expected_binary_fnv64 == 0
        || contract.expected_source_fnv64 != config.protocol_bindings.source_fnv64
        || contract.expected_binary_fnv64 != config.protocol_bindings.binary_fnv64
        || config.protocol_bindings.source_fnv64 != production_boolean_jet_source_fnv64()
        || contract.matched_control_moves.is_empty()
        || config.primary.objective != ProductionBooleanJetObjectiveSpec::wide_q47_q32_v2()
        || config.robustness_objective != ProductionBooleanJetObjectiveSpec::canonical_q15_v1()
        || config.reserved_document_start == 0
        || proposal_end > config.reserved_document_start
        || config.primary.transfer_document_start < config.reserved_document_start
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_confirmation_v2_protocol_mismatch",
        ));
    }

    let primary = audit_production_boolean_jet_confirmation_impl(
        model,
        tokens,
        token_stream_hash,
        config.primary.clone(),
    )?;
    let all_windows = document_windows_with_coordinates(tokens, config.primary.context_tokens);
    let proposal_windows = select_document_range(
        &all_windows,
        config.primary.proposal_document_start,
        config.primary.proposal_documents,
        config.primary.windows_per_document,
    )?;
    let transfer_windows = select_document_range(
        &all_windows,
        config.primary.transfer_document_start,
        config.primary.transfer_documents,
        config.primary.windows_per_document,
    )?;
    let mut trunk_model = model.clone();
    apply_moves(&mut trunk_model, &contract.trunk_moves)?;
    let mut head_model = model.clone();
    apply_moves(&mut head_model, &contract.head_moves)?;
    let mut joint_model = model.clone();
    apply_moves(&mut joint_model, &contract.trunk_moves)?;
    apply_moves(&mut joint_model, &contract.head_moves)?;
    let mut control_model = model.clone();
    apply_moves(&mut control_model, &contract.matched_control_moves)?;
    let mut head_control_model = head_model.clone();
    apply_moves(&mut head_control_model, &contract.matched_control_moves)?;
    let models = [model, &trunk_model, &head_model, &joint_model];
    let model_hashes = models.map(ProductionModelV1::model_hash);
    let proposal = build_confirmation_surface_for_objective(
        "proposal_objective_robustness",
        &proposal_windows,
        models,
        &control_model,
        &head_control_model,
        model_hashes,
        config.robustness_objective,
        contract,
        &config.primary,
    )?;
    let transfer = build_confirmation_surface_for_objective(
        "transfer_objective_robustness",
        &transfer_windows,
        models,
        &control_model,
        &head_control_model,
        model_hashes,
        config.robustness_objective,
        contract,
        &config.primary,
    )?;
    let proposal_branch_localization =
        build_branch_surface_trace("proposal", &proposal_windows, models)?;
    let transfer_branch_localization =
        build_branch_surface_trace("transfer", &transfer_windows, models)?;
    let primary_conditional = difference(
        primary.transfer.cube.vertices[3].nll_q20,
        primary.transfer.cube.vertices[2].nll_q20,
    );
    let robustness_conditional = difference(
        transfer.cube.vertices[3].nll_q20,
        transfer.cube.vertices[2].nll_q20,
    );
    let aggregate_direction_agrees = primary_conditional < 0 && robustness_conditional < 0;
    let document_direction_agrees = primary.transfer.conditional_sign_test.joint_wins
        > primary.transfer.conditional_sign_test.head_wins
        && transfer.conditional_sign_test.joint_wins > transfer.conditional_sign_test.head_wins;
    let transfer_direction = primary.transfer.conditional_sign_test.joint_wins
        > primary.transfer.conditional_sign_test.head_wins;
    let transfer_significance = sign_test_is_significant(
        primary.transfer.conditional_sign_test,
        config.primary.significance_numerator,
        config.primary.significance_denominator,
    )?;
    let minimum_non_ties = primary.transfer.conditional_sign_test.non_ties
        >= config.primary.minimum_independent_documents;
    let matched_control = primary
        .transfer
        .matched_control
        .as_ref()
        .is_some_and(|control| {
            control
                .structured_beats_control_sign_test
                .direction_supported
        });
    let objective_robustness = aggregate_direction_agrees && document_direction_agrees;
    let gates = ProductionBooleanJetDecisionGates {
        transfer_direction,
        transfer_significance,
        minimum_non_ties,
        objective_robustness,
        matched_control,
        source_binary_binding: true,
        optimizer_transition_tested: false,
        optimizer_change_authorized: false,
    };
    let stability_supported = gates.transfer_direction
        && gates.transfer_significance
        && gates.minimum_non_ties
        && gates.objective_robustness
        && gates.matched_control
        && gates.source_binary_binding;

    Ok(ProductionBooleanJetConfirmationV2Trace {
        protocol_version: ProductionBooleanJetProtocolVersion::StabilityV2,
        protocol_bindings: config.protocol_bindings,
        primary,
        robustness: ProductionBooleanJetObjectiveRobustnessTrace {
            objective: config.robustness_objective,
            proposal,
            transfer,
            aggregate_direction_agrees,
            document_direction_agrees,
        },
        proposal_branch_localization,
        transfer_branch_localization,
        gates,
        stability_supported,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_confirmation_surface_for_objective(
    name: &'static str,
    windows: &[super::alignment::DocumentWindow],
    models: [&ProductionModelV1; 4],
    control_model: &ProductionModelV1,
    head_control_model: &ProductionModelV1,
    model_hashes: [u64; 4],
    objective: ProductionBooleanJetObjectiveSpec,
    contract: &ProductionBooleanJetMoveContract,
    config: &ProductionBooleanJetConfirmationConfig,
) -> Result<ProductionBooleanJetConfirmationSurfaceTrace, TrainError> {
    let evaluations = models
        .map(|candidate| evaluate_surface_with_objective(candidate, windows, objective))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let cube = build_surface_trace(
        name,
        windows,
        &evaluations[0],
        &evaluations[1],
        &evaluations[2],
        &evaluations[3],
        contract.trunk_moves.len(),
        contract.head_moves.len(),
        model_hashes,
    )?;
    let conditional_sign_test = conditional_sign_test(
        &cube.documents,
        config.significance_numerator,
        config.significance_denominator,
    )?;
    let control = evaluate_surface_with_objective(control_model, windows, objective)?;
    let head_control = evaluate_surface_with_objective(head_control_model, windows, objective)?;
    let matched_control = Some(build_matched_control_trace(
        windows,
        &evaluations[0],
        &control,
        &head_control,
        &cube,
        contract.matched_control_moves.len(),
        control_model.model_hash(),
        head_control_model.model_hash(),
        config.significance_numerator,
        config.significance_denominator,
    )?);
    let documents = cube.documents.len();
    let document_start = cube
        .documents
        .first()
        .map_or(0, |document| document.document);
    Ok(ProductionBooleanJetConfirmationSurfaceTrace {
        cube,
        document_start,
        document_count: documents,
        windows_per_document: config.windows_per_document,
        conditional_sign_test,
        matched_control,
    })
}

fn build_branch_surface_trace(
    surface: &'static str,
    windows: &[super::alignment::DocumentWindow],
    models: [&ProductionModelV1; 4],
) -> Result<ProductionBooleanJetBranchSurfaceTrace, TrainError> {
    let names = ["empty", "trunk", "head", "trunk_head"];
    let mut vertices = Vec::with_capacity(4);
    for (name, model) in names.into_iter().zip(models) {
        let mut embedding_hash = FNV_OFFSET;
        let mut attention_residual_hashes = vec![FNV_OFFSET; model.config.layers];
        let mut layer_output_hashes = vec![FNV_OFFSET; model.config.layers];
        let mut final_features_hash = FNV_OFFSET;
        let mut logits_hash = FNV_OFFSET;
        for window in windows {
            let trace = forward_production_model_branch_hashes(model, &window.context)?;
            fold_hash_value(&mut embedding_hash, trace.embedding_hash);
            fold_hash_value(&mut final_features_hash, trace.final_features_hash);
            fold_hash_value(&mut logits_hash, trace.logits_hash);
            if trace.layers.len() != model.config.layers {
                return Err(TrainError::CoreRejected(
                    "production_boolean_jet_branch_layer_shape",
                ));
            }
            for (layer, boundary) in trace.layers.iter().enumerate() {
                fold_hash_value(
                    &mut attention_residual_hashes[layer],
                    boundary.attention_residual_hash,
                );
                fold_hash_value(&mut layer_output_hashes[layer], boundary.layer_output_hash);
            }
        }
        vertices.push(ProductionBooleanJetBranchVertexTrace {
            vertex: name,
            embedding_hash,
            attention_residual_hashes,
            layer_output_hashes,
            final_features_hash,
            logits_hash,
            first_divergent_boundary: None,
        });
    }
    let base = vertices[0].clone();
    for vertex in &mut vertices[1..] {
        vertex.first_divergent_boundary = first_divergent_boundary(&base, vertex);
    }
    Ok(ProductionBooleanJetBranchSurfaceTrace {
        surface,
        vertices: vertices
            .try_into()
            .map_err(|_| TrainError::CoreRejected("production_boolean_jet_branch_vertex_shape"))?,
    })
}

fn fold_hash_value(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash = fnv_byte(*hash, byte);
    }
}

fn first_divergent_boundary(
    base: &ProductionBooleanJetBranchVertexTrace,
    candidate: &ProductionBooleanJetBranchVertexTrace,
) -> Option<String> {
    if base.embedding_hash != candidate.embedding_hash {
        return Some("embedding".to_string());
    }
    for layer in 0..base.attention_residual_hashes.len() {
        if base.attention_residual_hashes[layer] != candidate.attention_residual_hashes[layer] {
            return Some(format!("layer_{layer}_attention_residual"));
        }
        if base.layer_output_hashes[layer] != candidate.layer_output_hashes[layer] {
            return Some(format!("layer_{layer}_output"));
        }
    }
    if base.final_features_hash != candidate.final_features_hash {
        return Some("final_features".to_string());
    }
    if base.logits_hash != candidate.logits_hash {
        return Some("logits".to_string());
    }
    None
}

pub fn freeze_production_boolean_jet_matched_control(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
    config: ProductionBooleanJetMatchedControlV2Config,
) -> Result<ProductionBooleanJetMatchedControlManifest, TrainError> {
    model.validate()?;
    validate_moves(model, trunk_moves, head_moves)?;
    let visibility_end = config
        .visibility_document_start
        .checked_add(config.visibility_documents)
        .ok_or(TrainError::InvalidConfig)?;
    if config.seed == 0
        || trunk_moves.is_empty()
        || config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.visibility_documents == 0
        || config.windows_per_document == 0
        || config.reserved_document_start == 0
        || visibility_end > config.reserved_document_start
        || config.protocol_bindings.source_fnv64 != production_boolean_jet_source_fnv64()
        || config.protocol_bindings.binary_fnv64 == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let all_windows = document_windows_with_coordinates(tokens, config.context_tokens);
    let visibility_windows = select_document_range(
        &all_windows,
        config.visibility_document_start,
        config.visibility_documents,
        config.windows_per_document,
    )?;
    let base_logits = evaluate_logits(model, &visibility_windows)?;
    let mut structured_model = model.clone();
    apply_moves(&mut structured_model, trunk_moves)?;
    let structured_logits = evaluate_logits(&structured_model, &visibility_windows)?;
    let structured_visibility_hash = visibility_pattern_hash(&base_logits, &structured_logits)?;
    let structured_saturation_margin_hash = saturation_margin_hash(model, trunk_moves)?;
    let mut accepted = None;
    for nonce in 0..256_u64 {
        let moves = select_matched_control_moves(
            model,
            trunk_moves,
            head_moves,
            config.seed ^ splitmix64(nonce),
        )?;
        if saturation_margin_hash(model, &moves)? != structured_saturation_margin_hash {
            continue;
        }
        let mut control_model = model.clone();
        apply_moves(&mut control_model, &moves)?;
        let control_logits = evaluate_logits(&control_model, &visibility_windows)?;
        let control_visibility_hash = visibility_pattern_hash(&base_logits, &control_logits)?;
        if control_visibility_hash == structured_visibility_hash {
            accepted = Some((moves, control_visibility_hash));
            break;
        }
    }
    let (moves, control_visibility_hash) = accepted.ok_or(TrainError::CoreRejected(
        "production_boolean_jet_visibility_matched_control_missing",
    ))?;
    let move_fingerprint = move_fingerprint(trunk_moves, head_moves);
    let manifest_hash = manifest_hash_v2(
        config.protocol_bindings.source_fnv64,
        config.protocol_bindings.binary_fnv64,
        model.model_hash(),
        model.tokenizer_hash,
        token_stream_hash,
        trunk_moves,
        head_moves,
        &moves,
    );
    Ok(ProductionBooleanJetMatchedControlManifest {
        seed: config.seed,
        model_hash: model.model_hash(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        source_fnv64: config.protocol_bindings.source_fnv64,
        binary_fnv64: config.protocol_bindings.binary_fnv64,
        visibility_window_hash: window_binding_hash(&visibility_windows),
        structured_visibility_hash,
        control_visibility_hash,
        structured_saturation_margin_hash,
        control_saturation_margin_hash: saturation_margin_hash(model, &moves)?,
        move_fingerprint,
        manifest_hash,
        moves,
    })
}

fn select_matched_control_moves(
    model: &ProductionModelV1,
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
    seed: u64,
) -> Result<Vec<ProductionBooleanJetMove>, TrainError> {
    let mut used = trunk_moves
        .iter()
        .chain(head_moves)
        .map(|movement| (movement.group_index, movement.coordinate))
        .collect::<BTreeSet<_>>();
    let mut moves = Vec::with_capacity(trunk_moves.len());
    for structured in trunk_moves {
        let target_margin = saturation_margin(model, structured)?;
        let mut candidates = Vec::new();
        for coordinate in 0..parameter_group_len(model, structured.group_index)? {
            if used.contains(&(structured.group_index, coordinate))
                || !can_perturb_both(model, structured.group_index, coordinate)
            {
                continue;
            }
            for delta in [-1_i8, 1_i8] {
                let candidate = ProductionBooleanJetMove {
                    block: "matched_control",
                    group: group_name(structured.group_index)?,
                    group_index: structured.group_index,
                    coordinate,
                    parameter_delta: delta,
                    coarse_gradient: 0,
                    selection_strata: vec![
                        "frozen_group_cardinality_width_visibility_margin_control",
                    ],
                    source_lane: "seeded-matched-random-control-v2",
                    move_kind: "model_only_unit_sign_probe",
                    canonical_order: 0,
                };
                if saturation_margin(model, &candidate)? == target_margin {
                    candidates.push((
                        splitmix64(
                            seed ^ (structured.group_index as u64).rotate_left(17)
                                ^ (coordinate as u64).rotate_left(31)
                                ^ u64::from(delta as u8),
                        ),
                        candidate,
                    ));
                }
            }
        }
        candidates.sort_unstable_by_key(|candidate| candidate.0);
        let candidate = candidates
            .into_iter()
            .map(|(_, movement)| movement)
            .find(|movement| !used.contains(&(movement.group_index, movement.coordinate)))
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_saturation_matched_control_missing",
            ))?;
        used.insert((candidate.group_index, candidate.coordinate));
        moves.push(candidate);
    }
    moves.sort_unstable_by_key(|movement| (movement.group_index, movement.coordinate));
    let order_offset = trunk_moves.len() + head_moves.len();
    for (index, movement) in moves.iter_mut().enumerate() {
        movement.canonical_order = order_offset + index;
    }
    Ok(moves)
}

fn parameter_group_len(model: &ProductionModelV1, group_index: usize) -> Result<usize, TrainError> {
    [
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
    ]
    .get(group_index)
    .copied()
    .ok_or(TrainError::InvalidConfig)
}

fn parameter_value_and_bounds(
    model: &ProductionModelV1,
    group_index: usize,
    coordinate: usize,
) -> Result<(i64, i64, i64), TrainError> {
    macro_rules! value {
        ($values:expr, $type:ty) => {{
            let value = i64::from(*$values.get(coordinate).ok_or(TrainError::InvalidConfig)?);
            (value, i64::from(<$type>::MIN), i64::from(<$type>::MAX))
        }};
    }
    Ok(match group_index {
        0 => value!(model.embeddings, i16),
        1 => value!(model.attention_rms_weights, i16),
        2 => value!(model.mlp_rms_weights, i16),
        3 => value!(model.final_rms_weights, i16),
        4 => value!(model.q_weights, i8),
        5 => value!(model.k_weights, i8),
        6 => value!(model.v_weights, i8),
        7 => value!(model.o_weights, i8),
        8 => value!(model.up_weights, i8),
        9 => value!(model.gate_weights, i8),
        10 => value!(model.down_weights, i8),
        11 => value!(model.output_weights, i16),
        12 => value!(model.output_bias_q8, i32),
        _ => return Err(TrainError::InvalidConfig),
    })
}

fn saturation_margin(
    model: &ProductionModelV1,
    movement: &ProductionBooleanJetMove,
) -> Result<u64, TrainError> {
    let (value, minimum, maximum) =
        parameter_value_and_bounds(model, movement.group_index, movement.coordinate)?;
    u64::try_from(if movement.parameter_delta > 0 {
        maximum - value
    } else {
        value - minimum
    })
    .map_err(|_| TrainError::InvalidConfig)
}

fn saturation_margin_hash(
    model: &ProductionModelV1,
    moves: &[ProductionBooleanJetMove],
) -> Result<u64, TrainError> {
    let mut margins = moves
        .iter()
        .map(|movement| saturation_margin(model, movement))
        .collect::<Result<Vec<_>, _>>()?;
    margins.sort_unstable();
    let mut hash = FNV_OFFSET;
    for margin in margins {
        for byte in margin.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    Ok(hash)
}

fn evaluate_logits(
    model: &ProductionModelV1,
    windows: &[super::alignment::DocumentWindow],
) -> Result<Vec<Vec<i32>>, TrainError> {
    windows
        .iter()
        .map(|window| {
            forward_production_model(model, &window.context).map(|forward| forward.logits_q8)
        })
        .collect()
}

fn visibility_pattern_hash(base: &[Vec<i32>], candidate: &[Vec<i32>]) -> Result<u64, TrainError> {
    if base.len() != candidate.len() {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_visibility_shape",
        ));
    }
    Ok(base
        .iter()
        .zip(candidate)
        .fold(FNV_OFFSET, |hash, (base, candidate)| {
            fnv_byte(hash, u8::from(base != candidate))
        }))
}

fn window_binding_hash(windows: &[super::alignment::DocumentWindow]) -> u64 {
    let mut hash = FNV_OFFSET;
    for window in windows {
        for value in [
            window.document as u64,
            window.context_start as u64,
            window.context_start.saturating_add(window.context.len()) as u64,
            u64::from(window.target),
        ] {
            for byte in value.to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
        }
    }
    hash
}

fn group_name(group_index: usize) -> Result<&'static str, TrainError> {
    [
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
    ]
    .get(group_index)
    .copied()
    .ok_or(TrainError::InvalidConfig)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn select_document_range(
    windows: &[super::alignment::DocumentWindow],
    document_start: usize,
    documents: usize,
    windows_per_document: usize,
) -> Result<Vec<super::alignment::DocumentWindow>, TrainError> {
    let document_end = document_start
        .checked_add(documents)
        .ok_or(TrainError::InvalidConfig)?;
    let mut selected = Vec::with_capacity(
        documents
            .checked_mul(windows_per_document)
            .ok_or(TrainError::InvalidConfig)?,
    );
    for document in document_start..document_end {
        let before = selected.len();
        selected.extend(
            windows
                .iter()
                .filter(|window| window.document == document)
                .take(windows_per_document)
                .cloned(),
        );
        if selected.len() - before != windows_per_document {
            return Err(TrainError::CoreRejected(
                "production_boolean_jet_confirmation_document_windows_missing",
            ));
        }
    }
    Ok(selected)
}

fn evaluate_surface_with_objective(
    model: &ProductionModelV1,
    windows: &[super::alignment::DocumentWindow],
    objective: ProductionBooleanJetObjectiveSpec,
) -> Result<SurfaceEval, TrainError> {
    objective.validate()?;
    let mut nll_q20 = 0_u64;
    let mut losses_q20 = Vec::with_capacity(windows.len());
    let mut logits = Vec::with_capacity(windows.len());
    for window in windows {
        let forward = forward_production_model(model, &window.context)?;
        let loss = match objective.algorithm {
            ProductionBooleanJetObjectiveAlgorithm::CanonicalQ15Lut => base2_softmax_nll_q20(
                &forward.logits_q8,
                window.target as usize,
                objective.zero_probability_floor_q20,
            ),
            ProductionBooleanJetObjectiveAlgorithm::WideQ47LogitAnchored => {
                match objective.fractional_bits {
                    20 => base2_softmax_nll_q47_q20(
                        &forward.logits_q8,
                        window.target as usize,
                        objective.zero_probability_floor_q20,
                    ),
                    32 => base2_softmax_nll_q47_q32(
                        &forward.logits_q8,
                        window.target as usize,
                        objective.zero_probability_floor_q20,
                    ),
                    _ => None,
                }
            }
        }
        .ok_or(TrainError::CoreRejected(
            "production_boolean_jet_objective_rejected",
        ))?;
        nll_q20 = nll_q20.checked_add(loss).ok_or(TrainError::CoreRejected(
            "production_boolean_jet_nll_accumulator_overflow",
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

#[allow(clippy::too_many_arguments)]
fn build_matched_control_trace(
    windows: &[super::alignment::DocumentWindow],
    base: &SurfaceEval,
    control: &SurfaceEval,
    head_control: &SurfaceEval,
    structured_cube: &ProductionBooleanJetSurfaceTrace,
    control_move_count: usize,
    control_model_hash: u64,
    head_control_model_hash: u64,
    significance_numerator: u128,
    significance_denominator: u128,
) -> Result<ProductionBooleanJetMatchedControlSurfaceTrace, TrainError> {
    if [
        base.losses_q20.len(),
        control.losses_q20.len(),
        head_control.losses_q20.len(),
    ]
    .into_iter()
    .any(|length| length != windows.len())
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_matched_control_loss_shape",
        ));
    }
    let mut by_document = BTreeMap::<usize, [u64; 2]>::new();
    for (index, window) in windows.iter().enumerate() {
        let losses = by_document.entry(window.document).or_insert([0; 2]);
        losses[0] =
            losses[0]
                .checked_add(control.losses_q20[index])
                .ok_or(TrainError::CoreRejected(
                    "production_boolean_jet_matched_control_nll_overflow",
                ))?;
        losses[1] = losses[1]
            .checked_add(head_control.losses_q20[index])
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_matched_control_nll_overflow",
            ))?;
    }
    let mut documents = Vec::with_capacity(structured_cube.documents.len());
    for structured in &structured_cube.documents {
        let [control_nll_q20, head_control_nll_q20] = by_document
            .get(&structured.document)
            .copied()
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_matched_control_document_missing",
            ))?;
        let head_nll_q20 = structured.vertex_nll_q20[2];
        let conditional_control_after_head_q20 = difference(head_control_nll_q20, head_nll_q20);
        documents.push(ProductionBooleanJetMatchedControlDocumentTrace {
            document: structured.document,
            control_nll_q20,
            head_control_nll_q20,
            conditional_control_after_head_q20,
            structured_minus_control_conditional_q20: structured.conditional_trunk_after_head_q20
                - conditional_control_after_head_q20,
        });
    }
    let control_deltas = documents
        .iter()
        .map(|document| document.conditional_control_after_head_q20)
        .collect::<Vec<_>>();
    let comparison_deltas = documents
        .iter()
        .map(|document| document.structured_minus_control_conditional_q20)
        .collect::<Vec<_>>();
    Ok(ProductionBooleanJetMatchedControlSurfaceTrace {
        control_vertex: vertex(
            "matched_control",
            base,
            control,
            control_move_count,
            control_model_hash,
        ),
        head_control_vertex: vertex(
            "head_matched_control",
            base,
            head_control,
            control_move_count,
            head_control_model_hash,
        ),
        documents,
        conditional_control_sign_test: sign_test_deltas(
            &control_deltas,
            significance_numerator,
            significance_denominator,
        )?,
        structured_beats_control_sign_test: sign_test_deltas(
            &comparison_deltas,
            significance_numerator,
            significance_denominator,
        )?,
    })
}

fn conditional_sign_test(
    documents: &[ProductionBooleanJetDocumentTrace],
    significance_numerator: u128,
    significance_denominator: u128,
) -> Result<ProductionBooleanJetSignTest, TrainError> {
    let deltas = documents
        .iter()
        .map(|document| document.conditional_trunk_after_head_q20)
        .collect::<Vec<_>>();
    sign_test_deltas(&deltas, significance_numerator, significance_denominator)
}

fn sign_test_deltas(
    deltas: &[i128],
    significance_numerator: u128,
    significance_denominator: u128,
) -> Result<ProductionBooleanJetSignTest, TrainError> {
    let joint_wins = deltas.iter().filter(|&&delta| delta < 0).count();
    let head_wins = deltas.iter().filter(|&&delta| delta > 0).count();
    let ties = deltas.len().saturating_sub(joint_wins + head_wins);
    let non_ties = joint_wins + head_wins;
    if non_ties >= 128 {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_sign_test_too_many_documents",
        ));
    }
    let (exact_p_numerator, exact_p_denominator) = exact_two_sided_sign_p(joint_wins, head_wins);
    let p_per_million = exact_p_numerator
        .checked_mul(1_000_000)
        .ok_or(TrainError::CoreRejected(
            "production_boolean_jet_sign_test_scale_overflow",
        ))?
        .checked_add(exact_p_denominator / 2)
        .ok_or(TrainError::CoreRejected(
            "production_boolean_jet_sign_test_scale_overflow",
        ))?
        / exact_p_denominator;
    let direction_supported = joint_wins > head_wins
        && exact_p_numerator
            .checked_mul(significance_denominator)
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_sign_test_threshold_overflow",
            ))?
            <= significance_numerator
                .checked_mul(exact_p_denominator)
                .ok_or(TrainError::CoreRejected(
                    "production_boolean_jet_sign_test_threshold_overflow",
                ))?;
    Ok(ProductionBooleanJetSignTest {
        joint_wins,
        head_wins,
        ties,
        non_ties,
        exact_p_numerator,
        exact_p_denominator,
        p_per_million: u64::try_from(p_per_million).map_err(|_| {
            TrainError::CoreRejected("production_boolean_jet_sign_test_scale_exceeds_u64")
        })?,
        direction_supported,
    })
}

fn sign_test_is_significant(
    test: ProductionBooleanJetSignTest,
    significance_numerator: u128,
    significance_denominator: u128,
) -> Result<bool, TrainError> {
    Ok(test
        .exact_p_numerator
        .checked_mul(significance_denominator)
        .ok_or(TrainError::CoreRejected(
            "production_boolean_jet_sign_test_threshold_overflow",
        ))?
        <= significance_numerator
            .checked_mul(test.exact_p_denominator)
            .ok_or(TrainError::CoreRejected(
                "production_boolean_jet_sign_test_threshold_overflow",
            ))?)
}

fn exact_two_sided_sign_p(left_wins: usize, right_wins: usize) -> (u128, u128) {
    let n = left_wins + right_wins;
    if n == 0 {
        return (1, 1);
    }
    let tail = left_wins.min(right_wins);
    let mut coefficient = 1_u128;
    let mut tail_sum = 1_u128;
    for k in 1..=tail {
        coefficient = coefficient * (n - k + 1) as u128 / k as u128;
        tail_sum += coefficient;
    }
    let denominator = 1_u128 << n;
    let numerator = tail_sum.saturating_mul(2).min(denominator);
    let divisor = gcd_u128(numerator, denominator);
    (numerator / divisor, denominator / divisor)
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn selection_strata(sample: &super::ProductionGradientAlignmentSample) -> Vec<&'static str> {
    let mut strata = Vec::new();
    if sample.selected_for_union_activity {
        strata.push("union_activity");
    }
    for (&selected, name) in sample
        .selected_for_rescue_strata
        .iter()
        .zip(RESCUE_STRATUM_NAMES)
    {
        if selected {
            strata.push(name);
        }
    }
    strata
}

fn validate_moves(
    model: &ProductionModelV1,
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
) -> Result<(), TrainError> {
    if trunk_moves
        .iter()
        .any(|movement| movement.block != "trunk" || movement.group_index != TRUNK_GROUP_INDEX)
        || head_moves
            .iter()
            .any(|movement| movement.block != "head" || movement.group_index < HEAD_GROUP_START)
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_block_membership_mismatch",
        ));
    }
    let mut coordinates = BTreeSet::new();
    for (order, movement) in trunk_moves.iter().chain(head_moves).enumerate() {
        if !matches!(movement.parameter_delta, -1 | 1)
            || movement.canonical_order != order
            || movement.move_kind != "model_only_unit_sign_probe"
            || movement.source_lane
                != ProductionGradientProposalLane::MassCorrectedNormalized.as_str()
            || !coordinates.insert((movement.group_index, movement.coordinate))
            || !can_perturb_both(model, movement.group_index, movement.coordinate)
        {
            return Err(TrainError::CoreRejected(
                "production_boolean_jet_invalid_or_colliding_move",
            ));
        }
    }
    Ok(())
}

fn validate_confirmation_move_family(
    model: &ProductionModelV1,
    contract: &ProductionBooleanJetMoveContract,
) -> Result<(), TrainError> {
    validate_moves(model, &contract.trunk_moves, &contract.head_moves)?;
    if contract.matched_control_moves.len() != contract.trunk_moves.len()
        || contract.matched_control_moves.is_empty()
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_unmatched_control_cardinality",
        ));
    }
    let mut structured_groups = contract
        .trunk_moves
        .iter()
        .map(|movement| movement.group_index)
        .collect::<Vec<_>>();
    let mut control_groups = contract
        .matched_control_moves
        .iter()
        .map(|movement| movement.group_index)
        .collect::<Vec<_>>();
    structured_groups.sort_unstable();
    control_groups.sort_unstable();
    if structured_groups != control_groups {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_unmatched_control_groups",
        ));
    }
    let mut coordinates = contract
        .trunk_moves
        .iter()
        .chain(&contract.head_moves)
        .map(|movement| (movement.group_index, movement.coordinate))
        .collect::<BTreeSet<_>>();
    let order_offset = contract.trunk_moves.len() + contract.head_moves.len();
    for (index, movement) in contract.matched_control_moves.iter().enumerate() {
        if movement.block != "matched_control"
            || movement.source_lane != "seeded-matched-random-control-v2"
            || movement.move_kind != "model_only_unit_sign_probe"
            || movement.canonical_order != order_offset + index
            || !matches!(movement.parameter_delta, -1 | 1)
            || !coordinates.insert((movement.group_index, movement.coordinate))
            || !can_perturb_both(model, movement.group_index, movement.coordinate)
        {
            return Err(TrainError::CoreRejected(
                "production_boolean_jet_invalid_matched_control",
            ));
        }
    }
    Ok(())
}

fn apply_moves(
    model: &mut ProductionModelV1,
    moves: &[ProductionBooleanJetMove],
) -> Result<(), TrainError> {
    for movement in moves {
        set_parameter_delta(
            model,
            movement.group_index,
            movement.coordinate,
            movement.parameter_delta,
        )?;
    }
    Ok(())
}

fn move_fingerprint(
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
) -> u64 {
    let mut hash = FNV_OFFSET;
    for (block_tag, moves) in [(b'T', trunk_moves), (b'H', head_moves)] {
        hash = fnv_byte(hash, block_tag);
        for movement in moves {
            for byte in (movement.group_index as u64).to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
            for byte in (movement.coordinate as u64).to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
            hash = fnv_byte(hash, movement.parameter_delta as u8);
        }
    }
    hash
}

fn manifest_hash(
    model_hash: u64,
    tokenizer_hash: u64,
    token_stream_hash: u64,
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
    matched_control_moves: &[ProductionBooleanJetMove],
) -> u64 {
    let mut hash = FNV_OFFSET;
    for value in [model_hash, tokenizer_hash, token_stream_hash] {
        for byte in value.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    let mut blocks = vec![(b'T', trunk_moves), (b'H', head_moves)];
    if !matched_control_moves.is_empty() {
        blocks.push((b'R', matched_control_moves));
    }
    for (block_tag, moves) in blocks {
        hash = fnv_byte(hash, block_tag);
        for movement in moves {
            for value in [
                movement.group_index as u64,
                movement.coordinate as u64,
                movement.canonical_order as u64,
            ] {
                for byte in value.to_le_bytes() {
                    hash = fnv_byte(hash, byte);
                }
            }
            hash = fnv_byte(hash, movement.parameter_delta as u8);
            for byte in movement
                .source_lane
                .bytes()
                .chain(movement.move_kind.bytes())
            {
                hash = fnv_byte(hash, byte);
            }
        }
    }
    hash
}

fn manifest_hash_v2(
    source_fnv64: u64,
    binary_fnv64: u64,
    model_hash: u64,
    tokenizer_hash: u64,
    token_stream_hash: u64,
    trunk_moves: &[ProductionBooleanJetMove],
    head_moves: &[ProductionBooleanJetMove],
    matched_control_moves: &[ProductionBooleanJetMove],
) -> u64 {
    let v1 = manifest_hash(
        model_hash,
        tokenizer_hash,
        token_stream_hash,
        trunk_moves,
        head_moves,
        matched_control_moves,
    );
    let mut hash = FNV_OFFSET;
    for byte in b"nsrl.production_boolean_jet_stability.v2" {
        hash = fnv_byte(hash, *byte);
    }
    for value in [source_fnv64, binary_fnv64, v1] {
        for byte in value.to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
    }
    hash
}

const fn fnv_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

fn mobius_coefficients(losses: &[u64]) -> Result<Vec<i128>, TrainError> {
    if losses.is_empty() || !losses.len().is_power_of_two() {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_mobius_shape",
        ));
    }
    let mut coefficients = losses.iter().copied().map(i128::from).collect::<Vec<_>>();
    let rank = losses.len().trailing_zeros() as usize;
    for bit in 0..rank {
        let bit_mask = 1_usize << bit;
        for mask in 0..losses.len() {
            if mask & bit_mask != 0 {
                coefficients[mask] = coefficients[mask]
                    .checked_sub(coefficients[mask ^ bit_mask])
                    .ok_or(TrainError::CoreRejected(
                        "production_boolean_jet_mobius_overflow",
                    ))?;
            }
        }
    }
    Ok(coefficients)
}

fn mobius_reconstructs(losses: &[u64], coefficients: &[i128]) -> bool {
    if losses.len() != coefficients.len() || losses.is_empty() || !losses.len().is_power_of_two() {
        return false;
    }
    (0..losses.len()).all(|mask| {
        let mut sum = 0_i128;
        let mut subset = mask;
        loop {
            let Some(next) = sum.checked_add(coefficients[subset]) else {
                return false;
            };
            sum = next;
            if subset == 0 {
                break;
            }
            subset = (subset - 1) & mask;
        }
        sum == i128::from(losses[mask])
    })
}

#[allow(clippy::too_many_arguments)]
fn build_surface_trace(
    surface: &'static str,
    windows: &[super::alignment::DocumentWindow],
    base: &SurfaceEval,
    trunk: &SurfaceEval,
    head: &SurfaceEval,
    joint: &SurfaceEval,
    trunk_move_count: usize,
    head_move_count: usize,
    model_hashes: [u64; 4],
) -> Result<ProductionBooleanJetSurfaceTrace, TrainError> {
    if [
        base.losses_q20.len(),
        trunk.losses_q20.len(),
        head.losses_q20.len(),
        joint.losses_q20.len(),
    ]
    .into_iter()
    .any(|length| length != windows.len())
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_window_loss_shape",
        ));
    }
    let vertices = [
        vertex("empty", base, base, 0, model_hashes[0]),
        vertex("trunk", base, trunk, trunk_move_count, model_hashes[1]),
        vertex("head", base, head, head_move_count, model_hashes[2]),
        vertex(
            "trunk_head",
            base,
            joint,
            trunk_move_count.saturating_add(head_move_count),
            model_hashes[3],
        ),
    ];
    let minimum = vertices
        .iter()
        .map(|vertex| vertex.nll_q20)
        .min()
        .expect("rank-two cube has four vertices");
    let minimizing_vertices = vertices
        .iter()
        .filter(|vertex| vertex.nll_q20 == minimum)
        .map(|vertex| vertex.vertex)
        .collect::<Vec<_>>();
    let minimizing_mask = vertices
        .iter()
        .enumerate()
        .filter(|(_, vertex)| vertex.nll_q20 == minimum)
        .fold(0_u8, |mask, (index, _)| mask | (1_u8 << index));
    let losses = [base.nll_q20, trunk.nll_q20, head.nll_q20, joint.nll_q20];
    let coefficients = mobius_coefficients(&losses)?;
    let mu_trunk_q20 = coefficients[1];
    let mu_head_q20 = coefficients[2];
    let mu_trunk_head_q20 = coefficients[3];
    let joint_delta_q20 = difference(joint.nll_q20, base.nll_q20);
    let reconstruction_verified = mobius_reconstructs(&losses, &coefficients);
    if !reconstruction_verified {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_mobius_reconstruction",
        ));
    }
    let gamma_one_q20 = mu_trunk_q20.min(mu_head_q20);
    let boolean_one_prime = gamma_one_q20 >= 0;
    let escape_order = if trunk.nll_q20 < base.nll_q20 || head.nll_q20 < base.nll_q20 {
        Some(1)
    } else if joint.nll_q20 < base.nll_q20 {
        Some(2)
    } else {
        None
    };
    let mut by_document = BTreeMap::<usize, (usize, [u64; 4])>::new();
    for (index, window) in windows.iter().enumerate() {
        let entry = by_document.entry(window.document).or_insert((0, [0; 4]));
        entry.0 = entry.0.checked_add(1).ok_or(TrainError::CoreRejected(
            "production_boolean_jet_document_window_overflow",
        ))?;
        for (sum, loss) in entry.1.iter_mut().zip([
            base.losses_q20[index],
            trunk.losses_q20[index],
            head.losses_q20[index],
            joint.losses_q20[index],
        ]) {
            *sum = sum.checked_add(loss).ok_or(TrainError::CoreRejected(
                "production_boolean_jet_document_nll_overflow",
            ))?;
        }
    }
    let documents = by_document
        .into_iter()
        .map(|(document, (windows, losses))| {
            let coefficients = mobius_coefficients(&losses)
                .expect("four document losses form a valid rank-two table");
            let mu_trunk_q20 = coefficients[1];
            let mu_head_q20 = coefficients[2];
            let mu_trunk_head_q20 = coefficients[3];
            let reconstruction_verified = mobius_reconstructs(&losses, &coefficients);
            ProductionBooleanJetDocumentTrace {
                document,
                windows,
                vertex_nll_q20: losses,
                mu_trunk_q20,
                mu_head_q20,
                mu_trunk_head_q20,
                conditional_trunk_after_head_q20: difference(losses[3], losses[2]),
                reconstruction_verified,
            }
        })
        .collect::<Vec<_>>();
    if documents
        .iter()
        .any(|document| !document.reconstruction_verified)
    {
        return Err(TrainError::CoreRejected(
            "production_boolean_jet_document_mobius_reconstruction",
        ));
    }

    Ok(ProductionBooleanJetSurfaceTrace {
        surface,
        vertices,
        mu_trunk_q20,
        mu_head_q20,
        mu_trunk_head_q20,
        joint_delta_q20,
        gamma_one_q20,
        boolean_one_prime,
        escape_order,
        minimizing_mask,
        minimizing_vertices,
        reconstruction_verified,
        documents,
    })
}

fn difference(left: u64, right: u64) -> i128 {
    i128::from(left) - i128::from(right)
}

fn vertex(
    name: &'static str,
    base: &SurfaceEval,
    value: &SurfaceEval,
    applied_moves: usize,
    model_hash: u64,
) -> ProductionBooleanJetVertex {
    ProductionBooleanJetVertex {
        vertex: name,
        nll_q20: value.nll_q20,
        function_visible: name != "empty" && value.logits != base.logits,
        applied_moves,
        parameter_saturation_count: 0,
        model_hash,
        function_hash: function_hash(&value.logits),
    }
}

fn function_hash(logits: &[Vec<i32>]) -> u64 {
    let mut hash = FNV_OFFSET;
    for window in logits {
        for byte in (window.len() as u64).to_le_bytes() {
            hash = fnv_byte(hash, byte);
        }
        for value in window {
            for byte in value.to_le_bytes() {
                hash = fnv_byte(hash, byte);
            }
        }
    }
    hash
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
        .expect("writing Boolean-jet window binding cannot fail");
    }
    json.push(']');
}

fn push_moves(json: &mut String, moves: &[ProductionBooleanJetMove]) {
    json.push('[');
    for (index, movement) in moves.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"block\":\"{}\",\"group\":\"{}\",\"group_index\":{},",
                "\"coordinate\":{},\"parameter_delta\":{},\"coarse_gradient\":{},",
                "\"source_lane\":\"{}\",\"move_kind\":\"{}\",\"canonical_order\":{},",
                "\"selection_strata\":["
            ),
            movement.block,
            movement.group,
            movement.group_index,
            movement.coordinate,
            movement.parameter_delta,
            movement.coarse_gradient,
            movement.source_lane,
            movement.move_kind,
            movement.canonical_order,
        )
        .expect("writing Boolean-jet move cannot fail");
        for (stratum_index, stratum) in movement.selection_strata.iter().enumerate() {
            if stratum_index != 0 {
                json.push(',');
            }
            write!(json, "\"{stratum}\"").expect("writing selection stratum cannot fail");
        }
        json.push_str("]}");
    }
    json.push(']');
}

fn push_legacy_moves(json: &mut String, moves: &[ProductionBooleanJetMove]) {
    json.push('[');
    for (index, movement) in moves.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"block\":\"{}\",\"group\":\"{}\",\"group_index\":{},",
                "\"coordinate\":{},\"parameter_delta\":{},\"coarse_gradient\":{},",
                "\"selection_strata\":["
            ),
            movement.block,
            movement.group,
            movement.group_index,
            movement.coordinate,
            movement.parameter_delta,
            movement.coarse_gradient,
        )
        .expect("writing legacy Boolean-jet move cannot fail");
        for (stratum_index, stratum) in movement.selection_strata.iter().enumerate() {
            if stratum_index != 0 {
                json.push(',');
            }
            write!(json, "\"{stratum}\"").expect("writing legacy selection stratum cannot fail");
        }
        json.push_str("]}");
    }
    json.push(']');
}

fn push_surface(json: &mut String, surface: &ProductionBooleanJetSurfaceTrace) {
    write!(json, "{{\"surface\":\"{}\",\"vertices\":[", surface.surface,)
        .expect("writing Boolean-jet surface cannot fail");
    for (index, vertex) in surface.vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"vertex\":\"{}\",\"nll_q20\":{},\"function_visible\":{},",
                "\"applied_moves\":{},\"parameter_saturation_count\":{},",
                "\"model_hash\":\"0x{:016x}\",\"function_hash\":\"0x{:016x}\"}}"
            ),
            vertex.vertex,
            vertex.nll_q20,
            vertex.function_visible,
            vertex.applied_moves,
            vertex.parameter_saturation_count,
            vertex.model_hash,
            vertex.function_hash,
        )
        .expect("writing Boolean-jet vertex cannot fail");
    }
    write!(
        json,
        concat!(
            "],\"mobius_q20\":{{\"trunk\":{},\"head\":{},\"trunk_head\":{}}},",
            "\"joint_delta_q20\":{},\"gamma_one_q20\":{},",
            "\"boolean_one_prime\":{},\"escape_order\":"
        ),
        surface.mu_trunk_q20,
        surface.mu_head_q20,
        surface.mu_trunk_head_q20,
        surface.joint_delta_q20,
        surface.gamma_one_q20,
        surface.boolean_one_prime,
    )
    .expect("writing Boolean-jet coefficients cannot fail");
    match surface.escape_order {
        Some(order) => write!(json, "{order}").expect("writing escape order cannot fail"),
        None => json.push_str("null"),
    }
    write!(
        json,
        ",\"minimizing_vertex_bitset\":{},\"minimizing_vertices\":[",
        surface.minimizing_mask,
    )
    .expect("writing minimizing bitset cannot fail");
    for (index, vertex) in surface.minimizing_vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"{vertex}\"").expect("writing minimizing vertex cannot fail");
    }
    write!(
        json,
        "],\"reconstruction_verified\":{},\"documents\":[",
        surface.reconstruction_verified,
    )
    .expect("writing reconstruction status cannot fail");
    for (index, document) in surface.documents.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"document\":{},\"windows\":{},",
                "\"vertex_nll_q20\":{{\"empty\":{},\"trunk\":{},\"head\":{},",
                "\"trunk_head\":{}}},",
                "\"mobius_q20\":{{\"trunk\":{},\"head\":{},\"trunk_head\":{}}},",
                "\"conditional_trunk_after_head_q20\":{},",
                "\"reconstruction_verified\":{}}}"
            ),
            document.document,
            document.windows,
            document.vertex_nll_q20[0],
            document.vertex_nll_q20[1],
            document.vertex_nll_q20[2],
            document.vertex_nll_q20[3],
            document.mu_trunk_q20,
            document.mu_head_q20,
            document.mu_trunk_head_q20,
            document.conditional_trunk_after_head_q20,
            document.reconstruction_verified,
        )
        .expect("writing document Boolean coefficients cannot fail");
    }
    json.push_str("]}");
}

fn push_legacy_surface(json: &mut String, surface: &ProductionBooleanJetSurfaceTrace) {
    write!(json, "{{\"surface\":\"{}\",\"vertices\":[", surface.surface)
        .expect("writing legacy Boolean-jet surface cannot fail");
    for (index, vertex) in surface.vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"vertex\":\"{}\",\"nll_q20\":{},\"function_visible\":{},",
                "\"applied_moves\":{},\"parameter_saturation_count\":{}}}"
            ),
            vertex.vertex,
            vertex.nll_q20,
            vertex.function_visible,
            vertex.applied_moves,
            vertex.parameter_saturation_count,
        )
        .expect("writing legacy Boolean-jet vertex cannot fail");
    }
    write!(
        json,
        concat!(
            "],\"mobius_q20\":{{\"trunk\":{},\"head\":{},\"trunk_head\":{}}},",
            "\"joint_delta_q20\":{},\"boolean_one_prime\":{},\"escape_order\":"
        ),
        surface.mu_trunk_q20,
        surface.mu_head_q20,
        surface.mu_trunk_head_q20,
        surface.joint_delta_q20,
        surface.boolean_one_prime,
    )
    .expect("writing legacy Boolean-jet coefficients cannot fail");
    match surface.escape_order {
        Some(order) => write!(json, "{order}").expect("writing legacy escape order cannot fail"),
        None => json.push_str("null"),
    }
    json.push_str(",\"minimizing_vertices\":[");
    for (index, vertex) in surface.minimizing_vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"{vertex}\"").expect("writing legacy minimizing vertex cannot fail");
    }
    json.push_str("]}");
}

fn push_objective(json: &mut String, objective: ProductionBooleanJetObjectiveSpec) {
    write!(
        json,
        concat!(
            "{{\"algorithm\":\"{}\",\"version\":{},\"fractional_bits\":{},",
            "\"zero_probability_floor_units\":{},\"aggregation\":\"{}\"}}"
        ),
        objective.algorithm.as_str(),
        objective.version,
        objective.fractional_bits,
        objective.zero_probability_floor_q20,
        objective.aggregation.as_str(),
    )
    .expect("writing objective cannot fail");
}

fn push_branch_surface(json: &mut String, surface: &ProductionBooleanJetBranchSurfaceTrace) {
    write!(json, "{{\"surface\":\"{}\",\"vertices\":[", surface.surface)
        .expect("writing branch surface cannot fail");
    for (index, vertex) in surface.vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"vertex\":\"{}\",\"embedding_hash\":\"0x{:016x}\",",
                "\"attention_residual_hashes\":["
            ),
            vertex.vertex, vertex.embedding_hash,
        )
        .expect("writing branch vertex cannot fail");
        push_hashes(json, &vertex.attention_residual_hashes);
        json.push_str("],\"layer_output_hashes\":[");
        push_hashes(json, &vertex.layer_output_hashes);
        write!(
            json,
            concat!(
                "],\"final_features_hash\":\"0x{:016x}\",",
                "\"logits_hash\":\"0x{:016x}\",\"first_divergent_boundary\":"
            ),
            vertex.final_features_hash, vertex.logits_hash,
        )
        .expect("writing branch terminal hashes cannot fail");
        match &vertex.first_divergent_boundary {
            Some(boundary) => {
                write!(json, "\"{boundary}\"").expect("writing divergent boundary cannot fail")
            }
            None => json.push_str("null"),
        }
        json.push('}');
    }
    json.push_str("]}");
}

fn push_hashes(json: &mut String, hashes: &[u64]) {
    for (index, hash) in hashes.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(json, "\"0x{hash:016x}\"").expect("writing boundary hash cannot fail");
    }
}

fn push_confirmation_surface_units(
    json: &mut String,
    surface: &ProductionBooleanJetConfirmationSurfaceTrace,
    fractional_bits: u8,
) {
    write!(
        json,
        concat!(
            "{{\"fractional_bits\":{},\"document_block\":{{\"start\":{},",
            "\"count\":{},\"windows_per_document\":{}}},\"cube\":"
        ),
        fractional_bits,
        surface.document_start,
        surface.document_count,
        surface.windows_per_document,
    )
    .expect("writing unit surface header cannot fail");
    push_surface_units(json, &surface.cube);
    json.push_str(",\"conditional_sign_test\":");
    push_sign_test(json, surface.conditional_sign_test);
    if let Some(control) = &surface.matched_control {
        json.push_str(",\"matched_control\":");
        push_matched_control_units(json, control);
    }
    json.push('}');
}

fn push_surface_units(json: &mut String, surface: &ProductionBooleanJetSurfaceTrace) {
    write!(json, "{{\"surface\":\"{}\",\"vertices\":[", surface.surface)
        .expect("writing unit cube header cannot fail");
    for (index, vertex) in surface.vertices.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"vertex\":\"{}\",\"nll_units\":{},\"function_visible\":{},",
                "\"applied_moves\":{},\"parameter_saturation_count\":{},",
                "\"model_hash\":\"0x{:016x}\",\"function_hash\":\"0x{:016x}\"}}"
            ),
            vertex.vertex,
            vertex.nll_q20,
            vertex.function_visible,
            vertex.applied_moves,
            vertex.parameter_saturation_count,
            vertex.model_hash,
            vertex.function_hash,
        )
        .expect("writing unit vertex cannot fail");
    }
    write!(
        json,
        concat!(
            "],\"mobius_units\":{{\"trunk\":{},\"head\":{},\"trunk_head\":{}}},",
            "\"joint_delta_units\":{},\"gamma_one_units\":{},",
            "\"boolean_one_prime\":{},\"reconstruction_verified\":{},",
            "\"documents\":["
        ),
        surface.mu_trunk_q20,
        surface.mu_head_q20,
        surface.mu_trunk_head_q20,
        surface.joint_delta_q20,
        surface.gamma_one_q20,
        surface.boolean_one_prime,
        surface.reconstruction_verified,
    )
    .expect("writing unit coefficients cannot fail");
    for (index, document) in surface.documents.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"document\":{},\"windows\":{},",
                "\"vertex_nll_units\":{{\"empty\":{},\"trunk\":{},",
                "\"head\":{},\"trunk_head\":{}}},",
                "\"mobius_units\":{{\"trunk\":{},\"head\":{},\"trunk_head\":{}}},",
                "\"conditional_trunk_after_head_units\":{},",
                "\"reconstruction_verified\":{}}}"
            ),
            document.document,
            document.windows,
            document.vertex_nll_q20[0],
            document.vertex_nll_q20[1],
            document.vertex_nll_q20[2],
            document.vertex_nll_q20[3],
            document.mu_trunk_q20,
            document.mu_head_q20,
            document.mu_trunk_head_q20,
            document.conditional_trunk_after_head_q20,
            document.reconstruction_verified,
        )
        .expect("writing unit document cannot fail");
    }
    json.push_str("]}");
}

fn push_matched_control_units(
    json: &mut String,
    control: &ProductionBooleanJetMatchedControlSurfaceTrace,
) {
    write!(
        json,
        concat!(
            "{{\"control_vertex\":{{\"nll_units\":{},\"function_visible\":{}}},",
            "\"head_control_vertex\":{{\"nll_units\":{},\"function_visible\":{}}},",
            "\"documents\":["
        ),
        control.control_vertex.nll_q20,
        control.control_vertex.function_visible,
        control.head_control_vertex.nll_q20,
        control.head_control_vertex.function_visible,
    )
    .expect("writing unit controls cannot fail");
    for (index, document) in control.documents.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"document\":{},\"control_nll_units\":{},",
                "\"head_control_nll_units\":{},",
                "\"conditional_control_after_head_units\":{},",
                "\"structured_minus_control_conditional_units\":{}}}"
            ),
            document.document,
            document.control_nll_q20,
            document.head_control_nll_q20,
            document.conditional_control_after_head_q20,
            document.structured_minus_control_conditional_q20,
        )
        .expect("writing unit control document cannot fail");
    }
    json.push_str("],\"conditional_control_sign_test\":");
    push_sign_test(json, control.conditional_control_sign_test);
    json.push_str(",\"structured_beats_control_sign_test\":");
    push_sign_test(json, control.structured_beats_control_sign_test);
    json.push('}');
}

fn push_confirmation_surface(
    json: &mut String,
    surface: &ProductionBooleanJetConfirmationSurfaceTrace,
) {
    write!(
        json,
        concat!(
            "{{\"document_block\":{{\"start\":{},\"count\":{},",
            "\"windows_per_document\":{}}},\"cube\":"
        ),
        surface.document_start, surface.document_count, surface.windows_per_document,
    )
    .expect("writing confirmation document block cannot fail");
    push_surface(json, &surface.cube);
    write!(
        json,
        concat!(
            ",\"conditional_sign_test\":{{\"joint_wins\":{},\"head_wins\":{},",
            "\"ties\":{},\"non_ties\":{},\"exact_p_numerator\":{},",
            "\"exact_p_denominator\":{},\"p_per_million\":{},",
            "\"direction_supported\":{}}}"
        ),
        surface.conditional_sign_test.joint_wins,
        surface.conditional_sign_test.head_wins,
        surface.conditional_sign_test.ties,
        surface.conditional_sign_test.non_ties,
        surface.conditional_sign_test.exact_p_numerator,
        surface.conditional_sign_test.exact_p_denominator,
        surface.conditional_sign_test.p_per_million,
        surface.conditional_sign_test.direction_supported,
    )
    .expect("writing confirmation sign test cannot fail");
    let Some(control) = surface.matched_control.as_ref() else {
        json.push('}');
        return;
    };
    write!(
        json,
        concat!(
            ",\"matched_control\":{{",
            "\"control_vertex\":{{\"nll_q20\":{},\"function_visible\":{},",
            "\"model_hash\":\"0x{:016x}\",\"function_hash\":\"0x{:016x}\"}},",
            "\"head_control_vertex\":{{\"nll_q20\":{},\"function_visible\":{},",
            "\"model_hash\":\"0x{:016x}\",\"function_hash\":\"0x{:016x}\"}},",
            "\"documents\":["
        ),
        control.control_vertex.nll_q20,
        control.control_vertex.function_visible,
        control.control_vertex.model_hash,
        control.control_vertex.function_hash,
        control.head_control_vertex.nll_q20,
        control.head_control_vertex.function_visible,
        control.head_control_vertex.model_hash,
        control.head_control_vertex.function_hash,
    )
    .expect("writing matched-control vertices cannot fail");
    for (index, document) in control.documents.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        write!(
            json,
            concat!(
                "{{\"document\":{},\"control_nll_q20\":{},",
                "\"head_control_nll_q20\":{},",
                "\"conditional_control_after_head_q20\":{},",
                "\"structured_minus_control_conditional_q20\":{}}}"
            ),
            document.document,
            document.control_nll_q20,
            document.head_control_nll_q20,
            document.conditional_control_after_head_q20,
            document.structured_minus_control_conditional_q20,
        )
        .expect("writing matched-control document cannot fail");
    }
    json.push_str("],\"conditional_control_sign_test\":");
    push_sign_test(json, control.conditional_control_sign_test);
    json.push_str(",\"structured_beats_control_sign_test\":");
    push_sign_test(json, control.structured_beats_control_sign_test);
    json.push_str("}}");
}

fn push_sign_test(json: &mut String, test: ProductionBooleanJetSignTest) {
    write!(
        json,
        concat!(
            "{{\"negative_wins\":{},\"positive_wins\":{},\"ties\":{},",
            "\"non_ties\":{},\"exact_p_numerator\":\"{}\",",
            "\"exact_p_denominator\":\"{}\",",
            "\"p_per_million\":{},\"direction_supported\":{}}}"
        ),
        test.joint_wins,
        test.head_wins,
        test.ties,
        test.non_ties,
        test.exact_p_numerator,
        test.exact_p_denominator,
        test.p_per_million,
        test.direction_supported,
    )
    .expect("writing sign test cannot fail");
}

#[cfg(test)]
mod tests {
    use nsrl_core::SoftmaxNormalization;
    use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

    use super::*;
    use crate::production::ProductionModelConfig;

    #[test]
    fn fast_mobius_transform_reconstructs_rank_three_cube() {
        let losses = [2_u64, 3, 3, 4, 3, 4, 4, 1];
        let coefficients = mobius_coefficients(&losses).expect("Möbius coefficients");
        assert_eq!(coefficients, [2, 1, 1, 0, 1, 0, 0, -4]);
        assert!(mobius_reconstructs(&losses, &coefficients));
    }

    #[test]
    fn exact_sign_test_excludes_ties_and_reduces_the_rational() {
        assert_eq!(exact_two_sided_sign_p(4, 0), (1, 8));
        assert_eq!(exact_two_sided_sign_p(3, 1), (5, 8));
        assert_eq!(exact_two_sided_sign_p(0, 0), (1, 1));
        let documents = [
            document_with_conditional_delta(-1),
            document_with_conditional_delta(-2),
            document_with_conditional_delta(0),
            document_with_conditional_delta(3),
        ];
        let test = conditional_sign_test(&documents, 1, 20).expect("sign test");
        assert_eq!(test.joint_wins, 2);
        assert_eq!(test.head_wins, 1);
        assert_eq!(test.ties, 1);
        assert_eq!(test.non_ties, 3);
    }

    #[test]
    fn mobius_transform_reconstructs_known_pair_and_triple_tables() {
        let pair = [10, 13, 8, 4];
        let pair_mu = mobius_coefficients(&pair).expect("pair transform");
        assert_eq!(pair_mu, [10, 3, -2, -7]);
        assert!(mobius_reconstructs(&pair, &pair_mu));

        let triple = [20, 23, 18, 17, 24, 30, 19, 9];
        let triple_mu = mobius_coefficients(&triple).expect("triple transform");
        assert!(mobius_reconstructs(&triple, &triple_mu));
        assert_eq!(triple_mu.len(), 8);
        assert_eq!(triple_mu[7], -12);
    }

    fn document_with_conditional_delta(delta: i128) -> ProductionBooleanJetDocumentTrace {
        ProductionBooleanJetDocumentTrace {
            document: 0,
            windows: 1,
            vertex_nll_q20: [0; 4],
            mu_trunk_q20: 0,
            mu_head_q20: 0,
            mu_trunk_head_q20: 0,
            conditional_trunk_after_head_q20: delta,
            reconstruction_verified: true,
        }
    }

    #[test]
    fn move_family_rejects_noncanonical_colliding_and_boundary_atoms() {
        let config = ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 2,
            hidden_dim: 48,
            context_tokens: 16,
        };
        let mut model = ProductionModelV1::new_initial(config, 0x1234, 11).expect("model");
        model.final_rms_weights[0] = 0;
        let atom = |canonical_order| ProductionBooleanJetMove {
            block: "trunk",
            group: "final_rms",
            group_index: TRUNK_GROUP_INDEX,
            coordinate: 0,
            parameter_delta: -1,
            coarse_gradient: 1,
            selection_strata: vec!["test"],
            source_lane: "mass-corrected-normalized-rhu",
            move_kind: "model_only_unit_sign_probe",
            canonical_order,
        };

        let mut noncanonical = atom(1);
        assert!(validate_moves(&model, &[noncanonical.clone()], &[]).is_err());
        noncanonical.canonical_order = 0;
        let mut collision = noncanonical.clone();
        collision.block = "head";
        collision.canonical_order = 1;
        assert!(validate_moves(&model, &[noncanonical.clone()], &[collision]).is_err());

        model.final_rms_weights[0] = i16::MAX;
        assert!(validate_moves(&model, &[noncanonical], &[]).is_err());
    }

    #[test]
    fn rank_two_audit_is_exact_deterministic_and_restores_the_model() {
        let config = ProductionModelConfig {
            vocab_size: 320,
            d_model: 16,
            heads: 4,
            layers: 2,
            hidden_dim: 48,
            context_tokens: 16,
        };
        let mut model = ProductionModelV1::new_initial(config, 0x1234, 11).expect("model");
        // Keep repeated saturation margins in this tiny fixture so the
        // matched-control search exercises visibility matching rather than
        // failing because randomized production RMS initialization gives all
        // sixteen coordinates distinct margins.
        model.final_rms_weights.fill(30_000);
        model.initialize_output_weights(2).expect("output weights");
        let original = model.clone();
        let tokens = vec![
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
            BOS_TOKEN_ID,
            310,
            311,
            312,
            313,
            314,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            315,
            316,
            317,
            318,
            319,
            EOS_TOKEN_ID,
        ];
        let training = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 1,
            epochs: 1,
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            ..ProductionFullTrainConfig::default()
        };
        let alignment_config = ProductionGradientAlignmentConfig {
            proposal_windows: 1,
            transfer_windows: 1,
            documents_per_surface: 0,
            rescue_stratified_sampling: true,
            include_mass_corrected_no_rescue: true,
            include_systematic_fixed_mass: false,
            coordinates_per_group: 1,
            sample_seed: 19,
        };
        let alignment = audit_production_gradient_alignment(
            &model,
            &tokens,
            0x5678,
            training,
            alignment_config,
        )
        .expect("alignment");
        let expected_trunk_moves = alignment
            .samples
            .iter()
            .filter(|sample| sample.group_index == TRUNK_GROUP_INDEX)
            .count();
        let expected_head_moves = alignment
            .samples
            .iter()
            .filter(|sample| sample.group_index >= HEAD_GROUP_START)
            .count();
        assert!(expected_trunk_moves > 0);
        assert!(expected_head_moves > 0);
        let audit_config = ProductionBooleanJetRankTwoConfig {
            alignment: alignment_config,
            expected_trunk_moves,
            expected_head_moves,
            expected_move_fingerprint: 0,
        };
        let left =
            audit_production_boolean_jet_rank_two(&model, &tokens, 0x5678, training, audit_config)
                .expect("Boolean-jet audit");
        let right =
            audit_production_boolean_jet_rank_two(&model, &tokens, 0x5678, training, audit_config)
                .expect("Boolean-jet replay");
        assert_eq!(model, original);
        assert_eq!(left, right);
        for surface in [&left.proposal, &left.transfer] {
            assert_eq!(
                surface.joint_delta_q20,
                surface.mu_trunk_q20 + surface.mu_head_q20 + surface.mu_trunk_head_q20
            );
            assert!(!surface.minimizing_vertices.is_empty());
            assert!(
                surface
                    .vertices
                    .iter()
                    .all(|vertex| vertex.parameter_saturation_count == 0)
            );
        }
        let json = left.to_json_line();
        assert!(json.contains("nsrl.production_boolean_jet.v1"));
        assert!(json.contains("post_hoc_v3_calibration"));
        assert!(json.contains("\"trunk_head\""));
        let legacy_json = left.to_legacy_json_line();
        assert!(legacy_json.contains("nsrl.production_boolean_jet_rank_two.v1"));
        assert!(!legacy_json.contains("manifest_hash"));

        let move_contract = ProductionBooleanJetMoveContract {
            protocol_version: ProductionBooleanJetProtocolVersion::ConfirmationV1,
            analysis_role: ProductionBooleanJetAnalysisRole::Confirmation,
            expected_source_fnv64: 0,
            expected_binary_fnv64: 0,
            expected_base_model_hash: left.model_hash,
            expected_tokenizer_hash: left.tokenizer_hash,
            expected_token_stream_hash: left.token_stream_hash,
            expected_move_fingerprint: left.move_fingerprint,
            expected_manifest_hash: left.manifest_hash,
            trunk_moves: left.trunk_moves.clone(),
            head_moves: left.head_moves.clone(),
            matched_control_moves: Vec::new(),
        };
        let confirmation_config = ProductionBooleanJetConfirmationConfig {
            context_tokens: 4,
            objective: ProductionBooleanJetObjectiveSpec::wide_q47_v1(),
            move_contract,
            proposal_document_start: 2,
            proposal_documents: 1,
            transfer_document_start: 3,
            transfer_documents: 1,
            windows_per_document: 1,
            minimum_independent_documents: 1,
            significance_numerator: 1,
            significance_denominator: 20,
        };
        let confirmation =
            audit_production_boolean_jet_confirmation(&model, &tokens, 0x5678, confirmation_config)
                .expect("document-disjoint confirmation");
        assert_eq!(model, original);
        assert_eq!(
            confirmation.analysis_role,
            ProductionBooleanJetAnalysisRole::Confirmation
        );
        assert_eq!(
            confirmation.objective,
            ProductionBooleanJetObjectiveSpec::wide_q47_v1()
        );
        assert_eq!(confirmation.proposal.cube.documents[0].document, 2);
        assert_eq!(confirmation.transfer.cube.documents[0].document, 3);
        assert_eq!(confirmation.manifest_hash, left.manifest_hash);
        assert!(confirmation.matched_control_moves.is_empty());
        assert!(!confirmation.optimizer_change_authorized);

        let protocol_bindings = ProductionBooleanJetProtocolBindings {
            source_fnv64: production_boolean_jet_source_fnv64(),
            binary_fnv64: 0x55aa,
        };
        let control_manifest = freeze_production_boolean_jet_matched_control(
            &model,
            &tokens,
            0x5678,
            &left.trunk_moves,
            &left.head_moves,
            ProductionBooleanJetMatchedControlV2Config {
                context_tokens: 4,
                visibility_document_start: 2,
                visibility_documents: 1,
                windows_per_document: 1,
                reserved_document_start: 3,
                seed: 19,
                protocol_bindings,
            },
        )
        .expect("freeze visibility- and margin-matched control");
        assert_eq!(
            control_manifest.structured_visibility_hash,
            control_manifest.control_visibility_hash
        );
        assert_eq!(
            control_manifest.structured_saturation_margin_hash,
            control_manifest.control_saturation_margin_hash
        );
        let v2 = audit_production_boolean_jet_confirmation_v2(
            &model,
            &tokens,
            0x5678,
            ProductionBooleanJetConfirmationV2Config {
                primary: ProductionBooleanJetConfirmationConfig {
                    context_tokens: 4,
                    objective: ProductionBooleanJetObjectiveSpec::wide_q47_q32_v2(),
                    move_contract: ProductionBooleanJetMoveContract {
                        protocol_version: ProductionBooleanJetProtocolVersion::StabilityV2,
                        analysis_role: ProductionBooleanJetAnalysisRole::Confirmation,
                        expected_source_fnv64: protocol_bindings.source_fnv64,
                        expected_binary_fnv64: protocol_bindings.binary_fnv64,
                        expected_base_model_hash: left.model_hash,
                        expected_tokenizer_hash: left.tokenizer_hash,
                        expected_token_stream_hash: left.token_stream_hash,
                        expected_move_fingerprint: left.move_fingerprint,
                        expected_manifest_hash: control_manifest.manifest_hash,
                        trunk_moves: left.trunk_moves.clone(),
                        head_moves: left.head_moves.clone(),
                        matched_control_moves: control_manifest.moves.clone(),
                    },
                    proposal_document_start: 2,
                    proposal_documents: 1,
                    transfer_document_start: 3,
                    transfer_documents: 1,
                    windows_per_document: 1,
                    minimum_independent_documents: 1,
                    significance_numerator: 1,
                    significance_denominator: 20,
                },
                robustness_objective: ProductionBooleanJetObjectiveSpec::canonical_q15_v1(),
                protocol_bindings,
                reserved_document_start: 3,
            },
        )
        .expect("v2 stability confirmation");
        assert_eq!(v2.primary.objective.fractional_bits, 32);
        assert!(!v2.gates.optimizer_transition_tested);
        assert!(!v2.gates.optimizer_change_authorized);
        assert_eq!(v2.transfer_branch_localization.vertices.len(), 4);
        let v2_json = v2.to_json_line();
        assert!(v2_json.contains("nsrl.production_boolean_jet_stability_confirmation.v2"));
        assert!(v2_json.contains("\"nll_units\""));
        assert!(v2_json.contains("\"decision_gates\""));
    }
}
