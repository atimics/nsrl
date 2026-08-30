//! Variable-vocabulary production decoder artifact and bounded integer smoke runtime.
//!
//! This module is separate from MT5/MT6 so their byte-vocabulary artifacts and
//! frozen proof semantics remain unchanged.

use std::collections::BTreeSet;
use std::fmt::Write;

use nsrl_core::{
    DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS, FixedScale, GatedMlpI16Params, GatedMlpWorkspace,
    LinearAttentionWorkspace, LinearI16I8Params, MASKED_LOGIT, SelfAttentionI16Params,
    SoftmaxNormalization, base2_exp_neg_q15, base2_softmax_i32_q15, base2_softmax_i32_q31,
    base2_softmax_i32_q31_with_normalization, base2_softmax_nll_millibits,
    gated_mlp_i16_q15_checked, linear_attention_i16_q15_checked, rms_norm_i16_q15_checked,
    round_shift_rhu_i64, saturate_i16,
};
use nsrl_corpus::subword::{BOS_TOKEN_ID, EOS_TOKEN_ID};

use crate::{PRODUCTION_MODEL_V1_MAGIC, TrainError};

mod alignment;
mod boolean_jet;
mod generation;
mod margin_training;
mod numeric_contract;
mod structure_audit;
mod training;
pub use alignment::{
    ProductionGradientAlignmentConfig, ProductionGradientAlignmentGate,
    ProductionGradientAlignmentSample, ProductionGradientAlignmentSummary,
    ProductionGradientAlignmentTrace, ProductionGradientLaneHealth, ProductionGradientLaneSample,
    ProductionGradientLaneTrace, ProductionGradientSurfaceDelta, ProductionGradientWindowBinding,
    audit_production_gradient_alignment,
};
pub use boolean_jet::{
    PRODUCTION_BOOLEAN_JET_RESERVED_DOCUMENT_START, ProductionBooleanJetAggregationRule,
    ProductionBooleanJetAnalysisRole, ProductionBooleanJetBranchSurfaceTrace,
    ProductionBooleanJetBranchVertexTrace, ProductionBooleanJetConfirmationConfig,
    ProductionBooleanJetConfirmationSurfaceTrace, ProductionBooleanJetConfirmationTrace,
    ProductionBooleanJetConfirmationV2Config, ProductionBooleanJetConfirmationV2Trace,
    ProductionBooleanJetDecisionGates, ProductionBooleanJetDocumentTrace,
    ProductionBooleanJetMatchedControlDocumentTrace, ProductionBooleanJetMatchedControlManifest,
    ProductionBooleanJetMatchedControlSurfaceTrace, ProductionBooleanJetMatchedControlV2Config,
    ProductionBooleanJetMove, ProductionBooleanJetMoveContract,
    ProductionBooleanJetObjectiveAlgorithm, ProductionBooleanJetObjectiveRobustnessTrace,
    ProductionBooleanJetObjectiveSpec, ProductionBooleanJetProtocolBindings,
    ProductionBooleanJetProtocolVersion, ProductionBooleanJetRankTwoConfig,
    ProductionBooleanJetRankTwoTrace, ProductionBooleanJetSignTest,
    ProductionBooleanJetSurfaceTrace, ProductionBooleanJetVertex,
    audit_production_boolean_jet_confirmation, audit_production_boolean_jet_confirmation_v2,
    audit_production_boolean_jet_rank_two, freeze_production_boolean_jet_matched_control,
    production_boolean_jet_binary_fnv64, production_boolean_jet_source_fnv64,
};
pub use generation::{
    PRODUCTION_GENERATION_SCHEMA, ProductionDecoder, ProductionGenerationConfig,
    ProductionGenerationStepTrace, ProductionGenerationTrace, generate_production_model,
};
pub use margin_training::{
    ProductionMarginEvaluation, ProductionMarginOptimizerStateV1, ProductionMarginTrainConfig,
    ProductionMarginTrainTrace, train_production_target_margin,
};
pub use numeric_contract::{
    PRODUCTION_ACTIVATION_FRACTIONAL_BITS, PRODUCTION_LOGIT_FRACTIONAL_BITS,
    PRODUCTION_RMS_SQUARE_FRACTIONAL_BITS, ProductionAttentionBounds,
    ProductionBackwardEdgeContract, ProductionNumericContract, ProductionParameterUpdateContract,
    ProductionProjectionContract, ProductionRoundingRule, ProductionTrainingNumericContract,
};
pub use structure_audit::{
    ProductionAtomicDocumentCoefficients, ProductionAtomicDocumentRange,
    ProductionAtomicObjectiveTrace, ProductionAtomicSourceBinding,
    ProductionAtomicStructureContract, ProductionAtomicStructureRole,
    ProductionAtomicStructureTrace, ProductionBoundaryTaxonomy,
    ProductionDocumentRepresentationDiscrepancy, ProductionExchangeTrace,
    ProductionInteractionTailTrace, ProductionInteractionWidthTrace,
    ProductionRepresentationConcordance, ProductionRepresentationDiscrepancy,
    audit_production_atomic_structure, freeze_production_atomic_structure_contract,
};
pub use training::{
    DirectFeatureTrainConfig, DirectFeatureTrainTrace, DirectHeadCoordinateDirection,
    DirectHeadCrossDocumentAuditConfig, DirectHeadCrossDocumentAuditTrace,
    DirectHeadCrossDocumentDirectionTrace, DirectHeadCrossDocumentSampleTrace,
    DirectHeadCrossDocumentSurfaceTrace, DirectHeadTrainConfig, DirectTrainWindowBinding,
    ProductionBackwardQuantization, ProductionDescentRejectedBatchTrace, ProductionFullTrainConfig,
    ProductionFullTrainTrace, ProductionGradientProposalLane, ProductionOptimizerStateV2,
    ProductionRejectedBatchTrace, ProductionSignedBlockSelectionTrace,
    audit_production_direct_head_cross_document, train_production_direct_feature,
    train_production_direct_head_search, train_production_full_smoke,
};

pub const PRODUCTION_MODEL_V1_SCHEMA: &str = "nsrl.production_model.v1";
pub const PRODUCTION_MODEL_V1_VERSION: u32 = 1;
const PRODUCTION_RMS_EPSILON: u64 = 1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub hidden_dim: usize,
    pub context_tokens: usize,
}

impl ProductionModelConfig {
    pub fn profile(id: &str) -> Option<Self> {
        match id {
            "p10m" => Some(Self {
                vocab_size: 8_192,
                d_model: 256,
                heads: 8,
                layers: 6,
                hidden_dim: 768,
                context_tokens: 256,
            }),
            "p20m" => Some(Self {
                vocab_size: 8_192,
                d_model: 384,
                heads: 8,
                layers: 8,
                hidden_dim: 1_152,
                context_tokens: 256,
            }),
            "p30m" => Some(Self {
                vocab_size: 8_192,
                d_model: 448,
                heads: 8,
                layers: 8,
                hidden_dim: 1_344,
                context_tokens: 256,
            }),
            _ => None,
        }
    }

    pub fn profile_id(self) -> Option<&'static str> {
        ["p10m", "p20m", "p30m"]
            .into_iter()
            .find(|&id| Self::profile(id) == Some(self))
    }

    pub fn validate(self) -> Result<(), TrainError> {
        if self.vocab_size <= EOS_TOKEN_ID as usize
            || self.d_model == 0
            || self.heads == 0
            || !self.d_model.is_multiple_of(self.heads)
            || self.layers == 0
            || self.hidden_dim == 0
            || self.context_tokens == 0
            || [
                self.vocab_size,
                self.d_model,
                self.heads,
                self.layers,
                self.hidden_dim,
                self.context_tokens,
            ]
            .into_iter()
            .any(|value| u32::try_from(value).is_err())
        {
            return Err(TrainError::InvalidConfig);
        }
        numeric_contract::validate_config_numeric_bounds(self)
            .map_err(|_| TrainError::InvalidConfig)?;
        self.parameter_count().ok_or(TrainError::InvalidConfig)?;
        Ok(())
    }

    pub fn parameter_count(self) -> Option<usize> {
        let embeddings_and_output = self.vocab_size.checked_mul(self.d_model)?.checked_mul(2)?;
        let attention = self.d_model.checked_mul(self.d_model)?.checked_mul(4)?;
        let mlp = self.d_model.checked_mul(self.hidden_dim)?.checked_mul(3)?;
        let rms = self.d_model.checked_mul(2)?;
        embeddings_and_output
            .checked_add(
                self.layers
                    .checked_mul(attention.checked_add(mlp)?.checked_add(rms)?)?,
            )?
            .checked_add(self.d_model)?
            .checked_add(self.vocab_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionProjectionScales {
    pub qkv_shift: u8,
    pub o_shift: u8,
    pub up_shift: u8,
    pub gate_shift: u8,
    pub down_shift: u8,
    pub output_shift: u8,
}

impl Default for ProductionProjectionScales {
    fn default() -> Self {
        Self {
            qkv_shift: 8,
            o_shift: 8,
            up_shift: 10,
            gate_shift: 10,
            down_shift: 12,
            output_shift: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionModelV1 {
    pub config: ProductionModelConfig,
    pub tokenizer_hash: u64,
    pub initialization_seed: u64,
    pub scales: ProductionProjectionScales,
    pub embeddings: Vec<i16>,
    pub attention_rms_weights: Vec<i16>,
    pub mlp_rms_weights: Vec<i16>,
    pub final_rms_weights: Vec<i16>,
    pub q_weights: Vec<i8>,
    pub k_weights: Vec<i8>,
    pub v_weights: Vec<i8>,
    pub o_weights: Vec<i8>,
    pub up_weights: Vec<i8>,
    pub gate_weights: Vec<i8>,
    pub down_weights: Vec<i8>,
    pub output_weights: Vec<i16>,
    pub output_bias_q8: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionForward {
    pub features_q15: Vec<i16>,
    pub logits_q8: Vec<i32>,
    pub probabilities_q15: Vec<i16>,
    pub residual_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionLayerBoundaryHashes {
    pub layer: usize,
    pub attention_residual_hash: u64,
    pub layer_output_hash: u64,
    pub attention_residual_saturation_count: usize,
    pub mlp_residual_saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionForwardBranchHashes {
    pub embedding_hash: u64,
    pub layers: Vec<ProductionLayerBoundaryHashes>,
    pub final_features_hash: u64,
    pub logits_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionEvalTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub mistakes: usize,
    pub total_millibits: u64,
    pub mean_millibits: u64,
    pub residual_saturation_count: usize,
    pub model_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionCanonicalEvalTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub mistakes: usize,
    pub total_nll_millibits: u64,
    pub mean_nll_millibits: u64,
    pub zero_probability_floor_millibits: u64,
    pub zero_probability_windows: usize,
    pub residual_saturation_count: usize,
    pub model_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionComparisonTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub forward_scales: ProductionProjectionScales,
    pub source_model_hash: u64,
    pub candidate_model_hash: u64,
    pub source_mistakes: usize,
    pub candidate_mistakes: usize,
    pub source_total_millibits: u64,
    pub candidate_total_millibits: u64,
    pub total_millibits_delta: i64,
    pub feature_changed_windows: usize,
    pub feature_delta_l1: u64,
    pub logits_changed_windows: usize,
    pub logit_changed_values: usize,
    pub logit_delta_l1: u64,
    pub target_logit_changed_windows: usize,
    pub probabilities_changed_windows: usize,
    pub probability_changed_values: usize,
    pub probability_delta_l1: u64,
    pub target_probability_changed_windows: usize,
    pub prediction_changed_windows: usize,
    pub improved_loss_windows: usize,
    pub worsened_loss_windows: usize,
    pub equal_loss_windows: usize,
    pub source_residual_saturation_count: usize,
    pub candidate_residual_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionProbabilityPrecisionRow {
    pub fractional_bits: u8,
    pub uniform_probability_floor: u32,
    pub source_target_probability_min: u32,
    pub source_target_probability_max: u32,
    pub source_target_unique_values: usize,
    pub source_target_zero_windows: usize,
    pub candidate_target_probability_min: u32,
    pub candidate_target_probability_max: u32,
    pub candidate_target_unique_values: usize,
    pub candidate_target_zero_windows: usize,
    pub source_zero_probability_values: usize,
    pub candidate_zero_probability_values: usize,
    pub source_probability_mass_error_l1: u64,
    pub source_probability_mass_error_max: u64,
    pub candidate_probability_mass_error_l1: u64,
    pub candidate_probability_mass_error_max: u64,
    pub probability_changed_windows: usize,
    pub probability_changed_values: usize,
    pub probability_delta_l1: u64,
    pub target_probability_changed_windows: usize,
    pub target_probability_delta_l1: u64,
    pub source_total_microbits: u64,
    pub candidate_total_microbits: u64,
    pub total_microbits_delta: i64,
    pub improved_loss_windows: usize,
    pub worsened_loss_windows: usize,
    pub equal_loss_windows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionProbabilityResolutionTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub forward_scales: ProductionProjectionScales,
    pub source_model_hash: u64,
    pub candidate_model_hash: u64,
    pub logit_changed_windows: usize,
    pub target_logit_changed_windows: usize,
    pub q15_requantization_exact: bool,
    pub source_residual_saturation_count: usize,
    pub candidate_residual_saturation_count: usize,
    pub precision_rows: Vec<ProductionProbabilityPrecisionRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionProbabilityNormalizationRow {
    pub normalization: &'static str,
    pub reciprocal_fractional_bits: u8,
    pub probability: ProductionProbabilityPrecisionRow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionProbabilityNormalizationTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub probability_fractional_bits: u8,
    pub forward_scales: ProductionProjectionScales,
    pub source_model_hash: u64,
    pub candidate_model_hash: u64,
    pub logit_changed_windows: usize,
    pub target_logit_changed_windows: usize,
    pub source_residual_saturation_count: usize,
    pub candidate_residual_saturation_count: usize,
    pub normalization_rows: Vec<ProductionProbabilityNormalizationRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProductionNormalizationErrorTrace {
    pub probability_changed_values: usize,
    pub probability_error_l1: u64,
    pub probability_error_max: u32,
    pub target_error_windows: usize,
    pub target_error_l1: u64,
    pub target_error_max: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionNormalizationSignalMethodTrace {
    pub normalization: &'static str,
    pub reciprocal_fractional_bits: u8,
    pub target_changed_window_indices: Vec<usize>,
    pub source_error_vs_exact: ProductionNormalizationErrorTrace,
    pub candidate_error_vs_exact: ProductionNormalizationErrorTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionNormalizationTargetPair {
    pub normalization: &'static str,
    pub source_probability_q23: u32,
    pub candidate_probability_q23: u32,
    pub delta_q23: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionNormalizationSignalWindow {
    pub window_index: usize,
    pub target_token: u32,
    pub source_target_logit_q8: i32,
    pub candidate_target_logit_q8: i32,
    pub source_target_weight_q15: u16,
    pub candidate_target_weight_q15: u16,
    pub source_normalization_sum: u64,
    pub candidate_normalization_sum: u64,
    pub target_probabilities: Vec<ProductionNormalizationTargetPair>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionNormalizationSignalAttributionTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub probability_fractional_bits: u8,
    pub forward_scales: ProductionProjectionScales,
    pub source_model_hash: u64,
    pub candidate_model_hash: u64,
    pub logit_changed_windows: usize,
    pub target_logit_changed_windows: usize,
    pub source_residual_saturation_count: usize,
    pub candidate_residual_saturation_count: usize,
    pub methods: Vec<ProductionNormalizationSignalMethodTrace>,
    pub window_attributions: Vec<ProductionNormalizationSignalWindow>,
}

impl ProductionEvalTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_eval.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{},\"mistakes\":{},",
                "\"total_millibits\":{},\"mean_millibits\":{}}},",
                "\"health\":{{\"residual_saturation_count\":{}}},",
                "\"model_hash\":\"0x{:016x}\"}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.mistakes,
            self.total_millibits,
            self.mean_millibits,
            self.residual_saturation_count,
            self.model_hash,
        )
    }
}

impl ProductionCanonicalEvalTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_canonical_eval.v2\",",
                "\"objective\":\"integer_base2_softmax_nll_millibits\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{},\"mistakes\":{},",
                "\"total_nll_millibits\":{},\"mean_nll_millibits\":{},",
                "\"zero_probability_floor_millibits\":{},\"zero_probability_windows\":{}}},",
                "\"invariants\":{{\"normalization_independent\":true,\"logit_shift_invariant\":true}},",
                "\"health\":{{\"residual_saturation_count\":{}}},",
                "\"model_hash\":\"0x{:016x}\"}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.mistakes,
            self.total_nll_millibits,
            self.mean_nll_millibits,
            self.zero_probability_floor_millibits,
            self.zero_probability_windows,
            self.residual_saturation_count,
            self.model_hash,
        )
    }
}

impl ProductionComparisonTrace {
    pub fn to_json_line(self) -> String {
        format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_functional_comparison.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{}}},",
                "\"forward_shifts\":{{\"qkv\":{},\"o\":{},\"up\":{},\"gate\":{},",
                "\"down\":{},\"output\":{}}},",
                "\"models\":{{\"source_hash\":\"0x{:016x}\",\"candidate_hash\":\"0x{:016x}\"}},",
                "\"quality\":{{\"source_mistakes\":{},\"candidate_mistakes\":{},",
                "\"source_total_millibits\":{},\"candidate_total_millibits\":{},",
                "\"total_millibits_delta\":{},\"improved_loss_windows\":{},",
                "\"worsened_loss_windows\":{},\"equal_loss_windows\":{}}},",
                "\"functional_delta\":{{\"feature_changed_windows\":{},\"feature_delta_l1\":{},",
                "\"logits_changed_windows\":{},\"logit_changed_values\":{},\"logit_delta_l1\":{},",
                "\"target_logit_changed_windows\":{},\"probabilities_changed_windows\":{},",
                "\"probability_changed_values\":{},\"probability_delta_l1\":{},",
                "\"target_probability_changed_windows\":{},\"prediction_changed_windows\":{}}},",
                "\"health\":{{\"source_residual_saturation_count\":{},",
                "\"candidate_residual_saturation_count\":{}}}}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.forward_scales.qkv_shift,
            self.forward_scales.o_shift,
            self.forward_scales.up_shift,
            self.forward_scales.gate_shift,
            self.forward_scales.down_shift,
            self.forward_scales.output_shift,
            self.source_model_hash,
            self.candidate_model_hash,
            self.source_mistakes,
            self.candidate_mistakes,
            self.source_total_millibits,
            self.candidate_total_millibits,
            self.total_millibits_delta,
            self.improved_loss_windows,
            self.worsened_loss_windows,
            self.equal_loss_windows,
            self.feature_changed_windows,
            self.feature_delta_l1,
            self.logits_changed_windows,
            self.logit_changed_values,
            self.logit_delta_l1,
            self.target_logit_changed_windows,
            self.probabilities_changed_windows,
            self.probability_changed_values,
            self.probability_delta_l1,
            self.target_probability_changed_windows,
            self.prediction_changed_windows,
            self.source_residual_saturation_count,
            self.candidate_residual_saturation_count,
        )
    }
}

impl ProductionProbabilityResolutionTrace {
    pub fn to_json_line(&self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"nsrl.production_probability_resolution_audit.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{}}},",
                "\"forward_shifts\":{{\"qkv\":{},\"o\":{},\"up\":{},\"gate\":{},",
                "\"down\":{},\"output\":{}}},",
                "\"models\":{{\"source_hash\":\"0x{:016x}\",",
                "\"candidate_hash\":\"0x{:016x}\"}},",
                "\"logit_signal\":{{\"changed_windows\":{},",
                "\"target_changed_windows\":{}}},",
                "\"compatibility\":{{\"q15_requantization_exact\":{}}},",
                "\"health\":{{\"source_residual_saturation_count\":{},",
                "\"candidate_residual_saturation_count\":{}}},\"precisions\":["
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.forward_scales.qkv_shift,
            self.forward_scales.o_shift,
            self.forward_scales.up_shift,
            self.forward_scales.gate_shift,
            self.forward_scales.down_shift,
            self.forward_scales.output_shift,
            self.source_model_hash,
            self.candidate_model_hash,
            self.logit_changed_windows,
            self.target_logit_changed_windows,
            self.q15_requantization_exact,
            self.source_residual_saturation_count,
            self.candidate_residual_saturation_count,
        );
        for (index, row) in self.precision_rows.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                concat!(
                    "{{\"fractional_bits\":{},\"uniform_probability_floor\":{},",
                    "\"source_target\":{{\"min\":{},\"max\":{},\"unique_values\":{},",
                    "\"zero_windows\":{}}},",
                    "\"candidate_target\":{{\"min\":{},\"max\":{},\"unique_values\":{},",
                    "\"zero_windows\":{}}},",
                    "\"mass\":{{\"source_zero_values\":{},\"candidate_zero_values\":{},",
                    "\"source_error_l1\":{},\"source_error_max\":{},",
                    "\"candidate_error_l1\":{},\"candidate_error_max\":{}}},",
                    "\"delta\":{{\"probability_changed_windows\":{},",
                    "\"probability_changed_values\":{},\"probability_delta_l1\":{},",
                    "\"target_probability_changed_windows\":{},",
                    "\"target_probability_delta_l1\":{}}},",
                    "\"quality\":{{\"source_total_microbits\":{},",
                    "\"candidate_total_microbits\":{},\"total_microbits_delta\":{},",
                    "\"improved_loss_windows\":{},\"worsened_loss_windows\":{},",
                    "\"equal_loss_windows\":{}}}}}"
                ),
                row.fractional_bits,
                row.uniform_probability_floor,
                row.source_target_probability_min,
                row.source_target_probability_max,
                row.source_target_unique_values,
                row.source_target_zero_windows,
                row.candidate_target_probability_min,
                row.candidate_target_probability_max,
                row.candidate_target_unique_values,
                row.candidate_target_zero_windows,
                row.source_zero_probability_values,
                row.candidate_zero_probability_values,
                row.source_probability_mass_error_l1,
                row.source_probability_mass_error_max,
                row.candidate_probability_mass_error_l1,
                row.candidate_probability_mass_error_max,
                row.probability_changed_windows,
                row.probability_changed_values,
                row.probability_delta_l1,
                row.target_probability_changed_windows,
                row.target_probability_delta_l1,
                row.source_total_microbits,
                row.candidate_total_microbits,
                row.total_microbits_delta,
                row.improved_loss_windows,
                row.worsened_loss_windows,
                row.equal_loss_windows,
            )
            .expect("writing JSON to String cannot fail");
        }
        output.push_str("]}\n");
        output
    }
}

impl ProductionProbabilityNormalizationTrace {
    pub fn to_json_line(&self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"nsrl.production_probability_normalization_audit.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{},",
                "\"probability_fractional_bits\":{}}},",
                "\"forward_shifts\":{{\"qkv\":{},\"o\":{},\"up\":{},\"gate\":{},",
                "\"down\":{},\"output\":{}}},",
                "\"models\":{{\"source_hash\":\"0x{:016x}\",",
                "\"candidate_hash\":\"0x{:016x}\"}},",
                "\"logit_signal\":{{\"changed_windows\":{},",
                "\"target_changed_windows\":{}}},",
                "\"health\":{{\"source_residual_saturation_count\":{},",
                "\"candidate_residual_saturation_count\":{}}},\"normalizations\":["
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.probability_fractional_bits,
            self.forward_scales.qkv_shift,
            self.forward_scales.o_shift,
            self.forward_scales.up_shift,
            self.forward_scales.gate_shift,
            self.forward_scales.down_shift,
            self.forward_scales.output_shift,
            self.source_model_hash,
            self.candidate_model_hash,
            self.logit_changed_windows,
            self.target_logit_changed_windows,
            self.source_residual_saturation_count,
            self.candidate_residual_saturation_count,
        );
        for (index, row) in self.normalization_rows.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            let probability = row.probability;
            write!(
                output,
                concat!(
                    "{{\"normalization\":\"{}\",\"reciprocal_fractional_bits\":{},",
                    "\"uniform_probability_floor\":{},",
                    "\"source_target\":{{\"min\":{},\"max\":{},\"unique_values\":{},",
                    "\"zero_windows\":{}}},",
                    "\"candidate_target\":{{\"min\":{},\"max\":{},\"unique_values\":{},",
                    "\"zero_windows\":{}}},",
                    "\"mass\":{{\"source_zero_values\":{},\"candidate_zero_values\":{},",
                    "\"source_error_l1\":{},\"source_error_max\":{},",
                    "\"candidate_error_l1\":{},\"candidate_error_max\":{}}},",
                    "\"delta\":{{\"probability_changed_windows\":{},",
                    "\"probability_changed_values\":{},\"probability_delta_l1\":{},",
                    "\"target_probability_changed_windows\":{},",
                    "\"target_probability_delta_l1\":{}}},",
                    "\"quality\":{{\"source_total_microbits\":{},",
                    "\"candidate_total_microbits\":{},\"total_microbits_delta\":{},",
                    "\"improved_loss_windows\":{},\"worsened_loss_windows\":{},",
                    "\"equal_loss_windows\":{}}}}}"
                ),
                row.normalization,
                row.reciprocal_fractional_bits,
                probability.uniform_probability_floor,
                probability.source_target_probability_min,
                probability.source_target_probability_max,
                probability.source_target_unique_values,
                probability.source_target_zero_windows,
                probability.candidate_target_probability_min,
                probability.candidate_target_probability_max,
                probability.candidate_target_unique_values,
                probability.candidate_target_zero_windows,
                probability.source_zero_probability_values,
                probability.candidate_zero_probability_values,
                probability.source_probability_mass_error_l1,
                probability.source_probability_mass_error_max,
                probability.candidate_probability_mass_error_l1,
                probability.candidate_probability_mass_error_max,
                probability.probability_changed_windows,
                probability.probability_changed_values,
                probability.probability_delta_l1,
                probability.target_probability_changed_windows,
                probability.target_probability_delta_l1,
                probability.source_total_microbits,
                probability.candidate_total_microbits,
                probability.total_microbits_delta,
                probability.improved_loss_windows,
                probability.worsened_loss_windows,
                probability.equal_loss_windows,
            )
            .expect("writing JSON to String cannot fail");
        }
        output.push_str("]}\n");
        output
    }
}

fn write_normalization_error_json(output: &mut String, error: ProductionNormalizationErrorTrace) {
    write!(
        output,
        concat!(
            "{{\"probability_changed_values\":{},\"probability_error_l1\":{},",
            "\"probability_error_max\":{},\"target_error_windows\":{},",
            "\"target_error_l1\":{},\"target_error_max\":{}}}"
        ),
        error.probability_changed_values,
        error.probability_error_l1,
        error.probability_error_max,
        error.target_error_windows,
        error.target_error_l1,
        error.target_error_max,
    )
    .expect("writing JSON to String cannot fail");
}

impl ProductionNormalizationSignalAttributionTrace {
    pub fn to_json_line(&self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"nsrl.production_probability_normalization_signal_attribution.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",",
                "\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"evaluation\":{{\"context_tokens\":{},\"windows\":{},",
                "\"probability_fractional_bits\":{}}},",
                "\"forward_shifts\":{{\"qkv\":{},\"o\":{},\"up\":{},\"gate\":{},",
                "\"down\":{},\"output\":{}}},",
                "\"models\":{{\"source_hash\":\"0x{:016x}\",",
                "\"candidate_hash\":\"0x{:016x}\"}},",
                "\"logit_signal\":{{\"changed_windows\":{},",
                "\"target_changed_windows\":{}}},",
                "\"health\":{{\"source_residual_saturation_count\":{},",
                "\"candidate_residual_saturation_count\":{}}},\"methods\":["
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.probability_fractional_bits,
            self.forward_scales.qkv_shift,
            self.forward_scales.o_shift,
            self.forward_scales.up_shift,
            self.forward_scales.gate_shift,
            self.forward_scales.down_shift,
            self.forward_scales.output_shift,
            self.source_model_hash,
            self.candidate_model_hash,
            self.logit_changed_windows,
            self.target_logit_changed_windows,
            self.source_residual_saturation_count,
            self.candidate_residual_saturation_count,
        );
        for (index, method) in self.methods.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                "{{\"normalization\":\"{}\",\"reciprocal_fractional_bits\":{},\"target_changed_window_indices\":[",
                method.normalization, method.reciprocal_fractional_bits,
            )
            .expect("writing JSON to String cannot fail");
            for (window_index, changed_window) in
                method.target_changed_window_indices.iter().enumerate()
            {
                if window_index > 0 {
                    output.push(',');
                }
                write!(output, "{changed_window}").expect("writing JSON to String cannot fail");
            }
            output.push_str("],\"source_error_vs_exact\":");
            write_normalization_error_json(&mut output, method.source_error_vs_exact);
            output.push_str(",\"candidate_error_vs_exact\":");
            write_normalization_error_json(&mut output, method.candidate_error_vs_exact);
            output.push('}');
        }
        output.push_str("],\"window_attributions\":[");
        for (index, window) in self.window_attributions.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            write!(
                output,
                concat!(
                    "{{\"window_index\":{},\"target_token\":{},",
                    "\"target_logit_q8\":{{\"source\":{},\"candidate\":{},\"changed\":{}}},",
                    "\"target_weight_q15\":{{\"source\":{},\"candidate\":{},\"changed\":{}}},",
                    "\"normalization_sum\":{{\"source\":{},\"candidate\":{},\"changed\":{}}},",
                    "\"target_probabilities_q23\":["
                ),
                window.window_index,
                window.target_token,
                window.source_target_logit_q8,
                window.candidate_target_logit_q8,
                window.source_target_logit_q8 != window.candidate_target_logit_q8,
                window.source_target_weight_q15,
                window.candidate_target_weight_q15,
                window.source_target_weight_q15 != window.candidate_target_weight_q15,
                window.source_normalization_sum,
                window.candidate_normalization_sum,
                window.source_normalization_sum != window.candidate_normalization_sum,
            )
            .expect("writing JSON to String cannot fail");
            for (pair_index, pair) in window.target_probabilities.iter().enumerate() {
                if pair_index > 0 {
                    output.push(',');
                }
                write!(
                    output,
                    concat!(
                        "{{\"normalization\":\"{}\",\"source\":{},",
                        "\"candidate\":{},\"delta\":{}}}"
                    ),
                    pair.normalization,
                    pair.source_probability_q23,
                    pair.candidate_probability_q23,
                    pair.delta_q23,
                )
                .expect("writing JSON to String cannot fail");
            }
            output.push_str("]}");
        }
        output.push_str("]}\n");
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionSmokeConfig {
    pub context_tokens: usize,
    pub max_windows: usize,
    pub epochs: usize,
    pub feature_shift: u8,
    pub bias_step_q8: i32,
    pub margin_q8: i32,
    pub spread_windows: bool,
}

impl Default for ProductionSmokeConfig {
    fn default() -> Self {
        Self {
            context_tokens: 4,
            max_windows: 8,
            epochs: 2,
            feature_shift: 13,
            bias_step_q8: 4,
            margin_q8: 8,
            spread_windows: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProductionSmokeTrace {
    pub profile: &'static str,
    pub parameter_count: usize,
    pub tokenizer_hash: u64,
    pub token_stream_hash: u64,
    pub context_tokens: usize,
    pub windows: usize,
    pub epochs: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub updates: usize,
    pub weight_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub spread_windows: bool,
}

impl ProductionSmokeTrace {
    pub fn to_json_line(self) -> String {
        let mut output = format!(
            concat!(
                "{{\"schema\":\"nsrl.production_model_smoke.v1\",",
                "\"profile\":\"{}\",\"parameter_count\":{},",
                "\"bindings\":{{\"tokenizer_hash\":\"0x{:016x}\",\"token_stream_hash\":\"0x{:016x}\"}},",
                "\"training\":{{\"scope\":\"output_head_perceptron\",\"context_tokens\":{},\"windows\":{},\"epochs\":{},\"updates\":{}}},",
                "\"evaluation\":{{\"initial_mistakes\":{},\"final_mistakes\":{}}},",
                "\"health\":{{\"weight_saturation_count\":{},\"residual_saturation_count\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\"}},",
                "\"known_non_claims\":[\"output_head_only_smoke_not_full_layer_training\",\"not_float_twin\",\"not_open_generation_quality\"]}}\n"
            ),
            self.profile,
            self.parameter_count,
            self.tokenizer_hash,
            self.token_stream_hash,
            self.context_tokens,
            self.windows,
            self.epochs,
            self.updates,
            self.initial_mistakes,
            self.final_mistakes,
            self.weight_saturation_count,
            self.residual_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
        );
        if self.spread_windows {
            output = output.replace(
                ",\"known_non_claims\"",
                ",\"window_selection\":\"deterministic_uniform_target_rank_over_all_documents\",\"known_non_claims\"",
            );
        }
        output
    }
}

impl ProductionModelV1 {
    pub fn new_initial(
        config: ProductionModelConfig,
        tokenizer_hash: u64,
        initialization_seed: u64,
    ) -> Result<Self, TrainError> {
        config.validate()?;
        if tokenizer_hash == 0 {
            return Err(TrainError::InvalidConfig);
        }
        let matrix = checked_product(config.d_model, config.d_model)?;
        let up = checked_product(config.d_model, config.hidden_dim)?;
        let down = checked_product(config.hidden_dim, config.d_model)?;
        let embeddings = initial_i16_tensor(
            checked_product(config.vocab_size, config.d_model)?,
            initialization_seed ^ 0x6a09_e667_f3bc_c909,
            512,
        );
        let model = Self {
            config,
            tokenizer_hash,
            initialization_seed,
            scales: ProductionProjectionScales::default(),
            embeddings,
            attention_rms_weights: vec![30_000; checked_product(config.layers, config.d_model)?],
            mlp_rms_weights: vec![30_000; checked_product(config.layers, config.d_model)?],
            final_rms_weights: positive_initial_i16(
                config.d_model,
                initialization_seed ^ 0xc0ff_ee12_3456_789a,
                1000,
                31000,
            ),
            q_weights: stacked_identity(config.layers, config.d_model, 16),
            k_weights: stacked_identity(config.layers, config.d_model, 16),
            v_weights: stacked_identity(config.layers, config.d_model, 16),
            o_weights: stacked_identity(config.layers, config.d_model, 8),
            up_weights: initial_i8_tensor(
                checked_product(config.layers, up)?,
                initialization_seed ^ 0x510e_527f_ade6_82d1,
                2,
            ),
            gate_weights: initial_i8_tensor(
                checked_product(config.layers, up)?,
                initialization_seed ^ 0x9b05_688c_2b3e_6c1f,
                2,
            ),
            down_weights: initial_i8_tensor(
                checked_product(config.layers, down)?,
                initialization_seed ^ 0x1f83_d9ab_fb41_bd6b,
                1,
            ),
            output_weights: vec![0_i16; checked_product(config.vocab_size, config.d_model)?],
            output_bias_q8: vec![0_i32; config.vocab_size],
        };
        debug_assert_eq!(model.q_weights.len(), config.layers * matrix);
        model.validate()?;
        Ok(model)
    }

    pub fn initialize_output_weights(&mut self, amplitude: i16) -> Result<(), TrainError> {
        if amplitude < 0 {
            return Err(TrainError::InvalidConfig);
        }
        self.output_weights = initial_i16_tensor(
            checked_product(self.config.vocab_size, self.config.d_model)?,
            self.initialization_seed ^ 0x3c6e_f372_fe94_f82b,
            amplitude,
        );
        self.validate()
    }

    pub fn validate(&self) -> Result<(), TrainError> {
        self.config.validate()?;
        if self.tokenizer_hash == 0 || scale_shifts(self.scales).iter().any(|&shift| shift > 30) {
            return Err(TrainError::InvalidModel(
                "invalid production model metadata",
            ));
        }
        ProductionNumericContract::derive(self.config, self.scales)
            .map_err(TrainError::InvalidModel)?;
        let config = self.config;
        let matrix = checked_product(config.d_model, config.d_model)?;
        let rms = checked_product(config.layers, config.d_model)?;
        let up = checked_product(
            config.layers,
            checked_product(config.d_model, config.hidden_dim)?,
        )?;
        let down = checked_product(
            config.layers,
            checked_product(config.hidden_dim, config.d_model)?,
        )?;
        if self.embeddings.len() != checked_product(config.vocab_size, config.d_model)?
            || self.attention_rms_weights.len() != rms
            || self.mlp_rms_weights.len() != rms
            || self.final_rms_weights.len() != config.d_model
            || self.q_weights.len() != checked_product(config.layers, matrix)?
            || self.k_weights.len() != self.q_weights.len()
            || self.v_weights.len() != self.q_weights.len()
            || self.o_weights.len() != self.q_weights.len()
            || self.up_weights.len() != up
            || self.gate_weights.len() != up
            || self.down_weights.len() != down
            || self.output_weights.len() != checked_product(config.vocab_size, config.d_model)?
            || self.output_bias_q8.len() != config.vocab_size
        {
            return Err(TrainError::InvalidModel("invalid production model shape"));
        }
        let actual_parameters = self
            .embeddings
            .len()
            .checked_add(self.attention_rms_weights.len())
            .and_then(|value| value.checked_add(self.mlp_rms_weights.len()))
            .and_then(|value| value.checked_add(self.final_rms_weights.len()))
            .and_then(|value| value.checked_add(self.q_weights.len()))
            .and_then(|value| value.checked_add(self.k_weights.len()))
            .and_then(|value| value.checked_add(self.v_weights.len()))
            .and_then(|value| value.checked_add(self.o_weights.len()))
            .and_then(|value| value.checked_add(self.up_weights.len()))
            .and_then(|value| value.checked_add(self.gate_weights.len()))
            .and_then(|value| value.checked_add(self.down_weights.len()))
            .and_then(|value| value.checked_add(self.output_weights.len()))
            .and_then(|value| value.checked_add(self.output_bias_q8.len()))
            .ok_or(TrainError::InvalidModel(
                "production parameter count overflow",
            ))?;
        if Some(actual_parameters) != config.parameter_count() {
            return Err(TrainError::InvalidModel(
                "production parameter count mismatch",
            ));
        }
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.config.parameter_count().unwrap_or(usize::MAX)
    }

    pub fn numeric_contract(&self) -> Result<ProductionNumericContract, TrainError> {
        ProductionNumericContract::derive(self.config, self.scales)
            .map_err(TrainError::InvalidModel)
    }

    pub fn model_hash(&self) -> u64 {
        fnv1a(&self.bytes_without_checksum())
    }

    pub fn try_to_bytes(&self) -> Result<Vec<u8>, TrainError> {
        self.validate()?;
        let mut bytes = self.bytes_without_checksum();
        let checksum = fnv1a(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TrainError> {
        if bytes.len() < 8 + 4 + 6 * 4 + 8 + 8 + 6 + 8 || &bytes[..8] != PRODUCTION_MODEL_V1_MAGIC {
            return Err(TrainError::InvalidModel("bad NSRLPM1 artifact magic"));
        }
        let checksum_offset = bytes.len() - 8;
        let expected_checksum = u64::from_le_bytes(
            bytes[checksum_offset..]
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad NSRLPM1 checksum"))?,
        );
        if fnv1a(&bytes[..checksum_offset]) != expected_checksum {
            return Err(TrainError::InvalidModel("bad NSRLPM1 checksum"));
        }
        let mut offset = 8;
        if read_u32(bytes, &mut offset)? != PRODUCTION_MODEL_V1_VERSION {
            return Err(TrainError::InvalidModel("unsupported NSRLPM1 version"));
        }
        let config = ProductionModelConfig {
            vocab_size: read_u32(bytes, &mut offset)? as usize,
            d_model: read_u32(bytes, &mut offset)? as usize,
            heads: read_u32(bytes, &mut offset)? as usize,
            layers: read_u32(bytes, &mut offset)? as usize,
            hidden_dim: read_u32(bytes, &mut offset)? as usize,
            context_tokens: read_u32(bytes, &mut offset)? as usize,
        };
        config.validate()?;
        let tokenizer_hash = read_u64(bytes, &mut offset)?;
        let initialization_seed = read_u64(bytes, &mut offset)?;
        let shifts = take(bytes, &mut offset, 6)?;
        let scales = ProductionProjectionScales {
            qkv_shift: shifts[0],
            o_shift: shifts[1],
            up_shift: shifts[2],
            gate_shift: shifts[3],
            down_shift: shifts[4],
            output_shift: shifts[5],
        };
        let matrix = checked_product(config.d_model, config.d_model)?;
        let rms = checked_product(config.layers, config.d_model)?;
        let up = checked_product(
            config.layers,
            checked_product(config.d_model, config.hidden_dim)?,
        )?;
        let down = checked_product(
            config.layers,
            checked_product(config.hidden_dim, config.d_model)?,
        )?;
        let model = Self {
            config,
            tokenizer_hash,
            initialization_seed,
            scales,
            embeddings: read_i16(
                bytes,
                &mut offset,
                checked_product(config.vocab_size, config.d_model)?,
            )?,
            attention_rms_weights: read_i16(bytes, &mut offset, rms)?,
            mlp_rms_weights: read_i16(bytes, &mut offset, rms)?,
            final_rms_weights: read_i16(bytes, &mut offset, config.d_model)?,
            q_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            k_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            v_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            o_weights: read_i8(bytes, &mut offset, checked_product(config.layers, matrix)?)?,
            up_weights: read_i8(bytes, &mut offset, up)?,
            gate_weights: read_i8(bytes, &mut offset, up)?,
            down_weights: read_i8(bytes, &mut offset, down)?,
            output_weights: read_i16(
                bytes,
                &mut offset,
                checked_product(config.vocab_size, config.d_model)?,
            )?,
            output_bias_q8: read_i32(bytes, &mut offset, config.vocab_size)?,
        };
        if offset != checksum_offset {
            return Err(TrainError::InvalidModel("wrong NSRLPM1 artifact length"));
        }
        model.validate()?;
        Ok(model)
    }

    fn bytes_without_checksum(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PRODUCTION_MODEL_V1_MAGIC);
        bytes.extend_from_slice(&PRODUCTION_MODEL_V1_VERSION.to_le_bytes());
        for value in [
            self.config.vocab_size,
            self.config.d_model,
            self.config.heads,
            self.config.layers,
            self.config.hidden_dim,
            self.config.context_tokens,
        ] {
            bytes.extend_from_slice(&(value as u32).to_le_bytes());
        }
        bytes.extend_from_slice(&self.tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&self.initialization_seed.to_le_bytes());
        bytes.extend_from_slice(&scale_shifts(self.scales));
        extend_i16(&mut bytes, &self.embeddings);
        extend_i16(&mut bytes, &self.attention_rms_weights);
        extend_i16(&mut bytes, &self.mlp_rms_weights);
        extend_i16(&mut bytes, &self.final_rms_weights);
        extend_i8(&mut bytes, &self.q_weights);
        extend_i8(&mut bytes, &self.k_weights);
        extend_i8(&mut bytes, &self.v_weights);
        extend_i8(&mut bytes, &self.o_weights);
        extend_i8(&mut bytes, &self.up_weights);
        extend_i8(&mut bytes, &self.gate_weights);
        extend_i8(&mut bytes, &self.down_weights);
        extend_i16(&mut bytes, &self.output_weights);
        for value in &self.output_bias_q8 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

pub fn forward_production_model(
    model: &ProductionModelV1,
    context: &[u32],
) -> Result<ProductionForward, TrainError> {
    let (features_q15, residual_saturation_count) = production_features(model, context)?;
    let logits_q8 = output_logits(model, &features_q15)?;
    let mut probabilities_q15 = vec![0_i16; model.config.vocab_size];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15)
        .ok_or(TrainError::CoreRejected("production_softmax"))?;
    Ok(ProductionForward {
        features_q15,
        logits_q8,
        probabilities_q15,
        residual_saturation_count,
    })
}

/// Executes the exact production forward path while recording hashes at the
/// residual branch boundaries. The returned hashes are diagnostic only and do
/// not participate in arithmetic or alter the deployed forward result.
pub fn forward_production_model_branch_hashes(
    model: &ProductionModelV1,
    context: &[u32],
) -> Result<ProductionForwardBranchHashes, TrainError> {
    let mut embedding_hash = 0_u64;
    let mut layers = Vec::with_capacity(model.config.layers);
    let (features, _) =
        production_features_observed(model, context, Some((&mut embedding_hash, &mut layers)))?;
    let logits = output_logits(model, &features)?;
    Ok(ProductionForwardBranchHashes {
        embedding_hash,
        layers,
        final_features_hash: hash_i16_slice(&features),
        logits_hash: hash_i32_slice(&logits),
    })
}

pub fn evaluate_production_model(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionEvalTrace, TrainError> {
    model.validate()?;
    if context_tokens == 0
        || context_tokens > model.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut mistakes = 0_usize;
    let mut total_millibits = 0_u64;
    let mut residual_saturation_count = 0_usize;
    for (context, target) in &windows {
        let output = forward_production_model(model, context)?;
        let predicted = argmax(&output.logits_q8);
        mistakes = mistakes.saturating_add(usize::from(predicted != *target as usize));
        total_millibits = total_millibits.saturating_add(q15_negative_log2_millibits(
            output.probabilities_q15[*target as usize],
        ));
        residual_saturation_count =
            residual_saturation_count.saturating_add(output.residual_saturation_count);
    }
    let count = windows.len() as u64;
    Ok(ProductionEvalTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        mistakes,
        total_millibits,
        mean_millibits: total_millibits.saturating_add(count / 2) / count,
        residual_saturation_count,
        model_hash: model.model_hash(),
    })
}

pub fn evaluate_production_model_canonical_nll(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
    zero_probability_floor_millibits: u64,
) -> Result<ProductionCanonicalEvalTrace, TrainError> {
    model.validate()?;
    if context_tokens == 0
        || context_tokens > model.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut mistakes = 0_usize;
    let mut total_nll_millibits = 0_u64;
    let mut zero_probability_windows = 0_usize;
    let mut residual_saturation_count = 0_usize;
    for (context, target) in &windows {
        let output = forward_production_model(model, context)?;
        let target = *target as usize;
        let predicted = argmax(&output.logits_q8);
        mistakes = mistakes.saturating_add(usize::from(predicted != target));
        let max_logit = output
            .logits_q8
            .iter()
            .copied()
            .max()
            .ok_or(TrainError::CoreRejected("production_canonical_nll"))?;
        let target_weight = base2_exp_neg_q15(output.logits_q8[target].saturating_sub(max_logit));
        zero_probability_windows = zero_probability_windows
            .checked_add(usize::from(target_weight == 0))
            .ok_or(TrainError::CoreRejected(
                "production_canonical_eval_counter_overflow",
            ))?;
        let window_nll_millibits = base2_softmax_nll_millibits(
            &output.logits_q8,
            target,
            zero_probability_floor_millibits,
        )
        .ok_or(TrainError::CoreRejected("production_canonical_nll"))?;
        total_nll_millibits = total_nll_millibits
            .checked_add(window_nll_millibits)
            .ok_or(TrainError::CoreRejected(
                "production_canonical_eval_nll_overflow",
            ))?;
        residual_saturation_count = residual_saturation_count
            .checked_add(output.residual_saturation_count)
            .ok_or(TrainError::CoreRejected(
                "production_canonical_eval_counter_overflow",
            ))?;
    }
    let count = windows.len() as u64;
    Ok(ProductionCanonicalEvalTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        mistakes,
        total_nll_millibits,
        mean_nll_millibits: total_nll_millibits.checked_add(count / 2).ok_or(
            TrainError::CoreRejected("production_canonical_eval_nll_overflow"),
        )? / count,
        zero_probability_floor_millibits,
        zero_probability_windows,
        residual_saturation_count,
        model_hash: model.model_hash(),
    })
}

pub fn evaluate_production_model_canonical_nll_default_floor(
    model: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionCanonicalEvalTrace, TrainError> {
    evaluate_production_model_canonical_nll(
        model,
        tokens,
        token_stream_hash,
        context_tokens,
        max_windows,
        DEFAULT_ZERO_PROBABILITY_NLL_MILLIBITS,
    )
}

pub fn compare_production_models(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionComparisonTrace, TrainError> {
    source.validate()?;
    candidate.validate()?;
    if source.config != candidate.config
        || source.scales != candidate.scales
        || source.tokenizer_hash != candidate.tokenizer_hash
        || context_tokens == 0
        || context_tokens > source.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= source.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let mut trace = ProductionComparisonTrace {
        profile: source.config.profile_id().unwrap_or("custom"),
        parameter_count: source.parameter_count(),
        tokenizer_hash: source.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        forward_scales: source.scales,
        source_model_hash: source.model_hash(),
        candidate_model_hash: candidate.model_hash(),
        source_mistakes: 0,
        candidate_mistakes: 0,
        source_total_millibits: 0,
        candidate_total_millibits: 0,
        total_millibits_delta: 0,
        feature_changed_windows: 0,
        feature_delta_l1: 0,
        logits_changed_windows: 0,
        logit_changed_values: 0,
        logit_delta_l1: 0,
        target_logit_changed_windows: 0,
        probabilities_changed_windows: 0,
        probability_changed_values: 0,
        probability_delta_l1: 0,
        target_probability_changed_windows: 0,
        prediction_changed_windows: 0,
        improved_loss_windows: 0,
        worsened_loss_windows: 0,
        equal_loss_windows: 0,
        source_residual_saturation_count: 0,
        candidate_residual_saturation_count: 0,
    };
    for (context, target) in &windows {
        let source_output = forward_production_model(source, context)?;
        let candidate_output = forward_production_model(candidate, context)?;
        let target = *target as usize;
        let source_prediction = argmax(&source_output.logits_q8);
        let candidate_prediction = argmax(&candidate_output.logits_q8);
        trace.source_mistakes = trace
            .source_mistakes
            .saturating_add(usize::from(source_prediction != target));
        trace.candidate_mistakes = trace
            .candidate_mistakes
            .saturating_add(usize::from(candidate_prediction != target));
        trace.prediction_changed_windows = trace
            .prediction_changed_windows
            .saturating_add(usize::from(source_prediction != candidate_prediction));

        let source_loss = q15_negative_log2_millibits(source_output.probabilities_q15[target]);
        let candidate_loss =
            q15_negative_log2_millibits(candidate_output.probabilities_q15[target]);
        trace.source_total_millibits = trace.source_total_millibits.saturating_add(source_loss);
        trace.candidate_total_millibits = trace
            .candidate_total_millibits
            .saturating_add(candidate_loss);
        if candidate_loss < source_loss {
            trace.improved_loss_windows = trace.improved_loss_windows.saturating_add(1);
        } else if candidate_loss > source_loss {
            trace.worsened_loss_windows = trace.worsened_loss_windows.saturating_add(1);
        } else {
            trace.equal_loss_windows = trace.equal_loss_windows.saturating_add(1);
        }

        if source_output.features_q15 != candidate_output.features_q15 {
            trace.feature_changed_windows = trace.feature_changed_windows.saturating_add(1);
        }
        for (&left, &right) in source_output
            .features_q15
            .iter()
            .zip(&candidate_output.features_q15)
        {
            trace.feature_delta_l1 = trace
                .feature_delta_l1
                .saturating_add(u64::from(left.abs_diff(right)));
        }
        if source_output.logits_q8 != candidate_output.logits_q8 {
            trace.logits_changed_windows = trace.logits_changed_windows.saturating_add(1);
        }
        for (&left, &right) in source_output
            .logits_q8
            .iter()
            .zip(&candidate_output.logits_q8)
        {
            if left != right {
                trace.logit_changed_values = trace.logit_changed_values.saturating_add(1);
                trace.logit_delta_l1 = trace
                    .logit_delta_l1
                    .saturating_add(u64::from(left.abs_diff(right)));
            }
        }
        trace.target_logit_changed_windows =
            trace
                .target_logit_changed_windows
                .saturating_add(usize::from(
                    source_output.logits_q8[target] != candidate_output.logits_q8[target],
                ));
        if source_output.probabilities_q15 != candidate_output.probabilities_q15 {
            trace.probabilities_changed_windows =
                trace.probabilities_changed_windows.saturating_add(1);
        }
        for (&left, &right) in source_output
            .probabilities_q15
            .iter()
            .zip(&candidate_output.probabilities_q15)
        {
            if left != right {
                trace.probability_changed_values =
                    trace.probability_changed_values.saturating_add(1);
                trace.probability_delta_l1 = trace
                    .probability_delta_l1
                    .saturating_add(u64::from(left.abs_diff(right)));
            }
        }
        trace.target_probability_changed_windows = trace
            .target_probability_changed_windows
            .saturating_add(usize::from(
                source_output.probabilities_q15[target]
                    != candidate_output.probabilities_q15[target],
            ));
        trace.source_residual_saturation_count = trace
            .source_residual_saturation_count
            .saturating_add(source_output.residual_saturation_count);
        trace.candidate_residual_saturation_count = trace
            .candidate_residual_saturation_count
            .saturating_add(candidate_output.residual_saturation_count);
    }
    trace.total_millibits_delta = if trace.candidate_total_millibits >= trace.source_total_millibits
    {
        i64::try_from(trace.candidate_total_millibits - trace.source_total_millibits)
            .unwrap_or(i64::MAX)
    } else {
        -i64::try_from(trace.source_total_millibits - trace.candidate_total_millibits)
            .unwrap_or(i64::MAX)
    };
    Ok(trace)
}

#[derive(Debug)]
struct ProbabilityPrecisionAccumulator {
    fractional_bits: u8,
    source_target_values: BTreeSet<u32>,
    candidate_target_values: BTreeSet<u32>,
    source_target_probability_min: u32,
    source_target_probability_max: u32,
    source_target_zero_windows: usize,
    candidate_target_probability_min: u32,
    candidate_target_probability_max: u32,
    candidate_target_zero_windows: usize,
    source_zero_probability_values: usize,
    candidate_zero_probability_values: usize,
    source_probability_mass_error_l1: u64,
    source_probability_mass_error_max: u64,
    candidate_probability_mass_error_l1: u64,
    candidate_probability_mass_error_max: u64,
    probability_changed_windows: usize,
    probability_changed_values: usize,
    probability_delta_l1: u64,
    target_probability_changed_windows: usize,
    target_probability_delta_l1: u64,
    source_total_microbits: u64,
    candidate_total_microbits: u64,
    improved_loss_windows: usize,
    worsened_loss_windows: usize,
    equal_loss_windows: usize,
}

impl ProbabilityPrecisionAccumulator {
    fn new(fractional_bits: u8) -> Self {
        Self {
            fractional_bits,
            source_target_values: BTreeSet::new(),
            candidate_target_values: BTreeSet::new(),
            source_target_probability_min: u32::MAX,
            source_target_probability_max: 0,
            source_target_zero_windows: 0,
            candidate_target_probability_min: u32::MAX,
            candidate_target_probability_max: 0,
            candidate_target_zero_windows: 0,
            source_zero_probability_values: 0,
            candidate_zero_probability_values: 0,
            source_probability_mass_error_l1: 0,
            source_probability_mass_error_max: 0,
            candidate_probability_mass_error_l1: 0,
            candidate_probability_mass_error_max: 0,
            probability_changed_windows: 0,
            probability_changed_values: 0,
            probability_delta_l1: 0,
            target_probability_changed_windows: 0,
            target_probability_delta_l1: 0,
            source_total_microbits: 0,
            candidate_total_microbits: 0,
            improved_loss_windows: 0,
            worsened_loss_windows: 0,
            equal_loss_windows: 0,
        }
    }

    fn finish(self, vocab_size: usize) -> ProductionProbabilityPrecisionRow {
        ProductionProbabilityPrecisionRow {
            fractional_bits: self.fractional_bits,
            uniform_probability_floor: ((1_u64 << self.fractional_bits) / vocab_size as u64) as u32,
            source_target_probability_min: self.source_target_probability_min,
            source_target_probability_max: self.source_target_probability_max,
            source_target_unique_values: self.source_target_values.len(),
            source_target_zero_windows: self.source_target_zero_windows,
            candidate_target_probability_min: self.candidate_target_probability_min,
            candidate_target_probability_max: self.candidate_target_probability_max,
            candidate_target_unique_values: self.candidate_target_values.len(),
            candidate_target_zero_windows: self.candidate_target_zero_windows,
            source_zero_probability_values: self.source_zero_probability_values,
            candidate_zero_probability_values: self.candidate_zero_probability_values,
            source_probability_mass_error_l1: self.source_probability_mass_error_l1,
            source_probability_mass_error_max: self.source_probability_mass_error_max,
            candidate_probability_mass_error_l1: self.candidate_probability_mass_error_l1,
            candidate_probability_mass_error_max: self.candidate_probability_mass_error_max,
            probability_changed_windows: self.probability_changed_windows,
            probability_changed_values: self.probability_changed_values,
            probability_delta_l1: self.probability_delta_l1,
            target_probability_changed_windows: self.target_probability_changed_windows,
            target_probability_delta_l1: self.target_probability_delta_l1,
            source_total_microbits: self.source_total_microbits,
            candidate_total_microbits: self.candidate_total_microbits,
            total_microbits_delta: signed_delta(
                self.candidate_total_microbits,
                self.source_total_microbits,
            ),
            improved_loss_windows: self.improved_loss_windows,
            worsened_loss_windows: self.worsened_loss_windows,
            equal_loss_windows: self.equal_loss_windows,
        }
    }

    fn observe(&mut self, source_q31: &[u32], candidate_q31: &[u32], target: usize) {
        debug_assert_eq!(source_q31.len(), candidate_q31.len());
        debug_assert!(target < source_q31.len());
        let bits = self.fractional_bits;
        let scale = 1_u64 << bits;
        let mut source_mass = 0_u64;
        let mut candidate_mass = 0_u64;
        let mut vector_changed = false;
        for (&source_wide, &candidate_wide) in source_q31.iter().zip(candidate_q31) {
            let source_probability = quantize_probability_q31(source_wide, bits);
            let candidate_probability = quantize_probability_q31(candidate_wide, bits);
            source_mass = source_mass.saturating_add(u64::from(source_probability));
            candidate_mass = candidate_mass.saturating_add(u64::from(candidate_probability));
            self.source_zero_probability_values = self
                .source_zero_probability_values
                .saturating_add(usize::from(source_probability == 0));
            self.candidate_zero_probability_values = self
                .candidate_zero_probability_values
                .saturating_add(usize::from(candidate_probability == 0));
            if source_probability != candidate_probability {
                vector_changed = true;
                self.probability_changed_values = self.probability_changed_values.saturating_add(1);
                self.probability_delta_l1 = self.probability_delta_l1.saturating_add(u64::from(
                    source_probability.abs_diff(candidate_probability),
                ));
            }
        }
        self.probability_changed_windows = self
            .probability_changed_windows
            .saturating_add(usize::from(vector_changed));
        let source_mass_error = source_mass.abs_diff(scale);
        let candidate_mass_error = candidate_mass.abs_diff(scale);
        self.source_probability_mass_error_l1 = self
            .source_probability_mass_error_l1
            .saturating_add(source_mass_error);
        self.source_probability_mass_error_max = self
            .source_probability_mass_error_max
            .max(source_mass_error);
        self.candidate_probability_mass_error_l1 = self
            .candidate_probability_mass_error_l1
            .saturating_add(candidate_mass_error);
        self.candidate_probability_mass_error_max = self
            .candidate_probability_mass_error_max
            .max(candidate_mass_error);

        let source_target = quantize_probability_q31(source_q31[target], bits);
        let candidate_target = quantize_probability_q31(candidate_q31[target], bits);
        self.source_target_values.insert(source_target);
        self.candidate_target_values.insert(candidate_target);
        self.source_target_probability_min = self.source_target_probability_min.min(source_target);
        self.source_target_probability_max = self.source_target_probability_max.max(source_target);
        self.candidate_target_probability_min =
            self.candidate_target_probability_min.min(candidate_target);
        self.candidate_target_probability_max =
            self.candidate_target_probability_max.max(candidate_target);
        self.source_target_zero_windows = self
            .source_target_zero_windows
            .saturating_add(usize::from(source_target == 0));
        self.candidate_target_zero_windows = self
            .candidate_target_zero_windows
            .saturating_add(usize::from(candidate_target == 0));
        self.target_probability_changed_windows = self
            .target_probability_changed_windows
            .saturating_add(usize::from(source_target != candidate_target));
        self.target_probability_delta_l1 = self
            .target_probability_delta_l1
            .saturating_add(u64::from(source_target.abs_diff(candidate_target)));

        let source_loss = negative_log2_microbits(source_target, bits);
        let candidate_loss = negative_log2_microbits(candidate_target, bits);
        self.source_total_microbits = self.source_total_microbits.saturating_add(source_loss);
        self.candidate_total_microbits = self
            .candidate_total_microbits
            .saturating_add(candidate_loss);
        if candidate_loss < source_loss {
            self.improved_loss_windows = self.improved_loss_windows.saturating_add(1);
        } else if candidate_loss > source_loss {
            self.worsened_loss_windows = self.worsened_loss_windows.saturating_add(1);
        } else {
            self.equal_loss_windows = self.equal_loss_windows.saturating_add(1);
        }
    }
}

/// Compare the frozen Q15 objective surface with wider views of the exact same
/// integer logits. No model arithmetic, weights, data order, or logits change;
/// only the retained fractional probability bits differ.
pub fn audit_production_probability_resolution(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionProbabilityResolutionTrace, TrainError> {
    const PRECISION_BITS: [u8; 5] = [15, 19, 23, 27, 31];

    source.validate()?;
    candidate.validate()?;
    if source.config != candidate.config
        || source.scales != candidate.scales
        || source.tokenizer_hash != candidate.tokenizer_hash
        || context_tokens == 0
        || context_tokens > source.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= source.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut accumulators = PRECISION_BITS
        .into_iter()
        .map(ProbabilityPrecisionAccumulator::new)
        .collect::<Vec<_>>();
    let mut logit_changed_windows = 0_usize;
    let mut target_logit_changed_windows = 0_usize;
    let mut q15_requantization_exact = true;
    let mut source_residual_saturation_count = 0_usize;
    let mut candidate_residual_saturation_count = 0_usize;
    let mut source_q31 = vec![0_u32; source.config.vocab_size];
    let mut candidate_q31 = vec![0_u32; source.config.vocab_size];

    for (context, target) in &windows {
        let source_output = forward_production_model(source, context)?;
        let candidate_output = forward_production_model(candidate, context)?;
        let target = *target as usize;
        base2_softmax_i32_q31(&source_output.logits_q8, &mut source_q31).ok_or(
            TrainError::CoreRejected("production_probability_audit_source"),
        )?;
        base2_softmax_i32_q31(&candidate_output.logits_q8, &mut candidate_q31).ok_or(
            TrainError::CoreRejected("production_probability_audit_candidate"),
        )?;

        logit_changed_windows = logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8 != candidate_output.logits_q8,
        ));
        target_logit_changed_windows = target_logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8[target] != candidate_output.logits_q8[target],
        ));
        source_residual_saturation_count = source_residual_saturation_count
            .saturating_add(source_output.residual_saturation_count);
        candidate_residual_saturation_count = candidate_residual_saturation_count
            .saturating_add(candidate_output.residual_saturation_count);

        for (index, &wide) in source_q31.iter().enumerate() {
            q15_requantization_exact &= quantize_probability_q31(wide, 15)
                == u32::try_from(source_output.probabilities_q15[index]).unwrap_or(0);
        }
        for (index, &wide) in candidate_q31.iter().enumerate() {
            q15_requantization_exact &= quantize_probability_q31(wide, 15)
                == u32::try_from(candidate_output.probabilities_q15[index]).unwrap_or(0);
        }

        for accumulator in &mut accumulators {
            accumulator.observe(&source_q31, &candidate_q31, target);
        }
    }

    Ok(ProductionProbabilityResolutionTrace {
        profile: source.config.profile_id().unwrap_or("custom"),
        parameter_count: source.parameter_count(),
        tokenizer_hash: source.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        forward_scales: source.scales,
        source_model_hash: source.model_hash(),
        candidate_model_hash: candidate.model_hash(),
        logit_changed_windows,
        target_logit_changed_windows,
        q15_requantization_exact,
        source_residual_saturation_count,
        candidate_residual_saturation_count,
        precision_rows: accumulators
            .into_iter()
            .map(|row| row.finish(source.config.vocab_size))
            .collect(),
    })
}

/// Compare reciprocal normalization implementations on the exact same frozen
/// logits. All rows retain Q23 probabilities; only the reciprocal path changes.
pub fn audit_production_probability_normalization(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionProbabilityNormalizationTrace, TrainError> {
    const PROBABILITY_FRACTIONAL_BITS: u8 = 23;
    const NORMALIZATIONS: [(&str, SoftmaxNormalization, u8); 4] = [
        ("legacy_q31_lut", SoftmaxNormalization::LegacyQ31Lut, 31),
        ("q47_lut", SoftmaxNormalization::Q47Lut, 47),
        ("q47_newton1", SoftmaxNormalization::Q47Newton1, 47),
        ("q47_exact_division", SoftmaxNormalization::Q47Exact, 47),
    ];

    source.validate()?;
    candidate.validate()?;
    if source.config != candidate.config
        || source.scales != candidate.scales
        || source.tokenizer_hash != candidate.tokenizer_hash
        || context_tokens == 0
        || context_tokens > source.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= source.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut accumulators = NORMALIZATIONS
        .iter()
        .map(|_| ProbabilityPrecisionAccumulator::new(PROBABILITY_FRACTIONAL_BITS))
        .collect::<Vec<_>>();
    let mut logit_changed_windows = 0_usize;
    let mut target_logit_changed_windows = 0_usize;
    let mut source_residual_saturation_count = 0_usize;
    let mut candidate_residual_saturation_count = 0_usize;
    let mut source_q31 = vec![0_u32; source.config.vocab_size];
    let mut candidate_q31 = vec![0_u32; source.config.vocab_size];

    for (context, target) in &windows {
        let source_output = forward_production_model(source, context)?;
        let candidate_output = forward_production_model(candidate, context)?;
        let target = *target as usize;

        logit_changed_windows = logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8 != candidate_output.logits_q8,
        ));
        target_logit_changed_windows = target_logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8[target] != candidate_output.logits_q8[target],
        ));
        source_residual_saturation_count = source_residual_saturation_count
            .saturating_add(source_output.residual_saturation_count);
        candidate_residual_saturation_count = candidate_residual_saturation_count
            .saturating_add(candidate_output.residual_saturation_count);

        for ((_, normalization, _), accumulator) in NORMALIZATIONS.iter().zip(&mut accumulators) {
            base2_softmax_i32_q31_with_normalization(
                &source_output.logits_q8,
                &mut source_q31,
                *normalization,
            )
            .ok_or(TrainError::CoreRejected(
                "production_probability_normalization_audit_source",
            ))?;
            base2_softmax_i32_q31_with_normalization(
                &candidate_output.logits_q8,
                &mut candidate_q31,
                *normalization,
            )
            .ok_or(TrainError::CoreRejected(
                "production_probability_normalization_audit_candidate",
            ))?;
            accumulator.observe(&source_q31, &candidate_q31, target);
        }
    }

    Ok(ProductionProbabilityNormalizationTrace {
        profile: source.config.profile_id().unwrap_or("custom"),
        parameter_count: source.parameter_count(),
        tokenizer_hash: source.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        probability_fractional_bits: PROBABILITY_FRACTIONAL_BITS,
        forward_scales: source.scales,
        source_model_hash: source.model_hash(),
        candidate_model_hash: candidate.model_hash(),
        logit_changed_windows,
        target_logit_changed_windows,
        source_residual_saturation_count,
        candidate_residual_saturation_count,
        normalization_rows: NORMALIZATIONS
            .into_iter()
            .zip(accumulators)
            .map(
                |((normalization, _, reciprocal_fractional_bits), accumulator)| {
                    ProductionProbabilityNormalizationRow {
                        normalization,
                        reciprocal_fractional_bits,
                        probability: accumulator.finish(source.config.vocab_size),
                    }
                },
            )
            .collect(),
    })
}

fn observe_normalization_error(
    error: &mut ProductionNormalizationErrorTrace,
    probabilities_q31: &[u32],
    exact_q31: &[u32],
    target: usize,
    fractional_bits: u8,
) {
    debug_assert_eq!(probabilities_q31.len(), exact_q31.len());
    for (&probability, &exact) in probabilities_q31.iter().zip(exact_q31) {
        let probability = quantize_probability_q31(probability, fractional_bits);
        let exact = quantize_probability_q31(exact, fractional_bits);
        let difference = probability.abs_diff(exact);
        error.probability_changed_values = error
            .probability_changed_values
            .saturating_add(usize::from(difference > 0));
        error.probability_error_l1 = error
            .probability_error_l1
            .saturating_add(u64::from(difference));
        error.probability_error_max = error.probability_error_max.max(difference);
    }
    let target_probability = quantize_probability_q31(probabilities_q31[target], fractional_bits);
    let exact_target = quantize_probability_q31(exact_q31[target], fractional_bits);
    let target_difference = target_probability.abs_diff(exact_target);
    error.target_error_windows = error
        .target_error_windows
        .saturating_add(usize::from(target_difference > 0));
    error.target_error_l1 = error
        .target_error_l1
        .saturating_add(u64::from(target_difference));
    error.target_error_max = error.target_error_max.max(target_difference);
}

fn target_softmax_weight_q15(logits_q8: &[i32], target: usize) -> Option<u16> {
    let target_logit = *logits_q8.get(target)?;
    if target_logit == MASKED_LOGIT {
        return None;
    }
    let max_logit = logits_q8
        .iter()
        .copied()
        .filter(|&logit| logit != MASKED_LOGIT)
        .max()?;
    Some(base2_exp_neg_q15(target_logit.saturating_sub(max_logit)) as u16)
}

/// Attribute the target-probability changes produced by the legacy and Newton
/// reciprocal paths against rounded exact Q47 division on the same frozen
/// logits. Only Q23 observation changes; model arithmetic and artifacts remain
/// read-only.
pub fn audit_production_probability_normalization_signal_attribution(
    source: &ProductionModelV1,
    candidate: &ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    context_tokens: usize,
    max_windows: usize,
) -> Result<ProductionNormalizationSignalAttributionTrace, TrainError> {
    const PROBABILITY_FRACTIONAL_BITS: u8 = 23;
    const METHODS: [(&str, SoftmaxNormalization, u8); 3] = [
        ("legacy_q31_lut", SoftmaxNormalization::LegacyQ31Lut, 31),
        ("q47_newton1", SoftmaxNormalization::Q47Newton1, 47),
        ("q47_exact_division", SoftmaxNormalization::Q47Exact, 47),
    ];

    source.validate()?;
    candidate.validate()?;
    if source.config != candidate.config
        || source.scales != candidate.scales
        || source.tokenizer_hash != candidate.tokenizer_hash
        || context_tokens == 0
        || context_tokens > source.config.context_tokens
        || max_windows == 0
        || tokens
            .iter()
            .any(|&token| token as usize >= source.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = document_windows(tokens, context_tokens, max_windows);
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let mut methods = METHODS
        .into_iter()
        .map(|(normalization, _, reciprocal_fractional_bits)| {
            ProductionNormalizationSignalMethodTrace {
                normalization,
                reciprocal_fractional_bits,
                target_changed_window_indices: Vec::new(),
                source_error_vs_exact: ProductionNormalizationErrorTrace::default(),
                candidate_error_vs_exact: ProductionNormalizationErrorTrace::default(),
            }
        })
        .collect::<Vec<_>>();
    let mut window_attributions = Vec::new();
    let mut logit_changed_windows = 0_usize;
    let mut target_logit_changed_windows = 0_usize;
    let mut source_residual_saturation_count = 0_usize;
    let mut candidate_residual_saturation_count = 0_usize;
    let mut source_exact = vec![0_u32; source.config.vocab_size];
    let mut candidate_exact = vec![0_u32; source.config.vocab_size];
    let mut source_work = vec![0_u32; source.config.vocab_size];
    let mut candidate_work = vec![0_u32; source.config.vocab_size];

    for (window_index, (context, target_token)) in windows.iter().enumerate() {
        let source_output = forward_production_model(source, context)?;
        let candidate_output = forward_production_model(candidate, context)?;
        let target = *target_token as usize;
        let source_sum = base2_softmax_i32_q31_with_normalization(
            &source_output.logits_q8,
            &mut source_exact,
            SoftmaxNormalization::Q47Exact,
        )
        .ok_or(TrainError::CoreRejected(
            "production_normalization_attribution_exact_source",
        ))?;
        let candidate_sum = base2_softmax_i32_q31_with_normalization(
            &candidate_output.logits_q8,
            &mut candidate_exact,
            SoftmaxNormalization::Q47Exact,
        )
        .ok_or(TrainError::CoreRejected(
            "production_normalization_attribution_exact_candidate",
        ))?;
        let source_target_weight = target_softmax_weight_q15(&source_output.logits_q8, target)
            .ok_or(TrainError::CoreRejected(
                "production_normalization_attribution_source_weight",
            ))?;
        let candidate_target_weight =
            target_softmax_weight_q15(&candidate_output.logits_q8, target).ok_or(
                TrainError::CoreRejected("production_normalization_attribution_candidate_weight"),
            )?;

        logit_changed_windows = logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8 != candidate_output.logits_q8,
        ));
        target_logit_changed_windows = target_logit_changed_windows.saturating_add(usize::from(
            source_output.logits_q8[target] != candidate_output.logits_q8[target],
        ));
        source_residual_saturation_count = source_residual_saturation_count
            .saturating_add(source_output.residual_saturation_count);
        candidate_residual_saturation_count = candidate_residual_saturation_count
            .saturating_add(candidate_output.residual_saturation_count);

        let mut target_probabilities = Vec::with_capacity(METHODS.len());
        let mut any_target_changed = false;
        for ((normalization_name, normalization, _), method) in METHODS.iter().zip(&mut methods) {
            base2_softmax_i32_q31_with_normalization(
                &source_output.logits_q8,
                &mut source_work,
                *normalization,
            )
            .ok_or(TrainError::CoreRejected(
                "production_normalization_attribution_source",
            ))?;
            base2_softmax_i32_q31_with_normalization(
                &candidate_output.logits_q8,
                &mut candidate_work,
                *normalization,
            )
            .ok_or(TrainError::CoreRejected(
                "production_normalization_attribution_candidate",
            ))?;
            observe_normalization_error(
                &mut method.source_error_vs_exact,
                &source_work,
                &source_exact,
                target,
                PROBABILITY_FRACTIONAL_BITS,
            );
            observe_normalization_error(
                &mut method.candidate_error_vs_exact,
                &candidate_work,
                &candidate_exact,
                target,
                PROBABILITY_FRACTIONAL_BITS,
            );
            let source_target =
                quantize_probability_q31(source_work[target], PROBABILITY_FRACTIONAL_BITS);
            let candidate_target =
                quantize_probability_q31(candidate_work[target], PROBABILITY_FRACTIONAL_BITS);
            let changed = source_target != candidate_target;
            any_target_changed |= changed;
            if changed {
                method.target_changed_window_indices.push(window_index);
            }
            target_probabilities.push(ProductionNormalizationTargetPair {
                normalization: normalization_name,
                source_probability_q23: source_target,
                candidate_probability_q23: candidate_target,
                delta_q23: i64::from(candidate_target) - i64::from(source_target),
            });
        }
        if any_target_changed {
            window_attributions.push(ProductionNormalizationSignalWindow {
                window_index,
                target_token: *target_token,
                source_target_logit_q8: source_output.logits_q8[target],
                candidate_target_logit_q8: candidate_output.logits_q8[target],
                source_target_weight_q15: source_target_weight,
                candidate_target_weight_q15: candidate_target_weight,
                source_normalization_sum: source_sum,
                candidate_normalization_sum: candidate_sum,
                target_probabilities,
            });
        }
    }

    Ok(ProductionNormalizationSignalAttributionTrace {
        profile: source.config.profile_id().unwrap_or("custom"),
        parameter_count: source.parameter_count(),
        tokenizer_hash: source.tokenizer_hash,
        token_stream_hash,
        context_tokens,
        windows: windows.len(),
        probability_fractional_bits: PROBABILITY_FRACTIONAL_BITS,
        forward_scales: source.scales,
        source_model_hash: source.model_hash(),
        candidate_model_hash: candidate.model_hash(),
        logit_changed_windows,
        target_logit_changed_windows,
        source_residual_saturation_count,
        candidate_residual_saturation_count,
        methods,
        window_attributions,
    })
}

fn quantize_probability_q31(probability_q31: u32, fractional_bits: u8) -> u32 {
    debug_assert!((1..=31).contains(&fractional_bits));
    let shift = 31_u8.saturating_sub(fractional_bits);
    let rounded = round_shift_rhu_i64(i64::from(probability_q31), shift);
    let maximum = (1_u64 << fractional_bits) - 1;
    u64::try_from(rounded).unwrap_or(0).min(maximum) as u32
}

fn negative_log2_microbits(probability: u32, fractional_bits: u8) -> u64 {
    if probability == 0 {
        return 64_000_000;
    }
    let integer_log2 = 31_u32.saturating_sub(probability.leading_zeros());
    let mut normalized = u64::from(probability) << (31 - integer_log2);
    let mut fractional_q24 = 0_u64;
    for bit in (0..24).rev() {
        normalized = ((u128::from(normalized) * u128::from(normalized)) >> 31) as u64;
        if normalized >= (2_u64 << 31) {
            normalized >>= 1;
            fractional_q24 |= 1_u64 << bit;
        }
    }
    let loss_q24 = (u64::from(fractional_bits.saturating_sub(integer_log2 as u8)) << 24)
        .saturating_sub(fractional_q24);
    loss_q24.saturating_mul(1_000_000).saturating_add(1 << 23) >> 24
}

fn signed_delta(candidate: u64, source: u64) -> i64 {
    if candidate >= source {
        i64::try_from(candidate - source).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(source - candidate).unwrap_or(i64::MAX)
    }
}

fn q15_negative_log2_millibits(probability: i16) -> u64 {
    let probability = u32::try_from(probability).unwrap_or(0);
    if probability == 0 {
        return 32_000;
    }
    let integer_log2 = 31_u32.saturating_sub(probability.leading_zeros());
    let mut normalized = u64::from(probability) << (31 - integer_log2);
    let mut fractional_q20 = 0_u64;
    for bit in (0..20).rev() {
        normalized = ((u128::from(normalized) * u128::from(normalized)) >> 31) as u64;
        if normalized >= (2_u64 << 31) {
            normalized >>= 1;
            fractional_q20 |= 1_u64 << bit;
        }
    }
    let loss_q20 =
        (u64::from(15_u32.saturating_sub(integer_log2)) << 20).saturating_sub(fractional_q20);
    loss_q20.saturating_mul(1_000).saturating_add(1 << 19) >> 20
}

pub fn train_production_output_smoke(
    model: &mut ProductionModelV1,
    tokens: &[u32],
    token_stream_hash: u64,
    config: ProductionSmokeConfig,
) -> Result<ProductionSmokeTrace, TrainError> {
    model.validate()?;
    if config.context_tokens == 0
        || config.context_tokens > model.config.context_tokens
        || config.max_windows == 0
        || config.epochs == 0
        || config.feature_shift > 15
        || tokens
            .iter()
            .any(|&token| token as usize >= model.config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let windows = if config.spread_windows {
        spread_document_windows(tokens, config.context_tokens, config.max_windows)
    } else {
        document_windows(tokens, config.context_tokens, config.max_windows)
    };
    if windows.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let initial_model_hash = model.model_hash();
    let mut cached = Vec::with_capacity(windows.len());
    let mut targets = Vec::with_capacity(windows.len());
    let mut residual_saturation_count = 0_usize;
    for (context, target) in &windows {
        let (features, saturation) = production_features(model, context)?;
        residual_saturation_count = residual_saturation_count.saturating_add(saturation);
        cached.push(features);
        targets.push(*target);
    }
    let initial_mistakes = count_mistakes(model, &cached, &targets)?;
    let mut updates = 0_usize;
    let mut weight_saturation_count = 0_usize;
    for _ in 0..config.epochs {
        for (features, &target) in cached.iter().zip(&targets) {
            let logits = output_logits(model, features)?;
            let predicted = argmax(&logits);
            let target_index = target as usize;
            let competitor = if predicted == target_index {
                argmax_except(&logits, target_index)
            } else {
                predicted
            };
            if predicted == target_index
                && logits[target_index] >= logits[competitor].saturating_add(config.margin_q8)
            {
                continue;
            }
            updates = updates.saturating_add(1);
            for (dim, &feature) in features.iter().enumerate() {
                let mut delta = i32::from(feature) >> config.feature_shift;
                if delta == 0 && feature != 0 {
                    delta = i32::from(feature.signum());
                }
                let target_offset = target_index * model.config.d_model + dim;
                let competitor_offset = competitor * model.config.d_model + dim;
                weight_saturation_count = weight_saturation_count
                    .saturating_add(update_i16(&mut model.output_weights[target_offset], delta));
                weight_saturation_count = weight_saturation_count.saturating_add(update_i16(
                    &mut model.output_weights[competitor_offset],
                    -delta,
                ));
            }
            model.output_bias_q8[target_index] =
                model.output_bias_q8[target_index].saturating_add(config.bias_step_q8);
            model.output_bias_q8[competitor] =
                model.output_bias_q8[competitor].saturating_sub(config.bias_step_q8);
        }
    }
    let final_mistakes = count_mistakes(model, &cached, &targets)?;
    Ok(ProductionSmokeTrace {
        profile: model.config.profile_id().unwrap_or("custom"),
        parameter_count: model.parameter_count(),
        tokenizer_hash: model.tokenizer_hash,
        token_stream_hash,
        context_tokens: config.context_tokens,
        windows: windows.len(),
        epochs: config.epochs,
        initial_mistakes,
        final_mistakes,
        updates,
        weight_saturation_count,
        residual_saturation_count,
        initial_model_hash,
        final_model_hash: model.model_hash(),
        spread_windows: config.spread_windows,
    })
}

pub fn decode_bound_token_stream(
    bytes: &[u8],
    tokenizer_hash: u64,
    vocab_size: usize,
) -> Result<(Vec<u32>, u64), TrainError> {
    if bytes.len() < 24 || &bytes[..8] != b"NSRLTOK1" {
        return Err(TrainError::InvalidModel("bad NSRLTOK1 header"));
    }
    let artifact_hash = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad NSRLTOK1 tokenizer hash"))?,
    );
    if artifact_hash != tokenizer_hash {
        return Err(TrainError::InvalidModel("NSRLTOK1 tokenizer hash mismatch"));
    }
    let token_count = usize::try_from(u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad NSRLTOK1 token count"))?,
    ))
    .map_err(|_| TrainError::InvalidModel("NSRLTOK1 token count overflow"))?;
    if bytes.len() != 24_usize.saturating_add(token_count.saturating_mul(4)) {
        return Err(TrainError::InvalidModel("wrong NSRLTOK1 length"));
    }
    let mut tokens = Vec::with_capacity(token_count);
    for chunk in bytes[24..].chunks_exact(4) {
        let token = u32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| TrainError::InvalidModel("truncated NSRLTOK1 token"))?,
        );
        if token as usize >= vocab_size {
            return Err(TrainError::InvalidModel(
                "NSRLTOK1 token exceeds model vocabulary",
            ));
        }
        tokens.push(token);
    }
    Ok((tokens, fnv1a(&bytes[24..])))
}

fn production_features(
    model: &ProductionModelV1,
    context: &[u32],
) -> Result<(Vec<i16>, usize), TrainError> {
    production_features_observed(model, context, None)
}

fn production_features_observed(
    model: &ProductionModelV1,
    context: &[u32],
    mut observer: Option<(&mut u64, &mut Vec<ProductionLayerBoundaryHashes>)>,
) -> Result<(Vec<i16>, usize), TrainError> {
    model.validate()?;
    let config = model.config;
    if context.is_empty()
        || context.len() > config.context_tokens
        || context
            .iter()
            .any(|&token| token as usize >= config.vocab_size)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut hidden = Vec::with_capacity(context.len() * config.d_model);
    for &token in context {
        let start = token as usize * config.d_model;
        hidden.extend_from_slice(&model.embeddings[start..start + config.d_model]);
    }
    if let Some((embedding_hash, _)) = observer.as_mut() {
        **embedding_hash = hash_i16_slice(&hidden);
    }
    let seq_len = context.len();
    let total = checked_product(seq_len, config.d_model)?;
    let matrix = checked_product(config.d_model, config.d_model)?;
    let up_matrix = checked_product(config.d_model, config.hidden_dim)?;
    let down_matrix = checked_product(config.hidden_dim, config.d_model)?;
    let mut residual_saturation_count = 0_usize;
    let qkv_scales = scales(config.d_model, model.scales.qkv_shift);
    let o_scales = scales(config.d_model, model.scales.o_shift);
    let up_scales = scales(config.hidden_dim, model.scales.up_shift);
    let gate_scales = scales(config.hidden_dim, model.scales.gate_shift);
    let down_scales = scales(config.d_model, model.scales.down_shift);
    for layer in 0..config.layers {
        let rms = layer * config.d_model..(layer + 1) * config.d_model;
        let attention_input = rms_rows(
            &hidden,
            &model.attention_rms_weights[rms.clone()],
            config.d_model,
        )?;
        let range = layer * matrix..(layer + 1) * matrix;
        let params = SelfAttentionI16Params {
            q: linear_params(
                &model.q_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            k: linear_params(
                &model.k_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            v: linear_params(
                &model.v_weights[range.clone()],
                &qkv_scales,
                config.d_model,
                config.d_model,
            ),
            o: linear_params(
                &model.o_weights[range],
                &o_scales,
                config.d_model,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            heads: config.heads,
            causal: true,
        };
        let head_dim = config.d_model / config.heads;
        let mut q = vec![0_i16; total];
        let mut k = vec![0_i16; total];
        let mut v = vec![0_i16; total];
        let mut attention_context = vec![0_i16; total];
        let mut state_kv = vec![0_i64; config.heads * head_dim * head_dim];
        let mut key_sums = vec![0_i64; config.heads * head_dim];
        let mut attention_output = vec![0_i16; total];
        linear_attention_i16_q15_checked(
            &attention_input,
            params,
            LinearAttentionWorkspace {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                context: &mut attention_context,
                state_kv: &mut state_kv,
                key_sums: &mut key_sums,
            },
            &mut attention_output,
        )
        .ok_or(TrainError::CoreRejected("production_linear_attention"))?;
        let mut attention_residual = vec![0_i16; total];
        let attention_residual_saturation_count =
            add_residual(&hidden, &attention_output, &mut attention_residual);
        residual_saturation_count =
            residual_saturation_count.saturating_add(attention_residual_saturation_count);
        let attention_residual_hash = observer
            .as_ref()
            .map_or(0, |_| hash_i16_slice(&attention_residual));
        let mlp_input = rms_rows(
            &attention_residual,
            &model.mlp_rms_weights[rms],
            config.d_model,
        )?;
        let up_range = layer * up_matrix..(layer + 1) * up_matrix;
        let down_range = layer * down_matrix..(layer + 1) * down_matrix;
        let mlp_params = GatedMlpI16Params {
            up: linear_params(
                &model.up_weights[up_range.clone()],
                &up_scales,
                config.d_model,
                config.hidden_dim,
            ),
            gate: linear_params(
                &model.gate_weights[up_range],
                &gate_scales,
                config.d_model,
                config.hidden_dim,
            ),
            down: linear_params(
                &model.down_weights[down_range],
                &down_scales,
                config.hidden_dim,
                config.d_model,
            ),
            seq_len,
            d_model: config.d_model,
            hidden_dim: config.hidden_dim,
        };
        let mut up = vec![0_i16; seq_len * config.hidden_dim];
        let mut gate = vec![0_i16; seq_len * config.hidden_dim];
        let mut gated = vec![0_i16; seq_len * config.hidden_dim];
        let mut mlp_output = vec![0_i16; total];
        gated_mlp_i16_q15_checked(
            &mlp_input,
            mlp_params,
            GatedMlpWorkspace {
                up: &mut up,
                gate: &mut gate,
                gated: &mut gated,
            },
            &mut mlp_output,
        )
        .ok_or(TrainError::CoreRejected("production_gated_mlp"))?;
        let mlp_residual_saturation_count =
            add_residual(&attention_residual, &mlp_output, &mut hidden);
        residual_saturation_count =
            residual_saturation_count.saturating_add(mlp_residual_saturation_count);
        if let Some((_, layers)) = observer.as_mut() {
            layers.push(ProductionLayerBoundaryHashes {
                layer,
                attention_residual_hash,
                layer_output_hash: hash_i16_slice(&hidden),
                attention_residual_saturation_count,
                mlp_residual_saturation_count,
            });
        }
    }
    let start = (seq_len - 1) * config.d_model;
    let mut features = vec![0_i16; config.d_model];
    rms_norm_i16_q15_checked(
        &hidden[start..start + config.d_model],
        &model.final_rms_weights,
        PRODUCTION_RMS_EPSILON,
        &mut features,
    )
    .ok_or(TrainError::CoreRejected("production_final_rms"))?;
    Ok((features, residual_saturation_count))
}

fn hash_i16_slice(values: &[i16]) -> u64 {
    values.iter().fold(FNV_OFFSET, |mut hash, value| {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn hash_i32_slice(values: &[i32]) -> u64 {
    values.iter().fold(FNV_OFFSET, |mut hash, value| {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    })
}

fn output_logits(model: &ProductionModelV1, features: &[i16]) -> Result<Vec<i32>, TrainError> {
    if features.len() != model.config.d_model {
        return Err(TrainError::InvalidConfig);
    }
    let mut logits = vec![0_i32; model.config.vocab_size];
    for (token, logit) in logits.iter_mut().enumerate() {
        let start = token * model.config.d_model;
        // The validated numeric contract proves this contiguous i16 dot product
        // cannot overflow i64, so this is byte-equivalent to saturating addition
        // while remaining vectorization-friendly.
        let accumulator = features
            .iter()
            .zip(&model.output_weights[start..start + model.config.d_model])
            .map(|(&feature, &weight)| i64::from(feature) * i64::from(weight))
            .sum::<i64>();
        let shifted = accumulator >> model.scales.output_shift;
        *logit = (shifted.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
            .saturating_add(model.output_bias_q8[token]);
    }
    Ok(logits)
}

fn document_windows(
    tokens: &[u32],
    context_tokens: usize,
    max_windows: usize,
) -> Vec<(Vec<u32>, u32)> {
    let mut windows = Vec::new();
    let mut document = Vec::new();
    let mut in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document && document.len() > context_tokens {
                for start in 0..document.len() - context_tokens {
                    windows.push((
                        document[start..start + context_tokens].to_vec(),
                        document[start + context_tokens],
                    ));
                    if windows.len() >= max_windows {
                        return windows;
                    }
                }
            }
            document.clear();
            in_document = false;
        } else if in_document {
            document.push(token);
        }
    }
    windows
}

fn spread_document_windows(
    tokens: &[u32],
    context_tokens: usize,
    max_windows: usize,
) -> Vec<(Vec<u32>, u32)> {
    let mut total_windows = 0_usize;
    let mut document_tokens = 0_usize;
    let mut in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document_tokens = 0;
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document {
                total_windows =
                    total_windows.saturating_add(document_tokens.saturating_sub(context_tokens));
            }
            document_tokens = 0;
            in_document = false;
        } else if in_document {
            document_tokens = document_tokens.saturating_add(1);
        }
    }
    let selected = max_windows.min(total_windows);
    if selected == 0 {
        return Vec::new();
    }
    let ranks = if selected == 1 {
        vec![total_windows / 2]
    } else {
        (0..selected)
            .map(|index| {
                ((index as u128) * ((total_windows - 1) as u128) / ((selected - 1) as u128))
                    as usize
            })
            .collect::<Vec<_>>()
    };
    let mut windows = Vec::with_capacity(selected);
    let mut rank_cursor = 0_usize;
    let mut current_rank = 0_usize;
    let mut document = Vec::new();
    in_document = false;
    for &token in tokens {
        if token == BOS_TOKEN_ID {
            document.clear();
            in_document = true;
        } else if token == EOS_TOKEN_ID {
            if in_document && document.len() > context_tokens {
                for start in 0..document.len() - context_tokens {
                    if rank_cursor < ranks.len() && current_rank == ranks[rank_cursor] {
                        windows.push((
                            document[start..start + context_tokens].to_vec(),
                            document[start + context_tokens],
                        ));
                        rank_cursor += 1;
                        if rank_cursor == ranks.len() {
                            return windows;
                        }
                    }
                    current_rank = current_rank.saturating_add(1);
                }
            }
            document.clear();
            in_document = false;
        } else if in_document {
            document.push(token);
        }
    }
    windows
}

fn count_mistakes(
    model: &ProductionModelV1,
    features: &[Vec<i16>],
    targets: &[u32],
) -> Result<usize, TrainError> {
    let mut mistakes = 0_usize;
    for (features, &target) in features.iter().zip(targets) {
        mistakes = mistakes.saturating_add(usize::from(
            argmax(&output_logits(model, features)?) != target as usize,
        ));
    }
    Ok(mistakes)
}

fn rms_rows(input: &[i16], weights: &[i16], d_model: usize) -> Result<Vec<i16>, TrainError> {
    let mut output = vec![0_i16; input.len()];
    for (input, output) in input
        .chunks_exact(d_model)
        .zip(output.chunks_exact_mut(d_model))
    {
        rms_norm_i16_q15_checked(input, weights, PRODUCTION_RMS_EPSILON, output)
            .ok_or(TrainError::CoreRejected("production_rms"))?;
    }
    Ok(output)
}

fn add_residual(left: &[i16], right: &[i16], output: &mut [i16]) -> usize {
    let mut saturation = 0_usize;
    for index in 0..output.len() {
        let value = i64::from(left[index]) + i64::from(right[index]);
        saturation = saturation.saturating_add(usize::from(
            value < i64::from(i16::MIN) || value > i64::from(i16::MAX),
        ));
        output[index] = saturate_i16(value);
    }
    saturation
}

fn linear_params<'a>(
    weights: &'a [i8],
    scales: &'a [FixedScale],
    input_dim: usize,
    output_dim: usize,
) -> LinearI16I8Params<'a> {
    LinearI16I8Params {
        weights,
        bias: None,
        scales,
        input_dim,
        output_dim,
    }
}

fn scales(count: usize, shift: u8) -> Vec<FixedScale> {
    vec![
        FixedScale {
            multiplier: 1,
            right_shift: shift,
        };
        count
    ]
}

fn scale_shifts(scales: ProductionProjectionScales) -> [u8; 6] {
    [
        scales.qkv_shift,
        scales.o_shift,
        scales.up_shift,
        scales.gate_shift,
        scales.down_shift,
        scales.output_shift,
    ]
}

fn stacked_identity(layers: usize, d_model: usize, strength: i8) -> Vec<i8> {
    let mut values = vec![0_i8; layers * d_model * d_model];
    for layer in 0..layers {
        let start = layer * d_model * d_model;
        for dim in 0..d_model {
            values[start + dim * d_model + dim] = strength;
        }
    }
    values
}

fn initial_i8_tensor(count: usize, seed: u64, amplitude: i8) -> Vec<i8> {
    let span = i16::from(amplitude) * 2 + 1;
    (0..count)
        .map(|index| {
            let value = splitmix64(seed ^ index as u64);
            ((value % span as u64) as i16 - i16::from(amplitude)) as i8
        })
        .collect()
}

fn initial_i16_tensor(count: usize, seed: u64, amplitude: i16) -> Vec<i16> {
    let span = i32::from(amplitude) * 2 + 1;
    (0..count)
        .map(|index| {
            let value = splitmix64(seed ^ index as u64) % span as u64;
            (value as i32 - i32::from(amplitude)) as i16
        })
        .collect()
}

fn positive_initial_i16(count: usize, seed: u64, min: i16, max: i16) -> Vec<i16> {
    let span = (max as u32).saturating_sub(min as u32).saturating_add(1) as u64;
    (0..count)
        .map(|index| {
            let value = splitmix64(seed ^ index as u64) % span;
            (min as u32)
                .saturating_add(value as u32)
                .min(i16::MAX as u32) as i16
        })
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn argmax(values: &[i32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn argmax_except(values: &[i32], excluded: usize) -> usize {
    values
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != excluded)
        .max_by_key(|&(index, value)| (*value, core::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn update_i16(value: &mut i16, delta: i32) -> usize {
    let updated = i64::from(*value) + i64::from(delta);
    let saturated = updated < i64::from(i16::MIN) || updated > i64::from(i16::MAX);
    *value = saturate_i16(updated);
    usize::from(saturated)
}

fn checked_product(left: usize, right: usize) -> Result<usize, TrainError> {
    left.checked_mul(right).ok_or(TrainError::InvalidConfig)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |mut hash, &byte| {
        hash ^= u64::from(byte);
        hash.wrapping_mul(FNV_PRIME)
    })
}

fn extend_i8(bytes: &mut Vec<u8>, values: &[i8]) {
    bytes.extend(values.iter().map(|&value| value as u8));
}

fn extend_i16(bytes: &mut Vec<u8>, values: &[i16]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, count: usize) -> Result<&'a [u8], TrainError> {
    let end = offset.checked_add(count).ok_or(TrainError::InvalidModel(
        "production artifact offset overflow",
    ))?;
    let value = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("truncated production artifact"))?;
    *offset = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, TrainError> {
    Ok(u32::from_le_bytes(
        take(bytes, offset, 4)?
            .try_into()
            .map_err(|_| TrainError::InvalidModel("truncated production u32"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, TrainError> {
    Ok(u64::from_le_bytes(
        take(bytes, offset, 8)?
            .try_into()
            .map_err(|_| TrainError::InvalidModel("truncated production u64"))?,
    ))
}

fn read_i8(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, TrainError> {
    Ok(take(bytes, offset, count)?
        .iter()
        .map(|&value| value as i8)
        .collect())
}

fn read_i16(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, TrainError> {
    let raw = take(
        bytes,
        offset,
        count
            .checked_mul(2)
            .ok_or(TrainError::InvalidModel("production i16 length overflow"))?,
    )?;
    Ok(raw
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_i32(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i32>, TrainError> {
    let raw = take(
        bytes,
        offset,
        count
            .checked_mul(4)
            .ok_or(TrainError::InvalidModel("production i32 length overflow"))?,
    )?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn frozen_profiles_match_scaling_plan_parameter_counts() {
        for (profile, expected_parameters) in [
            ("p10m", 9_317_632),
            ("p20m", 21_641_600),
            ("p30m", 28_229_056),
        ] {
            let config = ProductionModelConfig::profile(profile).unwrap();
            assert_eq!(config.parameter_count(), Some(expected_parameters));
            config.validate().expect("advertised profile must validate");
        }
    }

    #[test]
    fn production_artifact_round_trips_and_binds_tokenizer() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 7).expect("model");
        let bytes = model.try_to_bytes().expect("serialize");
        assert_eq!(
            ProductionModelV1::from_bytes(&bytes).expect("decode"),
            model
        );
        let mut corrupt = bytes;
        corrupt[32] ^= 1;
        assert!(ProductionModelV1::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn deterministic_output_initialization_activates_the_head() {
        let mut left = ProductionModelV1::new_initial(tiny_config(), 0x1234, 7).expect("model");
        let mut right = left.clone();
        left.initialize_output_weights(1).expect("initialize");
        right.initialize_output_weights(1).expect("initialize");
        assert_eq!(left.output_weights, right.output_weights);
        assert!(left.output_weights.iter().any(|&value| value != 0));
        assert!(left.output_weights.iter().all(|&value| value.abs() <= 1));
        assert!(left.initialize_output_weights(-1).is_err());
    }

    #[test]
    fn production_forward_and_smoke_training_accept_u32_tokens() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let context = [300, 301, 302, 303];
        let forward = forward_production_model(&model, &context).expect("forward");
        assert_eq!(forward.logits_q8.len(), 320);
        assert_eq!(forward.probabilities_q15.len(), 320);
        let branches =
            forward_production_model_branch_hashes(&model, &context).expect("branch health");
        assert_eq!(branches.layers.len(), model.config.layers);
        assert_eq!(
            branches
                .layers
                .iter()
                .map(|layer| {
                    layer.attention_residual_saturation_count + layer.mlp_residual_saturation_count
                })
                .sum::<usize>(),
            forward.residual_saturation_count
        );
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace = train_production_output_smoke(
            &mut model,
            &tokens,
            0x5678,
            ProductionSmokeConfig {
                context_tokens: 4,
                max_windows: 2,
                epochs: 4,
                ..ProductionSmokeConfig::default()
            },
        )
        .expect("smoke train");
        assert_eq!(trace.windows, 2);
        assert_ne!(trace.initial_model_hash, trace.final_model_hash);
        assert!(trace.final_mistakes <= trace.initial_mistakes);
    }

    #[test]
    fn spread_windows_cover_the_full_document_stream_deterministically() {
        let tokens = [
            BOS_TOKEN_ID,
            10,
            11,
            12,
            13,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            20,
            21,
            22,
            23,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            30,
            31,
            32,
            33,
            EOS_TOKEN_ID,
        ];
        let expected = vec![(vec![10, 11], 12), (vec![20, 21], 22), (vec![31, 32], 33)];
        assert_eq!(spread_document_windows(&tokens, 2, 3), expected);
        assert_eq!(spread_document_windows(&tokens, 2, 3), expected);
        assert_eq!(spread_document_windows(&tokens, 2, 99).len(), 6);
        assert!(spread_document_windows(&tokens, 8, 3).is_empty());
    }

    #[test]
    fn spread_window_selection_is_explicit_in_smoke_trace() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            304,
            305,
            306,
            307,
            EOS_TOKEN_ID,
        ];
        let trace = train_production_output_smoke(
            &mut model,
            &tokens,
            0x5678,
            ProductionSmokeConfig {
                context_tokens: 2,
                max_windows: 2,
                epochs: 1,
                spread_windows: true,
                ..ProductionSmokeConfig::default()
            },
        )
        .expect("spread smoke train");
        assert!(trace.spread_windows);
        assert!(trace.to_json_line().contains(
            "\"window_selection\":\"deterministic_uniform_target_rank_over_all_documents\""
        ));
    }

    #[test]
    fn production_eval_is_deterministic_and_reports_millibits() {
        assert_eq!(q15_negative_log2_millibits(16_384), 1_000);
        assert_eq!(q15_negative_log2_millibits(8_192), 2_000);
        assert_eq!(q15_negative_log2_millibits(0), 32_000);
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let left = evaluate_production_model(&model, &tokens, 0x5678, 4, 2).expect("eval");
        let right = evaluate_production_model(&model, &tokens, 0x5678, 4, 2).expect("eval");
        assert_eq!(left, right);
        assert_eq!(left.windows, 2);
        assert!(left.total_millibits > 0);
        assert!(left.to_json_line().contains("mean_millibits"));
    }

    #[test]
    fn canonical_production_eval_uses_normalization_independent_nll() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let left =
            evaluate_production_model_canonical_nll_default_floor(&model, &tokens, 0x5678, 4, 2)
                .expect("canonical eval");
        let right =
            evaluate_production_model_canonical_nll_default_floor(&model, &tokens, 0x5678, 4, 2)
                .expect("canonical eval replay");
        assert_eq!(left, right);
        assert_eq!(left.mean_nll_millibits, 8_322);
        assert_eq!(left.zero_probability_windows, 0);
        let json = left.to_json_line();
        assert!(json.contains("nsrl.production_model_canonical_eval.v2"));
        assert!(json.contains("\"normalization_independent\":true"));
    }

    #[test]
    fn production_comparison_reports_functional_equality_exactly() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace =
            compare_production_models(&model, &model, &tokens, 0x5678, 4, 2).expect("compare");
        assert_eq!(trace.windows, 2);
        assert_eq!(trace.total_millibits_delta, 0);
        assert_eq!(trace.feature_changed_windows, 0);
        assert_eq!(trace.logits_changed_windows, 0);
        assert_eq!(trace.probabilities_changed_windows, 0);
        assert_eq!(trace.equal_loss_windows, 2);
        assert!(trace.to_json_line().contains("functional_delta"));
    }

    #[test]
    fn probability_resolution_audit_preserves_q15_and_orders_precisions() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace = audit_production_probability_resolution(&model, &model, &tokens, 0x5678, 4, 2)
            .expect("audit");
        assert_eq!(trace.windows, 2);
        assert!(trace.q15_requantization_exact);
        assert_eq!(
            trace
                .precision_rows
                .iter()
                .map(|row| row.fractional_bits)
                .collect::<Vec<_>>(),
            [15, 19, 23, 27, 31]
        );
        assert!(trace.precision_rows.iter().all(|row| {
            row.probability_changed_windows == 0
                && row.target_probability_changed_windows == 0
                && row.equal_loss_windows == 2
        }));
        assert!(trace.to_json_line().contains("q15_requantization_exact"));
    }

    #[test]
    fn probability_normalization_audit_orders_methods_and_preserves_equality() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace =
            audit_production_probability_normalization(&model, &model, &tokens, 0x5678, 4, 2)
                .expect("audit");
        assert_eq!(trace.windows, 2);
        assert_eq!(trace.probability_fractional_bits, 23);
        assert_eq!(
            trace
                .normalization_rows
                .iter()
                .map(|row| row.normalization)
                .collect::<Vec<_>>(),
            [
                "legacy_q31_lut",
                "q47_lut",
                "q47_newton1",
                "q47_exact_division"
            ]
        );
        assert!(trace.normalization_rows.iter().all(|row| {
            row.probability.probability_changed_windows == 0
                && row.probability.target_probability_changed_windows == 0
                && row.probability.equal_loss_windows == 2
        }));
        assert!(trace.to_json_line().contains("q47_exact_division"));
    }

    #[test]
    fn probability_normalization_signal_attribution_uses_exact_ceiling() {
        let model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 11).expect("model");
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, 305, EOS_TOKEN_ID];
        let trace = audit_production_probability_normalization_signal_attribution(
            &model, &model, &tokens, 0x5678, 4, 2,
        )
        .expect("audit");
        assert_eq!(trace.windows, 2);
        assert_eq!(trace.probability_fractional_bits, 23);
        assert_eq!(
            trace
                .methods
                .iter()
                .map(|method| method.normalization)
                .collect::<Vec<_>>(),
            ["legacy_q31_lut", "q47_newton1", "q47_exact_division"]
        );
        assert!(
            trace
                .methods
                .iter()
                .all(|method| method.target_changed_window_indices.is_empty())
        );
        assert_eq!(
            trace.methods[2].source_error_vs_exact,
            ProductionNormalizationErrorTrace::default()
        );
        assert_eq!(
            trace.methods[2].candidate_error_vs_exact,
            ProductionNormalizationErrorTrace::default()
        );
        assert!(trace.window_attributions.is_empty());
        assert!(trace.to_json_line().contains("window_attributions"));
    }

    #[test]
    fn wider_probability_precision_can_expose_a_q15_tie() {
        let left = 262_144_u32;
        let right = 266_240_u32;
        assert_eq!(
            quantize_probability_q31(left, 15),
            quantize_probability_q31(right, 15)
        );
        assert_ne!(
            quantize_probability_q31(left, 19),
            quantize_probability_q31(right, 19)
        );
        assert_eq!(negative_log2_microbits(16_384, 15), 1_000_000);
        assert_eq!(negative_log2_microbits(8_192, 15), 2_000_000);
    }

    #[test]
    fn token_stream_loader_rejects_wrong_binding_and_vocab() {
        let tokenizer_hash = 0x1234_u64;
        let tokens = [BOS_TOKEN_ID, 300, EOS_TOKEN_ID];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"NSRLTOK1");
        bytes.extend_from_slice(&tokenizer_hash.to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        assert_eq!(
            decode_bound_token_stream(&bytes, tokenizer_hash, 320)
                .unwrap()
                .0,
            tokens
        );
        assert!(decode_bound_token_stream(&bytes, tokenizer_hash + 1, 320).is_err());
        assert!(decode_bound_token_stream(&bytes, tokenizer_hash, 300).is_err());
    }

    #[test]
    fn production_full_backward_moves_every_group_and_resumes_bound_state() {
        let mut initial = ProductionModelV1::new_initial(tiny_config(), 0x1234, 19).expect("model");
        initial.initialize_output_weights(1).expect("output init");
        initial.scales.output_shift = 14;
        initial.validate().expect("scaled model");
        let tokens = [
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            305,
            306,
            EOS_TOKEN_ID,
        ];
        let config = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 4,
            spread_windows: true,
            targets_per_window: 2,
            epochs: 2,
            q_learning_rate_shift: Some(17),
            k_learning_rate_shift: Some(21),
            down_learning_rate_shift: Some(16),
            output_backward_shift: Some(8),
            ..ProductionFullTrainConfig::default()
        };
        let mut uninterrupted_model = initial.clone();
        let (uninterrupted_trace, uninterrupted_state) =
            train_production_full_smoke(&mut uninterrupted_model, &tokens, 0x5678, config, None)
                .expect("full train");
        assert!(
            uninterrupted_trace
                .gradient_nonzero_count
                .iter()
                .sum::<u64>()
                > 0
        );
        assert!(uninterrupted_trace.schedule_complete);
        assert!(uninterrupted_trace.spread_windows);
        assert_eq!(uninterrupted_trace.targets_per_window, 2);
        assert!(uninterrupted_trace.supervised_targets > 0);
        assert!(!uninterrupted_trace.reject_saturated_batch);
        assert!(uninterrupted_trace.rejected_batch.is_none());
        assert!(
            !uninterrupted_trace
                .to_json_line()
                .contains("\"transaction\"")
        );
        assert!(uninterrupted_trace.to_json_line().contains(
            "\"window_selection\":\"deterministic_uniform_target_rank_over_all_documents\""
        ));
        assert!(
            uninterrupted_trace
                .to_json_line()
                .contains("\"target_policy\":\"causal_suffix_mean_v1\"")
        );
        assert_eq!(uninterrupted_trace.output_backward_shift, 8);
        assert_eq!(uninterrupted_trace.learning_rate_shifts[4], 18);
        assert_eq!(uninterrupted_trace.learning_rate_shifts[5], 22);
        assert_eq!(uninterrupted_trace.learning_rate_shifts[10], 17);
        assert_ne!(
            uninterrupted_trace.initial_model_hash,
            uninterrupted_trace.final_model_hash
        );

        let parallel_config = ProductionFullTrainConfig {
            training_workers: 4,
            ..config
        };
        let mut parallel_model = initial.clone();
        let (parallel_trace, parallel_state) = train_production_full_smoke(
            &mut parallel_model,
            &tokens,
            0x5678,
            parallel_config,
            None,
        )
        .expect("parallel full train");
        assert_eq!(parallel_model, uninterrupted_model);
        assert_eq!(parallel_state, uninterrupted_state);
        assert_eq!(parallel_trace.training_workers, 4);
        assert_eq!(parallel_trace.movement_l1, uninterrupted_trace.movement_l1);
        assert_eq!(
            parallel_trace.gradient_nonzero_count,
            uninterrupted_trace.gradient_nonzero_count
        );
        assert!(
            parallel_trace
                .to_json_line()
                .contains("\"training_workers\":4")
        );

        let mut resumed_model = initial.clone();
        let partial_config = ProductionFullTrainConfig {
            max_optimizer_steps: 1,
            ..config
        };
        let (partial_trace, partial_state) =
            train_production_full_smoke(&mut resumed_model, &tokens, 0x5678, partial_config, None)
                .expect("partial train");
        assert!(!partial_trace.schedule_complete);
        let bytes = partial_state.try_to_bytes().expect("optimizer bytes");
        let decoded = ProductionOptimizerStateV2::from_bytes(&bytes).expect("optimizer decode");
        assert_eq!(decoded, partial_state);
        let mut corrupt = bytes;
        corrupt[80] ^= 1;
        assert!(ProductionOptimizerStateV2::from_bytes(&corrupt).is_err());
        let (_, resumed_state) = train_production_full_smoke(
            &mut resumed_model,
            &tokens,
            0x5678,
            parallel_config,
            Some(decoded),
        )
        .expect("resume with a different worker count");
        assert_eq!(resumed_model, uninterrupted_model);
        assert_eq!(resumed_state, uninterrupted_state);
        assert!(
            train_production_full_smoke(
                &mut resumed_model,
                &tokens,
                0x5679,
                config,
                Some(resumed_state)
            )
            .is_err()
        );

        let embedding_flush_config = ProductionFullTrainConfig {
            flush_batched_embedding_residuals: true,
            ..config
        };
        let mut embedding_flush_uninterrupted_model = initial.clone();
        let (embedding_flush_trace, embedding_flush_uninterrupted_state) =
            train_production_full_smoke(
                &mut embedding_flush_uninterrupted_model,
                &tokens,
                0x5678,
                embedding_flush_config,
                None,
            )
            .expect("embedding flush full train");
        assert!(embedding_flush_trace.flush_batched_embedding_residuals);
        assert!(
            embedding_flush_trace
                .to_json_line()
                .contains("\"embedding_residual_flush\":\"all_batch_touched_tokens\"")
        );
        let mut embedding_flush_resumed_model = initial.clone();
        let (_, embedding_flush_partial_state) = train_production_full_smoke(
            &mut embedding_flush_resumed_model,
            &tokens,
            0x5678,
            ProductionFullTrainConfig {
                max_optimizer_steps: 1,
                ..embedding_flush_config
            },
            None,
        )
        .expect("partial embedding flush train");
        let (_, embedding_flush_resumed_state) = train_production_full_smoke(
            &mut embedding_flush_resumed_model,
            &tokens,
            0x5678,
            embedding_flush_config,
            Some(embedding_flush_partial_state),
        )
        .expect("resumed embedding flush train");
        assert_eq!(
            embedding_flush_resumed_model,
            embedding_flush_uninterrupted_model
        );
        assert_eq!(
            embedding_flush_resumed_state,
            embedding_flush_uninterrupted_state
        );
        assert!(
            embedding_flush_resumed_state
                .validate_binding(&embedding_flush_resumed_model, 0x5678, config)
                .is_err()
        );

        let stochastic_config = ProductionFullTrainConfig {
            backward_quantization: ProductionBackwardQuantization::LateStochastic,
            backward_stochastic_seed: 0x5eed,
            ..config
        };
        let mut stochastic_uninterrupted_model = initial.clone();
        let (stochastic_trace, stochastic_uninterrupted_state) = train_production_full_smoke(
            &mut stochastic_uninterrupted_model,
            &tokens,
            0x5678,
            stochastic_config,
            None,
        )
        .expect("stochastic full train");
        assert_eq!(
            stochastic_trace.backward_quantization,
            ProductionBackwardQuantization::LateStochastic
        );
        assert!(stochastic_trace.backward_stochastic_round_up_count > 0);
        assert!(
            stochastic_trace
                .to_json_line()
                .contains("\"backward_quantization\":\"late-stochastic\"")
        );

        let mut stochastic_resumed_model = initial;
        let stochastic_partial_config = ProductionFullTrainConfig {
            max_optimizer_steps: 1,
            ..stochastic_config
        };
        let (_, stochastic_partial_state) = train_production_full_smoke(
            &mut stochastic_resumed_model,
            &tokens,
            0x5678,
            stochastic_partial_config,
            None,
        )
        .expect("stochastic partial train");
        let (_, stochastic_resumed_state) = train_production_full_smoke(
            &mut stochastic_resumed_model,
            &tokens,
            0x5678,
            stochastic_config,
            Some(stochastic_partial_state),
        )
        .expect("stochastic resume");
        assert_eq!(stochastic_resumed_model, stochastic_uninterrupted_model);
        assert_eq!(stochastic_resumed_state, stochastic_uninterrupted_state);
        assert!(
            stochastic_resumed_state
                .validate_binding(
                    &stochastic_resumed_model,
                    0x5678,
                    ProductionFullTrainConfig {
                        backward_stochastic_seed: 0x5eee,
                        ..stochastic_config
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn production_full_saturation_guard_rejects_batch_atomically() {
        let mut model = ProductionModelV1::new_initial(tiny_config(), 0x1234, 23).expect("model");
        model.initialize_output_weights(1).expect("output init");
        model.output_bias_q8.fill(i32::MAX);
        model.validate().expect("saturation fixture");
        let initial = model.clone();
        let initial_hash = model.model_hash();
        let tokens = [BOS_TOKEN_ID, 300, 301, 302, 303, 304, EOS_TOKEN_ID];
        let config = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 1,
            epochs: 1,
            batch_windows: 1,
            max_optimizer_steps: 1,
            output_bias_learning_rate_shift: Some(0),
            reject_saturated_batch: true,
            ..ProductionFullTrainConfig::default()
        };

        let (trace, state) = train_production_full_smoke(&mut model, &tokens, 0x5678, config, None)
            .expect("guarded full train");
        let rejected = trace.rejected_batch.expect("saturated batch rejected");

        assert_eq!(model, initial);
        assert_eq!(model.model_hash(), initial_hash);
        assert_eq!(state.bound_model_hash, initial_hash);
        assert_eq!(state.step, 0);
        assert_eq!(state.next_epoch, 0);
        assert_eq!(state.next_window, 0);
        assert!(state.residuals.iter().all(|&residual| residual == 0));
        assert_eq!(trace.optimizer_steps, 0);
        assert_eq!(trace.total_optimizer_step, 0);
        assert_eq!(trace.supervised_targets, 0);
        assert_eq!(trace.initial_model_hash, initial_hash);
        assert_eq!(trace.final_model_hash, initial_hash);
        assert!(!trace.schedule_complete);
        assert_eq!(trace.weight_saturation_count, 0);
        assert_eq!(trace.gradient_saturation_count, 0);
        assert_eq!(trace.residual_saturation_count, 0);
        assert_eq!(rejected.attempted_total_optimizer_step, 1);
        assert_eq!(rejected.start_epoch, 0);
        assert_eq!(rejected.start_window, 0);
        assert_eq!(rejected.windows, 1);
        assert_eq!(rejected.supervised_targets, 1);
        assert!(rejected.weight_saturation_count > 0);
        assert!(rejected.saturation_by_group[12] > 0);
        let json = trace.to_json_line();
        assert!(json.contains("\"saturation_policy\":\"reject_batch_stop\""));
        assert!(json.contains("\"saturated_batch_rejected_atomically\":true"));
    }

    #[test]
    fn production_descent_guard_rejects_regression_and_resumes_exactly() {
        let mut initial = ProductionModelV1::new_initial(tiny_config(), 0x1234, 29).expect("model");
        initial.initialize_output_weights(1).expect("output init");
        initial.scales.output_shift = 14;
        initial.validate().expect("scaled model");
        let tokens = [
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            304,
            EOS_TOKEN_ID,
            BOS_TOKEN_ID,
            300,
            301,
            302,
            303,
            305,
            EOS_TOKEN_ID,
        ];
        let config = ProductionFullTrainConfig {
            context_tokens: 4,
            max_windows: 2,
            epochs: 1,
            matrix_learning_rate_shift: 62,
            vector_learning_rate_shift: 62,
            embedding_learning_rate_shift: 62,
            output_learning_rate_shift: 54,
            final_rms_learning_rate_shift: Some(62),
            output_bias_learning_rate_shift: Some(0),
            output_backward_shift: Some(8),
            probability_gradient_fractional_bits: 23,
            probability_normalization: SoftmaxNormalization::Q47Newton1,
            batch_windows: 1,
            max_optimizer_steps: 2,
            evaluation_windows: 2,
            descent_guard_windows: 1,
            descent_guard_signed_representation_blocks: true,
            descent_guard_signed_representation_zero_saturation: true,
            ..ProductionFullTrainConfig::default()
        };

        let mut uninterrupted = initial.clone();
        let (trace, uninterrupted_state) =
            train_production_full_smoke(&mut uninterrupted, &tokens, 0x5678, config, None)
                .expect("guarded train");
        assert_eq!(uninterrupted, initial);
        assert!(trace.schedule_complete);
        assert_eq!(trace.optimizer_steps, 2);
        assert_eq!(trace.descent_guard_evaluated_batches, 2);
        assert_eq!(trace.descent_guard_accepted_batches, 0);
        assert_eq!(trace.descent_guard_rejected_batches, 2);
        assert!(trace.descent_guard_signed_representation_blocks);
        assert!(trace.descent_guard_signed_representation_zero_saturation);
        assert_eq!(trace.signed_block_evaluated_batches, 0);
        assert_eq!(
            trace.descent_guard_initial_nll_millibits,
            trace.descent_guard_final_nll_millibits
        );
        assert!(trace.movement_l1.iter().all(|&movement| movement == 0));
        assert!(
            trace
                .descent_guard_last_rejected_batch
                .expect("rejected batch")
                .movement_l1[12]
                > 0
        );
        let json = trace.to_json_line();
        assert!(
            json.contains("\"descent_guard_policy\":\"reject_worsening_update_consume_batch\"")
        );
        assert!(json.contains("\"training_only_descent_guard_enabled\":true"));
        assert!(json.contains("\"signed_block_trust_region_enabled\":true"));
        assert!(json.contains("\"signed_block_zero_guard_residual_saturation_enabled\":true"));
        assert!(json.contains("\"signed_block_zero_saturation_feasibility_enforced\":true"));

        let mut resumed = initial.clone();
        let partial_config = ProductionFullTrainConfig {
            max_optimizer_steps: 1,
            ..config
        };
        let (partial_trace, partial_state) =
            train_production_full_smoke(&mut resumed, &tokens, 0x5678, partial_config, None)
                .expect("guarded partial train");
        assert_eq!(partial_trace.descent_guard_rejected_batches, 1);
        let (_, resumed_state) =
            train_production_full_smoke(&mut resumed, &tokens, 0x5678, config, Some(partial_state))
                .expect("guarded resume");
        assert_eq!(resumed, uninterrupted);
        assert_eq!(resumed_state, uninterrupted_state);
    }
}
