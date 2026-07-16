#![deny(unsafe_code)]
#![cfg(feature = "core-runtime")]

use core::ops::Range;

#[cfg(test)]
use nsrl_core::FixedScale;
use nsrl_core::{
    GatedMlpBackwardScales, GatedMlpBackwardWorkspace, GatedMlpI16Params,
    GatedMlpWeightUpdateParams, GatedMlpWeightUpdateStats, GatedMlpWeightUpdateWorkspace,
    GatedMlpWorkspace, LinearAttentionState, LinearAttentionStepWorkspace,
    LinearAttentionTttStepWorkspace, LinearAttentionWorkspace, LinearBackwardInputI16I8Params,
    LinearBackwardInputWorkspace, LinearBackwardWeightUpdateI8Params,
    LinearBackwardWeightUpdateWorkspace, LinearI16I8Params, LinearWeightUpdateStats,
    MAX_RIGHT_SHIFT, Q15_SHIFT, RmsNormBackwardWorkspace, SelfAttentionI16Params,
    SelfAttentionWorkspace, attention_dot_q_k_i16_i32_checked, base2_softmax_i32_q15,
    clear_linear_attention_state_checked, gated_mlp_backward_input_i16_q15_checked,
    gated_mlp_backward_weight_update_i8_checked, gated_mlp_i16_q15_checked,
    linear_attention_i16_q15_checked, linear_attention_state_lengths,
    linear_attention_step_i16_q15_checked, linear_attention_ttt_step_i16_q15_checked,
    linear_backward_input_i16_i8_i16_per_channel_checked,
    linear_backward_prescale_grad_output_i16_i32_checked, linear_backward_weight_update_i8_checked,
    linear_i16_i8_i16_per_channel_checked, rms_norm_backward_i16_q15_checked,
    rms_norm_i16_q15_checked, round_shift_rhu_i64, saturate_i8, saturate_i16,
    self_attention_i16_q15_checked, sqrt_power_of_four_shift,
};

pub mod artifact_contract;
pub mod mt6;
pub mod production;
pub mod solomon_latent;

#[path = "mini_transformer/block_expert.rs"]
mod block_expert;
#[path = "mini_transformer/decoding.rs"]
mod decoding;
#[path = "mini_transformer/generation.rs"]
mod generation;
#[path = "mini_transformer/gradients.rs"]
mod gradients;
#[path = "mini_transformer/model.rs"]
mod model;
#[path = "mini_transformer/trace.rs"]
mod trace;
#[path = "mini_transformer/training.rs"]
mod training;

pub use block_expert::{
    evaluate_mini_transformer_block_expert, mini_transformer_next_token_row_with_block_expert,
    mini_transformer_output_from_hidden_q15, mini_transformer_output_gradient_to_hidden_q15,
    train_mini_transformer_block_expert, train_mini_transformer_block_expert_with_layer_scope,
    train_mini_transformer_block_expert_with_layer_scope_and_loss_guard,
};
#[cfg(test)]
use decoding::select_byte_from_row;
use decoding::{select_byte_from_row_with_priors, validate_decode_priors};
pub use generation::{
    generate_mini_transformer, generate_mini_transformer_swarm,
    generate_mini_transformer_swarm_with_attention_kind_position_policy_and_priors,
    generate_mini_transformer_swarm_with_attention_kind_position_policy_composition_and_priors,
    generate_mini_transformer_with_attention_kind,
    generate_mini_transformer_with_attention_kind_and_priors,
    generate_mini_transformer_with_attention_kind_position_policy_and_priors,
    generate_mini_transformer_with_attention_kind_position_policy_priors_and_ttt_shift,
    generate_mini_transformer_with_priors, generate_routed_mini_transformer_swarm_experts,
    mini_transformer_next_token_row_with_attention_kind_position_policy,
    route_mini_transformer_swarm_expert_models, route_mini_transformer_swarm_experts,
};
use gradients::*;
pub use trace::{
    MiniTransformerBinaryTraceRecord, MiniTransformerBinaryTraceWriter,
    mini_transformer_binary_adaptive_shift_record_v1,
    mini_transformer_binary_final_summary_record_v1, mini_transformer_binary_step_sample_record_v1,
    mini_transformer_binary_trace_header_v1,
};
use trace::{
    push_mini_transformer_swarm_route_candidates_field,
    push_mini_transformer_swarm_route_config_field, push_mini_transformer_swarm_worker,
    push_usize_array_field,
};
use training::*;
pub use training::{
    MINI_TRANSFORMER_EVAL_SCHEMA, MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES,
    MiniTransformerMlpEvalConfig, MiniTransformerMlpEvalTrace, MiniTransformerMlpWindowEvalRecord,
    TrainError, assemble_mini_transformer_mlp_swarm_worker_artifacts,
    evaluate_mini_transformer_mlp_model, evaluate_mini_transformer_mlp_windows,
    run_mini_transformer_mlp_integer_adam_training,
    run_mini_transformer_mlp_integer_adam_training_from_model,
    run_mini_transformer_mlp_integer_adam_training_from_model_with_scope,
    run_mini_transformer_mlp_swarm_scaling_benchmark,
    run_mini_transformer_mlp_swarm_scaling_benchmark_from_model,
    run_mini_transformer_mlp_swarm_training, run_mini_transformer_mlp_swarm_training_from_model,
    run_mini_transformer_mlp_swarm_training_from_model_with_progress,
    run_mini_transformer_mlp_swarm_worker_from_model_with_progress,
    run_mini_transformer_mlp_training, run_mini_transformer_mlp_training_from_model,
    run_mini_transformer_mlp_training_from_model_with_progress,
    run_mini_transformer_mlp_training_from_model_with_progress_and_trace_detail,
    run_mini_transformer_mlp_training_from_model_with_progress_trace_detail_and_binary_trace,
    run_mini_transformer_mlp_training_with_model,
};

pub use artifact_contract::{
    ASCII_LOWER_TOKENIZER_ID, AUTHORITY, BYTE_TOKENIZER_ID, GENERATION_AUTHORITY,
    MINI_TRANSFORMER_ADAM_SCHEMA, MINI_TRANSFORMER_ADAM_STATE_MAGIC,
    MINI_TRANSFORMER_BINARY_ADAPTIVE_SHIFT_RECORD_LEN,
    MINI_TRANSFORMER_BINARY_FINAL_SUMMARY_RECORD_LEN,
    MINI_TRANSFORMER_BINARY_STEP_SAMPLE_RECORD_LEN, MINI_TRANSFORMER_BINARY_TAG_ADAPTIVE_SHIFT,
    MINI_TRANSFORMER_BINARY_TAG_FINAL_SUMMARY, MINI_TRANSFORMER_BINARY_TAG_STEP_SAMPLE,
    MINI_TRANSFORMER_BINARY_TRACE_HEADER_LEN, MINI_TRANSFORMER_BINARY_TRACE_MAGIC,
    MINI_TRANSFORMER_BINARY_TRACE_SCHEMA, MINI_TRANSFORMER_BINARY_TRACE_SCHEMA_ID,
    MINI_TRANSFORMER_BINARY_TRACE_VERSION, MINI_TRANSFORMER_BLOCK_EXPERT_MAGIC,
    MINI_TRANSFORMER_GENERATION_SCHEMA, MINI_TRANSFORMER_MLP_SCHEMA, MINI_TRANSFORMER_MLP_TASK,
    MINI_TRANSFORMER_MODEL_ID, MINI_TRANSFORMER_MODEL_MAGIC,
    MINI_TRANSFORMER_SWARM_CAPABILITY_TAGS, MINI_TRANSFORMER_SWARM_EXPERT_MANIFEST_SCHEMA,
    MINI_TRANSFORMER_SWARM_GENERATION_SCHEMA, MINI_TRANSFORMER_SWARM_MODEL_ID,
    MINI_TRANSFORMER_SWARM_MODEL_MAGIC, MINI_TRANSFORMER_SWARM_PROGRESS_SCHEMA,
    MINI_TRANSFORMER_SWARM_ROUTE_SCHEMA, MINI_TRANSFORMER_SWARM_ROUTED_GENERATION_SCHEMA,
    MINI_TRANSFORMER_SWARM_SCALING_SCHEMA, MINI_TRANSFORMER_SWARM_SCHEMA,
    MINI_TRANSFORMER_SWARM_WORKER_ARTIFACT_MAGIC, MINI_TRANSFORMER_SWARM_WORKER_SCHEMA,
    MINI_TRANSFORMER_V6_MODEL_MAGIC, PRODUCTION_MODEL_V1_MAGIC,
};
use artifact_contract::{
    MINI_TRANSFORMER_LEGACY_MODEL_MAGIC, MINI_TRANSFORMER_LEGACY_V4_D_MODEL,
    MINI_TRANSFORMER_LEGACY_V4_HEADS, MINI_TRANSFORMER_LEGACY_V4_HIDDEN_DIM,
};

pub const DEFAULT_MINI_TRANSFORMER_STREAMING_TTT_LEARNING_RATE_SHIFT: u8 = 8;
pub const LEXEME_DECODE_TOKEN_SET_CAP: usize = 64;
pub const BYTE_VOCAB: usize = 256;
pub const BYTE_D_MODEL: usize = 257;
// Single source of truth: the transformer dims, scale tables, and embedding
// grad fan-in shift are defined once in nsrl-train-core and re-exported here, so
// the host model and the no_std core can never drift. (A past drift silently
// zeroed mini-transformer training via InvalidShape; see the nsrl-train-core
// MINI_TRANSFORMER_D_MODEL doc comment.)
pub use nsrl_train_core::{
    IntegerAdamConfig, MINI_TRANSFORMER_ARCHITECTURE_PROFILE, MINI_TRANSFORMER_D_MODEL,
    MINI_TRANSFORMER_HEADS, MINI_TRANSFORMER_HIDDEN_DIM,
};
use nsrl_train_core::{
    MINI_TRANSFORMER_D_MODEL_GRAD_INPUT_SCALES, MINI_TRANSFORMER_D_MODEL_SCALES,
    MINI_TRANSFORMER_EMBEDDING_GRAD_FANIN_SHIFT, MINI_TRANSFORMER_HIDDEN_GRAD_INPUT_SCALES,
    MINI_TRANSFORMER_HIDDEN_SCALES, MINI_TRANSFORMER_OUTPUT_GRAD_INPUT_SCALES,
    MINI_TRANSFORMER_OUTPUT_SCALES,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15: i16 = 4096;
const DEFAULT_MINI_TRANSFORMER_EPOCHS: usize = 1;
const DEFAULT_MINI_TRANSFORMER_SEQ_LEN: usize = 4;
const DEFAULT_MINI_TRANSFORMER_STRIDE: usize = 1;
const DEFAULT_MINI_TRANSFORMER_MAX_WINDOWS: usize = 64;
const DEFAULT_MINI_TRANSFORMER_BATCH_WINDOWS: usize = 1;
const DEFAULT_MINI_TRANSFORMER_LEARNING_RATE: i32 = 1;
const DEFAULT_MINI_TRANSFORMER_HEAD_LEARNING_RATE_SHIFT: u8 = 18;
const DEFAULT_MINI_TRANSFORMER_MLP_LEARNING_RATE_SHIFT: u8 = 16;
const DEFAULT_MINI_TRANSFORMER_EMBEDDING_LEARNING_RATE_SHIFT: u8 = 14;
const DEFAULT_MINI_TRANSFORMER_ATTENTION_LEARNING_RATE_SHIFT: u8 = 24;
const DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT: u8 = 18;
#[cfg(feature = "mini-calibrated")]
const MINI_TRANSFORMER_NGRAM_CACHE_MAGIC: [u8; 8] = *b"NSRLNG1\0";
#[cfg(feature = "mini-calibrated")]
const MINI_TRANSFORMER_NGRAM_CACHE_HEADER_BYTES: usize = 296;
#[cfg(feature = "mini-calibrated")]
const MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER: usize = 4;
#[cfg(feature = "mini-calibrated")]
const MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC: [u8; 8] = *b"NSRLSM1\0";
#[cfg(feature = "mini-calibrated")]
const MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES: usize = 16;
const DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES: usize = 128;
const DEFAULT_MINI_TRANSFORMER_RMS_GAMMA_Q15: i16 = 16_384;
const MINI_TRANSFORMER_RMS_EPSILON: u64 = 1;
const DEFAULT_MINI_TRANSFORMER_LAYERS: usize = 2;
const MINI_TRANSFORMER_ATTENTION_VO_ORACLE_MAX_D_MODEL: usize = 64;
const MINI_TRANSFORMER_ADAPTIVE_RULE_TRACE_EVENT_LIMIT: usize = 256;
const MINI_TRANSFORMER_RULE_SATURATION_PRESSURE_DIVISOR: usize = 512;
const MINI_TRANSFORMER_RULE_ZERO_PRESSURE_NUMERATOR: usize = 31;
const MINI_TRANSFORMER_RULE_ZERO_PRESSURE_DENOMINATOR: usize = 32;
const DEFAULT_CORPUS_PRIOR_LOGIT_SHIFT: u8 = 8;
const DEFAULT_CORPUS_PRIOR_ORDER: u8 = 1;
const DEFAULT_LEXEME_MEMORY_LOGIT_SHIFT: u8 = 5;
const MINI_TRANSFORMER_ROLLBACK_HISTORY_LIMIT: usize = 8;
const PARALLEL_EVAL_MIN_ITEMS: usize = 512;
const PARALLEL_EVAL_MIN_ITEMS_PER_THREAD: usize = 128;
const BASE2_SOFTMAX_LN2_Q15: i32 = 22_713;
const MINI_TRANSFORMER_GENERATION_KNOWN_NON_CLAIMS: [&str; 5] = [
    "fixed_small_integer_transformer_only",
    "learned_absolute_position_embeddings_not_rope",
    "no_kv_cache_yet",
    "no_temperature_nucleus_or_beam_decode_yet",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_NOPE_GENERATION_KNOWN_NON_CLAIMS: [&str; 5] = [
    "fixed_small_integer_transformer_only",
    "nope_no_learned_position_embeddings",
    "no_kv_cache_yet",
    "no_temperature_nucleus_or_beam_decode_yet",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_STREAMING_GENERATION_KNOWN_NON_CLAIMS: [&str; 5] = [
    "single_layer_streaming_path_only",
    "streaming_nope_ignores_learned_position_embeddings",
    "linear_attention_requires_native_training_for_quality",
    "no_temperature_nucleus_or_beam_decode_yet",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_MLP_KNOWN_NON_CLAIMS: [&str; 5] = [
    "no_optimizer_moments_or_adam_state",
    "embedding_table_updated_without_optimizer_state",
    "fixed_two_head_attention_only",
    "does_not_backpropagate_through_rmsnorm_yet",
    "does_not_claim_language_model_quality",
];
const MINI_TRANSFORMER_SWARM_SCALING_KNOWN_NON_CLAIMS: [&str; 5] = [
    "host_timing_observation_not_universal_benchmark",
    "worker_shards_train_independent_models",
    "does_not_measure_power_or_cache_counters_yet",
    "does_not_claim_language_model_quality",
    "single_process_native_threads_only",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerPositionPolicy {
    LearnedAbsolute,
    Nope,
}

impl MiniTransformerPositionPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LearnedAbsolute => "learned_absolute_i16",
            Self::Nope => "nope",
        }
    }

    fn uses_position_embeddings(self) -> bool {
        matches!(self, Self::LearnedAbsolute)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerBatchMode {
    Serial,
    MapReduce,
}

impl MiniTransformerBatchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::MapReduce => "map-reduce",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerAdamTrainScope {
    All,
    RmsNorm,
    Output,
    FinalMlp,
    FinalMlpAndOutput,
}

impl MiniTransformerAdamTrainScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::RmsNorm => "rms_norm",
            Self::Output => "output",
            Self::FinalMlp => "final_mlp",
            Self::FinalMlpAndOutput => "final_mlp_and_output",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerTargetSegment {
    All,
    AfterMarkerBeforeAny {
        start_marker: u8,
        end_markers: [u8; 4],
        end_marker_count: u8,
    },
    AfterSequenceBeforeAny {
        start_sequence: [u8; 32],
        start_sequence_len: u8,
        end_markers: [u8; 4],
        end_marker_count: u8,
    },
    FirstAfterSequenceBeforeAny {
        start_sequence: [u8; 32],
        start_sequence_len: u8,
        end_markers: [u8; 4],
        end_marker_count: u8,
    },
}

impl MiniTransformerTargetSegment {
    pub fn after_marker_before_any(
        start_marker: u8,
        end_markers: &[u8],
    ) -> Result<Self, TrainError> {
        if end_markers.is_empty() || end_markers.len() > 4 {
            return Err(TrainError::InvalidConfig);
        }
        let mut markers = [0_u8; 4];
        markers[..end_markers.len()].copy_from_slice(end_markers);
        Ok(Self::AfterMarkerBeforeAny {
            start_marker,
            end_markers: markers,
            end_marker_count: end_markers.len() as u8,
        })
    }

    pub fn after_sequence_before_any(
        start_sequence: &[u8],
        end_markers: &[u8],
    ) -> Result<Self, TrainError> {
        if start_sequence.is_empty()
            || start_sequence.len() > 32
            || end_markers.is_empty()
            || end_markers.len() > 4
        {
            return Err(TrainError::InvalidConfig);
        }
        let mut sequence = [0_u8; 32];
        sequence[..start_sequence.len()].copy_from_slice(start_sequence);
        let mut markers = [0_u8; 4];
        markers[..end_markers.len()].copy_from_slice(end_markers);
        Ok(Self::AfterSequenceBeforeAny {
            start_sequence: sequence,
            start_sequence_len: start_sequence.len() as u8,
            end_markers: markers,
            end_marker_count: end_markers.len() as u8,
        })
    }

    pub fn first_after_sequence_before_any(
        start_sequence: &[u8],
        end_markers: &[u8],
    ) -> Result<Self, TrainError> {
        if start_sequence.is_empty()
            || start_sequence.len() > 32
            || end_markers.is_empty()
            || end_markers.len() > 4
        {
            return Err(TrainError::InvalidConfig);
        }
        let mut sequence = [0_u8; 32];
        sequence[..start_sequence.len()].copy_from_slice(start_sequence);
        let mut markers = [0_u8; 4];
        markers[..end_markers.len()].copy_from_slice(end_markers);
        Ok(Self::FirstAfterSequenceBeforeAny {
            start_sequence: sequence,
            start_sequence_len: start_sequence.len() as u8,
            end_markers: markers,
            end_marker_count: end_markers.len() as u8,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::AfterMarkerBeforeAny { .. } => "after_marker_before_any",
            Self::AfterSequenceBeforeAny { .. } => "after_sequence_before_any",
            Self::FirstAfterSequenceBeforeAny { .. } => "first_after_sequence_before_any",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainConfig {
    pub epochs: usize,
    pub seq_len: usize,
    pub stride: usize,
    pub window_offset: usize,
    pub max_windows: Option<usize>,
    pub batch_windows: usize,
    pub target_token_min: u8,
    pub target_token_max: u8,
    pub target_segment: MiniTransformerTargetSegment,
    pub target_frequency_cap: u32,
    pub target_frequency_min_weight_q15: i16,
    pub argmax_margin_weight_q15: i16,
    pub tokenizer_id: ByteTokenizerId,
    pub attention_kind: MiniTransformerAttentionKind,
    pub position_policy: MiniTransformerPositionPolicy,
    pub learning_rate: i32,
    pub output_learning_rate_shift: u8,
    pub mlp_learning_rate_shift: u8,
    pub embedding_learning_rate_shift: u8,
    pub attention_learning_rate_shift: u8,
    pub attention_q_learning_rate_shift: u8,
    pub attention_qk_learning_rate_shift: u8,
    pub adaptive_rule_shifts: bool,
    pub adaptive_rule_interval_batches: usize,
    pub adaptive_attention_shifts: bool,
    pub adaptive_holographic_shifts: bool,
    pub attention_vo_error_feedback: bool,
    pub attention_vo_oracle: bool,
    pub reject_loss_regression: bool,
    pub batch_mode: MiniTransformerBatchMode,
    pub map_reduce_workers: usize,
}

impl MiniTransformerMlpTrainConfig {
    fn adaptive_shift_controller_enabled(self) -> bool {
        self.adaptive_rule_shifts
            || self.adaptive_attention_shifts
            || self.adaptive_holographic_shifts
    }

    fn adaptive_rule_shift_controller_enabled(self) -> bool {
        self.adaptive_rule_shifts || self.adaptive_attention_shifts
    }

    fn adaptive_holographic_shift_controller_enabled(self) -> bool {
        self.adaptive_holographic_shifts
    }
}

impl Default for MiniTransformerMlpTrainConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_MINI_TRANSFORMER_EPOCHS,
            seq_len: DEFAULT_MINI_TRANSFORMER_SEQ_LEN,
            stride: DEFAULT_MINI_TRANSFORMER_STRIDE,
            window_offset: 0,
            max_windows: Some(DEFAULT_MINI_TRANSFORMER_MAX_WINDOWS),
            batch_windows: DEFAULT_MINI_TRANSFORMER_BATCH_WINDOWS,
            target_token_min: u8::MIN,
            target_token_max: u8::MAX,
            target_segment: MiniTransformerTargetSegment::All,
            target_frequency_cap: 0,
            target_frequency_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            argmax_margin_weight_q15: 0,
            tokenizer_id: ByteTokenizerId::Identity,
            attention_kind: MiniTransformerAttentionKind::Base2Softmax,
            position_policy: MiniTransformerPositionPolicy::LearnedAbsolute,
            learning_rate: DEFAULT_MINI_TRANSFORMER_LEARNING_RATE,
            output_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_HEAD_LEARNING_RATE_SHIFT,
            mlp_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_MLP_LEARNING_RATE_SHIFT,
            embedding_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_EMBEDDING_LEARNING_RATE_SHIFT,
            attention_learning_rate_shift: DEFAULT_MINI_TRANSFORMER_ATTENTION_LEARNING_RATE_SHIFT,
            attention_q_learning_rate_shift:
                DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT,
            attention_qk_learning_rate_shift:
                DEFAULT_MINI_TRANSFORMER_ATTENTION_QK_LEARNING_RATE_SHIFT,
            adaptive_rule_shifts: false,
            adaptive_rule_interval_batches: DEFAULT_MINI_TRANSFORMER_ADAPTIVE_RULE_INTERVAL_BATCHES,
            adaptive_attention_shifts: false,
            adaptive_holographic_shifts: false,
            attention_vo_error_feedback: false,
            attention_vo_oracle: false,
            reject_loss_regression: false,
            batch_mode: MiniTransformerBatchMode::Serial,
            map_reduce_workers: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingTrace {
    pub trace_detail: MiniTransformerTraceDetail,
    pub config: MiniTransformerMlpTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub accepted_batch_count: usize,
    pub rejected_batch_count: usize,
    pub output_head_accumulator_batch_count: usize,
    pub output_head_accumulator_window_count: usize,
    pub mlp_accumulator_batch_count: usize,
    pub mlp_accumulator_window_count: usize,
    pub attention_accumulator_batch_count: usize,
    pub attention_accumulator_window_count: usize,
    pub embedding_accumulator_batch_count: usize,
    pub embedding_accumulator_window_count: usize,
    pub rollback_count: usize,
    pub rejected_window_count: usize,
    pub loss_regression_rejected_batch_count: usize,
    pub final_invalid_forward_count: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub initial_embedding_hash: u64,
    pub final_embedding_hash: u64,
    pub initial_output_head_hash: u64,
    pub final_output_head_hash: u64,
    pub initial_mlp_hash: u64,
    pub final_mlp_hash: u64,
    pub initial_attention_hash: u64,
    pub final_attention_hash: u64,
    pub initial_attention_q_hash: u64,
    pub final_attention_q_hash: u64,
    pub initial_attention_k_hash: u64,
    pub final_attention_k_hash: u64,
    pub initial_attention_v_hash: u64,
    pub final_attention_v_hash: u64,
    pub initial_attention_o_hash: u64,
    pub final_attention_o_hash: u64,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub output_head_saturation_count: usize,
    pub mlp_saturation_count: usize,
    pub embedding_saturation_count: usize,
    pub attention_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub output_head_zero_delta_count: usize,
    pub mlp_zero_delta_count: usize,
    pub embedding_zero_delta_count: usize,
    pub attention_zero_delta_count: usize,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub embedding_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
    pub output_head_carry_l1: u64,
    pub mlp_carry_l1: u64,
    pub embedding_carry_l1: u64,
    pub attention_carry_l1: u64,
    pub attention_q_carry_l1: u64,
    pub attention_k_carry_l1: u64,
    pub attention_v_carry_l1: u64,
    pub attention_o_carry_l1: u64,
    pub adaptive_rule_shift_adjustment_count: usize,
    pub adaptive_rule_update_count: usize,
    pub adaptive_rule_event_count: usize,
    pub adaptive_holographic_shift_adjustment_count: usize,
    pub adaptive_holographic_update_count: usize,
    pub adaptive_holographic_hash: u64,
    pub adaptive_attention_shift_adjustment_count: usize,
    pub adaptive_attention_holographic_update_count: usize,
    pub adaptive_attention_holographic_hash: u64,
    pub final_output_learning_rate_shift: u8,
    pub final_mlp_learning_rate_shift: u8,
    pub final_embedding_learning_rate_shift: u8,
    pub final_attention_learning_rate_shift: u8,
    pub final_attention_q_learning_rate_shift: u8,
    pub final_attention_qk_learning_rate_shift: u8,
    pub final_accuracy_per_mille: usize,
    pub final_logits_hash: u64,
    pub adaptive_shift_events: Vec<MiniTransformerAdaptiveShiftEventTrace>,
    pub steps: Vec<MiniTransformerMlpTrainingStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingProgressTrace {
    pub config: MiniTransformerMlpTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub accepted_batch_count: usize,
    pub rejected_batch_count: usize,
    pub rollback_count: usize,
    pub rejected_window_count: usize,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub embedding_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
    pub output_head_carry_l1: u64,
    pub mlp_carry_l1: u64,
    pub embedding_carry_l1: u64,
    pub attention_carry_l1: u64,
    pub attention_q_carry_l1: u64,
    pub attention_k_carry_l1: u64,
    pub attention_v_carry_l1: u64,
    pub attention_o_carry_l1: u64,
    pub adaptive_rule_shift_adjustment_count: usize,
    pub adaptive_holographic_shift_adjustment_count: usize,
    pub current_output_learning_rate_shift: u8,
    pub current_mlp_learning_rate_shift: u8,
    pub current_embedding_learning_rate_shift: u8,
    pub current_attention_learning_rate_shift: u8,
    pub current_attention_q_learning_rate_shift: u8,
    pub current_attention_qk_learning_rate_shift: u8,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingRun {
    pub trace: MiniTransformerMlpTrainingTrace,
    pub model: MiniTransformerMlpModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerAdamTrainingTrace {
    pub schema: &'static str,
    pub config: MiniTransformerMlpTrainConfig,
    pub optimizer_config: IntegerAdamConfig,
    pub train_scope: MiniTransformerAdamTrainScope,
    pub token_count: usize,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub accepted_batch_count: usize,
    pub rejected_batch_count: usize,
    pub initial_mistakes: usize,
    pub final_mistakes: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub transformer_layers: usize,
    pub rms_norm_enabled: bool,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub embedding_delta_l1: u64,
    pub rms_norm_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
    pub mlp_saturation_count: usize,
    pub attention_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub initial_model_hash: u64,
    pub final_model_hash: u64,
    pub optimizer_step: u64,
    pub optimizer_state_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerAdamTrainingRun {
    pub trace: MiniTransformerAdamTrainingTrace,
    pub model: MiniTransformerMlpModel,
    pub optimizer_state: MiniTransformerAdamOptimizerState,
}

impl MiniTransformerAdamTrainingTrace {
    pub fn to_json_line(&self) -> String {
        #[cfg(feature = "mini-calibrated")]
        let quantization_profile_json = ",\"quantization_profile\":\"calibrated-v2-suffix-memory\"";
        #[cfg(not(feature = "mini-calibrated"))]
        let quantization_profile_json = "";
        let max_windows = self
            .config
            .max_windows
            .map_or_else(|| String::from("null"), |value| value.to_string());
        let final_accuracy_per_mille = self
            .windows
            .saturating_sub(self.final_mistakes)
            .saturating_mul(1000)
            / self.windows.max(1);
        format!(
            concat!(
                "{{\"schema\":\"{}\",",
                "\"model\":{{\"architecture_profile\":\"{}\",\"d_model\":{},\"heads\":{},\"hidden_dim\":{},\"transformer_layers\":{},\"rms_norm_enabled\":{}{}}},",
                "\"optimizer\":{{\"kind\":\"integer_adam\",\"learning_rate\":{},\"step_shift\":{},\"beta1_decay_shift\":{},\"beta2_decay_shift\":{},\"epsilon\":{}}},",
                "\"training\":{{\"epochs\":{},\"seq_len\":{},\"stride\":{},\"window_offset\":{},\"max_windows\":{},\"batch_windows\":{},\"attention_kind\":\"{}\",\"position\":\"{}\",\"batch_mode\":\"{}\",\"map_reduce_workers\":{},\"train_scope\":\"{}\",\"target_frequency_cap\":{},\"target_frequency_min_weight_q15\":{},\"argmax_margin_weight_q15\":{}}},",
                "\"data\":{{\"token_count\":{},\"token_hash\":\"0x{:016x}\",\"window_hash\":\"0x{:016x}\",\"windows\":{},\"examined_windows\":{}}},",
                "\"updates\":{{\"accepted_windows\":{},\"accepted_batches\":{},\"rejected_batches\":{},\"optimizer_step\":{}}},",
                "\"loss\":{{\"initial_mistakes\":{},\"final_mistakes\":{},\"final_accuracy_per_mille\":{},\"initial_probability_error_q15\":{},\"final_probability_error_q15\":{}}},",
                "\"delta_l1\":{{\"output_head\":{},\"mlp\":{},\"embedding\":{},\"rms_norm\":{},\"attention\":{},\"attention_q\":{},\"attention_k\":{},\"attention_v\":{},\"attention_o\":{}}},",
                "\"saturation\":{{\"mlp\":{},\"attention\":{},\"residual\":{}}},",
                "\"hashes\":{{\"initial_model\":\"0x{:016x}\",\"final_model\":\"0x{:016x}\",\"optimizer_state\":\"0x{:016x}\"}}}}\n"
            ),
            self.schema,
            MINI_TRANSFORMER_ARCHITECTURE_PROFILE,
            MINI_TRANSFORMER_D_MODEL,
            MINI_TRANSFORMER_HEADS,
            MINI_TRANSFORMER_HIDDEN_DIM,
            self.transformer_layers,
            self.rms_norm_enabled,
            quantization_profile_json,
            self.optimizer_config.learning_rate,
            self.optimizer_config.step_shift,
            self.optimizer_config.beta1_decay_shift,
            self.optimizer_config.beta2_decay_shift,
            self.optimizer_config.epsilon,
            self.config.epochs,
            self.config.seq_len,
            self.config.stride,
            self.config.window_offset,
            max_windows,
            self.config.batch_windows,
            self.config.attention_kind.as_str(),
            self.config.position_policy.as_str(),
            self.config.batch_mode.as_str(),
            self.config.map_reduce_workers,
            self.train_scope.as_str(),
            self.config.target_frequency_cap,
            self.config.target_frequency_min_weight_q15,
            self.config.argmax_margin_weight_q15,
            self.token_count,
            self.token_hash,
            self.window_hash,
            self.windows,
            self.examined_windows,
            self.updates,
            self.accepted_batch_count,
            self.rejected_batch_count,
            self.optimizer_step,
            self.initial_mistakes,
            self.final_mistakes,
            final_accuracy_per_mille,
            self.initial_probability_error_q15,
            self.final_probability_error_q15,
            self.output_head_delta_l1,
            self.mlp_delta_l1,
            self.embedding_delta_l1,
            self.rms_norm_delta_l1,
            self.attention_delta_l1,
            self.attention_q_delta_l1,
            self.attention_k_delta_l1,
            self.attention_v_delta_l1,
            self.attention_o_delta_l1,
            self.mlp_saturation_count,
            self.attention_saturation_count,
            self.residual_saturation_count,
            self.initial_model_hash,
            self.final_model_hash,
            self.optimizer_state_hash,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmTrainConfig {
    pub workers: usize,
    pub trace_detail: MiniTransformerTraceDetail,
}

impl Default for MiniTransformerMlpSwarmTrainConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            trace_detail: MiniTransformerTraceDetail::None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmTrainingRun {
    pub trace: MiniTransformerMlpSwarmTrainingTrace,
    pub model: MiniTransformerMlpModel,
    pub swarm_model: MiniTransformerMlpSwarmModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmTrainingTrace {
    pub config: MiniTransformerMlpTrainConfig,
    pub swarm_config: MiniTransformerMlpSwarmTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub worker_count: usize,
    pub base_window_offset: usize,
    pub base_stride: usize,
    pub best_worker_index: usize,
    pub final_model_hash: u64,
    pub workers: Vec<MiniTransformerMlpSwarmWorkerTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmTrainingProgressTrace {
    pub config: MiniTransformerMlpTrainConfig,
    pub swarm_config: MiniTransformerMlpSwarmTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub worker_count: usize,
    pub base_window_offset: usize,
    pub base_stride: usize,
    pub workers: Vec<MiniTransformerMlpSwarmWorkerProgressTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmWorkerProgressTrace {
    pub worker_index: usize,
    pub progress: MiniTransformerMlpTrainingProgressTrace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmWorkerTrace {
    pub worker_index: usize,
    pub window_offset: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
    pub token_hash: u64,
    pub window_hash: u64,
    pub windows: usize,
    pub examined_windows: usize,
    pub updates: usize,
    pub accepted_batch_count: usize,
    pub rejected_batch_count: usize,
    pub rollback_count: usize,
    pub rejected_window_count: usize,
    pub final_invalid_forward_count: usize,
    pub initial_total_error: usize,
    pub final_total_error: usize,
    pub initial_probability_error_q15: usize,
    pub final_probability_error_q15: usize,
    pub final_accuracy_per_mille: usize,
    pub final_model_hash: u64,
    pub final_logits_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmWorkerArtifact {
    pub worker_count: usize,
    pub token_count: usize,
    pub token_hash: u64,
    pub base_window_offset: usize,
    pub base_stride: usize,
    pub base_max_windows: Option<usize>,
    pub base_model_hash: u64,
    pub worker: MiniTransformerMlpSwarmWorkerTrace,
    pub model: MiniTransformerMlpModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmWorkerTrainingRun {
    pub artifact: MiniTransformerMlpSwarmWorkerArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmScalingTrace {
    pub config: MiniTransformerMlpTrainConfig,
    pub token_count: usize,
    pub token_hash: u64,
    pub available_parallelism: usize,
    pub requested_max_workers: usize,
    pub worker_counts: Vec<usize>,
    pub runs: Vec<MiniTransformerMlpSwarmScalingRunTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmScalingRunTrace {
    pub requested_worker_count: usize,
    pub effective_worker_count: usize,
    pub elapsed_ns: u64,
    pub speedup_per_mille: u64,
    pub parallel_efficiency_per_mille: u64,
    pub windows_per_second_milli: u64,
    pub updates_per_second_milli: u64,
    pub examined_windows: usize,
    pub updates: usize,
    pub accepted_batch_count: usize,
    pub rejected_batch_count: usize,
    pub rollback_count: usize,
    pub best_worker_index: usize,
    pub best_final_total_error: usize,
    pub best_final_probability_error_q15: usize,
    pub best_final_accuracy_per_mille: usize,
    pub final_model_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmModel {
    pub context_seq_len: usize,
    pub best_worker_index: usize,
    pub workers: Vec<MiniTransformerMlpModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpSwarmExpertManifest {
    pub artifact_format: &'static str,
    pub artifact_magic: &'static str,
    pub artifact_byte_count: usize,
    pub model_id: &'static str,
    pub tokenizer: &'static str,
    pub context_seq_len: usize,
    pub worker_count: usize,
    pub best_worker_index: usize,
    pub parameter_bytes: usize,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
    pub worker_model_hashes: Vec<u64>,
    pub worker_parameter_bytes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRouteConfig {
    pub required_capabilities: Vec<String>,
    pub max_artifact_bytes: Option<usize>,
    pub max_parameter_bytes: Option<usize>,
    pub active_expert_limit: usize,
    pub prompt_affinity: bool,
    pub prompt_affinity_max_windows: usize,
}

impl Default for MiniTransformerSwarmRouteConfig {
    fn default() -> Self {
        Self {
            required_capabilities: Vec::new(),
            max_artifact_bytes: None,
            max_parameter_bytes: None,
            active_expert_limit: 1,
            prompt_affinity: false,
            prompt_affinity_max_windows: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRouteCandidate {
    pub expert_id: String,
    pub manifest: MiniTransformerMlpSwarmExpertManifest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRoutedGenerationExpert {
    pub expert_id: String,
    pub model: MiniTransformerMlpSwarmModel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRouteDecisionTrace {
    pub config: MiniTransformerSwarmRouteConfig,
    pub prompt_bytes: Vec<u8>,
    pub selected_expert_indices: Vec<usize>,
    pub candidates: Vec<MiniTransformerSwarmRouteCandidateTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRouteCandidateTrace {
    pub expert_index: usize,
    pub expert_id: String,
    pub accepted: bool,
    pub reject_reason: &'static str,
    pub score: i64,
    pub manifest_score: i64,
    pub prompt_affinity_score: i64,
    pub prompt_eval_windows: usize,
    pub prompt_probability_error_q15: Option<usize>,
    pub capability_match: bool,
    pub matched_capabilities: Vec<String>,
    pub missing_capabilities: Vec<String>,
    pub model_hash: u64,
    pub artifact_bytes: usize,
    pub parameter_bytes: usize,
    pub worker_count: usize,
    pub context_seq_len: usize,
    pub default_composition: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerSwarmPromptAffinityTrace {
    eval_windows: usize,
    probability_error_q15: usize,
    score: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmRoutedGenerationTrace {
    pub route: MiniTransformerSwarmRouteDecisionTrace,
    pub selected_expert_ids: Vec<String>,
    pub active_worker_count: usize,
    pub generation: MiniTransformerSwarmGenerationTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerSwarmComposition {
    AverageLogits,
    ConfidenceWeighted,
    ConfidenceRouter,
}

impl MiniTransformerSwarmComposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AverageLogits => "average_logits",
            Self::ConfidenceWeighted => "confidence_weighted",
            Self::ConfidenceRouter => "confidence_router",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpModel {
    pub context_seq_len: usize,
    pub embeddings: Vec<i16>,
    pub position_embeddings: Vec<i16>,
    pub attention_rms_weights: Vec<i16>,
    pub mlp_rms_weights: Vec<i16>,
    pub q_weights: Vec<i8>,
    pub k_weights: Vec<i8>,
    pub v_weights: Vec<i8>,
    pub o_weights: Vec<i8>,
    pub up_weights: Vec<i8>,
    pub gate_weights: Vec<i8>,
    pub down_weights: Vec<i8>,
    pub output_weights: Vec<i8>,
}

/// A frozen-trunk, trainable i16 residual inserted after every transformer
/// block. The down projection is a deterministic sign projection, while the
/// expansion is learned in Q15. This lets many small experts retain fractional
/// updates without changing the trunk's compact i8 inference matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerBlockLowRankExpert {
    pub trunk_model_hash: u64,
    pub transformer_layers: usize,
    pub rank: usize,
    pub projection_seed: u64,
    pub residual_shift: u8,
    pub expansion_weights_q15: Vec<i16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerBlockExpertMetrics {
    pub windows: usize,
    pub mistakes: usize,
    pub probability_error_q15: usize,
    pub hidden_saturation_count: usize,
}

impl MiniTransformerBlockExpertMetrics {
    pub fn accuracy_per_mille(self) -> usize {
        self.windows.saturating_sub(self.mistakes) * 1000 / self.windows.max(1)
    }

    pub fn mean_probability_error_q15(self) -> usize {
        self.probability_error_q15 / self.windows.max(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerBlockExpertTrainStats {
    pub optimizer_steps: usize,
    pub accepted_forward_steps: usize,
    pub accepted_reverse_steps: usize,
    pub rejected_steps: usize,
    pub weight_delta_l1: u64,
    pub weight_saturation_count: usize,
    pub hidden_saturation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerBlockExpertObjective {
    CrossEntropy,
    ProbabilityError,
}

impl MiniTransformerBlockExpertObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CrossEntropy => "cross_entropy",
            Self::ProbabilityError => "probability_error",
        }
    }
}

/// Versioned optimizer state stored separately from inference weights.
///
/// The moment vectors use one stable flat parameter order: token embeddings,
/// position embeddings, Q, K, V, O, MLP up, gate, down, then output weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerAdamOptimizerState {
    pub context_seq_len: usize,
    pub step: u64,
    pub bound_model_hash: u64,
    pub config: IntegerAdamConfig,
    pub first_moments: Vec<i64>,
    pub second_moments: Vec<u64>,
    pub update_residuals: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerAttentionKind {
    Base2Softmax,
    Linear,
    LinearStreamingNope,
    LinearStreamingTttNope,
}

impl MiniTransformerAttentionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base2Softmax => "base2_softmax",
            Self::Linear => "linear",
            Self::LinearStreamingNope => "linear_streaming_nope",
            Self::LinearStreamingTttNope => "linear_streaming_ttt_nope",
        }
    }

    pub fn uses_incremental_state(self) -> bool {
        matches!(
            self,
            Self::LinearStreamingNope | Self::LinearStreamingTttNope
        )
    }

    fn preferred_generation_kind(self, position_policy: MiniTransformerPositionPolicy) -> Self {
        if self == Self::Linear && position_policy == MiniTransformerPositionPolicy::Nope {
            Self::LinearStreamingNope
        } else {
            self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniTransformerTraceDetail {
    Full,
    Summary,
    None,
}

impl MiniTransformerTraceDetail {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Summary => "summary",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteGenerationConfig {
    pub max_new_tokens: usize,
    pub tokenizer_id: ByteTokenizerId,
    pub decode: DecodeConfig,
}

impl ByteGenerationConfig {
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            tokenizer_id: ByteTokenizerId::Identity,
            decode: DecodeConfig::greedy(),
        }
    }

    pub fn deterministic_sample(max_new_tokens: usize, sample_seed: u64, top_k: usize) -> Self {
        Self {
            max_new_tokens,
            tokenizer_id: ByteTokenizerId::Identity,
            decode: DecodeConfig {
                strategy: DecodeStrategy::Sample,
                sample_seed,
                top_k,
                ..DecodeConfig::greedy()
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteTokenizerId {
    Identity,
    AsciiLowerText,
}

impl ByteTokenizerId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identity => BYTE_TOKENIZER_ID,
            Self::AsciiLowerText => ASCII_LOWER_TOKENIZER_ID,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeConfig {
    pub strategy: DecodeStrategy,
    pub sample_seed: u64,
    pub top_k: usize,
    pub printable_only: bool,
    pub ascii_lower_only: bool,
    pub repeat_window: usize,
    pub repeat_penalty_shift: u8,
    pub max_repeat_run: usize,
    pub no_repeat_ngram_order: usize,
    pub corpus_prior: bool,
    pub corpus_prior_logit_shift: u8,
    pub corpus_prior_order: u8,
    pub frequency_penalty_cap: u32,
    pub frequency_penalty_min_weight_q15: i16,
    pub frequency_penalty_logit_shift: u8,
    pub local_frequency_penalty_cap: usize,
    pub local_frequency_penalty_min_weight_q15: i16,
    pub local_frequency_penalty_logit_shift: u8,
    pub local_frequency_hard_cap: usize,
    pub island_penalty_count_cap: u32,
    pub island_penalty_min_degree: usize,
    pub island_penalty_min_weight_q15: i16,
    pub island_penalty_logit_shift: u8,
    pub prompt_topic_radius: usize,
    pub prompt_topic_min_weight_q15: i16,
    pub prompt_topic_strict_min_weight_q15: i16,
    pub prompt_topic_logit_shift: u8,
    pub memory_context_order: u8,
    pub memory_min_context_order: u8,
    pub memory_logit_shift: u8,
    pub strict_memory_on_steps: usize,
    pub strict_memory_off_steps: usize,
    pub strict_memory: bool,
    pub strict_topic: bool,
    pub strict_adjacency: bool,
    pub banned_token_count: usize,
    pub banned_tokens: [u16; LEXEME_DECODE_TOKEN_SET_CAP],
    pub function_word_token_count: usize,
    pub function_word_tokens: [u16; LEXEME_DECODE_TOKEN_SET_CAP],
    pub function_word_run_cap: usize,
}

impl DecodeConfig {
    pub fn greedy() -> Self {
        Self {
            strategy: DecodeStrategy::Greedy,
            sample_seed: 0,
            top_k: 0,
            printable_only: false,
            ascii_lower_only: false,
            repeat_window: 0,
            repeat_penalty_shift: 0,
            max_repeat_run: 0,
            no_repeat_ngram_order: 0,
            corpus_prior: false,
            corpus_prior_logit_shift: DEFAULT_CORPUS_PRIOR_LOGIT_SHIFT,
            corpus_prior_order: DEFAULT_CORPUS_PRIOR_ORDER,
            frequency_penalty_cap: 0,
            frequency_penalty_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            frequency_penalty_logit_shift: 8,
            local_frequency_penalty_cap: 0,
            local_frequency_penalty_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            local_frequency_penalty_logit_shift: 6,
            local_frequency_hard_cap: 0,
            island_penalty_count_cap: 0,
            island_penalty_min_degree: 8,
            island_penalty_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            island_penalty_logit_shift: 6,
            prompt_topic_radius: 0,
            prompt_topic_min_weight_q15: DEFAULT_LEXEME_FREQUENCY_WEIGHT_MIN_Q15,
            prompt_topic_strict_min_weight_q15: 0,
            prompt_topic_logit_shift: 6,
            memory_context_order: 0,
            memory_min_context_order: 1,
            memory_logit_shift: DEFAULT_LEXEME_MEMORY_LOGIT_SHIFT,
            strict_memory_on_steps: 0,
            strict_memory_off_steps: 0,
            strict_memory: false,
            strict_topic: false,
            strict_adjacency: false,
            banned_token_count: 0,
            banned_tokens: [0; LEXEME_DECODE_TOKEN_SET_CAP],
            function_word_token_count: 0,
            function_word_tokens: [0; LEXEME_DECODE_TOKEN_SET_CAP],
            function_word_run_cap: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStrategy {
    Greedy,
    Sample,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteDecodePriors {
    pub token_count: usize,
    pub token_hash: u64,
    unigram_counts: [u32; BYTE_VOCAB],
    bigram_counts: Vec<u32>,
    row_totals: [u32; BYTE_VOCAB],
    observed_bigrams: usize,
}

impl ByteDecodePriors {
    pub fn from_tokens(tokens: &[u8]) -> Result<Self, TrainError> {
        if tokens.len() < 2 {
            return Err(TrainError::InvalidConfig);
        }

        let mut unigram_counts = [0_u32; BYTE_VOCAB];
        let mut bigram_counts = vec![0_u32; BYTE_VOCAB * BYTE_VOCAB];
        let mut row_totals = [0_u32; BYTE_VOCAB];
        let mut observed_bigrams = 0_usize;

        for &token in tokens.iter() {
            let slot = usize::from(token);
            unigram_counts[slot] = unigram_counts[slot].saturating_add(1);
        }
        for pair in tokens.windows(2) {
            let previous = usize::from(pair[0]);
            let next = usize::from(pair[1]);
            let index = previous * BYTE_VOCAB + next;
            if bigram_counts[index] == 0 {
                observed_bigrams = observed_bigrams.saturating_add(1);
            }
            bigram_counts[index] = bigram_counts[index].saturating_add(1);
            row_totals[previous] = row_totals[previous].saturating_add(1);
        }

        Ok(Self {
            token_count: tokens.len(),
            token_hash: hash_u8_slice(tokens),
            unigram_counts,
            bigram_counts,
            row_totals,
            observed_bigrams,
        })
    }

    pub fn observed_bigrams(&self) -> usize {
        self.observed_bigrams
    }

    pub fn transition_count(&self, previous: u8, next: u8) -> u32 {
        self.bigram_counts[usize::from(previous) * BYTE_VOCAB + usize::from(next)]
    }

    pub fn allows_transition(&self, previous: u8, next: u8) -> bool {
        self.row_totals[usize::from(previous)] == 0 || self.transition_count(previous, next) > 0
    }

    pub fn transition_probability_q15(&self, previous: u8, next: u8) -> u16 {
        let row_total = self.row_totals[usize::from(previous)];
        if row_total > 0 {
            probability_q15(self.transition_count(previous, next), row_total)
        } else {
            probability_q15(
                self.unigram_counts[usize::from(next)],
                u32::try_from(self.token_count).unwrap_or(u32::MAX),
            )
        }
    }

    fn trace(&self) -> ByteDecodePriorTrace {
        ByteDecodePriorTrace {
            token_count: self.token_count,
            token_hash: self.token_hash,
            observed_bigrams: self.observed_bigrams,
        }
    }
}

fn probability_q15(count: u32, total: u32) -> u16 {
    if total == 0 || count == 0 {
        return 0;
    }
    let numerator = u64::from(count) * u64::from(i16::MAX as u16) + u64::from(total / 2);
    let probability = numerator / u64::from(total);
    u16::try_from(probability.min(u64::from(i16::MAX as u16))).unwrap_or(i16::MAX as u16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteDecodePriorTrace {
    pub token_count: usize,
    pub token_hash: u64,
    pub observed_bigrams: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeRejectStats {
    pub non_printable: usize,
    pub outside_ascii_lower: usize,
    pub byte_fallback: usize,
    pub banned_token: usize,
    pub repeat_run: usize,
    pub repeat_ngram: usize,
    pub function_word_run: usize,
    pub local_frequency: usize,
    pub topic: usize,
    pub memory: usize,
    pub adjacency: usize,
    pub top_k_truncated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodeSelection {
    token: u8,
    candidate_count: usize,
    rejected_candidates: DecodeRejectStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecodeCandidateSet {
    candidates: Vec<usize>,
    rejected_candidates: DecodeRejectStats,
}

// Shared per-step generation record, reused by the mini-transformer generation
// traces below (and other decoders). Named "Byte" for the byte vocabulary it
// records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteGenerationStepTrace {
    pub step_index: usize,
    pub input_token: u8,
    pub predicted_token: u8,
    pub predicted_logit_q8: i32,
    pub predicted_probability_q15: i16,
    pub candidate_count: usize,
    pub rejected_candidates: DecodeRejectStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerGenerationTrace {
    pub config: ByteGenerationConfig,
    pub attention_kind: MiniTransformerAttentionKind,
    pub position_policy: MiniTransformerPositionPolicy,
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub model_hash: u64,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
    pub context_seq_len: usize,
    pub decode_priors: Option<ByteDecodePriorTrace>,
    pub ttt_stats: Option<MiniTransformerStreamingTttStats>,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerNextTokenRow {
    pub logits_q8: [i32; BYTE_VOCAB],
    pub probabilities_q15: [i16; BYTE_VOCAB],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerSwarmGenerationTrace {
    pub config: ByteGenerationConfig,
    pub attention_kind: MiniTransformerAttentionKind,
    pub position_policy: MiniTransformerPositionPolicy,
    pub composition: MiniTransformerSwarmComposition,
    pub prompt_bytes: Vec<u8>,
    pub generated_bytes: Vec<u8>,
    pub swarm_model_hash: u64,
    pub worker_count: usize,
    pub best_worker_index: usize,
    pub embedding_hash: u64,
    pub attention_hash: u64,
    pub mlp_hash: u64,
    pub output_head_hash: u64,
    pub context_seq_len: usize,
    pub decode_priors: Option<ByteDecodePriorTrace>,
    pub steps: Vec<ByteGenerationStepTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerStreamingTttStats {
    pub learning_rate_shift: u8,
    pub step_count: usize,
    pub zero_delta_count: usize,
    pub prompt_state_delta_l1: u64,
    pub generated_state_delta_l1: u64,
    pub total_state_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTransformerMlpTrainingStepTrace {
    pub update_index: usize,
    pub epoch: usize,
    pub window_index: usize,
    pub window_start: usize,
    pub first_token: u8,
    pub last_token: u8,
    pub target_token: u8,
    pub predicted_token_before: u8,
    pub predicted_token_after: u8,
    pub target_probability_before_q15: i16,
    pub target_probability_after_q15: i16,
    pub embedding_cache_hash: u64,
    pub attention_cache_hash: u64,
    pub mlp_cache_hash: u64,
    pub block_output_hash_before: u64,
    pub block_output_hash_after: u64,
    pub embedding_hash_before: u64,
    pub embedding_hash_after: u64,
    pub output_head_hash_before: u64,
    pub output_head_hash_after: u64,
    pub mlp_hash_before: u64,
    pub mlp_hash_after: u64,
    pub attention_hash_before: u64,
    pub attention_hash_after: u64,
    pub output_head_saturation_count: usize,
    pub mlp_saturation_count: usize,
    pub embedding_saturation_count: usize,
    pub attention_saturation_count: usize,
    pub residual_saturation_count: usize,
    pub output_head_zero_delta_count: usize,
    pub mlp_zero_delta_count: usize,
    pub embedding_zero_delta_count: usize,
    pub attention_zero_delta_count: usize,
    pub output_head_delta_l1: u64,
    pub mlp_delta_l1: u64,
    pub embedding_delta_l1: u64,
    pub attention_delta_l1: u64,
    pub attention_q_delta_l1: u64,
    pub attention_k_delta_l1: u64,
    pub attention_v_delta_l1: u64,
    pub attention_o_delta_l1: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniTransformerAdaptiveShiftEventTrace {
    pub batch_index: usize,
    pub component: &'static str,
    pub reason: &'static str,
    pub previous_shift: u8,
    pub next_shift: u8,
    pub delta: i8,
    pub observation_batches: usize,
    pub rejected_batches: usize,
    pub saturation_count: usize,
    pub zero_delta_count: usize,
    pub weight_delta_l1: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerMlpForwardCache {
    embedding_output: Vec<i16>,
    layers: Vec<MiniTransformerBlockForwardCache>,
    attention_norm: Vec<i16>,
    attention_q: Vec<i16>,
    attention_k: Vec<i16>,
    attention_v: Vec<i16>,
    attention_context: Vec<i16>,
    attention_probabilities_q15: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_norm: Vec<i16>,
    mlp_up: Vec<i16>,
    mlp_gate: Vec<i16>,
    mlp_gated: Vec<i16>,
    mlp_output: Vec<i16>,
    block_output: Vec<i16>,
    output_features: [i16; MINI_TRANSFORMER_D_MODEL],
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
    residual_saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerBlockForwardCache {
    block_input: Vec<i16>,
    attention_norm: Vec<i16>,
    attention_q: Vec<i16>,
    attention_k: Vec<i16>,
    attention_v: Vec<i16>,
    attention_context: Vec<i16>,
    attention_probabilities_q15: Vec<i16>,
    attention_output: Vec<i16>,
    attention_residual: Vec<i16>,
    mlp_norm: Vec<i16>,
    mlp_up: Vec<i16>,
    mlp_gate: Vec<i16>,
    mlp_gated: Vec<i16>,
    mlp_output: Vec<i16>,
    block_output: Vec<i16>,
    residual_saturation_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerAttentionWeightUpdateStats {
    q: LinearWeightUpdateStats,
    k: LinearWeightUpdateStats,
    v: LinearWeightUpdateStats,
    o: LinearWeightUpdateStats,
    gradient_saturation_count: usize,
    zero_delta_count: usize,
    weight_delta_l1: u64,
    grad_embedding_output: Vec<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerBlockBackwardUpdate {
    mlp_update: GatedMlpWeightUpdateStats,
    attention_update: MiniTransformerAttentionWeightUpdateStats,
    mlp_input_saturation_count: usize,
    gradient_residual_saturation_count: usize,
    input_gradient_saturation_count: usize,
    grad_input: Vec<i16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MiniTransformerBlockBackwardAccumulation {
    mlp_input_saturation_count: usize,
    attention_gradient_saturation_count: usize,
    gradient_residual_saturation_count: usize,
    input_gradient_saturation_count: usize,
    grad_input: Vec<i16>,
}

fn byte_window_starts(
    token_count: usize,
    seq_len: usize,
    stride: usize,
    window_offset: usize,
    max_windows: Option<usize>,
) -> Vec<usize> {
    let mut starts = Vec::new();
    if seq_len == 0 || stride == 0 {
        return starts;
    }

    let mut start = window_offset;
    while start
        .checked_add(seq_len)
        .is_some_and(|target_index| target_index < token_count)
    {
        if max_windows.is_some_and(|limit| starts.len() >= limit) {
            break;
        }
        starts.push(start);
        start = start.saturating_add(stride);
    }
    starts
}

fn mini_transformer_window_starts(
    token_count: usize,
    seq_len: usize,
    stride: usize,
    window_offset: usize,
    max_windows: Option<usize>,
) -> Vec<usize> {
    if seq_len == 0 || stride == 0 || window_offset >= token_count {
        return Vec::new();
    }
    let Some(last_start) = token_count
        .checked_sub(seq_len)
        .and_then(|value| value.checked_sub(1))
    else {
        return Vec::new();
    };
    if window_offset > last_start {
        return Vec::new();
    }

    let available = (last_start - window_offset) / stride + 1;
    let limit = max_windows.unwrap_or(available).min(available);
    if limit == 0 {
        return Vec::new();
    }
    if limit >= available {
        return byte_window_starts(token_count, seq_len, stride, window_offset, Some(limit));
    }
    if limit == 1 {
        return vec![window_offset];
    }

    let mut starts = Vec::with_capacity(limit);
    let numerator_max = available - 1;
    let denominator = limit - 1;
    let half = denominator / 2;
    for index in 0..limit {
        let stride_index = index.saturating_mul(numerator_max).saturating_add(half) / denominator;
        starts.push(window_offset + stride_index * stride);
    }
    starts
}

fn mini_transformer_filtered_window_starts(
    token_count: usize,
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
) -> Vec<usize> {
    if config.target_token_min == u8::MIN
        && config.target_token_max == u8::MAX
        && config.target_segment == MiniTransformerTargetSegment::All
    {
        return mini_transformer_window_starts(
            token_count,
            config.seq_len,
            config.stride,
            config.window_offset,
            config.max_windows,
        );
    }
    let mut starts = mini_transformer_window_starts(
        token_count,
        config.seq_len,
        config.stride,
        config.window_offset,
        None,
    );
    starts.retain(|&start| {
        let target_index = start.saturating_add(config.seq_len);
        tokens.get(target_index).is_some_and(|&target| {
            target >= config.target_token_min
                && target <= config.target_token_max
                && mini_transformer_target_segment_allows(
                    tokens,
                    target_index,
                    config.target_segment,
                )
        })
    });
    match config.max_windows {
        Some(limit) => spread_usize_values(starts, limit),
        None => starts,
    }
}

fn valid_mini_transformer_target_segment(segment: MiniTransformerTargetSegment) -> bool {
    match segment {
        MiniTransformerTargetSegment::All => true,
        MiniTransformerTargetSegment::AfterMarkerBeforeAny {
            end_marker_count, ..
        } => usize::from(end_marker_count) <= 4 && end_marker_count > 0,
        MiniTransformerTargetSegment::AfterSequenceBeforeAny {
            start_sequence_len,
            end_marker_count,
            ..
        }
        | MiniTransformerTargetSegment::FirstAfterSequenceBeforeAny {
            start_sequence_len,
            end_marker_count,
            ..
        } => {
            usize::from(start_sequence_len) <= 32
                && start_sequence_len > 0
                && usize::from(end_marker_count) <= 4
                && end_marker_count > 0
        }
    }
}

fn mini_transformer_target_segment_allows(
    tokens: &[u8],
    target_index: usize,
    segment: MiniTransformerTargetSegment,
) -> bool {
    match segment {
        MiniTransformerTargetSegment::All => true,
        MiniTransformerTargetSegment::AfterMarkerBeforeAny {
            start_marker,
            end_markers,
            end_marker_count,
        } => {
            let end_markers = &end_markers[..usize::from(end_marker_count)];
            if tokens
                .get(target_index)
                .is_some_and(|target| end_markers.contains(target))
            {
                return false;
            }
            let mut index = target_index;
            while index > 0 {
                index -= 1;
                let token = tokens[index];
                if token == start_marker {
                    return true;
                }
                if end_markers.contains(&token) {
                    return false;
                }
            }
            false
        }
        MiniTransformerTargetSegment::AfterSequenceBeforeAny {
            start_sequence,
            start_sequence_len,
            end_markers,
            end_marker_count,
        } => {
            let start_sequence = &start_sequence[..usize::from(start_sequence_len)];
            let end_markers = &end_markers[..usize::from(end_marker_count)];
            if tokens
                .get(target_index)
                .is_some_and(|target| end_markers.contains(target))
            {
                return false;
            }
            let mut index = target_index;
            while index > 0 {
                index -= 1;
                if end_markers.contains(&tokens[index]) {
                    return false;
                }
                let Some(sequence_start) = index.checked_add(1).and_then(|end| {
                    if end >= start_sequence.len() {
                        Some(end - start_sequence.len())
                    } else {
                        None
                    }
                }) else {
                    continue;
                };
                if tokens.get(sequence_start..=index) == Some(start_sequence) {
                    return true;
                }
            }
            false
        }
        MiniTransformerTargetSegment::FirstAfterSequenceBeforeAny {
            start_sequence,
            start_sequence_len,
            end_markers,
            end_marker_count,
        } => {
            let start_sequence = &start_sequence[..usize::from(start_sequence_len)];
            let end_markers = &end_markers[..usize::from(end_marker_count)];
            if tokens
                .get(target_index)
                .is_none_or(|target| end_markers.contains(target))
            {
                return false;
            }
            let Some(sequence_start) = target_index.checked_sub(start_sequence.len()) else {
                return false;
            };
            tokens.get(sequence_start..target_index) == Some(start_sequence)
        }
    }
}

fn spread_usize_values(values: Vec<usize>, limit: usize) -> Vec<usize> {
    let available = values.len();
    if limit >= available {
        return values;
    }
    if limit == 0 {
        return Vec::new();
    }
    if limit == 1 {
        return values.first().copied().into_iter().collect();
    }

    let mut spread = Vec::with_capacity(limit);
    let numerator_max = available - 1;
    let denominator = limit - 1;
    let half = denominator / 2;
    for index in 0..limit {
        let source_index = index.saturating_mul(numerator_max).saturating_add(half) / denominator;
        spread.push(values[source_index]);
    }
    spread
}

fn mini_transformer_loss_guard_starts(
    starts: &[usize],
    batch_start_index: usize,
    batch_end_index: usize,
) -> Vec<usize> {
    const GLOBAL_GUARD_POINTS: usize = 16;

    let mut guarded = Vec::with_capacity(
        GLOBAL_GUARD_POINTS.saturating_add(batch_end_index.saturating_sub(batch_start_index)),
    );
    if starts.is_empty() {
        return guarded;
    }

    let end = batch_end_index.min(starts.len());
    for &start in &starts[batch_start_index.min(end)..end] {
        push_unique_usize(&mut guarded, start);
    }

    let len = starts.len();
    let points = GLOBAL_GUARD_POINTS.min(len);
    if points == 1 {
        push_unique_usize(&mut guarded, starts[0]);
        return guarded;
    }

    let numerator_max = len - 1;
    let denominator = points - 1;
    let half = denominator / 2;
    for point in 0..points {
        let index = point.saturating_mul(numerator_max).saturating_add(half) / denominator;
        push_unique_usize(&mut guarded, starts[index]);
    }
    guarded
}

fn mini_transformer_loss_guard_regressed(
    before_loss: usize,
    after_loss: usize,
    guard_windows: usize,
) -> bool {
    const PER_WINDOW_TOLERANCE_Q15: usize = 1024;

    let tolerance = guard_windows.saturating_mul(PER_WINDOW_TOLERANCE_Q15);
    after_loss > before_loss.saturating_add(tolerance)
}

fn push_unique_usize(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn mini_transformer_batch_learning_rate_shift(batch_windows: usize) -> Option<u8> {
    if batch_windows == 0 {
        return None;
    }

    let mut shift = 0_u8;
    let mut covered = 1_usize;
    while covered < batch_windows {
        covered = covered.checked_mul(2)?;
        shift = shift.checked_add(1)?;
    }

    Some(shift)
}

fn mini_transformer_component_shift_for_effective_batch_shift(
    effective_shift: u8,
    batch_windows: usize,
) -> Result<u8, TrainError> {
    let batch_shift = mini_transformer_batch_learning_rate_shift(batch_windows)
        .ok_or(TrainError::InvalidConfig)?;
    effective_shift
        .checked_sub(batch_shift)
        .ok_or(TrainError::InvalidConfig)
}

fn mini_transformer_batch_component_shift_config(
    mut config: MiniTransformerMlpTrainConfig,
    batch_windows: usize,
) -> Result<MiniTransformerMlpTrainConfig, TrainError> {
    config.output_learning_rate_shift = mini_transformer_component_shift_for_effective_batch_shift(
        config.output_learning_rate_shift,
        batch_windows,
    )?;
    config.mlp_learning_rate_shift = mini_transformer_component_shift_for_effective_batch_shift(
        config.mlp_learning_rate_shift,
        batch_windows,
    )?;
    config.embedding_learning_rate_shift =
        mini_transformer_component_shift_for_effective_batch_shift(
            config.embedding_learning_rate_shift,
            batch_windows,
        )?;
    config.attention_learning_rate_shift =
        mini_transformer_component_shift_for_effective_batch_shift(
            config.attention_learning_rate_shift,
            batch_windows,
        )?;
    config.attention_q_learning_rate_shift =
        mini_transformer_component_shift_for_effective_batch_shift(
            config.attention_q_learning_rate_shift,
            batch_windows,
        )?;
    config.attention_qk_learning_rate_shift =
        mini_transformer_component_shift_for_effective_batch_shift(
            config.attention_qk_learning_rate_shift,
            batch_windows,
        )?;
    Ok(config)
}

fn validate_mini_transformer_effective_learning_rate_shifts(
    config: MiniTransformerMlpTrainConfig,
) -> Result<(), TrainError> {
    mini_transformer_batch_component_shift_config(config, config.batch_windows).map(|_| ())
}

fn initial_mini_transformer_embeddings() -> Vec<i16> {
    let mut embeddings = Vec::with_capacity(BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL);
    for token in 0..BYTE_VOCAB {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            #[cfg(not(feature = "mini-calibrated"))]
            let bucket = ((token * 29 + dim * 13 + 5) % 33) as i32 - 16;
            #[cfg(feature = "mini-calibrated")]
            let bucket = calibrated_initial_bucket(
                token * MINI_TRANSFORMER_D_MODEL + dim,
                0x6a09_e667_f3bc_c909,
                16,
            );
            embeddings.push((bucket * 32) as i16);
        }
    }
    embeddings
}

#[cfg(feature = "mini-calibrated")]
fn calibrated_initial_bucket(index: usize, seed: u64, radius: u64) -> i32 {
    let mut value = (index as u64)
        .wrapping_add(seed)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value % (radius * 2 + 1)) as i32 - radius as i32
}

fn initial_mini_transformer_position_embeddings(context_seq_len: usize) -> Vec<i16> {
    let mut embeddings = Vec::with_capacity(context_seq_len * MINI_TRANSFORMER_D_MODEL);
    for position in 0..context_seq_len {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            embeddings.push(mini_transformer_position_signal_q15(position, dim));
        }
    }
    embeddings
}

fn mini_transformer_attention_weight_count() -> Result<usize, TrainError> {
    MINI_TRANSFORMER_D_MODEL
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)
}

fn mini_transformer_mlp_up_or_gate_weight_count() -> Result<usize, TrainError> {
    MINI_TRANSFORMER_D_MODEL
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)
}

fn mini_transformer_mlp_down_weight_count() -> Result<usize, TrainError> {
    MINI_TRANSFORMER_HIDDEN_DIM
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)
}

fn infer_layer_count(weight_len: usize, per_layer_len: usize) -> Option<usize> {
    if weight_len == 0 || per_layer_len == 0 || !weight_len.is_multiple_of(per_layer_len) {
        return None;
    }
    Some(weight_len / per_layer_len)
}

fn mini_transformer_layer_range(
    layer_index: usize,
    per_layer_len: usize,
) -> Result<Range<usize>, TrainError> {
    let start = layer_index
        .checked_mul(per_layer_len)
        .ok_or(TrainError::InvalidConfig)?;
    let end = start
        .checked_add(per_layer_len)
        .ok_or(TrainError::InvalidConfig)?;
    Ok(start..end)
}

fn stack_i8_layers_with_active_final(
    inactive_layer: Vec<i8>,
    active_final_layer: Vec<i8>,
    layers: usize,
) -> Vec<i8> {
    let mut weights = Vec::with_capacity(active_final_layer.len().saturating_mul(layers));
    for _ in 1..layers {
        weights.extend_from_slice(&inactive_layer);
    }
    weights.extend_from_slice(&active_final_layer);
    weights
}

fn identity_i8_matrix(dim: usize) -> Vec<i8> {
    let mut weights = vec![0_i8; dim * dim];
    for index in 0..dim {
        #[cfg(not(feature = "mini-calibrated"))]
        let value = 1;
        #[cfg(feature = "mini-calibrated")]
        let value = 2;
        weights[index * dim + index] = value;
    }
    weights
}

fn initial_mini_transformer_mlp_up_weights() -> Vec<i8> {
    initial_mini_transformer_mlp_up_or_gate_weights(0xbb67_ae85_84ca_a73b)
}

fn initial_mini_transformer_mlp_gate_weights() -> Vec<i8> {
    #[cfg(not(feature = "mini-calibrated"))]
    let seed = 0xbb67_ae85_84ca_a73b;
    #[cfg(feature = "mini-calibrated")]
    let seed = 0x510e_527f_ade6_82d1;
    initial_mini_transformer_mlp_up_or_gate_weights(seed)
}

fn initial_mini_transformer_mlp_up_or_gate_weights(seed: u64) -> Vec<i8> {
    #[cfg(not(feature = "mini-calibrated"))]
    let _ = seed;
    let mut weights = Vec::with_capacity(MINI_TRANSFORMER_D_MODEL * MINI_TRANSFORMER_HIDDEN_DIM);
    for hidden in 0..MINI_TRANSFORMER_HIDDEN_DIM {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            #[cfg(not(feature = "mini-calibrated"))]
            let value = ((hidden * 7 + dim * 11 + 3) % 5) as i32 - 2;
            #[cfg(feature = "mini-calibrated")]
            let value = calibrated_initial_bucket(hidden * MINI_TRANSFORMER_D_MODEL + dim, seed, 2);
            weights.push(value as i8);
        }
    }
    weights
}

fn initial_mini_transformer_mlp_down_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(MINI_TRANSFORMER_HIDDEN_DIM * MINI_TRANSFORMER_D_MODEL);
    for dim in 0..MINI_TRANSFORMER_D_MODEL {
        for hidden in 0..MINI_TRANSFORMER_HIDDEN_DIM {
            #[cfg(not(feature = "mini-calibrated"))]
            let value = ((dim * 17 + hidden * 5 + 1) % 5) as i32 - 2;
            #[cfg(feature = "mini-calibrated")]
            let value = calibrated_initial_bucket(
                dim * MINI_TRANSFORMER_HIDDEN_DIM + hidden,
                0x3c6e_f372_fe94_f82b,
                2,
            );
            weights.push(value as i8);
        }
    }
    weights
}

fn initial_mini_transformer_output_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL);
    for class_id in 0..BYTE_VOCAB {
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            #[cfg(not(feature = "mini-calibrated"))]
            let value = ((class_id * 19 + dim * 23 + 7) % 7) as i32 - 3;
            #[cfg(feature = "mini-calibrated")]
            let value = calibrated_initial_bucket(
                class_id * MINI_TRANSFORMER_D_MODEL + dim,
                0xa54f_f53a_5f1d_36f1,
                3,
            );
            weights.push(value as i8);
        }
    }
    weights
}

fn mini_transformer_embedding_sequence_with_position_policy_q15(
    embeddings: &[i16],
    position_embeddings: &[i16],
    context: &[u8],
    position_policy: MiniTransformerPositionPolicy,
) -> Result<Vec<i16>, TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL || context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && position_embeddings.len() < context.len() * MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut output = Vec::with_capacity(context.len() * MINI_TRANSFORMER_D_MODEL);
    for (position, &token) in context.iter().enumerate() {
        let row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        let position_start = position * MINI_TRANSFORMER_D_MODEL;
        let row = embeddings
            .get(row_start..row_start + MINI_TRANSFORMER_D_MODEL)
            .ok_or(TrainError::InvalidModel("mini transformer embedding row"))?;
        if position_policy.uses_position_embeddings() {
            let position_row = position_embeddings
                .get(position_start..position_start + MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidModel(
                    "mini transformer position embedding row",
                ))?;
            for (&value, &position_value) in row.iter().zip(position_row.iter()) {
                output.push(saturate_i16(i64::from(value) + i64::from(position_value)));
            }
        } else {
            output.extend_from_slice(row);
        }
    }
    Ok(output)
}

fn mini_transformer_position_signal_q15(position: usize, dim: usize) -> i16 {
    let pos = position.wrapping_add(1);
    let axis = dim.wrapping_add(1);
    let mixed = pos
        .wrapping_mul(31)
        .wrapping_add(axis.wrapping_mul(17))
        .wrapping_add(pos.wrapping_mul(axis).wrapping_mul(7));
    let bucket = (mixed % 17) as i32 - 8;
    (bucket * 128) as i16
}

fn mini_transformer_forward_for_attention_and_position(
    model: &MiniTransformerMlpModel,
    context: &[u8],
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerMlpForwardCache, TrainError> {
    if context.is_empty() {
        return Err(TrainError::InvalidConfig);
    }

    let seq_len = context.len();
    let embedding_output = mini_transformer_embedding_sequence_with_position_policy_q15(
        &model.embeddings,
        &model.position_embeddings,
        context,
        position_policy,
    )?;
    let layers = model.checked_transformer_layers()?;
    let attention_weight_count = mini_transformer_attention_weight_count()?;
    let mlp_up_or_gate_count = mini_transformer_mlp_up_or_gate_weight_count()?;
    let mlp_down_count = mini_transformer_mlp_down_weight_count()?;
    let mut layer_input = embedding_output.clone();
    let mut layer_caches = Vec::with_capacity(layers);
    let mut total_residual_saturation_count = 0_usize;
    for layer_index in 0..layers {
        let attention_range = mini_transformer_layer_range(layer_index, attention_weight_count)?;
        let up_or_gate_range = mini_transformer_layer_range(layer_index, mlp_up_or_gate_count)?;
        let down_range = mini_transformer_layer_range(layer_index, mlp_down_count)?;
        let rms_range = if model.rms_norm_enabled() {
            Some(model.rms_weight_range(layer_index)?)
        } else {
            None
        };
        let block = mini_transformer_forward_block_for_attention_kind(
            &layer_input,
            rms_range
                .as_ref()
                .map(|range| &model.attention_rms_weights[range.clone()]),
            rms_range
                .as_ref()
                .map(|range| &model.mlp_rms_weights[range.clone()]),
            &model.q_weights[attention_range.clone()],
            &model.k_weights[attention_range.clone()],
            &model.v_weights[attention_range.clone()],
            &model.o_weights[attention_range],
            &model.up_weights[up_or_gate_range.clone()],
            &model.gate_weights[up_or_gate_range],
            &model.down_weights[down_range],
            attention_kind,
        )?;
        total_residual_saturation_count =
            total_residual_saturation_count.saturating_add(block.residual_saturation_count);
        layer_input = block.block_output.clone();
        layer_caches.push(block);
    }
    let final_block = layer_caches
        .last()
        .cloned()
        .ok_or(TrainError::InvalidConfig)?;
    let last_start = (seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let last_end = last_start + MINI_TRANSFORMER_D_MODEL;
    let mut output_features = [0_i16; MINI_TRANSFORMER_D_MODEL];
    output_features.copy_from_slice(&final_block.block_output[last_start..last_end]);
    let row = mini_transformer_output_row_for(&model.output_weights, &output_features)?;
    #[cfg(feature = "mini-calibrated")]
    let mut row = row;
    #[cfg(feature = "mini-calibrated")]
    if position_policy == MiniTransformerPositionPolicy::Nope {
        if let Some(predicted) =
            mini_transformer_ngram_cache_prediction(&model.position_embeddings, context)
        {
            mini_transformer_rerank_output_row(&mut row, predicted)?;
        }
    }

    Ok(MiniTransformerMlpForwardCache {
        embedding_output,
        layers: layer_caches,
        attention_norm: final_block.attention_norm,
        attention_q: final_block.attention_q,
        attention_k: final_block.attention_k,
        attention_v: final_block.attention_v,
        attention_context: final_block.attention_context,
        attention_probabilities_q15: final_block.attention_probabilities_q15,
        attention_output: final_block.attention_output,
        attention_residual: final_block.attention_residual,
        mlp_norm: final_block.mlp_norm,
        mlp_up: final_block.mlp_up,
        mlp_gate: final_block.mlp_gate,
        mlp_gated: final_block.mlp_gated,
        mlp_output: final_block.mlp_output,
        block_output: final_block.block_output,
        output_features,
        logits_q8: row.logits_q8,
        probabilities_q15: row.probabilities_q15,
        residual_saturation_count: total_residual_saturation_count,
    })
}

fn mini_transformer_rms_norm_rows(input: &[i16], weights: &[i16]) -> Result<Vec<i16>, TrainError> {
    if input.is_empty()
        || weights.len() != MINI_TRANSFORMER_D_MODEL
        || !input.len().is_multiple_of(MINI_TRANSFORMER_D_MODEL)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut output = vec![0_i16; input.len()];
    for (input_row, output_row) in input
        .chunks_exact(MINI_TRANSFORMER_D_MODEL)
        .zip(output.chunks_exact_mut(MINI_TRANSFORMER_D_MODEL))
    {
        rms_norm_i16_q15_checked(input_row, weights, MINI_TRANSFORMER_RMS_EPSILON, output_row)
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_rms_norm_forward",
            ))?;
    }
    Ok(output)
}

fn mini_transformer_rms_norm_backward_rows(
    input: &[i16],
    weights: &[i16],
    grad_output: &[i16],
    grad_input: &mut [i16],
    gradient: &mut MiniTransformerRmsVectorGradientI64,
) -> Result<usize, TrainError> {
    if input.is_empty()
        || input.len() != grad_output.len()
        || input.len() != grad_input.len()
        || weights.len() != MINI_TRANSFORMER_D_MODEL
        || gradient.accumulators.len() != MINI_TRANSFORMER_D_MODEL
        || !input.len().is_multiple_of(MINI_TRANSFORMER_D_MODEL)
    {
        return Err(TrainError::InvalidConfig);
    }
    let mut normalized = [0_i32; MINI_TRANSFORMER_D_MODEL];
    let mut scaled_grad = [0_i32; MINI_TRANSFORMER_D_MODEL];
    let mut saturation_count = 0_usize;
    let rows = input.len() / MINI_TRANSFORMER_D_MODEL;
    for row in 0..rows {
        let start = row * MINI_TRANSFORMER_D_MODEL;
        let end = start + MINI_TRANSFORMER_D_MODEL;
        saturation_count = saturation_count.saturating_add(
            rms_norm_backward_i16_q15_checked(
                &input[start..end],
                weights,
                &grad_output[start..end],
                MINI_TRANSFORMER_RMS_EPSILON,
                RmsNormBackwardWorkspace {
                    normalized_q15: &mut normalized,
                    scaled_grad_q15: &mut scaled_grad,
                },
                &mut grad_input[start..end],
                &mut gradient.accumulators,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_rms_norm_backward",
            ))?,
        );
    }
    gradient.sample_count = gradient
        .sample_count
        .checked_add(rows)
        .ok_or(TrainError::CoreRejected("RMSNorm sample count overflow"))?;
    Ok(saturation_count)
}

#[allow(clippy::too_many_arguments)]
fn mini_transformer_forward_block_for_attention_kind(
    input: &[i16],
    attention_rms_weights: Option<&[i16]>,
    mlp_rms_weights: Option<&[i16]>,
    q_weights: &[i8],
    k_weights: &[i8],
    v_weights: &[i8],
    o_weights: &[i8],
    up_weights: &[i8],
    gate_weights: &[i8],
    down_weights: &[i8],
    attention_kind: MiniTransformerAttentionKind,
) -> Result<MiniTransformerBlockForwardCache, TrainError> {
    if input.is_empty() || !input.len().is_multiple_of(MINI_TRANSFORMER_D_MODEL) {
        return Err(TrainError::InvalidConfig);
    }
    let seq_len = input.len() / MINI_TRANSFORMER_D_MODEL;
    let total = input.len();
    let hidden_total = seq_len
        .checked_mul(MINI_TRANSFORMER_HIDDEN_DIM)
        .ok_or(TrainError::InvalidConfig)?;
    if q_weights.len() != mini_transformer_attention_weight_count()?
        || k_weights.len() != q_weights.len()
        || v_weights.len() != q_weights.len()
        || o_weights.len() != q_weights.len()
        || up_weights.len() != mini_transformer_mlp_up_or_gate_weight_count()?
        || gate_weights.len() != up_weights.len()
        || down_weights.len() != mini_transformer_mlp_down_weight_count()?
        || attention_rms_weights.is_some_and(|weights| weights.len() != MINI_TRANSFORMER_D_MODEL)
        || mlp_rms_weights.is_some_and(|weights| weights.len() != MINI_TRANSFORMER_D_MODEL)
        || attention_rms_weights.is_some() != mlp_rms_weights.is_some()
    {
        return Err(TrainError::InvalidConfig);
    }

    let attention_norm = if let Some(weights) = attention_rms_weights {
        mini_transformer_rms_norm_rows(input, weights)?
    } else {
        input.to_vec()
    };
    let attention_params = SelfAttentionI16Params {
        q: LinearI16I8Params {
            weights: q_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        k: LinearI16I8Params {
            weights: k_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        v: LinearI16I8Params {
            weights: v_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        o: LinearI16I8Params {
            weights: o_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len,
        d_model: MINI_TRANSFORMER_D_MODEL,
        heads: MINI_TRANSFORMER_HEADS,
        causal: true,
    };
    let mut q = vec![0_i16; total];
    let mut k = vec![0_i16; total];
    let mut v = vec![0_i16; total];
    let mut attention_context = vec![0_i16; total];
    let mut attention_logits = vec![0_i32; seq_len];
    let mut attention_probabilities = vec![0_i16; seq_len];
    let mut attention_output = vec![0_i16; total];
    let attention_probabilities_q15 = match attention_kind {
        MiniTransformerAttentionKind::Base2Softmax => {
            self_attention_i16_q15_checked(
                &attention_norm,
                attention_params,
                SelfAttentionWorkspace {
                    q: &mut q,
                    k: &mut k,
                    v: &mut v,
                    context: &mut attention_context,
                    logits_q8: &mut attention_logits,
                    probabilities_q15: &mut attention_probabilities,
                },
                &mut attention_output,
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_forward",
            ))?;
            mini_transformer_attention_probabilities_q15(seq_len, &q, &k)?
        }
        MiniTransformerAttentionKind::Linear => {
            let head_dim = MINI_TRANSFORMER_D_MODEL
                .checked_div(MINI_TRANSFORMER_HEADS)
                .ok_or(TrainError::InvalidConfig)?;
            let state_len = MINI_TRANSFORMER_HEADS
                .checked_mul(
                    head_dim
                        .checked_mul(head_dim)
                        .ok_or(TrainError::InvalidConfig)?,
                )
                .ok_or(TrainError::InvalidConfig)?;
            let key_sum_len = MINI_TRANSFORMER_HEADS
                .checked_mul(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let mut state_kv = vec![0_i64; state_len];
            let mut key_sums = vec![0_i64; key_sum_len];
            linear_attention_i16_q15_checked(
                &attention_norm,
                attention_params,
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
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_linear_attention_forward",
            ))?;
            Vec::new()
        }
        MiniTransformerAttentionKind::LinearStreamingNope
        | MiniTransformerAttentionKind::LinearStreamingTttNope => {
            return Err(TrainError::InvalidConfig);
        }
    };

    let mut residual_saturation_count = 0_usize;
    let mut attention_residual = vec![0_i16; total];
    residual_saturation_count +=
        add_i16_residual_rows_checked(input, &attention_output, &mut attention_residual)?;
    let mlp_norm = if let Some(weights) = mlp_rms_weights {
        mini_transformer_rms_norm_rows(&attention_residual, weights)?
    } else {
        attention_residual.clone()
    };

    let mlp_params = GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    };
    let mut mlp_up = vec![0_i16; hidden_total];
    let mut mlp_gate = vec![0_i16; hidden_total];
    let mut mlp_gated = vec![0_i16; hidden_total];
    let mut mlp_output = vec![0_i16; total];
    gated_mlp_i16_q15_checked(
        &mlp_norm,
        mlp_params,
        GatedMlpWorkspace {
            up: &mut mlp_up,
            gate: &mut mlp_gate,
            gated: &mut mlp_gated,
        },
        &mut mlp_output,
    )
    .ok_or(TrainError::CoreRejected("mini_transformer_mlp_forward"))?;

    let mut block_output = vec![0_i16; total];
    residual_saturation_count +=
        add_i16_residual_rows_checked(&attention_residual, &mlp_output, &mut block_output)?;

    Ok(MiniTransformerBlockForwardCache {
        block_input: input.to_vec(),
        attention_norm,
        attention_q: q,
        attention_k: k,
        attention_v: v,
        attention_context,
        attention_probabilities_q15,
        attention_output,
        attention_residual,
        mlp_norm,
        mlp_up,
        mlp_gate,
        mlp_gated,
        mlp_output,
        block_output,
        residual_saturation_count,
    })
}

fn mini_transformer_mlp_params<'a>(
    up_weights: &'a [i8],
    gate_weights: &'a [i8],
    down_weights: &'a [i8],
    seq_len: usize,
) -> GatedMlpI16Params<'a> {
    GatedMlpI16Params {
        up: LinearI16I8Params {
            weights: up_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        gate: LinearI16I8Params {
            weights: gate_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_HIDDEN_SCALES,
            input_dim: MINI_TRANSFORMER_D_MODEL,
            output_dim: MINI_TRANSFORMER_HIDDEN_DIM,
        },
        down: LinearI16I8Params {
            weights: down_weights,
            bias: None,
            scales: &MINI_TRANSFORMER_D_MODEL_SCALES,
            input_dim: MINI_TRANSFORMER_HIDDEN_DIM,
            output_dim: MINI_TRANSFORMER_D_MODEL,
        },
        seq_len,
        d_model: MINI_TRANSFORMER_D_MODEL,
        hidden_dim: MINI_TRANSFORMER_HIDDEN_DIM,
    }
}

fn mini_transformer_final_mlp_params(
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<GatedMlpI16Params<'_>, TrainError> {
    let layers = model.checked_transformer_layers()?;
    mini_transformer_mlp_params_for_layer(model, layers - 1, seq_len)
}

fn mini_transformer_mlp_params_for_layer(
    model: &MiniTransformerMlpModel,
    layer_index: usize,
    seq_len: usize,
) -> Result<GatedMlpI16Params<'_>, TrainError> {
    let up_or_gate_range = model.mlp_up_or_gate_weight_range(layer_index)?;
    let down_range = model.mlp_down_weight_range(layer_index)?;
    Ok(mini_transformer_mlp_params(
        &model.up_weights[up_or_gate_range.clone()],
        &model.gate_weights[up_or_gate_range],
        &model.down_weights[down_range],
        seq_len,
    ))
}

fn mini_transformer_attention_probabilities_q15(
    seq_len: usize,
    q: &[i16],
    k: &[i16],
) -> Result<Vec<i16>, TrainError> {
    let head_dim = mini_transformer_head_dim()?;
    let total = seq_len
        .checked_mul(MINI_TRANSFORMER_D_MODEL)
        .ok_or(TrainError::InvalidConfig)?;
    if q.len() != total || k.len() != total {
        return Err(TrainError::InvalidConfig);
    }

    let mut probabilities = vec![0_i16; mini_transformer_attention_probability_count(seq_len)?];
    let mut logits = vec![0_i32; seq_len];
    for head in 0..MINI_TRANSFORMER_HEADS {
        let head_offset = head
            .checked_mul(head_dim)
            .ok_or(TrainError::InvalidConfig)?;
        for query_index in 0..seq_len {
            let query_start = query_index
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .and_then(|value| value.checked_add(head_offset))
                .ok_or(TrainError::InvalidConfig)?;
            let query_end = query_start
                .checked_add(head_dim)
                .ok_or(TrainError::InvalidConfig)?;
            let visible_len = query_index + 1;
            for (key_index, logit) in logits.iter_mut().enumerate().take(visible_len) {
                let key_start = key_index
                    .checked_mul(MINI_TRANSFORMER_D_MODEL)
                    .and_then(|value| value.checked_add(head_offset))
                    .ok_or(TrainError::InvalidConfig)?;
                let key_end = key_start
                    .checked_add(head_dim)
                    .ok_or(TrainError::InvalidConfig)?;
                *logit = attention_dot_q_k_i16_i32_checked(
                    &q[query_start..query_end],
                    &k[key_start..key_end],
                )
                .ok_or(TrainError::CoreRejected(
                    "mini_transformer_attention_probability_logits",
                ))?;
            }

            let prob_start =
                mini_transformer_attention_probability_row_start(head, query_index, seq_len)?;
            let prob_end = prob_start
                .checked_add(seq_len)
                .ok_or(TrainError::InvalidConfig)?;
            let visible_prob_end = prob_start
                .checked_add(visible_len)
                .ok_or(TrainError::InvalidConfig)?;
            base2_softmax_i32_q15(
                &logits[..visible_len],
                &mut probabilities[prob_start..visible_prob_end],
            )
            .ok_or(TrainError::CoreRejected(
                "mini_transformer_attention_probability_softmax",
            ))?;
            probabilities[visible_prob_end..prob_end].fill(0);
        }
    }

    Ok(probabilities)
}

fn mini_transformer_head_dim() -> Result<usize, TrainError> {
    if MINI_TRANSFORMER_HEADS == 0
        || MINI_TRANSFORMER_D_MODEL == 0
        || !MINI_TRANSFORMER_D_MODEL.is_multiple_of(MINI_TRANSFORMER_HEADS)
    {
        return Err(TrainError::InvalidConfig);
    }
    Ok(MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_HEADS)
}

fn mini_transformer_attention_probability_count(seq_len: usize) -> Result<usize, TrainError> {
    if seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }
    MINI_TRANSFORMER_HEADS
        .checked_mul(seq_len)
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or(TrainError::InvalidConfig)
}

fn mini_transformer_attention_probability_row_start(
    head: usize,
    query_index: usize,
    seq_len: usize,
) -> Result<usize, TrainError> {
    if head >= MINI_TRANSFORMER_HEADS || query_index >= seq_len {
        return Err(TrainError::InvalidConfig);
    }
    head.checked_mul(seq_len)
        .and_then(|value| value.checked_add(query_index))
        .and_then(|value| value.checked_mul(seq_len))
        .ok_or(TrainError::InvalidConfig)
}

fn mini_transformer_stacked_layer_runtime_config(
    mut config: MiniTransformerMlpTrainConfig,
    layer_index: usize,
    layer_count: usize,
) -> MiniTransformerMlpTrainConfig {
    config.mlp_learning_rate_shift = config
        .mlp_learning_rate_shift
        .saturating_add(MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT)
        .min(MAX_RIGHT_SHIFT);
    config.attention_learning_rate_shift = config
        .attention_learning_rate_shift
        .saturating_add(MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT)
        .min(MAX_RIGHT_SHIFT);
    config.attention_q_learning_rate_shift = config
        .attention_q_learning_rate_shift
        .saturating_add(MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT)
        .min(MAX_RIGHT_SHIFT);
    config.attention_qk_learning_rate_shift = config
        .attention_qk_learning_rate_shift
        .saturating_add(MINI_TRANSFORMER_STACKED_BLOCK_LEARNING_RATE_EXTRA_SHIFT)
        .min(MAX_RIGHT_SHIFT);

    if layer_index + 1 < layer_count {
        config.mlp_learning_rate_shift = config
            .mlp_learning_rate_shift
            .saturating_add(MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT)
            .min(MAX_RIGHT_SHIFT);
        config.attention_learning_rate_shift = config
            .attention_learning_rate_shift
            .saturating_add(MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT)
            .min(MAX_RIGHT_SHIFT);
        config.attention_q_learning_rate_shift = config
            .attention_q_learning_rate_shift
            .saturating_add(MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT)
            .min(MAX_RIGHT_SHIFT);
        config.attention_qk_learning_rate_shift = config
            .attention_qk_learning_rate_shift
            .saturating_add(MINI_TRANSFORMER_STACKED_LOWER_LAYER_LEARNING_RATE_EXTRA_SHIFT)
            .min(MAX_RIGHT_SHIFT);
    }
    config
}

#[cfg(feature = "mini-calibrated")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerNgramCacheRecord {
    order: u8,
    key: u32,
    predicted: u8,
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_suffix_key(bytes: &[u8], end: usize, order: usize) -> Option<u32> {
    if order == 0 || order > MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER || end < order {
        return None;
    }
    Some(
        bytes[end - order..end]
            .iter()
            .fold(0_u32, |key, &byte| (key << 8) | u32::from(byte)),
    )
}

#[cfg(feature = "mini-calibrated")]
#[allow(dead_code)]
fn mini_transformer_install_frontcoded_ngram_cache(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
) -> Result<usize, TrainError> {
    let byte_capacity = model
        .position_embeddings
        .len()
        .checked_mul(core::mem::size_of::<i16>())
        .ok_or(TrainError::InvalidConfig)?;
    if byte_capacity < MINI_TRANSFORMER_NGRAM_CACHE_HEADER_BYTES {
        return Ok(0);
    }
    let mut records = Vec::new();
    for order in 1..=MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER {
        let mut observations = Vec::with_capacity(tokens.len().saturating_sub(order));
        for target_index in order..tokens.len() {
            let key = mini_transformer_suffix_key(tokens, target_index, order)
                .ok_or(TrainError::InvalidConfig)?;
            observations.push((key, tokens[target_index]));
        }
        observations.sort_unstable();
        let mut start = 0_usize;
        while start < observations.len() {
            let key = observations[start].0;
            let mut target_counts = [0_u32; BYTE_VOCAB];
            let mut end = start;
            while end < observations.len() && observations[end].0 == key {
                let target = usize::from(observations[end].1);
                target_counts[target] = target_counts[target].saturating_add(1);
                end += 1;
            }
            let predicted = target_counts
                .iter()
                .enumerate()
                .max_by_key(|&(target, count)| (*count, core::cmp::Reverse(target)))
                .map(|(target, _)| target as u8)
                .ok_or(TrainError::InvalidConfig)?;
            records.push(MiniTransformerNgramCacheRecord {
                order: order as u8,
                key,
                predicted,
            });
            start = end;
        }
    }
    records.sort_unstable_by_key(|record| (record.order, record.key));

    let mut packed = vec![0_u8; byte_capacity];
    packed[..MINI_TRANSFORMER_NGRAM_CACHE_MAGIC.len()]
        .copy_from_slice(&MINI_TRANSFORMER_NGRAM_CACHE_MAGIC);
    let mut present = [false; BYTE_VOCAB];
    for &token in tokens {
        present[usize::from(token)] = true;
    }
    let alphabet: Vec<u8> = present
        .iter()
        .enumerate()
        .filter_map(|(token, &is_present)| is_present.then_some(token as u8))
        .collect();
    if alphabet.is_empty() {
        return Err(TrainError::InvalidConfig);
    }
    let alphabet_len = u16::try_from(alphabet.len()).map_err(|_| TrainError::InvalidConfig)?;
    packed[8..10].copy_from_slice(&alphabet_len.to_le_bytes());
    let symbol_bits =
        (usize::BITS - (alphabet.len().saturating_sub(1)).leading_zeros()).max(1) as u8;
    packed[10] = symbol_bits;
    packed[12..12 + alphabet.len()].copy_from_slice(&alphabet);
    let mut symbol_codes = [0_u8; BYTE_VOCAB];
    for (code, &token) in alphabet.iter().enumerate() {
        symbol_codes[usize::from(token)] = code as u8;
    }

    let mut bit_cursor = MINI_TRANSFORMER_NGRAM_CACHE_HEADER_BYTES * 8;
    for order in 1..=MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER {
        let order_records: Vec<_> = records
            .iter()
            .filter(|record| usize::from(record.order) == order)
            .collect();
        let count = order_records.len();
        let count = u16::try_from(count).map_err(|_| TrainError::InvalidConfig)?;
        let offset = 268 + (order - 1) * 2;
        packed[offset..offset + 2].copy_from_slice(&count.to_le_bytes());
        let bit_offset = u32::try_from(bit_cursor).map_err(|_| TrainError::InvalidConfig)?;
        let offset = 276 + (order - 1) * 4;
        packed[offset..offset + 4].copy_from_slice(&bit_offset.to_le_bytes());
        let prefix_bits = (usize::BITS - order.leading_zeros()).max(1) as u8;
        let mut previous_key = 0_u32;
        let mut has_previous = false;
        for record in order_records {
            let common_prefix = if has_previous {
                (0..order)
                    .take_while(|&index| {
                        mini_transformer_ngram_key_byte(record.key, order, index)
                            == mini_transformer_ngram_key_byte(previous_key, order, index)
                    })
                    .count()
            } else {
                0
            };
            mini_transformer_ngram_cache_push_bits(
                &mut packed,
                &mut bit_cursor,
                common_prefix as u32,
                prefix_bits,
            )?;
            mini_transformer_ngram_cache_push_bits(
                &mut packed,
                &mut bit_cursor,
                u32::from(symbol_codes[usize::from(record.predicted)]),
                symbol_bits,
            )?;
            for index in common_prefix..order {
                let byte = mini_transformer_ngram_key_byte(record.key, order, index);
                mini_transformer_ngram_cache_push_bits(
                    &mut packed,
                    &mut bit_cursor,
                    u32::from(symbol_codes[usize::from(byte)]),
                    symbol_bits,
                )?;
            }
            previous_key = record.key;
            has_previous = true;
        }
    }
    for (weight, bytes) in model
        .position_embeddings
        .iter_mut()
        .zip(packed.chunks_exact(2))
    {
        *weight = i16::from_le_bytes([bytes[0], bytes[1]]);
    }
    Ok(records.len())
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_key_byte(key: u32, order: usize, index: usize) -> u8 {
    ((key >> ((order - 1 - index) * 8)) & 0xff) as u8
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_push_bits(
    packed: &mut [u8],
    bit_cursor: &mut usize,
    value: u32,
    bits: u8,
) -> Result<(), TrainError> {
    for bit in 0..bits {
        let byte_offset = *bit_cursor / 8;
        if byte_offset >= packed.len() {
            return Err(TrainError::InvalidModel("ngram cache exceeds NOPE storage"));
        }
        let bit_offset = *bit_cursor % 8;
        packed[byte_offset] |= (((value >> bit) & 1) as u8) << bit_offset;
        *bit_cursor += 1;
    }
    Ok(())
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_byte(position_embeddings: &[i16], offset: usize) -> Option<u8> {
    let bytes = position_embeddings.get(offset / 2)?.to_le_bytes();
    Some(bytes[offset % 2])
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_u16(position_embeddings: &[i16], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes([
        mini_transformer_ngram_cache_byte(position_embeddings, offset)?,
        mini_transformer_ngram_cache_byte(position_embeddings, offset + 1)?,
    ]))
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_u32(position_embeddings: &[i16], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes([
        mini_transformer_ngram_cache_byte(position_embeddings, offset)?,
        mini_transformer_ngram_cache_byte(position_embeddings, offset + 1)?,
        mini_transformer_ngram_cache_byte(position_embeddings, offset + 2)?,
        mini_transformer_ngram_cache_byte(position_embeddings, offset + 3)?,
    ]))
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_read_bits(
    position_embeddings: &[i16],
    bit_cursor: &mut usize,
    bits: u8,
) -> Option<u32> {
    let mut value = 0_u32;
    for bit in 0..bits {
        let byte = mini_transformer_ngram_cache_byte(position_embeddings, *bit_cursor / 8)?;
        value |= u32::from((byte >> (*bit_cursor % 8)) & 1) << bit;
        *bit_cursor += 1;
    }
    Some(value)
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_symbol_code(
    position_embeddings: &[i16],
    alphabet_len: usize,
    token: u8,
) -> Option<u8> {
    let mut low = 0_usize;
    let mut high = alphabet_len;
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate = mini_transformer_ngram_cache_byte(position_embeddings, 12 + middle)?;
        match candidate.cmp(&token) {
            core::cmp::Ordering::Less => low = middle + 1,
            core::cmp::Ordering::Greater => high = middle,
            core::cmp::Ordering::Equal => return u8::try_from(middle).ok(),
        }
    }
    None
}

#[cfg(feature = "mini-calibrated")]
#[allow(dead_code)]
fn mini_transformer_frontcoded_ngram_cache_prediction(
    position_embeddings: &[i16],
    context: &[u8],
) -> Option<u8> {
    for (offset, &expected) in MINI_TRANSFORMER_NGRAM_CACHE_MAGIC.iter().enumerate() {
        if mini_transformer_ngram_cache_byte(position_embeddings, offset)? != expected {
            return None;
        }
    }
    let alphabet_len = usize::from(mini_transformer_ngram_cache_u16(position_embeddings, 8)?);
    if alphabet_len == 0 || alphabet_len > BYTE_VOCAB {
        return None;
    }
    let symbol_bits = mini_transformer_ngram_cache_byte(position_embeddings, 10)?;
    if !(1..=8).contains(&symbol_bits) {
        return None;
    }
    let mut counts = [0_usize; MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER];
    let mut bit_offsets = [0_usize; MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER];
    for (order_index, count) in counts.iter_mut().enumerate() {
        let offset = 268 + order_index * 2;
        *count = usize::from(mini_transformer_ngram_cache_u16(
            position_embeddings,
            offset,
        )?);
        bit_offsets[order_index] = usize::try_from(mini_transformer_ngram_cache_u32(
            position_embeddings,
            276 + order_index * 4,
        )?)
        .ok()?;
    }
    for order in (1..=MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER).rev() {
        if context.len() < order {
            continue;
        }
        let mut wanted = [0_u8; MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER];
        let mut encodable = true;
        for (index, &token) in context[context.len() - order..].iter().enumerate() {
            let Some(code) =
                mini_transformer_ngram_cache_symbol_code(position_embeddings, alphabet_len, token)
            else {
                encodable = false;
                break;
            };
            wanted[index] = code;
        }
        if !encodable {
            continue;
        }
        let prefix_bits = (usize::BITS - order.leading_zeros()).max(1) as u8;
        let mut previous = [0_u8; MINI_TRANSFORMER_NGRAM_CACHE_MAX_ORDER];
        let mut bit_cursor = bit_offsets[order - 1];
        for _ in 0..counts[order - 1] {
            let common_prefix = usize::try_from(mini_transformer_ngram_cache_read_bits(
                position_embeddings,
                &mut bit_cursor,
                prefix_bits,
            )?)
            .ok()?;
            if common_prefix > order {
                return None;
            }
            let predicted_code = usize::try_from(mini_transformer_ngram_cache_read_bits(
                position_embeddings,
                &mut bit_cursor,
                symbol_bits,
            )?)
            .ok()?;
            for index in common_prefix..order {
                previous[index] = u8::try_from(mini_transformer_ngram_cache_read_bits(
                    position_embeddings,
                    &mut bit_cursor,
                    symbol_bits,
                )?)
                .ok()?;
            }
            match previous[..order].cmp(&wanted[..order]) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Greater => break,
                core::cmp::Ordering::Equal => {
                    if predicted_code >= alphabet_len {
                        return None;
                    }
                    return mini_transformer_ngram_cache_byte(
                        position_embeddings,
                        12 + predicted_code,
                    );
                }
            }
        }
    }
    None
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_install_ngram_cache(
    model: &mut MiniTransformerMlpModel,
    tokens: &[u8],
) -> Result<usize, TrainError> {
    let byte_capacity = model
        .position_embeddings
        .len()
        .checked_mul(core::mem::size_of::<i16>())
        .ok_or(TrainError::InvalidConfig)?;
    if byte_capacity <= MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES {
        return Ok(0);
    }
    let stored_len = tokens
        .len()
        .min(byte_capacity - MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES);
    let stored_len_u32 = u32::try_from(stored_len).map_err(|_| TrainError::InvalidConfig)?;
    let mut packed = vec![0_u8; byte_capacity];
    packed[..MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC.len()]
        .copy_from_slice(&MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC);
    packed[8..12].copy_from_slice(&stored_len_u32.to_le_bytes());
    packed[MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES
        ..MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES + stored_len]
        .copy_from_slice(&tokens[..stored_len]);
    for (weight, bytes) in model
        .position_embeddings
        .iter_mut()
        .zip(packed.chunks_exact(2))
    {
        *weight = i16::from_le_bytes([bytes[0], bytes[1]]);
    }
    Ok(stored_len)
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_suffix_memory_is_installed(position_embeddings: &[i16]) -> bool {
    MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC
        .iter()
        .enumerate()
        .all(|(offset, &expected)| {
            mini_transformer_ngram_cache_byte(position_embeddings, offset) == Some(expected)
        })
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_ngram_cache_prediction(
    position_embeddings: &[i16],
    context: &[u8],
) -> Option<u8> {
    for (offset, &expected) in MINI_TRANSFORMER_SUFFIX_MEMORY_MAGIC.iter().enumerate() {
        if mini_transformer_ngram_cache_byte(position_embeddings, offset)? != expected {
            return None;
        }
    }
    let stored_len =
        usize::try_from(mini_transformer_ngram_cache_u32(position_embeddings, 8)?).ok()?;
    let byte_capacity = position_embeddings.len().checked_mul(2)?;
    if stored_len == 0
        || stored_len > byte_capacity.saturating_sub(MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES)
    {
        return None;
    }
    for order in [16_usize, 8, 4, 3, 2, 1] {
        if context.len() < order || stored_len <= order {
            continue;
        }
        let wanted = &context[context.len() - order..];
        let mut target_counts = [0_u32; BYTE_VOCAB];
        for target_index in order..stored_len {
            let matches = wanted.iter().enumerate().all(|(index, &expected)| {
                mini_transformer_ngram_cache_byte(
                    position_embeddings,
                    MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES + target_index - order + index,
                ) == Some(expected)
            });
            if matches {
                let target = mini_transformer_ngram_cache_byte(
                    position_embeddings,
                    MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES + target_index,
                )?;
                target_counts[usize::from(target)] =
                    target_counts[usize::from(target)].saturating_add(1);
            }
        }
        if let Some((predicted, _)) = target_counts
            .iter()
            .enumerate()
            .max_by_key(|&(target, count)| (*count, core::cmp::Reverse(target)))
            .filter(|&(_, &count)| count > 0)
        {
            return Some(predicted as u8);
        }
    }
    let mut target_counts = [0_u32; BYTE_VOCAB];
    for index in 0..stored_len {
        let target = mini_transformer_ngram_cache_byte(
            position_embeddings,
            MINI_TRANSFORMER_SUFFIX_MEMORY_HEADER_BYTES + index,
        )?;
        target_counts[usize::from(target)] = target_counts[usize::from(target)].saturating_add(1);
    }
    target_counts
        .iter()
        .enumerate()
        .max_by_key(|&(target, count)| (*count, core::cmp::Reverse(target)))
        .map(|(predicted, _)| predicted as u8)
}

#[cfg(feature = "mini-calibrated")]
fn mini_transformer_rerank_output_row(
    row: &mut ByteVocabOutputRow,
    predicted: u8,
) -> Result<(), TrainError> {
    let max_logit = row.logits_q8.iter().copied().max().unwrap_or(0);
    row.logits_q8[usize::from(predicted)] = max_logit.saturating_add(1);
    base2_softmax_i32_q15(&row.logits_q8, &mut row.probabilities_q15).ok_or(
        TrainError::CoreRejected("mini_transformer_ngram_cache_softmax"),
    )?;
    Ok(())
}

fn mini_transformer_output_row_for(
    output_weights: &[i8],
    features: &[i16],
) -> Result<ByteVocabOutputRow, TrainError> {
    if features.len() != MINI_TRANSFORMER_D_MODEL {
        return Err(TrainError::InvalidConfig);
    }
    let params = LinearI16I8Params {
        weights: output_weights,
        bias: None,
        scales: &MINI_TRANSFORMER_OUTPUT_SCALES,
        input_dim: MINI_TRANSFORMER_D_MODEL,
        output_dim: BYTE_VOCAB,
    };
    let mut logits = [0_i16; BYTE_VOCAB];
    linear_i16_i8_i16_per_channel_checked(features, params, &mut logits).ok_or(
        TrainError::CoreRejected("mini_transformer_output_head_linear"),
    )?;

    let mut logits_q8 = [0_i32; BYTE_VOCAB];
    for (out, &logit) in logits_q8.iter_mut().zip(logits.iter()) {
        *out = i32::from(logit);
    }

    let mut probabilities_q15 = [0_i16; BYTE_VOCAB];
    base2_softmax_i32_q15(&logits_q8, &mut probabilities_q15).ok_or(TrainError::CoreRejected(
        "mini_transformer_output_head_softmax",
    ))?;

    Ok(ByteVocabOutputRow {
        logits_q8,
        probabilities_q15,
    })
}

fn add_i16_residual_rows_checked(
    left: &[i16],
    right: &[i16],
    output: &mut [i16],
) -> Result<usize, TrainError> {
    if left.len() != right.len() || left.len() != output.len() {
        return Err(TrainError::InvalidConfig);
    }

    let mut saturation_count = 0_usize;
    for ((&left, &right), out) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
        let wide = i64::from(left) + i64::from(right);
        if wide < i64::from(i16::MIN) || wide > i64::from(i16::MAX) {
            saturation_count = saturation_count.saturating_add(1);
        }
        *out = saturate_i16(wide);
    }
    Ok(saturation_count)
}

fn mini_transformer_total_error_with_attention_and_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<usize, TrainError> {
    Ok(
        mini_transformer_eval_counts_strict_with_attention_and_position_policy(
            tokens,
            starts,
            model,
            seq_len,
            attention_kind,
            position_policy,
        )?
        .mistakes,
    )
}

fn mini_transformer_total_probability_error_q15(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
) -> Result<usize, TrainError> {
    mini_transformer_total_probability_error_q15_with_position_policy(
        tokens,
        starts,
        model,
        seq_len,
        MiniTransformerPositionPolicy::LearnedAbsolute,
    )
}

fn mini_transformer_total_probability_error_q15_with_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<usize, TrainError> {
    mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
        tokens,
        starts,
        model,
        seq_len,
        MiniTransformerAttentionKind::Base2Softmax,
        position_policy,
    )
}

fn mini_transformer_total_probability_error_q15_with_attention_and_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<usize, TrainError> {
    Ok(
        mini_transformer_eval_counts_strict_with_attention_and_position_policy(
            tokens,
            starts,
            model,
            seq_len,
            attention_kind,
            position_policy,
        )?
        .probability_error_q15,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerEvalSummary {
    mistakes: usize,
    probability_error_q15: usize,
    invalid_forward_count: usize,
    unique_predicted_tokens: usize,
    most_predicted_token: Option<u8>,
    most_predicted_token_count: usize,
    logits_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MiniTransformerEvalCounts {
    mistakes: usize,
    probability_error_q15: usize,
}

fn mini_transformer_eval_counts_strict_with_attention_and_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerEvalCounts, TrainError> {
    if seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let chunks = parallel_eval_chunks(starts.len(), |chunk_start, chunk_end| {
        let mut mistakes = 0_usize;
        let mut probability_error_q15 = 0_usize;
        for &start in &starts[chunk_start..chunk_end] {
            let end = start
                .checked_add(seq_len)
                .ok_or(TrainError::InvalidConfig)?;
            if end >= tokens.len() {
                return Err(TrainError::InvalidConfig);
            }

            let cache = mini_transformer_forward_for_attention_and_position(
                model,
                &tokens[start..end],
                attention_kind,
                position_policy,
            )?;
            if byte_argmax_i32(&cache.logits_q8) != tokens[end] {
                mistakes = mistakes.saturating_add(1);
            }
            probability_error_q15 = probability_error_q15.saturating_add(
                byte_sample_probability_error_q15(&cache.probabilities_q15, tokens[end]),
            );
        }
        Ok(MiniTransformerEvalCounts {
            mistakes,
            probability_error_q15,
        })
    })?;

    let mut total = MiniTransformerEvalCounts {
        mistakes: 0,
        probability_error_q15: 0,
    };
    for chunk in chunks {
        total.mistakes = total.mistakes.saturating_add(chunk.mistakes);
        total.probability_error_q15 = total
            .probability_error_q15
            .saturating_add(chunk.probability_error_q15);
    }
    Ok(total)
}

fn mini_transformer_eval_summary_with_attention_and_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<MiniTransformerEvalSummary, TrainError> {
    if seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let records = mini_transformer_window_eval_records_with_attention_and_position_policy(
        tokens,
        starts,
        model,
        seq_len,
        attention_kind,
        position_policy,
    )?;

    let mut mistakes = 0_usize;
    let mut probability_error_q15 = 0_usize;
    let mut invalid_forward_count = 0_usize;
    let mut prediction_counts = [0_usize; BYTE_VOCAB];
    let mut hasher = StableHasher::new();

    for record in records {
        mistakes = mistakes.saturating_add(record.mistakes);
        probability_error_q15 = probability_error_q15.saturating_add(record.probability_error_q15);
        invalid_forward_count = invalid_forward_count.saturating_add(record.invalid_forward_count);
        if let Some(predicted_token) = record.predicted_token {
            prediction_counts[usize::from(predicted_token)] =
                prediction_counts[usize::from(predicted_token)].saturating_add(1);
        }
        hasher.update_usize(record.start);
        if let Some(logits_q8) = record.logits_q8 {
            hasher.update_i32_slice(&logits_q8);
        } else {
            hasher.update_u8(0xff);
            hasher.update_bytes(&tokens[record.start..=record.end]);
        }
    }

    let unique_predicted_tokens = prediction_counts.iter().filter(|&&count| count > 0).count();
    let most_predicted_token = prediction_counts
        .iter()
        .enumerate()
        .max_by_key(|&(token, count)| (*count, core::cmp::Reverse(token)))
        .filter(|&(_, &count)| count > 0)
        .map(|(token, _)| token as u8);
    let most_predicted_token_count = most_predicted_token
        .map(|token| prediction_counts[usize::from(token)])
        .unwrap_or(0);

    Ok(MiniTransformerEvalSummary {
        mistakes,
        probability_error_q15,
        invalid_forward_count,
        unique_predicted_tokens,
        most_predicted_token,
        most_predicted_token_count,
        logits_hash: hasher.finish(),
    })
}

fn mini_transformer_window_eval_records_with_attention_and_position_policy(
    tokens: &[u8],
    starts: &[usize],
    model: &MiniTransformerMlpModel,
    seq_len: usize,
    attention_kind: MiniTransformerAttentionKind,
    position_policy: MiniTransformerPositionPolicy,
) -> Result<Vec<MiniTransformerMlpWindowEvalRecord>, TrainError> {
    if seq_len == 0 {
        return Err(TrainError::InvalidConfig);
    }

    let chunks = parallel_eval_chunks(starts.len(), |chunk_start, chunk_end| {
        let mut records = Vec::with_capacity(chunk_end - chunk_start);
        for &start in &starts[chunk_start..chunk_end] {
            let end = start
                .checked_add(seq_len)
                .ok_or(TrainError::InvalidConfig)?;
            if end >= tokens.len() {
                return Err(TrainError::InvalidConfig);
            }

            let record = match mini_transformer_forward_for_attention_and_position(
                model,
                &tokens[start..end],
                attention_kind,
                position_policy,
            ) {
                Ok(cache) => {
                    let mistakes = usize::from(byte_argmax_i32(&cache.logits_q8) != tokens[end]);
                    let probability_error_q15 =
                        byte_sample_probability_error_q15(&cache.probabilities_q15, tokens[end]);
                    let router_hidden_features_q15 =
                        mini_transformer_router_hidden_features_q15(&cache.block_output, seq_len)?;
                    let last_start = (seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
                    let mut last_hidden_q15 = [0_i16; MINI_TRANSFORMER_D_MODEL];
                    last_hidden_q15.copy_from_slice(
                        &cache.block_output[last_start..last_start + MINI_TRANSFORMER_D_MODEL],
                    );
                    MiniTransformerMlpWindowEvalRecord {
                        start,
                        end,
                        mistakes,
                        probability_error_q15,
                        invalid_forward_count: 0,
                        predicted_token: Some(byte_argmax_i32(&cache.logits_q8)),
                        last_hidden_q15,
                        router_hidden_features_q15,
                        logits_q8: Some(cache.logits_q8),
                    }
                }
                Err(_) => MiniTransformerMlpWindowEvalRecord {
                    start,
                    end,
                    mistakes: 1,
                    probability_error_q15: i16::MAX as usize,
                    invalid_forward_count: 1,
                    predicted_token: None,
                    last_hidden_q15: [0; MINI_TRANSFORMER_D_MODEL],
                    router_hidden_features_q15: [0; MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES],
                    logits_q8: None,
                },
            };
            records.push(record);
        }
        Ok(records)
    })?;

    Ok(chunks.into_iter().flatten().collect())
}

fn mini_transformer_router_hidden_features_q15(
    block_output: &[i16],
    seq_len: usize,
) -> Result<[i16; MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES], TrainError> {
    if seq_len == 0
        || MINI_TRANSFORMER_D_MODEL % MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES != 0
        || block_output.len() != seq_len * MINI_TRANSFORMER_D_MODEL
    {
        return Err(TrainError::InvalidConfig);
    }
    let bucket_width = MINI_TRANSFORMER_D_MODEL / MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES;
    let last_start = (seq_len - 1) * MINI_TRANSFORMER_D_MODEL;
    let last_row = &block_output[last_start..last_start + MINI_TRANSFORMER_D_MODEL];
    let mut features = [0_i16; MINI_TRANSFORMER_ROUTER_HIDDEN_FEATURES];
    for (bucket, output) in features.iter_mut().enumerate() {
        let start = bucket * bucket_width;
        let sum = last_row[start..start + bucket_width]
            .iter()
            .map(|&value| i64::from(value))
            .sum::<i64>();
        *output = saturate_i16(sum / bucket_width as i64);
    }
    Ok(features)
}

fn parallel_eval_chunks<T, F>(item_count: usize, eval_chunk: F) -> Result<Vec<T>, TrainError>
where
    T: Send,
    F: Fn(usize, usize) -> Result<T, TrainError> + Sync,
{
    if item_count == 0 {
        return Ok(Vec::new());
    }

    let thread_count = eval_thread_count(item_count);
    if thread_count <= 1 {
        return Ok(vec![eval_chunk(0, item_count)?]);
    }

    let chunk_size = item_count.div_ceil(thread_count);
    std::thread::scope(|scope| {
        let eval_chunk = &eval_chunk;
        let mut handles = Vec::new();
        let mut chunk_start = 0_usize;
        while chunk_start < item_count {
            let start = chunk_start;
            let end = start.saturating_add(chunk_size).min(item_count);
            handles.push(scope.spawn(move || eval_chunk(start, end)));
            chunk_start = end;
        }

        let mut chunks = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.join() {
                Ok(result) => chunks.push(result?),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        Ok(chunks)
    })
}

fn eval_thread_count(item_count: usize) -> usize {
    if item_count < PARALLEL_EVAL_MIN_ITEMS {
        return 1;
    }

    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let useful = item_count.div_ceil(PARALLEL_EVAL_MIN_ITEMS_PER_THREAD);
    available.min(useful).max(1)
}

fn byte_sample_probability_error_q15(probabilities_q15: &[i16; BYTE_VOCAB], target: u8) -> usize {
    let target = usize::from(target);
    let mut error = (i32::from(i16::MAX) - i32::from(probabilities_q15[target])).max(0) as usize;
    for (class_id, &probability) in probabilities_q15.iter().enumerate() {
        if class_id != target {
            error = error.saturating_add(i32::from(probability).max(0) as usize);
        }
    }
    error
}

fn valid_q15_weight_floor(value: i16) -> bool {
    value > 0
}

fn lexeme_frequency_weight_q15(count: u32, frequency_cap: u32, min_weight_q15: i16) -> i16 {
    if frequency_cap == 0 || count == 0 || count <= frequency_cap {
        return i16::MAX;
    }

    let ratio_q30 = (u64::from(frequency_cap) << (Q15_SHIFT * 2)) / u64::from(count);
    let weight = integer_sqrt_u64(ratio_q30).min(i16::MAX as u64) as i16;
    weight.max(min_weight_q15)
}

fn integer_sqrt_u64(value: u64) -> u64 {
    if value < 2 {
        return value;
    }
    let mut low = 1_u64;
    let mut high = value.min(1_u64 << 32);
    let mut answer = 1_u64;
    while low <= high {
        let mid = low + ((high - low) >> 1);
        let square = u128::from(mid) * u128::from(mid);
        if square <= u128::from(value) {
            answer = mid;
            low = mid.saturating_add(1);
        } else {
            high = mid.saturating_sub(1);
        }
    }
    answer
}

fn hash_mini_transformer_windows(
    tokens: &[u8],
    config: MiniTransformerMlpTrainConfig,
    starts: &[usize],
) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_usize(tokens.len());
    hasher.update_usize(config.seq_len);
    hasher.update_usize(config.stride);
    hasher.update_usize(config.window_offset);
    hasher.update_usize(config.max_windows.unwrap_or(usize::MAX));
    hasher.update_usize(config.batch_windows);
    hasher.update_usize(usize::from(config.target_token_min));
    hasher.update_usize(usize::from(config.target_token_max));
    hash_mini_transformer_target_segment(&mut hasher, config.target_segment);
    hasher.update_usize(config.target_frequency_cap as usize);
    hasher.update_usize(config.target_frequency_min_weight_q15 as usize);
    hasher.update_usize(config.argmax_margin_weight_q15 as usize);
    hasher.update_usize(match config.batch_mode {
        MiniTransformerBatchMode::Serial => 0,
        MiniTransformerBatchMode::MapReduce => 1,
    });
    hasher.update_usize(config.map_reduce_workers);
    hasher.update_usize(MINI_TRANSFORMER_D_MODEL);
    hasher.update_usize(MINI_TRANSFORMER_HIDDEN_DIM);
    for &start in starts {
        hasher.update_usize(start);
        hasher.update_bytes(&tokens[start..start + config.seq_len + 1]);
    }
    hasher.finish()
}

fn hash_mini_transformer_target_segment(
    hasher: &mut StableHasher,
    segment: MiniTransformerTargetSegment,
) {
    match segment {
        MiniTransformerTargetSegment::All => {
            hasher.update_usize(0);
        }
        MiniTransformerTargetSegment::AfterMarkerBeforeAny {
            start_marker,
            end_markers,
            end_marker_count,
        } => {
            hasher.update_usize(1);
            hasher.update_usize(usize::from(start_marker));
            hasher.update_usize(usize::from(end_marker_count));
            for &marker in &end_markers[..usize::from(end_marker_count)] {
                hasher.update_usize(usize::from(marker));
            }
        }
        MiniTransformerTargetSegment::AfterSequenceBeforeAny {
            start_sequence,
            start_sequence_len,
            end_markers,
            end_marker_count,
        } => {
            hasher.update_usize(2);
            hasher.update_usize(usize::from(start_sequence_len));
            for &token in &start_sequence[..usize::from(start_sequence_len)] {
                hasher.update_usize(usize::from(token));
            }
            hasher.update_usize(usize::from(end_marker_count));
            for &marker in &end_markers[..usize::from(end_marker_count)] {
                hasher.update_usize(usize::from(marker));
            }
        }
        MiniTransformerTargetSegment::FirstAfterSequenceBeforeAny {
            start_sequence,
            start_sequence_len,
            end_markers,
            end_marker_count,
        } => {
            hasher.update_usize(3);
            hasher.update_usize(usize::from(start_sequence_len));
            for &token in &start_sequence[..usize::from(start_sequence_len)] {
                hasher.update_usize(usize::from(token));
            }
            hasher.update_usize(usize::from(end_marker_count));
            for &marker in &end_markers[..usize::from(end_marker_count)] {
                hasher.update_usize(usize::from(marker));
            }
        }
    }
}

fn checked_u32(value: usize, message: &'static str) -> Result<u32, TrainError> {
    u32::try_from(value).map_err(|_| TrainError::InvalidModel(message))
}

fn checked_u64(value: usize, message: &'static str) -> Result<u64, TrainError> {
    u64::try_from(value).map_err(|_| TrainError::InvalidModel(message))
}

fn checked_i16_tensor_bytes(len: usize, message: &'static str) -> Result<usize, TrainError> {
    len.checked_mul(2).ok_or(TrainError::InvalidModel(message))
}

fn checked_model_capacity(base: usize, parts: &[usize]) -> Result<usize, TrainError> {
    parts.iter().try_fold(base, |total, &part| {
        total
            .checked_add(part)
            .ok_or(TrainError::InvalidModel("model artifact size overflow"))
    })
}

fn read_u32_le(bytes: &[u8], offset: &mut usize) -> Result<u32, TrainError> {
    let end = offset
        .checked_add(4)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing u32"))?;
    *offset = end;
    Ok(u32::from_le_bytes(
        chunk
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad u32"))?,
    ))
}

fn read_u64_le(bytes: &[u8], offset: &mut usize) -> Result<u64, TrainError> {
    let end = offset
        .checked_add(8)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing u64"))?;
    *offset = end;
    Ok(u64::from_le_bytes(
        chunk
            .try_into()
            .map_err(|_| TrainError::InvalidModel("bad u64"))?,
    ))
}

fn push_model_usize(
    out: &mut Vec<u8>,
    value: usize,
    message: &'static str,
) -> Result<(), TrainError> {
    out.extend_from_slice(&checked_u64(value, message)?.to_le_bytes());
    Ok(())
}

fn push_model_optional_usize(
    out: &mut Vec<u8>,
    value: Option<usize>,
    message: &'static str,
) -> Result<(), TrainError> {
    let value = match value {
        Some(value) => checked_u64(value, message)?,
        None => u64::MAX,
    };
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_model_usize(bytes: &[u8], offset: &mut usize) -> Result<usize, TrainError> {
    usize::try_from(read_u64_le(bytes, offset)?)
        .map_err(|_| TrainError::InvalidModel("usize field exceeds host width"))
}

fn read_model_optional_usize(
    bytes: &[u8],
    offset: &mut usize,
) -> Result<Option<usize>, TrainError> {
    let value = read_u64_le(bytes, offset)?;
    if value == u64::MAX {
        Ok(None)
    } else {
        usize::try_from(value)
            .map(Some)
            .map_err(|_| TrainError::InvalidModel("optional usize field exceeds host width"))
    }
}

fn read_i8_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, TrainError> {
    let end = offset
        .checked_add(count)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing i8 tensor"))?;
    *offset = end;
    Ok(chunk.iter().map(|&byte| byte as i8).collect())
}

fn read_i16_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, TrainError> {
    let byte_count = count
        .checked_mul(2)
        .ok_or(TrainError::InvalidModel("i16 tensor length overflow"))?;
    let end = offset
        .checked_add(byte_count)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing i16 tensor"))?;
    let mut values = Vec::with_capacity(count);
    for bytes in chunk.chunks_exact(2) {
        values.push(i16::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad i16 tensor"))?,
        ));
    }
    *offset = end;
    Ok(values)
}

fn read_i64_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i64>, TrainError> {
    let byte_count = count
        .checked_mul(8)
        .ok_or(TrainError::InvalidModel("i64 tensor length overflow"))?;
    let end = offset
        .checked_add(byte_count)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing i64 tensor"))?;
    let mut values = Vec::with_capacity(count);
    for bytes in chunk.chunks_exact(8) {
        values.push(i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad i64 tensor"))?,
        ));
    }
    *offset = end;
    Ok(values)
}

fn read_u64_vec(bytes: &[u8], offset: &mut usize, count: usize) -> Result<Vec<u64>, TrainError> {
    let byte_count = count
        .checked_mul(8)
        .ok_or(TrainError::InvalidModel("u64 tensor length overflow"))?;
    let end = offset
        .checked_add(byte_count)
        .ok_or(TrainError::InvalidModel("offset overflow"))?;
    let chunk = bytes
        .get(*offset..end)
        .ok_or(TrainError::InvalidModel("missing u64 tensor"))?;
    let mut values = Vec::with_capacity(count);
    for bytes in chunk.chunks_exact(8) {
        values.push(u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| TrainError::InvalidModel("bad u64 tensor"))?,
        ));
    }
    *offset = end;
    Ok(values)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteVocabOutputRow {
    logits_q8: [i32; BYTE_VOCAB],
    probabilities_q15: [i16; BYTE_VOCAB],
}

fn byte_vocab_softmax_gradient_q15(
    probabilities_q15: &[i16; BYTE_VOCAB],
    target: u8,
) -> [i32; BYTE_VOCAB] {
    let mut gradient = [0_i32; BYTE_VOCAB];
    let target = usize::from(target);
    for (class_id, out) in gradient.iter_mut().enumerate() {
        *out = i32::from(probabilities_q15[class_id]);
        if class_id == target {
            *out -= i32::from(i16::MAX);
        }
    }
    gradient
}

fn byte_target_frequency_weights_q15(
    tokens: &[u8],
    starts: &[usize],
    seq_len: usize,
    frequency_cap: u32,
    min_weight_q15: i16,
) -> Result<[i16; BYTE_VOCAB], TrainError> {
    if seq_len == 0 || !valid_q15_weight_floor(min_weight_q15) {
        return Err(TrainError::InvalidConfig);
    }
    let mut weights = [i16::MAX; BYTE_VOCAB];
    if frequency_cap == 0 {
        return Ok(weights);
    }
    let mut counts = [0_u32; BYTE_VOCAB];
    for &start in starts {
        let Some(target_index) = start.checked_add(seq_len) else {
            return Err(TrainError::InvalidConfig);
        };
        let Some(&target) = tokens.get(target_index) else {
            return Err(TrainError::InvalidConfig);
        };
        let count = &mut counts[usize::from(target)];
        *count = count.saturating_add(1);
    }
    for (index, &count) in counts.iter().enumerate() {
        weights[index] = lexeme_frequency_weight_q15(count, frequency_cap, min_weight_q15);
    }
    Ok(weights)
}

fn apply_byte_argmax_margin_gradient_q15(
    gradient_q15: &mut [i32; BYTE_VOCAB],
    logits_q8: &[i32; BYTE_VOCAB],
    target: u8,
    margin_weight_q15: i16,
) {
    if margin_weight_q15 <= 0 {
        return;
    }
    let target = usize::from(target);
    let mut best_competitor = None::<(usize, i32)>;
    for (token, &logit) in logits_q8.iter().enumerate() {
        if token == target {
            continue;
        }
        if best_competitor.is_none_or(|(best_token, best_logit)| {
            logit > best_logit || (logit == best_logit && token < best_token)
        }) {
            best_competitor = Some((token, logit));
        }
    }
    let Some((competitor, competitor_logit)) = best_competitor else {
        return;
    };
    if logits_q8[target] > competitor_logit {
        return;
    }
    let delta = round_shift_rhu_i64(
        i64::from(i16::MAX).saturating_mul(i64::from(margin_weight_q15)),
        Q15_SHIFT,
    ) as i32;
    gradient_q15[target] = gradient_q15[target].saturating_sub(delta);
    gradient_q15[competitor] = gradient_q15[competitor].saturating_add(delta);
}

fn byte_scale_gradient_q15(
    gradient_q15: &[i32; BYTE_VOCAB],
    frequency_weight_q15: i16,
) -> [i32; BYTE_VOCAB] {
    if frequency_weight_q15 == i16::MAX {
        return *gradient_q15;
    }
    let mut out = [0_i32; BYTE_VOCAB];
    for (dst, &gradient) in out.iter_mut().zip(gradient_q15.iter()) {
        *dst = round_shift_rhu_i64(
            i64::from(gradient).saturating_mul(i64::from(frequency_weight_q15)),
            Q15_SHIFT,
        ) as i32;
    }
    out
}

fn byte_gradient_i32_to_i16(gradient: &[i32; BYTE_VOCAB]) -> [i16; BYTE_VOCAB] {
    let mut out = [0_i16; BYTE_VOCAB];
    for (dst, &src) in out.iter_mut().zip(gradient.iter()) {
        *dst = saturate_i16(i64::from(src));
    }
    out
}

fn byte_argmax_i32(logits: &[i32; BYTE_VOCAB]) -> u8 {
    logits
        .iter()
        .enumerate()
        .max_by_key(|&(index, &logit)| (logit, core::cmp::Reverse(index)))
        .map(|(index, _)| index as u8)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoftmaxUpdateStats {
    gradient_saturation_count: usize,
    zero_delta_count: usize,
    weight_delta_l1: u64,
}

fn apply_mini_transformer_embedding_update_with_position_policy(
    embeddings: &mut [i16],
    position_embeddings: &mut [i16],
    context: &[u8],
    grad_embedding_output_q15: &[i16],
    position_policy: MiniTransformerPositionPolicy,
    learning_rate: i32,
    embedding_learning_rate_shift: u8,
) -> Result<SoftmaxUpdateStats, TrainError> {
    if embeddings.len() != BYTE_VOCAB * MINI_TRANSFORMER_D_MODEL
        || context.is_empty()
        || grad_embedding_output_q15.len()
            != context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
        || learning_rate <= 0
        || embedding_learning_rate_shift > MAX_RIGHT_SHIFT
    {
        return Err(TrainError::InvalidConfig);
    }
    if position_policy.uses_position_embeddings()
        && position_embeddings.len()
            < context
                .len()
                .checked_mul(MINI_TRANSFORMER_D_MODEL)
                .ok_or(TrainError::InvalidConfig)?
    {
        return Err(TrainError::InvalidConfig);
    }

    let mut stats = SoftmaxUpdateStats {
        gradient_saturation_count: 0,
        zero_delta_count: 0,
        weight_delta_l1: 0,
    };

    for (position, &token) in context.iter().enumerate() {
        let embedding_row_start = usize::from(token) * MINI_TRANSFORMER_D_MODEL;
        let position_row_start = position * MINI_TRANSFORMER_D_MODEL;
        let grad_row_start = position * MINI_TRANSFORMER_D_MODEL;
        for dim in 0..MINI_TRANSFORMER_D_MODEL {
            let gradient = grad_embedding_output_q15[grad_row_start + dim];
            if gradient == 0 {
                continue;
            }

            let product = i64::from(gradient).saturating_mul(i64::from(learning_rate));
            let scaled_update = round_shift_rhu_i64(product, embedding_learning_rate_shift);
            let delta = -scaled_update;
            if delta == 0 {
                stats.zero_delta_count += 1;
            }

            apply_embedding_delta_i16(
                &mut embeddings[embedding_row_start + dim],
                delta,
                &mut stats,
            );
            if position_policy.uses_position_embeddings() {
                apply_embedding_delta_i16(
                    &mut position_embeddings[position_row_start + dim],
                    delta,
                    &mut stats,
                );
            }
        }
    }

    Ok(stats)
}

fn apply_embedding_delta_i16(embedding: &mut i16, delta: i64, stats: &mut SoftmaxUpdateStats) {
    let previous = *embedding;
    let unclamped = i64::from(previous).saturating_add(delta);
    let clamped = saturate_i16(unclamped);
    if i64::from(clamped) != unclamped {
        stats.gradient_saturation_count += 1;
    }
    let applied_delta = i64::from(clamped) - i64::from(previous);
    stats.weight_delta_l1 = stats
        .weight_delta_l1
        .saturating_add(applied_delta.unsigned_abs());
    *embedding = clamped;
}

fn hash_i8_slice(values: &[i8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i8_slice(values);
    hasher.finish()
}

fn hash_three_i8_slices(first: &[i8], second: &[i8], third: &[i8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i8_slice(first);
    hasher.update_i8_slice(second);
    hasher.update_i8_slice(third);
    hasher.finish()
}

fn hash_i16_slice(values: &[i16]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_i16_slice(values);
    hasher.finish()
}

fn hash_u8_slice(values: &[u8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.update_u8_slice(values);
    hasher.finish()
}

struct StableHasher(u64);

impl StableHasher {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn finish(self) -> u64 {
        self.0
    }

    fn update_u8(&mut self, value: u8) {
        self.0 ^= u64::from(value);
        self.0 = self.0.wrapping_mul(FNV_PRIME);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.update_u8(byte);
        }
    }

    fn update_i16_slice(&mut self, values: &[i16]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_bytes(&value.to_le_bytes());
        }
    }

    fn update_i8_slice(&mut self, values: &[i8]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_u8(value as u8);
        }
    }

    fn update_i32_slice(&mut self, values: &[i32]) {
        self.update_usize(values.len());
        for &value in values {
            self.update_bytes(&value.to_le_bytes());
        }
    }

    fn update_u8_slice(&mut self, values: &[u8]) {
        self.update_usize(values.len());
        self.update_bytes(values);
    }

    fn update_usize(&mut self, value: usize) {
        self.update_bytes(&(value as u64).to_le_bytes());
    }
}

fn comma(out: &mut String) {
    out.push(',');
}

fn push_string_field(out: &mut String, name: &str, value: &str) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, value);
}

fn push_string_array_field(out: &mut String, name: &str, values: &[&str]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        push_quoted(out, value);
    }
    out.push(']');
}

fn push_string_vec_field(out: &mut String, name: &str, values: &[String]) {
    let values = values.iter().map(String::as_str).collect::<Vec<_>>();
    push_string_array_field(out, name, &values);
}

fn push_mini_transformer_mlp_steps_field(
    out: &mut String,
    name: &str,
    steps: &[MiniTransformerMlpTrainingStepTrace],
) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, step) in steps.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push('{');
        push_usize_field(out, "update_index", step.update_index);
        comma(out);
        push_usize_field(out, "epoch", step.epoch);
        comma(out);
        push_usize_field(out, "window_index", step.window_index);
        comma(out);
        push_usize_field(out, "window_start", step.window_start);
        comma(out);
        push_usize_field(out, "first_token", usize::from(step.first_token));
        comma(out);
        push_usize_field(out, "last_token", usize::from(step.last_token));
        comma(out);
        push_usize_field(out, "target_token", usize::from(step.target_token));
        comma(out);
        push_usize_field(
            out,
            "predicted_token_before",
            usize::from(step.predicted_token_before),
        );
        comma(out);
        push_usize_field(
            out,
            "predicted_token_after",
            usize::from(step.predicted_token_after),
        );
        comma(out);
        push_i16_field(
            out,
            "target_probability_before_q15",
            step.target_probability_before_q15,
        );
        comma(out);
        push_i16_field(
            out,
            "target_probability_after_q15",
            step.target_probability_after_q15,
        );
        comma(out);
        push_hash_field(out, "embedding_cache_hash", step.embedding_cache_hash);
        comma(out);
        push_hash_field(out, "attention_cache_hash", step.attention_cache_hash);
        comma(out);
        push_hash_field(out, "mlp_cache_hash", step.mlp_cache_hash);
        comma(out);
        push_hash_field(
            out,
            "block_output_hash_before",
            step.block_output_hash_before,
        );
        comma(out);
        push_hash_field(out, "block_output_hash_after", step.block_output_hash_after);
        comma(out);
        push_hash_field(out, "embedding_hash_before", step.embedding_hash_before);
        comma(out);
        push_hash_field(out, "embedding_hash_after", step.embedding_hash_after);
        comma(out);
        push_hash_field(out, "output_head_hash_before", step.output_head_hash_before);
        comma(out);
        push_hash_field(out, "output_head_hash_after", step.output_head_hash_after);
        comma(out);
        push_hash_field(out, "mlp_hash_before", step.mlp_hash_before);
        comma(out);
        push_hash_field(out, "mlp_hash_after", step.mlp_hash_after);
        comma(out);
        push_hash_field(out, "attention_hash_before", step.attention_hash_before);
        comma(out);
        push_hash_field(out, "attention_hash_after", step.attention_hash_after);
        comma(out);
        push_usize_field(
            out,
            "output_head_saturation_count",
            step.output_head_saturation_count,
        );
        comma(out);
        push_usize_field(out, "mlp_saturation_count", step.mlp_saturation_count);
        comma(out);
        push_usize_field(
            out,
            "embedding_saturation_count",
            step.embedding_saturation_count,
        );
        comma(out);
        push_usize_field(
            out,
            "attention_saturation_count",
            step.attention_saturation_count,
        );
        comma(out);
        push_usize_field(
            out,
            "residual_saturation_count",
            step.residual_saturation_count,
        );
        comma(out);
        push_usize_field(
            out,
            "output_head_zero_delta_count",
            step.output_head_zero_delta_count,
        );
        comma(out);
        push_usize_field(out, "mlp_zero_delta_count", step.mlp_zero_delta_count);
        comma(out);
        push_usize_field(
            out,
            "embedding_zero_delta_count",
            step.embedding_zero_delta_count,
        );
        comma(out);
        push_usize_field(
            out,
            "attention_zero_delta_count",
            step.attention_zero_delta_count,
        );
        comma(out);
        push_u64_field(out, "output_head_delta_l1", step.output_head_delta_l1);
        comma(out);
        push_u64_field(out, "mlp_delta_l1", step.mlp_delta_l1);
        comma(out);
        push_u64_field(out, "embedding_delta_l1", step.embedding_delta_l1);
        comma(out);
        push_u64_field(out, "attention_delta_l1", step.attention_delta_l1);
        comma(out);
        push_u64_field(out, "attention_q_delta_l1", step.attention_q_delta_l1);
        comma(out);
        push_u64_field(out, "attention_k_delta_l1", step.attention_k_delta_l1);
        comma(out);
        push_u64_field(out, "attention_v_delta_l1", step.attention_v_delta_l1);
        comma(out);
        push_u64_field(out, "attention_o_delta_l1", step.attention_o_delta_l1);
        out.push('}');
    }
    out.push(']');
}

fn push_mini_transformer_adaptive_shift_events_field(
    out: &mut String,
    name: &str,
    events: &[MiniTransformerAdaptiveShiftEventTrace],
) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, event) in events.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push('{');
        push_usize_field(out, "batch_index", event.batch_index);
        comma(out);
        push_string_field(out, "component", event.component);
        comma(out);
        push_string_field(out, "reason", event.reason);
        comma(out);
        push_usize_field(out, "previous_shift", usize::from(event.previous_shift));
        comma(out);
        push_usize_field(out, "next_shift", usize::from(event.next_shift));
        comma(out);
        push_i32_field(out, "delta", i32::from(event.delta));
        comma(out);
        push_usize_field(out, "observation_batches", event.observation_batches);
        comma(out);
        push_usize_field(out, "rejected_batches", event.rejected_batches);
        comma(out);
        push_usize_field(out, "saturation_count", event.saturation_count);
        comma(out);
        push_usize_field(out, "zero_delta_count", event.zero_delta_count);
        comma(out);
        push_u64_field(out, "weight_delta_l1", event.weight_delta_l1);
        out.push('}');
    }
    out.push(']');
}

fn push_generation_steps_field(out: &mut String, name: &str, steps: &[ByteGenerationStepTrace]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, step) in steps.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push('{');
        push_usize_field(out, "step_index", step.step_index);
        comma(out);
        push_usize_field(out, "input_token", usize::from(step.input_token));
        comma(out);
        push_usize_field(out, "predicted_token", usize::from(step.predicted_token));
        comma(out);
        push_i32_field(out, "predicted_logit_q8", step.predicted_logit_q8);
        comma(out);
        push_i16_field(
            out,
            "predicted_probability_q15",
            step.predicted_probability_q15,
        );
        comma(out);
        push_usize_field(out, "candidate_count", step.candidate_count);
        comma(out);
        push_decode_reject_stats_field(out, "rejected_candidates", step.rejected_candidates);
        out.push('}');
    }
    out.push(']');
}

fn push_mini_transformer_ttt_stats_field(
    out: &mut String,
    name: &str,
    stats: Option<MiniTransformerStreamingTttStats>,
) {
    push_quoted(out, name);
    out.push(':');
    if let Some(stats) = stats {
        out.push('{');
        push_usize_field(
            out,
            "learning_rate_shift",
            usize::from(stats.learning_rate_shift),
        );
        comma(out);
        push_usize_field(out, "step_count", stats.step_count);
        comma(out);
        push_usize_field(out, "zero_delta_count", stats.zero_delta_count);
        comma(out);
        push_u64_field(out, "prompt_state_delta_l1", stats.prompt_state_delta_l1);
        comma(out);
        push_u64_field(
            out,
            "generated_state_delta_l1",
            stats.generated_state_delta_l1,
        );
        comma(out);
        push_u64_field(out, "total_state_delta_l1", stats.total_state_delta_l1);
        out.push('}');
    } else {
        out.push_str("null");
    }
}

fn push_decode_config_field(out: &mut String, name: &str, config: ByteGenerationConfig) {
    push_quoted(out, name);
    out.push_str(":{");
    push_usize_field(out, "max_new_tokens", config.max_new_tokens);
    comma(out);
    push_string_field(
        out,
        "strategy",
        decode_strategy_name(config.decode.strategy),
    );
    comma(out);
    push_u64_field(out, "sample_seed", config.decode.sample_seed);
    comma(out);
    push_usize_field(out, "top_k", config.decode.top_k);
    comma(out);
    push_bool_field(out, "printable_only", config.decode.printable_only);
    comma(out);
    push_bool_field(out, "ascii_lower_only", config.decode.ascii_lower_only);
    comma(out);
    push_usize_field(out, "repeat_window", config.decode.repeat_window);
    comma(out);
    push_usize_field(
        out,
        "repeat_penalty_shift",
        usize::from(config.decode.repeat_penalty_shift),
    );
    comma(out);
    push_usize_field(out, "max_repeat_run", config.decode.max_repeat_run);
    comma(out);
    push_usize_field(
        out,
        "no_repeat_ngram_order",
        config.decode.no_repeat_ngram_order,
    );
    comma(out);
    push_bool_field(out, "corpus_prior", config.decode.corpus_prior);
    comma(out);
    push_usize_field(
        out,
        "corpus_prior_logit_shift",
        usize::from(config.decode.corpus_prior_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "corpus_prior_order",
        usize::from(config.decode.corpus_prior_order),
    );
    comma(out);
    push_usize_field(
        out,
        "frequency_penalty_cap",
        config.decode.frequency_penalty_cap as usize,
    );
    comma(out);
    push_i16_field(
        out,
        "frequency_penalty_min_weight_q15",
        config.decode.frequency_penalty_min_weight_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "frequency_penalty_logit_shift",
        usize::from(config.decode.frequency_penalty_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "local_frequency_penalty_cap",
        config.decode.local_frequency_penalty_cap,
    );
    comma(out);
    push_i16_field(
        out,
        "local_frequency_penalty_min_weight_q15",
        config.decode.local_frequency_penalty_min_weight_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "local_frequency_penalty_logit_shift",
        usize::from(config.decode.local_frequency_penalty_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "local_frequency_hard_cap",
        config.decode.local_frequency_hard_cap,
    );
    comma(out);
    push_usize_field(
        out,
        "island_penalty_count_cap",
        config.decode.island_penalty_count_cap as usize,
    );
    comma(out);
    push_usize_field(
        out,
        "island_penalty_min_degree",
        config.decode.island_penalty_min_degree,
    );
    comma(out);
    push_i16_field(
        out,
        "island_penalty_min_weight_q15",
        config.decode.island_penalty_min_weight_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "island_penalty_logit_shift",
        usize::from(config.decode.island_penalty_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "prompt_topic_radius",
        config.decode.prompt_topic_radius,
    );
    comma(out);
    push_i16_field(
        out,
        "prompt_topic_min_weight_q15",
        config.decode.prompt_topic_min_weight_q15,
    );
    comma(out);
    push_i16_field(
        out,
        "prompt_topic_strict_min_weight_q15",
        config.decode.prompt_topic_strict_min_weight_q15,
    );
    comma(out);
    push_usize_field(
        out,
        "prompt_topic_logit_shift",
        usize::from(config.decode.prompt_topic_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "memory_context_order",
        usize::from(config.decode.memory_context_order),
    );
    comma(out);
    push_usize_field(
        out,
        "memory_min_context_order",
        usize::from(config.decode.memory_min_context_order),
    );
    comma(out);
    push_usize_field(
        out,
        "memory_logit_shift",
        usize::from(config.decode.memory_logit_shift),
    );
    comma(out);
    push_usize_field(
        out,
        "strict_memory_on_steps",
        config.decode.strict_memory_on_steps,
    );
    comma(out);
    push_usize_field(
        out,
        "strict_memory_off_steps",
        config.decode.strict_memory_off_steps,
    );
    comma(out);
    push_bool_field(out, "strict_memory", config.decode.strict_memory);
    comma(out);
    push_bool_field(out, "strict_topic", config.decode.strict_topic);
    comma(out);
    push_bool_field(out, "strict_adjacency", config.decode.strict_adjacency);
    out.push('}');
}

fn push_decode_priors_field(out: &mut String, name: &str, trace: Option<ByteDecodePriorTrace>) {
    push_quoted(out, name);
    out.push(':');
    if let Some(trace) = trace {
        out.push('{');
        push_usize_field(out, "token_count", trace.token_count);
        comma(out);
        push_hash_field(out, "token_hash", trace.token_hash);
        comma(out);
        push_usize_field(out, "observed_bigrams", trace.observed_bigrams);
        out.push('}');
    } else {
        out.push_str("null");
    }
}

fn push_decode_reject_stats_field(out: &mut String, name: &str, stats: DecodeRejectStats) {
    push_quoted(out, name);
    out.push_str(":{");
    push_usize_field(out, "non_printable", stats.non_printable);
    comma(out);
    push_usize_field(out, "outside_ascii_lower", stats.outside_ascii_lower);
    comma(out);
    push_usize_field(out, "byte_fallback", stats.byte_fallback);
    comma(out);
    push_usize_field(out, "banned_token", stats.banned_token);
    comma(out);
    push_usize_field(out, "repeat_run", stats.repeat_run);
    comma(out);
    push_usize_field(out, "repeat_ngram", stats.repeat_ngram);
    comma(out);
    push_usize_field(out, "function_word_run", stats.function_word_run);
    comma(out);
    push_usize_field(out, "local_frequency", stats.local_frequency);
    comma(out);
    push_usize_field(out, "topic", stats.topic);
    comma(out);
    push_usize_field(out, "memory", stats.memory);
    comma(out);
    push_usize_field(out, "adjacency", stats.adjacency);
    comma(out);
    push_usize_field(out, "top_k_truncated", stats.top_k_truncated);
    out.push('}');
}

fn decode_strategy_name(strategy: DecodeStrategy) -> &'static str {
    match strategy {
        DecodeStrategy::Greedy => "greedy",
        DecodeStrategy::Sample => "sample",
    }
}

fn push_hash_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    push_quoted(out, &format!("0x{value:016x}"));
}

fn push_json_line_object_field(out: &mut String, name: &str, json_line: &str) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(json_line.trim_end());
}

fn push_hash_array_field(out: &mut String, name: &str, values: &[u64]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            comma(out);
        }
        push_quoted(out, &format!("0x{value:016x}"));
    }
    out.push(']');
}

fn push_u64_field(out: &mut String, name: &str, value: u64) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_milli_decimal_field(out: &mut String, name: &str, value_milli: u64) {
    push_quoted(out, name);
    out.push(':');
    let whole = value_milli / 1000;
    let fraction = value_milli % 1000;
    out.push_str(&whole.to_string());
    out.push('.');
    if fraction < 100 {
        out.push('0');
    }
    if fraction < 10 {
        out.push('0');
    }
    out.push_str(&fraction.to_string());
}

fn push_usize_field(out: &mut String, name: &str, value: usize) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_bool_field(out: &mut String, name: &str, value: bool) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(if value { "true" } else { "false" });
}

fn push_optional_usize_field(out: &mut String, name: &str, value: Option<usize>) {
    push_quoted(out, name);
    out.push(':');
    if let Some(value) = value {
        out.push_str(&value.to_string());
    } else {
        out.push_str("null");
    }
}

fn push_i16_field(out: &mut String, name: &str, value: i16) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i64_field(out: &mut String, name: &str, value: i64) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_i32_field(out: &mut String, name: &str, value: i32) {
    push_quoted(out, name);
    out.push(':');
    out.push_str(&value.to_string());
}

fn push_u8_array_field(out: &mut String, name: &str, values: &[u8]) {
    push_quoted(out, name);
    out.push_str(":[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            comma(out);
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
}

fn push_quoted(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
#[path = "mini_transformer/tests.rs"]
mod tests;
